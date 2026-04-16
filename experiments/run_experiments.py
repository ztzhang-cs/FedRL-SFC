# experiments/run_experiments.py
import random
from typing import Dict, List, Tuple, Optional
import copy
import networkx as nx
import numpy as np
import argparse
import inspect
from collections import defaultdict

from fedrl_vnf.models import Domain
from fedrl_vnf.topology import (
    build_random_domains,
    generate_random_sfc,
    build_toy_domains,
)
from fedrl_vnf.orchestrators import GlobalOrchestrator
from fedrl_vnf.heuristic import deploy_sfc_heuristic
from fedrl_vnf.baselines_fedgreedy import FederatedGreedyOrchestrator

from .plotting import (
    plot_convergence_delays,
    plot_acceptance,
    plot_blocking,
    plot_utilization,
    plot_attention_heatmap,
    plot_delay_bar,
    plot_acceptance_bar,
)

# =========================
# SFC 生成配置
# =========================
GEN_TRAIN_RANDOM = dict(min_len=4, max_len=7, cpu_demand_range=(5, 9), bw_demand_range=(10, 40))
GEN_EVAL_RANDOM  = dict(min_len=4, max_len=7, cpu_demand_range=(5, 9), bw_demand_range=(10, 40))

GEN_TRAIN_TOY = dict(min_len=4, max_len=6, cpu_demand_range=(5, 9), bw_demand_range=(15, 45))
GEN_EVAL_TOY  = dict(min_len=4, max_len=6, cpu_demand_range=(5, 9), bw_demand_range=(15, 45))


# =========================
# Reset utils
# =========================
def reset_all_domains(domains: Dict[str, Domain]):
    for d in domains.values():
        for n in d.nodes.values():
            n.cpu_used = 0.0
        if hasattr(d, "graph") and d.graph is not None:
            for _, _, data in d.graph.edges(data=True):
                for k in ("bw_used", "used_bw", "load", "util"):
                    if k in data:
                        data[k] = 0.0


def reset_domain_graph_usage(domain_graph: nx.DiGraph):
    for _, _, data in domain_graph.edges(data=True):
        for k in ("bw_used", "used_bw", "load", "util"):
            if k in data:
                data[k] = 0.0


def set_eval_mode(domains: Dict[str, Domain], is_eval: bool):
    for dom in domains.values():
        dom._attn_eval_mode = bool(is_eval)
        dom._global_eval_mode = bool(is_eval)


def _collect_utils(domains: Dict[str, Domain]) -> Tuple[float, float]:
    util = []
    for dom in domains.values():
        for node in dom.nodes.values():
            cap = float(getattr(node, "cpu_capacity", 0.0))
            if cap > 0:
                util.append(float(getattr(node, "cpu_used", 0.0)) / cap)
    if not util:
        return 0.0, 0.0
    return float(max(util)), float(np.mean(util))


# =========================
# SFC generation
# =========================
def gen_train_sfcs(
    domain_ids: List[str],
    ep: int,
    n_train: int,
    gen_train: dict,
    seed: int
):
    r = random.Random(seed + 10007 * (ep + 1))
    train_sfcs = []
    for i in range(n_train):
        sfc_id = f"TRAIN_ep{ep}_req{i}"
        train_sfcs.append(generate_random_sfc(sfc_id, domain_ids, **gen_train, rng=r))
    return train_sfcs


def build_fixed_eval_sfcs(
    domain_ids: List[str],
    n_eval: int,
    gen_eval: dict,
    seed: int,
    tag: str,
):
    """
    ✅ 固定 eval 集合：显著降低曲线方差
    """
    r = random.Random(seed + 777777)  # fixed seed independent of ep
    eval_sfcs = []
    for i in range(n_eval):
        sfc_id = f"{tag}_EVAL_FIXED_req{i}"
        eval_sfcs.append(generate_random_sfc(sfc_id, domain_ids, **gen_eval, rng=r))
    return eval_sfcs


# =========================
# Steady-state load helpers (departures / releases)
# =========================
def _extract_cpu_alloc_from_result(sfc, placement_result) -> List[Tuple[str, str, float]]:
    """
    从 PlacementResult 里提取本次请求占用的 (domain_id, node_id, cpu_demand) 列表，便于后续释放。
    """
    name2cpu = {v.name: float(getattr(v, "cpu_demand", 0.0)) for v in getattr(sfc, "vnfs", [])}
    alloc = []
    v2n = getattr(placement_result, "vnf_to_node", {}) or {}
    for vname, (did, nid) in v2n.items():
        alloc.append((did, nid, float(name2cpu.get(vname, 0.0))))
    return alloc


def _release_alloc(domains: Dict[str, Domain], alloc: List[Tuple[str, str, float]]):
    """
    释放一条已部署 SFC 占用的 cpu_used（简单减回去，做下界截断）。
    """
    for did, nid, cpu in alloc:
        try:
            node = domains[did].nodes[nid]
            node.cpu_used = max(0.0, float(node.cpu_used) - float(cpu))
        except Exception:
            # 保守：释放失败就跳过（不影响主流程）
            pass


# =========================
# Batch runners
# =========================
def _run_batch_with_sfcs(
    orchestrator,
    domains: Dict[str, Domain],
    sfcs,
    rng: Optional[random.Random] = None,
    enable_departures: bool = True,
    release_prob: float = 0.40,
    max_active: int = 40,
    fail_delay: float = 400.0,
):
    """
    ✅ 关键改动：
    - enable_departures=True：模拟在线负载（请求会结束释放资源），否则 cpu_used 只增不减 -> acc 必然很低
    - delay 使用 penalized average：失败也记一个大 delay，避免“成功多了 delay 反而升”的统计假象
    """
    rng = rng or random.Random(0)

    ep_delays = []
    ep_success = 0
    ep_fail = 0

    # active allocations for departure process
    active: List[List[Tuple[str, str, float]]] = []

    sfcs_local = copy.deepcopy(sfcs)
    for sfc_i in sfcs_local:
        # --- departure before next arrival ---
        if enable_departures and active and (rng.random() < float(release_prob)):
            j = rng.randrange(len(active))
            _release_alloc(domains, active.pop(j))

        try:
            res = orchestrator.deploy_sfc(sfc_i)
            ep_success += 1

            d = float(getattr(res, "total_delay", 0.0))
            ep_delays.append(d)

            if enable_departures:
                alloc = _extract_cpu_alloc_from_result(sfc_i, res)
                active.append(alloc)

                # hard cap: avoid unbounded growth even if release is unlucky
                if max_active is not None and max_active > 0 and len(active) > int(max_active):
                    _release_alloc(domains, active.pop(0))

        except RuntimeError:
            ep_fail += 1
            # penalize failed attempt as large delay
            ep_delays.append(float(fail_delay))

    avg_d = float(np.mean(ep_delays)) if ep_delays else float(fail_delay)
    max_u, avg_u = _collect_utils(domains)

    n = max(len(sfcs_local), 1)
    acc = ep_success / n
    blk = ep_fail / n
    return avg_d, acc, blk, max_u, avg_u

def _eval_heuristic_batch(
    domains,
    domain_graph,
    sfcs,
    k_paths: int,
    fail_delay: float = 400.0,
    enable_departures: bool = True,
    release_prob: float = 0.45,
    max_active: int = 60,
    rng: Optional[random.Random] = None,
):
    reset_all_domains(domains)
    reset_domain_graph_usage(domain_graph)
    set_eval_mode(domains, True)

    rng = rng or random.Random(0)
    active: List[List[Tuple[str, str, float]]] = []

    delays_pen = []
    success = 0
    fail = 0

    for sfc in copy.deepcopy(sfcs):
        # departure before next arrival
        if enable_departures and active and (rng.random() < float(release_prob)):
            j = rng.randrange(len(active))
            _release_alloc(domains, active.pop(j))

        d, vmap = deploy_sfc_heuristic(
            sfc, domains, domain_graph,
            k_paths=k_paths,
            choose_best_path=False,   # ✅公平：非-oracle
            return_mapping=True,      # ✅拿映射做 departure
        )

        if d >= 1e8:
            fail += 1
            delays_pen.append(float(fail_delay))
            continue

        success += 1
        delays_pen.append(float(d))

        if enable_departures:
            # build alloc list directly from vmap
            name2cpu = {v.name: float(getattr(v, "cpu_demand", 0.0)) for v in getattr(sfc, "vnfs", [])}
            alloc = []
            for vname, (did, nid) in (vmap or {}).items():
                alloc.append((did, nid, float(name2cpu.get(vname, 0.0))))
            active.append(alloc)

            if max_active is not None and max_active > 0 and len(active) > int(max_active):
                _release_alloc(domains, active.pop(0))

    avg_delay = float(np.mean(delays_pen)) if delays_pen else float(fail_delay)
    max_u, avg_u = _collect_utils(domains)
    acc = success / len(sfcs) if sfcs else 0.0
    blk = fail / len(sfcs) if sfcs else 1.0
    return avg_delay, acc, blk, max_u, avg_u



def _safe_set_mode(orchestrator, is_eval: bool, eval_action_samples: int = 1):
    if not hasattr(orchestrator, "_supports_eval_action_samples"):
        try:
            sig = inspect.signature(orchestrator.set_mode)
            orchestrator._supports_eval_action_samples = ("eval_action_samples" in sig.parameters)
        except Exception:
            orchestrator._supports_eval_action_samples = False

    try:
        if is_eval and orchestrator._supports_eval_action_samples:
            orchestrator.set_mode(True, eval_action_samples=eval_action_samples)
        else:
            orchestrator.set_mode(is_eval)
    except TypeError:
        orchestrator.set_mode(is_eval)


def _run_orch_batch(
    orchestrator,
    domains,
    graph,
    sfcs,
    eval_mode: bool,
    eval_action_samples: int = 1,
    do_reset: bool = True,
    rng: Optional[random.Random] = None,
    enable_departures: bool = True,
    release_prob: float = 0.40,
    max_active: int = 40,
    fail_delay: float = 400.0,
):
    _safe_set_mode(orchestrator, eval_mode, eval_action_samples=eval_action_samples)

    if do_reset:
        reset_all_domains(domains)
        reset_domain_graph_usage(graph)

    set_eval_mode(domains, eval_mode)
    return _run_batch_with_sfcs(
        orchestrator, domains, sfcs,
        rng=rng,
        enable_departures=enable_departures,
        release_prob=release_prob,
        max_active=max_active,
        fail_delay=fail_delay,
    )


# =========================
# Core experiment loop (shared)
# =========================
def _init_logs(method_names: List[str]):
    logs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m in method_names:
        for phase in ("train", "eval"):
            for k in ("delay", "acc", "block", "max_util", "avg_util"):
                logs[m][phase][k]
    return logs


def _push_metrics(logs, method: str, phase: str, tup):
    d, acc, blk, maxu, avgu = tup
    logs[method][phase]["delay"].append(float(d))
    logs[method][phase]["acc"].append(float(acc))
    logs[method][phase]["block"].append(float(blk))
    logs[method][phase]["max_util"].append(float(maxu))
    logs[method][phase]["avg_util"].append(float(avgu))


def _method_id(method: str) -> int:
    if method == "local":
        return 11
    if method == "fed":
        return 22
    if method == "fedgreedy":
        return 33
    return 0


# =========================
# Random experiment
# =========================
def run_experiment_random(
    num_domains: int = 12,
    num_episodes: int = 200,
    sfc_per_episode: int = 120,
    eval_sfc_per_episode: int = 120,
    k_paths: int = 3,
    fedagg_interval: int = 3,
    seed: int = 42,
    eval_fixed: bool = True,

    # ✅ 新增：稳态负载控制（强烈建议开）
    enable_departures: bool = True,
    release_prob_train: float = 0.35,
    release_prob_eval: float = 0.60,
    max_active_train: int = 60,
    max_active_eval: int = 60,
    fail_delay: float = 450.0,
):
    random.seed(seed)
    np.random.seed(seed)

    domains_base, domain_graph_clean = build_random_domains(num_domains=num_domains)
    domain_ids = list(domains_base.keys())

    gen_train = dict(GEN_TRAIN_RANDOM)
    gen_eval  = dict(GEN_EVAL_RANDOM)

    fixed_eval_sfcs = build_fixed_eval_sfcs(
        domain_ids, eval_sfc_per_episode, gen_eval, seed=seed, tag="RANDOM"
    ) if eval_fixed else None

    doms = {
        "heuristic": {did: copy.deepcopy(dom) for did, dom in domains_base.items()},
        "local":    {did: copy.deepcopy(dom) for did, dom in domains_base.items()},
        "fed":      {did: copy.deepcopy(dom) for did, dom in domains_base.items()},
        "fedgreedy":{did: copy.deepcopy(dom) for did, dom in domains_base.items()},
    }
    graphs = {
        "heuristic": copy.deepcopy(domain_graph_clean),
        "local":     copy.deepcopy(domain_graph_clean),
        "fed":       copy.deepcopy(domain_graph_clean),
        "fedgreedy": copy.deepcopy(domain_graph_clean),
    }

    orchestrators = {
        "local": GlobalOrchestrator(doms["local"], graphs["local"], k_paths=k_paths),
        "fed":   GlobalOrchestrator(doms["fed"],   graphs["fed"],   k_paths=k_paths),
        "fedgreedy": FederatedGreedyOrchestrator(
            doms["fedgreedy"], graphs["fedgreedy"], k_paths=k_paths, fedavg_interval=fedagg_interval
        ),
    }

    logs = _init_logs(["local", "fed", "fedgreedy"])
    delays_h, acc_h, block_h, max_util_h, avg_util_h = [], [], [], [], []
    fed_attn_rows = []

    FED_WARMUP = 10

    for ep in range(num_episodes):
        train_sfcs = gen_train_sfcs(domain_ids, ep, sfc_per_episode, gen_train, seed)
        if eval_fixed:
            eval_sfcs = fixed_eval_sfcs
        else:
            r = random.Random(seed + 99991 + 1009 * (ep + 1))
            eval_sfcs = [
                generate_random_sfc(f"EVAL_ep{ep}_req{i}", domain_ids, **gen_eval, rng=r)
                for i in range(eval_sfc_per_episode)
            ]

        # --- Heuristic (EVAL) ---
        rng_h = random.Random(seed + 500000 + 1009 * (ep + 1) + 99)
        h = _eval_heuristic_batch(
            doms["heuristic"], graphs["heuristic"], eval_sfcs,
            k_paths=k_paths,
            fail_delay=fail_delay,
            enable_departures=enable_departures,
            release_prob=release_prob_eval,
            max_active=max_active_eval,
            rng=rng_h,
        )

        delays_h.append(h[0]); acc_h.append(h[1]); block_h.append(h[2]); max_util_h.append(h[3]); avg_util_h.append(h[4])

        # --- epsilon schedules ---
        orchestrators["local"].update_epsilon(
            episode=ep, start_eps=0.8, min_eps=0.02, warmup_eps_episodes=5, decay_per_episode=0.025
        )
        orchestrators["fed"].update_epsilon(
            episode=ep, start_eps=0.8, min_eps=0.02, warmup_eps_episodes=5, decay_per_episode=0.025
        )

        # --- Methods loop ---
        for method in ("local", "fed", "fedgreedy"):
            orch = orchestrators[method]
            dom = doms[method]
            gr  = graphs[method]

            mid = _method_id(method)

            # RNG for reproducible departure decisions
            rng_train = random.Random(seed + 300000 + 1009 * (ep + 1) + mid)
            rng_eval  = random.Random(seed + 400000 + 1009 * (ep + 1) + mid)

            # TRAIN
            t = _run_orch_batch(
                orch, dom, gr, train_sfcs,
                eval_mode=False,
                eval_action_samples=1,
                rng=rng_train,
                enable_departures=enable_departures,
                release_prob=release_prob_train,
                max_active=max_active_train,
                fail_delay=fail_delay,
            )
            _push_metrics(logs, method, "train", t)

            # aggregation
            if method == "fed":
                if (ep + 1) >= FED_WARMUP and (ep + 1) % fedagg_interval == 0:
                    orch.federated_aggregation()
            elif method == "fedgreedy":
                if (ep + 1) % fedagg_interval == 0:
                    orch.federated_aggregation()

            # EVAL (deterministic policy, but with steady-state departures)
            e = _run_orch_batch(
                orch, dom, gr, eval_sfcs,
                eval_mode=True,
                eval_action_samples=1,
                rng=rng_eval,
                enable_departures=enable_departures,
                release_prob=release_prob_eval,
                max_active=max_active_eval,
                fail_delay=fail_delay,
            )
            _push_metrics(logs, method, "eval", e)

        # --- peakedness heatmap ---
        row = []
        orch_fed = orchestrators["fed"]
        if hasattr(orch_fed, "domain_orchs"):
            for did in domain_ids:
                o = orch_fed.domain_orchs[did]
                cnt = float(getattr(o, "_peakedness_cnt", 0))
                row.append(float(getattr(o, "_peakedness_sum", 0.0)) / cnt if cnt > 0 else np.nan)
                o._peakedness_sum = 0.0
                o._peakedness_cnt = 0
        fed_attn_rows.append(row if row else [np.nan] * len(domain_ids))

        print(
            f"[Random Ep {ep+1}/{num_episodes}] "
            f"H acc={acc_h[-1]:.3f} | L acc={logs['local']['eval']['acc'][-1]:.3f} | "
            f"F acc={logs['fed']['eval']['acc'][-1]:.3f} | G acc={logs['fedgreedy']['eval']['acc'][-1]:.3f}"
        )

    results = {
        "heuristic": {"delay": delays_h, "acc": acc_h, "block": block_h, "max_util": max_util_h, "avg_util": avg_util_h},
        "local": {
            "delay": logs["local"]["eval"]["delay"],
            "delay_train": logs["local"]["train"]["delay"],
            "delay_eval": logs["local"]["eval"]["delay"],
            "acc": logs["local"]["eval"]["acc"],
            "block": logs["local"]["eval"]["block"],
            "max_util": logs["local"]["eval"]["max_util"],
            "avg_util": logs["local"]["eval"]["avg_util"],
            "acc_train": logs["local"]["train"]["acc"],
            "block_train": logs["local"]["train"]["block"],
            "max_util_train": logs["local"]["train"]["max_util"],
            "avg_util_train": logs["local"]["train"]["avg_util"],
        },
        "fed": {
            "delay": logs["fed"]["eval"]["delay"],
            "delay_train": logs["fed"]["train"]["delay"],
            "delay_eval": logs["fed"]["eval"]["delay"],
            "acc": logs["fed"]["eval"]["acc"],
            "block": logs["fed"]["eval"]["block"],
            "max_util": logs["fed"]["eval"]["max_util"],
            "avg_util": logs["fed"]["eval"]["avg_util"],
            "acc_train": logs["fed"]["train"]["acc"],
            "block_train": logs["fed"]["train"]["block"],
            "max_util_train": logs["fed"]["train"]["max_util"],
            "avg_util_train": logs["fed"]["train"]["avg_util"],
            "attn_mat": np.asarray(fed_attn_rows, dtype=float),
            "attn_xlabels": domain_ids,
        },
        "fedgreedy": {
            "delay": logs["fedgreedy"]["eval"]["delay"],
            "delay_train": logs["fedgreedy"]["train"]["delay"],
            "delay_eval": logs["fedgreedy"]["eval"]["delay"],
            "acc": logs["fedgreedy"]["eval"]["acc"],
            "block": logs["fedgreedy"]["eval"]["block"],
            "max_util": logs["fedgreedy"]["eval"]["max_util"],
            "avg_util": logs["fedgreedy"]["eval"]["avg_util"],
            "acc_train": logs["fedgreedy"]["train"]["acc"],
            "block_train": logs["fedgreedy"]["train"]["block"],
            "max_util_train": logs["fedgreedy"]["train"]["max_util"],
            "avg_util_train": logs["fedgreedy"]["train"]["avg_util"],
        },
    }
    return results


# =========================
# Toy experiment
# =========================
def run_experiment_toy(
    num_episodes: int = 150,
    sfc_per_episode: int = 40,
    eval_sfc_per_episode: int = 80,
    k_paths: int = 3,
    fedagg_interval: int = 3,
    seed: int = 42,
    eval_fixed: bool = True,

    # ✅ 新增：稳态负载控制（Toy 更需要）
    enable_departures: bool = True,
    release_prob_train: float = 0.40,
    release_prob_eval: float = 0.60,
    max_active_train: int = 25,
    max_active_eval: int = 25,
    fail_delay: float = 320.0,
):
    random.seed(seed)
    np.random.seed(seed)

    domains_base, domain_graph_clean = build_toy_domains()
    domain_ids = list(domains_base.keys())

    gen_train = dict(GEN_TRAIN_TOY)
    gen_eval  = dict(GEN_EVAL_TOY)

    fixed_eval_sfcs = build_fixed_eval_sfcs(
        domain_ids, eval_sfc_per_episode, gen_eval, seed=seed, tag="TOY"
    ) if eval_fixed else None

    doms = {
        "heuristic": {did: copy.deepcopy(dom) for did, dom in domains_base.items()},
        "local":    {did: copy.deepcopy(dom) for did, dom in domains_base.items()},
        "fed":      {did: copy.deepcopy(dom) for did, dom in domains_base.items()},
        "fedgreedy":{did: copy.deepcopy(dom) for did, dom in domains_base.items()},
    }
    graphs = {
        "heuristic": copy.deepcopy(domain_graph_clean),
        "local":     copy.deepcopy(domain_graph_clean),
        "fed":       copy.deepcopy(domain_graph_clean),
        "fedgreedy": copy.deepcopy(domain_graph_clean),
    }

    orchestrators = {
        "local": GlobalOrchestrator(doms["local"], graphs["local"], k_paths=k_paths, agent_lr=0.06, agent_init_eps=0.7),
        "fed":   GlobalOrchestrator(doms["fed"],   graphs["fed"],   k_paths=k_paths, agent_lr=0.06, agent_init_eps=0.7),
        "fedgreedy": FederatedGreedyOrchestrator(
            doms["fedgreedy"], graphs["fedgreedy"], k_paths=k_paths, fedavg_interval=fedagg_interval
        ),
    }

    logs = _init_logs(["local", "fed", "fedgreedy"])
    delays_h, acc_h, block_h, max_util_h, avg_util_h = [], [], [], [], []
    fed_attn_rows = []

    FED_WARMUP = 10

    for ep in range(num_episodes):
        train_sfcs = gen_train_sfcs(domain_ids, ep, sfc_per_episode, gen_train, seed)
        if eval_fixed:
            eval_sfcs = fixed_eval_sfcs
        else:
            r = random.Random(seed + 99991 + 1009 * (ep + 1))
            eval_sfcs = [
                generate_random_sfc(f"EVAL_ep{ep}_req{i}", domain_ids, **gen_eval, rng=r)
                for i in range(eval_sfc_per_episode)
            ]

        rng_h = random.Random(seed + 500000 + 1009 * (ep + 1) + 99)
        h = _eval_heuristic_batch(
            doms["heuristic"], graphs["heuristic"], eval_sfcs,
            k_paths=k_paths,
            fail_delay=fail_delay,
            enable_departures=enable_departures,
            release_prob=release_prob_eval,
            max_active=max_active_eval,
            rng=rng_h,
        )

        delays_h.append(h[0]); acc_h.append(h[1]); block_h.append(h[2]); max_util_h.append(h[3]); avg_util_h.append(h[4])

        orchestrators["local"].update_epsilon(
            episode=ep, start_eps=0.8, min_eps=0.02, warmup_eps_episodes=5, decay_per_episode=0.025
        )
        orchestrators["fed"].update_epsilon(
            episode=ep, start_eps=0.8, min_eps=0.02, warmup_eps_episodes=5, decay_per_episode=0.025
        )

        for method in ("local", "fed", "fedgreedy"):
            orch = orchestrators[method]
            dom = doms[method]
            gr  = graphs[method]

            mid = _method_id(method)
            rng_train = random.Random(seed + 130000 + 1009 * (ep + 1) + mid)
            rng_eval  = random.Random(seed + 140000 + 1009 * (ep + 1) + mid)

            t = _run_orch_batch(
                orch, dom, gr, train_sfcs,
                eval_mode=False,
                eval_action_samples=1,
                rng=rng_train,
                enable_departures=enable_departures,
                release_prob=release_prob_train,
                max_active=max_active_train,
                fail_delay=fail_delay,
            )
            _push_metrics(logs, method, "train", t)

            if method == "fed":
                if (ep + 1) >= FED_WARMUP and (ep + 1) % fedagg_interval == 0:
                    orch.federated_aggregation()
            elif method == "fedgreedy":
                if (ep + 1) % fedagg_interval == 0:
                    orch.federated_aggregation()

            e = _run_orch_batch(
                orch, dom, gr, eval_sfcs,
                eval_mode=True,
                eval_action_samples=1,
                rng=rng_eval,
                enable_departures=enable_departures,
                release_prob=release_prob_eval,
                max_active=max_active_eval,
                fail_delay=fail_delay,
            )
            _push_metrics(logs, method, "eval", e)

        row = []
        orch_fed = orchestrators["fed"]
        if hasattr(orch_fed, "domain_orchs"):
            for did in domain_ids:
                o = orch_fed.domain_orchs[did]
                cnt = float(getattr(o, "_peakedness_cnt", 0))
                row.append(float(getattr(o, "_peakedness_sum", 0.0)) / cnt if cnt > 0 else np.nan)
                o._peakedness_sum = 0.0
                o._peakedness_cnt = 0
        fed_attn_rows.append(row if row else [np.nan] * len(domain_ids))

        print(
            f"[Toy Ep {ep+1}/{num_episodes}] "
            f"H acc={acc_h[-1]:.3f} | L acc={logs['local']['eval']['acc'][-1]:.3f} | "
            f"F acc={logs['fed']['eval']['acc'][-1]:.3f} | G acc={logs['fedgreedy']['eval']['acc'][-1]:.3f}"
        )

    results = {
        "heuristic": {"delay": delays_h, "acc": acc_h, "block": block_h, "max_util": max_util_h, "avg_util": avg_util_h},
        "local": {
            "delay": logs["local"]["eval"]["delay"],
            "delay_train": logs["local"]["train"]["delay"],
            "delay_eval": logs["local"]["eval"]["delay"],
            "acc": logs["local"]["eval"]["acc"],
            "block": logs["local"]["eval"]["block"],
            "max_util": logs["local"]["eval"]["max_util"],
            "avg_util": logs["local"]["eval"]["avg_util"],
            "acc_train": logs["local"]["train"]["acc"],
            "block_train": logs["local"]["train"]["block"],
            "max_util_train": logs["local"]["train"]["max_util"],
            "avg_util_train": logs["local"]["train"]["avg_util"],
        },
        "fed": {
            "delay": logs["fed"]["eval"]["delay"],
            "delay_train": logs["fed"]["train"]["delay"],
            "delay_eval": logs["fed"]["eval"]["delay"],
            "acc": logs["fed"]["eval"]["acc"],
            "block": logs["fed"]["eval"]["block"],
            "max_util": logs["fed"]["eval"]["max_util"],
            "avg_util": logs["fed"]["eval"]["avg_util"],
            "acc_train": logs["fed"]["train"]["acc"],
            "block_train": logs["fed"]["train"]["block"],
            "max_util_train": logs["fed"]["train"]["max_util"],
            "avg_util_train": logs["fed"]["train"]["avg_util"],
            "attn_mat": np.asarray(fed_attn_rows, dtype=float),
            "attn_xlabels": domain_ids
        },
        "fedgreedy": {
            "delay": logs["fedgreedy"]["eval"]["delay"],
            "delay_train": logs["fedgreedy"]["train"]["delay"],
            "delay_eval": logs["fedgreedy"]["eval"]["delay"],
            "acc": logs["fedgreedy"]["eval"]["acc"],
            "block": logs["fedgreedy"]["eval"]["block"],
            "max_util": logs["fedgreedy"]["eval"]["max_util"],
            "avg_util": logs["fedgreedy"]["eval"]["avg_util"],
            "acc_train": logs["fedgreedy"]["train"]["acc"],
            "block_train": logs["fedgreedy"]["train"]["block"],
            "max_util_train": logs["fedgreedy"]["train"]["max_util"],
            "avg_util_train": logs["fedgreedy"]["train"]["avg_util"],
        },
    }
    return results


def save_and_plot(results, out_npz: str):
    h = results["heuristic"]
    l = results["local"]
    f = results["fed"]
    g = results["fedgreedy"]

    save_dict = dict(
        delay_h=np.array(h["delay"]),
        delay_l=np.array(l["delay"]),
        delay_f=np.array(f["delay"]),
        delay_g=np.array(g["delay"]),
        delay_l_train=np.array(l.get("delay_train", l["delay"])),
        delay_f_train=np.array(f.get("delay_train", f["delay"])),
        delay_g_train=np.array(g.get("delay_train", g["delay"])),
        acc_h=np.array(h["acc"]),
        acc_l=np.array(l["acc"]),
        acc_f=np.array(f["acc"]),
        acc_g=np.array(g["acc"]),
        block_h=np.array(h["block"]),
        block_l=np.array(l["block"]),
        block_f=np.array(f["block"]),
        block_g=np.array(g["block"]),
        max_util_h=np.array(h["max_util"]),
        max_util_l=np.array(l["max_util"]),
        max_util_f=np.array(f["max_util"]),
        max_util_g=np.array(g["max_util"]),
        avg_util_h=np.array(h["avg_util"]),
        avg_util_l=np.array(l["avg_util"]),
        avg_util_f=np.array(f["avg_util"]),
        avg_util_g=np.array(g["avg_util"]),
    )

    for key, obj in (("local", l), ("fed", f), ("fedgreedy", g)):
        if "acc_train" in obj:
            save_dict[f"acc_{key}_train"] = np.array(obj["acc_train"])
        if "block_train" in obj:
            save_dict[f"block_{key}_train"] = np.array(obj["block_train"])
        if "max_util_train" in obj:
            save_dict[f"max_util_{key}_train"] = np.array(obj["max_util_train"])
        if "avg_util_train" in obj:
            save_dict[f"avg_util_{key}_train"] = np.array(obj["avg_util_train"])

    np.savez(out_npz, **save_dict)
    print(f"Results saved to {out_npz}")

    plot_convergence_delays(
        h["delay"], l["delay"], f["delay"], g["delay"],
        l_train=l.get("delay_train"),
        f_train=f.get("delay_train"),
        g_train=g.get("delay_train"),
        show_train=False,
    )
    plot_acceptance(h["acc"], l["acc"], f["acc"], g["acc"])
    plot_blocking(h["block"], l["block"], f["block"], g["block"])
    plot_utilization(
        h["max_util"], l["max_util"], f["max_util"], g["max_util"],
        h["avg_util"], l["avg_util"], f["avg_util"], g["avg_util"],
    )

    delay_dict = {"Heuristic": h["delay"], "Local RL": l["delay"], "FedRL": f["delay"], "FedGreedy": g["delay"]}
    acc_dict   = {"Heuristic": h["acc"],   "Local RL": l["acc"],   "FedRL": f["acc"],   "FedGreedy": g["acc"]}

    plot_delay_bar(delay_dict, tail=30, save_path="results/delay_bar.png")
    plot_acceptance_bar(acc_dict, tail=30, save_path="results/acceptance_bar.png")

    if "attn_mat" in results.get("fed", {}):
        plot_attention_heatmap(
            results["fed"]["attn_mat"],
            x_labels=results["fed"].get("attn_xlabels", None),
            y_labels=results["fed"].get("attn_ylabels", None),
            title="FedRL attention heatmap",
            save_path="results/attn_heatmap.png",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy", "random"], default="toy")
    parser.add_argument("--eval_fixed", action="store_true", help="Use fixed eval set (recommended)")
    args = parser.parse_args()

    if args.mode == "toy":
        print(">>> Running TOY topology experiment...")
        results = run_experiment_toy(eval_fixed=args.eval_fixed)
        save_and_plot(results, "results/results_toy.npz")
    else:
        print(">>> Running RANDOM topology experiment...")
        results = run_experiment_random(eval_fixed=args.eval_fixed)
        save_and_plot(results, "results/results_random.npz")


if __name__ == "__main__":
    main()

