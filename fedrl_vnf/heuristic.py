from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np

from .models import Domain, VNF, SFC


def _cpu_free(node) -> float:
    cap = float(getattr(node, "cpu_capacity", 0.0))
    used = float(getattr(node, "cpu_used", 0.0))
    return max(0.0, cap - used)


def _proc_delay(used: float, demand: float, cap: float, alpha: float, power: float) -> float:
    util_after = float((used + demand) / (cap + 1e-9))
    util_after = float(np.clip(util_after, 0.0, 1.5))
    return float(alpha * (util_after ** power))


class DomainHeuristicOrchestrator:
    """
    域内启发式：greedy（与 RL delay 口径尽量一致）
    step_delay = d_cur + pdelay
    cost = step_delay + w_exit * d_exit + w_util * util_after + slack_bar + ingress_pen
    """
    def __init__(
        self,
        domain: Domain,
        proc_delay_alpha: Optional[float] = None,
        proc_delay_pow: Optional[float] = None,
        w_exit: float = 0.35,
        w_util: float = 0.30,
        slack_eps: float = 0.04,
        slack_lambda: float = 0.20,
        use_proc_delay: bool = True,
        ingress_penalty: float = 0.35,
    ):
        self.domain = domain

        # ✅ 同步环境参数（你 topology 里 _set_domain_delay_profile 挂的）
        if proc_delay_alpha is None:
            proc_delay_alpha = getattr(domain, "proc_delay_alpha", 10.0)
        if proc_delay_pow is None:
            proc_delay_pow = getattr(domain, "proc_delay_pow", 2.0)

        self.proc_delay_alpha = float(proc_delay_alpha)
        self.proc_delay_pow = float(proc_delay_pow)

        self.w_exit = float(w_exit)
        self.w_util = float(w_util)

        self.slack_eps = float(slack_eps)
        self.slack_lambda = float(slack_lambda)

        self.use_proc_delay = bool(use_proc_delay)

        # ✅ 轻微“反诱饵”：热点域 ingress_bias>0，避免一直往 ingress 塞
        self.ingress_bias = float(getattr(domain, "ingress_bias", 0.0))
        self.ingress_penalty = float(ingress_penalty)

    def place_segment(
        self,
        vnfs: List[VNF],
        entry_node: str,
        exit_node: str,
    ) -> Tuple[Dict[str, str], float]:
        g = self.domain.graph
        nodes = self.domain.nodes
        cur = entry_node

        total_delay = 0.0
        mapping: Dict[str, str] = {}
        if not vnfs:
            return mapping, 0.0

        for vnf in vnfs:
            demand = float(getattr(vnf, "cpu_demand", 0.0))

            best_nid = None
            best_cost = float("inf")
            best_step_delay = float("inf")

            for nid, node in nodes.items():
                cap = float(getattr(node, "cpu_capacity", 0.0)) + 1e-9
                used = float(getattr(node, "cpu_used", 0.0))
                free = _cpu_free(node)

                if free < demand:
                    continue

                try:
                    d_cur = float(nx.shortest_path_length(g, cur, nid, weight="delay"))
                    d_exit = float(nx.shortest_path_length(g, nid, exit_node, weight="delay"))
                except nx.NetworkXNoPath:
                    continue

                pdelay = 0.0
                if self.use_proc_delay:
                    pdelay = _proc_delay(used, demand, cap, self.proc_delay_alpha, self.proc_delay_pow)

                util_after = float((used + demand) / cap)

                # slack barrier（轻）
                slack = float((free - demand) / cap)
                slack_bar = 0.0
                if slack < self.slack_eps:
                    slack_bar = (self.slack_eps - slack) / max(self.slack_eps, 1e-9)
                slack_bar *= self.slack_lambda

                # 轻微反诱饵：热点域 ingress 节点加一点惩罚
                ingress_pen = 0.0
                if self.ingress_bias > 0.0 and nid == self.domain.ingress_node:
                    ingress_pen = self.ingress_penalty * self.ingress_bias

                step_delay = d_cur + pdelay
                cost = step_delay + self.w_exit * d_exit + self.w_util * util_after + slack_bar + ingress_pen

                if cost < best_cost:
                    best_cost = cost
                    best_nid = nid
                    best_step_delay = step_delay

            if best_nid is None:
                return {}, 1e9

            nodes[best_nid].cpu_used += demand
            total_delay += float(best_step_delay)
            mapping[vnf.name] = best_nid
            cur = best_nid

        try:
            total_delay += float(nx.shortest_path_length(g, cur, exit_node, weight="delay"))
        except nx.NetworkXNoPath:
            return {}, 1e9

        return mapping, float(total_delay)


def _assign_vnfs_to_domains(vnfs: List[VNF], domains: Dict[str, Domain], domain_path: List[str]) -> Dict[str, str]:
    """
    简单比例分配：按各域 free_cpu 分配 VNF 个数
    """
    n_vnfs = len(vnfs)
    cpu_list = [max(domains[did].total_cpu_free(), 0.01) for did in domain_path]
    cpu_sum = sum(cpu_list)

    counts = [int(n_vnfs * c / cpu_sum) for c in cpu_list]
    assigned = sum(counts)
    i = 0
    while assigned < n_vnfs:
        counts[i % len(counts)] += 1
        assigned += 1
        i += 1

    vnf_to_domain: Dict[str, str] = {}
    cur = 0
    for did, cnt in zip(domain_path, counts):
        for _ in range(cnt):
            if cur >= n_vnfs:
                break
            vnf_to_domain[vnfs[cur].name] = did
            cur += 1
    for j in range(cur, n_vnfs):
        vnf_to_domain[vnfs[j].name] = domain_path[-1]
    return vnf_to_domain


def _path_score_non_oracle(
    path: List[str],
    domains: Dict[str, Domain],
    domain_graph: nx.DiGraph,
    cpu_penalty: float = 22.0,
) -> float:
    """
    非 oracle 的路径打分：不模拟部署，只用
    inter_domain_delay + cpu_penalty * sum(1/(free_cpu + eps))
    """
    inter = 0.0
    for u, v in zip(path[:-1], path[1:]):
        inter += float(domain_graph[u][v].get("delay", 0.0))

    inv_cpu = 0.0
    for did in path:
        free = float(max(domains[did].total_cpu_free(), 0.01))
        inv_cpu += 1.0 / (free + 1e-6)

    return float(inter + cpu_penalty * inv_cpu)

def deploy_sfc_heuristic(
    sfc: SFC,
    domains: Dict[str, Domain],
    domain_graph: nx.DiGraph,
    k_paths: int = 3,
    proc_delay_alpha: Optional[float] = None,
    proc_delay_pow: Optional[float] = None,
    use_proc_delay: bool = True,
    choose_best_path: bool = False,  # True=oracle（不建议公平对比）
    return_mapping: bool = False,    # ✅新增：给 departures 用
):
    def _snapshot_cpu():
        return {did: {nid: float(n.cpu_used) for nid, n in dom.nodes.items()} for did, dom in domains.items()}

    def _restore_cpu(snap):
        for did, ns in snap.items():
            if did not in domains:
                continue
            for nid, used in ns.items():
                if nid in domains[did].nodes:
                    domains[did].nodes[nid].cpu_used = float(used)

    def _simulate_one_path(path: List[str]):
        """
        在“当前 domains 状态”上真实部署（会修改 cpu_used）。
        成功返回 (delay, vnf_to_node_map)，失败返回 (1e9, {}).
        """
        total_delay = 0.0
        vnf_to_node: Dict[str, Tuple[str, str]] = {}

        name_to_idx = {v.name: i for i, v in enumerate(sfc.vnfs)}
        vnf_to_domain = _assign_vnfs_to_domains(sfc.vnfs, domains, path)

        cur_failed = False
        for did in path:
            dom = domains[did]
            entry = dom.ingress_node
            exitn = dom.egress_node

            seg_vnfs = [v for v in sfc.vnfs if vnf_to_domain.get(v.name) == did]
            seg_vnfs.sort(key=lambda v: name_to_idx.get(v.name, 10**9))

            orch = DomainHeuristicOrchestrator(
                dom,
                proc_delay_alpha=proc_delay_alpha,
                proc_delay_pow=proc_delay_pow,
                use_proc_delay=use_proc_delay,
            )

            seg_map, seg_delay = orch.place_segment(seg_vnfs, entry_node=entry, exit_node=exitn)
            if seg_delay >= 1e9:
                cur_failed = True
                break

            # seg_map: vname -> nid
            for vname, nid in (seg_map or {}).items():
                vnf_to_node[vname] = (did, nid)

            total_delay += float(seg_delay)

        if cur_failed:
            return 1e9, {}

        # inter-domain delay
        for u, v in zip(path[:-1], path[1:]):
            total_delay += float(domain_graph[u][v].get("delay", 0.0))

        return float(total_delay), vnf_to_node

    # ---------- generate candidate paths ----------
    try:
        gen = nx.shortest_simple_paths(domain_graph, sfc.src_domain, sfc.dst_domain, weight="delay")
    except nx.NetworkXNoPath:
        return (1e9, {}) if return_mapping else 1e9

    paths = []
    for p in gen:
        paths.append(p)
        if len(paths) >= int(max(1, k_paths)):
            break
    if not paths:
        return (1e9, {}) if return_mapping else 1e9

    # ✅默认非-oracle：先选一条“更稳”的路径，然后真实部署并提交资源
    if not choose_best_path:
        best = None
        best_s = float("inf")
        for p in paths:
            s = _path_score_non_oracle(p, domains, domain_graph)
            if s < best_s:
                best_s = s
                best = p
        path = best if best is not None else paths[0]

        snap = _snapshot_cpu()
        try:
            d, vmap = _simulate_one_path(path)
        except Exception:
            d, vmap = 1e9, {}

        if d >= 1e8:
            _restore_cpu(snap)  # ✅失败才回滚
            return (1e9, {}) if return_mapping else 1e9

        # ✅成功：不回滚，提交 cpu_used
        return (d, vmap) if return_mapping else d

    # ---------- oracle mode (for debugging only) ----------
    base = _snapshot_cpu()
    best_delay = float("inf")
    best_map: Dict[str, Tuple[str, str]] = {}

    for path in paths:
        _restore_cpu(base)
        try:
            d, vmap = _simulate_one_path(path)
        except Exception:
            d, vmap = 1e9, {}

        if d < best_delay:
            best_delay = float(d)
            best_map = dict(vmap)

    _restore_cpu(base)  # oracle 不改变环境
    if best_delay >= 1e8:
        return (1e9, {}) if return_mapping else 1e9
    return (best_delay, best_map) if return_mapping else best_delay



