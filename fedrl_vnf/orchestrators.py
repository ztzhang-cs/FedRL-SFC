# fedrl_vnf/orchestrators.py
from typing import Dict, List, Tuple, Optional
import random
import math
import numpy as np
import networkx as nx

from .models import Domain, VNF, SFC, PlacementResult
from .agents import DomainRLAgent


# =========================================================
# AttnPoolNodeScorer
# =========================================================
class AttnPoolNodeScorer:
    """
    attention + pooling（numpy）
    Personalized-FL: Wo = Wo_global + Wo_local
    """
    def __init__(self, feature_dim: int):
        self.F = int(feature_dim)
        self.D = 16
        rng = np.random.default_rng(2025)

        self.Wq = np.zeros((self.F, self.D), dtype=np.float32)
        self.Wk = np.zeros((self.F, self.D), dtype=np.float32)
        self.Wv = np.zeros((self.F, self.D), dtype=np.float32)

        key_idx = {
            "cpu_free_ratio": 0,
            "cpu_util_now": 1,
            "degree_norm": 2,
            "dist_from_cur_norm": 3,
            "dist_to_exit_norm": 4,
            "vnf_demand_ratio": 5,
            "cpu_slack_ratio": 6,
        }

        for name, idx in key_idx.items():
            base = 0.0
            if name in ["dist_from_cur_norm", "dist_to_exit_norm"]:
                base = 1.2
            elif name in ["cpu_util_now"]:
                base = 1.0
            elif name in ["cpu_free_ratio", "cpu_slack_ratio"]:
                base = 0.8
            elif name in ["vnf_demand_ratio"]:
                base = 0.6
            elif name in ["degree_norm"]:
                base = 0.3

            direction = rng.normal(0, 1.0, size=(self.D,)).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            self.Wq[idx] = base * direction
            self.Wk[idx] = base * direction
            self.Wv[idx] = (base * 1.2) * direction

        self.q_pool = rng.normal(0, 1.0, size=(self.D,)).astype(np.float32)
        self.q_pool = self.q_pool / (np.linalg.norm(self.q_pool) + 1e-6)

        self.Wo_global = rng.normal(0, 0.05, size=(2 * self.D,)).astype(np.float32)
        self.Wo_local = np.zeros((2 * self.D,), dtype=np.float32)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-9)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        scale = 1.0 / math.sqrt(self.D)
        logits = (Q @ K.T) * scale
        A = self._softmax(logits, axis=1)
        H = A @ V

        pool_logits = H @ self.q_pool
        pool_alpha = self._softmax(pool_logits, axis=0)
        domain_emb = (pool_alpha[:, None] * H).sum(axis=0)
        return H, domain_emb, pool_alpha

    def node_z(self, H: np.ndarray, domain_emb: np.ndarray) -> np.ndarray:
        N = H.shape[0]
        domain_tile = np.repeat(domain_emb[None, :], repeats=N, axis=0)
        return np.concatenate([H, domain_tile], axis=1)

    def effective_Wo(self) -> np.ndarray:
        return self.Wo_global + self.Wo_local

    def node_costs(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.effective_Wo()


# =========================================================
# DomainOrchestrator (intra-domain placement)
# =========================================================
class DomainOrchestrator:
    def __init__(self, domain: "Domain"):
        self.domain = domain

        self._F = 7
        self._scorer = AttnPoolNodeScorer(feature_dim=self._F)

        self.W_HEUR = 0.95
        self.W_ATTN = 0.05

        self.pg_lr = 0.015
        self.pg_wd = 0.001
        
        # 初始温度降下来，避免完全随机分配
        self.tau = 0.20

        self.action_samples = 1

        rng = np.random.RandomState(abs(hash(domain.id)) % (2**32))
        self.delay_pref = float(rng.uniform(0.6, 1.4))
        self.util_pref  = float(rng.uniform(0.6, 1.4))

        self._traj = []
        self._peakedness_sum = 0.0
        self._peakedness_cnt = 0

        self.proc_delay_alpha = float(getattr(domain, "proc_delay_alpha", 10.0))
        self.proc_delay_pow   = float(getattr(domain, "proc_delay_pow", 2.0))

        self._warmup_episodes = 15
        self._ramp_episodes = 80

    def set_action_samples(self, n: int):
        self.action_samples = int(max(1, n))

    def set_training_progress(self, episode: int):
        e = int(max(0, episode))
        if e <= self._warmup_episodes:
            attn = 0.05
        else:
            t = min(1.0, (e - self._warmup_episodes) / float(max(1, self._ramp_episodes)))
            # RL 权重上限控制在 0.35，保证 Heuristic 的防拥塞机制始终起基础作用
            attn = 0.05 + 0.60 * t  

        self.W_ATTN = float(np.clip(attn, 0.02, 0.65))
        self.W_HEUR = 1.0 - self.W_ATTN
        
        # 修复 Bug 1：大幅降低温度，让 Softmax 真正偏向高分节点，而不是随机逛街
        self.tau = float(np.clip(0.20 - 0.18 * min(1.0, e / 100.0), 0.02, 0.20))

    @staticmethod
    def _softmax_vec(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-9)

    @staticmethod
    def _cpu_free(node_obj) -> float:
        cap = float(getattr(node_obj, "cpu_capacity", 0.0))
        used = float(getattr(node_obj, "cpu_used", 0.0))
        return max(0.0, cap - used)

    def _proc_delay(self, used: float, demand: float, cap: float) -> float:
        util_after = float((used + demand) / (cap + 1e-9))
        util_after = float(np.clip(util_after, 0.0, 1.5))
        return float(self.proc_delay_alpha * (util_after ** self.proc_delay_pow))

    def _record_pool_peakedness(self, pool_alpha):
        alpha = np.asarray(pool_alpha, dtype=np.float32).reshape(-1)
        s = float(alpha.sum())
        if s <= 1e-9:
            alpha = np.ones_like(alpha, dtype=np.float32) / max(1, len(alpha))
        else:
            alpha = alpha / (s + 1e-9)

        N = max(1, len(alpha))
        ent = -float(np.sum(alpha * np.log(alpha + 1e-9)))
        ent_norm = ent / float(np.log(N + 1e-9))
        peakedness = float(np.clip(1.0 - ent_norm, 0.0, 1.0))

        self._peakedness_sum += peakedness
        self._peakedness_cnt += 1

    def update_from_reward(self, advantage: float):
        if not self._traj:
            return

        adv = float(advantage)
        Wo_local = self._scorer.Wo_local
        grad = np.zeros_like(Wo_local, dtype=np.float32)

        # 吸收 W_A 和 tau 的缩放因子，保持梯度更新平稳
        coef = -1.0
        for z_a, Ez in self._traj:
            g = coef * (z_a - Ez)
            if g.shape != grad.shape:
                g = g.reshape(grad.shape)
            grad += adv * g

        grad /= max(1, len(self._traj))
        grad = np.clip(grad, -3.0, 3.0)

        Wo_new = (1.0 - self.pg_wd) * Wo_local + self.pg_lr * grad
        Wo_new = np.clip(Wo_new, -2.0, 2.0)

        self._scorer.Wo_local = Wo_new
        self._traj.clear()

    def place_segment(self, vnfs, entry_node=None, exit_node=None):
        if not vnfs:
            return {}, 0.0

        g = self.domain.graph
        nodes = self.domain.nodes
        cur = entry_node or self.domain.ingress_node
        exit_node = exit_node or self.domain.egress_node

        total_delay = 0.0
        seg_map = {}
        self._traj.clear()

        try:
            all_nids = list(nodes.keys())
            sample = all_nids if len(all_nids) <= 8 else random.sample(all_nids, 8)
            max_dist = 1.0
            for s in sample:
                sp = nx.single_source_dijkstra_path_length(g, s, weight="delay")
                if sp:
                    max_dist = max(max_dist, max(sp.values()))
            graph_delay_scale = float(max_dist) + 1e-6
        except Exception:
            graph_delay_scale = 50.0

        degs = dict(g.degree())
        deg_max = float(max(degs.values()) if degs else 1.0) + 1e-6

        SLACK_EPS = 0.04
        SLACK_LAMBDA = 0.20
        ENTROPY_MIX = 0.01

        for vnf in vnfs:
            demand = float(getattr(vnf, "cpu_demand", 0.0))

            cand = []
            for nid, node in nodes.items():
                if self._cpu_free(node) < demand:
                    continue
                try:
                    d_cur = float(nx.shortest_path_length(g, cur, nid, weight="delay"))
                    d_exit = float(nx.shortest_path_length(g, nid, exit_node, weight="delay"))
                except nx.NetworkXNoPath:
                    continue
                cand.append((nid, d_cur, d_exit))

            if not cand:
                raise RuntimeError("no feasible node")

            nid_list: List[str] = []
            heur_costs: List[float] = []
            X: List[List[float]] = []
            proc_delays: List[float] = []

            for nid, d_cur, d_exit in cand:
                node = nodes[nid]
                cap = float(getattr(node, "cpu_capacity", 0.0)) + 1e-6
                used = float(getattr(node, "cpu_used", 0.0))
                free = self._cpu_free(node)

                pdelay = self._proc_delay(used=used, demand=demand, cap=cap)
                proc_delays.append(float(pdelay))

                dist_from_cur_norm = float(d_cur) / graph_delay_scale
                dist_to_exit_norm  = float(d_exit) / graph_delay_scale

                cpu_free_ratio = free / cap
                cpu_util_now = used / cap
                degree_norm = float(degs.get(nid, 0)) / deg_max
                vnf_demand_ratio = demand / cap
                cpu_slack_ratio = (free - demand) / cap

                X.append([
                    cpu_free_ratio, cpu_util_now, degree_norm,
                    dist_from_cur_norm, dist_to_exit_norm,
                    vnf_demand_ratio, cpu_slack_ratio,
                ])

                util_after = float((used + demand) / cap)
                util_after = float(np.clip(util_after, 0.0, 1.5))

                eff_step = float(d_cur) + float(pdelay)
                eff_to_exit = float(d_exit)

                slack = float(cpu_slack_ratio)
                slack_bar = 0.0
                if slack < SLACK_EPS:
                    slack_bar = float((SLACK_EPS - slack) / max(SLACK_EPS, 1e-6))
                slack_bar *= SLACK_LAMBDA

                delay_term = self.delay_pref * (0.60 * eff_step + 0.40 * eff_to_exit)
                util_term  = self.util_pref  * util_after
                heur = 0.90 * delay_term + 0.10 * (util_term ** 1.5)+ slack_bar

                nid_list.append(nid)
                heur_costs.append(float(heur))

            heur_costs = np.asarray(heur_costs, dtype=np.float32)
            X = np.asarray(X, dtype=np.float32)

            H, emb, pool_alpha = self._scorer.forward(X)
            self._record_pool_peakedness(pool_alpha)

            Z = self._scorer.node_z(H, emb)
            attn_costs = self._scorer.node_costs(Z).astype(np.float32)

            # 修复 Bug 2：废除动态破坏梯度的 _standardize_to，直接做静态拉伸
            attn_costs = attn_costs * 5.0
            scores = self.W_HEUR * heur_costs + self.W_ATTN * attn_costs

            is_eval = bool(getattr(self.domain, "_attn_eval_mode", False))
            if is_eval:
                idx = int(np.argmin(scores))
                probs = np.zeros_like(scores, dtype=np.float32)
                probs[idx] = 1.0
            else:
                probs = self._softmax_vec(-scores / max(self.tau, 1e-6))
                if ENTROPY_MIX > 0:
                    uni = np.ones_like(probs, dtype=np.float32) / float(len(probs))
                    probs = (1.0 - ENTROPY_MIX) * probs + ENTROPY_MIX * uni
                    probs = probs / (probs.sum() + 1e-9)
                idx = int(np.random.choice(len(scores), p=probs))

            chosen = nid_list[idx]

            Ez = (probs[:, None] * Z).sum(axis=0)
            self._traj.append((Z[idx], Ez))

            nodes[chosen].cpu_used += demand

            link_d = float(nx.shortest_path_length(g, cur, chosen, weight="delay"))
            total_delay += link_d + float(proc_delays[idx])

            seg_map[vnf.name] = chosen
            cur = chosen

        total_delay += float(nx.shortest_path_length(g, cur, exit_node, weight="delay"))
        self.domain.last_delay = float(total_delay)
        return seg_map, float(total_delay)


# =========================================================
# GlobalOrchestrator (inter-domain path + federated)
# =========================================================
class GlobalOrchestrator:
    def __init__(
        self,
        domains: Dict[str, "Domain"],
        domain_graph: nx.DiGraph,
        k_paths: int = 3,
        agent_lr: float = 0.05,
        agent_init_eps: float = 0.8,
        fed_mu: float = 0.5,
        fed_mu_agent: Optional[float] = None,
        fed_mu_wo: Optional[float] = None,
        fed_warmup_episodes: int = 10,
        fed_perf_temp: float = 0.8,
        fed_uniform_mix: float = 0.10,
    ):
        self.domains = domains
        self.domain_graph = domain_graph
        self.k_paths = int(k_paths)
        self.agent_lr = float(agent_lr)
        self.agent_init_eps = float(agent_init_eps)

        base_mu = float(np.clip(fed_mu, 0.0, 1.0))
        if fed_mu_agent is None:
            fed_mu_agent = min(0.55, base_mu)         
        if fed_mu_wo is None:
            fed_mu_wo = min(0.35, 0.40 * base_mu)     

        self.fed_mu_agent = float(np.clip(fed_mu_agent, 0.0, 1.0))
        self.fed_mu_wo = float(np.clip(fed_mu_wo, 0.0, 1.0))
        self.fed_warmup_episodes = int(max(0, fed_warmup_episodes))

        self.fed_perf_temp = float(max(0.5, fed_perf_temp)) 
        self.fed_uniform_mix = float(np.clip(fed_uniform_mix, 0.0, 0.5))

        self._cur_episode = 0

        self.agents: Dict[str, "DomainRLAgent"] = {
            did: DomainRLAgent(
                domain_id=did,
                num_actions=self.k_paths,
                lr=self.agent_lr,
                epsilon=self.agent_init_eps,
            )
            for did in domains.keys()
        }
        self.domain_orchs: Dict[str, "DomainOrchestrator"] = {
            did: DomainOrchestrator(dom) for did, dom in domains.items()
        }

        self.is_eval = False
        self._train_eps_cache = float(self.agent_init_eps)

        self.reward_baseline = 0.0
        self.baseline_beta = 0.90

        self._dom_baseline: Dict[str, float] = {did: 0.0 for did in domains.keys()}
        self._dom_baseline_beta = 0.90

        self._dom_perf_ema: Dict[str, float] = {did: 0.0 for did in domains.keys()}
        self._perf_beta = 0.90

    def set_mode(self, is_eval: bool, eval_action_samples: int = 1):
        self.is_eval = bool(is_eval)
        for did, dom in self.domains.items():
            dom._attn_eval_mode = self.is_eval
            orch = self.domain_orchs.get(did, None)
            if orch is not None:
                orch.set_action_samples(eval_action_samples if self.is_eval else 1)

        if self.is_eval:
            for agent in self.agents.values():
                agent.epsilon = 0.0
        else:
            for agent in self.agents.values():
                agent.epsilon = float(self._train_eps_cache)

    def update_epsilon(
        self, episode: int, start_eps: float = None, min_eps: float = 0.05,
        warmup_eps_episodes: int = 20, decay_per_episode: float = 0.01,
    ):
        self._cur_episode = int(max(0, episode))
        if start_eps is None:
            start_eps = self.agent_init_eps

        if episode < warmup_eps_episodes:
            eps = float(start_eps)
        else:
            eps = float(start_eps) - float(decay_per_episode) * float(episode - warmup_eps_episodes)
            eps = max(float(min_eps), eps)

        self._train_eps_cache = float(eps)
        if not self.is_eval:
            for agent in self.agents.values():
                agent.epsilon = float(eps)
            for orch in self.domain_orchs.values():
                orch.set_training_progress(episode)

    def k_shortest_domain_paths(self, src: str, dst: str, k: int) -> List[List[str]]:
        try:
            gen = nx.shortest_simple_paths(self.domain_graph, src, dst, weight="delay")
        except nx.NetworkXNoPath:
            return []
        paths = []
        for path in gen:
            paths.append(path)
            if len(paths) >= k:
                break
        return paths

    def assign_vnfs_to_domains(self, sfc: "SFC", domain_path: List[str]) -> Dict[str, str]:
        vnfs = sfc.vnfs
        n_vnfs = len(vnfs)
        cpu_list = [max(self.domains[did].total_cpu_free(), 0.01) for did in domain_path]
        cpu_sum = sum(cpu_list)
        raw_counts = [int(n_vnfs * c / cpu_sum) for c in cpu_list]
        assigned = sum(raw_counts)

        i = 0
        while assigned < n_vnfs:
            raw_counts[i % len(raw_counts)] += 1
            assigned += 1
            i += 1

        vnf_to_domain: Dict[str, str] = {}
        cur = 0
        for did, cnt in zip(domain_path, raw_counts):
            for _ in range(cnt):
                if cur >= n_vnfs:
                    break
                vnf_to_domain[vnfs[cur].name] = did
                cur += 1

        for j in range(cur, n_vnfs):
            vnf_to_domain[vnfs[j].name] = domain_path[-1]
        return vnf_to_domain

    def _perf_weights(self, dids: List[str]) -> np.ndarray:
        perfs = np.array([max(self._dom_perf_ema.get(did, 0.0), -5.0) for did in dids], dtype=np.float32)
        temp = float(self.fed_perf_temp)
        logits = (perfs - perfs.max()) / temp
        w = np.exp(logits).astype(np.float32)
        w = w / (w.sum() + 1e-9)

        um = float(self.fed_uniform_mix)
        if um > 0:
            uni = np.ones_like(w, dtype=np.float32) / float(len(w))
            w = (1.0 - um) * w + um * uni
            w = w / (w.sum() + 1e-9)
        return w

    def federated_aggregation(self):
        if self.is_eval:
            return
        if int(self._cur_episode) < int(self.fed_warmup_episodes):
            return

        self._federated_aggregation_agents(mu=self.fed_mu_agent)
        self._domain_federated_aggregation_Wo(mu=self.fed_mu_wo)

    def _federated_aggregation_agents(self, mu: float):
        if not self.agents:
            return
        mu = float(np.clip(mu, 0.0, 1.0))
        if mu <= 0.0:
            return

        dids = list(self.agents.keys())
        qs = np.stack([self.agents[did].get_params() for did in dids], axis=0).astype(np.float32)
        w = self._perf_weights(dids)
        q_global = np.zeros_like(qs[0], dtype=np.float32)
        for wi, qi in zip(w, qs):
            q_global += float(wi) * qi

        for did in dids:
            q_local = self.agents[did].get_params().astype(np.float32)
            self.agents[did].set_params((1.0 - mu) * q_local + mu * q_global)

    def _domain_federated_aggregation_Wo(self, mu: float):
        orchs = list(self.domain_orchs.values())
        if not orchs:
            return
        mu = float(np.clip(mu, 0.0, 1.0))
        if mu <= 0.0:
            return

        dids = [orch.domain.id for orch in orchs]
        Ws = [orch._scorer.effective_Wo().astype(np.float32) for orch in orchs]
        w = self._perf_weights(dids)
        W_global = np.zeros_like(Ws[0], dtype=np.float32)
        for wi, Wi in zip(w, Ws):
            W_global += float(wi) * Wi

        for orch in orchs:
            orch._scorer.Wo_global = W_global.copy()
            orch._scorer.Wo_local *= (1.0 - mu)

    # -------------------------
    # deploy_sfc
    # -------------------------
    def deploy_sfc(self, sfc: "SFC") -> "PlacementResult":
        paths = self.k_shortest_domain_paths(sfc.src_domain, sfc.dst_domain, self.k_paths)
        if not paths:
            raise RuntimeError(f"No domain path from {sfc.src_domain} to {sfc.dst_domain}")

        # 为 Bandit 增加基础的准入过滤，避免盲目冲击已被挤爆的路径
        valid_actions = []
        for i, p in enumerate(paths):
            path_is_full = False
            for did in p:
                if self.domains[did].total_cpu_free() < 2.0:
                    path_is_full = True
                    break
            if not path_is_full:
                valid_actions.append(i)
                
        if not valid_actions:
            valid_actions = list(range(len(paths)))

        agent = self.agents[sfc.src_domain]
        action = agent.select_action(valid_actions)
        domain_path = paths[action]

        vnf_to_domain = self.assign_vnfs_to_domains(sfc, domain_path)
        domain_to_vnfs: Dict[str, List["VNF"]] = {did: [] for did in domain_path}
        for v in sfc.vnfs:
            domain_to_vnfs[vnf_to_domain[v.name]].append(v)

        vnf_to_node: Dict[str, Tuple[str, str]] = {}
        total_delay = 0.0

        backup = {
            did: {nid: n.cpu_used for nid, n in dom.nodes.items()}
            for did, dom in self.domains.items()
        }

        used_nodes = set()
        orch_used: Dict[str, "DomainOrchestrator"] = {}

        # 严厉且明确的失败惩罚
        FAILURE_REWARD_GLOBAL = -15.0
        FAILURE_REWARD_LOCAL = -15.0

        seg_delay_by_dom: Dict[str, float] = {}
        seg_nodes_by_dom: Dict[str, List[str]] = {}

        try:
            name_to_idx = {v.name: i for i, v in enumerate(sfc.vnfs)}

            for did in domain_path:
                orch = self.domain_orchs[did]
                orch_used[did] = orch
                dom = self.domains[did]

                seg_vnfs = sorted(domain_to_vnfs[did], key=lambda v: name_to_idx.get(v.name, 10**9))
                seg_map, seg_delay = orch.place_segment(
                    seg_vnfs,
                    entry_node=dom.ingress_node,
                    exit_node=dom.egress_node,
                )

                seg_delay_by_dom[did] = float(seg_delay)
                seg_nodes_by_dom[did] = list(seg_map.values())

                total_delay += float(seg_delay)
                for vname, nid in seg_map.items():
                    vnf_to_node[vname] = (did, nid)
                    used_nodes.add((did, nid))

            inter_delay = 0.0
            for u, v in zip(domain_path[:-1], domain_path[1:]):
                inter_delay += float(self.domain_graph[u][v].get("delay", 0.0))
            total_delay += inter_delay

            max_util = 0.0
            if used_nodes:
                max_util = float(
                    max(
                        float(self.domains[did].nodes[nid].cpu_used) /
                        (float(self.domains[did].nodes[nid].cpu_capacity) + 1e-9)
                        for did, nid in used_nodes
                    )
                )

            # 让常规延迟(如 80ms)正好为 1.0 的基准。延迟长了会变成负 Reward
            FIXED_DELAY_SCALE = 120.0
            cost = 1.5 * (float(total_delay) / FIXED_DELAY_SCALE) + 1.0 * float(max_util)
            reward_global = float(6.0 - cost)

            if not self.is_eval:
                base_g = float(self.reward_baseline)
                self.reward_baseline = self.baseline_beta * base_g + (1.0 - self.baseline_beta) * reward_global
                
                # 全局 Bandit 直接吃真实打分
                agent.update(reward_global)

                for did, orch in orch_used.items():
                    d_seg = float(seg_delay_by_dom.get(did, 0.0))
                    
                    # 修复：补回 maxu_d 的计算
                    dom_nodes = seg_nodes_by_dom.get(did, [])
                    if dom_nodes:
                        maxu_d = float(
                            max(
                                float(self.domains[did].nodes[nid].cpu_used) /
                                (float(self.domains[did].nodes[nid].cpu_capacity) + 1e-9)
                                for nid in dom_nodes
                            )
                        )
                    else:
                        maxu_d = 0.0
                    
                    FIXED_DELAY_LOCAL = 30.0
                    cost_d = 2.0 * (float(d_seg) / FIXED_DELAY_LOCAL) + 1.5 * float(maxu_d)
                    reward_d = float(5.0 - cost_d)

                    b_d = float(self._dom_baseline[did])
                    adv_d = float(np.clip(reward_d - b_d, -10.0, 10.0))
                    self._dom_baseline[did] = self._dom_baseline_beta * b_d + (1.0 - self._dom_baseline_beta) * reward_d

                    orch.update_from_reward(adv_d)

                    prev = float(self._dom_perf_ema.get(did, 0.0))
                    self._dom_perf_ema[did] = self._perf_beta * prev + (1.0 - self._perf_beta) * float(reward_d)

        except RuntimeError:
            for did, nodes_state in backup.items():
                for nid, used in nodes_state.items():
                    self.domains[did].nodes[nid].cpu_used = used

            if not self.is_eval:
                base_g = float(self.reward_baseline)
                self.reward_baseline = self.baseline_beta * base_g + (1.0 - self.baseline_beta) * float(FAILURE_REWARD_GLOBAL)
                
                agent.update(float(FAILURE_REWARD_GLOBAL))

                for did, orch in orch_used.items():
                    b_d = float(self._dom_baseline[did])
                    adv_d = float(np.clip(float(FAILURE_REWARD_LOCAL) - b_d, -10.0, 10.0))
                    self._dom_baseline[did] = self._dom_baseline_beta * b_d + (1.0 - self._dom_baseline_beta) * float(FAILURE_REWARD_LOCAL)
                    orch.update_from_reward(adv_d)

                    prev = float(self._dom_perf_ema.get(did, 0.0))
                    self._dom_perf_ema[did] = self._perf_beta * prev + (1.0 - self._perf_beta) * float(FAILURE_REWARD_LOCAL)

            raise

        return PlacementResult(
            sfc_id=sfc.id,
            domain_path=domain_path,
            vnf_to_domain=vnf_to_domain,
            vnf_to_node=vnf_to_node,
            total_delay=float(total_delay),
        )