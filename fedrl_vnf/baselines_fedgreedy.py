import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx

from .models import Domain, VNF, SFC, PlacementResult
from .baselines import (
    _k_shortest_paths,
    _backup_cpu_used,
    _restore_cpu_used,
    _assign_vnfs_to_domains_cpu_proportional,
    _sort_segment_vnfs_as_sfc_order,
)

class FederatedGreedyOrchestrator:
    """
    Federated Greedy baseline（公平版）：
    - domain_path：只选最短路 paths[0]（不 trial 多路径）
    - 域内：greedy 选节点，cost = w_delay*(link + proc + w_exit*d_exit) + w_util*util_after
    - train：本地轻量更新 w_delay/w_util，并定期 FedAvg
    - eval：冻结，不更新不聚合
    """

    def __init__(
        self,
        domains: Dict[str, "Domain"],
        domain_graph: nx.DiGraph,
        k_paths: int = 3,
        fedavg_interval: int = 3,
        init_w_delay: float = 0.7,
        init_w_util: float = 0.3,
        lr: float = 0.05,
        target_max_util: float = 0.75,
        target_delay: float = 20.0,
        # ✅ 与 RL 同口径的 proc delay
        proc_delay_alpha: float = 18.0,
        proc_delay_pow: float = 2.0,
        use_proc_delay: bool = True,
        # ✅ 出口引导（避免贪近路）
        w_exit: float = 0.35,
        # ✅ util 硬阈值（可调）
        util_hard_limit: float = 0.90,
    ):
        self.domains = domains
        self.domain_graph = domain_graph
        self.k_paths = int(k_paths)

        self.fedavg_interval = int(fedavg_interval)
        self.lr = float(lr)
        self.target_max_util = float(target_max_util)
        self.target_delay = float(target_delay)

        self.proc_delay_alpha = float(proc_delay_alpha)
        self.proc_delay_pow = float(proc_delay_pow)
        self.use_proc_delay = bool(use_proc_delay)

        self.w_exit = float(w_exit)
        self.util_hard_limit = float(util_hard_limit)

        self.local_w: Dict[str, np.ndarray] = {
            did: np.array([init_w_delay, init_w_util], dtype=np.float64)
            for did in domains.keys()
        }

        self._step = 0
        self.is_eval = False

    def set_mode(self, is_eval: bool):
        self.is_eval = bool(is_eval)

    def federated_aggregation(self):
        """FedAvg：把各域 local_w 平均成一个全局，再下发回去。"""
        if self.is_eval or not self.local_w:
            return
        W = np.stack(list(self.local_w.values()), axis=0)  # [D,2]
        avg = np.mean(W, axis=0)
        for did in self.local_w:
            self.local_w[did] = avg.copy()

    @staticmethod
    def _cpu_free(node) -> float:
        cap = float(getattr(node, "cpu_capacity", 0.0))
        used = float(getattr(node, "cpu_used", 0.0))
        return max(0.0, cap - used)

    def _proc_delay(self, used: float, demand: float, cap: float) -> float:
        util_after = float((used + demand) / (cap + 1e-9))
        util_after = float(np.clip(util_after, 0.0, 1.5))
        return float(self.proc_delay_alpha * (util_after ** self.proc_delay_pow))

    # ---------- 域内放置（贪心） ----------
    def _place_segment_greedy(
        self,
        domain: "Domain",
        vnfs: List["VNF"],
        entry_node: Optional[str],
        exit_node: Optional[str],
        w_delay: float,
        w_util: float,
    ) -> Tuple[Dict[str, str], float, List[str]]:
        if not vnfs:
            return {}, 0.0, []

        g = domain.graph
        nodes = domain.nodes
        cur = entry_node or domain.ingress_node
        egress = exit_node or domain.egress_node

        seg_map: Dict[str, str] = {}
        total = 0.0
        used_nodes: List[str] = []

        for v in vnfs:
            demand = float(getattr(v, "cpu_demand", 0.0))

            best_nid = None
            best_score = float("inf")
            best_step_delay = float("inf")

            for nid, node in nodes.items():
                if self._cpu_free(node) < demand:
                    continue

                cap = float(getattr(node, "cpu_capacity", 0.0)) + 1e-9
                used_now = float(getattr(node, "cpu_used", 0.0))
                util_after = float((used_now + demand) / cap)

                # ✅ 可控的硬阈值：太满直接跳过
                if util_after > self.util_hard_limit:
                    continue

                try:
                    d_cur = float(nx.shortest_path_length(g, cur, nid, weight="delay"))
                    d_exit = float(nx.shortest_path_length(g, nid, egress, weight="delay"))
                except nx.NetworkXNoPath:
                    continue

                pdelay = 0.0
                if self.use_proc_delay:
                    pdelay = self._proc_delay(used=used_now, demand=demand, cap=cap)

                # ✅ 与 RL 同口径：link + proc，同时加入出口引导
                step_delay = d_cur + pdelay
                score = float(w_delay) * float(step_delay + self.w_exit * d_exit) + float(w_util) * float(util_after)

                if score < best_score:
                    best_score = score
                    best_nid = nid
                    best_step_delay = step_delay

            if best_nid is None:
                raise RuntimeError(f"[FedGreedy {domain.id}] no feasible node for {v.name}")

            nodes[best_nid].cpu_used += demand
            total += float(best_step_delay)
            seg_map[v.name] = best_nid
            used_nodes.append(best_nid)
            cur = best_nid

        # exit hop（不加proc，跟你 RL 版本一致）
        try:
            exit_d = float(nx.shortest_path_length(g, cur, egress, weight="delay"))
        except nx.NetworkXNoPath:
            raise RuntimeError(f"[FedGreedy {domain.id}] no path to egress {egress}")

        total += exit_d
        return seg_map, float(total), used_nodes

    # ---------- 本地更新（轻量规则） ----------
    def _local_update(self, domain_path: List[str], total_delay: float, used_nodes: List[Tuple[str, str]], ok: bool):
        if self.is_eval:
            return

        if used_nodes:
            max_util = float(
                max(
                    float(getattr(self.domains[did].nodes[nid], "cpu_used", 0.0))
                    / (float(getattr(self.domains[did].nodes[nid], "cpu_capacity", 0.0)) + 1e-9)
                    for did, nid in used_nodes
                )
            )
        else:
            max_util = 0.0

        for did in set(domain_path):
            w = self.local_w[did].copy()
            w_delay, w_util = float(w[0]), float(w[1])

            if not ok:
                w_util += self.lr * 1.0
            else:
                if max_util > self.target_max_util:
                    w_util += self.lr * (max_util - self.target_max_util)
                if total_delay > self.target_delay:
                    w_delay += self.lr * ((total_delay - self.target_delay) / max(self.target_delay, 1e-9))

            w_delay = float(np.clip(w_delay, 0.05, 0.95))
            w_util = float(np.clip(w_util, 0.05, 0.95))
            s = w_delay + w_util
            w_delay, w_util = w_delay / s, w_util / s
            self.local_w[did] = np.array([w_delay, w_util], dtype=np.float64)

        self._step += 1
        if self.fedavg_interval > 0 and (self._step % self.fedavg_interval == 0):
            self.federated_aggregation()

    # ---------- 对外接口 ----------
    def deploy_sfc(self, sfc: "SFC") -> "PlacementResult":
        sfc = copy.deepcopy(sfc)

        paths = _k_shortest_paths(self.domain_graph, sfc.src_domain, sfc.dst_domain, self.k_paths)
        if not paths:
            raise RuntimeError(f"[FedGreedy] No domain path {sfc.src_domain}->{sfc.dst_domain}")

        # ✅ 真正贪心 baseline：只选最短路径
        domain_path = list(paths[0])

        backup = _backup_cpu_used(self.domains)

        try:
            vnf_to_domain = _assign_vnfs_to_domains_cpu_proportional(self.domains, sfc, domain_path)
            domain_to_vnfs: Dict[str, List["VNF"]] = {did: [] for did in domain_path}
            for v in sfc.vnfs:
                domain_to_vnfs[vnf_to_domain[v.name]].append(v)

            vnf_to_node: Dict[str, Tuple[str, str]] = {}
            used_nodes: List[Tuple[str, str]] = []
            total_delay = 0.0

            for did in domain_path:
                dom = self.domains[did]
                seg_vnfs = _sort_segment_vnfs_as_sfc_order(sfc, domain_to_vnfs[did])

                w_delay, w_util = map(float, self.local_w[did])
                seg_map, seg_delay, used_local = self._place_segment_greedy(
                    dom, seg_vnfs, dom.ingress_node, dom.egress_node, w_delay, w_util
                )

                total_delay += float(seg_delay)
                for vname, nid in seg_map.items():
                    vnf_to_node[vname] = (did, nid)
                for nid in used_local:
                    used_nodes.append((did, nid))

            # ✅ inter-domain delay：不要只给 FedGreedy 加噪声（否则不公平也不稳定）
            inter = 0.0
            for u, v in zip(domain_path[:-1], domain_path[1:]):
                inter += float(self.domain_graph[u][v].get("delay", 0.0))
            total_delay += float(inter)

            self._local_update(domain_path, float(total_delay), used_nodes, ok=True)

            return PlacementResult(
                sfc_id=sfc.id,
                domain_path=domain_path,
                vnf_to_domain=vnf_to_domain,
                vnf_to_node=vnf_to_node,
                total_delay=float(total_delay),
            )

        except RuntimeError:
            _restore_cpu_used(self.domains, backup)
            self._local_update(domain_path, 1e9, [], ok=False)
            raise

