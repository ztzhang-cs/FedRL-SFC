# fedrl_vnf/baselines.py
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx

from .models import Domain, VNF, SFC, PlacementResult


@dataclass
class _DeployTrialResult:
    ok: bool
    total_delay: float
    vnf_to_domain: Dict[str, str]
    vnf_to_node: Dict[str, Tuple[str, str]]
    used_nodes: List[Tuple[str, str]]  # (did, nid)
    domain_path: List[str]


def _backup_cpu_used(domains: Dict[str, Domain]) -> Dict[str, Dict[str, float]]:
    return {
        did: {nid: node.cpu_used for nid, node in dom.nodes.items()}
        for did, dom in domains.items()
    }


def _restore_cpu_used(domains: Dict[str, Domain], backup: Dict[str, Dict[str, float]]) -> None:
    for did, state in backup.items():
        dom = domains[did]
        for nid, used in state.items():
            dom.nodes[nid].cpu_used = used


def _k_shortest_paths(domain_graph: nx.DiGraph, src: str, dst: str, k: int) -> List[List[str]]:
    try:
        gen = nx.shortest_simple_paths(domain_graph, src, dst, weight="delay")
    except nx.NetworkXNoPath:
        return []
    paths: List[List[str]] = []
    for p in gen:
        paths.append(p)
        if len(paths) >= k:
            break
    return paths


def _assign_vnfs_to_domains_cpu_proportional(
    domains: Dict[str, Domain],
    sfc: SFC,
    domain_path: List[str],
) -> Dict[str, str]:
    # 复用你现在 GlobalOrchestrator 的“按 CPU 余量比例分配 VNF 到域”的逻辑
    vnfs = sfc.vnfs
    n_vnfs = len(vnfs)

    cpu_list = [max(domains[did].total_cpu_free(), 0.01) for did in domain_path]
    cpu_sum = float(sum(cpu_list))
    raw_counts = [int(n_vnfs * c / cpu_sum) for c in cpu_list]

    assigned = sum(raw_counts)
    idx = 0
    while assigned < n_vnfs:
        raw_counts[idx % len(raw_counts)] += 1
        assigned += 1
        idx += 1

    vnf_to_domain: Dict[str, str] = {}
    cur = 0
    for did, cnt in zip(domain_path, raw_counts):
        for _ in range(cnt):
            if cur >= n_vnfs:
                break
            vnf_to_domain[vnfs[cur].name] = did
            cur += 1
    for i in range(cur, n_vnfs):
        vnf_to_domain[vnfs[i].name] = domain_path[-1]
    return vnf_to_domain


def _sort_segment_vnfs_as_sfc_order(sfc: SFC, segment_vnfs: List[VNF]) -> List[VNF]:
    name_to_idx = {v.name: i for i, v in enumerate(sfc.vnfs)}
    return sorted(segment_vnfs, key=lambda v: name_to_idx[v.name])


class DFSCOrchestrator:
    """
    DFSC-style 分布式编排 baseline（可跑、接口一致）：
    ...
    """

    def __init__(
        self,
        domains: Dict[str, Domain],
        domain_graph: nx.DiGraph,
        k_paths: int = 3,
        w_path_delay: float = 0.7,
        w_util: float = 0.3,
        load_penalty_scale: float = 10.0,
        path_delay_scale: float = 5.0,
    ):
        self.domains = domains
        self.domain_graph = domain_graph
        self.k_paths = k_paths

        self.w_path_delay = w_path_delay
        self.w_util = w_util
        self.load_penalty_scale = load_penalty_scale
        self.path_delay_scale = path_delay_scale

        # NEW: for unified interface (train/eval), DFSC itself is deterministic
        self.is_eval = False

    # NEW: unified interface
    def set_mode(self, is_eval: bool):
        self.is_eval = bool(is_eval)

    def federated_aggregation(self):
        if self.is_eval:
            return
        vals = list(self.pred_util.values())
        if not vals:
            return
        avg = float(np.mean(vals))
        for did in self.pred_util:
            self.pred_util[did] = avg

    def _place_segment_cost_aware(
        self,
        domain: Domain,
        vnfs: List[VNF],
        entry_node: Optional[str],
        exit_node: Optional[str],
    ) -> Tuple[Dict[str, str], float, List[str]]:
        """
        域内 cost-aware greedy：
        - 候选节点：cpu_free >= demand
        - score = w_path_delay*(scaled shortest path delay) + w_util*(scaled util after)
        返回：vnf->node, segment_delay, used_node_ids_in_this_domain
        """
        if not vnfs:
            return {}, 0.0, []

        g = domain.graph
        nodes = domain.nodes
        cur = entry_node or domain.ingress_node
        egress = exit_node or domain.egress_node

        seg_map: Dict[str, str] = {}
        total = 0.0
        used: List[str] = []

        for v in vnfs:
            cands = [nid for nid, n in nodes.items() if n.cpu_free >= v.cpu_demand]
            if not cands:
                raise RuntimeError(f"[DFSC {domain.id}] no resource for {v.name}")

            best_nid = None
            best_score = float("inf")
            best_path = float("inf")

            for nid in cands:
                node = nodes[nid]
                try:
                    pdelay = nx.shortest_path_length(g, cur, nid, weight="delay")
                except nx.NetworkXNoPath:
                    continue

                util_after = (node.cpu_used + v.cpu_demand) / max(node.cpu_capacity, 1e-9)
                score = (
                    self.w_path_delay * (self.path_delay_scale * pdelay)
                    + self.w_util * (self.load_penalty_scale * util_after)
                )
                if score < best_score:
                    best_score = score
                    best_nid = nid
                    best_path = pdelay

            if best_nid is None:
                raise RuntimeError(f"[DFSC {domain.id}] no path for {v.name}")

            nodes[best_nid].cpu_used += v.cpu_demand
            total += best_path
            seg_map[v.name] = best_nid
            used.append(best_nid)
            cur = best_nid

        try:
            exit_delay = nx.shortest_path_length(g, cur, egress, weight="delay")
        except nx.NetworkXNoPath:
            raise RuntimeError(f"[DFSC {domain.id}] no path to egress {egress}")

        total += exit_delay
        return seg_map, total, used

    def deploy_sfc(self, sfc: SFC) -> PlacementResult:
        sfc = copy.deepcopy(sfc)
        paths = _k_shortest_paths(self.domain_graph, sfc.src_domain, sfc.dst_domain, self.k_paths)
        if not paths:
            raise RuntimeError(f"[DFSC] No domain path {sfc.src_domain}->{sfc.dst_domain}")

        # 保存“本次请求开始时”的资源状态
        base_backup = _backup_cpu_used(self.domains)

        best_cost = float("inf")
        best_path: Optional[List[str]] = None
        best_vnf_to_domain: Optional[Dict[str, str]] = None

        # ----------------------------
        # 1) search / evaluate：每条候选 path 都从同一 base_backup 开始试
        # ----------------------------
        for domain_path in paths:
            _restore_cpu_used(self.domains, base_backup)  # 关键：trial 前回到同一初始状态

            try:
                vnf_to_domain = _assign_vnfs_to_domains_cpu_proportional(self.domains, sfc, domain_path)
                domain_to_vnfs: Dict[str, List[VNF]] = {did: [] for did in domain_path}
                for v in sfc.vnfs:
                    domain_to_vnfs[vnf_to_domain[v.name]].append(v)

                vnf_to_node: Dict[str, Tuple[str, str]] = {}
                used_nodes: List[Tuple[str, str]] = []
                total_delay = 0.0

                # 域内部署（会修改 cpu_used，但这是 trial）
                for did in domain_path:
                    dom = self.domains[did]
                    seg_vnfs = _sort_segment_vnfs_as_sfc_order(sfc, domain_to_vnfs[did])

                    seg_map, seg_delay, used_local = self._place_segment_cost_aware(
                        dom, seg_vnfs, dom.ingress_node, dom.egress_node
                    )
                    total_delay += seg_delay

                    for vname, nid in seg_map.items():
                        vnf_to_node[vname] = (did, nid)
                    for nid in used_local:
                        used_nodes.append((did, nid))

                # 域间链路延迟
                inter = 0.0
                for u, v in zip(domain_path[:-1], domain_path[1:]):
                    inter += float(self.domain_graph[u][v].get("delay", 0.0))
                total_delay += inter

                # cost：delay + 热点惩罚（max util）
                if used_nodes:
                    max_util = float(
                        max(
                            self.domains[did].nodes[nid].cpu_used / max(self.domains[did].nodes[nid].cpu_capacity, 1e-9)
                            for did, nid in used_nodes
                        )
                    )
                else:
                    max_util = 0.0

                cost = float(total_delay + 50.0 * max_util)

                if cost < best_cost:
                    best_cost = cost
                    best_path = list(domain_path)
                    best_vnf_to_domain = dict(vnf_to_domain)

            except RuntimeError:
                # trial 失败：继续下一个 path（资源会在下轮 trial 前 restore）
                continue

        if best_path is None or best_vnf_to_domain is None:
            _restore_cpu_used(self.domains, base_backup)
            raise RuntimeError("[DFSC] all candidate paths failed")

        # ----------------------------
        # 2) commit：回到 base_backup，然后只执行 best_path 真正“落地”
        # ----------------------------
        _restore_cpu_used(self.domains, base_backup)

        vnf_to_domain = best_vnf_to_domain
        domain_to_vnfs: Dict[str, List[VNF]] = {did: [] for did in best_path}
        for v in sfc.vnfs:
            domain_to_vnfs[vnf_to_domain[v.name]].append(v)

        vnf_to_node: Dict[str, Tuple[str, str]] = {}
        total_delay = 0.0
        used_nodes: List[Tuple[str, str]] = []

        for did in best_path:
            dom = self.domains[did]
            seg_vnfs = _sort_segment_vnfs_as_sfc_order(sfc, domain_to_vnfs[did])

            seg_map, seg_delay, used_local = self._place_segment_cost_aware(
                dom, seg_vnfs, dom.ingress_node, dom.egress_node
            )
            total_delay += seg_delay
            for vname, nid in seg_map.items():
                vnf_to_node[vname] = (did, nid)
            for nid in used_local:
                used_nodes.append((did, nid))

        inter = 0.0
        for u, v in zip(best_path[:-1], best_path[1:]):
            inter += float(self.domain_graph[u][v].get("delay", 0.0))
        total_delay += inter

        return PlacementResult(
            sfc_id=sfc.id,
            domain_path=best_path,
            vnf_to_domain=vnf_to_domain,
            vnf_to_node=vnf_to_node,
            total_delay=float(total_delay),
        )
class FLPredictOrchestrator:
    """
    “Federated resource usage prediction + placement” baseline（可跑、接口一致）

    论文公平性说明：
    - train 模式：在线更新 EWMA + 定期 FedAvg（模拟“训练/运行期学习”）
    - eval  模式：冻结 predictor（不 EWMA，不 FedAvg），只用固定 pred_util 做 placement
      这样与 RL 的 “policy frozen evaluation” 对齐，避免 eval 期间继续学习导致不公平。
    """

    def __init__(
        self,
        domains: Dict[str, Domain],
        domain_graph: nx.DiGraph,
        k_paths: int = 3,
        ewma_alpha: float = 0.2,
        fedavg_interval: int = 3,
        w_delay: float = 0.7,
        w_pred_util: float = 0.3,
    ):
        self.domains = domains
        self.domain_graph = domain_graph
        self.k_paths = k_paths

        self.ewma_alpha = ewma_alpha
        self.fedavg_interval = fedavg_interval
        self.w_delay = w_delay
        self.w_pred_util = w_pred_util

        self.pred_util: Dict[str, float] = {did: 0.0 for did in domains.keys()}
        self._step = 0

        # NEW: mode switch
        self.is_eval = False

    # NEW: unified interface
    def set_mode(self, is_eval: bool):
        self.is_eval = bool(is_eval)

    def federated_aggregation(self):
        if self.is_eval:
            return
        vals = list(self.pred_util.values())
        if not vals:
            return
        avg = float(np.mean(vals))
        for did in self.pred_util:
            self.pred_util[did] = avg


    def _observe_domain_avg_util(self, dom: Domain) -> float:
        utils = [n.cpu_used / max(n.cpu_capacity, 1e-9) for n in dom.nodes.values()]
        return float(np.mean(utils)) if utils else 0.0

    def _update_predictors(self):
        if self.is_eval:
            return  # NEW: eval 时冻结 predictor

        # 用当前观测更新 EWMA
        for did, dom in self.domains.items():
            obs = self._observe_domain_avg_util(dom)
            old = self.pred_util.get(did, 0.0)
            a = self.ewma_alpha
            self.pred_util[did] = (1.0 - a) * old + a * obs

        self._step += 1
        if self.fedavg_interval > 0 and (self._step % self.fedavg_interval == 0):
            self.federated_aggregation()


    def _place_segment_pred_aware(
        self,
        domain: Domain,
        vnfs: List[VNF],
        entry_node: Optional[str],
        exit_node: Optional[str],
        pred_domain_util: float,
    ) -> Tuple[Dict[str, str], float, List[str]]:
        if not vnfs:
            return {}, 0.0, []

        g = domain.graph
        nodes = domain.nodes
        cur = entry_node or domain.ingress_node
        egress = exit_node or domain.egress_node

        seg_map: Dict[str, str] = {}
        total = 0.0
        used: List[str] = []

        # 预测利用率越高，越惩罚“再往高util节点塞”
        pred_boost = 1.0 + 2.0 * pred_domain_util  # 可调

        for v in vnfs:
            cands = [nid for nid, n in nodes.items() if n.cpu_free >= v.cpu_demand]
            if not cands:
                raise RuntimeError(f"[FLPredict {domain.id}] no resource for {v.name}")

            best_nid = None
            best_score = float("inf")
            best_path = float("inf")

            for nid in cands:
                node = nodes[nid]
                try:
                    pdelay = nx.shortest_path_length(g, cur, nid, weight="delay")
                except nx.NetworkXNoPath:
                    continue

                util_after = (node.cpu_used + v.cpu_demand) / max(node.cpu_capacity, 1e-9)

                # 预测感知：对 util_after 施加 pred_boost
                score = self.w_delay * pdelay + self.w_pred_util * (pred_boost * util_after)

                if score < best_score:
                    best_score = score
                    best_nid = nid
                    best_path = pdelay

            if best_nid is None:
                raise RuntimeError(f"[FLPredict {domain.id}] no path for {v.name}")

            nodes[best_nid].cpu_used += v.cpu_demand
            total += best_path
            seg_map[v.name] = best_nid
            used.append(best_nid)
            cur = best_nid

        try:
            exit_delay = nx.shortest_path_length(g, cur, egress, weight="delay")
        except nx.NetworkXNoPath:
            raise RuntimeError(f"[FLPredict {domain.id}] no path to egress {egress}")
        total += exit_delay

        return seg_map, total, used

    def deploy_sfc(self, sfc: SFC) -> PlacementResult:
        sfc = copy.deepcopy(sfc)
        paths = _k_shortest_paths(self.domain_graph, sfc.src_domain, sfc.dst_domain, self.k_paths)
        if not paths:
            raise RuntimeError(f"[FLPredict] No domain path {sfc.src_domain}->{sfc.dst_domain}")

        base_backup = _backup_cpu_used(self.domains)

        best_cost = float("inf")
        best_path: Optional[List[str]] = None
        best_vnf_to_domain: Optional[Dict[str, str]] = None

        # ----------------------------
        # 1) search / evaluate：每条 path 都从同一 base_backup 起步
        # ----------------------------
        for domain_path in paths:
            _restore_cpu_used(self.domains, base_backup)

            try:
                vnf_to_domain = _assign_vnfs_to_domains_cpu_proportional(self.domains, sfc, domain_path)
                domain_to_vnfs: Dict[str, List[VNF]] = {did: [] for did in domain_path}
                for v in sfc.vnfs:
                    domain_to_vnfs[vnf_to_domain[v.name]].append(v)

                vnf_to_node: Dict[str, Tuple[str, str]] = {}
                used_nodes: List[Tuple[str, str]] = []
                total_delay = 0.0

                for did in domain_path:
                    dom = self.domains[did]
                    seg_vnfs = _sort_segment_vnfs_as_sfc_order(sfc, domain_to_vnfs[did])

                    seg_map, seg_delay, used_local = self._place_segment_pred_aware(
                        dom,
                        seg_vnfs,
                        dom.ingress_node,
                        dom.egress_node,
                        pred_domain_util=float(self.pred_util.get(did, 0.0)),
                    )
                    total_delay += seg_delay
                    for vname, nid in seg_map.items():
                        vnf_to_node[vname] = (did, nid)
                    for nid in used_local:
                        used_nodes.append((did, nid))

                inter = 0.0
                for u, v in zip(domain_path[:-1], domain_path[1:]):
                    inter += float(self.domain_graph[u][v].get("delay", 0.0))
                total_delay += inter

                pred_path_util = float(np.mean([self.pred_util.get(did, 0.0) for did in domain_path])) if domain_path else 0.0
                cost = float(total_delay + 30.0 * pred_path_util)

                if cost < best_cost:
                    best_cost = cost
                    best_path = list(domain_path)
                    best_vnf_to_domain = dict(vnf_to_domain)

            except RuntimeError:
                continue

        if best_path is None or best_vnf_to_domain is None:
            _restore_cpu_used(self.domains, base_backup)
            # NEW: eval 不更新 predictor（避免测试期学习）
            if not self.is_eval:
                self._update_predictors()
            raise RuntimeError("[FLPredict] all candidate paths failed")

        # ----------------------------
        # 2) commit：回到 base_backup，重放 best_path 真正落地
        # ----------------------------
        _restore_cpu_used(self.domains, base_backup)

        vnf_to_domain = best_vnf_to_domain
        domain_to_vnfs: Dict[str, List[VNF]] = {did: [] for did in best_path}
        for v in sfc.vnfs:
            domain_to_vnfs[vnf_to_domain[v.name]].append(v)

        vnf_to_node: Dict[str, Tuple[str, str]] = {}
        total_delay = 0.0

        for did in best_path:
            dom = self.domains[did]
            seg_vnfs = _sort_segment_vnfs_as_sfc_order(sfc, domain_to_vnfs[did])

            seg_map, seg_delay, _ = self._place_segment_pred_aware(
                dom,
                seg_vnfs,
                dom.ingress_node,
                dom.egress_node,
                pred_domain_util=float(self.pred_util.get(did, 0.0)),
            )
            total_delay += seg_delay
            for vname, nid in seg_map.items():
                vnf_to_node[vname] = (did, nid)

        inter = 0.0
        for u, v in zip(best_path[:-1], best_path[1:]):
            inter += float(self.domain_graph[u][v].get("delay", 0.0))
        total_delay += inter

        # NEW: eval 不更新 predictor（避免测试期学习）
        if not self.is_eval:
            self._update_predictors()

        return PlacementResult(
            sfc_id=sfc.id,
            domain_path=best_path,
            vnf_to_domain=vnf_to_domain,
            vnf_to_node=vnf_to_node,
            total_delay=float(total_delay),
        )