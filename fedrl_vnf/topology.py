from typing import Dict, List, Tuple, Optional
import random
import networkx as nx

from .models import Domain, Node, VNF, SFC


# =========================================================
# Processing/queueing delay profile (environment-side)
# =========================================================
def _set_domain_delay_profile(dom: Domain, is_hot: bool):
    """
    给 Domain 挂一些“环境参数”，不破坏你现有接口。
    orchestrator 若读取不到也没关系（用默认值）。
    """
    # 热点域：拥塞造成的处理时延更强
    dom.proc_delay_alpha = 35.0 if is_hot else 8.0
    dom.proc_delay_pow = 3.0 if is_hot else 2.0

    # 热点域：入口更“诱饵”，让贪心更爱走
    dom.ingress_bias = 1.0 if is_hot else 0.0


def build_random_domains(
    num_domains: int,
    min_nodes_per_domain: int = 6,
    max_nodes_per_domain: int = 10,
    cpu_range: Tuple[int, int] = (18, 30),
    link_delay_range: Tuple[float, float] = (2.0, 6.0),
) -> Tuple[Dict[str, Domain], nx.DiGraph]:
    domains: Dict[str, Domain] = {}

    hot_set = set(range(1, num_domains - 1, 3))  # 1,4,7,...

    for d_idx in range(num_domains):
        did = f"D{d_idx}"
        g = nx.DiGraph()
        is_hot = (d_idx in hot_set)

        # 热点域更大，但并不意味着更强：入口小核多、出口少量大核
        if is_hot:
            n_nodes = random.randint(12, 16)
        else:
            n_nodes = random.randint(min_nodes_per_domain, max_nodes_per_domain)

        nodes: Dict[str, Node] = {}
        ids: List[str] = []

        # ---------- CPU 分布 ----------
        for i in range(n_nodes):
            nid = f"{did}_N{i}"

            if is_hot:
                # 入口区：容量小但“看起来能放几个”，容易拥塞
                if i < max(3, n_nodes // 3):
                    cpu_cap = random.randint(6, 8)
                # 出口区：少量大核“救命点”
                elif i > n_nodes - max(2, n_nodes // 5) - 1:
                    cpu_cap = random.randint(22, 28)
                else:
                    cpu_cap = random.randint(9, 13)
            else:
                cpu_cap = random.randint(*cpu_range)

            nodes[nid] = Node(id=nid, cpu_capacity=cpu_cap)
            g.add_node(nid)
            ids.append(nid)

        split = max(4, len(ids) // 2)
        laneA = ids[:split]
        laneB = ids[split:]

        def add_undir(u: str, v: str, delay: float):
            g.add_edge(u, v, delay=delay, bw=100)
            g.add_edge(v, u, delay=delay, bw=100)

        # ---------- laneA：诱饵（前段很快，后段很慢） ----------
        for i in range(len(laneA) - 1):
            if is_hot:
                # 前半段更快，强诱饵；后半段更慢，强惩罚
                d = random.uniform(0.9, 1.6) if i < len(laneA)//2 else random.uniform(7.5, 12.5)
            else:
                d = random.uniform(*link_delay_range)
            add_undir(laneA[i], laneA[i + 1], d)

        # ---------- laneB：更稳（出口友好） ----------
        for i in range(len(laneB) - 1):
            if is_hot:
                d = random.uniform(2.8, 4.4)
            else:
                d = random.uniform(*link_delay_range)
            add_undir(laneB[i], laneB[i + 1], d)

        # ---------- 贵跨道：避免随机捷径冲掉陷阱 ----------
        if laneA and laneB:
            add_undir(laneA[len(laneA)//2], laneB[0], delay=random.uniform(16.0, 24.0))

        # ---------- 额外边 ----------
        if is_hot:
            # 热点域：只允许 lane 内少量捷径
            extra = random.randint(1, 2)
            for _ in range(extra):
                if len(laneA) >= 3:
                    u, v = random.sample(laneA, 2)
                    g.add_edge(u, v, delay=random.uniform(1.1, 2.2), bw=100)
                if len(laneB) >= 3:
                    u, v = random.sample(laneB, 2)
                    g.add_edge(u, v, delay=random.uniform(2.0, 3.4), bw=100)
        else:
            # 非热点域：更通畅稳定
            extra = random.randint(len(ids)//2, len(ids))
            for _ in range(extra):
                u, v = random.sample(ids, 2)
                g.add_edge(u, v, delay=random.uniform(*link_delay_range), bw=100)

        ingress = laneA[0] if laneA else ids[0]
        egress = laneB[-1] if laneB else ids[-1]

        dom = Domain(id=did, graph=g, nodes=nodes, ingress_node=ingress, egress_node=egress)
        _set_domain_delay_profile(dom, is_hot=is_hot)

        domains[did] = dom

    # ---------- 域间图：主链 + 跳跃，热点走廊更香（诱饵） ----------
    dg = nx.DiGraph()
    for d_idx in range(num_domains):
        dg.add_node(f"D{d_idx}")

    def add_inter(u: str, v: str, delay: float):
        dg.add_edge(u, v, delay=delay)
        dg.add_edge(v, u, delay=delay)

    for d_idx in range(num_domains - 1):
        u = f"D{d_idx}"
        v = f"D{d_idx+1}"
        if d_idx in hot_set or (d_idx + 1) in hot_set:
            add_inter(u, v, delay=random.uniform(1.6, 2.9))
        else:
            add_inter(u, v, delay=random.uniform(6.2, 10.2))

    for d_idx in range(num_domains - 2):
        u = f"D{d_idx}"
        v = f"D{d_idx+2}"
        add_inter(u, v, delay=random.uniform(7.5, 11.5))

    return domains, dg


def generate_random_sfc(
    sfc_id: str,
    domain_ids: List[str],
    min_len: int = 4,
    max_len: int = 7,
    cpu_demand_range: Tuple[int, int] = (5, 9),
    bw_demand_range: Tuple[int, int] = (10, 30),
    rng: Optional[random.Random] = None,
) -> SFC:
    """
    让 workload 更“可学习”：
    - 同一个 src_domain 产生更固定的业务偏好（异质性更强 => Fed 有意义）
    - 保留 burst（尖峰）但概率更稳定，避免纯噪声
    """
    rng = rng or random
    src_domain, dst_domain = rng.sample(domain_ids, 2)

    length = rng.randint(min_len, max_len)
# ✅ burst 仍保留，但概率更稳，避免 batch 变成纯噪声
    if rng.random() < 0.12:   # 原 0.22
        length = min(max_len + 3, length + 3)

    src_idx = int(src_domain[1:])
    typ = src_idx % 3  # 0 CPU-heavy, 1 balanced, 2 spiky

    def sample_cpu():
        lo, hi = cpu_demand_range
        if typ == 0:
            return rng.randint(max(lo, (lo + hi)//2), hi + 2)
        # ✅ spiky：概率与幅度稍降，避免无释放时“必炸”
        if typ == 2 and rng.random() < 0.18:     # 原 0.28
            return hi + rng.randint(2, 4)        # 原 3~6
        return rng.randint(lo, hi)

    def sample_bw():
        lo, hi = bw_demand_range
        if typ == 2 and rng.random() < 0.15:     # 原 0.22
            return hi + rng.randint(6, 16)       # 原 8~24
        return rng.randint(lo, hi)


    vnfs = [VNF(name=f"f{i}", cpu_demand=sample_cpu(), bw_demand=sample_bw()) for i in range(length)]
    return SFC(id=sfc_id, vnfs=vnfs, src_domain=src_domain, dst_domain=dst_domain)


def build_toy_domains() -> Tuple[Dict[str, Domain], nx.DiGraph]:
    domains: Dict[str, Domain] = {}

    def make_domain(did: str, is_hot: bool) -> Domain:
        g = nx.DiGraph()
        nodes: Dict[str, Node] = {}

        n_nodes = 12 if is_hot else 6
        ids = []
        for i in range(n_nodes):
            nid = f"{did}_N{i}"
            if is_hot:
    # ✅ 保证最小容量 >= 5，否则 demand>=5 时这些节点永远不可用
                if i < n_nodes // 3:
                    cpu = random.randint(5, 7)      # 原来 3~6
                elif i > n_nodes - n_nodes // 4 - 1:
                    cpu = random.randint(12, 18)    # 保持
                else:
                    cpu = random.randint(6, 10)     # 原来 4~8，稍微抬一点让“中段”不至于全废
            else:
                cpu = random.randint(18, 26)


            nodes[nid] = Node(id=nid, cpu_capacity=cpu)
            g.add_node(nid)
            ids.append(nid)

        split = max(3, len(ids)//2)
        A = ids[:split]
        B = ids[split:]

        def add_undir(u, v, d):
            g.add_edge(u, v, delay=d, bw=100)
            g.add_edge(v, u, delay=d, bw=100)

        # A 段：诱饵更快，后段更慢
        for i in range(len(A)-1):
            if is_hot:
                d = random.uniform(0.9, 1.7) if i < len(A)//2 else random.uniform(6.5, 10.5)
            else:
                d = random.uniform(3.0, 6.0)
            add_undir(A[i], A[i+1], d)

        # B 段：更稳
        for i in range(len(B)-1):
            d = random.uniform(2.8, 4.5) if is_hot else random.uniform(3.0, 6.0)
            add_undir(B[i], B[i+1], d)

        if A and B:
            add_undir(A[len(A)//2], B[0], random.uniform(14.0, 22.0) if is_hot else random.uniform(4.0, 7.0))

        extra = 2 if is_hot else 10
        for _ in range(extra):
            u, v = random.sample(ids, 2)
            d = random.uniform(1.4, 2.6) if is_hot else random.uniform(3.0, 6.0)
            g.add_edge(u, v, delay=d, bw=100)

        ingress = A[0]
        egress = B[-1] if B else ids[-1]
        dom = Domain(id=did, graph=g, nodes=nodes, ingress_node=ingress, egress_node=egress)
        _set_domain_delay_profile(dom, is_hot=is_hot)
        return dom

    domains["D0"] = make_domain("D0", is_hot=False)
    domains["D3"] = make_domain("D3", is_hot=False)
    domains["D4"] = make_domain("D4", is_hot=False)
    domains["D5"] = make_domain("D5", is_hot=False)

    domains["D1"] = make_domain("D1", is_hot=True)
    domains["D2"] = make_domain("D2", is_hot=True)

    dg = nx.DiGraph()
    for did in domains:
        dg.add_node(did)

    def add_inter(u, v, d):
        dg.add_edge(u, v, delay=d)
        dg.add_edge(v, u, delay=d)

    add_inter("D0", "D1", 2.6)
    add_inter("D1", "D2", 2.6)
    add_inter("D2", "D3", 2.6)

    add_inter("D0", "D4", 6.8)
    add_inter("D4", "D5", 6.8)
    add_inter("D5", "D3", 6.8)

    return domains, dg
