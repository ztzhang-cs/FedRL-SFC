from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import networkx as nx


@dataclass
class VNF:
    name: str
    cpu_demand: float
    bw_demand: float
    parallel_group: Optional[int] = None  # 暂时不用


@dataclass
class SFC:
    id: str
    vnfs: List[VNF]
    src_domain: str
    dst_domain: str


@dataclass
class Node:
    id: str
    cpu_capacity: float
    cpu_used: float = 0.0

    @property
    def cpu_free(self) -> float:
        return self.cpu_capacity - self.cpu_used


@dataclass
class Domain:
    id: str
    graph: nx.DiGraph          # 域内拓扑
    nodes: Dict[str, Node]
    ingress_node: str
    egress_node: str

    def total_cpu_free(self) -> float:
        return sum(n.cpu_free for n in self.nodes.values())


@dataclass
class PlacementResult:
    sfc_id: str
    domain_path: List[str]
    vnf_to_domain: Dict[str, str]             # VNF.name -> domain.id
    vnf_to_node: Dict[str, Tuple[str, str]]   # VNF.name -> (domain.id, node.id)
    total_delay: float
