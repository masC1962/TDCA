from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import networkx as nx


class NodeType(str, Enum):
    KG = "kg"
    STATE = "state"
    MEMORY = "memory"


class EdgeType(str, Enum):
    STATE_TRANSITION = "state_transition"
    SUPPORTS = "supports"
    RECALLS = "recalls"
    REFINES = "refines"
    VERIFIES = "verifies"
    DERIVES = "derives"


@dataclass
class RetrievedContext:
    item_id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    content: str
    depth: int = 0
    parent_id: Optional[str] = None

    value: float = 0.0
    temperature: float = 0.0
    expanded: bool = False
    visit_count: int = 0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def frontier_key(self) -> tuple:
        return (self.temperature, self.value, -self.depth)

    @property
    def persistent(self) -> bool:
        return self.node_type in {NodeType.KG, NodeType.MEMORY}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "value": self.value,
            "temperature": self.temperature,
            "expanded": self.expanded,
            "visit_count": self.visit_count,
            "persistent": self.persistent,
            "score_breakdown": self.score_breakdown,
            "metadata": self.metadata,
        }


class HeteroGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, node_type=node.node_type.value)

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def add_edge(self, src_id: str, dst_id: str, edge_type: EdgeType, weight: float = 1.0) -> None:
        if src_id not in self.nodes or dst_id not in self.nodes:
            return
        self.graph.add_edge(src_id, dst_id, edge_type=edge_type.value, weight=weight)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.graph.remove_node(node_id)
            self.nodes.pop(node_id, None)

    def state_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.STATE]

    def memory_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.MEMORY]

    def kg_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.KG]

    def frontier(self) -> List[Node]:
        return [n for n in self.state_nodes() if not n.expanded]

    def all_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def ancestor_chain(self, node_id: str) -> Set[str]:
        keep: Set[str] = set()
        current = node_id
        while current and current in self.nodes and current not in keep:
            keep.add(current)
            current = self.nodes[current].parent_id
        return keep

    def export_json(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "edge_type": data.get("edge_type"),
                    "weight": data.get("weight", 1.0),
                }
                for u, v, data in self.graph.edges(data=True)
            ],
        }
