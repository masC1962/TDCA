"""Dynamic Reasoning Hypergraph TDCA.

This package is deliberately parallel to the frozen static reasoner.  It reuses
providers, retrieval and experiment infrastructure without importing or mutating
the static execution engine.
"""

from .config import DynamicResearchConfig
from .graph import DynamicReasoningHypergraph, GraphBudgetExceeded, GraphInvariantError

__all__ = ["DynamicResearchConfig", "DynamicReasoningHypergraph", "GraphBudgetExceeded", "GraphInvariantError"]
