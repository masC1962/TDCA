"""Training-free Dynamic Reasoning Hypergraph v2.

V2 is intentionally isolated from the frozen v1 implementation.  Its defining
mechanism is graph-state-driven computation allocation, not a renamed operation
scheduler.
"""

from .config import DynamicV2ResearchConfig
from .graph import DynamicReasoningHypergraphV2

__all__ = ["DynamicReasoningHypergraphV2", "DynamicV2ResearchConfig"]
