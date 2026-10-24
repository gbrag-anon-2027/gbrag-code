"""
Edge-construction configuration module.

Controls whether edges between E-level balls (boundary bridges) are built in the
granular-ball hierarchy graph.
"""
from enum import Enum


class EdgeMode(Enum):
    WITH_EDGES = "with_edges"        # build edges; use full graph topology for multi-hop retrieval
    WITHOUT_EDGES = "without_edges"  # skip edges; answer from A-level propositions directly


# Global configuration (can be changed at runtime from the main script)
CURRENT_EDGE_MODE = EdgeMode.WITH_EDGES


def set_edge_mode(mode: EdgeMode):
    """Dynamically set the edge mode at runtime."""
    global CURRENT_EDGE_MODE
    CURRENT_EDGE_MODE = mode


def is_edge_enabled() -> bool:
    """Return whether edge construction is currently enabled."""
    return CURRENT_EDGE_MODE == EdgeMode.WITH_EDGES
