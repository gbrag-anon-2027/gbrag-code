"""Central configuration for GBRAG runs.

The released pipeline is the v18 dual-pool framework (the old D-level routing
has been removed). The experiment id is kept for output file naming and trace
bookkeeping; override it with the GBRAG_EXPERIMENT_ID environment variable.
"""
import os
from dataclasses import dataclass

DEFAULT_EXPERIMENT_ID = "DCB_v18_dual_pool"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = DEFAULT_EXPERIMENT_ID


def get_experiment_config():
    """Return the active experiment configuration."""
    experiment_id = os.environ.get("GBRAG_EXPERIMENT_ID", DEFAULT_EXPERIMENT_ID).strip()
    return ExperimentConfig(experiment_id or DEFAULT_EXPERIMENT_ID)
