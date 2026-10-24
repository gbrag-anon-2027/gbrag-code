"""Central, validated configuration for GBRAG layer and edge ablations.

The registry below is aligned with the ablation matrix of the paper:

| Ablation row in paper       | experiment_id           |
|-----------------------------|-------------------------|
| Full model                  | DCB_edges               |
| w/o Cross-ball Edges        | DCB_no_edges            |
| w/o D-level Routing         | CB_edges                |
| w/o Entity Activation       | DB                      |
| Uniform Edge Costs          | DCB_edges_uniform_cost  |
| w/o Path-ordered Assembly   | DCB_edges_no_path       |

Additional ids remain legal but are not part of the ablation table:
DC_edges (w/o B-level Filtering; row removed from the slim table), flat
(w/o Hierarchical Routing; reported inline instead of in the table),
D/B/C_* (early layer matrix kept for diagnostics).

Flag tuple layout: (use_d, use_c, use_b, use_edges,
                    uniform_edge_cost, use_path_search, flat_retrieval)
"""
from dataclasses import dataclass
import os


LEGAL_EXPERIMENTS = {
    # Traditional layer ablations: run through _run_layer_ablation_pipeline
    "D": (True, False, False, False, False, True, False),
    "B": (False, False, True, False, False, True, False),
    "DB": (True, False, True, False, False, True, False),
    "C_no_edges": (False, True, False, False, False, True, False),
    "C_edges": (False, True, False, True, False, True, False),
    "DC_no_edges": (True, True, False, False, False, True, False),
    "DC_edges": (True, True, False, True, False, True, False),
    "CB_no_edges": (False, True, True, False, False, True, False),
    "CB_edges": (False, True, True, True, False, True, False),
    "DCB_no_edges": (True, True, True, False, False, True, False),
    # Full model + full-pipeline variants: run through the Graph_Rag_pipeline main path
    "DCB_edges": (True, True, True, True, False, True, False),
    "DCB_edges_uniform_cost": (True, True, True, True, True, True, False),
    "DCB_edges_no_path": (True, True, True, True, False, False, False),
    "flat": (False, False, False, False, False, True, True),
    # P1 variant: B-ball entry + two-stage (lexical -> semantic) in-ball recall
    # + global fallback. Not a formal ablation row; output suffix DCB_v2.
    "DCB_v2": (True, True, True, True, False, True, False),
    # P2 variant: P1 with a fixed fallback trigger condition.
    "DCB_v3": (True, True, True, True, False, True, False),
    # P3 variant: P1 two-stage recall + edge-activated context (bridge propositions
    # attached as a [multi-hop clue bridge] section). Output suffix DCB_v4.
    "DCB_v4": (True, True, True, True, False, True, False),
    # P4 variant: v4 + a similarity hard threshold GBRAG_P1_BRIDGE_MIN_SIM
    # (default 0.60, aligned with the original DIRECT_EDGE_THRESHOLD)
    # plus deduplication against candidate facts. Output suffix DCB_v5.
    "DCB_v5": (True, True, True, True, False, True, False),
    # v18 current framework (no D-level routing): dual-pool evidence context
    # + per-module ablation switches.
    "DCB_v18_dual_pool": (False, True, True, True, False, True, False),
}

# The v6 answer-prompt experiment deliberately keeps the v5 retrieval strategy.
if os.environ.get("GBRAG_EXPERIMENT_ID", "").strip() == "DCB_v6_answer_prompt":
    LEGAL_EXPERIMENTS["DCB_v6_answer_prompt"] = LEGAL_EXPERIMENTS["DCB_v5"]

# Whether an id runs the full pipeline is decided by is_full_pipeline_id;
# traditional layer ablations (D/B/DB/C_*/DC_*/CB_*/DCB_no_edges) run the
# ablation pipeline instead.


def is_full_pipeline_id(experiment_id: str) -> bool:
    """flat/uniform_cost/no_path, v18 and the full model run the main pipeline.

    Note: the traditional layer-ablation pipeline (_run_layer_ablation_pipeline)
    was removed together with the D-level routing; its code is archived."""
    if str(experiment_id).strip() == "DCB_v18_dual_pool":
        return True
    config = ExperimentConfig.from_id(experiment_id)
    return config.flat_retrieval or config.uniform_edge_cost or not config.use_path_search \
        or (config.use_d and config.use_c and config.use_b and config.use_edges)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    use_d: bool
    use_c: bool
    use_b: bool
    use_edges: bool
    uniform_edge_cost: bool = False
    use_path_search: bool = True
    flat_retrieval: bool = False

    @classmethod
    def from_id(cls, experiment_id):
        experiment_id = str(experiment_id).strip()
        if experiment_id not in LEGAL_EXPERIMENTS:
            raise ValueError(
                f"Illegal experiment config {experiment_id!r}; legal values: "
                + ", ".join(LEGAL_EXPERIMENTS)
            )
        return cls(experiment_id, *LEGAL_EXPERIMENTS[experiment_id])


def get_experiment_config():
    """Default preserves the current full GBRAG behavior."""
    return ExperimentConfig.from_id(
        os.environ.get("GBRAG_EXPERIMENT_ID", "DCB_v18_dual_pool")
    )
