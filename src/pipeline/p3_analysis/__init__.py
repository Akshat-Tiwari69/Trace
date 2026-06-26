"""P3 · Criticality + resilience lane (Shaivi).

Betweenness centrality to find chokepoints, node-ablation stress tests, and a
finite **global-efficiency** Resilience Index (``docs/PRD.md`` G3; metric locked
in ``docs/Tracker.md`` §8). CPU-only, classical Python.
"""

from src.pipeline.p3_analysis.criticality import (
    BetweennessCache,
    annotate_criticality,
    annotate_cut_structure,
    benchmark_betweenness,
    compute_betweenness,
    rank_table,
)
from src.pipeline.p3_analysis.resilience import (
    ablation_curve,
    global_efficiency,
    resilience_index,
)

__all__ = [
    "BetweennessCache",
    "annotate_criticality",
    "annotate_cut_structure",
    "benchmark_betweenness",
    "compute_betweenness",
    "rank_table",
    "ablation_curve",
    "global_efficiency",
    "resilience_index",
]
