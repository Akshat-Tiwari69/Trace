"""Resilience — global efficiency under node ablation (the finite metric).

The project's resilience metric is **global efficiency** (Latora & Marchiori
2001), chosen deliberately over the raw average-path-length ratio because it
**stays finite when the graph disconnects** (``docs/PRD.md`` NFR3, decision
locked in ``docs/Tracker.md`` §8 — never revert this).

Global efficiency::

    E(G) = (1 / (N·(N-1))) · Σ_{i≠j} 1 / d(i, j)

with ``d`` the shortest-path distance weighted by ``length_m``. Disconnected
pairs contribute ``1/∞ = 0`` rather than blowing up — so removing the bridge
between two halves simply drops ``E`` smoothly. The **Resilience Index** is the
ratio ``E(perturbed) / E(baseline)`` ∈ ``[0, 1]``.

We also expose an **ablation curve** (remove nodes one by one, targeted vs.
random) — the sanity check that betweenness finds genuinely critical nodes:
targeted removal must degrade efficiency faster than random (``docs/Evaluation.md``).
"""

from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def global_efficiency(
    graph: "nx.Graph",
    weight: str = "length_m",
    k: int | None = None,
    seed: int = 42,
) -> float:
    """Weighted global efficiency of ``graph`` (finite even when disconnected).

    Returns 0.0 for graphs with fewer than two nodes. Runs Dijkstra from each
    source; unreachable pairs contribute 0 (the whole reason this metric is used).

    Exact cost is all-pairs Dijkstra — O(N·(E log V)) — which is fine for a city
    sub-region but becomes the bottleneck on very large graphs (``docs/TRD.md``
    performance; ``RiskRegister.md`` T-3). Pass ``k`` to estimate efficiency from
    ``k`` randomly sampled sources instead of all N (the same k-sample trick used
    for approximate betweenness); ``k=None`` (default) stays exact so committed
    artifacts don't drift.
    """
    import networkx as nx

    if k is not None and k <= 0:
        raise ValueError("k must be a positive sample size (or None for exact)")

    n = graph.number_of_nodes()
    if n < 2:
        return 0.0

    nodes = list(graph.nodes)
    if k is not None and k < n:
        sources = random.Random(seed).sample(nodes, k)
        norm = k * (n - 1)  # unbiased estimate: mean per-source efficiency
    else:
        sources = nodes
        norm = n * (n - 1)

    total = 0.0
    for source in sources:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
        for target, dist in lengths.items():
            if source != target and dist > 0:
                total += 1.0 / dist
    return total / norm


def resilience_index(
    graph: "nx.Graph",
    removed_nodes: list[int],
    weight: str = "length_m",
    baseline_efficiency: float | None = None,
    k: int | None = None,
) -> dict:
    """Resilience after removing ``removed_nodes``: ``E(perturbed)/E(baseline)``.

    Returns a dict with the baseline/perturbed efficiencies, the finite
    ``resilience_index`` ratio, and ``largest_cc_fraction`` (how much of the
    network is still in one piece). Operates on a copy — the input is untouched.

    ``k`` forwards to :func:`global_efficiency` for k-sample estimation on large
    graphs. Note both efficiencies are then estimates from *independently* sampled
    sources (the baseline and perturbed graphs have different node sets), so the
    ratio is approximate and noisier; use ``k=None`` (default) for an exact,
    directly-comparable ratio.
    """
    base = (
        global_efficiency(graph, weight, k=k)
        if baseline_efficiency is None
        else baseline_efficiency
    )

    perturbed = graph.copy()
    perturbed.remove_nodes_from(removed_nodes)
    eff = global_efficiency(perturbed, weight, k=k)

    ri = (eff / base) if base > 0 else 0.0
    return {
        "removed": list(removed_nodes),
        "n_removed": len(removed_nodes),
        "baseline_efficiency": base,
        "perturbed_efficiency": eff,
        "resilience_index": ri,
        "travel_time_delta_pct": _travel_time_delta_pct(ri),
        "largest_cc_fraction": _largest_cc_fraction(perturbed),
    }


def _largest_cc_fraction(graph: "nx.Graph") -> float:
    """Fraction of nodes in the largest connected component (0–1)."""
    import networkx as nx

    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return largest / n


def _travel_time_delta_pct(resilience_index: float) -> float:
    """Rough average travel-time increase implied by an efficiency drop.

    Efficiency is the mean of inverse path lengths, so its reciprocal tracks
    mean travel time: a drop to ``RI`` implies ~``(1/RI - 1)`` longer trips. A
    readable headline number for the dashboard (``docs/UserJourney.md`` Flow B);
    the exact per-route delta is computed live on click by P4.
    """
    if resilience_index <= 0:
        return float("inf")
    return 100.0 * (1.0 / resilience_index - 1.0)


@dataclasses.dataclass
class AblationPoint:
    """One step of an ablation curve."""

    n_removed: int
    efficiency: float
    resilience_index: float
    largest_cc_fraction: float


def ablation_curve(
    graph: "nx.Graph",
    order: str = "targeted",
    betweenness: dict[int, float] | None = None,
    steps: int | None = None,
    weight: str = "length_m",
    seed: int = 42,
    k: int | None = None,
    sequence: list[int] | None = None,
) -> list[AblationPoint]:
    """Remove nodes one at a time and trace how efficiency degrades.

    ``order='targeted'`` removes highest-betweenness nodes first (needs
    ``betweenness``); ``order='random'`` removes in a shuffled order. Pass an
    explicit ``sequence`` to use a custom removal order (e.g. a flood scenario,
    ``order`` is then ignored). The targeted-vs-random pair is the sanity check in
    ``docs/Evaluation.md``. ``k`` forwards to :func:`global_efficiency` for
    k-sample estimation, so the per-step recompute stays cheap on large graphs.
    """
    import networkx as nx

    base = global_efficiency(graph, weight, k=k)
    nodes = list(graph.nodes)

    if sequence is not None:
        sequence = list(sequence)
    elif order == "targeted":
        if betweenness is None:
            raise ValueError("order='targeted' requires a betweenness dict")
        sequence = sorted(nodes, key=lambda n: betweenness.get(n, 0.0), reverse=True)
    elif order == "random":
        rng = random.Random(seed)
        sequence = nodes[:]
        rng.shuffle(sequence)
    else:
        raise ValueError("order must be 'targeted' or 'random'")

    if steps is not None:
        sequence = sequence[:steps]

    curve = [AblationPoint(0, base, 1.0 if base > 0 else 0.0, _largest_cc_fraction(graph))]
    working = graph.copy()
    for i, node in enumerate(sequence, start=1):
        working.remove_node(node)
        eff = global_efficiency(working, weight, k=k)
        curve.append(
            AblationPoint(
                n_removed=i,
                efficiency=eff,
                resilience_index=(eff / base) if base > 0 else 0.0,
                largest_cc_fraction=_largest_cc_fraction(working),
            )
        )
    return curve
