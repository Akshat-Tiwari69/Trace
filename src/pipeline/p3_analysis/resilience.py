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


AUTO_EFFICIENCY_NODE_THRESHOLD = 256
# Stable-node, seed-42 calibration across full 25-removal targeted and random
# curves: k=224 missed the frozen <=0.02 RI-error budget; k=256 passed.
AUTO_EFFICIENCY_SAMPLES = 256
EFFICIENCY_SEED = 42
RANDOM_REMOVAL_SEED = 43


def _stable_nodes(graph: "nx.Graph") -> list:
    """Node order that is reproducible across equivalent graph insert orders."""
    return sorted(graph.nodes, key=lambda node: (type(node).__name__, repr(node)))


def efficiency_sampling_policy(
    graph: "nx.Graph",
    k: int | None = None,
    seed: int = EFFICIENCY_SEED,
    exact: bool = False,
) -> dict:
    """Resolve and disclose the interactive global-efficiency policy.

    Offline evaluators still call :func:`global_efficiency` directly for exact
    results. Interactive/API analysis uses exact routing through 256 active nodes
    and a fixed, deterministic source sample above that threshold.
    """
    if k is not None and k <= 0:
        raise ValueError("k must be a positive sample size (or None for automatic)")
    active_nodes = sum(1 for _, degree in graph.degree() if degree > 0)
    n = graph.number_of_nodes()
    resolved = None if exact else k
    if resolved is None and not exact and active_nodes > AUTO_EFFICIENCY_NODE_THRESHOLD:
        resolved = min(AUTO_EFFICIENCY_SAMPLES, n)
    if resolved is not None:
        resolved = min(resolved, n)
        if resolved >= n:
            resolved = None
    return {
        "method": "sampled" if resolved is not None else "exact",
        "k": resolved,
        "seed": seed,
        "active_nodes": active_nodes,
    }


def _sample_sources(graph: "nx.Graph", k: int, seed: int) -> list:
    return random.Random(seed).sample(_stable_nodes(graph), k)


def global_efficiency(
    graph: "nx.Graph",
    weight: str = "length_m",
    k: int | None = None,
    seed: int = 42,
    sources: list | None = None,
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

    ``sources`` (optional) supplies an **explicit source-node list** and takes
    precedence over ``k`` — used by :func:`ablation_curve` so every step of a
    k-sampled curve is estimated from the *same* fixed source set (comparable
    numbers, no per-step resampling noise). An empty ``sources`` list returns
    0.0 (no usable estimate).
    """
    import networkx as nx

    if k is not None and k <= 0:
        raise ValueError("k must be a positive sample size (or None for exact)")

    n = graph.number_of_nodes()
    if n < 2:
        return 0.0

    nodes = _stable_nodes(graph)
    if sources is not None:
        if not sources:
            return 0.0
        if len(set(sources)) != len(sources) or not set(sources).issubset(graph):
            raise ValueError("sources must be unique nodes from the graph")
        norm = len(sources) * (n - 1)
    elif k is not None and k < n:
        sources = _sample_sources(graph, k, seed)
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
    seed: int = EFFICIENCY_SEED,
) -> dict:
    """Resilience after removing ``removed_nodes``: ``E(perturbed)/E(baseline)``.

    Returns a dict with the baseline/perturbed efficiencies, the finite
    ``resilience_index`` ratio, and ``largest_cc_fraction`` (how much of the
    network is still in one piece). Operates on a copy — the input is untouched.

    ``k`` forwards to :func:`global_efficiency` for k-sample estimation on large
    graphs. Failed nodes remain in the baseline node universe, so the deterministic
    node order and seed select the same sources for both efficiencies. A supplied
    ``baseline_efficiency`` must use the same sampling protocol.
    """
    removed_nodes = _validated_removals(graph, removed_nodes)
    fixed_sources = (
        _sample_sources(graph, k, seed)
        if k is not None and k < graph.number_of_nodes()
        else None
    )
    base = (
        global_efficiency(graph, weight, sources=fixed_sources)
        if baseline_efficiency is None
        else baseline_efficiency
    )

    if base <= 0:
        raise ValueError(
            "baseline global efficiency is 0 — graph is degenerate/disconnected; "
            "upstream data invalid"
        )

    perturbed = graph.copy()
    # Keep the baseline node universe and isolate failed nodes. This preserves the
    # N(N-1) normaliser and makes removed nodes contribute unreachable pairs,
    # rather than shrinking the denominator and allowing RI > 1.
    failed_edges = []
    for node in removed_nodes:
        if not perturbed.has_node(node):
            continue
        if perturbed.is_multigraph():
            failed_edges.extend(perturbed.edges(node, keys=True))
        else:
            failed_edges.extend(perturbed.edges(node))
    perturbed.remove_edges_from(failed_edges)
    eff = global_efficiency(perturbed, weight, sources=fixed_sources)

    ri = eff / base
    policy = {
        "method": "sampled" if fixed_sources is not None else "exact",
        "k": len(fixed_sources) if fixed_sources is not None else None,
        "seed": seed,
    }
    return {
        "removed": list(removed_nodes),
        "n_removed": len(removed_nodes),
        "baseline_efficiency": base,
        "perturbed_efficiency": eff,
        "resilience_index": ri,
        "efficiency_loss_pct": 100.0 * (1.0 - ri),
        "largest_cc_fraction": _largest_cc_fraction(perturbed),
        "active_largest_cc_fraction": _largest_cc_fraction(
            perturbed, graph.number_of_nodes() - len(removed_nodes)
        ),
        "efficiency_method": policy["method"],
        "efficiency_k": policy["k"],
        "efficiency_seed": policy["seed"],
    }


def _validated_removals(graph: "nx.Graph", removed_nodes: list) -> list:
    removed = list(removed_nodes)
    if len(set(removed)) != len(removed):
        raise ValueError("removed_nodes contains duplicate nodes")
    unknown = [node for node in removed if node not in graph]
    if unknown:
        raise ValueError(f"removed_nodes contains unknown nodes: {unknown}")
    return removed


def _largest_cc_fraction(graph: "nx.Graph", denominator: int | None = None) -> float:
    """Largest component divided by the baseline or caller-supplied universe."""
    import networkx as nx

    n = graph.number_of_nodes() if denominator is None else denominator
    if n == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return largest / n


@dataclasses.dataclass
class AblationPoint:
    """One step of an ablation curve."""

    n_removed: int
    efficiency: float
    resilience_index: float
    largest_cc_fraction: float
    active_largest_cc_fraction: float


def ablation_curve(
    graph: "nx.Graph",
    order: str = "targeted",
    betweenness: dict[int, float] | None = None,
    steps: int | None = None,
    weight: str = "length_m",
    seed: int = 42,
    source_seed: int | None = None,
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
    ``seed`` controls random removal order; ``source_seed`` independently controls
    sampled efficiency sources (and defaults to ``seed`` for compatibility).

    Failed nodes remain as isolates, preserving the baseline node universe and
    normaliser at every step. When ``k`` is set, the source nodes are sampled
    **once** and reused unchanged, so the estimates remain directly comparable.
    Raises ``ValueError`` when the baseline efficiency is 0 (degenerate graph) —
    a silent 0.0 would be indistinguishable from "network destroyed".
    """
    import networkx as nx

    nodes = _stable_nodes(graph)
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative or None")

    # Fixed source set for k-sampling: drawn once, reused (minus removed nodes)
    # at every step so successive efficiencies are comparable estimates.
    fixed_sources: list | None = None
    if k is not None and 0 < k < len(nodes):
        fixed_sources = _sample_sources(
            graph, k, seed if source_seed is None else source_seed
        )

    def _efficiency(g: "nx.Graph") -> float:
        if fixed_sources is None:
            return global_efficiency(g, weight, k=k)
        return global_efficiency(g, weight, sources=fixed_sources)

    base = _efficiency(graph)
    if base <= 0:
        raise ValueError(
            "baseline global efficiency is 0 — graph is degenerate/disconnected; "
            "upstream data invalid"
        )

    if sequence is not None:
        sequence = _validated_removals(graph, sequence)
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

    baseline_fraction = _largest_cc_fraction(graph)
    curve = [
        AblationPoint(
            0, base, 1.0 if base > 0 else 0.0, baseline_fraction, baseline_fraction
        )
    ]
    working = graph.copy()
    for i, node in enumerate(sequence, start=1):
        incident = (
            working.edges(node, keys=True)
            if working.is_multigraph()
            else working.edges(node)
        )
        working.remove_edges_from(list(incident))
        eff = _efficiency(working)
        curve.append(
            AblationPoint(
                n_removed=i,
                efficiency=eff,
                resilience_index=(eff / base) if base > 0 else 0.0,
                largest_cc_fraction=_largest_cc_fraction(working),
                active_largest_cc_fraction=_largest_cc_fraction(
                    working, graph.number_of_nodes() - i
                ),
            )
        )
    return curve
