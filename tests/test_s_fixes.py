"""Regression tests for the A36 S-effort fixes (bugs.md).

Grouped in one module because each fix is a few lines in a different file:
fail-loud guards (analyze/resilience/APLS/rank_table/load_checkpoint), the
AOI sanitizer, atomic writes, the zero-length-edge invariant, deterministic
parallel-branch collapse, and the foreground-biased training crop.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.p2_graph.config import GraphConfig, sanitize_aoi
from src.pipeline.p2_graph.graph_io import atomic_write, load_graphml, save_graphml


# --------------------------------------------------------------------------- #
# AOI sanitization (bugs.md §5A)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["../../evil", "", "UPPER", "a b", "x" * 65, "a/b"])
def test_sanitize_aoi_rejects_unsafe_ids(bad):
    with pytest.raises(ValueError):
        sanitize_aoi(bad)


def test_sanitize_aoi_accepts_valid_ids():
    assert sanitize_aoi("panaji_demo-2") == "panaji_demo-2"


def test_graphconfig_rejects_traversal_aoi(tmp_path):
    with pytest.raises(ValueError):
        GraphConfig(aoi="../escape", interim_dir=tmp_path, processed_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Atomic writes + zero-length-edge invariant (bugs.md §5A / §4)
# --------------------------------------------------------------------------- #
def test_atomic_write_failure_leaves_target_untouched(tmp_path):
    target = tmp_path / "artifact.json"
    target.write_text("previous good content")

    def _exploding_writer(tmp):
        tmp.write_text("partial")
        raise RuntimeError("crash mid-write")

    with pytest.raises(RuntimeError):
        atomic_write(target, _exploding_writer)
    assert target.read_text() == "previous good content"
    assert not (tmp_path / "artifact.json.tmp").exists()


def test_loader_rejects_zero_length_edges(tmp_path):
    import networkx as nx

    g = nx.Graph()
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=0.0, y=0.0)
    g.add_edge(1, 2, length_m=0.0, geometry=[[0.0, 0.0], [0.0, 0.0]], is_bridged=False)
    path = tmp_path / "bad.graphml"
    save_graphml(g, path)

    with pytest.raises(ValueError, match="length_m"):
        load_graphml(path)


# --------------------------------------------------------------------------- #
# Fail-loud P3 guards (bugs.md §7)
# --------------------------------------------------------------------------- #
def test_analyze_refuses_degenerate_graph(tmp_path):
    import networkx as nx

    from src.pipeline.p3_analysis.analyze import analyze

    cfg = GraphConfig(aoi="degenerate", interim_dir=tmp_path, processed_dir=tmp_path)
    g = nx.Graph()
    g.add_node(1, x=0.0, y=0.0)
    save_graphml(g, cfg.graphml_path)

    with pytest.raises(ValueError, match="upstream P1/P2 failure"):
        analyze(cfg)


def test_resilience_index_raises_on_zero_baseline():
    import networkx as nx

    from src.pipeline.p3_analysis.resilience import resilience_index

    g = nx.Graph()
    g.add_nodes_from([1, 2])  # no edges → baseline efficiency 0
    with pytest.raises(ValueError, match="baseline global efficiency is 0"):
        resilience_index(g, (1,))


def test_apls_scores_zero_when_nothing_is_reachable():
    import networkx as nx

    from src.pipeline.p3_analysis.apls import _apls_oneway

    src = nx.Graph()
    src.add_nodes_from(range(6))  # no edges: every sampled pair is unreachable
    assert _apls_oneway(src, src, snap={}, n_samples=20, weight="length_m", seed=1) == 0.0


def test_rank_table_requires_annotation_first():
    import networkx as nx

    from src.pipeline.p3_analysis.criticality import rank_table

    g = nx.Graph()
    g.add_node(7, x=1.0, y=2.0)  # no 'betweenness' attribute — annotate not run
    with pytest.raises(ValueError, match="annotate_criticality"):
        rank_table(g, {7: 0.5})


# --------------------------------------------------------------------------- #
# Parallel-branch preservation under MultiGraph (bugs.md §4 — A37 migration)
# --------------------------------------------------------------------------- #
def test_skeleton_parallel_branches_preserved(capsys):
    from src.pipeline.p2_graph.skeleton_graph import skeleton_to_graph

    # Two junctions A(6,2) and B(6,10) joined by TWO branches — a straight
    # 8-px chord and a longer diagonal detour over the apex (2,6). Stubs on
    # both sides make A and B true 3-way junctions (diagonal bends are not
    # junction pixels, so the detour stays one branch).
    #
    # Pre-A37 the simple graph collapsed these to one (keep-shortest) — silently
    # destroying a redundant alternate route that the resilience metric rewards.
    # The MultiGraph now keeps BOTH branches as keyed edges between A and B.
    skel = np.zeros((13, 13), np.uint8)
    skel[6, 0:13] = 1  # stub — A — chord — B — stub
    for r, c in [(5, 3), (4, 4), (3, 5), (2, 6), (3, 7), (4, 8), (5, 9)]:
        skel[r, c] = 1  # the diagonal detour (length ~11.3)

    graph = skeleton_to_graph(skel, transform=None, resolution_m=1.0)

    # 4 edges: two stubs + BOTH A–B branches (chord ~8 px + detour ~11.3 px).
    assert graph.number_of_edges() == 4
    lengths = sorted(d["length_m"] for _, _, d in graph.edges(data=True))
    # the two longest are the two A–B branches; the chord AND the detour survive
    assert any(l < 9.0 for l in lengths[-2:]), "chord branch lost"
    assert any(l > 9.0 for l in lengths[-2:]), "detour branch lost"
    assert "1 parallel branch(es) kept" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Checkpoint meta guard (bugs.md §7)
# --------------------------------------------------------------------------- #
def test_load_checkpoint_refuses_missing_meta(tmp_path):
    import torch

    from src.pipeline.p1_segment.model import load_checkpoint

    path = tmp_path / "no_meta.pt"
    torch.save({"state_dict": {}, "meta": {}}, path)
    with pytest.raises(ValueError, match="refusing to guess"):
        load_checkpoint(path)


# --------------------------------------------------------------------------- #
# Foreground-biased crop (bugs.md §3)
# --------------------------------------------------------------------------- #
def _write_sparse_pair(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), np.uint8)
    mask[250:253, :] = 255  # one thin road stripe → uniform 64px crops mostly miss it
    sat, msk = tmp_path / "t_sat.jpg", tmp_path / "t_mask.png"
    Image.fromarray(image).save(sat)
    Image.fromarray(mask, mode="L").save(msk)
    return sat, msk


def test_foreground_bias_yields_road_containing_crops(tmp_path):
    import random

    from src.pipeline.p1_segment.dataset import RoadTileDataset, build_train_transform

    pairs = [_write_sparse_pair(tmp_path)]
    ds = RoadTileDataset(
        pairs,
        build_train_transform(size=64, occlusion=False),
        crops_per_image=8,
        foreground_bias=1.0,
    )
    random.seed(42)
    for i in range(8):
        _, mask = ds[i]
        assert float(mask.sum()) > 0, f"crop {i} contains no road despite bias=1.0"


def test_foreground_bias_validation():
    from src.pipeline.p1_segment.dataset import RoadTileDataset

    with pytest.raises(ValueError, match="foreground_bias"):
        RoadTileDataset([("a", "b")], foreground_bias=1.5)
