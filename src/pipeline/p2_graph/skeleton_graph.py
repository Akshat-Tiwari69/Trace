"""Mask → skeleton → NetworkX graph (Phase II, first half).

Turns a binary road mask into a vector graph, mirroring the classical pipeline
in ``docs/Research.md`` → *Methodology*: ``skimage.morphology.skeletonize`` thins
the mask to 1-px centrelines, then **sknw** builds a NetworkX graph from the
skeleton, and we attach the schema attributes (``x, y, degree, type`` on nodes;
``length_m, geometry`` on edges — see ``docs/Schema.md``).

Coordinates: the skeleton is in pixel space. When an alignment ``transform`` is
supplied (from ``osm_mask.build_grid`` via the mask manifest), node geometry is
placed in the grid's **metric UTM** world so ``length_m`` and the healing
distance/angle maths are all true metres. Without a transform we fall back to
pixel coordinates scaled by ``resolution_m``. Reprojection to WGS84 lon/lat for
mapping happens once, *after* healing, via :func:`reproject_graph_to_wgs84`.

``skimage``/``sknw``/``pyproj`` are imported lazily so the pure-logic helpers and
their tests stay importable without the heavy geo stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from affine import Affine


# Node type labels (``docs/Schema.md`` → type ∈ {intersection, endpoint, bridged})
TYPE_INTERSECTION = "intersection"
TYPE_ENDPOINT = "endpoint"
TYPE_BRIDGED = "bridged"


def mask_to_skeleton(mask01: np.ndarray) -> np.ndarray:
    """Thin a binary {0,1} road mask to a 1-px-wide skeleton (bool array)."""
    from skimage.morphology import skeletonize

    if mask01.ndim != 2:
        raise ValueError("mask must be 2-D (H×W)")
    binary = np.asarray(mask01) > 0
    return skeletonize(binary)


def _classify(degree: int) -> str:
    """Map a node's degree to its schema ``type``."""
    if degree == 1:
        return TYPE_ENDPOINT
    return TYPE_INTERSECTION  # degree ≥ 2 (sknw collapses pure degree-2 chains)


def skeleton_to_graph(
    skeleton: np.ndarray,
    transform: "Affine | None" = None,
    resolution_m: float = 1.0,
) -> "nx.Graph":
    """Build a clean, **metric** NetworkX graph from a skeleton image.

    Parameters
    ----------
    skeleton : 2-D bool/int array
        1-px-wide centrelines (output of :func:`mask_to_skeleton`).
    transform :
        Affine pixel→world transform from the mask manifest (a metric UTM grid).
        If given, node ``x, y`` and edge geometry are in metres. If ``None``,
        coordinates are pixels scaled by ``resolution_m``.
    resolution_m :
        Ground sampling distance, used only in the no-transform fallback.

    Returns
    -------
    nx.Graph
        Nodes carry ``x, y`` (metric), ``degree, type``; edges carry
        ``length_m`` (metres), ``geometry`` (a metric ``[[x, y], ...]`` polyline)
        and ``is_bridged=False``. Reproject to lon/lat with
        :func:`reproject_graph_to_wgs84` before writing for the map.
    """
    import networkx as nx
    import sknw

    skel = np.asarray(skeleton).astype(np.uint16)
    raw = sknw.build_sknw(skel, multi=False)

    def pixel_to_metric(row: float, col: float) -> tuple[float, float]:
        """Pixel (row, col) → metric world (x, y), or scaled pixels if no grid."""
        if transform is not None:
            x, y = transform * (col + 0.5, row + 0.5)  # affine takes (col, row)
            return float(x), float(y)
        return col * resolution_m, row * resolution_m

    graph = nx.Graph()
    for node_id, data in raw.nodes(data=True):
        row, col = float(data["o"][0]), float(data["o"][1])  # sknw 'o' = (y, x)
        mx, my = pixel_to_metric(row, col)
        graph.add_node(int(node_id), x=mx, y=my)

    for u, v, data in raw.edges(data=True):
        pts = np.asarray(data["pts"], dtype=float)  # (row, col) polyline
        metric = [list(pixel_to_metric(r, c)) for r, c in pts]
        if len(metric) >= 2:
            arr = np.asarray(metric)
            seg = np.diff(arr, axis=0)
            length_m = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        else:
            length_m = 0.0
        graph.add_edge(
            int(u),
            int(v),
            length_m=length_m,
            geometry=metric,
            is_bridged=False,
        )

    _annotate_degree_and_type(graph)
    return graph


def reproject_graph_to_wgs84(graph: "nx.Graph", crs: object) -> None:
    """Reproject node ``x, y`` and edge ``geometry`` from ``crs`` to WGS84, in place.

    ``length_m`` is left untouched (it was measured in the source metric CRS and
    stays the routing weight). Call this once, after healing, before output.
    """
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    for _, data in graph.nodes(data=True):
        lon, lat = to_lonlat(data["x"], data["y"])
        data["x"], data["y"] = float(lon), float(lat)
    for _, _, data in graph.edges(data=True):
        data["geometry"] = [
            [float(lon), float(lat)]
            for lon, lat in (to_lonlat(x, y) for x, y in data["geometry"])
        ]


def _annotate_degree_and_type(graph: "nx.Graph") -> None:
    """(Re)compute ``degree`` and ``type`` on every node, in place."""
    for node_id in graph.nodes:
        deg = graph.degree(node_id)
        graph.nodes[node_id]["degree"] = int(deg)
        graph.nodes[node_id]["type"] = _classify(deg)
