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

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from affine import Affine


# Node type labels (``docs/Schema.md`` → type ∈ {intersection, endpoint, bridged})
TYPE_INTERSECTION = "intersection"
TYPE_ENDPOINT = "endpoint"
TYPE_BRIDGED = "bridged"


def ensure_metric_transform(transform, crs, width: int, height: int):
    """Return an affine/CRS pair whose world coordinates are metres.

    Metre-based projected inputs pass through. Geographic and non-metre projected
    rasters get a locally linearised affine in their centre-point UTM zone,
    preventing degrees/feet from being labelled and pruned/healed as metres.
    """
    if transform is None or crs is None:
        return transform, crs

    from affine import Affine
    from pyproj import CRS, Transformer

    source_crs = CRS.from_user_input(crs)
    axes = source_crs.axis_info[:2]
    if (source_crs.is_projected and len(axes) == 2
            and all(abs(float(axis.unit_conversion_factor) - 1.0) < 1e-9 for axis in axes)):
        return transform, crs

    cx, cy = width / 2.0, height / 2.0
    sx, sy = transform * (cx, cy)
    lon, lat = Transformer.from_crs(source_crs, 4326, always_xy=True).transform(sx, sy)
    zone = max(1, min(60, int((lon + 180.0) // 6.0) + 1))
    metric_crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    project = Transformer.from_crs(source_crs, metric_crs, always_xy=True).transform

    x0, y0 = project(*transform * (cx, cy))
    xc, yc = project(*transform * (cx + 1.0, cy))
    xr, yr = project(*transform * (cx, cy + 1.0))
    a, d = xc - x0, yc - y0
    b, e = xr - x0, yr - y0
    metric_transform = Affine(a, b, x0 - a * cx - b * cy,
                              d, e, y0 - d * cx - e * cy)
    return metric_transform, metric_crs


def mask_to_skeleton(mask01: np.ndarray) -> np.ndarray:
    """Thin a binary {0,1} road mask to a 1-px-wide skeleton (bool array)."""
    from skimage.morphology import skeletonize

    if mask01.ndim != 2:
        raise ValueError("mask must be 2-D (H×W)")
    binary = np.asarray(mask01) > 0
    return skeletonize(binary)


def mask_to_skeleton_with_distance(mask01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Skeleton + a pixel distance-to-background field for road-width recovery.

    Returns ``(skeleton, distance)`` where ``distance[r, c]`` is the Euclidean
    distance from a road pixel to the nearest background pixel — i.e. the road
    **half-width** at the centreline. Sampled along each edge in
    :func:`skeleton_to_graph` to populate the optional ``width_m`` edge attribute.

    We keep ``skeletonize`` for the topology (the validated S3/S4/S5 graph) and
    take the distance transform of the mask separately, rather than swapping to
    ``medial_axis`` — same width signal, zero change to the graph structure.
    """
    import cv2

    if mask01.ndim != 2:
        raise ValueError("mask must be 2-D (H×W)")
    binary = (np.asarray(mask01) > 0).astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant")
    distance = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    return mask_to_skeleton(binary), distance


def _classify(degree: int) -> str:
    """Map a node's degree to its schema ``type``."""
    if degree == 1:
        return TYPE_ENDPOINT
    return TYPE_INTERSECTION  # degree ≥ 2 (sknw collapses pure degree-2 chains)


def skeleton_to_graph(
    skeleton: np.ndarray,
    transform: "Affine | None" = None,
    resolution_m: float = 1.0,
    distance: np.ndarray | None = None,
) -> "nx.MultiGraph":
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
    distance :
        Optional pixel distance-to-background field (``H×W``, from
        :func:`mask_to_skeleton_with_distance`). When given, each edge gets a
        ``width_m`` attribute = mean road half-width along its centreline × 2 ×
        pixel size. Purely additive — the topology is unchanged.

    Returns
    -------
    nx.MultiGraph
        Parallel skeleton branches (loops, dual carriageways) are kept as
        **distinct keyed edges** rather than collapsed to one — redundant
        alternate routes are exactly what the resilience metric rewards, so
        dropping them silently biased ``global_efficiency``. Each edge key is an
        ``int`` assigned in build order. Nodes carry ``x, y`` (metric),
        ``degree, type``; edges carry ``length_m`` (metres), ``geometry`` (a
        metric ``[[x, y], ...]`` polyline), ``is_bridged=False`` and (if
        ``distance`` given) ``width_m``. Reproject to lon/lat with
        :func:`reproject_graph_to_wgs84` before writing for the map.
    """
    import networkx as nx
    import sknw

    skel = np.asarray(skeleton).astype(np.uint16)
    # multi=True so parallel skeleton branches (loops, dual carriageways) survive
    # as keyed edges instead of being silently dropped inside sknw. The returned
    # MultiGraph preserves all of them; the old keep-shortest collapse biased
    # the headline resilience metric (A37).
    raw = sknw.build_sknw(skel, multi=True)

    # Area-equivalent pixel scale is rotation-invariant and gives one explicit
    # scalar width policy for the uncommon anisotropic grid.
    if transform is not None:
        pixel_size_m = math.sqrt(abs(float(transform.a * transform.e - transform.b * transform.d)))
    else:
        pixel_size_m = float(resolution_m)
    dist = np.asarray(distance) if distance is not None else None

    def pixel_to_metric(row: float, col: float) -> tuple[float, float]:
        """Pixel (row, col) → metric world (x, y), or scaled pixels if no grid."""
        if transform is not None:
            x, y = transform * (col + 0.5, row + 0.5)  # affine takes (col, row)
            return float(x), float(y)
        return col * resolution_m, row * resolution_m

    def edge_width_m(pts: np.ndarray) -> float | None:
        """Mean road width (metres) sampled from the distance field along ``pts``."""
        if dist is None or len(pts) == 0:
            return None
        rows = np.clip(np.round(pts[:, 0]).astype(int), 0, dist.shape[0] - 1)
        cols = np.clip(np.round(pts[:, 1]).astype(int), 0, dist.shape[1] - 1)
        half_width_px = float(dist[rows, cols].mean())  # distance-to-edge = half width
        return round(2.0 * half_width_px * pixel_size_m, 3)

    graph = nx.MultiGraph()
    for node_id, data in raw.nodes(data=True):
        row, col = float(data["o"][0]), float(data["o"][1])  # sknw 'o' = (y, x)
        mx, my = pixel_to_metric(row, col)
        graph.add_node(int(node_id), x=mx, y=my)

    # Count parallel branches (same junction pair, ≥2 keyed edges) for a build
    # report — kept (not collapsed) in the MultiGraph, but surfaced so the
    # heal/simplify reports stay honest about how many survived.
    from collections import defaultdict
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    next_node_id = max(graph.nodes, default=-1) + 1
    rings_preserved = 0
    self_loop_dropped = 0
    for u, v, data in raw.edges(data=True):
        ui, vi = int(u), int(v)
        pts = np.asarray(data["pts"], dtype=float)  # (row, col) polyline
        metric = [list(pixel_to_metric(r, c)) for r, c in pts]

        if len(metric) < 2:
            self_loop_dropped += int(ui == vi)
            continue

        u_xy = [float(graph.nodes[ui]["x"]), float(graph.nodes[ui]["y"])]
        v_xy = [float(graph.nodes[vi]["x"]), float(graph.nodes[vi]["y"])]
        if ui != vi:
            forward = np.hypot(*(np.asarray(metric[0]) - u_xy)) + np.hypot(
                *(np.asarray(metric[-1]) - v_xy)
            )
            reverse = np.hypot(*(np.asarray(metric[0]) - v_xy)) + np.hypot(
                *(np.asarray(metric[-1]) - u_xy)
            )
            if reverse < forward:
                metric.reverse()
                pts = pts[::-1]
            metric[0], metric[-1] = u_xy, v_xy
        else:
            metric[0] = metric[-1] = u_xy

        arr = np.asarray(metric, dtype=float)
        segment_lengths = np.hypot(*np.diff(arr, axis=0).T)
        length_m = float(segment_lengths.sum())
        width = edge_width_m(pts)

        def add_route(a: int, b: int, geometry: list[list[float]], length: float) -> None:
            attrs = {"length_m": length, "geometry": geometry, "is_bridged": False}
            if width is not None:
                attrs["width_m"] = width
            graph.add_edge(a, b, **attrs)
            pair_counts[(min(a, b), max(a, b))] += 1

        if ui != vi:
            add_route(ui, vi, metric, length_m)
            continue

        if length_m <= 0.0:
            self_loop_dropped += 1
            continue

        # A positive sknw self-loop is a real closed road. Split it at half its
        # route length into two keyed edges so shortest paths can traverse either
        # side of the ring without retaining a degenerate self-loop.
        halfway = length_m / 2.0
        cumulative = np.cumsum(segment_lengths)
        segment = int(np.searchsorted(cumulative, halfway, side="left"))
        before = float(cumulative[segment - 1]) if segment else 0.0
        fraction = (halfway - before) / float(segment_lengths[segment])
        midpoint = (arr[segment] + fraction * (arr[segment + 1] - arr[segment])).tolist()
        first = [point.copy() for point in metric[:segment + 1]] + [midpoint]
        second = [midpoint] + [point.copy() for point in metric[segment + 1:]]
        split_node = next_node_id
        next_node_id += 1
        graph.add_node(split_node, x=float(midpoint[0]), y=float(midpoint[1]))
        add_route(ui, split_node, first, halfway)
        add_route(split_node, ui, second, length_m - halfway)
        rings_preserved += 1

    parallel_kept = sum(c - 1 for c in pair_counts.values() if c > 1)
    if parallel_kept or rings_preserved or self_loop_dropped:
        msg = f"[skeleton_graph] built MultiGraph: {parallel_kept} parallel branch(es) kept"
        if rings_preserved:
            msg += f", {rings_preserved} closed ring(s) preserved"
        if self_loop_dropped:
            msg += f", {self_loop_dropped} degenerate self-loop(s) dropped"
        print(msg)

    _annotate_degree_and_type(graph)
    return graph


def build_metric_to_pixel(transform: "Affine | None" = None, resolution_m: float = 1.0):
    """Return the exact inverse of :func:`skeleton_to_graph`'s ``pixel_to_metric``.

    Healing's probability-map corridor check needs to look up the
    P1 prob raster at points sampled along a candidate bridge's metric geometry —
    this maps a bridge point ``(x, y)`` back to the ``(row, col)`` pixel it came
    from, using the same ``transform``/``resolution_m`` the graph was built with.
    """
    if transform is not None:
        def metric_to_pixel(x: float, y: float) -> tuple[float, float]:
            col_half, row_half = ~transform * (x, y)  # inverse of transform * (col+.5, row+.5)
            return row_half - 0.5, col_half - 0.5
    else:
        def metric_to_pixel(x: float, y: float) -> tuple[float, float]:
            return y / resolution_m, x / resolution_m
    return metric_to_pixel


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


def prune_degenerate_edges(graph: "nx.Graph", min_edge_len_m: float) -> int:
    """Drop self-loops and sub-``min_edge_len_m`` edges; remove orphaned nodes.

    Skeletonisation can emit zero/near-zero-length edges (self-loops at junction
    pixels, single-pixel spurs). Left in, their tiny ``length_m`` weights skew the
    weighted shortest-path metrics (betweenness, global efficiency). Returns the
    number of edges removed. Run before healing, then re-annotate degree/type.
    """
    # Drop self-loops and sub-threshold branches. MultiGraph edges are addressed
    # by (u, v, key); a plain Graph by (u, v). ``keys=True`` is only valid on a
    # MultiGraph, so branch on the graph type.
    multi = graph.is_multigraph()
    if multi:
        to_remove = [
            (u, v, k)
            for u, v, k, data in graph.edges(data=True, keys=True)
            if u == v or float(data.get("length_m", 0.0)) < min_edge_len_m
        ]
    else:
        to_remove = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if u == v or float(data.get("length_m", 0.0)) < min_edge_len_m
        ]
    graph.remove_edges_from(to_remove)
    graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])
    _annotate_degree_and_type(graph)
    return len(to_remove)
