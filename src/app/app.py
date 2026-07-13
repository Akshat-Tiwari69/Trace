"""Interactive Streamlit dashboard for road-network resilience."""

from dataclasses import dataclass
import hashlib
import io
from itertools import combinations
import json
import logging
from math import inf, isfinite
import os
from pathlib import Path
import time
import urllib.error

import branca.colormap as cm
import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from PIL import Image

from src.app.modal_client import (
    MODAL_SEG_URL,
    EndpointBusyError,
    call as _call_modal_seg,
)
from src.pipeline.p3_analysis.resilience import resilience_index

# Refuse to decode anything above 4096x4096 px (PIL decompression-bomb guard
# for the public upload path; PIL errors at 2x this pixel count).
Image.MAX_IMAGE_PIXELS = 4096 * 4096

# Single source of truth for every colour in the app: the CSS theme, the map
# legend, edge/marker styling, the charts, and the PNG export all read from
# here. The ramp_* stops are cividis clipped to its upper range (0.3-1.0),
# colourblind-safe and legible on dark tiles; precomputed once from
# matplotlib.cm.cividis(np.linspace(0.3, 1.0, 4)) and hardcoded.
TOKENS: dict[str, str] = {
    "bg": "#0B1220",
    "surface": "#121C30",
    "surface_2": "#16233B",
    "border": "#24334E",
    "border_strong": "#33456A",
    "text": "#E2E8F0",
    "muted": "#8FA3BF",
    "amber": "#F59E0B",
    "amber_deep": "#D97706",
    "blue": "#38BDF8",
    "white": "#FFFFFF",
    "selected": "#56B4E9",
    "disabled": "#D55E00",
    "reroute": "#E69F00",
    "spof": "#FB7185",
    "ramp_0": "#4F576C",
    "ramp_1": "#848279",
    "ramp_2": "#C0B16A",
    "ramp_3": "#FEE838",
}

SINGLE_MODE = "Single junction"
FLOOD_MODE = "Flood area / junction set"


def find_repo_root() -> Path:
    """Find the repository root using its Tracker file as a stable marker."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "docs" / "Tracker.md").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


REPO_ROOT = find_repo_root()
SAMPLE_GEOJSON = REPO_ROOT / "data" / "sample" / "panaji_demo_graph.geojson"
SAMPLE_CRITICALITY = REPO_ROOT / "data" / "sample" / "panaji_demo_criticality.csv"

# Cache fingerprint derived from the data source, threaded through the cached
# graph builder and ablation so a future second dataset can't silently serve
# Panaji results from a stale cache entry.
DATA_FINGERPRINT = SAMPLE_GEOJSON.relative_to(REPO_ROOT).as_posix()


def _truthy(value: object) -> bool:
    """Interpret a GeoJSON/CSV flag as a bool — robust to bool, str, or NaN."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return value is True or value == 1


@dataclass(frozen=True)
class RouteResult:
    """Representative route before and after a junction failure."""

    origin: int
    destination: int
    baseline_path: tuple[int, ...]
    rerouted_path: tuple[int, ...] | None
    baseline_length_m: float
    rerouted_length_m: float | None
    travel_time_delta_pct: float
    delay_segments: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SimulationResult:
    """Metrics and route state produced by a node/area ablation (1+ nodes)."""

    disabled_nodes: tuple[int, ...]
    resilience_index: float
    largest_cc_fraction: float
    route: RouteResult | None


@st.cache_data
def load_sample_data() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Load and validate the sample artifacts defined in Tracker section 4."""
    features = gpd.read_file(SAMPLE_GEOJSON)
    criticality = pd.read_csv(SAMPLE_CRITICALITY)

    required_features = {
        "feature_type",
        "geometry",
        "node_id",
        "u",
        "v",
        "length_m",
        "is_bridged",
    }
    required_criticality = {"node_id", "betweenness", "rank", "is_critical"}
    if not required_features.issubset(features.columns):
        missing = required_features - set(features.columns)
        raise ValueError(f"GeoJSON is missing columns: {sorted(missing)}")
    if not required_criticality.issubset(criticality.columns):
        missing = required_criticality - set(criticality.columns)
        raise ValueError(f"Criticality CSV is missing columns: {sorted(missing)}")

    return features, criticality


@st.cache_data
def load_resilience_curve() -> pd.DataFrame | None:
    """Load the resilience degradation curve (P3 contract) if available."""
    path = REPO_ROOT / "data" / "sample" / "panaji_demo_resilience.csv"
    if path.is_file():
        curve = pd.read_csv(path)
        ri_columns = {"targeted_resilience_index", "random_resilience_index"}
        if not ri_columns.issubset(curve.columns):
            return None
        values = curve[list(ri_columns)].apply(pd.to_numeric, errors="coerce")
        if (
            not np.isfinite(values.to_numpy()).all()
            or not values.ge(0).all().all()
            or not values.le(1).all().all()
        ):
            return None
        return curve
    return None


def split_features(
    features: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return validated node and edge feature tables."""
    nodes = features[features["feature_type"] == "node"]
    edges = features[features["feature_type"] == "edge"]
    if nodes.empty or edges.empty:
        raise ValueError("Sample GeoJSON must contain node and edge features")
    if nodes["node_id"].isna().any() or edges[["u", "v", "length_m"]].isna().any().any():
        raise ValueError("Sample GeoJSON contains incomplete graph features")
    return nodes, edges


@st.cache_resource(show_spinner="Building routable graph...")
def graph_from_features(fingerprint: str, _features: gpd.GeoDataFrame) -> nx.MultiGraph:
    """Convert the map-ready GeoJSON features into a routable graph.

    ``fingerprint`` identifies the data source in the cache key.
    ``_features`` is underscore-prefixed so Streamlit skips hashing the
    (unhashable) GeoDataFrame for the cache key — required under cache_resource.

    MultiGraph (not Graph): the sample GeoJSON can carry parallel edges between
    the same (u, v) — loops, dual carriageways — and ``add_edge`` on a plain
    Graph silently overwrites them, collapsing the redundancy the resilience
    metric is meant to reward (A37 MultiGraph contract).
    """
    nodes, edges = split_features(_features)
    graph = nx.MultiGraph()
    for _, node in nodes.iterrows():
        node_id = int(node["node_id"])
        graph.add_node(
            node_id,
            x=float(node.geometry.x),
            y=float(node.geometry.y),
            betweenness=float(node.get("betweenness", 0.0)),
            is_critical=_truthy(node.get("is_critical")),
            is_articulation=_truthy(node.get("is_articulation")),
        )
    for _, edge in edges.iterrows():
        graph.add_edge(
            int(edge["u"]),
            int(edge["v"]),
            length_m=float(edge["length_m"]),
            is_bridged=_truthy(edge.get("is_bridged")),
            is_bridge=_truthy(edge.get("is_bridge")),
            coordinates=tuple(
                (float(longitude), float(latitude))
                for longitude, latitude in edge.geometry.coords
            ),
        )
    return graph


def _min_edge_data(graph: nx.MultiGraph, u: int, v: int) -> dict:
    """Return the shortest (``length_m``-minimal) parallel edge's data for (u, v).

    ``nx.shortest_path(weight="length_m")`` routes over the min-weight parallel
    edge between a pair of nodes, so anything reconstructing that route's length
    or geometry must agree on the *same* edge — otherwise a longer/shorter
    parallel twin (loop, dual carriageway) would silently mismatch the path
    Dijkstra actually chose (A37 MultiGraph contract).
    """
    return min(graph[u][v].values(), key=lambda data: float(data["length_m"]))


def path_length(graph: nx.MultiGraph, path: tuple[int, ...] | list[int]) -> float:
    """Return a path's total length in metres."""
    return sum(
        float(_min_edge_data(graph, start, end)["length_m"])
        for start, end in zip(path, path[1:])
    )


def edge_key(start: int, end: int) -> tuple[int, int]:
    """Return an order-independent edge identifier."""
    return min(start, end), max(start, end)


def representative_reroute(graph: nx.MultiGraph, disabled_node: int) -> RouteResult | None:
    """Find the most affected finite detour across a disabled junction."""
    neighbours = list(graph.neighbors(disabled_node))
    if len(neighbours) < 2:
        return None

    perturbed = graph.copy()
    perturbed.remove_node(disabled_node)
    finite_candidates: list[RouteResult] = []
    disconnected_candidate: RouteResult | None = None

    for origin, destination in combinations(neighbours, 2):
        baseline_path = tuple(
            nx.shortest_path(graph, origin, destination, weight="length_m")
        )
        if disabled_node not in baseline_path:
            continue
        baseline_length = path_length(graph, baseline_path)
        if baseline_length <= 0:
            continue

        try:
            rerouted_path = tuple(
                nx.shortest_path(perturbed, origin, destination, weight="length_m")
            )
        except nx.NetworkXNoPath:
            disconnected_candidate = RouteResult(
                origin=origin,
                destination=destination,
                baseline_path=baseline_path,
                rerouted_path=None,
                baseline_length_m=baseline_length,
                rerouted_length_m=None,
                travel_time_delta_pct=inf,
                delay_segments=(),
            )
            continue

        rerouted_length = path_length(perturbed, rerouted_path)
        travel_delta = 100.0 * (rerouted_length / baseline_length - 1.0)
        baseline_edges = {
            edge_key(start, end) for start, end in zip(baseline_path, baseline_path[1:])
        }
        delay_segments = []
        for start, end in zip(rerouted_path, rerouted_path[1:]):
            if edge_key(start, end) not in baseline_edges:
                contribution = (
                    100.0 * float(_min_edge_data(graph, start, end)["length_m"]) / baseline_length
                )
                delay_segments.append((f"{start}–{end}", contribution))

        finite_candidates.append(
            RouteResult(
                origin=origin,
                destination=destination,
                baseline_path=baseline_path,
                rerouted_path=rerouted_path,
                baseline_length_m=baseline_length,
                rerouted_length_m=rerouted_length,
                travel_time_delta_pct=travel_delta,
                delay_segments=tuple(
                    sorted(delay_segments, key=lambda item: item[1], reverse=True)[:5]
                ),
            )
        )

    if finite_candidates:
        return max(finite_candidates, key=lambda route: route.travel_time_delta_pct)
    return disconnected_candidate


@st.cache_data(show_spinner="Simulating failure...")
def simulate_ablation(graph_fingerprint: str, _graph: nx.MultiGraph, nodes: tuple[int, ...]) -> SimulationResult:
    """Disable nodes and compute resilience plus a representative reroute (if single node)."""
    metrics = resilience_index(_graph, removed_nodes=list(nodes))
    route = representative_reroute(_graph, nodes[0]) if len(nodes) == 1 else None

    return SimulationResult(
        disabled_nodes=nodes,
        resilience_index=float(metrics["resilience_index"]),
        largest_cc_fraction=float(metrics["largest_cc_fraction"]),
        route=route,
    )


def semantic_legend() -> folium.Element:
    """Create a labelled map legend for semantic route states.

    Folds the criticality ramp in as a CSS gradient bar (A39 vector-layer
    cosmetic) so the encoding lives in one place instead of this HTML legend
    plus a separate branca colormap bar plastered on the map corner.
    """
    ramp_css = ", ".join(TOKENS[f"ramp_{i}"] for i in range(4))
    html = f"""
    <div style="position: fixed; bottom: 36px; right: 12px; z-index: 9999;
                background: rgba(13,21,38,.92); color: {TOKENS["text"]}; padding: 10px 14px;
                border: 1px solid {TOKENS["border"]}; border-radius: 10px; font-size: 12px;
                line-height: 1.75; font-family: 'Fira Sans', sans-serif;
                box-shadow: 0 8px 24px rgba(2,6,17,.5); backdrop-filter: blur(6px);">
      <b style="font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
                color: {TOKENS["muted"]};">Network states</b><br>
      <span style="color:{TOKENS["selected"]}">●</span> selected junction<br>
      <span style="color:{TOKENS["disabled"]}">●</span> / <span style="color:{TOKENS["disabled"]}">┄</span> disabled junction / links (dashed)<br>
      <b style="color:{TOKENS["reroute"]}; font-size: 14px;">━</b> rerouted path (thick solid)<br>
      <span style="color:{TOKENS["ramp_2"]}">┄</span> healed road (dashed = inferred)<br>
      <span style="color:{TOKENS["spof"]}">●</span> / <span style="color:{TOKENS["spof"]}">━</span> single-point-of-failure<br>
      <div style="margin-top: 6px;">
        <span style="font-size: 10.5px; letter-spacing: .04em; color: {TOKENS["muted"]};">
          Road criticality (low → high)
        </span>
        <div style="width: 150px; height: 10px; border-radius: 5px; margin-top: 3px;
                    border: 1px solid {TOKENS["border"]};
                    background: linear-gradient(to right, {ramp_css});"></div>
      </div>
    </div>
    """
    return folium.Element(html)


def add_rerouted_path(road_map: folium.Map, graph: nx.MultiGraph, route: RouteResult) -> None:
    """Draw the rerouted path last so its orange highlight stays visible."""
    if route.rerouted_path is None:
        return
    for start, end in zip(route.rerouted_path, route.rerouted_path[1:]):
        # Same min-length parallel edge the route was computed over, so the
        # drawn geometry matches the routed segment (see _min_edge_data).
        coordinates = _min_edge_data(graph, start, end)["coordinates"]
        folium.PolyLine(
            [(latitude, longitude) for longitude, latitude in coordinates],
            color=TOKENS["reroute"],
            weight=7,
            opacity=1.0,
            tooltip=f"Rerouted road {start}–{end}",
        ).add_to(road_map)


@st.cache_data(show_spinner=False)
def compute_edge_styles(
    fingerprint: str,
    _edges: gpd.GeoDataFrame,
    _scores: dict[int, float],
    _colour_scale: cm.LinearColormap,
    disabled_nodes: tuple[int, ...],
    show_healed: bool,
    show_spof: bool,
) -> gpd.GeoDataFrame:
    """Vectorized per-edge style columns for the single GeoJson road layer.

    Replaces the old per-edge ``folium.PolyLine`` loop (A39 vector-layer optimization): what
    actually dominated rerun time at scale was instantiating ~500 individual
    Folium objects (each its own Jinja template + id), not the score→colour
    lookup — so this builds one styled GeoDataFrame via pandas/numpy ops
    instead of a Python ``for`` loop, and the caller renders it as one
    ``folium.GeoJson`` layer.

    Cache-keyed on ``(fingerprint, disabled_nodes, show_healed, show_spof)``:
    ``_edges``/``_scores``/``_colour_scale`` are pinned to the data
    fingerprint (same invariant ``simulate_ablation``/``generate_geojson_export``
    already rely on elsewhere in this file), so they're safe to leave
    unhashed. ``disabled_nodes`` is threaded through as an explicit hashable
    argument — not read from session state inside this function — so a new
    ablation always invalidates the cache instead of risking a stale style.
    """
    edges = _edges
    is_bridged = (
        edges["is_bridged"].map(_truthy)
        if "is_bridged" in edges.columns
        else pd.Series(False, index=edges.index)
    )
    if not show_healed:
        edges = edges.loc[~is_bridged]
        is_bridged = is_bridged.loc[edges.index]

    is_bridge = (
        edges["is_bridge"].map(_truthy)
        if "is_bridge" in edges.columns
        else pd.Series(False, index=edges.index)
    )
    start = edges["u"].astype(int)
    end = edges["v"].astype(int)
    score = np.maximum(
        start.map(_scores).fillna(0.0).astype(float).to_numpy(),
        end.map(_scores).fillna(0.0).astype(float).to_numpy(),
    )
    is_disabled = (start.isin(disabled_nodes) | end.isin(disabled_nodes)).to_numpy()
    is_spof = (is_bridge & show_spof).to_numpy()
    is_bridged_arr = is_bridged.to_numpy()

    # Precedence mirrors the original if/elif chain exactly: disabled beats
    # spof beats the criticality ramp for colour/state; spof beats
    # disabled-or-bridged for weight; disabled beats spof for opacity.
    colour = np.select(
        [is_disabled, is_spof],
        [TOKENS["disabled"], TOKENS["spof"]],
        default=pd.Series(score).map(_colour_scale).to_numpy(),
    )
    state = np.select(
        [is_disabled, is_spof, is_bridged_arr],
        ["disabled link", "critical bridge", "healed link"],
        default="observed link",
    )
    weight = np.select(
        [is_spof, is_disabled | is_bridged_arr],
        [5, 4],
        default=3,
    )
    # Per-edge confidence (A39, only present when P1 persisted a probability
    # map): a low-confidence edge is drawn fainter so an occluded/uncertain
    # road doesn't read as equally "observed" as a clean detection. Clipped to
    # [0.35, 1.0] — never fully invisible, since a faint road is still real
    # information the user shouldn't lose entirely. Missing confidence values
    # (e.g. an edge added after the column existed) default to full trust (1.0)
    # rather than fading an edge we have no reason to doubt. Disabled/spof
    # states keep their own fixed opacity — that channel already carries a
    # different meaning (simulation state / bridge criticality) and shouldn't
    # be diluted by detection confidence.
    if "confidence" in edges.columns:
        confidence = edges["confidence"].astype(float).fillna(1.0).to_numpy()
        default_opacity = np.clip(confidence, 0.35, 1.0)
    else:
        default_opacity = 0.85
    opacity = np.select(
        [is_disabled, is_spof],
        [0.45, 0.95],
        default=default_opacity,
    )
    dash_array = np.where(is_bridged_arr | is_disabled, "8 6", None)

    styled = edges[["geometry"]].copy()
    styled["u"] = start.to_numpy()
    styled["v"] = end.to_numpy()
    styled["score"] = np.round(score, 3)
    styled["state"] = state
    styled["color"] = colour
    styled["weight"] = weight.astype(int)
    styled["opacity"] = opacity
    styled["dash_array"] = dash_array
    return gpd.GeoDataFrame(styled, geometry="geometry", crs=edges.crs)


def _edge_style_function(feature: dict) -> dict:
    """Per-feature Leaflet style read from the pre-computed style columns."""
    props = feature["properties"]
    style = {
        "color": props["color"],
        "weight": props["weight"],
        "opacity": props["opacity"],
    }
    if props.get("dash_array"):
        style["dashArray"] = props["dash_array"]
    return style


def build_map(
    features: gpd.GeoDataFrame,
    criticality: pd.DataFrame,
    graph: nx.MultiGraph,
    selected_node: int,
    simulation: SimulationResult | None,
    show_critical: bool,
    show_healed: bool,
    show_spof: bool,
    failure_mode: str,
    center: list[float] | None = None,
    zoom: float | None = None,
    minimal: bool = False,
) -> folium.Map:
    """Build the map with criticality, selection, failure, and reroute states.

    ``center``/``zoom`` (when both given) restore the user's last viewport so the
    map doesn't snap back to the whole-network bounds on every rerun — the
    canonical streamlit-folium round-trip (A39 viewport persistence).

    ``minimal`` (A39 Briefing tab): a cheaper, chrome-less variant —
    single tile layer, no layer control, no draw tool, no full legend (ramp
    only), and only critical + disabled junctions as dots (ignores the
    show_critical/show_spof toggles and selected_node, which the Briefing tab
    doesn't expose). Edge styling/rerouting is unchanged — that's the
    criticality/simulation story itself, not chrome.
    """
    nodes, edges = split_features(features)
    scores = criticality.set_index("node_id")["betweenness"].to_dict()
    maximum = max(float(criticality["betweenness"].max()), 1e-9)
    # Colourblind-safe ramp: cividis clipped to its upper range (0.3-1.0) so
    # low-criticality roads stay visible against the dark tiles (see TOKENS).
    colour_scale = cm.LinearColormap(
        colors=[TOKENS["ramp_0"], TOKENS["ramp_1"], TOKENS["ramp_2"], TOKENS["ramp_3"]],
        vmin=0.0,
        vmax=maximum,
        caption="Road criticality (endpoint betweenness: low to high)",
    )
    disabled_nodes = simulation.disabled_nodes if simulation else ()

    road_map = folium.Map(
        location=center or [nodes.geometry.y.mean(), nodes.geometry.x.mean()],
        zoom_start=zoom or 15,  # fallback if the bounds below are degenerate
        tiles=None,
        control_scale=True,
    )
    # Only auto-fit to the whole network when we have no remembered viewport —
    # otherwise honour the user's last pan/zoom (A39 viewport persistence).
    if center is None or zoom is None:
        min_x, min_y, max_x, max_y = (float(value) for value in nodes.total_bounds)
        if all(isfinite(value) for value in (min_x, min_y, max_x, max_y)) and (
            min_x < max_x or min_y < max_y
        ):
            road_map.fit_bounds([[min_y, min_x], [max_y, max_x]])
    folium.TileLayer("CartoDB dark_matter", name="Dark map").add_to(road_map)
    if not minimal:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri — Source: Esri, Maxar, Earthstar Geographics",
            name="Satellite",
        ).add_to(road_map)

    if not minimal and failure_mode == FLOOD_MODE:
        from folium.plugins import Draw
        Draw(
            export=False,
            position='topleft',
            draw_options={
                'polyline': False,
                'rectangle': True,
                'circle': False,
                'circlemarker': False,
                'marker': False,
                'polygon': True
            },
            edit_options={'poly': {'allowIntersection': False}}
        ).add_to(road_map)

    # Single vectorized GeoJson layer (A39) replaces one
    # folium.PolyLine per edge — style columns are recomputed from the live
    # simulation/toggle state on every call (see compute_edge_styles), so
    # disabled/rerouted state never goes stale.
    styled_edges = compute_edge_styles(
        DATA_FINGERPRINT, edges, scores, colour_scale, disabled_nodes, show_healed, show_spof,
    )
    folium.GeoJson(
        styled_edges,
        name="Road network",
        style_function=_edge_style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["u", "v", "score", "state"],
            aliases=["From junction", "To junction", "Criticality", "State"],
            localize=True,
        ),
    ).add_to(road_map)

    critical_ids = set(criticality.loc[criticality["is_critical"].map(_truthy), "node_id"].astype(int))
    articulation_ids = set()
    if "is_articulation" in criticality.columns:
        articulation_ids = set(criticality.loc[criticality["is_articulation"].map(_truthy), "node_id"].astype(int))

    nodes_to_show = set()
    if minimal:
        nodes_to_show.update(critical_ids)
    else:
        if show_critical:
            nodes_to_show.update(critical_ids)
        if show_spof:
            nodes_to_show.update(articulation_ids)
        nodes_to_show.add(selected_node)
    nodes_to_show.update(disabled_nodes)

    if nodes_to_show:
        # Plain CircleMarkers in a named group: clustering hid the ~36 markers
        # at city zoom and left an unnamed entry in the layer control.
        marker_group = folium.FeatureGroup(name="Critical junctions")
        for _, node in nodes[nodes["node_id"].isin(nodes_to_show)].iterrows():
            node_id = int(node["node_id"])
            score = float(scores.get(node_id, 0.0))
            is_art = node_id in articulation_ids
            if node_id in disabled_nodes:
                colour, radius, label = TOKENS["disabled"], 9, "Disabled junction"
            elif node_id == selected_node:
                colour, radius, label = TOKENS["selected"], 8, "Selected junction"
            elif is_art and show_spof:
                colour, radius, label = TOKENS["spof"], 7, "Articulation point"
            else:
                colour, radius, label = colour_scale(score), 5, "Critical junction"

            folium.CircleMarker(
                location=[float(node.geometry.y), float(node.geometry.x)],
                radius=radius,
                color=TOKENS["white"],
                weight=2 if node_id == selected_node or node_id in disabled_nodes else 1,
                fill=True,
                fill_color=colour,
                fill_opacity=1.0,
                tooltip=f"{label} {node_id} · score {score:.3f}",
            ).add_to(marker_group)
        marker_group.add_to(road_map)

    if simulation and simulation.route:
        add_rerouted_path(road_map, graph, simulation.route)
    if minimal:
        road_map.get_root().html.add_child(ramp_legend())
    else:
        folium.LayerControl(position="topright").add_to(road_map)
        # The criticality ramp is now folded into semantic_legend() as a CSS
        # gradient bar (A39) — no separate branca bar.
        road_map.get_root().html.add_child(semantic_legend())
    return road_map


def ramp_legend() -> folium.Element:
    """Chrome-less legend for the A39 Briefing tab: ramp only, no
    network-states list — the Briefing map has no disabled/reroute toggles to
    explain beyond what the single "Simulate the worst failure" button does."""
    ramp_css = ", ".join(TOKENS[f"ramp_{i}"] for i in range(4))
    html = f"""
    <div style="position: fixed; bottom: 20px; right: 12px; z-index: 9999;
                background: rgba(13,21,38,.92); color: {TOKENS["text"]}; padding: 8px 12px;
                border: 1px solid {TOKENS["border"]}; border-radius: 10px; font-size: 11px;
                font-family: 'Fira Sans', sans-serif; box-shadow: 0 8px 24px rgba(2,6,17,.5);">
      <span style="font-size: 10px; letter-spacing: .04em; color: {TOKENS["muted"]};">
        Road criticality (low → high)
      </span>
      <div style="width: 140px; height: 9px; border-radius: 5px; margin-top: 3px;
                  border: 1px solid {TOKENS["border"]};
                  background: linear-gradient(to right, {ramp_css});"></div>
    </div>
    """
    return folium.Element(html)


def nearest_critical_node(
    nodes: gpd.GeoDataFrame,
    critical_ids: set[int],
    clicked: dict | None,
) -> int | None:
    """Resolve a Folium click to a nearby critical junction."""
    if not clicked or "lat" not in clicked or "lng" not in clicked:
        return None
    candidates = nodes[nodes["node_id"].isin(critical_ids)].copy()
    if candidates.empty:
        return None
    candidates["distance"] = (
        (candidates.geometry.y - float(clicked["lat"])) ** 2
        + (candidates.geometry.x - float(clicked["lng"])) ** 2
    )
    nearest = candidates.loc[candidates["distance"].idxmin()]
    if float(nearest["distance"]) > 0.0003**2:
        return None
    return int(nearest["node_id"])


def render_charts(simulation: SimulationResult) -> None:
    """Render Design.md's delay-contributor chart for the active reroute."""
    route = simulation.route
    if route is None:
        if len(simulation.disabled_nodes) > 1:
            st.info("Area-based flood active. Single-path rerouting analysis is disabled.")
        else:
            st.warning("This junction has no through-route to reroute.")
        return
    if route.rerouted_path is None:
        st.error(
            f"No alternate route remains between junctions {route.origin} and "
            f"{route.destination}."
        )
        return

    if route.delay_segments:
        st.caption("Top delay contributors on the detour")
        delays = pd.DataFrame(
            route.delay_segments,
            columns=["Road", "Delay contribution (%)"],
        ).set_index("Road")
        st.bar_chart(delays, color=TOKENS["reroute"], height=190)


@st.cache_data(show_spinner=False)
def generate_geojson_export(
    fingerprint: str,
    _features: gpd.GeoDataFrame,
    disabled_nodes: tuple[int, ...],
) -> str:
    """Generate GeoJSON of the current network state (cached per ablation)."""
    export_df = _features.copy()
    if disabled_nodes:
        is_disabled_node = (export_df["feature_type"] == "node") & (export_df["node_id"].isin(disabled_nodes))
        export_df.loc[is_disabled_node, "status"] = "disabled"
        is_disabled_edge = (export_df["feature_type"] == "edge") & ((export_df["u"].isin(disabled_nodes)) | (export_df["v"].isin(disabled_nodes)))
        export_df.loc[is_disabled_edge, "status"] = "disabled"
    return export_df.to_json()


@st.cache_data(show_spinner=False)
def generate_summary_png(
    simulation: SimulationResult | None,
    critical_nodes: pd.DataFrame,
    resilience_curve: pd.DataFrame | None
) -> bytes:
    """Generate a high-res PNG summary of the current state."""
    fig = Figure(figsize=(10, 8))
    fig.patch.set_facecolor(TOKENS["bg"])

    fig.suptitle("Route Resilience - Network Summary", color=TOKENS["text"], fontsize=20, y=0.95)

    ax_metrics = fig.add_subplot(2, 2, 1)
    ax_metrics.axis('off')
    ax_metrics.set_facecolor(TOKENS["bg"])

    ri = simulation.resilience_index if simulation else 1.0
    ax_metrics.text(0.1, 0.7, f"Resilience Index:\n{ri:.3f}", color=TOKENS["text"], fontsize=18, fontweight='bold')

    if simulation and simulation.route:
        delay = f"+{simulation.route.travel_time_delta_pct:.1f}%"
    elif simulation and len(simulation.disabled_nodes) > 1:
        delay = "N/A"
    else:
        delay = "0.0%"
    ax_metrics.text(0.1, 0.3, f"Travel Time Impact:\n{delay}", color=TOKENS["reroute"], fontsize=18, fontweight='bold')

    ax_table = fig.add_subplot(2, 2, 2)
    ax_table.axis('off')
    top5 = critical_nodes.head(5)[["node_id", "betweenness"]].copy()
    top5["betweenness"] = top5["betweenness"].round(3)
    table = ax_table.table(
        cellText=top5.values,
        colLabels=["Node ID", "Score"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(TOKENS["surface"] if row > 0 else TOKENS["border_strong"])
        cell.set_text_props(color=TOKENS["text"])
        cell.set_edgecolor(TOKENS["bg"])

    if resilience_curve is not None:
        ax_curve = fig.add_subplot(2, 1, 2)
        ax_curve.set_facecolor(TOKENS["bg"])
        ax_curve.tick_params(colors=TOKENS["text"])
        for spine in ax_curve.spines.values():
            spine.set_color(TOKENS["text"])

        curve_data = resilience_curve.set_index("n_removed")
        if "targeted_resilience_index" in curve_data.columns and "random_resilience_index" in curve_data.columns:
            ax_curve.plot(curve_data.index, curve_data["targeted_resilience_index"], color=TOKENS["disabled"], label='Targeted')
            ax_curve.plot(curve_data.index, curve_data["random_resilience_index"], color=TOKENS["selected"], label='Random')
            ax_curve.legend(facecolor=TOKENS["surface"], edgecolor=TOKENS["text"], labelcolor=TOKENS["text"])

        ax_curve.set_xlabel("Nodes Removed", color=TOKENS["text"])
        ax_curve.set_ylabel("Resilience Index", color=TOKENS["text"])
        ax_curve.set_title("Resilience Degradation Curve", color=TOKENS["text"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    return buf.getvalue()


def _on_failure_mode_change() -> None:
    """Clear flood-area ablations when the failure mode moves away from flood."""
    if (
        st.session_state.get("failure_mode") != FLOOD_MODE
        and st.session_state.get("ablation_source") == "flood"
    ):
        st.session_state["disabled_nodes"] = ()
        st.session_state["ablation_source"] = None


def _clear_last_map_click() -> None:
    """Forget the last map click so re-clicking the same spot works after a manual pick."""
    st.session_state["last_map_click"] = None


def _run_top_chokepoint_demo(critical_nodes: pd.DataFrame) -> None:
    """Shared A39 demo-CTA logic: disable the #1-ranked chokepoint.

    Used by both the Briefing tab's "Simulate the worst failure" button and the
    Analysis tab's "Run demo" button so the ablation itself lives in one place.
    """
    st.session_state["disabled_nodes"] = (int(critical_nodes.iloc[0]["node_id"]),)
    st.session_state["ablation_source"] = "single"
    st.rerun()


def _reset_simulation() -> None:
    """Reset state shared by the Analysis and Briefing tabs (A39)."""
    st.session_state["disabled_nodes"] = ()
    st.session_state["ablation_source"] = None
    st.session_state["reset_counter"] = st.session_state.get("reset_counter", 0) + 1
    st.session_state.pop("map_center", None)  # reframe to the whole network
    st.session_state.pop("map_zoom", None)
    st.rerun()


def _explain_failure(simulation: SimulationResult) -> str:
    """One-sentence plain-language explanation of an ablation (A39 Briefing tab)."""
    node = simulation.disabled_nodes[0]
    efficiency_loss = (1.0 - simulation.resilience_index) * 100
    if simulation.largest_cc_fraction < 0.99:
        return (
            f"Losing junction {node} splits the network — "
            f"{1 - simulation.largest_cc_fraction:.0%} of junctions become unreachable."
        )
    route = simulation.route
    if route and route.rerouted_path and isfinite(route.travel_time_delta_pct):
        return (
            f"Losing junction {node} cuts network efficiency by {efficiency_loss:.0f}% "
            f"and forces a {route.travel_time_delta_pct:.0f}% longer detour on the worst-hit route."
        )
    return f"Losing junction {node} cuts network efficiency by {efficiency_loss:.0f}%."


def render_briefing(
    features: gpd.GeoDataFrame,
    criticality: pd.DataFrame,
    graph: nx.MultiGraph,
    simulation: SimulationResult | None,
) -> None:
    """A39 Briefing tab: the judge/demo persona's one-screen pitch.

    Full-width hero, chrome-less map (build_map(minimal=True)), three headline
    metrics, and a single button that reuses the Analysis tab's demo-CTA logic
    (_run_top_chokepoint_demo) rather than duplicating the ablation.
    """
    nodes, edges = split_features(features)
    critical_nodes = criticality[criticality["is_critical"].map(_truthy)].sort_values("rank")

    st.subheader("See what one failure costs this city")
    st.caption(
        "Route Resilience extracts a routable road network from satellite imagery "
        "and shows which junctions the network can least afford to lose."
    )

    ri = simulation.resilience_index if simulation else 1.0
    metric_junctions, metric_links, metric_ri = st.columns(3)
    metric_junctions.metric("Junctions", f"{len(nodes):,}")
    metric_links.metric("Road links", f"{len(edges):,}")
    metric_ri.metric(
        "Resilience Index",
        f"{ri:.3f}",
        delta=f"{(ri - 1.0) * 100:.1f}%" if simulation else None,
        delta_color="normal",
        help="Global efficiency after failure divided by baseline global efficiency.",
    )

    selected_for_map = (
        simulation.disabled_nodes[0] if simulation else int(critical_nodes.iloc[0]["node_id"])
    )
    briefing_map = build_map(
        features,
        criticality,
        graph,
        selected_for_map,
        simulation,
        show_critical=True,
        show_healed=True,
        show_spof=False,
        failure_mode=SINGLE_MODE,
        minimal=True,
    )
    # No returned_objects / viewport round-trip: the Briefing map is static
    # (fit-bounds every rerun), keyed only off the shared reset counter.
    reset_n = st.session_state.get("reset_counter", 0)
    st_folium(
        briefing_map,
        height=420,
        use_container_width=True,
        returned_objects=[],
        key=f"briefing_map_{reset_n}",
    )

    if simulation:
        st.success(_explain_failure(simulation))
        if st.button("Reset", key="briefing_reset"):
            _reset_simulation()
    elif st.button(
        "Simulate the worst failure",
        type="primary",
        use_container_width=True,
        help="Knock out the network's top-ranked chokepoint and see the impact.",
        key="briefing_simulate",
    ):
        _run_top_chokepoint_demo(critical_nodes)


def render_export_controls(
    features: gpd.GeoDataFrame,
    simulation: SimulationResult | None,
    critical_nodes: pd.DataFrame,
    resilience_curve: pd.DataFrame | None,
) -> None:
    """A39 export downloads, moved from the flat panel into a popover."""
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        geojson_data = generate_geojson_export(
            DATA_FINGERPRINT, features, simulation.disabled_nodes if simulation else ()
        )
        st.download_button(
            label="Download GeoJSON",
            data=geojson_data,
            file_name="network_state.geojson",
            mime="application/geo+json",
            use_container_width=True,
            help="Download the current map state including disabled nodes and rerouted edges."
        )
    with export_col2:
        png_data = generate_summary_png(simulation, critical_nodes, resilience_curve)
        st.download_button(
            label="Download Summary",
            data=png_data,
            file_name="resilience_summary.png",
            mime="image/png",
            use_container_width=True,
            help="Download a high-resolution PNG summarizing metrics, curve, and top critical nodes."
        )


def render_panel(
    features: gpd.GeoDataFrame,
    criticality: pd.DataFrame,
    simulation: SimulationResult | None,
    resilience_curve: pd.DataFrame | None = None,
) -> None:
    """A39 Analysis panel: metrics + export popover, then
    Scenario/Rankings/Curves sub-tabs. Reorganized from one flat panel — the
    logic in each sub-tab is unchanged, just relocated."""
    nodes, edges = split_features(features)
    critical_nodes = criticality[criticality["is_critical"].map(_truthy)].sort_values("rank")
    critical_ids = critical_nodes["node_id"].astype(int).tolist()
    scores = criticality.set_index("node_id")["betweenness"].to_dict()
    ranks = criticality.set_index("node_id")["rank"].to_dict()

    st.caption(
        "Route Resilience maps a city's road network from satellite imagery and "
        "shows which junctions the network can least afford to lose."
    )

    ri = simulation.resilience_index if simulation else 1.0
    route = simulation.route if simulation else None
    travel_delta = route.travel_time_delta_pct if route else 0.0
    ri_column, travel_column = st.columns(2)
    ri_column.metric(
        "Resilience Index",
        f"{ri:.3f}",
        delta=f"{(ri - 1.0) * 100:.1f}%" if simulation else None,
        delta_color="normal",
        help="Global efficiency after failure divided by baseline global efficiency.",
    )
    if simulation and len(simulation.disabled_nodes) > 1:
        travel_value = "N/A"
    else:
        travel_value = "Route cut" if simulation and not isfinite(travel_delta) else f"+{travel_delta:.1f}%"

    travel_column.metric(
        "Travel-time impact",
        travel_value,
        help="Exact route-length change at constant speed; no speed data is assumed.",
    )
    st.progress(
        min(max(ri, 0.0), 1.0),
        text=f"Network efficiency retained: {ri:.0%}",
    )
    if simulation:
        n_removed = len(simulation.disabled_nodes)
        ri_context = "Resilience Index scale: 1.00 = intact network."
        if (
            resilience_curve is not None
            and {"n_removed", "random_resilience_index"}.issubset(resilience_curve.columns)
        ):
            match = resilience_curve.loc[resilience_curve["n_removed"] == n_removed]
            if not match.empty:
                random_ri = float(match.iloc[0]["random_resilience_index"])
                ri_context = (
                    f"Removing {n_removed} junction(s) at random typically retains "
                    f"{random_ri:.0%} efficiency — this scenario retains {ri:.0%}."
                )
        st.caption(ri_context)

    with st.popover("Export", use_container_width=True):
        render_export_controls(features, simulation, critical_nodes, resilience_curve)

    scenario_tab, rankings_tab, curves_tab = st.tabs(["Scenario", "Rankings", "Curves"])
    with scenario_tab:
        render_scenario_tab(
            criticality, edges, critical_nodes, critical_ids, scores, ranks, simulation, route
        )
    with rankings_tab:
        render_rankings_tab(critical_nodes, nodes, edges)
    with curves_tab:
        render_curves_tab(resilience_curve)


def render_scenario_tab(
    criticality: pd.DataFrame,
    edges: gpd.GeoDataFrame,
    critical_nodes: pd.DataFrame,
    critical_ids: list[int],
    scores: dict[int, float],
    ranks: dict[int, float],
    simulation: SimulationResult | None,
    route: RouteResult | None,
) -> None:
    """A39 scenario tab: failure-mode picker, closures, toggles and status."""
    if st.button(
        "Run demo: disable the #1 chokepoint",
        type="primary",
        use_container_width=True,
        help="One click: knock out the top-ranked junction and watch the network react.",
    ):
        _run_top_chokepoint_demo(critical_nodes)

    st.subheader("Scenario controls")
    failure_mode = st.radio(
        "Failure mode",
        [SINGLE_MODE, FLOOD_MODE],
        key="failure_mode",
        on_change=_on_failure_mode_change,
        horizontal=True,
    )
    current_closures = st.session_state.get("disabled_nodes", ())
    if failure_mode == FLOOD_MODE:
        st.caption(
            "Draw an area on the map, or use the keyboard-accessible junction "
            "list below. Both paths run the same multi-junction failure analysis."
        )
        all_ids = criticality.sort_values("rank")["node_id"].astype(int).tolist()
        manual_nodes = st.multiselect(
            "Junctions inside the failed area",
            all_ids,
            default=(
                list(current_closures)
                if st.session_state.get("ablation_source") == "flood"
                else []
            ),
            key="manual_flood_nodes",
            format_func=lambda node: (
                f"#{int(ranks[node])} · Junction {node} · score {scores[node]:.3f}"
            ),
        )
        apply_col, clear_col = st.columns(2)
        if apply_col.button(
            "Apply junction set",
            type="primary",
            use_container_width=True,
            disabled=not manual_nodes,
        ):
            st.session_state["disabled_nodes"] = tuple(sorted(map(int, manual_nodes)))
            st.session_state["ablation_source"] = "flood"
            log_event("simulate_flood_nodes", n_closed=len(manual_nodes))
            st.rerun()
        if clear_col.button("Clear area failure", use_container_width=True):
            _reset_simulation()

    selected = st.selectbox(
        "Junction to disable",
        critical_ids,
        key="selected_node",
        on_change=_clear_last_map_click,
        disabled=failure_mode == FLOOD_MODE,
        format_func=lambda node: (
            f"#{int(ranks[node])} · Junction {node} · score {scores[node]:.3f}"
        ),
    )
    st.caption(
        "Click a critical junction on the map or choose one above."
        if failure_mode == SINGLE_MODE
        else "Single-junction selection is disabled while area failure is active."
    )
    # Selected-junction details as text, so the info doesn't live only in
    # hover tooltips (screen readers / touch devices).
    selected_row = criticality.loc[criticality["node_id"] == selected]
    if not selected_row.empty:
        is_art = (
            _truthy(selected_row.iloc[0].get("is_articulation"))
            if "is_articulation" in criticality.columns
            else False
        )
        degree = int(((edges["u"] == selected) | (edges["v"] == selected)).sum())
        st.caption(
            f"Junction {selected} · rank #{int(ranks[selected])} · "
            f"score {scores[selected]:.3f} · degree {degree} · "
            + (
                "articulation point (single point of failure)"
                if is_art
                else "not an articulation point"
            )
        )

    single_active = current_closures and st.session_state.get("ablation_source") == "single"
    add_more = bool(single_active)  # once one junction is down, the button accumulates

    simulate_column, reset_column = st.columns(2)
    if simulate_column.button(
        "Add closure" if add_more else "Simulate closure",
        type="primary",
        use_container_width=True,
        help=("Add the selected junction to the active closures (compound disaster)."
              if add_more else
              "Disable the selected junction and recompute routes and resilience."),
        disabled=failure_mode == FLOOD_MODE,
    ):
        # Accumulate closures so users can model a compound disaster (A39).
        base = set(current_closures) if add_more else set()
        st.session_state["disabled_nodes"] = tuple(sorted(base | {int(selected)}))
        st.session_state["ablation_source"] = "single"
        log_event("simulate_closure", n_closed=len(base) + 1, compound=add_more)
        st.rerun()
    if reset_column.button("Reset", use_container_width=True):
        _reset_simulation()

    # Active-closure chips: each removable so a compound scenario can be pared back.
    if single_active and len(current_closures) >= 1:
        st.caption("Active closures — click to remove:")
        chip_cols = st.columns(min(len(current_closures), 4))
        for i, node in enumerate(current_closures):
            if chip_cols[i % len(chip_cols)].button(f"✕ {node}", key=f"rm_closure_{node}"):
                remaining = tuple(n for n in current_closures if n != node)
                st.session_state["disabled_nodes"] = remaining
                if not remaining:
                    st.session_state["ablation_source"] = None
                st.rerun()

    layer_one, layer_two, layer_three = st.columns(3)
    layer_one.checkbox("Critical nodes", value=True, key="show_critical")
    layer_two.checkbox("Healed roads", value=True, key="show_healed")
    layer_three.checkbox("Single-points-of-failure", value=True, key="show_spof")

    if simulation:
        nodes_str = ", ".join(map(str, simulation.disabled_nodes))
        if simulation.largest_cc_fraction < 0.99:
            st.error(
                f"Network split: {1 - simulation.largest_cc_fraction:.0%} of "
                "junctions isolated from the main network."
            )
        else:
            st.warning(f"Junction(s) {nodes_str} disabled.")
        st.caption(
            f"Largest connected network: {simulation.largest_cc_fraction:.1%} of "
            "remaining junctions."
        )
        if route and route.rerouted_path:
            st.write(
                f"Reroute **{route.origin} → {route.destination}**: "
                f"{route.baseline_length_m:.0f} m → {route.rerouted_length_m:.0f} m"
            )
        render_charts(simulation)
    else:
        st.info("No simulation running. Select a junction, then simulate its closure.")


def render_curves_tab(resilience_curve: pd.DataFrame | None) -> None:
    """A39 curves tab: the resilience degradation chart."""
    if resilience_curve is None:
        st.info("Resilience curve not available for this dataset.")
        return
    st.subheader("Resilience Degradation Curve")
    curve_data = resilience_curve.set_index("n_removed")
    if "targeted_resilience_index" in curve_data.columns and "random_resilience_index" in curve_data.columns:
        chart_data = curve_data.rename(columns={
            "targeted_resilience_index": "Targeted failure",
            "random_resilience_index": "Random failure"
        })[["Targeted failure", "Random failure"]]
    else:
        chart_data = curve_data
    st.line_chart(
        chart_data,
        color=[TOKENS["disabled"], TOKENS["selected"]] if len(chart_data.columns) == 2 else None,
        height=200,
    )


def render_rankings_tab(
    critical_nodes: pd.DataFrame, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
) -> None:
    """A39 rankings tab: the selectable chokepoint leaderboard."""
    st.subheader("Top critical junctions")
    # Numeric scores (not stringified) so sorting is true-numeric and the score
    # bar renders; clicking a row selects that junction and recentres the map (A39).
    ranked = critical_nodes[["rank", "node_id", "betweenness"]].copy()
    max_bw = max(float(ranked["betweenness"].max()), 1e-9)
    table_event = st.dataframe(
        ranked,
        hide_index=True,
        use_container_width=True,
        height=240,
        on_select="rerun",
        selection_mode="single-row",
        key="rank_table",
        column_config={
            "betweenness": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=max_bw, format="%.3f",
            ),
        },
    )
    selected_rows = table_event.selection.rows if table_event and table_event.selection else []
    if selected_rows:
        picked = int(ranked.iloc[selected_rows[0]]["node_id"])
        if picked != int(st.session_state.get("selected_node", -1)):
            st.session_state["selected_node"] = picked
            # Recentre on the picked junction (the viewport round-trip keeps zoom).
            match = nodes[nodes["node_id"].astype(int) == picked]
            if not match.empty:
                st.session_state["map_center"] = [
                    float(match.geometry.y.iloc[0]), float(match.geometry.x.iloc[0]),
                ]
            st.rerun()
    st.caption(f"Network: {len(nodes):,} junctions · {len(edges):,} road links")


def apply_design_theme() -> None:
    """Apply the A30 design system: dark ops-dashboard, Fira Sans/Code, amber accent.

    Works WITH `.streamlit/config.toml` (native-widget dark tokens); this layer
    adds the typography, card system, and interaction polish on top. All colour
    values are generated from TOKENS as CSS custom properties.
    """
    token_vars = "".join(
        f"            --rr-{name.replace('_', '-')}: {value};\n"
        for name, value in TOKENS.items()
    )
    css_body = """
          html, body, [data-testid="stAppViewContainer"] *:not(code):not(pre) {
            font-family: 'Fira Sans', -apple-system, 'Segoe UI', sans-serif;
          }

          /* Layered backdrop: deep navy with faint blue/amber radial glows */
          .stApp {
            background:
              radial-gradient(1100px 550px at 88% -10%, rgba(56,189,248,.07), transparent 60%),
              radial-gradient(900px 500px at -8% 108%, rgba(245,158,11,.05), transparent 55%),
              linear-gradient(180deg, #0B1220 0%, #0D1526 100%);
          }
          .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1560px; }

          /* Section headers become uppercase micro-labels with an amber tick */
          [data-testid="stAppViewContainer"] h3 {
            font-size: .9rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .12em;
            color: #B9C7DD !important;
            border-left: 3px solid var(--rr-amber);
            padding: .1rem 0 .1rem .6rem;
            margin-top: .4rem;
          }

          /* Brand header */
          .rr-header { display: flex; align-items: center; gap: 13px; padding: 2px 0 4px; }
          .rr-logo {
            width: 42px; height: 42px; flex: 0 0 42px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #1E3A8A 0%, #0EA5E9 120%);
            box-shadow: 0 6px 18px rgba(14,165,233,.28);
          }
          .rr-header h1 { font-size: 1.42rem; font-weight: 700; letter-spacing: -.015em; margin: 0; line-height: 1.15; color: var(--rr-text); }
          .rr-header .rr-sub { margin: 2px 0 0; color: var(--rr-muted); font-size: .85rem; }
          .rr-chip {
            margin-left: auto; display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 11px; border-radius: 999px; font-size: .7rem; font-weight: 600; letter-spacing: .1em;
            border: 1px solid var(--rr-border-strong); background: var(--rr-surface); color: var(--rr-muted);
          }
          .rr-chip-active {
            border-color: rgba(245,158,11,.45); background: rgba(245,158,11,.14); color: #FDE68A;
          }
          .rr-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--rr-amber); box-shadow: 0 0 8px var(--rr-amber); animation: rr-pulse 2.2s ease-in-out infinite; }
          @keyframes rr-pulse { 0%,100% { opacity: 1 } 50% { opacity: .4 } }

          /* Metric cards: elevated surface, amber signal edge, tabular numerals */
          [data-testid="stMetric"] {
            position: relative; overflow: hidden;
            background: linear-gradient(180deg, var(--rr-surface-2) 0%, var(--rr-surface) 100%);
            border: 1px solid var(--rr-border);
            border-radius: 12px;
            padding: 14px 16px 11px;
            box-shadow: 0 4px 16px rgba(2,6,17,.35);
            transition: border-color .2s ease;
          }
          [data-testid="stMetric"]:hover { border-color: var(--rr-border-strong); }
          [data-testid="stMetric"]::before {
            content: ""; position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
            background: linear-gradient(180deg, var(--rr-amber), rgba(245,158,11,.12));
          }
          [data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: .11em; font-size: .7rem !important; font-weight: 500; color: var(--rr-muted) !important; }
          [data-testid="stMetricValue"] { font-family: 'Fira Code', monospace; font-variant-numeric: tabular-nums; font-size: 1.85rem; color: #F1F5FB; }
          [data-testid="stMetricDelta"] { font-family: 'Fira Code', monospace; font-size: .82rem; }

          /* Buttons */
          .stButton > button, .stDownloadButton > button {
            border-radius: 10px; font-weight: 600; cursor: pointer;
            transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, filter .16s ease;
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, var(--rr-amber) 0%, var(--rr-amber-deep) 100%);
            color: #1A1205; border: none;
            box-shadow: 0 6px 18px rgba(245,158,11,.22);
          }
          .stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(245,158,11,.32); filter: brightness(1.06); }
          .stButton > button[kind="primary"]:active { transform: translateY(0); }
          .stButton > button[kind="secondary"], .stDownloadButton > button {
            background: var(--rr-surface); color: var(--rr-text); border: 1px solid var(--rr-border-strong);
          }
          .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {
            border-color: var(--rr-amber); color: #FDE68A;
          }

          /* Inputs / selects (BaseWeb) */
          div[data-baseweb="select"] > div {
            background: var(--rr-surface) !important;
            border-color: var(--rr-border) !important;
            border-radius: 10px !important;
            transition: border-color .16s ease;
          }
          div[data-baseweb="select"] > div:hover { border-color: var(--rr-border-strong) !important; }

          /* Cards: expander, dataframe, alerts, charts */
          [data-testid="stExpander"] {
            border: 1px solid var(--rr-border); border-radius: 12px;
            background: var(--rr-surface); overflow: hidden;
          }
          [data-testid="stExpander"] summary { font-weight: 600; }
          [data-testid="stExpander"] summary:hover { color: var(--rr-amber); }
          [data-testid="stDataFrame"] { border: 1px solid var(--rr-border); border-radius: 12px; overflow: hidden; }
          [data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--rr-border); }

          /* The Folium map iframe becomes a framed panel. Scoped to st_folium only
             (Streamlit injects hidden utility iframes that must stay unstyled), and
             wrapper-relative: streamlit-folium's bidirectional frontend can inflate
             the iframe height attribute, so 100% follows each component's requested
             wrapper height (420px Briefing, 640px Analysis) without global forcing. */
          iframe[title="streamlit_folium.st_folium"] {
            height: 100% !important;
            border-radius: 14px;
            border: 1px solid var(--rr-border) !important;
            box-shadow: 0 10px 30px rgba(2,6,17,.45);
          }

          [data-testid="stCaptionContainer"] { color: var(--rr-muted) !important; }
          hr { border-color: var(--rr-border) !important; }

          ::-webkit-scrollbar { width: 10px; height: 10px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: var(--rr-border); border-radius: 8px; border: 2px solid var(--rr-bg); }
          ::-webkit-scrollbar-thumb:hover { background: var(--rr-border-strong); }

          :focus-visible { outline: 2px solid var(--rr-blue) !important; outline-offset: 2px; }

          @media (prefers-reduced-motion: reduce) {
            * { transition: none !important; animation: none !important; }
          }
    """
    st.markdown(
        "<style>\n"
        "          @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');\n"
        + f"          :root {{\n{token_vars}          }}\n"
        + css_body
        + "\n        </style>",
        unsafe_allow_html=True,
    )


# A36 minimal observability: structured logs to stderr (picked up
# by journald on the deployed box) + a lightweight usage counter, so "how many
# people used it / how many uploads failed" is answerable without extra infra.
log = logging.getLogger("trace.app")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [trace.app] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def log_event(event: str, **fields: object) -> None:
    """Emit one structured usage/telemetry line (event + key=value fields)."""
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    log.info("%s %s", event, extra)


MAX_UPLOAD_MB = 11  # source bytes; base64 transport expands this to ~14.7 MiB
UPLOAD_GUIDANCE = (
    "Runs the deployed SegFormer model on a serverless T4 (Modal). The first "
    "call after idle takes ~30 s to warm up; subsequent calls are near-instant. "
    "**Best results at ~0.5 m/pixel** (Google-Earth neighbourhood zoom, roads "
    "4–10 px wide) — heavily zoomed-in or zoomed-out captures degrade extraction."
)
UPLOAD_DISCLOSURE = (
    "Your original image bytes leave this server and are sent to Modal's GPU "
    "infrastructure solely for road-mask inference. Platform/request logs may "
    "exist; this project does not offer a contractual retention guarantee. Do "
    "not upload classified, restricted, personal, or otherwise sensitive "
    "imagery. For those datasets, run the local pipeline instead."
)


@st.cache_data(show_spinner=False, ttl=3600)
def _call_modal_seg_cached(image_bytes: bytes) -> tuple[bytes, float | None]:
    """Cache GPU results for an hour so Streamlit reruns never re-POST an image."""
    return _call_modal_seg(image_bytes)


def _seg_error_message(error: Exception) -> str:
    """Map an endpoint failure to a human-readable message (never a raw repr)."""
    if isinstance(error, EndpointBusyError):
        return str(error)
    if isinstance(error, urllib.error.HTTPError):
        if error.code in (401, 403):
            return (
                "The GPU endpoint rejected the request — its access key may be "
                "misconfigured on this deployment."
            )
        if error.code == 413:
            return "Image too large for the GPU endpoint — crop or downscale it and retry."
        return f"The GPU endpoint returned an error (HTTP {error.code}) — try again in a minute."
    if isinstance(error, TimeoutError) or (
        isinstance(error, urllib.error.URLError)
        and isinstance(getattr(error, "reason", None), TimeoutError)
    ):
        return (
            "The request timed out — the first call after idle takes ~30 s while "
            "the GPU wakes up. Please try again."
        )
    if isinstance(error, urllib.error.URLError):
        return "GPU endpoint unreachable — it may be redeploying; try again in a minute."
    return (
        "Road extraction failed — the endpoint sent an unexpected response; "
        "try again in a minute."
    )


def render_live_detection() -> None:
    """Upload a satellite image → segment roads on the serverless GPU (Modal).

    The heavy model runs off-box on a T4 that scales to zero. Without
    MODAL_SEG_URL the section still renders (disabled) and points at the
    hosted demo instead of hiding entirely.
    """
    st.subheader("Analyze your own imagery")
    st.warning(UPLOAD_DISCLOSURE)
    if not MODAL_SEG_URL:
        st.info(
            "GPU inference is not configured on this instance — try the hosted "
            "demo at trace.tiwaribabu.in to analyze your own imagery."
        )
        st.file_uploader(
            "Upload a satellite / aerial image to extract its road network",
            type=["png", "jpg", "jpeg"],
            key="live_detection_upload",
            disabled=True,
        )
        st.caption(UPLOAD_GUIDANCE)
        return
    consent = st.checkbox(
        "I am authorized to send this image to Modal for processing",
        key="modal_upload_consent",
    )
    upload = st.file_uploader(
        "Upload a satellite / aerial image to extract its road network",
        type=["png", "jpg", "jpeg"],
        key="live_detection_upload",
        disabled=not consent,
    )
    st.caption(UPLOAD_GUIDANCE)
    if upload is None:
        return

    image_bytes = upload.getvalue()
    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"Image is larger than {MAX_UPLOAD_MB} MB — crop or downscale it "
            "and try again."
        )
        return
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            width, height = probe.size
    except Image.DecompressionBombError:
        st.error(
            "Image has too many pixels (limit 4096×4096) — crop or downscale "
            "it and try again."
        )
        return
    except Exception:  # noqa: BLE001 — any unreadable file gets the same friendly hint
        st.error("Could not read that file as an image — please upload a PNG or JPEG.")
        return
    if min(width, height) < 256:
        st.warning(
            "That image is quite small (under 256 px) — extraction may miss roads. "
            "Best results at ~0.5 m/pixel neighbourhood crops."
        )
    elif max(width, height) > 3000:
        st.warning(
            "That image is very large — extraction works best on neighbourhood-scale "
            "crops at ~0.5 m/pixel."
        )

    with st.status("Extracting road network…", expanded=True) as status:
        st.write(f"Image received: {width}×{height} px, {len(image_bytes) / 1_048_576:.1f} MB.")
        st.write("Waking the GPU (first call after idle takes ~30 s)…")
        try:
            mask_png, threshold = _call_modal_seg_cached(image_bytes)
            orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mask = Image.open(io.BytesIO(mask_png)).convert("L")
        except Exception as error:  # noqa: BLE001 — every failure maps to a human message
            status.update(label="Road extraction failed", state="error", expanded=True)
            st.error(_seg_error_message(error))
            log_event("upload_failed", kind=type(error).__name__, px=f"{width}x{height}")
            return
        st.write("Road mask received.")
        status.update(label="Road network extracted", state="complete", expanded=False)
        log_event("upload_ok", px=f"{width}x{height}", mb=round(len(image_bytes) / 1_048_576, 2))

    overlay = np.asarray(orig).copy()
    overlay[np.asarray(mask) > 0] = [255, 0, 0]

    col1, col2, col3 = st.columns(3)
    # use_column_width (not use_container_width): st.image doesn't accept the
    # latter on the pinned Streamlit 1.38 — it was added in a later release.
    col1.image(orig, caption="Input", use_column_width=True)
    thr = f"thr {threshold}" if threshold is not None else "roads"
    col2.image(mask, caption=f"Road mask ({thr})", use_column_width=True)
    col3.image(overlay, caption="Overlay", use_column_width=True)

    _render_upload_analysis(np.asarray(orig), np.asarray(mask))


@st.cache_resource
def _ensure_upload_worker() -> None:
    """Start the A39 filesystem job-queue worker once per process.

    `ensure_worker()` is already idempotent on its own (module flag + lock,
    src/app/job_queue.py) — this cache_resource wrapper is a second belt so a
    Streamlit rerun doesn't even re-enter the function to find that out.
    """
    from src.app import job_queue

    job_queue.ensure_worker()


def _render_upload_analysis(orig_rgb: np.ndarray, mask_gray: np.ndarray) -> None:
    """A39 upload loop: mask → queued analysis → resilience.

    Submits the mask to the filesystem job queue rather than calling
    analyze_mask() synchronously in the request thread: a queued job survives
    a Streamlit worker restart, and concurrent uploads wait in line instead of
    bouncing off the A37 semaphore's outright "busy, retry" warning. Image-space
    (the upload isn't georeferenced), so the network is drawn over the user's
    own image.
    """
    from src.app import job_queue
    from src.app.upload_analysis import render_graph_overlay

    st.markdown("#### Network resilience of your imagery")
    gsd = st.slider(
        "Approx. ground resolution (m/pixel)", 0.1, 2.0, 0.5, 0.05,
        help="Uploads aren't georeferenced — this scales road lengths. ~0.5 m/px "
             "matches neighbourhood-zoom satellite captures.",
    )

    _ensure_upload_worker()

    # A new upload or a changed resolution slider both mean "this is a
    # different analysis" — resubmit instead of showing a stale job's result.
    job_key = (hashlib.md5(mask_gray.tobytes()).hexdigest(), round(gsd, 3))
    if st.session_state.get("upload_job_key") != job_key:
        job_queue.cleanup()  # opportunistic housekeeping on every new submit
        binary = (mask_gray > 0).astype(np.uint8)
        st.session_state["upload_job_id"] = job_queue.submit(binary, resolution_m=gsd)
        st.session_state["upload_job_key"] = job_key
        st.session_state.pop("upload_job_result", None)

    job_id = st.session_state["upload_job_id"]

    if "upload_job_result" not in st.session_state:
        state = job_queue.status(job_id)
        if state["status"] == "queued":
            pos = job_queue.position(job_id)
            st.info(f"Queued for analysis — position {pos} in line…")
            time.sleep(2)
            st.rerun()
        elif state["status"] == "running":
            with st.spinner("Building the routable graph and scoring resilience…"):
                time.sleep(2)
            st.rerun()
        elif state["status"] == "failed":
            st.error(f"Analysis failed: {state['error']}")
            log_event("upload_analysis_failed", error=state["error"])
            if st.button("Retry analysis", key="retry_upload_analysis"):
                st.session_state.pop("upload_job_key", None)  # forces a resubmit above
                st.rerun()
            return
        else:  # done
            st.session_state["upload_job_result"] = job_queue.result(job_id)

    result = st.session_state["upload_job_result"]
    retained = result.resilience_index
    m1, m2, m3 = st.columns(3)
    m1.metric("Junctions", f"{result.n_nodes:,}")
    m2.metric("Road links", f"{result.n_edges:,}")
    m3.metric(
        "Efficiency if #1 junction fails", f"{retained:.0%}",
        delta=f"-{1 - retained:.0%}", delta_color="normal",
        help="Global efficiency retained after the single most critical junction is lost.",
    )
    st.progress(retained, text=f"Network efficiency retained under worst single failure: {retained:.0%}")

    fig = render_graph_overlay(orig_rgb, result)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        f"{result.summary['critical_junctions']} critical junctions · "
        f"{result.summary['articulation_points']} single-points-of-failure · "
        "the ringed junction is the worst chokepoint. Lengths assume "
        f"~{gsd:g} m/pixel (no georeference on uploads)."
    )
    st.dataframe(
        result.criticality[["rank", "node_id", "betweenness", "is_critical", "is_articulation"]].head(10),
        hide_index=True, use_container_width=True,
    )
    log_event("upload_analysis_ok", nodes=result.n_nodes, edges=result.n_edges,
              ri=round(retained, 3))


def main() -> None:
    """Render the interactive F2 dashboard."""
    st.set_page_config(
        page_title="Route Resilience",
        page_icon="🛰️",
        layout="wide",
        menu_items={
            "About": "Route Resilience — road-network resilience from satellite imagery. Demo AOI: Panaji.",
            "Get help": None,
            "Report a bug": None,
        },
    )
    apply_design_theme()
    try:
        features, criticality = load_sample_data()
        graph = graph_from_features(DATA_FINGERPRINT, features)
        resilience_curve = load_resilience_curve()
    except FileNotFoundError as error:
        st.error(
            "Sample artifacts not found. Please ensure the pipeline has generated "
            f"the sample data, or verify `data/sample/` exists. Details: {error}"
        )
        st.stop()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not load or parse the sample network: {error}")
        st.stop()

    critical_ids = set(
        criticality.loc[criticality["is_critical"].map(_truthy), "node_id"].astype(int)
    )
    if "selected_node" not in st.session_state:
        st.session_state["selected_node"] = int(
            criticality.sort_values("rank").iloc[0]["node_id"]
        )
    disabled_nodes = st.session_state.get("disabled_nodes", ())
    # Computed once per rerun and shared by the Briefing and Analysis tabs
    # (both need the current ablation's metrics) rather than re-simulating —
    # simulate_ablation is cache_data-keyed on disabled_nodes anyway, but this
    # avoids even the cache-lookup duplication (A39).
    if disabled_nodes:
        with st.spinner("Simulating failure…"):
            simulation = simulate_ablation(DATA_FINGERPRINT, graph, disabled_nodes)
    else:
        simulation = None

    # Full-width brand header with an honest state chip: the pulsing accent
    # chip only appears while a simulation is actually active.
    chip = (
        '<span class="rr-chip rr-chip-active"><span class="rr-dot"></span>SIM ACTIVE</span>'
        if disabled_nodes
        else '<span class="rr-chip">DEMO · PANAJI</span>'
    )
    st.markdown(
        f"""
        <div class="rr-header">
          <div class="rr-logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#F8FAFC"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round" role="img"
                 aria-label="Road network glyph">
              <circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="12" cy="18" r="2.2"/>
              <path d="M6.7 7.6 10.6 16M17.3 7.6 13.4 16M7.2 6h9.6"/>
            </svg>
          </div>
          <div>
            <h1>Route Resilience</h1>
            <p class="rr-sub">Panaji network &middot; live junction-failure simulation</p>
          </div>
          {chip}
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_briefing, tab_analysis, tab_upload, tab_method = st.tabs(
        ["Briefing", "Analysis", "Your imagery", "Methodology"]
    )
    with tab_briefing:
        render_briefing(features, criticality, graph, simulation)
    with tab_analysis:
        render_dashboard_view(
            features, criticality, graph, resilience_curve, critical_ids, simulation
        )
    with tab_upload:
        render_live_detection()
    with tab_method:
        render_methodology()


def render_methodology() -> None:
    """Methodology tab: metric definitions and committed evidence (A45-C6).

    Surfaces the eval artifacts already committed under ``data/sample/`` (IoU,
    APLS, graph quality) that otherwise live only in docs a judge never opens.
    """
    import json

    st.subheader("How Route Resilience works")
    st.markdown(
        "1. **Extract** — a fine-tuned SegFormer road segmenter turns satellite "
        "imagery into a road mask.\n"
        "2. **Vectorise** — the mask is skeletonised, gap-healed, and simplified "
        "into a routable graph (junctions + road links).\n"
        "3. **Analyse** — betweenness centrality flags chokepoint junctions; the "
        "**Resilience Index** is the network's *global efficiency* after a failure "
        "relative to intact (1.00 = no degradation, lower = worse)."
    )
    st.caption(
        "Resilience uses global efficiency (mean inverse shortest-path length), which "
        "stays finite even when a failure disconnects the network — unlike raw "
        "average path length."
    )

    st.subheader("Model & network quality (committed evaluation evidence)")
    sample_dir = SAMPLE_GEOJSON.parent
    reports = {
        "Segmentation (DeepGlobe v1 historical holdout)": "segmentation_eval.json",
        "Routing similarity — APLS": "panaji_demo_apls.json",
        "Graph quality": "panaji_demo_graph_eval.json",
    }
    shown = False
    for label, fname in reports.items():
        path = sample_dir / fname
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        shown = True
        with st.expander(label, expanded=False):
            st.json(data, expanded=False)
    if not shown:
        st.info("Evaluation artifacts are generated by the pipeline into `data/sample/`.")

    prov_path = sample_dir / "panaji_demo_provenance.json"
    if prov_path.exists():
        try:
            st.caption(f"Model provenance: {json.loads(prov_path.read_text())}")
        except (OSError, json.JSONDecodeError):
            pass

    st.caption(
        "Accessibility: the map is mouse-driven, but every junction is also "
        "reachable via the junction picker and the ranked table in the Analysis tab, "
        "and all colour states carry text labels."
    )


def render_dashboard_view(
    features, criticality, graph, resilience_curve, critical_ids, simulation
) -> None:
    """The A39 Analysis tab: the planner's full tooling — network map
    + control panel. ``simulation`` is computed once in main() and shared with
    the Briefing tab rather than re-simulated here."""
    show_critical = st.session_state.get("show_critical", True)
    show_healed = st.session_state.get("show_healed", True)
    show_spof = st.session_state.get("show_spof", True)
    failure_mode = st.session_state.get("failure_mode", SINGLE_MODE)

    # Remembered viewport (A39): fed into every map so pan/zoom survives
    # reruns. Reset clears it (via reset_counter) so the map reframes deliberately.
    map_center = st.session_state.get("map_center")
    map_zoom = st.session_state.get("map_zoom")

    sim_map = build_map(
        features,
        criticality,
        graph,
        int(st.session_state["selected_node"]),
        simulation,
        show_critical,
        show_healed,
        show_spof,
        failure_mode,
        center=map_center,
        zoom=map_zoom,
    )

    map_column, panel_column = st.columns([6.5, 3.5], gap="medium")

    # Stable keys (only the reset counter) so the iframe is NOT remounted on every
    # selection/toggle — the round-tripped center/zoom keeps the view in place.
    reset_n = st.session_state.get("reset_counter", 0)
    with map_column:
        # Side-by-side baseline/simulation comparison mode was removed
        # (A39): the two panes never synced pan/zoom, doubled the
        # chrome, and streamlit-folium's own docs flag DualMap as flaky —
        # the orange reroute + dimmed disabled edges already show the
        # before/after story in this one map.
        map_state = st_folium(
            sim_map,
            height=640,
            use_container_width=True,
            returned_objects=["last_object_clicked", "all_drawings", "center", "zoom"],
            key=f"network_map_{reset_n}",
        )

        st.caption(
            "Brighter roads connect more critical junctions · dashed = inferred/healed · "
            "orange = reroute · red = disabled. The network overlay renders even if "
            "basemap tiles fail to load."
        )

    # Persist the viewport the component reports so the next rerun rebuilds the map
    # where the user left it (guarded against the None first render).
    if map_state:
        _center = map_state.get("center")
        if isinstance(_center, dict) and "lat" in _center and "lng" in _center:
            st.session_state["map_center"] = [_center["lat"], _center["lng"]]
        if map_state.get("zoom") is not None:
            st.session_state["map_zoom"] = map_state["zoom"]

    clicked = map_state.get("last_object_clicked") if map_state else None
    drawings = map_state.get("all_drawings") if map_state else None

    if failure_mode == FLOOD_MODE and drawings is not None:
        from shapely.geometry import shape
        flooded_nodes = set()
        polygons_drawn = False
        nodes_gdf, _ = split_features(features)
        for drawing in drawings:
            geom_type = drawing.get("geometry", {}).get("type")
            if geom_type in ("Polygon", "MultiPolygon"):
                polygons_drawn = True
                geom = shape(drawing["geometry"])
                inside = nodes_gdf[nodes_gdf.geometry.within(geom)]["node_id"].astype(int).tolist()
                flooded_nodes.update(inside)

        if polygons_drawn and not flooded_nodes:
            with map_column:
                st.warning(
                    "No junctions found inside the drawn area — try a larger or "
                    "more precise polygon."
                )
        new_disabled = tuple(sorted(flooded_nodes))
        if new_disabled != st.session_state.get("disabled_nodes", ()):
            st.session_state["disabled_nodes"] = new_disabled
            st.session_state["ablation_source"] = "flood" if new_disabled else None
            st.rerun()
    click_signature = (
        (round(float(clicked["lat"]), 7), round(float(clicked["lng"]), 7))
        if clicked and "lat" in clicked and "lng" in clicked
        else None
    )
    if click_signature and click_signature != st.session_state.get("last_map_click"):
        st.session_state["last_map_click"] = click_signature
        nodes, _ = split_features(features)
        selected = nearest_critical_node(nodes, critical_ids, clicked)
        if selected is not None:
            st.session_state["selected_node"] = selected
            st.rerun()
        else:
            st.toast(
                "No critical junction near that click — the highlighted dots are selectable.",
                icon="🎯",
            )

    with panel_column:
        # Fixed-height scroll region so the panel scrolls independently beside
        # the 640-px map instead of stretching the whole page.
        with st.container(height=640):
            render_panel(features, criticality, simulation, resilience_curve)

    st.caption(
        "Route Resilience v1 · demo data: Panaji sample network (precomputed) · "
        "inferred segments shown dashed"
    )


if __name__ == "__main__":
    main()
