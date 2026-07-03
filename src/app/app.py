"""Interactive Streamlit dashboard for road-network resilience."""

from dataclasses import dataclass
from itertools import combinations
from math import inf, isfinite
import os
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import FastMarkerCluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

from src.pipeline.p3_analysis.resilience import resilience_index


def find_repo_root() -> Path:
    """Find the repository root using its Tracker file as a stable marker."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "docs" / "Tracker.md").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


REPO_ROOT = find_repo_root()
SAMPLE_GEOJSON = REPO_ROOT / "data" / "sample" / "panaji_demo_graph.geojson"
SAMPLE_CRITICALITY = REPO_ROOT / "data" / "sample" / "panaji_demo_criticality.csv"


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
        return pd.read_csv(path)
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
def graph_from_features(_features: gpd.GeoDataFrame) -> nx.Graph:
    """Convert the map-ready GeoJSON features into a routable graph.

    ``_features`` is underscore-prefixed so Streamlit skips hashing the
    (unhashable) GeoDataFrame for the cache key — required under cache_resource.
    """
    nodes, edges = split_features(_features)
    graph = nx.Graph()
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


def path_length(graph: nx.Graph, path: tuple[int, ...] | list[int]) -> float:
    """Return a path's total length in metres."""
    return sum(
        float(graph.edges[start, end]["length_m"])
        for start, end in zip(path, path[1:])
    )


def edge_key(start: int, end: int) -> tuple[int, int]:
    """Return an order-independent edge identifier."""
    return min(start, end), max(start, end)


def representative_reroute(graph: nx.Graph, disabled_node: int) -> RouteResult | None:
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
                    100.0 * float(graph.edges[start, end]["length_m"]) / baseline_length
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
def simulate_ablation(graph_fingerprint: str, _graph: nx.Graph, nodes: tuple[int, ...]) -> SimulationResult:
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
    """Create a labelled map legend for semantic route states."""
    html = """
    <div style="position: fixed; bottom: 36px; right: 12px; z-index: 9999;
                background: rgba(13,21,38,.92); color: #E2E8F0; padding: 10px 14px;
                border: 1px solid #24334E; border-radius: 10px; font-size: 12px;
                line-height: 1.75; font-family: 'Fira Sans', sans-serif;
                box-shadow: 0 8px 24px rgba(2,6,17,.5); backdrop-filter: blur(6px);">
      <b style="font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
                color: #8FA3BF;">Network states</b><br>
      <span style="color:#56B4E9">●</span> selected junction<br>
      <span style="color:#D55E00">●</span> disabled junction / links<br>
      <span style="color:#E69F00">━</span> rerouted path<br>
      <span style="color:#aaaaaa">┄</span> healed road<br>
      <span style="color:#FB7185">●</span> / <span style="color:#FB7185">━</span> single-point-of-failure
    </div>
    """
    return folium.Element(html)


def add_rerouted_path(road_map: folium.Map, graph: nx.Graph, route: RouteResult) -> None:
    """Draw the rerouted path last so its orange highlight stays visible."""
    if route.rerouted_path is None:
        return
    for start, end in zip(route.rerouted_path, route.rerouted_path[1:]):
        coordinates = graph.edges[start, end]["coordinates"]
        folium.PolyLine(
            [(latitude, longitude) for longitude, latitude in coordinates],
            color="#E69F00",
            weight=7,
            opacity=1.0,
            tooltip=f"Rerouted road {start}–{end}",
        ).add_to(road_map)


def build_map(
    features: gpd.GeoDataFrame,
    criticality: pd.DataFrame,
    graph: nx.Graph,
    selected_node: int,
    simulation: SimulationResult | None,
    show_critical: bool,
    show_healed: bool,
    show_spof: bool,
    scenario: str,
) -> folium.Map:
    """Build the map with criticality, selection, failure, and reroute states."""
    nodes, edges = split_features(features)
    scores = criticality.set_index("node_id")["betweenness"].to_dict()
    maximum = max(float(criticality["betweenness"].max()), 1e-9)
    # Dark-basemap-readable ramp (deep teal → sky → green → yellow): the old
    # viridis-style ramp started at near-black purple, so low-criticality roads
    # were invisible against the dark tiles.
    colour_scale = cm.LinearColormap(
        colors=["#155E75", "#0EA5E9", "#4ADE80", "#FDE047"],
        vmin=0.0,
        vmax=maximum,
        caption="Road criticality (endpoint betweenness: low to high)",
    )
    disabled_nodes = simulation.disabled_nodes if simulation else ()

    road_map = folium.Map(
        location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()],
        zoom_start=15,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer("CartoDB dark_matter", name="Dark map").add_to(road_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Satellite",
    ).add_to(road_map)
    
    if scenario == "Flood":
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

    for _, edge in edges.iterrows():
        is_bridged = _truthy(edge.get("is_bridged"))
        is_bridge = _truthy(edge.get("is_bridge"))
        if is_bridged and not show_healed:
            continue
        start, end = int(edge["u"]), int(edge["v"])
        score = max(float(scores.get(start, 0.0)), float(scores.get(end, 0.0)))
        is_disabled = start in disabled_nodes or end in disabled_nodes
        is_spof = is_bridge and show_spof

        if is_disabled:
            colour = "#D55E00"
            state = "disabled link"
        elif is_spof:
            # Muted rose (was neon magenta #ff00ff, which drowned the whole
            # network — most Panaji links are flagged is_bridge).
            colour = "#FB7185"
            state = "critical bridge"
        else:
            colour = colour_scale(score)
            state = "healed link" if is_bridged else "observed link"

        coordinates = [
            (latitude, longitude) for longitude, latitude in edge.geometry.coords
        ]
        folium.PolyLine(
            coordinates,
            color=colour,
            weight=5 if is_spof else (4 if is_disabled or is_bridged else 3),
            opacity=0.45 if is_disabled else (0.95 if is_spof else 0.85),
            dash_array="8 6" if is_bridged or is_disabled else None,
            tooltip=f"Road {start}–{end} · criticality {score:.3f} · {state}",
        ).add_to(road_map)

    critical_ids = set(criticality.loc[criticality["is_critical"].map(_truthy), "node_id"].astype(int))
    articulation_ids = set()
    if "is_articulation" in criticality.columns:
        articulation_ids = set(criticality.loc[criticality["is_articulation"].map(_truthy), "node_id"].astype(int))

    nodes_to_show = set()
    if show_critical:
        nodes_to_show.update(critical_ids)
    if show_spof:
        nodes_to_show.update(articulation_ids)
    
    nodes_to_show.add(selected_node)
    nodes_to_show.update(disabled_nodes)

    if nodes_to_show:
        cluster_data = []
        for _, node in nodes[nodes["node_id"].isin(nodes_to_show)].iterrows():
            node_id = int(node["node_id"])
            score = float(scores.get(node_id, 0.0))
            is_art = node_id in articulation_ids
            if node_id in disabled_nodes:
                colour, radius, label = "#D55E00", 9, "Disabled junction"
            elif node_id == selected_node:
                colour, radius, label = "#56B4E9", 8, "Selected junction"
            elif is_art and show_spof:
                colour, radius, label = "#FB7185", 7, "Articulation point"
            else:
                colour, radius, label = colour_scale(score), 5, "Critical junction"
                
            cluster_data.append([
                float(node.geometry.y),
                float(node.geometry.x),
                radius,
                2 if node_id == selected_node or node_id in disabled_nodes else 1,
                colour,
                f"{label} {node_id} &middot; score {score:.3f}"
            ])

        callback = """
        function (row) {
            var marker = L.circleMarker([row[0], row[1]], {
                radius: row[2],
                color: '#ffffff',
                weight: row[3],
                fill: true,
                fillColor: row[4],
                fillOpacity: 1.0
            });
            marker.bindTooltip(row[5]);
            return marker;
        };
        """
        FastMarkerCluster(cluster_data, callback=callback).add_to(road_map)

    if simulation and simulation.route:
        add_rerouted_path(road_map, graph, simulation.route)
    folium.LayerControl(position="topright").add_to(road_map)
    colour_scale.add_to(road_map)
    road_map.get_root().html.add_child(semantic_legend())
    return road_map


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
    """Render Design.md's live travel-impact and delay-contributor charts."""
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

    st.subheader("Travel impact")
    trend = pd.DataFrame(
        {
            "Disabled junctions": [0, 1],
            "Travel-time increase (%)": [0.0, route.travel_time_delta_pct],
        }
    )
    st.line_chart(
        trend,
        x="Disabled junctions",
        y="Travel-time increase (%)",
        color="#E69F00",
        height=180,
    )

    if route.delay_segments:
        st.caption("Top delay contributors on the detour")
        delays = pd.DataFrame(
            route.delay_segments,
            columns=["Road", "Delay contribution (%)"],
        ).set_index("Road")
        st.bar_chart(delays, color="#E69F00", height=190)


def generate_geojson_export(features: gpd.GeoDataFrame, disabled_nodes: tuple[int, ...]) -> str:
    """Generate GeoJSON string of the current network state."""
    export_df = features.copy()
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
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 8))
    fig.patch.set_facecolor('#1E1E2E')
    
    fig.suptitle("Route Resilience - Network Summary", color='white', fontsize=20, y=0.95)
    
    ax_metrics = fig.add_subplot(2, 2, 1)
    ax_metrics.axis('off')
    ax_metrics.set_facecolor('#1E1E2E')
    
    ri = simulation.resilience_index if simulation else 1.0
    ax_metrics.text(0.1, 0.7, f"Resilience Index:\n{ri:.3f}", color='white', fontsize=18, fontweight='bold')
    
    if simulation and simulation.route:
        delay = f"+{simulation.route.travel_time_delta_pct:.1f}%"
    elif simulation and len(simulation.disabled_nodes) > 1:
        delay = "N/A"
    else:
        delay = "0.0%"
    ax_metrics.text(0.1, 0.3, f"Travel Time Impact:\n{delay}", color='#E69F00', fontsize=18, fontweight='bold')
    
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
        cell.set_facecolor('#2D2D3D' if row > 0 else '#4C4C6D')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#1E1E2E')

    if resilience_curve is not None:
        ax_curve = fig.add_subplot(2, 1, 2)
        ax_curve.set_facecolor('#1E1E2E')
        ax_curve.tick_params(colors='white')
        for spine in ax_curve.spines.values():
            spine.set_color('white')
            
        curve_data = resilience_curve.set_index("n_removed")
        if "targeted_resilience_index" in curve_data.columns and "random_resilience_index" in curve_data.columns:
            ax_curve.plot(curve_data.index, curve_data["targeted_resilience_index"], color='#D55E00', label='Targeted')
            ax_curve.plot(curve_data.index, curve_data["random_resilience_index"], color='#56B4E9', label='Random')
            ax_curve.legend(facecolor='#2D2D3D', edgecolor='white', labelcolor='white')
        
        ax_curve.set_xlabel("Nodes Removed", color='white')
        ax_curve.set_ylabel("Resilience Index", color='white')
        ax_curve.set_title("Resilience Degradation Curve", color='white')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    return buf.getvalue()



def render_panel(
    features: gpd.GeoDataFrame,
    criticality: pd.DataFrame,
    simulation: SimulationResult | None,
    resilience_curve: pd.DataFrame | None = None,
) -> None:
    """Render controls, metrics, ranked hotspots, and live charts."""
    nodes, edges = split_features(features)
    critical_nodes = criticality[criticality["is_critical"].map(_truthy)].sort_values("rank")
    critical_ids = critical_nodes["node_id"].astype(int).tolist()
    scores = criticality.set_index("node_id")["betweenness"].to_dict()
    ranks = criticality.set_index("node_id")["rank"].to_dict()

    st.markdown(
        """
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
          <span class="rr-chip"><span class="rr-dot"></span>LIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ri = simulation.resilience_index if simulation else 1.0
    route = simulation.route if simulation else None
    travel_delta = route.travel_time_delta_pct if route else 0.0
    ri_column, travel_column = st.columns(2)
    ri_column.metric(
        "Resilience Index",
        f"{ri:.3f}",
        delta=f"{(ri - 1.0) * 100:.1f}%" if simulation else None,
        delta_color="inverse",
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

    st.subheader("Scenario controls")
    view_mode = st.radio("View Mode", ["Interactive Map", "Side-by-Side Comparison"], horizontal=True, key="view_mode")
    st.selectbox("Region", ["Panaji demo"], disabled=True)
    scenario = st.selectbox("Scenario", ["Road closure", "Accident", "Flood"], key="scenario")
    # TODO: wire scenario-specific behavior into routing logic
    selected = st.selectbox(
        "Junction to disable",
        critical_ids,
        key="selected_node",
        format_func=lambda node: (
            f"#{int(ranks[node])} · Junction {node} · score {scores[node]:.3f}"
        ),
    )
    st.caption("Click a critical junction on the map or choose one above.")

    simulate_column, reset_column = st.columns(2)
    if simulate_column.button(
        "Simulate closure",
        type="primary",
        use_container_width=True,
        help="Disable the selected junction and recompute routes and resilience.",
    ):
        st.session_state["disabled_nodes"] = (int(selected),)
        st.rerun()
    if reset_column.button("Reset", use_container_width=True):
        st.session_state["disabled_nodes"] = ()
        st.session_state["reset_counter"] = st.session_state.get("reset_counter", 0) + 1
        st.rerun()

    layer_one, layer_two, layer_three = st.columns(3)
    layer_one.checkbox("Critical nodes", value=True, key="show_critical")
    layer_two.checkbox("Healed roads", value=True, key="show_healed")
    layer_three.checkbox("Single-points-of-failure", value=True, key="show_spof")

    if simulation:
        nodes_str = ", ".join(map(str, simulation.disabled_nodes))
        st.success(f"Junction(s) {nodes_str} disabled.")
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

    if resilience_curve is not None:
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
            color=["#D55E00", "#56B4E9"] if len(chart_data.columns) == 2 else None,
            height=200,
        )

    st.subheader("Top critical junctions")
    top_nodes = critical_nodes[["rank", "node_id", "betweenness"]].head(5).copy()
    top_nodes["betweenness"] = top_nodes["betweenness"].map(lambda value: f"{value:.3f}")
    st.dataframe(top_nodes, hide_index=True, use_container_width=True)
    st.caption(f"Network: {len(nodes):,} junctions · {len(edges):,} road links")

    st.divider()
    st.subheader("Export & Reports")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        geojson_data = generate_geojson_export(features, simulation.disabled_nodes if simulation else ())
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


def apply_design_theme() -> None:
    """Apply the A30 design system: dark ops-dashboard, Fira Sans/Code, amber accent.

    Works WITH `.streamlit/config.toml` (native-widget dark tokens); this layer
    adds the typography, card system, and interaction polish on top.
    """
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

          :root {
            --rr-bg: #0B1220;
            --rr-surface: #121C30;
            --rr-surface-2: #16233B;
            --rr-border: #24334E;
            --rr-border-strong: #33456A;
            --rr-text: #E2E8F0;
            --rr-muted: #8FA3BF;
            --rr-amber: #F59E0B;
            --rr-amber-deep: #D97706;
            --rr-blue: #38BDF8;
          }

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
            font-size: .82rem !important;
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
          .rr-header .rr-sub { margin: 2px 0 0; color: var(--rr-muted); font-size: .8rem; }
          .rr-chip {
            margin-left: auto; display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 11px; border-radius: 999px; font-size: .7rem; font-weight: 600; letter-spacing: .1em;
            border: 1px solid rgba(52,211,153,.35); background: rgba(16,185,129,.12); color: #6EE7B7;
          }
          .rr-dot { width: 7px; height: 7px; border-radius: 50%; background: #34D399; box-shadow: 0 0 8px #34D399; animation: rr-pulse 2.2s ease-in-out infinite; }
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
             height-clamped: streamlit-folium's bidirectional frontend inflates the
             iframe height attribute (observed 3068px for a 760px map), which was
             stretching the page with dead space below the map and panel. */
          iframe[title="streamlit_folium.st_folium"] {
            height: 640px !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


MODAL_SEG_URL = os.environ.get("MODAL_SEG_URL")


def _call_modal_seg(image_bytes: bytes) -> tuple[bytes, float | None]:
    """POST an image to the Modal GPU endpoint; return (mask PNG bytes, threshold)."""
    import base64
    import json
    import urllib.request

    body = json.dumps(
        {
            "image_b64": base64.b64encode(image_bytes).decode(),
            "key": os.environ.get("MODAL_SEG_KEY", ""),
        }
    ).encode()
    req = urllib.request.Request(
        MODAL_SEG_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        out = json.load(resp)
    if "mask_png_b64" not in out:
        raise RuntimeError(out.get("error", "unexpected response from endpoint"))
    return base64.b64decode(out["mask_png_b64"]), out.get("threshold")


def render_live_detection() -> None:
    """Upload a satellite image → segment roads on the serverless GPU (Modal).

    Hidden unless MODAL_SEG_URL is set, so local dev without the endpoint is
    unaffected. The heavy model runs off-box on a T4 that scales to zero.
    """
    if not MODAL_SEG_URL:
        return
    st.subheader("Live road detection — serverless GPU")
    upload = st.file_uploader(
        "Upload a satellite / aerial image to extract its road network",
        type=["png", "jpg", "jpeg"],
        key="live_detection_upload",
    )
    if upload is None:
        st.caption(
            "Runs the deployed SegFormer model on a serverless T4 (Modal). The first "
            "call after idle takes ~30 s to warm up; subsequent calls are near-instant. "
            "**Best results at ~0.5 m/pixel** (Google-Earth neighbourhood zoom, roads "
            "4–10 px wide) — heavily zoomed-in or zoomed-out captures degrade extraction."
        )
        return
    image_bytes = upload.getvalue()
    with st.spinner("Segmenting roads on GPU…"):
        try:
            mask_png, threshold = _call_modal_seg(image_bytes)
        except Exception as error:  # noqa: BLE001 — surface any endpoint failure to the user
            st.error(f"Inference failed: {error}")
            return

    import io

    import numpy as np
    from PIL import Image

    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_png)).convert("L")
    overlay = np.asarray(orig).copy()
    overlay[np.asarray(mask) > 0] = [255, 0, 0]

    col1, col2, col3 = st.columns(3)
    # use_column_width (not use_container_width): st.image doesn't accept the
    # latter on the pinned Streamlit 1.38 — it was added in a later release.
    col1.image(orig, caption="Input", use_column_width=True)
    thr = f"thr {threshold}" if threshold is not None else "roads"
    col2.image(mask, caption=f"Road mask ({thr})", use_column_width=True)
    col3.image(overlay, caption="Overlay", use_column_width=True)


def main() -> None:
    """Render the interactive F2 dashboard."""
    st.set_page_config(page_title="Route Resilience", page_icon="🛰️", layout="wide")
    apply_design_theme()
    try:
        features, criticality = load_sample_data()
        graph = graph_from_features(features)
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
    with st.spinner("Simulating failure…") if disabled_nodes else st.empty():
        simulation = (
            simulate_ablation("panaji_demo_v1", graph, disabled_nodes)
            if disabled_nodes
            else None
        )

    show_critical = st.session_state.get("show_critical", True)
    show_healed = st.session_state.get("show_healed", True)
    show_spof = st.session_state.get("show_spof", True)
    scenario = st.session_state.get("scenario", "Road closure")
    
    sim_map = build_map(
        features,
        criticality,
        graph,
        int(st.session_state["selected_node"]),
        simulation,
        show_critical,
        show_healed,
        show_spof,
        scenario,
    )

    view_mode = st.session_state.get("view_mode", "Interactive Map")
    map_column, panel_column = st.columns([6.5, 3.5], gap="medium")

    with map_column:
        if view_mode == "Side-by-Side Comparison":
            baseline_map = build_map(
                features,
                criticality,
                graph,
                int(st.session_state["selected_node"]),
                None,
                show_critical,
                show_healed,
                show_spof,
                scenario,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline Network**")
                baseline_map_state = st_folium(
                    baseline_map,
                    height=640,
                    use_container_width=True,
                    returned_objects=["last_object_clicked"],
                    key=f"baseline_map_{st.session_state.get('reset_counter', 0)}_{st.session_state['selected_node']}_{show_critical}_{show_healed}_{show_spof}_{scenario}",
                )
            with col2:
                st.markdown("**Post-Failure Network**")
                map_state = st_folium(
                    sim_map,
                    height=640,
                    use_container_width=True,
                    returned_objects=["last_object_clicked", "all_drawings"],
                    key=f"network_map_{st.session_state.get('reset_counter', 0)}_{st.session_state['selected_node']}_{show_critical}_{show_healed}_{show_spof}_{scenario}",
                )
        else:
            baseline_map_state = None
            map_state = st_folium(
                sim_map,
                height=640,
                use_container_width=True,
                returned_objects=["last_object_clicked", "all_drawings"],
                key=f"network_map_{st.session_state.get('reset_counter', 0)}_{st.session_state['selected_node']}_{show_critical}_{show_healed}_{show_spof}_{scenario}",
            )

        st.caption(
            "Brighter roads connect more critical junctions · dashed = healed · "
            "orange = reroute · red = disabled"
        )
        render_live_detection()

    clicked = map_state.get("last_object_clicked") if map_state else None
    drawings = map_state.get("all_drawings") if map_state else None

    if scenario == "Flood" and drawings is not None:
        from shapely.geometry import shape
        flooded_nodes = set()
        nodes_gdf, _ = split_features(features)
        for drawing in drawings:
            geom_type = drawing.get("geometry", {}).get("type")
            if geom_type in ("Polygon", "MultiPolygon"):
                geom = shape(drawing["geometry"])
                inside = nodes_gdf[nodes_gdf.geometry.within(geom)]["node_id"].astype(int).tolist()
                flooded_nodes.update(inside)
        
        new_disabled = tuple(sorted(flooded_nodes))
        if new_disabled != st.session_state.get("disabled_nodes", ()):
            st.session_state["disabled_nodes"] = new_disabled
            st.rerun()
    if not clicked and baseline_map_state:
        clicked = baseline_map_state.get("last_object_clicked")
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

    with panel_column:
        render_panel(features, criticality, simulation, resilience_curve)


if __name__ == "__main__":
    main()
