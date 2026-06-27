"""Interactive Streamlit dashboard for road-network resilience."""

from dataclasses import dataclass
from itertools import combinations
from math import inf, isfinite
from pathlib import Path

import io
from PIL import Image, ImageDraw, ImageFont

import branca.colormap as cm
import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

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
    """Metrics and route state produced by one node ablation."""

    disabled_node: int
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


@st.cache_data(show_spinner=False)
def graph_from_features(_features: gpd.GeoDataFrame) -> nx.Graph:
    """Convert the map-ready GeoJSON features into a routable graph."""
    nodes, edges = split_features(_features)
    graph = nx.Graph()
    for _, node in nodes.iterrows():
        node_id = int(node["node_id"])
        graph.add_node(
            node_id,
            x=float(node.geometry.x),
            y=float(node.geometry.y),
            betweenness=float(node.get("betweenness", 0.0)),
            is_critical=bool(node.get("is_critical", False)),
        )
    for _, edge in edges.iterrows():
        graph.add_edge(
            int(edge["u"]),
            int(edge["v"]),
            length_m=float(edge["length_m"]),
            is_bridged=bool(edge.get("is_bridged", False)),
            coordinates=tuple(
                (float(longitude), float(latitude))
                for longitude, latitude in edge.geometry.coords
            ),
        )
    return graph


def generate_summary_png(criticality: pd.DataFrame, simulation: SimulationResult | None) -> bytes:
    """Generate a one-page PNG summary of the network criticality and current resilience."""
    img = Image.new("RGB", (800, 600), color=(18, 18, 18))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None
    
    # Title
    draw.text((30, 30), "Route Resilience - Network Summary Report", fill=(255, 255, 255), font=font)
    
    # Simulation Status
    y_offset = 80
    draw.text((30, y_offset), "Simulation Status:", fill=(200, 200, 200), font=font)
    y_offset += 30
    if simulation:
        draw.text((50, y_offset), f"- Disabled Node: {simulation.disabled_node}", fill=(255, 75, 75), font=font)
        y_offset += 25
        draw.text((50, y_offset), f"- Resilience Index: {simulation.resilience_index:.4f}", fill=(0, 212, 255), font=font)
        y_offset += 25
        draw.text((50, y_offset), f"- Connected Component Size: {simulation.largest_cc_fraction * 100:.1f}%", fill=(255, 255, 255), font=font)
        y_offset += 25
        if simulation.route:
            draw.text((50, y_offset), f"- Impact: {simulation.route.travel_time_delta_pct:+.1f}% travel time", fill=(255, 255, 255), font=font)
    else:
        draw.text((50, y_offset), "Baseline Network (No failure simulated).", fill=(255, 255, 255), font=font)
        
    # Criticality Table
    y_offset += 60
    draw.text((30, y_offset), "Top 10 Critical Junctions:", fill=(200, 200, 200), font=font)
    y_offset += 30
    draw.text((50, y_offset), f"{'Rank':<10} {'Node ID':<15} {'Betweenness':<15}", fill=(255, 255, 255), font=font)
    y_offset += 20
    draw.line([(50, y_offset), (350, y_offset)], fill=(100, 100, 100), width=1)
    y_offset += 10
    
    top_nodes = criticality.head(10)
    for _, row in top_nodes.iterrows():
        draw.text((50, y_offset), f"{row['rank']:<10} {int(row['node_id']):<15} {row['betweenness']:.4f}", fill=(255, 255, 255), font=font)
        y_offset += 25

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


@st.cache_data(show_spinner=False)
def simulate_ablation(graph_fingerprint: str, _graph: nx.Graph, node: int) -> SimulationResult:
    """Disable one node and compute resilience plus a representative reroute."""
    metrics = resilience_index(_graph, removed_nodes=[node])
    return SimulationResult(
        disabled_node=node,
        resilience_index=float(metrics["resilience_index"]),
        largest_cc_fraction=float(metrics["largest_cc_fraction"]),
        route=representative_reroute(_graph, node),
    )


def semantic_legend() -> folium.Element:
    """Create a labelled map legend for semantic route states."""
    html = """
    <div style="position: fixed; bottom: 36px; right: 12px; z-index: 9999;
                background: #1e1e2e; color: #ffffff; padding: 10px 12px;
                border: 1px solid #555; border-radius: 6px; font-size: 12px;">
      <b>Network states</b><br>
      <span style="color:#00d4ff">●</span> selected junction<br>
      <span style="color:#ff4b4b">●</span> disabled junction / links<br>
      <span style="color:#ff8c42">━</span> rerouted path<br>
      <span style="color:#aaaaaa">┄</span> healed road
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
            color="#ff8c42",
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
) -> folium.Map:
    """Build the map with criticality, selection, failure, and reroute states."""
    nodes, edges = split_features(features)
    scores = criticality.set_index("node_id")["betweenness"].to_dict()
    maximum = max(float(criticality["betweenness"].max()), 1e-9)
    colour_scale = cm.LinearColormap(
        colors=["#440154", "#31688e", "#35b779", "#fde725"],
        vmin=0.0,
        vmax=maximum,
        caption="Road criticality (endpoint betweenness: low to high)",
    )
    disabled_node = simulation.disabled_node if simulation else None

    road_map = folium.Map(
        location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()],
        zoom_start=15,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    for _, edge in edges.iterrows():
        is_bridged = bool(edge["is_bridged"])
        if is_bridged and not show_healed:
            continue
        start, end = int(edge["u"]), int(edge["v"])
        score = max(float(scores.get(start, 0.0)), float(scores.get(end, 0.0)))
        is_disabled = disabled_node in {start, end}
        colour = "#ff4b4b" if is_disabled else colour_scale(score)
        state = "disabled link" if is_disabled else (
            "healed link" if is_bridged else "observed link"
        )
        coordinates = [
            (latitude, longitude) for longitude, latitude in edge.geometry.coords
        ]
        folium.PolyLine(
            coordinates,
            color=colour,
            weight=4 if is_disabled or is_bridged else 3,
            opacity=0.45 if is_disabled else 0.85,
            dash_array="8 6" if is_bridged or is_disabled else None,
            tooltip=f"Road {start}–{end} · criticality {score:.3f} · {state}",
        ).add_to(road_map)

    critical_ids = set(criticality.loc[criticality["is_critical"], "node_id"].astype(int))
    if show_critical:
        for _, node in nodes[nodes["node_id"].isin(critical_ids)].iterrows():
            node_id = int(node["node_id"])
            score = float(scores.get(node_id, 0.0))
            if node_id == disabled_node:
                colour, radius, label = "#ff4b4b", 9, "Disabled junction"
            elif node_id == selected_node:
                colour, radius, label = "#00d4ff", 8, "Selected junction"
            else:
                colour, radius, label = colour_scale(score), 5, "Critical junction"
            folium.CircleMarker(
                location=[node.geometry.y, node.geometry.x],
                radius=radius,
                color="#ffffff",
                weight=2 if node_id in {selected_node, disabled_node} else 1,
                fill=True,
                fill_color=colour,
                fill_opacity=1.0,
                tooltip=f"{label} {node_id} · score {score:.3f}",
            ).add_to(road_map)

    if simulation and simulation.route:
        add_rerouted_path(road_map, graph, simulation.route)
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
        color="#ff8c42",
        height=180,
    )

    if route.delay_segments:
        st.caption("Top delay contributors on the detour")
        delays = pd.DataFrame(
            route.delay_segments,
            columns=["Road", "Delay contribution (%)"],
        ).set_index("Road")
        st.bar_chart(delays, color="#ff8c42", height=190)


def render_panel(
    features: gpd.GeoDataFrame,
    criticality: pd.DataFrame,
    simulation: SimulationResult | None,
    resilience_curve: pd.DataFrame | None = None,
) -> None:
    """Render controls, metrics, ranked hotspots, and live charts."""
    nodes, edges = split_features(features)
    critical_nodes = criticality[criticality["is_critical"]].sort_values("rank")
    critical_ids = critical_nodes["node_id"].astype(int).tolist()
    scores = criticality.set_index("node_id")["betweenness"].to_dict()
    ranks = criticality.set_index("node_id")["rank"].to_dict()

    st.title("Route Resilience")
    st.caption("Panaji demo · live junction-failure simulation")

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
    travel_value = "Route cut" if simulation and not isfinite(travel_delta) else f"+{travel_delta:.1f}%"
    travel_column.metric(
        "Travel-time impact",
        travel_value,
        help="Exact route-length change at constant speed; no speed data is assumed.",
    )

    st.subheader("Scenario controls")
    view_mode = st.radio("View Mode", ["Interactive Map", "Side-by-Side Comparison"], horizontal=True, key="view_mode")
    st.selectbox("Region", ["Panaji demo"], disabled=True)
    scenario = st.selectbox("Scenario", ["Road closure", "Accident", "Flood"])
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
        st.session_state["disabled_node"] = int(selected)
        st.rerun()
    if reset_column.button("Reset", use_container_width=True):
        st.session_state["disabled_node"] = None
        st.rerun()

    layer_one, layer_two = st.columns(2)
    layer_one.checkbox("Critical nodes", value=True, key="show_critical")
    layer_two.checkbox("Healed roads", value=True, key="show_healed")

    if simulation:
        st.success(f"Junction {simulation.disabled_node} is disabled.")
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
            color=["#ff4b4b", "#00d4ff"] if len(chart_data.columns) == 2 else None,
            height=200,
        )

    st.subheader("Top critical junctions")
    top_nodes = critical_nodes[["rank", "node_id", "betweenness"]].head(5).copy()
    top_nodes["betweenness"] = top_nodes["betweenness"].map(lambda value: f"{value:.3f}")
    st.dataframe(
        top_nodes,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Export / Report")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Download GeoJSON",
            data=features.to_json(),
            file_name="network_export.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
    with col_dl2:
        png_data = generate_summary_png(criticality, simulation)
        st.download_button(
            label="Download Summary PNG",
            data=png_data,
            file_name="resilience_summary.png",
            mime="image/png",
            use_container_width=True,
        )
    st.caption(f"Network: {len(nodes):,} junctions · {len(edges):,} road links")


def apply_design_theme() -> None:
    """Apply the restrained dark mission-control styling from Design.md."""
    st.markdown(
        """
        <style>
          .stApp { background: #121212; }
          [data-testid="stMetric"] {
            background: #1e1e2e;
            border: 1px solid #343447;
            border-radius: 8px;
            padding: 10px 12px;
          }
          [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
          .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the interactive F2 dashboard."""
    st.set_page_config(page_title="Route Resilience", layout="wide")
    apply_design_theme()
    try:
        features, criticality = load_sample_data()
        graph = graph_from_features(features)
        resilience_curve = load_resilience_curve()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not load the sample network: {error}")
        st.stop()

    critical_ids = set(
        criticality.loc[criticality["is_critical"], "node_id"].astype(int)
    )
    if "selected_node" not in st.session_state:
        st.session_state["selected_node"] = int(
            criticality.sort_values("rank").iloc[0]["node_id"]
        )
    disabled_node = st.session_state.get("disabled_node")
    with st.spinner("Simulating junction failure…") if disabled_node is not None else st.empty():
        simulation = (
            simulate_ablation("panaji_demo_v1", graph, int(disabled_node))
            if disabled_node is not None
            else None
        )

    show_critical = st.session_state.get("show_critical", True)
    show_healed = st.session_state.get("show_healed", True)
    show_spof = st.session_state.get("show_spof", False)
    sim_map = build_map(
        features,
        criticality,
        graph,
        int(st.session_state["selected_node"]),
        simulation,
        show_critical,
        show_healed,
        show_spof,
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
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline Network**")
                st_folium(
                    baseline_map,
                    height=500,
                    use_container_width=True,
                    key=f"baseline_map_{disabled_node}_{show_critical}_{show_healed}_{show_spof}",
                )
            with col2:
                st.markdown("**Post-Failure Network**")
                map_state = st_folium(
                    sim_map,
                    height=500,
                    use_container_width=True,
                    returned_objects=["last_object_clicked"],
                    key=f"network_map_{disabled_node}_{show_critical}_{show_healed}_{show_spof}",
                )
        else:
            map_state = st_folium(
                sim_map,
                height=760,
                use_container_width=True,
                returned_objects=["last_object_clicked"],
                key=f"network_map_{disabled_node}_{show_critical}_{show_healed}_{show_spof}",
            )
        
        st.caption(
            "Brighter roads connect more critical junctions · dashed = healed · "
            "orange = reroute · red = disabled"
        )
        
    clicked = map_state.get("last_object_clicked") if 'map_state' in locals() and map_state else None
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
