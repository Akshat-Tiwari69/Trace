"""Streamlit dashboard scaffold for the committed sample road network."""

from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


def find_repo_root() -> Path:
    """Find the repository root using its Tracker file as a stable marker."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "docs" / "Tracker.md").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


REPO_ROOT = find_repo_root()
SAMPLE_GEOJSON = REPO_ROOT / "data" / "sample" / "panaji_demo_graph.geojson"
SAMPLE_CRITICALITY = REPO_ROOT / "data" / "sample" / "panaji_demo_criticality.csv"


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


def build_map(features: gpd.GeoDataFrame, criticality: pd.DataFrame) -> folium.Map:
    """Build a dark Folium map with roads coloured by endpoint criticality."""
    nodes, edges = split_features(features)

    scores = criticality.set_index("node_id")["betweenness"].to_dict()
    maximum = max(float(criticality["betweenness"].max()), 1e-9)
    colour_scale = cm.LinearColormap(
        colors=["#440154", "#31688e", "#35b779", "#fde725"],
        vmin=0.0,
        vmax=maximum,
        caption="Road criticality (endpoint betweenness: low to high)",
    )

    road_map = folium.Map(
        location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()],
        zoom_start=15,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    for _, edge in edges.iterrows():
        score = max(
            float(scores.get(edge.get("u"), 0.0)),
            float(scores.get(edge.get("v"), 0.0)),
        )
        coordinates = [(latitude, longitude) for longitude, latitude in edge.geometry.coords]
        is_bridged = bool(edge.get("is_bridged", False))
        folium.PolyLine(
            coordinates,
            color=colour_scale(score),
            weight=4 if is_bridged else 3,
            opacity=0.9,
            dash_array="8 6" if is_bridged else None,
            tooltip=(
                f"Road {edge.get('u')}–{edge.get('v')} · "
                f"criticality {score:.3f} · "
                f"{'healed link' if is_bridged else 'observed link'}"
            ),
        ).add_to(road_map)

    critical_ids = set(criticality.loc[criticality["is_critical"], "node_id"])
    for _, node in nodes[nodes["node_id"].isin(critical_ids)].iterrows():
        score = float(scores.get(node["node_id"], 0.0))
        folium.CircleMarker(
            location=[node.geometry.y, node.geometry.x],
            radius=5,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color=colour_scale(score),
            fill_opacity=1.0,
            tooltip=f"Critical junction {node['node_id']} · score {score:.3f}",
        ).add_to(road_map)

    colour_scale.add_to(road_map)
    return road_map


def render_panel(features: gpd.GeoDataFrame, criticality: pd.DataFrame) -> None:
    """Render the read-only F1 summary panel."""
    nodes, edges = split_features(features)
    critical_nodes = criticality[criticality["is_critical"]].sort_values("rank")

    st.title("Route Resilience")
    st.caption("Sample network · Panaji demo")

    node_metric, road_metric = st.columns(2)
    node_metric.metric("Junctions", f"{len(nodes):,}")
    road_metric.metric("Road links", f"{len(edges):,}")

    st.subheader("Criticality overview")
    st.write(
        "Brighter roads connect more critical junctions. Dashed roads are healed "
        "links inferred by the graph pipeline."
    )
    st.metric("Critical junctions", f"{len(critical_nodes):,}")

    st.subheader("Top critical junctions")
    top_nodes = critical_nodes[["rank", "node_id", "betweenness"]].head(5).copy()
    top_nodes["betweenness"] = top_nodes["betweenness"].map(lambda value: f"{value:.3f}")
    st.dataframe(top_nodes, hide_index=True, use_container_width=True)

    st.info("Failure simulation and live resilience metrics arrive in F2.")


def main() -> None:
    """Render the F1 dashboard."""
    st.set_page_config(page_title="Route Resilience", layout="wide")
    try:
        features, criticality = load_sample_data()
        road_map = build_map(features, criticality)
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Could not load the sample network: {error}")
        st.stop()

    map_column, panel_column = st.columns([6.5, 3.5], gap="medium")
    with map_column:
        st_folium(road_map, height=720, use_container_width=True)
    with panel_column:
        render_panel(features, criticality)


if __name__ == "__main__":
    main()
