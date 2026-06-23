import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import branca.colormap as cm
from pathlib import Path

# --- Configuration ---
st.set_page_config(
    page_title="Route Resilience",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Data Loading ---
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent.parent.parent
    geojson_path = base_dir / "data" / "sample" / "panaji_demo_graph.geojson"
    csv_path = base_dir / "data" / "sample" / "panaji_demo_criticality.csv"
    
    gdf = gpd.read_file(geojson_path)
    df_crit = pd.read_csv(csv_path)
    
    return gdf, df_crit

def main():
    try:
        gdf, df_crit = load_data()
    except Exception as e:
        st.error(f"Failed to load sample data: {e}")
        return

    # Filter features
    nodes = gdf[gdf.geometry.type == 'Point']
    edges = gdf[gdf.geometry.type == 'LineString']

    # --- Layout ---
    col_map, col_panel = st.columns([6.5, 3.5])

    with col_panel:
        st.title("Route Resilience")
        st.subheader("Criticality Analysis")
        
        # Mock metrics for F1 (matches S1 sample output logs)
        resilience_index = 0.642
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Global Resilience Index", value=f"{resilience_index:.3f}")
        with col2:
            st.metric(label="Avg Travel Time Impact", value="+0.0%")
        
        st.markdown("### Top Critical Nodes")
        # Ensure we sort by rank if available, otherwise by betweenness
        top_nodes = df_crit[df_crit['is_critical'] == True].sort_values(
            by='rank' if 'rank' in df_crit.columns else 'betweenness',
            ascending=True if 'rank' in df_crit.columns else False
        ).head(5)
        st.dataframe(
            top_nodes[['node_id', 'betweenness', 'rank'] if 'rank' in top_nodes.columns else ['node_id', 'betweenness']],
            hide_index=True, 
            use_container_width=True
        )
        
        st.markdown("### Scenario Settings")
        st.selectbox("Select Scenario", ["Flood", "Accident", "Closure"])

    with col_map:
        if not nodes.empty:
            center_lat = nodes.geometry.y.mean()
            center_lon = nodes.geometry.x.mean()
        else:
            # Fallback coordinate if nodes are empty
            center_lat, center_lon = 15.4909, 73.8278

        # Dark theme map as per Design.md
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='cartodb dark_matter')
        
        # Color map for nodes
        vmax = nodes['betweenness'].max() if not nodes['betweenness'].empty and pd.notnull(nodes['betweenness'].max()) else 1.0
        # Viridis-like colourblind-safe ramp: blue -> teal -> yellow -> red
        colormap = cm.LinearColormap(colors=['#440154', '#31688e', '#35b779', '#fde725'], vmin=0, vmax=vmax)
        colormap.caption = 'Criticality (Betweenness)'
        m.add_child(colormap)
        
        # Add edges first so they are under nodes
        if not edges.empty:
            folium.GeoJson(
                edges,
                style_function=lambda feature: {
                    'color': '#555555',
                    'weight': 2,
                    'opacity': 0.7
                }
            ).add_to(m)
        
        # Add nodes
        for _, row in nodes.iterrows():
            b = row.get('betweenness', 0)
            color = colormap(b) if pd.notnull(b) else 'grey'
            is_crit = row.get('is_critical', False)
            radius = 6 if is_crit else 2
            
            # Using CircleMarker for junctions
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"Node {row.get('node_id', 'Unknown')}<br>Crit: {b:.4f}"
            ).add_to(m)
            
        st_folium(m, use_container_width=True, height=700)

if __name__ == "__main__":
    main()
