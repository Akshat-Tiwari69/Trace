"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { useEffect, useRef, useState } from "react";
import maplibregl, { type FilterSpecification, type GeoJSONSource, type StyleSpecification } from "maplibre-gl";

import type { GeoJsonCollection } from "@/lib/api";
import type { WorkspaceMode } from "@/lib/types";

type Layers = { critical: boolean; bridged: boolean; spof: boolean };

type Props = {
  graph: GeoJsonCollection;
  bounds: [number, number, number, number];
  mode: WorkspaceMode;
  layers: Layers;
  selectedNode: number | null;
  removedNodes: number[];
  compareValue: number;
  onSelectNode: (nodeId: number) => void;
};

const EDGE_FILTER: FilterSpecification = ["==", ["get", "feature_type"], "edge"];
const BASEMAP_STYLE = "https://tiles.openfreemap.org/styles/positron";
const ROAD_SHIELD_LAYERS = new Set([
  "highway-shield-non-us",
  "highway-shield-us-interstate",
  "road_shield_us",
]);

export function nullSafeBasemapStyle(
  _previous: StyleSpecification | undefined,
  next: StyleSpecification,
): StyleSpecification {
  return {
    ...next,
    layers: next.layers.map((layer) => {
      if (!ROAD_SHIELD_LAYERS.has(layer.id) || !("filter" in layer) || !Array.isArray(layer.filter)) return layer;
      const conditions = (layer.filter[0] === "all" ? layer.filter.slice(1) : [layer.filter]) as FilterSpecification[];
      const numericRefLength: FilterSpecification = ["==", ["typeof", ["get", "ref_length"]], "number"];
      const filter = ["all", numericRefLength, ...conditions] as unknown as FilterSpecification;
      return { ...layer, filter };
    }),
  };
}

function nodeFilter(property: string, value: unknown): FilterSpecification {
  return ["all", ["==", ["get", "feature_type"], "node"], ["==", ["get", property], value]] as FilterSpecification;
}

function failedFilter(nodes: number[]): FilterSpecification {
  if (!nodes.length) return ["==", 1, 0];
  return [
    "all",
    EDGE_FILTER,
    ["any", ["in", ["get", "u"], ["literal", nodes]], ["in", ["get", "v"], ["literal", nodes]]],
  ] as FilterSpecification;
}

export default function NetworkMap({
  graph,
  bounds,
  mode,
  layers,
  selectedNode,
  removedNodes,
  compareValue,
  onSelectNode,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const selectRef = useRef(onSelectNode);
  const stateRef = useRef({ layers, mode, removedNodes, selectedNode, compareValue });
  const [basemapWarning, setBasemapWarning] = useState(false);

  useEffect(() => { selectRef.current = onSelectNode; }, [onSelectNode]);
  useEffect(() => {
    stateRef.current = { layers, mode, removedNodes, selectedNode, compareValue };
  }, [compareValue, layers, mode, removedNodes, selectedNode]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      bounds: [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
      fitBoundsOptions: { padding: 56, duration: 0 },
      cooperativeGestures: true,
      fadeDuration: 0,
      attributionControl: false,
      maxZoom: 19,
      minZoom: 11,
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");

    map.on("error", (event) => {
      const message = String(event.error?.message ?? "");
      if (/style|tile|source|sprite|glyph/i.test(message)) setBasemapWarning(true);
    });
    map.on("load", () => {
      const canvas = map.getCanvas();
      canvas.setAttribute("aria-label", "Interactive road resilience map of Panaji");
      canvas.setAttribute("role", "region");
      map.addSource("network", { type: "geojson", data: graph as never });
      map.addLayer({
        id: "roads",
        type: "line",
        source: "network",
        filter: EDGE_FILTER,
        paint: {
          "line-color": "#173f37",
          "line-opacity": 0.82,
          "line-width": ["interpolate", ["linear"], ["zoom"], 11, 1.1, 16, 3.2],
        },
      });
      map.addLayer({
        id: "bridged-roads",
        type: "line",
        source: "network",
        filter: ["all", ["==", ["get", "feature_type"], "edge"], ["==", ["get", "is_bridged"], true]] as FilterSpecification,
        paint: {
          "line-color": "#318b74",
          "line-width": ["interpolate", ["linear"], ["zoom"], 11, 2, 16, 4.5],
          "line-dasharray": [1.4, 1.2],
        },
      });
      map.addLayer({
        id: "failed-roads",
        type: "line",
        source: "network",
        filter: failedFilter(stateRef.current.removedNodes),
        paint: {
          "line-color": "#f05a3c",
          "line-opacity": 0.95,
          "line-width": ["interpolate", ["linear"], ["zoom"], 11, 2.8, 16, 6],
          "line-dasharray": [0.4, 1.2],
        },
      });
      map.addLayer({
        id: "node-hit",
        type: "circle",
        source: "network",
        filter: nodeFilter("is_critical", true),
        paint: { "circle-radius": 12, "circle-opacity": 0 },
      });
      map.addLayer({
        id: "critical-nodes",
        type: "circle",
        source: "network",
        filter: nodeFilter("is_critical", true),
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 3.2, 16, 7.5],
          "circle-color": [
            "interpolate", ["linear"], ["coalesce", ["get", "betweenness"], 0],
            0, "#71c2a2", 0.16, "#e5b84e", 0.34, "#f05a3c",
          ],
          "circle-stroke-color": "#f4f0e6",
          "circle-stroke-width": 1.4,
          "circle-opacity": 0.92,
        },
      });
      map.addLayer({
        id: "spof-nodes",
        type: "circle",
        source: "network",
        filter: nodeFilter("is_articulation", true),
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 5, 16, 10],
          "circle-color": "rgba(229,184,78,.2)",
          "circle-stroke-color": "#7d5910",
          "circle-stroke-width": 2,
        },
      });
      map.addLayer({
        id: "failed-nodes",
        type: "circle",
        source: "network",
        filter: nodeFilter("node_id", -1),
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 7, 16, 13],
          "circle-color": "#f05a3c",
          "circle-stroke-color": "#0b1413",
          "circle-stroke-width": 2.5,
        },
      });
      map.addLayer({
        id: "selected-node",
        type: "circle",
        source: "network",
        filter: nodeFilter("node_id", stateRef.current.selectedNode ?? -1),
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 8, 16, 14],
          "circle-color": "rgba(11,20,19,.08)",
          "circle-stroke-color": "#0b1413",
          "circle-stroke-width": 3,
        },
      });
      map.on("mouseenter", "node-hit", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "node-hit", () => { map.getCanvas().style.cursor = ""; });
      map.on("click", "node-hit", (event) => {
        const nodeId = Number(event.features?.[0]?.properties?.node_id);
        if (Number.isInteger(nodeId)) selectRef.current(nodeId);
      });
      const state = stateRef.current;
      map.setFilter("failed-roads", failedFilter(state.removedNodes));
      map.setFilter("failed-nodes", ["all", ["==", ["get", "feature_type"], "node"], ["in", ["get", "node_id"], ["literal", state.removedNodes]]] as FilterSpecification);
      map.setFilter("selected-node", nodeFilter("node_id", state.selectedNode ?? -1));
      map.setLayoutProperty("critical-nodes", "visibility", state.layers.critical ? "visible" : "none");
      map.setLayoutProperty("bridged-roads", "visibility", state.layers.bridged ? "visible" : "none");
      map.setLayoutProperty("spof-nodes", "visibility", state.layers.spof ? "visible" : "none");
    });
    map.setStyle(BASEMAP_STYLE, { transformStyle: nullSafeBasemapStyle });
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [bounds, graph]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.isStyleLoaded()) return;
    (map.getSource("network") as GeoJSONSource | undefined)?.setData(graph as never);
  }, [graph]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getLayer("failed-roads")) return;
    map.setFilter("failed-roads", failedFilter(removedNodes));
    map.setFilter("failed-nodes", ["all", ["==", ["get", "feature_type"], "node"], ["in", ["get", "node_id"], ["literal", removedNodes]]] as FilterSpecification);
    map.setFilter("selected-node", nodeFilter("node_id", selectedNode ?? -1));
    map.setLayoutProperty("critical-nodes", "visibility", layers.critical ? "visible" : "none");
    map.setLayoutProperty("bridged-roads", "visibility", layers.bridged ? "visible" : "none");
    map.setLayoutProperty("spof-nodes", "visibility", layers.spof ? "visible" : "none");
    const comparison = mode === "compare" ? compareValue / 100 : 1;
    map.setPaintProperty("roads", "line-opacity", mode === "compare" ? 1 - comparison * 0.55 : 0.82);
    map.setPaintProperty("failed-roads", "line-opacity", Math.max(0.12, comparison));
  }, [compareValue, layers, mode, removedNodes, selectedNode]);

  return (
    <div className="map-frame">
      <div ref={containerRef} className="network-map" />
      <p className="sr-only">Use arrow keys to pan the map, plus and minus to zoom. All critical junctions are also available in the ranking table.</p>
      {mode === "compare" ? (
        <div className="compare-map-labels" aria-hidden="true">
          <span>Baseline</span><span>Scenario</span>
          <i style={{ left: `${compareValue}%` }} />
        </div>
      ) : null}
      {basemapWarning ? (
        <div className="map-warning" role="status">Basemap tiles are unavailable. Network data remains interactive.</div>
      ) : null}
    </div>
  );
}
