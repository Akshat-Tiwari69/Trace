"use client";

import { useRef } from "react";

import type { AoiSummary, GeoJsonCollection, SimulationResult } from "@/lib/api";
import { criticalityCsv, scenarioGeoJson } from "@/lib/model";

type Props = {
  summary: AoiSummary;
  graph: GeoJsonCollection;
  removedNodes: number[];
  simulation: SimulationResult | null;
};

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function ExportMenu({ summary, graph, removedNodes, simulation }: Props) {
  const details = useRef<HTMLDetailsElement>(null);
  const stem = `${summary.aoi}-${removedNodes.length ? "scenario" : "baseline"}`;

  function done() {
    if (details.current) details.current.open = false;
  }

  return (
    <details className="export-menu" ref={details}>
      <summary>Export <svg aria-hidden="true" viewBox="0 0 16 16"><path d="M8 2v9m0 0 3-3m-3 3L5 8M3 14h10" /></svg></summary>
      <div>
        <p>Portable evidence, generated in your browser.</p>
        <button type="button" onClick={() => {
          download(
            `${stem}.geojson`,
            JSON.stringify(scenarioGeoJson(graph, removedNodes, simulation), null, 2),
            "application/geo+json",
          );
          done();
        }}>
          <span>GeoJSON</span><small>Network + scenario state</small>
        </button>
        <button type="button" onClick={() => {
          download(
            `${stem}-criticality.csv`,
            criticalityCsv(summary.critical_nodes, removedNodes),
            "text/csv;charset=utf-8",
          );
          done();
        }}>
          <span>CSV</span><small>Ranked junction evidence</small>
        </button>
      </div>
    </details>
  );
}
