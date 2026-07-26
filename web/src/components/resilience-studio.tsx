"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CompareControl } from "@/components/compare-control";
import { ExportMenu } from "@/components/export-menu";
import { LayerToggles } from "@/components/layer-toggles";
import { MetricStrip } from "@/components/metric-strip";
import { ModeSwitcher } from "@/components/mode-switcher";
import { RankingsTable } from "@/components/rankings-table";
import { ResilienceChart } from "@/components/resilience-chart";
import { ScenarioPanel } from "@/components/scenario-panel";
import { UploadDialog } from "@/components/upload-dialog";
import {
  ApiError,
  fetchAoi,
  fetchGraph,
  runSimulation,
  type AoiSummary,
  type GeoJsonCollection,
  type SimulationResult,
} from "@/lib/api";
import { formatMetric } from "@/lib/model";
import type { WorkspaceMode } from "@/lib/types";

const NetworkMap = dynamic(() => import("@/components/network-map"), {
  ssr: false,
  loading: () => <div className="map-loading" role="status"><span /> Preparing the field atlas…</div>,
});

const MODES = new Set<WorkspaceMode>(["explore", "stress", "compare", "recover"]);

function readInitialState() {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  const requestedMode = params.get("mode") as WorkspaceMode | null;
  const selectedParam = params.get("junction");
  const selected = selectedParam == null ? Number.NaN : Number(selectedParam);
  const failed = (params.get("failed") ?? "")
    .split(",")
    .map(Number)
    .filter(Number.isInteger);
  return {
    mode: requestedMode && MODES.has(requestedMode) ? requestedMode : "explore" as WorkspaceMode,
    selected: Number.isInteger(selected) ? selected : null,
    failed: [...new Set(failed)].sort((a, b) => a - b),
  };
}

export function ResilienceStudio() {
  const [summary, setSummary] = useState<AoiSummary | null>(null);
  const [graph, setGraph] = useState<GeoJsonCollection | null>(null);
  const [mode, setMode] = useState<WorkspaceMode>("explore");
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [removedNodes, setRemovedNodes] = useState<number[]>([]);
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [layers, setLayers] = useState({ critical: true, bridged: true, spof: true });
  const [compareValue, setCompareValue] = useState(58);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const simulationRun = useRef(0);

  useEffect(() => {
    const initial = readInitialState();
    let active = true;
    if (initial) queueMicrotask(() => {
      if (!active) return;
      setMode(initial.mode);
      setSelectedNode(initial.selected);
      setRemovedNodes(initial.failed);
    });
    fetchAoi("panaji_demo")
      .then(async (value) => ({ value, graph: await fetchGraph(value.graph_url) }))
      .then(({ value, graph: graphData }) => {
        if (!active) return;
        setSummary(value);
        setGraph(graphData);
        const selectable = new Set(value.critical_nodes.map((row) => row.node_id));
        setSelectedNode((current) => current != null && selectable.has(current)
          ? current
          : value.critical_nodes[0]?.node_id ?? null);
        setRemovedNodes((current) => current.filter((nodeId) => selectable.has(nodeId)));
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "The atlas could not be loaded");
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!summary) return;
    const params = new URLSearchParams();
    if (mode !== "explore") params.set("mode", mode);
    if (selectedNode != null) params.set("junction", String(selectedNode));
    if (removedNodes.length) params.set("failed", removedNodes.join(","));
    const query = params.toString();
    window.history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
  }, [mode, removedNodes, selectedNode, summary]);

  const selected = useMemo(
    () => summary?.critical_nodes.find((row) => row.node_id === selectedNode) ?? null,
    [selectedNode, summary],
  );

  const runFor = useCallback(async (nodes: number[]) => {
    const run = ++simulationRun.current;
    if (!nodes.length) {
      setSimulation(null);
      setBusy(false);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const next = await runSimulation("panaji_demo", nodes);
      if (run === simulationRun.current) setSimulation(next);
    } catch (reason) {
      if (run === simulationRun.current) {
        setError(reason instanceof ApiError ? reason.message : "The scenario could not be completed");
      }
    } finally {
      if (run === simulationRun.current) setBusy(false);
    }
  }, []);

  const toggleRemoved = useCallback((nodeId: number) => {
    const next = removedNodes.includes(nodeId)
      ? removedNodes.filter((value) => value !== nodeId)
      : [...removedNodes, nodeId].sort((a, b) => a - b);
    setRemovedNodes(next);
    if (mode === "recover") void runFor(next);
  }, [mode, removedNodes, runFor]);

  const changeMode = useCallback((next: WorkspaceMode) => {
    setMode(next);
    if (next === "recover") void runFor(removedNodes);
  }, [removedNodes, runFor]);

  useEffect(() => {
    const restore = () => {
      const initial = readInitialState();
      if (!initial) return;
      setMode(initial.mode);
      setSelectedNode(initial.selected);
      setRemovedNodes(initial.failed);
      if (initial.mode === "recover") void runFor(initial.failed);
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, [runFor]);

  const resetScenario = useCallback(() => {
    setRemovedNodes([]);
    setSimulation(null);
  }, []);

  if (error && !summary) {
    return (
      <main className="fatal-state">
        <span className="brand-mark">TRACE</span>
        <h1>The field atlas is temporarily unavailable.</h1>
        <p>{error}</p>
        <button type="button" onClick={() => window.location.reload()}>Try again</button>
      </main>
    );
  }

  if (!summary || !graph) {
    return (
      <main className="app-loading" aria-busy="true">
        <div className="loading-brand"><span>TRACE</span><small>Road resilience field atlas</small></div>
        <div className="loading-rule"><i /></div>
        <p>Loading validated network evidence…</p>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace-panel">Skip to workspace</a>
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">TRACE</span>
          <span><strong>Route resilience</strong><small>Network field atlas</small></span>
        </div>
        <ModeSwitcher mode={mode} onChange={changeMode} />
        <nav className="top-actions" aria-label="Project actions">
          <Link href="/methodology">Method</Link>
          <ExportMenu summary={summary} graph={graph} removedNodes={removedNodes} simulation={simulation} />
          <button type="button" className="upload-trigger" onClick={() => setUploadOpen(true)}>
            Analyze imagery
            <svg aria-hidden="true" viewBox="0 0 18 18"><path d="M9 3v12M3 9h12" /></svg>
          </button>
        </nav>
      </header>

      <main className="workspace" id="workspace-panel" role="tabpanel" aria-labelledby={`mode-tab-${mode}`}>
        <aside className="left-rail" aria-label="Analysis controls">
          <section className="place-card">
            <div><span className="live-dot" /> Verified sample</div>
            <p className="eyebrow">Area of interest · IN-GA</p>
            <h1>{summary.label}</h1>
            <p>15.49° N&nbsp; / &nbsp;73.83° E</p>
          </section>

          <ScenarioPanel
            mode={mode}
            selected={selected}
            criticalNodes={summary.critical_nodes}
            removed={removedNodes}
            busy={busy}
            onToggleRemoved={toggleRemoved}
            onRun={() => void runFor(removedNodes)}
            onReset={resetScenario}
          />
          {mode === "compare" ? <CompareControl value={compareValue} onChange={setCompareValue} /> : null}
          <LayerToggles layers={layers} onChange={setLayers} />

          <section className="ranking-card">
            <div className="section-heading">
              <div><span className="eyebrow">Priority index</span><h2>Critical junctions</h2></div>
              <span>{summary.critical_count} flagged</span>
            </div>
            <RankingsTable
              rows={summary.critical_nodes.slice(0, 12)}
              selectedNode={selectedNode}
              onSelect={setSelectedNode}
            />
          </section>
        </aside>

        <section className="map-stage" aria-label="Network map workspace">
          <MetricStrip
            nodeCount={summary.node_count}
            edgeCount={summary.edge_count}
            criticalCount={summary.critical_count}
            simulation={simulation}
          />
          <NetworkMap
            graph={graph}
            bounds={summary.bounds}
            mode={mode}
            layers={layers}
            selectedNode={selectedNode}
            removedNodes={removedNodes}
            compareValue={compareValue}
            onSelectNode={setSelectedNode}
          />
          <div className="map-legend" aria-label="Map legend">
            <span><i className="legend-road" /> Road network</span>
            <span><i className="legend-critical" /> Critical</span>
            <span><i className="legend-failed" /> Failed</span>
            <span><i className="legend-inferred" /> Inferred</span>
          </div>
        </section>

        <aside className="insight-rail" aria-label="Selected junction insight">
          <div className="insight-topline"><span>Field insight</span><span>#{selected?.rank ?? "—"}</span></div>
          {selected ? (
            <>
              <div className="junction-glyph" aria-hidden="true"><span /><span /><span /><i /></div>
              <p className="eyebrow">Selected junction</p>
              <h2>J-{selected.node_id}</h2>
              <p className="insight-lede">
                This junction carries {formatMetric(selected.betweenness * 100, { digits: 1 })}% of normalized routing centrality in the sample network.
              </p>
              <dl className="insight-metrics">
                <div><dt>Criticality</dt><dd>{formatMetric(selected.betweenness, { digits: 3 })}</dd></div>
                <div><dt>Network rank</dt><dd>{selected.rank} / {summary.node_count}</dd></div>
                <div><dt>Failure role</dt><dd>{selected.is_articulation ? "Single point" : "High-flow"}</dd></div>
              </dl>
              <div className="analyst-note">
                <span>Why it matters</span>
                <p>{selected.is_articulation
                  ? "Removing it separates at least one reachable part of the road graph. Prioritize redundancy nearby."
                  : "Many shortest routes converge here, so disruption can redistribute travel across the wider network."}</p>
              </div>
              <button type="button" className="button-primary full" onClick={() => {
                setMode("stress");
                setRemovedNodes((current) => current.includes(selected.node_id)
                  ? current
                  : [...current, selected.node_id].sort((a, b) => a - b));
              }}>
                Test this junction
              </button>
              {simulation?.representative_route ? (
                <div className="route-note" role="status">
                  <span>Representative reroute</span>
                  <strong>{simulation.representative_route.disconnected
                    ? "Connection lost"
                    : `+${formatMetric(simulation.representative_route.travel_time_delta_pct, { digits: 1, suffix: "%" })}`}</strong>
                  <p>J-{simulation.representative_route.origin} → J-{simulation.representative_route.destination}</p>
                </div>
              ) : null}
            </>
          ) : <p>Select a critical junction from the map or ranking.</p>}
          <footer><span>Metric</span><strong>Baseline-normalized global efficiency</strong></footer>
        </aside>

        <section className="timeline-panel" aria-label="Resilience timeline">
          <div className="timeline-copy">
            <span className="eyebrow">Progressive stress</span>
            <h2>How quickly does the city fragment?</h2>
            <p>Targeted loss should fall faster than the seeded random reference.</p>
          </div>
          <ResilienceChart rows={summary.resilience_curve} activeStep={removedNodes.length} />
          <div className="timeline-readout">
            <span>Scenario step</span>
            <strong>{String(removedNodes.length).padStart(2, "0")}</strong>
            <small>{removedNodes.length ? "junctions removed" : "baseline"}</small>
          </div>
        </section>
      </main>

      {error && summary ? <div className="toast" role="alert">{error}<button type="button" onClick={() => setError(null)}>Dismiss</button></div> : null}
      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} />
    </div>
  );
}
