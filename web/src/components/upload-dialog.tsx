"use client";

/* Blob previews are already local user files; Next image optimization cannot improve them. */
/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useRef, useState, type ChangeEvent, type CSSProperties, type FormEvent, type PointerEvent } from "react";

import {
  fetchAnalysis,
  fetchAnalysisResult,
  fetchGraph,
  runAnalysisSimulation,
  submitAnalysis,
  type AnalysisJob,
  type AnalysisResult,
  type GeoJsonCollection,
  type SimulationResult,
} from "@/lib/api";
import type { CriticalNode } from "@/lib/types";
import { formatMetric } from "@/lib/model";

const MAX_BYTES = 11 * 1024 * 1024;
const MAX_SIDE = 4096;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png"]);

type Props = { open: boolean; onClose: () => void };

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

type UploadState = { job: AnalysisJob | null; busy: boolean; hasFile: boolean; consent: boolean; elapsed: number };

// Says what happens next. The upload request itself waits for GPU segmentation
// (a cold start can take ~30 s), so that stage needs its own visible progress.
export function statusCopy({ job, busy, hasFile, consent, elapsed }: UploadState) {
  if (!job) {
    if (busy) return `Uploading and extracting roads on the GPU · ${elapsed} s (a cold start can take ~30 s)`;
    if (!hasFile) return "Choose an image to begin";
    return consent ? "Ready to extract" : "Confirm processing consent to continue";
  }
  if (job.status === "queued") {
    return job.position > 0 ? `Queued · ${job.position} ahead` : "Queued · next in line";
  }
  if (job.status === "running") return "Extracting and testing the road network";
  return job.status === "done" ? "Analysis complete" : "Analysis failed";
}

export function UploadDialog({ open, onClose }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const runRef = useRef(0);
  const previewRef = useRef<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<[number, number] | null>(null);
  const [resolution, setResolution] = useState(0.5);
  const [consent, setConsent] = useState(false);
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [resultGraph, setResultGraph] = useState<GeoJsonCollection | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [failed, setFailed] = useState<CriticalNode | null>(null);
  const [stress, setStress] = useState<SimulationResult | null>(null);
  const [stressError, setStressError] = useState<string | null>(null);

  useEffect(() => {
    if (!busy) return;
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [busy]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
      queueMicrotask(() => fileInputRef.current?.focus());
    } else if (!open && dialog.open) {
      if (typeof dialog.close === "function") dialog.close();
      else dialog.removeAttribute("open");
    }
  }, [open]);

  useEffect(() => () => {
    runRef.current += 1;
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
  }, []);

  function rememberFile(next: File | null) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = next ? URL.createObjectURL(next) : null;
    setPreview(previewRef.current);
    setFile(next);
  }

  async function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const next = event.target.files?.[0] ?? null;
    setError(null);
      setResult(null);
      setResultGraph(null);
    setJob(null);
    if (!next) {
      rememberFile(null);
      return;
    }
    if (!IMAGE_TYPES.has(next.type)) {
      event.target.value = "";
      rememberFile(null);
      setError("Choose a PNG or JPEG image.");
      return;
    }
    if (next.size > MAX_BYTES) {
      event.target.value = "";
      rememberFile(null);
      setError("That image is larger than the 11 MiB upload limit.");
      return;
    }
    if (typeof createImageBitmap === "function") {
      const bitmap = await createImageBitmap(next);
      const tooLarge = bitmap.width > MAX_SIDE || bitmap.height > MAX_SIDE;
      setImageSize([bitmap.width, bitmap.height]);
      bitmap.close();
      if (tooLarge) {
        event.target.value = "";
        rememberFile(null);
        setError("Crop or resize the image to 4096 × 4096 pixels or smaller.");
        return;
      }
    }
    rememberFile(next);
  }

  async function analyze(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Choose a satellite image first.");
      fileInputRef.current?.focus();
      return;
    }
    if (!consent) {
      setError("Confirm external processing before uploading.");
      return;
    }
    const run = ++runRef.current;
    setBusy(true);
    setElapsed(0);
    setError(null);
    setResult(null);
    try {
      const queued = await submitAnalysis(file, resolution);
      if (run !== runRef.current) return;
      setJob(queued);
      const statusUrl = queued.status_url ?? `/api/v1/analyses/${queued.id}`;
      let current = queued;
      while (current.status === "queued" || current.status === "running") {
        await wait(1200);
        if (run !== runRef.current) return;
        current = await fetchAnalysis(statusUrl);
        setJob(current);
      }
      if (current.status === "failed") throw new Error(current.error || "The road analysis failed.");
      const resultUrl = current.result_url ?? queued.result_url;
      if (!resultUrl) throw new Error("The completed analysis did not provide a result URL.");
      const value = await fetchAnalysisResult(resultUrl);
      if (run !== runRef.current) return;
      setResult(value);
      setResultGraph(await fetchGraph(value.graph_url).catch(() => null));
    } catch (reason) {
      if (run === runRef.current) {
        setError(reason instanceof Error ? reason.message : "The image could not be analyzed.");
      }
    } finally {
      if (run === runRef.current) setBusy(false);
    }
  }

  async function failJunction(row: CriticalNode) {
    if (!result) return;
    setFailed(row);
    setStress(null);
    setStressError(null);
    try {
      const value = await runAnalysisSimulation(result.id, [row.node_id]);
      setStress(value);
    } catch (reason) {
      setStressError(reason instanceof Error ? reason.message : "The stress test could not be completed.");
    }
  }

  function reset() {
    runRef.current += 1;
    setFailed(null);
    setStress(null);
    setStressError(null);
    rememberFile(null);
    setJob(null);
    setResult(null);
    setResultGraph(null);
    setBusy(false);
    setError(null);
    setConsent(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function clickBackdrop(event: PointerEvent<HTMLDialogElement>) {
    if (event.target !== event.currentTarget) return;
    const box = event.currentTarget.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) {
      onClose();
    }
  }

  return (
    <dialog
      className="upload-dialog"
      ref={dialogRef}
      aria-labelledby="upload-title"
      onCancel={(event) => { event.preventDefault(); onClose(); }}
      onPointerDown={clickBackdrop}
    >
      <div className="dialog-grid">
        <header className="dialog-heading">
          <div><span className="eyebrow">Live extraction · queued safely</span><h2 id="upload-title">Bring your own terrain.</h2></div>
          <button className="dialog-close" type="button" aria-label="Close imagery analysis" onClick={onClose}>×</button>
        </header>

        {result ? (
          <section className="upload-result" aria-live="polite">
            <div className="result-stamp"><span>Analysis</span><strong>Complete</strong><small>Image-space network · no fake georeference</small></div>
            {preview ? (
              <UploadOverlay
                image={preview}
                graph={resultGraph}
                resolution={result.resolution_m}
                size={imageSize}
                onImageSize={setImageSize}
                failed={failed}
              />
            ) : null}
            <div className="result-score"><span>Worst-junction resilience</span><strong>{formatMetric(result.resilience_index, { digits: 3 })}</strong><i style={{ "--result": result.resilience_index } as CSSProperties} /></div>
            <dl>
              <div><dt>Junctions</dt><dd>{result.n_nodes}</dd></div>
              <div><dt>Road links</dt><dd>{result.n_edges}</dd></div>
              <div><dt>Critical</dt><dd>{result.summary.critical_junctions ?? "—"}</dd></div>
              <div><dt>Single points</dt><dd>{result.summary.articulation_points ?? "—"}</dd></div>
              <div><dt>Worst junction</dt><dd>{result.top_node == null ? "—" : `J-${result.top_node}`}</dd></div>
              <div><dt>Scale estimate</dt><dd>{result.resolution_m} m/px</dd></div>
            </dl>
            {result.criticality.length ? (
              <section className="upload-stress" aria-labelledby="upload-stress-title">
                <h3 id="upload-stress-title">Stress-test this network</h3>
                <p>Fail one of its most critical junctions to see what the network loses.</p>
                <div className="upload-stress-options">
                  {[...result.criticality].sort((a, b) => a.rank - b.rank).slice(0, 5).map((row) => (
                    <button key={row.node_id} type="button" aria-pressed={failed?.node_id === row.node_id} onClick={() => void failJunction(row)}>
                      J-{row.node_id}
                    </button>
                  ))}
                </div>
                <p role="status" aria-live="polite">
                  {stressError ?? (failed && !stress ? `Failing J-${failed.node_id}…` : stress && failed
                    ? `Failing J-${failed.node_id} costs ${formatMetric(stress.efficiency_loss * 100, { digits: 1 })}% of network efficiency (RI ${formatMetric(stress.resilience_index, { digits: 3 })}); ${formatMetric((stress.active_largest_cc_fraction ?? stress.largest_cc_fraction) * 100, { digits: 0 })}% of the remaining junctions stay connected.`
                    : "")}
                </p>
              </section>
            ) : null}
            <div className="result-actions">
              <a href={result.exports.geojson} download>Download GeoJSON</a>
              <a href={result.exports.json} download>Download evidence JSON</a>
              <button type="button" onClick={reset}>Analyze another image</button>
            </div>
          </section>
        ) : (
          <form className="upload-form" onSubmit={analyze}>
            <label className={`upload-drop ${preview ? "has-preview" : ""}`}>
              {preview ? <img src={preview} alt="Selected satellite image preview" /> : <span className="upload-cross" aria-hidden="true">＋</span>}
              <span><strong>{file?.name ?? "Choose a satellite image"}</strong><small>PNG or JPEG · up to 11 MiB and 4096² px</small></span>
              <input ref={fileInputRef} type="file" accept="image/png,image/jpeg" disabled={busy} onChange={(event) => void chooseFile(event)} />
            </label>

            <div className="resolution-control">
              <label htmlFor="resolution">Estimated ground resolution</label>
              <output htmlFor="resolution">{resolution.toFixed(1)} m / pixel</output>
              <input id="resolution" type="range" min="0.1" max="2" step="0.1" value={resolution} disabled={busy} onChange={(event) => setResolution(Number(event.target.value))} />
              <p>Use 0.5 m/px for a typical neighbourhood-scale satellite capture. This scale affects distance estimates.</p>
            </div>

            <label className="consent-row">
              <input type="checkbox" checked={consent} disabled={busy} onChange={(event) => setConsent(event.target.checked)} />
              <span>I confirm I may process this image and consent to encrypted transfer to the project’s authenticated Modal GPU endpoint.</span>
            </label>

            <div className="upload-submit-row">
              <p role="status" aria-live="polite"><span className={busy ? "status-pulse" : ""} />{statusCopy({ job, busy, hasFile: file != null, consent, elapsed })}</p>
              <button className="button-primary" type="submit" disabled={busy}>{busy ? "Analyzing…" : "Extract road network"}</button>
            </div>
            {error ? <div className="form-error" role="alert">{error}</div> : null}
          </form>
        )}

        <footer className="dialog-footnote"><span>Privacy boundary</span><p>The public server retains only short-lived job artifacts. Upload coordinates remain in honest image space.</p></footer>
      </div>
    </dialog>
  );
}

function UploadOverlay({
  image,
  graph,
  resolution,
  size,
  onImageSize,
  failed,
}: {
  image: string;
  graph: GeoJsonCollection | null;
  resolution: number;
  size: [number, number] | null;
  onImageSize: (size: [number, number]) => void;
  failed: CriticalNode | null;
}) {
  const path = useMemo(() => {
    if (!graph) return "";
    return graph.features.flatMap((feature) => {
      if (feature.properties.feature_type !== "edge" || !Array.isArray(feature.geometry.coordinates)) return [];
      const points = feature.geometry.coordinates as unknown[];
      const valid = points.filter((point): point is [number, number] => (
        Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1])
      ));
      if (valid.length < 2) return [];
      return [`M${valid.map(([x, y]) => `${x / resolution},${y / resolution}`).join("L")}`];
    }).join("");
  }, [graph, resolution]);

  const critical = useMemo(() => graph?.features.filter((feature) => (
    feature.properties.feature_type === "node" && feature.properties.is_critical === true
  )) ?? [], [graph]);

  return (
    <figure className="upload-overlay">
      <img src={image} alt="Uploaded satellite image with extracted road network" onLoad={(event) => {
        if (!size) onImageSize([event.currentTarget.naturalWidth, event.currentTarget.naturalHeight]);
      }} />
      {size && graph ? (
        <svg viewBox={`0 0 ${size[0]} ${size[1]}`} preserveAspectRatio="none" aria-hidden="true">
          <path d={path} />
          {critical.map((feature, index) => {
            const coordinates = feature.geometry.coordinates as [number, number];
            return <circle key={String(feature.properties.node_id ?? index)} cx={coordinates[0] / resolution} cy={coordinates[1] / resolution} r="5" />;
          })}
          {failed ? <circle className="failed" cx={failed.x / resolution} cy={failed.y / resolution} r="11" /> : null}
        </svg>
      ) : null}
      <figcaption>{graph ? "Extracted network overlay" : "Network overlay unavailable · downloads remain valid"}</figcaption>
    </figure>
  );
}
