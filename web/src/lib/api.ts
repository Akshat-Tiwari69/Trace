import type { CriticalNode, ResiliencePoint } from "@/lib/types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");
const AOI_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export type AoiSummary = {
  aoi: string;
  label: string;
  coordinate_system: string;
  bounds: [number, number, number, number];
  node_count: number;
  edge_count: number;
  critical_count: number;
  baseline_efficiency: number;
  graph_url: string;
  critical_nodes: CriticalNode[];
  resilience_curve: ResiliencePoint[];
  evidence: Record<string, unknown>;
  [key: string]: unknown;
};

export type SimulationResult = {
  aoi: string;
  removed_node_ids: number[];
  resilience_index: number;
  efficiency_loss: number;
  largest_cc_fraction: number;
  active_largest_cc_fraction?: number;
  efficiency_method?: "exact" | "sampled";
  efficiency_sample_size?: number | null;
  efficiency_seed?: number | null;
  representative_route?: {
    origin: number;
    destination: number;
    baseline_length_m: number;
    rerouted_length_m: number | null;
    travel_time_delta_pct: number | null;
    disconnected: boolean;
    disconnected_pairs?: number;
    [key: string]: unknown;
  } | null;
  route?: unknown;
  [key: string]: unknown;
};

export type AnalysisStatus = "queued" | "running" | "done" | "failed";

export type AnalysisJob = {
  id: string;
  job_id: string;
  status: AnalysisStatus;
  position: number;
  status_url?: string;
  result_url?: string;
  created_utc?: string;
  started_utc?: string | null;
  finished_utc?: string | null;
  error?: string | null;
};

export type AnalysisResult = {
  id: string;
  status: "done";
  n_nodes: number;
  n_edges: number;
  resolution_m: number;
  resilience_index: number;
  top_node: number | null;
  summary: {
    critical_junctions?: number;
    articulation_points?: number;
    worst_junction?: number | null;
    [key: string]: unknown;
  };
  criticality: CriticalNode[];
  graph_url: string;
  exports: { json: string; geojson: string };
};

export type GeoJsonGeometry = {
  type: string;
  coordinates: unknown;
};

export type GeoJsonFeature = {
  type: "Feature";
  geometry: GeoJsonGeometry;
  properties: Record<string, unknown>;
};

export type GeoJsonCollection = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
  [key: string]: unknown;
};

function checkedAoi(aoi: string): string {
  if (!AOI_PATTERN.test(aoi)) throw new ApiError("Invalid area identifier", 400);
  return aoi;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { accept: "application/json", ...init?.headers },
    signal: init?.signal ?? AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new ApiError(body?.detail ?? "The request could not be completed", response.status);
  }
  return response.json() as Promise<T>;
}

function checkedAnalysisUrl(path: string): string {
  if (!path.startsWith("/api/v1/analyses/")) {
    throw new ApiError("Invalid analysis URL", 400);
  }
  return path;
}

export function fetchAoi(aoi: string): Promise<AoiSummary> {
  return fetchJson(`/api/v1/aois/${checkedAoi(aoi)}`);
}

export function fetchGraph(url: string): Promise<GeoJsonCollection> {
  if (!url.startsWith("/api/v1/aois/") && !url.startsWith("/api/v1/analyses/")) {
    throw new ApiError("Invalid graph URL", 400);
  }
  return fetchJson(url);
}

export function runSimulation(aoi: string, removedNodeIds: readonly number[]): Promise<SimulationResult> {
  const removed_node_ids = [...new Set(removedNodeIds.filter(Number.isInteger))].sort((a, b) => a - b);
  return fetchJson("/api/v1/simulations", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ aoi: checkedAoi(aoi), removed_node_ids }),
  });
}

export function submitAnalysis(
  image: File,
  resolutionM: number,
  signal?: AbortSignal,
): Promise<AnalysisJob> {
  const body = new FormData();
  body.set("image", image);
  body.set("resolution_m", String(resolutionM));
  body.set("confirm_external_processing", "true");
  return fetchJson("/api/v1/analyses", { method: "POST", body, signal });
}

export function fetchAnalysis(path: string, signal?: AbortSignal): Promise<AnalysisJob> {
  return fetchJson(checkedAnalysisUrl(path), { cache: "no-store", signal });
}

export function fetchAnalysisResult(path: string, signal?: AbortSignal): Promise<AnalysisResult> {
  return fetchJson(checkedAnalysisUrl(path), { cache: "no-store", signal });
}
