import type { AtlasProvenance } from "@/lib/api";

export type MetricFormat = {
  digits?: number;
  prefix?: string;
  suffix?: string;
};

export function clampResilience(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export function scenarioKey(nodeIds: readonly number[]): string {
  const ids = [...new Set(nodeIds.filter(Number.isInteger))].sort((a, b) => a - b);
  return ids.length ? ids.join(".") : "baseline";
}

// URL lists like `failed=86,278`. Only whole digit runs count: Number("") is 0,
// so a naive split would turn an empty value into junction 0.
export function parseNodeIds(value: string | null): number[] {
  const ids = (value ?? "").split(",").filter((part) => /^\d+$/.test(part)).map(Number).filter(Number.isSafeInteger);
  return [...new Set(ids)].sort((a, b) => a - b);
}

export function sourceLabel(atlas: AtlasProvenance): string {
  if (atlas.source === "imagery") return `Extracted from imagery · ${atlas.model?.release ?? "unknown model"}`;
  return "OpenStreetMap roads";
}

// Only imagery atlases have a model; OSM networks are not model output.
export function trainingLabel(atlas: AtlasProvenance): string | null {
  if (atlas.source !== "imagery" || atlas.seen_in_training == null) return null;
  return atlas.seen_in_training ? "Area inside the training corpus" : "Area unseen by the model";
}

const cityName = (label: string) => label.split(", ").at(-1) ?? label;

// Imagery first: it is the product's own extraction; OSM is the reference.
export function groupAtlases<T extends AtlasProvenance & { aoi: string; label: string }>(rows: readonly T[]): T[][] {
  const groups = new Map<string, T[]>();
  for (const row of rows) groups.set(row.area ?? row.aoi, [...(groups.get(row.area ?? row.aoi) ?? []), row]);
  return [...groups.values()]
    .map((group) => [...group].sort((a, b) => (a.source === "imagery" ? 0 : 1) - (b.source === "imagery" ? 0 : 1)))
    // Labels read "Neighbourhood, City": order by city.
    .sort((a, b) => cityName(a[0].label).localeCompare(cityName(b[0].label)));
}

export function formatCoordinates([west, south, east, north]: readonly number[]): string {
  const lat = (south + north) / 2;
  const lon = (west + east) / 2;
  return `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"} / ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}`;
}

// ?city= opens that atlas; anything else lands on the picker.
export function initialAoi(search: string): string | null {
  const city = new URLSearchParams(search).get("city");
  return city && /^[a-z0-9][a-z0-9_-]{0,63}$/.test(city) ? city : null;
}

export function formatMetric(value: number | null | undefined, format: MetricFormat = {}): string {
  if (value == null || !Number.isFinite(value)) return "Not available";
  const { digits = 2, prefix = "", suffix = "" } = format;
  return `${prefix}${value.toFixed(digits)}${suffix}`;
}

type ScenarioFeature = {
  type: "Feature";
  geometry: { type: string; coordinates: unknown };
  properties: Record<string, unknown>;
};

type ScenarioCollection = {
  type: "FeatureCollection";
  features: ScenarioFeature[];
  scenario: {
    removed_node_ids: number[];
    resilience_index: number | null;
    efficiency_loss: number | null;
  };
  [key: string]: unknown;
};

export function scenarioGeoJson(
  graph: { type: "FeatureCollection"; features: ScenarioFeature[]; [key: string]: unknown },
  removedNodeIds: readonly number[],
  simulation?: { resilience_index: number; efficiency_loss: number } | null,
): ScenarioCollection {
  const removed = new Set(removedNodeIds);
  const features = graph.features.map((feature) => {
    const { properties } = feature;
    const failed = properties.feature_type === "node"
      ? removed.has(Number(properties.node_id))
      : properties.feature_type === "edge"
        ? removed.has(Number(properties.u)) || removed.has(Number(properties.v))
        : false;
    return { ...feature, properties: { ...properties, scenario_status: failed ? "failed" : "active" } };
  });
  return {
    ...graph,
    type: "FeatureCollection",
    features,
    scenario: {
      removed_node_ids: [...removed].sort((a, b) => a - b),
      resilience_index: simulation?.resilience_index ?? null,
      efficiency_loss: simulation?.efficiency_loss ?? null,
    },
  };
}

function csvCell(value: unknown): string {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function criticalityCsv(
  rows: ReadonlyArray<{
    rank: number;
    node_id: number;
    betweenness: number;
    is_critical: boolean;
    is_articulation: boolean;
  }>,
  removedNodeIds: readonly number[],
): string {
  const removed = new Set(removedNodeIds);
  const header = ["rank", "node_id", "betweenness", "is_critical", "is_articulation", "scenario_status"];
  const body = rows.map((row) => [
    row.rank,
    row.node_id,
    row.betweenness,
    row.is_critical,
    row.is_articulation,
    removed.has(row.node_id) ? "failed" : "active",
  ].map(csvCell).join(","));
  return [header.join(","), ...body].join("\r\n") + "\r\n";
}
