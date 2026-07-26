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
