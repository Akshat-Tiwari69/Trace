import { describe, expect, it } from "vitest";

import { clampResilience, criticalityCsv, formatMetric, scenarioGeoJson, scenarioKey } from "@/lib/model";

describe("workspace model helpers", () => {
  it("produces one stable key for the same set of failed nodes", () => {
    expect(scenarioKey([278, 86, 261])).toBe(scenarioKey([261, 278, 86]));
  });

  it("bounds resilience against invalid presentation values", () => {
    expect(clampResilience(-0.2)).toBe(0);
    expect(clampResilience(1.4)).toBe(1);
    expect(clampResilience(Number.NaN)).toBe(0);
  });

  it("formats missing metrics without pretending they are zero", () => {
    expect(formatMetric(null, { digits: 2 })).toBe("Not available");
    expect(formatMetric(0.965066, { digits: 3 })).toBe("0.965");
  });

  it("exports failed nodes and incident links without mutating the source graph", () => {
    const graph = {
      type: "FeatureCollection" as const,
      features: [
        { type: "Feature" as const, geometry: { type: "Point", coordinates: [0, 0] }, properties: { feature_type: "node", node_id: 7 } },
        { type: "Feature" as const, geometry: { type: "LineString", coordinates: [[0, 0], [1, 1]] }, properties: { feature_type: "edge", u: 7, v: 8 } },
      ],
    };

    const exported = scenarioGeoJson(graph, [7], { resilience_index: 0.8, efficiency_loss: 0.2 });
    expect(exported.features.map((feature) => feature.properties.scenario_status)).toEqual(["failed", "failed"]);
    expect(graph.features[0].properties).not.toHaveProperty("scenario_status");
    expect(exported.scenario.resilience_index).toBe(0.8);
  });

  it("creates a deterministic criticality CSV with scenario state", () => {
    const csv = criticalityCsv([
      { rank: 1, node_id: 7, betweenness: 0.3, is_critical: true, is_articulation: false },
    ], [7]);
    expect(csv).toContain("rank,node_id,betweenness,is_critical,is_articulation,scenario_status\r\n");
    expect(csv).toContain("1,7,0.3,true,false,failed\r\n");
  });
});
