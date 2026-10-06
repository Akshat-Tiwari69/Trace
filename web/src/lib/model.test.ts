import { describe, expect, it } from "vitest";

import { clampResilience, criticalityCsv, formatCoordinates, formatMetric, groupAtlases, initialAoi, parseNodeIds, scenarioGeoJson, scenarioKey, sourceLabel, trainingLabel } from "@/lib/model";

describe("workspace model helpers", () => {
  it("never turns an empty or malformed URL list into junction 0", () => {
    expect(parseNodeIds(null)).toEqual([]);
    expect(parseNodeIds("")).toEqual([]);
    expect(parseNodeIds("278,,x,-3,1.5")).toEqual([278]);
    expect(parseNodeIds("278,0,278")).toEqual([0, 278]);
    expect(parseNodeIds(`278,9007199254740993,${"9".repeat(400)}`)).toEqual([278]);
  });

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

describe("atlas helpers", () => {
  const osm = { aoi: "delhi_cp_osm", area: "delhi_cp", label: "Connaught Place, Delhi", source: "osm" as const, model: null, seen_in_training: null };
  const imagery = { ...osm, aoi: "delhi_cp_imagery", source: "imagery" as const, model: { release: "a4-roadseg-v4", checkpoint: "road_v4.pt", sha256: "x" }, seen_in_training: false };

  it("names the source and only claims training status for model output", () => {
    expect(sourceLabel(osm)).toBe("OpenStreetMap roads");
    expect(sourceLabel(imagery)).toBe("Extracted from imagery · a4-roadseg-v4");
    expect(trainingLabel(osm)).toBeNull();
    expect(trainingLabel(imagery)).toBe("Area unseen by the model");
    expect(trainingLabel({ ...imagery, seen_in_training: true })).toBe("Area inside the training corpus");
  });

  it("groups an area's sources with imagery first, areas by name", () => {
    const pune = { ...osm, aoi: "pune_osm", area: "pune", label: "Shivajinagar, Pune" };
    const groups = groupAtlases([pune, osm, imagery]);
    expect(groups.map((group) => group.map((row) => row.aoi))).toEqual([["delhi_cp_imagery", "delhi_cp_osm"], ["pune_osm"]]);
  });

  it("opens Panaji for links that predate the picker", () => {
    expect(initialAoi("")).toBeNull();
    expect(initialAoi("?city=pune_shivajinagar_osm&mode=stress")).toBe("pune_shivajinagar_osm");
    expect(initialAoi("?mode=stress&failed=278")).toBe("panaji_demo");
    expect(initialAoi("?city=../etc")).toBeNull();
  });

  it("formats the area centre", () => {
    expect(formatCoordinates([73.823, 15.488, 73.842, 15.501])).toBe("15.49° N / 73.83° E");
  });
});
