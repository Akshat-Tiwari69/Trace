import { describe, expect, it } from "vitest";
import type { StyleSpecification } from "maplibre-gl";

import { nullSafeBasemapStyle } from "@/components/network-map";

describe("OpenFreeMap style", () => {
  it("rejects road shields without a numeric ref_length before comparing it", () => {
    const style = {
      version: 8,
      sources: {},
      layers: [
        "highway-shield-non-us",
        "highway-shield-us-interstate",
        "road_shield_us",
      ].map((id) => ({
        id,
        type: "symbol" as const,
        source: "openmaptiles",
        filter: ["all", ["<=", ["get", "ref_length"], 6]],
      })),
    } as StyleSpecification;

    const transformed = nullSafeBasemapStyle(undefined, style);

    expect(transformed.layers.map((layer) => "filter" in layer ? layer.filter : null)).toEqual([
      ["all", ["==", ["typeof", ["get", "ref_length"]], "number"], ["<=", ["get", "ref_length"], 6]],
      ["all", ["==", ["typeof", ["get", "ref_length"]], "number"], ["<=", ["get", "ref_length"], 6]],
      ["all", ["==", ["typeof", ["get", "ref_length"]], "number"], ["<=", ["get", "ref_length"], 6]],
    ]);
  });
});
