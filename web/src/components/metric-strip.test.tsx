import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MetricStrip } from "@/components/metric-strip";

describe("MetricStrip", () => {
  it("shows a placeholder, not an error, for the active component at baseline", () => {
    render(<MetricStrip nodeCount={573} edgeCount={828} criticalCount={57} simulation={null} />);
    expect(screen.getByText("Baseline intact")).toBeInTheDocument();
    expect(screen.getByLabelText("No failures simulated")).toHaveTextContent("—");
    expect(screen.queryByText("Not available")).not.toBeInTheDocument();
  });
});

describe("MetricStrip stale result", () => {
  it("does not present an out-of-date result as current", () => {
    const simulation = { aoi: "a", removed_node_ids: [7], resilience_index: .985, efficiency_loss: .015, largest_cc_fraction: .99 };
    render(<MetricStrip nodeCount={1} edgeCount={1} criticalCount={1} simulation={simulation} stale />);
    expect(screen.getByText("Out of date: run again")).toBeInTheDocument();
    expect(screen.queryByText(/efficiency lost/)).not.toBeInTheDocument();
  });
});
