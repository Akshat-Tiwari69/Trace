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
