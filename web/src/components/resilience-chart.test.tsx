import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResilienceChart } from "@/components/resilience-chart";

const curve = [
  { n_removed: 0, targeted_resilience_index: 1, random_resilience_index: 1 },
  { n_removed: 1, targeted_resilience_index: 0.965, random_resilience_index: 0.99 },
];

describe("ResilienceChart", () => {
  it("pairs the visual curve with an accessible data table", () => {
    render(<ResilienceChart rows={curve} activeStep={1} />);
    expect(screen.getByRole("img", { name: /resilience under progressive junction loss/i })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Resilience curve data" })).toBeInTheDocument();
    expect(screen.getByText("0.965")).toBeInTheDocument();
  });
});
