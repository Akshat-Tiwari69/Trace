import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ScenarioPanel } from "@/components/scenario-panel";

const criticalNodes = [
  { node_id: 7, rank: 1, betweenness: .3, is_critical: true, is_articulation: false, x: 0, y: 0 },
  { node_id: 3, rank: 2, betweenness: .2, is_critical: true, is_articulation: false, x: 0, y: 0 },
];

describe("ScenarioPanel", () => {
  it("orders recovery work by measured impact instead of node number", () => {
    render(<ScenarioPanel
      mode="recover"
      selected={null}
      criticalNodes={criticalNodes}
      removed={[3, 7]}
      busy={false}
      onToggleRemoved={vi.fn()}
      onRun={vi.fn()}
      onReset={vi.fn()}
    />);

    const items = screen.getAllByRole("listitem");
    expect(within(items[0]).getByText("J-7")).toBeInTheDocument();
    expect(within(items[1]).getByText("J-3")).toBeInTheDocument();
  });
});
