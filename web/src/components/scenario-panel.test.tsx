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
      simulation={null}
      stale={false}
      onToggleRemoved={vi.fn()}
      onRun={vi.fn()}
      onReset={vi.fn()}
    />);

    const items = screen.getAllByRole("listitem");
    expect(within(items[0]).getByText("J-7")).toBeInTheDocument();
    expect(within(items[1]).getByText("J-3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Restore J-7" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Restore J-3" })).toBeInTheDocument();
  });

  it("counts a single offline junction in the singular", () => {
    render(<ScenarioPanel
      mode="recover"
      selected={null}
      criticalNodes={criticalNodes}
      removed={[7]}
      busy={false}
      simulation={null}
      stale={false}
      onToggleRemoved={vi.fn()}
      onRun={vi.fn()}
      onReset={vi.fn()}
    />);

    expect(screen.getByRole("heading", { name: "1 junction offline" })).toBeInTheDocument();
  });

  it("does not offer to add a junction that is already in the failure set", () => {
    render(<ScenarioPanel
      mode="stress"
      selected={criticalNodes[0]}
      criticalNodes={criticalNodes}
      removed={[7]}
      busy={false}
      simulation={null}
      stale={false}
      onToggleRemoved={vi.fn()}
      onRun={vi.fn()}
      onReset={vi.fn()}
    />);

    expect(screen.getByText("J-7 is in the failure set.")).toBeInTheDocument();
    expect(screen.queryByText(/ready to add/)).not.toBeInTheDocument();
  });
});

describe("ScenarioPanel stress result", () => {
  const simulation = {
    aoi: "panaji_demo", removed_node_ids: [7], resilience_index: .985, efficiency_loss: .015,
    largest_cc_fraction: .99, active_largest_cc_fraction: .99,
  };
  const props = {
    mode: "stress" as const, selected: criticalNodes[0], criticalNodes, removed: [7], busy: false,
    onToggleRemoved: vi.fn(), onRun: vi.fn(), onReset: vi.fn(),
  };

  it("shows the run's loss next to the controls", () => {
    render(<ScenarioPanel {...props} simulation={simulation} stale={false} />);
    expect(screen.getByText("1.5%")).toBeInTheDocument();
    expect(screen.getByText("0.985")).toBeInTheDocument();
    expect(screen.queryByText(/failure set changed/i)).not.toBeInTheDocument();
  });

  it("flags a result that no longer matches the failure set", () => {
    render(<ScenarioPanel {...props} removed={[3, 7]} simulation={simulation} stale />);
    expect(screen.getByText(/failure set changed/i)).toBeInTheDocument();
  });
});
