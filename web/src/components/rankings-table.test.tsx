import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RankingsTable } from "@/components/rankings-table";

const rows = [
  { node_id: 278, rank: 1, betweenness: 0.330746, is_critical: true, is_articulation: false, x: 73.82, y: 15.49 },
  { node_id: 261, rank: 2, betweenness: 0.264219, is_critical: true, is_articulation: false, x: 73.83, y: 15.50 },
];

describe("RankingsTable", () => {
  it("keeps the map ranking available as a semantic selection list", () => {
    const onSelect = vi.fn();
    render(<RankingsTable rows={rows} selectedNode={278} onSelect={onSelect} />);

    expect(screen.getByRole("table", { name: "Critical junction ranking" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /junction 278/i })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: /junction 261/i }));
    expect(onSelect).toHaveBeenCalledWith(261);
  });
});
