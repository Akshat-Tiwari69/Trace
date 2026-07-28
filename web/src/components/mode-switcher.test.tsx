import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ModeSwitcher } from "@/components/mode-switcher";

describe("ModeSwitcher", () => {
  it("exposes the four workspace modes as keyboard-friendly tabs", () => {
    const onChange = vi.fn();
    render(<ModeSwitcher mode="explore" onChange={onChange} />);

    expect(screen.getByRole("tablist", { name: "Workspace mode" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /explore/i })).toHaveAttribute("aria-selected", "true");
    fireEvent.click(screen.getByRole("tab", { name: /stress/i }));
    expect(onChange).toHaveBeenCalledWith("stress");

    const explore = screen.getByRole("tab", { name: /explore/i });
    explore.focus();
    fireEvent.keyDown(explore, { key: "ArrowRight" });
    expect(onChange).toHaveBeenLastCalledWith("stress");
    expect(screen.getByRole("tab", { name: /stress/i })).toHaveFocus();
  });
});
