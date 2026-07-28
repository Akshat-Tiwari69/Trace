"use client";

import { useRef, type KeyboardEvent } from "react";

import type { WorkspaceMode } from "@/lib/types";

const MODES: Array<{ id: WorkspaceMode; label: string; cue: string }> = [
  { id: "explore", label: "Explore", cue: "Read the network" },
  { id: "stress", label: "Stress", cue: "Test a failure" },
  { id: "compare", label: "Compare", cue: "See before and after" },
  { id: "recover", label: "Recover", cue: "Plan restoration" },
];

type Props = {
  mode: WorkspaceMode;
  onChange: (mode: WorkspaceMode) => void;
};

export function ModeSwitcher({ mode, onChange }: Props) {
  const tabs = useRef<Array<HTMLButtonElement | null>>([]);

  function move(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? MODES.length - 1
        : (index + delta + MODES.length) % MODES.length;
    onChange(MODES[next].id);
    tabs.current[next]?.focus();
  }

  return (
    <div className="mode-switcher" role="tablist" aria-label="Workspace mode">
      {MODES.map((item, index) => (
        <button
          key={item.id}
          className="mode-tab"
          id={`mode-tab-${item.id}`}
          ref={(node) => { tabs.current[index] = node; }}
          type="button"
          role="tab"
          aria-selected={mode === item.id}
          aria-controls="workspace-panel"
          tabIndex={mode === item.id ? 0 : -1}
          onClick={() => onChange(item.id)}
          onKeyDown={(event) => move(event, index)}
        >
          <span className="mode-index" aria-hidden="true">0{index + 1}</span>
          <span>{item.label}</span>
          <span className="sr-only"> — {item.cue}</span>
        </button>
      ))}
    </div>
  );
}
