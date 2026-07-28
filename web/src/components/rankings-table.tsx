"use client";

import type { CSSProperties } from "react";

import type { CriticalNode } from "@/lib/types";

type Props = {
  rows: CriticalNode[];
  selectedNode: number | null;
  onSelect: (nodeId: number) => void;
};

export function RankingsTable({ rows, selectedNode, onSelect }: Props) {
  const maxScore = Math.max(...rows.map((row) => row.betweenness), 1);
  return (
    <div className="ranking-scroll">
      <table className="ranking-table" aria-label="Critical junction ranking">
        <thead>
          <tr><th>Rank</th><th>Junction</th><th>Criticality</th></tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.node_id} data-selected={selectedNode === row.node_id || undefined}>
              <td><span className="rank-number">{String(row.rank).padStart(2, "0")}</span></td>
              <td>
                <button
                  type="button"
                  className="ranking-pick"
                  aria-label={`Junction ${row.node_id}, rank ${row.rank}`}
                  aria-pressed={selectedNode === row.node_id}
                  onClick={() => onSelect(row.node_id)}
                >
                  J-{row.node_id}
                  {row.is_articulation ? <span className="spof-mark">SPOF<span className="sr-only">, single point of failure</span></span> : null}
                </button>
              </td>
              <td>
                <span
                  className="score-meter"
                  style={{ "--score": row.betweenness / maxScore } as CSSProperties}
                >
                  <span>{row.betweenness.toFixed(3)}</span>
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
