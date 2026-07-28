import type { ResiliencePoint } from "@/lib/types";

type Props = {
  rows: ResiliencePoint[];
  activeStep?: number;
};

const WIDTH = 420;
const HEIGHT = 112;
const PAD = 10;

function linePath(rows: ResiliencePoint[], field: "targeted_resilience_index" | "random_resilience_index") {
  const maxStep = Math.max(rows.at(-1)?.n_removed ?? 1, 1);
  return rows.map((row, index) => {
    const x = PAD + (row.n_removed / maxStep) * (WIDTH - PAD * 2);
    const y = PAD + (1 - row[field]) * (HEIGHT - PAD * 2);
    return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

export function ResilienceChart({ rows, activeStep = 0 }: Props) {
  const maxStep = Math.max(rows.at(-1)?.n_removed ?? 1, 1);
  const markerX = PAD + (Math.min(activeStep, maxStep) / maxStep) * (WIDTH - PAD * 2);
  return (
    <div className="resilience-chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Resilience under progressive junction loss"
        preserveAspectRatio="none"
      >
        <defs>
          <pattern id="targeted-hatch" width="6" height="6" patternUnits="userSpaceOnUse">
            <path d="M0 6L6 0" stroke="currentColor" strokeWidth="1" opacity=".18" />
          </pattern>
        </defs>
        <line className="chart-grid" x1={PAD} y1={PAD} x2={WIDTH - PAD} y2={PAD} />
        <line className="chart-grid" x1={PAD} y1={HEIGHT / 2} x2={WIDTH - PAD} y2={HEIGHT / 2} />
        <line className="chart-grid" x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} />
        <path className="curve-random" d={linePath(rows, "random_resilience_index")} />
        <path className="curve-targeted" d={linePath(rows, "targeted_resilience_index")} />
        <line className="chart-marker" x1={markerX} x2={markerX} y1={PAD} y2={HEIGHT - PAD} />
      </svg>
      <div className="chart-key" aria-hidden="true">
        <span><i className="key-targeted" /> Targeted</span>
        <span><i className="key-random" /> Random reference</span>
      </div>
      <table className="sr-only" aria-label="Resilience curve data">
        <caption>Resilience curve data</caption>
        <thead><tr><th>Removed</th><th>Targeted</th><th>Random</th></tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.n_removed}>
              <td>{row.n_removed}</td>
              <td>{row.targeted_resilience_index.toFixed(3)}</td>
              <td>{row.random_resilience_index.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
