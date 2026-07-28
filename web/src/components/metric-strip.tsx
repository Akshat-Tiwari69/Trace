import { formatMetric } from "@/lib/model";
import type { SimulationResult } from "@/lib/api";

type Props = {
  nodeCount: number;
  edgeCount: number;
  criticalCount: number;
  simulation: SimulationResult | null;
};

export function MetricStrip({ nodeCount, edgeCount, criticalCount, simulation }: Props) {
  const ri = simulation?.resilience_index ?? 1;
  const loss = simulation?.efficiency_loss ?? 0;
  return (
    <section className="metric-strip" aria-label="Network status">
      <div className="metric-primary" aria-live="polite">
        <span className="metric-kicker">Resilience index</span>
        <strong>{formatMetric(ri, { digits: 3 })}</strong>
        <span className="metric-delta" data-loss={loss > 0 || undefined}>
          {loss > 0 ? `${formatMetric(loss * 100, { digits: 1 })}% efficiency lost` : "Baseline intact"}
        </span>
      </div>
      <div><span>Junctions</span><strong>{nodeCount}</strong></div>
      <div><span>Road links</span><strong>{edgeCount}</strong></div>
      <div><span>Critical</span><strong>{criticalCount}</strong></div>
      <div><span>Active component</span><strong>{formatMetric(simulation?.active_largest_cc_fraction ?? simulation?.largest_cc_fraction, { digits: 3 })}</strong></div>
    </section>
  );
}
