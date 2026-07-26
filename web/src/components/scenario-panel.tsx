import type { CriticalNode, WorkspaceMode } from "@/lib/types";

type Props = {
  mode: WorkspaceMode;
  selected: CriticalNode | null;
  criticalNodes: CriticalNode[];
  removed: number[];
  busy: boolean;
  onToggleRemoved: (nodeId: number) => void;
  onRun: () => void;
  onReset: () => void;
};

export function ScenarioPanel({ mode, selected, criticalNodes, removed, busy, onToggleRemoved, onRun, onReset }: Props) {
  if (mode === "explore") {
    return (
      <section className="context-card">
        <span className="eyebrow">Field note</span>
        <h2>{selected ? `Junction J-${selected.node_id}` : "Select a junction"}</h2>
        <p>
          {selected
            ? `Ranked ${selected.rank} of the network by routing criticality. Inspect it on the map or move to Stress to test its loss.`
            : "Choose any marked junction to inspect the role it plays in the city network."}
        </p>
      </section>
    );
  }

  if (mode === "recover") {
    const rank = new Map(criticalNodes.map((node) => [node.node_id, node.rank]));
    const recoveryOrder = [...removed].sort((a, b) => (rank.get(a) ?? Number.MAX_SAFE_INTEGER) - (rank.get(b) ?? Number.MAX_SAFE_INTEGER));
    return (
      <section className="context-card">
        <span className="eyebrow">Recovery queue</span>
        <h2>{removed.length ? `${removed.length} junctions offline` : "Network restored"}</h2>
        {removed.length ? (
          <ol className="recovery-list">
            {recoveryOrder.map((nodeId, index) => (
              <li key={nodeId}>
                <span>Priority {index + 1}</span><strong>J-{nodeId}</strong>
                <button type="button" onClick={() => onToggleRemoved(nodeId)}>Restore</button>
              </li>
            ))}
          </ol>
        ) : <p>No failed junctions remain in this scenario.</p>}
      </section>
    );
  }

  return (
    <section className="context-card">
      <span className="eyebrow">Scenario builder</span>
      <h2>{mode === "compare" ? "Baseline / disruption" : "Junction failure"}</h2>
      <p>{selected ? `J-${selected.node_id} is ready to add to the failure set.` : "Select a map junction or ranking row first."}</p>
      <div className="scenario-actions">
        <button
          type="button"
          className="button-secondary"
          disabled={!selected}
          onClick={() => selected && onToggleRemoved(selected.node_id)}
        >
          {selected && removed.includes(selected.node_id) ? "Remove from failure" : "Add selected failure"}
        </button>
        <button type="button" className="button-primary" disabled={!removed.length || busy} onClick={onRun}>
          {busy ? "Running…" : "Run stress test"}
        </button>
      </div>
      {removed.length > 0 && (
        <div className="failure-chips" aria-label="Failed junctions">
          {removed.map((nodeId) => (
            <button key={nodeId} type="button" onClick={() => onToggleRemoved(nodeId)}>
              J-{nodeId}<span aria-hidden="true"> ×</span><span className="sr-only"> remove</span>
            </button>
          ))}
          <button type="button" className="clear-chip" onClick={onReset}>Clear all</button>
        </div>
      )}
    </section>
  );
}
