export type WorkspaceMode = "explore" | "stress" | "compare" | "recover";

export type CriticalNode = {
  node_id: number;
  rank: number;
  betweenness: number;
  is_critical: boolean;
  is_articulation: boolean;
  x: number;
  y: number;
};

export type ResiliencePoint = {
  n_removed: number;
  targeted_resilience_index: number;
  random_resilience_index: number;
  [key: string]: number;
};
