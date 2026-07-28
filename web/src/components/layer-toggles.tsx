type Layers = {
  critical: boolean;
  bridged: boolean;
  spof: boolean;
};

type Props = {
  layers: Layers;
  onChange: (layers: Layers) => void;
};

const OPTIONS: Array<{ key: keyof Layers; label: string; description: string }> = [
  { key: "critical", label: "Critical junctions", description: "High network betweenness" },
  { key: "bridged", label: "Inferred links", description: "Gap-healed road segments" },
  { key: "spof", label: "Single points", description: "Articulation junctions" },
];

export function LayerToggles({ layers, onChange }: Props) {
  return (
    <fieldset className="layer-list">
      <legend>Map layers</legend>
      {OPTIONS.map((option) => (
        <label key={option.key} className="layer-toggle">
          <span className={`layer-swatch swatch-${option.key}`} aria-hidden="true" />
          <span><strong>{option.label}</strong><small>{option.description}</small></span>
          <input
            type="checkbox"
            checked={layers[option.key]}
            onChange={(event) => onChange({ ...layers, [option.key]: event.target.checked })}
          />
          <span className="toggle-ui" aria-hidden="true" />
        </label>
      ))}
    </fieldset>
  );
}
