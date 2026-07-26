type Props = {
  value: number;
  onChange: (value: number) => void;
};

export function CompareControl({ value, onChange }: Props) {
  return (
    <div className="compare-control">
      <div><span>Baseline</span><span>Disruption</span></div>
      <label>
        <span className="sr-only">Comparison position</span>
        <input
          type="range"
          min="0"
          max="100"
          value={value}
          aria-valuetext={`${value}% baseline view`}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </label>
      <p>Drag to blend the network state. Camera and scale remain locked.</p>
    </div>
  );
}
