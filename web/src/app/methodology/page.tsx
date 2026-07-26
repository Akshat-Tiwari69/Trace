import Link from "next/link";

export default function MethodologyPage() {
  return (
    <main className="method-page">
      <header>
        <Link href="/" className="back-link">← Return to field atlas</Link>
        <span className="brand-mark">TRACE</span>
        <p className="eyebrow">Methodology · evidence · limits</p>
        <h1>A network claim should survive the route.</h1>
        <p className="method-intro">
          TRACE converts aligned road detections into a routable MultiGraph, identifies high-dependency junctions, and measures what remains reachable when those junctions fail.
        </p>
      </header>
      <section className="method-grid">
        <article>
          <span>01</span><h2>Detect</h2>
          <p>A PyTorch segmentation endpoint produces a binary road mask. The public host never runs the GPU model; authenticated Modal is the only hosted inference boundary.</p>
        </article>
        <article>
          <span>02</span><h2>Build</h2>
          <p>The mask is skeletonized into a keyed MultiGraph. Metric geometry, parallel links, inferred repairs and coordinate provenance remain explicit.</p>
        </article>
        <article>
          <span>03</span><h2>Stress</h2>
          <p>Failures isolate nodes inside the original baseline universe. This prevents the denominator from shrinking and making a damaged network look healthier.</p>
        </article>
        <article>
          <span>04</span><h2>Measure</h2>
          <p>Resilience is perturbed global efficiency divided by baseline global efficiency. Disconnected pairs contribute zero, so the index stays finite in [0, 1].</p>
        </article>
      </section>
      <section className="formula-card">
        <p className="eyebrow">Locked project metric</p>
        <div className="formula">RI = E(G<sub>perturbed</sub>) / E(G<sub>baseline</sub>)</div>
        <p>Global efficiency averages inverse shortest-path distance. It is not a raw average-path-length ratio.</p>
      </section>
      <section className="limits-section">
        <div><p className="eyebrow">What this sample proves</p><h2>A contract-shaped, CPU-runnable planning demonstration.</h2></div>
        <ul>
          <li>Panaji sample graph and analysis artifacts are committed and reproducible.</li>
          <li>Mumbai model work is a repeatedly consulted development benchmark, not an untouched final test.</li>
          <li>No claim of live traffic, universal city generalization, or authoritative emergency routing is made.</li>
          <li>Graph-first research candidates stay out of production until metrics, provenance, licensing and deployment checks all pass.</li>
        </ul>
      </section>
      <footer><span>Route Resilience</span><Link href="/">Open the atlas →</Link></footer>
    </main>
  );
}
