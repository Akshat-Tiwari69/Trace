"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { UploadDialog } from "@/components/upload-dialog";
import { fetchAtlases, type AtlasListing } from "@/lib/api";
import { formatMetric, groupAtlases, sourceLabel, trainingLabel } from "@/lib/model";

export function CityPicker() {
  const [atlases, setAtlases] = useState<AtlasListing[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);

  useEffect(() => {
    let active = true;
    fetchAtlases()
      .then((value) => { if (active) setAtlases(value.aois); })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "The city list could not be loaded");
      });
    return () => { active = false; };
  }, []);

  return (
    <div className="picker-shell">
      <header className="topbar picker-topbar">
        <div className="brand-lockup">
          <span className="brand-mark">TRACE</span>
          <span><strong>Route resilience</strong><small>Network field atlas</small></span>
        </div>
        <nav className="top-actions" aria-label="Project actions">
          <Link href="/methodology" prefetch={false}>Method</Link>
          <button type="button" className="upload-trigger" onClick={() => setUploadOpen(true)}>
            Analyze<span className="label-extra"> imagery</span>
            <svg aria-hidden="true" viewBox="0 0 18 18"><path d="M9 3v12M3 9h12" /></svg>
          </button>
        </nav>
      </header>

      <main className="picker" id="cities">
        <section className="picker-intro">
          <span className="eyebrow">Choose a city</span>
          <h1>Which junction would break the city?</h1>
          <p>
            Each area is a road network extracted from satellite imagery by the current model, or taken from
            OpenStreetMap for reference. Open one to find its critical junctions and stress-test them.
          </p>
        </section>

        {error ? <p className="form-error" role="alert">{error}</p> : null}
        {!atlases && !error ? <p className="picker-loading" role="status">Loading cities…</p> : null}

        {atlases ? (
          <ul className="city-grid">
            {groupAtlases(atlases).map((group) => (
              <li key={group[0].area ?? group[0].aoi} className="city-card">
                <span className="eyebrow">{group[0].region ?? "Area"}</span>
                <h2>{group[0].label}</h2>
                {group.map((atlas) => (
                  <a key={atlas.aoi} className="city-source" href={`/?city=${encodeURIComponent(atlas.aoi)}`}>
                    <strong>{sourceLabel(atlas)}</strong>
                    {trainingLabel(atlas) ? <small>{trainingLabel(atlas)}</small> : null}
                    <span>
                      {atlas.node_count} junctions · worst single failure −
                      {formatMetric(atlas.worst_single_loss * 100, { digits: 1, suffix: "%" })}
                    </span>
                  </a>
                ))}
              </li>
            ))}
          </ul>
        ) : null}
      </main>
      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} />
    </div>
  );
}
