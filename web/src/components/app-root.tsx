"use client";

import { useEffect, useState } from "react";

import { CityPicker } from "@/components/city-picker";
import { ResilienceStudio } from "@/components/resilience-studio";
import { initialAoi } from "@/lib/model";

// The static export has one page: ?city= opens a city's studio, no city shows the picker.
export function AppRoot() {
  const [aoi, setAoi] = useState<string | null | undefined>(undefined);

  useEffect(() => {
    queueMicrotask(() => setAoi(initialAoi(window.location.search)));
  }, []);

  if (aoi === undefined) return null;
  return aoi ? <ResilienceStudio key={aoi} aoi={aoi} /> : <CityPicker />;
}
