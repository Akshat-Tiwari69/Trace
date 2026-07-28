import type { MetadataRoute } from "next";

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "TRACE — Route Resilience Field Atlas",
    short_name: "TRACE",
    description: "Explore and stress-test an urban road network from satellite-derived evidence.",
    start_url: "/",
    display: "standalone",
    background_color: "#f4f0e6",
    theme_color: "#0b1413",
    icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml" }],
  };
}
