// maplibre-gl 6 resolves its worker next to the chunk that bundled it, which
// Turbopack does not emit. Ship the worker (and the shared module it imports)
// as static files under a versioned directory, so upgrades never reuse a cached
// copy; network-map.tsx points setWorkerUrl() at them.
import { copyFileSync, mkdirSync, readFileSync, rmSync } from "node:fs";

const source = new URL("../node_modules/maplibre-gl/", import.meta.url);
const { version } = JSON.parse(readFileSync(new URL("package.json", source), "utf8"));
const root = new URL("../public/maplibre/", import.meta.url);
const target = new URL(`${version}/`, root);
rmSync(root, { recursive: true, force: true });
mkdirSync(target, { recursive: true });
for (const name of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) {
  copyFileSync(new URL(`dist/${name}`, source), new URL(name, target));
}
