import { gzipSync } from "node:zlib";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { extname, join, relative, resolve, sep } from "node:path";

const OUT = resolve("out");
const LIMITS = {
  // Next 16 + React 19's measured route/runtime floor is ~197 KiB gzip.
  "Non-map JavaScript": 205,
  // maplibre-gl 6's module worker re-downloads its ~145 KiB shared module (F15).
  "Map JavaScript": 430,
  "Total JavaScript": 615,
  CSS: 35,
  Fonts: 100,
  "Initial shell": 650,
};

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}

function gzipKiB(path) {
  return gzipSync(readFileSync(path)).byteLength / 1024;
}

const emitted = files(OUT);
// Turbopack also copies every maplibre .mjs into _next/static/media; nothing requests
// those copies. The worker the map really loads is public/maplibre/<version>/.
const scripts = emitted.filter((path) => [".js", ".mjs"].includes(extname(path))
  && !relative(OUT, path).split(sep).join("/").startsWith("_next/static/media/"));
const mapScripts = scripts.filter((path) => /maplibre|mercatorcoordinate|webglcontextattributes/i.test(readFileSync(path, "utf8")));
const mapSet = new Set(mapScripts);
const css = emitted.filter((path) => extname(path) === ".css");
const fonts = emitted.filter((path) => /\.(?:woff2?|ttf|otf)$/i.test(path));
const index = readFileSync(join(OUT, "index.html"), "utf8");
const initialAssets = [...index.matchAll(/(?:src|href)="([^"?]+\.(?:js|css))/g)]
  .map((match) => join(OUT, match[1].replace(/^\//, "")))
  .filter((path) => statSync(path).isFile());

const values = {
  "Non-map JavaScript": scripts.filter((path) => !mapSet.has(path)).reduce((sum, path) => sum + gzipKiB(path), 0),
  "Map JavaScript": mapScripts.reduce((sum, path) => sum + gzipKiB(path), 0),
  "Total JavaScript": scripts.reduce((sum, path) => sum + gzipKiB(path), 0),
  CSS: css.reduce((sum, path) => sum + gzipKiB(path), 0),
  Fonts: fonts.reduce((sum, path) => sum + statSync(path).size / 1024, 0),
  "Initial shell": gzipSync(index).byteLength / 1024 + initialAssets.reduce((sum, path) => sum + gzipKiB(path), 0),
};

let failed = false;
for (const [label, limit] of Object.entries(LIMITS)) {
  const value = values[label];
  const okay = value <= limit;
  failed ||= !okay;
  console.log(`${okay ? "PASS" : "FAIL"} ${label.padEnd(23)} ${value.toFixed(1).padStart(7)} KiB / ${limit} KiB`);
}
console.log(`Measured ${scripts.length} JS, ${css.length} CSS and ${fonts.length} font assets under ${relative(process.cwd(), OUT)}.`);
if (failed) process.exitCode = 1;
