import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const nodes = [
  { node_id: 278, rank: 1, betweenness: .331, is_critical: true, is_articulation: true, x: 73.824, y: 15.5 },
  { node_id: 261, rank: 2, betweenness: .264, is_critical: true, is_articulation: false, x: 73.834, y: 15.49 },
];
const graph = {
  type: "FeatureCollection",
  features: [
    ...nodes.map((node) => ({ type: "Feature", geometry: { type: "Point", coordinates: [node.x, node.y] }, properties: { feature_type: "node", ...node } })),
    { type: "Feature", geometry: { type: "LineString", coordinates: [[73.824, 15.5], [73.834, 15.49]] }, properties: { feature_type: "edge", u: 278, v: 261, is_bridged: false } },
  ],
};
const summary = {
  aoi: "panaji_demo", label: "Panaji, Goa", coordinate_system: "EPSG:4326",
  bounds: [73.82, 15.48, 73.84, 15.51], node_count: 364, edge_count: 535,
  critical_count: 2, baseline_efficiency: .001, graph_url: "/api/v1/aois/panaji_demo/graph",
  critical_nodes: nodes,
  resilience_curve: [
    { n_removed: 0, targeted_resilience_index: 1, random_resilience_index: 1 },
    { n_removed: 1, targeted_resilience_index: .8, random_resilience_index: .96 },
  ],
  evidence: {},
};

async function mockApi(page: Page) {
  let statusCalls = 0;
  await page.route("https://tiles.openfreemap.org/styles/positron", (route) => route.fulfill({
    json: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#dfe7e1" } }] },
  }));
  await page.route("**/api/v1/**", async (route) => {
    const { pathname } = new URL(route.request().url());
    if (pathname === "/api/v1/aois/panaji_demo") return route.fulfill({ json: summary });
    if (pathname === "/api/v1/aois/panaji_demo/graph") return route.fulfill({ json: graph });
    if (pathname === "/api/v1/simulations") return route.fulfill({ json: {
      aoi: "panaji_demo", removed_node_ids: [278], resilience_index: .8, efficiency_loss: .2,
      largest_cc_fraction: .76, active_largest_cc_fraction: .76,
      representative_route: { origin: 10, destination: 11, baseline_length_m: 100, rerouted_length_m: 145, travel_time_delta_pct: 45, disconnected: false },
    } });
    if (pathname === "/api/v1/analyses" && route.request().method() === "POST") return route.fulfill({ status: 202, json: {
      id: "a".repeat(32), job_id: "a".repeat(32), status: "queued", position: 0,
      status_url: `/api/v1/analyses/${"a".repeat(32)}`, result_url: `/api/v1/analyses/${"a".repeat(32)}/result`,
    } });
    if (pathname === `/api/v1/analyses/${"a".repeat(32)}`) {
      statusCalls += 1;
      return route.fulfill({ json: { id: "a".repeat(32), job_id: "a".repeat(32), status: statusCalls > 0 ? "done" : "running", position: 0, result_url: `/api/v1/analyses/${"a".repeat(32)}/result` } });
    }
    if (pathname.endsWith("/result")) return route.fulfill({ json: {
      id: "a".repeat(32), status: "done", n_nodes: 42, n_edges: 57, resolution_m: .5,
      resilience_index: .812, top_node: 7, summary: { critical_junctions: 4, articulation_points: 2 },
      criticality: [], graph_url: `/api/v1/analyses/${"a".repeat(32)}/graph`,
      exports: { json: `/api/v1/analyses/${"a".repeat(32)}/result`, geojson: `/api/v1/analyses/${"a".repeat(32)}/graph` },
    } });
    if (pathname.endsWith("/graph")) return route.fulfill({ json: { type: "FeatureCollection", features: [] } });
    return route.abort();
  });
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Panaji, Goa" })).toBeVisible();
});

test("keyboard modes, failure simulation, export and accessibility", async ({ page }) => {
  const explore = page.getByRole("tab", { name: /Explore/ });
  await explore.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: /Stress/ })).toHaveAttribute("aria-selected", "true");
  await page.getByRole("button", { name: "Add selected failure" }).click();
  await page.getByRole("button", { name: "Run stress test" }).click();
  await expect(page.getByText("20.0% efficiency lost")).toBeVisible();

  await page.getByText("Export", { exact: true }).click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: /GeoJSON/ }).click();
  expect((await download).suggestedFilename()).toContain("scenario.geojson");

  const violations = await new AxeBuilder({ page }).analyze();
  expect(violations.violations.filter((item) => ["serious", "critical"].includes(item.impact ?? ""))).toEqual([]);
});

test("upload dialog supports Escape and completes the queued workflow", async ({ page }) => {
  await page.getByRole("button", { name: /Analyze imagery/ }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();

  await page.getByRole("button", { name: /Analyze imagery/ }).click();
  await page.getByLabel(/Choose a satellite image/).setInputFiles({
    name: "roads.png", mimeType: "image/png",
    buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=", "base64"),
  });
  await page.getByLabel(/I confirm I may process/).check();
  await page.getByRole("button", { name: "Extract road network" }).click();
  await expect(page.getByText("Worst-junction resilience")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText("0.812")).toBeVisible();
});

test("reflows without horizontal overflow at target widths", async ({ page }) => {
  for (const width of [375, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: width === 375 ? 812 : 900 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
});

test("map canvas fills its frame", async ({ page }) => {
  const frame = await page.locator(".map-frame").boundingBox();
  const map = await page.locator(".network-map").boundingBox();

  expect(frame?.height).toBeGreaterThan(0);
  expect(map?.height).toBe(frame?.height);
});
