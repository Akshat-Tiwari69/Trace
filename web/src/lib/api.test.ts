import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchAoi, runSimulation, submitAnalysis, UPLOAD_TIMEOUT_MS } from "@/lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("API client", () => {
  it("waits for the server's GPU call when uploading instead of the 10 s default", async () => {
    const timeout = vi.spyOn(AbortSignal, "timeout");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "a".repeat(32), status: "queued" }), { status: 202 }),
    ));

    await submitAnalysis(new File(["x"], "roads.png", { type: "image/png" }), 0.5);
    expect(timeout).toHaveBeenCalledWith(UPLOAD_TIMEOUT_MS);
    // modal_client.call worst case: 3 attempts x 120 s + 1.5 s + 3 s backoff.
    expect(UPLOAD_TIMEOUT_MS).toBeGreaterThan(3 * 120_000 + 4_500);
  });

  it("loads the validated sample contract", async () => {
    const payload = { aoi: "panaji_demo", graph_url: "/api/v1/aois/panaji_demo/graph" };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), { status: 200, headers: { "content-type": "application/json" } }),
    ));

    await expect(fetchAoi("panaji_demo")).resolves.toEqual(payload);
    expect(fetch).toHaveBeenCalledWith("/api/v1/aois/panaji_demo", expect.objectContaining({ signal: expect.any(AbortSignal) }));
  });

  it("sends sorted node IDs and preserves server validation errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Unknown node IDs: 9999" }), {
        status: 422,
        headers: { "content-type": "application/json" },
      }),
    ));

    await expect(runSimulation("panaji_demo", [278, 86, 278])).rejects.toEqual(
      expect.objectContaining({ status: 422, message: "Unknown node IDs: 9999" }),
    );
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/simulations",
      expect.objectContaining({ body: JSON.stringify({ aoi: "panaji_demo", removed_node_ids: [86, 278] }) }),
    );
  });

  it("submits the explicit external-processing consent contract", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "job", status: "queued" }), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    ));
    const image = new File(["png"], "roads.png", { type: "image/png" });

    await submitAnalysis(image, 0.5);

    const init = vi.mocked(fetch).mock.calls[0][1];
    const body = init?.body as FormData;
    expect(body.get("image")).toBe(image);
    expect(body.get("resolution_m")).toBe("0.5");
    expect(body.get("confirm_external_processing")).toBe("true");
  });
});
