import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchAoi, runSimulation, submitAnalysis } from "@/lib/api";

afterEach(() => vi.unstubAllGlobals());

describe("API client", () => {
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
