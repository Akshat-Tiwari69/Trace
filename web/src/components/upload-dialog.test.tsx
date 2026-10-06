import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { statusCopy, UploadDialog } from "@/components/upload-dialog";

describe("UploadDialog", () => {
  it("uses the native modal boundary and rejects an unsupported file before upload", async () => {
    const onClose = vi.fn();
    render(<UploadDialog open onClose={onClose} />);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("open");

    const input = screen.getByLabelText(/Choose a satellite image/);
    fireEvent.change(input, { target: { files: [new File(["x"], "roads.txt", { type: "text/plain" })] } });
    expect(await screen.findByRole("alert")).toHaveTextContent("Choose a PNG or JPEG image");

    fireEvent(dialog, new Event("cancel", { cancelable: true }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});

describe("statusCopy", () => {
  const idle = { job: null, busy: false, hasFile: false, consent: false, elapsed: 0 };

  it("tells the user the next step instead of 'Preparing upload'", () => {
    expect(statusCopy(idle)).toBe("Choose an image to begin");
    expect(statusCopy({ ...idle, hasFile: true })).toBe("Confirm processing consent to continue");
    expect(statusCopy({ ...idle, hasFile: true, consent: true })).toBe("Ready to extract");
  });

  it("names the GPU stage with elapsed time while the upload waits", () => {
    expect(statusCopy({ ...idle, busy: true, hasFile: true, consent: true, elapsed: 12 })).toMatch(/GPU · 12 s/);
  });
});
