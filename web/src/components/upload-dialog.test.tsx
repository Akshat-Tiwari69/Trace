import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { UploadDialog } from "@/components/upload-dialog";

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
