"""A10 — clean a predicted road mask before it becomes a graph.

The raw model mask carries tiny false-positive blobs and pin-hole gaps. This
module removes that noise **without touching the real road network**, so P2
(skeletonize → graph) starts from a cleaner mask. It operates on binary ``{0,1}``
2-D ``uint8`` masks — the output of :func:`predict_large` — and is called by
``predict.py`` via ``--postprocess``.

Design rule: be **conservative**. The default chain only drops small
disconnected components (lossless on the large, connected real network) plus an
optional gentle close to bridge pin-holes, so IoU is preserved (Tracker A10:
"tiny false components drop … without IoU loss > 0.005"). Erosive ops (opening,
spur pruning) are opt-in because they can nibble thin 1-px roads.
"""
from __future__ import annotations

import numpy as np


def remove_small_components(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Drop connected road blobs smaller than ``min_size`` pixels (false positives).

    Lossless on the real network: only whole components below the threshold are
    removed, so large/connected roads are untouched.
    """
    from skimage.morphology import remove_small_objects

    cleaned = remove_small_objects(mask.astype(bool), min_size=min_size, connectivity=2)
    return cleaned.astype(np.uint8)


def fill_small_holes(mask: np.ndarray, area_threshold: int = 50) -> np.ndarray:
    """Fill small background holes inside roads (pin-holes from thresholding)."""
    from skimage.morphology import remove_small_holes

    filled = remove_small_holes(mask.astype(bool), area_threshold=area_threshold, connectivity=2)
    return filled.astype(np.uint8)


def morphological_cleanup(mask: np.ndarray, open_radius: int = 0, close_radius: int = 0) -> np.ndarray:
    """Binary opening (remove specks) then closing (bridge tiny gaps).

    ``open_radius``/``close_radius`` are disk radii in pixels; ``0`` skips that
    step. Opening is erosive — keep it ``0`` for thin roads.
    """
    from skimage.morphology import binary_closing, binary_opening, disk

    out = mask.astype(bool)
    if open_radius > 0:
        out = binary_opening(out, disk(open_radius))
    if close_radius > 0:
        out = binary_closing(out, disk(close_radius))
    return out.astype(np.uint8)


def otsu_threshold(prob: np.ndarray) -> np.ndarray:
    """Binarize a float probability/score map with Otsu's method (data-adaptive).

    A drop-in alternative to a fixed threshold when the score distribution is
    bimodal. A degenerate (constant) map yields all-background.
    """
    from skimage.filters import threshold_otsu

    prob = np.asarray(prob, dtype=float)
    if prob.max() <= prob.min():
        return np.zeros(prob.shape, dtype=np.uint8)
    return (prob > threshold_otsu(prob)).astype(np.uint8)


def _neighbour_count(binary: np.ndarray) -> np.ndarray:
    """8-connected foreground-neighbour count per pixel."""
    from scipy.ndimage import convolve

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return convolve(binary.astype(np.uint8), kernel, mode="constant")


def prune_skeleton_spurs(skeleton: np.ndarray, max_spur_len: int = 5) -> np.ndarray:
    """Remove short dead-end branches (spurs) from a 1-px skeleton.

    Walks inward from each skeleton endpoint until it reaches a branch point; if
    the branch is no longer than ``max_spur_len`` it is removed (a junction nub
    may remain — graph-level stub pruning handles the rest). Long branches and
    through-lines are preserved.
    """
    skel = skeleton.astype(bool).copy()
    counts = _neighbour_count(skel)
    height, width = skel.shape

    def neighbours(row: int, col: int) -> list[tuple[int, int]]:
        out = []
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row or d_col:
                    r, c = row + d_row, col + d_col
                    if 0 <= r < height and 0 <= c < width and skel[r, c]:
                        out.append((r, c))
        return out

    to_remove: set[tuple[int, int]] = set()
    endpoints = [tuple(p) for p in np.argwhere(skel & (counts == 1))]
    for endpoint in endpoints:
        path = [endpoint]
        previous, current = None, endpoint
        while len(path) <= max_spur_len:
            forward = [n for n in neighbours(*current) if n != previous]
            if len(forward) != 1:
                break  # junction or dead end reached
            nxt = forward[0]
            if counts[nxt] >= 3:  # spur meets a branch point — short enough to drop
                to_remove.update(path)
                break
            path.append(nxt)
            previous, current = current, nxt

    for row, col in to_remove:
        skel[row, col] = False
    return skel.astype(np.uint8)


def postprocess_mask(
    mask: np.ndarray,
    min_size: int = 50,
    open_radius: int = 0,
    close_radius: int = 0,
    fill_holes: int = 0,
) -> np.ndarray:
    """Conservative cleanup chain for a predicted binary road mask.

    Order: morphological open/close → drop small components → fill small holes.
    Defaults only remove small components (IoU-safe); enable the others per AOI.
    """
    out = morphological_cleanup(mask, open_radius=open_radius, close_radius=close_radius)
    out = remove_small_components(out, min_size=min_size)
    if fill_holes > 0:
        out = fill_small_holes(out, area_threshold=fill_holes)
    return out
