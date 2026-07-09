"""Segmentation model: build, checkpoint I/O, and CPU inference (task A4).

We fine-tune a **SegFormer MiT encoder + U-Net decoder** via
``segmentation_models_pytorch`` (smp ships the MiT/SegFormer encoders but
not a standalone Segformer decoder, and the U-Net decoder gives full-resolution
masks directly). The module supports several MiT encoders (mit_b0…mit_b3);
the **deployed v3.2 checkpoint uses MiT-B3 + SCSE U-Net**.
``encoder_weights="imagenet"`` fine-tunes pretrained weights —
never trains from scratch (``docs/Rules.md``).

``predict_mask`` is the pipeline's ``predict(tile) -> mask_array`` (``TRD.md``):
a trained model turns one RGB tile into a binary {0,1} road mask on CPU, which
P2 then skeletonises.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch

# Single source of truth for the deployed segmentation checkpoint (A36).
# Every CLI default / help text / error message should reference these, not a
# hard-coded path, so a re-deploy is a one-line change here.
DEPLOYED_CHECKPOINT = "models/road_pan.pt"
DEPLOYED_RELEASE = "a4-roadseg-v3.2"

# ImageNet stats — the MiT encoders are pretrained on ImageNet, so train-time
# augmentation and inference must normalise with the same constants.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# smp decoders that build with the MiT (SegFormer) encoders. UnetPlusPlus and
# DeepLabV3+ do NOT support MiT encoders (verified), so they're excluded.
_ARCHITECTURES = {"unet": smp.Unet, "manet": smp.MAnet, "fpn": smp.FPN}


def build_model(
    encoder: str = "mit_b0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    arch: str = "unet",
    decoder_attention_type: str | None = None,
) -> torch.nn.Module:
    """Build the SegFormer-MiT road segmenter.

    ``arch`` picks the smp decoder (``unet`` default; also ``manet``/``fpn``).
    ``decoder_attention_type`` (e.g. ``"scse"``) adds attention to the U-Net
    decoder blocks — a cheap accuracy boost; only valid for ``arch="unet"``.
    ``encoder_weights="imagenet"`` fine-tunes pretrained weights; ``None`` gives
    a random encoder (tests/offline). Raw logits out (loss/metrics apply sigmoid).
    """
    arch = arch.lower()
    if arch not in _ARCHITECTURES:
        raise ValueError(f"arch must be one of {sorted(_ARCHITECTURES)}, got {arch!r}")
    kwargs: dict[str, Any] = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    if decoder_attention_type:
        if arch != "unet":
            raise ValueError("decoder_attention_type is only supported for arch='unet'")
        kwargs["decoder_attention_type"] = decoder_attention_type
    return _ARCHITECTURES[arch](**kwargs)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, unwrapping DataParallel/DDP if present."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    meta: dict[str, Any] | None = None,
    train_state: dict[str, Any] | None = None,
) -> None:
    """Save model weights + metadata (encoder, metrics, config) to ``path``.

    Unwraps DataParallel/DDP so the saved keys match a plain model on reload.
    ``train_state`` (optimizer/scaler/epoch/… for A19 ``--resume``) is stored only
    when given, so plain weight checkpoints stay byte-for-byte as before.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {"state_dict": _unwrap(model).state_dict(), "meta": meta or {}}
    if train_state is not None:
        blob["train_state"] = train_state
    torch.save(blob, path)


def load_checkpoint_blob(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Read a checkpoint dict, preferring torch's safe ``weights_only=True`` path.

    Checkpoints written by :func:`save_checkpoint` hold only tensors and
    JSON-safe primitives (``str``/``int``/``float``/``bool``/``list``/``dict``),
    so they load under the restricted unpickler with no loss of function.
    Older/hand-rolled checkpoints (e.g. pre-A19 files with raw optimizer/
    scheduler objects at the top level) can't be safely unpickled that way;
    those fall back to a full pickle load with a one-time warning, since
    re-serialising the already-deployed checkpoint is out of scope here.
    ``map_location`` (default "cpu") keeps GPU-trained checkpoints CPU-safe.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(
            f"{path}: legacy pickled checkpoint (weights_only=True load failed) — "
            "falling back to a full pickle load. Re-save with save_checkpoint() to "
            "enable safe loading.",
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def load_train_state(path: str | Path, map_location: str = "cpu") -> dict[str, Any] | None:
    """Return the A19 training state saved next to a checkpoint, or ``None``.

    Companion to :func:`load_checkpoint` (which rebuilds the model): this pulls the
    optimizer/scaler/epoch/best/history blob so a run can resume mid-training.
    """
    ckpt = load_checkpoint_blob(path, map_location=map_location)
    return ckpt.get("train_state")


def load_checkpoint(
    path: str | Path,
    encoder: str | None = None,
    map_location: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the model (random encoder, no download) and load saved weights.

    The encoder/arch default to whatever the checkpoint's ``meta`` says (so a
    mit_b3 Unet+scse checkpoint rebuilds correctly); pass ``encoder`` to override.
    A checkpoint with **no** ``meta`` is refused (``ValueError``) rather than
    silently rebuilt as mit_b0/unet — unless ``encoder`` explicitly asserts the
    architecture.
    """
    ckpt = load_checkpoint_blob(path, map_location=map_location)
    meta = ckpt.get("meta") or {}
    if not meta and encoder is None:
        raise ValueError(f"{path} has no 'meta' — cannot verify architecture; refusing to guess")
    enc = encoder or meta.get("encoder", "mit_b0")
    model = build_model(
        encoder=enc,
        encoder_weights=None,
        arch=meta.get("arch", "unet"),
        decoder_attention_type=meta.get("decoder_attention_type"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


def _to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8/float image → normalised (1,3,H,W) float tensor."""
    arr = image.astype(np.float32)
    # uint8 is always 0–255 (even a near-black tile whose max is ≤1); floats are
    # 0–1 by contract, so only rescale those if they clearly exceed that range.
    if image.dtype == np.uint8 or arr.max() > 1.0:
        arr /= 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


@torch.no_grad()
def _dihedral_tta_prob(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Mean sigmoid over the 8 D4 symmetries (4 rotations × optional h-flip).

    Aerial roads have no canonical orientation, so averaging the prediction over
    the dihedral group is a cheap, retrain-free accuracy boost (task A7). Each
    transform is undone before averaging so the probabilities line up.
    """
    probs = []
    for k in range(4):  # 0/90/180/270° rotations
        xr = torch.rot90(x, k, dims=(2, 3))
        probs.append(torch.rot90(torch.sigmoid(net(xr)), -k, dims=(2, 3)))
        xf = torch.flip(xr, dims=(3,))  # + horizontal flip
        probs.append(torch.rot90(torch.flip(torch.sigmoid(net(xf)), dims=(3,)), -k, dims=(2, 3)))
    return torch.stack(probs).mean(dim=0)


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str = "cpu",
    threshold: float = 0.5,
    tta: bool = False,
) -> np.ndarray:
    """Predict a binary {0,1} road mask for one RGB tile (``predict(tile)``).

    ``image`` is ``HxWx3`` (uint8 0–255 or float 0–1). Returns a ``uint8``
    ``HxW`` mask of 0/1, the §4 contract shape P2 consumes. ``tta`` averages over
    the 8 D4 symmetries (retrain-free; ~8× compute).
    """
    net = _unwrap(model).to(device)
    net.eval()
    x = _to_chw_tensor(image).to(device)
    prob = _dihedral_tta_prob(net, x) if tta else torch.sigmoid(net(x))
    return (prob[0, 0].cpu().numpy() >= threshold).astype(np.uint8)


@torch.no_grad()
def predict_prob(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str = "cpu",
    tta: bool = False,
) -> np.ndarray:
    """Per-pixel road **probability** map (``HxW`` float in [0,1]) for one tile.

    Same forward pass as :func:`predict_mask` without the threshold — lets callers
    sweep thresholds (A21) or blend overlapping windows (A27) before binarising.
    """
    net = _unwrap(model).to(device)
    net.eval()
    x = _to_chw_tensor(image).to(device)
    prob = _dihedral_tta_prob(net, x) if tta else torch.sigmoid(net(x))
    return prob[0, 0].cpu().numpy().astype(np.float32)


def _auto_batch(device: str, batch_size: int | None) -> int:
    """Resolve the tile batch size (A28). GPU throughput scales with batch, but a
    CPU conv already uses every core — batching there only adds overhead — so the
    default is 16 on an accelerator and 1 (serial) on CPU. Pass an int to override.
    """
    if batch_size is not None:
        return batch_size
    return 1 if str(device) == "cpu" else 16


@torch.no_grad()
def predict_prob_batch(
    model: torch.nn.Module,
    images: list[np.ndarray],
    device: str = "cpu",
    tta: bool = False,
    batch_size: int | None = None,
) -> list[np.ndarray]:
    """Probability maps for a list of **same-size** tiles in batched forwards (A28).

    Same per-tile result as calling :func:`predict_prob` in a loop, but runs the
    model on chunks of ``batch_size`` tiles at once (~B× GPU throughput). Identical
    outputs under ``model.eval()`` — BatchNorm uses running stats, so each sample is
    independent of its batch-mates. All ``images`` must share ``H×W``. ``batch_size``
    defaults to device-aware (:func:`_auto_batch`): serial on CPU, 16 on GPU.
    """
    if not images:
        return []
    batch_size = _auto_batch(device, batch_size)
    net = _unwrap(model).to(device)
    net.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        x = torch.cat([_to_chw_tensor(im) for im in chunk], dim=0).to(device)
        prob = _dihedral_tta_prob(net, x) if tta else torch.sigmoid(net(x))
        out.extend(prob[i, 0].cpu().numpy().astype(np.float32) for i in range(prob.shape[0]))
    return out


def _hann2d(size: int) -> np.ndarray:
    """2-D Hann window (+small floor so edge pixels still contribute)."""
    w = np.hanning(size)
    return np.outer(w, w).astype(np.float32) + 1e-3


@torch.no_grad()
def predict_large_prob(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 512,
    stride: int | None = None,
    device: str = "cpu",
    tta: bool = False,
    batch_size: int | None = None,
) -> np.ndarray:
    """Blended probability map for a large image via **overlapping** windows.

    Slides ``tile_size`` windows at ``stride`` (default 75 % overlap), weights each
    tile's probabilities by a Hann window and normalises — so roads crossing tile
    seams don't break (A27). Threshold the returned map **once**. Reuses padding
    for edge windows. Windows run batched (A28, device-aware) rather than one
    forward per tile — the blended map is unchanged.
    """
    stride = stride or max(1, tile_size * 3 // 4)
    h, w = image.shape[:2]
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    window = _hann2d(tile_size)
    ys = sorted({*range(0, max(1, h - tile_size + 1), stride), max(0, h - tile_size)})
    xs = sorted({*range(0, max(1, w - tile_size + 1), stride), max(0, w - tile_size)})
    tiles: list[np.ndarray] = []
    places: list[tuple[int, int, int, int]] = []  # (y0, x0, th, tw) per window
    for y0 in ys:
        for x0 in xs:
            tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:  # pad edge windows
                padded = np.zeros((tile_size, tile_size, image.shape[2]), image.dtype)
                padded[:th, :tw] = tile
                tile = padded
            tiles.append(tile)
            places.append((y0, x0, th, tw))
    probs = predict_prob_batch(model, tiles, device=device, tta=tta, batch_size=batch_size)
    for (y0, x0, th, tw), prob in zip(places, probs):
        win = window[:th, :tw]
        acc[y0 : y0 + th, x0 : x0 + tw] += prob[:th, :tw] * win
        wsum[y0 : y0 + th, x0 : x0 + tw] += win
    return acc / np.maximum(wsum, 1e-6)


@torch.no_grad()
def predict_large_raster(
    model: torch.nn.Module,
    image_path,
    tile_size: int = 512,
    threshold: float = 0.5,
    device: str = "cpu",
    tta: bool = False,
    window_px: int = 2048,
    overlap_px: int = 256,
) -> np.ndarray:
    """Binary road mask for a raster too large to hold in RAM (bugs.md §5H).

    Streams overlapping ``window_px`` windows off disk (rasterio/PIL windowed
    reads), runs the blended :func:`predict_large_prob` on each, thresholds it,
    and OR-merges the binary result into a single full-size ``uint8`` mask — the
    only full-resolution array ever allocated. Roads are sparse, so an OR at the
    window overlaps closes seams without dropping links. Returns the ``{0,1}``
    mask for the whole raster.
    """
    from src.pipeline.p1_segment.raster_io import iter_windows, raster_dimensions

    height, width = raster_dimensions(image_path)
    mask = np.zeros((height, width), np.uint8)
    for window, r0, c0 in iter_windows(image_path, window_px=window_px, overlap_px=overlap_px):
        prob = predict_large_prob(model, window, tile_size=tile_size, device=device, tta=tta)
        wmask = (prob >= threshold).astype(np.uint8)
        h, w = wmask.shape
        region = mask[r0 : r0 + h, c0 : c0 + w]
        np.bitwise_or(region, wmask, out=region)  # merge; keeps roads at seams
    return mask


@torch.no_grad()
def predict_large(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 512,
    device: str = "cpu",
    threshold: float = 0.5,
    tta: bool = False,
) -> np.ndarray:
    """Predict a binary {0,1} road mask for a whole (large) RGB image.

    The model only sees ``tile_size`` windows, so tile the image, predict each
    tile, and stitch the results back to the original ``HxW`` — the §4 contract
    ``data/interim/{aoi}_mask.png`` that P2 consumes. Reuses A3's ``tile_array``.
    ``tta`` applies D4 test-time augmentation per tile.
    """
    from src.pipeline.p1_segment.osm_mask import tile_array

    net = _unwrap(model).to(device)
    h, w = image.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    for t in tile_array(image, tile_size):
        pred = predict_mask(net, t.data, device=device, threshold=threshold, tta=tta)
        th, tw = min(tile_size, h - t.y0), min(tile_size, w - t.x0)
        full[t.y0 : t.y0 + th, t.x0 : t.x0 + tw] = pred[:th, :tw]  # drop padding
    return full
