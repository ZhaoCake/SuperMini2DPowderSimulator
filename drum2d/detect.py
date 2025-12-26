"""drum2d.detect

OpenCV baseline detector.

Reads a case directory containing video.mp4 (+ optional gt.npz/meta.json) and outputs:
- detect.npz: theta_hat (T,), line_kb (T,2), (optional) surface_points
- detect_preview.mp4: overlay visualization

No deep learning; only numpy/opencv/scipy/matplotlib/tqdm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Support both:
# - package run: python -m drum2d.main ...
# - script run inside drum2d/: python main.py ...
try:
    from .utils import (
        circle_mask,
        draw_circle_boundary,
        draw_line_kb,
        draw_polyline,
        ema_filter,
        rad2deg,
        ransac_line_fit,
        read_json,
        set_global_seed,
    )
except ImportError:  # pragma: no cover
    from utils import (
        circle_mask,
        draw_circle_boundary,
        draw_line_kb,
        draw_polyline,
        ema_filter,
        rad2deg,
        ransac_line_fit,
        read_json,
        set_global_seed,
    )


@dataclass
class DetectParams:
    """Detector hyperparameters."""

    blur_ksize: int = 7  # odd
    thresh_method: str = "otsu"  # otsu|adaptive
    ransac_resid_thresh: float = 2.8
    ema_alpha: float = 0.25

    # extra knobs (kept minimal)
    morph_ksize: int = 3
    adaptive_block: int = 31
    adaptive_c: int = 2


def _odd_ksize(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def _threshold_powder(gray: np.ndarray, method: str, params: DetectParams) -> np.ndarray:
    """Return boolean mask for powder (darker region)."""
    method = str(method).lower()
    if method == "otsu":
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu gives white for >thr; powder is darker => invert
        powder = thr == 0
        return powder

    if method == "adaptive":
        blk = _odd_ksize(params.adaptive_block)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blk,
            C=int(params.adaptive_c),
        )
        powder = thr == 0
        return powder

    raise ValueError(f"Unknown thresh_method: {method}")


def _extract_surface_points(powder: np.ndarray, circle: np.ndarray) -> np.ndarray:
    """For each x column, find top-most powder pixel inside circle."""
    h, w = powder.shape
    valid_x = np.where(circle.any(axis=0))[0]
    pts = []
    for x in valid_x:
        col = powder[:, x] & circle[:, x]
        ys = np.where(col)[0]
        if ys.size == 0:
            continue
        y = int(ys.min())
        pts.append((float(x), float(y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def detect_case(
    case_dir: str | Path,
    params: Optional[DetectParams] = None,
    seed: int = 0,
    write_preview: bool = True,
) -> Path:
    """Run detection for a case directory."""

    case_dir = Path(case_dir)
    if params is None:
        params = DetectParams()

    meta_path = case_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {case_dir}")
    meta = read_json(meta_path)

    cfg = meta.get("cfg", {})
    w = int(cfg.get("w", 512))
    h = int(cfg.get("h", 512))
    r = float(cfg.get("r", 220.0))
    cx = w / 2.0
    cy = h / 2.0

    video_path = case_dir / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video.mp4 in {case_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        # fallback: will grow arrays dynamically
        n_frames = 0

    circle = circle_mask(h, w, cx, cy, r)

    rng = set_global_seed(seed)

    theta_hat = []
    line_kb = []
    surface_points = []

    # preview writer
    preview_writer = None
    if write_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_path = case_dir / "detect_preview.mp4"
        preview_writer = cv2.VideoWriter(str(preview_path), fourcc, float(fps) if fps > 0 else 30.0, (w, h))
        if not preview_writer.isOpened():
            raise RuntimeError("Failed to open preview VideoWriter")

    # optional gt overlay
    gt = None
    gt_path = case_dir / "gt.npz"
    if gt_path.exists():
        gt = np.load(gt_path, allow_pickle=False)
        theta_gt = gt["theta_gt"].astype(np.float32)
        surface_gt = gt["surface_xy"].astype(np.float32)
    else:
        theta_gt = None
        surface_gt = None

    blur_k = _odd_ksize(params.blur_ksize)
    morph_k = _odd_ksize(params.morph_ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))

    pbar = tqdm(total=n_frames if n_frames > 0 else None, desc=f"detect {case_dir.name}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_k > 1:
            gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        powder = _threshold_powder(gray, params.thresh_method, params)
        powder &= circle

        # Morphological cleanup
        powder_u8 = (powder.astype(np.uint8) * 255)
        powder_u8 = cv2.morphologyEx(powder_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        powder_u8 = cv2.morphologyEx(powder_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        powder = powder_u8 > 0

        pts = _extract_surface_points(powder, circle)

        if pts.shape[0] >= 2:
            k, b, inliers = ransac_line_fit(pts, resid_thresh=params.ransac_resid_thresh, rng=rng)
        else:
            k, b = 0.0, float(cy)

        theta = rad2deg(math.atan(float(k)))
        theta_hat.append(theta)
        line_kb.append((float(k), float(b)))

        # keep a sparse subset of points for optional saving
        if pts.shape[0] > 0:
            step = max(1, pts.shape[0] // 120)
            surface_points.append(pts[::step])
        else:
            surface_points.append(np.zeros((0, 2), dtype=np.float32))

        # preview overlay
        if preview_writer is not None:
            vis = frame.copy()
            draw_circle_boundary(vis, cx, cy, r, color=(0, 255, 255), thickness=2)

            # draw detection line over circle extent
            x0 = int(max(0, cx - r))
            x1 = int(min(w - 1, cx + r))
            draw_line_kb(vis, float(k), float(b), x0, x1, color=(0, 255, 0), thickness=2)

            if surface_gt is not None and idx < surface_gt.shape[0]:
                draw_polyline(vis, surface_gt[idx], color=(0, 0, 255), thickness=2)

            # text
            txt = f"theta_hat={theta:+.2f} deg"
            if theta_gt is not None and idx < theta_gt.shape[0]:
                txt += f" | theta_gt={float(theta_gt[idx]):+.2f}"
            cv2.putText(vis, txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            preview_writer.write(vis)

        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if preview_writer is not None:
        preview_writer.release()

    theta_hat = np.asarray(theta_hat, dtype=np.float32)
    line_kb = np.asarray(line_kb, dtype=np.float32)

    # temporal smoothing (EMA) on k and b
    if theta_hat.size > 0 and params.ema_alpha > 0:
        k_s = ema_filter(line_kb[:, 0], params.ema_alpha)
        b_s = ema_filter(line_kb[:, 1], params.ema_alpha)
        line_kb[:, 0] = k_s
        line_kb[:, 1] = b_s
        theta_hat = np.asarray([rad2deg(math.atan(float(k))) for k in k_s], dtype=np.float32)

    # pack variable-length surface points into (T,N,2) with padding
    max_n = max((p.shape[0] for p in surface_points), default=0)
    if max_n > 0:
        sp = np.full((len(surface_points), max_n, 2), np.nan, dtype=np.float32)
        for i, p in enumerate(surface_points):
            if p.size == 0:
                continue
            sp[i, : p.shape[0], :] = p
    else:
        sp = np.zeros((len(surface_points), 0, 2), dtype=np.float32)

    np.savez_compressed(
        case_dir / "detect.npz",
        theta_hat=theta_hat.astype(np.float32),
        line_kb=line_kb.astype(np.float32),
        surface_points=sp.astype(np.float32),
    )

    return case_dir
