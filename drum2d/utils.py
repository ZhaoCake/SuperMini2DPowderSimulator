"""drum2d.utils

Small utilities shared across simulator/detector/evaluator.

Only depends on: numpy, opencv-python, scipy, matplotlib, tqdm (and stdlib).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def ensure_dir(path: os.PathLike | str) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: os.PathLike | str, obj: Dict[str, Any]) -> None:
    """Write json with UTF-8 and pretty formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: os.PathLike | str) -> Dict[str, Any]:
    """Read json file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int) -> np.random.Generator:
    """Set a reproducible RNG and return a Generator."""
    # numpy Generator is the main source of randomness in this project
    return np.random.default_rng(int(seed))


def circle_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Return boolean mask for pixels inside a circle."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def circle_y_bounds_for_x(xs: np.ndarray, cx: float, cy: float, r: float) -> Tuple[np.ndarray, np.ndarray]:
    """For each x, return y_top/y_bottom circle boundary (float)."""
    dx2 = (xs - cx) ** 2
    inside = r**2 - dx2
    inside = np.maximum(inside, 0.0)
    dy = np.sqrt(inside)
    return cy - dy, cy + dy


def clamp(a: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, a)))


def deg2rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return float(rad) * 180.0 / math.pi


def line_from_theta(cx: float, cy: float, theta_deg: float, y0: Optional[float] = None) -> Tuple[float, float]:
    """Return (k,b) for a line y = kx + b given angle theta_deg.

    theta is the angle between the line and +x axis in image coordinates (y down).
    y0 if provided means the line passes through (cx, y0), else through (cx, cy).
    """
    k = math.tan(deg2rad(theta_deg))
    y_anchor = cy if y0 is None else float(y0)
    b = y_anchor - k * cx
    return float(k), float(b)


def ransac_line_fit(
    points_xy: np.ndarray,
    resid_thresh: float,
    rng: np.random.Generator,
    n_iters: int = 80,
    min_inliers: int = 30,
) -> Tuple[float, float, np.ndarray]:
    """Fit y=kx+b using simple RANSAC on 2D points.

    Returns (k,b,inlier_mask).
    Residual is vertical distance |y - (kx+b)|.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must be (N,2)")

    n = points_xy.shape[0]
    if n < 2:
        return 0.0, float(points_xy[:, 1].mean()) if n == 1 else 0.0, np.zeros((n,), dtype=bool)

    xs = points_xy[:, 0]
    ys = points_xy[:, 1]

    best_inliers = None
    best_count = -1
    best_kb = (0.0, float(np.median(ys)))

    for _ in range(int(n_iters)):
        i1, i2 = rng.integers(0, n, size=2)
        if i1 == i2:
            continue
        x1, y1 = float(xs[i1]), float(ys[i1])
        x2, y2 = float(xs[i2]), float(ys[i2])
        if abs(x2 - x1) < 1e-6:
            continue
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        resid = np.abs(ys - (k * xs + b))
        inliers = resid <= float(resid_thresh)
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_kb = (float(k), float(b))

    if best_inliers is None or best_count < int(min_inliers):
        # fallback: plain least squares on all points
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        kb, *_ = np.linalg.lstsq(A, ys, rcond=None)
        k, b = float(kb[0]), float(kb[1])
        resid = np.abs(ys - (k * xs + b))
        inliers = resid <= float(resid_thresh)
        return k, b, inliers

    # refine using least squares on inliers
    xin = xs[best_inliers]
    yin = ys[best_inliers]
    A = np.stack([xin, np.ones_like(xin)], axis=1)
    kb, *_ = np.linalg.lstsq(A, yin, rcond=None)
    k, b = float(kb[0]), float(kb[1])
    return k, b, best_inliers


def ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average filter for 1D array."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    alpha = float(alpha)
    alpha = clamp(alpha, 0.0, 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.shape[0]):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def draw_circle_boundary(img_bgr: np.ndarray, cx: float, cy: float, r: float, color=(0, 255, 255), thickness: int = 2) -> None:
    cv2.circle(img_bgr, (int(round(cx)), int(round(cy))), int(round(r)), color, int(thickness), lineType=cv2.LINE_AA)


def draw_polyline(img_bgr: np.ndarray, pts_xy: np.ndarray, color=(0, 0, 255), thickness: int = 2) -> None:
    if pts_xy.size == 0:
        return
    pts = np.round(pts_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], isClosed=False, color=color, thickness=int(thickness), lineType=cv2.LINE_AA)


def draw_line_kb(img_bgr: np.ndarray, k: float, b: float, x0: int, x1: int, color=(0, 255, 0), thickness: int = 2) -> None:
    y0 = k * x0 + b
    y1 = k * x1 + b
    cv2.line(
        img_bgr,
        (int(x0), int(round(y0))),
        (int(x1), int(round(y1))),
        color,
        int(thickness),
        lineType=cv2.LINE_AA,
    )
