"""drum2d.evaluate

Evaluation and parameter grid search.

Outputs per-case:
- report.png

Outputs for grid search:
- summary.csv
- summary.png

Metrics:
- MAE_theta
- RMSE_theta
- P95_abs_err

Optional extra indicators:
- theta_std
- surface_roughness_rms (from detector surface points residuals)
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Support both:
# - package run: python -m drum2d.main ...
# - script run inside drum2d/: python main.py ...
try:
    from .detect import DetectParams, detect_case
    from .utils import draw_circle_boundary, draw_line_kb, draw_polyline, read_json
except ImportError:  # pragma: no cover
    from detect import DetectParams, detect_case
    from utils import draw_circle_boundary, draw_line_kb, draw_polyline, read_json


def _metrics(theta_gt: np.ndarray, theta_hat: np.ndarray) -> Dict[str, float]:
    n = min(theta_gt.size, theta_hat.size)
    if n == 0:
        return {"MAE_theta": float("nan"), "RMSE_theta": float("nan"), "P95_abs_err": float("nan")}
    e = (theta_hat[:n] - theta_gt[:n]).astype(np.float32)
    ae = np.abs(e)
    mae = float(ae.mean())
    rmse = float(np.sqrt((e * e).mean()))
    p95 = float(np.quantile(ae, 0.95))
    return {"MAE_theta": mae, "RMSE_theta": rmse, "P95_abs_err": p95}


def _roughness_from_surface_points(surface_points: np.ndarray, k: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Compute RMS residual of surface points to fitted line per-frame and average."""
    if surface_points.size == 0:
        return None
    T = surface_points.shape[0]
    rms = []
    for i in range(T):
        pts = surface_points[i]
        if pts.size == 0:
            continue
        valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
        pts = pts[valid]
        if pts.shape[0] < 5:
            continue
        resid = pts[:, 1] - (k[i] * pts[:, 0] + b[i])
        rms.append(float(np.sqrt(np.mean(resid**2))))
    if not rms:
        return None
    return float(np.mean(rms))


def evaluate_case(case_dir: str | Path) -> Path:
    """Evaluate a case and save report.png."""
    case_dir = Path(case_dir)

    gt_path = case_dir / "gt.npz"
    det_path = case_dir / "detect.npz"
    meta_path = case_dir / "meta.json"
    video_path = case_dir / "video.mp4"

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing gt.npz in {case_dir}")
    if not det_path.exists():
        raise FileNotFoundError(f"Missing detect.npz in {case_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {case_dir}")

    meta = read_json(meta_path)
    cfg = meta.get("cfg", {})
    w = int(cfg.get("w", 512))
    h = int(cfg.get("h", 512))
    r = float(cfg.get("r", 220.0))
    cx = w / 2.0
    cy = h / 2.0

    gt = np.load(gt_path, allow_pickle=False)
    det = np.load(det_path, allow_pickle=False)

    theta_gt = gt["theta_gt"].astype(np.float32)
    surface_gt = gt["surface_xy"].astype(np.float32)

    theta_hat = det["theta_hat"].astype(np.float32)
    line_kb = det["line_kb"].astype(np.float32)
    surface_points = det.get("surface_points", np.zeros((theta_hat.size, 0, 2), np.float32)).astype(np.float32)

    m = _metrics(theta_gt, theta_hat)
    theta_std = float(np.std(theta_hat)) if theta_hat.size else float("nan")
    rough = _roughness_from_surface_points(surface_points, line_kb[:, 0], line_kb[:, 1])

    # pick a representative frame
    T = min(theta_gt.size, theta_hat.size, surface_gt.shape[0])
    idx = T // 2 if T > 0 else 0

    # read frame
    frame_bgr = None
    if video_path.exists() and T > 0:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            frame_bgr = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    if frame_bgr is None:
        frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    overlay = frame_bgr.copy()
    draw_circle_boundary(overlay, cx, cy, r, color=(0, 255, 255), thickness=2)
    if idx < surface_gt.shape[0]:
        draw_polyline(overlay, surface_gt[idx], color=(0, 0, 255), thickness=2)

    if idx < line_kb.shape[0]:
        x0 = int(max(0, cx - r))
        x1 = int(min(w - 1, cx + r))
        draw_line_kb(overlay, float(line_kb[idx, 0]), float(line_kb[idx, 1]), x0, x1, color=(0, 255, 0), thickness=2)

    cv2.putText(
        overlay,
        f"theta_gt={float(theta_gt[idx]):+.2f}  theta_hat={float(theta_hat[idx]):+.2f}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # matplotlib report
    fig = plt.figure(figsize=(12, 9), dpi=140)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_title("Overlay (GT surface in red, detected line in green)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    n = min(theta_gt.size, theta_hat.size)
    tt = np.arange(n)
    ax2.plot(tt, theta_gt[:n], label="theta_gt", linewidth=1.8)
    ax2.plot(tt, theta_hat[:n], label="theta_hat", linewidth=1.4)
    ax2.set_title("Theta (deg)")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("deg")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    err = (theta_hat[:n] - theta_gt[:n])
    ax3.plot(tt, np.abs(err), linewidth=1.4)
    ax3.set_title(f"Abs error |hat-gt|  MAE={m['MAE_theta']:.2f}  RMSE={m['RMSE_theta']:.2f}  P95={m['P95_abs_err']:.2f}")
    ax3.set_xlabel("frame")
    ax3.set_ylabel("deg")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    txt = [
        f"Regime: {meta.get('resolved', {}).get('regime', 'n/a')}",
        f"rpm={cfg.get('rpm', 'n/a')}  fill={cfg.get('fill', 'n/a')}",
        f"noise_sigma={cfg.get('noise_sigma', 'n/a')}  blur_sigma={cfg.get('blur_sigma', 'n/a')}  glare={cfg.get('glare_strength', 'n/a')}",
        f"theta_std (hat) = {theta_std:.3f}",
        f"surface_roughness_rms = {rough:.3f}" if rough is not None else "surface_roughness_rms = n/a",
        "Trend note: higher rpm (esp. cataracting) should increase theta_std/roughness.",
    ]
    ax4.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=10)
    ax4.set_title("Indicators")
    ax4.axis("off")

    fig.tight_layout()
    out_path = case_dir / "report.png"
    fig.savefig(out_path)
    plt.close(fig)

    return case_dir


def grid_search(
    root_out_dir: str | Path,
    cases: Sequence[str | Path],
    param_grid: Dict[str, Sequence[Any]],
    seed: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Scan detector parameters and summarize MAE across cases.

    Args:
        root_out_dir: output folder (for summary.csv/png).
        cases: list of case directories.
        param_grid: dict of parameter name -> list of values.

    Returns:
        best_params, rows
    """

    root_out_dir = Path(root_out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]

    rows: List[Dict[str, Any]] = []

    best = None
    best_score = float("inf")

    for combo in itertools.product(*values):
        p = DetectParams()
        for k, v in zip(keys, combo):
            if not hasattr(p, k):
                raise ValueError(f"DetectParams has no field: {k}")
            setattr(p, k, v)

        maes = []
        rmses = []
        p95s = []
        for case in cases:
            case = Path(case)
            detect_case(case, params=p, seed=seed, write_preview=False)
            gt = np.load(case / "gt.npz", allow_pickle=False)
            det = np.load(case / "detect.npz", allow_pickle=False)
            m = _metrics(gt["theta_gt"].astype(np.float32), det["theta_hat"].astype(np.float32))
            maes.append(m["MAE_theta"])
            rmses.append(m["RMSE_theta"])
            p95s.append(m["P95_abs_err"])

        mean_mae = float(np.mean(maes))
        mean_rmse = float(np.mean(rmses))
        mean_p95 = float(np.mean(p95s))

        row: Dict[str, Any] = {"mean_MAE_theta": mean_mae, "mean_RMSE_theta": mean_rmse, "mean_P95_abs_err": mean_p95}
        for k, v in zip(keys, combo):
            row[k] = v
        rows.append(row)

        if mean_mae < best_score:
            best_score = mean_mae
            best = row

    # write CSV
    csv_path = root_out_dir / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r0 in rows:
            wri.writerow(r0)

    # plot
    if rows:
        fig = plt.figure(figsize=(10, 5), dpi=140)
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(len(rows))
        y = np.array([r["mean_MAE_theta"] for r in rows], dtype=np.float32)
        ax.bar(x, y, color="#4C78A8")
        ax.set_title("Grid search: mean MAE_theta per param combo")
        ax.set_xlabel("combo index")
        ax.set_ylabel("deg")
        ax.grid(True, axis="y", alpha=0.25)
        if best is not None:
            bi = int(np.argmin(y))
            ax.text(bi, float(y[bi]), f" best={y[bi]:.2f}", ha="left", va="bottom")
        fig.tight_layout()
        fig.savefig(root_out_dir / "summary.png")
        plt.close(fig)

    best_params = {k: best[k] for k in keys} if best is not None else {}
    return best_params, rows
