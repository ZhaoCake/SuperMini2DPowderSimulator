"""drum2d.main

One-click CLI entry.

You can run either:
- From repo root (recommended):
    - python -m drum2d.main simulate --out outputs/case_001 --seed 0 --seconds 8 --fps 30 --rpm 25 --fill 0.55 --noise 8 --blur 1.0 --glare 0.15 --regime auto
    - python -m drum2d.main detect --case outputs/case_001
    - python -m drum2d.main eval --case outputs/case_001
    - python -m drum2d.main demo --out outputs/demo_run --n_cases 12 --seconds 6

- Or inside drum2d/:
    - python main.py simulate ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Support both:
# - package run: python -m drum2d.main ...
# - script run inside drum2d/: python main.py ...
try:
    from .detect import DetectParams, detect_case
    from .evaluate import evaluate_case, grid_search
    from .simulate import SimConfig, simulate_case
    from .utils import ensure_dir
except ImportError:  # pragma: no cover
    from detect import DetectParams, detect_case
    from evaluate import evaluate_case, grid_search
    from simulate import SimConfig, simulate_case
    from utils import ensure_dir


def _add_sim_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, help="Output directory for case")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--rpm", type=float, default=25.0)
    p.add_argument("--fill", type=float, default=0.55)
    p.add_argument("--noise", type=float, default=8.0, dest="noise_sigma")
    p.add_argument("--blur", type=float, default=1.0, dest="blur_sigma")
    p.add_argument("--glare", type=float, default=0.15, dest="glare_strength")
    p.add_argument("--regime", type=str, default="auto")


def _add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--case", required=True, help="Case directory")
    p.add_argument("--blur_ksize", type=int, default=7)
    p.add_argument("--thresh_method", type=str, default="otsu", choices=["otsu", "adaptive"])
    p.add_argument("--ransac_resid_thresh", type=float, default=2.8)
    p.add_argument("--ema_alpha", type=float, default=0.25)


def _add_demo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, help="Output directory for demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_cases", type=int, default=12)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=30)


def cmd_simulate(args: argparse.Namespace) -> None:
    cfg = SimConfig(
        seconds=float(args.seconds),
        fps=int(args.fps),
        rpm=float(args.rpm),
        fill=float(args.fill),
        noise_sigma=float(args.noise_sigma),
        blur_sigma=float(args.blur_sigma),
        glare_strength=float(args.glare_strength),
        regime=str(args.regime),
    )
    simulate_case(args.out, cfg=cfg, seed=int(args.seed))
    print(f"[simulate] wrote: {args.out}")


def cmd_detect(args: argparse.Namespace) -> None:
    params = DetectParams(
        blur_ksize=int(args.blur_ksize),
        thresh_method=str(args.thresh_method),
        ransac_resid_thresh=float(args.ransac_resid_thresh),
        ema_alpha=float(args.ema_alpha),
    )
    detect_case(args.case, params=params, seed=0, write_preview=True)
    print(f"[detect] wrote: {Path(args.case) / 'detect.npz'}")


def cmd_eval(args: argparse.Namespace) -> None:
    evaluate_case(args.case)
    print(f"[eval] wrote: {Path(args.case) / 'report.png'}")


def cmd_demo(args: argparse.Namespace) -> None:
    out = ensure_dir(args.out)
    rng = np.random.default_rng(int(args.seed))

    cases: List[Path] = []

    # 1) sample cases
    for i in range(int(args.n_cases)):
        rpm = float(rng.uniform(8, 45))
        fill = float(rng.uniform(0.30, 0.75))
        noise_sigma = float(rng.uniform(3, 14))
        blur_sigma = float(rng.uniform(0.6, 1.6))
        glare_strength = float(rng.uniform(0.0, 0.22))
        regime = "auto"

        cfg = SimConfig(
            seconds=float(args.seconds),
            fps=int(args.fps),
            rpm=rpm,
            fill=fill,
            noise_sigma=noise_sigma,
            blur_sigma=blur_sigma,
            glare_strength=glare_strength,
            regime=regime,
        )

        case_dir = out / f"case_{i:03d}"
        simulate_case(case_dir, cfg=cfg, seed=int(args.seed) + i)
        cases.append(case_dir)

    # 2) detect + eval per case with a default parameter set
    base_params = DetectParams()
    for case_dir in cases:
        detect_case(case_dir, params=base_params, seed=0, write_preview=True)
        evaluate_case(case_dir)

    # 3) small grid search
    param_grid: Dict[str, List[Any]] = {
        "blur_ksize": [5, 7, 9],
        "thresh_method": ["otsu", "adaptive"],
        "ransac_resid_thresh": [2.0, 2.8, 3.6],
        "ema_alpha": [0.15, 0.25, 0.35],
    }

    best_params, _rows = grid_search(out, cases=cases, param_grid=param_grid, seed=0)

    # save best params to json for convenience
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[demo] done")
    print(f"[demo] summary: {(out / 'summary.png').as_posix()}")
    print(f"[demo] best params: {best_params}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="drum2d")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("simulate", help="simulate one case")
    _add_sim_args(p1)
    p1.set_defaults(func=cmd_simulate)

    p2 = sub.add_parser("detect", help="detect surface/angle for a case")
    _add_detect_args(p2)
    p2.set_defaults(func=cmd_detect)

    p3 = sub.add_parser("eval", help="evaluate a case")
    p3.add_argument("--case", required=True)
    p3.set_defaults(func=cmd_eval)

    p4 = sub.add_parser("demo", help="batch demo + grid search")
    _add_demo_args(p4)
    p4.set_defaults(func=cmd_demo)

    return ap


def main(argv: List[str] | None = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
