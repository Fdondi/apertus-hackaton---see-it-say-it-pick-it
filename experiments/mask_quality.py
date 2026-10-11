#!/usr/bin/env python3
"""Dev check of the object outlines against the ground-truth masks (never used by the pipeline).

    python experiments/mask_quality.py --data dataset [--crops out/]

For every ground-truth object: the detected object that overlaps it most, its class and the IoU of the two
outlines. An outline that misses the mug's handle or follows only its inside shows up as a low IoU.
With --crops, writes one crop per object with the true outline (green) and the detected one (magenta).
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter"), str(ROOT / "experiments")]
from check_perception import gt_class  # noqa: E402
from pipeline.llm import Apertus  # noqa: E402
from pipeline.perception import Perception  # noqa: E402
from scene_io import load_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    ap.add_argument("--crops", type=Path)
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()
    llm = Apertus(cache_dir=args.cache / "llm")
    per = Perception(llm, args.cache / "perception")
    if args.crops:
        args.crops.mkdir(parents=True, exist_ok=True)
    by_cls = defaultdict(list)
    for d in sorted(p for p in args.data.iterdir() if (p / "rgb.png").exists()):
        if args.scenes and d.name not in args.scenes:
            continue
        sc = load_scene(d)
        objs = per(sc.rgb)
        gt_dir = args.data / "ground_truth" / "masks"
        lab = cv2.imread(str(gt_dir / f"{d.name}.png"), cv2.IMREAD_UNCHANGED)
        names = json.loads((gt_dir / f"{d.name}.json").read_text())["objects"]
        line = []
        for k, n in names.items():
            gt = lab == int(k)
            best, iou = None, 0.0
            for o in objs:
                inter = int((o.mask & gt).sum())
                if inter:
                    j = inter / int((o.mask | gt).sum())
                    if j > iou:
                        best, iou = o, j
            ok = best is not None and best.cls == gt_class(n)
            by_cls[gt_class(n)].append(iou if ok else 0.0)
            line.append(f"{n}={iou:.2f}" + ("" if ok else f"[{best.cls if best else 'none'}]"))
            if args.crops and best is not None:
                ys, xs = np.nonzero(gt | best.mask)
                x0, y0 = max(0, xs.min() - 40), max(0, ys.min() - 40)
                crop = cv2.cvtColor(sc.rgb[y0: ys.max() + 40, x0: xs.max() + 40], cv2.COLOR_RGB2BGR).copy()
                for m, col in ((gt, (0, 200, 0)), (best.mask, (255, 0, 255))):
                    cs, _ = cv2.findContours(m[y0: ys.max() + 40, x0: xs.max() + 40].astype(np.uint8),
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(crop, cs, -1, col, 3)
                cv2.imwrite(str(args.crops / f"{d.name}_{n}_{iou:.2f}.jpg"), crop)
        print(f"{d.name}: " + "  ".join(line))
    print("\nmean IoU by class (0 when the class is wrong):")
    for c, v in sorted(by_cls.items()):
        print(f"  {c:18s} {np.mean(v):.2f}  min {np.min(v):.2f}  n={len(v)}")
    allv = [x for v in by_cls.values() for x in v]
    print(f"  {'all':18s} {np.mean(allv):.3f}")


if __name__ == "__main__":
    main()
