#!/usr/bin/env python3
"""Dev check of the perception step against the ground-truth outlines (never used by the pipeline).

    python experiments/check_perception.py --data dataset --out perception_check/

For every ground-truth object: is there a detected object of the right class whose grasp pixel
lies inside the true outline? Also writes one overlay per scene to look at.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
from pipeline.llm import Apertus  # noqa: E402
from pipeline.perception import Perception, draw  # noqa: E402
from scene_io import load_scene  # noqa: E402

GT_CLASS = {"red_cube": "red cube", "yellow_gear": "yellow gear", "mug": "mug", "mandarin": "mandarin",
            "banana": "banana", "white_part": "white part", "small_part": "small white part"}


def gt_class(name: str) -> str:
    return GT_CLASS.get(name) or ("red gear" if name.startswith("red_gear") else
                                  "grey metal part" if name.startswith("grey_part") else name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("perception_check"))
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    llm = Apertus(cache_dir=args.cache / "llm")
    per = Perception(llm, args.cache / "perception")
    hit = tot = 0
    for d in sorted(p for p in args.data.iterdir() if (p / "rgb.png").exists()):
        if args.scenes and d.name not in args.scenes:
            continue
        sc = load_scene(d)
        t = time.time()
        objs = per(sc.rgb)
        draw(sc.rgb, objs, args.out / f"{d.name}.jpg")
        gt_dir = args.data / "ground_truth" / "masks"
        lab = cv2.imread(str(gt_dir / f"{d.name}.png"), cv2.IMREAD_UNCHANGED)
        names = json.loads((gt_dir / f"{d.name}.json").read_text())["objects"]
        miss = []
        for k, n in names.items():
            tot += 1
            ok = any(o.cls == gt_class(n) and lab[o.grasp_pixel[1], o.grasp_pixel[0]] == int(k) for o in objs)
            hit += ok
            miss += [] if ok else [n]
        print(f"{d.name} {time.time() - t:5.1f}s  missed={miss}  found: " + ", ".join(
            o.id + (f"(on {'+'.join(o.on)})" if o.on else "") for o in objs))
        for line in per.log:
            print("      ", line)
    print(f"ground-truth objects found with the right class and a pixel on them: {hit}/{tot}")


if __name__ == "__main__":
    main()
