#!/usr/bin/env python3
"""Robustness of perception: run it on altered copies of dev photos (darker, brighter, warmer light,
blur, JPEG artefacts, camera shifted by a few cm) and compare the objects found with the original.

    python experiments/augment_check.py --data dataset --scenes scene_05 scene_09
"""
from __future__ import annotations

import argparse, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
from pipeline.llm import Apertus  # noqa: E402
from pipeline.perception import Perception  # noqa: E402
from scene_io import load_scene  # noqa: E402


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


AUGS = {
    "darker": lambda im: np.clip(im.astype(np.float32) * 0.75, 0, 255).astype(np.uint8),
    "brighter": lambda im: np.clip(im.astype(np.float32) * 1.2 + 10, 0, 255).astype(np.uint8),
    "warm light": lambda im: np.clip(im.astype(np.float32) * np.array([1.1, 1.0, 0.85]), 0, 255).astype(np.uint8),
    "blur": lambda im: cv2.GaussianBlur(im, (0, 0), 2.0),
    "jpeg 50": lambda im: cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                                                                 [cv2.IMWRITE_JPEG_QUALITY, 50])[1], 1), cv2.COLOR_BGR2RGB),
    "shift 60px": lambda im: shift(im, 60, -40),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--scenes", nargs="+", default=["scene_05", "scene_09", "scene_10"])
    ap.add_argument("--augs", nargs="*", default=list(AUGS))
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    args = ap.parse_args()
    llm = Apertus(cache_dir=args.cache / "llm")
    see = Perception(llm, args.cache / "perception")
    same = total = 0
    for sc in args.scenes:
        rgb = load_scene(args.data / sc).rgb
        ref = Counter(o.cls for o in see(rgb))
        print(f"{sc}: {dict(ref)}")
        for name in args.augs:
            got = Counter(o.cls for o in see(AUGS[name](rgb)))
            total += 1
            same += got == ref
            diff = {k: got[k] - ref[k] for k in set(got) | set(ref) if got[k] != ref[k]}
            print(f"   {name:11s} {'same' if got == ref else 'DIFFERENT ' + str(diff)}")
    print(f"\nsame objects as the original photo: {same}/{total}")


if __name__ == "__main__":
    main()
