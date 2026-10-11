#!/usr/bin/env python3
"""Stress test: instructions we wrote ourselves on the dev scenes (experiments/stress.jsonl), with the
answer we expect: plan (and which objects may be picked first), ask_user or refuse. Object ids are the
pipeline's own (see the overlays in <run>.details/). Checks that the language step generalises
beyond the 50 dev sentences.

    python experiments/stress.py --data dataset [--file experiments/stress.jsonl]
"""
from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
from pipeline.llm import Apertus  # noqa: E402
from pipeline.perception import Perception  # noqa: E402
from pipeline.scene_graph import SceneGraph  # noqa: E402
from run import answer  # noqa: E402
from scene_io import load_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--file", type=Path, default=ROOT / "experiments" / "stress.jsonl")
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    ap.add_argument("--votes", type=int, default=4)
    args = ap.parse_args()
    cases = [json.loads(l) for l in args.file.read_text(encoding="utf-8").splitlines() if l.strip()]
    llm = Apertus(cache_dir=args.cache / "llm")
    see = Perception(llm, args.cache / "perception")
    graphs = {}
    for sc in sorted({c["scene"] for c in cases}):
        s = load_scene(args.data / sc)
        g = SceneGraph(see(s.rgb), s.rgb)
        graphs[sc] = (g, g.stacking_facts())

    def run(c):
        g, facts = graphs[c["scene"]]
        return answer(llm, g, facts, c["instruction"], args.votes)[0]
    with ThreadPoolExecutor(6) as ex:
        decs = list(ex.map(run, cases))
    ok = 0
    for c, d in zip(cases, decs):
        good = d.kind == c["expect"] and (d.kind != "plan" or d.target in c["first"])
        ok += good
        if not good:
            print(f"FAIL {c['scene']} '{c['instruction']}': expected {c['expect']} {c['first'] or ''}, got {d.kind} "
                  f"{d.target or d.message[:90]}\n      {d.notes}")
    print(f"\n{ok}/{len(cases)} as expected")


if __name__ == "__main__":
    main()
