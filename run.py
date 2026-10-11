#!/usr/bin/env python3
"""See it, say it, pick it: one photo + one English instruction -> a safe robot action.

    docker compose up -d                                   # Apertus 1.5 8B on vLLM, port 8000
    python run.py --data dataset --out results.json        # any folder of scene_*/ (rgb.png + instructions.txt)
    python starter/evaluate.py results.json --data dataset

Pipeline (Apertus does every language step; SAM 2.1 only proposes regions):
  1. perception   SAM 2.1 proposes regions; Apertus names each one from a crop     pipeline/perception.py
  2. scene graph  trays, what lies on what, the hand, positions, distances          pipeline/scene_graph.py
  3. understand   Apertus turns the instruction into a structured command            pipeline/reason.py
  4. decide       resolve the command on the scene graph; safety rules; plan         pipeline/plan.py
Step 3 runs several times (one greedy reading, a few sampled ones) and the decisions are put to a vote.

Writes <out> (the answers) and <out stem>.details/ (scene graphs, overlays, per-instruction reasoning).
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
from pipeline.llm import Apertus  # noqa: E402
from pipeline.perception import MM_PER_PX, Perception, draw, is_it, on_top, set_scale  # noqa: E402
from pipeline.plan import CARE_PHRASE, Decision, consistent, decide, said  # noqa: E402
from pipeline.reason import danger_check, match_objects, parse, parse_samples  # noqa: E402
from pipeline.scene_graph import SceneGraph  # noqa: E402
from scene_io import load_scene, pixel_to_table  # noqa: E402


def table_scale(sc) -> float | None:
    """mm per pixel on the table at the photo's centre, from the scene's calibration (None without one)."""
    try:
        h, w = sc.rgb.shape[:2]
        a, b = pixel_to_table(sc, w // 2, h // 2), pixel_to_table(sc, w // 2 + 100, h // 2)
        return float(((b - a) ** 2).sum() ** 0.5) / 100
    except Exception:
        return None


def answer(llm: Apertus, g: SceneGraph, scene_text: str, text: str, votes: int) -> tuple[Decision, list[dict], list[Decision]]:
    # "put the red gear under the yellow gear in the grey tray": which gear, or where it goes? Only the scene can
    # tell, so the parser gets the stacking facts, but only for sentences with "under" (elsewhere they distract it)
    ctx = scene_text if said("under", text) else ""
    readings = []
    try:
        readings.append(parse(llm, text, ctx))
    except ValueError:            # no valid reply after three tries: the sampled readings still vote
        pass
    if votes:
        readings += parse_samples(llm, text, ctx, n=votes)
    # "other danger" from the long reading is checked by the short safety question: the long prompt sees danger in
    # "don't bump the mug on the way"; the short one is told that a request to avoid something is not dangerous
    if any(r.get("other_danger") for r in readings) and CARE_PHRASE.search(text.lower()) \
            and danger_check(llm, text) == "none":
        readings = [{**r, "other_danger": False} for r in readings]
    decisions = []
    for r in readings:
        try:
            decisions.append(decide(g, r, text))
        except (KeyError, TypeError, ValueError, IndexError) as e:     # a reading the rules cannot use
            decisions.append(Decision("ask_user", RETRY, notes=[f"reading not usable: {e!r}"]))
    if not decisions:
        return Decision("ask_user", RETRY, notes=["no reading"]), [], []
    # vote among the readings whose relations are all said in the sentence (the others misread it), if any
    ok = [d for d, r in zip(decisions, readings) if consistent(r, text)] or decisions
    tally = Counter((d.kind, d.target) for d in ok)
    best, count = tally.most_common(1)[0]
    if count == tally[(ok[0].kind, ok[0].target)]:      # a tie keeps the first (greedy if it is consistent)
        best = (ok[0].kind, ok[0].target)
    chosen = next(d for d in ok if (d.kind, d.target) == best)
    if chosen.kind == "plan":     # a second, separate look at safety before a plan goes out
        danger = danger_check(llm, text)
        cue = DANGER_CUES.get(danger)
        words = text.lower().replace(",", " ").split()
        free = [w for k, w in enumerate(words)       # not right after without / don't / not / never / avoid
                if not {"without", "don't", "dont", "not", "never", "avoid", "no"} & set(words[max(0, k - 2):k])]
        if danger != "none" and (cue is None or any(w.startswith(c) for c in cue.split() for w in free)):
            chosen = Decision("refuse", f"This asks the robot to {danger}: that could hurt someone or make an object "
                              "fall.", notes=chosen.notes + [f"safety check: {danger}"])
    return chosen, readings, decisions


RETRY = "I could not understand the instruction. Could you say it another way?"
# the dangerous verb must be in the sentence, or the second look is ignored (it can be over-cautious)
DANGER_CUES = {"throw or toss": "throw toss fling hurl", "drop": "drop let fall", "push, slide or knock": "push slide knock shove nudge",
               "pull out from under": "pull yank tug", "take from a person": "hand fingers my from", "put on or near a person":
               "hand head arm body me my", "move fast or swing": "fast quick swing wave rapid", "hold two objects at once":
               "same time together once simultaneous", "other danger": None}


def to_result(iid: str, d: Decision, g: SceneGraph) -> dict:
    if d.kind == "plan":
        acts = [dict(a, pixel=list(g.nodes[a["object"]].obj.grasp_pixel)) if a["action"] == "pick" else a for a in d.actions]
        return {"instruction_id": iid, "target_pixel": list(g.nodes[d.target].obj.grasp_pixel), "actions": acts}
    field = "question" if d.kind == "ask_user" else "reason"
    return {"instruction_id": iid, "target_pixel": None, "actions": [{"action": d.kind, field: d.message}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"), help="folder of scene folders")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--team", default="see-say-pick")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible server running Apertus")
    ap.add_argument("--model", help="model name on the server (default: the first one it serves)")
    ap.add_argument("--cache", type=Path, default=Path(".cache"), help="cache for SAM masks and Apertus replies")
    ap.add_argument("--votes", type=int, default=4, help="sampled readings of each instruction besides the greedy one")
    ap.add_argument("--scenes", nargs="*", help="only these scene folders")
    args = ap.parse_args()

    llm = Apertus(args.base_url, args.model, cache_dir=args.cache / "llm")
    see = Perception(llm, args.cache / "perception")
    details = args.out.parent / (args.out.stem + ".details")
    details.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / "rgb.png").exists()
                    and (not args.scenes or p.name in args.scenes))
    results, log, t0 = [], [], time.time()
    for d in scenes:
        sc = load_scene(d)
        t = time.time()
        scale = table_scale(sc)       # the dev camera gives 0.248: kept at 0.25; another camera setup changes it
        set_scale(scale if scale and abs(scale - 0.25) > 0.05 * 0.25 else MM_PER_PX)
        try:
            objs = see(sc.rgb)
        except ImportError:        # a missing library is a setup error: stop and say so
            raise
        except Exception as e:     # one bad photo must not stop the run: no objects, every answer says why
            print(f"{d.name}: perception failed ({e!r}); answering without objects")
            objs = []
        g = SceneGraph(objs, sc.rgb)
        g.ask_stacking(lambda a, b: on_top(llm, sc.rgb, a, b))
        g.matcher = lambda phrase, text, names: match_objects(llm, phrase, text, names)
        g.reader = lambda text: parse(llm, text)
        g.looker = lambda i, noun, g=g: is_it(llm, g.nodes[i].obj, noun)
        scene_text = g.stacking_facts()
        draw(sc.rgb, objs, details / f"{d.name}.jpg")
        (details / f"{d.name}.scene.txt").write_text(g.describe() + "\n\nperception notes:\n" + "\n".join(see.log) + "\n",
                                                     encoding="utf-8")
        with ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda x: answer(llm, g, scene_text, x[2], args.votes), sc.instructions))
        for (iid, cat, text), (dec, readings, decs) in zip(sc.instructions, outs):
            res = to_result(iid, dec, g)
            results.append(res)
            log.append({"instruction_id": iid, "instruction": text, "decision": dec.kind, "target": dec.target,
                        "message": dec.message, "actions": dec.actions, "notes": dec.notes,
                        "votes": [f"{x.kind}:{x.target}" for x in decs], "reading": readings[0] if readings else None})
            shown = dec.target if dec.kind == "plan" else dec.message[:70]
            print(f"  {iid}  {dec.kind:8s} {shown}   <- {text}")
        print(f"{d.name}: {len(objs)} objects ({', '.join(o.id for o in objs)}) in {time.time() - t:.0f}s")
    args.out.write_text(json.dumps({"team": args.team, "results": results}, indent=2), encoding="utf-8")
    (details / "answers.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in log) + "\n", encoding="utf-8")
    print(f"\n{len(results)} answers -> {args.out}   details -> {details}/   "
          f"({time.time() - t0:.0f}s, {llm.calls} Apertus calls, {llm.cache_hits} cached)")


if __name__ == "__main__":
    main()
