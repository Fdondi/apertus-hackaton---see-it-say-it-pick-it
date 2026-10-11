#!/usr/bin/env python3
"""Experiment: Apertus alone (no vision helper), but with a better prompt, constrained decoding
and the decision separated from the pointing. The A/B point between the baseline and the full
pipeline; see README "Experiments".

    python experiments/apertus_only.py --data dataset --out results_apertus_only.json [--thinking]
"""
from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
from pipeline.llm import Apertus, image_part  # noqa: E402
from scene_io import load_scene  # noqa: E402

SYSTEM = """You are the language brain of a UR5 robot arm with a two-finger gripper. A fixed camera looks straight down at the workcell.
Facts about the workcell:
- The grey tray is the grey plastic box on the left of the photo. The black tray is the large black area on the right of the photo.
- Objects that can appear: three identical red gears, a yellow gear, a red cube, a mug, a mandarin, a banana, a plate, a saucer, two grey metal parts, a white part, a small white part. Sometimes a person's hand or arm.
- The gripper holds one object at a time and opens at most 85 mm. The plate and the saucer are too wide to pick. The trays are only places to put things; they are never picked.
- Left, right, top and bottom mean as seen in the photo."""

DECIDE = """Instruction: "{text}"

Decide which kind of answer is correct.
- "plan": the instruction is clear, safe and possible. Instructions about several objects ("all", "everything", "both", "except") are plans. So are instructions that need another object moved out of the way first.
- "ask_user": it is not clear which object is meant: the instruction names one object but several objects in the photo match it equally well (for example "the red gear" when there are several red gears and nothing tells them apart), or it points with words like "that one" or "this".
- "refuse": the instruction is impossible or unsafe:
  * an object it names is not in the photo;
  * it cannot work: an object does not fit, it is too wide for the gripper (plate, saucer), or the instruction contradicts itself or forbids a move that is needed;
  * it is unsafe: throwing, dropping, pushing something off, pulling something out from under another object, or taking something from a person's hand.

Answer in JSON. "why" is one short sentence. "message" is the question to the user (ask_user) or the reason (refuse), or "" for a plan."""

PLAN = """Instruction: "{text}"

The photo is {w} x {h} pixels. A pixel [u, v] is column u counted from the left and row v counted from the top.
Plan the robot actions as a list of pick and place steps, one object at a time. If another object lies on the object to move, move that object away first.
target_pixel is the centre of the first object you pick, in this photo's pixels.
Answer in JSON."""

DECIDE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["why", "decision", "message"],
                 "properties": {"why": {"type": "string", "maxLength": 300},
                                "decision": {"enum": ["plan", "ask_user", "refuse"]},
                                "message": {"type": "string", "maxLength": 200}}}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["first_object", "target_pixel", "actions"],
               "properties": {"first_object": {"type": "string", "maxLength": 60},
                              "target_pixel": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2},
                              "actions": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
                                  "type": "object", "additionalProperties": False, "required": ["action", "object"],
                                  "properties": {"action": {"enum": ["pick", "place"]}, "object": {"type": "string", "maxLength": 60}}}}}}


def answer(llm, img, w, h, scale, iid, text, thinking):
    head = [{"role": "system", "content": SYSTEM}]
    user = lambda t: [{"role": "user", "content": [img, {"type": "text", "text": t}]}]
    d = llm.json(head + user(DECIDE.format(text=text)), DECIDE_SCHEMA, max_tokens=2000 if thinking else 300, thinking=thinking)
    if d["decision"] != "plan":
        field = "question" if d["decision"] == "ask_user" else "reason"
        msg = d["message"].strip() or d["why"]
        return {"instruction_id": iid, "target_pixel": None, "actions": [{"action": d["decision"], field: msg}]}, d
    p = llm.json(head + user(PLAN.format(text=text, w=w, h=h)), PLAN_SCHEMA, max_tokens=2000 if thinking else 400, thinking=thinking)
    u, v = p["target_pixel"]
    px = [min(int(round(u / scale)), 2447), min(int(round(v / scale)), 2047)]
    acts = p["actions"] if p["actions"][0]["action"] == "pick" else [{"action": "pick", "object": p["first_object"]}] + p["actions"]
    return {"instruction_id": iid, "target_pixel": px, "actions": acts}, {**d, **p}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--out", type=Path, default=Path("results_apertus_only.json"))
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()
    llm = Apertus(args.base_url)
    results, log = [], []
    for d in sorted(p for p in args.data.iterdir() if (p / "rgb.png").exists()):
        sc = load_scene(d)
        img, scale = image_part(sc.rgb, args.max_side)
        w, h = round(sc.rgb.shape[1] * scale), round(sc.rgb.shape[0] * scale)
        with ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda x: answer(llm, img, w, h, scale, x[0], x[2], args.thinking), sc.instructions))
        for (iid, cat, text), (res, info) in zip(sc.instructions, outs):
            results.append(res)
            log.append({"instruction_id": iid, "instruction": text, **info})
            print(f"{iid} {res['actions'][0]['action']:8s} {text}")
    args.out.write_text(json.dumps({"team": "apertus-only", "results": results}, indent=2))
    args.out.with_suffix(".log.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    print(f"{len(results)} answers -> {args.out}  ({llm.calls} calls, {llm.cache_hits} cached)")


if __name__ == "__main__":
    main()
