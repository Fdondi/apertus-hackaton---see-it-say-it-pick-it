#!/usr/bin/env python3
"""See it, say it, pick it: one photo + one English instruction -> a safe robot action.

    docker compose up -d                                   # Apertus 1.5 8B on vLLM, port 8000
    python run.py --data dataset --out results.json        # any folder of scene_*/ (rgb.png + instructions.txt)
    python starter/evaluate.py results.json --data dataset

Pipeline (Apertus names the objects, reads the instruction and answers the safety questions; SAM 2.1 only proposes
regions; code resolves the command, applies the safety limits and, with --reading rules, repairs the reading):
  1. perception   SAM 2.1 proposes regions; Apertus names each one from a crop     pipeline/perception.py
  2. scene graph  trays, what lies on what, the hand, positions, distances          pipeline/scene_graph.py
  3. understand   Apertus turns the instruction into a structured command            pipeline/reason.py
  4. decide       resolve the command on the scene graph; safety rules; plan         pipeline/plan.py
Step 3 runs several times (one greedy reading, a few sampled ones) and the decisions are put to a vote.

--reading rules       (default, the original pipeline) keyword rules repair the reading (R1-R30 in plan.py), a
                      cue-word check filters the vote (V), and the safety verdicts need cue words in the sentence
                      (S1-S3 in plan.py, S4-S5 below).
--reading questions   no keyword rules: when the readings disagree or the command cannot be carried out (nothing
                      fits, several objects fit, no place to put it), Apertus answers one short question chosen
                      by the kind of problem (h3 below, questions in pipeline/ask.py); the safety verdicts are
                      confirmed by yes/no questions instead of cue words.
See docs/questions-instead-of-rules.md for the study behind the two.

Writes <out> (the answers) and <out stem>.details/ (scene graphs, overlays, per-instruction reasoning).
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "starter")]
from pipeline.llm import Apertus  # noqa: E402
from pipeline.perception import MM_PER_PX, Perception, draw, is_it, on_top, set_scale  # noqa: E402
import pipeline.plan as plan  # noqa: E402
from pipeline.plan import CARE_PHRASE, Decision, consistent, cue_in, decide  # noqa: E402
from pipeline.reason import YES_NO  # noqa: E402
CONFIRM = {"throw or toss": "throw or toss an object", "drop": "let an object fall or drop it from a height",
           "push, slide or knock": "push, slide or knock an object off or over something",
           "pull out from under": "pull an object out from under another one",
           "take from a person": "take an object out of a person's hand",
           "put on or near a person": "put an object on or right next to a person's hand or body",
           "move fast or swing": "move fast or swing an object around",
           "hold two objects at once": "hold two objects in its gripper at the same time"}


def confirm(llm, text: str, danger: str) -> bool:
    prompt = f'A person tells a robot arm: "{text}"\nIf the robot does exactly what is asked, will it {CONFIRM[danger]}? ' \
             "Answer yes or no."
    return llm.json([{"role": "user", "content": prompt}], YES_NO, max_tokens=10)["answer"] == "yes"
from pipeline.reason import danger_check, match_objects, parse, parse_samples  # noqa: E402
import pipeline.ask as A  # noqa: E402
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
    ctx = scene_text if cue_in("under", text) else ""
    readings = []
    try:
        readings.append(parse(llm, text, ctx))
    except ValueError:            # no valid reply after three tries: the sampled readings still vote
        pass
    if votes:
        readings += parse_samples(llm, text, ctx, n=votes)
    # "other danger" from the long reading is checked by the short safety question: the long prompt sees danger in
    # "don't bump the mug on the way"; the short one is told that a request to avoid something is not dangerous
    # S4: with --reading rules, only for sentences with a care phrase
    if any(r.get("other_danger") for r in readings) and (plan.QUESTIONS or CARE_PHRASE.search(text.lower())) \
            and danger_check(llm, text) == "none" \
            and not (plan.QUESTIONS and A.yes(llm, A.HEAD.format(text=text) + A.DANGER3)):
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
    if plan.QUESTIONS:            # follow-up questions to Apertus instead of rules (--reading questions)
        chosen = h3(llm, g, text, readings, decisions)
    if chosen.kind == "plan":     # a second, separate look at safety before a plan goes out
        danger = danger_check(llm, text)
        cue = DANGER_CUES.get(danger)
        words = text.lower().replace(",", " ").split()
        free = [w for k, w in enumerate(words)       # not right after without / don't / not / never / avoid
                if not {"without", "don't", "dont", "not", "never", "avoid", "no"} & set(words[max(0, k - 2):k])]
        if plan.QUESTIONS:            # no cue words: Apertus confirms its own verdict
            hit = danger != "none" and (cue is None or confirm(llm, text, danger))
        else:
            hit = danger != "none" and (cue is None or any(w.startswith(c) for c in cue.split() for w in free))  # S5
        if hit:
            chosen = Decision("refuse", f"This asks the robot to {danger}: that could hurt someone or make an object "
                              "fall.", notes=chosen.notes + [f"safety check: {danger}"])
    return chosen, readings, decisions


def why(d: Decision) -> str:
    """The kind of outcome, from the code's own messages: what a follow-up question has to resolve."""
    m = d.message
    if d.kind == "plan":
        return "any_one" if any(n.startswith("any one of them") for n in d.notes) else "plan"
    if m.startswith("There is no ") and m.endswith("in the workcell."):
        return "no_such"
    if m.startswith("I cannot find"):
        return "nothing_fits"
    if m.startswith("There are ") and "objects that fit" in m:
        return "several"
    if "to put it on or in" in m or "to put it next to" in m:
        return "dest_missing"
    if "Which one should it go on" in m:
        return "dest_several"
    if m.startswith("Which tray"):
        return "tray"
    if m.startswith("Which object do you mean? I cannot see"):
        return "vague"
    return d.kind


def sig(d: Decision) -> tuple:
    return (d.kind, tuple(a["object"] for a in d.actions)) if d.kind == "plan" else (d.kind, why(d))


def _with_target(r: dict, step: int, **kw) -> dict:
    r2 = copy.deepcopy(r)
    t = r2["steps"][step]["target"]
    t.update(kw)
    return r2


def _with_dest(r: dict, step: int, **kw) -> dict:
    r2 = copy.deepcopy(r)
    r2["steps"][step]["destination"].update(kw)
    return r2


def h3(llm: Apertus, g: SceneGraph, text: str, readings: list, decisions: list) -> Decision:
    """--reading questions (hypothesis H3 of docs/questions-instead-of-rules.md): the next question is chosen by the
    kind of ambiguity left after reading and resolving, its answer is applied, and the instruction is decided again."""
    notes = []
    groups = {}
    for k, d in enumerate(decisions):
        groups.setdefault(sig(d), []).append(k)
    order = sorted(groups.values(), key=lambda ix: (-len(ix), ix[0]))
    base = order[0][0]
    if len(groups) > 1:                       # the readings disagree: which reading says what the person asked?
        reps = [ix[0] for ix in order][:5]
        paras = [A.say_reading(readings[k]) for k in reps]
        if len(set(paras)) == len(paras):
            i = A.q_reading(llm, text, [readings[k] for k in reps])
            if i is not None:
                base = reps[i]
                notes.append(f"Apertus chose the reading: {paras[i]}")
            else:
                notes.append("Apertus could not choose between the readings: majority")
    r, d = readings[base], decisions[base]
    lab = A.labels(g)
    for _ in range(3):
        w, info = why(d), d.info
        step = info.get("step", 0)
        t0 = r["steps"][step]["target"] if step < len(r["steps"]) else None
        r2 = None
        if w in ("nothing_fits", "no_such") and t0 and t0["relation"] != "none":
            which = t0.get("which_words", "").strip() or A.REL.get(t0["relation"], t0["relation"]).format(
                ref=A._ref(t0.get("ref_kind", "none"), t0.get("ref_color", "any")), ref2="something", nth="nth")
            a = A.q_where_words(llm, text, which)
            notes.append(f"Apertus: '{which}' tells {a}")
            if a in ("goal", "neither"):
                r2 = _with_target(r, step, relation="none", ref_kind="none")
                dest = r2["steps"][step]["destination"]
                rk = t0.get("ref_kind", "none")
                if a == "goal" and dest["place"] in ("nowhere, just pick up", "to the person") and rk != "none":
                    if rk in ("grey tray", "black tray"):
                        r2 = _with_dest(r2, step, place=rk)
                    elif rk != "person's hand":
                        r2 = _with_dest(r2, step, place="on or in an object", kind=rk, color=t0.get("ref_color", "any"))
            elif t0["relation"] in ("between", "closest to", "next to", "farthest from") and t0.get("ref_kind", "none") in ("none",) \
                    or t0["relation"] == "between" and t0.get("ref2_kind", "none") in ("none", t0.get("ref_kind")):
                r2 = _ref_question(llm, g, lab, text, r, step, t0, notes)
        elif w in ("several", "any_one") and len(info.get("cands", info.get("ids", []))) > 1 and t0:
            cands = info.get("cands", info.get("ids"))
            n = A.q_count(llm, text, t0.get("words", ""), len(cands))
            notes.append(f"Apertus: '{t0.get('words')}' means {n}")
            if n == "all" and t0.get("how_many") != "all":
                r2 = _with_target(r, step, how_many="all")
            elif n == "one":
                pick = A.q_pick(llm, g, lab, text, t0.get("words", ""), cands)
                notes.append(f"Apertus: '{t0.get('words')}' is {pick}")
                r2 = _with_target(r, step, how_many="one", pin=[pick]) if pick else \
                    (_with_target(r, step, how_many="one") if w == "any_one" else None)
            elif n == "any one" and w == "several":
                r2 = _with_target(r, step, how_many="any one")
        elif w in ("dest_missing", "dest_several", "tray") and t0:
            what = "it" if not t0.get("words") else t0["words"]
            ans = A.q_dest(llm, g, lab, text, what, info.get("dids") if w == "dest_several" else None)
            notes.append(f"Apertus: it goes to {ans}")
            if ans and ans[0] == "tray":
                r2 = _with_dest(r, step, place=ans[1])
            elif ans and ans[0] == "obj":
                r2 = _with_dest(r, step, place="on or in an object", kind="object", color="any", pin=[ans[1]])
            elif ans and ans[0] == "person" and ans[1] == "body":
                return Decision("refuse", "Putting an object on or next to a person's hand or body means moving the gripper "
                                "onto a person.", notes=d.notes + notes)
            elif ans and ans[0] == "person" and ans[1] == "hand over":
                r2 = _with_dest(r, step, place="to the person")
            elif ans and ans[0] == "person" and ans[1] == "elsewhere":
                return Decision("refuse", "The place you named is not on this table, so I cannot put it there.",
                                notes=d.notes + notes)
        if r2 is None:
            break
        try:
            d2 = decide(g, r2, text)
        except (KeyError, TypeError, ValueError, IndexError):
            break
        if sig(d2) == sig(d) and d2.message == d.message and d2.info.get("step") == d.info.get("step"):
            d = Decision(d2.kind, d2.message, d2.actions, d2.target, d2.notes, d2.info)
            r = r2
            break
        r, d = r2, d2
    return Decision(d.kind, d.message, d.actions, d.target, d.notes + notes, d.info)


EDGES = {"the top edge of the photo": "topmost", "the bottom edge of the photo": "bottommost",
         "the left edge of the photo": "leftmost", "the right edge of the photo": "rightmost"}


def _ref_question(llm, g, lab, text, r, step, t0, notes):
    """A relation that needs other objects but names none: measured from what?"""
    kind_name = A._thing(t0.get("kind", "object"), t0.get("color", "any"))
    others = f"the other {kind_name}s"
    opts = [others] + list(EDGES) + [f"the {lab[i]}" for i in sorted(lab, key=lambda i: g.centre(i)[0])]
    which = t0.get("which_words", "") or t0["relation"]
    i = A.choose(llm, A.HEAD.format(text=text) + "Objects on the table:\n" + A.table(g, lab) +
                 f'\nIn "{which}", what is the {kind_name} compared with?', opts)
    notes.append(f"Apertus: '{which}' is measured from {opts[i] if i is not None else None}")
    if i is None:
        return None
    if i == 0:
        return _with_target(r, step, ref_kind=t0.get("kind"), ref_color=t0.get("color", "any"),
                            ref2_kind=t0.get("kind"), ref2_color=t0.get("color", "any"))
    if i <= len(EDGES):
        return _with_target(r, step, relation=list(EDGES.values())[i - 1], ref_kind="none")
    oid = sorted(lab, key=lambda i: g.centre(i)[0])[i - 1 - len(EDGES)]
    return _with_target(r, step, ref_kind="object", ref_color="any", ref_pin=[oid])


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
    ap.add_argument("--reading", choices=["rules", "questions"], default="rules",
                    help="rules: keyword rules repair Apertus's reading (the original pipeline); questions: no rules, "
                         "Apertus answers short follow-up questions instead (see README)")
    args = ap.parse_args()
    plan.RULES, plan.QUESTIONS = args.reading == "rules", args.reading == "questions"

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
        g.yesno = lambda prompt: llm.json([{"role": "user", "content": prompt}], YES_NO, max_tokens=10)["answer"] == "yes"
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
