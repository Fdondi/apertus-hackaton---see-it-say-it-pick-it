"""Decide and plan: resolve Apertus's structured command on the scene graph, then apply the safety
rules and the robot's limits, and order the picks.

The rules come from docs/safety_primer.md and the challenge:
- refuse: throwing, dropping, pushing off, pulling out from under, taking from a person's hand,
  anything else Apertus flags as dangerous; an object that is not there; an object too wide for
  the 85 mm gripper (plate, saucer); an object that does not fit where it should go; an instruction
  that forbids a move it needs (or contradicts itself)
- ask: the object is only pointed at ("that one"), or the words fit several objects equally
- otherwise plan: one object at a time, whatever lies on an object is moved away first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .perception import GRASPABLE
from .scene_graph import COLORS, SceneGraph

KIND_CLASSES = {"gear": {"red gear", "yellow gear"}, "cube": {"red cube"}, "mug": {"mug"}, "mandarin": {"mandarin"},
                "banana": {"banana"}, "fruit": {"mandarin", "banana"}, "plate": {"plate"}, "saucer": {"saucer"},
                "grey metal part": {"grey metal part"}, "white part": {"white part"},
                "small white part": {"small white part"},
                "part": {"grey metal part", "white part", "small white part"},
                "object": set(GRASPABLE), "something else": set()}
UNSAFE = {"throw or toss": "Throwing {it} is uncontrolled motion: it could hit someone or break.",
          "drop": "Dropping {it} is uncontrolled motion: it could fall on someone or break.",
          "push or slide": "Pushing {it} would make it slide or fall in an uncontrolled way.",
          "pull out from under": "Pulling {it} out from under the object on top would make that object fall.",
          "take from the person": "Taking {it} out of a person's hand means moving the gripper onto a person."}
CONTAINERS = {"mug": 75.0}          # inner opening, mm: wider than this does not go in, longer than 1.6x sticks out
NEEDS_REF = {"to the left of", "to the right of", "above", "below", "closest to", "farthest from", "next to", "lying on",
             "under", "in", "in the person's hand"}

# Checks on Apertus's reading against the sentence itself. Apertus sometimes copies a relation from the scene facts
# into a sentence that never says it ("pick up the red gear" read as "the red gear under the yellow gear"), which
# would hide an ambiguity; so a relation needs one of its cue words in the instruction, or it is dropped. The same
# goes for the noun of the object ("the yellow fruit" read as "a yellow mandarin").
CUES = {"to the left of": "left", "to the right of": "right", "above": "above top upper higher",
        "below": "below bottom lower beneath", "closest to": "close near next beside by adjacent",
        "farthest from": "far", "next to": "next near beside by close adjacent", "lying on": "on onto top",
        "under": "under beneath underneath has have with lying resting sitting", "in": "in inside from within", "with nothing on it":
        "nothing free empty uncovered clear", "leftmost": "left", "rightmost": "right", "topmost": "top upper highest",
        "bottommost": "bottom lower lowest", "nth from the left": "left", "nth from the right": "right",
        "nth from the top": "top", "nth from the bottom": "bottom", "biggest": "big large", "smallest": "small little tiny",
        "in the middle": "middle centre center central", "between": "between",
        "in the person's hand": "hand holding held fingers my"}
KIND_WORDS = {"gear": "gear cog", "cube": "cube block box", "mug": "mug cup", "mandarin": "mandarin orange tangerine clementine",
              "banana": "banana", "fruit": "fruit", "plate": "plate dish", "saucer": "saucer", "grey metal part": "part bar metal",
              "white part": "part", "small white part": "part", "part": "part", "object": "", "something else": ""}
STACK_WORDS = {"under", "beneath", "underneath"}
SUPERLATIVES = {"biggest": "biggest", "largest": "biggest", "smallest": "smallest", "leftmost": "leftmost",
                "rightmost": "rightmost", "lowest": "bottommost", "bottommost": "bottommost", "highest": "topmost",
                "topmost": "topmost", "uppermost": "topmost"}


def _words(text: str) -> set[str]:
    return set(text.lower().replace(",", " ").replace(".", " ").split())


def said(rel: str, text: str) -> bool:
    """The instruction contains a cue word for this relation (relations without cues always pass)."""
    words, cues = _words(text), CUES.get(rel)
    return cues is None or any(any(w.startswith(c) for w in words) for c in cues.split())


def indefinite(text: str) -> bool:
    return bool({"a", "an", "any", "some", "one", "either"} & _words(text))


def consistent(cmd: dict, text: str) -> bool:
    """The reading uses what the sentence says: each relation has a cue word in it, each kind a noun in it, and a
    stacking word in it is used. Only these readings vote, when there are any."""
    rels = [st["target"]["relation"] for st in cmd["steps"]]
    if not all(said(r, text) for r in rels):
        return False
    words = _words(text)
    for st in cmd["steps"]:
        cues = KIND_WORDS.get(st["target"]["kind"], "").split()
        if cues and not any(w.startswith(c) for c in cues for w in words):
            return False
    if STACK_WORDS & words and "under" not in rels and cmd["action"] != "pull out from under":
        return False
    return True


@dataclass
class Decision:
    kind: str                                  # plan | ask_user | refuse
    message: str = ""                          # question or reason
    actions: list = field(default_factory=list)
    target: str | None = None                  # id of the first object picked
    notes: list = field(default_factory=list)  # how the decision was reached, for the report


def _matches(g: SceneGraph, i: str, kind: str, color: str) -> bool:
    cls = g.nodes[i].cls
    if cls not in KIND_CLASSES.get(kind, set()):
        return False
    return color in ("any", "other") or color in COLORS.get(cls, set())


def _all(g: SceneGraph, kind: str, color: str, graspable_only: bool = False) -> list[str]:
    found = [i for i in g.ids(graspable_only) if _matches(g, i, kind, color)]
    if not found and kind == "white part":     # with no white part on the table, "the white part" is the small one
        found = [i for i in g.ids(graspable_only) if _matches(g, i, "small white part", color)]
    return found


def _nearest(g: SceneGraph, cands: list[str], refs: list[str], far: bool = False) -> list[str]:
    if not cands or not refs:
        return []
    score = {c: min(g.dist_mm(c, r) for r in refs if r != c) for c in cands if any(r != c for r in refs)}
    if not score:
        return []
    best = (max if far else min)(score.values())
    return [c for c, s in score.items() if s == best]


def _side(g: SceneGraph, cands: list[str], refs: list[str], side: str) -> list[str]:
    """Objects on one side of the reference, nearest first; 'left' means as seen in the photo."""
    out = []
    for c in cands:
        for r in refs:
            if c == r:
                continue
            du, dv = g.centre(c) - g.centre(r)
            ok = {"to the left of": du < 0 and abs(du) >= 0.5 * abs(dv), "to the right of": du > 0 and abs(du) >= 0.5 * abs(dv),
                  "above": dv < 0 and abs(dv) >= 0.5 * abs(du), "below": dv > 0 and abs(dv) >= 0.5 * abs(du)}[side]
            if ok:
                out.append(c)
                break
    return _nearest(g, out, refs) if out else []


def _between(g: SceneGraph, cands: list[str], a: list[str], b: list[str]) -> list[str]:
    """Objects whose centre lies near the segment joining an A and a B, nearest to its middle first."""
    best = []
    for ra in a:
        for rb in b:
            p, q = g.centre(ra), g.centre(rb)
            seg = q - p
            L2 = float(seg @ seg) or 1.0
            for c in cands:
                if c in (ra, rb):
                    continue
                x = g.centre(c) - p
                s = float(x @ seg) / L2
                off = float(np.linalg.norm(x - s * seg))
                if 0.1 < s < 0.9 and off < 0.35 * np.sqrt(L2):
                    best.append((float(np.linalg.norm(g.centre(c) - (p + q) / 2)), c))
    best.sort()
    return [best[0][1]] if best else []


def resolve(g: SceneGraph, t: dict) -> list[str]:
    """The object ids an instruction's target description fits."""
    kind, color, rel = t["kind"], t["color"], t["relation"]
    if rel in NEEDS_REF and t.get("ref_kind", "none") == "none":
        rel = {"to the left of": "leftmost", "to the right of": "rightmost", "above": "topmost",
               "below": "bottommost"}.get(rel, "none")      # a relation to nothing: Apertus filled the field anyway
    cands = _all(g, kind, color, graspable_only=kind == "object")
    for e in t.get("except", []):
        cands = [c for c in cands if not _matches(g, c, e["kind"], e["color"])]
    rk, rc = t.get("ref_kind", "none"), t.get("ref_color", "any")
    refs = [] if rk in ("none", "grey tray", "black tray", "person's hand") else _all(g, rk, rc)
    if rk in KIND_CLASSES and rel not in ("none", "leftmost", "rightmost", "topmost", "bottommost", "biggest", "smallest",
                                          "with nothing on it", "in the middle") and not rel.startswith("nth"):
        cands = [c for c in cands if c not in refs or len(refs) > 1]      # "the gear lying on the other gear"
    n = max(1, t.get("n", 1) or 1)
    by_u = lambda ids: sorted(ids, key=lambda i: g.centre(i)[0])
    by_v = lambda ids: sorted(ids, key=lambda i: g.centre(i)[1])
    if rel == "none":
        return cands
    if rel == "in":
        if rk in ("grey tray", "black tray"):
            return [c for c in cands if g.nodes[c].tray == rk]
        return [c for c in cands if any(r in g.nodes[c].on for r in refs)]
    if rel == "lying on":
        return [c for c in cands if any(r in g.nodes[c].on for r in refs)]
    if rel == "under":
        return [c for c in cands if any(r in g.nodes[c].under for r in refs)]
    if rel == "with nothing on it":
        return [c for c in cands if not g.nodes[c].under]
    if rel == "in the person's hand":
        return [c for c in cands if g.nodes[c].touched_by_hand]
    if rel in ("to the left of", "to the right of", "above", "below"):
        if not refs:   # "the gear on the left" parsed as "to the left of" with nothing named
            rel = {"to the left of": "leftmost", "to the right of": "rightmost", "above": "topmost", "below": "bottommost"}[rel]
        else:
            return _side(g, cands, refs, rel)
    if rel in ("closest to", "next to"):
        return _nearest(g, cands, refs)
    if rel == "farthest from":
        return _nearest(g, cands, refs, far=True)
    if rel == "leftmost":
        return by_u(cands)[:1]
    if rel == "rightmost":
        return by_u(cands)[-1:]
    if rel == "topmost":
        return by_v(cands)[:1]
    if rel == "bottommost":
        return by_v(cands)[-1:]
    if rel.startswith("nth"):
        order = {"nth from the left": by_u(cands), "nth from the right": by_u(cands)[::-1],
                 "nth from the top": by_v(cands), "nth from the bottom": by_v(cands)[::-1]}[rel]
        return order[n - 1: n]
    if rel == "between":
        refs2 = _all(g, t.get("ref2_kind", "none"), t.get("ref2_color", "any")) if t.get("ref2_kind") in KIND_CLASSES else []
        return _between(g, [c for c in cands if c not in refs2], refs, refs2)
    if rel == "in the middle":       # the middle one along the direction the objects are spread out in
        if len(cands) < 3 or len(cands) % 2 == 0:
            return cands
        spread_u = np.ptp([g.centre(i)[0] for i in cands])
        spread_v = np.ptp([g.centre(i)[1] for i in cands])
        order = by_u(cands) if spread_u >= spread_v else by_v(cands)
        return [order[len(order) // 2]]
    if rel in ("biggest", "smallest"):
        s = sorted(cands, key=lambda i: g.nodes[i].obj.area)
        return s[-1:] if rel == "biggest" else s[:1]
    return cands


def _name(g: SceneGraph, i: str) -> str:
    return f"the {g.nodes[i].cls}"


def _list_positions(g: SceneGraph, ids: list[str]) -> str:
    """'the left one, the middle one and the right one' (or top/bottom): how a person would tell them apart."""
    us, vs = [g.centre(i)[0] for i in ids], [g.centre(i)[1] for i in ids]
    across = np.ptp(us) >= np.ptp(vs)
    n = len(ids)
    if n == 2:
        names = ["the left one", "the right one"] if across else ["the upper one", "the lower one"]
    elif n == 3:
        names = ["the left one", "the middle one", "the right one"] if across else ["the top one", "the middle one", "the bottom one"]
    else:
        side = "left" if across else "top"
        names = [f"the {k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'} from the {side}" for k in range(1, n + 1)]
    return ", ".join(names[:-1]) + " or " + names[-1]


def decide(g: SceneGraph, cmd: dict, text: str = "") -> Decision:
    notes = []
    steps = cmd["steps"]
    words = steps[0]["target"]["words"] or "the object"
    # 1. unsafe actions: refused whatever the scene
    if cmd["action"] in UNSAFE:
        return Decision("refuse", UNSAFE[cmd["action"]].format(it=words), notes=[f"action: {cmd['action']}"])
    if cmd.get("holds_two_at_once"):
        return Decision("refuse", "The gripper holds one object at a time, so it cannot carry these together. I can move "
                        "them one after the other.", notes=["two at once"])
    if any(st["destination"]["place"] == "onto or next to the person's hand or body" for st in steps):
        return Decision("refuse", "Putting an object on or next to a person's hand or body means moving the gripper onto "
                        "a person.", notes=["destination: the person"])
    if cmd["other_danger"]:
        return Decision("refuse", f"This could hurt someone or make an object fall: {cmd['meaning']}", notes=["other danger"])
    # 2. an object only pointed at
    named = any(st["target"]["relation"] != "none" and said(st["target"]["relation"], text) for st in steps)
    if cmd["vague_reference"] and not named:      # "the object in my hand" is pointed at, but also described
        return Decision("ask_user", "Which object do you mean? I cannot see where you are pointing. I can see: "
                        + ", ".join(sorted({g.nodes[i].cls for i in g.ids()})) + ".", notes=["vague reference"])
    keep = {i for k in cmd.get("keep_still", []) for i in _all(g, k["kind"], k["color"])}
    plan, moved = [], []
    for st in steps:
        t, d = st["target"], st["destination"]
        if t["relation"] in ("lying on", "in") and (
                (d["place"] == "on or in an object" and t["ref_kind"] == d["kind"])
                or (d["place"] in ("grey tray", "black tray") and t["ref_kind"] == d["place"])):
            t = {**t, "relation": "none", "ref_kind": "none"}   # "the mandarin" read as "the mandarin on the plate"
            notes.append("the description named the destination: dropped")
        if t["relation"] == "none":                     # "the biggest object" with the relation left empty
            sup = [r for w, r in SUPERLATIVES.items() if w in t["words"].lower().split()]
            if sup:
                t = {**t, "relation": sup[0]}
                notes.append(f"'{t['words']}': relation {sup[0]} from the words")
        if t["relation"] == "under" and t["ref_kind"] == "none" and said("with nothing on it", text):
            t = {**t, "relation": "with nothing on it"}   # "the red gear with nothing on it", read as "under"
        if text and not said(t["relation"], text) and t["relation"] in ("under", "lying on") \
                and said({"under": "below", "lying on": "above"}[t["relation"]], text):
            t = {**t, "relation": {"under": "below", "lying on": "above"}[t["relation"]]}   # "below X": lower in the photo
            notes.append(f"stacking word not in the instruction: read as '{t['relation']}' in the photo")
        if text and not said(t["relation"], text):
            notes.append(f"relation '{t['relation']}' is not in the instruction: dropped")
            t = {**t, "relation": "none", "ref_kind": "none"}
        ids = resolve(g, t)
        if not ids and t["relation"] == "under" and said("below", text) and not STACK_WORDS & _words(text):
            t = {**t, "relation": "below"}            # "below the red gears": lower in the photo, not stacked
            ids = resolve(g, t)
        notes.append(f"'{t['words']}' -> {ids}")
        if not ids:
            same_kind = _all(g, t["kind"], "any")
            if t["kind"] == "something else" or not _all(g, t["kind"], t["color"]):
                return Decision("refuse", f"There is no {t['words'].removeprefix('the ')} in the workcell.", notes=notes)
            return Decision("refuse", f"I cannot find {t['words']}: none of the {len(same_kind)} matching objects fits that "
                            "description.", notes=notes)
        if t["how_many"] != "all" and {"both", "all", "every", "each", "everything"} & _words(t["words"]):
            t = {**t, "how_many": "all"}               # "both grey parts", read as one
            notes.append("'both/all/every' in the words: all of them")
        if t["how_many"] == "any one" and text and not indefinite(text):
            t = {**t, "how_many": "one"}               # "the fruit" is one particular fruit, whatever Apertus said
            notes.append("'any one' without a, an, any or some: one particular object")
        if t["how_many"] == "any one" and len(ids) > 1:     # "a red gear": any will do; take one with nothing on it
            ids = sorted(ids, key=lambda i: (len(g.above(i)), g.centre(i)[0]))[:1]
            notes.append(f"any one of them: {ids[0]}")
        if t["how_many"] == "one" and len(ids) > 1:
            return Decision("ask_user", f"There are {len(ids)} objects that fit '{t['words']}'. Which one do you mean: "
                            f"{_list_positions(g, ids)}?", notes=notes)
        if any(g.nodes[i].touched_by_hand for i in ids):
            return Decision("refuse", f"{_name(g, ids[0]).capitalize()} is in or under a person's hand: the gripper must not "
                            "move onto a person.", notes=notes)
        wide = [i for i in ids if g.nodes[i].cls not in GRASPABLE]
        if wide:
            return Decision("refuse", f"{_name(g, wide[0]).capitalize()} is wider than the gripper's 85 mm opening, so the "
                            "robot cannot pick it up.", notes=notes)
        # destination
        dest = None
        if d["place"] == "a tray, not saying which":
            return Decision("ask_user", "Which tray do you mean, the grey tray on the left or the black tray on the right?",
                            notes=notes)
        if d["place"] == "next to an object":            # set it down in the same tray as that object, beside it
            dids = [i for i in _all(g, d["kind"], d["color"]) if i not in ids]
            if not dids:
                return Decision("refuse", f"There is no {d['kind']} to put it next to.", notes=notes)
            dest = g.nodes[dids[0]].tray.replace(" ", "_")
            notes.append(f"destination: next to {dids[0]}, in the {g.nodes[dids[0]].tray}")
        elif d["place"] in ("grey tray", "black tray"):
            dest = d["place"].replace(" ", "_")
        elif d["place"] == "on or in an object":
            dids = [i for i in _all(g, d["kind"], d["color"]) if i not in ids]
            notes.append(f"destination '{d['kind']}' -> {dids}")
            if not dids:
                return Decision("refuse", f"There is no {d['color'] + ' ' if d['color'] not in ('any', 'other') else ''}"
                                f"{d['kind']} to put it on.", notes=notes)
            if len(dids) > 1:
                return Decision("ask_user", f"There are {len(dids)} {d['kind']}s. Which one should it go on: "
                                f"{_list_positions(g, dids)}?", notes=notes)
            dest = dids[0]
            dcls = g.nodes[dest].cls
            if g.nodes[dest].touched_by_hand:
                return Decision("refuse", f"The {dcls} is in or under a person's hand: the gripper must not move onto a "
                                "person.", notes=notes)
            if dcls in CONTAINERS:
                big = [i for i in ids if g.nodes[i].obj.size_mm[1] > CONTAINERS[dcls]
                       or g.nodes[i].obj.size_mm[0] > 1.6 * CONTAINERS[dcls]]
                if big:
                    return Decision("refuse", f"{_name(g, big[0]).capitalize()} does not fit in the {dcls}.", notes=notes)
        # what has to move first, and the order
        order = []
        for i in sorted(ids, key=lambda i: -len(g.above(i))):
            for top in g.above(i) + [i]:
                if top not in order:
                    order.append(top)
        blockers = [o for o in order if o not in ids]
        clash = [o for o in order if o in keep]
        if clash:
            why = (f"you asked me to leave {_name(g, clash[0])} where it is, but it lies on {_name(g, ids[0])} and must "
                   "be moved first") if clash[0] in blockers else \
                f"the instruction asks both to move {_name(g, clash[0])} and to leave it where it is"
            return Decision("refuse", f"I cannot do this: {why}.", notes=notes)
        for o in order:
            if o in moved:
                continue
            moved.append(o)
            plan.append({"action": "pick", "object": o})
            if o in blockers:
                plan.append({"action": "place", "object": g.nodes[o].tray.replace(" ", "_")})   # set it aside, same tray
                notes.append(f"{o} lies on {ids} and is moved first")
            elif dest:
                plan.append({"action": "place", "object": dest})
    if not plan:
        return Decision("refuse", "There is nothing to move.", notes=notes)
    return Decision("plan", actions=plan, target=plan[0]["object"], notes=notes)
