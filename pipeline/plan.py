"""Decide and plan: resolve Apertus's structured command on the scene graph, then apply the safety
rules and the robot's limits, and order the picks.

The rules come from docs/safety_primer.md and the challenge:
- refuse: throwing, dropping, pushing off, pulling out from under, taking from a person's hand,
  anything else Apertus flags as dangerous; an object that is not there; an object too wide for
  the 85 mm gripper (plate, saucer); an object that does not fit where it should go; an instruction
  that forbids a move it needs (or contradicts itself)
- ask: the object is only pointed at ("that one"), or the words fit several objects equally
- otherwise plan: one object at a time, whatever lies on an object is moved away first.

Two readings of the instruction (run.py --reading): with "rules", keyword rules repair Apertus's structured
command against the sentence (each is guarded by RULES); with "questions", those rules are off, three of them
are replaced by yes/no questions to Apertus (person, two at once, an unknown name), and run.py asks follow-up
questions where the rules used to decide. The safety limits above apply in both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .perception import COLOR_WORDS, NEAR, OTHER
from .scene_graph import SceneGraph

# Which reading runs (run.py --reading). "rules": the keyword rules below repair Apertus's reading of the
# instruction. "questions": those rules are off and Apertus is asked short questions instead (run.py h3).
RULES = True
QUESTIONS = False

YN_PERSON = ('A person tells a robot arm: "{text}"\nDoes the person ask the robot to take an object out of a person\'s hand '
             '(an object a person is holding)? Answer yes or no.')
YN_TWO = ('A person tells a robot arm with a two-finger gripper: "{text}"\nDoes the person ask the robot to hold two or more '
          'objects in its gripper at the same time? Answer yes or no.')
YN_NAME = ('A person tells a robot arm: "{text}"\nOn the table there is {a} {name}{desc}. Can "{phrase}" mean the {name}? '
           'Answer yes or no.')

KIND_CLASSES = {"gear": {"red gear", "yellow gear"}, "cube": {"red cube"}, "mug": {"mug"}, "mandarin": {"mandarin"},
                "banana": {"banana"}, "fruit": {"mandarin", "banana"}, "plate": {"plate"}, "saucer": {"saucer"},
                "grey metal part": {"grey metal part"}, "white part": {"white part"},
                "small white part": {"small white part"},
                "part": {"grey metal part", "white part", "small white part"},
                "object": None, "something else": set()}          # None: any object
CATEGORIES = {"fruit", "part", "something else"}
NOT_COLOUR = re.compile(r"\b(isn'?t|is not|not|non-?|other than)\s+(\w+)")
# "starting with the smallest", "from right to left": the order of a set's picks
ORDER_WORDS = [(re.compile(r"right to left|from the right|rightmost first|right one first"), lambda c, a: -c[0]),
               (re.compile(r"left to right|from the left|leftmost first|left one first"), lambda c, a: c[0]),
               (re.compile(r"top to bottom|from the top|top one first"), lambda c, a: c[1]),
               (re.compile(r"bottom to top|from the bottom|bottom one first"), lambda c, a: -c[1]),
               (re.compile(r"(start|starting|begin|beginning) with the (smallest|small)|smallest (one |item |object )?first"),
                lambda c, a: a),
               (re.compile(r"(start|starting|begin|beginning) with the (biggest|largest|big)|(biggest|largest) (one |item |object )?first"),
                lambda c, a: -a)]
ORDINALS = {"first", "second", "third", "fourth", "fifth", "sixth", "last", "next-to-last", "penultimate"}
PRONOUNS = {"it", "this", "that", "them", "these", "those", "one", "this one", "that one", "the", "same"}
NEXT_WORDS = {"next", "beside", "near", "by", "adjacent", "alongside", "besides", "nearby"}
BARE_PRONOUNS = {"it", "this", "that", "them", "these", "those"}      # nothing else in the words: only pointed at
ANY_NOUNS = {"two", "three", "four", "five", "both", "other", "each", "all", "them", "these", "those",
             "object", "thing", "item", "one", "stuff", "everything", "anything", "something", "objects", "things",
             "whatever", "whichever",
             "items", "ones", "rest", "remaining", "others", "piece"}
GENERIC_NOUNS = {"object", "thing", "item", "one", "fruit", "part", "tool", "piece", "stuff", "everything", "anything",
                 "gear", "cube", "mug", "cup", "block", "cog"}       # words for a whole kind: "the gear" is any gear   # kinds an unknown object may belong to that only Apertus can tell
UNSAFE = {"throw or toss": "Throwing {it} is uncontrolled motion: it could hit someone or break.",
          "drop": "Dropping {it} is uncontrolled motion: it could fall on someone or break.",
          "push or slide": "Pushing {it} would make it slide or fall in an uncontrolled way.",
          "pull out from under": "Pulling {it} out from under the object on top would make that object fall.",
          "take from the person": "Taking {it} out of a person's hand means moving the gripper onto a person."}
CONTAINERS = {"mug": 75.0}          # inner opening, mm: wider than this does not go in, longer than 1.6x sticks out
INNER = 0.8                         # an unknown container's opening, as a share of its width seen from above
NEEDS_REF = {"to the left of", "to the right of", "above", "below", "closest to", "farthest from", "next to", "lying on",
             "under", "in", "in the person's hand"}

# Checks on Apertus's reading against the sentence itself. Apertus sometimes copies a relation from the scene facts
# into a sentence that never says it ("pick up the red gear" read as "the red gear under the yellow gear"), which
# would hide an ambiguity; so a relation needs one of its cue words in the instruction, or it is dropped. The same
# goes for the noun of the object ("the yellow fruit" read as "a yellow mandarin").
CUES = {"to the left of": "left", "to the right of": "right", "above": "above top upper higher",
        "below": "below bottom lower beneath", "closest to": "close near next beside by adjacent",
        "farthest from": "far", "next to": "next near beside by close adjacent", "lying on": "on onto top off",
        "under": "under beneath underneath has have with lying resting sitting standing stands support carries", "in": "in inside from within", "with nothing on it":
        "nothing free empty uncovered clear", "leftmost": "left", "rightmost": "right", "topmost": "top upper highest",
        "bottommost": "bottom lower lowest", "nth from the left": "left", "nth from the right": "right",
        "nth from the top": "top", "nth from the bottom": "bottom", "biggest": "big large", "smallest": "small little tiny",
        "in the middle": "middle centre center central", "between": "between",
        "in the person's hand": "hand holding held fingers my", "in the top left corner": "corner top upper left",
        "in the top right corner": "corner top upper right", "in the bottom left corner": "corner bottom lower left",
        "in the bottom right corner": "corner bottom lower right", "oriented left-right": "horizontal across sideways left",
        "oriented up-down": "vertical upright up down", "touching": "touch contact against",
        "not lying on": "not isn't off outside other", "not in": "not isn't outside other",
        "across both trays": "across both border straddl between over"}
# taking from a person needs the person in the sentence: "take the mandarin off the saucer" is an ordinary pick
PERSON_WORDS = {"hand", "hands", "fingers", "finger", "me", "my", "palm", "person", "holding", "holds", "held", "grip", "you"}
SIDES = {"top": "topmost", "upper": "topmost", "bottom": "bottommost", "lower": "bottommost", "left": "leftmost",
         "right": "rightmost"}
OPPOSITE = {"topmost": "bottommost", "bottommost": "topmost", "leftmost": "rightmost", "rightmost": "leftmost"}
# "keep the cube where it is", "without moving the mug", "don't touch the plate": a constraint. "without dropping it",
# "don't bump the mug" ask for care; Apertus sometimes lists those objects as ones to keep still, and a plan that moves
# them (the gear lying on the target) would be refused for nothing.
# Apertus's "keep still" and "two at once" stand unless the sentence only asks for care: losing a refusal costs
# about twice what a needless refusal costs (30 x 1/10 against 30 x 1/40 + 35/40 per instruction)
KEEP_PHRASE = re.compile(r"\b(without|not|don'?t|do not|never|avoid|no)\s+(\w+\s+){0,1}"
                         r"(mov|touch|disturb|shift|lift|pick|budg|displac|relocat|tak|grab|remov)"
                         r"|\b(leave|keep|let|must|has to|have to|should)\b.*\b(where|still|place|alone|untouched|stay|spot|"
                         r"remain|put|position)|\bstays?\b|\bremains?\b|\bin (its )?place\b|\bin its spot\b")
CARE_PHRASE = re.compile(r"\b(without|not|don'?t|do not|never|avoid|careful|carefully|gently)\b\s*(\w+\s+){0,2}"
                         r"(drop|bump|knock|hit|damag|spill|break|scratch|crush|tip|fall|let)")
TWO_AT_ONCE = re.compile(r"same time|together|at once|simultaneous|single|one (go|grab|grasp|grip|move|motion|trip|hand|"
                         r"pick|lift|scoop)|in one\b|both at")
# a noun in the words beats Apertus's generic "object": "the small part" is a part
HEAD_KIND = {"gear": "gear", "cog": "gear", "cube": "cube", "block": "cube", "mug": "mug", "cup": "mug",
             "mandarin": "mandarin", "tangerine": "mandarin", "clementine": "mandarin", "banana": "banana",
             "fruit": "fruit", "plate": "plate", "dish": "plate", "saucer": "saucer", "part": "part", "bar": "grey metal part"}
SAME_SIZE = 0.12          # areas within 12% count as the same size: "the big gear" among three equal gears is ambiguous
KIND_SYNONYMS = {"gear": "gear cog wheel sprocket", "cube": "cube block box brick square", "mug": "mug cup teacup",
                 "mandarin": "mandarin orange tangerine clementine citrus", "banana": "banana", "fruit": "fruit",
                 "plate": "plate dish", "saucer": "saucer", "grey metal part": "part bar metal bracket rail strip plate piece",
                 "white part": "part plate panel piece sheet", "small white part": "part block piece brick",
                 "part": "part piece component bracket bar"}
KIND_WORDS = {"gear": "gear cog", "cube": "cube block box", "mug": "mug cup", "mandarin": "mandarin orange tangerine clementine",
              "banana": "banana", "fruit": "fruit", "plate": "plate dish", "saucer": "saucer", "grey metal part": "part bar metal",
              "white part": "part", "small white part": "part", "part": "part", "object": "", "something else": ""}
STACK_WORDS = {"under", "beneath", "underneath"}
SUPERLATIVES = {"biggest": "biggest", "largest": "biggest", "smallest": "smallest", "leftmost": "leftmost",
                "bigger": "biggest", "larger": "biggest", "smaller": "smallest", "lower": "bottommost",
                "higher": "topmost",
                "rightmost": "rightmost", "lowest": "bottommost", "bottommost": "bottommost", "highest": "topmost",
                "topmost": "topmost", "uppermost": "topmost"}


def _words(text: str) -> set[str]:
    return set(text.lower().replace(",", " ").replace(".", " ").split())


def said(rel: str, text: str) -> bool:
    """With the rules on, a relation counts only if the instruction has a cue word for it; without them, always."""
    return cue_in(rel, text) if RULES else True  # V


def cue_in(rel: str, text: str) -> bool:
    """The instruction contains a cue word for this relation (relations without cues always pass)."""
    words, cues = _words(text), CUES.get(rel)
    return cues is None or any(any(w.startswith(c) for w in words) for c in cues.split())


def indefinite(text: str) -> bool:
    return bool({"a", "an", "any", "some", "one", "either"} & _words(text))


def consistent(cmd: dict, text: str) -> bool:
    """The reading uses what the sentence says: each relation has a cue word in it, each kind a noun in it, and a
    stacking word in it is used. Only these readings vote, when there are any."""
    if not RULES:  # V
        return True
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
    info: dict = field(default_factory=dict)   # where the decision stopped: step, target, candidates (for follow-up questions)


def _stem(w: str) -> str:
    return w[:-2] if w.endswith("es") and len(w) > 4 else w[:-1] if w.endswith("s") and len(w) > 3 else w


def _head(phrase: str) -> str:
    """The noun of a short name or phrase: its last word ("tennis ball" -> "ball")."""
    words = [w for w in phrase.lower().replace(",", " ").replace(".", " ").split() if w not in ("the", "a", "an", "one", "some", "any")]
    return _stem(words[-1]) if words else ""


CUT_WORDS = {"in", "on", "of", "from", "with", "that", "which", "who", "sitting", "lying", "resting", "next", "closest",
             "nearest", "farthest", "furthest", "to", "near", "below", "above", "under", "beneath", "underneath", "between",
             "at", "inside", "into", "onto", "by", "beside", "left", "right", "top", "bottom", "except", "but", "and",
             "just", "is", "are", "was", "that's", "thats", "standing", "touching", "placed", "located"}


COMPARATIVES = {"nearer", "closer", "farther", "further", "higher", "lower", "bigger", "smaller", "larger"}


def _noun(phrase: str) -> str:
    """The noun a target phrase names: the last word before its first preposition or relative word ("the second
    object from the bottom" -> "object", "the red one" -> "one")."""
    words = [w for w in phrase.lower().replace(",", " ").replace(".", " ").split() if w not in ("the", "a", "an")]
    out = []
    for w in words:
        if out and (w in CUT_WORDS or w in COMPARATIVES or (w.endswith("ly") and len(w) > 4)):   # "directly", "nearer"
            break
        out.append(w)
    return _stem(out[-1]) if out else ""


def _ref_phrase(which: str) -> str:
    """The object a relation phrase names: "closest to the computer mouse" -> "the computer mouse"."""
    w = " " + which.lower().strip() + " "
    cut = max(w.rfind(" the "), w.rfind(" a "), w.rfind(" an "))
    return w[cut:].strip() if cut >= 0 else which


def _colour_ok(g: SceneGraph, i: str, color: str) -> bool:
    n = g.nodes[i]
    if color in ("any", "other"):
        return True
    if n.cls == OTHER:              # seen colours: allow the neighbouring words (yellow/green, red/orange)
        return any(color in NEAR.get(c, {c}) for c in n.obj.colors)
    return color in n.obj.colors


def _matches(g: SceneGraph, i: str, kind: str, color: str, phrase: str = "") -> bool:
    n = g.nodes[i]
    if n.cls == OTHER:              # an object outside the workcell's list: by the noun of its name
        if kind == "object":
            ok = True
        elif kind == "something else":
            ok = bool(phrase) and (_noun(phrase) or _head(phrase)) in {_stem(w) for w in _words(n.name)}
        else:
            ok = any(_head(n.name).startswith(c) for c in KIND_WORDS.get(kind, "").split())
    else:
        classes = KIND_CLASSES.get(kind, set())
        ok = classes is None or n.cls in classes
    return ok and _colour_ok(g, i, color)


def _all(g: SceneGraph, kind: str, color: str, graspable_only: bool = False, phrase: str = "", text: str = "") -> list[str]:
    found = [i for i in g.ids(graspable_only) if _matches(g, i, kind, color, phrase)]
    if not found and kind == "white part":     # with no white part on the table, "the white part" is the small one
        found = [i for i in g.ids(graspable_only) if _matches(g, i, "small white part", color)]
    others = [i for i in g.ids(graspable_only) if g.nodes[i].cls == OTHER and i not in found and _colour_ok(g, i, color)]
    if kind in CATEGORIES and others and phrase and g.matcher and not (kind == "something else" and found):
        names = g.matcher(phrase, text, sorted({g.nodes[i].name for i in others}))     # Apertus: "the fruit" -> apple?
        found += [i for i in others if g.nodes[i].name in names]
    noun = _noun(phrase) or _head(phrase)
    looks = kind == "something else" and noun not in ANY_NOUNS | GENERIC_NOUNS or         (kind in KIND_SYNONYMS and kind not in ("fruit", "part") and color not in ("any", "other"))   # "the purple cube"
    if looks and not found and others and g.looker and noun:
        what = " ".join(w for w in _ref_phrase(phrase).split() if w not in ("the", "a", "an")) or noun
        found = [i for i in others if g.looker(i, what)]      # named "USB flash drive", but is it the padlock?
    head = _noun(phrase)
    if RULES and head and head not in GENERIC_NOUNS | ANY_NOUNS and found:  # R23
        exact = [i for i in found if head in {_stem(w) for w in _words(g.nodes[i].name)}]
        if exact:
            found = exact
        elif kind in KIND_SYNONYMS and not any(head.startswith(w) or w.startswith(head)
                                               for w in KIND_SYNONYMS[kind].split()):
            # "the apple" read as kind fruit: no object is an apple; "the cucumber" read as banana: maybe an unknown one
            found = _all(g, "something else", color, graspable_only, phrase, text) if phrase else []
    return found


def _phrase(t: dict) -> str:
    """The words that name the target, without the part that says which one ("the screwdriver next to the mug"
    -> "the screwdriver")."""
    words, which = t.get("words", ""), t.get("which_words", "")
    return words.replace(which, " ").strip() if which and which in words else words


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
            if ra == rb:
                continue
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


def resolve(g: SceneGraph, t: dict, text: str = "") -> list[str]:
    """The object ids an instruction's target description fits."""
    if t.get("pin"):                    # an object Apertus chose in a follow-up question
        return [i for i in t["pin"] if i in g.nodes]
    kind, color, rel = t["kind"], t["color"], t["relation"]
    if rel in NEEDS_REF and t.get("ref_kind", "none") == "none":
        side = [SIDES[w] for w in _words(t.get("which_words", "") or t.get("words", "")) if w in SIDES]
        if RULES and rel in ("closest to", "next to", "farthest from") and side:     # R24: "nearest the top of the picture"
            rel = side[0] if rel != "farthest from" else OPPOSITE[side[0]]
        else:
            rel = {"to the left of": "leftmost", "to the right of": "rightmost", "above": "topmost",
                   "below": "bottommost"}.get(rel, "none")      # a relation to nothing: Apertus filled the field anyway
    cands = _all(g, kind, color, graspable_only=kind == "object", phrase=_phrase(t), text=text)
    for e in t.get("except", []):
        cands = [c for c in cands if not _matches(g, c, e["kind"], e["color"])]
    rk, rc = t.get("ref_kind", "none"), t.get("ref_color", "any")
    refs = [] if rk in ("none", "grey tray", "black tray", "person's hand") else \
        _all(g, rk, rc, phrase=_ref_phrase(t.get("which_words", "")), text=text)     # "next to the tennis ball"
    refs = _qualified(g, refs, rk, t.get("which_words", ""))
    if t.get("ref_pin"):                # the reference object Apertus chose in a follow-up question
        refs = [i for i in t["ref_pin"] if i in g.nodes]
    if rk in KIND_CLASSES and rk != "something else" and rel not in ("none", "leftmost", "rightmost", "topmost", "bottommost", "biggest", "smallest",
                                          "with nothing on it", "in the middle") and not rel.startswith("nth"):
        cands = [c for c in cands if c not in refs or len(refs) > 1]      # "the gear lying on the other gear"
    if rk in ("grey tray", "black tray") and rel not in ("in", "not in", "closest to", "next to", "farthest from",
                                                         "to the left of", "to the right of", "above", "below"):
        cands = [c for c in cands if g.nodes[c].tray == rk] or cands     # "the middle object in the black tray"
        rk = "none"
    tray = _tray_said(t) if RULES else ""  # R25
    if tray and rel not in ("in", "not in") and rk not in ("grey tray", "black tray"):
        inside = [c for c in cands if g.nodes[c].tray == tray]
        cands = inside or cands
    n = max(1, t.get("n", 1) or 1)
    by_u = lambda ids: sorted(ids, key=lambda i: g.centre(i)[0])
    by_v = lambda ids: sorted(ids, key=lambda i: g.centre(i)[1])
    if rel == "none":
        return cands
    if rk in ("grey tray", "black tray") and rel in ("closest to", "next to", "farthest from"):
        d = {c: g.tray_dist_mm(c, rk) for c in cands}       # "the gear closest to the grey tray"
        if not d:
            return []
        best = (max if rel == "farthest from" else min)(d.values())
        return [c for c, v in d.items() if abs(v - best) <= 10]
    if rel.startswith("in the") and rel.endswith("corner"):
        su = -1 if "left" in rel else 1                       # the corner the object is nearest to, in the photo
        sv = -1 if "top" in rel else 1
        score = {c: su * g.centre(c)[0] + sv * g.centre(c)[1] for c in cands}
        return [max(score, key=score.get)] if score else []
    if rel in ("oriented left-right", "oriented up-down"):
        want = "h" if rel == "oriented left-right" else "v"
        return [c for c in cands if g.orientation(c) == want]
    if rel == "touching":
        touch = [c for c in cands if any(g.touching(c, r) for r in refs if r != c)]
        return touch or _nearest(g, cands, refs)
    if rel == "not lying on":
        return [c for c in cands if not any(r in g.nodes[c].on for r in refs)]
    if rel == "not in":
        if rk in ("grey tray", "black tray"):
            return [c for c in cands if g.nodes[c].tray != rk]
        return [c for c in cands if not any(r in g.nodes[c].on for r in refs)]
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
    if rel.startswith("nth") and n > len(cands):
        return []                   # "the third mandarin" when there are two
    if rel.startswith("nth"):
        order = {"nth from the left": by_u(cands), "nth from the right": by_u(cands)[::-1],
                 "nth from the top": by_v(cands), "nth from the bottom": by_v(cands)[::-1]}[rel]
        return order[n - 1: n]
    if rel == "between":
        refs2 = _all(g, t.get("ref2_kind", "none"), t.get("ref2_color", "any"), phrase=_ref_phrase(t.get("which_words", "")),
                     text=text) \
            if t.get("ref2_kind") in KIND_CLASSES else []
        refs2 = _qualified(g, refs2, t.get("ref2_kind", "none"), t.get("which_words", ""))
        if RULES and (not refs2 or t.get("ref2_kind") in (t.get("ref_kind"), "none")):  # R27
            refs2 = refs                      # "between the two mandarins", "between the other two"
        return _between(g, cands, refs, refs2)
    if rel == "across both trays":
        return [c for c in cands if g.spans_trays(c)]
    if rel == "in the middle" and len(cands) >= 5:      # a grid or a cluster: the one nearest the centre of the group
        pts = np.array([g.centre(i) for i in cands])
        mid = (pts.min(0) + pts.max(0)) / 2
        d = np.linalg.norm(pts - mid, axis=1)
        order = np.argsort(d)
        if d[order[1]] > 1.3 * d[order[0]]:
            return [cands[order[0]]]
    if rel == "in the middle":       # the middle one along the direction the objects are spread out in
        if len(cands) < 3 or len(cands) % 2 == 0:
            return cands
        spread_u = np.ptp([g.centre(i)[0] for i in cands])
        spread_v = np.ptp([g.centre(i)[1] for i in cands])
        order = by_u(cands) if spread_u >= spread_v else by_v(cands)
        return [order[len(order) // 2]]
    if rel in ("biggest", "smallest"):
        return _extreme(g, cands, rel)
    return cands


def _extreme(g: SceneGraph, ids: list[str], rel: str) -> list[str]:
    """The biggest or smallest; several when they are the same size (three identical red gears)."""
    if not ids:
        return []
    area = {i: g.nodes[i].obj.area for i in ids}
    best = (max if rel == "biggest" else min)(area.values())
    return [i for i in ids if abs(area[i] - best) <= SAME_SIZE * best]


def _tray_said(t: dict) -> str:
    """The tray the target's words place it in ("the second object from the bottom in the black tray"), if any."""
    words = " " + " ".join(t.get("words", "").lower().replace("gray", "grey").replace(",", " ").split()) + " "
    for tray in ("grey tray", "black tray"):
        if any(f" {p} the {tray} " in words for p in ("in", "inside", "from", "within", "of")):
            return tray
    return ""


def _qualified(g: SceneGraph, refs: list[str], kind: str, which: str) -> list[str]:
    """'between the top cube and the red gear': a side word just before the reference's noun picks one of them."""
    if len(refs) < 2 or not which or not RULES:  # R26
        return refs
    words = which.lower().replace(",", " ").split()
    nouns = KIND_WORDS.get(kind, "").split()
    for i, w in enumerate(words):
        if any(w.startswith(n) for n in nouns if n):
            before = [SIDES[x] for x in words[max(0, i - 2): i] if x in SIDES]
            if before:
                return _outermost(g, refs, before[-1])
    return refs


def _outermost(g: SceneGraph, ids: list[str], rel: str) -> list[str]:
    """The leftmost, rightmost, topmost or bottommost of these objects, in the photo."""
    axis, sign = {"leftmost": (0, 1), "rightmost": (0, -1), "topmost": (1, 1), "bottommost": (1, -1)}[rel]
    return sorted(ids, key=lambda i: sign * g.centre(i)[axis])[:1]


def _name(g: SceneGraph, i: str) -> str:
    return f"the {g.nodes[i].name}"


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


def _dest_phrase(where: str) -> str:
    """'on top of the red gear furthest to the left' -> 'the red gear furthest to the left'."""
    w = where.strip()
    for prep in ("on top of", "onto", "on to", "into", "inside", "on", "in", "next to", "beside", "to"):
        if w.lower().startswith(prep + " "):
            return w[len(prep) + 1:].strip()
    return w


def decide(g: SceneGraph, cmd: dict, text: str = "") -> Decision:
    ctx = {}
    d = _decide(g, cmd, text, ctx)
    d.info = ctx
    return d


def _decide(g: SceneGraph, cmd: dict, text: str, ctx: dict) -> Decision:
    notes = []
    # R30
    if RULES and text and _words(text) & {"in", "into", "inside", "on", "onto"} and not _words(text) & NEXT_WORDS \
            and any(st["destination"]["place"] == "next to an object" for st in cmd["steps"]):
        # "put the mug inside the small white part" read as "next to": the sentence says in or on, not beside
        cmd = {**cmd, "steps": [{**st, "destination": {**st["destination"], "place": "on or in an object"}}
                                if st["destination"]["place"] == "next to an object" else st for st in cmd["steps"]]}
        notes.append("destination: the sentence says in or on, not next to")
    steps = cmd["steps"]
    words = steps[0]["target"]["words"] or "the object"
    # 1. unsafe actions: refused whatever the scene
    if cmd["action"] == "take from the person" and text and (not g.yesno(YN_PERSON.format(text=text)) if QUESTIONS
                                                          else RULES and not PERSON_WORDS & _words(text)):  # S1
        cmd = {**cmd, "action": "put or move"}            # "take the mandarin off the saucer"
        notes.append("no person in the instruction: an ordinary pick")
    if cmd["action"] in UNSAFE:
        return Decision("refuse", UNSAFE[cmd["action"]].format(it=words), notes=[f"action: {cmd['action']}"])
    if cmd.get("holds_two_at_once") and (g.yesno(YN_TWO.format(text=text)) if QUESTIONS and text else
                                         (not RULES or not text or TWO_AT_ONCE.search(text.lower()))):  # S2
        return Decision("refuse", "The gripper holds one object at a time, so it cannot carry these together. I can move "
                        "them one after the other.", notes=["two at once"])
    if any(st["destination"]["place"] == "onto or next to the person's hand or body" for st in steps):
        return Decision("refuse", "Putting an object on or next to a person's hand or body means moving the gripper onto "
                        "a person.", notes=["destination: the person"])
    if cmd["other_danger"]:
        return Decision("refuse", f"This could hurt someone or make an object fall: {cmd['meaning']}", notes=["other danger"])
    # 2. an object only pointed at
    if RULES and all(_words(st["target"]["words"]) <= PRONOUNS for st in steps) and not cmd["vague_reference"]:  # R1
        cmd = {**cmd, "vague_reference": True}         # "put it in the other tray": "it" names nothing
    named = any(st["target"]["relation"] != "none" and said(st["target"]["relation"], text) for st in steps)
    # "take it off the gear": nothing but "it" names the object, whatever else the sentence says about it
    bare = RULES and all(_words(st["target"]["words"]) and _words(st["target"]["words"]) <= BARE_PRONOUNS for st in steps)  # R29
    if cmd["vague_reference"] and (not (named and RULES) or bare):      # R28: "the object in my hand" is pointed at, but also described
        return Decision("ask_user", "Which object do you mean? I cannot see where you are pointing. I can see: "
                        + ", ".join(sorted({g.nodes[i].name for i in g.ids()})) + ".", notes=["vague reference"])
    keep = {i for k in cmd.get("keep_still", []) for i in _all(g, k["kind"], k["color"])}
    if RULES and keep and text and CARE_PHRASE.search(text.lower()) and not KEEP_PHRASE.search(text.lower()):  # S3
        notes.append("nothing in the instruction says an object must stay: 'keep still' dropped")
        keep = set()
    plan, moved = [], []
    for st in steps:
        ctx["step"] = ctx.get("step", -1) + 1
        t, d = st["target"], st["destination"]
        neg = NOT_COLOUR.search(t["words"].lower() + " " + (text.lower() if t["color"] != "any" else ""))
        if RULES and neg and neg.group(2) in COLOR_WORDS and t["color"] in (neg.group(2), "any", "other"):  # R2
            t = {**t, "color": "any", "except": t.get("except", []) + [{"kind": t["kind"], "color": neg.group(2)}]}
            notes.append(f"'not {neg.group(2)}': every colour but {neg.group(2)}")
        noun = _noun(t["words"])
        if RULES and t["kind"] == "something else" and noun in ANY_NOUNS:  # R3
            t = {**t, "kind": "object"}                 # "the yellow object", "the red one"
        elif RULES and t["kind"] == "object" and noun and noun not in ANY_NOUNS | set(HEAD_KIND) | set(COLOR_WORDS):  # R3
            t = {**t, "kind": "something else"}         # "the tool", "the book": a name to look for
        for rk_key in ("ref_kind", "ref2_kind"):        # "closest to the screwdriver" read with ref_kind "saucer"
            rk = t.get(rk_key, "none")
            cues = KIND_WORDS.get(rk, "").split() or (["hand", "finger", "my", "palm"] if rk == "person's hand" else [])
            where = t.get("which_words", "")
            if _head(_ref_phrase(where)) in {"two", "three", "other", "others", "one", "ones", "them", "both", ""}:
                where = text                  # "between the other two": the noun is elsewhere in the sentence
            if RULES and cues and not any(w.startswith(c) for c in cues for w in _words(where or text)):  # R4
                t = {**t, rk_key: "something else"}
                notes.append(f"'{rk}' is not in '{t.get('which_words', '')}': an object named there")
        ref_noun = _head(t.get("which_words", ""))
        # R5
        if RULES and t["relation"] in NEEDS_REF - {"in"} and t.get("ref_kind", "none") == "none" and ref_noun \
                and ref_noun not in SIDES and ref_noun not in {"photo", "picture", "image", "tray", "table", "it", "them"}:
            t = {**t, "ref_kind": "something else"}     # "closest to the computer mouse" with no kind filled in
            notes.append(f"the reference '{t['which_words']}' is named there")
        if RULES and t["kind"] == "object" and HEAD_KIND.get(_noun(t["words"])):  # R6
            t = {**t, "kind": HEAD_KIND[_noun(t["words"])]}
            notes.append(f"'{_phrase(t)}': kind {t['kind']} from its noun")
        if RULES and t["how_many"] == "one" and t["words"].lower().split()[:1] in (["a"], ["an"], ["any"]):  # R7
            t = {**t, "how_many": "any one"}           # "put a red gear on the plate": any of them
        which = t.get("which_words", "").strip().lower()
        if RULES and (t["relation"] in ("lying on", "in", "above") and (  # R8
                (d["place"] == "on or in an object" and t["ref_kind"] == d["kind"])
                or (d["place"] in ("grey tray", "black tray") and t["ref_kind"] == d["place"])) \
                or (len(which.split()) > 1 and which in cmd.get("move_where", "").lower())):
            t = {**t, "relation": "none", "ref_kind": "none"}   # "the mandarin" read as "the mandarin on the plate"
            notes.append("the description named the destination: dropped")
        if RULES and t["relation"].startswith("nth") and not ORDINALS & _words(text) and not any(w[:1].isdigit() for w in _words(text)):  # R9
            t = {**t, "relation": {"left": "leftmost", "right": "rightmost", "top": "topmost",
                                   "bottom": "bottommost"}[t["relation"].split()[-1]]}   # "the right one of the two"
            notes.append(f"no ordinal word: read as {t['relation']}")
        if RULES and t["relation"] == "none" and {"uncovered", "free", "alone", "own", "unstacked", "bare"} & _words(t["words"] + " " + text):  # R10
            t = {**t, "relation": "with nothing on it"}    # "the uncovered red gear", "lying on its own"
            notes.append("'uncovered/free/on its own': with nothing on it")
        if RULES and t["relation"] == "none":                     # R11: "the biggest object" with the relation left empty
            sup = [r for w, r in SUPERLATIVES.items() if w in t["words"].lower().split()]
            if sup:
                t = {**t, "relation": sup[0]}
                notes.append(f"'{t['words']}': relation {sup[0]} from the words")
        if RULES and t["relation"] == "under" and t["ref_kind"] == "none" and said("with nothing on it", text):  # R12
            t = {**t, "relation": "with nothing on it"}   # "the red gear with nothing on it", read as "under"
        if text and not said(t["relation"], text) and t["relation"] in ("under", "lying on") \
                and said({"under": "below", "lying on": "above"}[t["relation"]], text):
            t = {**t, "relation": {"under": "below", "lying on": "above"}[t["relation"]]}   # "below X": lower in the photo
            notes.append(f"stacking word not in the instruction: read as '{t['relation']}' in the photo")
        if text and not said(t["relation"], text):
            notes.append(f"relation '{t['relation']}' is not in the instruction: dropped")
            t = {**t, "relation": "none", "ref_kind": "none"}
        ids = resolve(g, t, text)
        if RULES and not ids and t["relation"] == "under" and said("below", text) and not STACK_WORDS & _words(text):  # V
            t = {**t, "relation": "below"}            # "below the red gears": lower in the photo, not stacked
            ids = resolve(g, t, text)
        # R13
        if RULES and not ids and t["relation"] == "in" and t["ref_kind"] in ("grey tray", "black tray") \
                and d["place"] in ("nowhere, just pick up", "to the person") \
                and (cmd["action"] != "pick up" or {"need", "needs", "want", "wants", "should", "must"} & _words(text)):
            d = {**d, "place": t["ref_kind"]}         # "I need the red gear in the black tray": where it should go
            t = {**t, "relation": "none", "ref_kind": "none"}
            ids = resolve(g, t, text)
            notes.append(f"nothing is in the {d['place']}: read as where it goes")
        sup = [r for w, r in SUPERLATIVES.items() if w in _words(t["words"])]
        if RULES and len(ids) > 1 and sup and t["relation"] != sup[0]:   # R14: "the smallest object in the grey tray": both
            ids = _extreme(g, ids, sup[0]) if sup[0] in ("biggest", "smallest") else _outermost(g, ids, sup[0])
            notes.append(f"'{sup[0]}' applied after '{t['relation']}'")
        notes.append(f"'{t['words']}' -> {ids}")
        ctx.update(t=t, d=d, ids=list(ids))
        if not ids and QUESTIONS and text and not _all(g, t["kind"], t["color"], phrase=_phrase(t), text=text):
            from .perception import CLASS_DESC                    # "the grey workpiece": a name Apertus may know
            phrase = _phrase(t) or t["words"]
            names = sorted({g.nodes[i].name for i in g.ids(True)})
            yes = [n for n in names if g.yesno(YN_NAME.format(text=text, phrase=phrase, name=n, a="an" if n[:1] in "aeiou" else "a",
                                                             desc=f" ({CLASS_DESC[n]})" if n in CLASS_DESC else ""))]
            if yes:
                t = {**t, "kind": "object"}
                ids = [i for i in resolve(g, t, text) if g.nodes[i].name in yes]
                notes.append(f"Apertus: '{phrase}' can mean {yes} -> {ids}")
        if not ids:
            same_kind = _all(g, t["kind"], "any", phrase=_phrase(t), text=text)
            if not _all(g, t["kind"], t["color"], phrase=_phrase(t), text=text):
                return Decision("refuse", f"There is no {t['words'].removeprefix('the ')} in the workcell.", notes=notes)
            return Decision("refuse", f"I cannot find {t['words']}: none of the {len(same_kind)} matching objects fits that "
                            "description.", notes=notes)
        if RULES and t["how_many"] != "all" and {"both", "all", "every", "each", "everything"} & _words(t["words"]):  # R15
            t = {**t, "how_many": "all"}               # "both grey parts", read as one
            notes.append("'both/all/every' in the words: all of them")
        if RULES and t["how_many"] == "any one" and text and not indefinite(text):  # R16
            t = {**t, "how_many": "one"}               # "the fruit" is one particular fruit, whatever Apertus said
            notes.append("'any one' without a, an, any or some: one particular object")
        plural = _noun(t["words"]) + "s"
        # R17
        if RULES and t["how_many"] == "one" and len(ids) > 1 and {"both", "all", "every", "each"} & _words(text) \
                and any(w.startswith(plural) for w in _words(text)) and len(steps) <= 2:
            t = {**t, "how_many": "all"}                 # "both mandarins"
            notes.append(f"'{plural}' in the instruction: all of them")
        # R18
        if RULES and "both" in _words(text) and any(w.startswith(_noun(t["words"]) + "s") for w in _words(text)) \
                and len(_all(g, t["kind"], t["color"], phrase=_phrase(t), text=text)) > 2 and t["relation"] == "none":
            return Decision("ask_user", f"You said both, but I see {len(_all(g, t['kind'], t['color'], phrase=_phrase(t), text=text))} "
                            f"{_noun(t['words'])}s. Which two do you mean?", notes=notes)
        # R19
        if RULES and t["how_many"] == "one" and len(ids) > 1 and {"both", "all", "every", "each", "two", "three"} & _words(text) \
                and {"first", "second", "third", "other", "another"} & _words(t["words"]) \
                and not {"left", "right", "top", "bottom", "upper", "lower"} & _words(t["words"]):
            t = {**t, "how_many": "any one"}
            notes.append(f"'{t['words']}' is one of a group the instruction names: any not yet moved")
        if RULES and len(ids) > 1 and {"other", "remaining", "rest", "another"} & _words(t["words"]):  # R20
            rest = [i for i in ids if i not in moved and i not in keep]
            if rest:
                ids = rest
                notes.append(f"'other/remaining': not the ones kept or moved -> {ids}")
        ctx.update(t=t, cands=list(ids))
        if t["how_many"] == "any one" and len(ids) > 1:
            ids = [i for i in ids if i not in moved] or ids
        if t["how_many"] == "any one" and len(ids) > 1:     # "a red gear": any will do; take one with nothing on it
            first = next((k for rx, k in ORDER_WORDS if rx.search(text.lower())), None) if RULES else None  # R21
            ids = sorted(ids, key=lambda i: (len(g.above(i)), first(g.centre(i), g.nodes[i].obj.area) if first
                                             else g.centre(i)[0]))[:1]
            notes.append(f"any one of them: {ids[0]}")
        if t["how_many"] == "one" and len(ids) > 1:
            return Decision("ask_user", f"There are {len(ids)} objects that fit '{t['words']}'. Which one do you mean: "
                            f"{_list_positions(g, ids)}?", notes=notes)
        if any(g.nodes[i].touched_by_hand for i in ids):
            return Decision("refuse", f"{_name(g, ids[0]).capitalize()} is in or under a person's hand: the gripper must not "
                            "move onto a person.", notes=notes)
        if t["how_many"] == "all" and any(not g.nodes[i].obj.graspable for i in ids) \
                and any(g.nodes[i].obj.graspable for i in ids):
            notes.append(f"too wide to move, left in place: {[i for i in ids if not g.nodes[i].obj.graspable]}")
            ids = [i for i in ids if g.nodes[i].obj.graspable]
        wide = [i for i in ids if not g.nodes[i].obj.graspable]
        if wide:
            return Decision("refuse", f"{_name(g, wide[0]).capitalize()} is wider than the gripper's 85 mm opening, so the "
                            "robot cannot pick it up.", notes=notes)
        # destination
        dest = None
        if d["place"] == "a tray, not saying which":
            return Decision("ask_user", "Which tray do you mean, the grey tray on the left or the black tray on the right?",
                            notes=notes)
        if d["place"] == "next to an object":            # set it down in the same tray as that object, beside it
            dids = [i for i in _all(g, d["kind"], d["color"], phrase=cmd.get("move_where", ""), text=text) if i not in ids]
            if not dids:
                what = cmd.get("move_where", "") if d["kind"] == "something else" else d["kind"]
                return Decision("refuse", f"There is no {what} to put it next to.", notes=notes)
            dest = g.nodes[dids[0]].tray.replace(" ", "_")
            notes.append(f"destination: next to {dids[0]}, in the {g.nodes[dids[0]].tray}")
        elif d["place"] in ("grey tray", "black tray"):
            dest = d["place"].replace(" ", "_")
        elif d["place"] == "on or in an object":
            dphrase = _dest_phrase(cmd.get("move_where", ""))
            dkind = d["kind"]
            if RULES and KIND_SYNONYMS.get(dkind) and not any(w.startswith(c) for c in KIND_SYNONYMS[dkind].split()  # R22
                                                    for w in _words(text or dphrase)):
                dkind = "something else"            # "on the sponge" read as "saucer"
            if RULES and dkind == "object" and _noun(dphrase) and _noun(dphrase) not in ANY_NOUNS | set(COLOR_WORDS):  # R22
                dkind = HEAD_KIND.get(_noun(dphrase), "something else")      # "on the book"
            d = {**d, "kind": dkind}
            # move_where is one phrase for the whole command ("next to the mandarin on the plate"): only a place
            # outside the workcell's kinds is looked up by it
            dids = [i for i in _all(g, dkind, d["color"], phrase=dphrase if dkind in ("something else", "object") else "",
                                    text=text) if i not in ids]
            if d.get("pin"):                # a place Apertus chose in a follow-up question
                dids = [i for i in d["pin"] if i in g.nodes and i not in ids]
            notes.append(f"destination '{d['kind']}' -> {dids}")
            ctx.update(d=d, dids=list(dids))
            if not dids:
                what = dphrase.removeprefix("the ") if d["kind"] == "something else" else d["kind"]
                return Decision("refuse", f"There is no {d['color'] + ' ' if d['color'] not in ('any', 'other') else ''}"
                                f"{what} to put it on or in.", notes=notes)
            just_moved = [i for i in dids if i in moved]
            if len(dids) > 1 and len(just_moved) == 1:
                notes.append(f"destination: {just_moved[0]}, moved in an earlier step ('it')")
                dids = just_moved
            if len(dids) > 1 and g.reader and len(_words(cmd.get("move_where", ""))) > 3:
                which = _dest_phrase(cmd.get("move_where", ""))     # read the place like an object to pick
                try:
                    dt = g.reader(f"pick up {which}")["steps"][0]["target"]
                    narrowed = [i for i in resolve(g, dt, f"pick up {which}") if i in dids]
                except (KeyError, IndexError, TypeError, ValueError):
                    narrowed = []
                notes.append(f"destination '{which}' -> {narrowed}")
                dids = narrowed if len(narrowed) == 1 else dids
            if len(dids) > 1:
                return Decision("ask_user", f"There are {len(dids)} {d['kind']}s. Which one should it go on: "
                                f"{_list_positions(g, dids)}?", notes=notes)
            dest = dids[0]
            dcls = g.nodes[dest].cls
            if g.nodes[dest].touched_by_hand:
                return Decision("refuse", f"The {dcls} is in or under a person's hand: the gripper must not move onto a "
                                "person.", notes=notes)
            # "put the mug inside the small white part": a known object is a container only if the sentence says
            # in/into/inside right before it; an unknown one (a bowl, a box) by the place phrase as before
            kind_word = re.escape((d["kind"].split() or ["object"])[-1])
            inside = re.search(r"\b(in|into|inside)\s+(?:(?:the|a|an|that|this)\s+)?(?:\w+\s+){0,2}?" + kind_word + r"\b",
                               (text or "").lower())
            says_in = bool(inside) if dcls != OTHER else bool(_words(cmd.get("move_where", "")) & {"in", "into", "inside"})
            opening = CONTAINERS.get(dcls) or (INNER * g.nodes[dest].obj.size_mm[1] if says_in else None)
            if opening:
                big = [i for i in ids if g.nodes[i].obj.size_mm[1] > opening or g.nodes[i].obj.size_mm[0] > 1.6 * opening]
                if big:
                    return Decision("refuse", f"{_name(g, big[0]).capitalize()} does not fit in {_name(g, dest)}.",
                                    notes=notes)
        if t["how_many"] == "all" and dest in ("grey_tray", "black_tray") and len(ids) > 1:
            away = [i for i in ids if g.nodes[i].tray.replace(" ", "_") != dest]
            if away and len(away) < len(ids):
                notes.append(f"already in the {dest.replace('_', ' ')}: {[i for i in ids if i not in away]}")
                ids = away
        # what has to move first, and the order
        key = next((k for rx, k in ORDER_WORDS if rx.search(text.lower())), None) if len(ids) > 1 and RULES else None  # R21
        if key:
            ids = sorted(ids, key=lambda i: key(g.centre(i), g.nodes[i].obj.area))
            notes.append(f"order from the sentence: {ids}")
        order = []
        for i in sorted(ids, key=lambda i: -len(g.above(i)) if not key else 0):
            for top in g.above(i) + [i]:
                if top not in order:
                    order.append(top)
        blockers = [o for o in order if o not in ids]
        stuck = [o for o in blockers if not g.nodes[o].obj.graspable]
        if stuck:
            return Decision("refuse", f"{_name(g, stuck[0]).capitalize()} lies on {_name(g, ids[0])} and is too wide for "
                            "the gripper to move it out of the way.", notes=notes)
        if len(keep) > 1 and keep - set(ids):
            keep = keep - set(ids)            # the kept object is the one of that kind that is not named for moving
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
