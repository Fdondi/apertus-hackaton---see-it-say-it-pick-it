"""Follow-up questions to Apertus in place of keyword rules.

The reading (reason.py) gives a structured command; plan.decide() resolves it on the scene graph. When the readings
disagree, or the command cannot be carried out as read (nothing fits, several fit, no place to put it), the code does
not repair the reading with a keyword rule: it asks Apertus one short question chosen by the kind of problem, and
applies the answer. Each question is plain English about the sentence or about the objects on the table.
"""
from __future__ import annotations

import copy
import json

from .reason import YES_NO

ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
REL = {"to the left of": "to the left of {ref}", "to the right of": "to the right of {ref}", "above": "above {ref} in the photo",
       "below": "below {ref} in the photo", "closest to": "closest to {ref}", "farthest from": "farthest from {ref}",
       "next to": "next to {ref}", "lying on": "lying on top of {ref}", "under": "with {ref} lying on top of it",
       "in": "in {ref}", "with nothing on it": "with nothing lying on top of it", "leftmost": "furthest to the left",
       "rightmost": "furthest to the right", "topmost": "highest in the photo", "bottommost": "lowest in the photo",
       "nth from the left": "{nth} from the left", "nth from the right": "{nth} from the right",
       "nth from the top": "{nth} from the top", "nth from the bottom": "{nth} from the bottom",
       "in the middle": "in the middle", "between": "between {ref} and {ref2}", "biggest": "biggest", "smallest": "smallest",
       "in the person's hand": "held in the person's hand", "in the top left corner": "in the top left corner",
       "in the top right corner": "in the top right corner", "in the bottom left corner": "in the bottom left corner",
       "in the bottom right corner": "in the bottom right corner", "oriented left-right": "lying left to right",
       "oriented up-down": "lying up and down", "touching": "touching {ref}", "not lying on": "not lying on {ref}",
       "not in": "not in {ref}", "across both trays": "lying across both trays"}
PLACE = {"nowhere, just pick up": "and hold it", "grey tray": "into the grey tray", "black tray": "into the black tray",
         "a tray, not saying which": "into a tray (which one is not said)", "to the person": "and hand it to the person",
         "onto or next to the person's hand or body": "onto or next to the person's hand or body"}


def _thing(kind: str, color: str, words: str = "") -> str:
    c = "" if color in ("any", "other") else color + " "
    if kind == "something else":
        return f"{c}thing called '{words}'" if words else f"{c}other object"
    if kind == "object":
        return f"{c}object"
    return c + kind


def _ref(kind: str, color: str, words: str = "") -> str:
    if kind in ("none", ""):
        return "something"
    if kind in ("grey tray", "black tray"):
        return "the " + kind
    if kind == "person's hand":
        return "the person's hand"
    return "the " + _thing(kind, color, words)


def say_target(t: dict) -> str:
    qty = {"one": "the one particular", "any one": "any one", "all": "all of the"}.get(t.get("how_many"), "the")
    noun = _thing(t.get("kind", "object"), t.get("color", "any"), t.get("words", ""))
    if t.get("how_many") == "all" and not noun.endswith("s"):
        noun += "s"
    rel = t.get("relation", "none")
    out = f"{qty} {noun}"
    if rel != "none":
        out += " " + REL.get(rel, rel).format(ref=_ref(t.get("ref_kind", "none"), t.get("ref_color", "any")),
                                             ref2=_ref(t.get("ref2_kind", "none"), t.get("ref2_color", "any")),
                                             nth=ORD.get(t.get("n") or 1, f"{t.get('n')}th"))
    for e in t.get("except", []) or []:
        out += f", except the {_thing(e['kind'], e['color'])}"
    return out


def say_dest(d: dict) -> str:
    if d["place"] == "on or in an object":
        return f"onto or into the {_thing(d['kind'], d['color'])}"
    if d["place"] == "next to an object":
        return f"next to the {_thing(d['kind'], d['color'])}"
    return PLACE.get(d["place"], d["place"])


def say_reading(r: dict) -> str:
    verb = {"pick up": "pick up", "put or move": "move", "give to the person": "give", "throw or toss": "throw",
            "drop": "drop", "push or slide": "push", "pull out from under": "pull out from under",
            "take from the person": "take from the person's hand"}.get(r["action"], r["action"])
    steps = [f"{verb} {say_target(st['target'])} {say_dest(st['destination'])}".strip() for st in r["steps"]]
    out = "; then ".join(steps)
    if r.get("keep_still"):
        out += "; and leave " + " and ".join("the " + _thing(k["kind"], k["color"]) for k in r["keep_still"]) + " where it is"
    if r.get("vague_reference"):
        out += " (the object is only pointed at, not named)"
    if r.get("holds_two_at_once"):
        out += " (holding them all in the gripper at the same time)"
    if r.get("other_danger"):
        out += " (this is dangerous)"
    return out[:1].upper() + out[1:] + "."


# ----------------------------------------------------------------------------- the table, as Apertus sees it
def labels(g) -> dict:
    ids = sorted(g.ids(), key=lambda i: g.centre(i)[0])
    out = {}
    for i in ids:
        same = [j for j in ids if g.nodes[j].name == g.nodes[i].name]
        out[i] = g.nodes[i].name + (f" {same.index(i) + 1}" if len(same) > 1 else "")
    return out


def facts(g, i: str, lab: dict) -> str:
    n = g.nodes[i]
    f = [f"{g.region(i)} of the photo", f"in the {n.tray}"]
    f += [f"lies on the {lab[b]}" for b in n.on if b in lab]
    f += [f"the {lab[a]} lies on it" for a in n.under if a in lab] or ["nothing lies on it"]
    if n.touched_by_hand:
        f.append("a person's hand holds or touches it")
    if not n.obj.graspable:
        f.append("too wide for the gripper")
    return ", ".join(f)


def table(g, lab: dict, ids=None) -> str:
    ids = ids or sorted(lab, key=lambda i: g.centre(i)[0])
    return "\n".join(f"- {lab[i]}: {facts(g, i, lab)}" for i in ids)


# ----------------------------------------------------------------------------- asking
def _choice(llm, prompt: str, options: list[str], think: bool = True) -> int:
    """Index of the option Apertus picks (letters), after one or two sentences of reasoning."""
    letters = "ABCDEFGHIJKL"[:len(options)]
    body = prompt + "\n" + "\n".join(f"{c}. {o}" for c, o in zip(letters, options))
    props = {"answer": {"enum": list(letters)}}
    req = ["answer"]
    if think:
        body += "\nFirst say in one or two sentences what the words mean, then give the letter."
        props = {"reason": {"type": "string", "maxLength": 400}, **props}
        req = ["reason", "answer"]
    schema = {"type": "object", "additionalProperties": False, "required": req, "properties": props}
    return letters.index(llm.json([{"role": "user", "content": body}], schema, max_tokens=220)["answer"])


def choose(llm, prompt: str, options: list[str]) -> int | None:
    """Asked twice, options in opposite orders; the index only when both orders agree."""
    a = _choice(llm, prompt, options)
    rev = options[::-1]
    b = len(options) - 1 - _choice(llm, prompt, rev)
    return a if a == b else None


def yes(llm, prompt: str) -> bool:
    return llm.json([{"role": "user", "content": prompt + " Answer yes or no."}], YES_NO, max_tokens=10)["answer"] == "yes"


HEAD = 'A person tells a robot arm: "{text}"\n'


def q_reading(llm, text: str, readings: list[dict]) -> int | None:
    """The readings disagree: which plain-English reading says what the person asked?"""
    opts = [say_reading(r) for r in readings]
    return choose(llm, HEAD.format(text=text) + "Which of these says exactly what the person asks for, no more and no less?",
                  opts)


COUNT = ["one particular object (if several fit, the robot must ask which)", "any one of them (whichever)",
         "all of them"]


def q_count(llm, text: str, words: str, n: int) -> str | None:
    i = choose(llm, HEAD.format(text=text) + f'On the table, {n} objects fit "{words}". How many of them does the person '
               "want the robot to take?", COUNT)
    return None if i is None else ["one", "any one", "all"][i]


def q_where_words(llm, text: str, which: str) -> str | None:
    i = choose(llm, HEAD.format(text=text) + f'What do the words "{which}" tell the robot?',
               ["which object to take (where it is now)", "where the object should go", "neither"])
    return None if i is None else ["which", "goal", "neither"][i]


def q_pick(llm, g, lab: dict, text: str, words: str, ids: list[str]) -> str | None:
    """Several objects fit: which one do the words mean, or can the words not tell?"""
    opts = [f"the {lab[i]} ({facts(g, i, lab)})" for i in ids] + ["the words fit several of them equally well"]
    i = choose(llm, HEAD.format(text=text) + "Objects on the table:\n" + table(g, lab) +
               f'\nWhich object does "{words}" mean?', opts)
    if i is None or i == len(ids):
        return None
    return ids[i]


def q_dest(llm, g, lab: dict, text: str, what: str, ids: list[str] | None = None):
    """Where should it end up? -> ('tray', name) | ('obj', id) | ('person', kind) | None"""
    ids = ids if ids is not None else sorted(lab, key=lambda i: g.centre(i)[0])
    opts = ["the grey tray", "the black tray"] + [f"on or in the {lab[i]}" for i in ids] + \
           ["on, in or next to the person's hand, body or clothes", "handed to the person", "a place that is not on this table",
            "the instruction does not say"]
    i = choose(llm, HEAD.format(text=text) + "Objects on the table:\n" + table(g, lab) +
               f"\nWhere should the robot put {what}?", opts)
    if i is None:
        return None
    if i < 2:
        return ("tray", ["grey tray", "black tray"][i])
    i -= 2
    if i < len(ids):
        return ("obj", ids[i])
    return ("person", ["body", "hand over", "elsewhere", "unsaid"][i - len(ids)])


DANGER3 = ("If the robot does exactly what is asked, could an object fall, tip over, break or be thrown, or could the "
           "robot touch or hit a person?")
