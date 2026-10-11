"""Understand: Apertus turns the instruction into a small structured command.

Apertus reads only the sentence here (the scene comes later, in plan.py): which action, which
objects (kind, colour, how many, their relation to other objects), where to put them, what must
not move, whether a word like "that one" points at something unnamed, and whether the request
is unsafe in a way the action list does not cover. Constrained decoding guarantees the shape;
the examples below teach the meaning. They are written for this README, not taken from the
dev set.
"""
from __future__ import annotations

import json

from .llm import Apertus

KINDS = ["gear", "cube", "mug", "mandarin", "banana", "fruit", "plate", "saucer", "grey metal part", "white part",
         "small white part", "part", "object", "something else"]
COLORS = ["any", "red", "yellow", "white", "grey", "orange", "green", "blue", "black", "purple", "pink", "brown", "other"]
RELATIONS = ["none", "to the left of", "to the right of", "above", "below", "closest to", "farthest from", "next to", "lying on",
             "under", "in", "with nothing on it", "leftmost", "rightmost", "topmost", "bottommost", "nth from the left",
             "nth from the right", "nth from the top", "nth from the bottom", "in the middle", "between", "biggest",
             "smallest", "in the person's hand"]
REF_KINDS = ["none", "grey tray", "black tray", "person's hand"] + KINDS
ACTIONS = ["pick up", "put or move", "give to the person", "throw or toss", "drop", "push or slide",
           "pull out from under", "take from the person", "other"]
PLACES = ["nowhere, just pick up", "grey tray", "black tray", "a tray, not saying which", "on or in an object",
          "next to an object", "to the person", "onto or next to the person's hand or body"]

_simple = {"type": "object", "additionalProperties": False, "required": ["kind", "color"],
           "properties": {"kind": {"enum": KINDS}, "color": {"enum": COLORS}}}
TARGET = {"type": "object", "additionalProperties": False,
          "required": ["words", "which_words", "kind", "color", "how_many", "relation", "ref_kind", "ref_color", "n",
                       "except"],
          "properties": {"words": {"type": "string", "maxLength": 80},
                         "which_words": {"type": "string", "maxLength": 60},
                         "kind": {"enum": KINDS}, "color": {"enum": COLORS},
                         "how_many": {"enum": ["one", "any one", "all"]},
                         "relation": {"enum": RELATIONS}, "ref_kind": {"enum": REF_KINDS}, "ref_color": {"enum": COLORS},
                         "ref2_kind": {"enum": REF_KINDS}, "ref2_color": {"enum": COLORS},   # only for "between"
                         "n": {"type": "integer", "minimum": 0, "maximum": 6},
                         "except": {"type": "array", "maxItems": 3, "items": _simple}}}
DEST = {"type": "object", "additionalProperties": False, "required": ["place", "kind", "color"],
        "properties": {"place": {"enum": PLACES}, "kind": {"enum": KINDS}, "color": {"enum": COLORS}}}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["move_what", "move_where", "meaning", "action", "steps", "keep_still", "vague_reference",
                       "holds_two_at_once", "other_danger"],
          "properties": {"move_what": {"type": "string", "maxLength": 120},
                         "move_where": {"type": "string", "maxLength": 80},
                         "meaning": {"type": "string", "maxLength": 240},
                         "action": {"enum": ACTIONS},
                         "steps": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
                             "type": "object", "additionalProperties": False, "required": ["target", "destination"],
                             "properties": {"target": TARGET, "destination": DEST}}},
                         "keep_still": {"type": "array", "maxItems": 3, "items": _simple},
                         "vague_reference": {"type": "boolean"},
                         "holds_two_at_once": {"type": "boolean"},
                         "other_danger": {"type": "boolean"}}}

SYSTEM = """You translate instructions for a robot arm into JSON. The robot works in a cell with a grey tray (left) and a black tray (right), and these objects: red gears, a yellow gear, a red cube, a mug, a mandarin, a banana, a plate, a saucer, grey metal parts, a white part, a small white part. Sometimes a person's hand is in the cell.

Fields:
- move_what: the words naming the object(s) to act on, with every word that says where they are NOW ("the red gear under the yellow gear", "everything in the grey tray except the mug").
- move_where: the words saying where they must go ("in the black tray", "on the plate"), or "" if nothing is said. Only the last place phrase of a "put"/"move" sentence is where the object goes; a phrase like "under Y", "on Y", "in the grey tray" that comes right after the object name, before another place phrase, says where the object is now.
- meaning: the instruction in your own words, with every "it", "one" and "them" replaced by what it refers to.
- action: the verb. With several verbs ("pick up the mug and throw it"), the most dangerous one. "pick up", "grab", "take" (from the table) = pick up. "put", "place", "move", "bring", "transfer" = put or move. "give me", "hand me" = give to the person. "throw", "toss", "fling" = throw or toss. "drop", "let fall" = drop. "push", "slide", "knock", "shove" = push or slide. "pull/yank/take X out from under Y" = pull out from under. Taking something out of a person's hand = take from the person.
- steps: one step per object or group named for moving, in the order of the sentence. target = what to pick; destination = where it goes.
  - words: the words naming this target. which_words: the part of them that says which one, if any ("free", "on the left", "closest to the mug", "under the yellow gear", "second from the right"), or "".
  - kind: the noun. "cog" = gear, "block" = cube, "cup" = mug, "orange", "tangerine", "clementine" = mandarin, "dish" = plate, "metal bar", "grey part" = grey metal part, "thing", "item", "everything", "anything" = object, "fruit" = fruit, "part" = part. A noun that is not in the cell (bottle, pen, cylinder, ball, ...) = something else.
  - color: the colour word in the instruction, or "any".
  - how_many: "all" for all, every, everything, both, each, the gears (plural); "any one" for a, an, any, one of the, some (any single one will do); otherwise "one" (the, this, that: one particular object).
  - relation, ref_kind, ref_color, n: how the instruction picks the object out. "on the left" (no other object named) = leftmost; "the second from the right" = nth from the right with n = 2; "to the left of the mug" = to the left of, ref_kind mug; "closest to", "nearest to" = closest to; "next to", "beside" = next to; "lying on", "on top of", "sitting on" = lying on; "under", "beneath", "underneath" another object, or "with X on it", "that has X on top" = under, ref_kind X; "above X" and "below X" = higher or lower than X in the photo (above, below); "the biggest", "the largest" = biggest; "the smallest" = smallest; "the lowest", "at the bottom" = bottommost; "the highest", "at the top", "the upper" = topmost; "the middle one", "in the middle", "the centre one" = in the middle; "between X and Y" = between, ref_kind X, and ref2_kind Y (the only relation with a second object); "in the grey tray" when it says where the object is now = in, ref_kind grey tray; "with nothing on it", "that is free" = with nothing on it. "above" and "below" without another object mean higher or lower in the photo. Use relation none, ref_kind none, ref_color any, n 0 when nothing like this is said.
  - except: the objects excluded by "except", "but not", "other than".
  - destination.place: grey tray, black tray, "a tray, not saying which" (for "the tray", "a tray"), on or in an object (then kind and color of that object), next to an object ("next to the mug", "beside the plate"; kind and color of that object), onto or next to the person's hand or body ("on my hand", "next to my hand", "on my head"), to the person ("give me"), or nowhere, just pick up.
- keep_still: objects the instruction says must not move ("without moving the mug", "leave the cube where it is", "do not touch the plate"). If the words "leave it where it is" refer to the object being moved, put that object here too.
- vague_reference: true when the object is only pointed at ("that one", "this", "it", "the thing there") and neither named nor described ("the object in my hand", "the thing on the plate" are described: false).
- When the sentence is "put X under Y in/on Z", X is the object that lies under Y now, and Z is where it goes. A line "In the scene: ..." before the instruction says what lies on what.
- holds_two_at_once: true when the robot would have to hold two or more objects at the same time ("at the same time", "together in one grab", "both at once"). Moving them one after the other is false.
- other_danger: true when the request could hurt the person or make an object fall, fly or break in a way the action list does not already say (for example "put the cube on my head", "move the gear as fast as you can towards me", "stack everything as high as possible"). Ordinary picking and placing is not dangerous."""

EXAMPLES = [
    ("grab the banana",
     {"move_what": "the banana", "move_where": "", "meaning": "Pick up the banana.", "action": "pick up",
      "steps": [{"target": {"words": "the banana", "which_words": "", "kind": "banana", "color": "any", "how_many": "one", "relation": "none",
                            "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("place the cog that is nearest to the plate on the saucer",
     {"move_what": "the cog that is nearest to the plate", "move_where": "on the saucer", "meaning": "Put the gear closest to the plate on the saucer.", "action": "put or move",
      "steps": [{"target": {"words": "the cog that is nearest to the plate", "which_words": "nearest to the plate", "kind": "gear", "color": "any", "how_many": "one",
                            "relation": "closest to", "ref_kind": "plate", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "on or in an object", "kind": "saucer", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("pick up the third red cog from the left",
     {"move_what": "the third red cog from the left", "move_where": "", "meaning": "Pick up the red gear that is third counting from the left.", "action": "pick up",
      "steps": [{"target": {"words": "the third red cog from the left", "which_words": "third from the left", "kind": "gear", "color": "red", "how_many": "one",
                            "relation": "nth from the left", "ref_kind": "none", "ref_color": "any", "n": 3, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("transfer every part in the black tray to the grey tray, but not the white part",
     {"move_what": "every part in the black tray, but not the white part", "move_where": "to the grey tray", "meaning": "Move each part that is in the black tray, except the white part, into the grey tray.", "action": "put or move",
      "steps": [{"target": {"words": "every part in the black tray", "which_words": "in the black tray", "kind": "part", "color": "any", "how_many": "all",
                            "relation": "in", "ref_kind": "black tray", "ref_color": "any", "n": 0,
                            "except": [{"kind": "white part", "color": "any"}]},
                 "destination": {"place": "grey tray", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("pick up the item to the right of the banana",
     {"move_what": "the item to the right of the banana", "move_where": "", "meaning": "Pick up the object that is to the right of the banana in the photo.", "action": "pick up",
      "steps": [{"target": {"words": "the item to the right of the banana", "which_words": "to the right of the banana", "kind": "object", "color": "any", "how_many": "one",
                            "relation": "to the right of", "ref_kind": "banana", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("move the gear under the mandarin onto the saucer",
     {"move_what": "the gear under the mandarin", "move_where": "onto the saucer", "meaning": "A gear lies under the mandarin; move that gear onto the saucer.", "action": "put or move",
      "steps": [{"target": {"words": "the gear under the mandarin", "which_words": "under the mandarin", "kind": "gear", "color": "any", "how_many": "one",
                            "relation": "under", "ref_kind": "mandarin", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "on or in an object", "kind": "saucer", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("grab the grey part that has nothing on top of it",
     {"move_what": "the grey part that has nothing on top of it", "move_where": "",
      "meaning": "Pick up the grey metal part with no object lying on it.", "action": "pick up",
      "steps": [{"target": {"words": "the grey part that has nothing on top of it", "which_words": "that has nothing on top of it", "kind": "grey metal part", "color": "any",
                            "how_many": "one", "relation": "with nothing on it", "ref_kind": "none", "ref_color": "any",
                            "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("put a gear in the tray",
     {"move_what": "a gear", "move_where": "in the tray", "meaning": "Put any one gear into a tray; the tray is not named.",
      "action": "put or move",
      "steps": [{"target": {"words": "a gear", "which_words": "", "kind": "gear", "color": "any", "how_many": "any one", "relation": "none",
                            "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "a tray, not saying which", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("move the grey part that has the cube on it to the black tray",
     {"move_what": "the grey part that has the cube on it", "move_where": "to the black tray",
      "meaning": "The cube lies on a grey metal part; move that grey metal part to the black tray.", "action": "put or move",
      "steps": [{"target": {"words": "the grey part that has the cube on it", "which_words": "that has the cube on it", "kind": "grey metal part", "color": "any",
                            "how_many": "one", "relation": "under", "ref_kind": "cube", "ref_color": "any", "n": 0,
                            "except": []},
                 "destination": {"place": "black tray", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("pick up the item between the banana and the saucer",
     {"move_what": "the item between the banana and the saucer", "move_where": "",
      "meaning": "Pick up the object that lies between the banana and the saucer.", "action": "pick up",
      "steps": [{"target": {"words": "the item between the banana and the saucer", "which_words": "between the banana and the saucer", "kind": "object", "color": "any",
                            "how_many": "one", "relation": "between", "ref_kind": "banana", "ref_color": "any",
                            "ref2_kind": "saucer", "ref2_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("pick up the orange fruit",
     {"move_what": "the orange fruit", "move_where": "", "meaning": "Pick up the fruit that is orange.", "action": "pick up",
      "steps": [{"target": {"words": "the orange fruit", "which_words": "", "kind": "fruit", "color": "orange", "how_many": "one",
                            "relation": "none", "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("grab the banana and fling it into the grey tray",
     {"move_what": "the banana", "move_where": "into the grey tray",
      "meaning": "Pick up the banana and throw it into the grey tray.", "action": "throw or toss",
      "steps": [{"target": {"words": "the banana", "which_words": "", "kind": "banana", "color": "any", "how_many": "one",
                            "relation": "none", "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "grey tray", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("put the banana on my hand",
     {"move_what": "the banana", "move_where": "on my hand", "meaning": "Place the banana on the person's hand.",
      "action": "put or move",
      "steps": [{"target": {"words": "the banana", "which_words": "", "kind": "banana", "color": "any", "how_many": "one",
                            "relation": "none", "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "onto or next to the person's hand or body", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("grab the cube and the banana together in one go",
     {"move_what": "the cube and the banana", "move_where": "", "meaning": "Pick up the cube and the banana at the same time.",
      "action": "pick up",
      "steps": [{"target": {"words": "the cube", "which_words": "", "kind": "cube", "color": "any", "how_many": "one",
                            "relation": "none", "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}},
                {"target": {"words": "the banana", "which_words": "", "kind": "banana", "color": "any", "how_many": "one",
                            "relation": "none", "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": True, "other_danger": False}),
    ("hand me this",
     {"move_what": "this", "move_where": "to me", "meaning": "Give the person an object they point at but do not name.", "action": "give to the person",
      "steps": [{"target": {"words": "this", "which_words": "", "kind": "object", "color": "any", "how_many": "one", "relation": "none",
                            "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "to the person", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": True, "holds_two_at_once": False, "other_danger": False}),
    ("pick up the green bottle",
     {"move_what": "the green bottle", "move_where": "", "meaning": "Pick up a green bottle.", "action": "pick up",
      "steps": [{"target": {"words": "the green bottle", "which_words": "", "kind": "something else", "color": "green", "how_many": "one",
                            "relation": "none", "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("toss the mandarin into the black tray",
     {"move_what": "the mandarin", "move_where": "into the black tray", "meaning": "Throw the mandarin into the black tray.", "action": "throw or toss",
      "steps": [{"target": {"words": "the mandarin", "which_words": "", "kind": "mandarin", "color": "any", "how_many": "one", "relation": "none",
                            "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "black tray", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("put the cube lying on the white part in the grey tray",
     {"move_what": "the cube lying on the white part", "move_where": "in the grey tray", "meaning": "The cube lies on the white part; put that cube in the grey tray.", "action": "put or move",
      "steps": [{"target": {"words": "the cube lying on the white part", "which_words": "lying on the white part", "kind": "cube", "color": "any", "how_many": "one",
                            "relation": "lying on", "ref_kind": "white part", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "grey tray", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("put the white part under the banana in the black tray",
     {"move_what": "the white part under the banana", "move_where": "in the black tray", "meaning": "The white part is under the banana; move that white part into the black tray.", "action": "put or move",
      "steps": [{"target": {"words": "the white part under the banana", "which_words": "under the banana", "kind": "white part", "color": "any", "how_many": "one",
                            "relation": "under", "ref_kind": "banana", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "black tray", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("slide the banana off the plate",
     {"move_what": "the banana on the plate", "move_where": "", "meaning": "Push the banana so that it slides off the plate.", "action": "push or slide",
      "steps": [{"target": {"words": "the banana", "which_words": "off the plate", "kind": "banana", "color": "any", "how_many": "one", "relation": "lying on",
                            "ref_kind": "plate", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("move the yellow gear onto the saucer but keep the gear exactly where it is",
     {"move_what": "the yellow gear", "move_where": "onto the saucer", "meaning": "Move the yellow gear onto the saucer, and also leave the yellow gear where it is: a contradiction.",
      "action": "put or move",
      "steps": [{"target": {"words": "the yellow gear", "which_words": "", "kind": "gear", "color": "yellow", "how_many": "one", "relation": "none",
                            "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "on or in an object", "kind": "saucer", "color": "any"}}],
      "keep_still": [{"kind": "gear", "color": "yellow"}], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("take the banana and the cube to the grey tray without touching the mug",
     {"move_what": "the banana and the cube", "move_where": "to the grey tray", "meaning": "Put the banana in the grey tray, then the cube, and do not move the mug.", "action": "put or move",
      "steps": [{"target": {"words": "the banana", "which_words": "", "kind": "banana", "color": "any", "how_many": "one", "relation": "none",
                            "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "grey tray", "kind": "object", "color": "any"}},
                {"target": {"words": "the cube", "which_words": "", "kind": "cube", "color": "any", "how_many": "one", "relation": "none",
                            "ref_kind": "none", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "grey tray", "kind": "object", "color": "any"}}],
      "keep_still": [{"kind": "mug", "color": "any"}], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("yank the cube out from under the mandarin",
     {"move_what": "the cube under the mandarin", "move_where": "", "meaning": "Pull the cube out from under the mandarin, which would make the mandarin fall.", "action": "pull out from under",
      "steps": [{"target": {"words": "the cube", "which_words": "out from under the mandarin", "kind": "cube", "color": "any", "how_many": "one", "relation": "under",
                            "ref_kind": "mandarin", "ref_color": "any", "n": 0, "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
    ("grab the cube from my fingers",
     {"move_what": "the cube in the person's fingers", "move_where": "", "meaning": "Take the cube out of the person's hand.", "action": "take from the person",
      "steps": [{"target": {"words": "the cube", "which_words": "from my fingers", "kind": "cube", "color": "any", "how_many": "one",
                            "relation": "in the person's hand", "ref_kind": "person's hand", "ref_color": "any", "n": 0,
                            "except": []},
                 "destination": {"place": "nowhere, just pick up", "kind": "object", "color": "any"}}],
      "keep_still": [], "vague_reference": False, "holds_two_at_once": False, "other_danger": False}),
]


def messages(instruction: str, scene: str = "") -> list:
    msgs = [{"role": "system", "content": SYSTEM}]
    for text, ans in EXAMPLES:
        msgs += [{"role": "user", "content": f'Instruction: "{text}"'},
                 {"role": "assistant", "content": json.dumps(ans)}]
    ctx = f"In the scene: {scene}.\n" if scene else ""
    msgs.append({"role": "user", "content": f'{ctx}Instruction: "{instruction}"'})
    return msgs


def parse(llm: Apertus, instruction: str, scene: str = "") -> dict:
    """One reading of the instruction (greedy)."""
    return llm.json(messages(instruction, scene), SCHEMA, max_tokens=900)


def parse_samples(llm: Apertus, instruction: str, scene: str = "", n: int = 4) -> list[dict]:
    """More readings, sampled, for a vote. Replies cut off by max_tokens are skipped."""
    out = []
    for r in llm.complete(messages(instruction, scene), schema=SCHEMA, max_tokens=900, temperature=0.7, n=n):
        try:
            out.append(json.loads(r))
        except json.JSONDecodeError:
            pass
    return out


# ----------------------------------------------------------------------------- a second look at safety
DANGERS = ["none", "throw or toss", "drop", "push, slide or knock", "pull out from under", "take from a person",
           "put on or near a person", "move fast or swing", "hold two objects at once", "other danger"]
DANGER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["dangerous_action"],
                 "properties": {"dangerous_action": {"enum": DANGERS}}}
DANGER_PROMPT = """A robot arm with a two-finger gripper gets this instruction from a person standing at the table:
"{text}"
Does the instruction ask the robot to do something dangerous? Answer with the dangerous action it asks for, or "none".
- throw or toss; drop (let something fall); push, slide or knock something (off, over, away); pull something out from under another object; take something from a person's hand; put something on or near a person's hand or body; move fast or swing; hold two objects at once; any other danger to a person or of an object falling or breaking.
- Ordinary picking up and putting down, one object after another, is "none". So is a request to avoid something ("without dropping it", "do not touch the mug")."""


def danger_check(llm: Apertus, instruction: str) -> str:
    """One short question, separate from the long reading: the long prompt sometimes keeps only the first verb of
    "pick up the mug and throw it"."""
    return llm.json([{"role": "user", "content": DANGER_PROMPT.format(text=instruction)}], DANGER_SCHEMA,
                    max_tokens=30)["dangerous_action"]
