"""The scene graph: the objects found by perception plus the facts the instructions talk about.

- where each object is: grey tray (found by colour) or black tray (the dark area on the right)
- what lies on what: from the visible outlines. An object that lies on another hides part of it, so
  it overlaps the region the lower object would cover if it were whole (its enclosing circle for
  round objects, its convex hull otherwise), while the reverse does not happen
- what the person's hand touches
- positions in the photo and distances between objects in mm
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import perception
from .perception import OTHER, Obj, contact, workcell_roi

ROUND = {"red gear", "yellow gear", "mandarin", "plate", "saucer"}   # not the mug: its handle


@dataclass
class Node:
    obj: Obj
    tray: str = ""                                  # "grey tray" | "black tray"
    on: list[str] = field(default_factory=list)     # ids this object lies on
    under: list[str] = field(default_factory=list)  # ids lying on this object
    touched_by_hand: bool = False

    @property
    def id(self) -> str:
        return self.obj.id

    @property
    def cls(self) -> str:
        return self.obj.cls

    @property
    def name(self) -> str:
        return self.obj.name


class SceneGraph:
    def __init__(self, objs: list[Obj], rgb: np.ndarray):
        self.h, self.w = rgb.shape[:2]
        self.nodes = {o.id: Node(o) for o in objs}
        # which unknown objects a phrase like "the screwdriver" or "the fruit" can mean: (phrase, instruction,
        # names) -> names. run.py asks Apertus; without it, the last word of the name must match.
        self.matcher = None
        # reads a phrase like Apertus reads an instruction (run.py sets it): used for a place described by more
        # than its kind ("on the red gear furthest to the left")
        self.reader = None
        # looks at one object's crop: (id, noun) -> is it one? run.py asks Apertus
        self.looker = None
        self.grey_tray = find_grey_tray(rgb, objs)
        self._centre = {o.id: np.array(o.centroid, float) for o in objs}
        for n in self.nodes.values():
            u, v = n.obj.centroid
            n.tray = "grey tray" if self.grey_tray[v, u] else "black tray"
        self.unsure: list[tuple[str, str]] = []
        self._stacking(objs)
        hands = [o for o in objs if o.cls == "hand"]
        for n in self.nodes.values():
            n.touched_by_hand = n.cls != "hand" and any(contact(n.obj.mask, h.mask, r=15) > 0 for h in hands)

    # ------------------------------------------------------------------ stacking
    def _stacking(self, objs: list[Obj]):
        things = [o for o in objs if o.cls != "hand"]
        whole = {o.id: amodal(o) for o in things}
        for a in things:
            for b in things:
                if a is b or b.id in self.nodes[a.id].on or a.id in self.nodes[b.id].on:
                    continue
                if b.id in a.on:                      # perception saw a lying on b
                    self._put(a.id, b.id)
                    continue
                if contact(a.mask, b.mask, r=12) == 0:
                    continue
                ab = (a.mask & whole[b.id]).sum() / a.mask.sum()     # a covers where b would be
                ba = (b.mask & whole[a.id]).sum() / b.mask.sum()
                hides = (a.mask & whole[b.id]).sum() / whole[b.id].sum()    # share of b's whole extent a covers
                if (ab > 0.12 and ab > 2.5 * ba) or (hides > 0.15 and ba < 0.03):   # the banana across the cube
                    self._put(a.id, b.id)
                elif a.id < b.id and max(contact(a.mask, b.mask, r=12), contact(b.mask, a.mask, r=12)) > 0.12:
                    self.unsure.append((a.id, b.id))   # pressed against each other, the outlines do not tell

    def ask_stacking(self, ask):
        """Pairs whose outlines meet along a stretch but do not show which lies on top (a cube on the end of a bar
        hides the end, so the bar's outline just stops): ask(a, b) -> "a on b" | "b on a" | "side by side"."""
        for a, b in self.unsure:
            if b in self.nodes[a].on or a in self.nodes[b].on:
                continue
            ans = ask(self.nodes[a].obj, self.nodes[b].obj)
            if ans == "a on b":
                self._put(a, b)
            elif ans == "b on a":
                self._put(b, a)
        self.unsure = []

    def _put(self, top: str, bottom: str):
        self.nodes[top].on.append(bottom)
        self.nodes[bottom].under.append(top)

    # ------------------------------------------------------------------ queries
    def ids(self, graspable_only: bool = False) -> list[str]:
        return [i for i, n in self.nodes.items() if n.cls != "hand" and (not graspable_only or n.obj.graspable)]

    def centre(self, i: str) -> np.ndarray:
        return self._centre[i]

    def touching(self, a: str, b: str) -> bool:
        """The outlines meet (within 4 mm), or one lies on the other."""
        if b in self.nodes[a].on or a in self.nodes[b].on:
            return True
        return contact(self.nodes[a].obj.mask, self.nodes[b].obj.mask, r=16) > 0

    def spans_trays(self, i: str) -> bool:
        """Lies across the border between the two trays: a good part of it on each side."""
        m = self.nodes[i].obj.mask
        grey = (m & self.grey_tray).sum() / max(1, m.sum())
        return 0.15 < grey < 0.85

    def tray_dist_mm(self, i: str, tray: str) -> float:
        """How far the object's centre is from a tray (0 inside it). The trays stand side by side: columns decide."""
        cols = np.nonzero(self.grey_tray.any(0))[0]
        u = self.centre(i)[0]
        if not len(cols):
            return 0.0
        x0, x1 = cols.min(), cols.max()
        if tray == "grey tray":
            d = 0.0 if x0 <= u <= x1 else min(abs(u - x0), abs(u - x1))
        else:
            d = 0.0 if u > x1 else x1 - u
        return float(d) * perception.MM_PER_PX

    def orientation(self, i: str) -> str:
        """'h' if the object's long side runs left-right in the photo, 'v' if up-down, '' if it is about square."""
        cs, _ = cv2.findContours(self.nodes[i].obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (w, h), ang = cv2.minAreaRect(np.vstack(cs))
        if max(w, h) < 1.3 * max(1.0, min(w, h)):
            return ""
        long_dir = ang if w >= h else ang + 90          # degrees from the image x axis
        a = abs(((long_dir + 90) % 180) - 90)
        return "h" if a < 35 else "v" if a > 55 else ""

    def dist_mm(self, a: str, b: str) -> float:
        """Distance between centres, mm. (The gap between outlines ties too often: two gears touching the
        same mandarin are both 17 mm away; their centres differ by 2 mm, and that is what people judge.)"""
        return float(np.linalg.norm(self.centre(a) - self.centre(b))) * perception.MM_PER_PX

    def above(self, top: str) -> list[str]:
        """Everything lying on top, directly or indirectly, nearest the top first: what must move first."""
        out, todo = [], list(self.nodes[top].under)
        while todo:
            t = todo.pop()
            if t not in out:
                out.append(t)
                todo.extend(self.nodes[t].under)
        return out[::-1]

    def where(self, i: str) -> str:
        """'in the grey tray', 'on the plate (in the black tray)'"""
        n = self.nodes[i]
        if n.on:
            return f"on the {self.nodes[n.on[0]].name} {n.on[0]} ({n.tray})"
        return f"in the {n.tray}"

    def region(self, i: str) -> str:
        u, v = self.centre(i)
        col = "left" if u < self.w / 3 else "right" if u > 2 * self.w / 3 else "middle"
        row = "top" if v < self.h / 3 else "bottom" if v > 2 * self.h / 3 else "middle"
        return f"{row} {col}" if row != col else "centre"

    def stacking_facts(self) -> str:
        """What lies on what, in plain words and without ids: all the parser needs to read
        "put X under Y in Z" the right way. (A full description made Apertus invent relations.)"""
        facts = []
        for i, n in self.nodes.items():
            if n.on:
                below = [self.nodes[b].name for b in n.on]
                kinds = list(dict.fromkeys(below))
                what = " and ".join(f"{'a' if below.count(c) == 1 else below.count(c)} {c}{'s' if below.count(c) > 1 else ''}"
                                    for c in kinds)
                named = " and ".join(f"the {c}{'s are' if below.count(c) > 1 else ' is'}" for c in kinds)
                facts.append(f"the {n.name} lies on {what} ({named} under the {n.name})")
            if n.touched_by_hand:
                facts.append(f"the person's hand touches the {n.name}")
        return "; ".join(facts)

    def describe(self) -> str:
        lines = []
        for i, n in sorted(self.nodes.items(), key=lambda x: self.centre(x[0])[0]):
            if n.cls == "hand":
                lines.append(f"- {i}: a person's hand or arm reaching into the workcell")
                continue
            facts = [n.name, self.where(i), f"{self.region(i)} of the photo"]
            if n.under:
                facts.append("under " + ", ".join(n.under))
            if n.touched_by_hand:
                facts.append("touched or held by the person's hand")
            if not n.obj.graspable:
                facts.append("too wide for the gripper")
            lines.append(f"- {i}: " + "; ".join(facts))
        return "\n".join(lines)


def amodal(o: Obj) -> np.ndarray:
    """Where the object would be if nothing lay on it: its enclosing circle (round objects) or convex hull."""
    cs, _ = cv2.findContours(o.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack(cs)
    out = np.zeros(o.mask.shape, np.uint8)
    if o.cls in ROUND or (o.cls == OTHER and _round(pts, o.mask)):
        (x, y), r = cv2.minEnclosingCircle(pts)
        cv2.circle(out, (int(x), int(y)), int(r), 1, -1)
    else:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        typical = perception.SIZE_MM.get(o.cls, (0,))[0] / perception.MM_PER_PX
        if typical and max(w, h) < 0.92 * typical and max(w, h) > 2 * min(w, h):
            # a long object seen shorter than it is (a bar with a cube on one end): the hidden end lies along its axis,
            # on either side, up to its usual length
            long_ = 2 * typical - max(w, h)
            box = cv2.boxPoints(((cx, cy), (long_, h) if w >= h else (w, long_), ang)).astype(np.int32)
            cv2.fillConvexPoly(out, box, 1)
    return out.astype(bool)


def _round(pts: np.ndarray, mask: np.ndarray) -> bool:
    """An unknown object is round if its visible outline nearly fills its enclosing circle (a ball, a roll of tape)."""
    (_, _), r = cv2.minEnclosingCircle(pts)
    return float(mask.sum()) >= 0.75 * np.pi * r * r


def find_grey_tray(rgb: np.ndarray, objs: list[Obj]) -> np.ndarray:
    """The grey tray: the columns where most free pixels of the workcell are light, unsaturated grey.
    Falls back to the left half of the photo (where it stands in this workcell)."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    grey = (hsv[..., 1] < 45) & (hsv[..., 2] > 90) & (hsv[..., 2] < 215)
    free = workcell_roi(h, w).copy()
    for o in objs:
        free &= ~o.mask
    out = np.zeros((h, w), bool)
    cols = _run((grey & free).sum(0) / np.maximum(1, free.sum(0)))
    if cols is None:
        out[:, : w // 2] = True
        return out
    x0, x1 = cols
    out[:, x0:x1] = True        # the trays stand side by side: the columns decide (rows are noisy under shadows)
    return out


def _run(frac: np.ndarray, thr: float = 0.45, min_len: int = 300) -> tuple[int, int] | None:
    """Longest stretch where the (smoothed) share of grey pixels is above thr."""
    frac = np.convolve(frac, np.ones(25) / 25, mode="same")
    best, start = None, None
    for i, f in enumerate(np.append(frac, 0)):
        if f > thr and start is None:
            start = i
        elif f <= thr and start is not None:
            if i - start >= min_len and (best is None or i - start > best[1] - best[0]):
                best = (start, i)
            start = None
    return best
