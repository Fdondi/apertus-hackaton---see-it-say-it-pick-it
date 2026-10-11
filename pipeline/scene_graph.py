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

from .perception import GRASPABLE, MM_PER_PX, Obj, contact, workcell_roi

ROUND = {"red gear", "yellow gear", "mandarin", "plate", "saucer"}   # not the mug: its handle
COLORS = {"red gear": {"red"}, "yellow gear": {"yellow"}, "red cube": {"red"}, "mug": {"white", "green"},
          "mandarin": {"orange"}, "banana": {"yellow"}, "plate": {"white"}, "saucer": {"yellow", "white", "orange"},
          "grey metal part": {"grey", "silver"}, "white part": {"white"}, "small white part": {"white"}, "hand": set()}


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


class SceneGraph:
    def __init__(self, objs: list[Obj], rgb: np.ndarray):
        self.h, self.w = rgb.shape[:2]
        self.nodes = {o.id: Node(o) for o in objs}
        self.grey_tray = find_grey_tray(rgb, objs)
        self._centre = {o.id: np.array(o.centroid, float) for o in objs}
        for n in self.nodes.values():
            u, v = n.obj.centroid
            n.tray = "grey tray" if self.grey_tray[v, u] else "black tray"
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
                if ab > 0.12 and ab > 2.5 * ba:
                    self._put(a.id, b.id)

    def _put(self, top: str, bottom: str):
        self.nodes[top].on.append(bottom)
        self.nodes[bottom].under.append(top)

    # ------------------------------------------------------------------ queries
    def ids(self, graspable_only: bool = False) -> list[str]:
        return [i for i, n in self.nodes.items() if n.cls != "hand" and (not graspable_only or n.cls in GRASPABLE)]

    def centre(self, i: str) -> np.ndarray:
        return self._centre[i]

    def dist_mm(self, a: str, b: str) -> float:
        """Distance between centres, mm. (The gap between outlines ties too often: two gears touching the
        same mandarin are both 17 mm away; their centres differ by 2 mm, and that is what people judge.)"""
        return float(np.linalg.norm(self.centre(a) - self.centre(b))) * MM_PER_PX

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
            return f"on the {self.nodes[n.on[0]].cls} {n.on[0]} ({n.tray})"
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
                below = [self.nodes[b].cls for b in n.on]
                kinds = list(dict.fromkeys(below))
                what = " and ".join(f"{'a' if below.count(c) == 1 else below.count(c)} {c}{'s' if below.count(c) > 1 else ''}"
                                    for c in kinds)
                named = " and ".join(f"the {c}{'s are' if below.count(c) > 1 else ' is'}" for c in kinds)
                facts.append(f"the {n.cls} lies on {what} ({named} under the {n.cls})")
            if n.touched_by_hand:
                facts.append(f"the person's hand touches the {n.cls}")
        return "; ".join(facts)

    def describe(self) -> str:
        lines = []
        for i, n in sorted(self.nodes.items(), key=lambda x: self.centre(x[0])[0]):
            if n.cls == "hand":
                lines.append(f"- {i}: a person's hand or arm reaching into the workcell")
                continue
            facts = [n.cls, self.where(i), f"{self.region(i)} of the photo"]
            if n.under:
                facts.append("under " + ", ".join(n.under))
            if n.touched_by_hand:
                facts.append("touched or held by the person's hand")
            if n.cls not in GRASPABLE:
                facts.append("too wide for the gripper")
            lines.append(f"- {i}: " + "; ".join(facts))
        return "\n".join(lines)


def amodal(o: Obj) -> np.ndarray:
    """Where the object would be if nothing lay on it: its enclosing circle (round objects) or convex hull."""
    cs, _ = cv2.findContours(o.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack(cs)
    out = np.zeros(o.mask.shape, np.uint8)
    if o.cls in ROUND:
        (x, y), r = cv2.minEnclosingCircle(pts)
        cv2.circle(out, (int(x), int(y)), int(r), 1, -1)
    else:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


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
