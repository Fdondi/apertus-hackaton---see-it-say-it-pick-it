"""See: find every object in the photo and give it a name and a pixel mask.

1. SAM 2.1 (automatic mask generation, Apache 2.0) proposes regions, without names.
2. Cheap geometry drops what cannot be an object: specks, the trays themselves, the rails
   outside the workcell, thin strips of tray wall.
3. Apertus names each region from a crop with the region outlined, told the size of the
   region and of each workcell object (the camera is fixed: 0.25 mm per pixel on the table).
   It gives a first and a second choice.
4. Clean-up with what we know about the workcell: object sizes, how many of each object exist
   (three red gears, two grey metal parts, one of everything else), and that a region inside
   another object is either a piece of it (the hub of a gear, the centre of the saucer) or a
   separate object lying on it; Apertus decides that from a crop.

The result is cached per image (masks as packed bits), so only the first run costs time.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .llm import Apertus, image_part

VERSION = "p3"
MM_PER_PX = 0.25
# long side seen from above, mm: (typical, smallest, largest) for a fully visible object, measured on the dev photos
SIZE_MM = {"red gear": (91, 80, 105), "yellow gear": (48, 38, 60), "red cube": (87, 75, 105),
           "mug": (110, 85, 130), "mandarin": (74, 60, 90), "banana": (220, 160, 260), "plate": (193, 170, 250),
           "saucer": (124, 110, 145), "grey metal part": (135, 100, 210), "white part": (143, 120, 170),
           "small white part": (80, 50, 100)}
OBJECT_CLASSES = list(SIZE_MM)
COUNT = {**{c: 1 for c in OBJECT_CLASSES}, "red gear": 3, "grey metal part": 2}   # the workcell's inventory
GRASPABLE = set(OBJECT_CLASSES) - {"plate", "saucer"}
HAND, EMPTY, OTHER = "hand or arm", "tray or table", "other object"
LABELS = OBJECT_CLASSES + [HAND, EMPTY, OTHER]

CLASSIFY_PROMPT = """This is a crop of a photo of a robot workcell, taken from straight above. The magenta outline marks one region, about {long:.0f} x {short:.0f} mm.
Which object does the region inside the magenta outline belong to? The objects of the workcell, with their size seen from above:
red gear (about 90 mm), yellow gear (about 48 mm, small), red cube (about 87 mm), mug (about 110 mm with its handle), mandarin (about 74 mm), banana (about 220 mm long), plate (white, about 195 mm), saucer (yellow rim and white centre, about 125 mm), grey metal part (flat aluminium bar with holes, about 135 x 47 mm), white part (flat white plastic plate, about 143 x 98 mm), small white part (white block with a dark side, about 80 x 55 mm).
Other answers: "hand or arm" for a person's hand, arm, wrist or sleeve; "tray or table" when the region is empty (the floor or wall of a tray, the table, a mark, a shadow or a reflection); "other object" for any other object.
If the outline covers only a piece of an object (part of a gear, the inside of a mug, the centre of a saucer, the pattern of a plate), answer with that object. A region can be smaller than its object when something lies on it.
Give your best answer and a second choice."""
CLASSIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label", "second_choice"],
                   "properties": {"label": {"enum": LABELS}, "second_choice": {"enum": LABELS}}}

PART_PROMPT = """This crop shows a {parent}, seen from straight above. The magenta outline marks a region on it.
Is that region a separate object lying on the {parent} (for example a fruit, a gear or a part placed on it), or is it just a part of the {parent} itself (its centre, its rim, its pattern, its hub)?"""
NEXT_PROMPT = """This crop shows a {parent}, seen from straight above. The magenta outline marks a region next to it that looks like a {label}.
Is that region a separate object (perhaps partly hidden under the {parent}), or is it part of the {parent} itself (its side, wall, handle or rim) or its shadow?"""
NEXT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
               "properties": {"answer": {"enum": ["separate object", "part of it or its shadow"]}}}
PART_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
               "properties": {"answer": {"enum": ["separate object", "part of it"]}}}


@dataclass
class Obj:
    id: str
    cls: str
    mask: np.ndarray = field(repr=False)
    on: list = field(default_factory=list)   # ids of the objects this one lies on, found while naming
    score: float = 0.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @property
    def centroid(self) -> tuple[int, int]:
        ys, xs = np.nonzero(self.mask)
        return int(round(xs.mean())), int(round(ys.mean()))

    @property
    def grasp_pixel(self) -> tuple[int, int]:
        """The visible pixel deepest inside the object: always on it, and the safe point for the robot."""
        return deepest_point(self.mask)

    @property
    def size_mm(self) -> tuple[float, float]:
        return size_mm(self.mask)


# ----------------------------------------------------------------------------- mask helpers
def deepest_point(mask: np.ndarray) -> tuple[int, int]:
    m = np.pad(mask.astype(np.uint8), 1)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)[1:-1, 1:-1]
    v, u = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(u), int(v)


def size_mm(mask: np.ndarray) -> tuple[float, float]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(np.vstack(cs))
    return round(max(w, h) * MM_PER_PX, 1), round(min(w, h) * MM_PER_PX, 1)


def filled(mask: np.ndarray) -> np.ndarray:
    """The mask with its holes filled (outer contours only)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, cv2.FILLED)
    return out.astype(bool)


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def contact(a: np.ndarray, b: np.ndarray, r: int = 10) -> float:
    """Share of a's outline that lies within r px of b: 0 = apart, small = touching, large = cut by b."""
    ax0, ay0, ax1, ay1 = bbox_of(a)
    bx0, by0, bx1, by1 = bbox_of(b)
    if ax0 > bx1 + r or bx0 > ax1 + r or ay0 > by1 + r or by0 > ay1 + r:
        return 0.0
    x0, y0, x1, y1 = max(0, ax0 - r), max(0, ay0 - r), ax1 + r, ay1 + r
    sa, sb = a[y0:y1, x0:x1].astype(np.uint8), b[y0:y1, x0:x1].astype(np.uint8)
    edge = sa - cv2.erode(sa, np.ones((3, 3), np.uint8))
    near = cv2.dilate(sb, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    n = int(edge.sum())
    return float((edge & near).sum()) / n if n else 0.0


def inside_share(a: np.ndarray, region: np.ndarray) -> float:
    """Share of mask a that lies inside region."""
    return float((a & region).sum()) / max(1, int(a.sum()))


# ----------------------------------------------------------------------------- 1. proposals
class Segmenter:
    """SAM 2.1 automatic mask generation, loaded on first use."""

    def __init__(self, model: str = "facebook/sam2.1-hiera-large", device: str | None = None):
        self.model_id, self.device, self._gen = model, device, None

    def __call__(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        if self._gen is None:
            import torch
            from transformers import pipeline
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._gen = pipeline("mask-generation", model=self.model_id, device=dev, dtype=torch.float32)
        from PIL import Image
        # SAM sees 1024 pixels on the long side whatever we give it; giving it a 1024-pixel copy keeps the
        # mask post-processing small (at full size it took 27 GB of RAM). Masks are scaled back up.
        h, w = rgb.shape[:2]
        k = min(1.0, 1024 / max(h, w))
        small = cv2.resize(rgb, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else rgb
        # 48 x 48 seed points: the small white part on the grey tray is found only with this density
        out = self._gen(Image.fromarray(small), points_per_batch=64, points_per_crop=48,
                        pred_iou_thresh=0.8, stability_score_thresh=0.85)
        return [(cv2.resize(np.asarray(m, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool), float(s))
                for m, s in zip(out["masks"], out["scores"])]


# ----------------------------------------------------------------------------- 2. geometry filter
def workcell_roi(h: int, w: int) -> np.ndarray:
    """The two trays in the fixed camera: everything but the rails along the top-left and bottom edges,
    the tape at the right edge and the inner left wall of the grey tray (it mirrors the objects)."""
    roi = np.ones((h, w), bool)
    roi[: int(0.037 * h), : int(0.51 * w)] = False      # rails above the grey tray
    roi[int(0.915 * h):, :] = False                      # rails and robot base along the bottom
    roi[:, int(0.985 * w):] = False                      # tape on the right edge
    roi[:, : int(0.04 * w)] = False                      # inner left wall of the grey tray
    return roi


def prefilter(props: list[tuple[np.ndarray, float]], shape) -> list[tuple[np.ndarray, float]]:
    h, w = shape[:2]
    roi = workcell_roi(h, w)
    min_area, max_area = 6000, int(0.12 * h * w)        # 6000 px = 3.7 cm^2, below half a yellow gear; the plate is ~0.09 of the image
    keep = []
    for m, s in sorted(props, key=lambda x: -x[1]):
        a = int(m.sum())
        if not (min_area <= a <= max_area) or (m & roi).sum() < 0.6 * a:
            continue
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.vstack(cs))
        if min(rw, rh) < 35 or max(rw, rh) / max(1, min(rw, rh)) > 8:   # strips of tray wall, edges
            continue
        dup = [j for j, (k, _) in enumerate(keep)
               if (m & k).sum() > 0.85 * min(a, k.sum()) and abs(a - k.sum()) < 0.15 * max(a, k.sum())]
        if dup:   # near-duplicates: keep the more compact one (the other often has a strip of tray wall attached)
            j = dup[0]
            if compactness(m) > compactness(keep[j][0]) + 0.05:
                keep[j] = (m, s)
            continue
        keep.append((m, s))
    return keep


def compactness(mask: np.ndarray) -> float:
    """Area over convex-hull area: 1 for a convex blob, low for a blob with strips or tails."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(mask.sum()) / max(1.0, cv2.contourArea(cv2.convexHull(np.vstack(cs))))


def contrast(lab: np.ndarray, mask: np.ndarray, others: np.ndarray) -> float:
    """Colour difference (CIELAB) between a region and the free surface around it. Marks and moulded
    bumps of the tray floor score under 7; the dimmest real object (the metal bar on the grey tray) 20."""
    f = filled(mask).astype(np.uint8)
    near = cv2.dilate(f, np.ones((41, 41), np.uint8)).astype(bool) & ~cv2.dilate(f, np.ones((11, 11), np.uint8)).astype(bool)
    ring = near & ~others
    if ring.sum() < 200:
        ring = near
    return float(np.linalg.norm(lab[mask].mean(0) - lab[ring].mean(0)))


def drop_flat(props: list[tuple[np.ndarray, float]], rgb: np.ndarray, min_contrast: float = 8.0):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    union = np.zeros(rgb.shape[:2], bool)
    for m, _ in props:
        union |= m
    return [(m, s) for m, s in props if contrast(lab, m, union & ~m) >= min_contrast]


# ----------------------------------------------------------------------------- 3. naming
def crop_for(rgb: np.ndarray, mask: np.ndarray, around: np.ndarray | None = None) -> np.ndarray:
    """A crop around `around` (default: the mask) with the mask outlined in magenta."""
    x0, y0, x1, y1 = bbox_of(mask if around is None else around)
    pad = int(max(60, 0.35 * max(x1 - x0, y1 - y0)))
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(rgb.shape[1], x1 + pad), min(rgb.shape[0], y1 + pad)
    crop = rgb[Y0:Y1, X0:X1].copy()
    cs, _ = cv2.findContours(mask[Y0:Y1, X0:X1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cs, -1, (255, 0, 255), max(2, int(round(3 * max(crop.shape[:2]) / 384))))
    return crop


def ask_crop(llm: Apertus, crop: np.ndarray, prompt: str, schema: dict) -> dict:
    part, _ = image_part(crop, 384)
    return llm.json([{"role": "user", "content": [part, {"type": "text", "text": prompt}]}], schema, max_tokens=40)


def classify(llm: Apertus, rgb: np.ndarray, masks: list[np.ndarray]) -> list[dict]:
    def one(m):
        long, short = size_mm(filled(m))
        return ask_crop(llm, crop_for(rgb, m), CLASSIFY_PROMPT.format(long=long, short=short), CLASSIFY_SCHEMA)
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, masks))


def size_fits(cls: str, long_mm: float) -> bool:
    """Not larger than the object can be (a region can be smaller: something may lie on it)."""
    return cls not in SIZE_MM or long_mm <= SIZE_MM[cls][2]


def too_small(cls: str, long_mm: float) -> bool:
    return cls in SIZE_MM and long_mm < 0.6 * SIZE_MM[cls][1]


def pick_label(lab: dict, long_mm: float, short_mm: float, at_edge: bool = False) -> str:
    """First choice unless the region is too large for it; then the second choice. The plate and the saucer,
    and the three flat parts, are told apart by size and shape: Apertus mixes them up."""
    first, second = lab["label"], lab.get("second_choice", EMPTY)
    if size_fits(first, long_mm):
        label = first
    else:   # too large for its name: another object that size, or the object with its shadow (then dropped)
        label = second if second in OBJECT_CLASSES and size_fits(second, long_mm) else             HAND if second == HAND and at_edge else EMPTY      # an arm reaches in from the edge of the photo
    if label in ("plate", "saucer"):
        label = "plate" if long_mm > 160 else "saucer"
    if label in ("white part", "small white part", "grey metal part"):
        aspect = long_mm / max(short_mm, 1.0)
        if long_mm < 95 and aspect < 2.0:
            label = "small white part"
        elif long_mm >= 100 and aspect >= 2.1:
            label = "grey metal part"
        elif long_mm >= 100 and aspect < 1.75:
            label = "white part"
    return label


# ----------------------------------------------------------------------------- 4. clean-up
@dataclass(eq=False)
class Cand:
    mask: np.ndarray
    label: str
    second: str
    score: float
    on: list = field(default_factory=list)
    weak: bool = False      # named by Apertus's second choice: the first did not fit its size

    def __post_init__(self):
        self.refresh()

    def refresh(self):
        self.fill = filled(self.mask)
        self.long_mm, self.short_mm = size_mm(self.fill)

    def absorb(self, other: "Cand"):
        self.mask = self.mask | other.mask
        self.refresh()


def resolve(cands: list[Cand], rgb: np.ndarray, llm: Apertus, log: list | None = None) -> list[Cand]:
    log = log if log is not None else []
    real = [c for c in cands if c.label in OBJECT_CLASSES or c.label == HAND]

    # a) nesting: smaller regions inside a larger one are its pieces, or objects lying on it
    real.sort(key=lambda c: int(c.fill.sum()))
    kept: list[Cand] = []
    for c in real:
        inside = [k for k in kept if inside_share(k.mask, c.fill) >= 0.85]
        if c.weak and any(k.label != c.label for k in inside):
            log.append(f"dropped a {c.long_mm:.0f} mm region renamed '{c.label}': it holds a whole other object")
            continue          # too big for what Apertus saw in it: that object plus its surroundings
        for k in inside:
            if k.label == c.label or c.label == HAND or k.label == HAND:
                same = k.label == c.label
            else:
                ans = ask_crop(llm, crop_for(rgb, k.mask, around=c.mask), PART_PROMPT.format(parent=c.label), PART_SCHEMA)
                same = ans["answer"] == "part of it"
            if same:
                kept.remove(k)
                c.absorb(k)
                log.append(f"{k.label} ({k.long_mm:.0f} mm) is part of the {c.label}")
            else:
                k.on.append(c)
        kept.append(c)
    for c in kept:                       # a bottom object's mask must not include what lies on it
        for k in kept:
            if c in k.on:
                c.mask = c.mask & ~k.mask
        c.refresh()

    # b) regions smaller than their object can be: a fragment of a neighbour of the same kind, a side or
    #    shadow of the object they touch, a partly hidden object, or (free-standing) nothing
    small = [c for c in kept if c.label != HAND and c.long_mm < SIZE_MM[c.label][1]]
    anchors = [c for c in kept if c not in small]
    for f in sorted(small, key=lambda c: c.long_mm):
        same = [g for g in anchors if g.label == f.label]
        touching = sorted([(contact(f.mask, o.mask), o) for o in anchors if o.label != f.label], key=lambda x: -x[0])
        touching = [(t, o) for t, o in touching if t > 0]
        direct = [g for g in same if contact(f.mask, g.mask) > 0]
        via = [g for g in same if any(t > 0.1 and contact(g.mask, o.mask) > 0 for t, o in touching)]
        hand = [o for t, o in touching if o.label == HAND]
        if hand:      # too small to be what Apertus called it, and touching the arm: the wrist, a watch, a sleeve
            hand[0].absorb(f)
            log.append(f"a {f.long_mm:.0f} mm '{f.label}' region touching the arm is part of the person")
        elif direct or via:
            (direct or via)[0].absorb(f)
            log.append(f"{f.label} fragment ({f.long_mm:.0f} mm) joined to its neighbour")
        elif touching:
            parent = touching[0][1]
            ans = ask_crop(llm, crop_for(rgb, f.mask, around=f.mask | parent.mask),
                           NEXT_PROMPT.format(label=f.label, parent=parent.label), NEXT_SCHEMA)
            if ans["answer"] == "separate object":
                anchors.append(f)
                log.append(f"kept a {f.long_mm:.0f} mm {f.label} next to the {parent.label} (partly hidden)")
                continue
            log.append(f"dropped a {f.long_mm:.0f} mm {f.label}: side or shadow of the {parent.label}")
        elif too_small(f.label, f.long_mm):
            log.append(f"dropped {f.label} region of {f.long_mm:.0f} mm: too small and nothing hides it")
        else:
            anchors.append(f)
            continue
        kept.remove(f)

    # c) a single object split in pieces (the hand, the mug and its handle): join touching regions of a one-of class
    for c in sorted(kept, key=lambda c: -c.mask.sum()):
        if c not in kept or (COUNT.get(c.label, 1) > 1 and c.label != HAND):
            continue
        for g in [g for g in kept if g is not c and g.label == c.label and contact(g.mask, c.mask) > 0]:
            kept.remove(g)
            c.absorb(g)
            log.append(f"joined two touching {c.label} regions")

    # c2) a region that touches the arm and the edge of the photo is the person's sleeve: people reach in
    #     from the edge. (Asking Apertus here made it call the mug held in the hand "part of the person".)
    hands = [c for c in kept if c.label == HAND]
    for c in [c for c in kept if c.label != HAND]:
        near = [h for h in hands if contact(c.mask, h.mask, r=25) > 0]
        x0, y0, x1, y1 = bbox_of(c.mask)
        if near and (x0 <= 2 or y0 <= 2 or x1 >= rgb.shape[1] - 2 or y1 >= rgb.shape[0] - 2):
            kept.remove(c)
            near[0].absorb(c)
            log.append(f"a {c.long_mm:.0f} mm '{c.label}' region is part of the person")

    # d) more of a class than the workcell has: keep the best-sized ones, the rest take their second choice or go
    for cls, n in COUNT.items():
        group = [c for c in kept if c.label == cls]
        if len(group) <= n:
            continue
        typical = SIZE_MM[cls][0]
        group.sort(key=lambda c: (abs(c.long_mm - typical) / typical) - 0.05 * c.score)
        for c in group[n:]:
            alt = c.second if c.second in OBJECT_CLASSES and size_fits(c.second, c.long_mm) \
                and sum(k.label == c.second for k in kept) < COUNT.get(c.second, 1) else None
            log.append(f"too many {cls}: a {c.long_mm:.0f} mm one becomes {alt or 'nothing'}")
            if alt:
                c.label = alt
            else:
                kept.remove(c)
    # e) overlapping masks: the overlap belongs to the smaller object, which lies on the larger one
    #    (SAM often outlines a stack as one region)
    for a in sorted(kept, key=lambda c: c.mask.sum()):
        for b in kept:
            if b is a or b.mask.sum() <= a.mask.sum():
                continue
            ov = a.mask & b.mask
            if ov.any():
                b.mask = b.mask & ~a.mask
                if ov.sum() > 0.3 * a.mask.sum() and b not in a.on:
                    a.on.append(b)
    for c in kept:
        c.refresh()
    return [c for c in kept if c.mask.sum() > 0]


def name_objects(cands: list[Cand]) -> list[Obj]:
    """ids like red_gear_1, numbered left to right."""
    objs: list[Obj] = []
    ids: dict[int, str] = {}
    by_cls: dict[str, list[Cand]] = {}
    for c in cands:
        by_cls.setdefault(c.label, []).append(c)
    for cls, cs in by_cls.items():
        cs.sort(key=lambda c: np.nonzero(c.mask)[1].mean())
        base = "hand" if cls == HAND else cls.replace(" ", "_")
        for i, c in enumerate(cs, 1):
            ids[id(c)] = base if COUNT.get(cls, 1) == 1 else f"{base}_{i}"
    for c in cands:
        objs.append(Obj(ids[id(c)], "hand" if c.label == HAND else c.label, c.mask,
                        [ids[id(o)] for o in c.on if id(o) in ids], c.score))
    return sorted(objs, key=lambda o: o.centroid[0])


# ----------------------------------------------------------------------------- entry point
class Perception:
    def __init__(self, llm: Apertus, cache_dir: Path | None = Path(".cache/perception"), segmenter: Segmenter | None = None):
        self.llm, self.seg = llm, segmenter or Segmenter()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log: list[str] = []

    def proposals(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        key = hashlib.sha1(rgb.tobytes() + b"p4-1024").hexdigest()[:20]
        path = self.cache_dir / f"sam_{key}.npz" if self.cache_dir else None
        if path and path.exists():
            z = np.load(path)
            masks = np.unpackbits(z["bits"], axis=1)[:, : rgb.shape[0] * rgb.shape[1]].reshape(-1, *rgb.shape[:2]).astype(bool)
            return list(zip(masks, z["scores"].tolist()))
        props = prefilter(self.seg(rgb), rgb.shape)
        if path:
            bits = np.packbits(np.stack([m for m, _ in props]).reshape(len(props), -1), axis=1)
            np.savez_compressed(path, bits=bits, scores=np.array([s for _, s in props]))
        return props

    def __call__(self, rgb: np.ndarray) -> list[Obj]:
        props = drop_flat(self.proposals(rgb), rgb)
        labels = classify(self.llm, rgb, [m for m, _ in props])
        cands = []
        for (m, s), lab in zip(props, labels):
            c = Cand(m, EMPTY, lab.get("second_choice", EMPTY), s)
            x0, y0, x1, y1 = bbox_of(m)
            edge = x0 <= 2 or y0 <= 2 or x1 >= rgb.shape[1] - 2 or y1 >= rgb.shape[0] - 2
            c.label = pick_label(lab, c.long_mm, c.short_mm, edge)
            c.weak = c.label not in (lab["label"], EMPTY)
            cands.append(c)
        self.log = []
        return name_objects(resolve(cands, rgb, self.llm, self.log))


def draw(rgb: np.ndarray, objs: list[Obj], path: Path, width: int = 1224):
    """An overlay to check perception by eye."""
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    rng = np.random.default_rng(1)
    for o in objs:
        col = [int(c) for c in rng.integers(80, 255, 3)]
        cs, _ = cv2.findContours(o.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, col, 7)
        u, v = o.grasp_pixel
        cv2.circle(im, (u, v), 14, (0, 0, 255), -1)
        for t, c in ((10, (255, 255, 255)), (4, (0, 0, 0))):
            cv2.putText(im, o.id, (u - 90, v - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.0, c, t)
    h = int(im.shape[0] * width / im.shape[1])
    cv2.imwrite(str(path), cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA))
