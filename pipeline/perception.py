"""See: find every object in the photo and give it a name and a pixel mask.

1. SAM 2.1 (automatic mask generation, Apache 2.0) proposes regions, without names.
2. Cheap geometry drops what cannot be an object: specks, the trays themselves, the rails
   outside the workcell, thin strips of tray wall, flat marks on the tray floor.
3. Apertus names each region from a crop with the region outlined, told the size of the
   region and of each workcell object (the camera is fixed: 0.25 mm per pixel on the table).
   It says in a few words what it sees, picks a workcell object (or "other object"), a second
   choice and the colour. An object that is not one of the workcell's keeps Apertus's own words
   ("tennis ball"); a workcell shape in a colour it does not have ("blue cube") too.
4. Clean-up: a region larger than its object can be takes the second choice; a region inside
   another object is a piece of it (the hub of a gear, the centre of the saucer) or a separate
   object lying on it; a region next to an object is a part of it (the mug's handle), its shadow,
   or a separate object. Apertus answers these questions from a crop; colours check its answers.
   There is no fixed inventory: any number of objects of any kind can be found.

The result is cached per image (masks as packed bits), so only the first run costs time.
"""
from __future__ import annotations

import hashlib, re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .llm import Apertus, image_part

MM_PER_PX = 0.25        # on the table, for the workcell's fixed camera; set_scale() takes it from another calibration


def set_scale(mm_per_px: float):
    """Use another camera's scale (mm per pixel on the table). The sizes below are in mm, so they still hold."""
    global MM_PER_PX
    MM_PER_PX = float(mm_per_px)

# long side seen from above, mm: (typical, smallest, largest) for a fully visible object, measured on the dev photos
SIZE_MM = {"red gear": (91, 80, 105), "yellow gear": (48, 38, 60), "red cube": (87, 75, 105),
           "mug": (110, 85, 130), "mandarin": (74, 60, 90), "banana": (220, 160, 260), "plate": (193, 170, 250),
           "saucer": (124, 110, 145), "grey metal part": (135, 100, 210), "white part": (143, 120, 170),
           "small white part": (80, 50, 100)}
OBJECT_CLASSES = list(SIZE_MM)
GRASPABLE = set(OBJECT_CLASSES) - {"plate", "saucer"}
GRIP_MM = 85            # the gripper's opening; an unknown object narrower than GRIP_MM + 10 counts as graspable
HAND, EMPTY, OTHER = "hand or arm", "tray or table", "other object"
LABELS = OBJECT_CLASSES + [HAND, EMPTY, OTHER]
COLOR_WORDS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "white", "grey", "black", "silver"]
CLASS_COLORS = {"red gear": {"red"}, "yellow gear": {"yellow"}, "red cube": {"red"}, "mug": {"white", "green"},
                "mandarin": {"orange"}, "banana": {"yellow"}, "plate": {"white"}, "saucer": {"yellow", "white", "orange"},
                "grey metal part": {"grey", "silver"}, "white part": {"white"}, "small white part": {"white"}}
# colour words people (and Apertus) use for the same colour under this light
NEAR = {"red": {"red", "orange", "pink", "brown"}, "orange": {"orange", "red", "yellow", "brown"},
        "yellow": {"yellow", "orange", "green", "brown"}, "green": {"green", "yellow"}, "white": {"white", "grey", "silver"},
        "grey": {"grey", "silver", "white", "black"}, "silver": {"silver", "grey", "white"},
        "brown": {"brown", "orange", "red"}, "pink": {"pink", "red", "purple"}, "purple": {"purple", "pink", "blue"},
        "blue": {"blue", "purple"}, "black": {"black", "grey"}}

CLASSIFY_PROMPT = """This is a crop of a photo of a robot workcell, taken from straight above. The magenta outline marks one region, about {long:.0f} x {short:.0f} mm.
Which object does the region inside the magenta outline belong to? The objects of the workcell, with their size seen from above:
red gear (about 90 mm), yellow gear (about 48 mm, small), red cube (about 87 mm), mug (about 110 mm with its handle), mandarin (about 74 mm), banana (about 220 mm long), plate (white, about 195 mm), saucer (yellow rim and white centre, about 125 mm), grey metal part (flat aluminium bar with holes, about 135 x 47 mm), white part (flat white plastic plate, about 143 x 98 mm), small white part (white block with a dark side, about 80 x 55 mm).
Other answers: "hand or arm" for a person's hand, arm, wrist or sleeve; "tray or table" when the region is empty (the floor or wall of a tray, the table, a mark, a shadow or a reflection); "other object" for any other object.
If the outline covers only a piece of an object (part of a gear, the inside of a mug, the centre of a saucer, the pattern of a plate), answer with that object. A region can be smaller than its object when something lies on it.
Give your best answer and a second choice."""
CLASSIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label", "second_choice"],
                   "properties": {"label": {"enum": LABELS}, "second_choice": {"enum": LABELS}}}
# a second, open question: the list above makes Apertus fit anything into a workcell class; this one lets it say
# "tennis ball". Asked separately, so the answers for the workcell objects stay what they were.
DESCRIBE_PROMPT = """This is a crop of a photo taken from straight above. The magenta outline marks one region, about {long:.0f} x {short:.0f} mm.
What is inside the magenta outline? Name the object in two or three words, and give its main colour."""
DESCRIBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["what", "color"],
                   "properties": {"what": {"type": "string", "maxLength": 40}, "color": {"enum": COLOR_WORDS}}}

# words of the open description that agree with a workcell class (Apertus describes the mug's inside as "green
# liquid" and the small white part as an "electronic device"); other words make it ask which of the two it is
GEAR_WORDS = "gear cog wheel sprocket sun star flower toothed"      # "glittery sun-shaped object": the yellow gear
CLASS_WORDS = {"red gear": GEAR_WORDS, "yellow gear": GEAR_WORDS,
               "red cube": "cube block box brick square cubic", "mug": "mug cup liquid coffee tea drink beverage",
               "mandarin": "mandarin orange tangerine clementine citrus", "banana": "banana",
               "plate": "plate dish", "saucer": "saucer plate dish",
               "grey metal part": "metal bar bracket plate part rail aluminium aluminum strip profile",
               "white part": "plate panel sheet board tile part plastic",
               "small white part": "block part piece plastic box device connector component brick"}
ROUND_CLASSES = {"plate", "saucer", "mandarin"}      # whole and round; gears are often half hidden, so not here
GENERIC_WORDS = {"object", "item", "thing", "piece", "part", "region", "shape", "area", "surface", "section"}
CLASS_DESC = {"red gear": "a red gear, about 90 mm", "yellow gear": "a small yellow gear, about 48 mm",
              "red cube": "a red cube, about 87 mm", "mug": "a mug, about 110 mm with its handle",
              "mandarin": "a mandarin, about 74 mm", "banana": "a banana, about 220 mm long",
              "plate": "a white plate, about 195 mm", "saucer": "a saucer with a yellow rim and a white centre, about 125 mm",
              "grey metal part": "a flat aluminium bar with holes, about 135 x 47 mm",
              "white part": "a flat white plastic plate, about 143 x 98 mm",
              "small white part": "a white block with a dark side, about 80 x 55 mm"}
# two differently worded checks of the class answer; both must say no before the open description wins (on the
# dev photos each alone rejected one workcell object, together none; they rejected 11 of 14 unseen objects)
VERIFY_PROMPTS = [("""This is a crop of a photo taken from straight above. The magenta outline marks one region, about {long:.0f} x {short:.0f} mm.
Is the object inside the magenta outline {desc}? Answer yes or no.""", ["yes", "no"]),
                  ("""This is a crop of a photo taken from straight above. The magenta outline marks one region, about {long:.0f} x {short:.0f} mm.
Someone says it is {desc}. Is that right, or is it a different kind of object?""", ["right", "a different kind of object"])]


def agrees(label: str, what: str) -> bool:
    """Do the open description's words fit the class answer (or say nothing either way)?"""
    words = set(re.findall(r"[a-z]+", what.lower().replace("gray", "grey"))) - set(COLOR_WORDS) - FILLER
    if not words or words <= GENERIC_WORDS:
        return True
    return any(c in w for w in words for c in CLASS_WORDS.get(label, label).split())      # "teacup" has "cup"


PART_PROMPT = """This crop shows a {parent}, seen from straight above. The magenta outline marks a region on it.
Is that region a separate object lying on the {parent} (for example a fruit, a gear or a part placed on it), or is it just a part of the {parent} itself (its centre, its rim, its pattern, its hub)?"""
NEXT_PROMPT = """This crop shows a {parent}, seen from straight above. The magenta outline marks a region next to it that looks like a {label}.
Is that region a separate object (perhaps partly hidden under the {parent}), a part of the {parent} itself (its side, wall, handle or rim), or only the {parent}'s shadow or a reflection on the surface?"""
NEXT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
               "properties": {"answer": {"enum": ["separate object", "part of it", "its shadow or a reflection"]}}}
PART_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
               "properties": {"answer": {"enum": ["separate object", "part of it"]}}}


@dataclass
class Obj:
    id: str
    cls: str                                 # a workcell object ("red gear"), "other object", or "hand"
    mask: np.ndarray = field(repr=False)
    on: list = field(default_factory=list)   # ids of the objects this one lies on, found while naming
    score: float = 0.0
    name: str = ""                           # what it is called: "red gear", "blue cube", "tennis ball"
    colors: frozenset = frozenset()
    crop: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.name = self.name or self.cls
        self.colors = frozenset(self.colors or CLASS_COLORS.get(self.cls, ()))

    @property
    def graspable(self) -> bool:
        """The workcell's plate and saucer are too wide for the 85 mm gripper; an unknown object is judged by its
        width seen from above (a little wider than the gripper is allowed: the top of a tall object looks larger)."""
        if self.cls in SIZE_MM:
            return self.cls in GRASPABLE
        return self.cls != "hand" and self.size_mm[1] <= GRIP_MM + 10

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


MIN_AREA = 6000      # px; 6000 = 3.7 cm^2, below half a yellow gear


def prefilter(props: list[tuple[np.ndarray, float]], shape) -> list[tuple[np.ndarray, float]]:
    h, w = shape[:2]
    roi = workcell_roi(h, w)
    # 6000 px = 3.7 cm^2, below half a yellow gear. Up to 0.18 of the photo: the plate is 0.09, a book would be 0.15;
    # the floors of the trays, the largest regions SAM finds, are 0.21 and more
    min_area, max_area = MIN_AREA, int(0.18 * h * w)
    keep = []
    for m, s in sorted(props, key=lambda x: -x[1]):
        a = int(m.sum())
        if not (min_area <= a <= max_area) or (m & roi).sum() < 0.6 * a:
            continue
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.vstack(cs))
        # tray walls: strips about 390 x 15 mm, or shorter where an object lies across the wall; they run along the
        # image axes. A pen or a screwdriver (about 150 x 10 mm) lying at an angle is kept
        thin = max(rw, rh) / max(1, min(rw, rh)) > 8
        ang = abs(((cv2.minAreaRect(np.vstack(cs))[2] + 45) % 90) - 45)        # 0 = along an image axis
        if min(rw, rh) < 30 or (thin and max(rw, rh) * MM_PER_PX > 300) or \
                (thin and ang < 4 and max(rw, rh) * MM_PER_PX > 150):
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


def edge_gap(lab: np.ndarray, inner: "Cand", outer: "Cand", r: int = 15) -> float:
    """Colour difference across the outline of a region lying inside another: a band just inside its outline
    against the outer object's pixels just outside it (the plate's inner disc: white against white)."""
    x0, y0, x1, y1 = bbox_of(inner.fill)
    X0, Y0, X1, Y1 = max(0, x0 - r), max(0, y0 - r), x1 + r, y1 + r
    f = inner.fill[Y0:Y1, X0:X1].astype(np.uint8)
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    band_in = (f.astype(bool) & ~cv2.erode(f, k).astype(bool)) & inner.mask[Y0:Y1, X0:X1]
    band_out = cv2.dilate(f, k).astype(bool) & ~f.astype(bool) & outer.mask[Y0:Y1, X0:X1]
    if band_in.sum() < 50 or band_out.sum() < 50:
        return 0.0
    win = lab[Y0:Y1, X0:X1]
    return float(np.linalg.norm(win[band_in].mean(0) - win[band_out].mean(0)))


def same_material(lab: np.ndarray, inner: "Cand", outer: "Cand", kept: list) -> bool:
    """The inner region (without what lies on it) has the outer object's colour: only lighter or darker, as the
    plain well of a plate is against its patterned rim. A saucer on a plate differs in colour (its yellow rim)."""
    held = np.zeros(inner.mask.shape, bool)
    for j in kept:
        if j is not inner and j is not outer and inside_share(j.mask, inner.fill) >= 0.5:
            held |= j.mask
    own, rest = inner.mask & ~held, outer.mask & ~inner.fill
    if own.sum() < 500 or rest.sum() < 500:
        return False
    d = lab[own].mean(0) - lab[rest].mean(0)
    return bool(np.hypot(d[1], d[2]) < 6 and abs(d[0]) < 25)


# workcell objects with a strong colour: on the dev photos at least 70% of their pixels are saturated (chroma > 20);
# the white plate, the parts and the tray none
COLOURFUL = {"red gear", "yellow gear", "red cube", "mug", "mandarin", "banana", "saucer"}


def pixel_colour(rgb: np.ndarray, mask: np.ndarray) -> str:
    """The colour word for the median colour of the region's well-lit pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask]
    if len(hsv) == 0:
        return ""
    hsv = hsv[hsv[:, 2] > np.percentile(hsv[:, 2], 25)]          # leave out the shaded side
    h, s, v = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if s < 45:
        return "white" if v > 170 else "grey" if v > 70 else "black"
    if v < 60:
        return "black"
    for top, name in ((8, "red"), (20, "orange"), (34, "yellow"), (85, "green"), (128, "blue"), (150, "purple"),
                      (170, "pink"), (180, "red")):
        if h < top:
            return "brown" if name in ("orange", "red") and v < 120 else name
    return ""


def saturated_share(lab: np.ndarray, mask: np.ndarray) -> float:
    px = lab[mask]
    return float((np.hypot(px[:, 1] - 128, px[:, 2] - 128) > 20).mean()) if len(px) else 0.0


def colour_gap(lab: np.ndarray, frag: np.ndarray, other: np.ndarray, r: int = 25) -> float:
    """Colour difference (CIELAB) between a region and the part of a neighbour within r px of it: what the
    eye compares to tell whether a piece belongs to the object it touches (the white handle against the white
    rim of the mug, not against its green inside)."""
    x0, y0, x1, y1 = bbox_of(frag)
    X0, Y0, X1, Y1 = max(0, x0 - r), max(0, y0 - r), x1 + r, y1 + r
    near = cv2.dilate(frag[Y0:Y1, X0:X1].astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)
    ring = near & other[Y0:Y1, X0:X1] & ~frag[Y0:Y1, X0:X1]
    if ring.sum() < 50:
        return 99.0
    return float(np.linalg.norm(lab[frag].mean(0) - lab[Y0:Y1, X0:X1][ring].mean(0)))


def surface_shift(lab: np.ndarray, frag: np.ndarray, parent: np.ndarray, r: int = 30) -> np.ndarray:
    """Colour (CIELAB) of a region minus that of the free surface around it (not the parent object)."""
    x0, y0, x1, y1 = bbox_of(frag)
    X0, Y0, X1, Y1 = max(0, x0 - r), max(0, y0 - r), x1 + r, y1 + r
    f = frag[Y0:Y1, X0:X1]
    near = cv2.dilate(f.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool) & ~f & ~parent[Y0:Y1, X0:X1]
    win = lab[Y0:Y1, X0:X1]
    return win[f].mean(0) - (win[near].mean(0) if near.sum() > 50 else win[f].mean(0))


def shadow_like(lab: np.ndarray, frag: np.ndarray, parent: np.ndarray) -> bool:
    """Darker than the surface around it, grey, and unlike the object where they meet."""
    shift = surface_shift(lab, frag, parent)
    return bool(shift[0] < -15 and np.hypot(*shift[1:]) < 8 and colour_gap(lab, frag, parent) > 40)


def piece_of(lab: np.ndarray, frag: np.ndarray, parent: np.ndarray) -> bool:
    """A region next to an object, which Apertus calls a part of it: is it really (the mug's handle), or the
    object's shadow, or a patch of tray floor? Apertus says "part of it" for all three, so the pixels decide:
    a part stands out from the surface, and a shadow is darker than the surface, grey, and unlike the object
    where they meet (lightness 33-47 below the tray floor in the dev photos; handles 48-114 above it)."""
    shift = surface_shift(lab, frag, parent)
    return bool(np.linalg.norm(shift) >= 20 and not shadow_like(lab, frag, parent) and colour_gap(lab, frag, parent) < 50)


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


STACK_PROMPT = """This crop of a photo taken from straight above shows two objects that touch: the {a} (magenta outline) and the {b} (cyan outline).
Does the {a} lie on top of the {b}, does the {b} lie on top of the {a}, or do they lie side by side on the table?"""


def on_top(llm: Apertus, rgb: np.ndarray, a: "Obj", b: "Obj") -> str:
    """Apertus on two touching objects: "a on b", "b on a" or "side by side"."""
    crop = crop_for(rgb, a.mask, around=a.mask | b.mask)
    x0, y0, x1, y1 = bbox_of(a.mask | b.mask)
    pad = int(max(60, 0.35 * max(x1 - x0, y1 - y0)))
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    cs, _ = cv2.findContours(b.mask[Y0:Y0 + crop.shape[0], X0:X0 + crop.shape[1]].astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cs, -1, (0, 255, 255), max(2, int(round(3 * max(crop.shape[:2]) / 384))))
    opts = [f"the {a.name} lies on the {b.name}", f"the {b.name} lies on the {a.name}", "side by side"]
    if a.name == b.name:
        opts[:2] = ["the magenta one lies on the cyan one", "the cyan one lies on the magenta one"]
    schema = {"type": "object", "additionalProperties": False, "required": ["answer"],
              "properties": {"answer": {"enum": opts}}}
    ans = ask_crop(llm, crop, STACK_PROMPT.format(a=a.name, b=b.name), schema)["answer"]
    return {opts[0]: "a on b", opts[1]: "b on a"}.get(ans, "side by side")


IS_IT_PROMPT = """This is a crop of a photo taken from straight above. Is the object inside the magenta outline {a} {noun}? Answer yes or no."""


def is_it(llm: Apertus, obj: "Obj", noun: str) -> bool:
    """A yes/no look at one object: Apertus recognises a padlock it named "USB flash drive" when asked."""
    if obj.crop is None:
        return False
    schema = {"type": "object", "additionalProperties": False, "required": ["answer"],
              "properties": {"answer": {"enum": ["yes", "no"]}}}
    a = "an" if noun[:1] in "aeiou" else "a"
    return ask_crop(llm, obj.crop, IS_IT_PROMPT.format(a=a, noun=noun), schema)["answer"] == "yes"


def ask_crop(llm: Apertus, crop: np.ndarray, prompt: str, schema: dict) -> dict:
    part, _ = image_part(crop, 384)
    return llm.json([{"role": "user", "content": [part, {"type": "text", "text": prompt}]}], schema, max_tokens=40)


def classify(llm: Apertus, rgb: np.ndarray, masks: list[np.ndarray]) -> list[dict]:
    """{"label", "second_choice"} per region, plus {"what", "color"} for the regions that are objects."""
    def one(m):
        long, short = size_mm(filled(m))
        crop = crop_for(rgb, m)
        out = ask_crop(llm, crop, CLASSIFY_PROMPT.format(long=long, short=short), CLASSIFY_SCHEMA)
        if out["label"] not in (EMPTY, HAND) or out["second_choice"] not in (EMPTY, HAND):
            out.update(ask_crop(llm, crop, DESCRIBE_PROMPT.format(long=long, short=short), DESCRIBE_SCHEMA))
        what, label = clean_name(out.get("what", "")), out["label"]
        # a whole object of the class's size whose open description names something else ("tennis ball" for
        # "mandarin"): is it really the class? (A piece smaller than its object is left to the clean-up rules.)
        whole = compactness(m) > 0.8 and (label not in ROUND_CLASSES | {"red gear", "yellow gear"} or long < 1.25 * short)
        if label in CLASS_WORDS and what and not agrees(label, what) and SIZE_MM[label][1] <= long <= SIZE_MM[label][2] \
                and whole and colour_fits(label, out.get("color", ""), what):
            no = []
            for prompt, answers in VERIFY_PROMPTS:
                schema = {"type": "object", "additionalProperties": False, "required": ["answer"],
                          "properties": {"answer": {"enum": answers}}}
                ans = ask_crop(llm, crop, prompt.format(long=long, short=short, desc=CLASS_DESC[label]), schema)
                no.append(ans["answer"] == answers[1])
                if not no[-1]:
                    break
            out["choice"] = "description" if all(no) else "class"
        return out
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, masks))


def size_fits(cls: str, long_mm: float) -> bool:
    """Not larger than the object can be (a region can be smaller: something may lie on it)."""
    return cls not in SIZE_MM or long_mm <= SIZE_MM[cls][2]


def too_small(cls: str, long_mm: float) -> bool:
    return cls in SIZE_MM and long_mm < 0.6 * SIZE_MM[cls][1]


# the open description's nouns for the workcell's objects ("gear wheel", "orange", "coffee cup")
FAMILY = {**{w: ("red gear", "yellow gear") for w in ("gear", "cog", "sprocket", "sun", "star", "flower", "toothed")},
          "cube": ("red cube",),
          "block": ("red cube",), "mug": ("mug",), "cup": ("mug",), "mandarin": ("mandarin",), "orange": ("mandarin",),
          "tangerine": ("mandarin",), "clementine": ("mandarin",), "banana": ("banana",), "plate": ("plate",),
          "dish": ("plate",), "saucer": ("saucer",)}
GENERIC_HEADS = {"wheel", "piece", "part", "object", "toy", "fruit", "tooth", "teeth", "hub", "rim", "edge", "half",
                 "section", "segment", "handle", "decoration", "ornament", "design", "figure", "emblem", "shape"}       # "gear tooth": a piece of a gear


def from_description(label: str, what: str, color: str, long_mm: float) -> str:
    """The class the open description names, if it names a workcell object of another family than the class
    answer and that object can have this size: Apertus told to pick from the list called a yellow gear on the
    black tray a "grey metal part", and asked openly, "gear", yellow. Otherwise the label stays."""
    words = re.findall(r"[a-z]+", what.lower())
    if not words or label in (HAND, EMPTY):
        return label
    fam = [w for w in words if w in FAMILY]
    if not fam or (words[-1] not in FAMILY and words[-1] not in GENERIC_HEADS):
        return label            # "orange ball", "gear-shaped cookie": another object, or not sure
    options = [c for c in FAMILY[fam[-1]] if SIZE_MM[c][1] <= long_mm <= SIZE_MM[c][2]]   # a whole one, this size
    if label in FAMILY[fam[-1]] or not options:
        return label
    if len(options) > 1:        # red or yellow gear: by colour, then by size
        by_colour = [c for c in options if any(color in NEAR.get(k, {k}) for k in CLASS_COLORS[c])]
        options = by_colour or sorted(options, key=lambda c: abs(long_mm - SIZE_MM[c][0]))
    return options[0]


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
KIND_NOUN = {"red gear": "gear", "yellow gear": "gear", "red cube": "cube", "grey metal part": "metal part",
             "white part": "part", "small white part": "part"}
FILLER = {"a", "an", "the", "of", "some", "piece", "part", "region", "inside", "outline", "magenta", "object", "item", "thing"}


def clean_name(what: str) -> str:
    """Apertus's few words, as a short name: "A blue plastic bottle cap." -> "blue plastic bottle cap"."""
    words = [w for w in re.findall(r"[a-z]+", what.lower().replace("gray", "grey")) if w not in FILLER]
    return " ".join(words[:4])


def described(what: str, color: str, long_mm: float) -> tuple[str, str]:
    """(label, name) of a region that is what the open description says, not the class answer: a workcell class if
    the description names one and the size fits; a workcell shape of another size ("big yellow gear"); else the
    description itself ("tennis ball")."""
    words = re.findall(r"[a-z]+", what.lower())
    fam = [w for w in words if w in FAMILY]
    if fam and (words[-1] in FAMILY or words[-1] in GENERIC_HEADS | GENERIC_WORDS):
        fits = [c for c in FAMILY[fam[-1]] if SIZE_MM[c][1] <= long_mm <= SIZE_MM[c][2]]
        if fits:
            return fits[0] if len(fits) == 1 else from_description(fits[0], what, color, long_mm), ""
        noun = KIND_NOUN.get(FAMILY[fam[-1]][0], FAMILY[fam[-1]][0])
        return OTHER, f"{color} {noun}"
    if words and words[-1] in GENERIC_WORDS and "square" in words:
        return OTHER, f"{color} cube"
    return OTHER, clean_name(what) or f"{color} object"


def colour_fits(label: str, color: str, what: str = "") -> bool:
    words = set(re.findall(r"[a-z]+", what.lower().replace("gray", "grey")))
    seen = ({color} if color else set()) | (words & set(COLOR_WORDS))
    return not seen or any(c in NEAR.get(k, {k}) for c in seen for k in CLASS_COLORS.get(label, ()))


def identity(label: str, what: str, color: str) -> tuple[str, str, frozenset]:
    """(class, name, colours) of a region Apertus named. A workcell class keeps its name unless both the colour
    Apertus gives and the colour words it used say another colour: then it is another object of that shape
    ("blue cube"), with Apertus's own words as its name."""
    words = set(re.findall(r"[a-z]+", what.lower().replace("gray", "grey")))
    seen = ({color} if color else set()) | (words & set(COLOR_WORDS))
    if label in CLASS_COLORS:
        own = CLASS_COLORS[label]
        if not seen or any(c in NEAR.get(k, {k}) for c in seen for k in own):
            return label, label, frozenset(own)
        fam = [w for w in re.findall(r"[a-z]+", what.lower()) if w in FAMILY]
        if (fam and label in FAMILY[fam[-1]]) or not clean_name(what) or \
                set(clean_name(what).split()) <= GENERIC_WORDS | set(COLOR_WORDS) | {"square", "round", "shaped"}:
            return OTHER, f"{color} {KIND_NOUN.get(label, label)}", frozenset(seen)      # "blue gear"
        return OTHER, described(what, color, 0)[1] or clean_name(what), frozenset(seen)  # "computer mouse"
    if label == OTHER:
        return OTHER, clean_name(what) or f"{color} object", frozenset(seen)
    return label, label, frozenset()


@dataclass(eq=False)
class Cand:
    mask: np.ndarray
    label: str
    second: str
    score: float
    on: list = field(default_factory=list)
    weak: bool = False      # named by Apertus's second choice: the first did not fit its size
    name: str = ""          # for "other object": Apertus's own words ("tennis ball"); else the class
    colors: frozenset = frozenset()
    first: str = ""         # Apertus's first answer, before the size checks
    hidden: bool = False    # the visible part of an object lying partly under another: may be smaller than its class

    def __post_init__(self):
        self.refresh()

    def refresh(self):
        self.fill = filled(self.mask)
        self.long_mm, self.short_mm = size_mm(self.fill)

    def absorb(self, other: "Cand"):
        self.mask = self.mask | other.mask
        self.refresh()


def _axis(c: "Cand") -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = np.nonzero(c.mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    mid = pts.mean(0)
    _, sv, vt = np.linalg.svd(pts[:: max(1, len(pts) // 4000)] - mid, full_matrices=False)
    return mid, vt[0], float(sv[0] / max(sv[1], 1e-6))


def end_to_end(a: "Cand", b: "Cand", max_gap_px: float) -> bool:
    """Two elongated regions on one line, the gap between their ends small."""
    (ma, da, ra), (mb, db, rb) = _axis(a), _axis(b)
    if ra < 2.5 or rb < 2.5:
        return False
    link = (mb - ma) / max(np.linalg.norm(mb - ma), 1e-6)
    if abs(da @ db) < 0.95 or abs(link @ da) < 0.95:            # within about 18 degrees
        return False
    d = cv2.distanceTransform((~cv2.dilate(a.mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
    return float(d[b.mask].min()) <= max_gap_px


def between(a: "Cand", b: "Cand", o: "Cand") -> bool:
    """Does o cover the gap between a and b (the line joining their centres crosses it)?"""
    ya, xa = np.nonzero(a.mask)
    yb, xb = np.nonzero(b.mask)
    p, q = np.array([xa.mean(), ya.mean()]), np.array([xb.mean(), yb.mean()])
    pts = [p + t * (q - p) for t in np.linspace(0.2, 0.8, 13)]
    return sum(bool(o.mask[int(y), int(x)]) for x, y in pts) >= 4


def stack_rest(cands: list[Cand], log: list) -> list[Cand]:
    """SAM often outlines a stack of equal objects (gears on gears) as one region, too big for one object, and
    the top one on its own. The region minus the top object is the visible rest of the objects underneath."""
    out = []
    for r in [c for c in cands if c.label == EMPTY and c.first in SIZE_MM and c.long_mm > SIZE_MM[c.first][2]]:
        tops = [k for k in cands if k.label == r.first and inside_share(k.mask, r.fill) >= 0.85
                and SIZE_MM[k.label][1] <= k.long_mm <= SIZE_MM[k.label][2]]
        if not tops or r.long_mm > 1.4 * SIZE_MM[r.first][2]:
            continue
        top = max(tops, key=lambda k: k.mask.sum())
        rest = (r.mask & ~cv2.dilate(top.mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)).astype(np.uint8)
        rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        n, comp = cv2.connectedComponents(rest)
        for i in range(1, n):
            m = comp == i
            if m.sum() >= 0.2 * top.mask.sum() and not any(inside_share(m, k.mask) > 0.5 for k in out):
                c = Cand(m, r.first, r.second, r.score, name=r.first, colors=frozenset(CLASS_COLORS[r.first]),
                         first=r.first, hidden=True)
                top.on.append(c)
                out.append(c)
                log.append(f"a stack outlined as one {r.long_mm:.0f} mm region: another {r.first} under the top one")
    return out


def resolve(cands: list[Cand], rgb: np.ndarray, llm: Apertus, log: list | None = None) -> list[Cand]:
    log = log if log is not None else []
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    real = [c for c in cands if c.label in OBJECT_CLASSES or c.label in (HAND, OTHER)]

    # a) nesting: smaller regions inside a larger one are its pieces, or objects lying on it
    real.sort(key=lambda c: int(c.fill.sum()))
    kept: list[Cand] = []
    for c in real:
        inside = [k for k in kept if inside_share(k.mask, c.fill) >= 0.85]
        if c.weak and any(k.label != c.label for k in inside):
            log.append(f"dropped a {c.long_mm:.0f} mm region renamed '{c.label}': it holds a whole other object")
            continue          # too big for what Apertus saw in it: that object plus its surroundings
        for k in inside:
            holds = any(j is not k and inside_share(j.mask, k.fill) >= 0.5 for j in kept)   # something lies on k
            if (k.label == c.label and c.label != OTHER) or c.label == HAND or k.label == HAND:
                same = k.label == c.label
            elif k.first == c.label and c.label in SIZE_MM and k.long_mm < SIZE_MM[c.label][1]:
                same = True     # an end of the grey bar the size rules called a small white part
            elif holds and k.label != OTHER and (edge_gap(lab, k, c) < 12 or same_material(lab, k, c, kept)):
                same = True     # the plate's inner disc with the mandarin on it: the plate's own colour
            elif k.label in SIZE_MM and SIZE_MM[k.label][1] <= k.long_mm <= SIZE_MM[k.label][2] \
                    and edge_gap(lab, k, c) > 25:
                same = False    # a whole yellow gear on the red cube: right size, another colour (Apertus: "part of it")
            else:
                ans = ask_crop(llm, crop_for(rgb, k.mask, around=c.mask), PART_PROMPT.format(parent=c.name), PART_SCHEMA)
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
    # pieces of one object seen on both sides of something lying across it (the two ends of a grey part under the
    # mandarin): same class, both touching the same object, together the size of one such object
    #   (each end alone is too short for a grey part, so the size rules may have renamed it: Apertus's first answer
    #   counts too)
    for a in [c for c in kept if c.label in SIZE_MM and c.long_mm < SIZE_MM[c.label][1]]:
        for b in [c for c in kept if c is not a and c.label in SIZE_MM and c.long_mm < SIZE_MM[c.label][1]]:
            if a not in kept or b not in kept:
                continue
            joint = size_mm(a.fill | b.fill)[0]
            cls = [k for k in dict.fromkeys([a.label, a.first]) if k in SIZE_MM and k in (b.label, b.first)
                   and SIZE_MM[k][1] <= joint <= SIZE_MM[k][2]]
            over = [o for o in kept if o.label not in (a.label, b.label, HAND) and contact(a.mask, o.mask) > 0.05
                    and contact(b.mask, o.mask) > 0.05 and o.long_mm < 1.2 * joint and between(a, b, o)]
            if over and cls:
                a.absorb(b)
                kept.remove(b)
                a.hidden, a.label = True, cls[0]
                a.name, a.colors = cls[0], frozenset(CLASS_COLORS[cls[0]])
                if a not in over[0].on:
                    over[0].on.append(a)
                log.append(f"two pieces of a {a.label} on both sides of the {over[0].label}: one {joint:.0f} mm object under it")
    small = [c for c in kept if c.label in SIZE_MM and c.long_mm < SIZE_MM[c.label][1] and not c.hidden]
    small += [c for c in kept if c.label == OTHER and any(
        contact(c.mask, o.mask) > 0.1 and o.fill.sum() > c.fill.sum() for o in kept if o is not c and o.label != HAND)]
    anchors = [c for c in kept if c not in small]
    for f in sorted(small, key=lambda c: c.long_mm):
        same = [g for g in anchors if g.label == f.label and f.label != OTHER]     # unknown objects are never assumed alike
        touching = sorted([(contact(f.mask, o.mask), o) for o in anchors if o not in same], key=lambda x: -x[0])
        touching = [(t, o) for t, o in touching if t > 0]
        direct = [g for g in same if contact(f.mask, g.mask) > 0]
        via = [g for g in same if any(t > 0.1 and contact(g.mask, o.mask) > 0 for t, o in touching)]
        hand = [o for t, o in touching if o.label == HAND]
        thing = [o for t, o in touching if o.label != HAND]
        if hand and thing and colour_gap(lab, f.mask, thing[0].mask) < colour_gap(lab, f.mask, hand[0].mask):
            hand = []     # touches the arm and an object, and looks like the object where they meet: the mug's handle
            touching = [(t, o) for t, o in touching if o.label != HAND]
        if hand:      # too small to be what Apertus called it, and touching the arm: the wrist, a watch, a sleeve
            hand[0].absorb(f)
            log.append(f"a {f.long_mm:.0f} mm '{f.label}' region touching the arm is part of the person")
        elif direct or via:
            (direct or via)[0].absorb(f)
            log.append(f"{f.label} fragment ({f.long_mm:.0f} mm) joined to its neighbour")
        elif touching:
            parent = touching[0][1]
            ans = ask_crop(llm, crop_for(rgb, f.mask, around=f.mask | parent.mask),
                           NEXT_PROMPT.format(label=f.name, parent=parent.name), NEXT_SCHEMA)
            if ans["answer"] == "separate object":
                anchors.append(f)
                log.append(f"kept a {f.long_mm:.0f} mm {f.label} next to the {parent.label} (partly hidden)")
                continue
            if ans["answer"] == "part of it" and piece_of(lab, f.mask, parent.mask):
                parent.absorb(f)
                log.append(f"a {f.long_mm:.0f} mm '{f.label}' region is part of the {parent.label} (its handle, side or rim)")
            elif f.label == OTHER and not shadow_like(lab, f.mask, parent.mask):
                anchors.append(f)       # an unknown object is kept unless the pixels say shadow: better asked about than lost
                log.append(f"kept a {f.long_mm:.0f} mm '{f.name}' next to the {parent.name}")
                continue
            else:
                log.append(f"dropped a {f.long_mm:.0f} mm {f.label}: shadow, reflection or tray next to the {parent.label}")
        elif too_small(f.label, f.long_mm):
            log.append(f"dropped {f.label} region of {f.long_mm:.0f} mm: too small and nothing hides it")
        else:
            anchors.append(f)
            continue
        kept.remove(f)

    # c) one object split in pieces (the hand and its sleeve, the mug and its handle, a banana cut by a shadow):
    #    touching regions of the same class whose union is no larger than one such object can be
    for c in sorted(kept, key=lambda c: -c.mask.sum()):
        if c not in kept or c.label == OTHER:
            continue
        for g in [g for g in kept if g is not c and g.label == c.label and contact(g.mask, c.mask) > 0]:
            if c.label != HAND and size_mm(c.fill | g.fill)[0] > SIZE_MM[c.label][2]:
                continue          # two red gears side by side
            kept.remove(g)
            c.absorb(g)
            log.append(f"joined two touching {c.label} regions")

    # c1) an unknown object SAM cut in two (the screwdriver's handle and its shaft): the same description, both long
    #     and thin, lying end to end with a gap of at most 20 mm
    for a in [c for c in kept if c.label == OTHER]:
        for b in [c for c in kept if c.label == OTHER and c is not a and c.name == a.name]:
            if a in kept and b in kept and end_to_end(a, b, 20 / MM_PER_PX):
                a.absorb(b)
                kept.remove(b)
                log.append(f"two pieces of one '{a.name}' joined")

    # c2) a region that touches the arm and the edge of the photo is the person's sleeve: people reach in
    #     from the edge; one the arm surrounds on most sides is worn (a watch, a bracelet). (Asking Apertus
    #     here made it call the mug held in the hand "part of the person".)
    hands = [c for c in kept if c.label == HAND]
    for c in [c for c in kept if c.label != HAND]:
        near = [h for h in hands if contact(c.mask, h.mask, r=25) > 0]
        x0, y0, x1, y1 = bbox_of(c.mask)
        if near and (x0 <= 2 or y0 <= 2 or x1 >= rgb.shape[1] - 2 or y1 >= rgb.shape[0] - 2
                     or contact(c.mask, near[0].mask, r=15) > 0.6):
            kept.remove(c)
            near[0].absorb(c)
            log.append(f"a {c.long_mm:.0f} mm '{c.label}' region is part of the person")

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
    """ids like red_gear_1 (numbered left to right when there are several) or tennis_ball."""
    ids: dict[int, str] = {}
    by_name: dict[str, list[Cand]] = {}
    for c in cands:
        by_name.setdefault("hand" if c.label == HAND else c.name, []).append(c)
    for name, cs in by_name.items():
        cs.sort(key=lambda c: np.nonzero(c.mask)[1].mean())
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "object"
        for i, c in enumerate(cs, 1):
            ids[id(c)] = base if len(cs) == 1 else f"{base}_{i}"
    objs = [Obj(ids[id(c)], "hand" if c.label == HAND else c.label, c.mask, [ids[id(o)] for o in c.on if id(o) in ids],
                c.score, "hand" if c.label == HAND else c.name, c.colors) for c in cands]
    return sorted(objs, key=lambda o: o.centroid[0])


# ----------------------------------------------------------------------------- entry point
class Perception:
    def __init__(self, llm: Apertus, cache_dir: Path | None = Path(".cache/perception"), segmenter: Segmenter | None = None):
        self.llm, self.seg = llm, segmenter or Segmenter()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log: list[str] = []
        self.fresh = False          # True: ignore the old filtered cache, run SAM and keep its raw output

    def proposals(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """SAM's regions after the geometry filter. SAM's raw output is what is cached, so the filter can change
        without running SAM again. (An older cache holds filtered regions only: used if there is no raw one.)"""
        digest = hashlib.sha1(rgb.tobytes() + b"p4-1024").hexdigest()[:20]
        raw = self.cache_dir / f"raw_{digest}.npz" if self.cache_dir else None
        old = self.cache_dir / f"sam_{digest}.npz" if self.cache_dir else None
        if raw and raw.exists():
            return prefilter(self._load(raw, rgb.shape), rgb.shape)
        if old and old.exists() and not self.fresh:
            return self._load(old, rgb.shape)
        props = [(m, s) for m, s in self.seg(rgb) if m.sum() >= 1000]
        if raw:
            bits = np.packbits(np.stack([m for m, _ in props]).reshape(len(props), -1), axis=1)
            np.savez_compressed(raw, bits=bits, scores=np.array([s for _, s in props]))
        return prefilter(props, rgb.shape)

    @staticmethod
    def _load(path: Path, shape) -> list[tuple[np.ndarray, float]]:
        z = np.load(path)
        n = shape[0] * shape[1]
        masks = np.unpackbits(z["bits"], axis=1)[:, :n].reshape(-1, *shape[:2]).astype(bool)
        return list(zip(masks, z["scores"].tolist()))

    def __call__(self, rgb: np.ndarray) -> list[Obj]:
        props = drop_flat(self.proposals(rgb), rgb)
        rgb_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        labels = classify(self.llm, rgb, [m for m, _ in props])
        cands = []
        for (m, s), lab in zip(props, labels):
            c = Cand(m, EMPTY, lab.get("second_choice", EMPTY), s)
            x0, y0, x1, y1 = bbox_of(m)
            edge = x0 <= 2 or y0 <= 2 or x1 >= rgb.shape[1] - 2 or y1 >= rgb.shape[0] - 2
            c.label = pick_label(lab, c.long_mm, c.short_mm, edge)
            c.weak = c.label not in (lab["label"], EMPTY)
            c.first = lab["label"]
            what, colour = lab.get("what", ""), lab.get("color", "")
            # the open description wins when both checks rejected the class answer, or when the class is round and the
            # region long and thin (a 179 x 41 mm "plate": the spoon)
            if lab.get("choice") == "description" or (
                    what and c.label in ROUND_CLASSES and c.long_mm > 2 * max(c.short_mm, 1)):
                label, name = described(what, colour, c.long_mm)
                c.weak = c.weak and lab.get("choice") != "description"   # a renamed region stays suspect
                c.label, c.name, c.colors = identity(label, what, colour)
                if name:
                    c.name = name
            else:
                described_cls = from_description(c.label, what, colour, c.long_mm)
                if described_cls != c.label:
                    c.label, c.weak = described_cls, False
                trusted = c.label in (lab["label"], described_cls)
                c.label, c.name, c.colors = identity(c.label, what if trusted else "", colour if trusted else "")
            if c.label in COLOURFUL and saturated_share(rgb_lab, m) < 0.3:
                c.label = EMPTY     # a colourless region cannot be the saucer (yellow rim) or a gear: the plate's well
            cands.append(c)
        self.log = []
        cands += stack_rest(cands, self.log)
        kept = resolve(cands, rgb, self.llm, self.log)
        for c in kept:
            if c.label == OTHER:
                c.colors = frozenset(c.colors | {pixel_colour(rgb, c.mask)} - {""})
        objs = name_objects(kept)
        for o in objs:
            o.crop = crop_for(rgb, o.mask)       # for questions about the object later ("is this a padlock?")
        return objs


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
