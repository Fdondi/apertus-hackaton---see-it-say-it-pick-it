# See It, Say It, Pick It: our pipeline

**Approach in one sentence.** SAM 2.1 cuts the photo into regions; Apertus names each region, reads
the instruction into a small structured command, and code resolves that command on a scene graph
and applies the safety rules.

Apertus does every language step: naming what it sees, understanding the instruction, deciding
which reading is meant. The one open vision model, SAM 2.1, only proposes regions without names.
Everything runs locally: Apertus 1.5 8B in vLLM (Docker), SAM 2.1 through `transformers`.

## Results on the dev set (50 instructions, `starter/evaluate.py`)

| | Grounding /35 | Safety /30 | Script total /65 |
|---|---|---|---|
| Baseline (`baseline/baseline.py`: Apertus reads the photo, plain prompt) | 13.1 | 2.0 | 15.1 |
| Apertus alone, better prompt + constrained JSON + decision split from pointing (`experiments/apertus_only.py`) | 11.4 | 6.7 | 18.1 |
| **This pipeline** (`run.py`) | **35.0** | **30.0** | **65.0** |

We tuned on the dev set, so 65/65 there says little about new scenes. To measure that, agents that
never saw our code wrote test sets in the official format (same scorer, same 80/20 mix of plans and
questions/refusals):

| Blind test set (written by an agent that never saw the code) | Size | First version | This version |
|---|---|---|---|
| Round 3: new sentences on the dev photos | 60 | 50.0 | **53.2** |
| Round 3: new scenes from the dev objects, two with objects the workcell does not have (kiwi, pen, battery, blue cube) | 42 | 32.2 | **48.6** |
| Round 2: new sentences on the dev photos | 80 | 46.1 | 52.3 |
| Round 2: new scenes composited from the dev objects (two mugs, five red gears, stacks, grids, an empty tray) | 40 | 25.3 | 47.3 |
| Round 2: scenes with objects the workcell does not have (SDXL-inpainted padlock, peach, rubber duck, dice…; recoloured and resized gears and cubes) | 40 | 14.2 | 37.0 |
| Dev set | 50 | 65.0 | **65.0** |

Each round was written after we froze the code and scored once; round 3 is the final code. (Round 2
with the code frozen when it was written: 49.2, 47.3, 37.0; the numbers in the table include fixes
found on round-1 data before anyone looked at round 2.) A targeted set of 27 tricky unsafe or
impossible sentences plus 7 harmless look-alikes (`experiments/safety_regression.jsonl`): 25/27 refused
and 6/7 planned, against 25/27 and 4/7 for the first version. After round 3 was scored we fixed three
wording gaps it showed ("between the other two", "the gear that isn't red", "starting with the
smallest" / "from right to left"); the dev set and the safety set did not change. A first round of the same
three kinds (each half used to find problems, half scored once before further fixes) gave, for its
held-back halves, 48.2 → 53.8, 31.5 → 38.4 and 12.6 → 37.0. What changed between the two versions
and what Apertus found hard is in `docs/report.md` (section 6). The test sets live outside the
repository (about 300 MB of photos); the scripts that made them are kept with them.

- **Perception on altered photos** (`experiments/augment_check.py`: darker, brighter, warmer light,
  blur, JPEG artefacts, camera shifted 60 px): the same objects in 27 of 30 cases; heavy blur lost
  the small white part twice, a darker photo lost it once.
- **Perception against the ground-truth outlines** (`experiments/check_perception.py`): all 32 objects
  found, named right, with the target pixel on them; no extra objects in any dev scene. Outline overlap
  with the ground truth (`experiments/mask_quality.py`): mean IoU 0.95; the mug's outline now includes
  its handle (mug IoU 0.90 → 0.94, worst case 0.79 → 0.89).
- **Our own early instruction sets** (`experiments/stress*.jsonl`, 114 sentences on the dev scenes) were
  written before the blind tests and refer to the object ids of the first version.

## The pipeline

```
photo ──► SAM 2.1 ──► ~30 unnamed regions ──► Apertus names each crop ──► clean-up ──► objects + masks
                                               (workcell list + open        (sizes, colours,
                                                description, checked)       parts vs objects)
objects ──► scene graph: grey/black tray, what lies on what, the person's hand, positions, distances

instruction ──► Apertus: structured command ──► resolve on the scene graph ──► safety rules ──► plan
                (5 readings, vote)               (which objects?)               ask / refuse    pick/place
```

| Step | File | Model | What it does |
|---|---|---|---|
| Regions | `pipeline/perception.py` | SAM 2.1 large | Automatic masks, 48×48 seed points. No names. SAM's raw output is cached, the filter below runs after loading |
| Filter | `pipeline/perception.py` | — | Drops specks, tray floors, tray walls (long strips along the image axes), rails, and regions with no colour contrast to the tray (marks, moulded bumps). A pen, a screwdriver or a book passes |
| Naming | `pipeline/perception.py` | **Apertus** | Each region as a crop with a magenta outline. (1) Apertus picks one of the 11 workcell objects, "hand or arm", "tray or table" or "other", plus a second choice, told the size of the region and of each object. (2) Asked openly, it names the object and its colour. (3) If the two disagree on a whole object of the right size, two differently worded yes/no checks ("is this a mandarin, about 74 mm?"); only if both say no does the open description win ("tennis ball"). A workcell shape in a colour it does not have becomes "blue gear", "yellow cube" |
| Clean-up | `pipeline/perception.py` | **Apertus** + geometry | Known sizes fix plate/saucer and the three flat parts; a colourful class (gear, cube, mandarin, saucer…) needs colourful pixels; pieces of an object (gear hub, inside of the mug, centre of the saucer) are joined to it; for a region inside another object Apertus decides "part of it" or "separate object lying on it" (a whole object of another colour is separate without asking); a region next to an object is its part (the mug's handle), its shadow, or another object: Apertus's answer, checked against the pixels (a shadow is darker and grey); the two ends of a bar under a mandarin are one bar; a stack SAM outlines as one region holds more objects under the top one; no fixed inventory: any number of objects of any kind |
| Completion | `pipeline/perception.py` | SAM again + **Apertus** | A region too small for what it was named (the valve's plug alone is 55 mm) is grown with the raw SAM regions that touch it, taken from a second SAM pass over a 640 px window around it at native resolution; the result counts only if it is the size of a workcell object, overlaps no whole object, is as wide as that object is, and Apertus names it. A region Apertus named a workcell object that is only up to 25 % too large (the cube seen from the side, 111 mm against 105) is kept if Apertus says yes to "is it a red cube?" without the size |
| Scene graph | `pipeline/scene_graph.py` | **Apertus** + geometry | Grey tray found by colour; stacking from outlines (an object on top covers where the lower one would be: enclosing circle, convex hull, or the usual length of a bar whose end is hidden); for pairs pressed together that the outlines leave open, Apertus is asked which lies on top; the hand (and what it wears); distances between centres in mm (scale from the calibration) |
| Understand | `pipeline/reason.py` | **Apertus** | The instruction becomes JSON (constrained decoding): action verb; per step the target (kind, colour, how many, relation to another object, exceptions) and the destination; what must not move; "that one" without a name; other dangers. 22 worked examples written for this purpose (not from the dev set) |
| Decide | `pipeline/plan.py` | **Apertus** (lookups) + rules | Resolves the target on the scene graph; refuses unsafe verbs, objects in a hand, absent objects, objects wider than the gripper (unknown objects: by their measured width), things that do not fit (unknown containers: by their size), contradictions; asks when the words fit several objects or nothing is named; otherwise plans, moving whatever lies on the target first. For objects outside the workcell list, Apertus answers yes/no per object: "can 'the tool' mean the wrench?", and, when no name matches, "is the object in this crop a padlock?" |
| Vote | `run.py` | **Apertus** ×5 | One greedy and four sampled readings; readings that use a relation the sentence never says are discarded; majority vote on the decision |
| Safety check | `pipeline/reason.py` | **Apertus** | Before a plan goes out, one short separate question: does the sentence ask for something dangerous (throw, drop, push, pull out from under, take from or put on a person, hold two objects at once)? The long reading sometimes keeps only the first verb of "pick up the mug and throw it" |

The answer's `target_pixel` is the visible pixel deepest inside the first object picked (the
maximum of the distance transform of its mask): always on the object, and the middle of its top
surface for the real robot.

## Run it (about 15 minutes, most of it downloads)

Needs an NVIDIA GPU with 16 GB or more (tested on an RTX 3090, 24 GB) and Docker with GPU support.

```
git clone <this repository> && cd see-it-say-it-pick-it
echo HF_TOKEN=hf_... > .env              # a Hugging Face token
docker compose up -d                     # Apertus 1.5 8B FP8 in vLLM on port 8000; first start downloads ~11 GB
python -m venv .venv && .venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python run.py --data dataset --out results.json             # SAM 2.1 (~0.9 GB) downloads on first use
python starter/evaluate.py results.json --data dataset
```

On any folder of scenes (one sub-folder per scene with `rgb.png` and `instructions.txt`):

```
python run.py --data heldout --out results_heldout.json
```

A first run takes about a minute per scene on the 3090 (SAM, ~60 short Apertus questions about the
regions, 5 readings per instruction): about 10 minutes for the 10 dev scenes. With the CPU build of torch
(what `pip install -r requirements.txt` gives on Windows) SAM runs on the CPU and a scene takes
about 2 minutes, with the same answers; Apertus still needs the GPU. SAM needs about 2 GB of RAM. Everything is cached in `.cache/` (masks, every Apertus reply), so a
second run takes seconds. `run.py --help` lists the options (`--base-url` for another server,
`--votes`, `--cache`).

Each run also writes `<out>.details/`: an overlay per scene with every object found and its grasp
pixel, the scene graph as text, and for every instruction the reading, the votes and how the
decision was reached. This is what the jury reads to check the vision step.

`docker-compose.yml` uses `--gpu-memory-utilization 0.62` so SAM fits next to Apertus on a 24 GB
card. The JSON grammar does not mask Apertus's turn-end tokens, so `pipeline/llm.py` biases them
away; without that, constrained replies stop mid-string (see `docs/report.md`).

## Files

| Path | |
|---|---|
| `run.py` | The one command |
| `pipeline/` | `llm.py` (client, constrained JSON, cache), `perception.py`, `scene_graph.py`, `reason.py`, `plan.py` |
| `experiments/` | `apertus_only.py` (the A/B point), `check_perception.py` (perception vs ground truth), `mask_quality.py` (outline IoU vs ground truth), `stress.py` with `stress.jsonl`, `stress2.jsonl`, `stress3.jsonl`, `augment_check.py` |
| `docs/report.md` | Where Apertus had difficulty |
| `docker-compose.yml` | Apertus in vLLM |

## Licences

Our code: Apache 2.0. Apertus 1.5: see its model card (swiss-ai). SAM 2.1: Apache 2.0. Data in
`dataset/`: CC BY 4.0.
