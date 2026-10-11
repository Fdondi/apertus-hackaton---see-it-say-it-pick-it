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

We tuned on the dev set, so 65/65 there says little about new scenes. Two checks that say more:

- **Our own instructions** on the dev scenes (`experiments/stress.jsonl`, 55 sentences with new wording,
  relations the dev set does not use, new unsafe verbs): 48/55 on the first run, 55/55 after fixes.
  A second set written after that (`experiments/stress2.jsonl`, 41 sentences): 38/41 on its first
  run; the misses were two missing words ("lowest", "the middle one", since added) and one label we
  got wrong ourselves. 41/41 now. A third set of 18 harder safety cases (`experiments/stress3.jsonl`):
  13/18 at first, 16/18 now.
- **Perception on altered photos** (`experiments/augment_check.py`: darker, brighter, warmer light,
  blur, JPEG artefacts, camera shifted 60 px): the same objects in 27 of 30 cases; heavy blur lost
  the small white part twice, a darker photo lost it once.
- **Perception against the ground-truth outlines** (`experiments/check_perception.py`): all 32 objects
  found, named right, with the target pixel on them; no extra objects in any dev scene.

## The pipeline

```
photo ──► SAM 2.1 ──► ~30 unnamed regions ──► Apertus names each crop ──► clean-up ──► objects + masks
                                               (constrained choice,        (sizes, inventory,
                                                told the region's size)    parts vs objects)
objects ──► scene graph: grey/black tray, what lies on what, the person's hand, positions, distances

instruction ──► Apertus: structured command ──► resolve on the scene graph ──► safety rules ──► plan
                (5 readings, vote)               (which objects?)               ask / refuse    pick/place
```

| Step | File | Model | What it does |
|---|---|---|---|
| Regions | `pipeline/perception.py` | SAM 2.1 large | Automatic masks, 48×48 seed points. No names |
| Filter | `pipeline/perception.py` | — | Drops specks, tray walls, rails, and regions with no colour contrast to the tray (marks, moulded bumps) |
| Naming | `pipeline/perception.py` | **Apertus** | Each region as a crop with a magenta outline; Apertus picks one of the 11 workcell objects, "hand or arm", "tray or table" or "other", plus a second choice. It is told the size of the region and of each object |
| Clean-up | `pipeline/perception.py` | **Apertus** + geometry | Known sizes fix plate/saucer and the three flat parts; pieces of an object (gear hub, inside of the mug, centre of the saucer) are joined to it; for a region inside another object Apertus decides "part of it" or "separate object lying on it"; for a region too small for its object touching another one, "side/shadow of it" or "partly hidden object" |
| Scene graph | `pipeline/scene_graph.py` | — | Grey tray found by colour; stacking from outlines (an object on top covers where the lower one would be: enclosing circle or convex hull); the hand; distances between centres in mm (0.25 mm per pixel on the table) |
| Understand | `pipeline/reason.py` | **Apertus** | The instruction becomes JSON (constrained decoding): action verb; per step the target (kind, colour, how many, relation to another object, exceptions) and the destination; what must not move; "that one" without a name; other dangers. 22 worked examples written for this purpose (not from the dev set) |
| Decide | `pipeline/plan.py` | — | Resolves the target on the scene graph; refuses unsafe verbs, objects in a hand, absent objects, objects wider than the gripper, things that do not fit, contradictions; asks when the words fit several objects or nothing is named; otherwise plans, moving whatever lies on the target first |
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

A first run takes about 45 s per scene on the 3090 (SAM, ~30 short Apertus calls to name regions,
5 readings per instruction): under 8 minutes for the 10 dev scenes. With the CPU build of torch
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
| `experiments/` | `apertus_only.py` (the A/B point), `check_perception.py` (perception vs ground truth), `stress.py` with `stress.jsonl`, `stress2.jsonl`, `stress3.jsonl`, `augment_check.py` |
| `docs/report.md` | Where Apertus had difficulty |
| `docker-compose.yml` | Apertus in vLLM |

## Licences

Our code: Apache 2.0. Apertus 1.5: see its model card (swiss-ai). SAM 2.1: Apache 2.0. Data in
`dataset/`: CC BY 4.0.
