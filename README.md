# See It, Say It, Pick It: my attempt

This repository is my attempt at the **Hack Apertus 2026 ZHAW challenge "See It, Say It, Pick It"**.
It is built on the challenge repository by Fatemeh Mohammadi Amin (ZHAW School of Engineering,
[Fatemeh-MA/see-it-say-it-pick-it](https://github.com/Fatemeh-MA/see-it-say-it-pick-it)). The dataset,
the scorer, the schema and the baseline are hers; the pipeline in `run.py` and `pipeline/` is mine.

**Approach in one sentence.** SAM 2.1 cuts the photo into regions, Apertus names each region and reads the
instruction into a small structured command, and code resolves that command on a scene graph, applies the
safety checks and orders the picks.

| | Script score (out of 65) on the 50 dev instructions |
|---|---|
| Baseline (Apertus reads the photo, plain prompt) | 15.1 |
| Apertus alone, better prompt and constrained JSON | 18.1 |
| **This pipeline** | **65.0** (tuned on these 50, see below for unseen data) |

The full description of the pipeline is in [SOLUTION.md](SOLUTION.md); where Apertus had difficulty is in
[docs/report.md](docs/report.md); the held-out run is in [heldout/REPORT.md](heldout/REPORT.md).

## The challenge, in short

One photo of a robot workcell and one English instruction (*"put the red gear under the yellow gear in the
grey tray"*) in; out: the pixel of the object the robot should pick first and the ordered pick/place actions,
or a question when the instruction is ambiguous, or a refusal when it is impossible or unsafe. Apertus must do
the language reasoning; everything runs locally. Scoring: 35 % grounding and 30 % safety are computed by
`starter/evaluate.py` (65 points), the rest is judged. Details: [EVALUATION.md](EVALUATION.md).

## Getting started

Serving Apertus was the hard part of the setup, so this section is longer than the challenge's original
quickstart. Tested on Windows 11 with Docker Desktop (WSL2) and an RTX 3090 (24 GB).

**You need:** an NVIDIA GPU with 16 GB or more, Docker with GPU support, a Hugging Face token, Python 3.11 or
3.12, and about 35 GB of free disk (the vLLM image is about 20 GB, the FP8 model about 11 GB).

```
git clone <this repository> && cd see-it-say-it-pick-it

# 1. Apertus 1.5 8B (FP8) in vLLM, port 8000
echo HF_TOKEN=hf_... > .env
docker compose up -d
curl http://localhost:8000/v1/models        # wait until this lists onprem-ai/Apertus-v1.5-8B-FP8 (minutes on first start)

# 2. the pipeline (SAM 2.1 runs in this Python; for a GPU, install torch from pytorch.org first)
python -m venv .venv
.venv/bin/pip install -r requirements.txt  # Windows: .venv\Scripts\pip
.venv/bin/python run.py --data dataset --out results.json
python starter/evaluate.py results.json --data dataset

# any folder of scenes (one sub-folder per scene with rgb.png and instructions.txt)
.venv/bin/python run.py --data heldout --out results_heldout.json

# the same, without keyword rules: Apertus answers follow-up questions instead (see "Two readings" below)
.venv/bin/python run.py --data dataset --out results_q.json --reading questions
```

A first run takes about a minute per scene; every SAM mask and every Apertus reply is cached in `.cache/`,
so a second run takes seconds. Each run also writes `<out>.details/` with an overlay per scene and, per
instruction, the reading, the votes and how the decision was reached.

### What made serving Apertus hard, and what this repository does about it

| Problem | What I saw | What to do |
|---|---|---|
| Which server | The pipeline needs Apertus 1.5 with images and constrained JSON | `docker-compose.yml` uses the vLLM image swiss-ai publishes for Apertus 1.5 (`ghcr.io/swiss-ai/vllm_apertus_1.5_release`) with `--chat-template-content-format string`, serving `onprem-ai/Apertus-v1.5-8B-FP8` |
| Constrained replies cut off | JSON replies stopped in the middle of a string | Apertus ends a turn with special tokens (ids 68 and 72) that vLLM's JSON grammar does not mask. `pipeline/llm.py` biases them away (`logit_bias` −100); any other client needs the same |
| Blank-line loops | In later experiments, a constrained reply filled the token budget with newlines | Show the model the shape of the reply: worked examples (as `pipeline/reason.py` does), or a one-line JSON template with placeholders, not real values (the model copies those) |
| GPU memory | SAM 2.1 and Apertus did not fit together on 24 GB | `--gpu-memory-utilization 0.62` in `docker-compose.yml` leaves room for SAM |
| Slow images | Encoding a photo took 28 s instead of 2.8 s | Another model (LM Studio) was sharing the GPU. Stop other GPU programs before a run |
| RAM growth on Windows | Over long sessions the container's memory grew to about 40 GB | Cap WSL2 memory in `%UserProfile%\.wslconfig` (`[wsl2]` `memory=32GB`), and `docker compose stop` when you are done |
| Thinking mode | Thoughts arrive inside the answer | They end with `<|inner_suffix|>`; `pipeline/llm.py` strips them. The pipeline does not use thinking mode |
| FP8 on Ampere | No speed-up on the 3090 | FP8 is weight-only there: it saves memory, not time (about 40 tokens/s) |
| Run-to-run variation | Two runs from an empty cache scored 63.4 and 65.0 | vLLM batches requests differently each run. The pipeline takes five readings per instruction and votes |
| Server stopped | `run.py` stops with "cannot reach Apertus" | With a warm cache you can re-run without the server: pass `--model onprem-ai/Apertus-v1.5-8B-FP8` so the client does not ask the server for its model name; a cache miss still stops the run instead of guessing |

No GPU at all: `python baseline/baseline.py --mock` runs the challenge's baseline loop without a model, to
check the data and the scorer.

## Results

Script score out of 65 for the code in this repository, in both readings (see the next section):

| Test set | Items | `rules` (default) | `questions` |
|---|---|---|---|
| Dev set (both tuned on it) | 50 | 65.0 | 65.0 |
| Round 1: new sentences / new scenes / objects the workcell lacks, each in two halves (mean of 6) | 320 | **57.1** | 50.6 |
| Round 2: new sentences on the dev photos | 80 | **56.1** | 53.3 |
| Round 2: new scenes from the dev objects | 40 | **53.5** | 48.8 |
| Round 2: objects the workcell lacks | 40 | **37.0** | 30.6 |
| Round 3: new sentences on the dev photos | 60 | 56.0 | **56.6** |
| Round 3: new scenes from the dev objects | 42 | 50.4 | **51.2** |
| Round 4: reworded and unsafe sentences | 105 | 46.5 | **51.6** |
| Probes: code-blind rewordings | 25 | 18.2 | **51.8** |
| Safety set: unsafe sentences and harmless look-alikes | 34 | 53.8 | **54.8** |
| Mean of the 14 validation sets | 746 | **51.0** | 50.2 |
| Held-out scenes (no ground truth; my judgement of the photos) | 40 | **39**: 31 of 32 plans, 8 of 8 questions or refusals | 36: 28 of 32 plans, 8 of 8 |

The blind sets were written in the official format by agents that never saw the code. The gap between dev and
fresh data (9 to 28 points on the blind rounds; 47 on the probes with `rules`) is the real result: the pipeline is
tuned. Each round was scored once when it was new
(those scores, and the first version's, are in [SOLUTION.md](SOLUTION.md)); rounds 1 to 3 were then used for
development, so only round 4, the probes and the held-out scenes are close to unseen, and round 4 also helped design
the yes/no questions of `questions`. The held-out differences between the readings are in
[heldout/REPORT.md](heldout/REPORT.md).

## Two readings: `--reading rules` or `--reading questions`

The original pipeline repairs Apertus's reading of the instruction with 36 keyword rules. Later I tried several
ways of asking Apertus instead (see Findings, and the study with before/after images for each case in
[docs/questions-instead-of-rules.md](docs/questions-instead-of-rules.md)). The best of them is in the repository as
`--reading questions`. Neither dominates the other (table above), so both are kept, and the rules stay the
default:

- **`rules`** is better on the sentences its rules were written against (rounds 1 and 2) and on the held-out
  scenes. It is the original pipeline. Its 36 rules are marked R1–R30, S1–S5 and V in comments in `pipeline/plan.py`
  and `run.py`.
- **`questions`** has no keyword rules for reading the instruction. When the readings disagree, or the command
  cannot be carried out as read (nothing fits, several objects fit, no place to put it), Apertus answers one
  short question chosen by the kind of problem, and the answer is applied (`pipeline/ask.py`, `h3()` in
  `run.py`). It does better on the freshest wording (round 4, the probes) and on the safety set, and costs
  about one extra Apertus call per instruction. One keyword gate stays in both modes: the reading gets the
  stacking facts only for sentences with a word like "under".

Both modes were checked against the experiments they come from: on all 15 test sets they give exactly the
same answers, without a single new Apertus call.

## Findings

**About Apertus.**
- Reading the photo directly is weak: the baseline scores 15.1. Apertus is much better at naming one cropped
  region at a time, and at reading the sentence, than at finding objects in a whole photo.
- It reads most instructions right, but slips in a large structured form: "with nothing on it" picked as
  "under", "both fruits" as "any one", "the mandarin on the plate" (where it should go) as where it is now.
- Attachment is genuinely ambiguous ("put the red gear under the yellow gear in the grey tray"); only the scene
  settles it, so the reading gets the stacking facts for such sentences, and nothing else from the scene,
  because more context made it invent relations.
- With several verbs it keeps the first ("pick it up and throw it"); a short separate safety question catches
  that.

**Rules against questions.** After building the pipeline I studied the 36 keyword rules that fix Apertus's
readings in `pipeline/plan.py` and `run.py`. The best variant is in the repository (`--reading questions`);
the others are described here but not included.
- Only 2 of the 36 rules change any answer on the 50 dev instructions, yet removing them costs 5 to 8 points
  on the sets they were written on, and 18 points on a safety set: they were tuned on test data.
- Short yes/no questions to Apertus ("does the person ask to take an object out of a person's hand?") replace
  the safety keyword lists and win most of that back.
- The best variant asks Apertus one plain question chosen by the kind of problem left after reading (the
  readings disagree, nothing fits, several objects fit, no place found). With **zero rules** it averages 50.2
  over 14 validation sets against 51.0 for the 36 rules, and beats them on the freshest sets (round 4 51.6
  against 46.5, a safety set 54.8 against 53.8). It also handled cases no rule anticipated ("the gear that is
  sitting on its own", "place the mug on my lap").
- Notes from a checklist before the reading, or a tree of small questions in place of the reading, did worse:
  extra text moves the reading unpredictably, and many small answers compound their errors.
- Questions work for an 8B model when they are about the sentence rather than the outcome, offer a way out
  ("the words fit several equally"), never state the code's guess as a fact, and ask for a one-sentence reason
  before a lettered answer, asked twice with the options reversed.
- Every variant still changed some held-out answers for the worse, so none replaces the committed rules
  before a fresh test.

## Repository

| Path | What it is |
|---|---|
| `run.py` | The one command: runs the pipeline on any folder of scenes |
| `pipeline/` | `llm.py` (client, constrained JSON, cache), `perception.py` (SAM 2.1 + Apertus naming), `scene_graph.py`, `reason.py` (reading the instruction), `plan.py` (decisions and safety; the rules of `--reading rules`, marked R1–R30, S1–S3 and V in comments), `ask.py` (the follow-up questions of `--reading questions`) |
| `docker-compose.yml` | Apertus 1.5 8B FP8 in vLLM |
| `SOLUTION.md` | The pipeline, step by step, and the test results |
| `docs/report.md` | Where Apertus had difficulty |
| `docs/questions-instead-of-rules.md` | The rules-against-questions study, one section per case with before/after images (in the folder of the same name) |
| `heldout/` | The five held-out scenes and `REPORT.md` |
| `experiments/` | Apertus-only comparison, perception checks, stress and safety sets |
| `dataset/`, `starter/`, `schema/`, `baseline/`, `examples/`, `EVALUATION.md`, `docs/DATA.md`, `docs/safety_primer.md` | From the challenge repository |

## Credits and licences

Challenge, dataset, scorer and baseline: Fatemeh Mohammadi Amin, ZHAW School of Engineering (mohm@zhaw.ch).
Pipeline: Francesco Dondi. Code: Apache 2.0. Data in `dataset/`: CC BY 4.0. Apertus 1.5: see its model card
(swiss-ai). SAM 2.1: Apache 2.0.
