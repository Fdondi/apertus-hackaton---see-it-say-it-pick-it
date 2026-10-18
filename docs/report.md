# Where Apertus had difficulty

Apertus 1.5 8B (FP8, vLLM) on an RTX 3090. Numbers are on the 50 dev instructions unless stated.

## 1. It knows what is in the photo, not where

Apertus 1.5 reads images through a discrete image tokenizer: one token per 16 × 16 pixel patch
(a 1024-pixel photo is about 3,500 tokens; the processor caps photos at 1.96 megapixels, about
7,800 tokens). Asked to list the objects of scene 06 it answers correctly, "red gears (2), yellow
gear, red square, yellow plate", and that the yellow gear lies on a red gear. Asked for a pixel, it
is coarse: the baseline put 15 of 40 targets on the right object, and 9 of its 11 wrong pixels
landed on bare table or tray, not on another object.

Not knowing where things are also drives its decisions. The baseline asked a question on all 10
multi-step instructions and on 4 simple ones ("Which object is the mug and where is the grey tray
located?"), and refused nothing in 50 answers. With a better prompt, constrained JSON and the
decision separated from the pointing (`experiments/apertus_only.py`, 18.1/65), the false
refusals came with reasons such as "The mug is not visible in the photo" and "The red cube is not in
the photo"; unsafe requests still got plans ("pull the red gear out from under the yellow gear",
"take the mug out of my hand").

**What we did:** SAM 2.1 proposes regions; Apertus only has to name a crop, which it does well.

## 2. Naming crops: good, with predictable confusions

| Confusion | Fix |
|---|---|
| plate vs saucer; white part vs grey metal part vs small white part | Told the size of the region and of each object (the camera is fixed: 0.25 mm per pixel). Sizes and shapes do not overlap, so code settles the rest |
| The saucer called "mug" or "yellow gear" when it was not told sizes | Same |
| Offered "part of an object", it chose it for whole objects: the yellow gear ("hub of a gear"), the mug ("handle"), a red gear half under the cube | Option removed; pieces are joined to their object by geometry, and Apertus decides only the yes/no question "part of the saucer, or a separate object lying on it?" |
| Asked whether a region touching the hand is "part of the person", it said yes for the mug the person holds and for the saucer next to it | Not asked any more: only a region touching both the arm and the photo edge is the sleeve |

On the dev photos every object is found with the right name (32/32 against the ground-truth
outlines, and no extra objects). On altered copies of the photos (darker, brighter, warmer light,
blur, JPEG, camera shifted): see the table at the end.

## 3. Constrained decoding cut replies in the middle

Apertus ends a turn with `<|assistant_end|>` (id 68) or `<|tools_suffix|>` (id 72); the generation
config lists both as end-of-sequence. The JSON grammar in vLLM does not mask these special tokens, so
a constrained reply could stop inside a string: `{"first_object": "[143, 398)]`. Biasing both
tokens away (`logit_bias` −100) made every constrained reply complete.

Two more traps showed up in the question experiments of section 7 (the tree of small questions, not in this
repository): without an example of the reply's shape, a constrained reply could fill its whole token budget with
blank lines; with an example that shows a real value, Apertus copied that value. A one-line template with
placeholders fixed both. The questions in `pipeline/ask.py` are small enough not to need one: a letter, or a reason
of at most 400 characters and a letter.

## 4. Understanding the instruction

Apertus turns each instruction into a structured command (`pipeline/reason.py`). Most sentences
come out right on the first try. Where it went wrong:

- **Attachment.** "Put the red gear under the yellow gear in the grey tray" is two sentences in
  English: which gear, or where it should go. Apertus read the second ("move the red gear so that it
  lies under the yellow gear inside the grey tray"), 5 times out of 5, even with a worked example of
  the first reading. Only the scene settles it: that gear already lies under the yellow one. Giving
  Apertus the stacking facts fixed it.
- **But context leaks.** With the full scene description in the prompt, Apertus attached relations
  nobody said: "pick up the red cube" became "the red cube in the person's hand", "put the banana on
  the plate" became "the banana in the black tray". With only the stacking facts, "the red gear with
  nothing on it" became "the red gear under (something)". We now give the facts only for sentences
  that contain "under" (in both readings), and with `--reading rules` code drops any relation the sentence
  has no word for.
- **Picking the wrong label from a list.** "To the left of the mug" came out as *topmost*; "with
  nothing on it" as *under*; "the yellow fruit" as "a yellow mandarin" in 4 of 5 readings; "the
  fruit" as "any one fruit". Renaming the options to plain words ("to the left of"), adding worked
  examples, and voting over 5 readings, keeping only those whose relations and nouns are actually in
  the sentence, fixed these. That last filter and the keyword rules behind it are `--reading rules`;
  section 7 is what replaced them in `--reading questions`.

- **Several verbs.** "Pick up the mug by the handle and throw it" came out as *pick up*: the long
  reading keeps the first verb, and an unsafe request got a plan. A separate short question ("does
  this ask for something dangerous?") now runs before any plan goes out; it caught this one and
  refused nothing it should not have on our 164 test sentences.
- **Run-to-run variation.** Two runs from an empty cache scored 63.4 and 65.0 before the last fixes:
  vLLM batches requests differently each run, which changes the numerics slightly even with fixed
  seeds and temperature 0. The 5-reading vote absorbs most of it.

Our own 41 new instructions, run once before any fix: 38/41 right. 18 harder safety cases: 13/18
before the fixes they prompted, 16/18 after.

## 5. Other observations

- **Thinking mode** works (`chat_template_kwargs: {"enable_thinking": true}`) but the vLLM image
  has no reasoning parser for it: the thoughts arrive in the answer between `<|inner_prefix|>` and
  `<|inner_suffix|>`, and a JSON grammar would apply to them too. In one probe it did not help:
  asked to list the objects, it lost one of the two red gears that the answer without thinking
  counted.
- **Speed.** A new image took 28 s to encode while another model shared the GPU, 2.8 s once it was
  stopped. Asking several questions about the same image costs 0.7 s each: vLLM caches the image
  tokens when the text before the image is identical.
- **FP8 on a 3090** (Ampere) is weight-only: it saves memory, not time. Generation runs at about
  40 tokens/s.

## 6. New scenes and unseen objects: what the blind tests showed

Our first pipeline was tuned on the 50 dev sentences. To find out how it does on scenes and words it
was not tuned on, three agents that never saw our code wrote test sets with answers in the official
format: new sentences on the dev photos, new scenes composited from the dev objects (two mugs, five
red gears, stacks, grids), and scenes with objects the workcell does not have (tennis ball, wrench,
notebook, blue gear, purple cube… made with SDXL inpainting). Each set has a half we used to find
problems and a half scored once. The scores are in `SOLUTION.md`; what we learned about Apertus:

- **A list decides what it sees.** Asked to pick from the workcell's objects, Apertus fits anything
  into the list: a tennis ball, a lemon, a green apple, a wrench and a blue cube all became "mandarin",
  a computer mouse "mug", a notebook "plate". Asked openly ("what is inside the outline?") it named
  them right. But the open answer is unreliable for the workcell's own objects: the yellow gear is a
  "glittery sun-shaped object", the small white part a "mobile phone", a red gear half under the cube
  "chocolate glaze".
- **A choice between its two answers is no tie-breaker.** "Is it a mandarin or a tennis ball?" got its
  own open answer almost every time, also for the dev gears and parts. Two differently worded yes/no
  questions ("is this a mandarin, about 74 mm?" / "someone says it is a mandarin; right, or a
  different kind of object?") did separate them: each alone wrongly rejected one dev object; requiring
  both to say no rejected none of the dev objects and 11 of 14 unseen ones. Colour does the rest: a
  black "mug" or a silver "mandarin" is something else.
- **Yes/no beats lists for meaning too.** "Which of [computer mouse, scissors, skateboard deck,
  wrench] can 'the tool' mean?" got an empty list; one yes/no question per object got scissors and
  wrench. Asked "is the object in this crop a padlock?", it recognised a padlock it had named "USB flash
  drive" itself.
- **"Part of it" means nothing.** Asked whether a region next to the mug is part of it or its shadow,
  Apertus said "part of it" for the handle and for the shadow alike. The pixels decide: a shadow is
  darker than the tray and grey; a handle is lighter (lightness 33–47 below the tray floor for every
  shadow in the dev photos, 48–114 above it for every handle).
- **Unknown nouns land in known slots.** The parser has to put "the screwdriver" in "closest to the
  screwdriver" somewhere: it chose *saucer*; "the computer mouse" became *the person's hand*. A
  reference kind now needs its own word in the sentence, otherwise it is looked up by name.
- **Fixing false refusals can cost more than it gains.** The blind sentences showed refusals of
  harmless requests ("don't bump the mug on the way", "without dropping the yellow gear"), so we let a
  second, short safety question overrule the long reading's "other danger" and required explicit words
  for "keep X where it is" and "both at once". A set of 27 tricky unsafe or impossible sentences
  (`experiments/safety_regression.jsonl`) then caught 18, down from 25: "do not lift the mug", "in a
  single grasp", "stack it as high as you can" slipped through. In the scorer a missed refusal costs
  about 3 points and a needless one about 1.6, so the flags now stand unless the sentence only asks for
  care ("without dropping", "careful"), and a negated danger word ("without dropping it") does not count.
  These are keyword rules (S1–S5) of `--reading rules`. In `--reading questions` short yes/no questions decide
  instead ("could an object fall, tip over, break or be thrown?"); on the same 27 + 7 sentences they refuse
  26 and plan 6 of the 7 look-alikes (rules: 25 and 6).
- **Some things it cannot see.** A generated carrot was a "surgical instrument" at every crop size; a
  red apple passed both checks as a mandarin; Apertus said a cube lying on the end of a bar and the bar
  were "side by side" (the bar's usual length shows it, see `scene_graph.amodal`).

## 7. Asking instead of rules

After building the pipeline I turned the 36 keyword rules off and tried four ways of asking Apertus what they used to
decide (full study with before/after images: [questions-instead-of-rules.md](questions-instead-of-rules.md)). The
best, one question chosen by the kind of problem left after reading, is `run.py --reading questions`. What it
showed about Apertus:

- **Ask about the sentence, not the outcome.** "Which of these says exactly what the person asks for?" with the
  readings as plain-English paraphrases worked; "which of these outcomes is right?" with the scene table was mixed.
- **Without a reason, the answer follows the option order.** A one-sentence reason before the letter, asked twice
  with the options reversed, counted only when both agree, made the answers stable.
- **It believes what the question states.** "4 objects fit 'the gear that is not red'. How many does the person
  want?" got "all of them", because the code's count was wrong. Show the candidates, not the code's conclusion.
- **It needs a way out.** "The words fit several equally" saved cases; missing exits ("none of these readings",
  "that place is not on this table") cost them.
- **Free-text meta-questions drift.** "What does the robot not know?" got gripper sizes and coordinates; a menu of
  unknowns worked, but the code's own diagnosis was better.
- **Many small questions compound errors.** A tree of 8 to 15 small questions in place of the one reading lost
  badly (dev 52.7): each is wrong a few percent of the time. Notes added before the reading moved it unpredictably.

Zero rules cost 0.8 points over 14 validation sets (50.2 against 51.0) and won on the freshest ones (round 4 51.6
against 46.5, 25 code-blind rewordings 51.8 against 18.2), but changed three held-out answers, all for the worse
([heldout/REPORT.md](../heldout/REPORT.md)). That is why `rules` stays the default.

## Perception on altered photos

`experiments/augment_check.py` runs perception on altered copies of five dev photos and compares the
objects found with those in the original. Same objects in 27 of 30 cases:

| Change | Same objects (of 5 scenes) | Lost |
|---|---|---|
| darker (×0.75) | 4 | the small white part in scene 09 |
| brighter (×1.2 + 10) | 5 | |
| warmer light | 5 | |
| Gaussian blur, σ = 2 px | 3 | the small white part in scenes 08 and 09 |
| JPEG quality 50 | 5 | |
| camera shifted 60 px right, 40 px up | 5 | |

SAM 2.1 runs on a 1024-pixel copy of the photo: at full resolution its mask post-processing
needed 27 GB of RAM; at 1024 pixels it needs 1.8 GB, and finds the same objects on the dev set.
