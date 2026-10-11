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
  that contain "under", and code drops any relation the sentence has no word for.
- **Picking the wrong label from a list.** "To the left of the mug" came out as *topmost*; "with
  nothing on it" as *under*; "the yellow fruit" as "a yellow mandarin" in 4 of 5 readings; "the
  fruit" as "any one fruit". Renaming the options to plain words ("to the left of"), adding worked
  examples, and voting over 5 readings, keeping only those whose relations and nouns are actually in
  the sentence, fixed these.

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
