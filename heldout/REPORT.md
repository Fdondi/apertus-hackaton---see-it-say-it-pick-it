# Held-out run (held_01 – held_05, 40 instructions)

`python run.py --data heldout`. The folder has no ground truth (only `instructions.txt` with the category), so
"right" and "wrong" below are my reading of the photos and the overlays, not a score.

| | First run (committed code f47f24d) | After the fixes (working tree) |
|---|---|---|
| Plans I judge right | 26 of 32 | 31 of 32 |
| Plans wrong or refused | 4 + 2 doubtful | 0 + 1 doubtful |
| Ask / refuse answered with the right action | 7 of 8 | 8 of 8 |

Rough script score (EVALUATION.md formulas, scaled to 32 plans and 8 non-plans): grounding about 30 → 34 of 35,
safety about 23 → 29 of 30. Cold-cache run (empty perception cache, as on release day) gives the same 40 answers.

## What was missed, and what fixed it

| id | instruction | before | now | cause and fix |
|---|---|---|---|---|
| held_03_i1 | pick up the small white part | refuse "no small white part" | pick | SAM broke the valve into pieces, none of them the object, and the largest was merged into the grey part. **Completion**: an undersized region is grown with SAM regions from a second look at a 640 px window around it; Apertus names the 83 × 58 mm result |
| held_03_i4 | put both grey parts and the small white part in the grey tray | refuse | plan, 3 objects | same |
| held_03_i7 | everything except the mug and the plate | 4 objects | 5 objects | same |
| held_03_i8 | put the mug inside the small white part | refuse (for the wrong reason) | refuse "does not fit" | the fit check only knew mugs and unknown objects as containers; now any object the sentence says "in/inside" |
| held_04_i7 | pick up the cube | refuse "no cube" | pick | not a merge: the cube was 111 mm seen from the side, the red cube window ends at 105 mm, so the size rule discarded a correctly named region. **Near miss**: up to 25 % too large and Apertus says yes to "is it a red cube?" without the size, it is kept |
| held_04_i3 | put the red gear under the cube on the plate | refuse | cube first, then the gear | same, plus "circular saw blade" (Apertus's name for the toothed crescent of the gear under the cube) now counts as a gear word |
| held_04_i4 | put the two red gears in the grey tray | one red gear | both red gears, blockers moved first | same |
| held_04_i8 | take it off the gear | refuse "no it" | ask_user | a target made only of "it/this/that/them" is always asked about, even when a relation ("off the gear") is said |

## Still doubtful
- **held_05_i4 "everything that is on the plate"**: plans the grey metal part, then the yellow gear. The part overlaps the
  plate's rim and lies on the red gear. If the ground truth lists only the yellow gear, the first pixel is a miss. The
  photo does not settle it.
- **held_02**: the mug's handle is kept as a separate `small_white_part` (an object lying on the mug). No instruction of
  this set depends on it; "pick up the small white part" on a scene like held_02 would grasp the handle.
- held_04: the plan puts the yellow gear and the cube on the black tray, which is a legal "somewhere else".

## Regression check of the fixes (same Apertus cache, official scorer, 86 scenes, 632 + 164 instructions)
Committed code vs. working tree: every set the same except two items that got better.
r1_rearr_hold 55.6 → 56.7 (`pick up the object directly below the cube`), r2_rearr 51.4 → 53.5 (`pick up the small
white part`). dev 65.0, r1 text 64.0 / 55.1, r1 novel 52.3 / 49.6, r2 text 56.1, r2 novel 37.0, r3 56.0 / 50.4, round 4
46.5, safety regression set 53.8, probe set 18.2: unchanged. The held-out scenes were used to find these fixes, so they
are no longer an unbiased measure; the 164 + 632 comparison items are.

## Tiling the photo (tried, not kept)
SAM at 1024 px sees the 2448 × 2048 photo at 0.42 scale. Overlapping 1024 px windows at native resolution (12 passes,
100 s per scene, 3.7 GB GPU, no RAM problem) and 800 px windows (30 passes, 240 s) gave 568 / 1080 raw regions on
held_03, but never the whole valve: SAM splits it into plug, body and cover at any resolution. Added to the normal
candidate list they also changed how other objects were named (the cube's neighbour crescent became "part of the
cube"). So tiling is used only where it is needed: one extra window of 640 px around a fragment that is too small for its name
(about 8 s each, cached with the photo's other proposals). No environment variable or flag: it is on by default.
