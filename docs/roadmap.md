# Gelem -- Roadmap

What we are building, why, and when it is usable. For the rules see `CLAUDE.md`;
for the components see `docs/architecture.md`; for media design and the ordered
work items see `docs/media_architecture.md`.

**Status:** the Measurement, Foundations, Media and Blendshape stages are
designed, and their work items are specified in `docs/media_architecture.md` §6.
Viewing, Analysis and Trial detection are described at the level of intent and
constraints only. **Do not treat the later stages as specified.**

**On naming.** Stages are named in words; only work items are numbered
(`P0.1`, `P1.6`, `P2.2`), and those identifiers live in
`docs/media_architecture.md` §6. Do not number the stages -- a previous attempt
made "Phase 0" mean Measurement here and Foundations there.

---

## 1. Why this plan exists

Development is driven by a real dataset rather than by imagining features. The
method: take one concrete study end to end, build what it needs, and generalise
each feature as it is built.

**The guard against the obvious failure mode** -- an application that fits one
study and nothing else -- is to name, before building any feature, the parameter
that makes it general:

- not "extract the trial number", but "read a marker from a region"
- not "split into trials", but "split by start/end columns"
- not "average 400-1600 ms", but "average over a parameterised window"
- trimming a video to a task is a segment, not a separate mode

A hardcoded emotion count, time window, or blendshape name inside a generic
component is a defect. `CLAUDE.md` states this as a hard rule.

---

## 2. The driving study

Participants were recorded performing tasks. The clearest task: produce seven
facial expressions across three blocks of 49 trials (7 expressions x 7 trials per
block).

- Block 1: the emotion name shown as text
- Block 2: text plus a photo of a person showing it (different person each trial)
- Block 3: photo only

Per participant: one full-length video plus a CSV describing the trials. Each
trial begins with its number displayed on screen, for synchronisation.

The research question is about **consistencies that hold within a participant and
not across participants** -- which is why the within- versus across-participant
contrast appears again as a constraint under "Analysis operators" in §5.

**Target datasets generally:** 10-10,000 videos of 2-300 minutes; commonly 20-120
videos of 10-120 minutes. Full numbers in `docs/media_architecture.md` §2.

---

## 3. The pipeline the researcher wants

This is the user story the whole plan serves. Nine steps, and the stage that
delivers each:

| # | Step | Delivered by | State |
|---|---|---|---|
| 1 | Open the full-length participant videos | -- | works |
| 2 | Trim each to the portion covering this task | Media (P1.6) | outside Gelem |
| 3 | Split each into per-trial pieces | Media (P1.6) | outside Gelem |
| 4 | Merge the trial CSV | Media (P1.5) | broken: one table, one key, no one-to-many |
| 5 | Produce a frame-level table | Media (P1.7) | works, but writes files |
| 6 | Extract blendshapes per frame | Blendshapes (P2.x) | works, too slow at scale |
| 7 | Aggregate to one row per trial over a time window | Media (P1.8) + existing | partly works |
| 8 | View seven trials of one expression stacked, per participant | Viewing | not built |
| 9 | Find which blendshape combinations predict condition | Analysis | not built |

Steps 2 and 3 are done outside Gelem today with a throwaway script, deliberately,
so the Media stage can concentrate on general capability. See "Trial detection
inside Gelem" in §5.

Note that browsing -- filter, sort, group, look at tiles -- is not a numbered step
because it happens continuously between all of them. It is the thing that has to
stay responsive at 530,000 rows, which is what most of Foundations exists for.

---

## 4. What "usable" means

The stopping rule, because without one the work items simply continue.

**Gelem is usable when a student can complete steps 1 through 7 unaided, on a
two-participant subset, without writing Python.**

Two-participant subset, not the whole dataset. Processing 240 hours is a compute
problem, not an architecture one, and waiting for it would confuse the two.

Steps 8 and 9 are the research payoff but not the usability bar -- a student who
reaches step 7 has a tidy trial-level table and can take it to R, which is what
the Analysis constraints in §5 assume anyway.

That criterion is testable, and it is what the stages below are ordered to reach.

---

## 5. The stages

The four designed stages are described here only by **what they are for and what
they unblock**. Their contents live in `docs/media_architecture.md` §6 and are not
repeated, because a restated list drifts out of agreement with the original.

### Measurement

**Why:** the media design rests on empirical claims -- how much faster keyframe
extraction is, how slow a scattered seek is, whether MediaPipe parallelises. Those
are assumptions until measured, and if one is off by an order of magnitude the
design changes rather than the estimate.

**Unblocks:** everything, cheaply. A few hours of throwaway scripts, and it can
delete a work item outright.

`docs/media_architecture.md` §6.0. Record measured values in §11 of that document
and mark them measured.

### Foundations -- `P0.1` to `P0.5`

**Why:** the media work cannot be built on the current code. Reading one row
copies the entire table, the picture cache cannot tell two media columns of the
same row apart, and there is no address type for anything to key on.

**Unblocks:** every later stage. Membership is by **dependency** -- other work sits
directly on top of these -- not by severity, which is why some genuinely bad
defects are *not* here.

`docs/media_architecture.md` §6.1.

### Media and data handling -- `P1.x`

**Why:** this is the general capability every future study needs, and the reason
the whole plan exists. Media values become **addresses into source files** rather
than paths to extracted files, so splitting a video into trials or frames writes
nothing to disk.

**Unblocks:** pipeline steps 2 through 5, and 7. After this stage a student can get
from raw participant videos to a trial-level table.

`docs/media_architecture.md` §§3-5 for the design, §6.2 for the ordered items.

**One thing to state here because it is easy to re-propose:** lineage between
derived tables is carried by **ordinary data columns** -- `participant_id`,
`trial_id`, `frame_index` -- as it would be in R. A `parent_row_id` / `parent_table`
pointer convention was proposed and **rejected**. Natural keys serve every need it
was meant to: "show me the clips behind these trial rows" is a join, navigating
upward is a join, and detecting that a re-run duplicated a level is a duplicate-key
check, which a surrogate pointer would not reveal at all. Authority:
`docs/architecture.md` §4.2.

### Blendshape extraction -- `P2.x`

**Why:** step 6 of the pipeline, and the analysis the driving study exists to
perform.

**Unblocks:** steps 8 and 9, which have nothing to display or model until
blendshapes run at scale.

Design settled; three parameters decided by measurement rather than argument.
`docs/media_architecture.md` §6.3, and `operators/CLAUDE.md` for the caching and
resumption rules -- both of which are narrower than they sound and are easy to
overstate.

### Viewing and comparison

**Not yet designed.** Intent and constraints only.

The need: compare the seven trials of one expression within a participant, and the
same expression across participants, fast enough to form impressions by eye. This
is step 8.

**Filmstrip first.** One row per trial, a frame every ~100 ms across the columns.
You see all seven trajectories at once rather than replaying them in sequence,
which is usually more informative for comparing expressions than playback is. It
is a display operator, so it needs no infrastructure beyond the Media stage.

**Playback second.** Native `QMediaPlayer` given a file and a time range --
`docs/media_architecture.md` §4.2. A handful of simultaneous players is fine;
dozens are not. An HTML page with `<video>` elements and one shared scrubber
remains a reasonable alternative for synchronised playback of many short clips.

Also belonging here: **a synchronisation QC view** -- each trial's detected onset
frame plus its neighbours -- needed once trial splitting happens inside Gelem.
Sync errors are silent: an onset off by a few frames appears as noise in the
window average, not as an error. This view should exist before anyone trusts
trial-level results.

### Analysis operators

**Not yet designed, deliberately.** These are constraints to build into whatever
they become, recorded now so they are not rediscovered late and expensively.

**Cross-validation folds must group by trial, never by frame.** Frames within a
trial are heavily autocorrelated. Splitting at frame level leaks the answer across
the fold boundary and reports accuracy that means nothing. The fold-grouping
column must be a mandatory parameter, defaulting to trial.

**A permutation baseline belongs in the operator's output**, not as a separate
step the researcher might skip. With 52 predictors and 7 classes there is ample
room to fool oneself.

**Within-participant and across-participant generalisation are the comparison of
interest**, not an afterthought. The research question is precisely about
consistencies that hold within a person and not between people, so the operator
should make that contrast a first-class output.

### Trial detection inside Gelem

**Deliberately postponed.** Clips are produced outside Gelem so the Media stage
can concentrate on general capability.

When it returns it is a UI feature rather than a research problem: the user drags
a box around where the trial number appears on a sample frame, and OCR reads that
region on every frame. A clean high-contrast digit in a known region is close to
always correct.

Generalised, this is "mark a region on a sample frame, then read from that region
on every frame" -- which also covers timestamps, condition labels and any other
on-screen marker. The `#r=` address form already expresses the region.

---

## 6. Effort

**These estimates predate the Foundations stage and are therefore low.** They were
made before external review found that Dataset access paths, artifact identity and
visible-row ownership all needed fixing first. Treat them as a floor, not a
budget, and re-estimate after Foundations is done and its actual cost is known.

| Item | Hours |
|---|---|
| Generalised merge (any table, any key, one-to-many expand) | 4 |
| Media addresses and resolver | 12 |
| Segment operator | 6 |
| Sequential blendshape pass | 4 |
| Trial aggregation with time windows | 4 |
| Filmstrip display operator | 4 |
| Onset QC view | 3 |
| ROI picker + OCR operator (Trial detection stage) | 8 |

The item most likely to overrun is the media resolver, and it is also the one most
worth keeping -- an undergraduate on a laptop cannot manage a quarter-million
frame files.

**A signal, not a schedule.** An item taking far longer than estimated means the
design is wrong, not that the estimate was. Stop and reconsider rather than push
through. The same applies to a specified test that cannot be written cleanly.

---

## 7. How this gets built

**Claude Code, in the repo, does the implementation.** It runs inside the
`(gelem)` environment, so it can write a file, run `pytest`, read the failure and
fix it without anything being relayed by hand. Desktop sessions cannot run the
test suite at all.

**Claude Desktop is for design, review, and scoping** the next work item. Connect
the repo folder there for reading, so design conversations reference current code
rather than a stale export. Avoid editing from there -- changes made without
running the tests create work for the next session.

**One work item per session, one branch each, review each diff before the next
starts.** That is the practical form of close supervision: not watching a long
autonomous run, but keeping each unit small enough that its diff is readable. The
handoff format, and which items get an external review, are in `CLAUDE.md` under
"Finishing a work item".

**Suggested first implementation item after Measurement:** the address module
(`P0.3`). Pure logic, no file I/O, no Qt -- parse a string, produce a structure,
format it back. The most readable diff in the plan, and reviewing it makes the
media model concrete before anything depends on it.
