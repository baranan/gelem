*SUPERSEDED -- see docs/media_architecture.md*

# Gelem -- Media Handling Architecture Plan

**Status:** design agreed, not yet implemented
**Audience:** Claude sessions working on Gelem over the next few weeks
**Author:** Y B (supervisor / sole developer for this phase)
**Date:** August 2026
**Revised 4 Aug 2026:** added segment thumbnails (§4.1c) after finding that
keyframe-spaced proxy thumbnails can show the wrong trial; added the measurement
pass (§6, item 0) which must run before any implementation.

---

## 0. How to use this document

This is the design Claude and Y B agreed on after working through the scale
requirements. It is the authoritative reference for **media handling** and
supersedes any contrary assumption in the original design doc or in the
student-era code.

Read this before proposing changes to: `column_types/renderers.py`,
`artifacts/artifact_store.py`, `operators/base.py` (`load_image`),
`operators/video_frames.py`, or anything that opens a media file.

Sections 1-5 are decisions with their reasoning. Section 6 is the work.
Section 9 is how to behave in these sessions.

**Rationale is included on purpose.** When a decision looks wrong during
implementation, check whether the reasoning still holds before changing it,
and say so explicitly rather than silently working around it.

---

## 1. What Gelem is becoming

Gelem is a desktop visual data explorer for psychology research. The medium-term
goal is a tool that **undergraduates can use with very little training**, to do
basic processing and analysis of video data before they know Python or machine
learning.

The typical study: videos of participants, split into smaller pieces, merged
with trial-level data files, then blendshapes compared across conditions --
often looking for consistencies within participants that do not hold across
participants.

Architecture remains the seven components (Dataset, QueryEngine, ArtifactStore,
ColumnTypeRegistry, OperatorRegistry, AppController, UI) with the existing
threading rule: DataFrames are mutated only on the main thread; background
workers emit results and AppController applies them.

---

## 2. Target datasets

These numbers drive every decision below. Do not re-derive them; do not design
for numbers outside this range without raising it.

| | Videos | Length each |
|---|---|---|
| Full range | 10 -- 10,000 | 2 -- 300 min |
| **Common case (design target)** | **20 -- 120** | **10 -- 120 min** |

Worst common case is therefore about **240 hours of footage**.

**"Handle" means:**

- View videos, or splits of them, as tiles (thumbnails) -- instantly, at any
  scroll position
- Play any of them, or any split of them, without noticeable lag
- Re-arrange the view (filter, sort, group) responsively
- Sometimes play a handful side by side
- Run analyses (currently blendshapes) over selected material

**"Handle" does not mean:** dense frame-by-frame analysis of the entire corpus.
That is compute-bound, not architecture-bound (see §5.3).

---

## 3. The central decision: media values are addresses

### 3.1 What changes

Today a media cell holds a **location** -- a path to a file that exists:

```
C:/data/p01.mp4
```

It becomes an **address** -- a description of which pixels are wanted, which
may correspond to no file at all:

```
C:/data/p01.mp4#f=1234
```

There is no file containing frame 1234. Something must produce it: open the
video, seek, decode.

### 3.2 The grammar

```
<path>                        the whole file (image or video)
<path>#f=<int>                one frame, by frame index
<path>#t=<sec>                one frame, by time in seconds
<path>#t=<start>-<end>        a time range (a "clip")
<path>#r=<x>,<y>,<w>,<h>      a rectangular region
<path>#t=<a>-<b>&r=<x,y,w,h>  combined
```

A bare path is a valid address, so **existing data keeps working by
construction**.

- **Point forms** (`#f=`, `#t=<sec>`) resolve to pixels.
- **Range forms** (`#t=a-b`) resolve to a representative frame for display,
  and are handed to the player for playback.
- Frame index is preferred for analysis (exact); time is preferred for ranges
  (natural, robust to frame-rate confusion).

### 3.3 The resolver

Exactly one module parses and resolves addresses. Its interface is roughly:

```python
parse(address)   -> MediaAddress          # structured form
format(addr)     -> str                   # back to string
resolve(address, purpose) -> pixels       # purpose: 'display' | 'analysis'
```

**Nothing else in Gelem may open a media file.** Today three places do, and all
three must route through the resolver:

1. `BaseOperator.load_image` -- analysis path
2. `column_types/renderers.py` (`_render_image`, `_video_first_frame_pixmap`)
   -- display path
3. `ArtifactStore._generate_thumbnails` / `_first_frame_as_pil`

### 3.4 Why

Splitting a video into 150 trials or 13,000 frames becomes **row creation only**
-- no files written, milliseconds instead of hours, and cheap to throw away and
redo. It also gives caching, decoder pooling, and resolution limits exactly one
home instead of being scattered.

### 3.5 Consequences to respect

- Splits are **not files**. Exporting split files is an explicit user action
  ("Export frames as files", "Export clips"), never a side effect of analysis.
- `VideoFramesOperator` is demoted to that export action. It is no longer how
  frame tables are made.
- Addresses must survive project save/load. `models/dataset.py` already
  rewrites paths relative to the project root (`_rel_if_inside` /
  `_abs_against`); the address grammar must be handled there, not just the bare
  path portion. **This is easy to miss and will silently break saved projects.**

---

## 4. Display, playback, analysis are three different paths

They have opposite requirements and must not share a mechanism.

| | Resolution | Access pattern | Caching | Tolerates approximation |
|---|---|---|---|---|
| **Display** | low (tile size) | random | aggressive | **yes** |
| **Playback** | full | sequential | none (player's job) | n/a |
| **Analysis** | full | strictly sequential | none | **no** |

The guiding principle: **display tolerates approximation, analysis does not.**
A thumbnail can be the nearest available sampled frame. A blendshape must come
from the exact frame, decoded properly.

### 4.1 Display -- why a proxy layer is required

Video stores complete images only occasionally (every 1-10 s) and only
differences in between. Producing frame N means jumping to the last complete
image before it and replaying forward.

- **Frames near each other** (all frames of one 3 s trial): one forward pass,
  a few ms per frame. Fast.
- **Frames scattered across a long video** (first frame of each of 150 trials
  across an hour): a separate seek each, 50-200 ms apiece. Filling one screen
  of thumbnails takes 10-30 seconds. **Unacceptable, and it is the most common
  browsing action.**

So there are **three thumbnail sources**, all feeding one cache. Which one is
correct depends on what is being shown, and using the wrong one produces wrong
pictures, not just slow ones.

**(a) Proxy layer -- for scrubbing across long videos only.**
Small thumbnails extracted **only at the video's own keyframes**, which
decoders can do while skipping all intermediate frames. This is roughly 100x
faster than full decoding: 240 hours goes from an overnight job to minutes.
Keyframe spacing (1-10 s) is about the right browsing granularity anyway.
Positions are irregular; show the nearest.

- Packed **one file per video plus an index**, never loose files. 864,000
  thumbnails as loose files would be fatal, especially on the Google Drive
  Streaming path. ~6 KB each, ~20 MB per hour of video.
- Built lazily per video, in the background, resumable.

**(b) Segment thumbnails -- for tiles representing clips or trials.**
**The proxy must not be used for this.** Keyframes are 1-10 s apart; a 3 s trial
may have no keyframe inside it at all, so the nearest proxy thumbnail can come
from a neighbouring trial. That is a wrong picture, not a slow one, and it would
be easy to miss because the tile still looks plausible.

Instead, capture each segment's representative frame **when the segment rows are
created**, during the sequential pass that operator is already making. Exact,
effectively free, and stored alongside the segment row.

**(c) Short-span decode -- for frame-level browsing inside a clip.**
The proxy is far too coarse to browse individual frames of a 3 s trial. Decode
the whole span once (~90 frames, a fraction of a second), cache every frame's
thumbnail. Prefetch when a trial is selected. Browsing within that trial is then
instant.

### 4.2 Playback -- keep it out of Python

Hand a native player (`QMediaPlayer`) the file path and a start/stop time. It
uses the container index to seek and hardware to decode. Sub-second regardless
of corpus size.

Playing a "split" is the source file plus two timestamps. Nothing is created.
A handful of simultaneous players is fine; dozens are not.

**Playback must not go through the resolver.** Different mechanism, different
performance characteristics, no shared code path.

### 4.3 Decoding, for the display and analysis paths

- Use **PyAV** rather than OpenCV for seeking. OpenCV's seek into a long file
  is poor.
- Keep a small **LRU pool of open decoders** keyed by file (4-8 handles).
- **Batch requests by source file and sort by frame index** before decoding.
  Thirty frames of one clip is one sequential pass, not thirty seeks. This is
  the single biggest lever for scroll smoothness.

### 4.4 Replace thread-per-request with a bounded pool

`ArtifactStore.request_thumbnail` currently spawns a raw `threading.Thread` per
call. At gallery scale that is thousands of threads. Replace with:

- 2-4 workers, fixed
- a priority queue where visible rows outrank prefetched rows
- **cancellation** -- when the viewport moves, drop stale requests

Without cancellation one fast scroll queues thousands of renders nobody will
see, and the gallery runs seconds behind the scrollbar. This is most of the
difference between "instant" and "broken", and it is independent of everything
else here.

---

## 5. Storage: tables stay in memory

### 5.1 The decision

Tables remain pandas DataFrames in memory. **Do not add DuckDB, Arrow
out-of-core, or any database engine.**

The realistic frame table: 40 participants x 147 trials x 3 s x 30 fps ≈
**530,000 rows**, roughly 300 MB with disciplined dtypes. Two million rows is
~1.2 GB and still workable.

### 5.2 Why not build for huge datasets from the start

There is no runtime cost -- a query engine is fast on small tables too. There
are two real costs, and both land where they hurt most:

**It makes the operator contract much harder to write.** With everything in
memory, mean blendshapes per trial is one line:

```python
def create_table(self, df, group_by=None):
    return df.groupby('trial_id').mean(numeric_only=True).reset_index()
```

Chunked, it becomes a three-method accumulator with partial-result combination
logic. Mean is easy that way; median, percentiles, and correlation matrices are
not. Operators are the component undergraduates write.

**It conflicts with progressive updates** (§5.4), which are worth keeping.

### 5.3 Why memory will not be the binding constraint

Ten million frames of blendshape extraction is roughly **70 hours of
computation**. You cannot accidentally build a frame table too large for RAM,
because the analysis that fills it is far slower than the memory it consumes.
Compute stops you long before memory does.

### 5.4 Progressive updates are a requirement, not a nicety

Operators return results row by row; AppController applies each on the main
thread; tiles repaint immediately. When a normal run takes 20+ minutes this
gives: results visible in seconds, mistakes caught early, the ability to cancel,
and partial results retained on stop or crash. Preserve this.

### 5.5 Keeping the door open

The migration to a database engine later must stay inside `Dataset`. That holds
if, and only if:

- **All table access goes through `Dataset`'s methods.** The UI already obeys
  this; extend it to operators.
- **Table updates have a batch path**, with per-row progressive update as a
  convenience layered on top -- not the only way data enters a table.
- **Frame-level rows are never created by default.** Splitting is scoped and
  deliberate. If eager frame explosion becomes the normal workflow, every
  operator will assume it and the migration becomes viral.
- **Dtypes are set centrally**, not inferred: float32 for blendshapes, int32
  for indices, categorical for repeated strings (source path, condition,
  participant). Inferred `object` columns are the usual memory culprit.

### 5.6 Make the migration trigger empirical

Log the timing of filter, sort, and group operations along with row count. The
decision to migrate should come from those numbers, not from an impression --
and they will also show *which* operations are slow, which is what tells you
what to push into a query engine.

---

## 6. Phase 1 -- work items, in order

Phase 1 is media and dataset handling. Phase 2 (§7) is the blendshape operator.
Items 1-2 are the keystone; item 3 is independent and can be done any time.

0. **Measurement pass -- do this before writing any of the below.**
   This design rests on empirical claims about video and inference speed. They are
   assumptions until measured, and if any is off by an order of magnitude the
   design changes. Throwaway scripts against one real hour-long video, about an
   hour of work total. Record the results in §11.

   - keyframe-only extraction versus full decode (claimed ~100x)
   - seeking to scattered frames (claimed 50-200 ms each)
   - frames inside an already-decoded span (claimed 2-5 ms)
   - MediaPipe per frame on this machine (claimed ~25 ms)
   - whether MediaPipe parallelises across threads (see §7)
   - **keyframe interval in the actual recordings** -- this determines how badly
     (b) above was needed, and whether the proxy is useful at all for these files

1. **Address module.** Grammar, `parse`, `format`, `MediaAddress` type.
   Pure logic, fully unit-testable, no I/O. *Done when:* every form in §3.2
   round-trips, and malformed addresses raise clear errors.

2. **Resolver.** Decoder pool, PyAV backend, `resolve(address, purpose)`.
   Route all three existing call sites (§3.3) through it.
   *Done when:* no `Image.open` / `cv2.VideoCapture` / `av.open` exists outside
   the resolver module, enforced by test.

3. **Bounded worker pool with priority and cancellation**, replacing
   thread-per-request in `ArtifactStore`. *Independent of 1-2; fixes a real bug
   at current scale.*

4. **Proxy layer.** Keyframe-only thumbnail extraction, packed one file per
   video with an index, built lazily in the background, resumable.

5. **Short-span decode cache** for frame-level browsing inside clips, with
   prefetch on selection.

6. **Segment operator.** Video rows -> time-range rows. Must emit a
   **time-within-segment** column (currently nothing produces this, and the
   400-1600 ms window analysis has no column to filter on without it). Must also
   capture each segment's representative thumbnail during the same sequential
   pass (§4.1b) -- without this, segment tiles show frames from the wrong segment.
   Trimming a video to a task is the same operation, not a separate feature.

7. **Frame operator.** Segment or video rows -> frame rows holding `#f=`
   addresses. Writes zero files.

8. **Demote `VideoFramesOperator`** to "Export frames as files".

9. **Central dtype policy**, applied wherever a table is created.

10. **Playback from an address** via `QMediaPlayer`, honouring time ranges.

11. **Generalise merging.** `Dataset.merge_csv` currently hardcodes a join onto
    the `frames` table against `file_name` and rejects one-to-many joins. Trial
    data is inherently one-to-many against a participant video row. Generalise
    to arbitrary table, arbitrary key, with explicit expand semantics.

---

## 7. Phase 2 -- blendshape extraction

Design settled; three parameters to be decided **by measurement**, not by
argument.

**Settled:**

- Runs as a **sequential pass over a clip or video**, not per-row random
  access. Sequential decoding is an order of magnitude faster than seeking,
  and it is what enables tracking mode.
- **Every frame is processed. No sampling by default.** (Y B's decision: never
  sacrifice data for compute.) Keep a sampling parameter for other users,
  defaulting to every frame.
- **Results are cached by frame address.** Re-runs skip completed frames. In
  practice this saves more wall-clock time than any speedup, because iteration
  is where the time goes.
- **Cancellable and resumable**, with partial results retained.
- The parallel unit is the **clip**, processed sequentially inside.

**To measure, before deciding:**

1. **Threads vs processes.** Python's GIL normally prevents parallel threads,
   but it is released while C++ code runs, and MediaPipe inference is C++.
   *Experiment:* time 200 frames on 1 thread, then 4. If wall clock drops,
   use threads -- same process, no serialisation, normal debugging, no Windows
   spawn issues, no per-worker model copies. Only pay for multiprocessing if
   threads do not deliver.
2. **Tracking mode vs per-image.** MediaPipe's video mode reuses the previous
   frame's result instead of re-detecting -- worth ~1.5-2x, requires ordered
   frames with increasing timestamps. But its output may differ slightly from
   fresh detection, and it can behave differently after losing a face. That is
   a **reproducibility** question, not just speed. Make it a flag, default to
   per-image, and test how far the two diverge on real data.
3. **Worker count default.** Each process loads its own MediaPipe model
   (hundreds of MB). Six workers is fine on a development machine and not fine
   on a student's 8 GB laptop. Default low, make it a setting.

**Robustness requirement:** parallelism is a strategy behind one interface, with
the serial path always present and always correct, and automatic fallback to
serial if a worker fails. A user with an unusual machine gets slow but correct
results, never a broken application.

---

## 8. Invariants and guardrail tests

**Every architectural rule here must become a failing test, not a sentence in a
document.** Claude reliably obeys a test that fails and reliably drifts from
prose. Extend the existing pattern in `tests/test_architecture_imports.py`,
`test_controller_async_contracts.py`, `test_operator_registry_boundaries.py`.

Write these:

- Only the resolver module may call `Image.open`, `cv2.VideoCapture`, or
  `av.open`. (import-graph test)
- Creating a frame table writes **zero** files.
- **Equivalence:** pixels obtained via an address equal pixels obtained by
  extracting that frame to JPEG and reading it back (within JPEG tolerance).
  *Build this first* -- it lets the old extraction path stand as a reference
  implementation until the virtual path agrees, then be deleted.
- **A segment's thumbnail comes from inside that segment's own time range.**
  (Guards the §4.1b failure: a proxy thumbnail from a neighbouring trial looks
  plausible and would otherwise go unnoticed.)
- The decoder pool never exceeds N open handles, including under concurrent
  access.
- A cancelled render request never invokes its callback.
- Peak memory during a scroll over a synthetic 1M-row table stays under a fixed
  ceiling. *(This one would have caught the flaw in the first version of this
  design.)*
- Serial and parallel blendshape extraction produce identical output on the
  same input.
- Every table created by any operator conforms to the dtype policy.
- Save project -> load project -> all addresses still resolve, including
  relative-path rewriting.

Test at **component seams, not internals**. Asserting on contracts (Dataset owns
all mutation, operators return dicts, workers never touch Qt) means internals
can be rewritten freely. Asserting on internals makes every refactor a
test-rewriting slog, and refactoring then stops happening.

---

## 9. Working rules for these sessions

Y B's stated concern is not that features are hard to build -- Claude builds
them quickly. It is keeping the result **modular, well-tested, and widely
enough conceived**, with freedom to make any change that helps long-term.

**For Claude:**

- **Propose refactors; do not patch.** Late in a long session there is a pull
  toward patching because refactoring feels disruptive. Resist it, and say
  explicitly when a clean fix requires touching more than the task at hand.
- **Do not treat existing code as a constraint.** Y B has explicitly authorised
  replacing it. If a decision looks wrong because of how something is currently
  built, say so rather than designing around it.
- **Confirm which component owns a feature before writing code.**
- **Flag anything crossing the main-thread / worker boundary.**
- **Write tests from the spec, not from the implementation.** Tests written
  after and alongside code tend to encode the implementation's quirks and pass
  vacuously.
- **Ask rather than assume** when this document or the design doc is ambiguous.
- **Watch for study-specific vocabulary leaking into general components.** Any
  hardcoded 300-1500 ms window, seven-emotion assumption, or blendshape-specific
  branch inside a generic component is a leak. Before building a feature, name
  the parameter that makes it general: ROI OCR is "read a marker from a
  region"; segmentation is "split by any start/end columns"; trimming is a
  segment, not a mode.
- **Do not front-load the whole codebase.** `gelem_codebase_for_claude.txt` is
  ~150k tokens and goes stale immediately. Read the specific files a question
  touches.

**For Y B:** ask "what would you rip out if you could?" at intervals. Claude
will not volunteer it.

---

## 10. Non-goals and deferred decisions

**Explicitly out of scope now:**

- Any database or out-of-core storage engine (§5). Revisit when §5.6 timings say so.
- Dense frame-by-frame analysis of the full 240-hour corpus -- compute-bound, not
  an architecture problem.
- GPU inference. Undermines "installs easily for undergraduates".
- Distributed or cluster execution.
- OCR / ROI trial-number detection. **Deliberately postponed by Y B**; short
  clips will be produced externally for now. When it returns, it is
  "let the user mark a region on a sample frame, then read from that region on
  every frame" -- and the `#r=` address form already expresses it.
- Avatar rendering (blocked on a usable VRM source).

**Deferred, with known cost:**

- **`row_id` is a string.** At 500k rows this costs ~50 MB, which is tolerable.
  At tens of millions it is fatal. Changing it is a wide refactor. Revisit only
  if the row-count trigger fires.

**Known pre-existing bug to fix early:** `ColumnTypeRegistry.infer_type()`
mistags columns whose values end in media extensions. A `video_file` column
holding `.mp4` filenames will be mistagged as `media_path` immediately in this
pipeline.

---

## 10a. Considered and rejected

These were weighed and turned down for stated reasons. Reversing one is allowed,
but do it knowingly -- say which reason no longer holds.

**Frame index and source path as two separate columns**, instead of one address
string. Cheaper in memory (categorical path plus int32 index) and directly
filterable and sortable. **Rejected because it breaks with more than one media
column per row**, which the design already supports (`GridTile` exists to show
`full_path` and `avatar_path` together). Two media columns would need four
physical columns plus a rule about which pair goes with which. Revisit only if
address-string memory is measured to be a real problem, and then as a logical
column backed by several physical ones, registered in `ColumnTypeRegistry`.

**Extending the renderer signature to receive the whole row**, so it could read
those separate columns. Rejected for the same reason, and unnecessary once the
address is self-contained. The existing
`render(value, size, mode, context)` contract is sufficient. Treat the fact that
the correct design required no change to that seam as evidence for it.

**A database engine (DuckDB over Parquet) from the start.** See §5.2. No runtime
cost on small data; rejected because it hardens the operator contract and
conflicts with progressive per-row updates. §5.5 lists what keeps the door open,
§5.6 makes the trigger empirical.

**HTML with `<video>` elements for side-by-side playback.** Still reasonable for
many short clips and gives a shared scrubber cheaply, but native `QMediaPlayer`
is better for scrubbing long files and is what §4.2 specifies. Revisit if
synchronised playback of many clips becomes a priority.

**A filmstrip view** (one row per trial, a frame every ~100 ms) as the primary way
to compare trials. Genuinely useful -- you see whole trajectories at once rather
than replaying them -- but it is a display operator, not infrastructure, so it is
out of phase 1. Not rejected on merit; deferred.

**Materialised clip files as the unit Gelem sees.** Clips are currently produced
externally, which is fine for getting moving. But once addresses express time
ranges, a clip need not be a file, and the external step becomes optional. Do not
build anything that assumes clips are files on disk.

---

## 11. Numbers worth not re-deriving

**All performance figures below are estimates until the §6 measurement pass runs.
Replace them with measured values and mark them as measured.**

| Quantity | Value |
|---|---|
| Common-case corpus | 20-120 videos, 10-120 min → up to ~240 h |
| Frames in 240 h @ 30 fps | ~26 M (never materialise all of these) |
| Realistic frame table (Y B's study) | ~530 k rows, ~300 MB |
| MediaPipe per frame, CPU | ~25 ms |
| Y B's study, every frame, 1 thread | ~4.4 h |
| Same, with tracking + 6 workers | ~30-45 min |
| Thumbnail, 150 px JPEG | ~6 KB |
| Proxy layer per hour of video | ~20 MB |
| Full decode of 1 h video | ~2-6 min |
| Keyframe-only extraction | ~100x faster than full decode |
| Seek to a scattered frame | 50-200 ms |
| Sequential frame in a decoded span | 2-5 ms |
