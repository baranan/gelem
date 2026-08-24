# Gelem -- Media Handling Architecture Plan

**Status:** design agreed, not yet implemented
**Audience:** Claude sessions working on Gelem over the next few weeks
**Author:** Y B (supervisor / sole developer for this phase)
**Date:** August 2026

**Revision history**

- *4 Aug 2026 (a):* added segment thumbnails (§4.1b) after finding that
  keyframe-spaced proxy thumbnails can show the wrong trial; added the measurement
  pass, which must run before any implementation.
- *4 Aug 2026 (b), after external review:* this document previously assumed the
  existing code was a sound foundation. It is not, in three specific ways, and a
  **Phase 0** now precedes the media work (§6.1). Four substantive corrections:
  the segment-thumbnail mechanism in §4.1b does not work for metadata-driven
  segmentation (§4.1b, rewritten); artifact identity must be the media address,
  not the row (§4.5, new); address semantics must be settled before the parser is
  written (§3.6, new); the JPEG equivalence test was an invalid reference (§7).
  The measurement pass is unchanged and still runs first.

---

## 0. How to use this document

This is the authoritative reference for **media handling** and supersedes any
contrary assumption in the original design doc or in the student-era code.

Read this before proposing changes to: `column_types/renderers.py`,
`artifacts/artifact_store.py`, `operators/base.py` (`load_image`),
`operators/video_frames.py`, or anything that opens a media file.

Sections 1-5 are decisions with their reasoning. Section 6 is the work. Section 9
is how to behave in these sessions.

**Rationale is included on purpose.** When a decision looks wrong during
implementation, check whether the reasoning still holds before changing it, and
say so explicitly rather than silently working around it.

**Statuses** follow `CLAUDE.md`. Nothing in this document is `[NOW]` -- none of it
is implemented. Work items are numbered `P0.n`, `P1.n`, `P2.n` and referenced from
the other documents.

---

## 1. What Gelem is becoming

Gelem is a desktop visual data explorer for psychology research. The medium-term
goal is a tool that **undergraduates can use with very little training**, to do
basic processing and analysis of video data before they know Python or machine
learning.

The typical study: videos of participants, split into smaller pieces, merged with
trial-level data files, then blendshapes compared across conditions -- often
looking for consistencies within participants that do not hold across
participants.

Architecture remains the seven components (see `docs/architecture.md`).

---

## 2. Target datasets

These numbers drive every decision below. Do not re-derive them; do not design for
numbers outside this range without raising it.

| | Videos | Length each |
|---|---|---|
| Full range | 10 -- 10,000 | 2 -- 300 min |
| **Common case (design target)** | **20 -- 120** | **10 -- 120 min** |

Worst common case is therefore about **240 hours of footage**.

**"Handle" means:**

- View videos, or splits of them, as tiles -- instantly, at any scroll position
- Play any of them, or any split, without noticeable lag
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

It becomes an **address** -- a description of which pixels are wanted, which may
correspond to no file at all:

```
C:/data/p01.mp4#f=1234
```

There is no file containing frame 1234. Something must produce it: open the video,
seek, decode.

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
- **Range forms** (`#t=a-b`) resolve to a representative frame for display, and
  are handed to the player for playback.
- Frame index is preferred for analysis (exact); time is preferred for ranges
  (natural, robust to frame-rate confusion).

### 3.3 The resolver

Exactly one module decodes media in Python.

**Do not make it image-only.** Audio is coming -- synchronised expression and
speech analysis is a natural next study -- and an interface that returns "pixels"
will have to be replaced rather than extended. Use explicit methods:

```python
parse(address)   -> MediaAddress          # structured form, pure logic
format(addr)     -> str                   # back to string

resolve_frame(address, purpose)      -> FramePayload   # 'display' | 'analysis'
decode_video_span(address, purpose)  -> iterator of FramePayload
decode_audio_span(address, purpose)  -> AudioPayload
```

Input requirements on operators become declarative in the same spirit: `metadata
only`, `image frame`, `video span`, `audio span`, `arbitrary media address` --
rather than the current boolean `requires_image`.

**Nothing else in Gelem decodes media.** Today three places do, and all three must
route through the resolver:

1. `BaseOperator.load_image` -- analysis path
2. `column_types/renderers.py` (`_render_image`, `_video_first_frame_pixmap`) --
   display path
3. `ArtifactStore._generate_thumbnails` / `_first_frame_as_pil`

### 3.4 Playback is the one explicit exception

`QMediaPlayer` receives a file path and a start/stop time and decodes natively.
It does not go through the resolver.

**Precise rule:** all Python-based decoding, for display and for analysis, goes
through `MediaResolver`. Native playback is handled by `PlaybackAdapter`, which
uses the **shared address parser** to obtain the source path and time range, and
then delegates decoding to Qt.

The parser is shared. The decoding path is not. An earlier version of this
document said both "nothing else may open a media file" and "playback must not go
through the resolver", which read as a contradiction.

### 3.5 Why addresses

Splitting a video into 150 trials or 13,000 frames becomes **row creation only**
-- no files written, milliseconds instead of hours, and cheap to throw away and
redo. It also gives caching, decoder pooling, and resolution limits exactly one
home instead of being scattered.

**Consequences to respect:**

- Splits are **not files**. Exporting split files is an explicit user action
  ("Export frames as files", "Export clips"), never a side effect of analysis.
- `VideoFramesOperator` is demoted to that export action. It is no longer how
  frame tables are made.
- Addresses must survive project save/load. `models/dataset.py` already rewrites
  paths relative to the project root (`_rel_if_inside` / `_abs_against`); the
  address grammar must be handled there, not just the bare path portion. **This is
  easy to miss and will silently break saved projects.**
- Only paths **inside** the project folder are made relative. External source
  paths stay absolute unless the user explicitly imports the file.

### 3.6 Semantics to settle before writing the parser

**New, and blocking for P0.3.** The grammar in §3.2 says what an address looks
like, not what it means. Each of the following silently returns the wrong frame if
left to the implementer's judgement, and "wrong frame" in this application means
wrong data in a published analysis.

Decide and write down, with a test for each:

1. **Escaping.** Source paths may contain `#`, `&`, `,` or `-`. Define the escape
   rule and round-trip it.
2. **Time-point rounding.** Does `#t=3.5` select the nearest frame, the first frame
   at or after 3.5 s, or the frame whose presentation interval contains 3.5 s?
3. **Range endpoints.** Is `#t=a-b` half-open `[a, b)` or inclusive at both ends?
4. **Representative frame for a range.** Which frame does a range resolve to for
   display -- first, midpoint, or a named policy? Different policies produce
   different thumbnails for the same segment, so this is part of the artifact
   cache key.
5. **Crop coordinates.** Pixels or normalised 0-1? Normalised survives a proxy
   resolution change; pixels do not.
6. **Orientation.** How is container rotation metadata applied, and is `#r=`
   expressed before or after rotation?
7. **Multi-stream files.** How is the video or audio stream selected when a file
   has several?
8. **Frame identity.** Is a frame identified by ordinal, by presentation timestamp
   (PTS), or both? **Variable-frame-rate files make ordinal and time disagree**,
   and phone recordings are frequently VFR.
9. **Canonical form.** Fixed ordering of components and fixed numeric formatting,
   so that two addresses meaning the same thing hash the same. Required for the
   cache to work at all.
10. **Time origin.** Is `#t=0` the container's time zero, the stream's start time,
    or the first presented frame? These differ routinely, and a stream start-time
    offset produces a constant error across an entire video -- plausible, uniform,
    and very hard to notice.
11. **Degenerate intervals.** Negative, reversed (`#t=5-2`), empty (`#t=3-3`), and
    out-of-range ranges. Each needs a defined behaviour: clamp, empty result, or
    error. Silently clamping a reversed range is the dangerous option.
12. **Frame ordinal after edit lists and stream selection.** Is the ordinal counted
    in the selected stream, before or after any container edit list is applied?
    A file with an edit list makes "frame 1450" ambiguous.
13. **Where approximation is permitted.** A point address must always resolve
    exactly. Approximation is confined to whole-video proxy browsing (§4.1a) and is
    never used for a segment tile (§4.1b) or for analysis. State this as a property
    of the address, so no future caller can quietly relax it.

**Internally, prefer integer microseconds or rational PTS to floating-point
seconds.** Float seconds accumulate error across a long video and make two
addresses that should be identical compare unequal.

Items 10 to 13 were added in the second review round. All four produce **plausible
but incorrect frame selection** rather than an error, which is the failure mode
this section exists to prevent.

---

## 4. Display, playback, analysis are three different paths

They have opposite requirements and must not share a mechanism.

| | Resolution | Access pattern | Caching | Tolerates approximation |
|---|---|---|---|---|
| **Display** | low (tile size) | random | aggressive | **yes** |
| **Playback** | full | sequential | none (player's job) | n/a |
| **Analysis** | full | strictly sequential | none | **no** |

The guiding principle: **display tolerates approximation, analysis does not.** A
thumbnail can be the nearest available sampled frame. A blendshape must come from
the exact frame, decoded properly.

### 4.1 Display -- why a proxy layer is required

Video stores complete images only occasionally (every 1-10 s) and only differences
in between. Producing frame N means jumping to the last complete image before it
and replaying forward.

- **Frames near each other** (all frames of one 3 s trial): one forward pass, a few
  ms per frame. Fast.
- **Frames scattered across a long video** (first frame of each of 150 trials
  across an hour): a separate seek each, 50-200 ms apiece. Filling one screen of
  thumbnails takes 10-30 seconds. **Unacceptable, and it is the most common
  browsing action.**

So there are **three thumbnail sources**, all feeding one cache. Which one is
correct depends on what is being shown, and using the wrong one produces wrong
pictures, not just slow ones.

**(a) Proxy layer -- for scrubbing across long videos only.**
Small thumbnails extracted **only at the video's own keyframes**, which decoders
can do while skipping all intermediate frames. Roughly 100x faster than full
decoding. Positions are irregular; show the nearest.

- Packed **one file per video plus an index**, never loose files. 864,000
  thumbnails as loose files would be fatal, especially on the Google Drive
  Streaming path. ~6 KB each, ~20 MB per hour of video.
- Built lazily per video, in the background, resumable.

**Keyframe interval is a property of the file, not an assumption.** The resolver
measures it per video on first access and decides **per video** whether a proxy is
worth building. Dense keyframes get a proxy; sparse ones skip it and fall back to
(c). A single global assumption about keyframe spacing would be wrong for some
user's files by construction. See §11 on which measurements generalise.

**(b) Segment thumbnails -- for tiles representing clips or trials.**
**Rewritten 4 Aug 2026 (b).**

**The proxy must not be used for this.** Keyframes may be seconds apart; a 3 s
trial may contain no keyframe at all, so the nearest proxy thumbnail can come from
a neighbouring trial. That is a wrong picture, not a slow one, and it would be
easy to miss because the tile still looks plausible.

The previous version said the segment operator should capture each representative
frame "during the sequential pass it is already making". **That assumed
segmentation decodes video.** A metadata-driven segmentation -- start and end
columns from a trial CSV, which is the common workflow -- decodes nothing. There
is no pass to piggyback on.

So exact segment thumbnails are an **ArtifactStore batch job**: collect the
outstanding segments, sort by source file and start time, and make one sequential
pass per video. This preserves the reason the original design was cheap (one pass,
not one seek per segment) without requiring an operator to have made it.

An operator that *is* decoding anyway may offer a decoded representative frame as
a hint. It never writes into ArtifactStore itself.

The guardrail test is unchanged: **a segment's thumbnail comes from inside that
segment's own time range.**

**(c) Short-span decode -- for frame-level browsing inside a clip.**
The proxy is far too coarse to browse individual frames of a 3 s trial. Decode the
whole span once (~90 frames, a fraction of a second), cache every frame's
thumbnail. Prefetch when a trial is selected. Browsing within that trial is then
instant.

### 4.2 Playback -- keep it out of Python

Hand `QMediaPlayer` the file path and a start/stop time. It uses the container
index to seek and hardware to decode. Sub-second regardless of corpus size.

Playing a "split" is the source file plus two timestamps. Nothing is created. A
handful of simultaneous players is fine; dozens are not.

### 4.3 Decoding, for the display and analysis paths

- Use **PyAV** rather than OpenCV for seeking. OpenCV's seek into a long file is
  poor.
- Keep a small **LRU pool of open decoders** keyed by file. *Provisional default
  4-8 handles* -- a setting, not an architectural constant. Open file handles and
  decoder memory are machine-dependent.
- **Batch requests by source file and sort by frame index** before decoding. Thirty
  frames of one clip is one sequential pass, not thirty seeks. This is the single
  biggest lever for scroll smoothness.

### 4.4 Replace thread-per-request with a bounded pool

`ArtifactStore.request_thumbnail` currently spawns a raw `threading.Thread` per
call. At gallery scale that is thousands of threads. Replace with:

- **a bounded pool** whose size is a setting. *Provisional default 2-4 workers.*
- a priority queue where visible rows outrank prefetched rows
- **cancellation** -- when the viewport moves, drop stale requests

**The numbers here are provisional defaults, not architecture.** Worker count,
handle count and cache ceiling are all machine-dependent, so by the rule in
`CLAUDE.md` they are settings with low defaults. What is architectural is that the
pool is *bounded*, prioritised and cancellable -- not any particular bound.
*(Second review round caught these as constants contradicting that rule.)*

Without cancellation one fast scroll queues thousands of renders nobody will see,
and the gallery runs seconds behind the scrollbar. This is most of the difference
between "instant" and "broken", and it is independent of everything else here.

### 4.5 Artifact identity is the address, not the row

**New, and this is a live bug, not only a scaling concern.**

`ArtifactStore` currently keys both its index and its memory cache on
`(row_id, artifact_type)`, and writes files named `{row_id}_{artifact_type}.jpg`.
That is wrong on at least five counts:

- **A row may hold several media columns.** `full_path` and `avatar_path` on one
  row collide in the index *and* on disk. `ImageTile` correctly passes
  `column_name` in the render context and `_render_image` ignores it. **Today a
  `GridTile` showing both columns displays the source photograph in both tiles.**
- The same `row_id` exists in more than one table, because
  `create_table_from_rows()` keeps ids when copying rows.
- Loading another project reuses the same ids, and `load_index()` replaces the disk
  index without clearing the in-memory image cache.
- A media value can change while the row keeps its id.
- Different representative-frame policies produce different pictures for the same
  segment (§3.6 item 4).

**The correct key is what the picture *is*:**

```
canonical media address
+ source fingerprint          (size and mtime; a stronger hash where needed)
+ artifact purpose / variant  (thumbnail, preview, proxy frame)
+ requested resolution or representative-frame policy
+ renderer / decoder cache version
```

The table, row and column identify **the UI subscriber waiting for the result** --
who to notify when the picture arrives -- never the picture itself.

This also answers media sharing for free: two tables pointing at the same video
share one cached thumbnail, with no bookkeeping.

### 4.6 The display contract

Demand-driven, and a bounded worker pool alone does not achieve it. Define it
explicitly:

1. A viewport change requests the visible rows plus a small prefetch margin.
2. A cache miss returns a placeholder **immediately**.
3. **No media is opened or decoded during a paint.** Today `_render_image` falls
   back to `Image.open` on the main thread, and `_render_video` runs
   `cv2.VideoCapture` on the main thread on *every* paint of *every* video tile,
   consulting no cache at all. That is the dominant current display cost.
4. ArtifactStore queues the request with viewport priority.
5. Stale off-screen requests are cancelled.
6. Workers return raw image data or a persisted cache artifact -- never a
   `QPixmap`.
7. `QPixmap` construction happens only on the main thread.
8. The ready notification carries enough context to repaint the right table, row
   and column.

The controller also currently requests a thumbnail for **every** row immediately
after loading, which must stop.

---

## 5. Storage: tables stay in memory

### 5.1 The decision

Tables remain pandas DataFrames in memory. **Do not add DuckDB, Arrow out-of-core,
or any database engine.**

The realistic frame table: 40 participants x 147 trials x 3 s x 30 fps ≈
**530,000 rows**, roughly 300 MB with disciplined dtypes. Two million rows is
~1.2 GB and still workable.

### 5.2 Why not build for huge datasets from the start

There is no runtime cost -- a query engine is fast on small tables too. There are
two real costs, and both land where they hurt most:

**It makes the operator contract much harder to write.** With everything in memory,
mean blendshapes per trial is one line:

```python
def create_table(self, df, group_by=None):
    return df.groupby('trial_id').mean(numeric_only=True).reset_index()
```

Chunked, it becomes a three-method accumulator with partial-result combination
logic. Mean is easy that way; median, percentiles, and correlation matrices are
not. Operators are the component undergraduates write.

**It conflicts with progressive updates** (§5.4), which are worth keeping.

### 5.3 Why memory will not be the binding constraint

Ten million frames of blendshape extraction is roughly **70 hours of computation**.
You cannot accidentally build a frame table too large for RAM, because the analysis
that fills it is far slower than the memory it consumes.

**This argument is about memory only.** It says nothing about access-path
complexity, and the current access paths are quadratic -- see §6.1. Those are a
defect in the implementation, not a reason to revisit the storage decision.

### 5.4 Progressive updates are a requirement, not a nicety

Operators return results row by row; AppController applies each on the main thread;
tiles repaint immediately. When a normal run takes 20+ minutes this gives: results
visible in seconds, mistakes caught early, and the ability to cancel with partial
results retained.

**Partial results survive cancellation, not a process crash.** They are in memory.
Crash recovery would need periodic checkpointing or a result journal, which is not
planned. Do not claim otherwise.

### 5.5 Keeping the door open

Migration to a database engine later stays inside `Dataset` **and the snapshot and
query interfaces** -- QueryEngine and operators consume pandas DataFrames today, so
those seams would need adapting too. That is a testable claim; "entirely inside
Dataset" was not.

It holds if, and only if:

- **All stored-table access goes through `Dataset`'s methods**, including the
  snapshot path operators use.
- **Table updates have a batch path**, with per-row progressive update as a
  convenience layered on top -- not the only way data enters a table.
- **Frame-level rows are never created by default.** Splitting is scoped and
  deliberate. If eager frame explosion becomes the normal workflow, every operator
  will assume it and the migration becomes viral.
- **Dtypes are set centrally and validated by Dataset**, not left to operators.

### 5.6 Dtype policy

Set explicitly at table creation and **validated by `Dataset` when it accepts a
table** (P1.8). Operators declare intended semantic types; Dataset does the final
conversion. Making each operator responsible for conformance guarantees drift.

Defaults: float32 for measurements, int32 for indices, categorical for repeated
strings (source path, condition, participant). Inferred `object` columns are the
usual memory culprit.

**Explicit exceptions.** Presentation timestamps, audio sample positions, absolute
frame ordinals and counters require int64 or float64. A blanket float32/int32
policy would quietly truncate time, which is exactly the class of error that
produces plausible wrong numbers.

### 5.7 Make the migration trigger empirical

Log the timing of filter, sort, and group operations along with row count. The
decision to migrate should come from those numbers, not from an impression -- and
they will also show *which* operations are slow, which is what tells you what to
push into a query engine.

---

## 6. The work

### 6.0 Measurement pass -- runs first

**Before Phase 0, not after it.** A few hours of throwaway scripts. Several claims
below are order-of-magnitude assumptions, and one result can delete a work item --
so measuring first is time that may save a week. Record results in §11.

**Use a handful of fixtures, not one file.** H.264 and H.265, constant and variable
frame rate, short and long GOP, and at least one phone recording. One video guides
the immediate implementation; it does not settle a ratio for every user's data.
See §11.

See §11 for which of these generalise beyond Y B's machine and which do not. In
brief: the ratios test structural properties of codecs and libraries and do
generalise; absolute per-frame times do not and must not drive design.

**Video and decoder -- gates Phase 1:**

- keyframe-only extraction versus full decode (claimed ~100x)
- seeking to scattered frames (claimed 50-200 ms each)
- frames inside an already-decoded span (claimed 2-5 ms)
- **keyframe interval in the actual recordings.** Not a value to hardcode -- see
  §4.1a, the resolver measures this per video at runtime. What Y B's files tell us
  is how common each case is, which decides whether the proxy layer (P1.3) is a
  Phase 1 item or a later optimisation.

**MediaPipe -- gates Phase 2:**

- does inference parallelise across threads? Time 200 frames on 1 thread, then 4.
  This is a property of whether the library releases the GIL, so it generalises.
- tracking mode versus per-image: how much faster, and **how far do the numbers
  diverge**. The second half is a reproducibility question, not a speed one.
- per-frame time on this machine. **Operational only.** It tells Y B whether his
  study is a 4-hour or a 40-hour run, which is worth knowing. No architectural
  decision depends on it, and it must not become a constant.

**If a measurement comes back ambiguous** -- 8x where the plausible answers were
100x or 3x -- it has not settled the question. Say so rather than rounding toward
the expected answer.

### 6.1 Phase 0 -- foundations the media work sits on

**Work item IDs are stable names, not an order.** Phase membership is by
**dependency** -- other work is built directly on top of it -- not by severity.
An item can move between phases; its ID does not change. Two items moved into
Phase 0 in the second review round for exactly this reason, and kept their names.

Five items:

| ID | Item | Why it is a prerequisite |
|---|---|---|
| P0.1 | Rule labelling, archive, green baseline | Nothing else can be trusted to obey rules whose status is unstated |
| P0.2 | Dataset access paths and result delivery | Every operator run and every progressive update goes through these |
| P0.3 | Address semantics and `MediaAddress` | P0.5 keys on a canonical address, which does not exist until this does |
| P0.4 | Controller ownership of visible row order | P0.5 must know what is on screen to prioritise and cancel |
| P0.5 | ArtifactStore identity and demand-driven display | The proxy, segment thumbnails and every tile attach here |

*Round-two corrections.* P0.3 was P1.1 and P0.4 was part of P1.13. Both were moved
after the reviewer pointed out that the original P0.3 (ArtifactStore) depended on
them: an address-keyed cache cannot be built before canonical addresses exist, and
a demand-driven renderer cannot prioritise a viewport whose contents live in a
widget's private list. This makes Phase 0 five items rather than three. That is
closer to the seven originally proposed, and the difference is now a stated
criterion rather than a judgement call -- table schemas, project paths, operator
descriptors and the remaining UI cleanup are still Phase 1, because nothing in
Phase 0 or in the media foundation is built on them.

**P0.1 Label every rule, archive student-era text.**
Give every rule in `CLAUDE.md` a `[NOW]` / `[TARGET]` / `[MIGRATING]` status, with
violation sites listed. Move the student guide and old workflow to `docs/archive/`
marked as historical. Get `python -m pytest` green -- `tests/test_renderer.py` has
two collection errors from manual checks named `test_*`. Cheap, and it is what
stops the next session either refactoring chaotically or treating every rule as
optional.

**P0.2 Dataset access paths and result delivery.**

1. Maintain a row-id index per table.
2. `get_row()` reads without copying the whole table.
3. Add `snapshot_rows(row_ids, columns=None)` -- one controlled copy for a run.
4. Stop materialising every row as a dict before starting a worker.
   `run_create_columns()` currently calls `get_row()` per row, so launching an
   operator over 530k rows performs 530k full-table copies.
5. Add `apply_row_updates(table_name, updates)` as the primary batch path; keep
   single-row update as a convenience over it.
6. Replace the list queues with `queue.SimpleQueue` or a locked `deque`.
7. Drain for a fixed time or item budget per tick; coalesce progress and repaint
   events.
8. Propagate `operation_id` through progress, completion, cancellation and
   stale-result rejection. It is generated and then discarded today.
9. Add `table_name` to `row_updated` and to artifact notifications.

**P0.3 Address semantics and the `MediaAddress` module.** *(was P1.1)* Settle §3.6
first, then grammar, `parse`, `format`, `MediaAddress`. Pure logic, no I/O, no
dependencies, fully unit-testable -- which is why it is cheap to do early and
expensive to do late. *Done when:* every form round-trips, canonical form is
stable, each §3.6 decision has a test, and malformed addresses raise clear errors.

**P0.4 Split ownership of "what is on screen".** *(was part of P1.13)* Two
different things are currently tangled in `GalleryWidget._row_ids`, and the split
matters -- putting all of it in the controller would drag layout logic there,
which is the opposite mistake:

- **The controller owns the ordered query result** -- which rows match the current
  filters, in what order. This is data, and it is what "Visible" scope for an
  operator, saving a filtered set, and prefetch priority all need.
- **The gallery owns viewport geometry** -- tile size, column count, scroll
  position -- and **reports the currently displayed index range** into that order.
  It does not own the order itself and is never asked for its private list.

So a render request is "rows `[i, j)` of the current result, plus a prefetch
margin", which the controller can resolve without knowing anything about tiles.
Includes the `None` versus `[]` fix in `_relayout()`. Everything else in P1.13
stays in Phase 1.

**P0.5 ArtifactStore identity and demand-driven display.** *(was P0.3)*
Implement §4.5 and §4.6. Address-based keys, `ArtifactCodec` separated from source
decoding (§7), bounded worker pool with priority and cancellation, no decoding in a
paint path, clear the memory cache on project load, cache size becomes a setting.
This fixes the avatar-tile bug and is the foundation the proxy and segment
thumbnails attach to.

### 6.2 Phase 1 -- media foundation

Listed in dependency order. IDs are names and do not follow it.

**P1.8 Table schemas, column roles, and central dtype validation.** Dataset owns a
`TableSchema` per table; `ColumnTypeRegistry` maps type tag to renderer only.
Dataset validates and normalises on accept. **First, because every table created
below is created against a schema** -- doing it after the segment and frame
operators means writing them twice.

Includes **column roles** (`identifier` / `index` / `measurement`) and the separate
**`carry_to_children`** flag, which together are how a splitting operator knows
what to copy onto derived rows. Without them, "carry the identifying columns" is
not implementable and every operator would guess differently. Identifiers and
indices are always carried; everything else defaults to carried, because dropping a
trial-level covariate such as reaction time from frame rows would destroy the
analysis those rows exist for. See `docs/architecture.md` §4.2. **Must be settled
before P1.6.**

**P1.9 ProjectPaths.** Temporary workspace for an unsaved project, project root,
artifacts and proxies, operator outputs, Save As migration, relative artifact
indexing, cleanup and cache-version invalidation. A value object injected into
ArtifactStore and every run -- not an eighth component. **Before the resolver**,
which needs somewhere to put proxies.

**P1.2 Resolver.** Decoder pool, PyAV backend, typed payloads (§3.3). Route all
three existing call sites through it. *Done when:* no source-media decode call
exists outside the resolver module, enforced by the test in §7 -- which permits
`ArtifactCodec` to read and write Gelem's own cached images.

**P1.12 Operator descriptor, run spec, and run context.** `OperatorDescriptor` and
immutable `OperatorRunSpec`; generated parameter dialogs; no Qt in operator
modules; parameters never stored on the operator instance.

**`OperatorRunContext` moves here from Phase 2**, at the reviewer's argument, with
a minimal payload and nothing else: **cancellation token, media resolver, result
sink, project paths**. `run.cache` and `run.log` are **not** part of it -- they
arrive in P2.2 and are optional, so an operator written now must work without them. The earlier plan
deferred the whole thing on the grounds that there was nothing to inject; once the
resolver and ProjectPaths exist there is, and the segment and frame operators below
need both -- plus cancellation, since they are the first operators that can run
long enough to want stopping.

**P1.3 Proxy layer.** Keyframe-only extraction, packed one file per video with an
index, lazy and resumable, **built only for videos whose measured keyframe interval
makes it worthwhile**. Scope depends on §6.0.

**P1.4 Short-span decode cache** for frame-level browsing inside clips, with
prefetch on selection.

**P1.7a Segment thumbnail batch job** in ArtifactStore, ordered by source and time
(§4.1b).

**P1.5 Generalise merging.** `Dataset.merge_csv` hardcodes a join onto `frames`
against `file_name` and rejects one-to-many joins. Trial data is inherently
one-to-many against a participant video row. Generalise to arbitrary table,
arbitrary key, with explicit expand semantics. **Before the segment operator**,
because trial-level metadata is what drives metadata-driven segmentation (§4.1b).

**P1.6 Segment operator.** Video rows -> time-range rows. **Carries every
`identifier` and `index` column unconditionally, plus every other source column
whose `carry_to_children` flag is true** (P1.8), and adds segment index, start, end
and duration. The second half is not optional wording: a trial-level covariate such
as reaction time is a `measurement`, and dropping it here would gut the frame table
downstream. Trimming a video to a task is the same
operation, not a separate feature. It does **not** write thumbnails -- see P1.7a.
Segment boundaries double as the reset boundaries for tracking-mode resumption
(P2.2).

**P1.7 Frame operator.** Segment or video rows -> frame rows holding `#f=`
addresses, with source ordinal or PTS, absolute source time, `time_within_segment`,
and the segment index. Writes zero files.

**P1.10 Playback adapter.** `QMediaPlayer` from an address, honouring time ranges,
sharing the parser and not the decoder.

**P1.14 Demote `VideoFramesOperator`** to "Export frames as files".

**P1.11 Operator registration and output contract.** Make
`operators_config.yaml` drive registration or delete it. Register the missing type
tags (`avatar_path`, `plot_image`) or map them to `media_address`. Settle
`html_path` versus `plot_html`. Delete the dead `operators/thumbnail.py` and
promote a real operator as the reference. Fix `self.output_dir` versus
`self._output_dir` in the docs.

**P1.13 Remaining UI boundary cleanup.** Replace the private-attribute access
listed in `CLAUDE.md` with the public methods that already exist -- `_op_registry`,
`_registry`, `_dataset`, `_active_table`, and `operator._group_by`. The visible-row
half of this moved to P0.4; the `_group_by` half is only meaningful after P1.12
gives parameters somewhere else to live.

### 6.3 Phase 2 -- blendshape extraction

Design settled; three parameters decided **by measurement**, not by argument.

**Settled:**

- Runs as a **sequential pass over a clip or video**, not per-row random access,
  through `iter_column_updates` (P2.1).
- **Every frame is processed. No sampling by default.** (Y B's decision: never
  sacrifice data for compute.) Keep a sampling parameter for other users,
  defaulting to every frame.
- **Cancellable**, with partial results retained. **Resumability is narrower and
  differs by mode** -- see P2.2. The two were previously stated as one property,
  which was too broad.
- The parallel unit is the **clip**, processed sequentially inside.
- Parallelism is a strategy behind one interface, with the serial path always
  present and always correct, and automatic fallback to serial if a worker fails.
  A user with an unusual machine gets slow but correct results, never a broken
  application.

**P2.1 Ordered execution interface.** `iter_column_updates` over a clip, with the
serial path as the reference implementation. `OperatorRunContext` arrives earlier,
in P1.12, with the minimal payload; `run.cache` and `run.log` are added by **P2.2**,
not here.

**P2.2 Result cache identity and resumption.**

**"Cache by frame address" is not sufficient.** For independent-frame mode the key
needs: media address, source fingerprint, operator and model versions, relevant
library version, parameter values, sampling policy, and colour/orientation
preprocessing. Without the model version, a MediaPipe upgrade silently serves stale
results.

**Tracking mode is cached at a different granularity.** When the model reuses the
previous frame's state, the result is not a function of the frame address -- it
depends on preceding frames and on where tracking state was last reset. A longer
per-frame key would still let a resumed run mix two tracking histories.

So tracking results are cached per **clip-run**, and a clip-run must be a
**deterministic sequence identity**, never an ephemeral run UUID -- a UUID would
make the cache unusable by construction, since no second run could ever hit it. A
clip-run identity is:

```
source address and range
+ reset-boundary policy and the boundary this sequence started from
+ the exact ordered frame set (sampling policy plus range, canonically expressed)
+ operator and model versions
+ parameters and preprocessing
```

**Resumption semantics, stated per mode.** A generator gives progressive output and
a place to check for cancellation. It does not give resumability, and the earlier
documents implied it did.

- **Independent-frame mode** resumes by skipping frames already in the cache. Cheap
  and exact.
- **Tracking mode** resumes only from a **reset boundary**. Restarting mid-clip
  from a cold model is not a resumption -- it produces different numbers.
- **Mid-clip continuation** would require serialising model state, which MediaPipe
  may not expose. Until it is shown to, do not plan on it.
- The practical default: on resume, replay from the last reset boundary and
  suppress outputs already stored. This costs recomputation but is reproducible,
  which is the property that matters here.

*Consequence for P1.6:* the segment operator's boundaries are the natural reset
boundaries, which is a further reason segments are a first-class row type rather
than a display convenience.

**P2.3 Threads versus processes**, decided by §6.0. If wall clock drops across
threads, use threads -- same process, no serialisation, normal debugging, no
Windows spawn issues, no per-worker model copies. Only pay for multiprocessing if
threads do not deliver.

**P2.4 Tracking mode as a flag**, defaulting to per-image, with a test of how far
the two diverge on real data.

**P2.5 Worker count** defaults low and is a setting. Each process loads its own
model (hundreds of MB). Six workers is fine on a development machine and not fine
on a student's 8 GB laptop.

---

## 7. Invariants and guardrail tests

**Every architectural rule here must become a failing test, not a sentence in a
document.** Extend the existing pattern in `tests/test_architecture_imports.py`,
`test_controller_async_contracts.py`, `test_operator_registry_boundaries.py`.

**Correctness of media:**

- **Source decoding versus artifact codec.** Only the resolver module may decode a
  **user's media file**. `ArtifactStore` must still read back the JPEGs it wrote --
  `get_pixmap()` calls `Image.open` on its own cache today, and should keep being
  able to. A blanket ban on `Image.open` outside the resolver would either forbid
  legitimate cache I/O or force the resolver to own ArtifactStore's internals.
  The test therefore names two permitted owners: **source media decode -> resolver
  only; derived artifact encode/decode -> `ArtifactCodec` only; everything else ->
  no image or video I/O at all.** (import-graph plus AST test.) *Caught in the
  second review round; the previous formulation was unimplementable.*
- **Equivalence against a lossless reference.** *Replaces the JPEG comparison in the
  previous version, which was invalid:* JPEG is lossy, so it cannot be the
  reference for an exactness claim and would conceal colour, frame-selection and
  conversion errors. Instead: a small **synthetic lossless video fixture with known
  frames**; compare random-access pixels against a trusted sequential decode; and
  separate tests for PTS and frame selection, colour channel order, orientation,
  and crops. The old extraction path may stay temporarily as a *display*
  compatibility check, not as the analysis reference.
- Every §3.6 semantic decision has a test.
- **A segment's thumbnail comes from inside that segment's own time range.**
- Creating a frame table writes **zero** files.
- Save project -> load project -> all addresses still resolve, including relative
  rewriting of the path portion.

**Boundaries:**

- No UI file accesses `controller._*`.
- No widget reads another widget's private state.
- Operator modules construct no Qt objects (after P1.12).
- Every declared output type tag is registered and valid.
- No synchronous media decode in any renderer paint path.
- Schemas are per table: the same column name may hold different types in two
  tables.
- A splitting operator carries down every source column marked `carry_to_children`,
  plus all `identifier` and `index` columns unconditionally. In particular, a
  trial-level covariate such as `reaction_time` survives a split into frame rows --
  a guardrail against the rule this document got wrong in its fourth revision.
- Every execution method accepts `run` and reads no per-run value from `self`.
- Artifact keys separate two media columns on one row, and separate projects.
- Project load clears the in-memory artifact cache.

**Resources and concurrency:**

- The decoder pool never exceeds **its configured limit**, whatever that is set to,
  including under concurrent access. Test the bound, not a specific number.
- The worker pool never exceeds its configured size.
- No machine-dependent quantity is a module-level constant: worker count, handle
  count and cache ceiling are all reachable as settings.
- A cancelled render request never invokes its callback.
- Result draining is bounded per tick.
- `operation_id` propagates through progress, completion and cancellation, and
  stale results are rejected.
- Peak memory during a scroll over a synthetic 1M-row table stays under **the
  configured ceiling**, tested at a low ceiling so the test is fast and the bound
  is what is asserted. *(This one would have caught the flaw in the first version
  of this design.)*
- Serial and parallel blendshape extraction produce identical output on the same
  input. (This is the test that catches a stateful model shared across clips, whose
  symptom is otherwise subtly wrong numbers rather than a failure.)
- An operator declaring `model_lifecycle = "per_sequence"` receives a distinct
  model instance per clip-run.
- A clip-run cache identity is stable across two separate runs with the same
  inputs. (Guards against an ephemeral UUID creeping into the key, which would
  silently disable the cache rather than break anything visibly.)
- Resuming an independent-frame run reproduces the un-resumed result exactly.
- Resuming a tracking run from a reset boundary reproduces the un-resumed result
  exactly; resuming mid-clip is refused rather than approximated.
- Every table accepted by Dataset conforms to its schema's dtype policy.

Test at **component seams, not internals.** Static AST checks suit forbidden
imports and private access; everything else goes through public seams. Asserting on
internals makes every refactor a test-rewriting slog, and refactoring then stops
happening.

---

## 8. Working rules for these sessions

Y B's stated concern is not that features are hard to build -- Claude builds them
quickly. It is keeping the result **modular, well-tested, and widely enough
conceived**, with freedom to make any change that helps long-term.

**For Claude:**

- **Propose refactors; do not patch.** Late in a long session there is a pull
  toward patching because refactoring feels disruptive. Resist it, and say
  explicitly when a clean fix requires touching more than the task at hand.
- **Do not treat existing code as a constraint.** Y B has explicitly authorised
  replacing it.
- **Check a rule's status before relying on it.** `[TARGET]` describes the future.
- **Confirm which component owns a feature before writing code.**
- **Flag anything crossing the main-thread / worker boundary.**
- **Write tests from the spec, not from the implementation.**
- **Ask rather than assume** when this document or the design doc is ambiguous.
- **Watch for study-specific vocabulary leaking into general components.** Before
  building a feature, name the parameter that makes it general: ROI OCR is "read a
  marker from a region"; segmentation is "split by any start/end columns"; trimming
  is a segment, not a mode.
- **A number that does not generalise becomes a setting or a runtime measurement,
  never a constant.**
- **Do not front-load the whole codebase.** `gelem_codebase_for_claude.txt` is a
  point-in-time review export, not a source of truth, and goes stale immediately.
  Read the specific files a question touches. Note it also states that every
  operator returns a dict, which is false -- `create_table()` returns a DataFrame.

**For Y B:** ask "what would you rip out if you could?" at intervals. Claude will
not volunteer it.

---

## 9. Non-goals and deferred decisions

**Explicitly out of scope now:**

- Any database or out-of-core storage engine (§5). Revisit when §5.7 timings say so.
- Dense frame-by-frame analysis of the full 240-hour corpus -- compute-bound, not
  an architecture problem.
- GPU inference. Undermines "installs easily for undergraduates".
- Distributed or cluster execution.
- Crash recovery of in-flight operator results. Cancellation is supported;
  surviving a process crash would need checkpointing and is not planned.
- OCR / ROI trial-number detection. **Deliberately postponed by Y B**; short clips
  will be produced externally for now. When it returns, it is "let the user mark a
  region on a sample frame, then read from that region on every frame" -- and the
  `#r=` address form already expresses it.
- Avatar rendering (blocked on a usable VRM source).
- Audio analysis. Not scheduled, but the resolver interface (§3.3) is shaped so it
  does not require replacing.

**Deferred, with known cost:**

- **`row_id` is a string.** At 500k rows this costs ~50 MB, which is tolerable. At
  tens of millions it is fatal. Changing it is a wide refactor. Revisit only if the
  row-count trigger fires.

---

## 10. Considered and rejected

These were weighed and turned down for stated reasons. Reversing one is allowed,
but do it knowingly -- say which reason no longer holds.

**Frame index and source path as two separate columns**, instead of one address
string. Cheaper in memory (categorical path plus int32 index) and directly
filterable and sortable. **Rejected because it breaks with more than one media
column per row**, which the design already supports (`GridTile` exists to show
`full_path` and `avatar_path` together). Two media columns would need four physical
columns plus a rule about which pair goes with which. Revisit only if
address-string memory is measured to be a real problem, and then as a logical
column backed by several physical ones.

**Surrogate lineage pointers (`source_row_id`) on derived rows.** Proposed during
review, and rejected by Y B: lineage belongs in ordinary data columns --
`participant_id, trial_id, frame_index` -- the way it would be handled in R. A
pointer graph would duplicate what those columns already say in a form no analysis
would use. Table-level lineage ("this table was made by segmenting that one")
belongs in the provenance log. Media sharing, which was the other motivation, is
solved by address-keyed artifacts (§4.5) instead.

**Extending the renderer signature to receive the whole row.** Rejected for the
same reason as separate columns, and unnecessary once the address is
self-contained. The existing `render(value, size, mode, context)` contract is
sufficient -- but note that `context` is a *subscriber identity*, not part of the
picture's identity (§4.5), which is precisely the distinction the current code gets
wrong.

**A database engine (DuckDB over Parquet) from the start.** See §5.2. No runtime
cost on small data; rejected because it hardens the operator contract and conflicts
with progressive per-row updates.

**HTML with `<video>` elements for side-by-side playback.** Still reasonable for
many short clips and gives a shared scrubber cheaply, but native `QMediaPlayer` is
better for scrubbing long files and is what §4.2 specifies. Revisit if synchronised
playback of many clips becomes a priority.

**A filmstrip view** (one row per trial, a frame every ~100 ms) as the primary way
to compare trials. Genuinely useful -- you see whole trajectories at once rather
than replaying them -- but it is a display operator, not infrastructure, so it is
out of Phase 1. Not rejected on merit; deferred.

**Materialised clip files as the unit Gelem sees.** Clips are currently produced
externally, which is fine for getting moving. But once addresses express time
ranges, a clip need not be a file, and the external step becomes optional. Do not
build anything that assumes clips are files on disk.

---

## 11. Numbers, and which of them generalise

**All performance figures below are estimates until §6.0 runs. Replace them with
measured values and mark them as measured.**

The distinction matters more than the numbers. A measurement taken on one machine
with one dataset can support a design decision only if what it tests is structural.

**Mechanism generalises; the ratio does not.** *Revised in the second review round,
which was right that the first version overclaimed.* The reason keyframe-skipping
is fast is structural and holds everywhere. **The measured ratio still varies with
codec, GOP length, container, storage device, library version and platform.** So a
single file supports a decision of the form "is this mechanism worth building" and
does **not** establish a number that holds for any user's data.

Where practical, measure across a few representative fixtures rather than one:
H.264 and H.265; constant and variable frame rate; short and long GOP; and at least
one phone recording, since phone files are both common in this work and the most
likely to be VFR.

Runtime adaptation remains the stronger design in every case: measuring a property
per file at load time (§4.1a) is more robust than any benchmark, however many
fixtures it used.

| Quantity | Estimate | Why it generalises |
|---|---|---|
| Keyframe-only extraction vs full decode | ~100x | Skipping intermediate-frame decode is inherent to inter-frame compression |
| Seek to a scattered frame | 50-200 ms | Same mechanism, any machine |
| Sequential frame in a decoded span | 2-5 ms | Same |
| MediaPipe parallelises across threads? | unknown | Depends on whether the library releases the GIL |
| Tracking vs per-image output divergence | unknown | A property of the model |

**Does not generalise -- must become a setting or a runtime measurement.**

| Quantity | Estimate | Handling |
|---|---|---|
| Keyframe interval | 1-10 s | Measured per video at runtime (§4.1a) |
| MediaPipe per frame, CPU | ~25 ms | Operational only; no decision depends on it |
| Worker count | -- | Setting, default low |
| Thumbnail cache size | 500 MB | Setting |

**Corpus and storage figures -- arithmetic, not measurement.**

| Quantity | Value |
|---|---|
| Common-case corpus | 20-120 videos, 10-120 min → up to ~240 h |
| Frames in 240 h @ 30 fps | ~26 M (never materialise all of these) |
| Realistic frame table (Y B's study) | ~530 k rows, ~300 MB |
| Thumbnail, 150 px JPEG | ~6 KB |
| Proxy layer per hour of video | ~20 MB |
| Full decode of 1 h video | ~2-6 min |
| Y B's study, every frame, 1 thread | ~4.4 h (follows from 25 ms/frame) |
| Same, with tracking + 6 workers | ~30-45 min (follows from the above) |

The last two inherit the per-frame estimate, so they are operational planning
figures for Y B's machine, not design inputs.
