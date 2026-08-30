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
- *26 Aug 2026 (a):* the measurement pass has started. The video fixture set is
  built and verified; contents and generation commands are in `docs/fixtures.md`.
  First result recorded: keyframe intervals in all three real recordings are dense
  (0.6 s to 2.0 s), which **weakens** the case for P1.3 rather than strengthening
  it (§6.0). Two §11 rows are now measured. No timing measurement has run yet.
- *26 Aug 2026 (b):* the video measurement pass completed (§6.0, §11). **P1.3 (the
  proxy layer) is rejected**, on the fill-time rule agreed before any number was
  seen -- moved to §10 with the full record needed to reverse it. **P1.7a
  (segment-thumbnail batch job) is rewritten**: sorted per-trial seeking replaces
  the full sequential decode pass, which the same measurement showed to be two to
  three orders of magnitude more expensive at any realistic trial density (§4.1b,
  §6.2). **§4.1a is folded into §10** -- the keyframe-gap signal it proposed
  measuring did not predict seek cost on the real recordings, and one sentence of
  the surviving principle (measure per file at load time; do not assume) stays in
  §4.1. **The §4.3 batching claim is downgraded to an open question**: no benefit
  was measured, but the method's mandatory warm-cache protocol cannot see the
  cold-cache effect the claim is actually about. **Approximation is removed from
  the design entirely**: with no proxy, no thumbnail source is approximate any
  longer, which simplifies §3.6 item 13 and the §4 opening table. Full method,
  machine, and raw numbers: `%USERPROFILE%\Documents\gelem.measure\RUNLOG.md`
  (outside the repo; see §6.0). MediaPipe measurements are unaffected and still
  outstanding, still gating Phase 2.
- *26 Aug 2026 (c), P0.3:* §3.6's twelve open questions are settled and the
  section renamed "Address semantics -- settled". `media/media_address.py`
  implements the pure-logic half (parsing, canonical form, frame selection
  given supplied timings, region-to-pixel arithmetic); `docs/fixtures.md`
  gains the lossless known-frame fixture §7 needed and a video-stream
  edit-list fixture for decision 12, both generated on demand rather than
  committed. Tests: `tests/test_media_address.py`.
- *27 Aug 2026, P0.4:* §6.1's P0.4 done. The controller owns one flat ordered
  query result plus group-boundary spans (`models/query_result.py`); the
  gallery holds no row ids and is given an index range into that order,
  reporting the range it displays back as absolute indices. Grouped mode is
  one flat order plus boundaries, not a separate structure. `result_changed`
  replaces the `gallery_updated` / `grouped_gallery_updated` signal pair;
  `save_filtered_as_table()` reuses the owned order instead of re-querying.
  Tests: `tests/test_visible_row_order.py`, `tests/test_ui_private_access.py`.

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
<path>#[v=<n>|a=<n>&](f=<int>|t=<sec>|t=<start>-<end>)[&r=<x,y,w,h>]   full form
```

**Escaping (§3.6 decision 1).** Inside `<path>`, `%` is written `%25` and `#`
is written `%23`. Nothing else is escaped. The fragment begins at the first
`#` that is not part of that escaping.

**Stream selector (§3.6 decision 7).** An optional `v=<n>` or `a=<n>` names a
stream explicitly, by zero-based index within streams of that kind. Omitted,
the default is the lowest-index stream of the requested type -- never the
container's "best" stream.

**Canonical component order (§3.6 decision 9).** When a fragment is written
out, its components appear in this fixed order: stream selector, then `f=`
or `t=`, then `r=`, joined by `&`. Parsing accepts any order; formatting
always produces this one.

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
3. `ArtifactStore._decode_source` / `_first_frame_as_pil` (was
   `_generate_thumbnails` until P0.5b-2i split out the decode)

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
- Addresses survive project save/load. `models/dataset.py`'s `_rewrite_media_cell`
  parses each media cell as an address, moves only the path portion
  (relative-if-inside on save, absolute on load), and re-formats, so an address
  survives a project round trip with its fragment (`#f=`, `#t=`, `#r=`, stream
  selector) intact. The path arithmetic is `media/media_address.py`'s
  `absolutise` / `relativise`. Built in P0.2c; tests
  `tests/test_dataset_address_paths.py`.
- Only paths **inside** the project folder are made relative. External source
  paths stay absolute unless the user explicitly imports the file.

### 3.6 Address semantics -- settled

**Settled 26 August 2026. Each decision below has a test, or a recorded
reason why it does not yet.** These were open questions whose common failure
is a plausible but wrong frame rather than an error, which is why they were
settled before the parser was written rather than during it.

Some decisions are pure logic and are tested by P0.3. Others describe what
the resolver must do when it decodes, and their tests are owed by P1.2. Each
decision says which.

**1. Escaping.** The fragment begins at the first `#` that is not part of an
escape. Inside the path portion `%` is written `%25` and `#` is written
`%23`, and nothing else is escaped: `&`, `,` and `-` carry meaning only
after the `#`, so a path containing them needs no treatment. A filesystem
path becomes an address only through `MediaAddress.from_path()`. No code
builds an address by joining strings.
*Why:* backslash escaping collides with Windows separators, and only two
characters are genuinely ambiguous.
*Tested by P0.3.*

**2. Time point.** `#t=X` selects the frame whose presentation interval
contains X -- the frame being displayed at that moment. The final frame's
interval runs to the end of the stream.
*Why:* the only rule that still works when frame durations vary, and it
matches what a person means by "what is on screen then".
*Tested by P0.3 against supplied frame timings; end-to-end test owed by P1.2.*

**3. Range endpoints.** Half-open. `#t=a-b` contains a frame whose
presentation time p satisfies `a <= p < b`.
*Why:* adjacent ranges then tile exactly. Inclusive ends place a boundary
frame in two ranges, so it is counted twice in both of their averages.
*Tested by P0.3 against supplied frame timings.*

**4. Representative frame for a range.** A named policy, default `first`. A
bare video path is treated as the range covering the whole file, so one
policy governs both cases. **Corrected 26 Aug 2026 (P0.3 amendment):** both
policies choose only from the frames the range actually contains under
decision 3 -- never a frame from outside it, even one whose display interval
still overlaps the range's start. `first` is the earliest frame in the
range. `midpoint` is the frame in the range nearest to `a + (b-a)/2`, ties
going to the earlier frame. A range containing no frame at all raises rather
than falling back to a frame outside it (decision 11). The policy is **not
part of the address string**. It is an argument at resolve time and part of
the artifact cache key (section 4.5).
*Why:* the original definition of `first` -- the frame selected by decision
2 at the range's start -- silently violated section 7's own invariant that a
segment's thumbnail comes from inside that segment's own time range,
whenever the range's start did not land exactly on a frame's own timestamp.
That is the common case: start times come from a data file, not from frame
arithmetic. Choosing only from the range's own members is what makes
decisions 3 and 4 agree, and the first frame still assumes nothing about the
content and needs no seek beyond the range's own start. Because the policy
is part of the cache key, changing the default later is safe -- existing
pictures are simply not reused.
*Tested by P0.3 for policy selection, and for the invariant that either
policy's choice is always a member of decision 3's own range.*
`canonical_key()` (decision 9) supplies only the address component of the
artifact cache key -- folding `policy` into the *full* key alongside the
other components section 4.5 lists (source fingerprint, purpose/variant,
cache version) is P0.5's job, not this module's, so "key distinctness" is
not something P0.3 can test and is not claimed here.
*Picture-level test owed by P1.2.*

**5. Region.** `#r=x,y,w,h`, four numbers in the closed interval 0 to 1,
expressed as fractions of the **upright** frame (decision 6). Conversion to
pixels rounds the four edges, not the origin and size separately: the left
edge is `round(x*W)`, the right edge is `round((x+w)*W)`, and the width is
their difference. Values outside 0 to 1, a zero or negative width or height,
or `x+w > 1` are errors.
*Why:* the same address must mean the same region whatever size the frame is
decoded at, so a region marked on a small preview is exact at full
resolution. Rounding edges rather than widths makes two adjacent regions
rejoin with no seam and no overlap.
*Note:* nothing in Gelem currently produces a region address. The likely
future sources are marking a region by hand, splitting a side-by-side
recording into two rows, and cropping to a detected face before analysis.
None is scheduled.
*Tested by P0.3 for parsing, validation and the pixel arithmetic;
picture-level test owed by P1.2.*

**6. Orientation.** The resolver always returns display-oriented frames,
applying the container's display matrix, for analysis exactly as for
display. Frame dimensions it reports are post-rotation. Regions are
expressed on the upright frame. Quarter turns and horizontal mirroring are
supported; any other display matrix raises rather than being ignored.
*Why:* a landmark model shown a sideways face returns garbage. Applying
rotation in exactly one place is the only way it is never forgotten. Note
that PyAV does not apply rotation automatically the way the ffmpeg command
line does; this is real work in the resolver.
*Fixture:* the phone recording (`docs/fixtures.md`) carries `rotation=90` --
stored landscape, displayed portrait -- so this is a real input, not a
hypothetical, once P1.2 exists to test against it.
*Owed by P1.2.*

**7. Stream selection.** The default is the **lowest-index stream** of the
requested type, not the container's "best" stream. An address may name
another explicitly with `&v=<n>` or `&a=<n>`. The canonical form omits the
selector when it names the default. Frame ordinals and times are counted
within the selected stream.
*Why:* "best stream" is a library heuristic that can change between library
versions, which would quietly change which camera was analysed.
*Tested by P0.3 for parsing and canonical form; selection test owed by P1.2.*

**8. Frame identity.** `#f=N` is the zero-based ordinal of a frame in
**presentation order** of the selected stream, counted after any edit list
(decision 12). `#t=` is a time. **The parser never converts between them**;
conversion requires the file and belongs to the resolver. Two addresses are
equal only if their canonical strings are equal, so `#f=100` and `#t=4.0`
are different addresses even on a file where they resolve to the same frame.
The cost is a duplicate cached picture, never a wrong one.
*Why:* multiplying a time by the nominal frame rate is the most likely
wrong-frame bug there is, and it is always wrong on phone video. Forbidding
the conversion in pure logic is what prevents it.
*Consequence for P1.2:* resolving `#f=N` on a variable-frame-rate file
requires a per-file index of frame times, built on first use. Per CLAUDE.md's
generality rule, this is the first feature needing a runtime-measured
per-file property, and that rule should cite it.
*Fixture:* the phone recording (`docs/fixtures.md`) is variable frame rate
with 98 distinct frame durations, so it is a real input for the per-file
index P1.2 must build, not a hypothetical.
*Tested by P0.3.*

**9. Canonical form.** Components appear in a fixed order: stream selector,
then `f` or `t`, then `r`, joined by `&`. Paths use forward slashes. Case is
**not** folded -- on Windows two spellings of one path therefore produce two
cache entries, which wastes space and never produces a wrong picture, and
that trade is accepted deliberately. Times are parsed as exact integer
microseconds and written with six decimal places; more than six decimal
places is an error rather than a rounding, because silently rounding a time
is the failure this section exists to prevent. Frame numbers are plain
integers with no sign and no leading zeros. Region values carry six decimal
places. An empty fragment (`path#`) is an error.
Two forms exist and must not be confused. `format()` produces the **stored**
form, whose path may be relative to the project. `canonical_key(project_root)`
produces the **key** form, with the path resolved and absolute, and that is
what the artifact cache hashes.
*Why:* two spellings of one address must hash identically or the cache
silently keeps two copies of every picture.
*Tested by P0.3.*

**10. Time origin.** **One clock per file.** Its zero is the first presented
frame of the primary video stream, meaning the lowest-index video stream.
Audio times are expressed on that same clock. `PlaybackAdapter` converts to
whatever the player expects. Times in an address are never negative.
*Why:* the phone recording in docs/fixtures.md has an edit list on its audio
stream, so its sound and picture do not begin together. Two clocks would put
them permanently out of step by a constant -- uniform, plausible, and very
hard to notice.
*Future need, not built:* if a study's recorded times are counted from
something other than the file's first frame, that offset is a fact about the
data and belongs in an ordinary column, never inside the meaning of an
address. Gelem holds no opinion about what a study's clock started from.
Nothing in the current plan applies such an offset.
*Tested by P0.3 for the rejection of negative times; the rest owed by P1.2.*

**11. Degenerate values.** Refused when the address is read: a reversed range
(`#t=5-2`), a zero-length range (`#t=3-3`), a negative time, a negative frame
number, and a region outside 0 to 1 or with zero area. Each raises with a
message naming the problem. **A reversed range is never reordered and a value
is never clamped**, because quietly correcting a broken data file makes it
produce plausible output forever.
Refused when the address is resolved: a start beyond the end of the stream,
a frame ordinal beyond the last frame, and a well-formed range (start < end)
that, against the file's real frame times, contains no frame at all --
distinct from the zero-length range above, which is refused earlier, at
parse. These raise for that row only; other rows are unaffected, which the
progressive result path already supports.
*Behaviour at row creation, documented here and built in P1.6, not now:* the
operator that creates range rows checks them against the files, reports how
many do not fit, and offers to use what exists or to cancel. If the user
proceeds, the **shortened range is written into the row**, so the stored
address states what will actually be read, and the row records that it was
shortened. The resolver itself never shortens anything.
*Why:* a row whose range overruns the recording still produces a
perfectly good-looking tile, because the tile shows the first frame. Only
the analysis is short. The problem is invisible unless something checks.
*Tested by P0.3 for the parse-time refusals and for the empty-range
resolve-time refusal.*

**12. Frame ordinal after an edit list.** The ordinal counts presented frames
of the selected stream, after the edit list is applied. Frame 0 and time 0
are therefore the same frame.
*Why:* any other choice puts frame numbers and times a constant distance
apart, which is decision 10's failure in a different disguise.
**Coverage, verified on the author's machine 26 Aug 2026 -- not verifiable on
every checkout.** No file in the committed fixture set has an edit list on
its video stream, so P0.3 attempts to produce one on demand: a
non-keyframe-aligned stream-copy cut of the Zoom recording
(`docs/fixtures.md`), checked afterwards with `ffprobe -v debug` rather than
assumed to have worked. On this attempt it did: the cut carries a genuine
video-stream edit list (`media time: 9000, duration: 90600` at the file's
1/30000 time base -- a 0.3 s pre-roll, matching the 0.3 s the requested start
sits past the nearest keyframe). This confirms a fixture with a video-stream
edit list can be produced and detected; it is **not** an end-to-end test
that frame numbering is correct against one, which still requires the
resolver and is owed by P1.2.
**This verification is weaker than the other twelve decisions'.** The test
needs `GELEM_FIXTURES` pointing at the real, uncommitted recordings
(`docs/fixtures.md`'s opening section: local disk only, never synced), so on
any other checkout -- CI included -- it unconditionally skips, which looks
identical to "not yet run" rather than "regressed". If this attempt stops
reproducing on a future ffmpeg version, nothing forces this note to be
corrected back to unverified; treat "Coverage, verified" here as true only
as of a run against `GELEM_FIXTURES` on this machine, not as a standing
guarantee the way the other decisions' test coverage is.

**13. Where approximation is permitted.** *Simplified 26 Aug 2026: the exception
named here was the whole-video proxy layer, which is rejected (§10) after the
measurement pass. No caller anywhere in Gelem now produces an approximate
picture -- segment tiles resolve exactly (§4.1b), short-span decode caches real
frames (§4.1), and there is no third source.* **A point address always resolves
exactly, everywhere, with no permitted approximation.** State this as an
unconditional property of the address, so no future caller can quietly relax
it.

**Internally, prefer integer microseconds or rational PTS to floating-point
seconds.** Float seconds accumulate error across a long video and make two
addresses that should be identical compare unequal.

---

## 4. Display, playback, analysis are three different paths

They have opposite requirements and must not share a mechanism.

| | Resolution | Access pattern | Caching | Tolerates approximation |
|---|---|---|---|---|
| **Display** | low (tile size) | random | aggressive | **no** *(was yes; see below)* |
| **Playback** | full | sequential | none (player's job) | n/a |
| **Analysis** | full | strictly sequential | none | **no** |

*Changed 26 Aug 2026.* The original guiding principle was **display tolerates
approximation, analysis does not**: a thumbnail could be the nearest available
sampled frame, where a blendshape had to come from the exact frame, decoded
properly. That distinction existed only to permit one thing -- the whole-video
proxy layer's nearest-keyframe substitution (§4.1a) -- and the proxy is rejected
after measurement (§10). With no proxy, nothing in Gelem produces an approximate
picture any more: segment tiles resolve to an exact frame inside their own time
range (§4.1b), and short-span decode caches real, exactly-decoded frames (§4.1).
The distinction in this table is now the same as §3.6 item 13's: a point address
always resolves exactly, for display exactly as much as for analysis. If a future
feature reintroduces an approximate display path, it must re-earn a "yes" here
explicitly, not inherit one left over from this table's original design.

### 4.1 Display -- two exact thumbnail sources

**Rewritten 26 Aug 2026, after the measurement pass.** This section described a
required whole-video proxy layer and its interaction with two other thumbnail
sources. The proxy is rejected on measurement (§10); there are now **two exact
thumbnail sources**, not three, and neither approximates.

Video stores complete images only occasionally (every 1-10 s) and only differences
in between. Producing frame N means jumping to the last complete image before it
and replaying forward.

- **Frames near each other** (all frames of one 3 s trial): one forward pass, a few
  ms per frame. Fast.
- **Frames scattered across a long video** (first frame of each of 150 trials
  across an hour): a separate seek each. **This was assumed to cost 50-200 ms per
  seek**, unacceptable at gallery scale, and was the reason a proxy layer was
  proposed. **Measured 26 Aug 2026** (§11): scattered-seek cost on Y B's three real
  recordings is 25-113 ms, and filling one screen of 30 tiles measures 0.4-1.7 s
  with the worker pool in §4.4, not the 10-30 s this section originally assumed.
  See §10 for the full record.

**(a) What's left of the proxy layer: one surviving principle.** *Folded into §10,
26 Aug 2026 -- this letter is kept only so `§4.1a` still resolves to something,
since `§4.1b` and `§4.1c` below are cited elsewhere in this document and are not
renumbered.* Measuring a property of a file at load time -- as the rejected
proxy's keyframe-interval check did -- is more robust than any global assumption
baked into the code, and that pattern applies wherever a per-file property is
going to drive a decision, not only to media. See §10 for the full proxy record
and for why the *specific* signal proposed here (keyframe interval, as a proxy for
seek cost) turned out not to predict what it was chosen to predict.

**(b) Segment thumbnails -- for a media cell holding a time range.**
**Rewritten 4 Aug 2026, then again 26 Aug 2026.**

The previous version said the segment operator should capture each representative
frame "during the sequential pass it is already making". **That assumed
segmentation decodes video.** A metadata-driven segmentation -- start and end
columns from a trial CSV, which is the common workflow -- decodes nothing. There
is no pass to piggyback on.

The version after that said this batch job should make one full sequential decode
pass per video, sorted by source file and start time, on the assumption that
seeking to each trial separately would be far more expensive. **Measurement 26
Aug 2026 showed the opposite**, sharply: on all three real recordings, sorted
per-trial seeking is cheaper than one full sequential pass by two to three orders
of magnitude at any realistic trial count. The crossover -- the trial density
above which a full pass would actually win -- is 16,800 to 58,600 trials **per
hour of video**, depending on the file. At 3,000 trials per hour, seeking is
about five times cheaper on the least favourable recording measured; at 300 per
hour, about fifty times. A sequential pass becomes correct again only above
roughly seventeen thousand trials in a single hour of recording -- a trial every
fifth of a second. No study drives anywhere near that. Full numbers:
`%USERPROFILE%\Documents\gelem.measure\section11_table.md`.

So exact segment thumbnails are an **ArtifactStore batch job**: collect the
outstanding segments, sort by source file and start time, and **seek to each
representative frame**. No full sequential pass.

An operator that *is* decoding anyway may offer a decoded representative frame as
a hint. It never writes into ArtifactStore itself.

The guardrail test is unchanged: **a segment's thumbnail comes from inside that
segment's own time range.**

**(c) Short-span decode -- for frame-level browsing inside a clip.**
Neither of the sources above suits browsing individual frames of a 3 s trial: (a)
no longer exists, and (b) gives one representative frame per segment, not every
frame in it. Decode the whole span once (~90 frames, a fraction of a second),
cache every frame's thumbnail. Prefetch when a trial is selected. Browsing within
that trial is then instant.

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
- **Batch requests by source file** before decoding, so repeated calls for the same
  file share one open container. Within a batch, decoding frames that are near
  each other in one forward pass rather than seeking to each is fast and
  uncontested -- thirty frames of one clip is one sequential pass, not thirty
  seeks (§4.1c).
- **Open question, downgraded 26 Aug 2026: does *sorting scattered requests* by
  frame index before servicing them add anything beyond that?** This document
  previously claimed it was "the single biggest lever for scroll smoothness" and
  was untested. Measured 26 Aug 2026 (§11): sorted order was 0.5-2.9% faster than
  scattered (random) order across all eight fixtures -- one file measured 7.4%
  *slower* sorted -- no meaningful benefit, under the warm-cache protocol the
  measurement pass required. **That protocol removes exactly the disk-I/O
  locality effect sorting is meant to exploit, so this result is silent on the
  cold-cache and Google Drive Streaming case the claim is actually about -- it is
  not a refutation.** Do not build scheduling complexity around this claim until
  a cold-cache variant tests it properly (outstanding, §6.0).

### 4.4 Replace thread-per-request with a bounded pool

*Partly done, P0.5b-2i (30 Aug 2026).* `ArtifactStore.request_thumbnail` no
longer spawns a raw thread per call. It runs jobs on a bounded `WorkerPool`
(`artifacts/worker_pool.py`), coalesces requests by canonical address (one
decode, many `(table, row)` subscribers), and cancels via a generation counter
that `reset()` bumps -- a stale job is guaranteed to leave no index entry, no
fingerprint-memo entry and to send no notification (a JPEG it had already
encoded can linger on disk with nothing pointing at it -- reclaiming it is
P0.5b-2ii). Worker count is a keyword-only `ArtifactStore` constructor
parameter, default 2. **Still outstanding, P0.5b-3:** priority ordering
(`get_displayed_ranges()` has no caller yet) and viewport cancellation. Tests:
`tests/test_request_queue.py`.

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
+ artifact purpose / variant  (thumbnail, preview, segment thumbnail)
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
2. A cache miss returns a placeholder **immediately**. **Done, P0.5b-3i:**
   `make_media_path_renderer`'s `render()` returns a grey placeholder pixmap on a
   thumbnail-mode miss.
3. **No media is opened or decoded during a paint.** **Done, P0.5b-3i:** the
   thumbnail path is cache-or-placeholder for both image and video tiles;
   `_render_image`'s `Image.open` fallback and `_video_first_frame_pixmap`'s
   `cv2.VideoCapture` are gone. On a miss the controller
   (`render_column_value`) queues one `ArtifactStore.request_thumbnail`. Detail
   mode still opens the source, by design.
4. ArtifactStore queues the request with viewport priority. **Still open,
   P0.5b-3ii:** the request is queued but FIFO, not viewport-prioritised
   (`get_displayed_ranges()` still has no caller).
5. Stale off-screen requests are cancelled. **Still open, P0.5b-3ii.**
6. Workers return raw image data or a persisted cache artifact -- never a
   `QPixmap`.
7. `QPixmap` construction happens only on the main thread.
8. The ready notification carries enough context to repaint the right table, row
   and column.

**Resolved, P0.5b-3i:** the controller no longer requests a thumbnail for every
row immediately after loading. `load_folder`, `load_csv_as_primary` and the
operator `create_table` result path queue nothing; requests are issued by
`render_column_value` as tiles paint.

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

*Status 26 Aug 2026: built and verified.* Eight files. Three real recordings -- a
Zoom study recording (h264, 1080p25), a legacy study recording (**mpeg4**, a third
codec family, 3200x1200 at 20 fps) and a phone recording (h264, variable frame
rate, `rotation=90`, audio-stream edit list). Five generated from a 20-minute
normalised excerpt of the Zoom file, crossing codec (h264 / hevc) with GOP length
(1 s / 10 s) at constant frame rate, plus one variable-frame-rate long-GOP file
whose keyframe gaps run from 10 s to 49 s.

Every generated file was verified against the container rather than trusted from
the encoder log. Contents, measured properties, generation commands and the
verification block are in `docs/fixtures.md`. The media itself is kept on local
disk outside the repo -- it contains participants and runs to about 2 GB -- and is
located through a `GELEM_FIXTURES` environment variable.

See §11 for which of these generalise beyond Y B's machine and which do not. In
brief: the ratios test structural properties of codecs and libraries and do
generalise; absolute per-frame times do not and must not drive design.

**Video and decoder -- gates Phase 1. Done, 26 Aug 2026.**

- **keyframe-only extraction versus full decode** (claimed ~100x). **Done.** Ratio
  tracks `gap_frames` closely, but with a fitted k of 1.1-1.35 across all three
  codec families -- well below the pre-registered guess of k = 2-5, because
  JPEG-encode cost (paid on both sides of the comparison, by design) dominates
  and compresses the ratio toward the theoretical ceiling. Full account, and why
  this is not a measurement error, in §11.
- **seeking to scattered frames** (claimed 50-200 ms each). **Done.** 25-113 ms on
  Y B's three real recordings, up to 199 ms on the sparsest synthetic fixture --
  not the flat 50-200 ms this document assumed for all cases. The fixed cost of a
  seek (index lookup, decoder flush/reinit) is 64% of total seek time on
  dense-keyframe files and 15% on sparse ones (§11).
- **frames inside an already-decoded span** (claimed 2-5 ms). **Done.** 5.5-15.9
  ms/frame (analysis mode, no encode) on the three real recordings -- higher than
  claimed, and not explained by resolution: the phone recording, not the
  higher-resolution legacy one, is the outlier (§11).
- **keyframe interval in the actual recordings. Done, 26 Aug 2026.** Not a value
  to hardcode -- §4.1a records the surviving principle that this kind of
  property is measured per file, not assumed. What Y B's files tell us is how
  common each case is.

  *Result.* All three real recordings are dense. Legacy study (mpeg4, 20 fps):
  0.600 s, perfectly uniform, 3582 keyframes. Phone (h264, VFR): 1.002 s mean,
  179 keyframes. Zoom (h264, 25 fps): 2.000 s mean, minimum 0.760 s because scene
  detection is active, 1338 keyframes.

  *Reading, decided 26 Aug 2026 once the seek and fill-time measurements
  completed.* **P1.3 is not built** (§10). All three real recordings measure well
  inside the "not built" bands of the fill-time rule agreed before any number was
  seen, and two of the three independently trip the 240-hour-corpus disk gate.
  What changed across this measurement pass is the answer, not just the prior.

  *One further observation.* No real recording in this corpus has a long GOP. The
  two long-GOP fixtures therefore represent **some other user's data, not Y B's**.
  Their job is to prove Gelem neither breaks nor shows wrong pictures on such
  files, not to serve as a performance baseline for this study.

**New, added 26 Aug 2026: sorted-versus-scattered batching benefit.** §4.3's
claim that sorting requests by frame index is "the single biggest lever for
scroll smoothness" was tested and no benefit was observed (0.5-2.9%, one file
slower sorted) -- but under the mandatory warm-cache protocol, which cannot see
the cold-cache disk-locality effect the claim is actually about. **Outstanding:**
a cold-cache variant is needed before that claim can be trusted either way (§4.3,
§11).

**MediaPipe -- outstanding, gates Phase 2 only.** Not yet run. **Note added
26 Aug 2026:** when these are eventually measured, take them **against whatever
MediaPipe version Phase 2 actually uses**, not against whatever version happens
to be installed at measurement time -- P2.2 makes model version part of the
result-cache identity, so a number measured against one version is not a
substitute for measuring against another, and stale numbers here would be
easy to mistake for current ones.

- does inference parallelise across threads? Time 200 frames on 1 thread, then 4.
  This is a property of whether the library releases the GIL, so it generalises.
  **Still gates Phase 2** -- decides threads versus processes (P2.3).
- tracking mode versus per-image: how much faster, and **how far do the numbers
  diverge**. The second half is a reproducibility question, not a speed one.
  **Still gates Phase 2** -- decides whether tracking is safe to default to
  (P2.4).
- per-frame time on this machine. **No longer operational, and no current
  decision depends on it.** This was going to tell Y B whether his driving
  study (§2) was a 4-hour or a 40-hour run -- moot, since that study's
  blendshapes are already extracted. Kept as an outstanding item because a
  future study will want the planning number, not because anything gates on
  it now. See §11 for the per-frame-cost row and the two run-time estimates
  that follow from it, all now marked as unverified estimates that no current
  decision depends on.

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
| P0.5 | ArtifactStore identity and demand-driven display | Segment thumbnails and every tile attach here |

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
10. Handle the address grammar in the project-relative path rewriting, so a
    saved project reopens with its addresses intact (section 3.5). Uses the
    module built in P0.3. **Done, P0.2c:** `models/dataset.py`'s
    `_rewrite_media_cell` parses each cell and moves only the path portion,
    using `media/media_address.py`'s `absolutise` / `relativise`; tests
    `tests/test_dataset_address_paths.py`.

**Done, P0.2b (27 August 2026)** -- items 6-9. The five list queues became four
`queue.SimpleQueue`s plus a single lock-guarded latest-progress value; each
queue is drained by at most `AppController._drain_budget` items per 50 ms tick
(a constructor parameter, not a constant), and a tick's per-row results are
grouped by table and applied with **one** `Dataset.apply_row_updates()` call and
**one** `rows_updated` emission per table. One `operation_id` is minted per run
in all three `run_*` methods and travels through every completion and error
callback. The controller keeps a registry of live runs keyed by `operation_id`,
cleared only by `load_folder()`, `load_csv_as_primary()` and `load_project()` --
**not** by `set_active_table()`. **"Stale" means the run is no longer live, never
"the user switched tables":** a result carries its own `table_name` and lands in
that table whatever is on screen. A per-row result from a dead run is dropped
silently; a `create_table` / `create_display` result from a dead run is not
stored and raises `error_occurred`. `apply_row_updates()` now returns the
row_ids it could not place; the controller accumulates them per run and reports
the count at completion. Notifications are frozen payloads (`RowsUpdated`,
`ThumbnailsReady` in `models/notifications.py`, no pandas/numpy/PIL import) and
the signals were renamed to the plural `rows_updated` / `thumbnails_ready`.
`ArtifactStore.request_thumbnail()` takes the table name and echoes it back.
`MainWindow` is the single place that checks a payload's table against
`AppController.get_active_table()` before repainting; each gallery makes one pass
over its mounted tiles per batch. The two `[MIGRATING]` worker-thread label
lookups are gone -- `BaseOperator.display_label` owns that chain. Supersession
of one run by another is deliberately not detected (that is P1.12). Tests:
`tests/test_result_delivery.py`, rewritten
`tests/test_controller_async_contracts.py`, and a signal-signature check added
to `tests/test_fake_controller_contract.py`.

**P0.2 therefore runs after P0.3** -- item 10 needs `MediaAddress` to exist.

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

**Done, 27 August 2026.** The controller owns a single `QueryResult`
(`models/query_result.py`): one flat ordered sequence of row ids plus, in
grouped mode, a tuple of `GroupSection(label, start, stop)` spans into that same
sequence. **Grouped mode does not change which rows are visible, only how they
are stacked**, so one flat order serves both views; the flat order in grouped
mode is `QueryEngine.apply_grouped()`'s dict concatenated in its existing order,
never a second `apply()` call. The UI is handed a `ResultLayout` that carries
`total` and the group spans but **no row ids**. Each gallery is given an
absolute half-open index range into the flat order and fetches the ids it needs
in one batch call per viewport update; it reports the range it currently shows
back as absolute indices plus its own `result_id` (`displayed_range_changed`),
under a viewport key `MainWindow` owns. The reported range is the **mounted**
window, which includes `GalleryWidget._BUFFER_ROWS` above and below the strictly
visible rows -- it is a superset of what is literally on screen, so P0.5 should
not add its own prefetch margin on top without accounting for this one. Every recompute mints a new `result_id`; a viewport report
naming a superseded id is dropped, and `get_displayed_ranges()` exposes what is
on screen for P0.5 to prioritise against. `save_filtered_as_table()` lost its
duplicate query entirely -- it now stores `QueryResult.row_ids` directly, so a
randomised on-screen order is saved as shown. Tests:
`tests/test_visible_row_order.py`, `tests/test_ui_private_access.py`.

**P0.5 ArtifactStore identity and demand-driven display.** *(was P0.3)*
Implement §4.5 and §4.6. Address-based keys, `ArtifactCodec` separated from source
decoding (§7), bounded worker pool with priority and cancellation, no decoding in a
paint path, clear the memory cache on project load, cache size becomes a setting.
This fixes the avatar-tile bug and is the foundation segment thumbnails and
every tile attach to. **The thumbnail path used to carry the project-reload
identity bug that P0.2b fixed for operator results** -- an in-flight worker
writing into the index after `reset()` while `load_folder()` re-minted the same
row ids. *Fixed P0.5b-2i:* `reset()` bumps a generation counter and a job does
its I/O into locals, committing the index and fingerprint memo in one
generation-checked lock hold, so a job `reset()` raced past commits nothing.
Address+fingerprint keying (P0.5b-1) already meant a straggler could not show a
wrong picture; P0.5b-2i means it also leaves no stale index or memo entry.

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

**P1.3 Proxy layer -- rejected 26 Aug 2026 on measurement.** See §10 for the full
record. This ID is retired, not reassigned.

**P1.4 Short-span decode cache** for frame-level browsing inside clips, with
prefetch on selection.

**P1.7a Segment thumbnail batch job** in ArtifactStore: collect outstanding
segments, sort by source file and start time, **seek to each representative
frame** (§4.1b). *Rewritten 26 Aug 2026 -- was one full sequential decode pass per
video; measurement showed sorted seeking cheaper by two to three orders of
magnitude at any realistic trial density. See §4.1b and §10.*

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
`html_path` versus `plot_html`. Promote a real operator as the thumbnail-era
reference (`operators/thumbnail.py` itself was deleted in P0.5b-2i). Fix
`self.output_dir` versus `self._output_dir` in the docs.

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
- **Fixture location, and why it is a rule rather than a detail.** The real
  fixtures contain participants and run to about 2 GB, so they are not in the repo
  and must not be. Tests locate them through a `GELEM_FIXTURES` environment
  variable and **skip rather than fail** when it is unset, so a student's checkout
  stays green without the media. The small synthetic lossless fixture is the
  opposite case: reproducible from a command and containing no participants, so it
  is generated on demand by the test fixture itself and may live in the repo. The
  dividing line is generated-and-small in the repo, real-and-large outside it. See
  `docs/fixtures.md`.
- Every §3.6 semantic decision has a test.
- **A segment's thumbnail comes from inside that segment's own time range.**
- Creating a frame table writes **zero** files.
- Save project -> load project -> all addresses still resolve, including relative
  rewriting of the path portion. **Owed by P0.2**, item 10 (§6.1) -- P0.3 supplies
  the address grammar this rewriting must handle, but does not do the rewriting.

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

**P1.3, the whole-video proxy layer. Rejected 26 Aug 2026, on the measurement
pass, against a rule agreed before any number was seen.** Full raw numbers, run
log and scripts: `%USERPROFILE%\Documents\gelem.measure\` (outside the repo;
`section11_table.md` and `RUNLOG.md`). Summary numbers are also in §11.

The rule: time to fill thirty tiles at a never-visited scroll position. Under
1 s -> not built, comfortable. 1-6 s -> not built, accepted as a deliberate
trade against the proxy's total cost. Over 6 s -> reconsider, after trying more
workers, reduced-resolution decode for the mpeg4 file, and prefetching ahead of
the scroll. Two gates defer it regardless of the time: build cost over 3 minutes
per hour of video, or proxy disk over 2 GB across a 240-hour corpus.

Measured fill times: **0.41 s** (Zoom recording), **0.37 s** (legacy mpeg4
recording), **1.65 s** (phone recording). All three land in the "not built"
bands.

Proxy disk across a 240-hour corpus, extrapolated from each recording's own
measured keyframe density and JPEG size: **0.93 GB** if the corpus looked like
the Zoom recording, **2.81 GB** if it looked like the legacy recording, **2.20
GB** if it looked like the phone recording. Two of the three real recording
types individually exceed the 2 GB gate on their own.

**K** -- how many approximate thumbnails a video's proxy would need to serve
before it repaid its own construction cost -- is **837** (Zoom), **5,093**
(legacy), **665** (phone). Even where the fill-time rule alone would already
say "not built" (as it does here), K says the payback bar for reviving this is
high: a proxy would need to be reused hundreds to thousands of times per hour of
source video.

**Costs beyond development time, weighed against a benefit that is at most
1.6 s of scroll comfort on this corpus:**

- **Disk.** Up to ~2.8 GB per real recording type, per the figures above, for a
  cache that exists solely to make an already-sub-2-second wait shorter.
- **Background decoding that competes with analysis runs.** Building a proxy
  means decoding video on a background worker while blendshape extraction wants
  the same CPU. That contention is real cost on top of the disk.
- **A permanent hazard, not a one-time one.** Every approximate picture is a
  chance it reaches a caller that needed an exact one. §3.6 item 13 and the §4
  display/analysis table exist specifically to police that boundary. Removing
  the proxy removes the hazard entirely rather than requiring it be policed
  forever.

**Condition for revisiting:** a corpus with genuinely long-GOP video (this
corpus has none -- see §6.0's "further observation" on the two synthetic
long-GOP fixtures), or a UI that draws thousands of approximate thumbnails from
a single video, such as a timeline scrubber. Neither exists in the plan today.
If one is proposed, re-run the measurement pass against it rather than assuming
these numbers still hold -- they are this corpus's numbers, not a general
result (§11).

**§4.1a, the keyframe-interval-decides-per-video-proxy signal. Folded into this
entry, 26 Aug 2026, because its only purpose was deciding P1.3.** One sentence
of the surviving principle -- measure a per-file property at load time rather
than assume it globally -- stays in §4.1a, because that pattern outlives the
proxy it was built for.

**The specific signal §4.1a proposed measuring does not predict what it was
chosen to predict.** It proposed classifying a video as cheap or expensive to
seek by its keyframe gap: dense gets a proxy, sparse falls back to (c). Measured
against the three real recordings, that ordering is wrong:

| Recording | Keyframe gap | Scattered-seek median |
|---|---|---|
| legacy (mpeg4) | 12 frames (0.6 s) -- densest | 25.2 ms -- cheapest |
| Zoom (h264) | 50 frames (2.0 s) -- sparsest of the three | 28.4 ms -- second cheapest |
| phone (h264, VFR) | 30 frames (~1.0 s) -- in between | **113.3 ms -- 4x either** |

The phone recording has a smaller keyframe gap than the Zoom recording --
denser, by this rule's own logic it should seek cheaper -- yet costs four times
as much per seek. Bitrate (20 Mbps against 0.86-2.98 Mbps for the other two)
looks like the better predictor on this evidence, though three files is not
enough to establish a replacement rule with confidence. **Anyone reviving a
proxy-like decision must measure seek cost directly, per file, rather than infer
it from keyframe spacing.** Full numbers and reasoning:
`%USERPROFILE%\Documents\gelem.measure\section11_table.md`.

---

## 11. Numbers, and which of them generalise

*Status 26 Aug 2026, measurement pass.* All rows below are **measured 26 Aug
2026 on Y B's machine** (Intel Core i9-14900K, 24C/32T, 128 GB RAM, Windows 11
Pro 10.0.26200, PyAV 18.1.0 / ffmpeg libs per `RUNLOG.md`), replacing the
estimate rows below with measured ones. Every timing is warm-cache: each
measurement unit ran twice per file, first discarded, second reported;
the first operation after every container open was separately discarded.
Full method, raw numbers, and two implementation bugs found and fixed during
this pass are in `%USERPROFILE%\Documents\gelem.measure\RUNLOG.md` and
`results_raw.json` (outside the repo; see §6.0). MediaPipe rows below remain
unmeasured estimates. The two structural questions (parallelism, tracking
divergence) still gate Phase 2; the per-frame-cost row and the two estimates
that follow from it do not gate anything currently -- see §6.0 for why.

**Pre-registered prediction, and what actually happened.** The prediction was
that the keyframe-only/full-decode ratio would be roughly `gap_frames / k`
with k between 2 and 5. The measured ratio does track `gap_frames`
closely and consistently across all eight files and three codec families --
**the mechanism is confirmed** -- but the fitted k is **1.1 to 1.35**, well
below the predicted range, meaning the achieved speedup sits much closer to
the theoretical ceiling (proportional to `gap_frames`) than expected. Two
causes were checked, per the method rules, before accepting this:

- **`skip_frame` not taking effect?** No. Keyframe counts recovered in the
  3-minute window match `docs/fixtures.md`'s stated values almost exactly
  (e.g. the phone file: 179 keyframes measured against 179 stated; the
  1 s-GOP pair: 181 measured against an expected 180; the 10 s-GOP pair: 19
  measured against an expected 18).
- **JPEG encoding dominating and swamping the decode difference?** Yes, in
  part, and this is the actual explanation for the low k. Measurement 3's
  display/analysis split shows resize+JPEG-encode is **59-73% of total
  per-frame cost** for the constant-frame-rate files (e.g. h265 1 s-GOP:
  22.4 ms display vs 6.0 ms analysis-only). Because this pass produces
  *identical output* on both the keyframe-only and full-decode sides (as the
  method requires, so the ratio isn't an artifact of unequal work), the
  shared encode cost is paid once per frame on both sides and compresses the
  ratio toward the frame-count ceiling. This is not a measurement error --
  it is the actual cost of the actual pipeline Gelem would run to build a
  proxy -- but it means k is a property of *this* pipeline (decode -> scale
  150 px -> JPEG q80), not of decode alone. A decode-only k would likely sit
  closer to the originally-guessed 2-5.

### Structural, generalises

| Quantity | Measured 26 Aug 2026 (Y B's machine) | Why it generalises |
|---|---|---|
| Keyframe-only vs full decode | ratio ≈ `gap_frames / k`. k = 1.15-1.35 (h264, n=5), 1.10-1.22 (hevc, n=2), 1.17 (mpeg4, n=1). Tracks `gap_frames` consistently across all 8 files. | Skipping inter-frame decode is inherent to how any decoder implements a predictive codec -- the mechanism holds on any machine. The *specific* k value does not generalise: it is bound to this pipeline's encode-dominated cost structure (see above), not to decode cost alone. |
| Seek fixed cost F0 (as a share of total scattered-seek time) | h264: F0 = 23.4 ms = **64%** of seek time on the dense (1 s-GOP) file, **15%** on the sparse (10 s-GOP) file. hevc: F0 = 14.5 ms, same 64%/15% split. | F0 is index lookup + decoder flush/reinit, a structural cost of any seek in any container/decoder. That it dominates for dense-keyframe files and shrinks proportionally for sparse ones is a property of the mechanism. The absolute F0 value (14-23 ms) is machine- and library-specific and will not generalise numerically. |
| Does PyAV release the GIL during decode? | **Yes, confirmed.** W > 1 at every worker count tested (never ≈1.0, which would mean no parallelism). Sub-linear: same-file W = 1.60 (2 workers) / 2.06 (4 workers, of ideal 4.0); different-files W = 1.36 / 1.72 (5-repeat median, tight range -- see `RUNLOG.md`). | Whether the C library releases the GIL during its C-level decode call is a fact about the binding, true on any machine. That scaling is sub-linear, and further reduced when threads decode different files/codecs simultaneously rather than one shared file, is a real, reproduced effect -- but the exact multipliers are this CPU's core count and scheduler, and will not generalise numerically. |
| Sorted vs scattered batching benefit | **0.5-2.9% faster sorted**, one file (h265 1 s-GOP) measured 7.4% *slower* sorted -- within noise. No meaningful benefit observed. | **Does not generalise, and cannot be read as "batching doesn't help."** The method's mandatory warm-cache protocol (discard first pass, report second) removes exactly the disk-I/O locality effect that sequential access is supposed to exploit. This result says the batching benefit is not visible *once the OS page cache is already warm* -- it is silent on the cold-cache / Google-Drive-Streaming case §4.3 is actually written for. A cold-cache variant would be needed to test the claim as stated. |
| MediaPipe parallelises across threads? | unknown -- not yet measured | Depends on whether the library releases the GIL; still gates Phase 2 (§6.0) |
| Tracking vs per-image output divergence | unknown -- not yet measured | A property of the model; still gates Phase 2 (§6.0) |

### Machine and file specific, must not become a constant

| File | Codec | Open cost (median, ms) | Scattered seek (median / IQR / max, ms) | Sorted seek (median, ms) | Sequential display (median / IQR, ms/frame) | Sequential analysis (median / IQR, ms/frame) | Proxy build cost (s/hour) | Proxy size (MB/hour) | T (s) | G\* (frames) |
|---|---|---|---|---|---|---|---|---|---|---|
| sid89_video.mp4 (Zoom) | h264 | 10.0 | 28.4 / [18.4, 31.7] / 57.3 | 27.8 | 20.64 / [18.89, 23.81] | 5.50 / [5.22, 6.42] | 23.8 | 3.95 | 0.41 | 1.4 |
| PID_031...mp4 (legacy) | mpeg4 | 3.2 | 25.2 / [23.8, 27.0] / 33.9 | 24.7 | 30.66 / [29.01, 33.18] | 6.65 / [6.42, 7.18] | 128.3 | 12.0 | 0.37 | 0.8 |
| VID_20260826...mp4 (phone) | h264 | 12.7 | 113.3 / [84.1, 147.5] / 203.5 | 112.7 | 27.32 / [25.72, 30.18] | 15.85 / [13.96, 18.82] | 75.4 | 9.40 | 1.65 | 4.1 |
| h264_cfr_gop1s.mp4 | h264 | 6.5 | 36.8 / [30.6, 40.8] / 67.2 | 36.9 | 22.09 / [20.33, 25.24] | 9.13 / [7.54, 12.01] | 53.4 | 7.06 | 0.54 | 1.7 |
| h264_cfr_gop10s.mp4 | h264 | 7.1 | 157.5 / [94.6, 220.4] / 347.8 | 153.0 | 21.81 / [20.21, 24.96] | 8.06 / [7.10, 10.50] | 6.3 | 0.72 | 2.29 | 7.2 |
| h265_cfr_gop1s.mp4 | hevc | 1.6 | 22.6 / [17.7, 26.7] / 34.2 | 24.3 | 22.40 / [20.30, 26.28] | 6.00 / [5.57, 6.81] | 47.0 | 7.06 | 0.33 | 1.0 |
| h265_cfr_gop10s.mp4 | hevc | 1.7 | 96.1 / [58.4, 130.9] / 186.6 | 95.1 | 22.54 / [20.32, 25.89] | 5.90 / [5.54, 6.54] | 5.2 | 0.72 | 1.40 | 4.3 |
| h264_vfr_longgop.mp4 | h264 (VFR) | 7.4 | 199.3 / [126.0, 266.0] / 386.7 | 195.5 | 21.23 / [19.89, 23.41] | 7.71 / [7.04, 9.34] | 4.3 | 0.47 | 2.90 | 9.4 |

Each depends on: keyframe density and codec (seek and build cost), frame
resolution and bitrate/entropy (sequential decode cost -- see the phone-file
note below), and this machine's single-thread decode speed and disk cache
state (all of it). None of it should become a setting default or a hardcoded
threshold without re-measuring on the machine and files it will actually run
against.

**Unexpected: the phone file, not the legacy file, is the outlier.** The
prediction was that the legacy 3200x1200 recording would differ sharply from
the 1080p files at full resolution. It doesn't -- its analysis-mode cost
(6.65 ms) is close to the Zoom file's (5.50 ms) despite ~1.85x the pixels.
The phone file's analysis-mode cost (15.85 ms) is the actual outlier, roughly
2.5x every other file including the higher-resolution legacy one. The phone
file is also the only one recorded at high bitrate (20 Mbps vs 0.86-2.98 Mbps
for the others) with real, irregular VFR -- entropy/bitrate, not resolution,
looks like the dominant driver of decode cost here, at least on this small
set. Worth another look before anything is built on the resolution-scales-cost
assumption.

**Full decode of 1 hour of video (decode-only: full-resolution decode plus RGB
conversion, no scaling, no encoding).** Corrected 26 Aug 2026. This lived in
the arithmetic table below as a ~2-6 min pre-registered estimate; moved here
because it is a measured, file-specific decode speed, not arithmetic, and does
not generalise across machines or codecs any more than the rest of this table
does. Same source as the decode-only full-pass figures in the P1.7a crossover
table (§4.1b): analysis-mode per-frame cost above, extrapolated to one hour at
each recording's own frame rate.

| Recording | Full decode of 1 h (decode-only) |
|---|---|
| sid89_video.mp4 (Zoom, 25 fps) | **8.2 min** |
| PID_031...mp4 (legacy, mpeg4, 20 fps) | **8.0 min** |
| VID_20260826...mp4 (phone, ~29.9 fps VFR) | **28.4 min** |

Measured on the three real recordings only -- not extrapolated to the five
synthetic fixtures, which is why this is a three-row table rather than a
column on the eight-row one above.

### Arithmetic, not measurement

| Quantity | Value |
|---|---|
| Common-case corpus | 20-120 videos, 10-120 min → up to ~240 h |
| Frames in 240 h @ 30 fps | ~26 M (never materialise all of these) |
| Realistic frame table (Y B's study) | ~530 k rows, ~300 MB |
| Thumbnail, 150 px JPEG | **Measured 26 Aug 2026: 2.0-2.6 KB** across the three real recordings (`section11_table.md`'s `avg_jpeg_bytes`), not ~6 KB. These recordings are visually soft, which is most of the gap. This number feeds the thumbnail cache size setting below -- **the setting's default assumption was too pessimistic by roughly a factor of three.** |
| Proxy disk, 240 h corpus, if entirely Zoom-like (sid89 density) | 0.93 GB |
| Proxy disk, 240 h corpus, if entirely legacy-like (PID_031 density) | **2.81 GB** |
| Proxy disk, 240 h corpus, if entirely phone-like (VID_20260826 density) | **2.20 GB** |
| Proxy disk, 240 h corpus, if entirely h264 1 s-GOP density | 1.65 GB |
| Proxy disk, 240 h corpus, if entirely h264/h265 10 s-GOP density | 0.17 GB |
| Proxy disk, 240 h corpus, if entirely VFR-longgop density | 0.11 GB |
| K (build cost per hour ÷ scattered seek time), sid89 | 837 |
| K, legacy (PID_031) | 5093 |
| K, phone | 665 |
| K, h264 1 s-GOP | 1451 |
| K, h264 10 s-GOP | 40 |
| K, h265 1 s-GOP | 2079 |
| K, h265 10 s-GOP | 54 |
| K, VFR-longgop | 22 |
| MediaPipe per frame, CPU | ~25 ms. **Unverified estimate; no current decision depends on it.** *(Status changed 26 Aug 2026.)* Was operational-only -- it would have told Y B whether his driving study (§2) was a 4-hour or 40-hour run -- but that reason is gone too: the blendshapes for that study are already extracted. Kept only as a placeholder until Phase 2 measures it for real, against whatever MediaPipe version Phase 2 actually uses (§6.0). |
| Y B's study, every frame, 1 thread | ~4.4 h. **Unverified estimate; no current decision depends on it, same status as the row above** -- the study it would have planned is already done. Also, independent of that: **this omits decode cost entirely.** Decode in analysis mode measured 5.5-15.9 ms/frame on the real recordings (table above), so this row would understate total per-frame cost by roughly 20 to 60 percent if it were still being used for planning. |
| Same, with tracking + 6 workers | ~30-45 min. **Unverified estimate; no current decision depends on it, same status as the two rows above.** Also, independent of that: it **assumes near-linear thread scaling across 6 workers**. PyAV, the one comparable measurement available, showed 2.06x at 4 workers, not 4x. That contradiction is noted, not corrected, since it's MediaPipe threads being assumed here, not PyAV's, and MediaPipe's thread scaling has not been measured. |
| Worker count | -- (setting, default low) |
| Thumbnail cache size | 500 MB (setting; see the corrected thumbnail-size row above for why its sizing assumption was too pessimistic) |

K is "how many thumbnails must be drawn from one hour of video before a
prepared cache repays its own construction." For every real recording, K is
in the hundreds to low thousands -- a proxy would need to be reused hundreds
to thousands of times per hour of source video before it pays for itself.
Under the T-driven verdict below this doesn't matter (P1.3 isn't being built
regardless), but if that verdict is revisited later, K is the number that
says the payback bar is high.

The real-corpus proxy-disk figure is not a single number: Y B's actual 240 h
corpus is presumably a mix of these three densities, not uniformly one type,
and its composition wasn't measured. **Two of the three real recording types
(legacy and phone) individually exceed the 2 GB gate on their own; only the
Zoom-density type does not.** A realistic mixed corpus is not comfortably
under the gate -- see the verdict below.

**The last three rows are unverified estimates that no current decision
depends on.** *(Changed 26 Aug 2026.)* They used to be operational planning
figures for Y B's machine -- how long his driving study's extraction would
take. That question is closed: the study's blendshapes are already extracted,
so there is nothing left for these numbers to plan. They stay in this table
only as a placeholder pending Phase 2 (§6.0), and must be re-measured against
whatever MediaPipe version Phase 2 actually uses before being trusted for
anything -- P2.2 makes model version part of the result-cache identity, so an
estimate taken against one version is not evidence about another.

---

### Verdict, stated mechanically

**Rule:** T < 1 s -> not built, comfortable. T in [1, 6] s -> not built,
accepted trade-off. T > 6 s -> reconsider (more workers / reduced-res decode
for mpeg4 / prefetch). Two gates defer P1.3 regardless of T: build cost > 180
s/hour, or 240 h-corpus proxy disk > 2 GB.

| Real recording | T | Band | Build-cost gate (>180 s/h)? | Proxy-disk gate (>2 GB @240h)? |
|---|---|---|---|---|
| sid89_video.mp4 (Zoom) | 0.41 s | **under 1 s** | No (23.8 s/h) | No (0.93 GB) |
| PID_031...mp4 (legacy, mpeg4) | 0.37 s | **under 1 s** | No (128.3 s/h) | **Yes (2.81 GB)** |
| VID_20260826...mp4 (phone) | 1.65 s | **1-6 s** | No (75.4 s/h) | **Yes (2.20 GB)** |

**P1.3 is not built.** All three real recordings land in the "not built"
bands on T alone (two comfortably under 1 s, the phone recording in the
accepted 1-6 s trade-off band). The disk gate independently confirms this for
two of the three recording types, and is ambiguous rather than clearly clear
for a realistic mixed corpus (see above) -- it does not change the verdict,
since T already settles it, but it removes any temptation to treat "T is
fine" as the last word if T is revisited later on a different machine. Full
record, including the reasoning for the rejection, is in §10.

No band is close enough to a boundary to flag as fragile except the phone
file's T = 1.65 s, which sits inside the 1-6 s band regardless of the exact
value -- moving W (the parallelism speedup used in T's denominator) between
the two measured values in this pass (2.06 vs the earlier noisy 3.31) moved T
between 1.03 s and 1.65 s without changing the band. The verdict is robust to
that noise; a specific point estimate of T for the phone file is not.

---

### Contradictions with the plan, found during this pass

**P1.7a (trial-thumbnail batch job, §4.1b) was contradicted, sharply, for all
three real recordings, and has been rewritten (§4.1b, §6.2).** The crossover --
how many trials per hour of video before a full sequential decode pass beats
sorted per-trial seeking -- comes out at:

| File | Full sequential pass (s/hour, decode-only) | Sorted seek, seek-only component (ms) | Crossover N\* (trials/hour) |
|---|---|---|---|
| sid89_video.mp4 | 494.7 | 17.55 | **~28,200** |
| PID_031...mp4 (legacy) | 478.7 | 8.16 | **~58,600** |
| VID_20260826...mp4 (phone) | 1705.8 | 101.68 | **~16,800** |

(Method: sorted-seek total time was split into its seek-only and JPEG-encode
components using measurement 2's own per-operation fields, since a
sequential pass only has to pay the encode cost at the N representative
frames it actually captures, not at every frame it decodes through. N\* =
full-pass decode-only cost ÷ (sorted-seek time − encode-only time). Full
detail in `analyze.py`.)

No real study has 16,000-59,000 trials in one hour of video: **for any
realistic trial count, sorted per-trial seeking beats a full sequential
decode.** Stated at specific, checkable densities rather than dramatised: at
3,000 trials per hour of video, seeking is about five times cheaper on the
phone recording (the least favourable of the three); at 300 per hour, about
fifty times. A full sequential pass would only become the right choice again
above roughly seventeen thousand trials in a single hour of recording -- a
trial every fifth of a second. §4.1b and P1.7a (§6.2) have been rewritten to
seek to each representative frame instead.

**§4.1a's dense-keyframes-get-a-proxy rule does not track which files
actually have expensive seeks.** The rule sorts files by keyframe gap and
expects seek cost to follow. It doesn't, for the three real recordings:

| File | Keyframe gap | Scattered seek median |
|---|---|---|
| PID_031 (legacy) | 12 frames (0.6 s) -- densest | 25.2 ms -- cheapest |
| sid89 (Zoom) | 50 frames (2.0 s) -- sparsest of the three | 28.4 ms -- second cheapest |
| VID_20260826 (phone) | 30 frames (~1.0 s) -- in between | 113.3 ms -- **4x more expensive than either** |

The phone file has a *smaller* keyframe gap than the Zoom recording (denser,
by the rule's own logic it should seek cheaper) yet costs four times as much
per seek. Bitrate looks like the better predictor on this evidence: the phone
file is encoded at 20 Mbps against 0.86-2.98 Mbps for the other two, meaning
substantially more entropy-coded data to parse per frame regardless of how
close the nearest keyframe is. **§4.1a's binary dense/sparse-by-keyframe-gap
rule would classify the phone file as cheap to seek (it's reasonably dense)
when it is in fact the most expensive file in the entire eight-file set.**
This doesn't necessarily mean measuring a per-video property at load time is
the wrong general approach -- it already avoids hardcoding a global
assumption, and that principle survives in §4.1a -- but the specific signal
it proposed measuring (keyframe interval alone) missed the dominant cost
driver on this evidence. Three real files is not enough to replace "keyframe
gap" with "bitrate" as the decision signal with confidence; it is enough to
say gap alone is not sufficient, and that this wasn't tested before now. Full
record: §10.
