# Gelem

Desktop visual data explorer for psychology research, in Python with PySide6/Qt.
Researchers load collections of videos, images, and metadata, then browse, filter,
group, and analyse them without writing code.

**Goal:** a tool undergraduates can use with very little training, to do basic
processing and analysis of video data before they know Python or machine learning.

Y B is the sole developer. This was previously a three-student project; that phase
is over. Claude now does the implementation work.

---

## Documentation

| File | Answers |
|---|---|
| `CLAUDE.md` (this file) | What rules must never be broken, and how is work handed back? |
| `docs/architecture.md` | What is Gelem and how do its parts relate? |
| `docs/media_architecture.md` | How do we handle video and scale? |
| `operators/CLAUDE.md` | How do I write an operator? |
| `docs/roadmap.md` | What are we building, why, and when is it usable? |

**Filenames are exact.** `operators/CLAUDE.md` must have that name or Claude Code
will not load it when working in that directory, and every operator rule will be
silently absent. `docs/archive/` holds superseded material -- historical only,
never guidance.

Each answers exactly one question. **Each subject has exactly one authoritative
document.** Another document may name a concept and point at the authority, but
must never restate the definition -- a restated definition eventually contradicts
the original.

**`docs/media_architecture.md` supersedes anything about media handling elsewhere.**
It is the current design and it changes how frames, clips, and thumbnails work.

---

## How to read the rules below

This file previously stated rules in the imperative without saying which were true
and which were goals. Several were contradicted by the code, and two were wrong as
written. Every rule now carries a status. **A rule without a status tag is a
documentation bug -- report it.**

**`[NOW]`** -- true today and guarded by a test. Breaking it is a defect. If a test
does not yet exist for a `[NOW]` rule, writing it is a Phase 0 task.

**`[TARGET -> item]`** -- not true yet. Must be true by the named work item. Do not
write new code that assumes it already holds, and do not add new violations.

**`[MIGRATING]`** -- known violations exist and are listed by file and line. The
list is closed: fix the listed sites, never add to them. A guardrail test should
fail on any site not on the list.

**Work item IDs (`P0.2`, `P1.6`, ...) are stable names, not an order.** Phase
membership can change; the ID does not. The order of work is in
`docs/media_architecture.md` §6, and the criterion for Phase 0 membership is
**dependency** -- other work is built directly on top of it -- not severity.

---

## The seven components

`Dataset` (owns all data) · `QueryEngine` (filters/sorts) · `ArtifactStore`
(derived-image cache) · `ColumnTypeRegistry` (display rules) · `OperatorRegistry`
(runs analyses) · `AppController` (wiring) · `UI` (widgets)

**`[MIGRATING]`** Components talk through defined public methods only and no
component reaches into another's internals. This is the goal, not the present
state -- the violation sites are listed under "UI" below. Stating it untagged was
itself an instance of the problem this file's status tags exist to fix.
*(Re-verified 26 Aug 2026: this rule carries no violation list of its own --
it points at the three `[MIGRATING]` rules under "UI" below, which are each
re-verified there.)*

---

## Hard rules

### Data ownership

- **`[NOW]`** Only `Dataset` may modify a **stored** table. A worker may freely
  build and mutate a DataFrame it created itself; that is not a stored table until
  `Dataset` accepts it.
- **`[NOW]`** `get_table()` returns a copy. Modifying it does not modify the stored
  table.
- **`[NOW]`** Reading one row must not copy the whole table. `get_row()` reads
  through a per-table row-id index and returns just that row. Made true by P0.2a.
  Tests: `tests/test_dataset_access_paths.py`
  (`test_get_row_does_not_call_get_table`,
  `test_run_create_columns_does_not_copy_table_per_row`).
- **`[NOW]`** Operators never access `Dataset` or `AppController`. They receive the
  data they need as arguments and return results.
- **`[MIGRATING]`** Every structural operation is recorded in
  `provenance.record()`. Load, merge, aggregate and table creation are recorded.
  **Operator runs are not**, so provenance is not yet sufficient to reproduce an
  analysis. See `docs/architecture.md` §8. Also: `save()` writes `provenance.json`
  before recording the save, so the save action is absent from the file it just
  wrote. *(Re-verified 26 Aug 2026: every `provenance.record()` call still lives
  in `models/dataset.py` -- none in `operators/` or `controller.py` -- and
  `save()` at `models/dataset.py:790-793` still writes the file before recording
  the save.)*
- **`[NOW]`** Paths **inside the project folder** are stored relative to it, so
  projects stay portable. Paths outside it stay absolute unless the user
  explicitly imports or copies the file into the project. The rewriting parses
  each media cell as an address and moves only the path portion -- the fragment
  (`#f=`, `#t=`, `#r=`, stream selector) is preserved exactly. P0.2c made it
  address-aware. `docs/media_architecture.md` §3.5 is the authority for the
  address-survival rule; tests: `tests/test_dataset_address_paths.py`.

### Row identity and lineage

- **`[MIGRATING]`** `row_id` is an opaque handle so the UI can name a row. It
  carries no meaning: do not parse it, sort by it, or infer anything from it.
  Known violation: `Dataset.load()` restores `_id_counter` with
  `int(df["row_id"].astype(int).max())`, which parses it. The counter should be
  stored in the project rather than recovered from the ids. *(Re-verified 26 Aug
  2026: still the only site -- `models/dataset.py:842`. No other file parses,
  sorts by, or infers meaning from `row_id`.)*
- **`[NOW]`** `row_id` is unique **within a table**. It is not unique across
  tables: `create_table_from_rows()` deliberately keeps ids when copying rows.
- **`[NOW]`** Row ids **are preserved by project save and load** -- `save()` writes
  them to Parquet and `load()` reads them straight back, so a saved project reopens
  with the same ids and a saved reference stays valid. They are **not** guaranteed
  across **re-import or dataset reconstruction**: `load_folder()` and
  `load_csv_as_primary()` reset the counter and mint new ids, so ids from a
  previous import mean nothing after one. Earlier documentation claimed row ids
  were globally unique and never reused, which was too strong; a previous revision
  then said they were "not stable across a reload", which was too weak and wrong
  about save/load.
- **`[TARGET -> P0.2b]`** Any public reference to a row -- signal, artifact request,
  controller method -- identifies it as `(table_name, row_id)`. `row_updated` and
  `thumbnail_ready` currently carry a bare `row_id`, so the UI cannot tell which
  table changed.
- **`[NOW]`** **Lineage is carried by ordinary data columns, not by surrogate
  pointers.** When an operator turns one row into many -- a video into segments, a
  segment into frames -- it carries the source's identifying columns down and adds
  its own index. `participant_id, trial_id, frame_index` is the model, exactly as
  it would be in R. Gelem does not maintain a parallel `source_row_id` graph.
- **`[TARGET -> P1.6, P1.7]`** Segment and frame operators must emit the columns
  that make this work: a segment index, a frame index, `time_within_segment`, and
  everything carried down from the source.
- **`[TARGET -> P1.8]`** What gets carried down is defined by two schema
  properties, `role` and `carry_to_children` -- **not by role alone**. Identifiers
  and indices are always carried; everything else defaults to carried, because a
  trial-level covariate such as `reaction_time` is a measurement *and* is required
  on every frame row. Full rule in `docs/architecture.md` §4.2.

### Threading

- **`[NOW]`** Tables owned by `Dataset` are mutated on the main thread only.
- **`[NOW]`** Background workers never call Qt and never create a `QPixmap`.
- **`[MIGRATING]`** Workers communicate only by placing results into the
  controller's result queues. They do not read controller or component state.
  Known violations: `controller.py` `_on_operator_setup_error` and
  `_on_operator_row_errors` both call `self._op_registry.get()` from the worker
  thread to look up a label. The label should travel with the error instead.
  *(Re-verified 26 Aug 2026: both are still called from
  `OperatorRegistry._run_create_columns_worker`, which runs on a
  `threading.Thread`; every other worker-bound controller callback
  -- `_on_item_complete`, `_on_progress`, `_on_create_columns_complete`,
  `_on_create_table_complete`, `_on_create_display_complete` -- only appends to a
  queue and reads no state, so the list is still closed at these two.)*
- **`[NOW]`** Worker callbacks **are** bound controller methods. This is correct
  and deliberate. An earlier version of this file said workers "never touch the
  controller," which forbade the actual design.
- **`[TARGET -> P0.2b]`** Draining the result queues is bounded -- a fixed time or
  item budget per timer tick. `_drain_queues()` currently empties every queue in
  one tick, which stalls the main thread during a large run, and uses `list.pop(0)`
  which is linear in queue length.

### UI

- **`[NOW]`** UI files never import pandas, PIL, numpy, mediapipe, or cv2.
- **`[MIGRATING]`** UI never reads a DataFrame. Known violation:
  `ui/main_window.py:415` (`columns=list(df.columns)`). *(Re-verified 24 Aug
  2026: this used to be two sites; the second one merged into the citation
  below and no longer separately reads a DataFrame.)* *(Re-verified 27 Aug 2026
  (P0.4): still the only `.columns`/`.iloc`/`.loc`/`DataFrame` site under `ui/`
  outside `ui/fake_controller.py`. Line moved from 401 to 415 as P0.4 added
  code above it; the site itself is unchanged and still belongs to P1.13.)*
- **`[MIGRATING]`** UI never touches private controller attributes. Known
  violations: `ui/main_window.py:290-292` (`_op_registry`),
  `ui/main_window.py:410-411` (`_dataset`, `_active_table`). Public equivalents
  already exist -- `get_all_row_ids()`, `get_column_type()`, `get_operator()` --
  so these are unfinished migrations, not missing API. *(Re-verified 24 Aug
  2026: `ui/filter_panel.py:199` no longer reaches into `_registry` -- it now
  calls the public `controller.get_column_type()` -- so that citation is
  removed. The `_op_registry` and `_dataset`/`_active_table` sites also
  shrank from three occurrences each to one.)* *(Re-verified 27 Aug 2026
  (P0.4): still one occurrence each, unchanged in content. Line numbers moved
  (`_op_registry` 276-278 -> 290-292, `_dataset`/`_active_table` 396-397 ->
  410-411) because P0.4 added code above them. These two sites are on the
  closed allowlist in `tests/test_ui_private_access.py`, which fails on any
  other foreign private read under `ui/`.)*
- **`[MIGRATING]`** No widget reads another component's private state. Known
  violation: `ui/main_window.py:448` reads `operator._group_by` back off the
  operator instance, where `base.py:331-334` stores it with `setattr`. This is
  the last site; P1.12 removes it when parameters stop being stored on the
  operator instance. *(P0.4, 27 Aug 2026: the three `ui/main_window.py` sites
  that read `GalleryWidget._row_ids` are gone -- the controller now owns the
  ordered result and the gallery is given an index range into it, holding no
  row ids. `GalleryWidget` also used to read `TileWidget._tile` at two sites;
  an earlier revision of this rule wrongly claimed "no other widget reads
  another widget's private attribute". That was false and is now fixed:
  `TileWidget` has a public `get_row_ids()`. The guard is
  `tests/test_ui_private_access.py`, an AST check with a closed allowlist of
  `_op_registry`, `_dataset`, `_active_table` and `_group_by`.)*
- **`[NOW]`** Renderers may import PIL and cv2 -- they are not UI files. Renderers
  never import from `ui/`.
- **`[NOW]`** Shared display components go in `shared_widgets/`, not inside `ui/`.
- **`[NOW]`** `None` and `[]` are different. For visible columns, `None` means "no
  preference set" and `[]` means "the user chose zero columns".
  `GalleryWidget._relayout()` (`ui/gallery_widget.py:453-463`) now distinguishes
  the two explicitly and correctly, and `AppController.get_effective_visible_columns()`
  /`has_visible_columns_preference()` carry the same distinction on the
  controller side. Guarded by
  `tests/test_visible_row_order.py::test_visible_columns_none_versus_empty`
  (added in P0.4, 27 Aug 2026). *(Re-verified 24 Aug 2026: this was
  `[MIGRATING]` with a known violation at `ui/gallery_widget.py:390`; the
  violation is fixed.)*

### Media

- **`[TARGET -> P1.2]`** **Only the media resolver decodes *source* media.** No
  `cv2.VideoCapture`, `av.open`, or `Image.open` **of a user's media file**
  anywhere else. Three places do today: `BaseOperator.load_image`,
  `column_types/renderers.py`, and `ArtifactStore._generate_thumbnails`.
- **`[NOW]`** **Reading and writing Gelem's own derived artifacts is a different
  operation and is not covered by that rule.** `ArtifactStore` reads back the JPEGs
  it wrote (`get_pixmap` calls `Image.open`), and must keep being able to. A
  guardrail that bans `Image.open` outright would either forbid legitimate cache
  I/O or push cache internals into the resolver. `[TARGET -> P0.5]` The two are
  separated by a narrow `ArtifactCodec`, so the guardrail can name the boundary
  precisely: source decoding is the resolver's, artifact encoding is
  `ArtifactCodec`'s, and nothing else opens an image at all.
- **`[TARGET -> P1.10]`** Native playback is the explicit exception. `QMediaPlayer`
  receives a file path and a time range directly. It shares the address **parser**
  with the resolver but not the decoding path.
- **`[TARGET -> P0.5]`** No media is opened or decoded during a paint. A cache miss
  returns a placeholder immediately and queues a request. Today
  `_render_video()` runs `cv2.VideoCapture` on the main thread on every paint of
  every video tile, and consults no cache at all.
- **`[TARGET -> P0.5]`** Derived images are identified by **media address**, not by
  row. See `docs/media_architecture.md` §4.5. The row, table and column identify
  the UI subscriber waiting for the picture, never the picture itself.
- **`[NOW]`** `media/media_address.py` gives the address grammar exact,
  guarded-by-test meaning: escaping, canonical form, frame/time-point and
  range semantics, region validation and pixel arithmetic, and which
  malformed or degenerate values are refused. Each rule is `docs/media_architecture.md`
  §3.6's authority; this file does not restate the decisions, only points at
  them. Tests: `tests/test_media_address.py`.
- **`[TARGET -> P1.2]`** What §3.6 settled but the resolver has not yet built:
  orientation is applied before anything else sees a frame (decision 6), the
  addressed stream is what actually gets decoded (decision 7), and a frame
  ordinal is resolved against a file's real per-frame timings, never a nominal
  frame rate (decision 8). `MediaAddress` only carries the address; nothing
  today decodes one.

### Generality

- **`[NOW]`** Watch for study-specific vocabulary in general components. A
  hardcoded 300-1500 ms window, a seven-emotion assumption, or a
  blendshape-specific branch inside a generic component is a leak. Before building
  a feature, name the parameter that makes it general.
- **`[TARGET -> P0.5]`** **A number that does not generalise across machines
  or datasets must become a setting or a runtime measurement -- never a constant in
  the code.** Worker count and cache size are both of this kind. Measuring one on
  Y B's machine tells you the shape of the common case, not a value to hardcode.
  Not true today: `DEFAULT_CACHE_MAX_BYTES` is a hardcoded 500 MB, and there is no
  worker pool to have a count. *(Keyframe interval was the third example here
  until 26 Aug 2026: it was measured per video specifically to decide whether to
  build a whole-video proxy layer, and that proxy is rejected -- see
  `docs/media_architecture.md` §10.)* This rule covers two different things --
  a number that varies **by machine** needs a setting (worker count, cache
  size, both above); a number that varies **by file or dataset** needs a
  runtime measurement instead, which is a different mechanism. Per-file seek
  cost was considered as an example, since the measurement pass found
  keyframe interval doesn't predict it -- but seek cost isn't measured
  anywhere in the current plan either; it only appears in
  `docs/media_architecture.md` §10 as what a future revival of proxy-like work
  would need to measure, which isn't a live case today. **The next feature
  that needs a runtime-measured per-file property should cite itself here**
  rather than this rule citing a hypothetical.
  **`[TARGET -> P1.2]`** The first real case, cited 26 Aug 2026: resolving
  `#f=N` against a variable-frame-rate file (decision 8,
  `docs/media_architecture.md` §3.6) needs a per-file index of frame
  presentation times, built on first use, because no nominal frame rate is
  trustworthy on VFR source. Not built yet -- `media/media_address.py`'s
  `select_frame()` takes frame times as a plain argument and never measures
  them itself; that measurement is the resolver's, in P1.2.

### Long-running work

- **`[TARGET -> P1.12]`** Long runs are cancellable, keeping partial results. **No
  cancellation mechanism exists today** -- there is no cancellation token and no
  check anywhere in the operator loop, so a started run always runs to completion.
  Any current statement that runs "can be cancelled" describes the target.
- **`[TARGET -> P2.2]`** Resumability is narrower than cancellability and must be
  stated per mode; see `operators/CLAUDE.md`. A generator gives progressive output
  and a cancellation point. It does not by itself give resumability.

### Contract mirroring

- **`[NOW]`** When a public method is added to `AppController`,
  `ui/fake_controller.py` must mirror it immediately.
  `tests/test_fake_controller_contract.py` guards this, and drift here is a
  recurring problem.

---

## Testing

Guardrail tests enforce the architecture: `tests/test_architecture_imports.py`,
`test_controller_async_contracts.py`, `test_operator_registry_boundaries.py`,
`test_fake_controller_contract.py`.

A failing guardrail test almost always means something reached across a component
boundary. **Fix the violation, never work around the test.**

**Every architectural rule must become a failing test, not a sentence in a
document.** When a design decision is made, write its test at that moment. Claude
reliably obeys a test that fails and reliably drifts from prose. This document's
`[NOW]` tags are a promise that a test exists; a `[NOW]` rule without one is
itself a defect.

**Test at component seams, not internals.** Asserting on contracts means internals
can be rewritten freely. Asserting on internals makes every refactor a
test-rewriting slog, and refactoring then stops happening. Static AST checks are
appropriate for forbidden imports and private-attribute access. Everything else
goes through public seams.

Run with `python -m pytest`. Environment is a `(gelem)` virtualenv activated via
`.\setup.ps1`, on Windows PowerShell. The repo sits inside a Google Drive Streaming
path, so allow a moment after branch checkouts before running tests.

**The baseline must be green before Phase 0 starts.** `tests/test_renderer.py`
used to have two collection errors from functions named `test_*` that were really
manual checks (`test_thumbnail`, `test_detail`, each requiring positional
arguments pytest tried to treat as fixtures). Fixed 24 Aug 2026 by renaming them
to `check_thumbnail` / `check_detail` -- the file is a standalone manual-check
script, not a pytest module, so nothing inside it should match `test_*`.

**Verified green 26 Aug 2026 (P0.1).** Deleting the stray untracked
`test_images/boxtest.png` (a 21st image with no matching `metadata.csv` row) fixed
the three `tests/test_dataset.py` failures this section used to describe; run
individually, `test_add_computed_column_correct_values`, `test_apply_sort`, and
`test_apply_grouped_all_rows_accounted` all pass. `test_images/` currently holds
exactly 20 `.jpg` files against 20 `metadata.csv` rows, so nothing stray remains.

**Verified green 27 Aug 2026 (P0.4).** `python -m pytest` collects 174 items with
zero collection errors and zero failures. (The 91 this section reported at P0.1
grew as P0.2a, P0.2c, P0.3 and now P0.4 each added their own test files.)

---

## Working rules for Claude

- **Propose refactors; do not patch.** Late in a long session there is a pull
  toward patching because refactoring feels disruptive. Resist it, and say
  explicitly when a clean fix needs to touch more than the immediate task.
- **Existing code is not a constraint.** Y B has authorised replacing it. If a
  design looks wrong because of how something is currently built, say so rather
  than designing around it.
- **Check a rule's status before relying on it.** A `[TARGET]` rule describes the
  future. Writing code that assumes it holds today produces silent breakage.
- **Confirm which component owns a feature before writing code.**
- **Flag anything crossing the main-thread / worker boundary.**
- **Write tests from the spec, not from the implementation.** Tests written
  afterwards tend to encode the implementation's quirks and pass vacuously.
- **Ask rather than assume** when a design document is ambiguous.
- **Prefer explicit, readable Python over terse idioms.** Y B's background is
  JavaScript and C. Comment each block before the block.
- **Use `--` rather than em dashes** in prose and comments.
- **Do not front-load the whole codebase.** `gelem_codebase_for_claude.txt` is a
  point-in-time review export, not a source of truth. It goes stale immediately.
  Read the specific files a question touches.

**For Y B:** ask "what would you rip out if you could?" at intervals. Claude will
not volunteer it.

---

## Starting a work item

**Before your first edit, check the current branch.** Each work item has its
own branch, created by Y B with `Start-Item <branch-name>`.

- If the branch is already the work item's branch, proceed.
- **If the branch is `main`, stop.** Do not edit a single file. Tell Y B to run
  `Start-Item <branch-name>` and wait.

**One work item per session, per branch.** The work-item prompt names the
branch it expects; if `git rev-parse --abbrev-ref HEAD` does not print that
name, something is wrong -- ask rather than guessing.

---

## Finishing a work item

**Claude Code never commits, merges, or pushes.** Y B does all of that himself, one command at a time. Claude Code's job ends at a working tree and a generated diff. If a step seems to need a commit, say so and stop.

**One work item per session, per branch.** Stop at the boundary. Do not
roll into the next item because it looks small.

**Never end with just "done".** Y B is working across three tools and will not
remember this procedure. Claude Code is responsible for reminding him. End every
completed work item with exactly this block, filled in:

```
## Work item complete: <ID>

**What changed**
- three bullets, maximum

**Verify it yourself**
- the **actual** final lines of the `python -m pytest` run, pasted verbatim
  (the collected count and the pass/fail summary), not a prediction of them --
  a reviewer must be able to tell what Claude observed from what it expects
- the exact commands for Y B to re-run, and what a pass looks like
- if there is something to check by eye in the app, an exact click list:
  which screen, which controls, and for each step what a pass looks like --
  cover every path that no automated test covers

**Diff**
git diff main > docs/review/<id>.diff
(state whether this has already been run)

**Review**
- <needed / not needed>, per the table below
- if needed: the exact bounded question to paste, and which of the four
  documents to attach
```

### Which items get an external review

Review a change that **establishes a contract other work is built on**. Skip
mechanical ones -- the tests are the check there.

| Item | Review? | Why |
|---|---|---|
| P0.1 | no | mechanical; the tests are the check |
| P0.2a | **yes** | Dataset access paths -- every operator run goes through them |
| P0.2c | no | a save/load path fix behind an already-reviewed module |
| P0.2b | **yes** | result delivery -- every progressive update goes through it |
| P0.3 | **yes, narrowly** | review the §3.6 semantic decisions, not the parser code |
| P0.4 | **yes** | well specified but not small; establishes the result-ownership contract P0.5 builds on |
| P0.5 | **yes** | segment thumbnails and every tile attach here |
| P1.8, P1.12 | **yes** | schema roles and the operator contract are contracts |
| everything else | judgement | apply the same test |

### How to route a review

1. **Claude Code, every item.** Run `/code-review` on the working diff before
   committing. Catches mechanical slips: a test that asserts nothing, a signature
   that drifted from the document, a swallowed exception.
2. **Claude Desktop**, for items marked yes. It has the repo mounted and holds the
   design reasoning, so it checks code against intent. It cannot run the tests.
3. **ChatGPT**, for the same items. Its value is independence -- it does not share
   Desktop's assumptions, which is how the `carry_to_children` defect was found.
   Give it the diff plus `CLAUDE.md` and the work-item text. **Never give it
   `gelem_codebase_for_claude.txt`.**

### Ask a bounded question

**Do not ask "review this, any problems?"** An open-ended request reliably produces
problems, at any level of quality, and never terminates. Six rounds of that on the
documents ended with two cross-reference typos. Code has far more surface for it
than prose.

Ask something answerable instead:

- "Does this preserve the documented behaviour of `get_table()` returning a copy?
  Check only that."
- "Do these tests fail if the rule they guard is violated? Describe a violation
  each one would miss."
- "§3.6 lists thirteen decisions. Which does this parser leave unspecified?"

The third form is the most valuable: it checks coverage against a list already
agreed, rather than inviting a new list.

**Escalate rather than comply.** If a review says the code contradicts a document,
that is not automatically the code's fault -- four rules in this file were
mislabelled and one was actively wrong. Say which you think it is; never bend the
code to match a document without saying so.

---

## Known defects

Verified against the code on 4 Aug 2026; re-verified against `main` on 24 Aug
2026 (P0.1). Each should become a failing test before or as it is fixed.

**Wrong output, not just slow:**

- A row with several media columns shares one cached image. `ArtifactStore` keys on
  `(row_id, artifact_type)` and writes `{row_id}_{artifact_type}.jpg`, so the entry
  collides in the index *and* on disk. `ImageTile` correctly passes `column_name`
  in the render context and `_render_image` ignores it. **A `GridTile` showing
  `full_path` and `avatar_path` displays the source photograph in both tiles.**
  Fixed by P0.5.
- `BlendshapeAvatarOperator` declares tag `avatar_path` and `PlotOperator` declares
  `plot_image`. Neither is a registered type, so `register_by_tag` raises,
  `controller.py:471-472` swallows it as a printed warning, and those columns
  render as "Unknown column".
- `ColumnTypeRegistry.infer_type()` mistags any column whose values end in a media
  extension. A column of `.mp4` filenames becomes `media_path`. This fires
  immediately in the new pipeline.
- `ArtifactStore.load_index()` replaces the disk index but leaves the in-memory
  image cache populated, so opening a second project can show the first project's
  pictures.
- `Dataset.load()` does not clear `ColumnTypeRegistry`, so column types from the
  previous project persist.

**Non-functional at target scale:**

- `Dataset.get_row()` calls `get_table()`, which copies the entire table.
  `AppController.run_create_columns()` calls `get_row()` once per selected row, so
  starting an operator over a 530k-row table performs 530k full-table copies.
- `Dataset.update_row()` scans the whole `row_id` column per result.
- `ArtifactStore.request_thumbnail()` spawns one raw `threading.Thread` per call.
- Controller result queues are lists drained with `pop(0)`, unbounded per tick.

**Dead or inconsistent:**

- `operators/thumbnail.py` is dead code. Its docstring says
  `ArtifactStore.request_thumbnail()` calls it; `_generate_thumbnails()`
  reimplements the whole thing inline and never touches it. Delete it and make a
  real operator the reference.
- `operators_config.yaml` claims to control which operators are enabled. `main.py`
  registers them manually and never reads the file. `StatsOperator` is registered
  in code and absent from the YAML.
- `operators/base.py` documents a `plot_html` result key; `ResultsPanel` and
  `PlotAdvancedOperator` use `html_path`.
- `Dataset.load_folder()` never registers `file_name` in the registry, unlike the
  CSV import paths.
- `_id_counter` assumes `row_id` parses as an int. No stale-file cleanup on
  re-save.
- Many module docstrings still assign files to Student A, B, or C. Remove as those
  files are touched. *(Done in `tests/test_renderer.py` 24 Aug 2026, the only
  file touched this session.)*

*(Re-verified 24 Aug 2026: the `self.output_dir` vs `self._output_dir` bullet
that used to be here is removed -- `operators/CLAUDE.md` no longer makes that
claim; it already documents `self._output_dir` and notes `self.output_dir`
"never existed". The other bullets in this section are still true as written.)*
