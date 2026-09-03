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
| `docs/known_defects.md` | What is currently broken, and which item fixes it? |

**Filenames are exact.** `operators/CLAUDE.md` must have that name or Claude Code
will not load it when working in that directory, and every operator rule will be
silently absent. `docs/archive/` holds superseded material -- historical only,
never guidance. `docs/archive/rule_verification_log.md` holds the dated history of
how the rules below reached their current wording; it is a record, not guidance,
and nothing in it overrides this file.

Each answers exactly one question. **Each subject has exactly one authoritative
document.** Another document may name a concept and point at the authority, but
must never restate the definition -- a restated definition eventually contradicts
the original.

**`docs/media_architecture.md` supersedes anything about media handling elsewhere.**
It is the current design and it changes how frames, clips, and thumbnails work.

---

## How to read the rules below

Every rule carries a status. **A rule without a status tag is a documentation bug
-- report it.**

**`[NOW]`** -- true today and guarded by a test. Breaking it is a defect. If a test
does not yet exist for a `[NOW]` rule, writing it is a Phase 0 task.

**`[TARGET -> item]`** -- not true yet. Must be true by the named work item. Do not
write new code that assumes it already holds, and do not add new violations.

**`[MIGRATING]`** -- known violations exist and are listed by file and line. The
list is closed: fix the listed sites, never add to them. A guardrail test should
fail on any site not on the list.

**Line numbers in violation lists drift.** They are re-verified when an item
touches the file. If a cited line does not contain what the rule says, the site
has moved -- search for it, fix the citation, and do not assume the rule is stale.

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
state. This rule carries no violation list of its own -- it points at the three
`[MIGRATING]` rules under "UI" below, which each carry theirs.

---

## Hard rules

### Data ownership

- **`[NOW]`** Only `Dataset` may modify a **stored** table. A worker may freely
  build and mutate a DataFrame it created itself; that is not a stored table until
  `Dataset` accepts it.
- **`[NOW]`** Every **stored** table has a `TableSchema`. `_commit_prepared` is
  the only caller of `_set_table` and is only ever handed a `_PreparedTable`
  from `_prepare_table`, so every table that reaches storage was validated; a
  non-conforming frame is refused with `SchemaRejection` and not stored, in
  whole or in part. (A test that assigns straight into `Dataset._tables`
  bypasses this; `schema_for()` then returns `None`.) `docs/architecture.md`
  §4.3 is the authority for the schema and its dtype policy. Guarded by an AST
  check and behaviour tests in `tests/test_dataset_schema.py`.
- **`[NOW]`** `Dataset.load()` is atomic: every parquet is read and validated
  before any in-memory state is replaced, so a failed load leaves the open
  project untouched -- tables, schemas, provenance log and id counter.
  `AppController.load_project()` calls `Dataset.load()` before it mutates any
  controller field, so a failed load never moves `_project_root` or drops live
  runs. Tests: `tests/test_project_load.py`.
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
  wrote. Sites: every `provenance.record()` call is in `models/dataset.py`; the
  save ordering is `models/dataset.py:790-793`. *(Re-verified 26 Aug 2026.)*
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
  Known violation, one site: `models/dataset.py:842`, where `Dataset.load()`
  restores `_id_counter` with `int(df["row_id"].astype(int).max())`. The counter
  should be stored in the project rather than recovered from the ids.
  *(Re-verified 26 Aug 2026.)*
- **`[NOW]`** `row_id` is unique **within a table**. It is not unique across
  tables: `create_table_from_rows()` deliberately keeps ids when copying rows.
- **`[NOW]`** `row_id` is exempt from every `TableSchema`: none of the three
  roles fits an opaque handle. `models/table_schema.py` never sees it -- the
  exemption is in one place, `Dataset._SCHEMA_EXEMPT_COLUMNS`. Tests:
  `tests/test_dataset_schema.py`.
- **`[NOW]`** Row ids **are preserved by project save and load** -- `save()` writes
  them to Parquet and `load()` reads them straight back, so a saved project reopens
  with the same ids and a saved reference stays valid. They are **not** guaranteed
  across **re-import or dataset reconstruction**: `load_folder()` and
  `load_csv_as_primary()` reset the counter and mint new ids, so ids from a
  previous import mean nothing after one.
- **`[NOW]`** Every **signal and artifact request** that refers to a row
  identifies it as `(table_name, row_id)`. Made true by P0.2b: `rows_updated`
  and `thumbnails_ready` carry a frozen payload (`models/notifications.py`)
  with `table_name` and a tuple of `row_ids`;
  `ArtifactStore.request_thumbnail(row_id, address, source_path, table_name)`
  (address and source_path added by P0.5b-1) holds the table and echoes it back
  as `on_thumbnail_ready(table_name, row_id)`.
  `MainWindow` -- not the controller, not each gallery -- is the single place
  that checks the payload's table against `AppController.get_active_table()`.
  Tests: `tests/test_result_delivery.py`
  (`test_rows_updated_is_a_batched_payload_with_table_name`,
  `test_live_result_lands_in_its_own_table_after_table_switch`),
  `tests/test_fake_controller_contract.py`
  (`test_signal_signatures_match_between_real_and_fake_controller`).
- **`[MIGRATING]`** The same `(table_name, row_id)` discipline for **controller
  methods** that take a bare `row_id`. Known remaining set, closed:
  `select_row(row_id)`, `get_result_index(row_id)` and
  `get_row(row_id, table_name=None)`. Each defaults to `self._active_table`, so
  a caller that has switched tables since it captured the id reads or selects
  the wrong row. This set has **no guardrail test yet**.
  *(P0.5b-1 removed `get_artifact_pixmap` from this set: it now takes a media
  address, not a row id -- the row never identified the picture, only the
  subscriber. `docs/media_architecture.md` §4.5.)*
- **`[NOW]`** **Lineage is carried by ordinary data columns, not by surrogate
  pointers.** When an operator turns one row into many -- a video into segments, a
  segment into frames -- it carries the source's identifying columns down and adds
  its own index. `participant_id, trial_id, frame_index` is the model, exactly as
  it would be in R. Gelem does not maintain a parallel `source_row_id` graph.
- **`[TARGET -> P1.6, P1.7]`** Segment and frame operators must emit the columns
  that make this work: a segment index, a frame index, `time_within_segment`, and
  everything carried down from the source.
- **`[TARGET -> P1.6]`** What gets carried down is defined by two schema
  properties, `role` and `carry_to_children` -- **not by role alone**. Identifiers
  and indices are always carried; everything else defaults to carried, because a
  trial-level covariate such as `reaction_time` is a measurement *and* is required
  on every frame row. Full rule in `docs/architecture.md` §4.2.

### Threading

- **`[NOW]`** Tables owned by `Dataset` are mutated on the main thread only.
- **`[NOW]`** Background workers never call Qt and never create a `QPixmap`.
- **`[NOW]`** Workers communicate only by placing results into the
  controller's result queues (or, for progress, overwriting a single latest
  value under a lock). They do not read controller or component state. The
  violation list is **empty**. Guarded by
  `tests/test_controller_async_contracts.py::test_worker_callbacks_touch_no_component_state`,
  an AST check that every worker-invoked controller callback reaches only
  a queue (`OperatorRegistry` boundaries are separately guarded by
  `tests/test_operator_registry_boundaries.py`).
- **`[NOW]`** Worker callbacks **are** bound controller methods. This is correct
  and deliberate.
- **`[NOW]`** Draining the result queues is bounded -- at most
  `AppController._drain_budget` items are taken from each queue per timer tick,
  and whatever is left is picked up on the next tick. Made true by P0.2b: the
  queues are `queue.SimpleQueue` (no `list.pop(0)`), progress is coalesced to a
  single latest value rather than queued, and per-row results for a tick are
  applied with one `Dataset.apply_row_updates()` call per table. `_drain_budget`
  is a constructor parameter, not a module constant. Tests:
  `tests/test_result_delivery.py` (`test_drain_is_bounded`,
  `test_drain_completes_over_later_ticks`, `test_progress_is_coalesced`,
  `test_no_drain_method_uses_pop_zero`),
  `tests/test_controller_async_contracts.py`.
- **`[NOW]`** `ArtifactStore` serves thumbnail requests on a **bounded**
  `WorkerPool` (`artifacts/worker_pool.py`), not a thread per call. Worker
  count is a keyword-only `ArtifactStore` constructor parameter with a low
  default (2) -- no longer a module constant. It is wired to a real setting:
  `main.py` passes the persisted `worker_count` in, and P0.5b-2ii-c2b2 gave the
  researcher a dialog to change it (see the Generality `[NOW]` rule on
  machine-dependent numbers). Requests naming the same canonical address are
  coalesced to one job with many subscribers; `reset()` bumps a generation
  counter. A job made stale by that bump is **guaranteed** to leave no index
  entry, no fingerprint-memo entry and to send no notification -- the job
  encodes into local variables and writes the index and memo in one
  generation-checked lock hold. What is **not** guaranteed: a stale job that
  had already encoded its JPEG leaves that file on disk with nothing pointing
  at it -- reclaiming it is P0.5b-2ii (see `docs/known_defects.md`, the
  append-only disk cache). Made true by P0.5b-2i.
  `docs/media_architecture.md` §4.4 is the authority. Demand-driven requests
  landed P0.5b-3i. Viewport cancellation landed P0.5b-3ii: the pool primitive
  is `WorkerPool.drop_pending(keep)` (there is no priority reorder -- survivors
  keep submit order, which is paint order), `ArtifactStore.set_wanted_addresses`
  turns an address set into a keep set, and `AppController._update_wanted_addresses`
  is its consumer, called on every gallery displayed-range report or clear.
  Tests: `tests/test_request_queue.py`, `tests/test_demand_driven_display.py`.

### UI

- **`[NOW]`** UI files never import pandas, PIL, numpy, mediapipe, or cv2.
- **`[MIGRATING]`** UI never reads a DataFrame. Known violation, one site:
  `ui/main_window.py:415` (`columns=list(df.columns)`), which belongs to P1.13.
  It is the only `.columns`/`.iloc`/`.loc`/`DataFrame` site under `ui/` outside
  `ui/fake_controller.py`. *(Re-verified 27 Aug 2026, P0.4.)*
- **`[MIGRATING]`** UI never touches private controller attributes. Known
  violations, one occurrence each: `ui/main_window.py:290-292` (`_op_registry`)
  and `ui/main_window.py:410-411` (`_dataset`, `_active_table`). Public
  equivalents already exist -- `get_all_row_ids()`, `get_column_type()`,
  `get_operator()` -- so these are unfinished migrations, not missing API. Both
  sites are on the closed allowlist in `tests/test_ui_private_access.py`, which
  fails on any other foreign private read under `ui/`.
  *(Re-verified 27 Aug 2026, P0.4.)*
- **`[MIGRATING]`** No widget reads another component's private state. Known
  violation, the last site: `ui/main_window.py:448` reads `operator._group_by`
  back off the operator instance, where `base.py:331-334` stores it with
  `setattr`. P1.12 removes it when parameters stop being stored on the operator
  instance. *(Re-verified 27 Aug 2026, P0.4.)*
- **`[NOW]`** Renderers may import PIL and cv2 -- they are not UI files. Renderers
  never import from `ui/`.
- **`[NOW]`** Shared display components go in `shared_widgets/`, not inside `ui/`.
- **`[NOW]`** `None` and `[]` are different. For visible columns, `None` means "no
  preference set" and `[]` means "the user chose zero columns".
  `GalleryWidget._relayout()` (`ui/gallery_widget.py:453-463`) distinguishes the
  two explicitly, and `AppController.get_effective_visible_columns()` /
  `has_visible_columns_preference()` carry the same distinction on the controller
  side. Guarded by
  `tests/test_visible_row_order.py::test_visible_columns_none_versus_empty`.

### Media

- **`[TARGET -> P1.2]`** **Only the media resolver decodes *source* media.** No
  `cv2.VideoCapture`, `av.open`, or `Image.open` **of a user's media file**
  anywhere else. Two places do today: `BaseOperator.load_image` and
  `ArtifactStore._decode_source`. (`column_types/renderers.py` came off this
  list in P0.5b-3i: thumbnail mode is cache-or-placeholder and decodes
  nothing, and detail mode loads through Qt's `QPixmap(path)`, not PIL or cv2.)
- **`[NOW]`** **Reading and writing Gelem's own derived artifacts is a different
  operation and is not covered by that rule.** `ArtifactStore` reads back the JPEGs
  it wrote, and must keep being able to. P0.5b-1 built the narrow `ArtifactCodec`
  (`artifacts/artifact_codec.py`): it is the only place a derived JPEG is
  encoded or read back, and it refuses any path outside the artifact cache root.
  The boundary is checked by behaviour, not source inspection -- a test hands it
  a source-media path and asserts the raise
  (`tests/test_artifact_identity.py::test_codec_refuses_path_outside_cache_root`).
  `[TARGET -> P1.2]` The matching half -- source decoding confined to the
  resolver, so that *nothing else opens an image at all* -- waits on the
  resolver. `column_types/renderers.py` stopped decoding source media in
  P0.5b-3i (its `_render_image` `Image.open` fallback and
  `_video_first_frame_pixmap` `cv2` path are gone). `ArtifactStore._decode_source`
  still decodes source media directly until the resolver lands.
- **`[NOW]`** The artifact cache directory is bound to the project folder on save
  and load, and swept to stay in bounds. `docs/media_architecture.md` §4.7 is the
  authority for where derived JPEGs live, why their one-way filenames force
  directory-driven eviction, and the sweep's three steps -- do not restate it
  here. `save_project` and `load_project` call
  `ArtifactStore.set_artifacts_dir(project_path / "artifacts")` then
  `reconcile_and_evict()` (main-thread only; on save both run before `save_index`
  writes the paths, or it would record files the sweep then deletes; on load the
  sweep is skipped unless `load_index()` reports the index authoritative). The
  sweep walks the directory, deletes orphaned and over-ceiling JPEGs, and drops
  index entries whose file is missing -- the last is what stops a reopened
  project showing a permanent grey tile for a cached-but-absent picture. Made
  true by P0.5b-2ii-a, -b1 and -b2. Tests: `tests/test_artifact_cache_location.py`,
  `tests/test_cache_sweep.py`.
- **`[TARGET -> P1.10]`** Native playback is the explicit exception. `QMediaPlayer`
  receives a file path and a time range directly. It shares the address **parser**
  with the resolver but not the decoding path.
- **`[NOW]`** No media is opened or decoded during a paint. In thumbnail mode
  `make_media_path_renderer`'s `render()` is cache-or-placeholder for **both**
  image and video tiles: a hit returns the cached picture touching no
  filesystem, a miss returns a grey placeholder pixmap immediately. It never
  stats, opens or decodes a source file. On a miss
  `AppController.render_column_value()` -- the one place that knows the row and
  table -- queues exactly one generation request through
  `ArtifactStore.request_thumbnail` (coalesced by canonical address, so a
  still-pending tile repainted many times adds no second job); the ready
  notification repaints the tile, which then hits. Detail mode is the
  deliberate exception and still opens the source (through Qt for images,
  `QMediaPlayer` for video). Made true by P0.5b-3i.
  `docs/media_architecture.md` §4.6 is the authority. Tests:
  `tests/test_demand_driven_display.py`;
  `tests/test_artifact_identity.py::test_load_folder_a_then_b_shows_no_a_picture`
  and `::test_load_project_a_then_b_shows_no_a_picture` exercise it end to end.
  P0.5b-3ii-b added viewport-scoped cancellation: the request queue is
  submit-order FIFO with no priority reordering, and a queued job whose
  address is no longer in the on-screen set is dropped -- see the
  `WorkerPool` rule above and `docs/media_architecture.md` §4.4.
- **`[NOW]`** Derived images are identified by an `ArtifactKey` -- canonical
  **media address**, source fingerprint, purpose, resolution,
  representative-frame policy, renderer cache version -- not by the row that
  asked. The row, table and column identify the UI subscriber waiting for the
  picture, never the picture itself. Made true by P0.5b-1:
  `media/artifact_key.py` is the key; `ArtifactStore` indexes, caches, names on
  disk and persists by it; a fingerprint memo keeps the paint-path lookup off
  `stat()`. The memo has three states -- absent (a lookup misses),
  seeded-unverified from `load_index` (served, and the next `request_thumbnail`
  re-stats it), verified from a fresh stat this session (served, and a
  duplicate request short-circuits). `ArtifactStore.load_index()`'s docstring is
  the single authority on what the seeded memo means for freshness; do not
  restate it anywhere. *(P0.5b-2i: the fresh stat and the memo write moved into
  `_run_job`'s generation-checked commit; the standalone `refresh_fingerprint`
  method is gone.)* See `docs/media_architecture.md` §4.5.
  Tests: `tests/test_artifact_identity.py`
  (`test_second_media_column_gets_its_own_cached_artifact`,
  `test_same_file_two_tables_share_one_cache_entry`,
  `test_time_range_in_address_changes_the_key`,
  `test_changed_source_fingerprint_changes_the_key`,
  `test_policy_is_part_of_the_stable_hash`,
  `test_same_row_id_two_tables_different_media_do_not_collide`,
  `test_persisted_fingerprint_is_re_stated_on_next_request`,
  `test_load_folder_a_then_b_shows_no_a_picture`,
  `test_load_project_a_then_b_shows_no_a_picture`),
  `tests/test_gallery_seam.py::test_two_media_columns_on_one_row_render_different_pictures`.
  The ready notification carries `(table_name, row_id)`, not the column that
  asked -- a deliberate simplification, not yet worth changing. See
  `docs/known_defects.md`.
- **`[NOW]`** `media/media_address.py` gives the address grammar exact,
  guarded-by-test meaning: escaping, canonical form, frame/time-point and
  range semantics, region validation and pixel arithmetic, and which
  malformed or degenerate values are refused. `docs/media_architecture.md`
  §3.6 is the authority for each decision; this file does not restate them,
  only points at them. Tests: `tests/test_media_address.py`.
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
- **`[NOW]`** **A number that does not
  generalise across machines or datasets must become a setting or a runtime
  measurement -- never a constant in the code.** The rule covers two different
  mechanisms: a number that varies **by machine** needs a setting; a number that
  varies **by file or dataset** needs a runtime measurement.
  The five machine-dependent numbers -- the in-memory cache ceiling, the disk
  cache ceiling, the worker count, the thumbnail largest side and the preview
  largest side --
  now come from `settings/` and are passed into the `ArtifactStore` constructor
  by `main.py` (P0.5b-2ii-c1). `docs/architecture.md` §9 is the single authority
  for them; do not restate the list. The module-level `DEFAULT_` constants that
  remain are fallbacks only. This tag flipped at P0.5b-2ii-c2b2: a researcher
  can now change every one of the five from **File -> Settings...**
  (`ui/settings_dialog.py`), which edits through
  `AppController.get_settings_fields()` / `apply_settings()` and submits only
  the fields that moved. Guarded by `tests/test_settings_dialog.py` (Layer A
  arithmetic and wording, plus an AST guardrail that the File menu wires a
  QAction to a handler constructing `SettingsDialog`) and, below the UI, by
  `tests/test_settings.py`.
- **`[TARGET -> P1.2]`** The first real per-file measurement case: resolving
  `#f=N` against a variable-frame-rate file (decision 8,
  `docs/media_architecture.md` §3.6) needs a per-file index of frame
  presentation times, built on first use, because no nominal frame rate is
  trustworthy on VFR source. Not built yet -- `media/media_address.py`'s
  `select_frame()` takes frame times as a plain argument and never measures
  them itself; that measurement is the resolver's, in P1.2. **The next feature
  that needs a runtime-measured per-file property should cite itself here.**

### Long-running work

- **`[TARGET -> P1.12]`** Long runs are cancellable, keeping partial results. **No
  cancellation mechanism exists today** -- there is no cancellation token and no
  check anywhere in the operator loop, so a started run always runs to completion.
  Any current statement that runs "can be cancelled" describes the target.
  *(The `WorkerPool` generation counter added in P0.5b-2i cancels **thumbnail
  jobs**, not operator runs. It does not flip this rule.)*
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
`test_fake_controller_contract.py`, `test_ui_private_access.py`.

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

**`tests/test_dataset.py` runs every test twice** -- it has module-level
`run_test()` calls as well as pytest collection. Put new tests in their own file.

Run with `python -m pytest`. Environment is a `(gelem)` virtualenv activated via
`.\setup.ps1`, on Windows PowerShell. The repo sits inside a Google Drive Streaming
path, so allow a moment after branch checkouts before running tests.

**The baseline must be green before an item starts.** The per-item green-baseline
history is in `docs/archive/rule_verification_log.md`.

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
- **Read only what the work item names, and read only the parts you need.** Use
  the Read tool's offset and limit on any file over about 1,000 lines --
  `docs/media_architecture.md` is about 100KB and must never be read whole. Do
  not survey the repository. A subagent is appropriate for two things: reading a
  large document and reporting back a specific section, and verification passes
  such as `/code-review`. Do not use one to search for files or to reconstruct
  context you were not given. If the named files are not enough, stop and say
  which file you need and why, rather than inferring.
- **Do not front-load the whole codebase.** `gelem_codebase_for_claude.txt` is a
  point-in-time review export, not a source of truth. It goes stale immediately.

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
  cover every path that no automated test covers. **A by-eye check must not
  exercise a stubbed `FakeController` method -- name real data instead.**
  Several `FakeController` methods (`set_active_table`, `save_filtered_as_table`,
  ...) only print or return canned rows, so a click list that runs through them
  proves nothing.

**Diff**
git diff main > docs/review/<id>.diff
(state whether this has already been run)

**Review**
- <needed / not needed>, per the rule below
- if needed: the exact bounded question to paste, and which of the four
  documents to attach
```

### Which items get an external review

**Review a change that establishes a contract other work is built on. Skip
mechanical ones -- the tests are the check there.** Schema roles, the operator
contract, dataset access paths, result delivery and anything every tile or
operator run flows through are contracts. **Claude Code does not get to declare a
review unnecessary**; if it thinks one is not needed, it says so and Y B decides.

### How to route a review

1. **Claude Code, every item.** Run `/code-review` on the working diff before
   committing. Catches mechanical slips: a test that asserts nothing, a signature
   that drifted from the document, a swallowed exception.
2. **Claude Desktop**, for items marked yes. It holds the design reasoning, so it
   checks code against intent. It cannot run the tests.
3. **ChatGPT**, for the same items. Its value is independence -- it does not share
   Desktop's assumptions, which is how the `carry_to_children` defect was found.
   Give it the diff plus `CLAUDE.md` and the work-item text. **Never give it
   `gelem_codebase_for_claude.txt`.** Its findings are evidence to verify, not a
   verdict: on P0.5b-1 it named the wrong test for a field and declared a
   guarded field unguarded, alongside two real findings.

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
agreed, rather than inviting a new list. **At least one check must ask what would
still pass if the change were entirely broken.**

**Escalate rather than comply.** If a review says the code contradicts a document,
that is not automatically the code's fault -- four rules in this file were
mislabelled and one was actively wrong. Say which you think it is; never bend the
code to match a document without saying so.

---

## Known defects

**Moved to `docs/known_defects.md`.** That file is the single authority for what
is currently broken and which work item fixes it. Do not restate a defect here.
When an item fixes something, search `docs/known_defects.md` for it -- a defects
list goes stale silently.
