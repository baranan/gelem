# Known defects

**This file is the single authority for what is currently broken in Gelem and
which work item fixes it.** `CLAUDE.md` states the rules; it does not restate a
defect. When an item fixes something, edit this file in the same item -- a
defects list goes stale silently.

Open defects are listed first, grouped by how they hurt. Fixed ones move to the
bottom, with the item that fixed them, and are never deleted: knowing that a
defect existed and how it was closed is worth keeping.

Verified against the code on 4 Aug 2026; re-verified against `main` on 24 Aug
2026 (P0.1) and updated by each item since. On 2 Sep 2026 (the
`docs-known-defects-sync` item) sixteen tracked defects were checked against the
code named for each; all sixteen were real and none was already listed under
other wording, so all sixteen were added below. That sync verified each new
claim by reading the named code -- it did not re-verify the pre-existing
entries, the "Fixed" section, or any line number already in the file. Each open
defect should become a failing test before or as it is fixed.

On 3 Sep 2026 (the `docs-p18-foldin` item) the P1.8a..P1.8c-2b rules were
written into `CLAUDE.md` and `docs/architecture.md`, and nine further defects
were checked against the code named for each and added below: two dtype/schema
gaps (an `object`-vs-`object` spec skips the contents scan; `take_schema_messages()`
has no consumer), two provenance-sidecar inconsistencies exposed by comparison
with the new `schemas.json` fallback, two test smells, a full-table copy on two
column-add paths, an unpinned `requirements.txt`, and a stray committed CSV. A
tenth entry -- that a project saved before P1.8b-1 holding an unsupported-dtype
column can no longer be opened -- was added **unverified**: it follows by reading
from the new reject-on-unsupported-dtype path, but no such project was produced
or opened to confirm it. That pass verified the nine checked claims by reading
the named code; it did not verify the tenth, and did not re-verify the
pre-existing entries, the "Fixed" section, or any line number already in the
file. One claim checked -- that `Dataset.confirm_merge` can raise
`SchemaRejection` with no handler -- was **not** added: `AppController.confirm_merge`
wraps the call in `try/except Exception` and `_accept_table` is atomic, so
nothing is left half-written. In the same pass three entries previously pointing
at the closed umbrella item P1.8 were reassigned: the `infer_type()` address half
to P1.8d, and the two OS-native / non-parsing media-cell entries to P1.8e.

---

## Open -- dead or inconsistent

- **`_id_counter` assumes `row_id` parses as an int**, and there is no stale-file
  cleanup on re-save. (The parsing half is also the `[MIGRATING]` violation under
  "Row identity and lineage" in `CLAUDE.md`.)
- **A non-string media cell is silently skipped** and is not counted by
  `_is_blank_cell()`.
- **A media cell that does not parse as an address shows a permanent grey
  placeholder.** Since P0.5b-3i the thumbnail renderer never decodes a source,
  and `render_column_value()` cannot key a demand request without a canonical
  address, so a file name with a literal `#` (which `parse()` reads as a
  fragment start) renders a placeholder and never a picture. Before P0.5b-3i
  the renderer's `Image.open` fallback still displayed it. **Partly closed by
  P1.8e:** a cell that enters through the folder scan, CSV import, or a
  newly-inferred column is now canonicalised at that point (`#` -> `%23`), so
  it parses and renders. Two gaps remain: a column already saved with a
  non-`media_path` tag keeps that tag on load (see "Canonicalisation at accept
  does not retro-fix..." below), and `load_csv_as_primary` still writes a
  typed fragment straight into `file_name` (next entry). Detail mode is
  unaffected.
- **`load_csv_as_primary` keeps an address fragment in `file_name` as literal
  text.** It builds `file_name` with `Path(str(path_val)).name`, which does not
  split on `#`, so an image-column cell `clip.mp4#f=1234` yields the file name
  `clip.mp4#f=1234`. P1.8e canonicalises `full_path` on this path but
  deliberately leaves `file_name` as the OS-native basename in both import
  paths (`tests/test_import_canonicalisation.py::test_load_folder_leaves_file_name_as_the_os_native_basename`).
  P1.8e did not fix this on purpose. Verified against the code 6 Sep 2026;
  no item assigned.
- **Canonicalisation at accept does not retro-fix an already-saved mis-tagged
  media column.** On load, the schema restored from `schemas.json` is
  authoritative for every column it names (`Dataset._prepare_table`, called
  with `schema=declared`), so a column saved with the wrong type tag -- an
  operator that declared a typo'd tag, or an early inference that read a
  `#`-bearing column as `text` -- keeps that wrong tag forever, and
  `_rewrite_media_column` runs only on columns already tagged `media_path`.
  P1.8e canonicalises a column's *values* only when the column is newly
  inferred at accept or is already `media_path`, so it cannot reach a
  mis-tagged saved column. This is a **non-goal** of P1.8e, not an oversight.
  The real fix is a researcher-facing re-tag action -- a way to correct a
  column's type tag on an open project -- which is not built and is not yet
  tracked as its own item; `CLAUDE.md`'s rule that an unregistered operator
  tag "costs a placeholder tile, never a dropped column or a raised exception"
  is the current accepted behaviour. Verified against the code 6 Sep 2026.
- **Many module docstrings still assign files to Student A, B, or C.** Remove as
  those files are touched. (Done in `tests/test_renderer.py`, 24 Aug 2026.)
- **The thumbnail-ready notification is row-grained, not column-grained.**
  `thumbnails_ready` / `ArtifactStore.on_thumbnail_ready` carry
  `(table_name, row_id)`. Artifact identity moved to the media address in
  P0.5b-1, but the ready signal was left as it was: a repaint of the row
  repaints every column of that row and each re-looks-up its own `ArtifactKey`,
  so a row with two media columns does one redundant cache hit per update. Held
  deliberately through P0.5b-3ii as sufficient; no item assigned.
  `CLAUDE.md`'s "Derived images are identified by an `ArtifactKey`" rule points
  here.
- **No gallery or tile carries a table identity of its own.** `ImageTile.render()`
  passes `controller.get_active_table()` into the render context, which is
  exactly the value `AppController.render_column_value()` would fall back to on
  its own, so the "caller states the table" half of the display contract carries
  no information yet. `ImageTile.render()` also calls `get_row(self.row_id)` with
  no table argument. `docs/media_architecture.md` §4.6 item 8 is marked "Half
  done" and is the authority; the remaining half needs a table identity injected
  at gallery construction and passed down to the tiles, and it only pays off once
  more than one table can be displayed at a time.
- **`SettingsDialog._refresh_from_controller()` is all but dead code.** `_on_ok()`
  calls it to reset the spin boxes to what the store actually kept, then
  immediately shows the "adjusted" message box (if any) and calls `self.accept()`,
  so the researcher only glimpses the refreshed numbers behind that box and never
  on an open dialog. Either keep the dialog open when `apply_settings()` returned
  messages, or drop the refresh.
- **`CLAUDE.md` tells Claude Code to activate the environment with `.\setup.ps1`,
  but `setup.ps1` is gitignored.** It is line 28 of `.gitignore` and is not
  tracked, so a fresh clone cannot follow its own setup instruction. No item
  assigned, and `CLAUDE.md` is not the file to change first.
- **`Dataset.take_schema_messages()` has no consumer.** Accept-time dtype
  adjustments are recorded to provenance and also accumulated as plain-English
  sentences for the researcher, but nothing in the app or the controller drains
  `take_schema_messages()`, so those sentences are never shown. Only tests call
  it. Added P1.8b-1.
- **A corrupt `provenance.json` still makes a project unopenable.**
  `Dataset.load()` does `json.loads(prov.read_text())` with no guard, so a
  malformed provenance sidecar raises and the project will not open -- which now
  contradicts the rule for `schemas.json`, where a corrupt sidecar is ignored
  with a message and the project opens by inference. Pre-existing; make the two
  consistent when something next touches `load()`.
- **A project with no `provenance.json` inherits the previously open project's
  log.** `Dataset.load()` calls `self.provenance.replace(...)` only when the
  file exists, so opening a bare project (parquet only) after another project
  leaves the first project's provenance log in place. Pre-existing.
- **A project saved before P1.8b-1 that holds a column of an unsupported dtype
  can no longer be opened.** *Unverified.* Since P1.8b-1 the accept path rejects
  a datetime64, timezone-aware datetime or nullable-extension dtype with
  `SchemaRejection`, and `load()` runs every parquet through that path, so such
  a project would now fail to open where it opened before -- a datetime column
  is the realistic case. Probably no such project exists: no current import path
  parses dates, so nothing writes a datetime column. Nobody has produced or
  opened one to check.

- **Double-clicking a tile opens the detail view on the wrong column.** After
  an operator adds a media column (for example the per-row plot images), the
  gallery can show that column in its tiles, but double-clicking a tile opens
  the detail view on the row's original media column instead of the one the
  tile was showing -- even when the original column is unchecked and not
  displayed anywhere. LIKELY CAUSE, not verified: the double-click seam
  carries only the row, so the detail view chooses a column itself rather
  than being told which one. Intended behaviour: the tile you double-clicked
  was showing one column, and that is the column the detail view opens; when
  a row has several media columns the detail view should additionally let the
  researcher switch between them, defaulting to the one double-clicked. This
  is an interface change, not a local fix. Found by eye 13 Sep 2026; no item
  assigned.
- **A missing numeric value displays as "nan" in the detail view.** This is
  correct underneath -- the blendshape columns are floating point, and a
  missing value in a float column is NaN, not None -- but a researcher
  reading "nan" has no way to know it means "this was not computed". A blank
  cell, or a dash, is the honest rendering. The fix belongs wherever a value
  becomes display text, not in the operator and not in the data. Found by
  Y B on 14 Sep 2026 while checking a cancelled run by eye; no item
  assigned.
- **P1.9b-1's Save-As output copy runs on the main thread and blocks the UI
  for its duration.** `AppController.save_project()` calls
  `models/output_copy.py::execute_output_copy()` synchronously, with no
  cancellation and no progress reporting -- a save that needs to move a lot
  of video is a frozen window until it finishes. Deliberate for this item
  (saving is already refused while any operator run is live, so there is no
  concurrent write for a worker thread to race), but a real limitation on a
  slow disk or a large project. The researcher-facing warning before a large
  copy (P1.9b-2) can at least tell them it is about to happen; moving the
  copy off the main thread is not scoped to either item.
- **A project saved before P1.9 (P1.9a's `ProjectPaths`) that holds
  operator-output cells pointing at the OS temp directory is not helped by
  P1.9b-1's copy.** Before P1.9a an operator's `output_dir` was a
  construction-time scratch folder under the codebase install directory or
  the OS temp directory, never re-rooted on save -- see
  `docs/review/p1.9-survey.md`, referenced from `models/project_paths.py`.
  P1.9b-1's copy plan only ever looks at cells under the CURRENT
  `ProjectPaths.outputs_dir`; a cell left over from before P1.9a lies
  somewhere else entirely and is invisible to it. Such a project still opens
  -- the cell still resolves as an absolute path exactly as it did before --
  but the file it names is never copied into the project and is exactly as
  exposed to being cleaned up by something else on the machine as it always
  was. No item assigned; nothing currently migrates a pre-P1.9 project's
  scratch-folder outputs into its `outputs_dir`.

## Open -- smells, no item assigned

- **`create_table_from_rows()` rebuilds a `set_index("row_id")` over the whole
  source table on every call** instead of reusing the positional index P0.2a
  maintains. O(n) per "save filtered set", and a second source of truth. It also
  has a duplicate-row hazard: `by_id.loc[...]` returns a row once per occurrence
  where the old `isin` de-duplicated. Safe under today's within-table uniqueness
  rule; **P1.5** is what would break it.
- **`tests/test_ui_private_access.py`'s allowlist is keyed by attribute name**,
  not by file and line, so a new `foo._dataset` anywhere under `ui/` would pass.
  `shared_widgets/` is not scanned at all.
- **`_apply_visible_columns()` hands the same list object** to both the gallery
  and the controller, so two attributes are one list. Inert today.
- **`run_create_columns` keeps `operation_id: str = ""`** because a direct-call
  test passes it positionally. The reasoning is backwards; the risk is near zero.
- **The "can this operator run in this mode?" guard is still duplicated,
  though narrower than it was.** P1.12d-2a/P1.12d-3 consolidated the
  `controller.py` side into one implementation, `_build_operator_run`
  (`controller.py:1491-1525`), called once each from `run_create_columns`,
  `run_create_table` and `run_create_display` -- the comments at
  `controller.py:1618-1620`, `:1771-1773` and `:1844-1846` are right that it is
  now the only such check *on that side*. But `OperatorRegistry` was not
  touched: `run_create_columns` (`operators/operator_registry.py:291-294`),
  `run_create_table` (`:603-606`) and `run_create_display` (`:702-705`) each
  still re-implement the identical `descriptor is None or
  descriptor.mode_for(...) is None` condition on their own, independently of
  each other and of `_build_operator_run`. Every call from `AppController`
  reaches `_build_operator_run` first, so these three registry checks never
  fire on that path; they still guard `OperatorRegistry`'s own public methods
  against a caller that skips the controller (a test, or any future direct
  caller). Four independent implementations of one condition, not the six the
  original wording implied, but a real duplication -- one shared
  implementation plus three separate ones would remove it. No item is
  assigned.
- **`FakeController._drain_thumb_queue` is unbounded and uses `pop(0)`**, so it
  behaves differently from the real controller. More broadly, `FakeController`'s
  table switching and filtered-set saving are not plausible:
  `set_active_table()` emits the same five rows whatever table is picked, and
  `save_filtered_as_table()` only prints. Both predate P0.4.
- **The Defect A tests call `MainWindow._apply_visible_columns`, a private slot**,
  because the Columns combo's handler has no public equivalent. The honest fix is
  a public method.
- **`export_csv()`, `run_create_table()` and `run_create_display()` deliver rows
  in the caller's `row_ids` order** rather than table order.
- **A broken or missing media path whose tile is on screen re-queues a failing
  worker job on every fresh repaint.** Since P0.5b-3i there is no main-thread
  `exists()` short-circuit and no negative cache, so a job that finds the source
  missing writes no index entry, `ArtifactStore.is_cached()` stays False, and
  repainting the tile re-submits it. Coalescing bounds it to one in-flight job
  per address and the failing job is microseconds. P0.5b-3ii's viewport
  cancellation bounds only the off-screen case: once the tile scrolls away its
  address leaves the wanted set and the queued job is dropped. A tile still on
  screen is inside the wanted set, so its job is never dropped and re-queues on
  every repaint. The remaining fix is a negative cache -- remember an address
  that failed to decode and do not re-queue it -- and no item is assigned.
- **Saving a project cancels in-flight thumbnail jobs without repainting their
  tiles.** `save_project` -> `ArtifactStore.set_artifacts_dir` bumps the
  generation (P0.5b-2ii-a) so no worker commits an old-root path into the index
  being saved. A side effect: any thumbnail still generating when the user hits
  Save is dropped and sends no `on_thumbnail_ready`, so its tile stays a grey
  placeholder until the next repaint re-queues it (scroll, resize, table
  switch). Self-healing and minor; no negative cache involved. No item assigned.
- **The first save of a project copies the whole scratch thumbnail cache on the
  main thread.** `set_artifacts_dir`'s migration loop is a synchronous
  `shutil.copy2` per indexed JPEG (P0.5b-2ii-a); for a large frame dataset on a
  slow filesystem this blocks the UI during Save. The file I/O is kept off
  `_lock` and before the directory swap, so a failure leaves the store
  consistent, but it is still synchronous. An async or move-based migration is a
  candidate for P0.5b-2ii-b. Also: a migration failure propagates after
  `Dataset.save()` has already written, so `save_project` reports "Failed to
  save project" over a folder that does hold a complete dataset.
- **No guardrail stops a future caller of `read_only_view()` from mutating the
  frame it returns.** Its docstring also leans on QueryEngine purity, but the
  test covers `apply()` only -- not `apply_grouped()` or `get_group_values()`.
- **The row-id index keeps the last occurrence** when a `row_id` appears twice,
  rather than raising.
- **`AppController.get_group_values()` ends in a bare `except Exception: return
  []`**, so a missing column silently becomes "this column has no values".
- **`.claude/settings.json` has no `permissions.deny` list** for
  `Bash(git commit *)` and `Bash(git push *)`. `.claude/` now exists; this is
  still not done.
- **`SweepResult.files_deleted` folds orphan cleanup in with ceiling eviction.**
  `reconcile_and_evict()` counts orphans (unreachable files the index never
  named) and ceiling evictions in one number, and `AppController.apply_settings()`
  reports that number to the researcher as "deleted N cached picture files" -- so
  it can over-state how many *cached* pictures were actually lost when a
  disk-ceiling change also swept orphans. Over-reporting was accepted as the
  safer error; the clean fix is a separate count on `SweepResult`.
- **`AppController.apply_settings()` assumes the `ArtifactStore`'s ceilings equal
  the persisted settings.** It compares `SettingsGateway.describe_fields()` before
  and after `save_values()` and pushes a ceiling into the store only when the
  persisted value moved, which is correct only while the store's ceiling and the
  persisted setting cannot diverge. That holds today -- `apply_settings()` is the
  only production caller of `set_memory_cache_max_bytes()` and
  `set_disk_cache_max_bytes()` -- but the invariant is load-bearing and
  unguarded, so anything that later sets a ceiling directly breaks it silently.
- **`SettingsDialog._join_with_and()` raises `IndexError` on an empty list.**
  `labels[-1]` is evaluated whenever `len(labels) != 1`. It is safe only because
  its single call site in `confirmation_text()` is guarded by `if restart_labels:`.
  One line to fix.
- **A `SettingsDialog` built over an `AppController` with no settings gateway
  raises `RuntimeError` out of a menu click.** `__init__` calls
  `controller.get_settings_fields()`, which raises when `_settings_gateway is
  None`, and the dialog does not catch it. No shipped path hits this -- `main.py`
  always wires the gateway and the `None` default exists only for test
  construction -- but nothing defends the menu action.
- **In `--fake-data` mode `SettingsDialog` never assigns `_fields`, `_spin_boxes`
  or `_initial_native`.** `__init__` returns after `_build_empty()` when there are
  no fields, so any later call into `_current_native()` or `_on_ok()` would be an
  `AttributeError`. `_on_ok` is unreachable in that mode today (only a Close
  button is wired), so this is robustness only.
- **The settings wiring guardrail does not pin that the gateway's store is the
  one `.load()` was called on.**
  `tests/test_settings.py::test_main_passes_a_settings_gateway_into_app_controller`
  checks only that the gateway's first positional argument is a name bound to *a*
  `SettingsStore(...)` call, so `main.py` binding two stores and loading settings
  from one while building the gateway over the other would still pass. The test
  comment says as much and leaves it "to the reader to confirm by eye". Three more
  lines whenever something next touches that test.
- **The negative assertions in `tests/test_request_queue.py` rest on
  `time.sleep(0.1)` to `0.3`.** Tests such as
  `test_drop_pending_removes_unkept_jobs_and_returns_their_keys` spin until the
  kept jobs finish, then sleep and assert a dropped job did *not* run -- a slow
  machine could let the dropped job slip in first. `tests/test_artifact_cache_location.py`
  uses only `threading.Event` for the same kind of check and is the pattern to
  copy.
- **`AppController._update_wanted_addresses()` is O(visible tiles) on the UI
  thread, once per mounted-window shift.** It runs on every displayed-range report
  or clear, resolving every visible media cell to a canonical address with pure
  path arithmetic (no I/O). A zero-millisecond single-shot `QTimer` would coalesce
  a burst of scroll reports into one pass, but the cost should be measured before
  that is built.
- **`ui/main_window.py::_clear_grouped_galleries()` calls `clear_displayed_range()`
  once per group key.** Each call runs `AppController._update_wanted_addresses()`
  in full, so tearing down a group-by view triggers one complete wanted-address
  recompute per group rather than one for the whole teardown.
- **`test_unparseable_media_cell_is_skipped_not_raised` monkeypatches
  `_resolve_media_cell` to raise.** In `tests/test_demand_driven_display.py`, the
  test replaces `controller._resolve_media_cell` with a stub that raises
  `MediaAddressError` for one cell, so it proves the `try/except` in
  `_update_wanted_addresses()` but not that a real cell containing a literal `#`
  actually reaches and trips it.
- **`load_folder()` inherits the previous project's artifacts directory.** It
  calls `ArtifactStore.reset()` but not `set_artifacts_dir()`; only
  `save_project` / `load_project` re-point the store. So after a project has been
  open in the session, loading a bare folder writes its thumbnails into that saved
  project's `artifacts/` directory (as orphans, since `reset()` cleared the
  index). Self-healing -- the saved project's next sweep deletes them -- but it is
  undocumented and makes manual cache testing treacherous.
- **A hard-killed encode leaves `<hash>.jpg.<pid>.<tid>.tmp` files that nothing
  reclaims.** `ArtifactCodec.write_jpeg()` writes to that temp name and
  `os.replace`s it into place, cleaning up only when the encode itself fails; a
  process killed mid-write leaves the temp file behind. The cache sweep owns only
  top-level `<hash>.jpg` files by design (`docs/media_architecture.md` §4.7), so
  these are never swept. A deliberate gap, not an accident; no item assigned.
- **`manual_testing/` is gitignored** (line 26 of `.gitignore`), so fixes or
  checks made there are never committed and vanish on a fresh clone. Whether the
  directory should be tracked has not been decided.
- **An `object` column checked against an `object` schema spec is accepted with
  no contents check.** `check_frame`'s `arrived == stored` short-circuit runs
  before the text-contents scan, so a column declared `object` that actually
  holds non-`str` values is stored; the scan only fires when an `object` column
  meets a *different* text spec. Added P1.8c-1.
- **`add_column` and `add_computed_column` copy the whole table for rollback
  safety.** Both do `_get_stored_table(...).copy()` so a `SchemaRejection`
  leaves the stored frame intact, where `apply_row_updates`'
  in-place-plus-targeted-rollback pattern would avoid the full copy.
- **`_assert_schema_matches_frame` in `tests/test_dataset_schema.py` cannot fail
  its column-order assertion.** It compares `schema_for(table)` against the
  columns of the frame the accept path itself stored, and `_prepare_table`
  builds the schema from exactly those columns in that order -- so only the
  "row_id absent" half of the helper can actually fail.
- **`tests/test_dataset.py` runs its checks twice, in two shapes.** 28 collected
  `test_` functions, plus 64 module-level `run_test(...)` calls that fire at
  import. `CLAUDE.md` already flags the double run; the count is recorded here.
- **`test_images2/othercolnames.csv` is committed and no prompt asked for it.**
  It landed in the P1.8c-2b commit and is referenced only by `create_test_csv.py`,
  not by any test. Harmless test data; delete it if a later item works in that
  area.
- **`requirements.txt` pins nothing -- eleven bare package names.** The
  supported-dtype set, the text-dtype rule and the empty-parquet cast were
  designed against pandas 3.0.2, numpy 2.4.4, pyarrow 23.0.1, Python 3.13.2. On
  pandas 2 a bare text column is `object`, not `str`, so a fresh clone can
  behave differently.
- **A TABLE-mode operator that declares `creates_table` cannot declare the
  role of a column it creates.** `OutputSpec.creates_table=True` requires
  `columns` to be empty (`operators/descriptor.py`'s `OutputSpec`), and
  `OutputColumn` carries only a name and a type tag, no `role` --
  `docs/architecture.md` §4.2's `role` / `carry_to_children` vocabulary has
  no way to reach a TABLE mode's own new columns. `operators/segment.py`
  (P1.6b) is the concrete case: `segment_index` is plainly an `index`
  column by §4.2's own definition, but the schema Dataset infers on accept
  has no declared role to read, so it falls back to `measurement`. Harmless
  today because every `measurement` column defaults `carry_to_children` to
  true (§4.2), so `segment_index` is still carried down to a later frame
  split -- but it would be silently dropped the moment carry narrowing
  exists (a `carry_columns` parameter on the frame operator, explicitly out
  of scope for P1.6b), since narrowing only ever touches `measurement`
  columns and an `index` column is supposed to be exempt from it. No item
  assigned.
- **A FRAME-requirement COLUMNS run reads only the `full_path` metadata key**
  (`operators/operator_registry.py`'s `_run_create_columns_worker`), never a
  column the researcher picked or a differently-named media_path column type
  inference tagged. This is narrower than `AppController._absolutise_media_columns`
  (P1.2c-2), which pre-resolves every column the active table's schema tags
  `media_path`, not only one named `full_path` -- an ADDRESS-requirement run
  already benefits from that (an operator can read any such column itself,
  as `tests/test_media_resolver.py::test_address_run_resolves_a_relative_cell_in_a_non_full_path_media_column`
  exercises), but a FRAME run's fixed key means a table whose media lives
  under another column name gets no decoded frame at all, silently, however
  the pre-resolution step improves. Which column a FRAME operator reads is a
  contract question -- letting an operator or its descriptor name the column,
  the same way `video_frames.py`'s `video_column` parameter does for a TABLE
  operator -- not a patch to this file. No item assigned.
- **The per-file frame-time index (`media/resolver.py`'s `_build_frame_index`)
  identifies a frame by its presentation time rounded to the nearest
  microsecond, not by its exact timestamp.** It refuses two presented frames
  whose raw ticks are an exact tie, and (review round 5) two presented frames
  whose raw ticks are distinct but round to the same microsecond -- but a
  file whose real per-frame timings are closer together than microsecond
  resolution allows (a container time_base finer than 1 tick per
  microsecond, encoding two frames within that same microsecond) would still
  trip this refusal on genuine, non-duplicate content. No real fixture has
  been found that does this, and no video encoder in ordinary use places two
  presented frames a fraction of a microsecond apart. The clean fix, if one
  ever does, is exact-timestamp frame identity (keyed on the raw tick or an
  exact rational, never a rounded microsecond value) rather than loosening
  the refusal. No item assigned.
- **Two threads that both miss the per-file frame-time index cache build it
  concurrently.** `MediaResolver._get_frame_index` checks the cache under
  `_frame_index_lock`, releases the lock, and (on a miss) calls the expensive
  `_build_frame_index` demux pass OUTSIDE the lock -- deliberately, so one
  slow build never blocks an unrelated file's lookup -- then re-acquires the
  lock only to store the result. Two threads racing to resolve the same
  file's first `#f=` address (or first `with_ordinals=True` span) can
  therefore both demux the whole file, and whichever finishes last simply
  overwrites the other's entry with an equivalent one -- wasted work, not a
  correctness bug, since the index is a pure function of the file. No item
  assigned.

## Open -- test-suite instability and process leaks

- **Native access violation during the full pytest run.** *Cause unverified.*
  Running the whole suite as one `python -m pytest` process on Windows dies,
  roughly one run in six, with a native access violation (exit code
  `0xC0000005`, which Python reports as `3221225477` or `-1073741819`) on the
  main thread. It strikes at the first moment in the run that a Qt widget is
  shown. The diagnostic record for this lives outside this repository, so
  everything a second developer needs is here:
  - **Symptom.** The pytest process exits with the native access-violation
    code and no Python traceback. Nothing is printed at the point of death; the
    run simply stops.
  - **Frequency.** About one full-suite run in six. It is not tied to any one
    test; a re-run usually passes.
  - **What five counted diagnostic rounds ruled out.** (1) The background
    worker threads -- the crash is on the main thread and reproduces with the
    worker pools quiescent. (2) PIL -- the PIL frames in the first traceback
    belonged to bystander worker threads, not the crashing main thread; with
    the worker pools forced to run inline on a single thread the crash still
    occurred in 10 runs out of 10. (3) Any single culprit module -- no one
    module, removed, makes the
    crash go away; it needs a co-occurrence of MediaPipe FaceLandmarker
    inference (`tests/test_blendshape_operator.py`), several data-layer test
    modules, and a widget being shown. (4) The loaded library set alone --
    loading the same libraries without running the tests does not crash. (5)
    Module imports alone, and QApplication creation timing -- importing every
    test module without running it does not crash, and moving when the
    `QApplication` is constructed does not change the rate. No standalone
    reproducer exists outside pytest, and an application-shaped script does not
    crash.
  - **Status.** The cause is UNVERIFIED and the investigation is deliberately
    closed. This is not being fixed.
  - **Mitigation.** `run_tests.py` at the repo root. It runs the suite as
    several independent pytest processes -- the non-widget modules together,
    each widget-touching module alone -- so a native crash in one process loses
    only that process's results and the rest of the suite still reports. Its
    summary table flags a group that hit this code as "NATIVE CRASH -- known
    defect" rather than "FAILED".
  - **Trigger for reopening.** The RUNNING APPLICATION (`python main.py`, or a
    packaged build) dying without a Python traceback. A flaky `run_tests.py`
    group that reports the native-crash code is the known, contained condition
    and is not cause to reopen. Only the same failure mode escaping into normal
    application use is.

- **`tests/test_renderer.py` builds a `QApplication` at module scope with no
  `QApplication.instance()` guard.** *Consequence unverified.* Because pytest
  imports every collected module before it runs any test, that module-level
  line -- not the `qapp` fixture in `tests/conftest.py` -- is what creates the
  process-wide `QApplication` in a normal full-suite run. Recorded here; not
  fixed in this item.

- **Nothing ever shuts down a `WorkerPool`.** *Consequence unverified.*
  `ArtifactStore` has no `shutdown()` method and no test or application code
  stops the pools it starts, so roughly 30 daemon threads survive to the end of
  a full test run and are only cleaned up by process exit.

- **`operators/blendshapes.py` never closes its `FaceLandmarker`.**
  *Consequence unverified.* The MediaPipe `FaceLandmarker` it creates is never
  `.close()`d. This was proven **not** to be the native crash above, but it is
  still a resource leak.

- **Two files under `tests/` are named `test_*.py` but are standalone
  manual-check scripts, not pytest modules.** *Consequence unverified.*
  `tests/test_renderer.py` and `tests/test_results_panel.py` do their work at
  import / under `__main__` and expose no `test_` functions, so pytest collects
  zero tests from them and exits 5. `run_tests.py` has to name them explicitly
  (`MANUAL_CHECK_MODULES`) to tell that apart from a real "nothing collected"
  failure such as a mistyped `-k`. The real fix is renaming them out of the
  `test_` namespace; not done in this item.

## Open -- questions, no item assigned

- **Should `_run_outcome()` compare rows applied against rows requested for
  every run, not only a cancelled one?** Today `AppController._run_outcome()`
  makes that comparison only inside its cancelled branch (COLUMNS mode): an
  UNcancelled run that silently produced less than it was asked for --
  for example a row whose image failed to load -- is still recorded
  "complete". Extending the comparison to every run is **not obviously
  correct**: whether a shortfall means "partial" depends on whether one row
  in always means one result out for COLUMNS mode, and an operator that
  legitimately writes nothing for some rows would be mislabelled. Settling
  it also needs a load-failure counter that nothing tracks today. Raised by
  Claude Code on 14 Sep 2026 and deferred deliberately -- this is a question,
  not a bug with a known fix.

---

## Fixed

- **`operators_config.yaml` claimed to control which operators are enabled but
  `main.py` registered them by hand and never read the file** (`StatsOperator`
  was registered in code and absent from the YAML). *(P1.11a:
  `operators/operator_config.py` reads the file and `main.py` keeps only the
  factories; drift or a malformed file raises `OperatorConfigError` at startup.
  `docs/architecture.md` §7 is the authority. Tests:
  `tests/test_operator_config.py`.)*
- **`operators/base.py` documented a `plot_html` result key** while `ResultsPanel`
  and `PlotAdvancedOperator` used `html_path`. *(P1.11b: `base.py` documents
  `html_path` and `ui/results_panel.py` reads that key only -- the `plot_html`
  fallback is gone.)*
- **`load_folder()` and `load_csv_as_primary()` wrote `str(path)`, OS-native**,
  so a fresh unsaved project's media cells were non-canonical (backslashes, an
  unescaped `#` or `%`) until the first save/load. *(P1.8e-2a: `load_folder`
  canonicalises every scanned path with `media_address.canonicalise_path` and
  `load_csv_as_primary` canonicalises each image-column cell with
  `canonicalise_cell`, so the stored cell is canonical from the first accept.
  Separators are normalised to forward slashes and `#`/`%` are escaped. Tests:
  `tests/test_import_canonicalisation.py`.)*
- **The on-disk artifact cache was append-only, and `load_index()` seeded index
  entries without checking the JPEG was present.** Nothing walked the artifacts
  directory, so a JPEG whose index entry was gone -- from a discarded old-format
  index, an `INDEX_FORMAT_VERSION` or `RENDERER_CACHE_VERSION` bump, a changed
  source fingerprint, `reset()` on never-saved artifacts, process exit, a
  corrupted index, or a generation-cancelled job that had already encoded its
  file -- was unreachable forever, because the on-disk name is a one-way hash.
  The reverse case: an indexed-but-absent entry (a partial sync, a deleted cache
  file, a foreign-OS absolute path) reopened with `is_cached()` reporting True,
  so demand-driven display queued no request and the tile stayed a permanent
  grey placeholder until the app restarted. *(P0.5b-2ii-b2:
  `ArtifactStore.reconcile_and_evict()` walks the directory on every save and
  load -- via the pure `artifacts/cache_sweep.py::plan_sweep` -- deletes
  orphaned and over-ceiling JPEGs, and drops index entries whose file is gone.
  On load the sweep runs only when `load_index()` reports the index as
  authoritative, so a transient failure reading `artifact_index.json` cannot
  turn into a full cache wipe. Disk ceiling defaults to
  `DEFAULT_DISK_CACHE_MAX_BYTES` (1 GiB), evicted oldest-mtime first; `main.py`
  now passes it (and the memory ceiling, worker count and thumbnail/preview
  sizes) from `settings/` (P0.5b-2ii-c1, `docs/architecture.md` §9), with only
  the editing dialog (c2) still missing. The sweep runs only at save and
  load, so the pre-project scratch folder (`%TEMP%\gelem_artifacts`, used by a
  session that never saves a project) is deliberately out of scope and still
  grows unbounded until the OS clears `%TEMP%`. `docs/media_architecture.md`
  §4.7 is the authority. Tests: `tests/test_cache_sweep.py`.)*
- **The media renderer decoded source files on the paint path, and the
  controller requested a thumbnail for every row on load.** `_render_image`
  fell back to `Image.open` and `_render_video` ran `cv2.VideoCapture` on the
  main thread on a cache miss; `load_folder`, `load_csv_as_primary` and the
  `create_table` result path each looped over the whole table queuing
  requests. *(P0.5b-3i: thumbnail mode is cache-or-placeholder and opens no
  source; `AppController.render_column_value` queues one request per painted
  tile on a miss; the three eager loops are gone. Tests:
  `tests/test_demand_driven_display.py`,
  `tests/test_artifact_identity.py`.)*
- **`ArtifactStore.request_thumbnail()` spawned one raw `threading.Thread` per
  call.** *(P0.5b-2i: requests run on a bounded `WorkerPool`
  (`artifacts/worker_pool.py`, default 2 workers, a keyword-only `ArtifactStore`
  constructor parameter), coalesced by canonical address, and cancelled by a
  generation counter that `reset()` bumps. Tests: `tests/test_request_queue.py`.)*
- **`operators/thumbnail.py` was dead code.** *(P0.5b-2i: deleted, together with
  its `main.py` import and registration and its `operators_config.yaml` entry.
  `ArtifactStore._run_job` was always the real path. Promoting a genuine
  reference operator in its place moved to P1.12 (P1.11b) -- it waits on the
  operator contract settling.)*
- **The purpose -> resolution mapping was computed in two places.**
  *(P0.5b-1-followups: `ArtifactStore._resolution_for` is now the public
  `ArtifactStore.resolution_for`, `column_types/renderers.py::_cached_thumbnail`
  calls it on the injected store instance, and the `column_types ->
  artifacts.artifact_store` constant import (`THUMBNAIL_RESOLUTION` /
  `PREVIEW_RESOLUTION`) is gone.)*
- **`media/artifact_key.py` imported `_POLICIES`, a private name, from
  `media_address`.** *(P0.5b-1-followups: `_POLICIES` is now the public
  `POLICIES`, in `media_address.__all__`; no private alias remains.)*
- **A row with several media columns shared one cached image**, and
  **`ArtifactStore.load_index()` could show the first project's pictures.**
  *(P0.5b-1: artifacts are keyed by `ArtifactKey` -- media address, fingerprint,
  purpose, resolution, policy, version -- so two media columns on one row and two
  projects no longer collide, and `load_project()` calls `_store.reset()` before
  `load_index()`. Tests: `tests/test_artifact_identity.py`.)*
- **`Dataset.get_row()` called `get_table()`**, and **`Dataset.update_row()`
  scanned the whole `row_id` column per result.** *(Both false since P0.2a;
  bullets removed 27 Aug 2026. `get_row()` reads through the row-id index and
  `update_row()` is a wrapper over `apply_row_updates()`, which uses it.)*
- **Controller result queues were lists drained with `pop(0)`, unbounded per
  tick.** *(P0.2b: the queues are `queue.SimpleQueue` and each drain is bounded
  by `AppController._drain_budget`.)*
- **`operators/CLAUDE.md` claimed `self.output_dir`.** *(Removed 24 Aug 2026; it
  documents `self._output_dir` and notes that `self.output_dir` never existed.)*
- **`GalleryWidget._relayout()` conflated `None` and `[]` for visible columns**
  (`ui/gallery_widget.py:390`). *(Fixed before 24 Aug 2026; the rule in
  `CLAUDE.md` is now `[NOW]` and guarded by
  `tests/test_visible_row_order.py::test_visible_columns_none_versus_empty`.)*
- **`ui/filter_panel.py:199` reached into `_registry`.** *(Fixed 24 Aug 2026; it
  calls the public `controller.get_column_type()`.)*
- **`ui/main_window.py` read `GalleryWidget._row_ids` at three sites, and
  `GalleryWidget` read `TileWidget._tile` at two.** *(P0.4: the controller owns
  the ordered result and the gallery is given an index range into it, holding no
  row ids; `TileWidget` has a public `get_row_ids()`.)*
- **Worker-bound callbacks read component state**
  (`_on_operator_setup_error`, `_on_operator_row_errors` called
  `self._op_registry.get()` from the worker thread to build a display label).
  *(P0.2b: `OperatorRegistry` passes the ready label into the callbacks. Since
  P1.12d-3 the worker computes that label once from
  `run.spec.mode_descriptor.label` -- the `BaseOperator.display_label` property
  that used to own the fallback chain is deleted. Guarded by
  `tests/test_controller_async_contracts.py::test_worker_callbacks_touch_no_component_state`.)*
- **Operator result columns declared unregistered types.**
  `BlendshapeAvatarOperator` declared tag `avatar_path` and `PlotOperator`
  declared `plot_image`. Neither was a registered type, so the old
  `register_by_tag` raised, `controller.py:471-472` swallowed it as a printed
  warning, and those columns rendered as "Unknown column". *(P1.8d-2b-2: both
  operators now declare `media_path`, a registered tag, and the mechanism they
  tripped is gone -- an operator's declared tag reaches the target table's
  `TableSchema` as a `ColumnHint(type_tag=...)`
  (`Dataset.apply_row_updates()`, `models/dataset.py`), and an unknown tag is
  stored on the schema as given rather than raising; a tag the registry has no
  renderer for only prints a once-per-run warning
  (`AppController.run_create_columns`, `controller.py`) and the column renders
  as a placeholder. Tests: `tests/test_operator_tag_hints.py`.)*
- **`ColumnTypeRegistry.infer_type()` mistagged any column whose values end in
  a media extension**, and separately **failed to recognise a media address**
  ending in a fragment such as `#f=1234`. *(P1.8d-1: `ColumnTypeRegistry.infer_type`
  is deleted; `infer_type_tag` in `models/table_schema.py` is the single
  authority now. Both halves were checked against the current code and are
  fixed, not just moved: a value is `media_path` only when its path portion
  both ends in a media extension AND carries a directory separator or an
  address fragment (`_looks_like_media_path`), so a bare filename such as
  `face.jpg` with neither is `text`, and `#f=1234` is recognised because the
  address is parsed before the extension check runs. Tests:
  `tests/test_table_schema.py`
  (`test_infer_type_tag_column_of_bare_media_filenames_is_text`,
  `test_infer_type_tag_media_value_with_a_frame_fragment_is_media_path`).)*
- **`Dataset.load()` did not clear `ColumnTypeRegistry`**, so column types from
  the previous project persisted. *(P1.8d-2b-1: `Dataset` no longer holds or
  reads a `ColumnTypeRegistry` reference at all, so there is nothing left for
  `load()` to clear -- a loaded project's column tags come from the schema
  `_commit_prepared` installs (restored from `schemas.json` when present, else
  re-inferred). Tests: `tests/test_dataset.py`
  (`test_load_without_column_types_json_keeps_full_path_media`),
  `tests/test_project_load.py`.)*
- **`Dataset.load_folder()` never registered `file_name` in the registry**,
  unlike the CSV import paths. *(P1.8d-2b-1: neither path registers anything
  in a registry any more -- `Dataset` holds no `ColumnTypeRegistry` reference
  at all, and `file_name`'s display tag now comes from `infer_type_tag` /
  `infer_schema` (`models/table_schema.py`) exactly like every other inferred
  column, `load_folder()` included. The asymmetry this entry described no
  longer exists on either path.)*
- **The toolbar's table-name dropdown stayed at its first-show width forever.**
  `QComboBox`'s default `AdjustToContentsOnFirstShow` policy sizes the widget
  once, at first show; every table name added afterwards -- by a load or an
  operator run, including a generated name such as `frames_expanded` -- was
  truncated regardless of how much toolbar space was free. *(P1.6b:
  `ui/main_window.py`'s `_table_combo` now carries
  `setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)`, so it
  resizes to its widest current entry on every repopulation, capped at
  `setMaximumWidth(260)` so one very long name cannot crowd out the rest of
  the toolbar row.)*
