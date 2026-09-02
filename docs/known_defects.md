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

---

## Open -- wrong output, not just slow

- **Operator result columns declare unregistered types.**
  `BlendshapeAvatarOperator` declares tag `avatar_path` and `PlotOperator`
  declares `plot_image`. Neither is a registered type, so `register_by_tag`
  raises, `controller.py:471-472` swallows it as a printed warning, and those
  columns render as "Unknown column".
- **`ColumnTypeRegistry.infer_type()` mistags any column whose values end in a
  media extension.** A column of `.mp4` filenames becomes `media_path`. It
  decides by `value.lower().endswith(ext)`, so it also **fails to recognise a
  media address**, which ends in a fragment such as `#f=1234`. This matters more
  since P0.5b-1, which keys the artifact cache on those address strings. Fixing
  the address half belongs with the schema work (P1.8).
- **`Dataset.load()` does not clear `ColumnTypeRegistry`**, so column types from
  the previous project persist.

## Open -- dead or inconsistent

- **`operators_config.yaml` claims to control which operators are enabled.**
  `main.py` registers them manually and never reads the file. `StatsOperator` is
  registered in code and absent from the YAML.
- **`operators/base.py` documents a `plot_html` result key**; `ResultsPanel` and
  `PlotAdvancedOperator` use `html_path`.
- **`Dataset.load_folder()` never registers `file_name` in the registry**, unlike
  the CSV import paths.
- **`_id_counter` assumes `row_id` parses as an int**, and there is no stale-file
  cleanup on re-save. (The parsing half is also the `[MIGRATING]` violation under
  "Row identity and lineage" in `CLAUDE.md`.)
- **`load_folder()` and `load_csv_as_primary()` write `str(path)`, OS-native**, so
  a fresh unsaved project's media cells are non-canonical until the first
  save/load. Belongs with the schema work (P1.8).
- **A non-string media cell is silently skipped** and is not counted by
  `_is_blank_cell()`.
- **A media cell that does not parse as an address shows a permanent grey
  placeholder.** Since P0.5b-3i the thumbnail renderer never decodes a source,
  and `render_column_value()` cannot key a demand request without a canonical
  address, so a file name with a literal `#` (which `parse()` reads as a
  fragment start) renders a placeholder and never a picture. Before P0.5b-3i
  the renderer's `Image.open` fallback still displayed it. The real fix is
  canonicalising cells at import (P1.8); detail mode is unaffected.
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
- **The "can this operator run in this mode?" guard is duplicated** between
  `controller.py` and `OperatorRegistry.run_*`, three sites each. Candidate for
  **P1.12**.
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

---

## Fixed

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
  reference operator in its place stays with P1.11.)*
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
  *(P0.2b: `BaseOperator.display_label` owns that fallback chain and
  `OperatorRegistry` passes the ready label into the callbacks. Guarded by
  `tests/test_controller_async_contracts.py::test_worker_callbacks_touch_no_component_state`.)*
