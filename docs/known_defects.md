# Known defects

**This file is the single authority for what is currently broken in Gelem and
which work item fixes it.** `CLAUDE.md` states the rules; it does not restate a
defect. When an item fixes something, edit this file in the same item -- a
defects list goes stale silently.

Open defects are listed first, grouped by how they hurt. Fixed ones move to the
bottom, with the item that fixed them, and are never deleted: knowing that a
defect existed and how it was closed is worth keeping.

Verified against the code on 4 Aug 2026; re-verified against `main` on 24 Aug
2026 (P0.1) and updated by each item since. Each open defect should become a
failing test before or as it is fixed.

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

## Open -- non-functional at target scale

- **The on-disk artifact cache is append-only, and nothing prunes it.** Before a
  project is saved or opened the cache is the shared pre-project scratch folder
  (`%TEMP%\gelem_artifacts`, from `main.py`); a save or load then binds the store
  to `project_path / "artifacts"` and migrates the index into it (P0.5b-2ii-a,
  `tests/test_artifact_cache_location.py`). Pruning the append-only cache is
  still open, **assigned to P0.5b-2ii-b**.

  P0.5b-1's content-addressed filenames (`{key.stable_hash()}.jpg`) removed the
  accidental overwrite bound that the old `{row_id}_{artifact_type}.jpg` naming
  provided. A changed source fingerprint, a `RENDERER_CACHE_VERSION` bump,
  `reset()`, `load_index()` discarding an old-format index, and (P0.5b-2i) a job
  cancelled by `reset()` after it had already encoded its JPEG all leave their
  JPEGs on disk. The last case is routine now: every project switch cancels
  in-flight thumbnail jobs.

  **Eviction must be directory-driven, not index-driven.** `ArtifactStore` never
  walks the artifacts directory -- it only reads its index -- so a JPEG with no
  index entry is invisible to Gelem forever and an index pass can never reclaim
  it. Events after which a JPEG becomes unreachable, as a checklist for the
  eviction design: an old-format index discarded, a future
  `INDEX_FORMAT_VERSION` bump, a `RENDERER_CACHE_VERSION` bump, `reset()` on
  never-saved artifacts, process exit, the index overwritten or corrupted, and a
  generation-cancelled job that had already written its file.

  The memory cache has a ceiling and LRU eviction; the disk cache has neither.
  **Assigned to P0.5b-2ii-b**; cache size becoming a setting is P0.5b-2ii-c.

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
  missing writes no index entry, `_artifact_is_cached()` stays False, and
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
- **Migrated artifact paths are stored absolute**, so moving a saved project
  folder breaks its thumbnail cache: `load_index` seeds the old absolute paths,
  the rebuilt codec rejects them as outside the new root, `get_pixmap` misses,
  but `_artifact_is_cached` still sees the index key and queues no regeneration,
  leaving permanent grey tiles. The fix is relative paths in `artifact_index.json`
  resolved against the store's current directory; not assigned.
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

---

## Fixed

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
