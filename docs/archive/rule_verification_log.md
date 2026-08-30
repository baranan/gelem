# Rule verification log

**Historical only, never guidance.** `CLAUDE.md` is the authority for every rule
and every current violation site. This file records how those rules reached their
present wording, when each `[MIGRATING]` list was last re-checked and what was
found, and what the green baseline was at each item. Nothing here overrides
`CLAUDE.md`, and a disagreement between the two means `CLAUDE.md` is right and
this log is stale.

It exists so that `CLAUDE.md` can stay short. `CLAUDE.md` is loaded into every
Claude Code session and into every subagent's context, so its length is paid for
many times over; this file is read only when someone asks how a rule got here.

---

## Why the rules carry status tags at all

`CLAUDE.md` originally stated rules in the imperative without saying which were
true and which were goals. Several were contradicted by the code, and two were
wrong as written. **Four rules have now been found mislabelled and one was
actively wrong.** That is why every rule carries `[NOW]`, `[TARGET -> item]` or
`[MIGRATING]`, and why an untagged rule is itself a documentation bug.

Wordings that were corrected, and are worth not re-introducing:

- **"Components talk through public methods only", stated untagged.** It was a
  goal, not the state. Stating it untagged was itself an instance of the problem
  the status tags exist to fix.
- **"Row ids are globally unique and never reused."** Too strong. A later
  revision said they were "not stable across a reload", which was too weak and
  wrong about save/load. The current `[NOW]` wording distinguishes save/load
  (preserved) from re-import (not preserved).
- **"Workers never touch the controller."** This forbade the actual design.
  Worker callbacks *are* bound controller methods; what they must not do is read
  controller or component state.
- **"The `(table_name, row_id)` discipline is done for controller methods."**
  Overreached -- P0.2b did the signals and the artifact request only. The
  controller-method set is still `[MIGRATING]`.
- **"No other widget reads another widget's private attribute."** False when
  written; `GalleryWidget` was reading `TileWidget._tile` at two sites. Fixed in
  P0.4.

---

## `[MIGRATING]` re-verification history

**UI never reads a DataFrame.** 24 Aug 2026: this used to be two sites; the
second merged into the citation and no longer separately reads a DataFrame.
27 Aug 2026 (P0.4): still the only `.columns`/`.iloc`/`.loc`/`DataFrame` site
under `ui/` outside `ui/fake_controller.py`. Line moved 401 -> 415 as P0.4 added
code above it; the site itself unchanged, still P1.13.

**UI never touches private controller attributes.** 24 Aug 2026:
`ui/filter_panel.py:199` no longer reaches into `_registry` -- it calls the public
`controller.get_column_type()` -- so that citation was removed. The
`_op_registry` and `_dataset`/`_active_table` sites shrank from three occurrences
each to one. 27 Aug 2026 (P0.4): still one occurrence each, unchanged in content;
line numbers moved (`_op_registry` 276-278 -> 290-292,
`_dataset`/`_active_table` 396-397 -> 410-411) because P0.4 added code above them.

**No widget reads another component's private state.** 27 Aug 2026 (P0.4): the
three `ui/main_window.py` sites reading `GalleryWidget._row_ids` are gone -- the
controller now owns the ordered result and the gallery is given an index range
into it. `GalleryWidget`'s two reads of `TileWidget._tile` are also gone;
`TileWidget` has a public `get_row_ids()`. The guard is
`tests/test_ui_private_access.py`, an AST check with a closed allowlist of
`_op_registry`, `_dataset`, `_active_table` and `_group_by`.

**`row_id` is opaque.** 26 Aug 2026: `models/dataset.py:842` is still the only
site; no other file parses, sorts by, or infers meaning from `row_id`.

**Provenance records every structural operation.** 26 Aug 2026: every
`provenance.record()` call still lives in `models/dataset.py` -- none in
`operators/` or `controller.py` -- and `save()` at `models/dataset.py:790-793`
still writes the file before recording the save.

**Components talk through public methods.** 26 Aug 2026: this rule carries no
violation list of its own; it points at the three UI rules above.

---

## Green-baseline history

- **24 Aug 2026.** `tests/test_renderer.py` had two collection errors from
  functions named `test_*` that were really manual checks (`test_thumbnail`,
  `test_detail`, each requiring positional arguments pytest tried to treat as
  fixtures). Renamed to `check_thumbnail` / `check_detail` -- the file is a
  standalone manual-check script, not a pytest module, so nothing inside it
  should match `test_*`.
- **26 Aug 2026 (P0.1), green.** Deleting the stray untracked
  `test_images/boxtest.png` (a 21st image with no matching `metadata.csv` row)
  fixed three `tests/test_dataset.py` failures. `test_images/` holds exactly 20
  `.jpg` files against 20 `metadata.csv` rows.
- **27 Aug 2026 (P0.4), green.** 174 items collected, zero errors, zero failures.
  (91 at P0.1; P0.2a, P0.2c, P0.3 and P0.4 each added test files.)
- **27 Aug 2026 (P0.2b), green.** 193 items. P0.2b added
  `tests/test_result_delivery.py` and grew
  `tests/test_controller_async_contracts.py` and
  `tests/test_fake_controller_contract.py`.
- **30 Aug 2026 (P0.5b-1), green.** 215 items.
- **30 Aug 2026 (P0.5b-1-followups), green.** 217 items, 0 xfailed.

---

## Superseded design decisions

- **Keyframe interval as a machine-dependent constant.** Until 26 Aug 2026 it was
  the third example under the Generality rule. It was measured per video
  specifically to decide whether to build a whole-video proxy layer, and that
  proxy is rejected -- see `docs/media_architecture.md` §10.
- **Per-file seek cost as the runtime-measurement example.** Considered, since the
  measurement pass found that keyframe interval does not predict it, but seek cost
  is not measured anywhere in the current plan. It appears in
  `docs/media_architecture.md` §10 only as something a future revival of
  proxy-like work would need. The live example is now VFR frame timings (P1.2).
- **Splitting P0.5b-1 into "keys and codec inside ArtifactStore" then "wire
  addresses into the render path".** Rejected: the first half alone changes
  nothing observable and does not flip the `xfail`, and a diff with no success
  criterion cannot be reviewed.
- **The per-item review table** (P0.1 no, P0.2a yes, P0.2c no, P0.2b yes, P0.3
  narrowly, P0.4 yes, P0.5 yes, P1.8/P1.12 yes). Replaced in `CLAUDE.md` by the
  rule the table encoded: review a change that establishes a contract other work
  is built on; skip mechanical ones.
