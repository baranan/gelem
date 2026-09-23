# Gelem -- Architecture

What Gelem is and how its parts relate. For the rules that must never be broken,
see `CLAUDE.md`. For video and scale, see `docs/media_architecture.md`, which
supersedes this document wherever they disagree about media.

**Statuses.** This document describes the target architecture. Where the code does
not yet match, the paragraph says so and names the work item, using the same tags
as `CLAUDE.md`: `[NOW]`, `[TARGET -> item]`, `[MIGRATING]`.

---

## 1. Purpose

Gelem is a desktop application for psychology researchers working with collections
of videos, images, and the metadata that goes with them. A typical study might
record videos of 150 participants, split them into trials, and need to browse,
filter, and analyse those pieces alongside study metadata -- experimental
condition, trial number, timestamp, reaction time.

Without Gelem, that means writing a custom Python script for every new study just
to answer "show me all frames from condition A, sorted by reaction time." Gelem
makes that point-and-click.

Gelem also supports studies with no media at all. A researcher can load a CSV of
computed features, run operators on it, and produce plots without a single image
file.

The medium-term goal is a tool usable by undergraduates with very little training,
before they know Python or machine learning.

---

## 2. The seven components

Each has one job and communicates with the others only through defined public
methods. No component reaches into another's internals.

**Dataset** -- The single source of truth. Owns every table and its schema, and is
the only place that may change stored data. The database layer.

**QueryEngine** -- Answers "which rows match these filters, in what order?" Reads
from Dataset, never writes to it. The search and filter engine.

**ArtifactStore** -- Cache of derived images, so the gallery stays fast. Keyed by
media address, not by row. The picture cache. It reads and writes its own cached
files through a narrow `ArtifactCodec`; that is a different operation from decoding
a user's source media, which only the resolver does.

**ColumnTypeRegistry** -- Knows how a *type* is displayed and filtered. It does not
know what type any particular column is; that belongs to the table's schema. The
display rules engine.

**OperatorRegistry** -- Manages analysis plugins and runs them. The analysis
engine.

**AppController** -- The thin wiring layer between the UI and everything else.
Receives events from the UI and calls the appropriate components. The coordinator.

**UI** -- Everything the researcher sees and clicks: gallery, filters, detail view,
menus.

Two more things exist but are not components -- they are values passed to
components:

**MediaAddress** -- the parsed form of a media value. See
`docs/media_architecture.md` §3.

**ProjectPaths** -- where this project's files live. A frozen, Qt-free value
object (`models/project_paths.py`), not a component: `root` (a saved project's
folder, or an unsaved project's workspace folder under the per-user app-data
folder -- never the OS temp directory), `artifacts_dir`, `outputs_dir` (every
operator writes under here, in a subfolder it names for itself -- there is no
operator-specific field), and `is_workspace`. Injected into ArtifactStore and
into every operator run (`run.paths`) so nothing invents its own directory;
`AppController` rebuilds and swaps it on `save_project()`/`load_project()`, at
the same point it already re-roots ArtifactStore. Made true by P1.9a.

`[NOW]` **Copying operator outputs already on disk into the researcher's
chosen folder happens on save.** `save_project()` refuses outright, before
planning, copying or writing anything, while any operator run is live
(`AppController.is_save_blocked()`) -- a live run's own later writes still
land under the OLD `outputs_dir` (it read `run.paths` once, at start), so a
copy planned now would miss them regardless, and the in-memory cell rewrite
below would otherwise give that live run a false "data changed" notice
(`CLAUDE.md`'s "Run provenance and conflict detection", `_superseded_input_tables`).
Otherwise, `models/output_copy.py::plan_output_copy` selects every media cell
whose absolute path lies under the CURRENT `outputs_dir`, and
`execute_output_copy` copies each to the same relative path under the new
one; `Dataset.rewrite_media_cell_paths()` then repoints the matching
in-memory cells at the copies through Dataset's normal accept path, before
`Dataset.save()` writes Parquet. A destination that already exists with the
same size is left alone; a different size is a conflict, and the whole save
is refused before anything is copied or written. Made true by P1.9b-1.
`[NOW]` The researcher-facing warning for a large or colliding copy is
built in `ui/output_copy_warning.py`, called from `ui/main_window.py`'s
save entry point around the same `plan_output_copy()` call. Made true by
P1.9b-2.

---

## 3. Threading model

This is the invariant most likely to be violated by accident, and the one most
often stated imprecisely.

- Tables **owned by Dataset** are mutated on the main thread only.
- A background worker may create and freely mutate a DataFrame of its own. That is
  not a stored table until Dataset accepts it. Forbidding this would forbid
  `create_table()`, which is the normal way an operator works.
- Workers never call Qt and never create a `QPixmap`.
- Workers hand results back through the controller's result queues. Callbacks are
  bound controller methods -- that is the design, not a violation. What workers
  must not do is *read* controller or component state. `[MIGRATING]`, see
  `CLAUDE.md`.
- `AppController` drains those queues on the main thread with a `QTimer`.
  `[TARGET -> P0.2]` The drain must be bounded per tick; today it empties
  everything in one pass, which stalls the UI during a large run.
- The UI never reads a DataFrame directly. It receives data through controller
  signals.

**The practical consequence worth preserving:** operators return results row by
row, the controller applies each one, and tiles repaint immediately. When a run
takes 20+ minutes, results are visible in seconds, mistakes are caught early, and
the run can be cancelled with partial results retained.

**Correction.** Earlier text claimed partial results "survive a stop or a crash."
They survive a **stop**. They are in memory, so a process crash loses them.
Surviving a crash requires periodic checkpointing or a result journal, which does
not exist. See §8.

---

## 4. Tables and schemas

### 4.1 Row identity

Every table has a `row_id` column. It is an **opaque handle** so that the UI, the
controller, and the artifact cache can name a row without knowing anything about
its contents.

That is all it is. It carries no meaning and is unique only **within its table** --
`create_table_from_rows()` deliberately keeps ids when copying rows, because those
rows are the same items.

**What is and is not guaranteed:**

- **Preserved by project save and load.** `save()` writes `row_id` to Parquet and
  `load()` reads it back unchanged, so reopening a saved project keeps every id and
  any reference stored alongside it stays valid.
- **Not guaranteed across re-import or dataset reconstruction.** `load_folder()`
  and `load_csv_as_primary()` reset `_id_counter` and mint new ids. Pointing the
  application at the same folder again may produce different ids for the same
  files.
- **Not unique across tables**, as above.

`[MIGRATING]` Earlier text described row ids as globally unique, never reused and
never changed, which was too strong. A subsequent revision said they were "not
stable across a reload", which was too weak and wrong about save/load. The three
statements above are what the code actually does.

`[TARGET -> P0.2]` Because a bare `row_id` does not identify a row, every public
reference uses `(table_name, row_id)`. `row_updated` and `thumbnail_ready`
currently carry only the id.

### 4.2 Lineage

When one row becomes many -- a video split into segments, a segment into frames --
**the connection is carried by ordinary data columns**, in the same way it would be
in an R data frame:

| row_id | participant_id | trial_id | frame_index | time_within_segment | frame_address |
|---|---|---|---|---|---|
| 004821 | p07 | 12 | 43 | 1.433 | `p07.mp4#f=1450` |

The identifying columns of the source are carried down; the operator adds its own
index. Analyses group and join on those columns.

**What gets carried down** `[TARGET -> P1.6]` -- this must be answered before the
segment and frame operators, or every splitting operator will guess differently.
P1.8 built the `TableSchema` container -- the `role` and `carry_to_children`
fields exist on it -- but did not decide these semantics, and nothing in the code
sets or reads either field yet.

**Two independent properties.** An earlier revision derived carry-down from the
role and got it badly wrong; see the correction note below.

**`role`** -- what kind of thing the column is:

- **`identifier`** -- names the entity this row belongs to (`participant_id`,
  `condition`)
- **`index`** -- this row's position within its parent (`trial_id`, `frame_index`)
- **`measurement`** -- a value observed or computed for this row (`reaction_time`,
  a blendshape score)

**`carry_to_children`** -- a separate boolean: does this value remain true of a row
derived from this one?

- `identifier` and `index` are **always** carried. They are what reconnects the
  pieces, and a split that dropped them would be unusable.
- Everything else **defaults to carried**, and an operator may narrow it with an
  explicit `carry_columns` parameter.

**The default is "carry" on purpose.** A parent row's attribute is constant across
its children, so carrying it down is semantically valid; the only cost is memory
and clutter. Dropping a covariate silently is a research error, and an extra column
is merely wasteful. Fail toward keeping the data -- this is the same principle as
"never sacrifice data for compute" in `docs/media_architecture.md` §6.3.

Narrowing is worth doing when the parent table is wide. Splitting a table that
carries 52 blendshape columns into 530,000 frame rows copies all 52 onto every
row, which is real memory for little gain.

> **Correction (fourth review round).** A previous version said `measurement`
> columns are never carried, on the grounds that "a participant-level mean copied
> onto every frame is meaningless". That is wrong, and wrong in the direction that
> destroys data. **Trial-level reaction time is a measurement, and it is the
> central covariate in the studies Gelem exists to support.** So are participant
> age and trait scores. Dropping them from frame rows would make the frame table
> useless for the analysis it was built for. The redundancy of a repeated
> parent-level value is a storage question, not a correctness one -- and it is
> sometimes exactly what you want, since deviation-from-participant-mean needs the
> mean present on each row.

Roles are set when a table is created and can be corrected by the user, because
Gelem cannot reliably infer which CSV column is a participant identifier. Sensible
defaults on import: the join key of a merge is an `identifier`, other numeric
columns are `measurement`, and `carry_to_children` is true.

When a table has no marked columns at all -- a bare folder load with no metadata --
nothing is carried, and the link to the source still survives in the address
itself, which contains the source path.

Gelem does **not** maintain a separate pointer graph of which row came from which
row. That would duplicate what these columns already say, in a form no analysis
would use. Where a purely structural link is needed -- "this table was produced by
segmenting that table" -- it belongs in the provenance log at table level, not in
a column.

Note that the source path is also embedded in the address itself, so even a
dataset with no metadata at all retains the link to its source file.

### 4.3 Schemas

`[NOW]` for the accept path, project save and load; `[TARGET -> P1.12]` for an
operator that declares its own output schema. Each table has a **`TableSchema`**
owned by Dataset:

- column name
- type tag (`media_path`, `numeric`, `text`, `boolean_flag`, ...)
- dtype
- **role** -- `identifier`, `index`, or `measurement`
- **`carry_to_children`** -- whether a splitting operator copies this column onto
  derived rows. Separate from `role`, because a trial-level measurement such as
  reaction time is both a measurement and essential on every frame row (§4.2).

The `role` and `carry_to_children` fields exist on the `TableSchema` value
object, but nothing in the code sets or reads either one yet: every column is a
`measurement` by default, §4.2 is the authority for what carrying down should
mean, and those semantics are still an open decision -- not settled here.

`[NOW]` `ColumnTypeRegistry` maps a **type tag** to its renderer, filter
control, and display rules. It does not store what type a named column is --
that lives on the column's own `TableSchema` above. `AppController` reads a
column's type tag off that table's schema and asks the registry only what the
tag renders as (`get_column_type()`, `render_column_value()`). `Dataset`
never writes to `ColumnTypeRegistry` and holds no reference to it at all.
Tests: `tests/test_gallery_seam.py`, `tests/test_operator_tag_hints.py`.

The split matters because a project holds many tables and only the second half is
genuinely global. Before P1.8d the registry instead mapped `column_name ->
ColumnType` for the whole project, so a `score` column that was numeric in one
table and text in another silently took whichever was registered last, and
switching tables emitted every column registered anywhere rather than the
active table's columns. That map is gone; `docs/known_defects.md` (Fixed) has
the detail.

Dtypes are set explicitly when a table is created, never inferred, and
**validated by Dataset when it accepts a table** -- not left to each operator to
get right. Defaults: float32 for measurements, int32 for indices, categorical for
repeated strings. Explicit exceptions: presentation timestamps, sample positions,
frame ordinals and counters need int64 or float64, and truncating them to 32 bits
would corrupt time.

`[NOW]` **Inference never narrows.** On an import path -- CSV import, folder
load, merge -- nothing is declared, so `infer_schema` keeps the dtype each
column arrived in: `int64` stays `int64`, `float64` stays `float64`, `bool`
stays `bool`, a text column keeps its arrival text dtype, and a column that
arrived categorical keeps `category`. The narrow defaults just above (float32, int32,
categorical) are guidance for whoever *declares* a schema -- an operator that
creates a table -- and do not license inference to narrow; a caller that wants a
narrower storage dtype passes `ColumnHint(dtype=...)`. Tests:
`tests/test_dataset_schema.py`.

`[NOW]` A media value is canonicalised as it enters storage, so a filename
containing a literal `#` is not demoted to `text`.
`docs/media_architecture.md` §3.6 is the authority for where and when. Made
true by P1.8e. Tests: `tests/test_import_canonicalisation.py`,
`tests/test_accept_canonicalisation.py`.

`[NOW]` **The three pandas text dtype names -- `object`, `string`, `str` -- are
one storage kind.** The schema compares text columns by kind, not by name, so a
text column arriving under one name against a schema declaring another is an
exact match: no adjustment, no conversion. One exception: `object` is the only
name that can physically hold a non-text value, so a column arriving as `object`
against a *different* text spec matches only if every non-null value is a Python
`str` (nulls are skipped; an empty or all-null column counts as text).

`[NOW]` `Dataset.strict_schema` is off in production and on for the whole pytest
suite -- a module-level assignment in `tests/conftest.py`, guarded by
`tests/test_dataset_schema.py::test_suite_runs_with_strict_schema_on`. With it
on, an "unexpected" dtype adjustment -- a width the frame declared that the
schema did not expect -- is refused instead of applied and recorded.

### 4.4 Storage

Tables live in memory as pandas DataFrames and are saved as Parquet. See
`docs/media_architecture.md` §5 for why they stay in memory and what would change
that.

`[NOW]` A saved project folder holds one Parquet file per table plus two JSON
sidecars: `provenance.json` (the provenance log) and `schemas.json` (each
stored table's serialised `TableSchema`, including every column's display type
tag, written since P1.8c-2a). `schemas.json` is written whenever at least one
stored table has a schema -- true for every table that reached storage through
the accept path -- and carries `format_version` 1; an unknown version or a
corrupt file is ignored -- the project still opens, with every table's schema
re-inferred and a message recorded -- and a table the file does not name is
re-inferred the same way. `models/table_schema.py`'s `to_dict` /
`schema_from_dict` are the authority on the per-table encoding.

`column_types.json` (a column-name -> type-tag map covering the whole project)
was retired by P1.8d: `save()` no longer writes it. `load()` still reads it,
but only as a fallback for a project saved before P1.8c-2a that has no usable
`schemas.json`; a project with neither file opens by inference. Tests:
`tests/test_dataset.py`, `tests/test_project_load.py`.

---

## 5. Column types and renderers

Every column has a registered type tag describing how its values display:

- **`media_path`** -- an image, a video, or a piece of one. The renderer
  dispatches internally. The cell's value may be a plain file path or a media
  address carrying a fragment (`#f=`, `#t=`, `#r=`, a stream selector); either
  form gets this tag. `docs/media_architecture.md` §3.6 is the authority on
  the address grammar.
- **`numeric`** -- a number, including timestamps, durations, computed scores.
- **`text`** -- any string. The filter panel shows toggle buttons when there are few
  unique values, a search box when there are many.
- **`boolean_flag`** -- True/False, shown as a tick or cross.

`[NOW]` A type tag an operator declares need not be registered -- it reaches the
column's `TableSchema` either way (P1.8d-2b-2, §4.3 above). An unregistered tag
costs the researcher a placeholder, not a crash: `AppController` prints a
once-per-run warning when `ColumnTypeRegistry` has no renderer for a declared
tag, and the column renders as "Unknown column" until the operator declares a
registered tag instead. `BlendshapeAvatarOperator` and `PlotOperator`, the two
operators that used to hit this with `avatar_path` and `plot_image`, both now
declare `media_path`. Tests: `tests/test_operator_tag_hints.py`.

Renderers live in `column_types/renderers.py` with the signature:

```python
render(value, size, mode='thumbnail', context=None) -> QPixmap | QWidget | None
```

`context` identifies **who is asking** -- table, row, column -- so an asynchronous
result can be delivered back to the right tile. It is not part of the identity of
the picture; see `docs/media_architecture.md` §4.5.

`mode` controls the return type:

- `'thumbnail'` -- for gallery tiles. Always a `QPixmap` scaled to the tile size.
- `'detail'` -- for `DetailWidget`. A `QWidget`: a zoomable view for images, a
  player for videos, a label for text.

`[TARGET -> P0.5]` A renderer never decodes media. On a cache miss it returns a
placeholder and the request is queued. Today `_render_image` falls back to
`Image.open` on the main thread, and `_render_video` decodes with OpenCV on every
paint with no cache at all.

**The key principle:** tiles and detail views never know whether a column holds an
image, a video, or a number. They call `controller.render_column_value()` and
display whatever comes back. All media-type logic lives inside the renderer.

---

## 6. Operators

Analysis plugins the researcher runs from a menu.

An operator is described by three things:

**`OperatorDescriptor`** -- `[NOW]` pure metadata: name, version,
human-readable description, supported execution modes, input requirements,
parameter specifications, output schema, whether results are deterministic and
cacheable, and **the model lifecycle** (`shared` / `per_worker` / `per_sequence`).
The last one is a concurrency contract, not documentation: a tracking model shared
across two clips interleaves state and returns subtly wrong numbers rather than
failing. See `operators/CLAUDE.md`. The Operators menu and the parameter dialog are
**generated from this**, and a run refuses to start for an operator that carries no
descriptor, or none for the requested mode. It is also what makes the planned
natural-language interface possible.

**`OperatorRunSpec`** -- `[NOW]` an immutable description of one run: operation
id, operator name, mode, mode descriptor, parameter values, target table. Built
and validated by `operators/run_context.py`: a value outside a declared
parameter's bounds, or a parameter the descriptor does not declare, raises
`OperatorRunError` at construction rather than reaching the operator.

**`OperatorRun`** -- `[NOW]` the object (`operators/run_context.py`) that
carries runtime services to an operator for one run: the parameter mapping, the
frozen table snapshots it may read, project directories, the cancellation check,
the model the runner built for it, the shared `MediaResolver` (`run.resolver`),
and the result sink. *(An earlier draft of this document called this
`OperatorRunContext` and listed a `resolver` among its services; neither name
was built at the time. `run.resolver` landed at P1.2c-1, the one `MediaResolver`
instance main.py also injects into `ArtifactStore` (section 9); the per-row
COLUMNS runner reads it for a `FRAME` requirement, not the operator itself.
`operators/run_context.py` is the authority for the real shape -- re-verified
17 Sep 2026, P1.2c-1.)*

### The uniform execution signature

`[NOW]` **Every execution method that exists takes the same final argument,
`run`.** There is no other channel by which parameters or runtime services reach
an operator.

```python
create_columns(row_id, media, metadata, run) -> dict
create_table(df, run)                        -> pd.DataFrame
create_display(df, run)                      -> dict
```

`[TARGET -> P2.1]` `iter_column_updates(rows, run)` is declared for shape in
`operators/CLAUDE.md` but is not implemented by any operator yet -- see below.

`run` is an `OperatorRun` carrying:

- `run.spec` -- the immutable `OperatorRunSpec` above
- `run.parameters` -- this run's parameter values, as declared in the descriptor.
  **This replaces `group_by=None` on `create_table`**, which was a hardcoded
  special case for one parameter, and replaces reading values off the operator
  instance.
- `run.cancelled()` -- the cancellation check. `[TARGET -> P1.12f-3]` The token
  behind it exists and is held on `AppController`'s live-run registry, but
  nothing calls `token.cancel()` yet -- see `CLAUDE.md`, "Long-running work".
- `run.paths` -- this project's directories
- `run.model` -- the model the runner built per the mode's `model_lifecycle`,
  or `None`
- `run.emit()` -- the result sink

Those are the whole of the context delivered by P1.12. `[TARGET -> P2.2]`
`run.cache` (result cache) and `run.log` (structured logging) arrive later and are
**optional** -- an operator must work without them, and a P1.12-era `run` will not
have them.

`[TARGET -> P2.1]` `iter_column_updates` is the streaming mode for work that must
walk media in order, yielding progressive updates while decoding each source
sequentially. Independent per-row work keeps using `create_columns`. A generator
gives progressive output and a cancellation point, **not resumability** -- that is
defined per mode in `docs/media_architecture.md` P2.2.

`[NOW]` **`ProjectPaths` is never stored on the operator.** It arrives as
`run.paths`. An operator is a singleton shared across runs, so anything
per-run held on `self` is a race between concurrent runs -- the same defect
as parameters on `self`, and the reason `run` exists at all. Made true by
P1.9a, which removed the `output_dir` constructor argument (and the
`self._output_dir` it fed) from every operator that had one; each now writes
under `run.paths.outputs_dir`, in a subfolder it names for itself.
`operators/CLAUDE.md`'s "Write only to `run.paths.outputs_dir`" rule is the
authority. *(The other consequence this section used to list -- `image`
becoming `media` -- is done: every operator's `create_columns` takes `media`,
not `image`.)*

`[NOW]` **Operator modules contain no Qt.** `CLAUDE.md`, "Data ownership", is the
authority for this rule and its guarding test; not restated here.

Full contract and template: `operators/CLAUDE.md`.

### Run provenance and conflict detection

`[NOW]` Every operator run appends one `"operator_run"` entry to
`ProvenanceLog` (`Dataset.record_operator_run()`, made true by P1.12f-1). The
entry carries the operator name, the execution mode, the mode's label, the
parameter values, the target table (COLUMNS mode only), every input table this
run read together with the write-ticket version it was at when the run
started, how many rows were requested and how many were actually applied, any
row ids a result could not be placed back onto, `superseded_tables` (see
below), and `cancellation_requested` (see below). `models/dataset.py`'s
`record_operator_run()` docstring is the authority for the exact field list;
this is a summary, not a restatement.

`outcome` is one of three values:

- `"complete"` -- every requested row produced a result and was applied.
- `"partial"` -- the run produced less than it was asked for: an error
  partway through, a result that could not be placed back onto a row that no
  longer exists, or -- since P1.12f-3 -- a cancelled run that actually fell
  short. For COLUMNS mode, falling short means applying fewer rows than were
  requested. A single-shot TABLE or DISPLAY run that was cancelled is always
  `"partial"` for a different reason: it cannot be interrupted mid-computation
  (`CLAUDE.md`, "Long-running work", is the authority for what Cancel does and
  does not interrupt), so it runs to completion and its result is discarded
  on arrival rather than stored or shown -- discarding the whole result is
  producing nothing.
- `"failed"` -- the run produced no applied results at all.

Cancellation and outcome answer two different questions, and the entry keeps
them separate. `cancellation_requested` records the fact that the researcher
clicked Cancel (`AppController.cancel_run()`); `outcome` records what the run
actually produced. A COLUMNS run that had already applied every row it was
asked for by the time the click reached the controller reads `outcome:
"complete"`, `cancellation_requested: true` -- clicking Cancel does not
retroactively make a finished result incomplete. See `docs/known_defects.md`
for the open question of whether an UNcancelled run that silently produced
less than it was asked for should be held to the same "partial" standard.

Not yet in the entry, and still open: operator and model **version**, and a
cache identity for the run -- both wait on `P2.2`'s caching design (§8).

**Supersession needs two checks, not one.** A run's input tables can move
under it while it is still running, and each check catches a different moment
a foreign write can land (`AppController._superseded_input_tables`):

1. **At arrival**, compare a table's current write-ticket version against the
   version this run **itself** last caused there (or, for a table it never
   wrote to, the version recorded at run start). This is what stops a COLUMNS
   run's own final write from making it look superseded by itself.
2. **Before each of the run's own writes**, latch the same comparison at that
   earlier moment. Without this, a foreign write landing between two of the
   run's own drain ticks is masked: by arrival time the run's own later write
   is the last thing that happened to the table, so check 1 alone would see
   nothing wrong.

The standing ruling behind both checks: **a false staleness note is advisory
and cheap; a missed one is a wrong number in a paper.** The two checks combine
with OR, not AND, so the rule errs toward over-reporting.

**The pre-emptive write/read warning** fires before a new run starts, if it
would read a table a live run is still writing, or write a table a live run is
still reading (`controller.write_read_conflict_warnings()`, P1.12f-2). It
compares at **table granularity, not row granularity**: the set of tables the
new run reads and writes against the set each live run reads and writes, so it
can warn even when the two runs' actual row selections do not overlap at all.
That over-warning is deliberate, for the same reason supersession errs toward
over-reporting -- a false pre-emptive warning costs a dismissed dialog; a
missed one costs a silently wrong result.

Blendshape extraction is one operator among several, not the point of the
application. Most operators work on ordinary numeric and tabular data.

---

## 7. Layout

```
models/          Dataset, TableSchema and QueryEngine
media/           MediaAddress, MediaResolver, PlaybackAdapter   [TARGET -> P1]
artifacts/       ArtifactStore
column_types/    ColumnTypeRegistry and render functions
operators/       Analysis plugins
ui/              All PySide6 widgets
ui/tiles/        Gallery tile types
shared_widgets/  Display components shared between UI and renderers
tests/           Functional and guardrail tests
docs/            This document and the media architecture plan
docs/archive/    Student-era guides, kept as history                [TARGET -> P0.1]
controller.py    AppController
main.py          Entry point
operators_config.yaml
```

`[NOW]` `operators_config.yaml` is the single authority for **which** operators
the application offers, and its entry order is the Operators menu order.
`operators/operator_config.py` reads the file and constructs the enabled
entries; `main.py` holds only the per-operator factories
(`OPERATOR_FACTORIES`) -- the knowledge of *how* to build each one -- and
registers whatever the file enables. Drift in either direction (an enabled
YAML entry with no factory, or a factory the file never names), a missing
file, or malformed YAML raises `OperatorConfigError` at startup and stops the
program rather than silently changing what the researcher can do. Made true
by P1.11a. Guarded by `tests/test_operator_config.py`
(`test_yaml_keys_equal_factory_keys`, plus the `build_enabled_operators`
drift tests). This is the authority for operator registration; other
documents point here.

---

## 8. Open architectural work

- **Media handling** is being reworked. See `docs/media_architecture.md`. Media
  values become addresses into source files rather than paths to extracted files.
- **Dataset access paths** are quadratic in the common case, and the artifact cache
  cannot distinguish two media columns on one row. These are Phase 0; see
  `docs/media_architecture.md` §6.
- **Merging** is too narrow. `Dataset.merge_csv` hardcodes a join onto the `frames`
  table against `file_name` and rejects one-to-many joins, which blocks merging
  trial-level data onto a participant video row. Needs generalising to arbitrary
  table, arbitrary key, with explicit expand semantics.
- **Visible-row and visible-column ownership.** `MainWindow` reconstructs state by
  interrogating widgets instead of `AppController` owning it. Two distinct
  problems, and the fix splits ownership rather than moving all of it: **the
  controller owns the ordered query result** (which rows, in what order -- data),
  while **the gallery keeps viewport geometry and reports the displayed index
  range** into that order. Moving layout knowledge into the controller would be the
  opposite mistake. Separately, visible-column state must distinguish `None` from
  `[]`. **This is Phase 0 (P0.4), not Phase 1** -- demand-driven rendering has to
  know what is on screen in order to prioritise and cancel, so it cannot be built
  while that knowledge lives in a widget's private list.
- **Provenance and reproducibility.** §6 ("Run provenance and conflict
  detection") is now the authority for what an operator run's provenance entry
  contains -- not restated here. What is still missing, and still blocks full
  reproducibility: operator and model **version** are not in the entry, filters
  and sorts are not recordable actions at all, and there is no cache identity.
  Session export as standalone Python is a renderer over the provenance log
  once a single "recordable action" abstraction covers runs, filters and sorts.
- **Crash recovery.** Progressive results survive cancellation, not a crash. If
  crash recovery is wanted, it needs periodic checkpointing or a result journal.
  Not currently planned; noted so the claim is not made loosely.
- **Natural-language terminal.** Planned after v0.1. `OperatorDescriptor` metadata
  is the reliability lever, which is why it is mandatory now rather than later.
- **Avatar rendering.** Turning blendshape values into a deformed 3D face. Blocked
  on a usable VRM source. Will attach through a column contract defined once that
  is solved.

---

## 9. Settings

**Single authority for the machine-tunable values.** Nothing else -- not
`media_architecture.md` §4.7, not `CLAUDE.md` -- restates this list. Added
P0.5b-2ii-c1. The editing dialog landed P0.5b-2ii-c2b2
(`ui/settings_dialog.py`, reached from **File -> Settings...**); a researcher
changes any of these values there. The two byte values are shown and edited in
MiB, and the dialog submits only the fields that actually changed.

### The six values

| Value | Default | Meaning | Takes effect |
|---|---|---|---|
| `picture_memory_max_bytes` | 500 MiB | Ceiling on the RAM the ArtifactStore's in-memory decoded-image cache may hold. Over it, the least recently used images are dropped; they regenerate from disk on next view. | **Immediately** |
| `picture_disk_max_bytes` | 1 GiB | Ceiling on the total size of the derived-JPEG files in a project's `artifacts/` folder. Over it, the oldest (by write time) are deleted and regenerate on demand. | **Immediately** |
| `worker_count` | 2 | How many background threads decode and resize source media for thumbnails and previews. Higher uses more CPU and RAM for faster gallery fill. | **On restart** |
| `max_open_decoders` | 6 | Ceiling on how many source video files the shared `MediaResolver` (media/resolver.py) may have open for decoding at once, across every caller -- `ArtifactStore`'s workers and the per-row COLUMNS runner alike, since P1.2c-1 they share the one resolver instance. Higher lets more files decode in parallel at the cost of more open file handles and memory. | **On restart** |
| `thumbnail_max_side` | 150 | Largest side, in pixels, of a gallery thumbnail. This number is the "thumbnail resolution" that enters the artifact key and decides, per tile, whether a tile asks for a thumbnail or a preview. | **On restart** |
| `preview_max_side` | 600 | Largest side, in pixels, of the larger preview image used for bigger tiles and quick previews. | **On restart** |

Each size is a single number, not a width-and-height pair: only the larger side
ever entered the artifact key, so a pair such as `700x100` could pass validation
while describing a picture whose real short side was nowhere near the resolution
the key claimed.

Bounds and the exact defaults live in `settings/settings.py` as module-level
constants; a saved value outside its bounds is clamped, an unparseable one falls
back to the default, and either way the app still starts. One cross-field rule:
if `preview_max_side` is smaller than `thumbnail_max_side`, the preview size is
set equal to the thumbnail size.

### Why some need a restart

The memory and disk ceilings are read every time the cache is checked or swept,
so a setter can lower them and immediately evict down to the new bound
(`ArtifactStore.set_memory_cache_max_bytes`, `set_disk_cache_max_bytes` -- each
does the eviction itself, it is not left to the caller).

Worker count is fixed when the `WorkerPool` builds its threads. Thumbnail and
preview sizes are read by worker threads without a lock, which is only safe
because they are written once in `ArtifactStore.__init__` and never mutated --
see the comment there. `max_open_decoders` is fixed when `MediaResolver` builds
its decoder pool (media/resolver.py's `_DecoderPool`). Changing any of these
four therefore needs a fresh process.

### How a value reaches a component

`main.py` builds a `QSettingsBackend`, wraps it in a `SettingsStore`, calls
`load()`, prints any correction messages, and passes the **plain values** into
the `ArtifactStore` constructor -- exactly as `worker_count` and
`disk_cache_max_bytes` were already passed. `max_open_decoders` is the one
exception to "straight into `ArtifactStore`": it builds the single
`MediaResolver` instance (P1.2c-1), which `main.py` then injects into both
`ArtifactStore` and `AppController` as a REQUIRED constructor argument -- no
default, no `None` fallback, so a caller that forgets it fails to construct
rather than silently decoding nothing. **No component receives the
`SettingsStore` or a `GelemSettings` object. `AppController` receives a
`SettingsGateway` and only passes calls through to it. No component imports
`settings/`.** Only `main.py` and the `settings/` package may import `settings/`
or `QSettings` (guarded by `tests/test_settings.py`). `settings/` is Qt-free
except `settings/qsettings_backend.py`, the one file that touches PySide6.

### The editing face

`SettingsGateway` (`settings/settings_gateway.py`, added P0.5b-2ii-c2b1) is the
plain-data face a settings dialog edits through. It is Qt-free and holds no
knowledge that a dialog exists. Two methods: `describe_fields()` returns one
`SettingField` per value -- name, label, help text, bounds (read from the
`*_RANGE` constants, never retyped), unit, restart-required flag, current value
-- and `save_values(mapping)` takes a **partial** update: it overlays the
caller's mapping onto the values currently in the store (a field the mapping
does not name keeps its current persisted value), runs the overlaid full mapping
through `GelemSettings.from_values` (so an out-of-range or unparseable value is
corrected and the preview-not-smaller-than-thumbnail rule is applied before
anything is written), persists the result through the store, and returns the
list of correction messages.

`main.py` builds the gateway over the same `SettingsStore` and passes it to
`AppController` as `settings_gateway=`. `AppController.get_settings_fields()` and
`apply_settings(values)` forward to it; `apply_settings` additionally pushes an
immediate-effect ceiling into the `ArtifactStore` only when that ceiling's value
actually changed -- the value the gateway reports before the save is compared
with the value it reports after, so a number the gateway corrected straight back
to what was already stored pushes nothing (the memory push evicts and the disk
push runs a full sweep, both on the main thread). A change to the disk ceiling in
either direction runs a sweep -- the cache can sit over its ceiling between
sweeps, so even raising the limit can evict genuinely cached pictures -- and when
that sweep deletes cached picture files `apply_settings` appends one
plain-English sentence naming the count to the messages it returns; the count
includes orphan cleanup as well as ceiling eviction, so it is not an exact
measure of what the researcher lost. The dialog that drives this,
`ui/settings_dialog.py` (P0.5b-2ii-c2b2), builds a spin box per field from
`get_settings_fields()` -- byte fields in MiB, the rest in their native unit --
records each starting value, and on OK passes `apply_settings()` only the
entries whose spin box moved. When any changed field is restart-only (the
worker count or either size) it first shows a three-button confirmation --
"Save and quit Gelem", "Save and keep working", "Cancel" -- and, when a size
changed, that confirmation also warns that every existing thumbnail and preview
becomes unreachable and regenerates from source. After the save it re-reads the
fields and shows any messages `apply_settings()` returned, including the
sentence about swept picture files, before quitting if quit was chosen.
