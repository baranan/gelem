# Writing an operator

An operator is a self-contained analysis plugin the researcher runs from the
Operators menu. It inherits from `BaseOperator`, describes itself, and implements
one or more execution methods. The menu is built from the description.

`operators/base.py` is the contract.

**Statuses** follow `CLAUDE.md`: `[NOW]` is true today, `[TARGET -> item]` is not
yet true and must be by the named item, `[MIGRATING]` has listed exceptions.

`[TARGET -> P1.12]` **There is currently no good reference operator.**
`operators/thumbnail.py` used to be described as one but was dead code -- it
was deleted in P0.5b-2i (`ArtifactStore` generates thumbnails inline in
`_run_job`, off a bounded worker pool). This is deliberately deferred, not an
oversight: operators differ in what they return -- new columns, a new table,
a result display, or a combination -- and in whether they need a parameter
dialog, so no single example can serve as the reference. A set of examples is
worth writing once a few real operators exist and P1.12 has settled the
operator contract. There are no students yet, so nothing depends on it now.

---

## Describing an operator

`[TARGET -> P1.12]` Every operator carries an **`OperatorDescriptor`**: name,
version, a human-readable description, which execution modes it supports, what
input it needs, what parameters it takes, what columns or results it produces, and
whether its output is deterministic and cacheable.

This is **mandatory, not optional**, for three reasons:

1. The Operators menu and the parameter dialog are generated from it.
2. Results cannot be cached correctly without the version and determinism
   information -- see "Caching" below.
3. A natural-language interface is planned, and this metadata is what will make it
   reliable. Filling it in later is far more expensive than writing it now.

### Parameters are declared, not drawn

`[TARGET -> P1.12]` An operator declares what its parameters *are* -- name, type,
allowed values, whether they name a column, default. The UI generates the dialog.

**Operator modules contain no Qt.** Today `plot_advanced.py`, `video_frames.py`,
`plot_operator.py` and the example in `base.py` construct `QDialog` and
`QComboBox` directly. That puts UI code in the analysis layer and makes an
operator impossible to run from a test, a script, or a future terminal.

**Parameter values never live on `self`.** They arrive for one run in
`run.parameters` -- a read-only mapping backed by an immutable `OperatorRunSpec`.
Read every per-run value from there, and never store one as an attribute on the
operator instance or read one back off it. Operators are singletons, so a value
on `self` is a race: two concurrent runs of the same operator would overwrite
each other's parameters.

---

## The execution methods

`[TARGET -> P1.12]` **All four take the same final argument, `run`.** It is the
only channel by which parameters and runtime services reach an operator.

```python
create_columns(row_id, media, metadata, run) -> dict
iter_column_updates(rows, run)               -> Iterator[tuple[str, dict]]
create_table(df, run)                        -> pd.DataFrame
create_display(df, run)                      -> dict
```

`run` gives you:

| | |
|---|---|
| `run.parameters` | this run's parameter values, as declared in `parameters` |
| `run.cancelled()` | check between units of work; return promptly if true |
| `run.resolver` | the only way to decode media |
| `run.model` | `[NOW]` the model the COLUMNS runner built for this run per the mode's `model_lifecycle`; `None` for `NONE`, and `None` in TABLE/DISPLAY modes (no runner builds one there yet). **Never build or cache a model on `self`** -- see "Where a model lives" |
| `run.paths` | this project's directories; **never store these on `self`** |
| `run.spec` | the immutable description of the run, including versions |
| `run.emit()` | the result sink |

`[TARGET -> P2.2]` `run.cache` and `run.log` arrive later and are **optional**.
Guard their use -- an operator must still work on a `run` that has neither.

Everything per-run comes from `run` and nothing per-run is stored on the operator.
An operator is a singleton shared across runs, so a value on `self` is a race
between two concurrent runs. That applies to parameters, to `ProjectPaths`, and to
anything else that differs run to run.

### Where a model lives

`[NOW]` **Never build or cache a model on the operator singleton.** "Load the
model once in `__init__`" is the obvious optimisation and it is wrong for exactly
the models Gelem runs. MediaPipe landmarkers hold internal state, are not
documented as thread-safe, and in tracking mode carry the previous frame's result
forward. Two runs processed in parallel through one shared model object would
interleave that state, and the symptom is subtly wrong numbers rather than a crash.
An operator is a singleton, so a model on `self` is shared by every concurrent run
whatever lifecycle the descriptor names.

Declare the lifecycle in the mode descriptor's `model_lifecycle`
(`operators/descriptor.py` -> `ModelLifecycle`) and provide a **factory**,
`build_model(self)`. The runner -- `OperatorRegistry`, the one component that
builds models -- calls the factory and hands the result back as **`run.model`**.
That is the one channel, for every lifecycle; an operator never builds or caches
a model itself.

**Today only the per-row COLUMNS runner honours this** (`_run_create_columns_worker`).
`run_create_table` and `run_create_display` build no model and leave `run.model`
as `None`; no TABLE or DISPLAY operator needs one yet. An operator in those modes
that comes to need a model extends the same mechanism to its runner rather than
loading one on `self`.

| `ModelLifecycle` | For | Who builds it, how often |
|---|---|---|
| `NONE` | the mode uses no model | nobody; `run.model` is `None` |
| `SHARED` | immutable, *demonstrated* thread-safe -- lookup tables, config, pure functions | the COLUMNS runner, once per application, cached on `OperatorRegistry` under a `threading.Lock`; reused across every run (the cache is not cleared on a project switch -- fine only while SHARED stays project-independent) |
| `PER_WORKER` | stateless-per-call inference that is expensive to load | the COLUMNS runner, once per worker, inside the worker before the row loop; the same instance serves every row of that run |
| `PER_SEQUENCE` | anything that tracks state across frames | nobody yet -- declared but refused, see below |

`SHARED` requires evidence, not assumption. If you do not know, choose
`PER_WORKER`; if the model tracks across frames, `PER_SEQUENCE` is the only correct
answer and P2.2's per-clip-run cache identity depends on it.

```python
# The runner calls this as often as the declared lifecycle requires:
# once per worker for PER_WORKER, once per application for SHARED, never
# for NONE. Raise OperatorSetupError here if a prerequisite -- an
# undownloaded model file -- is missing: the run aborts through
# on_setup_error before any row is processed.
model_lifecycle = ModelLifecycle.PER_WORKER   # in the ModeDescriptor

def build_model(self):
    return load_landmarker()
```

`[TARGET -> P2.1]` **`PER_SEQUENCE` is declared but refused today.** The per-row
COLUMNS runner sees one row at a time and has no sequence boundary to reset a
tracking model at, so `AppController.run_create_columns` refuses a run whose mode
declares it -- before any worker starts, in a message naming the operator and the
lifecycle -- and the worker keeps a defensive guard for one that slips through. An
operator that needs tracking state waits on the ordered-group runner
(`iter_column_updates`).

Building the model before the row loop also means a missing prerequisite is
reported before any work starts. The old lazy load inside `create_columns` only
raised on the first row whose media happened to decode; a run where every row
failed to decode reported success having processed zero rows and never mentioned
the missing model.

Guarded by `tests/test_model_lifecycle.py`.

### `create_columns(row_id, media, metadata, run) -> dict`

Runs **once per row, in a background thread**. Returns new column values for that
one row. `AppController` applies each dict to `Dataset` on the main thread, and the
tile repaints immediately.

Use for per-item work where rows are independent. `media` is a typed payload, not
necessarily a still image, so the same method serves video spans and audio later.
Declare what you need as the mode descriptor's `media_requirement`
(`operators/descriptor.py` -> `MediaRequirement`): `FRAME` gets a decoded frame,
`METADATA` and `ADDRESS` get `media = None`. There is no `requires_image` boolean
-- it was removed in P1.12d-2b-1.

Inside this method, use **only** the arguments given. Never touch `Dataset`, the
controller, or any Qt object. Never modify `metadata`.

### `iter_column_updates(rows, run) -> yields updates`

`[TARGET -> P2.1]` (run context from P1.12) Runs **once over an ordered group of
rows**, in a background
thread, yielding results progressively.

Use this when the work must walk media in order: sequential decoding is roughly an
order of magnitude faster than seeking to individual frames, and tracking modes
require frames in increasing-timestamp order. This cannot be expressed as repeated
independent calls to `create_columns`.

The serial implementation is the reference and must always be present and correct.
Parallelism is a strategy wrapped around this same contract, with automatic
fallback to serial if a worker fails. A user on an unusual machine gets slow but
correct results, never a broken application.

### `create_table(df, run) -> pd.DataFrame`

Runs **once with a DataFrame**, in a background thread. Returns a new DataFrame
that `AppController` stores as a new named table.

Work on a copy. Do not add `row_id` -- it is generated when the table is stored.

Mutating the DataFrame you created is fine, including in a worker thread. It is
yours until Dataset accepts it. What you must not do is mutate a DataFrame you
were handed.

Grouping comes from `run.parameters["group_by"]`. The old `group_by=None` argument
was a hardcoded special case for one parameter, which is exactly what a declared
parameter schema exists to avoid.

### `create_display(df, run) -> dict`

Runs **once with the selected rows**, in a background thread. Returns a result dict
shown in the Results panel. Never stored in any table.

Keys: `operator_name`, `artifact_path` (a PNG), `html_path` (an interactive page),
`summary` (a nested `{column: {stat: value}}` dict rendered as a table).

Note a trap. `summary` must be wrapped under that key; returning the nested dict
directly produces no visible output. The interactive-plot key is `html_path`
everywhere -- `base.py` documents it and `ResultsPanel` reads it (P1.11b removed
the old `plot_html` fallback).

---

## Rules

- **`[NOW]` Never call Qt** in any execution method. They all run off the main
  thread.
- **`[NOW]` Never read from or write to `Dataset` or the controller.** Return
  results.
- **`[NOW]` Never modify an input DataFrame.** Always `df.copy()`.
- **`[NOW]` Fail gracefully per row.** If a face is not detected, return `None`
  values rather than raising. One bad row must not kill a run.
- **`[NOW]` Produce files, return paths.** Save the image or HTML and return its
  path as a string. Do not return image data.
- **`[TARGET -> P1.9]` Write only to `run.paths`**, never into a source data folder
  and never to a path you chose yourself, and never to a path cached on `self`.
  Operators currently default to a global temp folder and store it as
  `self._output_dir`; older documentation called this `self.output_dir`, which
  never existed.
- **`[NOW]` A declared type tag need not be registered, but an unregistered one
  costs the researcher a placeholder.** The tag still reaches the column's
  `TableSchema`; `AppController` only prints a once-per-run warning when
  `ColumnTypeRegistry` has no renderer for it, and that column renders as
  "Unknown column" instead. Declare a registered tag -- reuse `media_path`,
  `numeric`, `text` or `boolean_flag`, or register a new one -- so the column
  actually displays. Tests: `tests/test_operator_tag_hints.py`.
- **`[TARGET -> P1.8]` Declare intended types; do not enforce dtypes yourself.**
  Dataset validates and normalises a table against its schema when it accepts it.
- **`[TARGET -> P1.2]` Never open a media file.** Use the resolver -- see
  `docs/media_architecture.md`.
- **`[TARGET -> P1.12]` Long runs must be cancellable, keeping partial results.**
  **No cancellation exists today** -- there is no token and no check anywhere in
  the operator loop, so a started run always runs to completion. Partial results
  will survive cancellation, but never a process crash.
- **`[TARGET -> P2.2]` Resumability is a separate, narrower promise.** See
  "Resuming a run" below. Do not state the two as one property.

---

## Media and splitting

Frames and clips are **addresses into source files**, not extracted files.
Operators that split video produce rows holding addresses and write no media
files. Exporting real files is a separate, explicit user action.

### Carry the lineage columns

An operator that turns one row into many must carry the source's identifying
columns down and add its own index. This is how everything downstream reconnects
the pieces -- ordinary tidy-data columns, not hidden pointers.

**Which columns those are is not a judgement call.** `[TARGET -> P1.8]`
`TableSchema` gives each column **two independent properties**:

| Property | Values | Meaning |
|---|---|---|
| `role` | `identifier` | names the entity the row belongs to (`participant_id`) |
| | `index` | position within the parent (`trial_id`, `frame_index`) |
| | `measurement` | a value observed or computed for this row (`reaction_time`) |
| `carry_to_children` | bool | whether a split copies this onto derived rows |

**Carry-down is not derived from the role.** A trial's reaction time is a
`measurement` and must appear on every frame row of that trial -- it is the
covariate the analysis turns on. Participant age and trait scores are the same.
Treating "measurement" as "do not carry" would silently gut the frame table.

The rule:

- `identifier` and `index` are **always** carried, and you add a new `index` at
  your own level.
- Everything else is carried when `carry_to_children` is true, **which is the
  default**. Dropping a covariate silently is a research error; an extra column is
  only wasteful. Fail toward keeping the data.
- Narrow it with an explicit `carry_columns` parameter when the source is wide --
  copying 52 blendshape columns onto 530,000 frame rows is real memory for little
  gain.
**Initial flags on a bare folder load.** "Everything defaults to carried" and
"nothing is carried when nothing is marked" pull against each other, so state the
starting point explicitly. `load_folder()` creates three columns and they are not
all alike:

| Column | Role | Carried? |
|---|---|---|
| `row_id` | not a data column | **no** -- children get their own |
| `full_path` | `media` | **no** -- superseded, the child derives its own address from it |
| `file_name` | `identifier` | **yes** -- tells a frame row which video it came from |

So a bare folder load carries `file_name` and nothing else, which is the useful
answer rather than "nothing". The source path also survives inside the address
itself regardless.

**Segment rows** carry: the source media address, segment start, segment end,
duration, a segment index, and **every `identifier` and `index` column from the
source plus every other source column flagged `carry_to_children`**.

**Frame rows** carry: the frame address, the source frame ordinal or presentation
timestamp, absolute time in the source, `time_within_segment`, the segment index,
and **the same carried set**.

Both summaries mean the full rule above, not just the identifying columns. Reaction
time, participant age and any other trial- or participant-level covariate travel
with the split.

`time_within_segment` belongs on **frame** rows. A segment row has no single time
within itself. Earlier text asked the segment operator to emit it, which was
incoherent; the point stands that nothing currently produces this column and the
windowed analyses need it.

### Segment thumbnails are not the operator's job

**Revised 4 Aug 2026, then again 26 Aug 2026 -- see `docs/media_architecture.md`
§4.1b, which is authoritative on this mechanism; the summary below points at it
rather than restating it in full.** An earlier version said a segment operator
must capture each segment's representative frame "during the sequential pass it
is already making." That assumed segmentation decodes video. **A metadata-driven
segmentation -- start and end columns from a trial CSV, which is the common
case -- decodes nothing.** There is no pass to piggyback on.

A later version said this batch job should make one full sequential decode pass
per video instead, on the assumption that seeking to each segment separately
would be far more expensive. The measurement pass (26 Aug 2026) found the
opposite by two to three orders of magnitude at any realistic trial density --
see `docs/media_architecture.md` §4.1b and §10 for the numbers.

So exact segment thumbnails are an **ArtifactStore batch job**: collect the
outstanding segments, sort by source file and start time, and **seek to each
representative frame**. No full sequential pass. An operator that happens to be
decoding anyway may *offer* a decoded representative frame, but never writes
into ArtifactStore itself.

The requirement is unchanged and still guarded by a test: **a segment's thumbnail
must come from inside that segment's own time range.** A frame from the wrong
segment -- from a sorting or off-by-one bug in which seek result gets attached to
which segment -- looks entirely plausible and would otherwise go unnoticed.

### Walk media in order

Sequential passes over a video are roughly an order of magnitude faster than
seeking to individual frames -- **when the operator needs every frame, or most of
them, in a contiguous span.** Operators that process video that way should walk
it in order rather than requesting frames one at a time. This is what
`iter_column_updates` exists for.

**This is not the segment-thumbnail case above, and the two should not be
conflated.** A segment-thumbnail batch job wants one frame per segment, often
scattered widely across an hour-long video -- a much sparser access pattern, and
measurement (§4.1b) found seeking wins there, sharply. The rule of thumb is about
density of frames actually needed, not about video decoding in general.

Read `docs/media_architecture.md` before writing anything that touches video.

---

## Caching results

`[TARGET -> P2.2]` A cached analysis result is only reusable if everything that
could have changed the number is in the key. For scientific output that means at
least: the media address, a fingerprint of the source file, the operator name and
version, the model version, the relevant library version, the parameter values,
the sampling policy, and any colour or orientation preprocessing.

Caching by frame address alone is not enough. A MediaPipe upgrade would silently
serve results from the old model.

**Tracking mode is different in kind.** When a model reuses the previous frame's
state, the result is not a function of the frame address at all -- it depends on
the preceding frames and on where tracking state was last reset. Tracking-mode
results are therefore cached **per clip-run, not per frame**. Caching them per
frame with a longer key would still let a resumed run mix results from two
different tracking histories, which produces a plausible but irreproducible number.

A "clip-run" must be a **deterministic sequence identity** -- source and range,
reset-boundary policy and starting boundary, the exact ordered frame set, operator
and model versions, parameters and preprocessing. **Never a run UUID.** A UUID
would guarantee a miss on every subsequent run, quietly disabling the cache rather
than breaking anything visibly.

## Resuming a run

`[TARGET -> P2.2]` A generator gives progressive output and a place to check for
cancellation. **It does not give resumability.** State the promise per mode:

- **Independent-frame mode** resumes by skipping frames already in the cache.
  Exact and cheap.
- **Tracking mode** resumes only from a **reset boundary**. Restarting mid-clip
  from a cold model is not a resumption; it produces different numbers.
- **Mid-clip continuation** would require serialising model state, which MediaPipe
  may not expose. Do not plan on it until it is shown to.
- Practical default: replay from the last reset boundary and suppress outputs
  already stored. This costs recomputation and buys reproducibility, which is the
  property that matters for published analysis.

Segment boundaries are the natural reset boundaries, which is one more reason
segments are a first-class row type rather than a display convenience.

---

## Generality

Before building an operator, name the parameter that makes it general. Study-
specific vocabulary inside a general component is a leak:

- not "extract trial number", but "read a marker from a region"
- not "split into trials", but "split by start/end columns"
- not "average the 300-1500 ms window", but "average over a parameterised window"

A hardcoded emotion count, time window, or blendshape name inside a generic
operator is a defect.

**And a number that does not generalise must be a setting or a runtime
measurement, never a constant.** Worker counts, batch sizes and cache limits differ
between a development machine and a student's 8 GB laptop. Default low.

---

## Template

A complete, runnable **COLUMNS** operator. It constructs today: copy the class,
change the names, and it will start. Two things look like mistakes and are not:

- **`media` is `None` unless the mode declares `media_requirement =
  MediaRequirement.FRAME`.** `METADATA` and `ADDRESS` both hand `create_columns`
  a `media` of `None`; the per-row runner refuses `VIDEO_SPAN` and `AUDIO_SPAN`
  before the run starts.
- **Parameters are never stored on `self`.** An operator instance is a singleton,
  so a value on `self` is a race -- two concurrent runs would overwrite each
  other. Every per-run value is read from `run.parameters`.

```python
from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    NumberParameter,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)


class MyOperator(BaseOperator):
    """One-line description shown to the researcher."""

    # ---- Identifier ---------------------------------------------------
    # The one non-descriptor attribute. `name` is the operator's unique
    # key -- in operators_config.yaml, in OPERATOR_FACTORIES, and in
    # OperatorRegistry. It must equal `descriptor.name` (checked when the
    # descriptor is built). Everything else the menu and the runner need
    # -- the menu label, the declared output columns -- lives only in the
    # descriptor below.
    name = "my_operator"

    # ---- Descriptor (mandatory: the run will not start without it) ----
    # AppController builds every run from the mode's ModeDescriptor and
    # refuses to start an operator that carries no descriptor (or none for
    # the requested mode). Since P1.12d-3 the descriptor is also the only
    # source of the menu label and the output columns.
    descriptor = OperatorDescriptor(
        name="my_operator",              # must equal the `name` attribute
        version="1.0",                   # part of the result-cache key --
                                         # bump it whenever a change could
                                         # alter the numbers produced
        description=(
            "For each row, computes my score from the face in the frame "
            "and writes it to the numeric column 'my_score'."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                # The Operators-menu entry for this mode. Your wording --
                # no test pins it.
                label="Compute my score",
                # Where the rows come from. ACTIVE_TABLE means "whatever
                # table is on screen" -- no dialog choice needed.
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                # What the runner decodes per row before calling
                # create_columns(). FRAME -> `media` is a decoded frame;
                # METADATA or ADDRESS -> `media` is None. VIDEO_SPAN and
                # AUDIO_SPAN are refused for a per-row COLUMNS run.
                media_requirement=MediaRequirement.FRAME,
                # Declared parameters. The researcher sets them in the
                # dialog built by get_parameters_dialog() below; the
                # operator reads their values from run.parameters.
                parameters=(
                    NumberParameter(
                        name="threshold",
                        label="Detection threshold",
                        minimum=0.0,
                        maximum=1.0,
                        decimals=2,
                        default=0.5,
                    ),
                ),
                # The columns this mode writes, one OutputColumn each.
                # Reuse a registered tag -- numeric, media_path, text,
                # boolean_flag -- or the column shows a placeholder.
                # tests/test_operator_descriptors_match.py pins each real
                # operator's (name, tag) pairs.
                output=OutputSpec(
                    columns=(
                        OutputColumn(name="my_score", type_tag="numeric"),
                    ),
                ),
                # This template uses no model, so build_model() is not
                # overridden and run.model is None. An operator that needs
                # one declares PER_WORKER or SHARED here and returns it
                # from build_model(); the runner builds it and hands it in
                # as run.model. Cross-frame tracking state needs
                # PER_SEQUENCE, which is declared but refused today -- see
                # "Where a model lives".
                model_lifecycle=ModelLifecycle.NONE,
                # Same inputs + parameters always give the same output, so
                # a cached result may be reused.
                deterministic=True,
                cacheable=True,
            ),
        ),
    )

    def create_columns(self, row_id, media, metadata, run):
        # Runs once per row, in a background thread.
        #
        # `media` is the decoded frame (numpy uint8, HxWx3, RGB) because
        # this mode declares FRAME; it is None for METADATA or ADDRESS.
        # `metadata` holds this row's existing column values -- read-only.
        #
        # Every per-run value comes from `run`. Read parameters from
        # run.parameters and NEVER store one on `self`: an operator
        # instance is a singleton, so two concurrent runs would overwrite
        # each other's parameter values.
        threshold = run.parameters["threshold"]
        score = compute_my_score(media, threshold)   # your analysis here
        return {"my_score": score}

    def get_parameters_dialog(self, parent=None, columns=None):
        # Nothing generates a dialog from the declared `parameters` yet
        # (that is later P1.12d/e work). Until then a parameterised
        # operator builds its own QDialog here and exposes
        # parameter_values() -> dict, keyed by the descriptor's parameter
        # names. MainWindow shows this after the scope dialog and passes
        # the dict to the controller, which validates it against the
        # descriptor. Store NOTHING on `self`.
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
        )

        dialog = QDialog(parent)
        dialog.setWindowTitle("Compute my score")
        form = QFormLayout(dialog)

        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.0, 1.0)
        threshold_spin.setSingleStep(0.05)
        threshold_spin.setValue(0.5)            # the descriptor's default
        form.addRow("Detection threshold:", threshold_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        # Hand the chosen values back BY NAME. Every key must be a
        # parameter the descriptor declares.
        chosen = {}

        def _store():
            chosen["threshold"] = threshold_spin.value()

        dialog.accepted.connect(_store)
        dialog.parameter_values = lambda: dict(chosen)
        return dialog
```

A second execution mode is a second `ModeDescriptor` in `modes=(...)` -- with its
own `label` and `output` -- plus the matching `create_table` / `create_display`
method; `create_table` reads its grouping column from `run.parameters`, not from
a special-cased argument.

`[NOW]` Registration is driven by `operators_config.yaml` (P1.11a). Add a new
operator in **both** places: an entry in `operators_config.yaml` (its position
sets the menu order) and a factory in `OPERATOR_FACTORIES` in
`operators/operator_config.py`. A disagreement between the two raises
`OperatorConfigError` at startup. `docs/architecture.md` §7 is the authority
for the mechanism.

`[NOW]` Then add the operator to `OPERATORS_UNDER_TEST` **and** `EXPECTED` in
`tests/test_operator_descriptors_match.py` -- the modes it declares and, for a
COLUMNS mode, its `(name, tag)` output columns -- with its own named test.
`test_every_factory_operator_is_pinned` asserts that set equals
`OPERATOR_FACTORIES`, so a new operator without an entry fails the baseline.

---

## Testing

Write a standalone script that builds a small input, calls the method directly, and
prints or opens the output. Check the numbers by hand. This should not require Qt;
if it does, the operator has UI code in it.

Then run the guardrail tests:

```
python -m pytest tests/test_architecture_imports.py
python -m pytest tests/test_controller_async_contracts.py
python -m pytest tests/test_operator_registry_boundaries.py
```

A guardrail failure almost always means the operator reached into `Dataset`, the
controller, or Qt. Fix the access rather than working around the test.
