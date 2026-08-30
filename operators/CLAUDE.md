# Writing an operator

An operator is a self-contained analysis plugin the researcher runs from the
Operators menu. It inherits from `BaseOperator`, describes itself, and implements
one or more execution methods. The menu is built from the description.

`operators/base.py` is the contract.

**Statuses** follow `CLAUDE.md`: `[NOW]` is true today, `[TARGET -> item]` is not
yet true and must be by the named item, `[MIGRATING]` has listed exceptions.

`[TARGET -> P1.11]` **There is currently no good reference operator.**
`operators/thumbnail.py` used to be described as one but was dead code -- it
was deleted in P0.5b-2i (`ArtifactStore` generates thumbnails inline in
`_run_job`, off a bounded worker pool). P1.11 still owes a real operator
promoted in its place.

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

**Parameter values never live on `self`.** They arrive as an immutable
`OperatorRunSpec` for one run. Storing them as attributes on the operator instance
is a live correctness bug: operators are singletons, so two concurrent runs of the
same operator overwrite each other's parameters, and `MainWindow` currently reads
`operator._group_by` back off the instance.

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

`[TARGET -> P1.12]` **Do not put a stateful model on the singleton.** "Load the
model once in `__init__`" is the obvious optimisation and it is wrong for exactly
the models Gelem runs. MediaPipe landmarkers hold internal state, are not
documented as thread-safe, and in tracking mode carry the previous frame's result
forward. Two clips processed in parallel through one shared model object would
interleave that state, and the symptom is subtly wrong numbers rather than a crash.

Declare the lifecycle in the descriptor and let the runner honour it:

| Lifecycle | For | Lives on |
|---|---|---|
| `shared` | immutable, demonstrated thread-safe -- lookup tables, config, pure functions | the singleton, in `__init__` |
| `per_worker` | stateless-per-call inference that is expensive to load | one instance per worker thread or process |
| `per_sequence` | anything with tracking state | one isolated instance per clip-run, reset at the sequence boundary |

`shared` requires evidence, not assumption. If you do not know, choose
`per_worker`; if the model tracks across frames, `per_sequence` is the only correct
answer and P2.2's per-clip-run cache identity depends on it.

Provide a **factory**, not an instance, so the runner can build models at the right
granularity:

```python
# The runner calls this as many times as the declared lifecycle requires.
model_lifecycle = "per_sequence"

def build_model(self):
    return load_landmarker()
```

*(Raised in the fifth review round. It does not block Phase 0 or the measurement
pass, since no parallel execution exists yet -- but the template is what gets
copied, so the wrong default would be baked in well before P2.3 decides on
threads.)*

### `create_columns(row_id, media, metadata, run) -> dict`

Runs **once per row, in a background thread**. Returns new column values for that
one row. `AppController` applies each dict to `Dataset` on the main thread, and the
tile repaints immediately.

Use for per-item work where rows are independent. `media` is a typed payload, not
necessarily a still image, so the same method serves video spans and audio later.
Declare what you need in the descriptor's input requirements; `media` is `None` for
metadata-only operators. (The current boolean `requires_image` is the narrow
version of this.)

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

Note two traps. `summary` must be wrapped under that key; returning the nested dict
directly produces no visible output. And the key is `html_path` -- `base.py`
documents `plot_html`, which `ResultsPanel` only accepts as legacy fallback.

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
- **`[TARGET -> P1.11]` Every declared type tag must be registered.** Declaring an
  unregistered tag currently raises, gets swallowed as a printed warning at
  `controller.py:471-472`, and the column renders as "Unknown column".
  `avatar_path` and `plot_image` are both in this state today.
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

```python
from operators.base import BaseOperator


class MyOperator(BaseOperator):
    """One-line description shown to the researcher."""

    # ---- Identity and self-description --------------------------------
    # Version is part of the result-cache key. Bump it whenever a change
    # could alter the numbers this operator produces.
    name        = "my_operator"
    version     = "1.0"
    description = "Computes my score from the face in each image."

    # ---- Menu placement -----------------------------------------------
    # Setting a label makes that method appear in the Operators menu.
    create_columns_label = "Compute my score"
    create_table_label   = None
    create_display_label = None

    # ---- Declared outputs ----------------------------------------------
    # (column name, type tag). The tag must be registered in
    # ColumnTypeRegistry or the column will not render.
    output_columns = [("my_score", "numeric")]

    # ---- Declared inputs -----------------------------------------------
    # What this operator needs handed to it. One of:
    #   'metadata'   -- no media at all
    #   'frame'      -- a single decoded frame
    #   'video_span' -- an ordered span, for sequential work
    #   'audio_span' -- an audio span
    #   'address'    -- the raw address, resolve it yourself
    # (The current boolean `requires_image` is the narrow version of this.)
    input_requirement = "frame"

    # ---- Declared parameters -------------------------------------------
    # The UI builds the dialog from this. No Qt in this file.
    # 'column' means the value is a column name, so the UI offers a
    # dropdown of the active table's columns.
    parameters = [
        {"name": "window_ms", "type": "number", "default": 500,
         "label": "Averaging window (ms)"},
        {"name": "group_by",  "type": "column", "default": None,
         "label": "Group by", "optional": True},
    ]

    # ---- Model lifecycle ------------------------------------------------
    # How often the runner should build a model. 'shared' only if the
    # object is immutable and demonstrably thread-safe; 'per_sequence' for
    # anything that tracks across frames. See "Where a model lives".
    model_lifecycle = "per_worker"

    def build_model(self):
        # A factory, not an instance. The runner calls it as many times as
        # the declared lifecycle requires, and hands the result to the
        # execution methods via `run.model`.
        return load_landmarker()

    def __init__(self):
        # Only immutable, genuinely shared resources belong here. No model
        # unless its lifecycle is 'shared'. No parameter values and no
        # ProjectPaths -- this object is a singleton and two runs can be in
        # flight at once.
        pass

    def create_columns(self, row_id, media, metadata, run):
        # Background thread. Everything per-run comes from `run`, including
        # the model instance built for this run's lifecycle.
        window = run.parameters["window_ms"]
        score  = compute_something(media, window, run.model)
        return {"my_score": score}

    def create_table(self, df, run):
        # Same `run` argument, same rule. Grouping is a declared parameter,
        # not a special-cased method argument.
        group_by = run.parameters.get("group_by")
        work     = df.copy()
        if group_by:
            return work.groupby(group_by).mean(numeric_only=True).reset_index()
        return work.mean(numeric_only=True).to_frame().T

    def iter_column_updates(self, rows, run):
        # Streaming mode: walk the media in order, yield as you go, and
        # check for cancellation between units of work. With a
        # 'per_sequence' lifecycle, run.model is isolated to this clip.
        for row_id, address in rows:
            if run.cancelled():
                return
            frame = run.resolver.resolve_frame(address, purpose="analysis")
            yield row_id, {"my_score": compute_something(
                frame, run.parameters["window_ms"], run.model
            )}
```

`[TARGET -> P1.11]` Registration should be driven by `operators_config.yaml`.
Today `main.py` registers operators manually and never reads the file, and the two
lists already disagree -- `StatsOperator` is registered in code and absent from the
YAML. Until that is fixed, add the operator in both places.

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
