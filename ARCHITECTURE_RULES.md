# Gelem Architecture Rules

This is the one-page rule reference for all contributors.
Read it before writing any new code. When in doubt, ask.

---

## The seven components

| Component | Owner | Responsibility |
|---|---|---|
| `Dataset` | Student B | The only component that may modify a DataFrame |
| `QueryEngine` | Student B | Filtering and sorting — read-only access to DataFrames |
| `ArtifactStore` | Core | Thumbnail and preview cache |
| `ColumnTypeRegistry` | Core | Maps column names to render functions |
| `OperatorRegistry` | Student C | Loads operators from config, dispatches background runs |
| `AppController` | Core | The only wiring layer — coordinates all other components |
| `UI` | Student A | Widgets only — no data logic |

---

## Threading rules

**Main thread:** All DataFrame reads and writes happen here. Qt widgets are created and updated here.

**Background threads:** Operators run here. They may read only from work items that were prepared and handed to them *before* the thread started. They must never call `Dataset` directly during execution.

**The handoff:** A background thread signals completion by calling the `on_item_complete` callback. `AppController` receives that signal and applies the result to `Dataset` on the main thread. No shortcutting this path.

**Async result payloads must always include:**
- `table_name` — the table that was active when the run was launched
- `row_id` — for per-row results

No component may infer the target table from controller state at drain time. If you write `self._active_table` inside a drain loop, that is a bug.

> ⚠️ **If you are writing code that touches both a background worker and `Dataset`, stop and ask.**

---

## Import rules — what each layer may import

### `ui/`
✅ May import: `controller.py` public methods and signals, `shared_widgets/`  
❌ Must never import: `pandas`, `PIL`, `cv2`, `mediapipe`, `Dataset`, `QueryEngine`, `OperatorRegistry`, `ColumnTypeRegistry`, `column_types/`, `operators/`, `artifacts/`

### `column_types/`
✅ May import: `shared_widgets/`, `PIL`, `cv2`, `numpy`, `PySide6`  
❌ Must never import: anything under `ui/`

#### Renderer rules

Renderers translate a column value into a displayable object.

- In `'thumbnail'` mode, they must return a `QPixmap`.
- In `'detail'` mode, they may return a `QWidget`.
- Renderers may use `shared_widgets/`, but must not depend on page-level UI modules (e.g. `main_window`, `detail_widget`, `gallery_widget`).
- Renderers must not read from `Dataset` or `QueryEngine`.
- If rendering depends on row-level information (e.g. a cached thumbnail keyed by `row_id`), that context must be passed in through the render call. Renderers must not reconstruct it themselves.

### `operators/`
✅ May import: `column_types/`, `shared_widgets/`, media libraries  
❌ Must never import: `ui/`, and must never call `Dataset` directly

### `shared_widgets/`

Contains reusable UI components that are independent of application state.

✅ May import: `PySide6` and standard-library modules only  
❌ Must never import: anything else in this project  
❌ Must not depend on: `Dataset`, `QueryEngine`, `OperatorRegistry`, or `AppController`  
❌ Must not assume: any specific column names or data schema

### `AppController`

`AppController` may orchestrate all components through their public APIs.

If a required operation is not available through a public API, add a new method to the appropriate component instead of accessing private attributes.

✅ May import: all components through their public APIs  
❌ Must never access `_private` attributes of any component — no exceptions

---

## Data flow rules

1. **`Dataset` is the only writer.** No operator, registry, or widget may assign to a DataFrame column directly. Return a dict; let `AppController` call `dataset.update_row()`.

2. **The UI never reads a DataFrame.** Widgets receive data through controller signals (`gallery_updated`, `row_selected`, etc.) only. If you find yourself writing `df[...]` inside `ui/`, something is wrong.

3. **`row_id` is permanent.** It is assigned once when a folder is loaded. It is never reused, never reassigned, never used as a positional index.

4. **Column registration happens before operator results arrive.** `AppController` registers output columns via `registry.register_by_tag(...)` before launching a `create_columns` operator, so tiles can show placeholders immediately.

5. **Render context flows in, not out.** If a renderer needs row-level information (e.g. `row_id` for cache lookups), the caller must pass it as a `context` dict through the render call chain. Renderers must not reach up into the controller or sideways into `Dataset` to retrieve it.

---

## Operator rules

- Every operator must subclass `BaseOperator` from `operators/base.py`.
- `create_columns` operators return a `dict` of `{column_name: value}` per row. They do not write to `Dataset`.
- `create_table` operators return a new `pd.DataFrame`. They do not write to `Dataset`.
- `create_display` operators return a displayable object (e.g. a matplotlib figure). They do not write to `Dataset`.
- Operator output columns are declared in `operators_config.yaml`, not hardcoded in the operator body.

---

## The two violations to never repeat

These are patterns that already appear in the skeleton as known bugs being fixed. **Do not copy them.**

```python
# ❌ Wrong — operator registry reaching into a private Dataset attribute
self._dataset._registry.register(...)

# ✅ Right — controller registers the column before launching the operator
self._registry.register_by_tag(column_name, tag)
self._op_registry.run_create_columns(...)
```

```python
# ❌ Wrong — UI tile reading Dataset through a private controller attribute
metadata = self._controller._dataset.get_row(self.row_id)

# ✅ Right — UI receives metadata through a controller signal
# (e.g. row_selected) and stores it locally
self._metadata = metadata_from_signal

```

---

## Quick checklist before you submit code

- [ ] Does any `ui/` file import `pandas`, `PIL`, `cv2`, or a `Dataset`/`QueryEngine` class? → Remove it.
- [ ] Does any operator body call `dataset.update_row()` or assign to a DataFrame? → Move that to `AppController`.
- [ ] Does any background thread call `Dataset` or read `self._dataset` during execution? → Snapshot the data before the thread starts.
- [ ] Does any code access `component._private_attr` from outside that component? → Use the public API, or add a new public method to the component.
- [ ] Does any async result callback use `self._active_table` to decide where to write? → The payload must carry `table_name` explicitly.
- [ ] Does any renderer import from `ui/` or call `Dataset`/`QueryEngine`? → Renderers receive all context they need through their parameters.
