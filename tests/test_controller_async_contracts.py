"""
tests/test_controller_async_contracts.py

Source-inspection (AST) checks on AppController's result-delivery
contract. They run without Qt or any real data.

P0.2b rewrite. The drain is no longer a single `_drain_queues` method
whose body can be grepped -- it is an orchestrator (`_drain_queues`) plus
one helper per queue (`_drain_thumbnails`, `_drain_item_results`,
`_drain_completions`) and a separate coalesced progress emit
(`_emit_progress_if_changed`, deliberately not `_drain`-prefixed because
it carries no budget). The old tests here inspected the text of a method
literally named `_drain_queues`; if the drain were split and this file
were not updated, all of them would keep passing while guarding nothing.
So the checks below apply to **every** method whose name starts with
`_drain`, whatever the split looks like.

Run with: pytest tests/test_controller_async_contracts.py
"""

import ast
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONTROLLER = ROOT / "controller.py"


def _functions() -> list[ast.FunctionDef]:
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _method_source(method_name: str) -> str | None:
    source = CONTROLLER.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(source, node)
    return None


def _method_arg_names(method_name: str) -> list[str] | None:
    for node in _functions():
        if node.name == method_name:
            return [a.arg for a in node.args.args]
    return None


def _drain_methods() -> dict[str, str]:
    """{name: source} for every method whose name starts with `_drain`."""
    source = CONTROLLER.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for node in _functions():
        if node.name.startswith("_drain"):
            out[node.name] = ast.get_source_segment(source, node)
    return out


# ── _on_item_complete carries the full per-row identity ──────────────────

def test_on_item_complete_carries_operation_id():
    """_on_item_complete must accept operation_id so a result from a run
    that is no longer live can be rejected.

    Would still pass if violated? No -- drop the arg and this fails.
    """
    args = _method_arg_names("_on_item_complete")
    assert args is not None, "_on_item_complete not found in controller.py"
    assert "operation_id" in args, (
        f"_on_item_complete is missing 'operation_id' -- got args: {args}"
    )


def test_on_item_complete_carries_table_name():
    """_on_item_complete must accept table_name so results land in the
    table they were computed against, not whatever is on screen.

    Would still pass if violated? No.
    """
    args = _method_arg_names("_on_item_complete")
    assert args is not None, "_on_item_complete not found in controller.py"
    assert "table_name" in args, (
        f"_on_item_complete is missing 'table_name' -- got args: {args}"
    )


def test_on_item_complete_enqueues_all_four_fields():
    """_on_item_complete must enqueue (operation_id, table_name, row_id,
    result) -- the whole tuple, or the drain cannot key on the run or
    place the row.

    Would still pass if violated? Only if a field name happened to appear
    elsewhere in the two-line method; it does not.
    """
    src = _method_source("_on_item_complete")
    assert src is not None, "_on_item_complete not found in controller.py"
    for field in ("operation_id", "table_name", "row_id", "result"):
        assert field in src, (
            f"_on_item_complete does not reference '{field}' -- the full "
            f"4-tuple must be enqueued"
        )


# ── The drain never keys staleness on the active table ──────────────────

def test_no_drain_method_reads_active_table():
    """No `_drain*` method may read self._active_table.

    Staleness is keyed on run liveness (_live_runs). A result carries its
    own table_name and belongs in that table whatever is on screen. If a
    drain helper read self._active_table, a result computed for one table
    while the user has switched to another would be dropped or misplaced.

    Would still pass if violated? No -- this is the non-vacuous
    replacement for the old test, which only inspected `_drain_queues`
    and would now inspect a method too small to contain the bug.
    """
    offenders = {
        name: src for name, src in _drain_methods().items()
        if "_active_table" in src
    }
    assert not offenders, (
        f"these drain methods read self._active_table: {sorted(offenders)}"
    )


def test_item_results_go_through_the_batch_write_path():
    """The item-result drain must apply updates via
    Dataset.apply_row_updates() (the batch path) and must not call the
    one-row update_row().

    Would still pass if violated? No -- a per-row update_row() loop in the
    drain would fail the `update_row(` check.
    """
    combined = "\n".join(_drain_methods().values())
    assert "apply_row_updates" in combined, (
        "no drain method calls Dataset.apply_row_updates() -- the batched "
        "write path is gone"
    )
    assert "update_row(" not in combined, (
        "a drain method calls Dataset.update_row() -- results must be "
        "batched into one apply_row_updates() call per table"
    )


def test_every_queue_drain_is_budget_bounded():
    """Every `_drain*` method except the orchestrator must reference
    self._drain_budget, so each queue is drained by a bounded number of
    items per tick rather than emptied.

    Would still pass if violated? No -- an unbounded `while True: get_nowait`
    helper would not mention _drain_budget.
    """
    unbounded = []
    for name, src in _drain_methods().items():
        if name == "_drain_queues":
            # The orchestrator only calls the helpers.
            continue
        if "_drain_budget" not in src:
            unbounded.append(name)
    assert not unbounded, (
        f"these drain methods are not bounded by self._drain_budget: "
        f"{sorted(unbounded)}"
    )


def test_no_drain_method_uses_list_pop_zero():
    """No `_drain*` method may use list.pop(0) (linear in queue length).

    Would still pass if violated? No. This asserts the property, so a
    SimpleQueue or a deque both pass; only a list drained from the front
    fails.
    """
    offenders = {
        name for name, src in _drain_methods().items() if "pop(0)" in src
    }
    assert not offenders, (
        f"these drain methods use list.pop(0): {sorted(offenders)}"
    )


# ── Worker-bound callbacks read no component state ──────────────────────

# Every callback the OperatorRegistry / ArtifactStore invoke from a
# background thread. CLAUDE.md: they may only put a result onto a queue
# (or, for _on_progress, overwrite a locked value). Reading a component
# would be a data race.
_WORKER_CALLBACKS = {
    "_on_thumbnail_ready",
    "_on_clip_frames_ready",
    "_on_item_complete",
    "_on_progress",
    "_on_run_log",
    "_on_create_columns_complete",
    "_on_operator_setup_error",
    "_on_operator_row_errors",
    "_on_create_table_complete",
    "_on_create_display_complete",
    "_on_operator_error",
}
_FORBIDDEN_ATTRS = {
    "_dataset", "_query", "_store", "_registry", "_op_registry",
    "_result", "_result_index", "_live_runs", "_active_table",
}


def test_worker_callbacks_touch_no_component_state():
    """A worker-thread callback must only reach `self.<a queue>` or
    `self._progress_lock` / `self._latest_progress` -- never a component
    or shared controller state.

    Would still pass if violated? No. This is the guardrail behind the
    `[NOW]` promise that CLAUDE.md's 'workers place results on queues and
    read nothing' rule now has no violation list -- e.g. it fails the
    moment _on_operator_setup_error goes back to self._op_registry.get().
    """
    for node in _functions():
        if node.name not in _WORKER_CALLBACKS:
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
                and sub.attr in _FORBIDDEN_ATTRS
            ):
                raise AssertionError(
                    f"{node.name} reads self.{sub.attr} from a worker "
                    f"thread -- it may only enqueue a result"
                )


# ── _WORKER_CALLBACKS itself must not silently fall behind the source ───

def _self_on_call_func_ids(tree: ast.AST) -> set:
    """id() of every AST node that is the immediate `.func` of a Call node
    -- e.g. the Attribute node in `self._on_x(...)`. Used to tell "this
    self._on_* reference IS the callee of a direct call" apart from every
    other way the same reference can appear.
    """
    return {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}


def _handed_off_self_on_names(tree: ast.AST) -> set:
    """Every `self._on_<name>` attribute name with at least one occurrence
    that is NOT the immediate callee of a direct call -- i.e. handed off
    as a plain value (assigned onto another object's attribute, or passed
    as a positional/keyword argument) rather than called by AppController
    itself. See test_worker_callbacks_set_matches_every_self_on_name_handed_to_another_component
    for why this is the rule and what it would miss.
    """
    call_func_ids = _self_on_call_func_ids(tree)
    names: set = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr.startswith("_on_")
            and id(node) not in call_func_ids
        ):
            names.add(node.attr)
    return names


def test_worker_callbacks_set_matches_every_self_on_name_handed_to_another_component():
    """_WORKER_CALLBACKS above is hand-maintained, and this repo has
    already hit the failure mode of a hand-maintained list silently
    walking less than it claims: run-indicator-2 added
    AppController._on_run_log as a new worker-thread callback (wired as
    OperatorRun's _log_fn, the same way _on_progress is wired as
    on_progress=) but never added it to _WORKER_CALLBACKS, so
    test_worker_callbacks_touch_no_component_state silently did not scan
    the newest worker-thread callback at all. This test ties the set to
    the source instead of trusting it to stay current by hand.

    RULE USED: in this codebase, a worker-thread callback is, by
    construction, a `self._on_*` bound method AppController hands to
    ANOTHER component to invoke later -- as a plain attribute assignment
    (`self._store.on_thumbnail_ready = self._on_thumbnail_ready`) or as
    an argument (`on_progress=self._on_progress`,
    `_log_fn=self._on_run_log`) -- never a name AppController only ever
    calls itself. So: walk controller.py's AST for every
    `self._on_<name>` attribute access; a name counts as a worker
    callback if at least one of its occurrences is NOT the immediate
    callee of a direct call. `_on_operator_complete` is the control case
    this rule has to get right: it IS called directly
    (`self._on_operator_complete(mode, payload)`, and its own docstring
    says "Called on the main thread from the completion-queue drain"),
    and every occurrence of it in the file is that same direct call, so
    it must NOT appear in the derived set -- and does not.

    WHAT THIS WOULD MISS: a worker-thread callback that does not follow
    the `self._on_*` naming convention, or one wired through something
    this AST walk does not see as a literal `self._on_<name>` attribute
    access -- built inside a lambda, fetched via getattr(), or assembled
    into a dict/functools.partial rather than passed as a plain
    argument. It would also miss a callback wired from OUTSIDE
    controller.py (this test reads only CONTROLLER_FILE). Every
    worker-thread callback in this codebase today follows the plain
    `self._on_*` handoff shape this rule looks for, so for the code as
    it stands this is a real, non-vacuous check -- but it is a
    structural proxy for "handed to another component to call later",
    not a proof that a name obeys every rule CLAUDE.md states for
    worker callbacks (it says nothing about which thread anything
    actually runs on, only about how the reference reaches that other
    component).
    """
    source = CONTROLLER.read_text(encoding="utf-8")
    tree = ast.parse(source)

    derived = _handed_off_self_on_names(tree)

    assert derived == _WORKER_CALLBACKS, (
        "self._on_* names handed off to another component in controller.py "
        "do not match the hand-maintained _WORKER_CALLBACKS set.\n"
        f"  handed off in the source but missing from _WORKER_CALLBACKS: "
        f"{sorted(derived - _WORKER_CALLBACKS)}\n"
        f"  in _WORKER_CALLBACKS but never handed off anywhere in the "
        f"source: {sorted(_WORKER_CALLBACKS - derived)}"
    )
