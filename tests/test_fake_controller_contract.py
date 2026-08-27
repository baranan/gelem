# tests/test_fake_controller_contract.py
#
# Guard test for real-vs-fake controller drift.
#
# FakeController promises in its own docstring to expose the same signals
# AND the same public methods as AppController. When someone adds a member
# to the real controller but forgets the fake, --fake-data breaks: a missing
# signal crashes at startup (during _connect_signals), and a missing method
# crashes later, the moment a UI widget calls it in fake-data mode.
#
# This happened once already: the "add-controller-public-methods" PR added
# one signal (active_table_changed) and three methods (get_all_row_ids,
# get_column_type, get_operator) to the real controller without updating the
# fake. The signal gap surfaced as a startup crash; the method gaps would
# have surfaced later as crashes-on-use. These two tests turn both kinds of
# silent drift into a clear, named test failure.

import ast
import pathlib

from PySide6.QtCore import QObject, Signal
from controller import AppController
from ui.fake_controller import FakeController

_ROOT = pathlib.Path(__file__).parent.parent
_CONTROLLER_FILE = _ROOT / "controller.py"
_FAKE_FILE = _ROOT / "ui" / "fake_controller.py"


# ── Helper: collect the Signal members declared on a class ────────────────
def _signal_names(cls):
    # A PySide6 Signal is a class attribute; isinstance(..., Signal) finds it.
    return {name for name in dir(cls) if isinstance(getattr(cls, name), Signal)}


# ── Helper: collect the PUBLIC METHOD names a class adds beyond QObject ────
def _public_method_names(cls):
    # We only care about the application contract, not the dozens of methods
    # every QObject inherits from Qt (blockSignals, deleteLater, ...). So we
    # subtract QObject's own members: whatever is left is what *this* class
    # (or AppController/FakeController) deliberately adds.
    qobject_members = set(dir(QObject))

    names = set()
    for name in dir(cls):
        # Skip inherited Qt plumbing and any private/dunder names.
        if name in qobject_members:
            continue
        if name.startswith("_"):
            continue
        attr = getattr(cls, name)
        # Methods are callable; Signals are not. This keeps signals out of the
        # method set so the two tests stay cleanly separated.
        if callable(attr):
            names.add(name)
    return names


# ── Test 1: signal parity ─────────────────────────────────────────────────
def test_fake_controller_exposes_all_real_signals():
    # Every signal the real controller declares must also exist on the fake,
    # or MainWindow._connect_signals() crashes at startup in fake-data mode.
    missing = _signal_names(AppController) - _signal_names(FakeController)
    assert not missing, (
        f"FakeController is missing signals that AppController declares: "
        f"{sorted(missing)}. Add them to ui/fake_controller.py so --fake-data "
        f"keeps working."
    )


# ── Test 2: public-method parity ──────────────────────────────────────────
def test_fake_controller_exposes_all_real_public_methods():
    # Every public method on the real controller must also exist on the fake,
    # or a UI widget that calls it will crash only in fake-data mode (and only
    # when that code path runs -- so it can hide for a long time).
    missing = _public_method_names(AppController) - _public_method_names(FakeController)
    assert not missing, (
        f"FakeController is missing public methods that AppController defines: "
        f"{sorted(missing)}. Add matching stubs to ui/fake_controller.py so "
        f"--fake-data keeps working."
    )


# ── Test 3: signal SIGNATURE parity (not just names) ─────────────────────
#
# Tests 1 and 2 compare names only. That misses a signal whose payload
# type drifted: changing `Signal(str)` to `Signal(object)` in the real
# controller but not the fake (or vice versa) leaves the names equal and
# the wiring subtly broken -- MainWindow would hand a payload object to a
# slot the fake still declares as `Signal(str)`. This test compares the
# argument list of each `Signal(...)` between the two files.

def _signal_arg_lists(path: pathlib.Path) -> dict[str, list[str]]:
    """{signal_name: [unparsed Signal() args]} for class-body Signals."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    specs: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        if not (isinstance(func, ast.Name) and func.id == "Signal"):
            continue
        arg_srcs = [ast.unparse(a) for a in value.args]
        for target in node.targets:
            if isinstance(target, ast.Name):
                specs[target.id] = arg_srcs
    return specs


def test_signal_signatures_match_between_real_and_fake_controller():
    real = _signal_arg_lists(_CONTROLLER_FILE)
    fake = _signal_arg_lists(_FAKE_FILE)

    common = set(real) & set(fake)
    assert common, "no signals found in common -- parser or file path is wrong"

    mismatched = {
        name: {"controller.py": real[name], "fake_controller.py": fake[name]}
        for name in sorted(common)
        if real[name] != fake[name]
    }
    assert not mismatched, (
        f"Signal payload types differ between the real and fake controller: "
        f"{mismatched}. They must be declared identically or --fake-data "
        f"wiring breaks silently."
    )
