"""
run_tests.py -- the one command for running Gelem's test suite.

    python run_tests.py            # run everything
    python run_tests.py -x -q      # extra args are forwarded to every pytest run

Extra arguments are appended to every one of the several pytest invocations.
Flags such as -x, -q, -v and --tb=short do the sensible thing. A test
SELECTOR (-k NAME, a node id, or a file path) does NOT: it is handed to every
group, so groups it does not match exit "nothing collected" and the run is
reported FAILED. To run a single test, call `python -m pytest` directly.

WHY THIS SCRIPT EXISTS (for a developer who has never heard this story)

Running the whole suite as a single `python -m pytest` process is unreliable
on Windows. Roughly one full-suite run in six, the pytest process dies with a
native access violation (exit code 0xC0000005) on the main thread, at the first
moment in the run that a Qt widget is shown. A five-round diagnostic traced the
crash to a co-occurrence of MediaPipe FaceLandmarker inference
(tests/test_blendshape_operator.py), several data-layer test modules, and a
widget being shown -- but never found the cause, and the investigation is
deliberately closed. See docs/known_defects.md, "Native access violation during
the full pytest run", for the full record.

This script does NOT fix that crash. It contains the damage. Instead of one
process whose death takes the whole run's results with it, the suite is split
into several independent pytest processes:

  - every test module that does NOT show a Qt widget runs together in one
    process;
  - every widget-touching module runs alone in its own process.

A native crash in one process then loses only that process's results. The other
groups still run and still report. At the end this script prints a summary
table and distinguishes a known native crash from an ordinary test failure.
"""

from __future__ import annotations

import re
import sys
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Which test modules run isolated.
#
# A test module runs in its own pytest process when it SHOWS a Qt widget, so
# that the native crash described in this file's docstring loses only that one
# module's results. Two mechanisms decide which modules those are.
#
# 1. An explicit declaration token. A module may carry ONE comment of the form
#
#        # run-tests: isolate -- <reason>
#        # run-tests: combined -- <reason>
#
#    anywhere in the file. When a token is present it DECIDES, whatever the
#    substring heuristic below would say. `isolate` forces the module into its
#    own process; `combined` forces it into the shared group. The reason after
#    "--" is required and non-empty -- it exists so a future reader learns WHY
#    the module opted out, which the old substring-only rule could not express.
#
# 2. The substring heuristic, used only when no token is present. The module
#    counts as widget-showing if its source text contains any of "PySide6",
#    "realize_widget", or ".show(" (conftest.py excluded). The first catches a
#    module that imports the Qt binding directly; the other two catch a module
#    that only realizes a widget through the shared fixtures in
#    tests/conftest.py without naming the binding.
#
# The heuristic is only a proxy for "shows a widget" and is wrong in both
# directions -- which is the reason the token exists:
#
#   - a module that merely NAMES "PySide6" inside a source-scan assertion, and
#     never imports it, is forced into isolation it does not need;
#   - a module that loads Qt transitively -- by importing something that
#     imports it -- names none of the three substrings and silently joins the
#     fragile combined group.
#
# The rule keys on a widget being SHOWN, not merely on a QApplication
# existing, because QApplication creation timing was counted and ruled out
# during the diagnostic -- showing a widget is the moment the crash strikes.
#
# WIDGET_MODULES below is the isolated set written out explicitly: it fixes the
# order the isolated groups run in, and tests/test_run_tests_grouping.py fails
# if it drifts from what the two mechanisms above actually derive, so a new Qt
# test module that is not declared here is caught rather than silently
# rejoining the fragile group.
# ---------------------------------------------------------------------------
WIDGET_MARKERS: tuple[str, ...] = ("PySide6", "realize_widget", ".show(")

WIDGET_MODULES: tuple[str, ...] = (
    "test_dataset_access_paths.py",
    "test_demand_driven_display.py",
    "test_fake_controller_contract.py",
    "test_gallery_seam.py",
    "test_operator_tag_hints.py",
    "test_renderer.py",
    "test_result_delivery.py",
    "test_results_panel.py",
    "test_settings.py",
    "test_settings_dialog.py",
    "test_visible_row_order.py",
)


# tests/test_run_tests_grouping.py necessarily contains the marker substrings
# and example "# run-tests:" lines in order to describe and test this rule, so
# it is excluded from the source scan and always runs in the combined group.
# Any future module that documents this rule at the top level (an example
# declaration on its own comment line, which the scan would read as real) must
# be added here as well.
GROUPING_SELF_EXCLUDED: frozenset[str] = frozenset({"test_run_tests_grouping.py"})


class RunTestsDeclarationError(RuntimeError):
    """A test module's `# run-tests:` declaration is malformed.

    Raised while classifying modules, so a bad declaration stops the suite
    runner instead of silently mis-grouping the module.
    """


# Matches one "# run-tests:" declaration. It must be its OWN comment line
# (only whitespace before the "#"), so the marker appearing mid-line inside a
# string literal or prose -- as it necessarily does in the test that documents
# this rule -- is not mistaken for a declaration. Everything after the colon
# to end of line is the payload, expected to read "<token> -- <reason>".
_DECLARATION_RE = re.compile(r"^[ \t]*#[ \t]*run-tests:[ \t]*(.*)$", re.MULTILINE)

_VALID_TOKENS = ("isolate", "combined")


def declared_isolation_token(source_text: str, module_name: str) -> str | None:
    """Return a module's declared grouping token, or None if it declares none.

    The token is `isolate` or `combined`. Raises RunTestsDeclarationError,
    naming the module, when:
      - the module carries more than one `# run-tests:` declaration;
      - the token is neither `isolate` nor `combined`;
      - the reason after `--` is missing or blank.
    """
    # Every "# run-tests:" comment in the file, payload only.
    payloads = [match.strip() for match in _DECLARATION_RE.findall(source_text)]
    if not payloads:
        return None

    # More than one declaration -- two conflicting tokens, or two of the same
    # -- is ambiguous and refused.
    if len(payloads) > 1:
        raise RunTestsDeclarationError(
            f"{module_name} carries {len(payloads)} '# run-tests:' "
            "declarations; a module may carry at most one"
        )

    payload = payloads[0]

    # The reason is mandatory, so the "--" separator must be present.
    if "--" not in payload:
        raise RunTestsDeclarationError(
            f"{module_name} has a '# run-tests:' declaration with no "
            "'-- <reason>'; a non-empty reason is required"
        )

    # Split on the first "--"; a reason may itself contain "--".
    token_part, reason_part = payload.split("--", 1)
    token = token_part.strip()
    reason = reason_part.strip()

    if token not in _VALID_TOKENS:
        raise RunTestsDeclarationError(
            f"{module_name} declares '# run-tests: {token}'; the token must "
            "be 'isolate' or 'combined'"
        )

    if not reason:
        raise RunTestsDeclarationError(
            f"{module_name} declares '# run-tests: {token}' with an empty "
            "reason; the text after '--' must say why"
        )

    return token


def source_text_names_a_widget(source_text: str) -> bool:
    """The substring heuristic: True when the text contains any widget marker."""
    return any(marker in source_text for marker in WIDGET_MARKERS)


def module_runs_isolated(source_text: str, module_name: str) -> bool:
    """Whether a module runs in its own process: its declaration token when it
    has one, otherwise the substring heuristic.
    """
    token = declared_isolation_token(source_text, module_name)
    if token is not None:
        return token == "isolate"
    return source_text_names_a_widget(source_text)


# The Windows access violation exit code, in both forms it can reach Python.
# GetExitCodeProcess returns the unsigned DWORD 0xC0000005; depending on the
# Python build and platform, subprocess may hand it back as the raw unsigned
# value or as its signed 32-bit interpretation.
NATIVE_CRASH_CODES = frozenset({3221225477, -1073741819})

# pytest's own exit codes. 0 is all-passed; 5 is "no tests were collected".
# Exit 5 is only acceptable for the two modules below: they are in the widget
# list because their text names the Qt binding, but they are standalone
# manual-check scripts (run directly with `python tests/<name>`) with no
# collectable test functions, so they always return 5. For every other group
# an exit 5 IS a failure -- e.g. a forwarded `-k` that matches nothing there,
# or a module that lost all its tests -- and must not be reported green.
PYTEST_NO_TESTS_COLLECTED = 5
MANUAL_CHECK_MODULES = frozenset({"test_renderer.py", "test_results_panel.py"})


# The repo root is wherever this script lives; the tests live beside it.
REPO_ROOT = Path(__file__).resolve().parent
TESTS_DIR = REPO_ROOT / "tests"


def discover_test_modules() -> list[str]:
    """Return the sorted filenames of every tests/test_*.py module.

    Discovery is by glob, not a hardcoded list, so a newly added non-widget
    test module joins the big group automatically.
    """
    return sorted(path.name for path in TESTS_DIR.glob("test_*.py"))


def scan_isolated_modules() -> set[str]:
    """Classify every tests/test_*.py module from its own source and return the
    filenames that run isolated.

    A module's `# run-tests:` token decides when present, otherwise the
    substring heuristic. Raises RunTestsDeclarationError on a malformed
    declaration anywhere in the suite, so a bad token stops the run.
    """
    isolated: set[str] = set()
    for name in discover_test_modules():
        if name in GROUPING_SELF_EXCLUDED:
            continue
        text = (TESTS_DIR / name).read_text(encoding="utf-8")
        if module_runs_isolated(text, name):
            isolated.add(name)
    return isolated


def build_groups() -> list[tuple[str, list[str]]]:
    """Return the ordered list of (group_name, pytest_target_args) to run.

    The first group is every non-widget module, run together as one pytest
    invocation. After it, one group per isolated module, each run alone.
    """
    all_modules = discover_test_modules()

    # Derive the isolated set from each module's own source (token or
    # heuristic). WIDGET_MODULES only fixes the run order and is drift-checked
    # by tests/test_run_tests_grouping.py.
    isolated = scan_isolated_modules()

    # Non-widget modules: everything discovered that is not isolated.
    non_widget = [name for name in all_modules if name not in isolated]

    groups: list[tuple[str, list[str]]] = []

    # One combined group for the non-widget modules. Passing the explicit file
    # paths (rather than just "tests/") keeps this group's membership exactly
    # the complement of the isolated set, even if pytest's own collection
    # rules would otherwise pick up something else in the directory.
    non_widget_targets = [str(TESTS_DIR / name) for name in non_widget]
    groups.append(("non-widget (combined)", non_widget_targets))

    # One isolated group per isolated module: WIDGET_MODULES order first, then
    # any module the scan isolates that is not yet listed there -- so a new,
    # undeclared Qt module is still isolated rather than silently dropped.
    ordered_isolated = [name for name in WIDGET_MODULES if name in isolated]
    ordered_isolated += sorted(isolated.difference(WIDGET_MODULES))
    for name in ordered_isolated:
        groups.append((name, [str(TESTS_DIR / name)]))

    return groups


def run_group(name: str, targets: list[str], forwarded_args: list[str]) -> int:
    """Run one pytest process for a group and return its exit code.

    stdout and stderr are inherited, so the developer sees ordinary pytest
    output for each group as it runs.
    """
    command = [sys.executable, "-m", "pytest", *targets, *forwarded_args]

    print()
    print("=" * 72)
    print(f"GROUP: {name}")
    print(f"  {' '.join(command)}")
    print("=" * 72, flush=True)

    completed = subprocess.run(command, cwd=str(REPO_ROOT))
    return completed.returncode


def _is_expected_no_tests(name: str, exit_code: int) -> bool:
    """True only when this group is a known manual-check script AND it
    returned pytest's "nothing collected" code. Any other group returning 5
    is a real problem (a forwarded selector matched nothing, or the module
    lost its tests) and is not excused here.
    """
    return (
        exit_code == PYTEST_NO_TESTS_COLLECTED
        and name in MANUAL_CHECK_MODULES
    )


def status_for(name: str, exit_code: int) -> str:
    """Turn a group's exit code into a plain-language status.

    A native access violation is called out as a known defect and kept
    distinct from an ordinary test failure.
    """
    if exit_code == 0:
        return "passed"
    if _is_expected_no_tests(name, exit_code):
        return "no tests collected (manual-check script)"
    if exit_code in NATIVE_CRASH_CODES:
        return "NATIVE CRASH -- known defect, see docs/known_defects.md"
    if exit_code == PYTEST_NO_TESTS_COLLECTED:
        return "FAILED -- no tests collected (unexpected for this group)"
    return "FAILED"


def group_is_ok(name: str, exit_code: int) -> bool:
    """Whether a group's exit code counts as success for the overall run.

    A clean pass (0) is fine; so is "no tests collected" (5) but only for the
    known manual-check scripts. Every other non-zero code -- an ordinary
    failure or the native crash -- is not.
    """
    return exit_code == 0 or _is_expected_no_tests(name, exit_code)


def print_summary(results: list[tuple[str, int]]) -> None:
    """Print the final summary table: group, exit code, status."""
    name_width = max(len(name) for name, _ in results)
    name_width = max(name_width, len("GROUP"))

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    header = f"{'GROUP':<{name_width}}  {'EXIT':>11}  STATUS"
    print(header)
    print("-" * len(header))
    for name, exit_code in results:
        status = status_for(name, exit_code)
        print(f"{name:<{name_width}}  {exit_code:>11}  {status}")
    print()


def main(argv: list[str]) -> int:
    """Run every group, print the summary, and return the overall exit code.

    Any arguments after the script name are forwarded verbatim to every pytest
    invocation.
    """
    forwarded_args = argv[1:]

    groups = build_groups()
    results: list[tuple[str, int]] = []

    for name, targets in groups:
        exit_code = run_group(name, targets, forwarded_args)
        results.append((name, exit_code))

    print_summary(results)

    # The whole run succeeds only if every group succeeded. Exit 0 is the only
    # pass; the sole exception is a known manual-check script returning "no
    # tests collected" (see group_is_ok).
    all_ok = all(group_is_ok(name, exit_code) for name, exit_code in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
