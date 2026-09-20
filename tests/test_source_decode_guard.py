"""
tests/test_source_decode_guard.py

P1.2c-2's guardrail for CLAUDE.md's Media rule: "Only the media resolver
decodes *source* media." Before this item, three places decoded a
user's media file directly -- BaseOperator.load_image (deleted, P1.2c-1),
ArtifactStore._decode_source (routed through MediaResolver, P1.2c-1), and
operators/video_frames.py's cv2.VideoCapture loop (routed through
MediaResolver, P1.2c-2). This test makes the rule enforceable instead of
a hand-maintained list that goes stale silently.

Walks every non-test .py file in the repository with ``ast`` -- not a
text search, which a comment or docstring naming cv2 or PIL would trip --
and fails if any of them CALLS cv2.VideoCapture, cv2.imread, av.open, or
PIL's Image.open, under any import spelling (a bare name from
``from X import Y``, a renamed module from ``import X as Y``, or a fully
qualified ``PIL.Image.open``). Two files are the deliberate exception and
excluded from the failing check (but still visited, so the file-count
floor below still counts them): media/resolver.py, the one module that
is allowed -- required -- to decode source media, and
artifacts/artifact_codec.py, which reads back JPEGs ArtifactStore itself
wrote (CLAUDE.md: "reading and writing Gelem's own derived artifacts is a
different operation").

The one other named exception is Qt loading a file for a full-size
DETAIL-MODE view -- QPixmap(path) or QImageReader(path) in
column_types/renderers.py's _render_image, and QMediaPlayer.setSource()
for video. Neither of those calls is cv2.VideoCapture, cv2.imread,
av.open, or PIL's Image.open, so this AST walk does not need to name them
as an exclusion -- they simply never match the target-call list.

Run with:
    python -m pytest tests/test_source_decode_guard.py
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

# Directories holding non-application scripts: dev-only manual test
# scripts (manual_testing/), one-off crash-diagnosis scripts kept for the
# record (docs/review/), and this test suite itself. None of these ships
# as part of the running application, so they are outside "every non-test
# .py file" this guard means to police.
_EXCLUDED_DIR_PARTS = {"tests", "manual_testing", "docs", "__pycache__", ".git"}

# Visited (so the file-count floor below still counts them) but never
# reported as a violation -- see the module docstring.
_EXEMPT_FILES = {
    ROOT / "media" / "resolver.py",
    ROOT / "artifacts" / "artifact_codec.py",
}

# The exact qualified calls this guard refuses, whatever local name or
# alias the source spells them with.
_FORBIDDEN_CALLS = {
    "cv2.VideoCapture",
    "cv2.imread",
    "av.open",
    "PIL.Image.open",
}


def _source_files() -> list[pathlib.Path]:
    """Every non-test .py file under the repository root."""
    files = []
    for path in ROOT.rglob("*.py"):
        relative_parts = path.relative_to(ROOT).parts
        if any(part in _EXCLUDED_DIR_PARTS for part in relative_parts):
            continue
        if path.name.startswith("test_"):
            continue
        files.append(path)
    return sorted(files)


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruct 'a.b.c' from a Name/Attribute chain such as a call's
    ``func``. None if the target is not a plain dotted name -- a
    subscript, a call result, or anything else this simple, static walk
    cannot resolve.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _collect_aliases(tree: ast.AST) -> dict[str, str]:
    """local name -> fully-qualified dotted path, from every Import and
    ImportFrom node anywhere in the file (module level or nested inside a
    function, exactly as tests/test_operators_are_qt_free.py's Qt-import
    walk does not limit itself to module level either).

    Covers every spelling the work item names:
      * ``import cv2`` / ``import cv2 as c``            -- module alias
      * ``import av`` / ``import av as a``               -- module alias
      * ``from PIL import Image`` / ``... as Img``       -- submodule alias
      * ``import PIL.Image`` / ``... as pi``             -- submodule alias
      * ``from cv2 import VideoCapture, imread``         -- direct function
      * ``from av import open``                          -- direct function
      * ``from PIL.Image import open``                   -- direct function
    A plain ``import cv2`` (no ``as``) needs no entry: the call site
    already reads ``cv2.VideoCapture(...)``, which _dotted_name()
    reconstructs correctly with no alias substitution at all.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                # A relative import ('from . import x') cannot reach an
                # external cv2/av/PIL module.
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = f"{node.module}.{alias.name}"
    return aliases


def _resolve_call(dotted: str, aliases: dict[str, str]) -> str:
    """The fully-qualified form of a call's dotted target, substituting
    any alias found for its head component. A bare name (e.g. a directly
    imported ``VideoCapture``) is looked up whole; a dotted chain (e.g.
    ``c.VideoCapture`` where ``c`` aliases ``cv2``) has only its head
    replaced, so the qualified attribute chain is preserved.
    """
    parts = dotted.split(".")
    head, rest = parts[0], parts[1:]
    if not rest:
        return aliases.get(head, head)
    if head in aliases:
        return aliases[head] + "." + ".".join(rest)
    return dotted


def _forbidden_calls_in_module(path: pathlib.Path) -> list[str]:
    """Every forbidden call this file makes, resolved to its qualified
    form (e.g. "cv2.VideoCapture"), in the order encountered.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _collect_aliases(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if dotted is None:
            continue
        resolved = _resolve_call(dotted, aliases)
        if resolved in _FORBIDDEN_CALLS:
            hits.append(resolved)
    return hits


def test_the_walk_visits_more_than_twenty_files():
    # A guardrail that inspects nothing passes vacuously. main.py,
    # controller.py, and every module under artifacts/, column_types/,
    # media/, models/, operators/, settings/, shared_widgets/ and ui/
    # comfortably clears twenty; a broken glob or an over-broad exclusion
    # list would drop well below it.
    files = _source_files()
    assert len(files) > 20, (
        f"expected to walk more than 20 non-test .py files, found "
        f"{len(files)}: {[str(f.relative_to(ROOT)) for f in files]}"
    )


def test_the_walk_catches_a_planted_violation(tmp_path):
    planted = tmp_path / "planted_violation.py"
    planted.write_text(
        "import cv2\n"
        "\n"
        "def decode(path):\n"
        "    return cv2.VideoCapture(path)\n",
        encoding="utf-8",
    )
    assert _forbidden_calls_in_module(planted) == ["cv2.VideoCapture"]


def test_the_walk_catches_every_named_import_spelling(tmp_path):
    planted = tmp_path / "every_spelling.py"
    planted.write_text(
        "import cv2 as c\n"
        "from cv2 import imread\n"
        "import av\n"
        "from PIL import Image\n"
        "import PIL.Image as pi\n"
        "from PIL.Image import open as pil_open\n"
        "\n"
        "def decode(path):\n"
        "    c.VideoCapture(path)\n"
        "    imread(path)\n"
        "    av.open(path)\n"
        "    Image.open(path)\n"
        "    pi.open(path)\n"
        "    pil_open(path)\n",
        encoding="utf-8",
    )
    hits = _forbidden_calls_in_module(planted)
    assert hits == [
        "cv2.VideoCapture",
        "cv2.imread",
        "av.open",
        "PIL.Image.open",
        "PIL.Image.open",
        "PIL.Image.open",
    ]


def test_no_source_file_decodes_media_outside_the_resolver_and_the_codec():
    violations: dict[str, list[str]] = {}
    for path in _source_files():
        if path in _EXEMPT_FILES:
            continue
        hits = _forbidden_calls_in_module(path)
        if hits:
            violations[str(path.relative_to(ROOT))] = hits

    assert not violations, (
        "these files decode source media directly instead of going "
        "through media.resolver.MediaResolver (CLAUDE.md, 'Media' -- "
        "'Only the media resolver decodes source media'):\n"
        + "\n".join(
            f"  {name}: {', '.join(hits)}" for name, hits in violations.items()
        )
    )
