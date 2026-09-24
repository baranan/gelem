"""
media/extensions.py

The one authoritative set of file extensions Gelem treats as media.

It lives in its own module so both the data layer
(models/table_schema.py, models/dataset.py) and anything under media/ can
import it without a cycle and without either side re-declaring the list.
Before P1.8d-2 there were two copies -- a dict in models/dataset.py and a
frozenset in models/table_schema.py -- which could drift apart: a format
added to one and not the other would be recognised by a folder scan but
tagged "text" when the same path appeared in a CSV column, or the reverse.

Standard library only. No pandas, no Qt, no project imports, so every
layer is free to depend on it.
"""

from __future__ import annotations

import pathlib

# Lowercase and dot-prefixed. Callers compare with
# `value.lower().endswith(ext)`, so a file whose extension differs only in
# case still matches. To support a new format, add its extension to
# IMAGE_EXTENSIONS or VIDEO_EXTENSIONS below and nowhere else.
#
# This is the one authoritative image/video split (P1.2c-2 review round 4).
# Before it, media/resolver.py and operators/operator_registry.py each held
# their own private copy of the image half, and column_types/renderers.py
# held a third, independent copy of both halves for the display layer --
# exactly the two-copies-drift failure this module's own docstring already
# warns about for MEDIA_EXTENSIONS. column_types/renderers.py's copy is
# deliberately left alone: it is the display layer's own local dispatch,
# not a caller of this module, and unifying it is a separate decision.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def _ends_with_extension_in(s: str, extensions: frozenset[str]) -> bool:
    """Shared body of ends_with_media_extension: true when `s` ends in a
    member of `extensions`, case-insensitively. Not exported -- callers
    outside this module go through ends_with_media_extension (against
    MEDIA_EXTENSIONS) so the extension set stays consulted in exactly the
    places this module already documents.
    """
    lowered = s.lower()
    return any(lowered.endswith(ext) for ext in extensions)


def ends_with_media_extension(s: str) -> bool:
    """True when `s` ends in a member of MEDIA_EXTENSIONS, case-insensitively.

    The one atomic extension check. Callers that already have a single
    candidate string in hand (a parsed path portion, one half of a
    cheaply-split cell) compare it against MEDIA_EXTENSIONS through this
    function only, so the extension set is consulted in exactly one place.
    """
    return _ends_with_extension_in(s, MEDIA_EXTENSIONS)


def _looks_like_extension_in(cell: str, extensions: frozenset[str]) -> str | None:
    """Shared body of looks_like_media_extension and
    looks_like_video_extension: the qualifying portion of `cell` -- EITHER
    the whole string OR the portion before its first literal '#' -- if
    either ends in a member of `extensions`, case-insensitively; None if
    neither does. See looks_like_media_extension's own docstring for why
    both spellings must be checked and why the '#' index is never taken
    from a lowered copy.
    """
    if _ends_with_extension_in(cell, extensions):
        return cell
    hash_index = cell.find("#")
    if hash_index == -1:
        return None
    before_hash = cell[:hash_index]
    if _ends_with_extension_in(before_hash, extensions):
        return before_hash
    return None


def looks_like_video_extension(cell: str) -> str | None:
    """Same rule as looks_like_media_extension, checked against
    VIDEO_EXTENSIONS only, not the full MEDIA_EXTENSIONS set.

    For a caller that must reject a still-image cell as "not a video" at
    this same cheap, pre-parse gate, specifically, rather than accepting
    it as media in general and discovering only much later, deep inside
    the resolver
    (which refuses to decode_video_span() an image path), that it was
    never a video. Using looks_like_media_extension there would let an
    image cell through the gate and have it counted among the resolver's
    own decode failures instead of the gate's "not a video" skip.
    """
    return _looks_like_extension_in(cell, VIDEO_EXTENSIONS)


def is_image_path(path: str) -> bool:
    """True when `path` -- an already-parsed PATH PORTION (never a raw,
    possibly-fragmented cell) -- names a still image, by its extension
    alone.

    This is the single authority for that specific test (P1.7-2 round 4):
    before it existed, media/resolver.py and operators/frame_operator.py
    each carried their own identical private copy, and column_types/
    renderers.py and operators/operator_registry.py each carry a further
    inline copy of the same check -- see those modules' own history for
    why unifying every one of them is a separate decision from this one.

    Unlike looks_like_video_extension / looks_like_media_extension above,
    this takes a bare path with no '#' fragment to strip -- a caller
    holding a raw, possibly-fragmented cell parses it first (through
    media/media_address.py) and passes the parsed address's own `.path`.
    """
    return pathlib.PurePosixPath(path).suffix.lower() in IMAGE_EXTENSIONS


def looks_like_media_extension(cell: str) -> str | None:
    """The qualifying portion of `cell` -- EITHER the whole string OR the
    portion before its first literal '#' -- if either ends in a media
    extension, case-insensitively; None if neither does.

    String operations only -- no media_address.parse(), no filesystem
    access. Safe to call on either a raw, unparsed cell (which may carry
    a '#' fragment) or an already-parsed, fragment-free path portion
    (which will not contain an unescaped '#', so only the whole-string
    check ever applies): the two callers this module was written for --
    models.table_schema._looks_like_media_path's Gate 1, and
    models.dataset's cheap pre-gate -- share this one implementation so
    the extension rule cannot drift between them (P1.8e-2b-1's Unicode
    bug existed only because there were, briefly, two copies). The
    pre-gate also uses the returned portion for its own Gate 2 (does
    THIS portion carry a directory separator), rather than re-deriving
    which half of the cell qualified.

    Both spellings must be checked because a media cell can end in its
    extension either way: "photos/img#2.png" ends in ".png" only as the
    WHOLE string (its pre-'#' portion, "photos/img", has no extension),
    while "clip.mp4#f=1234" ends in ".mp4" only in its pre-'#' portion
    (the whole string ends in "=1234").

    The '#' index is found in `cell` and used to slice `cell` -- never an
    index from one string used to slice a DIFFERENT one. str.lower() can
    change a string's LENGTH under Unicode case-folding (the Turkish
    dotted capital 'İ' lowercases to the two-codepoint 'i' +
    combining-dot-above): an index found by searching a lowered copy
    would land at the wrong offset if used to slice the original (or vice
    versa), silently cutting the extension short. Finding and slicing the
    SAME string sidesteps that -- the lowering that decides the match
    happens afterward, inside ends_with_media_extension, on the already
    correctly-sliced substring, where it can no longer disagree with an
    index computed elsewhere.
    """
    return _looks_like_extension_in(cell, MEDIA_EXTENSIONS)
