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

# Lowercase and dot-prefixed. Callers compare with
# `value.lower().endswith(ext)`, so a file whose extension differs only in
# case still matches. To support a new format, add its extension here and
# nowhere else.
MEDIA_EXTENSIONS = frozenset(
    {
        # images
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        # video
        ".mp4", ".mov", ".avi", ".mkv", ".webm",
    }
)


def ends_with_media_extension(s: str) -> bool:
    """True when `s` ends in a member of MEDIA_EXTENSIONS, case-insensitively.

    The one atomic extension check. Callers that already have a single
    candidate string in hand (a parsed path portion, one half of a
    cheaply-split cell) compare it against MEDIA_EXTENSIONS through this
    function only, so the extension set is consulted in exactly one place.
    """
    lowered = s.lower()
    return any(lowered.endswith(ext) for ext in MEDIA_EXTENSIONS)


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
    if ends_with_media_extension(cell):
        return cell
    hash_index = cell.find("#")
    if hash_index == -1:
        return None
    before_hash = cell[:hash_index]
    if ends_with_media_extension(before_hash):
        return before_hash
    return None
