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
