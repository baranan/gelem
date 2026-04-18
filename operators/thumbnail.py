"""
operators/thumbnail.py

ThumbnailOperator generates thumbnail and preview images for each item
in the dataset. This is the first operator that runs when a folder is
loaded — it is what makes the gallery display images rather than
gray placeholders.

Unlike most operators, ThumbnailOperator does not add columns to the
dataset. Instead it writes files to ArtifactStore. It is called
directly by ArtifactStore.request_thumbnail() rather than through
OperatorRegistry.

This operator is provided centrally as a reference implementation.
Student C can use this file as a template when writing new operators.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from operators.base import BaseOperator


class ThumbnailOperator(BaseOperator):
    """
    Generates thumbnail (150px) and preview (600px) images for gallery
    display. Writes them to ArtifactStore.

    This operator does not appear in the Operators menu because it runs
    automatically when a folder is loaded — the researcher never needs
    to run it manually. All label attributes are left as None.
    """

    name = "thumbnail"
    output_columns = []

    # No labels — this operator is internal and never shown in the menu.
    create_columns_label = None
    create_table_label   = None
    create_display_label = None

    THUMBNAIL_SIZE = (150, 150)
    PREVIEW_SIZE   = (600, 600)

    def generate(
        self,
        row_id: str,
        full_path: Path,
        artifact_store,
    ) -> bool:
        """
        Generates thumbnail and preview images for one item and stores
        them in ArtifactStore. Designed to run in a background thread.

        Args:
            row_id:         The item to generate thumbnails for.
            full_path:      Path to the full-resolution source image.
            artifact_store: The ArtifactStore instance to write into.

        Returns:
            True if successful, False if the file could not be processed.
        """
        try:
            from PIL import Image

            path = Path(str(full_path))
            if not path.exists():
                print(f"[ThumbnailOperator] File not found: {path}")
                return False

            with Image.open(path) as img:
                img = img.convert("RGB")

                # Generate thumbnail.
                thumb = img.copy()
                thumb.thumbnail(self.THUMBNAIL_SIZE, Image.LANCZOS)
                artifact_store.put(
                    row_id, "thumbnail",
                    np.array(thumb, dtype=np.uint8)
                )

                # Generate preview.
                preview = img.copy()
                preview.thumbnail(self.PREVIEW_SIZE, Image.LANCZOS)
                artifact_store.put(
                    row_id, "preview",
                    np.array(preview, dtype=np.uint8)
                )

            return True

        except Exception as e:
            print(f"[ThumbnailOperator] Error processing {full_path}: {e}")
            return False