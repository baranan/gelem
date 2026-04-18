"""
operators/blendshape_avatar.py

BlendshapeAvatarOperator renders a 3D face avatar deformed by a set
of blendshape values. The output is a rendered face image saved to
disk, with the file path stored as a new column called 'avatar_path'.

This operator depends on BlendshapeOperator having run first — it
reads blendshape columns from the row's metadata.

Student C is responsible for implementing this operator.

Dependencies:
    pip install mediapipe
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from operators.base import BaseOperator
from operators.blendshapes import BLENDSHAPE_NAMES


class BlendshapeAvatarOperator(BaseOperator):
    """
    Renders a neutral 3D face deformed by the blendshape values
    stored in the dataset for each frame.

    Output: one JPEG image per frame stored in the project's
    outputs/ folder. The file path is written as 'avatar_path'.
    """

    name = "blendshape_avatar"
    create_columns_label = "Render blendshape avatar"
    output_columns = [("avatar_path", "avatar_path")]
    requires_image = False  # Reads blendshape values from metadata.

    def __init__(self, output_dir: Path | None = None):
        """
        Creates the operator.

        Args:
            output_dir: Folder where avatar images will be saved.

        TODO (Student C): Initialise the mediapipe face renderer here,
        if needed.
        """
        import tempfile
        self._output_dir = output_dir or (
            Path(tempfile.gettempdir()) / "gelem_avatars"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def create_columns(
        self,
        row_id: str,
        image: np.ndarray | None,
        metadata: dict,
    ) -> dict:
        """
        Renders a face avatar for one frame using that frame's
        blendshape values.

        Args:
            row_id:   The row being processed.
            image:    Not used (requires_image = False).
            metadata: Must contain blendshape columns (bs_jawOpen etc.)
                      for this to produce a meaningful avatar.

        Returns:
            Dict with 'avatar_path' key pointing to the rendered image.
            Returns {'avatar_path': None} if blendshapes are missing.

        TODO (Student C): Implement this method.

        Suggested approach:
            1. Read blendshape values from metadata:
               values = {name: metadata.get(name, 0.0)
                         for name in BLENDSHAPE_NAMES}
            2. If all values are None or missing, return
               {'avatar_path': None}
            3. Use mediapipe to render a face mesh deformed by
               these blendshape values.
            4. Save the rendered image:
               output_path = self._output_dir / f'{row_id}_avatar.jpg'
               self.save_image(rendered_array, output_path)
            5. Return {'avatar_path': str(output_path)}
        """
        # PLACEHOLDER: creates a gray placeholder image.
        from PIL import Image
        output_path = self._output_dir / f"{row_id}_avatar.jpg"
        placeholder = Image.new("RGB", (256, 256), color=(180, 180, 180))
        placeholder.save(output_path, "JPEG")
        print(f"[BlendshapeAvatarOperator] PLACEHOLDER for {row_id}")
        return {"avatar_path": str(output_path)}