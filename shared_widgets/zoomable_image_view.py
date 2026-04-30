from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtCore import Qt


class ZoomableImageView(QGraphicsView):
    """
    A QGraphicsView that supports zoom via scroll wheel and
    pan via click-and-drag.

    This widget is used by the image renderer in 'detail' mode —
    the renderer imports and instantiates it directly.
    """

    # Absolute scale bounds relative to 1:1 pixels-on-screen. These are
    # safety limits; fitInView on show_pixmap places the initial zoom
    # somewhere inside the range regardless of image size.
    _MIN_SCALE = 0.18
    _MAX_SCALE = 3.5
    _ZOOM_STEP = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        from PySide6.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._pixmap_item = None

    def show_pixmap(self, pixmap: QPixmap) -> None:
        """
        Displays a QPixmap in the view, fitting it to the viewport.

        Args:
            pixmap: The QPixmap to display.
        """
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self.fitInView(
            self._scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def current_pixmap(self) -> QPixmap | None:
        """
        Returns the QPixmap currently displayed, or None if no image
        has been shown yet. Callers (e.g. DetailWidget's Save-as-PNG)
        should use this rather than reaching into the private
        _pixmap_item attribute.
        """
        if self._pixmap_item is None:
            return None
        return self._pixmap_item.pixmap()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Zooms in or out when the scroll wheel is used, clamped to the
        configured scale range.
        """
        factor = self._ZOOM_STEP if event.angleDelta().y() > 0 else 1 / self._ZOOM_STEP
        current = self.transform().m11()
        target  = current * factor

        if target < self._MIN_SCALE:
            factor = self._MIN_SCALE / current
        elif target > self._MAX_SCALE:
            factor = self._MAX_SCALE / current

        if factor != 1.0:
            self.scale(factor, factor)