from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtCore import Qt


class ZoomableImageView(QGraphicsView):
    """
    A QGraphicsView that supports zoom via scroll wheel and
    pan via click-and-drag.

    This widget is also used by the image renderer in 'detail' mode —
    the renderer imports and instantiates it directly. Keep it in this
    file (ui/detail_widget.py) so it stays with the UI layer.

    TODO (Student A): Implement smooth zoom and pan.
    """

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
        Zooms in or out when the scroll wheel is used.

        TODO (Student A): Implement smooth zoom with min/max limits.
        """
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)