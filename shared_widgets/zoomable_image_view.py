from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtCore import Qt, QTimer, Signal


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

    # Carries the new zoom factor relative to fit, emitted only from
    # wheelEvent so callers can tell a user-driven zoom apart from a
    # programmatic fit or reset.
    zoom_factor_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        # Panning is drag-hand only, so scrollbars are never needed for
        # navigation. Off also avoids a real fitInView glitch: fitInView
        # resets to 1:1 before measuring the viewport, and at 1:1 a
        # picture bigger than the pane would need scrollbars, which
        # shrinks the viewport it then measures -- producing a fit
        # slightly smaller than the pane can actually hold.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        from PySide6.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._pixmap_item = None
        # The absolute scale the most recent fit_to_view() produced --
        # the reference point a relative zoom factor is measured against.
        self._last_fit_scale: float | None = None
        # A zoom factor relative to fit, or None for plain fit. Set by a
        # caller that remembers a zoom across pictures (e.g. DetailWidget,
        # per active table); reapplied by reset_view().
        self._preferred_factor: float | None = None

    def show_pixmap(self, pixmap: QPixmap) -> None:
        """
        Displays a QPixmap in the view and schedules reset_view() (fit,
        then the preferred factor if one is set) two event-loop turns
        out.

        The fit is deferred rather than run here because this is usually
        called while the widget is still being constructed, before it is
        placed in its parent's layout -- fitInView against that stale
        size would compute the wrong scale. One deferred turn reaches
        the layout pass that first assigns the widget its real size;
        inserting a new widget into an existing layout schedules a
        second pass on top of that (the same reason
        tests/conftest.py's realize_widget() pumps the event queue
        twice), so the fit itself waits one turn further, for whichever
        of the two passes lands last.

        Both turns are scheduled with this view as their context object
        (QTimer.singleShot's context-object overload), not a bare
        lambda/bound method -- a caller that swaps in a new widget
        before the old one's deferred turns fire (e.g. double-clicking
        a different tile right away) can delete this view first, and
        without a context Qt would still invoke the callback against
        the now-deleted C++ object and raise. With a context, Qt drops
        the pending call instead once that object is destroyed.

        Args:
            pixmap: The QPixmap to display.
        """
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        QTimer.singleShot(
            0, self, lambda: QTimer.singleShot(0, self, self.reset_view)
        )

    def fit_to_view(self) -> None:
        """
        Scales and centers the current picture to fill the viewport --
        the same fit show_pixmap always performs. Public so a caller that
        set the picture through set_pixmap_keeping_view (which
        deliberately does not fit) can still ask for this fit once, e.g.
        the frame stepper (P1.4b part 2) fitting only on its first entry
        into frame view, after the page holding this view has its real
        on-screen size.

        No-op if no picture has been shown yet.
        """
        if self._pixmap_item is None:
            return
        self.fitInView(
            self._scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._last_fit_scale = self.transform().m11()

    def set_preferred_factor(self, factor: float | None) -> None:
        """
        Sets the zoom factor -- relative to fit -- that reset_view()
        applies after fitting. None means plain fit.
        """
        self._preferred_factor = factor

    def reset_view(self) -> None:
        """
        Fits the current picture to the viewport, then reapplies the
        preferred factor if one is set: the absolute scale becomes
        fit-scale times that factor, clamped into the same
        _MIN_SCALE/_MAX_SCALE range the wheel uses. The fit itself is
        never clamped.

        No-op (beyond the fit) if no preferred factor is set, or no
        picture has been shown yet.
        """
        self.fit_to_view()
        if self._preferred_factor is None or self._pixmap_item is None:
            return
        fit_scale = self._last_fit_scale
        if not fit_scale:
            return
        target = fit_scale * self._preferred_factor
        target = max(self._MIN_SCALE, min(self._MAX_SCALE, target))
        current = self.transform().m11()
        if current:
            factor = target / current
            if factor != 1.0:
                self.scale(factor, factor)

    def set_pixmap_keeping_view(self, pixmap: QPixmap) -> None:
        """
        Swaps the displayed QPixmap for a new one without resetting zoom
        or pan -- unlike show_pixmap's fitInView, which recenters and
        rescales every time. For the frame stepper (P1.4b part 2), which
        wants each stepped frame to appear at whatever zoom the
        researcher already chose.

        Args:
            pixmap: The QPixmap to display in place of the current one.
        """
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)

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
        configured scale range. Emits zoom_factor_changed with the
        resulting scale relative to the last fit -- the one place this
        view reports a user-driven zoom, as opposed to a programmatic
        fit or reset.
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
            if self._last_fit_scale:
                relative = self.transform().m11() / self._last_fit_scale
                self.zoom_factor_changed.emit(relative)