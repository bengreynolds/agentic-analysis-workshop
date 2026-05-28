from typing import Dict, Optional, List

from PySide6.QtCore import QPointF, QSize, Qt, QObject, QEvent, Slot
from PySide6.QtGui import QContextMenuEvent, QAction, QKeyEvent, QCursor
import pyqtgraph as pg
from PySide6.QtWidgets import QMenu, QGraphicsSceneMouseEvent, QGraphicsRectItem
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

from reachx.common import RxBodyPart, RX, RxCam
from reachx.data.datamanager import DataManager

_RX = RX()
""" Application-wide constants. """


class CamFrame(pg.PlotWidget):
    """
    A custom **PyQtGraph** ``PlotWidget`` tailored to display video frames from a ReachX rig camera and attach body part
    markers to individual frames for model training purposes. It can also display the predicted locations of detected
    markers on a frame for a session that has already been analyzed by a trained body part detection model. It is
    intended as a drop-in-place widget for displaying video from any of the available rig cameras.

    Frame images from a rig camera are 300x200 in legacy-cam rigs, and 256x256 in fixed-cam rigs. In either case, we
    scale them by 2 and add a 10-pixel border all around. So the widget is fixed in size at 620x420 when displaying
    content from a legacy-cam, and 532x532 for a fixed-cam. The coordinate system is laid out with the origin (0,0) at
    the top-left corner of the image, X increaseing rightward and Y downward. **It is vital that the Y-axis is inverted,
    otherwise the frame image will be upside down!**

    ReachX users define "annotated training data" for a body part detection ML network model by attaching body part
    markers to selected frames from the two camera videos recorded during at least one but typically several experiment
    sessions. ``CamFrame`` supports interactive features to attach a body part marker to the displayed frame:
     - When the mouse cursor is inside the widget, the normal cursor is replaced by H/V lines that intersect at the
       cursor's current location, marked by a small circle. A nearby text label displays the cursor position (X,Y) in
       image pixels -- X in [0, 299), Y in [0, 199).
     - A right-click raises a context menu to select a body part marker to place at the cursor position. There is also
       an option to clear all markers from the image. The user can move an existing marker simply by right-clicking
       somewhere else and selecting the relevent item from the context menu.
     - You can nudge the position of or delete an existing marker by left-clicking on it. The widget enters a transient
       "edit marker" mode: The crosshairs are hidden, the selected marker is highlighted, and the arrow keys can be
       used to fine-tune the marker's location in single-pixels increments; pressing Enter/Return restores normal
       operation. Alternatively, pressing the Backspace, Del or "X" key while in the "edit marker" mode will remove the
       selected marker and immediately return to normal operation.
     - Each body part marker is represented by a different symbol.

    To show/hide marker labels next to each displayed marker, use ``show_hide_marker_labels()``. ``CamFrame`` is
    initially configured with the marker labels hidden.

    To show/hide predicted body part locations, use ``show_predicted_markers()``. ``CamFrame`` is initially configured
    to show all predicted BP locations whenever there are any defined for the current frame.

    If no ML model is currently loaded in ReachX, then the body part marking facility described is disabled. The
    facility only applies to certain camera sources: ``RxCam.SIDE/FRONT`` for a legacy-cam session, ``RxCam.LEFT/RIGHT``
    for a fixed-cam session. Furthermore, the set of available body part markers depends on the camera displayed. The
    context menu described only exposes body part markers that can be placed on that camera's frames.

    **USAGE**
     - Simply construct an instance and install in a layout container as you would any other widget. The target camera
       (``RxCam``) and the application ``DataMananger`` singleton are passed to the constructor. ``CamFrame`` maintains
       a reference to the current loaded session (``SessionManager``) and responds appropriately whenever the current
       frame changes, or when a different session is loaded or there's a workspace switch.
     - ``CamFrame`` queries the loaded ML model for any body part markers attached to the current frame (eg, for a
       session that was previously marked). It also confirms any user-initiated change in the set of markers before
       actually rendering that change on the widget.
     - A flag passed to the constructor enables/disables the custom widget's interactive features.
     - You need two ``CamFrame`` objects to display ``RxCam.SIDE`` and ``RxCam.FRONT`` for a legacy-cam session, and
       another two to display ``RxCam.LEFT`` and ``RxCam.RIGHT`` for a fixed-cam session. Recommend using a stacked
       widget to switch between the two sets when the user switches from a legacy-cam to fixed-cam workspace or
       vice-versa.
    """
    _SCALE_FAC: int = 2
    """ The actual frame image is scaled by this factor in both width and height. """
    _NO_VIDEO_MSG = "No video available."
    """ This messsage appears in camera frame widget when no video is available. """
    _DROPPED_MSG = "Dropped frame."
    """ This message appears in camera frame widget when video is available but the current frame was dropped. """
    _OUT_OF_FRAME_XY: int = max(_RX.camera_frame_dims(False) + _RX.camera_frame_dims(True)) * 10
    """ An x- or y-coordinate value well outside the bounds of the view box for this camera frame widget. """
    _MARK_SZ: int = 16
    """ Size of bounding box for body part markers, in screen pixels. """

    def __init__(self, cam: RxCam, data_mgr: DataManager, interactive: bool = True):
        """
        Construct a ``CamFrame`` that displays the "current" video frame from the specified rig camera and, optionally,
        lets the user place body part markers at arbitrary locations on the frame image. Initially configured to
        display all model-predicted body part locations (if that information is available).

        :param cam: Camera identifier
        :param data_mgr: Singleton object that manages all data for ReachX.
        :param interactive: If True (the default), interactive features (dynamic cursor, body part markers) are
          enabled. Otherwise, the widget merely displays the current video frame from the specified camera source and
          does nothing else.
        """
        super().__init__()
        self._cam = cam
        """ The rig camera sourcing the video frames displayed here. """
        self._cam_w: int = 0
        """ Fixed width of a frame from the camera source (differs for legacy- vs fixed-cam). """
        self._cam_h: int = 0
        """ Fixed height of a frame from the camera source (differs for legacy- vs fixed-cam). """
        self._session = data_mgr.session_manager
        """ Encapsulates the current loaded session and its data, including video frames. """
        self._model_mgr = data_mgr.model_manager
        """ 
        Encapsulates the current body part detection ML model and its annotated training data, ie, body part
        markers attached to select frames from one or more experiment sessions. The interactive placement of body
        part markers is only enabled when a model is loaded in ReachX.
        """
        self._interactive = interactive
        """ If True, interactive features are enabled. This is set at construction time and cannot be changed. """
        self._cam_image = pg.ImageItem()
        """ Camera frame image container. """
        self._message_label: pg.LabelItem = pg.LabelItem(CamFrame._NO_VIDEO_MSG, size="20pt", color="#808080")
        """ This label is displayed centrally when video is not available (or current frame was dropped). """
        self._vline: Optional[pg.InfiniteLine] = None
        """ If interactive, this vertical line follows mouse when inside plot view box."""
        self._hline: Optional[pg.InfiniteLine] = None
        """ If interactive, this horizontal line follows mouse when inside plot view box."""
        self._cursor_dot: Optional[pg.TargetItem] = None
        """ 
        If interactive, this dot marks location of mouse cursor when inside plot view box and is labeled with
        the cursor coords (in the image's pixel coordinates).
        """
        self._markable_body_parts: List[RxBodyPart] = list()
        """ List of markable body parts on the camera source displayed (so we don't have to get it every time). """
        self._marker_menu: Optional[QMenu] = None
        """ Context menu for selecting the body part marker to be placed at cursor. """
        self._curr_markers: Dict[RxBodyPart, pg.TargetItem] = dict()
        """ Body part markers currently overlaid on the video frame image; initially empty. """
        self._marker_labels_on: bool = False
        """ If `True`, body part markers are labeled with marker nickname."""
        self._predicted_markers: Dict[RxBodyPart, pg.TargetItem] = dict()
        """ Predicted locations of any body part markers (analyzed session only); initially empty. """
        self._predictions_on: bool = True
        """ Whether predicted body part locations are marked. """
        self._hand_pellet_predictions_only: bool = False
        """ If ``True``, only show predicted locations of hand and pellet; else, all available predicted locations. """
        self._drag_start: Optional[QPointF] = None
        """ If not None, this is the starting point of an in-progress drag gesture in view (not image!) coords. """
        self._drag_rect: Optional[QGraphicsRectItem] = None
        """
        If interactive, this is the drag rectangle defined by the drag start position and the current cursor
        position. If no drag in progress, this is placed out of bounds.
        """
        self._edited_bp: Optional[RxBodyPart] = None
        """
        ID of selected body part marker when widget is in transient mode to fine-tune position of or delete that marker.
        None if widget is not currently in this mode.
        """

        # camera frame sizes are different for the fixed-cam vs legacy-cam rig setups!
        self._cam_w, self._cam_h = _RX.camera_frame_dims(self._cam.is_fixed_cam)
        self._markable_body_parts = RxBodyPart.markable_body_parts_on(self._cam)

        pi: pg.PlotItem = self.getPlotItem()
        pi.hideButtons()
        pi.hideAxis('left')
        pi.hideAxis('bottom')

        self._message_label.setParentItem(pi)
        self._message_label.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))

        vb: pg.ViewBox = self.getPlotItem().getViewBox()
        vb.setMenuEnabled(False)
        vb.setMouseEnabled(x=False, y=False)
        vb.setBorder(color='g', width=3)
        vb.setAspectLocked(True)
        vb.addItem(self._cam_image)
        vb.setRange(xRange=(0, self._cam_w * 2), yRange=(0, self._cam_h * 2), padding=0)

        # CRUCIAL: For image to be right side up, origin is at TL corner and Y increases DOWNWARD!
        vb.invertY(True)

        fixed_size = QSize(self._cam_w * 2 + 20, self._cam_h * 2 + 20)
        self.setMinimumSize(fixed_size)
        self.setMaximumSize(fixed_size)

        # interactive features...
        if self._interactive:
            self._vline = pg.InfiniteLine(pos=CamFrame._OUT_OF_FRAME_XY, angle=90, movable=False, pen='w')
            self._hline = pg.InfiniteLine(pos=CamFrame._OUT_OF_FRAME_XY, angle=0, movable=False, pen='w')
            self._cursor_dot = pg.TargetItem(
                symbol='o', brush=None, pen='w', movable=False, label=" ",
                labelOpts=dict(color='w', fill='k', offset=(10, -10), anchor=(0, 0)))
            self._marker_menu = QMenu(self)

            # NOTE: setting ignoreBounds is CRITICAL for any item placed outside the view box -- else the view box
            # range will adjust accordingly, and that will mess everything up!
            pi.addItem(self._vline, ignoreBounds=True)
            pi.addItem(self._hline, ignoreBounds=True)
            pi.addItem(self._cursor_dot, ignoreBounds=True)
            self._cursor_dot.setPos(CamFrame._OUT_OF_FRAME_XY, CamFrame._OUT_OF_FRAME_XY)
            self._vline.setZValue(2)   # so that lines and cursor dot are drawn on top of image!
            self._hline.setZValue(2)
            self._cursor_dot.setZValue(2)

            # hide cursor when inside widget. Instead, we draw H/V lines intersecting at cursor location
            self.setCursor(Qt.CursorShape.BlankCursor)
            self.installEventFilter(self)

            # left-click on an existing marker deletes it
            self.sceneObj.sigMouseClicked.connect(self._mouse_clicked)

            # custom context menu populated with body part labels. HOWEVER, we only expose the "side hand" labels for
            # the sideCam and the "front hand" labels for the frontCam.
            self._marker_menu.addAction("Clear Markers")
            self._marker_menu.addSeparator()
            for bp in self._markable_body_parts:
                self._marker_menu.addAction(str(bp))

            # rectangle for animating drag gesture -- initially placed out of bounds
            self._drag_rect = QGraphicsRectItem(self._OUT_OF_FRAME_XY, self._OUT_OF_FRAME_XY, 0, 0)
            self._drag_rect.setPen(pg.mkPen('r', width=2))
            self._drag_rect.setZValue(2)
            pi.addItem(self._drag_rect, ignoreBounds=True)

        self._update_frame()
        self._session.frame_ready.connect(self._update_frame)
        self._session.scorer_changed.connect(self._update_predicted_markers)

        data_mgr.session_loaded.connect(self._update_frame)
        data_mgr.active_workspace_switched.connect(self._update_frame)
        data_mgr.active_workspace_path_changed.connect(self._update_frame)

        # must update displayed BP markers upon switching to a different ML model, a different iteration of the current
        # model, or whenever the set of markers for the current model iteration changes.
        data_mgr.model_loaded.connect(self._update_markers)
        self._model_mgr.iter_loaded.connect(self._update_markers)
        self._model_mgr.markers_changed.connect(lambda _: self._update_markers())   # signal param ignored

    @Slot()
    def _update_frame(self) -> None:
        """
        Handler for the `frame_ready` signal: Updates the widget to display the image for the current frame, plus any
        body part markers attached to that frame. Also called whenever there's a change in the session loaded.

        Note that a camera in the fixed-cam setup will not be part of a legacy-cam session and vice versa. In this
        scenario, no action is taken.
        """
        if (self._session.is_fixed_cam is not None) and (self._session.is_fixed_cam != self._cam.is_fixed_cam):
            return

        img = self._session.current_frame(self._cam)
        if img is None:
            self._cam_image.clear()
            was_dropped = self._session.was_current_frame_dropped_on(self._cam)
            self._message_label.setText(self._DROPPED_MSG if was_dropped else self._NO_VIDEO_MSG)
        else:
            self._cam_image.setImage(img)
            self._cam_image.setRect(0, 0, self._cam_w * 2, self._cam_h * 2)
            self.autoRange()
            self._message_label.setText("")

        self._update_markers()
        self._update_predicted_markers()

    @Slot()
    def _update_markers(self) -> None:
        if not self._interactive:
            return
        if (self._session.is_fixed_cam is not None) and (self._session.is_fixed_cam != self._cam.is_fixed_cam):
            return

        if (self._session.id is None) or not self._model_mgr.model_loaded:
            markers = []
        else:
            # we only want the markers on the camera video shown in this widget!
            all_markers = self._model_mgr.get_markers_for_frame(self._session.id, self._session.current_frame_num)
            markers = [m for m in all_markers if m.cam == self._cam]

        # remove target items corresponding to parts not marked
        for part in self._markable_body_parts:
            if part not in [m.part for m in markers]:
                ti = self._curr_markers.pop(part, None)
                if ti is not None:
                    self.getPlotItem().removeItem(ti)

        # add or update target items corresponding to parts that are marked
        for m in markers:
            loc = QPointF(m.x_pix * CamFrame._SCALE_FAC, m.y_pix * CamFrame._SCALE_FAC)  # view coordinates!
            self._add_or_update_body_part_target_item(m.part, loc)

    def show_hide_marker_labels(self, show: bool) -> None:
        """
        Show/hide marker labels next to each displayed body part marker on the current frame.
        :param show ``True`` to show labels, ``False`` to hide them.
        """
        if show != self._marker_labels_on:
            self._marker_labels_on = show
            for part, ti in self._curr_markers.items():
                label_str = str(part) if self._marker_labels_on else None
                if self._marker_labels_on:
                    x_pix, y_pix = ti.pos().x() / CamFrame._SCALE_FAC, ti.pos().y() / CamFrame._SCALE_FAC
                    anchor = (0 if x_pix < self._cam_w / 2 else 1, 1 if y_pix < self._cam_h / 2 else 0)
                    offset = (10 if x_pix < self._cam_w / 2 else -10, 10 if y_pix < self._cam_h / 2 else -10)
                    label_opts = dict(color='w', fill='k', offset=offset, anchor=anchor)
                else:
                    label_opts = None
                ti.setLabel(label_str, label_opts)
            for part, ti in self._predicted_markers.items():
                label_str = str(part) if self._marker_labels_on else None
                if self._marker_labels_on:
                    x_pix, y_pix = ti.pos().x() / CamFrame._SCALE_FAC, ti.pos().y() / CamFrame._SCALE_FAC
                    anchor = (0 if x_pix < self._cam_w / 2 else 1, 1 if y_pix < self._cam_h / 2 else 0)
                    offset = (10 if x_pix < self._cam_w / 2 else -10, 10 if y_pix < self._cam_h / 2 else -10)
                    label_opts = dict(color='w', fill='k', offset=offset, anchor=anchor)
                else:
                    label_opts = None
                ti.setLabel(label_str, label_opts)

    def show_predicted_markers(self, show: bool, hand_pellet_only: bool) -> None:
        """
        Configure ``CamFrame`` to show/hide the predicted locations of body parts on the current frame if this
        information is available for the current session. Only sessions that have been analyzed by a body part detection
        model will have body part location predictions available.

        This feature is available only if the ``CamFrame`` is configured to be interactive.

        :param show: ``True`` to show predicted body part locations, ``False`` to hide them.
        :param hand_pellet_only: If ``True``, only hand and pellet locations are shown. If ``False``, all available body
            part location predictions are rendered.
        """
        if (show != self._predictions_on) or (hand_pellet_only != self._hand_pellet_predictions_only):
            self._predictions_on = show
            self._hand_pellet_predictions_only = hand_pellet_only
            self._update_predicted_markers()

    @Slot()
    def _update_predicted_markers(self) -> None:
        if not self._interactive:
            return
        if (self._session.is_fixed_cam is not None) and (self._session.is_fixed_cam != self._cam.is_fixed_cam):
            return

        predicted_locations = dict() if not self._predictions_on else (
            self._session.predicted_marker_locations_for_current_frame(self._cam, self._hand_pellet_predictions_only))

        # remove target items corresponding to parts for which there's no longer a prediction
        currently_shown = [part for part in self._predicted_markers.keys()]
        for part in currently_shown:
            if part not in predicted_locations:
                ti = self._predicted_markers.pop(part, None)
                if ti is not None:
                    self.getPlotItem().removeItem(ti)

        # add or update target items corresponding to parts that are marked
        for part, loc_tuple in predicted_locations.items():
            loc = QPointF(loc_tuple[0] * CamFrame._SCALE_FAC, loc_tuple[1] * CamFrame._SCALE_FAC)  # view coordinates!
            self._add_or_update_predicted_marker_target_item(part, loc)

    def _add_or_update_predicted_marker_target_item(self, part: RxBodyPart, loc: QPointF) -> None:
        """
        Helper method for ``_update_predicted_markers()``: Adds a target item to the plot to render the specified body
        part marker at the specified location (as predicted from analysis of session video). If there is already a
        target item for the specified part, merely update its position.

        To distinguish a user-placed body part marker from the predicated location for that body part, the predicted
        location symbol is 5 pixels larger and translucent rather than opaque.

        :param part: The body part identifier.
        :param loc: The marker location in view box coordinates: Origin at top left corner, 2 * image size.
        """
        if part in self._predicted_markers:
            self._predicted_markers[part].setPos(loc)
        else:
            shape, fill = RxBodyPart.marker_for(part)
            label_str = str(part) if self._marker_labels_on else None
            if self._marker_labels_on:
                x_pix, y_pix = loc.x() / CamFrame._SCALE_FAC, loc.y() / CamFrame._SCALE_FAC
                anchor = (0 if x_pix < self._cam_w / 2 else 1, 1 if y_pix < self._cam_h / 2 else 0)
                offset = (10 if x_pix < self._cam_w / 2 else -10, 10 if y_pix < self._cam_h / 2 else -10)
                label_opts = dict(color='w', fill='k', offset=offset, anchor=anchor)
            else:
                label_opts = None

            color = pg.mkColor(fill)
            color.setAlpha(128)
            ti = pg.TargetItem(pos=loc, size=self._MARK_SZ + 5, symbol=shape, brush=color,
                               pen=pg.mkPen('w', width=2), movable=False, label=label_str, labelOpts=label_opts)
            ti.setZValue(1)   # on top of image, but under cursor lines
            self._predicted_markers[part] = ti
            self.getPlotItem().addItem(ti)

    def _update_crosshairs(self, pos: Optional[QPointF] = None) -> None:
        """
        Show, hide or update the infinite crosshairs cursor.
        :param pos: The current cursor position in view coordinates. If None, the cursor elements are hidden. Else,
            update their position accordingly and set the cursor dot's label to reflect the position in image pixels
            (to get image pixel coordinates, divide by fixed scale factor).
        """
        if self._interactive:
            if pos is None:
                # we "hide" the cursor elements by putting them "out of bounds" and clearing the label
                self._vline.setPos(self._OUT_OF_FRAME_XY)
                self._hline.setPos(self._OUT_OF_FRAME_XY)
                self._cursor_dot.setPos(CamFrame._OUT_OF_FRAME_XY, CamFrame._OUT_OF_FRAME_XY)
                self._cursor_dot.label().setFormat(" ")
            elif self._edited_bp is None:   # while fine-tuning a marker's position, crosshairs are disabled
                self._vline.setPos(pos.x())
                self._hline.setPos(pos.y())
                self._cursor_dot.setPos(pos.x(), pos.y())
                x_pix, y_pix = pos.x() / CamFrame._SCALE_FAC, pos.y() / CamFrame._SCALE_FAC
                anchor = (0 if x_pix < self._cam_w / 2 else 1, 1 if y_pix < self._cam_h / 2 else 0)
                offset = (10 if x_pix < self._cam_w / 2 else -10, 10 if y_pix < self._cam_h / 2 else -10)
                self._cursor_dot.setLabel(f"({int(x_pix + 0.5)}, {int(y_pix + 0.5)})",
                                          labelOpts=dict(color='w', fill='k', offset=offset, anchor=anchor))

    def _update_drag(self, pos: Optional[QPointF] = None) -> None:
        """
        Start, stop, or update the rectangle-drawing drag gesture on the camera frame.
        :param pos: The current cursor position in view coordinates. If valid and a drag has not started, initialize
            the operation. If valid and a drag is in progress, update the size and position of the drag rectangle. If
            None and a drag is in progress, terminate the operation.
        :return:
        """
        if self._interactive:
            if pos is None:
                if self._drag_start is not None:
                    # self._test_tracking_after_drag()
                    self._drag_start = None
                    self._drag_rect.setRect(self._OUT_OF_FRAME_XY, self._OUT_OF_FRAME_XY, 0, 0)
            elif self._drag_start is None:
                # force start location to lie inside image rect
                x = max(0.0, min(float(self._SCALE_FAC * self._cam_w), pos.x()))
                y = max(0.0, min(float(self._SCALE_FAC * self._cam_h), pos.y()))
                self._drag_start = QPointF(x, y)
                self._drag_rect.setRect(self._drag_start.x(), self._drag_start.y(), 0, 0)
            else:
                # don't allow rectangle to follow cursor off image
                x = max(0.0, min(float(self._SCALE_FAC * self._cam_w), pos.x()))
                y = max(0.0, min(float(self._SCALE_FAC * self._cam_h), pos.y()))
                self._drag_rect.setRect(
                    min(self._drag_start.x(), x),
                    min(self._drag_start.y(), y),
                    abs(self._drag_start.x() - x),
                    abs(self._drag_start.y() - y)
                )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Custom handling of certain events on this widget:
          - When mouse leaves the widget, exit the transient "edit marker" mode (if necessary) and ensnure that the
            crosshairs cursor is hidden.
          - While in the "edit marker" mode: (1) override application-wide shortcuts since this widget needs to respond
            to the arrow keys to fine-tune the selected marker's location. (2) Update selected marker's location when
            any of the arrow keys are pressed. (3) Exit the mode if the Return/Enter key is pressed. (4) Delete the
            selected marker and exit the mode when the Backspace/Del/X key is pressed.

        Filter NOT installed if interactive features disabled.

        NOTE: If user drags mouse outside widget, the Leave event is not received b/c the widget is holding onto the
        mouse. We deal with this situation in `mouseMoveEvent()`.
        """
        if watched == self:
            if event.type() == QEvent.Type.Leave:
                self._exit_edit_marker_mode()
                self._update_crosshairs(None)
            elif self._edited_bp is not None:
                if event.type() == QEvent.Type.ShortcutOverride:
                    # circumvent all application shortcuts when widget is in the transient edit-marker mode, including
                    # the arrow keys that are used to fine-tune selected marker's location
                    event.accept()
                    return True
                elif event.type() == QEvent.Type.KeyPress:
                    assert isinstance(event, QKeyEvent)
                    key = QKeyEvent(event).key()
                    if key in [Qt.Key.Key_Backspace, Qt.Key.Key_Delete, Qt.Key.Key_X]:
                        self._model_mgr.detach_session_marker(self._session.id, self._session.current_frame_num,
                                                              which=(self._cam, self._edited_bp))
                        self._exit_edit_marker_mode()
                    elif key in [Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down]:
                        self._nudge_edited_marker(key)
                    elif key in [Qt.Key.Key_Enter, Qt.Key.Key_Return]:
                        self._exit_edit_marker_mode()

        return super().eventFilter(watched, event)

    def _nudge_edited_marker(self, key: int) -> None:
        """
        In the transient "edit marker" mode only: Update the selected body part marker by nudging up, down, left or
        right by one image pixel.
        :param key: Key code -- should be one of `Qt.Key.Key_Up, _Down, _Left, _Right`.
        """
        if self._edited_bp in self._curr_markers:
            marker = next((m for m in
                           self._model_mgr.get_markers_for_frame(self._session.id, self._session.current_frame_num)
                           if m.part == self._edited_bp and m.cam == self._cam), None)
            assert (marker is not None)

            x_pix = marker.x_pix + (-1 if key == Qt.Key.Key_Left else (1 if key == Qt.Key.Key_Right else 0))
            y_pix = marker.y_pix + (-1 if key == Qt.Key.Key_Up else (1 if key == Qt.Key.Key_Down else 0))
            self._model_mgr.attach_session_marker(self._session.id, self._session.current_frame_num,
                                                  self._cam, self._edited_bp, x_pix, y_pix)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """ If interactive features enabled, a left button press initiates the rectangle-drawing drag gesture. """
        if self._interactive and (event.button() == Qt.MouseButton.LeftButton):
            self._update_drag(self.getPlotItem().getViewBox().mapSceneToView(event.pos()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        IF interactive features enabled, update the "infinite crosshairs cursor" and an in-progress drag gesture
        appropriately. Detect when the mouse is dragged (left button still pressed) outside the widget, in which case
        turn off the crosshairs cursor but leave the drag rectangle in place.
        :param event: The mouse movement event.
        """
        if self._interactive:
            pi: pg.PlotItem = self.getPlotItem()
            pos: QPointF = event.pos()
            if pi.sceneBoundingRect().contains(pos):
                loc: QPointF = pi.getViewBox().mapSceneToView(pos)
                self._update_crosshairs(loc)
                if event.buttons() == Qt.MouseButton.LeftButton:
                    self._update_drag(loc)
            elif event.buttons() == Qt.MouseButton.LeftButton:
                self._update_crosshairs(None)
        super().mouseMoveEvent(event)  # Call the base class implementation

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """ If interactive features enabled, releasing the left mouse button terminates the ongoing drag gesture. """
        if self._interactive and (event.button() == Qt.MouseButton.LeftButton):
            self._update_drag(None)
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, evt: QContextMenuEvent) -> None:
        """
        Raise the body part marker context menu, but ONLY if interactive features enabled, and only if a body part
        detection ML model is currently loaded in ReachX.
        """
        if not (self._interactive and self._model_mgr.model_loaded):
            return

        selected: QAction = self._marker_menu.exec(evt.globalPos())
        if isinstance(selected, QAction):
            if selected.text() == "Clear Markers":
                self._model_mgr.detach_session_marker(self._session.id, self._session.current_frame_num, None)
            else:
                loc: QPointF = self.getPlotItem().getViewBox().mapSceneToView(evt.pos())
                # noinspection PyTypeChecker
                pix_loc: QPointF = loc / CamFrame._SCALE_FAC   # converts to image pixels
                part = RxBodyPart.from_nickname(selected.text())  # menu item text == body part nickname!
                if part is None:
                    return
                self._model_mgr.attach_session_marker(self._session.id, self._session.current_frame_num, self._cam,
                                                      part, int(pix_loc.x() + 0.5), int(pix_loc.y() + 0.5))

    def _add_or_update_body_part_target_item(self, part: RxBodyPart, loc: QPointF) -> None:
        """
        Helper method adds a target item to the plot to render the specified body part marker at the specified
        location. If there is already a target item for the specified part, merely update its position.
        :param part: The body part identifier.
        :param loc: The marker location in view box coordinates: Origin at top left corner, 2 * image size.
        """
        if part in self._curr_markers:
            self._curr_markers[part].setPos(loc)
        else:
            shape, fill = RxBodyPart.marker_for(part)
            label_str = str(part) if self._marker_labels_on else None
            if self._marker_labels_on:
                x_pix, y_pix = loc.x() / CamFrame._SCALE_FAC, loc.y() / CamFrame._SCALE_FAC
                anchor = (0 if x_pix < self._cam_w / 2 else 1, 1 if y_pix < self._cam_h / 2 else 0)
                offset = (10 if x_pix < self._cam_w / 2 else -10, 10 if y_pix < self._cam_h / 2 else -10)
                label_opts = dict(color='w', fill='k', offset=offset, anchor=anchor)
            else:
                label_opts = None

            ti = pg.TargetItem(pos=loc, size=self._MARK_SZ, symbol=shape, brush=fill, pen=pg.mkPen('w', width=2),
                               movable=False, label=label_str, labelOpts=label_opts)
            ti.setZValue(1)   # on top of image, but under cursor lines
            self._curr_markers[part] = ti
            self.getPlotItem().addItem(ti)

    @Slot(MouseClickEvent)
    def _mouse_clicked(self, evt: MouseClickEvent) -> None:
        """
        If user left-clicks on a body part marker, highlight the marker and enter the transient "edit marker" mode to
        fine-tune the marker's location (using the arrow keys) or, alternatively, delete it (the 'x', delete or
        backspace keys).
        """
        if not (self._interactive and self._model_mgr.model_loaded):
            return
        for bp, ti in self._curr_markers.items():
            ti_loc = ti.pos()
            ti_pix: pg.Point = self.getPlotItem().getViewBox().mapViewToScene(ti_loc)
            marker_half_sz = int(self._MARK_SZ / 2 + 2)
            if ((abs(ti_pix.x() - evt.pos().x()) <= marker_half_sz) and
                    (abs(ti_pix.y() - evt.pos().y()) <= marker_half_sz)):
                self._enter_edit_marker_mode(bp)
                return

    def _enter_edit_marker_mode(self, bp: RxBodyPart) -> None:
        if self._edited_bp is None:
            if bp not in self._curr_markers:
                return
            self._edited_bp = bp
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._update_crosshairs(None)
            # highlight corresponding target item to by inverting its stroke and fill colors
            ti: pg.TargetItem = self._curr_markers[self._edited_bp]
            ti.setPen(pg.mkPen(RxBodyPart.marker_for(self._edited_bp)[1], width=4))
            ti.setBrush('w')

    def _exit_edit_marker_mode(self) -> None:
        if self._edited_bp is not None:
            # if edited marker was not deleted, restore corresponding target item to normal appearance
            if self._edited_bp in self._curr_markers:
                ti: pg.TargetItem = self._curr_markers[self._edited_bp]
                ti.setPen('w', width=2)
                ti.setBrush(RxBodyPart.marker_for(self._edited_bp)[1])
            self._edited_bp = None
            self.clearFocus()
            # restore crosshairs if current mouse location still inside widget
            pos = self.mapFromGlobal(QCursor.pos())
            pos = self.mapToScene(pos)
            vb: pg.ViewBox = self.getPlotItem().getViewBox()
            loc: QPointF = vb.mapSceneToView(pos)
            x_range, y_range = vb.viewRange()
            if (x_range[0] < loc.x() < x_range[1]) and (y_range[0] < loc.y() < y_range[1]):
                self._update_crosshairs(loc)
