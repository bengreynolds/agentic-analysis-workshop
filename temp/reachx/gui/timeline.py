from enum import Enum
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSize, QObject, QEvent, QPointF, Slot, Qt, QPoint
from PySide6.QtGui import QPainterPath, QPen, QKeyEvent, QCursor
from PySide6.QtWidgets import QGraphicsSceneMouseEvent, QGraphicsPathItem, QGraphicsRectItem, QMenu, QDialog, \
    QVBoxLayout, QRadioButton, QGroupBox, QGridLayout, QDialogButtonBox
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

from reachx.data.datamanager import DataManager
from reachx.common import RxEvent, RxReach, RxReachSegment, RxHandPos
from reachx.uicommon import RxIcons

_EVENT2MARKER: Dict[RxEvent, str] = {
    RxEvent.T6000: 'blue',
    RxEvent.DETECTED: 'yellow',
    RxEvent.DELIVERY: 'green',
    RxEvent.T5000: 'red',
    RxEvent.REACHED: 'magenta'
}
""" 
Maps each defined session event to the fill color of the marker that represents that event in the timeline. The 
color strings must be values accepted as the 'brush' attributes to the PyQtGraph TargetItem that renders the marker.
"""


class _TLMode(Enum):
    """ Enumeration of the different operational modes of the ``TimeLineWidget``. """
    NORMAL = 1
    """ Normal mode. """
    ADD_REACH = 2
    """ 
    Add reach mode. User interactively defines new reach segments or edits an existing reach (by changing the position
    of one of its key frames). Panning enabled, in which the current frame follows the mouse cursor. 
    """
    def __str__(self) -> str:
        return ["", "[add/edit reach]"][self.value - 1]


class TimeLineWidget(pg.PlotWidget):
    """
    Custom PyQtGraph PlotWidget that displays a portion of the event timeline for the recorded session spanning +/-N
    frames on either side of the current frame number, marking all events and reach segments occuring within the
    visible timespan.

    Each recorded event (see ``RxEvent``) is indicated by a small triangle at the event frame; the triangle color
    reflects the event type. A hideable legend maps event type to symbol.

    Each visible reach segment is represented by a horizontal rectangular bar spanning the duration of the reach, with
    a vertical line marking the so-called "reachMax" keyframe within the segment. The bar color reflects the result of
    the reach: green ==> `RxReach.GRABBED`, darkgoldenrod ==> `.MISSED`, orange ==> `.DROPPED`, and red ==> `.STALLED`.

    A session has two categories of reach segments: the user-curated list and an auto-generated list. The latter is
    created by running a reach segmenation algorithm over the hand and pellet trajectories prepared when analyzing the
    session with a ReachX body part detection model, aka, a "scorer". The two categories are distinguished visually in
    two ways: the baseline for the auto-generated, or "scored", reach segments is above that of the curated segments,
    and the pen color outlining the rectangular bar for each reach segment is blue for a scored reach and white for a
    curated reach.

    Whenever the mouse cursor is inside the widget, a vertical line follows the cursor, labeled with the frame number
    that corresponds to the cursor position. If the user left-clicks, the widget requests that the corresponding frame
    number become the current frame -- a means by which the user has finer-grained control over the current frame.

    To allow the user to manually add, edit or delete CURATED reach segments, the widget has two modes of operation:
     - **Normal mode**: Crosshair cursor, with white vertical line following cursor as mouse moves inside the widget.
        - Left-click changes the current frame, as described above.
        - If mouse is inside widget and the vertical line cursor intersects a horizontal bar representing a scored
          reach segment, pressing the **C** key will add that reach to the curated set.
        - Similarly, if the vertical line intersects a horizontal bar representing a curated reach, pressing the **X**,
          **Delete**, or **Backspace** key will delete it. Or, if you press **E**, a popup dialog appears to let you
          change two parameters or that reach: the result code or the hand position relative to pellet at "reachMax".
          This is the same dialog that appears when you define a new reach segment in "add/edit reach" mode (see next).
     - **Add/edit reach mode**: Toggle between the Normal mode and this mode by pressing the 'A' key while the mouse is
       inside the widget.
        - A pin-shaped cursor appears AND the current frame starts tracking the mouse location as long as it remains
          inside the widget. As the mouse moves, note how the ``MainCameraView`` is updated.
        - The "current frame" marker, a white upward-pointing arrow normally fixed at the horizontal center, now follows
          the vertical time cursor.
        - To construct a new reach segment, the user left-clicks at the desired starting frame. Moves the mouse to the
          right and clicks again to mark the "reachMax" keyframe. Then moves the mouse further to the right and clicks
          to mark the segment's end. At this point a modal popup window appears to let the user choose: the result
          of the reach and the hand position relative to pellet at the  "reachMax" keyframe. You can cancel out of the
          popup if you want to discard the newly defined segment. If you hit OK and the newly defined segment is valid,
          a representation of the segment appears on the timeline, and the user can begin to define another segment.
        - While defining the segment, the vertical line cursor changes color with each click and a two-color rectangle
          anchored to the segment's start frame follows the cursor. The "goldenrod" portion of the rectangle spans
          between the "reachStart" and "reachMax" keyframes, while the  "cornflowerblue" portion spans between
          "reachMax" and "reachEnd".
        - If the cursor exits the widget bounds, any partially defined reach segment is discarded.
        - To modify an existing reach segment, the user left-clicks near one of its 3 keyframes. The two-color rectangle
          reappears. Depending on which keyframe is selected, one or both portions of the rectangle adjust as the user
          moves the keyframe to a new location. Clicking again updates the reach segment accordingly (as long as the
          new keyframe value satisfies 0 <= reachStart < reachMax < reachEnd < #totalFrames).
        - Note that the **C**, **E**, and **X/Delete/Backspace** hot key actions in **Normal** mode also work in this
          mode -- UNLESS the user is in the process of defining a reach segment as described.

     The op mode is "sticky" in the sense that, if the mouse leaves the widget and then re-enters, the op mode in
     effect when it last left the widget is restored. When the mouse is inside the widget, a small label in the top-R
     corner reflects the op mode in effect (for normal mode, there is no label).
    """

    TIME_SPANS = [50, 100, 200, 500, 1000, 5000]
    """ 
    The different time spans, T, supported by the timeline widget, in #frames. It displays the frame interval 
    [C-T .. C+T], where C is the current frame number.
    """
    _DEFAULT_SPAN = TIME_SPANS[3]
    """ The default time span. """
    _NO_VIDEO_MSG = "No video available."
    """ This messsage appears in camera frame widget when no video is available. """
    _FIXED_HT_PIX = 150
    """ Fixed height H of this widget in pixels."""
    _MIN_WIDTH_PIX = 600
    """ Minimum width of this widget in pixels. """
    _TICK_MIN_Y = 40
    """ Y-coord for tick mark bottoms. """
    _TICK_HT = 10
    """ Tick height in pixels. """
    _EVENT_MARKER_SZ = 15
    """ Height of rectangular event marker in pixels. """
    _EVENT_MARKER_Y = 60
    """ Y-coord for event marker locations. """
    _CURATED_REACH_BASE = 80
    """ Bottom edge of rectangular bar representing a curated reach segment lies at this Y-coordinate, in pixels. """
    _AUTOGEN_REACH_BASE = 110
    """ Bottom edge of rectangular bar representing an auto-generated reach segment, in pixels. """
    _REACH_SEG_H = 12
    """ Height of the rectangular bar representing a reach segment, in pixels. """
    _REACH_SEG_EXTRA = 4
    """ The vertical line through rectangular bar extends beyond its top and bottom edges by this much, in pixels. """
    _CURATED_BAR_PEN: QPen = pg.mkPen('w', width=2)
    """ Pen used to outline the rectangular bar representing a curated reach segment. """
    _AUTOGEN_BAR_PEN: QPen = pg.mkPen('cornflowerblue', width=3)
    """ Pen used to outline the rectangular bar representing an auto-generated reach segment. """
    _BORDER_PEN_NO_FOCUS: QPen = pg.mkPen('gray', width=2)
    """ Pen that outlines view box when mouse is outside widget."""
    _BORDER_PEN_FOCUS: QPen = pg.mkPen('deepskyblue', width=8)
    """ Pen that oulines view box when mouse is inside widget. """

    def __init__(self, data_mgr: DataManager):
        """
        Construct the timeline widget.
        :param data_mgr: Singleton object that manages all data for ReachX.
        """
        super().__init__()
        self._add_reach_dlg = _AddReachDlg(self)
        """ 
        Modal dialog raised to specify the result and relative hand position for a just-defined reach segment, or
        to edit the result and/or hand position for an existing reach segment.
        """
        self._session = data_mgr.session_manager
        """ Encapsulates the current loaded session and its data, including the current frame number, events, etc. """
        self._mode: _TLMode = _TLMode.NORMAL
        """ Current operational mode. """
        self._last_mode: _TLMode = _TLMode.NORMAL
        """ 
        The operational mode when the cursor last exited the widget -- so we can restore the mode when cursor
        re-enters.
        """
        self._mouse_inside = False
        """ Flag set whenever mouse is within the timeline widget's bounds. """
        self._message_label: pg.LabelItem = pg.LabelItem(self._NO_VIDEO_MSG, size="20pt", color="#808080")
        """ This label is displayed centrally when video timeline is not available. """
        self._mode_label: pg.LabelItem = pg.LabelItem("", size="12pt", color="deepskyblue")
        """ This label appears in TR corner when widget is in one of the special op modes. """
        self._vline = pg.InfiniteLine(pos=self._DEFAULT_SPAN*10, angle=90, movable=False, label="0",
                                      labelOpts=dict(position=0.05))
        """ Vertical line follows mouse cursor when it is inside the plot view box. Initially placed outside box. """
        self._vline_shown = False
        """ Flag set when the vertical cursor line is shown. """
        self._axis_polyline: Optional[pg.PlotDataItem] = None
        """ Renders the timeline "axis" as a sequence of tick marks and a horizontal line bisecting them. """
        self._curr_frame_marker = pg.TargetItem(
            pos=(0, self._TICK_MIN_Y-self._TICK_HT/2), size=self._TICK_HT+2, symbol='t1', brush='w', pen='w',
            movable=False, label="0",
            labelOpts=dict(color='w', fill=None, offset=(0, -self._TICK_HT/2 + 1), anchor=(0.5, 0)))
        """ Upward-pointing arrow marks location of current frame in timeline. """
        self._event_markers: List[pg.TargetItem] = list()
        """ Markers for any recorded session events that fall in the displayed time span. """
        self._reach_seg_bars: List[QGraphicsPathItem] = list()
        """ Rectangular bars spanning any defined reach segments that overlap with the displayed time span. """
        self._curr_frame = 0
        """ The current frame number. """
        self._num_frames = 0
        """ The total timeline duration in #frames. """
        self._span = self._DEFAULT_SPAN
        """ Display this many frames on either side of the current frame number. """
        self._show_legend = False
        """ If True, show event symbol/name for each event type in a single line near top of view box. """
        self._legend_items: List[pg.TargetItem] = list()
        """ The target items we use to render a legend near top of view box."""
        self._pan_frame: Optional[int] = None
        """ The current frame number while panning in the "add reach" mode. None otherwise. """
        self._reach_start: int = -1
        """ In "add reach" mode only, the starting frame for a new reach seg under construction. """
        self._reach_max: int = -1
        """ In "add reach" mode only, this is the 'reachMax' frame for a new reach seg under construction. """
        self._reach_rect1 = QGraphicsRectItem(0, self._FIXED_HT_PIX*2, 0, 0)
        """ 
        "Add reach" mode only: While selecting the "reachMax" key frame for a new reach segment under construction, this
        rect spans between the "reachStart" key frame and the current cursor position; once the "reachMax" key frame is 
        selected, the rect is fixed. While repositioning any of the 3 key frames of an **existing** reach segment, this 
        rect spans between the "reachStart" and "reachMax" frames -- one of which could be following the current cursor
        position if it was selected for modification; when modifying the "reachEnd" key frame, the rect is fixed. Placed
        out of bounds when not used.
        """
        self._reach_rect2 = QGraphicsRectItem(0, self._FIXED_HT_PIX*2, 0, 0)
        """ 
        "Add reach" mode only: While selecting the "reachEnd" key frame for a new reach segment under construction, this 
        rect spans between the "reachMax" key frame and the current cursor position; once "reachEnd" is selected, the 
        new reach segment is created. When repositioning any of the 3 key frame of an **existing** reach segment, this 
        rect spans between the "reachMax" and "reachEnd" key frames -- one of which could be following the current 
        cursor position if selected for modification; when modifying the "reachStart" key frame, the rect is fixed. 
        Placed out of bounds when not used.
        """
        self._reach_result_menu = QMenu(self)
        """ Context menu for selecting the result assigned to a just-defined reach segment. """
        self._reach_seg_edited: Optional[RxReachSegment] = None
        """ In "add reach" mode only: The existing reach segment currently being edited; else None. """
        self._reach_key_frame_edited: Optional[str] = None
        """ 
        In "add reach" mode only: Which key frame is being edited (attached to cursor) - "start", "max", or "end".
        None if not in use.
        """

        self._MODE2CURSOR: Dict[_TLMode, Union[QCursor, Qt.CursorShape]] = {
            _TLMode.NORMAL: Qt.CursorShape.CrossCursor,
            _TLMode.ADD_REACH: RxIcons.MARK_CURSOR,
        }
        """ Maps timeline's op mode to the cursor that appears in that mode. """

        pi: pg.PlotItem = self.getPlotItem()
        pi.hideButtons()
        pi.hideAxis('left')
        pi.hideAxis('bottom')

        pi.addItem(self._curr_frame_marker, ignoreBounds=True)
        pi.addItem(self._vline, ignoreBounds=True)
        self._vline.setZValue(2)  # so cursor line is drawn over everything else

        self._reach_rect1.setBrush(pg.mkBrush('gold'))
        self._reach_rect1.setPen(pg.mkPen(None))
        self._reach_rect2.setBrush(pg.mkBrush('cornflowerblue'))
        self._reach_rect2.setPen(pg.mkPen(None))
        pi.addItem(self._reach_rect1, ignoreBounds=True)
        pi.addItem(self._reach_rect2, ignoreBounds=True)
        self._reach_rect1.setZValue(1)  # so these are drawn over existing reach seg bars when editing a key frame
        self._reach_rect2.setZValue(2)

        self._update_axis_polyline()
        self._update_legend()

        self._message_label.setParentItem(pi)
        self._message_label.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))

        self._mode_label.setParentItem(pi)
        self._mode_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-3, 3))

        self._reach_result_menu.addAction("Discard")
        self._reach_result_menu.addSeparator()
        for res in RxReach:
            if res.is_result_code:
                self._reach_result_menu.addAction(str(res))

        vb: pg.ViewBox = pi.getViewBox()
        vb.setMenuEnabled(False)
        vb.setMouseEnabled(x=False, y=False)
        vb.setBorder(self._BORDER_PEN_NO_FOCUS)
        vb.setRange(xRange=(-self._span, self._span), yRange=(0, self._FIXED_HT_PIX), padding=0)

        self.installEventFilter(self)   # to catch when cursor leaves/enters widget
        self.sceneObj.sigMouseClicked.connect(self._mouse_clicked)
        self.setMinimumSize(QSize(self._MIN_WIDTH_PIX, self._FIXED_HT_PIX))
        self.setMaximumHeight(self._FIXED_HT_PIX)

        self.setCursor(self._MODE2CURSOR[self._mode])

        # a session could already be loaded before UI is up, so update timeline accordingly. Then connect to the
        # signals we care about from the session and data managers.
        self._refresh()
        self._session.frame_ready.connect(self._refresh)
        self._session.reaches_changed.connect(self._update_displayed_reach_segments)
        self._session.scorer_changed.connect(self._update_displayed_reach_segments)
        self._session.segmentation_result_changed.connect(self._update_displayed_reach_segments)
        data_mgr.session_loaded.connect(self._refresh)
        data_mgr.active_workspace_switched.connect(self._refresh)
        data_mgr.active_workspace_path_changed.connect(self._refresh)

    def _update_axis_polyline(self) -> None:
        """
        Helper method creates or updates the initial polyline that renders the timeline's axis. Be sure to call
        this whenever the time span changes.

        If T is the timespan, then the polyline consists of a horizontal line at Y=Y0 spanning [T, -T], plus a
        series of 50 vertical tick marks at [-T, -49*T/50,   0, ... 49*T/50, T]
        """
        col1 = np.arange(-self._span, self._span + 1, int(self._span / 50))
        col2 = np.tile(np.array([self._TICK_MIN_Y, self._TICK_MIN_Y + self._TICK_HT]), len(col1))
        col1 = np.repeat(col1, 2)
        time_line_pts = np.vstack((np.column_stack((col1, col2)),
                                   np.array([[-self._span, self._TICK_MIN_Y], [self._span, self._TICK_MIN_Y]])))
        if self._axis_polyline is None:
            self._axis_polyline = self.getPlotItem().plot(time_line_pts, connect='pairs', pen=pg.mkPen('w', width=2))
        else:
            self._axis_polyline.setData(time_line_pts)

    def _update_legend(self) -> None:
        """
        Helper method creates the target items that render the event symbol legend near the top of timeline view, or
        updates their positions. Be sure to call whenever the time span changes. (This is necessary because the event
        symbol legend is rendered as a set of `TargetItems` positioned horizontally IAW the current time span.)
        """
        if len(self._legend_items) == 0:
            t = -self._span + self._span/50
            for evt, color in _EVENT2MARKER.items():
                ti = pg.TargetItem(pos=(t, self._FIXED_HT_PIX + self._TICK_HT), size=self._TICK_HT,
                                   symbol='t', brush=color, pen='w', movable=False, label=str(evt),
                                   labelOpts=dict(color='w', fill=None, offset=(5, 0), anchor=(0, 0.5)))
                self.getPlotItem().addItem(ti)
                self._legend_items.append(ti)
                t += (self._span*2) / len(_EVENT2MARKER)
        else:
            t = -self._span + self._span/50
            for i in range(len(self._legend_items)):
                ti = self._legend_items[i]
                ti.setPos((t, self._FIXED_HT_PIX + self._TICK_HT))
                t += (self._span*2) / len(_EVENT2MARKER)

    def _update_displayed_events(self) -> None:
        """
        Helper method updates the displayed event markers. Be sure to call this method whenever the time span or the
        current frame number changes!

        DEVNOTE: The method reuses rendered marker items as needed rather than always clearing and recreating them
        (which would reduce performance during timed playback). Since the current frame is always at T=0 in the
        displayed time span, we subtract the current frame number from the event times to position markers correctly.
        """
        pi: pg.PlotItem = self.getPlotItem()
        events = self._session.events
        visible_events = events.get_all_events_between(self._curr_frame - self._span, self._curr_frame + self._span) \
            if events is not None else []
        evt: Tuple[RxEvent, int]
        for i, evt in enumerate(visible_events):
            color = _EVENT2MARKER[evt[0]]
            t_relative = evt[1] - self._curr_frame
            if i >= len(self._event_markers):
                ti = pg.TargetItem(pos=(t_relative, self._EVENT_MARKER_Y), size=self._EVENT_MARKER_SZ,
                                   symbol='t', brush=color, pen='w', movable=False)
                self._event_markers.append(ti)
                pi.addItem(ti)
            else:
                self._event_markers[i].setPos((t_relative, self._EVENT_MARKER_Y))
                self._event_markers[i].setBrush(color)

        #  Marker items that are no longer in use are simply positioned out of view.
        i = len(visible_events)
        while i < len(self._event_markers):
            self._event_markers[i].setPos((self._span*10, self._EVENT_MARKER_Y))
            i += 1

    def _update_displayed_reach_segments(self) -> None:
        """
        Helper method updates the rectangular bars spanning any defined reach segments that overlap with the current
        displayed time span. Be sure to call this method whenever the time span or the current frame number changes!

        Two categories of reach segments are displayed, user-curated reaches (editable by the user) and auto-generaated
        reaches. The latter are the result of reach segmentation on the hand/pellet trajectories generated by analysis
        with a body part detection ML model, aka, a "scorer". A given session may be analyzed by multiple scorers, but
        the auto-generated reaches are only shown for the currently selected scorer.

        The two categories of reaches are distinugished by their vertical placement and pen color: The auto-generated
        reaches appear above the curated ones, and the rectangular bars are outlined in yellow rather than white.

        DEVNOTE: The method reuses 'QGraphicsPathItems' rendering the reach segments as needed rather than always
        clearing and recreating them. Since the current frame is always at T=0 in the displayed time span, we subtract
        the current frame number from the reach segment start times to position segments correctly.
        """
        pi: pg.PlotItem = self.getPlotItem()
        t0, t1 = self._curr_frame - self._span, self._curr_frame + self._span
        visible_reaches = self._session.get_all_curated_reaches_in(t0, t1)

        reach: RxReachSegment
        for i, reach in enumerate(visible_reaches):
            color = reach.result.ui_color
            t = reach.frame - self._curr_frame
            path = QPainterPath()
            path.addRect(t, self._CURATED_REACH_BASE, reach.dur, self._REACH_SEG_H)
            path.moveTo(t + reach.max_delta, self._CURATED_REACH_BASE - self._REACH_SEG_EXTRA)
            path.lineTo(t + reach.max_delta, self._CURATED_REACH_BASE + self._REACH_SEG_H + self._REACH_SEG_EXTRA)

            if i >= len(self._reach_seg_bars):
                bar = QGraphicsPathItem(path)
                bar.setPen(self._CURATED_BAR_PEN)
                bar.setBrush(pg.mkBrush(color))
                self._reach_seg_bars.append(bar)
                pi.addItem(bar)
            else:
                self._reach_seg_bars[i].setPath(path)
                self._reach_seg_bars[i].setPen(self._CURATED_BAR_PEN)
                self._reach_seg_bars[i].setBrush(pg.mkBrush(color))

        # repeat for scored reaches
        n_used = len(visible_reaches)   # so we don't stomp on what we just prepared above!
        scored_reaches = self._session.get_all_scored_reaches_in(t0, t1)
        for i, reach in enumerate(scored_reaches):
            color = reach.result.ui_color
            t = reach.frame - self._curr_frame
            path = QPainterPath()
            path.addRect(t, self._AUTOGEN_REACH_BASE, reach.dur, self._REACH_SEG_H)
            path.moveTo(t + reach.max_delta, self._AUTOGEN_REACH_BASE - self._REACH_SEG_EXTRA)
            path.lineTo(t + reach.max_delta, self._AUTOGEN_REACH_BASE + self._REACH_SEG_H + self._REACH_SEG_EXTRA)

            if (i + n_used) >= len(self._reach_seg_bars):
                bar = QGraphicsPathItem(path)
                bar.setPen(self._AUTOGEN_BAR_PEN)
                bar.setBrush(pg.mkBrush(color))
                self._reach_seg_bars.append(bar)
                pi.addItem(bar)
            else:
                self._reach_seg_bars[i + n_used].setPath(path)
                self._reach_seg_bars[i + n_used].setPen(self._AUTOGEN_BAR_PEN)
                self._reach_seg_bars[i + n_used].setBrush(pg.mkBrush(color))

        #  reach segment bars that are no longer in use are simply reinitialized with an empty painter path
        i = len(visible_reaches) + len(scored_reaches)
        while i < len(self._reach_seg_bars):
            self._reach_seg_bars[i].setPath(QPainterPath())
            i += 1

    def _update_time_cursor(self, pos: Optional[QPointF] = None) -> None:
        """
        Show, hide, or update the time label for the vertical line that follows the mouse when it is inside this
        TimeLineWidget.
        :param pos: The current cursor position in view coordinates. If valid, update location of the vertical line and
            set its label to reflect the frame number at that location. If None, hide the line.
        """
        if pos is None:
            if self._vline_shown:
                self._vline.setPos(self.TIME_SPANS[-1]*10)   # positioned well outside the view
                self._vline_shown = False
        else:
            self._vline_shown = True
            self._vline.setPos(pos.x())
            # label not shown while panning; instead labeled arrow follows the vertical line
            if not self._panning:
                self._vline.label.setFormat(f"{int(self._curr_frame + pos.x() + 0.5)}")

    @property
    def _panning(self) -> bool:
        """ Is the timeline widget panning in the "add/edit reach" operational mode? """
        return self._mode == _TLMode.ADD_REACH

    def _update_pan(self, pos: QPointF) -> None:
        """
        Update panning state while in the "add/edit reach" mode.

        In this special op mode, the frame interval [N..M] spanned by the timeline is fixed, and the current frame
        follows the mouse location -- allowing the user to pan through the video and "visually select" the key frames
        for a new reach segment under construction, or to reposition a key frame of an existing segment. No action taken
        if panning is off.

        :param pos: Current cursor position in view coordinates. We only care about X-coordinate -- which is a +/-
            offset, as a fractional frame count, from T=0 at the horizontal center of the widget. If the frame number
            under the cursor does not match the current frame number, the current frame is updated.
        """
        if self._panning:
            self._pan_frame = int(self._curr_frame + pos.x() + 0.5)
            self._curr_frame_marker.setPos((pos.x(), self._TICK_MIN_Y - self._TICK_HT / 2))
            self._curr_frame_marker.label().setText(f"{self._pan_frame}")
            if self._pan_frame != self._session.current_frame_num:
                self._session.go_to_frame(self._pan_frame)

    def _refresh(self) -> None:
        """
        Update the timeline widget to reflect the current frame number and any session events and reach segments that
        occur within the time span displayed by the widget.

        NOTE: While panning in the "add/edit reach" mode, the timeline widget itself requests frame changes and updates
        its appearance prior to each change. If a frame change happens external to this widget, the panning state is
        adjusted accordingly.
        """
        # if panning and the frame change request came from here, do nothing. If panning and the frame change doesn't
        # match the current pan frame, we have to adjust the pan frame accordingly.
        if self._panning and (self._pan_frame == self._session.current_frame_num):
            return
        else:
            self._num_frames = self._session.total_frames
            old_curr = self._curr_frame
            self._curr_frame = max(0, self._session.current_frame_num)

            self._message_label.setText("" if self._num_frames > 0 else self._NO_VIDEO_MSG)
            if not self._panning:
                self._curr_frame_marker.label().setText(f"{self._curr_frame}")
                # if vertical cursor is currently shown, need to update its label
                if self._vline_shown:
                    self._vline.label.setFormat(f"{int(self._curr_frame + self._vline.value() + 0.5)}")
            else:
                delta = self._pan_frame - old_curr
                self._pan_frame = self._curr_frame + delta
                self._curr_frame_marker.label().setText(f"{self._pan_frame}")

            self._update_displayed_events()
            self._update_displayed_reach_segments()

    def _switch_mode(self, m: _TLMode) -> None:
        """
        Switch to the specified operational mode.
        :param m: The new op mode. No action taken if this matches the current mode.
        """
        if m != self._mode:
            is_panning = self._panning
            will_pan = (m == _TLMode.ADD_REACH)

            # if leaving "add/edit reach", reset animation
            if self._mode == _TLMode.ADD_REACH:
                self._update_reach_segment_under_construction(None)

            # stop panning if necessary
            if is_panning and not will_pan:
                self._curr_frame_marker.setPos((0, self._TICK_MIN_Y - self._TICK_HT / 2))
                self._curr_frame_marker.label().setText(f"{self._curr_frame}")
                self._vline.setPen('w')
                self._pan_frame = None

            self._mode = m
            self.setCursor(self._MODE2CURSOR[m])
            self._mode_label.setText(str(self._mode))

            # start panning if necessary
            if will_pan and not is_panning:
                self._vline.label.setFormat(" ")
                pi: pg.PlotItem = self.getPlotItem()
                p = pi.getViewBox().mapDeviceToView(self.mapFromGlobal(QCursor.pos()))
                self._update_pan(p)

            # reset upon entering "add/edit reach" mode.
            if self._mode == _TLMode.ADD_REACH:
                self._update_reach_segment_under_construction(None)

    def _update_reach_segment_under_construction(self, pos: Optional[QPointF] = None, clicked: bool = False) -> None:
        """
        In the "add/edit reach" op mode only, this method updates the widget state during the user-interactive
        construction of a new reach segment or repositioning of a key frame (start, max, or end) of an existing segment.

        **Usage**:
          - Call with `pos==None` to reset. The color of the vertical time cursor is set to green, and the two
            rectangles that represent the two parts (start to max, max to end) of a reach segment under construction or
            being modified are hidden.
          - After reset, call with clicked==True to initiate an add or edit operation. If the clicked frame is close
            enough to a key frame of an existing reach segment, that key frame becomes "attached" to the cursor and
            repositioning begins. Otherwise, construction of a new reach segment begins and the clicked frame is the
            starting frame for the new segment.
          - When defining a new reach segment, the second click marks the "max" key frame, and the third click marks
            the "end" key frame. At this point, the new reach is defined, and the two rectangles that animate the
            process are hidden.
          - When repositioning a key frame for an existing segment, the next click sets the new position for the key
            frame; if valid, the segment is modified accordingly and the two rectangles that animate the process are
            once again hidden.
          - During the user-interactive gesture, call with clicked==False to update the two rectangles that animate
            the processs whenever the cursor moves.
        By design, the "reachMax" key frame must be strictly greater than the start frame, and the end frame must be
        greater than the "reachMax" frame.

        When the end frame is selected for the reach segment under construction, a popup modal dialog is raised so the
        user can select the "result" and relative hand position (at "reachMax" keyframe) for the reach. A request to add
        the reach segment is passed to the SessionManager (which could reject it). Then, if the mouse is still inside
        the, it resumes "add/edit reach" mode (or it's configured to do so the next time the mouse reenters).

        DEVNOTE: We use the "fractional frame number" corresponding to the current cursor position when positioning and
        sizing the two rectangles that animate the add/edit operation. Prior to this change, when the timespan was
        small, using an integer frame number made the animation jumpy, and made it difficult to create a new reach that
        began one frame after a previous reach ended.

        :param pos: The current cursor position **in view coordinates**, or None. The X-coordinate is the offset in
            fractional frames from the horizontal center of the widget.
        :param clicked: True if called in response to a mouse click. Ignored if `pos==None`.
        """
        # need floating-point version of frame # for positioning/sizing animation rects
        frame_fp: float = None if pos is None else (self._curr_frame + pos.x())
        frame = None if pos is None else int(round(frame_fp))
        if frame is None:
            self._vline.setPen('g', width=3)
            self._reach_start, self._reach_max = -1, -1
            self._reach_seg_edited, self._reach_key_frame_edited = None, None
            self._reach_rect1.setRect(0, self._FIXED_HT_PIX*2, 0, 0)
            self._reach_rect2.setRect(0, self._FIXED_HT_PIX*2, 0, 0)
        elif clicked and (self._reach_start < 0) and (self._reach_seg_edited is None):
            # initiate add or edit operation: determine if user has clicked on a key frame of a visible reach segment
            visible_reaches = self._session.get_all_curated_reaches_in(self._curr_frame - self._span,
                                                                       self._curr_frame + self._span)
            for i, reach in enumerate(visible_reaches):
                if abs(reach.frame - frame_fp) <= 0.5:
                    self._reach_key_frame_edited = "start"
                elif abs(reach.max_frame - frame_fp) <= 2:
                    self._reach_key_frame_edited = "max"
                elif abs(reach.end_frame - frame_fp) <= 0.5:
                    self._reach_key_frame_edited = "end"
                if self._reach_key_frame_edited is not None:
                    self._reach_seg_edited = reach
                    break

            # if user has selected a reach segment to edit, begin an edit op... else an add op.
            if self._reach_seg_edited is not None:
                if self._reach_key_frame_edited == "start":
                    x1, w1 = frame_fp - self._curr_frame, self._reach_seg_edited.max_frame - frame_fp
                    x2, w2 = (self._reach_seg_edited.max_frame - self._curr_frame,
                              self._reach_seg_edited.end_frame - self._reach_seg_edited.max_frame)
                elif self._reach_key_frame_edited == "max":
                    self._vline.setPen('y', width=3)
                    x1, w1 = (self._reach_seg_edited.frame - self._curr_frame,
                              frame_fp - self._reach_seg_edited.frame)
                    x2, w2 = frame_fp - self._curr_frame, self._reach_seg_edited.end_frame - frame_fp
                else:
                    self._vline.setPen('b', width=3)
                    x1, w1 = self._reach_seg_edited.frame - self._curr_frame, self._reach_seg_edited.max_delta
                    x2, w2 = (self._reach_seg_edited.max_frame - self._curr_frame,
                              frame_fp - self._reach_seg_edited.max_frame)

                self._reach_rect1.setRect(x1, self._CURATED_REACH_BASE, w1, self._REACH_SEG_H)
                self._reach_rect2.setRect(x2, self._CURATED_REACH_BASE, w2, self._REACH_SEG_H)
            else:
                self._reach_start = frame
                self._vline.setPen('y', width=3)
        elif self._reach_start >= 0:
            # update state of a new reach segment under construction
            if self._reach_max < 0:
                if frame_fp <= self._reach_start:  # not valid: reachMax must be > reachStart!
                    return
                self._reach_rect1.setRect(self._reach_start - self._curr_frame, self._CURATED_REACH_BASE,
                                          frame_fp - self._reach_start, self._REACH_SEG_H)
                if clicked:
                    self._reach_max = frame
                    self._reach_rect1.setRect(self._reach_start - self._curr_frame, self._CURATED_REACH_BASE,
                                              self._reach_max - self._reach_start, self._REACH_SEG_H)
                    if frame_fp > self._reach_max:
                        self._reach_rect2.setRect(self._reach_max - self._curr_frame, self._CURATED_REACH_BASE,
                                                  frame_fp - self._reach_max, self._REACH_SEG_H)
                    self._vline.setPen('b', width=3)
            elif frame_fp > self._reach_max:   # reachEnd must always be > reachMax!
                if not clicked:
                    self._reach_rect2.setRect(self._reach_max - self._curr_frame, self._CURATED_REACH_BASE,
                                              frame_fp - self._reach_max, self._REACH_SEG_H)
                elif frame > self._reach_max:
                    # define the segment, assuming a successful result initially
                    # switch to normal mode before raising context menu, because user interaction with the menu will
                    # generate a mouse-leave event anyway. This will reset _reach_start and _reach_max, which is why we
                    # save the segment definition first!
                    seg = RxReachSegment(self._reach_start, self._reach_max - self._reach_start,
                                         frame - self._reach_start,
                                         RxReach.GRABBED, RxHandPos.UNSPECIFIED)
                    self._switch_mode(_TLMode.NORMAL)

                    # raise modal dialog near mouse click location to set reach result and relative hand position
                    pos.setY(self._FIXED_HT_PIX - 5)
                    scene_loc: QPointF = self.getPlotItem().getViewBox().mapViewToScene(pos)
                    updated_seg = self._add_reach_dlg.update_reach_segment(seg, self.mapToGlobal(scene_loc.toPoint()))
                    if updated_seg is not None:
                        self._session.add_curated_reach(updated_seg)

                    # at this point, if mouse is still inside, resume add-reach mode. If mouse outside, ensure we
                    # return to add-reach if mouse moves back inside
                    if self._mouse_inside:
                        self._switch_mode(_TLMode.ADD_REACH)
                    else:
                        self._last_mode = _TLMode.ADD_REACH
        elif self._reach_seg_edited is not None:
            # update state while repositioning a key frame of an existing segment
            # enforce requirements on edited key frame: 0 <= start < max < end < #totalFrames. If invalid, do nothing.
            if self._reach_key_frame_edited == "start":
                valid = (0 <= frame_fp < self._reach_seg_edited.max_frame)
            elif self._reach_key_frame_edited == "max":
                valid = self._reach_seg_edited.frame < frame_fp < self._reach_seg_edited.end_frame
            else:
                valid = self._reach_seg_edited.max_frame < frame_fp < self._session.total_frames
            if not valid:
                return

            if not clicked:
                # update the two rectangles that animate the reach segment being edited.
                if self._reach_key_frame_edited == "start":
                    x1, w1 = frame_fp - self._curr_frame, self._reach_seg_edited.max_frame - frame_fp
                    x2, w2 = (self._reach_seg_edited.max_frame - self._curr_frame,
                              self._reach_seg_edited.end_frame - self._reach_seg_edited.max_frame)
                elif self._reach_key_frame_edited == "max":
                    x1, w1 = self._reach_seg_edited.frame - self._curr_frame, frame_fp - self._reach_seg_edited.frame
                    x2, w2 = frame_fp - self._curr_frame, self._reach_seg_edited.end_frame - frame_fp
                else:
                    x1, w1 = self._reach_seg_edited.frame - self._curr_frame, self._reach_seg_edited.max_delta
                    x2, w2 = (self._reach_seg_edited.max_frame - self._curr_frame,
                              frame_fp - self._reach_seg_edited.max_frame)

                self._reach_rect1.setRect(x1, self._CURATED_REACH_BASE, w1, self._REACH_SEG_H)
                self._reach_rect2.setRect(x2, self._CURATED_REACH_BASE, w2, self._REACH_SEG_H)
            else:
                # attempt to replace edited reach with the reach resulting from change in chosen key frame. Here
                # we must use the integer frame number closest to the floating-point version
                kf = self._reach_key_frame_edited
                seg = self._reach_seg_edited
                if kf == "start":
                    new_reach = RxReachSegment(frame, seg.max_frame - frame, seg.end_frame - frame, seg.result,
                                               seg.hand_pos)
                elif kf == "max":
                    new_reach = RxReachSegment(seg.frame, frame - seg.frame, seg.dur, seg.result, seg.hand_pos)
                else:  # kf == "end"
                    new_reach = RxReachSegment(seg.frame, seg.max_delta, frame - seg.frame, seg.result, seg.hand_pos)
                self._session.replace_curated_reach(self._reach_seg_edited, new_reach)

                # regardless of success, reset the animation, ready for next add/edit op
                self._reach_seg_edited = None
                self._reach_key_frame_edited = None
                self._reach_rect1.setRect(0, self._FIXED_HT_PIX * 2, 0, 0)
                self._reach_rect2.setRect(0, self._FIXED_HT_PIX * 2, 0, 0)
                self._vline.setPen('g', width=3)

    @property
    def show_legend(self) -> bool:
        return self._show_legend

    @show_legend.setter
    def show_legend(self, show: bool) -> None:
        """
        Show or hide the timeline legend: event symbol and name for each type of session event.
        :param show: True to show legend, False to hide it.
        """
        if show == self._show_legend:
            return
        self._show_legend = show
        h = self._FIXED_HT_PIX + (2*self._TICK_HT if show else 0)
        self.getPlotItem().getViewBox().setRange(xRange=(-self._span, self._span), yRange=(0, h), padding=0)

    @property
    def timespan(self) -> int:
        """ The number of frames displayed either side of the current frame number, which is centered in timeline. """
        return self._span

    def change_timespan(self, span: int) -> None:
        """
        Change the current time span displayed in this timeline widget.
        :param span: The new span. Must be one of TimeLineWidget.TIME_SPANS, or no action is taken.
        """
        if (span != self._span) and (span in self.TIME_SPANS):
            self._span = span
            self._update_axis_polyline()
            self._update_legend()
            self._update_displayed_events()
            self._update_displayed_reach_segments()
            h = self._FIXED_HT_PIX + (2 * self._TICK_HT if self._show_legend else 0)
            self.getPlotItem().getViewBox().setRange(xRange=(-self._span, self._span), yRange=(0, h), padding=0)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Respond to mouse enter/leave events, and to key presses while the mouse is inside the widget:
         - Enter: Grab keyboard focus so we get key press events. Restore previous op mode if it was not normal mode.
         - Leave: Release keyboard focus, hide the vertical timeline cursor, and return to normal operation mode. If
           the timeline was panning in "add reach" mode, restore the current frame to what it was prior to starting
           panning.
         - KeyPress (only when mouse is inside widget):
            - The 'A' key toggles from "normal" to "add reach" mode and vice versa.
            - The 'C' key: If the vertical line cursor intersects a scored reach segment, pressing 'C' will add that
              reach to the current session's "curated" set.
            - The 'X', 'Delete' or 'Backspace' key: If the vertical line cursor intersects a curated reach segment,
              pressing any one of these keys will remove it from the current session's "curated" set.
            - The 'E' key: If the vertical line cursor intersects a curated reach segment, pressing 'E' will raise a
              modal dialog so you can change the result and/or hand position relative to the pellet for that reach.
            - Note that the 'C', 'X', 'Delete', 'Backspace' and 'E' key presses are active in both op modes, but are
              ignored in "add/edit reach" mode **IF** the user is in the process of defining a reach segment manually.
        """
        if watched == self:
            if event.type() == QEvent.Type.Enter:
                self._mouse_inside = True
                self.setFocus(Qt.FocusReason.MouseFocusReason)
                self.getPlotItem().getViewBox().setBorder(self._BORDER_PEN_FOCUS)
                if self._last_mode != _TLMode.NORMAL:
                    self._switch_mode(self._last_mode)
            elif event.type() == QEvent.Type.Leave:
                self._mouse_inside = False
                self.clearFocus()
                self.getPlotItem().getViewBox().setBorder(self._BORDER_PEN_NO_FOCUS)
                self._update_time_cursor(None)

                # remember op mode upon leaving so we can restore it when cursor reenters
                self._last_mode = self._mode

                # if we were panning, go back to what was the current frame before we started panning
                if self._panning:
                    self._session.go_to_frame(self._curr_frame)
                self._switch_mode(_TLMode.NORMAL)
            elif event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent) and self._mouse_inside:
                key = QKeyEvent(event).key()
                if key == Qt.Key.Key_A:
                    next_mode = _TLMode.ADD_REACH if self._mode == _TLMode.NORMAL else _TLMode.NORMAL
                    # if we were panning, go back to what was the current frame before we started panning
                    if self._panning:
                        self._session.go_to_frame(self._curr_frame)
                    self._switch_mode(next_mode)
                else:
                    ok = ((self._mode == _TLMode.NORMAL) or (self._reach_start < 0)) and self._vline_shown
                    if ok:
                        frame = int(self._curr_frame + self._vline.value() + 0.5)
                        if key == Qt.Key.Key_C:
                            self._session.add_scored_reach_containing_frame_to_curated_set(frame)
                        elif key in [Qt.Key.Key_X, Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:
                            self._session.delete_curated_reach_containing_frame(frame)
                        elif key == Qt.Key.Key_E:
                            seg = self._session.get_curated_reach_containing_frame(frame)
                            if isinstance(seg, RxReachSegment):
                                was_add_reach = (self._mode == _TLMode.ADD_REACH)
                                self._switch_mode(_TLMode.NORMAL)
                                pos = QPointF(self._vline.value(), self._FIXED_HT_PIX - 5)
                                scene_loc: QPointF = self.getPlotItem().getViewBox().mapViewToScene(pos)
                                updated_seg = (
                                    self._add_reach_dlg.update_reach_segment(seg,
                                                                             self.mapToGlobal(scene_loc.toPoint())))
                                if isinstance(updated_seg, RxReachSegment) and (updated_seg != seg):
                                    self._session.replace_curated_reach(seg, updated_seg)
                                if was_add_reach:
                                    if self._mouse_inside:
                                        self._switch_mode(_TLMode.ADD_REACH)
                                    else:
                                        self._last_mode = _TLMode.ADD_REACH

        return super().eventFilter(watched, event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """
        Whenever the mouse moves over the plot window, update the vertical "time cursor". Also, if panning is enabled,
        update the current frame to match the frame under the cursor. If currently adding a new reach segment or
        adjusting the key frame of an existing segment in the "add/edit reach" mode, update the two-color rectangle that
        animates the process.

        :param event: The mouse movement event.
        """
        pi: pg.PlotItem = self.getPlotItem()
        pos: QPointF = event.pos()
        if pi.sceneBoundingRect().contains(pos):
            loc: QPointF = pi.getViewBox().mapSceneToView(pos)
            self._update_time_cursor(loc)
            if self._panning:
                self._update_pan(loc)
                if self._mode == _TLMode.ADD_REACH:
                    self._update_reach_segment_under_construction(loc, False)
        super().mouseMoveEvent(event)

    @Slot(MouseClickEvent)
    def _mouse_clicked(self, evt: MouseClickEvent) -> None:
        """
        The effect of a left-click on the timeline depends on the current operational mode:
          - In normal mode: Make the frame at the cursor location the current frame.
          - In add-edit-reach mode: When adding a new segment, the first click marks the "reachStart" key frame, the
            next marks "reachMax", and the third click marks "reachEnd", completing the definition of the new segment.
            When repositioning a key frame for an existing segment, the first click selects the key frame to edit, and
            the next click sets its new location.
        """
        if self._num_frames > 0:
            # ignore any click outside scene rect!
            pos = evt.scenePos()
            if self.sceneBoundingRect().contains(pos):
                loc: QPointF = self.getPlotItem().getViewBox().mapSceneToView(pos)
                frame = int(self._curr_frame + loc.x() + 0.5)
                if self._mode == _TLMode.NORMAL:
                    self._session.go_to_frame(frame)
                elif self._mode == _TLMode.ADD_REACH:
                    self._update_reach_segment_under_construction(loc, True)


class _AddReachDlg(QDialog):
    """
    A modal dialog raised to let the user specify the result code and hand position relative to pellet at the
    'reachMax' keyframe for a newly defined or already existing curated reach segment.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set reach result")
        self.setModal(True)

        # group box of mutually exclusive options for reach segment result
        self._grabbed_rb = QRadioButton("grabbed")
        self._missed_rb = QRadioButton("missed")
        self._dropped_rb = QRadioButton("dropped")
        self._stalled_rb = QRadioButton("stalled")
        self._none_rb = QRadioButton("none")
        self._grabbed_rb.setChecked(True)

        result_gb = QGroupBox("Result")
        layout = QVBoxLayout()
        layout.addWidget(self._grabbed_rb)
        layout.addWidget(self._missed_rb)
        layout.addWidget(self._dropped_rb)
        layout.addWidget(self._stalled_rb)
        layout.addWidget(self._none_rb)
        result_gb.setLayout(layout)

        # group box of mutually exclusive options for hand position at "reach max"
        self._left_rb = QRadioButton("left of pellet")
        self._right_rb = QRadioButton("right of pellet")
        self._above_rb = QRadioButton("above pellet")
        self._below_rb = QRadioButton("below pellet")
        self._unspecified_rb = QRadioButton("unspecified")
        self._unspecified_rb.setChecked(True)

        hand_pos_gb = QGroupBox("Hand Position")
        layout = QVBoxLayout()
        layout.addWidget(self._left_rb)
        layout.addWidget(self._right_rb)
        layout.addWidget(self._above_rb)
        layout.addWidget(self._below_rb)
        layout.addWidget(self._unspecified_rb)
        hand_pos_gb.setLayout(layout)

        # Ok/Cancel dialog button box
        button_box = QDialogButtonBox()
        button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout = QGridLayout()
        main_layout.addWidget(result_gb, 0, 0)
        main_layout.addWidget(hand_pos_gb, 0, 1)
        main_layout.addWidget(button_box, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(main_layout)

    def update_reach_segment(self, seg: RxReachSegment, pos: QPoint) -> Optional[RxReachSegment]:
        """
        Raise this modal dialog to let the user set the result of a reach segment and the hand position relative to
        the pellet at the segment's "reach_max" keyframe.

        :param seg: The reach segment to update.
        :param pos: Desired location of top-left corner of dialog, in global/screen coordinates (pixels).
        :return: If None, user canceled out of dialog. Otherwise, the updated reach segment -- only the result and
            relative hand position parameters may be changed.
        """
        if seg.result == RxReach.GRABBED:
            self._grabbed_rb.setChecked(True)
        elif seg.result == RxReach.MISSED:
            self._missed_rb.setChecked(True)
        elif seg.result == RxReach.DROPPED:
            self._dropped_rb.setChecked(True)
        elif seg.result == RxReach.NONE:
            self._none_rb.setChecked(True)
        else:
            self._stalled_rb.setChecked(True)

        if seg.hand_pos == RxHandPos.LEFT:
            self._left_rb.setChecked(True)
        elif seg.hand_pos == RxHandPos.RIGHT:
            self._right_rb.setChecked(True)
        elif seg.hand_pos == RxHandPos.ABOVE:
            self._above_rb.setChecked(True)
        elif seg.hand_pos == RxHandPos.BELOW:
            self._below_rb.setChecked(True)
        else:
            self._unspecified_rb.setChecked(True)

        # move dialog TL corner to specified screen coordinates
        self.move(pos)

        # raise dialog and return reach segment with result and hand position modified IAW user choices
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        else:
            result: RxReach = RxReach.GRABBED
            if self._missed_rb.isChecked():
                result = RxReach.MISSED
            elif self._dropped_rb.isChecked():
                result = RxReach.DROPPED
            elif self._stalled_rb.isChecked():
                result = RxReach.STALLED
            elif self._none_rb.isChecked():
                result = RxReach.NONE
            hand_pos: RxHandPos = RxHandPos.UNSPECIFIED
            if self._left_rb.isChecked():
                hand_pos = RxHandPos.LEFT
            elif self._right_rb.isChecked():
                hand_pos = RxHandPos.RIGHT
            elif self._above_rb.isChecked():
                hand_pos = RxHandPos.ABOVE
            elif self._below_rb.isChecked():
                hand_pos = RxHandPos.BELOW
            return RxReachSegment(seg.frame, seg.max_delta, seg.dur, result, hand_pos)
