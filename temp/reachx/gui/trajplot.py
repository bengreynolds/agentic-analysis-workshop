from typing import Optional, List, Dict

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QPen, QColor, Qt

from reachx.common import RxReachSegment


_ENDPOINT_SIZE: int = 12
""" Size of symbols that may mark the start and end of a reach segment trajectory trace, in pixels. """
_FOCUS_LW: int = 10
""" Line width applied to the hand/pellet trajectories corresponding to the current focus segment. """
_FOCUS_COLOR: QColor = QColor('mediumpurple')
""" Color applied to the hand/pellet trajectories corresponding to the current focus segment. """


class _ReachSegTraj:
    def __init__(self, seg: RxReachSegment, hand: np.ndarray, pellet: np.ndarray) -> None:
        """

        :param seg: The reach segment represented.
        :param hand: The trajectory of the hand over the entire session. This must be a ``np.array((N, 4),
            dtype=np.float32)``, where N = the session duration in frames and the columns hold the X, Y, Z, and speed
            of the animal's hand.
        :param pellet: The trajectory of the pellet over the entire session. Same structure as ``hand``.
        """
        self._seg = seg
        self._sesh_len = hand.shape[0]
        """ Duration of recorded session in #frames (in case reach segment hits end of session). """
        self._hand: Optional[np.ndarray] = None
        """ Excerpted trajectory of hand during the reach segment. """
        self._pellet: Optional[np.ndarray] = None
        """ Excerpted trajectory of pellet during the reach segment. """
        self._pdi_hand: Optional[pg.PlotDataItem] = None
        """ The rendered hand trajectory during the reach segment. """
        self._pdi_hand_endpts: Optional[pg.PlotDataItem] = None
        """ Renders symbols marking the start and end of hand trajectory. """
        self._pdi_pellet: Optional[pg.PlotDataItem] = None
        """ The rendered pellet trajectory during the reach segment. """
        self._pdi_pellet_endpts: Optional[pg.PlotDataItem] = None
        """ Renders symbols marking the start and end of pellet trajectory. """

        self._build(hand, pellet)

    def _build(self, hand: np.ndarray, pellet: np.ndarray) -> None:
        """
        Helper method saves slices of the specified hand and pellet trajectories during the reach segment, then
        prepares a renderable trace --``PlotDataItem`` -- for both the hand trajectory and the pellet trajectory. All
        trace lines are initially white 2px solid lines; the hand trace is configured to be shown, but the pellet trace
        is hidden. Endpoint symbols at start and end of each trajectory are NOT shown.

        :param hand: Hand trajectory -- ``(N,4) np.float32`` Numpy array -- over entire session of N frames.
        :param pellet: Pellet trajectory, similar to ``hand``.
        """
        start = max(0, self._seg.frame)
        end = min(self._sesh_len - 1, self._seg.end_frame)
        self._hand = hand[start:end + 1, :]
        self._pellet = pellet[start:end + 1, :]

        pen = pg.mkPen('w', width=2)
        y = self._hand[:, 1]
        z = self._hand[:, 2]

        symbols: List[str] = ['o', 's']
        symbol_sizes: List[int] = [0, 0]
        symbol_color = pg.mkColor('w')

        self._pdi_hand = pg.PlotDataItem(y, z, pen=pen)
        self._pdi_hand_endpts = pg.PlotDataItem([y[0], y[-1]], [z[0], z[-1]], pen=None, symbol=symbols,
                                                symbolSize=symbol_sizes, symbolBrush=symbol_color)

        y = self._pellet[:, 1]
        z = self._pellet[:, 2]
        self._pdi_pellet = pg.PlotDataItem(y, z, pen=pen)
        self._pdi_pellet.hide()
        self._pdi_pellet_endpts = pg.PlotDataItem([y[0], y[-1]], [z[0], z[-1]], pen=None, symbol=symbols,
                                                  symbolSize=symbol_sizes, symbolBrush=symbol_color)
        self._pdi_pellet_endpts.hide()

    @property
    def plot_data_items(self) -> List[pg.PlotDataItem]:
        return [self._pdi_hand, self._pdi_hand_endpts, self._pdi_pellet, self._pdi_pellet_endpts]

    @property
    def hand_y(self) -> np.ndarray:
        return self._hand[:, 1]

    @property
    def hand_z(self) -> np.ndarray:
        return self._hand[:, 2]

    @property
    def pellet_y(self) -> np.ndarray:
        return self._pellet[:, 1]

    @property
    def pellet_z(self) -> np.ndarray:
        return self._pellet[:, 2]

    def update(
            self, show_hand: Optional[bool] = None, show_pellet: Optional[bool] = None,
            show_endpts: Optional[bool] = None, hand_pen: Optional[QPen] = None, pellet_pen: Optional[QPen] = None,
            z_value: Optional[int] = None, playback_frame: Optional[int] = None
            ) -> None:
        """
        Update the rendered hand and/or pellet trace for this reach segment trajectory IAW the specified changes.
        :param show_hand: ``True/False`` to show/hide the hand trajectory, or ``None`` if no change.
        :param show_pellet: ``True/False`` to show/hide the pellet trajectory, or ``None`` if no change.
        :param show_endpts: ``True/False`` to show/hide symbols marking the start and end of hand and pellet traces, or
            ``None`` if no change.
        :param hand_pen: New pen for drawing hand trace, or ``None`` if no change.
        :param pellet_pen: New pen for drawing pellet trace, or ``None`` if no change.
        :param z_value: New Z-value for both hand and pellet trace, or ``None`` if no change.
        :param playback_frame: When playing back reach segment trajectories, this is the # of frames elapsed since
            start of segment. Of course, if it exceeds the segment duration, the entire trace is drawn. If -1, then
            ensure entire trace is drawn. ``None`` if no change.
        """
        if show_hand is not None:
            self._pdi_hand.setVisible(show_hand)
            self._pdi_hand_endpts.setVisible(show_hand)
        if show_pellet is not None:
            self._pdi_pellet.setVisible(show_pellet)
            self._pdi_pellet_endpts.setVisible(show_pellet)
        if hand_pen is not None:
            self._pdi_hand.setPen(hand_pen)
            self._pdi_hand_endpts.setSymbolBrush(hand_pen.color())
        if pellet_pen is not None:
            self._pdi_pellet.setPen(pellet_pen)
            self._pdi_pellet_endpts.setSymbolBrush(pellet_pen.color())
        if z_value is not None:
            self._pdi_hand.setZValue(z_value)
            self._pdi_hand_endpts.setZValue(z_value)
            self._pdi_pellet.setZValue(z_value)
            self._pdi_pellet_endpts.setZValue(z_value)
        if show_endpts is not None:
            # the end-of-seg marker is not shown if we're not rendering the entire segment!
            size_start = _ENDPOINT_SIZE if show_endpts else 0
            size_end = size_start if len(self._pdi_hand.xData) >= len(self._hand[:, 1]) else 0
            self._pdi_hand_endpts.setSymbolSize([size_start, size_end])
            self._pdi_pellet_endpts.setSymbolSize([size_start, size_end])
        if playback_frame is not None:
            d_len = len(self._hand[:, 1])
            t = d_len if playback_frame < 0 else min(playback_frame + 1, d_len)
            self._pdi_hand.setData(x=self._hand[:t, 1], y=self._hand[:t, 2])
            self._pdi_pellet.setData(x=self._pellet[:t, 1], y=self._pellet[:t, 2])
            endpts_on = self._pdi_hand_endpts.opts['symbolSize'][0] > 0
            size_start = _ENDPOINT_SIZE if endpts_on else 0
            size_end = size_start if endpts_on and (t == d_len) else 0
            self._pdi_hand_endpts.setSymbolSize([size_start, size_end])
            self._pdi_pellet_endpts.setSymbolSize([size_start, size_end])


class TrajPlot2D(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self._show_hand: bool = True
        """ Whether the hand trajectory traces are drawn. """
        self._hand_color: QColor = pg.mkColor('w')
        """ Color for hand trajectory traces (when not coloring by result). Defaults to opaque white. """
        self._color_by_result: bool = False
        """ If set, use reach segment result color for hand trajectory traces. Defaults to ``False``. """
        self._hand_line_style = Qt.PenStyle.SolidLine
        """ Line style for hand trajectory traces. Defaults to solid line."""
        self._hand_line_width = 2
        """ Line width for hand trajectory traces. Defaults to 2px. """
        self._show_pellet: bool = False
        """ Whether the pellet trajectory traces are drawn. """
        self._pellet_color: QColor = pg.mkColor('w')
        """ Color for pellet trajectory traces. Defaults to opaque white. """
        self._pellet_line_style = Qt.PenStyle.SolidLine
        """ Line style for pellet trajectory traces. Defaults to solid line. """
        self._pellet_line_width = 2
        """ Line width for pellet trajectory traces. Defaults to 2px. """

        self._show_endpts: bool = False
        """ Whether symbols are drawn at start and end of each hand/pellet trace. """
        self._show_hand_avg: bool = False
        """ Whether average hand trajectory trace is drawn (when applicable). """
        self._hand_avg_color: QColor = pg.mkColor('w')
        """ Color for average hand trajectory trace. Defaults to opaque white. """
        self._show_pellet_avg: bool = False
        """ Whether average pellet trajectory is drawn (when applicable). """
        self._pellet_avg_color: QColor = pg.mkColor('w')
        """ Color for average pellet trajectory. Defaults to opaque white. """

        self._all_trajectories: Dict[RxReachSegment, _ReachSegTraj] = dict()
        """ All reach segment trajectories (only some of which may be selected for display). """

        self._displayed_segs: List[RxReachSegment] = list()
        """ List of reach segments currently displayed in the trajectory plot. """

        self._focus_seg: Optional[RxReachSegment] = None
        """ The current focus segment (if any). It is highlighted with a thicker pen. """

        self._live_frame: Optional[int] = None
        """ 
        Current frame number in 'live mode', or None if live mode is off. This mode displays only the hand and/or
        pellet traces for the reach segment containing this frame, if there is one. Also, symbols mark the
        hand and pellet location along the respective trace at that point in time.
        """
        self._live_seg: Optional[RxReachSegment] = None
        """ The reach segment (if any) containing the current frame numbe in 'live mode', if enabled. """

        self._display_to_frame: Optional[int] = None
        """ 
        If not ``None``, this is the frame number T (elapsed since segment start) such that the interval [0..T] is 
        rendered for the hand and/or pellet traces of all reach segments currently selected for display. Ignored in 
        "live mode". 
        """

        self._pdi_hand_avg = pg.PlotDataItem()
        """ The averaged hand trace. """
        self._pdi_pellet_avg = pg.PlotDataItem()
        """ The averaged pellet trace. """
        self._ti_live_hand = pg.TargetItem(symbol='d', size=12, brush='w')
        """ Marks location of hand at current frame number in 'live mode', if enabled. """
        self._ti_live_pellet = pg.TargetItem(symbol='d', size=12, brush='w')

        # set up the PDIs for the average hand and pellet traces, initially with empty data and configured so they
        # appear on top of individual traces
        pen = pg.mkPen(color=self._hand_color, style=Qt.PenStyle.SolidLine, width=self._hand_line_width + 2)
        self._pdi_hand_avg.setPen(pen)
        self._pdi_hand_avg.setVisible(self._show_hand_avg)
        self._pdi_hand_avg.setZValue(1)
        self.addItem(self._pdi_hand_avg)
        pen = pg.mkPen(color=self._pellet_color, style=Qt.PenStyle.SolidLine, width=self._pellet_line_width + 2)
        self._pdi_pellet_avg.setPen(pen)
        self._pdi_pellet_avg.setVisible(self._show_pellet_avg)
        self._pdi_pellet_avg.setZValue(1)
        self.addItem(self._pdi_pellet_avg)

        # add the target items drawn at the live hand/pellet locations. Initially, they are hidden
        self._ti_live_hand.setVisible(False)
        self._ti_live_hand.setZValue(2)   # so it's drawn on top of the hand trace
        self.addItem(self._ti_live_hand)
        self._ti_live_pellet.setVisible(False)
        self._ti_live_pellet.setZValue(2)
        self.addItem(self._ti_live_pellet)

        # initial configuration: axis labels, grid on, invert Y axis (else things are upside down
        self.setLabel('bottom', 'Y (mm)')
        self.setLabel('left', 'Z (mm)')
        self.showGrid(x=True, y=True, alpha=0.5)
        self.getPlotItem().invertY(True)

    @property
    def grid_on(self) -> bool:
        """ ``True/False`` if the X/Y grid lines are on/off. """
        pi: pg.PlotItem = self.getPlotItem()
        return pi.getAxis('bottom').grid is not False

    @grid_on.setter
    def grid_on(self, on: bool) -> None:
        """
        Turn the X/Y grid lines on or off.
        :param on: ``True/False`` to turn grid lines on/off.
        """
        if on != self.grid_on:
            self.showGrid(x=on, y=on, alpha=0.5)

    @property
    def show_hand(self) -> bool:
        return self._show_hand

    @show_hand.setter
    def show_hand(self, show: bool) -> None:
        if show != self.show_hand:
            self._show_hand = show
            for seg, traj in self._all_trajectories.items():
                traj.update(show_hand=show)

    @property
    def hand_color(self) -> QColor:
        return self._hand_color

    @hand_color.setter
    def hand_color(self, color: QColor) -> None:
        if color != self.hand_color:
            self._hand_color = color
            if not self._color_by_result:
                self._on_hand_pen_changed()

    def _on_hand_pen_changed(self) -> None:
        for seg, traj in self._all_trajectories.items():
            color = _FOCUS_COLOR if seg == self._focus_seg else \
                (seg.result.ui_color if self._color_by_result else self._hand_color)
            line_w = _FOCUS_LW if seg == self._focus_seg else self._hand_line_width
            traj.update(hand_pen=pg.mkPen(color=color, style=self._hand_line_style, width=line_w),
                        z_value=1 if seg == self._focus_seg else 0)

    @property
    def hand_line_style(self) -> Qt.PenStyle:
        return self._hand_line_style

    @hand_line_style.setter
    def hand_line_style(self, style: Qt.PenStyle) -> None:
        if style != self.hand_line_style:
            self._hand_line_style = style
            self._on_hand_pen_changed()

    @property
    def hand_line_width(self) -> int:
        return self._hand_line_width

    @hand_line_width.setter
    def hand_line_width(self, width: int) -> None:
        if width != self.hand_line_width:
            self._hand_line_width = width
            self._on_hand_pen_changed()
            # the average hand trace line is 2px wider than the individual traces
            self._pdi_hand_avg.setPen(pg.mkPen(color=self._hand_avg_color, style=Qt.PenStyle.SolidLine,
                                               width=self._hand_line_width + 2))

    @property
    def color_by_result(self) -> bool:
        return self._color_by_result

    @color_by_result.setter
    def color_by_result(self, enable: bool) -> None:
        if enable != self._color_by_result:
            self._color_by_result = enable
            self._on_hand_pen_changed()

    @property
    def show_pellet(self) -> bool:
        return self._show_pellet

    @show_pellet.setter
    def show_pellet(self, show: bool) -> None:
        if show != self.show_pellet:
            self._show_pellet = show
            for seg, traj in self._all_trajectories.items():
                traj.update(show_pellet=show)

    @property
    def pellet_color(self) -> QColor:
        return self._pellet_color

    @pellet_color.setter
    def pellet_color(self, color: QColor) -> None:
        if color != self.pellet_color:
            self._pellet_color = color
            self._on_pellet_pen_changed()

    def _on_pellet_pen_changed(self) -> None:
        for seg, traj in self._all_trajectories.items():
            pp = pg.mkPen(color=_FOCUS_COLOR if seg == self._focus_seg else self._pellet_color,
                          style=self._pellet_line_style,
                          width=_FOCUS_LW if seg == self._focus_seg else self._pellet_line_width)
            traj.update(pellet_pen=pp, z_value=1 if seg == self._focus_seg else 0)

    @property
    def pellet_line_style(self) -> Qt.PenStyle:
        return self._pellet_line_style

    @pellet_line_style.setter
    def pellet_line_style(self, style: Qt.PenStyle) -> None:
        if style != self.pellet_line_style:
            self._pellet_line_style = style
            self._on_pellet_pen_changed()

    @property
    def pellet_line_width(self) -> int:
        return self._pellet_line_width

    @pellet_line_width.setter
    def pellet_line_width(self, width: int) -> None:
        if width != self.pellet_line_width:
            self._pellet_line_width = width
            self._on_pellet_pen_changed()
            # the average pellet trace line is 2px wider than the individual traces
            self._pdi_pellet_avg.setPen(pg.mkPen(color=self._pellet_avg_color, style=Qt.PenStyle.SolidLine,
                                                 width=self._pellet_line_width + 2))

    @property
    def show_endpts(self) -> bool:
        return self._show_endpts

    @show_endpts.setter
    def show_endpts(self, show: bool) -> None:
        if show != self.show_endpts:
            self._show_endpts = show
            for seg, traj in self._all_trajectories.items():
                traj.update(show_endpts=show)

    @property
    def show_hand_avg(self) -> bool:
        return self._show_hand_avg

    @show_hand_avg.setter
    def show_hand_avg(self, show: bool) -> None:
        if show != self.show_hand_avg:
            self._show_hand_avg = show
            if show:
                self._update_average_traces(pellet=False)
            self._pdi_hand_avg.setVisible(show)

    @property
    def hand_avg_color(self) -> QColor:
        return self._hand_avg_color

    @hand_avg_color.setter
    def hand_avg_color(self, color: QColor) -> None:
        if color != self.hand_avg_color:
            self._hand_avg_color = color
            self._pdi_hand_avg.setPen(pg.mkPen(color=color, style=Qt.PenStyle.SolidLine,
                                               width=self._hand_line_width + 2))

    @property
    def show_pellet_avg(self) -> bool:
        return self._show_pellet_avg

    @show_pellet_avg.setter
    def show_pellet_avg(self, show: bool) -> None:
        if show != self.show_pellet_avg:
            self._show_pellet_avg = show
            if show:
                self._update_average_traces(hand=False)
            self._pdi_pellet_avg.setVisible(show)

    @property
    def pellet_avg_color(self) -> QColor:
        return self._pellet_avg_color

    @pellet_avg_color.setter
    def pellet_avg_color(self, color: QColor) -> None:
        if color != self.pellet_avg_color:
            self._pellet_avg_color = color
            self._pdi_pellet_avg.setPen(pg.mkPen(color=color, style=Qt.PenStyle.SolidLine,
                                                 width=self._pellet_line_width + 2))

    @property
    def live_frame(self) -> Optional[int]:
        return self._live_frame

    @live_frame.setter
    def live_frame(self, frame: int) -> None:
        if frame != self._live_frame:
            old = self._live_frame
            self._live_frame = frame
            self._update_live_mode(old)

    def update_focus_segment(self, seg: Optional[RxReachSegment]) -> None:
        if seg == self._focus_seg:
            return
        if self._focus_seg is not None:
            traj = self._all_trajectories[self._focus_seg]
            color = self._focus_seg.result.ui_color if self._color_by_result else self._hand_color
            hp = pg.mkPen(color=color, style=self._hand_line_style, width=self._hand_line_width)
            pp = pg.mkPen(color=self._pellet_color, style=self._pellet_line_style, width=self._pellet_line_width)
            traj.update(hand_pen=hp, pellet_pen=pp, z_value=0)

        self._focus_seg = seg
        if self._focus_seg is None:
            return
        traj = self._all_trajectories[self._focus_seg]
        hp = pg.mkPen(color=_FOCUS_COLOR, style=self._hand_line_style, width=_FOCUS_LW)
        pp = pg.mkPen(color=_FOCUS_COLOR, style=self._pellet_line_style, width=_FOCUS_LW)
        traj.update(hand_pen=hp, pellet_pen=pp, z_value=1)

    def update_display_to_frame(self, frame: Optional[int]) -> None:
        if self._live_frame is not None:
            return
        if frame == self._display_to_frame:
            return
        self._display_to_frame = frame
        for seg, traj in self._all_trajectories.items():
            traj.update(playback_frame=-1 if self._display_to_frame is None else self._display_to_frame)

    def rebuild(self, segments: List[RxReachSegment], hand_traj: np.ndarray, pellet_traj: np.ndarray) -> None:
        """
        Rebuild the content of this 2D plot of reach segment trajectories. All trajectories are configured IAW the
        widget's current properties. Live mode is turned off, and there's no focus segment.

        NOTE: There must at least one reach segment, Both hand and pellet trajectories must be non-empty. If not, then
        the plot is reset and nothing is displayed,

        :param segments: All available reach segments. Assume all are initially selected for display.
        :param hand_traj: The hand trajectory throughout session: (N,4) np.float32 Numpy array, with columns holding
            the X, Y, Z, and speed of hand in each fraem.
        :param pellet_traj: The pellet trajectory throughout session, analogous to ``hand_traj``.
        """
        self.disableAutoRange()   # disabling auto-range during rebuild reduced execution time by ~50%

        # reset
        if len(self._all_trajectories) > 0:
            # Remove PDIs for all current hand/pellet traces
            for _, traj in self._all_trajectories.items():
                for pdi in traj.plot_data_items:
                    self.removeItem(pdi)
            self._all_trajectories.clear()
            self._displayed_segs.clear()
        if self._live_frame is not None:
            self._live_frame = None
            self._live_seg = None
            self._ti_live_hand.setVisible(False)
            self._ti_live_pellet.setVisible(False)
        self._focus_seg = None
        self._pdi_hand_avg.setVisible(False)
        self._pdi_pellet_avg.setVisible(False)

        # nothing to plot! We require at least one segment and non-empty hand/pellet trajectories.
        if len(segments) == 0 or hand_traj.size == 0 or pellet_traj.size == 0:
            return

        hand_pen = pg.mkPen(color=self._hand_color, style=self._hand_line_style, width=self._hand_line_width)
        pellet_pen = pg.mkPen(color=self._pellet_color, style=self._pellet_line_style,
                              width=self._pellet_line_width)

        for seg in segments:
            traj = _ReachSegTraj(seg, hand_traj, pellet_traj)
            if self._color_by_result:
                hand_pen = pg.mkPen(color=seg.result.ui_color, style=self._hand_line_style,
                                    width=self._hand_line_width)
            traj.update(show_hand=self._show_hand, show_pellet=self._show_pellet, show_endpts=self._show_endpts,
                        hand_pen=hand_pen, pellet_pen=pellet_pen)
            self._all_trajectories[seg] = traj
            self._displayed_segs.append(seg)
            for pdi in traj.plot_data_items:
                self.addItem(pdi)

        # update in the PlotDataItems for the average hand and pellet traces, but only update their data IF they
        # are currently shown.
        self._update_average_traces()
        self._pdi_hand_avg.setVisible(self._show_hand_avg)
        self._pdi_pellet_avg.setVisible(self._show_pellet_avg)

        self.enableAutoRange()   # re-enable auto range after rebuild

    def update_displayed_segments(self, displayed: List[RxReachSegment]) -> None:
        self._displayed_segs.clear()
        self._displayed_segs = [seg for seg in displayed if seg in self._all_trajectories]

        # in live mode, only the live segment traces are shown, even if that segment is not in the displayed set!
        if self._live_frame is not None:
            return

        for seg, traj in self._all_trajectories.items():
            show_hand = (seg in self._displayed_segs) and self._show_hand
            show_pellet = (seg in self._displayed_segs) and self._show_pellet
            traj.update(show_hand=show_hand, show_pellet=show_pellet)

        self._update_average_traces()

    def _update_average_traces(self, hand: bool = True, pellet: bool = True) -> None:
        if len(self._displayed_segs) == 0:
            return
        max_dur = max(s.dur for s in self._displayed_segs)
        shape = (max_dur, len(self._displayed_segs))
        if hand and self._show_hand_avg:
            if max_dur <= 0:
                self._pdi_hand_avg.setData(x=[], y=[])
            else:
                data_y = np.full(shape, np.nan, dtype=np.float32)
                data_z = np.full(shape, np.nan, dtype=np.float32)
                for i, seg in enumerate(self._displayed_segs):
                    span = min(seg.dur, max_dur)
                    traj = self._all_trajectories[seg]
                    data_y[:span, i] = traj.hand_y[:span]
                    data_z[:span, i] = traj.hand_z[:span]
                self._pdi_hand_avg.setData(x=np.nanmean(data_y, axis=1), y=np.nanmean(data_z, axis=1))
        if pellet and self._show_pellet_avg:
            if max_dur <= 0:
                self._pdi_pellet_avg.setData(x=[], y=[])
            else:
                data_y = np.full(shape, np.nan, dtype=np.float32)
                data_z = np.full(shape, np.nan, dtype=np.float32)
                for i, seg in enumerate(self._displayed_segs):
                    span = min(seg.dur, max_dur)
                    traj = self._all_trajectories[seg]
                    data_y[:span, i] = traj.pellet_y[:span]
                    data_z[:span, i] = traj.pellet_z[:span]
                self._pdi_pellet_avg.setData(x=np.nanmean(data_y, axis=1), y=np.nanmean(data_z, axis=1))

    def _update_live_mode(self, prev_live_frame: Optional[int]) -> None:
        """
        Update trajectory plot's "live mode" feature. It may have just been turned on, turned off, or the current
        live frame updated.

        :param prev_live_frame: Previous value of the live frame. ``None`` --> live mode off.
        """
        if isinstance(prev_live_frame, int) and isinstance(self._live_frame, int):
            if self._live_frame == prev_live_frame:
                return
            live_seg = self._seg_containing_frame(self._live_frame)
            if self._live_seg != live_seg:
                if self._live_seg is not None:
                    traj = self._all_trajectories[self._live_seg]
                    traj.update(show_hand=False, show_pellet=False)
                self._live_seg = live_seg
                if live_seg is not None:
                    traj = self._all_trajectories[live_seg]
                    traj.update(show_hand=self._show_hand, show_pellet=self._show_pellet)
            if self._live_seg is not None:
                t = self._live_frame - self._live_seg.frame
                traj = self._all_trajectories[self._live_seg]
                self._ti_live_hand.setPos(traj.hand_y[t], traj.hand_z[t])
                self._ti_live_hand.setVisible(self._show_hand)
                self._ti_live_pellet.setPos(traj.pellet_y[t], traj.pellet_z[t])
                self._ti_live_pellet.setVisible(self._show_pellet)
            else:
                self._ti_live_hand.setVisible(False)
                self._ti_live_pellet.setVisible(False)
        if (prev_live_frame is None) and isinstance(self._live_frame, int):
            self._live_seg = self._seg_containing_frame(self._live_frame)
            for seg, traj in self._all_trajectories.items():
                traj.update(show_hand=self._show_hand and (seg == self._live_seg),
                            show_pellet=self._show_pellet and (seg == self._live_seg))
            if self._live_seg is not None:
                t = self._live_frame - self._live_seg.frame
                traj = self._all_trajectories[self._live_seg]
                self._ti_live_hand.setPos(traj.hand_y[t], traj.hand_z[t])
                self._ti_live_hand.setVisible(self._show_hand)
                self._ti_live_pellet.setPos(traj.pellet_y[t], traj.pellet_z[t])
                self._ti_live_pellet.setVisible(self._show_pellet)
            # average traces NOT shown in live mode
            self._pdi_hand_avg.setVisible(False)
            self._pdi_pellet_avg.setVisible(False)
        elif (self._live_frame is None) and isinstance(prev_live_frame, int):
            self._live_seg = None
            self._ti_live_hand.setVisible(False)
            self._ti_live_pellet.setVisible(False)
            for seg, traj in self._all_trajectories.items():
                displayed = seg in self._displayed_segs
                traj.update(show_hand=displayed and self._show_hand, show_pellet=displayed and self._show_pellet)
            # restore average traces as well
            self._update_average_traces()
            self._pdi_hand_avg.setVisible(self._show_hand_avg)
            self._pdi_pellet_avg.setVisible(self._show_pellet_avg)

    def _seg_containing_frame(self, frame: int) -> Optional[RxReachSegment]:
        return next((seg for seg in self._displayed_segs if seg.frame <= frame <= seg.end_frame), None)
