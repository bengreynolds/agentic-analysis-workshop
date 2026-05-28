"""
analysisview.py - Trajectory analysis window/view for plotting reach trajectories.

Implementation of a **trajectory analysis view** that is housed in a separate floating window rather being docked
within ReachX's main application window. Key capabilities:

- Curated vs scored reach toggle with per-reach selection and filtering.
- Default 2D Y/Z plotting with optional 3D view (when backend available).
- Playback timeline and figure export (PNG/SVG/PDF/EPS), plus MP4 export with fallback to PNG sequence.
- Live mode to sync plotting with the current frame in the main application window, showing only the reach containing
  that frame and a tracer to the current position.

CREDITS: Ben Reynolds wrote much of the original code, and used Agentic to adapt it to the ReachX UI framework.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

# noinspection PyPackageRequirements
import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Signal, Slot, \
    QPoint, QTimer
from PySide6.QtGui import QBrush, QColor, QImage, QStandardItemModel, QCloseEvent, QShortcut, QKeySequence, \
    QFontMetrics
from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, \
    QCheckBox, QSpinBox, QDoubleSpinBox, QTableView, QHeaderView, QAbstractItemView, QFileDialog, \
    QMessageBox, QStackedWidget, QStatusBar, QMainWindow, QDockWidget, QColorDialog, QFrame, QGroupBox, QSizePolicy

from reachx.common import RxReachSegment, RxReach, RxHandPos
from reachx.gui.trajplot import TrajPlot2D
from reachx.uicommon import RxIcons
from reachx.config.app_log import get_application_logger
from reachx.data.datamanager import DataManager
from reachx.gui.baseview import BaseView

_LOG = get_application_logger()
""" Module reference to ReachX's main application logger. """

# at import time, check whether PyQtGraph OpenGL support is available for doing 3D rendering.
try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception as e:
    _LOG.warning(f"OpenGL support not available. 3D view disabled in Trajectory Analysis Window:   \n{e}")
    gl = None
    _HAS_GL = False


@dataclass
class _ReachRow:
    """ Encapsulates a row in the reach metrics table. """
    segment: RxReachSegment
    """ The reach segment. """
    selected: bool = True
    """ True if row is currently selected. """
    metrics: Optional[_ReachMetrics] = None
    """ Metrics associated with the reach. """


@dataclass
class _ReachMetrics:
    """ Reach segment metrics. Note that a given metric will be `None` if it could not be/has not been computed.  """
    dx: Optional[float] = None
    """ Hand-pellet separation in X at end of reach, in mm. """
    dy: Optional[float] = None
    """ Hand-pellet separation in Y at end of reach, in mm. """
    dz: Optional[float] = None
    """ Hand-pellet separation in Z at end of reach, in mm. """
    dist: Optional[float] = None
    """ Hand-pellet separation in 3D space at end of reach, in mm. """


def _format_metric(val: Optional[float]) -> str:
    """
    Format a reach metric value for tabular display.
    :param val: The value. If None or not a number, returns '-'. Else, returns string representation of floating-point
        value with 2 decimal digits.
    :raise TypeError: If val is not an instance of `float`.
    """
    if val is None:
        return "-"
    try:
        if math.isnan(val):
            return "-"
    except TypeError:
        pass
    return f"{val:.2f}"


class _ReachMetricsTM(QAbstractTableModel):
    """
    Table model for the reach segment metrics table. It displays the reach segment parameters as well as computed
    metrics (``_ReachMetrics``), and includes a column to select/deselect individual reaches for plotting.
    """
    selection_changed: Signal = Signal()
    """ The current selection has changed. """
    focus_changed: Signal = Signal()
    """ Signal emitted whenever the focus row in the table changes. """

    COLUMNS: List[str] = [
        "", "Frame", "\u0394 Max", "Dur", "Result", "Hand Pos", "dX", "dY", "dZ", "Dist"
    ]

    METRIC_FIELDS: List[str] = ["dx", "dy", "dz", "dist"]

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[_ReachRow] = list()
        """ The table rows. """
        self._focus_row_idx: int = -1

    def set_segments(self, segments: List[RxReachSegment]) -> None:
        """
        Set the list of reach segments encapsulated in this model. All segments are initially selected with no metrics
        computed, and the ``selection_changed`` signal is emitted.

        :param segments: Reach segment list, possibly empty.
        """
        self.beginResetModel()
        self._rows.clear()
        self._rows = [_ReachRow(seg, True) for seg in segments]
        self.endResetModel()
        self.selection_changed.emit()

    def toggle_selected_state_of_row(self, row: int):
        check_state = Qt.CheckState.Unchecked if self.is_selected(row) else Qt.CheckState.Checked
        self.setData(self.index(row, 0), check_state, Qt.ItemDataRole.CheckStateRole)

    def set_focus_row(self, focus_idx: int):
        # since only one row can have the focus at a time, need to repaint the previous focus row as well as the row
        # gaining the focus.
        if (-1 <= focus_idx < self.rowCount()) and (focus_idx != self._focus_row_idx):
            old_focus_idx = self._focus_row_idx
            self._focus_row_idx = focus_idx
            min_row = self._focus_row_idx if old_focus_idx < 0 else min(old_focus_idx, self._focus_row_idx)
            max_row = self._focus_row_idx if old_focus_idx < 0 else max(old_focus_idx, self._focus_row_idx)
            top_left = self.index(min_row, 0)
            bot_right = self.index(max_row, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bot_right, [Qt.ItemDataRole.BackgroundRole])
            self.focus_changed.emit()

    def rowCount(self, /, parent=...) -> int:
        return len(self._rows)

    def columnCount(self, /, parent=...) -> int:
        return len(self.COLUMNS)

    def headerData(self, section: int, orientation, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return super().headerData(section, orientation, role)

    def index(self, row, column, /, parent=...):
        return self.createIndex(row, column)

    # Pycharm bug: Misses the pure virtual parent(self, index) in QAbstractItemModel
    # noinspection PyMethodOverriding
    def parent(self, idx: QModelIndex) -> QModelIndex:
        return QModelIndex()

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsEnabled
        flags |= Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index: QModelIndex, /, role=...):
        if not index.isValid():
            return None
        row = index.row()
        if not (0 <= row < len(self._rows)):
            return None
        seg = self._rows[row].segment
        selected = self._rows[row].selected
        metrics = self._rows[row].metrics
        has_focus = row == self._focus_row_idx

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return ""
            if index.column() == 1:
                return str(seg.frame)
            if index.column() == 2:
                return f"+{seg.max_delta}"
            if index.column() == 3:
                return str(seg.dur)
            if index.column() == 4:
                return seg.result.short_name
            if index.column() == 5:
                return str(seg.hand_pos)
            if index.column() >= 6:
                metric_idx = index.column() - 6
                if 0 <= metric_idx < len(self.METRIC_FIELDS) and metrics is not None:
                    val = getattr(metrics, self.METRIC_FIELDS[metric_idx], None)
                    return _format_metric(val)
                return "-"
            return ""

        if role == Qt.ItemDataRole.UserRole:
            if index.column() == 0:
                return 1 if selected else 0
            if index.column() == 1:
                return int(seg.frame)
            if index.column() == 2:
                return int(seg.max_delta)
            if index.column() == 3:
                return int(seg.dur)
            if index.column() == 4:
                return int(seg.result.value)
            if index.column() == 5:
                return int(seg.hand_pos.value)
            if index.column() >= 6:
                metric_idx = index.column() - 6
                if 0 <= metric_idx < len(self.METRIC_FIELDS) and metrics is not None:
                    return getattr(metrics, self.METRIC_FIELDS[metric_idx], None)
            return None

        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0:
            return Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        if role == Qt.ItemDataRole.BackgroundRole and index.column() == 4:
            color = QColor(seg.result.ui_color)
            color.setAlphaF(0.5)
            return QBrush(color)

        if role == Qt.ItemDataRole.BackgroundRole and (has_focus or not selected):
            # note: focus background color overrides unselected background color
            if has_focus:
                return QBrush(QColor(0, 0, 255, 60))
            else:
                return QBrush(QColor(255, 0, 0, 60))

        return None

    def setData(self, index: QModelIndex, value, /, role=...):
        if not index.isValid():
            return False
        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            row = index.row()
            if 0 <= row < len(self._rows):
                self._rows[row].selected = (value == Qt.CheckState.Checked)
                left = self.index(row, 0)
                right = self.index(row, self.columnCount() - 1)
                self.dataChanged.emit(left, right)
                self.selection_changed.emit()
                return True
        return False

    def set_all_selected(self, selected: bool) -> None:
        if len(self._rows) == 0:
            return
        self._rows = [_ReachRow(r.segment, selected, r.metrics) for r in self._rows]
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._rows) - 1, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right)
        self.selection_changed.emit()

    def segment_at(self, row: int) -> Optional[RxReachSegment]:
        if 0 <= row < len(self._rows):
            return self._rows[row].segment
        return None

    @property
    def focus_segment(self) -> Optional[RxReachSegment]:
        if 0 <= self._focus_row_idx < len(self._rows):
            return self._rows[self._focus_row_idx].segment
        return None

    @property
    def focus_row(self) -> int:
        return self._focus_row_idx

    def is_selected(self, row: int) -> bool:
        return 0 <= row < len(self._rows) and self._rows[row].selected

    def selected_segments(self) -> List[RxReachSegment]:
        return [r.segment for r in self._rows if r.selected]

    def segments(self) -> List[RxReachSegment]:
        return [r.segment for r in self._rows]

    def row_for_segment(self, seg: RxReachSegment) -> int:
        for idx, row in enumerate(self._rows):
            if row.segment == seg:
                return idx
        return -1

    def set_metrics(self, metrics: Dict[RxReachSegment, _ReachMetrics]) -> None:
        for row in self._rows:
            row.metrics = metrics.get(row.segment)
        if len(self._rows) == 0:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._rows) - 1, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right)

    def metrics_for_segment(self, seg: RxReachSegment) -> Optional[_ReachMetrics]:
        for row in self._rows:
            if row.segment == seg:
                return row.metrics
        return None


class _ReachFilterProxy(QSortFilterProxyModel):
    """ Sort-and-filter proxy implements sorting and filtering of the contents of ``_ReachMetricsTM``. """
    def __init__(self) -> None:
        super().__init__()
        self._result_filter: Optional[RxReach] = None
        """ If not None, accept only reach segments with the specified result. """
        self._hand_pos_filter: Optional[RxHandPos] = None
        """ If not None, accept only reach segments with the specified hand position at 'reachMax'. """
        self._dur_min: int = 0
        """ Accept only reach segments with duration >= this value. """
        self._dur_max: int = 0
        """ Accept only reach segments with duration <= this value. """
        self._delta_min: int = 0
        """ Accept only reach segments with 'reachMax' occurring at least this many frames after reach start. """
        self._delta_max: int = 0
        """ Accept only reach segments with 'reachMax' occurring at most this many frames after reach start. """

    def set_result_filter(self, res: Optional[RxReach]) -> None:
        self._result_filter = res
        self.invalidateFilter()

    def set_dur_range(self, min_val: int, max_val: int) -> None:
        self._dur_min = min_val
        self._dur_max = max_val
        self.invalidateFilter()

    def set_hand_pos_filter(self, pos: Optional[RxHandPos]) -> None:
        self._hand_pos_filter = pos
        self.invalidateFilter()

    def set_delta_range(self, min_val: int, max_val: int) -> None:
        self._delta_min = min_val
        self._delta_max = max_val
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        src = self.sourceModel()
        src_cast: _ReachMetricsTM = src if isinstance(src, _ReachMetricsTM) else None
        if src_cast is None:
            return False

        seg = src_cast.segment_at(source_row)
        if not isinstance(seg, RxReachSegment):
            return False

        if isinstance(self._result_filter, RxReach) and seg.result != self._result_filter:
            return False
        if isinstance(self._hand_pos_filter, RxHandPos) and seg.hand_pos != self._hand_pos_filter:
            return False
        if self._dur_min > 0 and seg.dur < self._dur_min:
            return False
        if 0 < self._dur_max < seg.dur:
            return False
        if self._delta_min > 0 and seg.max_delta < self._delta_min:
            return False
        if 0 < self._delta_max < seg.max_delta:
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        # disable sorting on the first column ("Select")
        if self.sortColumn() == 0:
            return False

        left_val = self.sourceModel().data(left, Qt.ItemDataRole.UserRole)
        right_val = self.sourceModel().data(right, Qt.ItemDataRole.UserRole)
        try:
            return left_val < right_val
        except Exception:
            return str(left_val) < str(right_val)


_NCOLS: int = 4
""" Number of columns in hand or pellet trajectory data array. """
_COL_X: int = 0
""" Index of column in hand or pellet trajectory corresponding to the X-coordinate vector. """
_COL_Y: int = 1
""" Index of column in hand or pellet trajectory corresponding to the X-coordinate vector. """
_COL_Z: int = 2
""" Index of column in hand or pellet trajectory corresponding to the X-coordinate vector. """
_COL_SPEED: int = 3
""" Index of column in hand or pellet trajectory corresponding to the X-coordinate vector. """


class AnalysisView(BaseView):
    """
    The trajectory analysis view -- for plotting reach trajectories based on curated or scored reach segments for the
    current experiment session.
    """
    _SOURCE_CURATED = "Curated reaches"
    _SOURCE_SCORED = "Scored reaches"

    def __init__(self, data_manager: DataManager, status_bar: QStatusBar) -> None:
        super().__init__("Trajectory Analysis", None, data_manager)

        self._hand_traj: np.ndarray = np.zeros((0, _NCOLS), dtype=np.float32)
        """ 3D trajectory (X, Y, Z, speed) of right hand for the current loaded session; empty if trajectory N/A. """
        self._pellet_traj: np.ndarray = np.zeros((0, _NCOLS), dtype=np.float32)
        """ 3D trajectory (X, Y, Z, speed) of pellet for the current loaded session; empty if trajectory N/A. """
        self._export_segments_override: Optional[List[RxReachSegment]] = None
        """ 
        While exporting video of the focused reach trajectory, this contains the focus segment. See
        ``_selected_visible_segments``.
        """

        self._last_status: str = ""
        """ Last status message displayed. """
        self._status_bar: QStatusBar = status_bar
        """ A reference to the containing window's status bar. This view controls the status bar's content. """

        self._status_label = QLabel("Analysis idle.")
        """ Status bar label reflecting status of trajectory analysis view. """
        self._session_status_label = QLabel("- | -")
        """ Status bar label reflecting session ID and current scorer name for the current loaded session (if any). """
        self._status_bar.setSizeGripEnabled(False)
        self._status_bar.setFixedHeight(22)
        self._status_bar.addWidget(self._session_status_label, 1)
        self._status_bar.addPermanentWidget(self._status_label)

        self._build_plotting_panel()

        self._panel_widgets: List[Tuple[str, QWidget]] = [
            ("Reach Selection", self._build_reach_metrics_panel()),
        ]
        """ Widget panels that are dockable around the central widget plotting reach segment trajectories. """

        self._connect_signals()
        self._reload_all()

    def _build_plotting_panel(self) -> None:
        """
        Constructs the plotting panel, housing the stacked 2D and 3D (if available) plot widgets, along with
        various controls governing plotting behavior and appearance. This panel is installed as the "view container"
        for the ``AnalysisView``, which as the central widget in the parent ``AnalysisWindow``.
        """
        self._plot_mode_combo = QComboBox()
        """ Combo box selects 2D or 3D plotting mode. 3D plotting may not be available. """
        self._plot_mode_combo.addItem("2D (Y/Z)")
        self._plot_mode_combo.addItem("3D (XYZ)")
        if not _HAS_GL:
            self._disable_combo_item(self._plot_mode_combo, 1, "3D plotting backend unavailable")

        self._show_grid = QCheckBox("Grid")
        """ Checkbox to show/hide the grid in the 2D plot of reach trajectories. """
        self._show_grid.setToolTip("Show/hide plot grid lines")
        self._show_grid.setChecked(True)
        self._lock_aspect = QCheckBox("Lock aspect")
        """ Checkbox to either lock or unlock aspect ratio of the 2D plot of reach trajectories. """
        self._lock_aspect.setToolTip("Preserve aspect ratio such that Y and Z are scaled equally.")
        self._show_endpoints = QCheckBox("Endpoints")
        """ Checkbox to show/hide markers at the endpoints of each hand or pellet trajectory plotted. """
        self._show_endpoints.setToolTip("Show/hide endpoint markers on each hand or pellet trajectory plotted.")
        self._live_enabled = QCheckBox("Live")
        """ 
        If checked, plot the single reach trajectory containing the session's current frame, and render a marker
        along that trajectory at that frame.
        """
        self._live_enabled.setToolTip("If checked, render the reach trajectory containing the current \n"
                                      "frame and place a marker at that frame.")
        self._playback_enabled = QCheckBox("Playback:")
        """ Checkbox enables playback of all reach trajectories using the accompanying slider. """
        self._playback_enabled.setToolTip("Use slider or step forward/back buttons to playback selected reaches "
                                          "in unison. \nNot available in live mode.")
        self._forward_btn = QPushButton(RxIcons.STEP_FWD, "")
        """ Press this button to increment current playback frame when playback mode is enabled. """
        self._back_btn = QPushButton(RxIcons.STEP_BACK, "")
        """ Press this button to decrement current playback frame when playback mode is enabled. """
        self._playback_frame_readout = QLabel("0")
        """ Static label displays elapsed frame number since start of the reach trajectories selected for playback. """
        self._playback_frame: int = 0
        """ When playback enabled, the current elapsed frame number since start of reach trajectory playback. """
        self._playback_dur: int = 0
        """ When playback enabled, duration of longest reach trajectory selected for playback. """

        # initially, playback feature is turned off
        self._playback_enabled.setChecked(False)
        self._forward_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._forward_btn.setStyleSheet("QPushButton { border: none; }")
        self._forward_btn.setEnabled(False)
        self._forward_btn.setToolTip("Advance to next frame in displayed reaches (hot key = '.')")
        self._back_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._back_btn.setStyleSheet("QPushButton { border: none; }")
        self._back_btn.setEnabled(False)
        self._back_btn.setToolTip("Rewind to previous frame in displayed reaches (hot key = ',')")

        # want fixed-width readout wide enough to accommodate "0000"
        self._playback_frame_readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._playback_frame_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        font_metrics = QFontMetrics(self._playback_frame_readout.font())
        self._playback_frame_readout.setFixedWidth(font_metrics.horizontalAdvance("0000"))
        self._playback_frame_readout.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._traj_plot_2d = TrajPlot2D()
        """ A customized PlotWidget provides the 2D view of reach trajectories. """

        self._plot_stack = QStackedWidget()
        """ Stack widget lets user see 2D or 3D plot (if available). """
        self._plot_stack.addWidget(self._traj_plot_2d)
        if _HAS_GL:
            self._plot_3d = gl.GLViewWidget()
            """ OpenGL plot widget for 3D rendering, if available."""
            self._plot_stack.addWidget(self._plot_3d)
        else:
            placeholder = QLabel("3D plotting backend not available")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plot_stack.addWidget(placeholder)

        self._show_hand = QCheckBox("Hand: ")
        """ Checkbox to show/hide trajectories of hand during each reach segment. """
        self._show_hand.setChecked(True)
        self._hand_color_picker = CompactColorPicker()
        """ Custom PB raises dialog to choose hand trajectory color, including alpha channel. """
        self._hand_line_style = QComboBox()
        """ Selects line style for rendering hand trajectories. """
        self._hand_line_style.addItems(["Solid", "Dash", "Dot"])
        self._hand_line_style.setToolTip("Select line style for rendering hand trajectories.")
        self._hand_line_width = QSpinBox()
        """ Selects line width for rendering hand trajectories. """
        self._hand_line_width.setRange(1, 10)
        self._hand_line_width.setValue(2)
        self._hand_line_width.setToolTip("Select line width for rendering hand trajectories.")
        self._hand_avg = QCheckBox("Avg?:")
        """ Show/hide the average hand trajectory across all selected reaches. """
        self._hand_avg.setToolTip("Show/hide the averaged hand trajectory")
        self._hand_avg_color_picker = CompactColorPicker()
        """ Custom PB raises dialog to choose color for average hand trajectory, including alpha channel. """
        self._color_by_result = QCheckBox("Color by result")
        """ If checked, hand trajectory pen color reflects the result of that reach. """

        self._show_pellet = QCheckBox("Pellet")
        """ Checkbox to show/hide trajectories of pellet during each reach segment. """
        self._show_pellet.setChecked(False)
        self._pellet_color_picker = CompactColorPicker()
        """ Custom PB raises dialog to choose pellet trajectory color, including alpha channel. """
        self._pellet_line_style = QComboBox()
        """ Selects line style for rendering pellet trajectories. """
        self._pellet_line_style.addItems(["Solid", "Dash", "Dot"])
        self._pellet_line_style.setToolTip("Select line style for rendering pellet trajectories.")
        self._pellet_line_width = QSpinBox()
        """ Selects line width for rendering pellet trajectories. """
        self._pellet_line_width.setRange(1, 10)
        self._pellet_line_width.setValue(2)
        self._pellet_line_width.setToolTip("Select line width for rendering pellet trajectories.")
        self._pellet_avg = QCheckBox("Avg?:")
        """ Show/hide the average pellet trajectory across all selected reaches. """
        self._pellet_avg.setToolTip("Show/hide the averaged pellet trajectory")
        self._pellet_avg_color_picker = CompactColorPicker()
        """ Custom PB raises dialog to choose color for average pellet trajectory, including alpha channel. """

        # widgets for exporting trajectory plot as a figure
        self._fig_format = QComboBox()
        """ Combo box selects the output format for the trajectory plot figure. """
        self._fig_format.addItems(["png", "pdf", "eps", "svg"])
        self._fig_width = QSpinBox()
        """ Editable spin box selects figure width in pixels. """
        self._fig_width.setRange(100, 10000)
        self._fig_width.setToolTip("Figure width in pixels [100..10000]")
        self._fig_height = QSpinBox()
        """ Editable spin box selects figure height in pixels."""
        self._fig_height.setRange(100, 10000)
        self._fig_height.setToolTip("Figure height in pixels [100..10000]")
        self._fig_dpi = QSpinBox()
        """ Editable spin box selects figure resolution in dots per inch (pdf/eps only). """
        self._fig_dpi.setRange(72, 600)
        self._fig_dpi.setValue(300)
        self._fig_dpi.setToolTip("Figure resolution in dots per inch [72..600]. ")
        self._fig_gain = QDoubleSpinBox()
        """ Editable spin box selects figure's color gain (png only). """
        self._fig_gain.setRange(0.1, 5.0)
        self._fig_gain.setSingleStep(0.1)
        self._fig_gain.setValue(1.0)
        self._fig_gain.setToolTip("Color gain for PNG only [0.1 .. 5.0].")
        self._fig_gamma = QDoubleSpinBox()
        """ Editable spin box selects figure's color gamma (png only). """
        self._fig_gamma.setRange(0.1, 5.0)
        self._fig_gamma.setSingleStep(0.1)
        self._fig_gamma.setValue(1.0)
        self._fig_gamma.setToolTip("Color gamma for PNG only [0.1 .. 5.0].")

        self._fig_export_btn = QPushButton("Export")
        """ Button pressed to export 2D trajectory plot to file. """

        # widgets for exporting playback of focussed reach's trajectory as a video
        self._video_fps = QSpinBox()
        """ Editable spin box selects video frames-per-second. """
        self._video_fps.setRange(1, 120)
        self._video_fps.setValue(30)
        self._video_fps.setToolTip("Video frames-per-second [30 to 120].")
        self._video_fallback = QComboBox()
        """ Combo box selects alternate if MP4 codec not available. """
        self._video_fallback.addItems(["No fallback", "PNG sequence"])
        self._video_fallback.setToolTip("Select fallback option if unable to export as MP4 file.")
        self._video_gain = QDoubleSpinBox()
        """ Spin box selects color gain for video output. """
        self._video_gain.setRange(0.1, 5.0)
        self._video_gain.setSingleStep(0.1)
        self._video_gain.setValue(1.0)
        self._video_gain.setToolTip("Video color gain [0.1 .. 5.0].")
        self._video_gamma = QDoubleSpinBox()
        """ Spin box selects color gamma for video output. """
        self._video_gamma.setRange(0.1, 5.0)
        self._video_gamma.setSingleStep(0.1)
        self._video_gamma.setValue(1.0)
        self._video_gamma.setToolTip("Video color gamma [0.1 .. 5.0].")

        self._video_export_btn = QPushButton("Export MP4")
        """ Button pressed to export video of the focussed reach trajectory. """

        plot_layout = QVBoxLayout()
        plot_row = QHBoxLayout()
        plot_row.addWidget(self._plot_mode_combo)
        plot_row.addWidget(self._show_grid)
        plot_row.addWidget(self._lock_aspect)
        plot_row.addWidget(self._show_endpoints)

        plot_row.addSpacing(10)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(2)
        plot_row.addWidget(separator)
        plot_row.addSpacing(10)

        plot_row.addWidget(self._live_enabled)

        plot_row.addSpacing(10)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(2)
        plot_row.addWidget(separator)
        plot_row.addSpacing(10)

        plot_row.addWidget(self._playback_enabled)
        plot_row.addWidget(self._back_btn)
        plot_row.addWidget(self._playback_frame_readout)
        plot_row.addWidget(self._forward_btn)
        plot_row.addStretch(1)
        plot_layout.addLayout(plot_row)

        plot_layout.addWidget(self._plot_stack, stretch=1)

        hand_pellet_layout = QGridLayout()
        hand_pellet_layout.setSpacing(10)
        hand_pellet_layout.addWidget(self._show_hand, 0, 0)
        hand_pellet_layout.addWidget(self._hand_color_picker, 0, 1)
        hand_pellet_layout.addWidget(self._hand_line_style, 0, 2)
        hand_pellet_layout.addWidget(self._hand_line_width, 0, 3)
        hand_pellet_layout.addWidget(self._hand_avg, 0, 4)
        hand_pellet_layout.addWidget(self._hand_avg_color_picker, 0, 5)
        hand_pellet_layout.addWidget(self._color_by_result, 0, 6)
        hand_pellet_layout.addWidget(self._show_pellet, 0, 8)
        hand_pellet_layout.addWidget(self._pellet_color_picker, 0, 9)
        hand_pellet_layout.addWidget(self._pellet_line_style, 0, 10)
        hand_pellet_layout.addWidget(self._pellet_line_width, 0, 11)
        hand_pellet_layout.addWidget(self._pellet_avg, 0, 12)
        hand_pellet_layout.addWidget(self._pellet_avg_color_picker, 0, 13)
        hand_pellet_layout.setColumnStretch(7, 1)
        plot_layout.addLayout(hand_pellet_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(2)
        separator.setMinimumHeight(20)
        plot_layout.addWidget(separator)

        export_groups = QHBoxLayout()
        group_box = QGroupBox("Export trajectory plot")
        export_fig_layout = QGridLayout()
        export_fig_layout.setContentsMargins(2, 2, 2, 2)
        export_fig_layout.setHorizontalSpacing(5)
        export_fig_layout.setVerticalSpacing(8)
        export_fig_layout.addWidget(self._make_label("Format", self._fig_format), 0, 0)
        export_fig_layout.addWidget(self._fig_format, 0, 1)
        export_fig_layout.addWidget(self._make_label("W (px)", self._fig_width), 0, 2)
        export_fig_layout.addWidget(self._fig_width, 0, 3)
        export_fig_layout.addWidget(self._make_label("H (px)", self._fig_height), 0, 4)
        export_fig_layout.addWidget(self._fig_height, 0, 5)
        export_fig_layout.addWidget(self._make_label("DPI", self._fig_dpi), 1, 0)
        export_fig_layout.addWidget(self._fig_dpi, 1, 1)
        export_fig_layout.addWidget(self._make_label("Gain", self._fig_gain), 1, 2)
        export_fig_layout.addWidget(self._fig_gain, 1, 3)
        export_fig_layout.addWidget(self._make_label("Gamma", self._fig_gamma), 1, 4)
        export_fig_layout.addWidget(self._fig_gamma, 1, 5)
        export_fig_layout.addWidget(self._fig_export_btn, 0, 6, 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        export_fig_layout.setColumnStretch(6, 1)
        group_box.setLayout(export_fig_layout)
        export_groups.addWidget(group_box)

        group_box = QGroupBox("Export playback of focussed trajectory")
        export_vid_layout = QGridLayout()
        export_vid_layout.setContentsMargins(2, 2, 2, 2)
        export_vid_layout.setHorizontalSpacing(5)
        export_vid_layout.setVerticalSpacing(8)
        export_vid_layout.addWidget(self._make_label("FPS", self._video_fps), 0, 0)
        export_vid_layout.addWidget(self._video_fps, 0, 1)
        export_vid_layout.addWidget(self._make_label("Fallback", self._video_fallback), 0, 2)
        export_vid_layout.addWidget(self._video_fallback, 0, 3)
        export_vid_layout.addWidget(self._make_label("Gain", self._video_gain), 1, 0)
        export_vid_layout.addWidget(self._video_gain, 1, 1)
        export_vid_layout.addWidget(self._make_label("Gamma", self._video_gamma), 1, 2)
        export_vid_layout.addWidget(self._video_gamma, 1, 3)
        export_vid_layout.addWidget(self._video_export_btn, 0, 4, 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        export_vid_layout.setColumnStretch(4, 1)
        group_box.setLayout(export_vid_layout)
        export_groups.addWidget(group_box)

        plot_layout.addLayout(export_groups)

        main_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout, stretch=1)
        self.view_container.setLayout(main_layout)

    def _build_reach_metrics_panel(self) -> QWidget:
        self._reach_source_combo = QComboBox()
        """ Combo box to select source for reach segments displayed: scored or user-curated. """
        self._reach_source_combo.addItems([self._SOURCE_CURATED, self._SOURCE_SCORED])
        self._segmentation_result_combo = QComboBox()
        """ Combo box to select which scored-reach result is displayed. """
        self._segmentation_result_combo.addItem("No reach results")
        self._segmentation_result_combo.setEnabled(False)

        self._endpoint_mode_combo = QComboBox()
        """ Combo box selects criterion for the "endpoint frame" of a reach segment (for computing metrics). """
        self._endpoint_mode_combo.addItem("Reach Max", "max")
        self._endpoint_mode_combo.addItem("Reach End", "end")
        self._endpoint_mode_combo.addItem("Min Distance", "min_dist")

        self._pellet_stop_speed_combo = QDoubleSpinBox()
        """ Spin box selects pellet speed threshold to consider pellet as having stopped moving. """
        self._pellet_stop_speed_combo.setRange(0.0, 1.0)
        self._pellet_stop_speed_combo.setSingleStep(0.001)
        self._pellet_stop_speed_combo.setDecimals(4)
        self._pellet_stop_speed_combo.setValue(0.01)

        self._reach_model = _ReachMetricsTM()
        """ Reach segment metrics table model. """
        self._reach_proxy = _ReachFilterProxy()
        """ Sort filter proxy model wrapping the reach segment metrics table model. """
        self._reach_proxy.setSourceModel(self._reach_model)

        self._reach_table = QTableView()
        """ The reach segment metrics table. """
        self._reach_table.setModel(self._reach_proxy)
        self._reach_table.setSortingEnabled(True)
        self._reach_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._reach_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._reach_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._reach_table.verticalHeader().setVisible(False)
        self._reach_table.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)
        self._reach_table.setAlternatingRowColors(False)

        # infrastructure to veto sorting on column 0 ("Select" column)
        self._reach_table.horizontalHeader().setSortIndicator(1, Qt.SortOrder.AscendingOrder)
        self._reach_table.horizontalHeader().sortIndicatorChanged.connect(self._on_sort_change)
        self._last_sort_col = 1
        """ Last valid sort column for the reach segment metrics table. """
        self._last_sort_order = Qt.SortOrder.AscendingOrder
        """ Last valid sort order for the reach segment metrics table. """

        self._focus_prev_btn = QPushButton(RxIcons.PREV, "")
        """ Push button to put focus on reach segment prior to the current focus segment. """
        self._focus_prev_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._focus_prev_btn.setStyleSheet("QPushButton { border: none; }")
        self._focus_prev_btn.setToolTip("Go to reach segment preceding current focus segment.")

        self._focus_next_btn = QPushButton(RxIcons.NEXT, "")
        """ Push button to put focus on reach segment after current focus segment. """
        self._focus_next_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._focus_next_btn.setStyleSheet("QPushButton { border: none; }")
        self._focus_next_btn.setToolTip("Go to reach segment after current focus segment.")

        self._select_all_btn = QPushButton("Select all")
        """ Pressing this button selects all segments displayed in reach metrics table."""
        self._deselect_all_btn = QPushButton("Deselect all")
        """ Pressing this button deselects all segments displayed in reach metrics table. """

        self._result_filter_combo = QComboBox()
        """ Combo box to specify reach segment result for filtering the reach metrics table. """
        self._result_filter_combo.addItem("All results")
        for res in [RxReach.GRABBED, RxReach.MISSED, RxReach.DROPPED, RxReach.STALLED, RxReach.NONE]:
            self._result_filter_combo.addItem(res.short_name, res)

        self._handpos_filter_combo = QComboBox()
        """ Combo box to specify reach segment hand position for filtering the reach metrics table. """
        self._handpos_filter_combo.addItem("All positions")
        for pos in RxHandPos:
            self._handpos_filter_combo.addItem(str(pos), pos)

        self._dur_min_spin = QSpinBox()
        """ Spin box to specify minimum acceptable reach duration for filtering purposes. """
        self._dur_max_spin = QSpinBox()
        """ Spin box to specify maximum acceptable reach duration for filtering purposes."""
        self._delta_min_spin = QSpinBox()
        """ Spin box to specify minimum delta to "reachMax", for filtering purposes. """
        self._delta_max_spin = QSpinBox()
        """ Spin box to specify maximum delta to "reachMax", for filtering purposes. """
        for spin in [self._dur_min_spin, self._dur_max_spin, self._delta_min_spin, self._delta_max_spin]:
            spin.setRange(0, 10000)
            spin.setSingleStep(1)

        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        source_row = QGridLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.setHorizontalSpacing(10)
        source_row.addWidget(self._make_label("Source", self._reach_source_combo), 0, 0)
        source_row.addWidget(self._reach_source_combo, 0, 1)
        source_row.addWidget(self._make_label("Result", self._segmentation_result_combo), 0, 2)
        source_row.addWidget(self._segmentation_result_combo, 0, 3)
        source_row.setColumnStretch(1, 1)
        source_row.setColumnStretch(3, 2)
        layout.addLayout(source_row)

        group_box = QGroupBox("Filter reaches")
        filter_layout = QGridLayout()
        filter_layout.setContentsMargins(2, 2, 2, 2)
        filter_layout.setHorizontalSpacing(5)
        filter_layout.setVerticalSpacing(8)
        filter_layout.addWidget(self._make_label("Result", self._result_filter_combo), 0, 0)
        filter_layout.addWidget(self._result_filter_combo, 0, 1)
        filter_layout.addWidget(self._make_label("Hand pos", self._handpos_filter_combo), 1, 0)
        filter_layout.addWidget(self._handpos_filter_combo, 1, 1)
        filter_layout.addWidget(QLabel("Reach Duration:"), 0, 3)
        filter_layout.addWidget(self._dur_min_spin, 0, 4)
        filter_layout.addWidget(QLabel("to"), 0, 5)
        filter_layout.addWidget(self._dur_max_spin, 0, 6)
        filter_layout.addWidget(QLabel("Delta Reach Max:"), 1, 3)
        filter_layout.addWidget(self._delta_min_spin, 1, 4)
        filter_layout.addWidget(QLabel("to"), 1, 5)
        filter_layout.addWidget(self._delta_max_spin, 1, 6)
        filter_layout.setColumnStretch(2, 1)
        group_box.setLayout(filter_layout)
        layout.addWidget(group_box)

        group_box = QGroupBox("Reach metrics options")
        metrics_layout = QGridLayout()
        metrics_layout.setContentsMargins(2, 2, 2, 2)
        metrics_layout.setHorizontalSpacing(5)
        metrics_layout.setVerticalSpacing(8)
        metrics_layout.addWidget(self._make_label("Endpoint", self._endpoint_mode_combo), 0, 0)
        metrics_layout.addWidget(self._endpoint_mode_combo, 0, 1)
        metrics_layout.addWidget(self._make_label("Pellet stop <= (mm/ms)", self._pellet_stop_speed_combo), 0, 3)
        metrics_layout.addWidget(self._pellet_stop_speed_combo, 0, 4)
        metrics_layout.setColumnStretch(2, 1)
        group_box.setLayout(metrics_layout)
        layout.addWidget(group_box)

        layout.addWidget(self._reach_table)

        nav_select_row = QHBoxLayout()
        nav_select_row.addWidget(self._select_all_btn)
        nav_select_row.addWidget(self._deselect_all_btn)
        nav_select_row.addStretch(1)
        nav_select_row.addWidget(self._focus_prev_btn)
        nav_select_row.addWidget(self._focus_next_btn)
        layout.addLayout(nav_select_row)

        panel.setLayout(layout)
        return panel

    @Slot(int, Qt.SortOrder)
    def _on_sort_change(self, col: int, sort_order: Qt.SortOrder):
        """ Prevents sorting on column 0 of the reach metrics table. """
        self._reach_table.horizontalHeader().blockSignals(True)
        if col == 0:
            self._reach_table.sortByColumn(self._last_sort_col, self._last_sort_order)
        else:
            self._last_sort_col = col
            self._last_sort_order = sort_order

        self._reach_table.horizontalHeader().blockSignals(False)

    def _connect_signals(self) -> None:
        self._reach_source_combo.currentTextChanged.connect(self._on_reach_source_changed)
        self._segmentation_result_combo.currentIndexChanged.connect(self._on_segmentation_result_combo_changed)
        self._endpoint_mode_combo.currentIndexChanged.connect(self._on_metrics_changed)
        self._pellet_stop_speed_combo.valueChanged.connect(self._on_metrics_changed)
        self._select_all_btn.clicked.connect(lambda: self._reach_model.set_all_selected(True))
        self._deselect_all_btn.clicked.connect(lambda: self._reach_model.set_all_selected(False))
        self._reach_model.selection_changed.connect(self._on_selection_changed)
        self._reach_model.focus_changed.connect(self._on_focus_reach_changed)

        self._result_filter_combo.currentIndexChanged.connect(self._on_filters_changed)
        self._handpos_filter_combo.currentIndexChanged.connect(self._on_filters_changed)
        self._dur_min_spin.valueChanged.connect(self._on_filters_changed)
        self._dur_max_spin.valueChanged.connect(self._on_filters_changed)
        self._delta_min_spin.valueChanged.connect(self._on_filters_changed)
        self._delta_max_spin.valueChanged.connect(self._on_filters_changed)

        self._plot_mode_combo.currentIndexChanged.connect(self._on_plot_mode_changed)
        self._show_endpoints.checkStateChanged.connect(lambda _: self._on_toggle_show_endpoints())
        self._color_by_result.checkStateChanged.connect(lambda _: self._on_toggle_color_by_result())
        self._show_hand.checkStateChanged.connect(lambda _: self._on_toggle_hand())
        self._show_pellet.checkStateChanged.connect(lambda _: self._on_toggle_pellet())
        self._hand_line_style.currentIndexChanged.connect(lambda _: self._on_line_style_changed(hand=True))
        self._hand_line_width.valueChanged.connect(lambda _: self._on_line_width_changed(hand=True))
        self._hand_color_picker.color_changed.connect(lambda: self._on_line_color_changed(hand=True))
        self._hand_avg.checkStateChanged.connect(lambda _: self._on_toggle_avg_trace(hand=True))
        self._hand_avg_color_picker.color_changed.connect(lambda: self._on_avg_line_color_changed(hand=True))
        self._pellet_line_style.currentIndexChanged.connect(lambda _: self._on_line_style_changed(hand=False))
        self._pellet_line_width.valueChanged.connect(lambda _: self._on_line_width_changed(hand=False))
        self._pellet_color_picker.color_changed.connect(lambda: self._on_line_color_changed(hand=False))
        self._pellet_avg.checkStateChanged.connect(lambda _: self._on_toggle_avg_trace(hand=False))
        self._pellet_avg_color_picker.color_changed.connect(lambda: self._on_avg_line_color_changed(hand=False))
        self._show_grid.checkStateChanged.connect(self._on_toggle_grid)
        self._lock_aspect.checkStateChanged.connect(self._on_toggle_aspect)

        self._playback_enabled.checkStateChanged.connect(self._on_playback_toggled)
        self._forward_btn.clicked.connect(self._on_advance_playback_frame)
        self._back_btn.clicked.connect(self._on_rewind_playback_frame)

        # keyboard shortcuts for playback back/forward buttons, active when view's container has focus.
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Period), self.view_container, self._on_advance_playback_frame)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Comma), self.view_container, self._on_rewind_playback_frame)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.view_container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)   # shortcuts don't work without this!

        self._live_enabled.checkStateChanged.connect(self._on_live_toggled)

        self._fig_export_btn.clicked.connect(self._export_figure)
        self._video_export_btn.clicked.connect(self._export_movie)

        self._reach_table.clicked.connect(self._on_table_clicked)
        self._focus_prev_btn.clicked.connect(lambda: self._jump_focus(-1))
        self._focus_next_btn.clicked.connect(lambda: self._jump_focus(1))

        # self.data_manager.active_workspace_switched.connect(self._reload_all)
        self.data_manager.session_loaded.connect(self._reload_all)
        self.data_manager.session_manager.scorer_changed.connect(self._reload_all)
        self.data_manager.session_manager.segmentation_result_changed.connect(self._on_segmentation_result_changed)
        self.data_manager.session_manager.reaches_changed.connect(self._reload_reaches_only)
        self.data_manager.session_manager.frame_ready.connect(self._on_frame_ready)

    @Slot()
    def _on_toggle_show_endpoints(self) -> None:
        self._traj_plot_2d.show_endpts = self._show_endpoints.isChecked()
        self._refresh_plot()

    @Slot()
    def _on_toggle_color_by_result(self) -> None:
        self._traj_plot_2d.color_by_result = self._color_by_result.isChecked()
        self._refresh_plot()

    @Slot()
    def _on_toggle_hand(self) -> None:
        self._traj_plot_2d.show_hand = self._show_hand.isChecked()
        self._refresh_plot()

    @Slot()
    def _on_toggle_pellet(self) -> None:
        if self._show_pellet.isChecked() and self._pellet_traj is None:
            self._show_pellet.setChecked(False)
            QMessageBox.information(self.view_container, "Pellet trajectory unavailable",
                                    "Pellet trajectory file not found for current scorer.")
            return
        self._traj_plot_2d.show_pellet = self._show_pellet.isChecked()
        self._refresh_plot()

    def _on_line_style_changed(self, hand: bool):
        style = {
            "Solid": Qt.PenStyle.SolidLine, "Dash": Qt.PenStyle.DashLine, "Dot": Qt.PenStyle.DotLine
        }.get((self._hand_line_style if hand else self._pellet_line_style).currentText(), Qt.PenStyle.SolidLine)
        if hand:
            self._traj_plot_2d.hand_line_style = style
        else:
            self._traj_plot_2d.pellet_line_style = style
        self._refresh_plot()

    def _on_line_width_changed(self, hand: bool):
        if hand:
            self._traj_plot_2d.hand_line_width = self._hand_line_width.value()
        else:
            self._traj_plot_2d.pellet_line_width = self._pellet_line_width.value()
        self._refresh_plot()

    def _on_line_color_changed(self, hand: bool):
        if hand:
            self._traj_plot_2d.hand_color = self._hand_color_picker.color
        else:
            self._traj_plot_2d.pellet_color = self._pellet_color_picker.color
        self._refresh_plot()

    def _on_toggle_avg_trace(self, hand: bool):
        if hand:
            self._traj_plot_2d.show_hand_avg = self._hand_avg.isChecked()
        else:
            self._traj_plot_2d.show_pellet_avg = self._pellet_avg.isChecked()
        self._refresh_plot()

    def _on_avg_line_color_changed(self, hand: bool):
        if hand:
            self._traj_plot_2d.hand_avg_color = self._hand_avg_color_picker.color
        else:
            self._traj_plot_2d.pellet_avg_color = self._pellet_avg_color_picker.color
        self._refresh_plot()

    def _on_toggle_grid(self) -> None:
        show = self._show_grid.isChecked()
        self._traj_plot_2d.grid_on = show

    def _on_toggle_aspect(self) -> None:
        locked = self._lock_aspect.isChecked()
        self._traj_plot_2d.setAspectLocked(locked)

    def panels(self) -> List[Tuple[str, QWidget]]:
        return self._panel_widgets.copy()

    @staticmethod
    def _make_label(text: str, buddy: Optional[QWidget] = None) -> QLabel:
        if not text.endswith(":"):
            text = f"{text}:"
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet("font-style: italic; font-weight: 500;")
        if buddy is not None:
            label.setBuddy(buddy)
        return label

    @staticmethod
    def _disable_combo_item(combo: QComboBox, idx: int, tooltip: str) -> None:
        if 0 <= idx < combo.count():
            combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        model = combo.model()
        if isinstance(model, QStandardItemModel):
            item = model.item(idx)
            if item:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

    @Slot()
    def _reload_all(self) -> None:
        self._load_session_info()
        self._load_trajectories()
        self._reload_segmentation_result_combo()
        self._reload_reaches_only()
        self._sync_export_size()
        self._refresh_plot()

    @Slot()
    def _on_segmentation_result_changed(self) -> None:
        self._reload_segmentation_result_combo()
        self._reload_reaches_only()

    def _reload_segmentation_result_combo(self) -> None:
        self._segmentation_result_combo.blockSignals(True)
        self._segmentation_result_combo.clear()
        results = self.data_manager.session_manager.segmentation_results
        curr_idx = self.data_manager.session_manager.current_segmentation_result_index
        if len(results) == 0:
            self._segmentation_result_combo.addItem("No reach results")
            self._segmentation_result_combo.setCurrentIndex(0)
            self._segmentation_result_combo.setEnabled(False)
        else:
            self._segmentation_result_combo.addItems([r.display_name for r in results])
            self._segmentation_result_combo.setCurrentIndex(min(len(results) - 1, max(0, curr_idx)))
            self._segmentation_result_combo.setEnabled(self._reach_source_combo.currentText() == self._SOURCE_SCORED)
        self._segmentation_result_combo.blockSignals(False)

    @Slot()
    def _reload_reaches_only(self) -> None:
        segments = self._get_reach_segments()
        self._traj_plot_2d.rebuild(segments, self._hand_traj, self._pellet_traj)
        self._reach_model.set_segments(segments)
        self._reset_filter_ranges(segments)
        self._recompute_metrics()
        self._update_availability()
        self._refresh_plot()

    def _load_session_info(self) -> None:
        sesh = self.data_manager.current_session_id
        scorer = self.data_manager.session_manager.current_scorer
        sesh_text = str(sesh) if sesh is not None else "-"
        scorer_text = scorer.short_name if scorer is not None else "-"
        self._session_status_label.setText(f"Session: {sesh_text}  |  Scorer: {scorer_text}")

    def _load_trajectories(self) -> None:
        self._hand_traj = np.zeros((0, _NCOLS), dtype=np.float32)
        self._pellet_traj = np.zeros((0, _NCOLS), dtype=np.float32)
        hand, pellet = self.data_manager.session_manager.load_hand_pellet_trajectories_for_current_scorer()
        if hand is not None:
            self._hand_traj = hand
        if pellet is not None:
            self._pellet_traj = pellet

    def _get_reach_segments(self) -> List[RxReachSegment]:
        if self._reach_source_combo.currentText() == self._SOURCE_SCORED:
            return self.data_manager.session_manager.get_scored_reaches()
        return self.data_manager.session_manager.get_curated_reaches()

    def _reset_filter_ranges(self, segments: List[RxReachSegment]) -> None:
        if len(segments) == 0:
            for spin in [self._dur_min_spin, self._dur_max_spin, self._delta_min_spin, self._delta_max_spin]:
                spin.blockSignals(True)
                spin.setValue(0)
                spin.blockSignals(False)
            return

        dur_vals = [s.dur for s in segments]
        delta_vals = [s.max_delta for s in segments]
        max_dur = max(dur_vals)
        max_delta = max(delta_vals)

        for spin, val in [(self._dur_min_spin, 0), (self._dur_max_spin, max_dur),
                          (self._delta_min_spin, 0), (self._delta_max_spin, max_delta)]:
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self._on_filters_changed()

    def _update_availability(self) -> None:
        reasons = list()
        if not self.data_manager.session_manager.session_loaded:
            reasons.append("no session loaded")
        # if self.data_manager.fixed_cam_mode and not self.data_manager.session_manager.calibration_ok:
        #    reasons.append("calibration missing")
        if self._reach_model.rowCount() == 0:
            reasons.append("no reaches available")
        if self._hand_traj.size == 0:
            reasons.append("hand trajectory not available")

        pellet_ok = self._pellet_traj.size > 0
        self._show_pellet.setEnabled(pellet_ok)
        if not pellet_ok:
            self._show_pellet.setChecked(False)
        for widget in [self._endpoint_mode_combo, self._pellet_stop_speed_combo]:
            widget.setEnabled(pellet_ok)

        enabled = len(reasons) == 0
        self._plot_stack.setEnabled(enabled)
        live = self._live_enabled.isChecked()
        is_2d = self._plot_mode_combo.currentText().startswith("2D")
        self._playback_enabled.setEnabled(enabled and not live)
        self._forward_btn.setEnabled(enabled and self._playback_enabled.isChecked() and not live)
        self._back_btn.setEnabled(enabled and self._playback_enabled.isChecked() and not live)
        self._video_export_btn.setEnabled(enabled and is_2d and (not live) and
                                          (self._reach_model.focus_segment is not None))
        self._fig_export_btn.setEnabled(enabled and is_2d)

        msg = "Ready." if enabled else f"Analysis unavailable: {', '.join(reasons)}."
        self._set_status(msg)

    def _set_status(self, msg: str) -> None:
        if msg != self._last_status:
            self._status_label.setText(msg)
            self._last_status = msg

    def _sync_export_size(self) -> None:
        size = self._traj_plot_2d.size()
        if size.width() > 0 and size.height() > 0:
            self._fig_width.setValue(size.width())
            self._fig_height.setValue(size.height())

    @Slot()
    def _on_selection_changed(self) -> None:
        self._update_playback_range()
        self._traj_plot_2d.update_displayed_segments(self._selected_visible_segments())
        self._refresh_plot()

    @Slot()
    def _on_focus_reach_changed(self) -> None:
        self._update_availability()
        self._traj_plot_2d.update_focus_segment(self._reach_model.focus_segment)
        self._refresh_plot()

    @Slot()
    def _on_filters_changed(self) -> None:
        res = self._result_filter_combo.currentData()
        self._reach_proxy.set_result_filter(res if isinstance(res, RxReach) else None)
        pos = self._handpos_filter_combo.currentData()
        self._reach_proxy.set_hand_pos_filter(pos if isinstance(pos, RxHandPos) else None)
        self._reach_proxy.set_dur_range(self._dur_min_spin.value(), self._dur_max_spin.value())
        self._reach_proxy.set_delta_range(self._delta_min_spin.value(), self._delta_max_spin.value())
        self._update_playback_range()
        self._traj_plot_2d.update_displayed_segments(self._selected_visible_segments())
        self._traj_plot_2d.update_focus_segment(self._reach_model.focus_segment)
        self._refresh_plot()

    @Slot()
    def _on_metrics_changed(self) -> None:
        self._recompute_metrics()

    @Slot()
    def _on_reach_source_changed(self) -> None:
        self._reload_segmentation_result_combo()
        # 200ms gives the combo box dropdown time to close before the time-consuming reload occurs (50ms not enough!)
        QTimer.singleShot(200, self._reload_reaches_only)

    @Slot(int)
    def _on_segmentation_result_combo_changed(self, idx: int) -> None:
        self.data_manager.session_manager.change_current_segmentation_result(idx)

    @Slot()
    def _on_plot_mode_changed(self) -> None:
        self._plot_stack.setCurrentIndex(self._plot_mode_combo.currentIndex())
        self._refresh_plot()

    @Slot()
    def _on_playback_toggled(self) -> None:
        enabled = self._playback_enabled.isChecked()
        self._update_playback_range()
        self._playback_frame_readout.setText(f"{self._playback_frame}")
        for w in [self._back_btn, self._forward_btn]:
            w.setEnabled(enabled)
        self._traj_plot_2d.update_display_to_frame(self._playback_frame if enabled else None)
        self._refresh_plot()

    @Slot()
    def _on_advance_playback_frame(self) -> None:
        if (not self._playback_enabled.isChecked()) or (self._playback_frame >= self._playback_dur):
            return
        self._playback_frame += 1
        self._playback_frame_readout.setText(f"{self._playback_frame}")
        self._traj_plot_2d.update_display_to_frame(self._playback_frame)
        self._refresh_plot()

    @Slot()
    def _on_rewind_playback_frame(self) -> None:
        if (not self._playback_enabled.isChecked()) or (self._playback_frame <= 0):
            return
        self._playback_frame -= 1
        self._playback_frame_readout.setText(f"{self._playback_frame}")
        self._traj_plot_2d.update_display_to_frame(self._playback_frame)
        self._refresh_plot()

    @Slot()
    def _on_live_toggled(self) -> None:
        refreshed = False
        if self._live_enabled.isChecked():
            if self._playback_enabled.isChecked():
                self._playback_enabled.setChecked(False)
                self._on_playback_toggled()
                refreshed = True
        self._update_availability()
        if not refreshed:
            self._refresh_plot()

        curr_frame = self.data_manager.session_manager.current_frame_num
        curr_frame = None if (curr_frame < 0 or not self._live_enabled.isChecked()) else curr_frame
        self._traj_plot_2d.live_frame = curr_frame

    @Slot()
    def _on_frame_ready(self) -> None:
        curr = self.data_manager.session_manager.current_frame_num
        if self._reach_source_combo.currentText() == self._SOURCE_SCORED:
            curr_seg = self.data_manager.session_manager.get_scored_reach_containing_frame(curr)
        else:
            curr_seg = self.data_manager.session_manager.get_curated_reach_containing_frame(curr)
        focus_row = self._reach_model.row_for_segment(curr_seg)
        if focus_row != self._reach_model.focus_row:
            self._reach_model.set_focus_row(focus_row)   # changing the focus row also refreshes the plot!
        elif self._live_enabled.isChecked():
            self._traj_plot_2d.live_frame = None if curr < 0 else curr
            self._refresh_plot()

    @Slot(QModelIndex)
    def _on_table_clicked(self, idx: QModelIndex) -> None:
        if not idx.isValid():
            return
        src_idx = self._reach_proxy.mapToSource(idx)
        if not src_idx.isValid():
            return
        if src_idx.column() == 0:
            self._reach_model.toggle_selected_state_of_row(src_idx.row())
        else:
            self._reach_model.set_focus_row(src_idx.row())

    def _update_playback_range(self) -> None:
        segments = self._selected_visible_segments()
        self._playback_frame = 0
        self._playback_dur = max([s.dur for s in segments], default=0)

    def _recompute_metrics(self) -> None:
        metrics = self._compute_reach_metrics(self._reach_model.segments())
        self._reach_model.set_metrics(metrics)

    def _compute_reach_metrics(self, segments: List[RxReachSegment]) -> Dict[RxReachSegment, _ReachMetrics]:
        """
        Compute metrics for each of the specified reach segments.
        :param segments: List of reach segments.
        :return: Dictionary of reach segments and their respective metrics. If hand or pellet trajectory is not
            available, all metrics are set to None.
        """
        metrics: Dict[RxReachSegment, _ReachMetrics] = {seg: _ReachMetrics() for seg in segments}
        if self._hand_traj.size == 0 or self._pellet_traj.size == 0 or len(segments) == 0:
            return metrics

        max_len = min(self._hand_traj.shape[0], self._pellet_traj.shape[0])
        if max_len <= 0:
            return metrics

        pellet_speed = self._pellet_traj[:, _COL_SPEED]
        stop_thresh = abs(self._pellet_stop_speed_combo.value())

        raw_diffs: Dict[RxReachSegment, np.ndarray] = {}
        for seg in segments:
            start = max(0, seg.frame)
            end = min(max_len - 1, seg.end_frame)
            if end <= start:
                continue
            pellet_ref = self._pellet_reference_for(seg, pellet_speed, stop_thresh, max_len)
            if pellet_ref is None:
                continue
            endpoint = self._endpoint_frame_for(seg, pellet_ref, max_len)
            if endpoint is None:
                continue
            hand_x = self._hand_traj[endpoint, _COL_X]
            hand_y = self._hand_traj[endpoint, _COL_Y]
            hand_z = self._hand_traj[endpoint, _COL_Z]
            hand_xyz = np.array([hand_x, hand_y, hand_z], dtype=np.float32)
            raw_diffs[seg] = hand_xyz - pellet_ref

        if len(raw_diffs) == 0:
            return metrics

        grabbed = [vec for seg, vec in raw_diffs.items() if seg.result == RxReach.GRABBED]
        mean_offset = np.mean(grabbed, axis=0) if len(grabbed) > 0 else np.zeros(3, dtype=np.float32)

        corrected: Dict[RxReachSegment, np.ndarray] = {}
        for seg, vec in raw_diffs.items():
            corrected_vec = vec - mean_offset
            corrected[seg] = corrected_vec
            m = metrics[seg]
            m.dx = float(corrected_vec[0])
            m.dy = float(corrected_vec[1])
            m.dz = float(corrected_vec[2])
            m.dist = float(np.linalg.norm(corrected_vec))

        return metrics

    def _pellet_reference_for(self, seg: RxReachSegment, pellet_speed: np.ndarray,
                              stop_thresh: float, max_len: int) -> Optional[np.ndarray]:
        start = max(0, seg.frame - 100)
        end = min(max_len - 1, seg.frame)
        if end < start:
            return None
        speed_window = pellet_speed[start:end + 1]
        stable = np.abs(speed_window) <= stop_thresh
        if np.any(stable):
            stable_idx = np.where(stable)[0]
            first_stable = stable_idx[0]
            use_mask = stable & (np.arange(len(stable)) >= first_stable)
            frames = np.arange(start, end + 1)[use_mask]
        else:
            fallback_start = max(0, seg.frame - 20)
            fallback_end = min(max_len - 1, seg.frame)
            frames = np.arange(fallback_start, fallback_end + 1)
        if frames.size == 0:
            return None

        x = self._pellet_traj[frames, _COL_X]
        y = self._pellet_traj[frames, _COL_Y]
        z = self._pellet_traj[frames, _COL_Z]
        return np.array([np.nanmean(x), np.nanmean(y), np.nanmean(z)], dtype=np.float32)

    def _endpoint_frame_for(self, seg: RxReachSegment, pellet_ref: np.ndarray, max_len: int) -> Optional[int]:
        mode = self._endpoint_mode_combo.currentData()
        if mode == "max":
            return max(0, min(max_len - 1, seg.max_frame))
        if mode == "end":
            return max(0, min(max_len - 1, seg.end_frame))
        start = max(0, seg.frame)
        end = min(max_len - 1, seg.end_frame)
        if end <= start:
            return None

        hand_xyz = self._hand_traj[start:end + 1][:, :_NCOLS-1]
        dists = np.linalg.norm(hand_xyz - pellet_ref, axis=1)
        if dists.size == 0:
            return None
        if np.all(np.isnan(dists)):
            return None
        min_idx = int(np.nanargmin(dists))
        return start + min_idx

    def _selected_visible_segments(self) -> List[RxReachSegment]:
        if isinstance(self._export_segments_override, list):
            return self._export_segments_override
        out: List[RxReachSegment] = list()
        for row in range(self._reach_proxy.rowCount()):
            proxy_idx = self._reach_proxy.index(row, 0)
            src_idx = self._reach_proxy.mapToSource(proxy_idx)
            if self._reach_model.is_selected(src_idx.row()):
                seg = self._reach_model.segment_at(src_idx.row())
                if isinstance(seg, RxReachSegment):
                    out.append(seg)
        return out

    def _get_focus_segment(self) -> Optional[RxReachSegment]:
        if not self.data_manager.session_manager.session_loaded:
            return None
        curr = int(self.data_manager.session_manager.current_frame_num)
        if curr < 0:
            return None
        if self._reach_source_combo.currentText() == self._SOURCE_SCORED:
            segments = self.data_manager.session_manager.get_all_scored_reaches_in(curr, curr)
        else:
            segments = self.data_manager.session_manager.get_all_curated_reaches_in(curr, curr)
        if len(segments) != 1:
            return None
        return segments[0]

    def _jump_focus(self, delta: int) -> None:
        segments = self._reach_model.segments()
        if len(segments) == 0:
            return
        row = self._reach_model.focus_row
        if row < 0:
            row = 0
        new_idx = max(0, min(len(segments) - 1, row + delta))
        target_seg = segments[new_idx]
        self.data_manager.session_manager.go_to_frame(target_seg.frame)

    def _get_live_segment(self) -> Tuple[Optional[RxReachSegment], Optional[str], Optional[int]]:
        if not self.data_manager.session_manager.session_loaded:
            return None, "Live: no session loaded.", None
        curr = int(self.data_manager.session_manager.current_frame_num)
        if curr < 0:
            return None, "Live: no current frame.", curr

        if self._reach_source_combo.currentText() == self._SOURCE_SCORED:
            segments = self.data_manager.session_manager.get_all_scored_reaches_in(curr, curr)
        else:
            segments = self.data_manager.session_manager.get_all_curated_reaches_in(curr, curr)

        if len(segments) == 0:
            _LOG.debug(f"Live: no reach at frame {curr}.")
            return None, "Live: no reach at current frame.", curr
        if len(segments) > 1:
            _LOG.warning(f"Live: overlapping reaches at frame {curr} (count={len(segments)}).")
            return None, f"Live: overlapping reaches at frame {curr}.", curr
        return segments[0], None, curr

    def _refresh_plot(self) -> None:
        if self._plot_mode_combo.currentIndex() == 1 and _HAS_GL:
            self._refresh_plot_3d()
        # NOTE: No action taken if the TrajPlot2D is currently shown! It is refreshed more efficiently as
        # properties are changed.

    def _refresh_plot_3d(self) -> None:
        if not _HAS_GL:
            return
        self._plot_3d.clear()
        if self._hand_traj.size == 0:
            return
        if not self._show_hand.isChecked() and not self._show_pellet.isChecked():
            return

        live = self._live_enabled.isChecked()
        live_curr: Optional[int] = None
        if live:
            seg, err, curr = self._get_live_segment()
            if err is not None:
                self._set_status(err)
                return
            segments = [seg] if seg is not None else []
            live_curr = curr
            if len(segments) == 0:
                self._set_status("Live: no reach selected.")
                return
            self._set_status("Live: ready.")
        else:
            segments = self._selected_visible_segments()
            if len(segments) == 0:
                self._set_status("No reaches selected.")
                return
            self._set_status("Ready.")

        max_len = self._hand_traj.shape[0]
        playback = self._playback_enabled.isChecked() and not live
        t = self._playback_frame if playback else None

        for seg in segments:
            start = max(0, seg.frame)
            end = min(max_len - 1, seg.end_frame)
            if playback and t is not None:
                end = min(end, start + t)
            if end <= start:
                continue

            if self._show_hand.isChecked():
                x = self._hand_traj[start:end + 1, _COL_X]
                y = self._hand_traj[start:end + 1, _COL_Y]
                z = self._hand_traj[start:end + 1, _COL_Z]
                pts = np.vstack([x, y, z]).T
                self._plot_3d.addItem(gl.GLLinePlotItem(pos=pts, color=self._gl_color(seg, False), width=1.5))
                if live and isinstance(live_curr, int):
                    tracer_end = min(end, live_curr)
                    if tracer_end >= start:
                        x_t = self._hand_traj[start:tracer_end + 1, _COL_X]
                        y_t = self._hand_traj[start:tracer_end + 1, _COL_Y]
                        z_t = self._hand_traj[start:tracer_end + 1, _COL_Z]
                        pts_t = np.vstack([x_t, y_t, z_t]).T
                        color = self._gl_tracer_color(seg, False)
                        self._plot_3d.addItem(gl.GLLinePlotItem(pos=pts_t, color=color, width=1.0))
                        if hasattr(gl, "GLScatterPlotItem"):
                            self._plot_3d.addItem(gl.GLScatterPlotItem(pos=pts_t[-1:], color=color, size=8))

            if self._show_pellet.isChecked() and self._pellet_traj.size > 0:
                x = self._pellet_traj[start:end + 1, _COL_X]
                y = self._pellet_traj[start:end + 1, _COL_Y]
                z = self._pellet_traj[start:end + 1, _COL_Z]
                pts = np.vstack([x, y, z]).T
                self._plot_3d.addItem(gl.GLLinePlotItem(pos=pts, color=self._gl_color(seg, True), width=1.0))
                if live and isinstance(live_curr, int):
                    tracer_end = min(end, live_curr)
                    if tracer_end >= start:
                        x_t = self._pellet_traj[start:tracer_end + 1, _COL_X]
                        y_t = self._pellet_traj[start:tracer_end + 1, _COL_Y]
                        z_t = self._pellet_traj[start:tracer_end + 1, _COL_Z]
                        pts_t = np.vstack([x_t, y_t, z_t]).T
                        color = self._gl_tracer_color(seg, True)
                        self._plot_3d.addItem(gl.GLLinePlotItem(pos=pts_t, color=color, width=0.8))
                        if hasattr(gl, "GLScatterPlotItem"):
                            self._plot_3d.addItem(gl.GLScatterPlotItem(pos=pts_t[-1:], color=color, size=6))

        if (not live) and self._show_hand.isChecked() and self._hand_avg.isChecked():
            avg = self._average_trajectory(self._hand_traj, segments, _COL_X, _COL_Y, playback, t, z_col=_COL_Z)
            if avg is not None:
                x_avg, y_avg, z_avg = avg
                pts = np.vstack([x_avg, y_avg, z_avg]).T
                self._plot_3d.addItem(gl.GLLinePlotItem(pos=pts, color=self._avg_gl_color(False), width=2.0))

        if ((not live) and self._show_pellet.isChecked() and (self._pellet_traj is not None) and
                self._pellet_avg.isChecked()):
            avg = self._average_trajectory(self._pellet_traj, segments, _COL_X, _COL_Y, playback, t, z_col=_COL_Z)
            if avg is not None:
                x_avg, y_avg, z_avg = avg
                pts = np.vstack([x_avg, y_avg, z_avg]).T
                self._plot_3d.addItem(gl.GLLinePlotItem(pos=pts, color=self._avg_gl_color(True), width=1.5))

    def _pen_for(self, seg: RxReachSegment, pellet: bool) -> pg.QtGui.QPen:
        if pellet:
            color = self._pellet_color_picker.color
        else:
            if self._color_by_result.isChecked():
                color = QColor(seg.result.ui_color)
            else:
                color = self._hand_color_picker.color
        style = {
            "Solid": Qt.PenStyle.SolidLine,
            "Dash": Qt.PenStyle.DashLine,
            "Dot": Qt.PenStyle.DotLine
        }.get((self._pellet_line_style if pellet else self._hand_line_style).currentText(),
              Qt.PenStyle.SolidLine)
        width = (self._pellet_line_width if pellet else self._hand_line_width).value()
        return pg.mkPen(color=color, width=width, style=style)

    def _tracer_pen_for(self, seg: RxReachSegment, pellet: bool) -> pg.QtGui.QPen:
        base = self._pen_for(seg, pellet)
        color = QColor(base.color())
        color.setAlpha(min(120, color.alpha()))
        width = max(1.0, base.widthF() * 0.5)
        return pg.mkPen(color=color, width=width, style=base.style())

    def _gl_color(self, seg: RxReachSegment, pellet: bool) -> Tuple[float, float, float, float]:
        if pellet:
            color = self._pellet_color_picker.color
        else:
            if self._color_by_result.isChecked():
                color = QColor(seg.result.ui_color)
            else:
                color = self._hand_color_picker.color
        return color.redF(), color.greenF(), color.blueF(), color.alphaF()

    def _gl_tracer_color(self, seg: RxReachSegment, pellet: bool) -> Tuple[float, float, float, float]:
        r, g, b, a = self._gl_color(seg, pellet)
        return r, g, b, min(a, 0.4)

    def _avg_pen(self, pellet: bool) -> pg.QtGui.QPen:
        if pellet:
            color = self._pellet_avg_color_picker.color
            color.setAlpha(min(255, color.alpha() + 30))
            width = self._pellet_line_width.value() + 2
        else:
            color = self._hand_avg_color_picker.color
            color.setAlpha(min(255, color.alpha() + 30))
            width = self._hand_line_width.value() + 2
        return pg.mkPen(color=color, width=width, style=Qt.PenStyle.SolidLine)

    def _avg_gl_color(self, pellet: bool) -> Tuple[float, float, float, float]:
        if pellet:
            color = self._pellet_avg_color_picker.color
            color.setAlpha(min(255, color.alpha() + 30))
        else:
            color = self._hand_avg_color_picker.color
            color.setAlpha(min(255, color.alpha() + 30))
        return color.redF(), color.greenF(), color.blueF(), color.alphaF()

    @staticmethod
    def _average_trajectory(traj: np.ndarray, segments: List[RxReachSegment], x_col: int, y_col: int,
                            playback: bool, t: Optional[int], z_col: Optional[int] = None):
        if len(segments) == 0:
            return None
        max_dur = max(s.dur for s in segments)
        if playback and isinstance(t, int):
            max_dur = min(max_dur, t)
        if max_dur <= 0:
            return None

        n = max_dur + 1
        if z_col is None:
            data_x = np.full((n, len(segments)), np.nan, dtype=np.float32)
            data_y = np.full((n, len(segments)), np.nan, dtype=np.float32)
            for i, seg in enumerate(segments):
                start = max(0, seg.frame)
                end = min(traj.shape[0] - 1, seg.end_frame, start + max_dur)
                if end <= start:
                    continue
                span = end - start + 1
                data_x[:span, i] = traj[start:end + 1, x_col]
                data_y[:span, i] = traj[start:end + 1, y_col]
            return np.nanmean(data_x, axis=1), np.nanmean(data_y, axis=1)
        else:
            data_x = np.full((n, len(segments)), np.nan, dtype=np.float32)
            data_y = np.full((n, len(segments)), np.nan, dtype=np.float32)
            data_z = np.full((n, len(segments)), np.nan, dtype=np.float32)
            for i, seg in enumerate(segments):
                start = max(0, seg.frame)
                end = min(traj.shape[0] - 1, seg.end_frame, start + max_dur)
                if end <= start:
                    continue
                span = end - start + 1
                data_x[:span, i] = traj[start:end + 1, x_col]
                data_y[:span, i] = traj[start:end + 1, y_col]
                data_z[:span, i] = traj[start:end + 1, z_col]
            return np.nanmean(data_x, axis=1), np.nanmean(data_y, axis=1), np.nanmean(data_z, axis=1)

    def _export_figure(self) -> None:
        if self._hand_traj.size == 0:
            QMessageBox.information(self.view_container, "Export unavailable", "No trajectory data available.")
            return
        fmt = self._fig_format.currentText()
        dst, _ = QFileDialog.getSaveFileName(self.view_container, "Export figure", f"trajectory.{fmt}",
                                             f"*.{fmt}")
        if not dst:
            return

        w = self._fig_width.value()
        h = self._fig_height.value()
        gain = self._fig_gain.value()
        gamma = self._fig_gamma.value()
        try:
            if fmt == "png":
                if gain == 1.0 and gamma == 1.0:
                    from pyqtgraph.exporters import ImageExporter
                    exporter = ImageExporter(self._traj_plot_2d.plotItem)
                    exporter.params['width'] = w
                    exporter.params['height'] = h
                    exporter.export(dst)
                else:
                    self._export_processed_png(dst, w, h, gain, gamma)
                return
            if fmt == "svg":
                from pyqtgraph.exporters import SVGExporter
                exporter = SVGExporter(self._traj_plot_2d.plotItem)
                exporter.params['width'] = w
                exporter.params['height'] = h
                exporter.export(dst)
                return
            if fmt in ["pdf", "eps"]:
                self._export_vector_with_qprinter(dst, fmt)
                return
        except Exception as exc:
            QMessageBox.warning(self.view_container, "Export failed", f"Export failed: {exc}")

    def _export_vector_with_qprinter(self, dst: str, fmt: str) -> None:
        try:
            from PySide6.QtPrintSupport import QPrinter
            from PySide6.QtGui import QPainter
        except Exception as exc:
            QMessageBox.warning(self.view_container, "Export failed", f"Printer backend unavailable: {exc}")
            return

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        if fmt == "pdf":
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        else:
            if not hasattr(QPrinter.OutputFormat, "PostScriptFormat"):
                QMessageBox.warning(self.view_container, "Export unavailable",
                                    "EPS export not supported by this Qt build.")
                return
            # noinspection PyUnresolvedReferences
            printer.setOutputFormat(QPrinter.OutputFormat.PostScriptFormat)

        printer.setOutputFileName(dst)
        printer.setResolution(self._fig_dpi.value())
        # NOTE: Exception thrown upon calling setPageMargins()
        # printer.setPageMargins(QMargins(), QPrinter.Unit.Millimeter)

        painter = QPainter(printer)
        try:
            self._traj_plot_2d.plotItem.scene().render(painter)
        finally:
            painter.end()

    def _export_movie(self) -> None:
        seg = self._reach_model.focus_segment
        if (seg is None) or not RxReachSegment.is_valid(seg):
            QMessageBox.information(self.view_container, "Export unavailable", "Focus reach unspecified or invalid.")
            return
        if self._hand_traj.size == 0:
            QMessageBox.information(self.view_container, "Export unavailable", "No trajectory data available.")
            return
        dst, _ = QFileDialog.getSaveFileName(self.view_container, "Export playback (MP4)", "trajectory.mp4", "*.mp4")
        if not dst:
            return

        fps = self._video_fps.value()
        gain = self._video_gain.value()
        gamma = self._video_gamma.value()

        writer = None
        writer_size: Optional[Tuple[int, int]] = None

        # remember playback state so we can restore after export
        playback_was_enabled = self._playback_enabled.isChecked()
        save_playback_frame = self._playback_frame
        save_playback_dur = self._playback_dur

        self._traj_plot_2d.update_displayed_segments([seg])
        self._playback_enabled.setChecked(True)
        max_dur = seg.dur
        self._playback_frame = 0
        self._playback_dur = max_dur

        # for a smooth-looking movie, need to freeze the axis ranges of the 2D plot widget
        max_len = self._hand_traj.shape[0]
        start = max(0, seg.frame)
        end = min(max_len - 1, seg.end_frame)
        if not self._show_pellet.isChecked():
            y = self._hand_traj[start:end + 1, _COL_Y]
            z = self._hand_traj[start:end + 1, _COL_Z]
        else:
            y = np.concatenate(
                (self._hand_traj[start:end + 1, _COL_Y], self._pellet_traj[start:end + 1, _COL_Y]))
            z = np.concatenate(
                (self._hand_traj[start:end + 1, _COL_Z], self._pellet_traj[start:end + 1, _COL_Z]))
        min_y = int(np.min(y))
        max_y = int(np.max(y))
        min_z = int(np.min(z))
        max_z = int(np.max(z))
        self._traj_plot_2d.setXRange(min_y-1, max_y+1)
        self._traj_plot_2d.setYRange(min_z-1, max_z+1)

        try:
            for t in range(0, max_dur + 1):
                self._playback_frame = t
                self._traj_plot_2d.update_display_to_frame(self._playback_frame)
                img = self._traj_plot_2d.grab().toImage()
                img = img.convertToFormat(img.Format.Format_RGB888)
                buf = img.bits()
                frame = np.frombuffer(buf, dtype=np.uint8, count=img.sizeInBytes())
                frame = frame.reshape((img.height(), img.width(), 3))
                frame = self._apply_gain_gamma(frame, gain, gamma)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if writer is None:
                    frame, writer_size = self._ensure_even_frame(frame)
                    writer, _ = self._init_video_writer(dst, fps, writer_size)
                    if writer is None:
                        self._handle_video_fallback(dst, max_dur, fps, gain, gamma)
                        return
                if writer_size is not None:
                    frame = self._ensure_frame_size(frame, writer_size)
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
                self._maybe_fallback_on_missing_mp4(dst, max_dur, fps, gain, gamma)
            self._playback_enabled.setChecked(playback_was_enabled)
            self._playback_frame, self._playback_dur = save_playback_frame, save_playback_dur
            self._export_segments_override = None
            self._traj_plot_2d.update_displayed_segments(self._selected_visible_segments())
            self._traj_plot_2d.update_display_to_frame(self._playback_frame if playback_was_enabled else None)
            self._traj_plot_2d.getViewBox().autoRange()  # reenable autoranging of axes

    def _handle_video_fallback(self, dst: str, max_dur: int, _fps: int, gain: float, gamma: float) -> None:
        """
        Handler called to export playback of a reach trajectory as a sequnce of PNG files (one per frame).

        :param dst: The destination file, which should have a ".mp4" extension. That is removed to form the destination
            directory for the PNG files comprising the PNG sequence.
        :param max_dur: The number of frames to write.
        :param _fps: Unused. Kept in case we add support for additional fallback options when MP4 is unavailable.
        :param gain: Video color gain.
        :param gamma: Video color gamma.
        """
        choice = self._video_fallback.currentText()
        if choice != "PNG sequence":
            QMessageBox.warning(self.view_container, "Export failed",
                                "MP4 export failed (codec unavailable). No fallback selected.")
            return

        out_dir = Path(dst).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        playback_was_enabled = self._playback_enabled.isChecked()
        save_playback_frame, save_playback_dur = self._playback_frame, self._playback_dur
        self._playback_enabled.setChecked(True)
        try:
            for t in range(0, max_dur + 1):
                self._playback_frame = t
                self._traj_plot_2d.update_display_to_frame(self._playback_frame)
                file_path = Path(out_dir, f"frame_{t:05d}.png")
                self._export_processed_png(str(file_path), 0, 0, gain, gamma)
        finally:
            self._playback_enabled.setChecked(playback_was_enabled)
            self._playback_frame, self._playback_dur = save_playback_frame, save_playback_dur
            self._traj_plot_2d.update_display_to_frame(self._playback_frame if playback_was_enabled else None)
        QMessageBox.information(self.view_container, "Export complete",
                                f"Saved PNG sequence to: {str(out_dir)}")

    @staticmethod
    def _reach_label(seg: RxReachSegment) -> str:
        return f"Start {seg.frame} | Dur {seg.dur} | Max {seg.max_delta} | {seg.result.short_name}"

    @staticmethod
    def _apply_gain_gamma(frame: np.ndarray, gain: float, gamma: float) -> np.ndarray:
        if gain == 1.0 and gamma == 1.0:
            return frame
        data = frame.astype(np.float32) / 255.0
        data = np.clip(data * gain, 0.0, 1.0)
        if gamma != 1.0:
            data = np.power(data, 1.0 / gamma)
        return (data * 255.0).astype(np.uint8)

    def _export_processed_png(self, dst: str, width: int, height: int, gain: float, gamma: float) -> None:
        img = self._traj_plot_2d.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        if width > 0 and height > 0:
            img = img.scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        buf = img.bits()
        frame = np.frombuffer(buf, dtype=np.uint8, count=img.sizeInBytes())
        frame = frame.reshape((img.height(), img.width(), 3))
        frame = self._apply_gain_gamma(frame, gain, gamma)
        out_img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_RGB888)
        out_img.copy().save(dst)

    @staticmethod
    def _init_video_writer(dst: str, fps: int, size: Tuple[int, int]):
        for tag in ["mp4v", "avc1", "H264", "X264"]:
            # noinspection PyUnresolvedReferences
            fourcc = cv2.VideoWriter_fourcc(*tag)
            writer = cv2.VideoWriter(dst, fourcc, fps, size)
            if writer.isOpened():
                _LOG.info(f"MP4 export using fourcc '{tag}' at {size[0]}x{size[1]}.")
                return writer, (tag, size)
            writer.release()
        _LOG.warning("MP4 export failed to open a VideoWriter for available codecs.")
        return None, None

    @staticmethod
    def _ensure_even_frame(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = frame.shape[:2]
        new_w = w - (w % 2)
        new_h = h - (h % 2)
        if new_w != w or new_h != h:
            frame = cv2.resize(frame, (new_w, new_h))
        return frame, (new_w, new_h)

    @staticmethod
    def _ensure_frame_size(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != size:
            frame = cv2.resize(frame, size)
        return frame

    def _maybe_fallback_on_missing_mp4(self, dst: str, max_dur: int, fps: int, gain: float, gamma: float) -> None:
        out = Path(dst)
        if out.is_file() and out.stat().st_size > 0:
            return
        _LOG.warning(f"MP4 export did not create output: {dst}")
        if self._video_fallback.currentText() == "PNG sequence":
            self._handle_video_fallback(dst, max_dur, fps, gain, gamma)
        else:
            QMessageBox.warning(self.view_container, "Export failed",
                                "MP4 export did not produce a file. Select PNG sequence fallback or check codecs.")


class AnalysisWindow(QMainWindow):
    """
    The trajectory analysis window, a secondary window in ReachX that houses ``AnalysisView``.
    """
    def __init__(self, parent: QMainWindow, data_mgr: DataManager, on_close: Callable[[], None]):
        """
        Construct and configure the trajectory analysis window.
        :param parent: Parent window -- this must be the ReachX main application window.
        :param data_mgr: The ReachX data manager, needed to configure the embedded ``AnalysisView``.
        :param on_close: If this is a callable, it will be invoked when the trajectory analysis window is closed.
            Use this mechanism to update application state accordingly (eg, the main window menu bar item
            by which the trajectory analysis window is opened).
        """
        super().__init__(parent)
        self._on_close: Callable[[], None] = on_close
        """ If callable, this method is invoked when the trajectory analysis window is closed. """

        self._analysis_view = AnalysisView(data_mgr, self.statusBar())
        """ The trajectory analysis view housed in this window. The view controls contents of window's status bar. """

        self.setWindowTitle(self._analysis_view.title)
        self.setCentralWidget(self._analysis_view.view_container)

        # put the control panels in docking widgets around the trajectory analysis view (central widget).
        self.setDockNestingEnabled(True)
        for title, widget in self._analysis_view.panels():
            dock = QDockWidget(title, self)
            dock.setStyleSheet("""
                QDockWidget::title {
                    background: lightsteelblue;
                    text-align: center;
                    border-bottom: 1px solid steelblue
                }
            """)
            dock.setObjectName(f"{title}-ANALYSIS-DOCK")
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
            dock.setFloating(False)
            dock.setWidget(widget)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.setMinimumSize(800, 600)

    def closeEvent(self, event: QCloseEvent) -> None:
        if callable(self._on_close):
            self._on_close()
        super().closeEvent(event)


class CompactColorPicker(QPushButton):
    """
    A pushbutton color picker that raises a dialog to choose the color.
    """
    color_changed = Signal()
    """ Signals that a different color has been selected in the color picker. """
    _color_dlg: Optional[QColorDialog] = None
    """ Modal color picker dialog shared by all instances, created on first use. """

    def __init__(self, initial_color="white", parent=None):
        super().__init__("", parent)
        self.setFixedWidth(30)  # Make it compact
        self.setFixedHeight(20)
        self._color = QColor(initial_color)
        self._update_style()
        self.clicked.connect(self._choose_color)

    @property
    def color(self) -> QColor:
        return self._color

    def _update_style(self):
        # Set background color and border radius for a modern look
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color.name(format=QColor.NameFormat.HexArgb)};
                border: 1px solid gray;
                border-radius: 4px;
            }}
            QPushButton:pressed {{
                background-color: {self._color.darker(110).name()};
            }}
        """)

    @Slot()
    def _choose_color(self):
        # create the shared color picker dialog on first use
        if self._color_dlg is None:
            self._color_dlg = QColorDialog(self)
            options = QColorDialog.ColorDialogOption.ShowAlphaChannel
            # MacOS ignores attempts to move its native color dialog, so don't use it.
            if sys.platform == "darwin":
                options |= QColorDialog.ColorDialogOption.DontUseNativeDialog
            self._color_dlg.setOptions(options)
            self._color_dlg.setWindowTitle("Select a color")
        elif self._color_dlg.parent() != self:
            self._color_dlg.setParent(self)

        # move dialog so that its level with and just to right of invoking color picker
        # TODO: ISSSUE - This does not work consistently on MacOS. The first 2 times OK, thereafter, the
        #   dialog appears 56 pixels below where it should. What about on Linux/Windows?
        button_pos = self.mapToGlobal(QPoint(0, 0))
        self._color_dlg.move(button_pos.x() + self.width() + 10, button_pos.y())

        self._color_dlg.setCurrentColor(self._color)
        if self._color_dlg.exec():
            color = self._color_dlg.currentColor()
            if color.isValid() and (color != self._color):
                self._color = color
                self._update_style()
                get_application_logger().info("Emitting color_changed!")
                self.color_changed.emit()
