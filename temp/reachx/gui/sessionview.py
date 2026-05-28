from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
from PySide6.QtCore import Slot, QSize, Qt, QAbstractItemModel, QModelIndex, QItemSelectionModel, QItemSelection
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QPushButton, QDialog, QDialogButtonBox, QWidget, \
    QGridLayout, QLabel, QTabWidget, QTableView, QHeaderView, QSpacerItem, QSizePolicy, QTreeView, QHBoxLayout, \
    QFrame, QLineEdit, QCheckBox, QGroupBox, QFileDialog

from reachx.config.app_log import get_application_logger
from reachx.common import RxSessionID, RxEvent, RxReachSegment, RxReachSegControls
from reachx.uicommon import RxIcons
from reachx.data.datamanager import DataManager
from reachx.data.sessionmgr import CamSource, SessionEventList, SessionManager
from reachx.gui.baseview import BaseView
from reachx.modeling.segmentation.registry import SegmentationModelRecord


class SessionView(BaseView):
    """
    This view displays information about the current loaded session and lets the user switch to a different session in
    the current active workspace.
    """
    _NO_SELECTION: str = "<None selected>"
    """ Pushbutton label indicating that no session has been selected. """
    _NONE_FOUND: str = "No sessions found!"
    """ Pushbutton label indicates that no sessions were found in the active workspace. """
    _NO_SCORERS: str = "No analysis results"
    """ Item displayed in combo box selecting the session scorer when no scorers are available. """
    _NO_SEGMENTATION_MODELS: str = "No segmentation models"
    """ Item displayed in combo box selecting the reach segmentation model when no models are available. """

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__('Session', None, data_manager)
        self._session_btn = QPushButton(SessionView._NO_SELECTION)
        """ Push button raises modal dialog to select the current session. Label reflects the current session. """
        self._scorer_cb = QComboBox()
        """ Combo box by which user selects the current scorer for the current session. """
        self._segmentation_result_cb = QComboBox()
        """ Combo box by which user selects the scored-reach result for the current scorer. """
        self._available_segmentation_models: List[SegmentationModelRecord] = list()
        """ Internal cache of externally trained reach segmentation models in the active workspace. """
        self._segmentation_model_cb = QComboBox()
        """ Combo box by which user selects the ML segmentation model to use for this session scorer. """
        self._register_segmentation_model_btn = QPushButton("Register...")
        """ Pressing this button registers an externally trained reach segmentation model folder. """
        self._reach_seg_btn = QPushButton("Find Reaches...")
        """
        Pressing this button initiates background job to find all reach segments for the current session using the
        analysis results from the current scorer.
        """
        self._ml_reach_seg_btn = QPushButton("Find ML Reaches...")
        """
        Pressing this button initiates externally trained model-based reach segmentation for the current scorer.
        """
        self._analyze_btn = QPushButton("Analyze this session")
        """ 
        Pressing this button initiates background job that uses the current ML model to predict body part locations
        per frame in the current session's two required camera videos. 
        """
        self._batch_analyze_btn = QPushButton("Multisession Analysis...")
        """ 
        Pressing this button raises a dialog by which user selects any number of sessions to analyze (detect
        body parts and find reach segments).
        """
        self._tab_widget = QTabWidget()
        """ Tab panel organizes the widgets displaying information for the current session. """
        self._cameras_tab = CameraTab(self)
        """ Info on camera videos recorded during current session. """
        self._events_tab = EventsTab(self)
        """ Events recorded during the current session. """
        self._reaches_tab = ReachesTab(self)
        """ Reach segments defined for the current session. """

        self._session_select_dlg = _SessionSelectDialog(self._session_btn)
        """ Dialog raised to select the session to load. """
        self._analysis_dlg = _AnalysisDlg(self._batch_analyze_btn)
        """ 
        Dialog raised to set up a multisession analysis job or to edit the current reach segmentation control 
        parameters prior to finding the reach segments for the current session.
        """

        self._session_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        self._session_btn.clicked.connect(self.select_session)

        # when no session is loaded, disable the combo box that selects a scorer
        self._scorer_cb.addItem(self._NO_SCORERS)
        self._scorer_cb.setCurrentIndex(0)
        self._scorer_cb.setEnabled(False)
        self._scorer_cb.currentIndexChanged.connect(self._on_scorer_index_changed)

        self._segmentation_result_cb.addItem("No reach results")
        self._segmentation_result_cb.setCurrentIndex(0)
        self._segmentation_result_cb.setEnabled(False)
        self._segmentation_result_cb.currentIndexChanged.connect(self._on_segmentation_result_index_changed)

        self._segmentation_model_cb.addItem(self._NO_SEGMENTATION_MODELS)
        self._segmentation_model_cb.setCurrentIndex(0)
        self._segmentation_model_cb.setEnabled(False)
        self._segmentation_model_cb.currentIndexChanged.connect(self._on_segmentation_model_index_changed)
        self._register_segmentation_model_btn.clicked.connect(self._on_register_segmentation_model)

        self._analyze_btn.clicked.connect(lambda: self._analyze_current_session(True))
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setToolTip("Use currently selected detection model to predict per-frame body part locations "
                                     "for this experiment session")
        self._reach_seg_btn.clicked.connect(lambda: self._analyze_current_session(False))
        self._reach_seg_btn.setEnabled(False)
        self._reach_seg_btn.setToolTip("Perform automated reach segmentation on this session using predicted body "
                                       "part locations from the current scorer")
        self._ml_reach_seg_btn.clicked.connect(self._on_find_ml_reaches)
        self._ml_reach_seg_btn.setEnabled(False)
        self._ml_reach_seg_btn.setToolTip("Perform externally trained model-based reach segmentation for this scorer")
        self._batch_analyze_btn.clicked.connect(self._on_batch_analyze)
        self._batch_analyze_btn.setEnabled(False)
        self._batch_analyze_btn.setToolTip(
            "Analyze (body part detection followed by reach segmentation) multiple sessions at a time")

        self._tab_widget.addTab(self._cameras_tab, "Cameras")
        self._tab_widget.addTab(self._events_tab, "Events")
        self._tab_widget.addTab(self._reaches_tab, "Reaches")

        main_layout = QGridLayout()
        main_layout.setSpacing(1)
        main_layout.addWidget(QLabel("Current Session:"), 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self._session_btn, 0, 2)
        main_layout.addWidget(QLabel("Analyzed By:"), 1, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grp = QHBoxLayout()
        grp.setSpacing(5)
        grp.addWidget(self._scorer_cb, stretch=1)
        grp.addWidget(self._reach_seg_btn)
        main_layout.addLayout(grp, 1, 2)
        main_layout.addWidget(QLabel("Scored Reaches:"), 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self._segmentation_result_cb, 2, 2)
        main_layout.addWidget(QLabel("Segmentation Model:"), 3, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grp = QHBoxLayout()
        grp.setSpacing(5)
        grp.addWidget(self._segmentation_model_cb, stretch=1)
        grp.addWidget(self._register_segmentation_model_btn)
        grp.addWidget(self._ml_reach_seg_btn)
        main_layout.addLayout(grp, 3, 2)
        grp = QHBoxLayout()
        grp.setSpacing(10)
        grp.addWidget(self._analyze_btn)
        grp.addWidget(self._batch_analyze_btn)
        main_layout.addLayout(grp, 4, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(2)
        separator.setMinimumHeight(20)
        main_layout.addWidget(separator, 5, 0, 1, 3)
        main_layout.addWidget(self._tab_widget, 6, 0, 1, 3)
        main_layout.setColumnMinimumWidth(1, 5)
        main_layout.setColumnStretch(2, 1)
        main_layout.setRowStretch(6, 1)
        self.view_container.setLayout(main_layout)

        # connect to relevant signals from DataManager, SessionManager
        self.data_manager.sessions_available.connect(self._refresh_session_btn)
        self.data_manager.active_workspace_switched.connect(self._reload)
        self.data_manager.active_workspace_path_changed.connect(self._reload)
        self.data_manager.session_loaded.connect(self._reload)
        self.data_manager.session_manager.scorer_list_changed.connect(self._reload_scorers)
        self.data_manager.session_manager.scorer_changed.connect(self._reload_segmentation_results)
        self.data_manager.session_manager.segmentation_result_changed.connect(self._reload_segmentation_results)
        self.data_manager.model_loaded.connect(lambda: self._refresh_analysis_buttons())
        self.data_manager.model_manager.iter_loaded.connect(lambda: self._refresh_analysis_buttons())
        self.data_manager.segmentation_model_loaded.connect(self._reload_segmentation_models)

        self._reload()

    @Slot()
    def _reload(self) -> None:
        self._refresh_session_btn()
        self._reload_segmentation_models()
        self._reload_scorers()
        self._reload_segmentation_results()
        for tab in [self._cameras_tab, self._events_tab, self._reaches_tab]:
            tab.reload()
        self._refresh_analysis_buttons()

    @Slot()
    def _refresh_session_btn(self) -> None:
        sesh = self.data_manager.current_session_id
        empty = not self.data_manager.has_recorded_sessions
        self._session_btn.setText(
            SessionView._NONE_FOUND if empty else (SessionView._NO_SELECTION if sesh is None else str(sesh)))

    @Slot()
    def _reload_scorers(self) -> None:
        self._scorer_cb.currentIndexChanged.disconnect(self._on_scorer_index_changed)
        self._scorer_cb.clear()
        scorers = self.data_manager.session_manager.scorers
        curr_idx = self.data_manager.session_manager.current_scorer_index
        if len(scorers) == 0:
            self._scorer_cb.addItem(self._NO_SCORERS)
            self._scorer_cb.setCurrentIndex(0)
            self._scorer_cb.setEnabled(False)
        else:
            self._scorer_cb.addItems([s.short_name for s in scorers])
            self._scorer_cb.setCurrentIndex(curr_idx)
            self._scorer_cb.setEnabled(True)
        self._scorer_cb.currentIndexChanged.connect(self._on_scorer_index_changed)
        self._refresh_analysis_buttons()

    @Slot()
    def _reload_segmentation_models(self) -> None:
        self._available_segmentation_models.clear()
        self._available_segmentation_models.extend(self.data_manager.existing_segmentation_models)
        current = self.data_manager.current_segmentation_model

        self._segmentation_model_cb.blockSignals(True)
        self._segmentation_model_cb.clear()
        if len(self._available_segmentation_models) == 0:
            self._segmentation_model_cb.addItem(self._NO_SEGMENTATION_MODELS)
            self._segmentation_model_cb.setCurrentIndex(0)
            self._segmentation_model_cb.setEnabled(False)
        else:
            self._segmentation_model_cb.addItems(
                [record.display_name for record in self._available_segmentation_models])
            if current in self._available_segmentation_models:
                self._segmentation_model_cb.setCurrentIndex(self._available_segmentation_models.index(current))
            else:
                self._segmentation_model_cb.setCurrentIndex(0)
            self._segmentation_model_cb.setEnabled(True)
        self._segmentation_model_cb.blockSignals(False)
        self._refresh_analysis_buttons()

    @Slot()
    def _reload_segmentation_results(self) -> None:
        self._segmentation_result_cb.blockSignals(True)
        self._segmentation_result_cb.clear()
        results = self.data_manager.session_manager.segmentation_results
        curr_idx = self.data_manager.session_manager.current_segmentation_result_index
        if len(results) == 0:
            self._segmentation_result_cb.addItem("No reach results")
            self._segmentation_result_cb.setCurrentIndex(0)
            self._segmentation_result_cb.setEnabled(False)
        else:
            self._segmentation_result_cb.addItems([r.display_name for r in results])
            self._segmentation_result_cb.setCurrentIndex(min(len(results) - 1, max(0, curr_idx)))
            self._segmentation_result_cb.setEnabled(True)
        self._segmentation_result_cb.blockSignals(False)
        self._refresh_analysis_buttons()

    @Slot()
    def select_session(self) -> None:
        """ Raise a modal dialog by which user can select a session from the current active workspace. """
        sessions = self.data_manager.sessions()
        if len(sessions) == 0:
            return
        selected: Optional[RxSessionID] = (
            self._session_select_dlg.select_session(sessions, self.data_manager.current_session_id))
        if selected is not None:
            self.data_manager.load_session(selected)

    @Slot(int)
    def _on_scorer_index_changed(self, idx: int) -> None:
        """
        Forwards the request to change the scorer to the SessionManager.
        :param idx: The 0-based index of the scorer.
        """
        self.data_manager.session_manager.change_current_scorer(idx)

    @Slot(int)
    def _on_segmentation_result_index_changed(self, idx: int) -> None:
        self.data_manager.session_manager.change_current_segmentation_result(idx)

    @Slot(int)
    def _on_segmentation_model_index_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._available_segmentation_models):
            self.data_manager.load_segmentation_model(self._available_segmentation_models[idx])

    def _refresh_analysis_buttons(self) -> None:
        """ Refresh enable state of the three analysis action buttons based on current application state. """
        trained = self.data_manager.model_manager.is_trained()
        curr_sesh_loaded = self.data_manager.session_manager.session_loaded
        self._analyze_btn.setEnabled(trained and curr_sesh_loaded)
        self._reach_seg_btn.setEnabled(self.data_manager.can_find_reaches_for_current_session())
        self._register_segmentation_model_btn.setEnabled(self.data_manager.active_workspace is not None)
        self._ml_reach_seg_btn.setEnabled(
            self.data_manager.can_find_reaches_for_current_session()
            and self.data_manager.current_segmentation_model is not None)
        self._batch_analyze_btn.setEnabled(trained)

    @Slot()
    def _on_find_ml_reaches(self) -> None:
        self.data_manager.find_ml_reach_segments_for_current_session()

    @Slot()
    def _on_register_segmentation_model(self) -> None:
        model_root = QFileDialog.getExistingDirectory(
            self.view_container,
            "Select segmentation model folder",
            str(self.data_manager.active_workspace.segmentation_model_root.absolute()),
        )
        if len(model_root) > 0 and self.data_manager.register_segmentation_model(Path(model_root)):
            self._reload_segmentation_models()

    @Slot()
    def _on_batch_analyze(self) -> None:
        """
        Raise a dialog by which user selects one or more experiment sessions to analyze with the current model, then
        optionallly use the predicted body part locations to perform reach segmentation for each session.
        """
        if not self.data_manager.model_manager.is_trained():
            get_application_logger().info("Cannot proceed: Current body part detection model untrained.")
            return
        all_sessions = self.data_manager.sessions()
        if len(all_sessions) == 0:
            get_application_logger().info("Cannot proceed: No sessions in workspace")
            return
        selected_sids, seg_ctrls = (
            self._analysis_dlg.select_sessions_to_analyze(all_sessions,
                                                          self.data_manager.active_workspace.reach_seg_ctrls))
        if len(selected_sids) > 0:
            find_reaches = isinstance(seg_ctrls, RxReachSegControls)
            if find_reaches:
                self.data_manager.active_workspace.reach_seg_ctrls = seg_ctrls
            self.data_manager.analyze_sessions(selected_sids, 15, True)

    def _analyze_current_session(self, bp_detect: bool = True) -> None:
        """
        Initiate a long-running background task to perform body part detection OR automated reach segmentation for the
        currently loaded experiment session.

        :param bp_detect: If True, perform body part detection analysis, else automated reach segmentation. Note that
            reach segmentation can only be done if predicted per-frame body part locations are already available for
            the session. Default = True.
        """
        if not self.data_manager.session_manager.session_loaded:
            get_application_logger().info("Cannot proceed: No session to analyze.")
            return
        sesh_ids: List[RxSessionID] = [self.data_manager.session_manager.id]
        if bp_detect:
            self.data_manager.analyze_sessions(sesh_ids)
        else:
            ctrls = self._analysis_dlg.edit_reach_seg_controls(self.data_manager.active_workspace.reach_seg_ctrls)
            if ctrls is None:
                return
            else:
                # update reach seg control param values for active workspace before starting the task!
                self.data_manager.active_workspace.reach_seg_ctrls = ctrls
                self.data_manager.find_reach_segments_for_current_session()


class _CamInfoModel(QAbstractItemModel):
    """
    A table model exposing information about all recorded camera videos in the curreently loaded session.

    USAGE:
    - Pass the singleton SessionManager to the constructor, and set the model in a QTableView.
    - Call reload() whenever a different session is loaded.
    """
    _COL_HEADERS = ["Camera", "Frame Rate/Size", '#Frames', '#Dropped']

    def __init__(self, sesh_mgr: SessionManager, /):
        super().__init__()
        self._session = sesh_mgr
        """ The singleton session manager is the source for all information about the current session. """
        self._cameras: List[CamSource] = list()
        """ The list of available cameras for the current session. Initially empty. """

    def reload(self) -> None:
        self.beginResetModel()
        self._cameras.clear()
        for cam_id in self._session.available_camera_ids:
            self._cameras.append(self._session.get_camera(cam_id))
        self.endResetModel()

    def rowCount(self, /, parent=...):
        return len(self._cameras)

    def columnCount(self, /, parent=...):
        return len(self._COL_HEADERS)

    def data(self, index, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole:
            out = ''
            row, col = index.row(), index.column()
            if 0 <= row < self.rowCount() and 0 <= col < 4:
                cam = self._cameras[row]
                if col == 0:
                    out = str(cam.id)
                elif col == 1:
                    out = f"{cam.frame_rate}Hz @ {int(cam.frame_size[0])}x{int(cam.frame_size[1])}"
                elif col == 2:
                    out = str(cam.frame_count)
                else:
                    out = str(cam.num_dropped_frames)
            return out
        elif role == Qt.ItemDataRole.CheckStateRole:
            return None
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(self, section: int, orientation, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COL_HEADERS[section]
        else:
            return super().headerData(section, orientation, role)

    def index(self, row, column, /, parent=...):
        return self.createIndex(row, column)

    # Pycharm bug: Misses the pure virtual parent(self, index) in QAbstractItemModel
    # noinspection PyMethodOverriding
    def parent(self, idx: QModelIndex) -> QModelIndex:
        return QModelIndex()


_EVENT_TYPES: List[RxEvent] = [evt for evt in RxEvent]
""" All defined types of events time-stamped while recording a ReachX experiment session. """


class _EventInfoModel(QAbstractItemModel):
    """
    Table model that can be configured to display either summary counts for each type of ``RxEvent`` recorded during the
    current loaded experiment session, or a detailed summary of occurrence frames for any one event type, or all
    eveents.

    USAGE:
    - Pass the singleton SessionManager to the constructor, and set the model in a QTableView.
    - Call reload() whenever a different session is loaded.
    """
    _SUMMARY_COLS = ["Event", "Count"]
    _EVENTLIST_COLS = ["Event", "Frame #"]
    _RIGHT_VCENTER = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

    def __init__(self, sesh_mgr: SessionManager, summary: bool, evt: Optional[RxEvent] = None):
        """
        Construct the event information model.
        :param sesh_mgr: The singleton encapsulating the current loaded session
        :param summary: If True, model presents a summary of event counts for each type of event.
        :param evt: Ignored if `summary == True`. Otherwise, model lists all occurrences of the event specifiied.
           If `evt` is None, then it lists ALL recorded events.
        """
        super().__init__()
        self._session_manager = sesh_mgr
        """ The singleton session manager is the source for all information about the current session. """
        self._is_summary = summary
        self._evt = evt
        self._events: Optional[SessionEventList] = None
        """ The full recorded event list for the current session. Initially None. """

        self._per_event_frames: List[int] = list()
        """ All occurrences of a single specified event, IF model is configured accordingly. Else empty."""

    def reload(self) -> None:
        self.beginResetModel()
        self._events = self._session_manager.events
        self._per_event_frames.clear()
        if isinstance(self._evt, RxEvent):
            self._per_event_frames = self._events.get_all_occurrences_of(self._evt)
        self.endResetModel()

    def filter_on_event(self, evt: RxEvent) -> None:
        if not (self._is_summary or (self._events is None)):
            self.beginResetModel()
            self._evt = evt
            self._per_event_frames.clear()
            if isinstance(self._evt, RxEvent):
                self._per_event_frames = self._events.get_all_occurrences_of(self._evt)
            self.endResetModel()

    @Slot(QModelIndex)
    def on_cell_clicked_event(self, table_index: QModelIndex) -> None:
        """
        Whenever user clicks on a row in the event detail list, request that the session manager go to the
        corresponding frame number -- unless video playback is active.
        :param table_index: Table index that was clicked; should correspond to a particular recorded event.
        """
        # do nothing if playback in progress or if this is the summary table model
        if self._session_manager.playback_in_progress or self._is_summary:
            return
        row = table_index.row()
        n = self._events.get(row)[1] if self._evt is None else self._per_event_frames[row]
        self._session_manager.go_to_frame(n)

    def find_nearest_row_to_current_frame(self) -> int:
        """
        Return nearest row in detail list corresponding to an event that occurred during or immediately
        after the current frame. Not valid when video playback is active.
        :return: Row index of an event in detail list, as described. Returns -1 if model is configured to
            display summary list, or if video playback is active, or if row index not found.
        """
        if self._session_manager.playback_in_progress or self._is_summary:
            return -1
        if isinstance(self._evt, RxEvent):
            curr = self._session_manager.current_frame_num
            row = int(np.searchsorted(self._per_event_frames, curr, side='right'))
            if (row > 0) and (self._per_event_frames[row-1] == curr):
                row = row - 1
        else:
            row = self._events.index_of_event_frame_near(self._session_manager.current_frame_num)
        return row if 0 <= row < self.rowCount() else -1

    def rowCount(self, /, parent=...) -> int:
        """ For the event summary, a row for each event type, then a last row that shows total # of events. """
        return 0 if self._events is None else \
            ((len(RxEvent) + 1) if self._is_summary else self._events.event_count(self._evt))

    def columnCount(self, /, parent=...):
        return len(self._SUMMARY_COLS if self._is_summary else self._EVENTLIST_COLS)

    def data(self, index, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole:
            out = ''
            row, col = index.row(), index.column()
            if 0 <= row < self.rowCount() and 0 <= col < self.columnCount():
                if self._is_summary:
                    if 0 <= row < len(RxEvent):
                        evt = _EVENT_TYPES[row]
                        out = str(evt if col == 0 else self._events.event_count(evt))
                    else:
                        out = "TOTAL" if col == 0 else self._events.count
                elif self._evt is None:
                    evt_tuple = self._events.get(row)
                    out = str(evt_tuple[col])
                else:
                    out = str(self._evt) if col == 0 else str(self._per_event_frames[row])
            return out
        elif role == Qt.ItemDataRole.CheckStateRole:
            return None
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if index.column() == 0:
                return self._RIGHT_VCENTER
            else:
                return Qt.AlignmentFlag.AlignCenter if self._is_summary else self._RIGHT_VCENTER
        return None

    def headerData(self, section: int, orientation, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._SUMMARY_COLS[section] if self._is_summary else self._EVENTLIST_COLS[section]
        else:
            return super().headerData(section, orientation, role)

    def index(self, row, column, /, parent=...):
        return self.createIndex(row, column)

    # Pycharm bug: Misses the pure virtual parent(self, index) in QAbstractItemModel
    # noinspection PyMethodOverriding
    def parent(self, idx: QModelIndex) -> QModelIndex:
        return QModelIndex()


class _ReachInfoModel(QAbstractItemModel):
    """
    A table model that exposes information about either user-curated or scorer-generated reach segments for the current
    experiment session.

    The user-curated reach segment list is under the user's control, whereas the scorer-generated list is fixed --
    though the user can copy any or all reach segments from the scorer list to the curated list.

    In either case, the model defines a 4-column table listing the reach segments in chronological order, one row per
    segment. The 4 columns display the starting frame, the # of frames separating the "reachMax" event from segment
    start, the segment duration, and the result.

    USAGE:
    - Pass the singleton SessionManager to the constructor, and set the model in a QTableView.
    - Call reload() whenever a different session is loaded, a different session scorer is chosen (a session can be
      analyzed by multiple scorers), or there's a change in the user-curated reach segments list.
    """
    _COLUMNS = ["Frame", "\u0394 Max", "Dur", "Result"]

    def __init__(self, sesh_mgr: SessionManager, curated: bool) -> None:
        super().__init__()
        self._session_manager = sesh_mgr
        """ The singleton session manager is the source for all information about the current session. """
        self._curated = curated
        """ 
        If True, model displays the list of user-curated reach segments for the current experiment session, else
        the list of reach segments auto-detected by the current session scorer (if any).
        """
        self._segments: List[RxReachSegment] = list()
        """ A copy of the reach segments displayed by the model. """

    def reload(self) -> None:
        self.beginResetModel()
        self._segments = self._session_manager.get_curated_reaches() if self._curated else (
            self._session_manager.get_scored_reaches())
        self.endResetModel()

    @Slot(QModelIndex)
    def on_cell_clicked_event(self, table_index: QModelIndex) -> None:
        """
        Whenever user clicks on a row in the reach segment table, request that the session manager go to the
        corresponding frame number.
        :param table_index: Table index that was clicked; should correspond to a particular reach segment.
        """
        row = table_index.row()
        try:
            reach = self._segments[row]
            self._session_manager.go_to_frame(reach.frame)
        except IndexError:
            pass

    def rowCount(self, /, parent=...) -> int:
        return len(self._segments)

    def columnCount(self, /, parent=...):
        return len(self._COLUMNS)

    def data(self, index, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole:
            out = ''
            row, col = index.row(), index.column()
            if 0 <= row < self.rowCount() and 0 <= col < self.columnCount():
                try:
                    reach = self._segments[row]
                    col1 = f"+{reach.max_delta} {str(reach.hand_pos)}" if self._curated else f"+{reach.max_delta}"
                    out = str([reach.frame, col1, reach.dur, reach.result.short_name][col])
                except IndexError:
                    pass
            return out
        elif role == Qt.ItemDataRole.CheckStateRole:
            return None
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(self, section: int, orientation, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLUMNS[section]
        else:
            return super().headerData(section, orientation, role)

    def index(self, row, column, /, parent=...):
        return self.createIndex(row, column)

    # Pycharm bug: Misses the pure virtual parent(self, index) in QAbstractItemModel
    # noinspection PyMethodOverriding
    def parent(self, idx: QModelIndex) -> QModelIndex:
        return QModelIndex()


class CameraTab(QWidget):
    """
    This tab panel displays a single table with information on the camera videos recorded during the session,
    including the # of dropped frames in each video..
    """
    def __init__(self, parent: SessionView):
        super().__init__()
        self._camera_info = _CamInfoModel(parent.data_manager.session_manager)
        """ Camera information summary. """

        cam_view = QTableView()
        cam_view.setModel(self._camera_info)
        cam_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        cam_view.verticalHeader().setVisible(False)
        cam_view.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)

        layout = QGridLayout()
        layout.addWidget(cam_view, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 0)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 2)
        layout.addItem(QSpacerItem(0, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding), 1, 1)
        layout.setRowStretch(1, 1)
        self.setLayout(layout)

        self.reload()

    def reload(self) -> None:
        self._camera_info.reload()


class EventsTab(QWidget):
    """
    This tab panel houses two tables:
        1. Event count per event type summary.
        2. Full listing of all events (RxEvent) recorded during a session, or events of one particular type. This is
        configured so that, if the user clicks on an event row, the event frame becomes the current frame.

    Under the hood, it uses a `QTableView` for each table, each with a simple table model that queries the
    `SessionManager` for the displayed contents. Call reload() whenever a different session is loaded.
    """
    _SHOW_ALL: str = "All events"

    def __init__(self, parent: SessionView):
        super().__init__()
        self._event_counts = _EventInfoModel(parent.data_manager.session_manager, True)
        """ Event counts summary. """
        self._event_detail = _EventInfoModel(parent.data_manager.session_manager, False, None)
        """ Model for the event listing (all events, or all of a particular type). """
        self._detail_view = QTableView()
        """ The table view displaying the detailed event listing. """
        self._event_combo = QComboBox()
        """ Chooses which event type to show in the event listing table, or 'all' for all recorded events. """

        summary_view = QTableView()
        summary_view.setModel(self._event_counts)
        summary_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        summary_view.verticalHeader().setVisible(False)
        summary_view.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)

        self._detail_view.setModel(self._event_detail)
        self._detail_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._detail_view.verticalHeader().setVisible(False)
        self._detail_view.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)
        self._detail_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)

        # clicking on an event will trigger a request to make the event time the current frame
        self._detail_view.clicked.connect(self._event_detail.on_cell_clicked_event)

        self._event_combo.addItem(self._SHOW_ALL)
        self._event_combo.insertSeparator(1)
        self._event_combo.addItems([str(e) for e in RxEvent])
        self._event_combo.currentIndexChanged.connect(self._on_detail_event_changed)

        layout = QGridLayout()
        layout.addWidget(QLabel("Summary"), 0, 1,
                         alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(summary_view, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._event_combo, 0, 2, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self._detail_view, 1, 2, 2, 1)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 0)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 3)
        layout.addItem(QSpacerItem(0, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding), 3, 1)
        layout.setRowStretch(3, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(3, 1)

        self.setLayout(layout)

        self.reload()

        # to update row selection in detail table as current frame changes
        parent.data_manager.session_manager.frame_ready.connect(self._on_frame_ready)

    def reload(self) -> None:
        self._event_counts.reload()
        self._event_detail.reload()

    @Slot(int)
    def _on_detail_event_changed(self, idx: int) -> None:
        """
        Whenever user selects a different event, the event details table is updated to reflect the occurrences of that
        event.
        :param idx: The current index.
        """
        # the separator in the drop down list is at index 1 -- so we subtract 2 to get an index into _EVENT_TYPES!
        idx -= 2
        evt = _EVENT_TYPES[idx] if 0 <= idx < len(_EVENT_TYPES) else None
        self._event_detail.filter_on_event(evt)

    @Slot()
    def _on_frame_ready(self) -> None:
        row = self._event_detail.find_nearest_row_to_current_frame()
        if row >= 0:
            self._detail_view.selectRow(row)
            self._detail_view.scrollTo(self._event_detail.index(row, 0),
                                       QTableView.ScrollHint.PositionAtCenter)


class ReachesTab(QWidget):
    """
    This tab panel houses two tables: The left-hand table displays the list of user-curated reach segments, while the
    right-hand table displays the list of reach segments algorithmically generated based on analysis results from a
    body part detection ML model, aka, the session "scorer".

    Under the hood, it uses a ``QTableView`` for each table, each with a simple table model that queries the
    ``SessionManager`` for the displayed contents. Call reload() whenever a different session is loaded, whenever a
    different session scorer is selected, or whenever the list of user-curated reach segments is changed (users can
    add, delete, and edit reach segments in the curated list).
    """
    _SHOW_ALL: str = "All events"

    def __init__(self, parent: SessionView):
        super().__init__()
        self._sesh_mgr = parent.data_manager.session_manager
        """ App singleton that manages the current loaded experiment session. """
        self._curated_tm = _ReachInfoModel(self._sesh_mgr, True)
        """ Model listing all curated reach segments in chronological order. """
        self._scored_tm = _ReachInfoModel(self._sesh_mgr, False)
        """ Model listing all scorer-generated reach segments in chronological order. """
        self._curated_table = QTableView()
        """ The table view displaying all curated reach segments. """
        self._scored_table = QTableView()
        """ The table view displaying all scorer-generated reach segments. """
        self._add_all_btn = QPushButton(RxIcons.NEW, "")
        """ Pressing this button adds all scorer-generated reach segments (if any) to the curated set. """
        self._delete_all_btn = QPushButton(RxIcons.TRASH, "")
        """ Pressing this button deletes all curated reach segments for the current session. """

        self._curated_table.setModel(self._curated_tm)
        self._curated_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._curated_table.verticalHeader().setVisible(False)
        self._curated_table.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)
        self._curated_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._curated_table.clicked.connect(self._curated_tm.on_cell_clicked_event)

        self._scored_table.setModel(self._scored_tm)
        self._scored_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._scored_table.verticalHeader().setVisible(False)
        self._scored_table.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)
        self._scored_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._scored_table.clicked.connect(self._scored_tm.on_cell_clicked_event)

        self._add_all_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._add_all_btn.setStyleSheet("QPushButton { border: none; }")
        self._add_all_btn.setToolTip("Add all scorer-generated reach segments to the curated set")
        self._add_all_btn.clicked.connect(lambda: self._sesh_mgr.add_scored_reaches_to_curated_set())

        self._delete_all_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._delete_all_btn.setStyleSheet("QPushButton { border: none; }")
        self._delete_all_btn.setToolTip("Remove all curated reach segments")
        self._delete_all_btn.clicked.connect(lambda: self._sesh_mgr.delete_all_curated_reaches())

        layout = QGridLayout()
        group = QHBoxLayout()
        group.addWidget(QLabel("Curated"))
        group.addWidget(self._delete_all_btn)
        layout.addLayout(group, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._curated_table, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        group = QHBoxLayout()
        group.addWidget(QLabel("Auto-generated"))
        group.addWidget(self._add_all_btn)
        layout.addLayout(group, 0, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._scored_table, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 0)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 3)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(0, 1)
        self.setLayout(layout)

        self.reload()

        # to refresh both tables whenever the set of reach segments changes
        self._sesh_mgr.reaches_changed.connect(self._on_reaches_changed)
        self._sesh_mgr.scorer_changed.connect(self._on_scorer_changed)
        self._sesh_mgr.segmentation_result_changed.connect(self._on_scorer_changed)

    def reload(self) -> None:
        self._curated_tm.reload()
        self._scored_tm.reload()

    def _on_reaches_changed(self) -> None:
        """ Reload curated table when the list of curated reach segments changes in any way. """
        self._curated_tm.reload()

    def _on_scorer_changed(self) -> None:
        """ Reload the scorer-generated table whenever the current session scorer changes. """
        self._scored_tm.reload()


class _SessionTreeModel(QAbstractItemModel):
    """
    A hierarchical model of all session folders found under a workspace's session root.

    This model organizes the "session tree" with rig names at the top level, then all session dates per rig at the
    second level, then all session numbers per date at the third level. Note that this is different from how the
    session folders are organized in the file system (date -> rig -> session num).

    There is only one column at each level -- displaying the name of the item at that level: rig names at level 0,
    session dates in YYYYMMDD format at level 1, and "sessionNNN" at level 2. Rig names at level 0 are ordered
    alphabetically, session dates at level 1 chronologically, and session numbers at level 2 in ascending order.
    """

    class Node:
        """ A node in the session tree model. """
        def __init__(self, datum: Union[str, RxSessionID], parent: Optional[_SessionTreeModel.Node] = None):
            """
            Construct a node in the 3-level session tree.
            :param datum: For a node corresponding to a rig, this should be the rig name; for a session date, the
                date string 'YYYYMMDD'; for a particular session, the session identifier.
            :param parent: The parent node (this must be None for the root node itself).
            """
            self._parent = parent
            self._datum = datum
            self._children: List[_SessionTreeModel.Node] = []

        def add_child(self, child: _SessionTreeModel.Node) -> None:
            self._children.append(child)

        def child_at(self, row) -> Optional[_SessionTreeModel.Node]:
            return None if (row < 0 or row >= len(self._children)) else self._children[row]

        def num_children(self):
            return len(self._children)

        def datum(self) -> Union[str, RxSessionID]:
            return self._datum

        def parent(self):
            return self._parent

        def row_in_parent(self):
            return 0 if (self._parent is None) else self._parent._children.index(self)

    def __init__(self, session_ids: List[RxSessionID]):
        """
        Construct the session folder tree model.
        :param session_ids: List of all sessions to be presented by the model.
        """
        super().__init__()
        self._root: _SessionTreeModel.Node = _SessionTreeModel.Node("root")
        """ The root node of the session tree (not exposed for display). """
        self._setup_tree(session_ids)

    def _setup_tree(self, session_ids: List[RxSessionID]):
        rigs = list({sid.rig for sid in session_ids})
        rigs.sort()
        for rig in rigs:
            rig_node = _SessionTreeModel.Node(rig, self._root)
            sids_for_rig = [sid for sid in session_ids if sid.rig == rig]
            dates = list({sid.date for sid in sids_for_rig})
            dates.sort()
            for d in dates:
                date_node = _SessionTreeModel.Node(d, rig_node)
                sids_for_date = [sid for sid in sids_for_rig if sid.date == d]
                sids_for_date.sort(key=lambda k: k.num)
                for sid in sids_for_date:
                    date_node.add_child(_SessionTreeModel.Node(sid, date_node))
                rig_node.add_child(date_node)
            self._root.add_child(rig_node)

    def find_session(self, sid: RxSessionID) -> QModelIndex:
        """
        Find the model index for the session tree leaf node corresponding to the specified session.
        :param sid: Session ID.
        :return: The corresponding model index; invalid if not found.
        """
        for i in range(self._root.num_children()):
            rig_node = self._root.child_at(i)
            for j in range(rig_node.num_children()):
                date_node = rig_node.child_at(j)
                for k in range(date_node.num_children()):
                    if date_node.child_at(k).datum() == sid:
                        return self.createIndex(k, 0, date_node.child_at(k))
        return QModelIndex()

    def get_session_id_at(self, index: QModelIndex) -> Optional[RxSessionID]:
        """
        Get the session ID at the specified location in the model, if any.
        :param index: Model index.
        :return: If index corresponds to a leaf node, then return the corresponding session ID; else return None.
        """
        datum = self._get_node(index).datum()
        return datum if isinstance(datum, RxSessionID) else None

    def print_tree(self) -> None:
        """ For testing purposes. """
        for i in range(self._root.num_children()):
            rig_node = self._root.child_at(i)
            print(f"{rig_node.datum()}")
            for j in range(rig_node.num_children()):
                date_node = rig_node.child_at(j)
                print(f"  |--- {date_node.datum()}")
                for k in range(date_node.num_children()):
                    sid: RxSessionID = date_node.child_at(k).datum()
                    print(f"  |---    |--- session{sid.num}")

    def index(self, row, column, parent=QModelIndex()) -> QModelIndex:
        if column != 0:   # there is only one column in our model
            return QModelIndex()
        parent_node = self._get_node(parent)
        child_node = parent_node.child_at(row)
        if child_node:
            return self.createIndex(row, column, child_node)
        return QModelIndex()

    def _get_node(self, index: QModelIndex) -> _SessionTreeModel.Node:
        if index.isValid():
            node = index.internalPointer()
            if node:
                return node
        return self._root

    # Pycharm bug: Misses the pure virtual parent(self, index) in QAbstractItemModel
    # noinspection PyMethodOverriding
    def parent(self, child: QModelIndex) -> QModelIndex:
        if not child.isValid():
            return QModelIndex()

        child_node = self._get_node(child)
        parent_node = child_node.parent()
        return QModelIndex() if (parent_node == self._root) \
            else self.createIndex(parent_node.row_in_parent(), 0, parent_node)

    def columnCount(self, /, parent=...) -> int:
        """ There is only one column in the model, regardless the hierarchical level. """
        return 1

    def rowCount(self, /, parent=...) -> int:
        parent_node = self._get_node(parent)
        return parent_node.num_children()

    def data(self, index, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole:
            if not index.isValid():
                return None
            else:
                node = self._get_node(index)
                if node == self._root:
                    return None
                else:
                    datum = node.datum()
                    return f"session{datum.num}" if isinstance(datum, RxSessionID) else str(datum)
        elif role == Qt.ItemDataRole.CheckStateRole:
            return None
        return None

    def headerData(self, section: int, orientation, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return "Sessions"
        else:
            return super().headerData(section, orientation, role)


class _SessionSelectDialog(QDialog):
    """
    Modal dialog by which user selects a different session to load.

    USAGE: Construct the dialog object and call select_session() to raise it, supplying the entire list of sessions
    available in the active workspace. The sessions are displayed in a pseudo-file explorer like hierarchical tree
    with three levels: rig name at the top level, session dates on the second level, and session "folders" on the
    third level. The user can select any session folder then hit the "Ok" button to confirm selection, or as a
    short-cut, simply double-click on the desired session.
    """
    def __init__(self, parent: Optional[QWidget], /):
        super().__init__(parent)
        self._session_chosen: Optional[RxSessionID] = None
        """ The session currently selected on the dialog. On close, this is set to None if user cancelled. """
        self._session_tree: Optional[_SessionTreeModel] = None
        """ 
        Model of the session tree with 3 levels: rig --> date --> session number. The leaf nodes at the third
        level correspond to session IDs.
        """
        self._session_tree_view = QTreeView()
        """ The displayed session tree. """
        self._button_box = QDialogButtonBox()
        """ 
        The OK and Cancel buttons for the dialog. We disable OK whenever the current selection in the session
        tree is not a leaf node corresponding to a session ID.
        """

        self.setWindowTitle("Select a session to view")

        self._button_box = QDialogButtonBox()
        self._button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self._session_tree_view, stretch=1)
        layout.addWidget(self._button_box)
        self.setLayout(layout)
        self.setMinimumSize(QSize(400, 600))

        self._session_tree_view.doubleClicked.connect(self._on_double_clicked)

    def select_session(
            self, sessions: List[RxSessionID], initial: Optional[RxSessionID] = None) -> Optional[RxSessionID]:

        if len(sessions) == 0:
            return None

        self._session_chosen = None

        self._session_tree = _SessionTreeModel(sessions)
        self._session_tree_view.setModel(self._session_tree)
        if initial in sessions:
            model_index = self._session_tree.find_session(initial)
            if model_index.isValid():
                self._session_tree_view.selectionModel().select(
                    model_index,
                    QItemSelectionModel.SelectionFlag.ClearAndSelect
                )
                self._session_tree_view.scrollTo(model_index)

        # can't do this in constructor because selection model is created after model is set on treeview
        self._session_tree_view.selectionModel().selectionChanged.connect(self._on_selection_changed)

        # if user cancels, then no session selected
        if self.exec() != QDialog.DialogCode.Accepted:
            self._session_chosen = None

        self._session_tree_view.selectionModel().selectionChanged.disconnect(self._on_selection_changed)

        return self._session_chosen

    @Slot(QModelIndex)
    def _on_double_clicked(self, index: QModelIndex) -> None:
        sid = self._session_tree.get_session_id_at(index)
        if sid:
            self._session_chosen = sid
            self.accept()

    @Slot(QItemSelection, QItemSelection)
    def _on_selection_changed(self, selected: QItemSelection, _: QItemSelection) -> None:
        """
        If the current selection in the session tree does or does not correspond to a session ID, enable or disable
        the dialog's OK button.
        """
        indices = selected.indexes()
        enable = False
        self._session_chosen = None
        if len(indices) == 1:
            sid = self._session_tree.get_session_id_at(indices[0])
            if sid:
                self._session_chosen = sid
                enable = True
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enable)


class _AnalysisDlg(QDialog):
    """
    Reconfigurable modal dialog raised when either the "Find Reaches..." or "Multisession Analysis" button is clicked.

    The dialog contains two widget frames. One selects a batch of sessions to analyze using the current body part
    detection model. A second frame exposes the reach segmentation control parameters, ``RxReachSegControls``, for
    the active workspace in a series of ``QLineEdit`` widgets. Below these frames is a button box with **Start** and
    **Cancel** buttons.

    When the "Find Reaches..." button is clicked, only the second frame is visible. The user can leave the parameter
    values as they are, or make some changes.

    When the "Multisession Analysis..." button is clicked, only the first frame is visible initially. This frame
    contains widgets for selecting the batch of sessions to be analyzed. It includes a check box to run the reach
    segmentation algorithm on each session after that session is successfully analyzed. The check box is initially
    unchecked. If the user checks it, the second frame is made visible so that the user can make any desired changes to
    the active workspace's reach segmentation control parameters.

    In either configuration, pressing the **Start** or **Cancel** button extinguishes the dialog. Pressing the former
    button confirms the user wishes to start the configured task.

    **USAGE**:
     - Construct the dialog.
     - Call ``edit_reach_seg_controls()`` when "Find Reaches..." button is pressed.
     - Call ``select_sessions_to_analyze()`` when "Multisession Analysis..." button is pressed.
    """

    def __init__(self, parent: Optional[QWidget], /):
        super().__init__(parent)
        self.setWindowTitle("Multisession Analysis")

        self._seg_ctrls = RxReachSegControls()
        """ The reach segmentation control parameters edited in this dialog. """
        self._sessions: List[RxSessionID] = list()
        """ The list of all sessions from which user will select a subset to analyze. """

        self._sesh_sel_panel = QGroupBox("Sessions to analyze")
        """ Panel housing widgets for selecting which experimnent sessions to analyze. """
        self._seg_ctrl_panel = QGroupBox("Reach segmentation parameters")
        """ Panel housing widgets for the reach segmentation control parameters. """
        self._button_box = QDialogButtonBox()
        """  The Start and Cancel buttons for the dialog. """

        self._rig_combo = QComboBox()
        """ Combo box selects the experiment rig. """
        self._date_min_combo = QComboBox()
        """ Combo box selects the minimum experiment date. """
        self._date_max_combo = QComboBox()
        """ Combo box select the maximum experiment date. """
        self._sesh_table = QTableView()
        """ Table listing all sessions in workspace that match the current filters (experiment rig and date range). """
        self._sesh_table_model = _MatchingSessionsTM()
        """ Simple table model for the matching sessions table. """
        self._clear_btn = QPushButton("0 sessions selected")
        """ Button title reflects # sessions selected; press it to clear the current selection. """
        self._find_reaches_chkbox = QCheckBox("Find reach segments also?")
        """ When this box is checked, the frame housing reach seg controls is made visible. """

        # numeric edit widgets - one for each reach seg control param
        self._edit_init_speed = QLineEdit()
        self._edit_distZ_hand_start = QLineEdit()
        self._edit_dirchange_speed = QLineEdit()
        self._edit_drop_speed = QLineEdit()
        self._edit_drop_distZ = QLineEdit()
        self._edit_drop_distY = QLineEdit()
        self._edit_dist_end = QLineEdit()
        self._edit_confidence = QLineEdit()
        self._edit_dist_to_origin = QLineEdit()
        self._edit_min_frame = QLineEdit()
        self._edit_max_frame = QLineEdit()

        self._sesh_table.setModel(self._sesh_table_model)
        self._sesh_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._sesh_table.verticalHeader().setVisible(False)
        self._sesh_table.verticalHeader().setDefaultSectionSize(25)
        self._sesh_table.setMinimumHeight(25*5)
        self._sesh_table.setMaximumHeight(25*20)
        self._sesh_table.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)
        self._sesh_table.setSelectionMode(QTableView.SelectionMode.MultiSelection)

        self._clear_btn.setToolTip("Clear the current selection in the matching sessions table")
        self._clear_btn.clicked.connect(lambda: self._sesh_table.selectionModel().clearSelection())

        self._find_reaches_chkbox.setToolTip("Check this box to perform reach segmentation on each session analyzed")

        self._button_box.addButton("Start", QDialogButtonBox.ButtonRole.AcceptRole)
        self._button_box.accepted.connect(self.accept)
        self._button_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self._button_box.rejected.connect(self.reject)

        self._edit_init_speed.setToolTip("Hand-to-pellet approach speed threshold in mm/s. Range: [-0.1 .. -0.01].")
        self._edit_init_speed.setValidator(QDoubleValidator(-0.1, -0.01, 3))
        self._edit_init_speed.setFixedWidth(100)
        self._edit_distZ_hand_start.setToolTip("Minimum Z-coordinate separation between hand and pellet origin at "
                                               "reach initiation, in mm. Range: [1..9]")
        self._edit_distZ_hand_start.setValidator(QDoubleValidator(1, 9, 1))
        self._edit_distZ_hand_start.setFixedWidth(100)
        self._edit_dirchange_speed.setToolTip("Hand speed threshold indicating direction change since "
                                              "reach started, in mm/s. Range: [0.01 .. 0.1].")
        self._edit_dirchange_speed.setValidator(QDoubleValidator(0.01, 0.1, 3))
        self._edit_dirchange_speed.setFixedWidth(100)
        self._edit_drop_speed.setToolTip("Pellet speed exceeding this threshold in mm/s indicates pellet was dropped. "
                                         "Range: [0.1 .. 0.5].")
        self._edit_drop_speed.setValidator(QDoubleValidator(0.1, 0.5, 3))
        self._edit_drop_speed.setFixedWidth(100)
        validator = QDoubleValidator(-10, -2, 1)
        self._edit_drop_distZ.setToolTip("If pellet Z-coordinate (mm) WRT pellet origin is < this threshold, pellet "
                                         "was dropped. Range: [-10 .. -2].")
        self._edit_drop_distZ.setValidator(validator)
        self._edit_drop_distZ.setFixedWidth(100)
        self._edit_drop_distY.setToolTip("If pellet Y-coordinate (mm) WRT pellet origin is < this threshold, pellet "
                                         "was dropped. Range: [-10 .. -2].")
        self._edit_drop_distY.setValidator(validator)
        self._edit_drop_distY.setFixedWidth(100)
        self._edit_dist_end.setToolTip("Min distance of hand from pellet origin at reach's end, in mm. Range: [2..10].")
        self._edit_dist_end.setValidator(QDoubleValidator(2, 10, 1))
        self._edit_dist_end.setFixedWidth(100)
        self._edit_confidence.setToolTip("Minimum confidence value required for 'detected' hand and pellet locations, "
                                         "in [0..1].")
        self._edit_confidence.setValidator(QDoubleValidator(0, 1, 2))
        self._edit_confidence.setFixedWidth(100)
        self._edit_dist_to_origin.setToolTip("Minimum distance of pellet from pellet origin to indicate pellet was "
                                             "grabbed, in mm. Range: [1..5].")
        self._edit_dist_to_origin.setValidator(QDoubleValidator(1, 5, 1))
        self._edit_dist_to_origin.setFixedWidth(100)
        self._edit_min_frame.setToolTip("Minimum acceptable length of a reach segment (# video frames >= 5).")
        self._edit_min_frame.setValidator(QIntValidator(10, 50))
        self._edit_min_frame.setFixedWidth(100)
        self._edit_max_frame.setToolTip("Maximum acceptable length of a reach segment (# video frames >= 5).")
        self._edit_max_frame.setValidator(QIntValidator(25, 200))
        self._edit_max_frame.setFixedWidth(100)

        # layout the panel for selecting the sessions to analyze...
        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Rig:"))
        row.addWidget(self._rig_combo, stretch=1)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Dates:"))
        row.addWidget(self._date_min_combo, stretch=1)
        row.addWidget(QLabel(" to "))
        row.addWidget(self._date_max_combo, stretch=1)
        layout.addLayout(row)
        layout.addWidget(self._sesh_table, stretch=1)
        layout.addWidget(self._clear_btn, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self._sesh_sel_panel.setLayout(layout)

        # layout the panel housing the reach segmentation controls...
        layout = QVBoxLayout()
        grid = QGridLayout()
        grid.setSpacing(2)

        grid.addWidget(QLabel("Reach Initiation Speed (mm/s):"), 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_init_speed, 0, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Reach Initiation Hand-Pellet Dist (mm):"), 1, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_distZ_hand_start, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Reach Speed Dir Change Threshold (mm/s):"), 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_dirchange_speed, 2, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Pellet Drop Speed Threshold (mm/s):"), 3, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_drop_speed, 3, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Pellet Drop Dist Threshold, Z (mm/s):"), 4, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_drop_distZ, 4, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Pellet Drop Dist Threshold, Y (mm/s):"), 5, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_drop_distY, 5, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Reach End Hand-Pellet Origin Dist (mm):"), 6, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_dist_end, 6, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Minimum confidence score (0..1):"), 7, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_confidence, 7, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Min pellet displacement for grab (mm):"), 8, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_dist_to_origin, 8, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Minimum reach segment length:"), 9, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_min_frame, 9, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Maximum reach segment length:"), 10, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._edit_max_frame, 10, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        grid.setColumnMinimumWidth(1, 5)
        layout.addLayout(grid)
        layout.addStretch(1)
        self._seg_ctrl_panel.setLayout(layout)

        # layout the two panels and button box in dialog
        grid = QGridLayout()
        grid.addWidget(self._sesh_sel_panel, 0, 0)
        grid.addWidget(self._find_reaches_chkbox, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self._seg_ctrl_panel, 0, 1, 2, 1)
        grid.addWidget(self._button_box, 2, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(grid)

        self._connect_or_disconnect_handlers()

    def _connect_or_disconnect_handlers(self, connect: bool = True) -> None:
        """
        Connect/disconnect handlers for the relevant widgets on the session selection panel. We disconnect handlers
        when reloading certain widgets to avoid unintended effects.
        :param connect: True to connect, False to disconnnect. Default = True.
        """
        if connect:
            self._rig_combo.currentTextChanged.connect(self._on_rig_selected)
            self._date_min_combo.currentTextChanged.connect(self._on_date_min_selected)
            self._date_max_combo.currentTextChanged.connect(self._on_date_max_selected)
            self._sesh_table.selectionModel().selectionChanged.connect(self._on_selection_changed)
            self._find_reaches_chkbox.checkStateChanged.connect(self._on_toggle_find_reaches)
        else:
            self._rig_combo.currentTextChanged.disconnect(self._on_rig_selected)
            self._date_min_combo.currentTextChanged.disconnect(self._on_date_min_selected)
            self._date_max_combo.currentTextChanged.disconnect(self._on_date_max_selected)
            self._sesh_table.selectionModel().selectionChanged.disconnect(self._on_selection_changed)
            self._find_reaches_chkbox.checkStateChanged.disconnect(self._on_toggle_find_reaches)

    @Slot(str)
    def _on_rig_selected(self, _: str) -> None:
        """
        Whenever a different rig is selected, reload the min and max date combos with the available recording dates
        for that rig. Then select the full range of available dates, reload the table model of matching sessions, and
        select all of those sessions initially.
        """
        self._connect_or_disconnect_handlers(connect=False)
        self._stuff_date_combos()
        self._update_matching_sessions_table()
        self._select_all_matching_sessions()
        self._stuff_readout()
        self._connect_or_disconnect_handlers()

    @Slot(str)
    def _on_date_min_selected(self, min_date: str) -> None:
        """
        Whenever the minimum date changes, correct the current maximum date if it is no longer valid, then reload the
        table model of matching sessions and select all of those sessions.
        :param min_date: The date selected in the minimum date combo.
        """
        self._connect_or_disconnect_handlers(connect=False)
        if self._date_max_combo.currentText() < min_date:
            self._date_max_combo.setCurrentIndex(self._date_max_combo.count() - 1)
        self._update_matching_sessions_table()
        self._select_all_matching_sessions()
        self._stuff_readout()
        self._connect_or_disconnect_handlers()

    @Slot(str)
    def _on_date_max_selected(self, max_date: str) -> None:
        """
        Whenever the maximum date changes, correct the current minimum date if it is no longer valid, then reload the
        table model of matching sessions and select all of those sessions.
        :param max_date: The date selected in the maximum date combo.
        """
        self._connect_or_disconnect_handlers(connect=False)
        if self._date_min_combo.currentText() > max_date:
            self._date_min_combo.setCurrentIndex(0)
        self._update_matching_sessions_table()
        self._select_all_matching_sessions()
        self._stuff_readout()
        self._connect_or_disconnect_handlers()

    @Slot(QItemSelection, QItemSelection)
    def _on_selection_changed(self, _selected: QItemSelection, _deselected: QItemSelection) -> None:
        """
        Whenever selection changes in the matching sessions table, update the readout label indicating how many
        sessions are currently selected.
        """
        self._stuff_readout()

    @Slot(Qt.CheckState)
    def _on_toggle_find_reaches(self, _: Qt.CheckState) -> None:
        """ When the `Find Reaches?' box is checked, the panel housing segmenation controls is revealed. """
        self._seg_ctrl_panel.setVisible(self._find_reaches_chkbox.isChecked())

    def select_sessions_to_analyze(self, session_ids: List[RxSessionID], seg_ctrls: RxReachSegControls) \
            -> Tuple[List[RxSessionID], Optional[RxReachSegControls]]:
        """
        Configure and raise this dialog so user can select a batch of sessions to analyze with the current body part
        detection model and, optionally, to perform reach segmentation on each analyzed session.

        Initially, the dialog is configured for selecting sessions for analysis and perform reach segmentation. Specify
        the target experiment rig and a range of experiment dates, and the set of sessions matching those "filters" is
        displayed in a table from which you can choose any number of sessions to analyze. If you do NOT want to do
        reach segmentation as well, uncheck the relevant box, and the panel housing the segmentation control
        parameters is hidden.

        :param session_ids: List of all experiment session identifiers from which selected sessions are chosen. If
            empty, the dialog is not raised.
        :param seg_ctrls: The current reach segmentation control parameters.
        :return: A 2-tuple: ([], None) if user canceled out of dialog; (sessions, None) if user selected sessions to
            analyze but elected not to perform reach segmentation on the sessions; (sessions, updated seg params) if
            user selected sessions to both analyze and do reach segmentation.
        """
        if len(session_ids) == 0:
            get_application_logger().info("Empty session list!")
            return [], None

        # configure dialog for display
        self._sessions.clear()
        self._sessions.extend(session_ids)
        self._seg_ctrls = seg_ctrls
        self._stuff_reach_seg_controls()
        self._sesh_sel_panel.setVisible(True)
        self._seg_ctrl_panel.setVisible(True)
        self._find_reaches_chkbox.setVisible(True)

        self._connect_or_disconnect_handlers(connect=False)
        self._find_reaches_chkbox.setChecked(True)
        rig_names: List[str] = list(set(sid.rig for sid in self._sessions))
        rig_names.sort()
        self._rig_combo.clear()
        self._rig_combo.addItems(rig_names)
        self._rig_combo.setCurrentIndex(0)
        self._stuff_date_combos()
        self._update_matching_sessions_table()
        self._select_all_matching_sessions()
        self._stuff_readout()
        self._connect_or_disconnect_handlers()

        # raise dialog and return selected session list and (possibly) updated segmentation control parameters, unless
        # user canceled
        if self.exec() != QDialog.DialogCode.Accepted:
            return [], None
        else:
            selected_indexes = self._sesh_table.selectionModel().selectedIndexes()
            sel_rows = sorted(list({index.row() for index in selected_indexes}))
            out = self._sesh_table_model.get_selected_sessions(sel_rows)
            updated_seg_ctrls = None
            if len(out) > 0 and self._find_reaches_chkbox.isChecked():
                updated_seg_ctrls = self._get_reach_seg_controls()
            return out, updated_seg_ctrls

    def _stuff_date_combos(self) -> None:
        """
        Reload the minimum and maximum date combo boxes with all available recording dates for the selected rig name.
        The selected date in each combo box is initially set to span the full range of recording dates for that rig.

        NOTE: Since session date strings are in YYYYMMDD format, ascending alphabetic order matches chronological order!
        """
        rig_selected = self._rig_combo.currentText()
        dates: List[str] = list(set(sid.date for sid in self._sessions if sid.rig == rig_selected))
        dates.sort()
        self._date_min_combo.clear()
        self._date_min_combo.addItems(dates)
        self._date_max_combo.clear()
        self._date_max_combo.addItems(dates)
        self._date_min_combo.setCurrentIndex(0)
        self._date_max_combo.setCurrentIndex(len(dates) - 1)

    def _update_matching_sessions_table(self) -> None:
        """
        Reloads the matching sessions table whenever there's a change in the rig, minimum date, or maximum date combo
        boxes.
        """
        rig_selected = self._rig_combo.currentText()
        min_date = self._date_min_combo.currentText()
        max_date = self._date_max_combo.currentText()
        self._sesh_table_model.reload(self._sessions, rig_selected, min_date, max_date)

    def _select_all_matching_sessions(self) -> None:
        """ Select all sessions listed in the matching sessions table. """
        n = self._sesh_table_model.rowCount()
        start = self._sesh_table_model.index(0, 0)
        end = self._sesh_table_model.index(n - 1, 0)
        sel_range = QItemSelection(start, end)
        self._sesh_table.selectionModel().select(sel_range, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _stuff_readout(self) -> None:
        """ Update the label of the push button to reflect # of sessions selected from the matching sessions table. """
        n = len(self._sesh_table.selectionModel().selectedRows())
        self._clear_btn.setText(f"{n} sessions selected")

    def _stuff_reach_seg_controls(self) -> None:
        """ Load the numeric widgets housing the reach segmentation control parmeter values. """
        self._edit_init_speed.setText(f"{self._seg_ctrls.reach_init_speed:.3f}")
        self._edit_distZ_hand_start.setText(f"{self._seg_ctrls.distZ_hand_start:.1f}")
        self._edit_dirchange_speed.setText(f"{self._seg_ctrls.reach_dirchange_speed:.3f}")
        self._edit_drop_speed.setText(f"{self._seg_ctrls.pellet_drop_speed:.3f}")
        self._edit_drop_distZ.setText(f"{self._seg_ctrls.pellet_drop_distZ:.1f}")
        self._edit_drop_distY.setText(f"{self._seg_ctrls.pellet_drop_distY:.1f}")
        self._edit_dist_end.setText(f"{self._seg_ctrls.dist_thresh_end:.1f}")
        self._edit_confidence.setText(f"{self._seg_ctrls.confidence:.2f}")
        self._edit_dist_to_origin.setText(f"{self._seg_ctrls.pellet_dist_to_origin:.1f}")
        self._edit_min_frame.setText(str(self._seg_ctrls.min_frame))
        self._edit_max_frame.setText(str(self._seg_ctrls.max_frame))

    def _get_reach_seg_controls(self) -> Optional[RxReachSegControls]:
        """
        Retrieve reach segmentation control parameter values from the corresponding widgets in the dialog.
        :return: The segmentation control parameters, or None if an exception occurs (should not happen!).
        """
        try:
            out = RxReachSegControls(
                reach_init_speed=float(self._edit_init_speed.text()),
                distZ_hand_start=float(self._edit_distZ_hand_start.text()),
                reach_dirchange_speed=float(self._edit_dirchange_speed.text()),
                pellet_drop_speed=float(self._edit_drop_speed.text()),
                pellet_drop_distZ=float(self._edit_drop_distZ.text()),
                pellet_drop_distY=float(self._edit_drop_distY.text()),
                dist_thresh_end=float(self._edit_dist_end.text()),
                confidence=float(self._edit_confidence.text()),
                pellet_dist_to_origin=float(self._edit_dist_to_origin.text()),
                min_frame=int(self._edit_min_frame.text()),
                max_frame=int(self._edit_max_frame.text())
            )
            return out
        except ValueError:
            get_application_logger().warning("Invalid reach seg control parameter!")
            return None

    def edit_reach_seg_controls(self, seg_ctrls: RxReachSegControls) -> Optional[RxReachSegControls]:
        """
        Configure and raise this dialog to edit the control parameters governing the ReachX reach segmentation algorithm
        prior to initiating a segmentation task.

        :param seg_ctrls: The current reach segmentation control parameters.
        :return: None if user canceled out of dialog, else the updated segmentation control parameters to use for the
            segmentation task (possibly unchanged).
        """
        # the session selection panel is hidden in this configuration
        self._sesh_sel_panel.setVisible(False)
        self._find_reaches_chkbox.setVisible(False)
        self._seg_ctrl_panel.setVisible(True)

        # stuff the widgets with the segmenation control param values
        self._seg_ctrls = seg_ctrls
        self._stuff_reach_seg_controls()

        # raise dialog and return updated segmentation control parameters unless user canceled
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        else:
            return self._get_reach_seg_controls()


class _MatchingSessionsTM(QAbstractItemModel):
    """ Table model for the filtered sessions table in the session selection panel of ``_AnalysisDlg``. """
    _COL_HEADERS = ['Matching sessions']

    def __init__(self):
        super().__init__()
        self._matching_sessions: List[RxSessionID] = list()
        """ The list of sessions that match the currently selected rig and date range. """

    def get_selected_sessions(self, sel_rows: List[int]) -> List[RxSessionID]:
        return [self._matching_sessions[i] for i in sel_rows]

    def reload(self, sessions: List[RxSessionID], rig: str, min_date: str, max_date: str) -> None:
        self.beginResetModel()
        self._matching_sessions.clear()
        for sid in sessions:
            if sid.rig == rig and min_date <= sid.date <= max_date:
                self._matching_sessions.append(sid)
        self.endResetModel()

    def rowCount(self, /, parent=...):
        return len(self._matching_sessions)

    def columnCount(self, /, parent=...):
        return len(self._COL_HEADERS)

    def data(self, index, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole:
            out = ''
            row, col = index.row(), index.column()
            if 0 <= row < self.rowCount() and (col == 0):
                sid = self._matching_sessions[row]
                out = f"{sid.date}/session{sid.num}"
            return out
        elif role == Qt.ItemDataRole.CheckStateRole:
            return None
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft
        return None

    def headerData(self, section: int, orientation, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COL_HEADERS[section]
        else:
            return super().headerData(section, orientation, role)

    def index(self, row, column, /, parent=...):
        return self.createIndex(row, column)

    # Pycharm bug: Misses the pure virtual parent(self, index) in QAbstractItemModel
    # noinspection PyMethodOverriding
    def parent(self, idx: QModelIndex) -> QModelIndex:
        return QModelIndex()
