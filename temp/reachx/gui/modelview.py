from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Slot, Qt, QRegularExpression, QSize
from PySide6.QtGui import QShortcut, QKeySequence, QRegularExpressionValidator
from PySide6.QtWidgets import QWidget, QTableView, QPushButton, QHeaderView, QGridLayout, QLabel, QSpacerItem, \
    QSizePolicy, QComboBox, QHBoxLayout, QTabWidget, QMessageBox, QDialog, QLineEdit, QDialogButtonBox, QVBoxLayout, \
    QFrame

from reachx.data.modelmgr import BodyPartMarker
from reachx.common import RxCam, RxBodyPart, RxSessionID, RxModelID, RxNetworkType
from reachx.uicommon import RxIcons
from reachx.data.datamanager import DataManager
from reachx.gui.baseview import BaseView


class _GetModelNameDialog(QDialog):
    """  Reusable modal dialog to query user for the project name to assign to a new body part detection ML model. """
    def __init__(self, parent: Optional[QWidget], /):
        super().__init__(parent)
        self._prj_name_edit = QLineEdit()
        """ The user enters the project name here. """
        self._button_box = QDialogButtonBox()
        """ 
        The OK and Cancel buttons for the dialog. We disable OK whenever the current model is invalid or matches the
        project name of an existing ML model.
        """
        self._existing_project_names: List[str] = list()
        """ While dialog raised, the list of project names for all existing ML models in the active workspace. """

        self.setWindowTitle(f"New body part detection model")

        self._button_box = QDialogButtonBox()
        self._button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        label = QLabel("Project:")
        self._prj_name_edit.setToolTip("Enter a project name, 5-20 alphanumeric characters, starting with a letter")
        regex = QRegularExpression("^[a-zA-Z][a-zA-Z0-9]{0,19}$")
        self._prj_name_edit.setValidator(QRegularExpressionValidator(regex, self._prj_name_edit))

        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self._prj_name_edit, stretch=1)
        layout.addLayout(row)
        layout.addWidget(self._button_box)
        self.setLayout(layout)
        self.setMinimumSize(QSize(400, 200))
        self.setMaximumSize(QSize(400, 200))

    def select_model_name(self, existing_models: List[RxModelID]) -> Optional[str]:
        """
        Raise modal dialog to query user for a project name for a new ML model.
        :param existing_models: List of existing models in the active workspace. The project name for the new model
            cannot match any of these.
        :return: The candiate project name for the new ML model, or None if user canceled operation.
        """
        self._existing_project_names.clear()
        self._existing_project_names.extend([m.project_name for m in existing_models])

        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._prj_name_edit.setText("")
        self._prj_name_edit.setPlaceholderText("ReachXModel2")

        self._prj_name_edit.textChanged.connect(self._on_text_changed)
        res: int = self.exec()
        self._prj_name_edit.textChanged.disconnect(self._on_text_changed)

        return self._prj_name_edit.text() if res == QDialog.DialogCode.Accepted else None

    @Slot(str)
    def _on_text_changed(self, text: str) -> None:
        enable = (5 <= len(text) <= 20) and (text not in self._existing_project_names)
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enable)


class ModelView(BaseView):
    """
    This view displays information about the currently loaded body part detection ML model. It provides controls to
    let the user choose from among the set of available models in the active workspace, create a new ML model; select
    among the existing iterations of the selected model, create a new iteration, and delete the currently selected
    iteration. A tab panel in the view shows the annotated training data for the currently selected model iteration,
    which is the collecction of body part markers attached to one or more "marked" experiment sessions.
    """

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__('Body Part Detection Model', None, data_manager)

        self._available_models: List[RxModelID] = list()
        """ Internal cache of identifiers for all available ML models in the current active workspace. """
        self._curr_model: Optional[RxModelID] = None
        """ Identifier for the currently loaded ML model; None if no model is loaded. """

        self._sel_model_combo = QComboBox()
        """ Selects the currently active body part detection ML model. """
        self._new_model_btn = QPushButton(RxIcons.NEW, "")
        """ Pressing this button triggers creation of a new ML model. """
        self._del_model_btn = QPushButton(RxIcons.TRASH, "")
        """ Pressing this button deletes the currently active ML model entirely (with user confirmation). """
        self._sel_iter_combo = QComboBox()
        """ Selects the current iteration for the currently active body part detection ML model. """
        self._new_iter_btn = QPushButton(RxIcons.NEW, "")
        """ Pressing this button triggers creation of a new iteration of the currently active ML model. """
        self._del_iter_btn = QPushButton(RxIcons.TRASH, "")
        """ Pressing this button deletes the current iteration of the currently active ML model (with confirmation). """
        self._train_btn = QPushButton("Train")
        """ 
        Pressing this button raises a modal progress dialog and launches a background task to train the active 
        model iteration.
        """
        self._net_type_combo = QComboBox()
        """ Selects the neural net architecture on which the model is trained. """

        self._tab_widget = QTabWidget()
        """ Tab widget organizes panels displaying information for the active iteration of the current ML model. """
        self._markers_tab = _MarkersTab(self)
        """ 
        All annotated training data for the selected iteration of the current model -- body part markers attached to one
        or more experiment sessions -- are exposed on this tab panel.
        """
        self._get_model_name_dlg = _GetModelNameDialog(self._new_model_btn)
        """ Simple reusable dialog to query user for the project name when creating a new ML model. """

        self._sel_model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._sel_model_combo.currentTextChanged.connect(self._on_model_selected)
        self._new_model_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._new_model_btn.setStyleSheet("QPushButton { border: none; }")
        self._new_model_btn.setToolTip("Create and load a new detection model")
        self._new_model_btn.clicked.connect(self._on_create_new_model)
        self._new_model_btn.setEnabled(False)
        self._del_model_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._del_model_btn.setStyleSheet("QPushButton { border: none; }")
        self._del_model_btn.setToolTip("Delete this BP detection model permanently")
        self._del_model_btn.clicked.connect(self._on_delete_curr_model)
        self._del_model_btn.setEnabled(False)
        self._sel_iter_combo.currentTextChanged.connect(self._on_iteration_selected)
        self._new_iter_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._new_iter_btn.setStyleSheet("QPushButton { border: none; }")
        self._new_iter_btn.setToolTip("Create and load a new model iteration (copies BP markers from current iteration")
        self._new_iter_btn.clicked.connect(self._on_create_new_iteration)
        self._new_iter_btn.setEnabled(False)
        self._del_iter_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._del_iter_btn.setStyleSheet("QPushButton { border: none; }")
        self._del_iter_btn.setToolTip("Delete this model iteration permanently")
        self._del_iter_btn.clicked.connect(self._on_delete_curr_iteration)
        self._del_iter_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._on_start_training)
        self._train_btn.setEnabled(False)
        self._train_btn.setToolTip("Train/retrain current model iteration using selected neural net architecture")
        self._net_type_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._net_type_combo.addItems([str(t) for t in RxNetworkType])

        self._tab_widget.addTab(self._markers_tab, "Training Data")

        main_layout = QGridLayout()
        main_layout.setSpacing(1)
        grp = QHBoxLayout()
        grp.setSpacing(5)
        grp.addWidget(QLabel("Model"))
        grp.addWidget(self._sel_model_combo, stretch=1)
        grp.addWidget(self._new_model_btn)
        grp.addWidget(self._del_model_btn)
        grp.addSpacing(50)
        main_layout.addLayout(grp, 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grp = QHBoxLayout()
        grp.addStretch(1)
        grp.addWidget(QLabel("Iteration"))
        grp.addWidget(self._sel_iter_combo)
        grp.addWidget(self._new_iter_btn)
        grp.addWidget(self._del_iter_btn)
        main_layout.addLayout(grp, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        grp = QHBoxLayout()
        grp.setSpacing(5)
        grp.addWidget(self._train_btn)
        grp.addWidget(QLabel("Neural Net"))
        grp.addWidget(self._net_type_combo)
        main_layout.addLayout(grp, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(2)
        separator.setMinimumHeight(20)
        main_layout.addWidget(separator, 2, 0, 1, 2)
        main_layout.addWidget(self._tab_widget, 3, 0, 1, 2)
        main_layout.setColumnStretch(0, 1)
        main_layout.setRowStretch(3, 1)
        self.view_container.setLayout(main_layout)

        # connect to relevant signals from DataManager, ModelManager
        self.data_manager.active_workspace_switched.connect(self._reload)
        self.data_manager.active_workspace_path_changed.connect(self._reload)
        self.data_manager.model_loaded.connect(self._reload)
        self.data_manager.model_manager.iter_loaded.connect(self._on_iter_loaded)
        self.data_manager.model_manager.markers_changed.connect(self._on_markers_changed)

        self._reload()

    @Slot()
    def _on_start_training(self) -> None:
        """ Initiate long-running background task to train the current model iteration. """
        if self.data_manager.model_manager.can_train():
            if self.data_manager.model_manager.is_trained():
                btn = QMessageBox.question(
                    self._del_model_btn, "", "Are you sure you want to retrain this model?\nRetraining will "
                                             "overwrite the previously trained model's weights.")
                if btn == QMessageBox.StandardButton.No:
                    return
            self.data_manager.train_current_model(net_type=RxNetworkType(self._net_type_combo.currentText()))

    @Slot()
    def _reload(self) -> None:
        self._available_models.clear()
        self._available_models.extend(self.data_manager.existing_models)
        self._curr_model = self.data_manager.model_manager.id

        # repopulate model selection combo with project names for all available models
        self._sel_model_combo.currentTextChanged.disconnect(self._on_model_selected)
        self._sel_model_combo.clear()
        if len(self._available_models) > 0:
            self._sel_model_combo.addItems([m.project_name for m in self._available_models])
            if self._curr_model in self._available_models:
                self._sel_model_combo.setCurrentIndex(self._available_models.index(self._curr_model))
            else:
                self._curr_model = None
        self._sel_model_combo.currentTextChanged.connect(self._on_model_selected)

        self._reload_iteration_select_combo()

        # refresh enable state of the pushbuttons and reload the Markers tab content
        self._new_model_btn.setEnabled(self.data_manager.can_create_model())
        self._del_model_btn.setEnabled(self.data_manager.can_delete_current_model())
        self._new_iter_btn.setEnabled(self._curr_model is not None)
        self._del_iter_btn.setEnabled(self._sel_iter_combo.currentIndex() > -1)
        self._train_btn.setEnabled(self.data_manager.model_manager.can_train())
        self._markers_tab.reset()

    def _reload_iteration_select_combo(self) -> None:
        """
        Helper method reloads the contents of the iteration selector to ensure it is synced with the list of
        available iterations for the current model, and the identity of the currently loaded iteration.
        """
        self._sel_iter_combo.currentTextChanged.disconnect(self._on_iteration_selected)
        self._sel_iter_combo.clear()
        if self._curr_model is not None:
            self._sel_iter_combo.addItems(
                [str(iter_num) for iter_num in self.data_manager.model_manager.existing_iterations])
            curr_iter = self.data_manager.model_manager.current_iteration
            if curr_iter is not None:
                self._sel_iter_combo.setCurrentText(str(curr_iter))
        self._sel_iter_combo.currentTextChanged.connect(self._on_iteration_selected)

    @Slot()
    def _on_iter_loaded(self) -> None:
        """
        Handler for the signal ModelManager.iter_loaded, indicating the current active iteration of the current
        ML model has changed. This signal is also emitted when a new iteration is created or when the current active
        iteration is removed. In either scenario, a different iteration is loaded and the set of available iterations
        changes, so we need to check and update the state of the iteration selection combo if necessary.
        """
        iterations = self.data_manager.model_manager.existing_iterations
        if len(iterations) != self._sel_iter_combo.count():
            self._reload_iteration_select_combo()
        self._train_btn.setEnabled(self.data_manager.model_manager.can_train())
        self._markers_tab.reset()
        self._load_current_marked_session_if_necessary()

    @Slot(str)
    def _on_model_selected(self, text: str) -> None:
        """
        Handle user selection of a model from the combo box listing all models in the active workspace.

        :param text: The combo box's current text, which is the chosen model's project name.
        """
        if self.data_manager.can_load_model():
            for model_id in self._available_models:
                if model_id.project_name == text:
                    self.data_manager.load_model(model_id)
                    self._load_current_marked_session_if_necessary()

    def _load_current_marked_session_if_necessary(self) -> None:
        """
        If the current marked session selected in the Markers tab does not correspond to the session currently
        loaded for display in ReachX, then load the marked session, displaying the first marked frame initially.
        """
        sesh_id = self._markers_tab.current_marked_session
        if (sesh_id is not None) and (sesh_id != self.data_manager.session_manager.id):
            first_marked = self.data_manager.model_manager.marked_frames_for_session(sesh_id)[0]
            self.data_manager.load_session(sesh_id, first_marked)

    @Slot()
    def _on_create_new_model(self) -> None:
        """ Raise dialog to query user for the new model's project name, then create and load the new model. """
        proj_name = self._get_model_name_dlg.select_model_name(self.data_manager.existing_models)
        if isinstance(proj_name, str):
            self.data_manager.create_and_load_new_model(proj_name)

    @Slot(str)
    def _on_iteration_selected(self, text: str) -> None:
        """
        Handle user selection of a different model iteration from the combo box listing all existing iterations of the
        current ML model.
        :param text: The combo box's current text, whith is the selected iteration number in string form.
        """
        try:
            iter_num = int(text)
            if iter_num in self.data_manager.model_manager.existing_iterations:
                self.data_manager.model_manager.load_iteration(iter_num)
        except Exception:
            pass

    @Slot()
    def _on_create_new_iteration(self) -> None:
        """ Create a new iteration of the current ML model. The new iteration will become the active one. """
        if self._curr_model is not None:
            self.data_manager.model_manager.create_and_load_new_iteration(copy=True)

    @Slot()
    def _on_delete_curr_iteration(self) -> None:
        try:
            iter_num = int(self._sel_iter_combo.currentText())
            if iter_num == self.data_manager.model_manager.current_iteration:
                btn = QMessageBox.question(
                    self._del_iter_btn, "", "Are you sure?\nDestroying an ML model iteration cannot be undone.")
                if btn == QMessageBox.StandardButton.Yes:
                    self.data_manager.model_manager.remove_current_iteration_permanently()
        except Exception:
            pass

    @Slot()
    def _on_delete_curr_model(self) -> None:
        if self.data_manager.can_delete_current_model():
            try:
                btn = QMessageBox.question(
                    self._del_model_btn, "", "Are you sure?\nDestroying an ML model cannot be undone.")
                if btn == QMessageBox.StandardButton.Yes:
                    self.data_manager.delete_current_model()
            except Exception:
                pass

    @Slot()
    def _on_markers_changed(self) -> None:
        """
        Whenever the annotated training data for the active model iteration changes, refresh the enable state of the
        **Train** button. A minimum number of hand and pellet markers must be defined before training is permitted.
        """
        self._train_btn.setEnabled(self.data_manager.model_manager.can_train())


class _MarkerInfoModel(QAbstractItemModel):
    """
    A table model displaying information about the annotated training data for a body part detection ML model. The
    model training data is a collection of body part locations on selected frames from the two required camera videos
    of one or more experiment sessions.

    The table model supports two possible configurations, specified at construction time:
      - A summary of the total # of frames marked, per body part and camera source, across all marked sessions included
        in the training data.
      - A detailed listing of all body part markers defined on a particular marked session. A single `BodyPartMarker`
        consists of the frame number, camera identifier, body part identifier, and its location (x, y) on the frame.

    USAGE:
     - Pass the singleton `ModelManager` to the constructor, and set the model in a QTableView.
     - Call reload() whenever a different model iteration is made the current iteration (including when a different
       model project is loaded), or whenever the marked session changes (for the detailed configuration).
    """
    _DETAIL_COLS = ["Frame", "Cam", "Body Part", "Location"]

    def __init__(self, data_mgr: DataManager, is_summary: bool = True):
        """
        Construct the body part marker information table model.
        :param data_mgr: The singleton encapsulating all data and state in ReachX, including helper objects that
            manage the current body part detection ML model and the current experiment session.
        :param is_summary: If True, the table model presents a summary of the marker counts for each body part type on
            both the two required cameras across all marked sessions in the ML model's annotated training data. Else,
            it lists the frame number, camera source, body part type, and location for each individual marker
            attached to a particular marked session (which is initially None, so model is empty).
        """
        super().__init__()
        self._data_mgr = data_mgr
        """ 
        The singleton data manager is the ultimate source for all information about the current ML model and the
        current experiment session. 
        """
        self._is_summary = is_summary
        """ True for the summary count configuration, False to list all body part markers for a single session. """
        self._marked_sesh_id: Optional[RxSessionID] = None
        """ The identifier for the marked session - for the detailed listing configuration only. """
        self._markers: List[BodyPartMarker] = list()
        """ 
        List of all body part markers defined on a particular marked session, in chronological order. One marker per 
        row in the table model. NOT USED if model configured to display only total counts per body part.
        """
        self._body_parts: List[RxBodyPart] = RxBodyPart.defined_body_parts(False)
        """ List of all defined body parts for a session. Depends on the model installed: fixed-cam or legacy-cam. """
        self._summary_cols: List[str] = ["Body Part", str(RxCam.SIDE), str(RxCam.FRONT)]
        """ Colummn labels for the summary count config. Last two are different for fixed- vs legacy-cam model. """

    @property
    def marked_session_id(self) -> Optional[RxSessionID]:
        """
        For detailed configuration of the table model, this is the identifier to the marked session for which all
        attached body part markers are listed, or None if no marked session installed in model. For the summary count
        configuration of the table model, this is irrelevant and is always None.
        """
        return self._marked_sesh_id

    def set_marked_session(self, marked_sesh: Optional[RxSessionID]) -> None:
        """
        Set the identifier of the marked experiment session for which all attached body part markers should be listed
        by the "detailed" configuration of this table model. This method has no effect for the summary configuration.
        :param marked_sesh: The session ID. If None, then there is no marked session and the table model will be empty.
        """
        if not self._is_summary:
            self.beginResetModel()
            self._marked_sesh_id = marked_sesh
            self._markers.clear()
            if marked_sesh is not None:
                model_mgr = self._data_mgr.model_manager
                for frame in model_mgr.marked_frames_for_session(marked_sesh):
                    self._markers.extend(model_mgr.get_markers_for_frame(marked_sesh, frame))
            self.endResetModel()

    def reload(self, sesh_updated: Optional[RxSessionID] = None) -> None:
        """
        Reload the table model. Intended to be called when the collection of body part markers attached to the current
        ML model have changed in some way. Behavior depends on the table model configuation.
        - For the summary count configuration, the model is simply reset (no information is cached anyway).
        - For the detailed configuration, the table model is reloaded only if the specified session ID is not None and
          corresponds to the marked session exposed by the table model. To change the session for which markers are
          displayed by the model, call `set_marker_session()` instead.
        :param sesh_updated: Identifier of the marked session that has had a body part marker attached or detached.
            Ignored if table model is in the summary configuration.
        :return:
        """
        if self._is_summary:
            self.beginResetModel()
            self._body_parts.clear()
            fixed = self._data_mgr.model_manager.is_fixed_cam
            self._body_parts = RxBodyPart.defined_body_parts(fixed)
            self._summary_cols[1] = str(RxCam.LEFT) if fixed else str(RxCam.SIDE)
            self._summary_cols[2] = str(RxCam.RIGHT) if fixed else str(RxCam.FRONT)
            self.endResetModel()
        elif not self._is_summary and (sesh_updated is not None) and (self._marked_sesh_id == sesh_updated):
            self.beginResetModel()
            self._markers.clear()
            model_mgr = self._data_mgr.model_manager
            for frame in model_mgr.marked_frames_for_session(self._marked_sesh_id):
                self._markers.extend(model_mgr.get_markers_for_frame(self._marked_sesh_id, frame))
            self.endResetModel()

    def get_row_for_current_frame_if_marked(self) -> int:
        """
        For the configuration exposing all body part markers on a given marked session: If the current session on
        display is this marked session, and the current displayed frame corresponds to a body part-marked frame from
        that session, then return the model row of the first defined body part marker on that frame.
        :return: The row number, or -1 if the marked session for the table model is not the current session, or if
            the current displayed frame has no attached body part markers.
        """
        out = -1
        if (not self._is_summary) and (self._marked_sesh_id == self._data_mgr.session_manager.id):
            curr = self._data_mgr.session_manager.current_frame_num
            out = next((i for i, m in enumerate(self._markers) if m.frame == curr), -1)
        return out

    @Slot(QModelIndex)
    def on_cell_clicked_event(self, table_index: QModelIndex) -> None:
        """
        Request that the session manager go to the frame number on the row clicked. The marked  session is loaded if it
        is not already the current displayed session. Not applicable to the summary version of this model.

        No action taken if the current session is in an active playback state.

        :param table_index: Table index that was clicked; should correspond to a particular body part marker
        """
        # do nothing if playback in progress or if this is the summary table model configuration
        sesh_mgr = self._data_mgr.session_manager
        if sesh_mgr.playback_in_progress or self._is_summary:  # or (sesh_mgr.id != self._marked_sesh_id):
            return

        # get the frame number for the row clicked
        frame_num = -1
        row = table_index.row()
        if 0 <= row < len(self._markers):
            frame_num = self._markers[row].frame

        # if the currently displayed session is NOT the marked session exposed in the table, then we have to load that
        # session with the desired initial frame; else, just update current frame for the current session
        if frame_num > -1:
            if sesh_mgr.id != self._marked_sesh_id:
                self._data_mgr.load_session(self._marked_sesh_id, frame_num)
            else:
                sesh_mgr.go_to_frame(frame_num)

    def rowCount(self, /, parent=...) -> int:
        return len(self._body_parts) if self._is_summary else len(self._markers)

    def columnCount(self, /, parent=...):
        return len(self._summary_cols if self._is_summary else self._DETAIL_COLS)

    def data(self, model_idx, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole:
            model_mgr = self._data_mgr.model_manager
            out = ''
            row, col = model_idx.row(), model_idx.column()
            if 0 <= row < self.rowCount() and 0 <= col < self.columnCount():
                if self._is_summary:
                    bp = self._body_parts[row]
                    cams = [RxCam.LEFT, RxCam.RIGHT] if model_mgr.is_fixed_cam else [RxCam.SIDE, RxCam.FRONT]
                    out = str(bp if col == 0 else model_mgr.num_markers(cams[col-1], bp))
                else:
                    marker = self._markers[row]
                    if col < 3:
                        out = str(marker.frame if col == 0 else (marker.cam if col == 1 else marker.part))
                    else:
                        out = f"({marker.x_pix},{marker.y_pix})"
            return out
        elif role == Qt.ItemDataRole.CheckStateRole:
            return None
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(self, section: int, orientation, /, role=...):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            header: List[str] = self._summary_cols if self._is_summary else self._DETAIL_COLS
            return header[section]
        else:
            return super().headerData(section, orientation, role)

    def index(self, row, column, /, parent=...):
        return self.createIndex(row, column)

    # Pycharm bug: Misses the pure virtual parent(self, index) in QAbstractItemModel
    # noinspection PyMethodOverriding
    def parent(self, idx: QModelIndex) -> QModelIndex:
        return QModelIndex()


class _MarkersTab(QWidget):
    """
    This tab panel displays annotated training data for the current iteration of the body part detection ML model
    currently loaded in ReachX. That data consists of the collection of all body part markers defined on selected
    frames from the required videos recorded during one or more experiment sessions.

    Along the top of the panel is a combo box for selecting a marked session, and all body part markers attached to that
    session are displayed in the table immediately below the combo box. That table has 4 columns: "Frame", "Cam",
    "Body Part", and "Location". The last column displays the (X,Y) location of the body part marker in that row.

    To the right of the combo box are two pushbuttons:
     - "Discard all": Discard all training data for the current iteration of the loaded ML model.
     - "Discard session": Discard the currently selected marked session from the training data.

    Below the "detail table" is a pair of pushbuttons by which the user can switch to the next or previous marked
    frame in the training data. Since that data can include marked frames across multiple experiment sessions, pressing
    the "next marked frame" button, for example, could involve loading a different session for display. The notion of
    next or previous marked frame is always with respect to the current frame of the currently displayed session, which
    is not necessarily one of the marked sessions in the training data. For more information, see the `ModelManager`
    method `get_next_marked_frame_starting_at()`. Note that the two pushbuttons are associated with global application
    shortcuts: **PageUp** = go to previous marked frame; **PageDown** = go to next marked frame.

    At the bottom of the tab panel is a summary table view that shows the body part marker counts per body part type and
    camera source across all marked sessions included in the training data.

    See class `_MarkerInfoModel`, which serves at the table model for both table views described above.
    """
    def __init__(self, parent: ModelView):
        super().__init__()
        self._data_mgr = parent.data_manager
        """ 
        Keep a reference to the ReachX data manager. We'll need this, eg, to switch among the marked sessions 
        comprising the current ML model's annoated training data.
        """
        self._marked_sesh_ids: List[RxSessionID] = list()
        """ 
        Cached copy of the session identifiers for all marked sessions in the annotated training data for the 
        active iteration of the currently loaded ML model. Will be empty if no model is loaded, or if the training data
        is empty.
        """

        self._sesh_combo = QComboBox()
        """ Combo box selects the marked session for which body part markers are listed in the details table view. """
        self._discard_all_btn = QPushButton("Discard All")
        """ Pressing this button discards all training data for the current iteration of the current ML model. """
        self._discard_sesh_btn = QPushButton(RxIcons.TRASH, "")
        """ 
        Pressing this button removes all markers for the current session from the training data for the current
        iteration of the current ML model.
        """
        self._marker_counts_info = _MarkerInfoModel(self._data_mgr, True)
        """ 
        Table model summarizing body part marker counts across all marked sessions in the annotated training data
        for the current iteration of the current ML model. 
        """
        self._marked_session_info = _MarkerInfoModel(self._data_mgr, False)
        """ Model exposes all body part markers attached to a given marked session, in chrono order by frame number. """
        self._detail_table = QTableView()
        """ The table view displaying all body part locations attached to the currently selected marked session. """
        self._next_mark_btn = QPushButton(RxIcons.NEXT, "")
        """ Advances to the next marked frame. """
        self._prev_mark_btn = QPushButton(RxIcons.PREV, "")
        """ Rewinds to the previous marked frame. """

        self._discard_all_btn.clicked.connect(self._discard_all_markers)
        self._discard_all_btn.setToolTip("Discard ALL training data and start over")
        self._discard_all_btn.setEnabled(False)

        self._discard_sesh_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._discard_sesh_btn.setStyleSheet("QPushButton { border: none; }")
        self._discard_sesh_btn.setToolTip("Discard all markers for the marked session selected")
        self._discard_sesh_btn.clicked.connect(self._discard_curr_marked_sesh)
        self._discard_sesh_btn.setEnabled(False)

        self._sesh_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._sesh_combo.currentIndexChanged.connect(self._on_marked_session_selected)

        self._detail_table.setModel(self._marked_session_info)
        self._detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._detail_table.verticalHeader().setVisible(False)
        self._detail_table.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)
        self._detail_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._detail_table.clicked.connect(self._marked_session_info.on_cell_clicked_event)

        self._next_mark_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._next_mark_btn.setStyleSheet("QPushButton { border: none; }")
        self._next_mark_btn.setToolTip("Advance to next marked frame (shortcut: PageUp )")
        self._next_mark_btn.clicked.connect(self._go_to_next_marked_frame)
        self._prev_mark_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._prev_mark_btn.setStyleSheet("QPushButton { border: none; }")
        self._prev_mark_btn.setToolTip("Rewind to previous marked frame (shortcut: PageDown )")
        self._prev_mark_btn.clicked.connect(self._go_to_previous_marked_frame)

        shortcut = QShortcut(QKeySequence(Qt.Key.Key_PageUp), self, self._go_to_next_marked_frame)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_PageDown), self, self._go_to_previous_marked_frame)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)

        button_row = QHBoxLayout()
        button_row.setSpacing(1)
        button_row.addStretch(1)
        button_row.addWidget(QLabel("Marked Frames:"))
        button_row.addWidget(self._prev_mark_btn)
        button_row.addWidget(self._next_mark_btn)

        summary_view = QTableView()
        summary_view.setModel(self._marker_counts_info)
        summary_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        summary_view.verticalHeader().setVisible(False)
        summary_view.setSizeAdjustPolicy(QTableView.SizeAdjustPolicy.AdjustToContents)

        layout = QGridLayout()
        h_grp = QHBoxLayout()
        h_grp.setSpacing(1)
        h_grp.addWidget(self._sesh_combo, stretch=1)
        h_grp.addWidget(self._discard_sesh_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(h_grp, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addLayout(button_row, 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._detail_table, 1, 1, 3, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(summary_view, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._discard_all_btn, 2, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addItem(QSpacerItem(5, 5, QSizePolicy.Policy.MinimumExpanding,
                                   QSizePolicy.Policy.MinimumExpanding), 3, 2)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 0)
        layout.addItem(QSpacerItem(5, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum), 0, 3)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(3, 1)
        layout.addItem(QSpacerItem(0, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding), 4, 0)
        layout.setRowStretch(4, 1)
        self.setLayout(layout)

        self.reset()

        # to update row selection in the detail table whenever the current displayed frame corresponds to a row
        # in that table. REM - the currently displayed session may NOT be a marked session at all!
        self._data_mgr.session_loaded.connect(self._on_frame_ready)
        self._data_mgr.session_manager.frame_ready.connect(self._on_frame_ready)

        # refresh tab panel whenever whenever the annotated training data (aka, the body part markers) changes,
        self._data_mgr.model_manager.markers_changed.connect(self.on_markers_changed)

    def reset(self, focus_sesh: Optional[RxSessionID] = None) -> None:
        """
        Reload the tab panel entirely IAW the current active ML model iteration. **Call this method upon switching to
        a different ML model or a different iteration of the same model.**
        :param focus_sesh: If not None, this is the ID of the marked session that should be selected in the combo box
            and displayed in the detail table. Otherwise, the first marked session (if any) gets the focus.
        """
        self._marked_sesh_ids.clear()
        self._marked_sesh_ids.extend(self._data_mgr.model_manager.marked_sessions)
        has_markers = len(self._marked_sesh_ids) > 0
        focus_idx = -1
        if has_markers:
            focus_idx = 0 if (focus_sesh not in self._marked_sesh_ids) else self._marked_sesh_ids.index(focus_sesh)

        # repopulate combo box with list of marked sessions, if any
        self._sesh_combo.currentIndexChanged.disconnect(self._on_marked_session_selected)
        self._sesh_combo.clear()
        for sesh_id in self._marked_sesh_ids:
            self._sesh_combo.addItem(str(sesh_id))
        if focus_idx > -1:
            self._sesh_combo.setCurrentIndex(focus_idx)
        self._sesh_combo.currentIndexChanged.connect(self._on_marked_session_selected)

        self._marker_counts_info.reload()
        self._marked_session_info.set_marked_session(self._marked_sesh_ids[focus_idx] if has_markers else None)

        self._discard_sesh_btn.setEnabled(has_markers)
        self._discard_all_btn.setEnabled(has_markers)

        self._refresh_marked_frame_nav_buttons()

    @property
    def current_marked_session(self) -> Optional[RxSessionID]:
        """ Identier of the marked session for which body part markers are currently listed on this tab panel. """
        return self._marked_session_info.marked_session_id

    def _refresh_marked_frame_nav_buttons(self) -> None:
        """
        Refresh the enable state of the buttons that navigate to the next or previous marked frame in the
        complete set of body part markers attached to marked sessions as currently displayed on this tab panel.
        """
        curr_sesh = self._data_mgr.session_manager.id
        curr_frame = self._data_mgr.session_manager.current_frame_num

        next_sesh, _ = self._data_mgr.model_manager.get_next_marked_frame_starting_at(curr_sesh, curr_frame)
        prev_sesh, _ = self._data_mgr.model_manager.get_prev_marked_frame_starting_at(curr_sesh, curr_frame)
        self._next_mark_btn.setEnabled(next_sesh is not None)
        self._prev_mark_btn.setEnabled(prev_sesh is not None)

    def _go_to_next_marked_frame(self) -> None:
        curr_sesh = self._data_mgr.session_manager.id
        curr_frame = self._data_mgr.session_manager.current_frame_num
        sesh_id, frame = self._data_mgr.model_manager.get_next_marked_frame_starting_at(curr_sesh, curr_frame)
        if sesh_id is not None:
            self._data_mgr.load_session(sesh_id, frame)

    def _go_to_previous_marked_frame(self) -> None:
        curr_sesh = self._data_mgr.session_manager.id
        curr_frame = self._data_mgr.session_manager.current_frame_num
        sesh_id, frame = self._data_mgr.model_manager.get_prev_marked_frame_starting_at(curr_sesh, curr_frame)
        if sesh_id is not None:
            self._data_mgr.load_session(sesh_id, frame)

    @Slot(RxSessionID)
    def on_markers_changed(self, sesh_id: Optional[RxSessionID]) -> None:
        """
        Handler for the `markers_changed` signal from ModelManager. Possible scenarios:
         - A marker was attached to an experiment session that was not previously marked.
         - A marker was added to an already marked session.
         - A marker was removed from a marked session, and some markers remain.
         - All markers were removed from a marked session, or the last attached marker was removed -- in which case
           that session is no longer a marked session!
         - All marked sessions were removed from the current ML model's training data -- a complete reset.

        :param sesh_id: The identifier of the affected session. A marker may have been attached or detached from that
            session, OR all markers attached to that session were discarded. If a session no longer has any attached
            markers, it is removed from the set of marked sessions for a model iteration. If None, then all marked
            sessions have been removed from the annotated training data for the current ML model iteration.
        """
        if sesh_id is None:
            self.reset()
        if sesh_id not in self._marked_sesh_ids:
            # first marker attached to a session not previously marked
            self.reset(focus_sesh=sesh_id)
        elif sesh_id not in self._data_mgr.model_manager.marked_sessions:
            # last marker removed from a session -- so it's no longer one of the marked sessions! If any marked
            # sessions remain, the reset will put the focus on one of them. Since ther marker removal happened in this
            # view, we update the displayed session to match the marked session that now has the focus
            self.reset(focus_sesh=None)
            sesh_id = self.current_marked_session
            if (sesh_id is not None) and (sesh_id != self._data_mgr.session_manager.id):
                first_marked = self._data_mgr.model_manager.marked_frames_for_session(sesh_id)[0]
                self._data_mgr.load_session(sesh_id, first_marked)

        else:
            # marker added to an already marked session, or removed from a session that still has attached markers
            if sesh_id != self._marked_session_info.marked_session_id:
                self._marked_session_info.set_marked_session(sesh_id)
            else:
                self._marked_session_info.reload(sesh_id)
            self._marker_counts_info.reload()
            self._refresh_marked_frame_nav_buttons()

    @Slot()
    def _on_frame_ready(self) -> None:
        """
        Handler called whenever a session is loaded for display, or whenever the current displayed frame changes.
         - If the currently displayed session is one of the marked sessions, sync the current index of the combo
           box that chooses the marked session; also, ensure the detail table reflects the markers for that session.
         - If the current displayed frame corresponds to a marked frame, select the corresponding row in the detail
           table.
         - Refresh enable state of the marked frame nav buttons.
         - If the currently displayed session is NOT a marked session, do nothing.
        """
        # if the current displayed session is a marked session, make sure the current combo box selection is
        # synced. We have to temporarily disconnect from the combo box currentIndexChanged signal, since we're
        # programmatically changing the index rather than the user doing it.
        curr_sesh = self._data_mgr.session_manager.id
        if curr_sesh not in self._marked_sesh_ids:
            return

        idx = self._marked_sesh_ids.index(curr_sesh)
        if idx != self._sesh_combo.currentIndex():
            self._sesh_combo.currentIndexChanged.disconnect(self._on_marked_session_selected)
            self._sesh_combo.setCurrentIndex(idx)
            self._sesh_combo.currentIndexChanged.connect(self._on_marked_session_selected)

        if curr_sesh != self.current_marked_session:
            self._marked_session_info.set_marked_session(curr_sesh)

        row = self._marked_session_info.get_row_for_current_frame_if_marked()
        if row >= 0:
            self._detail_table.selectRow(row)
            self._detail_table.scrollTo(
                self._marked_session_info.index(row, 0), QTableView.ScrollHint.PositionAtCenter)
        self._refresh_marked_frame_nav_buttons()

    @Slot()
    def _discard_all_markers(self) -> None:
        if self._data_mgr.model_manager.has_marked_sessions:
            self._data_mgr.model_manager.discard_all_markers()

    @Slot()
    def _discard_curr_marked_sesh(self) -> None:
        focus_sesh = self._marked_session_info.marked_session_id
        if isinstance(focus_sesh, RxSessionID):
            self._data_mgr.model_manager.discard_all_markers(focus_sesh)

    @Slot(int)
    def _on_marked_session_selected(self, index: int) -> None:
        """
        Handler called whenever user selects a different marked session from the combo box on this tab panel. The
        table listing all body part markers for the current selected session is updated accordingly. In addition,
        the session is loaded for the display, with the initial frame set to to the first marked frame in that session.

        :param index: The selected index, or -1 if the current index was reset or the combo box is empty.
        """
        if index == -1:
            self._marked_session_info.set_marked_session(None)
        elif 0 <= index < len(self._marked_sesh_ids):
            sesh_id = self._marked_sesh_ids[index]
            first_marked_frame = self._data_mgr.model_manager.marked_frames_for_session(sesh_id)[0]
            self._data_mgr.load_session(sesh_id, first_marked_frame)
            self._marked_session_info.set_marked_session(sesh_id)
            self._on_frame_ready()  # to ensure row is selected and marked frame nav buttons are refreshed
