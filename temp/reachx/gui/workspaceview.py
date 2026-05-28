from pathlib import Path
from typing import Optional, Callable, Tuple

from PySide6.QtCore import Slot, Qt, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator, QFontMetrics
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, \
    QGridLayout, QFileDialog, QDialog, QWidget, QLineEdit, QRadioButton, QButtonGroup, QDialogButtonBox, QLayout

from reachx.config.workspace import Workspace
from reachx.data.datamanager import DataManager
from reachx.gui.baseview import BaseView


class _CreateWSDialog(QDialog):
    """
    Modal dialog by which user creates a new workspace.

    USAGE: Construct the dialog object and call ``create_workspace()`` to raise it. User enters a (unique) name for the
    workspace, and specifies its type (fixed- or legacy-cam) or elects to make a copy of the currently active
    workspace. Once user confirms or cancels out of the dialog, the method returns the required information to
    create the new workspace.
    """
    def __init__(self, check_ws_name: Callable[[str], bool], parent: Optional[QWidget], /):
        super().__init__(parent)
        self._ws_name_edit = QLineEdit()
        """ Single-line edit specifying desired name for the new workspace. """
        self._fixed_radio = QRadioButton("Fixed-cam")
        """ Radio button selected to create a fixed-cam workspace with default paths. """
        self._legacy_radio = QRadioButton("Legacy-cam")
        """ Radio button selected to create a legacy-cam workspace with default paths. """
        self._copy_radio = QRadioButton("Copy active workspace")
        """ Radio button selected to create a copy of the currently active workspace, albeit with a unique name. """
        self._button_box = QDialogButtonBox()
        """ 
        The OK and Cancel buttons for the dialog. We disable OK whenever the workspace name is invalid or
        duplicates an existing workspace.
        """
        self._check_name_fcn = check_ws_name
        """ Callable function which validates a candidate workspace name. """

        self._ws_name_edit.setPlaceholderText("workspace_1")
        self._ws_name_edit.setToolTip("Enter workspace name. Must be unique, 1-25 characters long.\nOnly alphanumeric "
                                      "dharacters and underscare are allowed.")
        validator = QRegularExpressionValidator(QRegularExpression("^[a-zA-Z0-9_]{1,25}$"), self)
        self._ws_name_edit.setValidator(validator)
        self._ws_name_edit.setMaxLength(25)
        self._ws_name_edit.textChanged.connect(self._on_text_changed)

        font_metrics = QFontMetrics(self._ws_name_edit.font())
        min_w = font_metrics.horizontalAdvance("X" * 25) + 20
        self._ws_name_edit.setMinimumWidth(min_w)

        btn_group = QButtonGroup(self)
        self._legacy_radio.setChecked(True)
        btn_group.addButton(self._fixed_radio)
        btn_group.addButton(self._legacy_radio)
        btn_group.addButton(self._copy_radio)

        self._button_box = QDialogButtonBox()
        self._button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        name_row.addWidget(self._ws_name_edit, stretch=1)
        layout.addLayout(name_row, stretch=1)
        layout.addWidget(self._fixed_radio, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._legacy_radio, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._copy_radio, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._button_box)
        self.setLayout(layout)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        self.setWindowTitle("Create new workspace")
        self.setMinimumSize(400, 200)

    @Slot(str)
    def _on_text_changed(self, ws_name: str) -> None:
        """ Disable/enable dialog 'OK' button if the specified workspace name is invalid/valid. """
        enabled = self._check_name_fcn(ws_name)
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)

    def create_workspace(self) -> Tuple[str, int]:
        """
        Raise the modal dialog to get the name for the new workspace, and whether it should be fixed-cam, legacy-cam,
        or a copy of the currently active workspace.
        :return: A 2-tuple ``(ws_name, create_mode)``, where ``ws_name`` is the name for the new workspace. If
            ``create_mode == 0``, make a copy of the active workspace using the new name. If > 0, create a fixed-cam
            workspace with default root paths. If < 0, create a legacy-cam workspace with default root paths. Returns
            ``("", 0)`` if user canceled out of dialog.
        """
        self._ws_name_edit.setText("")   # invalid workspace name, so "OK" button should be disabled
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._copy_radio.setChecked(True)
        if self.exec() != QDialog.DialogCode.Accepted:
            return "", 0
        else:
            ws_name = self._ws_name_edit.text()
            create_mode = 0 if self._copy_radio.isChecked() else 1 if self._fixed_radio.isChecked() else -1
            return ws_name, create_mode


class WorkspaceView(BaseView):
    """
    This view displays and manages all defined application workspaces.

    A **workspace** is a collection of user-specified settings which govern application behavior -- particularly the
    file system root directories where ReachX searches for sessions and body part detection models belonging to the
    workspace. See ``Workspace`` for more information.

    As of v0.6.0, there are two types of workspaces: fixed-cam and legacy-cam. A ReachX fixed-cam workspace only
    contains sessions recorded on a fixed-cam rig, and only allows fixed-cam models trained on fixed-cam sessions.
    Analogously for a legacy-cam workspace.

    The view also maintains the notion of the **current active workspace**. A combo box lets you switch to a different
    workspace -- unless vetoed by ``DataManager``. You can also edit the workspace's session, model, segmentation model,
    and calibration root paths here. The calibration root applies only to a fixed-cam workspace.
    """

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__('Workspaces', None, data_manager)
        self._ws_combo = QComboBox()
        """ 
        Combo box selects the application's current active workspace, the settings of which are edited/displayed in the 
        various tabs in this view. Combo box disabled if a workspace switch is currently prohibited.
        """
        self._create_btn = QPushButton("New")
        """ 
        Push button adds a new workpace and makes it the current active workspace. Disabled if a workspace
        switch is currently prohibited.
        """
        self._delete_btn = QPushButton("Delete")
        """ Push button deletes the current active workspace. Disabled if deletion is not possible. """

        self.session_dir_readout = QLabel()
        """ Label reflects the session root directory for the active workspace. """
        self.session_dir_btn = QPushButton("Change...")
        """ Click to change session root directory for the active workspace. """
        self.model_dir_readout = QLabel()
        """ Label reflects the model root directory for the active workspace. """
        self.model_dir_btn = QPushButton("Change...")
        """ Click to change model root directory for the active workspace. """
        self.segmentation_model_dir_readout = QLabel()
        """ Label reflects the segmentation model root directory for the active workspace. """
        self.segmentation_model_dir_btn = QPushButton("Change...")
        """ Click to change segmentation model root directory for the active workspace. """
        self.calibration_dir_readout = QLabel()
        """ Label reflects the calibration root directory for the active workspace. Empty for legacy-cam workspace. """
        self.calibration_dir_btn = QPushButton("Change...")
        """ Click to change calibration root directory for the active workspace. Disabled for legacy-cam workspace. """

        self._create_ws_dlg = _CreateWSDialog(self.data_manager.is_valid_and_unique_workspace, self._create_btn)
        """ Modal dialog raised to get name and configuration for a new workspace"""

        self._ws_combo.currentIndexChanged.connect(self.on_workspace_selection_changed)
        self._create_btn.clicked.connect(self._on_create_workspace)
        self._delete_btn.clicked.connect(self._on_delete_workspace)

        self.session_dir_readout.setStyleSheet("QLabel { font-weight: bold; }")
        self.session_dir_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self.session_dir_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.session_dir_btn.clicked.connect(lambda: self._on_select_directory(self.session_dir_btn))

        self.model_dir_readout.setStyleSheet("QLabel { font-weight: bold; }")
        self.model_dir_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self.model_dir_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.model_dir_btn.clicked.connect(lambda: self._on_select_directory(self.model_dir_btn))

        self.segmentation_model_dir_readout.setStyleSheet("QLabel { font-weight: bold; }")
        self.segmentation_model_dir_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self.segmentation_model_dir_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.segmentation_model_dir_btn.clicked.connect(lambda: self._on_select_directory(
            self.segmentation_model_dir_btn))

        self.calibration_dir_readout.setStyleSheet("QLabel { font-weight: bold; }")
        self.calibration_dir_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self.calibration_dir_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.calibration_dir_btn.clicked.connect(lambda: self._on_select_directory(self.calibration_dir_btn))

        main_layout = QVBoxLayout()
        control_line = QHBoxLayout()
        control_line.addWidget(self._ws_combo, stretch=2)
        control_line.addWidget(self._create_btn)
        control_line.addWidget(self._delete_btn)
        main_layout.addLayout(control_line)

        grid = QGridLayout()
        grid.addWidget(QLabel("Sessions:"), 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.session_dir_readout, 0, 2)
        grid.addWidget(self.session_dir_btn, 0, 4)
        grid.addWidget(QLabel("Models:"), 1, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.model_dir_readout, 1, 2)
        grid.addWidget(self.model_dir_btn, 1, 4)
        grid.addWidget(QLabel("Segmentation Models:"), 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.segmentation_model_dir_readout, 2, 2)
        grid.addWidget(self.segmentation_model_dir_btn, 2, 4)
        grid.addWidget(QLabel("Calibrations:"), 3, 0, alignment=Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.calibration_dir_readout, 3, 2)
        grid.addWidget(self.calibration_dir_btn, 3, 4)
        grid.setColumnMinimumWidth(1, 5)
        grid.setColumnMinimumWidth(3, 5)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(4, 1)
        main_layout.addLayout(grid)

        self.view_container.setLayout(main_layout)

        self._reload()

    def _refresh(self) -> None:
        """ Refresh enable state of certain widgets depending on current application state. """
        switch_allowed = self.data_manager.can_switch_workspaces()
        self._ws_combo.setEnabled(switch_allowed)
        self._create_btn.setEnabled(switch_allowed)

        self._delete_btn.setEnabled(self.data_manager.can_delete_active_workspace())

        ws = self.data_manager.active_workspace
        self.session_dir_readout.setText(str(ws.session_root.absolute()))
        self.model_dir_readout.setText(str(ws.model_root.absolute()))
        self.segmentation_model_dir_readout.setText(str(ws.segmentation_model_root.absolute()))
        self.calibration_dir_readout.setText(str(ws.calibration_root.absolute()) if ws.is_fixed_cam else "")
        self.calibration_dir_btn.setEnabled(ws.is_fixed_cam)

    def _reload(self) -> None:
        """
        Reload the contents of the workspace view. Call this method at application startup (after configuration load),
        and whenever a workspace configuration is added or removed.
        """
        ws_names = self.data_manager.existing_workspaces
        ws_active = self.data_manager.active_workspace.name

        # on reload, always make sure combo box current selection is the active workspace
        select_index = next((i for i in range(len(ws_names)) if ws_names[i] == ws_active), -1)
        assert select_index > -1

        self._ws_combo.currentIndexChanged.disconnect(self.on_workspace_selection_changed)
        self._ws_combo.clear()
        self._ws_combo.addItems(ws_names)
        self._ws_combo.setCurrentIndex(select_index)
        self._ws_combo.currentIndexChanged.connect(self.on_workspace_selection_changed)

        self._refresh()

    @Slot(int)
    def on_workspace_selection_changed(self, index: int) -> None:
        if index != -1:
            restore = not self.data_manager.switch_to_workspace(self._ws_combo.currentText())
            if restore:
                self._ws_combo.currentIndexChanged.disconnect(self.on_workspace_selection_changed)
                self._ws_combo.setCurrentText(self.data_manager.active_workspace.name)
                self._ws_combo.currentIndexChanged.connect(self.on_workspace_selection_changed)
            self._refresh()

    @Slot()
    def _on_create_workspace(self) -> None:
        """
        Handler requests that the data manager create a new workspace with default settings. If successful, the new
        workspace becomes the active workspace. No action taken if a workspace switch is not currently permissible.
        """
        ws_name, create_mode = self._create_ws_dlg.create_workspace()
        if len(ws_name) == 0:
            return
        if create_mode == 0:
            ok = self.data_manager.create_new_workspace(True, ws_name, False)
        else:
            ok = self.data_manager.create_new_workspace(False, ws_name, create_mode > 0)
        if ok:
            self._reload()

    @Slot()
    def _on_delete_workspace(self) -> None:
        """
        Handler requests that the data manager delete the active workspace, if possible. If successful, the view is
        updated accordingly.
        """
        if self.data_manager.delete_active_workspace():
            self._reload()

    def _on_select_directory(self, btn: QPushButton) -> None:
        """
        Raise a modal dialog by which the user can change the session or model root directory for the current active
        workspace. No action taken if the active workspace may not be edited at this time.

        :param btn: The push button that triiggered this handler. Indicates which root directory to change.
        """
        mgr = self.data_manager
        curr_dir: Path
        snippet: str
        if btn == self.session_dir_btn:
            curr_dir, snippet, key = mgr.active_workspace.session_root, 'session data', Workspace.SESH_ROOT
        elif btn == self.model_dir_btn:
            curr_dir, snippet, key = mgr.active_workspace.model_root, 'body part detection models', Workspace.MODEL_ROOT
        elif btn == self.segmentation_model_dir_btn:
            curr_dir, snippet, key = mgr.active_workspace.segmentation_model_root, 'reach segmentation models', \
                Workspace.SEGMENTATION_MODEL_ROOT
        elif btn == self.calibration_dir_btn:
            curr_dir, snippet, key = mgr.active_workspace.calibration_root, 'calibration files', Workspace.CALIB_ROOT
        else:
            return

        # NOTE: Caption does not appear on native MacOS file dialog
        parent_dir: Path = curr_dir.parent if curr_dir.exists() else Path.home()
        dir_result: str = QFileDialog.getExistingDirectory(
            self._ws_combo, f"Select root directory for {snippet}", str(parent_dir.absolute()))
        if dir_result != "":
            # since we verified we can edit workspace before raising file dialog, this should always work
            mgr.update_active_workspace_path(Path(dir_result), key)
            if key == Workspace.SESH_ROOT:
                self.session_dir_readout.setText(str(mgr.active_workspace.session_root.absolute()))
            elif key == Workspace.MODEL_ROOT:
                self.model_dir_readout.setText(str(mgr.active_workspace.model_root.absolute()))
            elif key == Workspace.SEGMENTATION_MODEL_ROOT:
                self.segmentation_model_dir_readout.setText(
                    str(mgr.active_workspace.segmentation_model_root.absolute()))
            elif key == Workspace.CALIB_ROOT and mgr.active_workspace.is_fixed_cam:
                self.calibration_dir_readout.setText(str(mgr.active_workspace.calibration_root.absolute()))
