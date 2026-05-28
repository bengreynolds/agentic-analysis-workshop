from importlib import resources as impresources
from typing import Optional

from PySide6.QtCore import QObject, QCoreApplication, QSize, Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QMessageBox, QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout, QMenu, \
    QApplication, QDockWidget, QCheckBox

import reachx.assets as reach_assets
from reachx.gui.analysisview import AnalysisWindow
from reachx.config.app_log import get_application_logger
from reachx.gui.logview import LogView
from reachx.gui.modelview import ModelView
from reachx.gui.othercamview import OtherCamView
from reachx.gui.sessionview import SessionView
from reachx.gui.helpview import HelpView
from reachx.gui.maincamview import MainCamView
from reachx.gui.workspaceview import WorkspaceView
from reachx.common import RX, RxPreferences
from reachx.data.datamanager import DataManager

_RX = RX()
""" Application-wide constants. """


class _PreferencesDlg(QDialog):
    """
    Modal dialog by which user view and/or updates selected application-wide preferences.

    See ``RxPreferences`` for the collection of user preferences exposed in this dialog.
    """
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)

        self._use_gpu_chk = QCheckBox("Use GPU for session analysis (if GPU support available)")
        """ Checkbox enables use of GPU when analyzing session video with a trained body part detection model. """
        self._enable_pp_chk = QCheckBox("Analyze a session videos in parallel (rather than sequentially)")
        """ Checkbox enables parallel analysis of session videos in two separate processes. """

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self._use_gpu_chk)
        layout.addWidget(self._enable_pp_chk)
        layout.addWidget(button_box)
        self.setLayout(layout)

        self.setWindowTitle("Edit Preferences")

    def edit_preferences(self, prefs: RxPreferences) -> RxPreferences:
        """
        Raise this dialog to edit the user's current ReachX preferences.

        :param prefs: Current preferences.
        :return: Updated preferences (may or may not have been changed).
        """
        assert isinstance(prefs, RxPreferences)

        self._use_gpu_chk.setChecked(prefs.use_gpu)
        self._enable_pp_chk.setChecked(prefs.enable_pp)

        self.exec()

        return RxPreferences(use_gpu=self._use_gpu_chk.isChecked(), enable_pp=self._enable_pp_chk.isChecked())


class ViewManager(QObject):
    """
    The application's model view controller.

    It constructs and manages views in the main application window. It maintains and passes all views a reference to the
    ``DataManager`` (aka, the "model"), the source for all application configuration, data and state displayed in views.
    """

    def __init__(self, main_window: QMainWindow, mgr: DataManager):
        super().__init__(None)
        self._main_window = main_window
        """ Reference to main application window -- to update standard UI elements like status bar and window title."""
        self._data_manager = mgr
        """ 
        Reads, writes, generates, caches and otherwise manages all data sources displayed in the application. It
        also encapulates application configuration, including workspaces.
        """
        self._about_dlg = self._create_about_dialog()
        """ The application's 'About' dialog. Created once and reused throughout application runtime. """
        self._prefs_dlg = _PreferencesDlg(main_window)
        """ The applications's 'Preferences' dialog. Created once and reused throughout application runtime. """
        self._traj_analysis_window = AnalysisWindow(main_window, mgr, self._on_trajectory_analysis_window_closed)
        """ Secondary application window housing the reach trajectory analysis view. """

        self._main_cam_view = MainCamView(self._data_manager)
        """ 
        The primary application view, housing the side and front camera views. This view serves as the central
        widget in the main window and cannot be hidden. 
        """
        self._session_view = SessionView(self._data_manager)
        """ View manages widget that selects the current loaded session and shows stats for that session. """
        self._model_view = ModelView(self._data_manager)
        """ View for selecting, updating and using a body part detection ML model defied in the active workspace. """
        self._workspace_view = WorkspaceView(self._data_manager)
        """ Application workspace view. """
        self._help_view = HelpView(self._data_manager)
        """ A read-only view encapsulating a very brief user guide for the application. """
        self._othercams_view = OtherCamView(self._data_manager)
        """ Non-interactive view housing the 'stim' and 'fast' cam views. Rarely used. """
        self._log_view = LogView(self._data_manager)
        """ View housing the application log. """

        self._all_views = [self._main_cam_view, self._model_view, self._session_view, self._workspace_view,
                           self._othercams_view, self._log_view, self._help_view]
        """ List of all views housed in docking widgets within the main application window. """

        # actions and menus
        self._quit_action: Optional[QAction] = None
        self._about_action: Optional[QAction] = None
        self._about_qt_action: Optional[QAction] = None
        self._prefs_action: Optional[QAction] = None
        self._undo_reach_op: Optional[QAction] = None
        """ Action undoes most recent add/delete/replace operation on the current session's curated reaches. """
        self._traj_analysis_window_toggle_action: Optional[QAction] = None
        """ Action toggles the visible state of the trajectory analysis window. """

        self._file_menu: Optional[QMenu] = None
        self._view_menu: Optional[QMenu] = None
        self._edit_menu: Optional[QMenu] = None
        self._help_menu: Optional[QMenu] = None

        self._construct_ui()

        # connect to signals from our data manager
        self._data_manager.active_workspace_switched.connect(self.on_workspace_switch)
        self._data_manager.session_manager.reaches_changed.connect(self._on_update_undo_action)
        self._data_manager.session_loaded.connect(self._on_update_undo_action)
        self._main_window.setMinimumSize(800, 600)
        self._restore_from_settings()
        self._main_window.setWindowTitle(self.main_window_title)

    def _construct_ui(self) -> None:
        """
        Builds and lays outs the UI within the main application window. This method must be called at startup, prior to
        showing the window and prior to restoring its state from user settings. It creates and configures UI actions and
        menus, installs the primary camera view as the central widget in the application window, and installs all other
        views in dock widgets.
            By default, all views are initially docked along the right edge, but the user can dock a view to any edge,
        or float the view in a separate window. Nesting is permitted. Each dock widget is assigned a unique name
        '<title>-DOCK', where <title> is the title of the view it contains, so that its state can be saved to and
        restored from user settings. Hence it is critical to set up the dock widgets before restoring the application
        window's state from those settings!
        """
        get_application_logger().info("Building UI.")

        self._main_window.setCentralWidget(self._main_cam_view.view_container)

        # actions
        self._quit_action = QAction("&Quit", parent=self._main_window, statusTip=f"Quit {_RX.APP_NAME}",
                                    shortcut=QKeySequence("Ctrl+Q"))
        # noinspection PyUnresolvedReferences
        self._quit_action.triggered.connect(self.quit)

        self._about_action = QAction("&About", parent=self._main_window, statusTip=f"About {_RX.APP_NAME}")
        # noinspection PyUnresolvedReferences
        self._about_action.triggered.connect(self._about)

        self._about_qt_action = QAction("About Qt", parent=self._main_window, statusTip="About the Qt library")
        # noinspection PyUnresolvedReferences
        self._about_qt_action.triggered.connect(QApplication.aboutQt)

        self._prefs_action = QAction("&Preferences...", parent=self._main_window,
                                     statusTip=f"Edit user preferences for {_RX.APP_NAME}")
        # noinspection PyUnresolvedReferences
        self._prefs_action.triggered.connect(self._edit_preferences)

        self._undo_reach_op = QAction("Undo", parent=self._main_window, statusTip="Undo",
                                      shortcut=QKeySequence("Ctrl+Z"))
        # noinspection PyUnresolvedReferences
        self._undo_reach_op.triggered.connect(self.undo)

        self._traj_analysis_window_toggle_action = QAction("Trajectory Analysis", parent=self._main_window)
        self._traj_analysis_window_toggle_action.setCheckable(True)
        # noinspection PyUnresolvedReferences
        self._traj_analysis_window_toggle_action.triggered.connect(self._toggle_traj_analysis_window)

        # menus - note that I couldn't get tool tips to show for menu actions on MacOS.
        # the two File menu items end up in the Application menu on MacOS
        self._file_menu = self._main_window.menuBar().addMenu("&File")
        self._file_menu.addAction(self._prefs_action)
        self._file_menu.addSeparator()
        self._file_menu.addAction(self._quit_action)

        # the View menu controls the visibility of all dockable views.
        # Note that the first view, the main camera view, is the central widget and NOT a dockable view
        self._view_menu = self._main_window.menuBar().addMenu("&View")
        self._view_menu.addAction(self._traj_analysis_window_toggle_action)
        self._view_menu.addSeparator()

        for v in self._all_views[1:]:
            dock = QDockWidget(v.title, self._main_window)
            dock.setStyleSheet("""
                QDockWidget::title {
                    background: lightsteelblue;
                    text-align: center;
                    border-bottom: 1px solid steelblue
                }
            """)
            dock.setObjectName(f"{v.title}-DOCK")
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setWidget(v.view_container)
            self._main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

            # the OtherCamView and LogView are hidden by default
            if isinstance(v, OtherCamView) or isinstance(v, LogView):
                dock.setHidden(True)
            # dock widget holding the user guide is separated from the other views and hidden by default
            if isinstance(v, HelpView):
                self._view_menu.addSeparator()
                dock.setHidden(True)

            self._view_menu.addAction(dock.toggleViewAction())

        self._main_window.setDockNestingEnabled(True)
        self._main_window.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

        # not much of an "Edit" menu -- just exposes ability to undo last operation on current session's curated reaches
        self._edit_menu = self._main_window.menuBar().addMenu("&Edit")
        self._edit_menu.addAction(self._undo_reach_op)

        self._main_window.menuBar().addSeparator()

        self._help_menu = self._main_window.menuBar().addMenu("&Help")
        # under the hood, these are automatically put in the "Apple" menu on Mac OS X
        self._help_menu.addAction(self._about_action)
        self._help_menu.addAction(self._about_qt_action)

        # status bar
        self._main_window.statusBar().showMessage("Ready")

    def _create_about_dialog(self) -> QDialog:
        """
        Helper method creates the application's simple **About** dialog, which simply display the contents of a
        markdown file in the application asses folder.
        """
        dlg = QDialog(self._main_window)
        dlg.setWindowTitle(f"About {_RX.APP_NAME}")

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(dlg.accept)

        # noinspection PyTypeChecker
        inp_file = (impresources.files(reach_assets) / 'about.md')
        with inp_file.open("r") as f:
            markdown = f.read()
        about_browser = QTextBrowser()
        about_browser.setReadOnly(True)
        about_browser.setOpenExternalLinks(True)
        about_browser.setMarkdown(markdown)

        layout = QVBoxLayout()
        layout.addWidget(about_browser)
        layout.addWidget(button_box)
        dlg.setLayout(layout)
        dlg.setMinimumSize(QSize(600, 400))
        return dlg

    def _about(self) -> None:
        """
        Handler for the 'About <application name>' menu command. It raises a modal message dialog describing the
        application.
        """
        self._about_dlg.exec()

    @Slot()
    def _edit_preferences(self) -> None:
        self._data_manager.preferences = self._prefs_dlg.edit_preferences(self._data_manager.preferences)

    def _restore_from_settings(self) -> None:
        """
        Helper method called at application startup to restore certain login-user-specific preferences -- including the
        GUI layout and the most recently used workspace -- that were saved when the application last exited.

        **This must not be called until after the main window and all views have been realized (but not shown).**
        """
        geometry, window_state = self._data_manager.gui_state
        if not geometry.isEmpty():
            self._main_window.restoreGeometry(geometry)
        else:
            self._main_window.setGeometry(200, 200, 800, 600)

        if not window_state.isEmpty():
            self._main_window.restoreState(window_state)

        geometry, window_state, show = self._data_manager.trajectory_window_settings
        if not geometry.isEmpty():
            self._traj_analysis_window.restoreGeometry(geometry)
        else:
            self._traj_analysis_window.setGeometry(300, 300, 800, 600)

        if not window_state.isEmpty():
            self._traj_analysis_window.restoreState(window_state)

        if show:
            self._toggle_traj_analysis_window(checked=True)
            self._traj_analysis_window_toggle_action.setChecked(True)

    def _save_to_settings(self) -> None:
        """
        Helper method called just prior to application exit to persist certain login-user-specific preferences --
        including the GUI layout.
        """
        self._data_manager.gui_state = (self._main_window.saveGeometry(), self._main_window.saveState(version=0))
        self._data_manager.trajectory_window_settings = \
            (self._traj_analysis_window.saveGeometry(), self._traj_analysis_window.saveState(version=0),
             self._traj_analysis_window.isVisible())

    @property
    def main_window_title(self) -> str:
        """
        String to be displayed in the title bar of the main application window. This reflects the name of the current
        workspace (which should always be defined.
        """
        return (f"{_RX.APP_NAME} (Workspace: {self._data_manager.active_workspace.name} "
                f"{'[fixed-cam]' if self._data_manager.active_workspace.is_fixed_cam else ''})")

    def quit(self) -> None:
        """
        Handler for the Exit/Quit menu command. Unless user vetoes the operation, performs some cleanup/shutdown work,
        saves the current workspace settings, and calls exit() on the main application object.
        """
        res = QMessageBox.question(self._main_window, "Exit", "Are you sure you want to quit?")
        if res == QMessageBox.StandardButton.Yes:
            get_application_logger().info("User initiated application shutdown")
            # TODO: prepare for shutdown. Veto if a long-running operation is still in progress??
            self._save_to_settings()
            self._data_manager.on_exit()
            QCoreApplication.instance().exit(0)

    @Slot()
    def on_workspace_switch(self) -> None:
        """ Respond to user changing the active workspace. Here we only update main window title accordingly. """
        self._main_window.setWindowTitle(self.main_window_title)

    @Slot()
    def undo(self) -> None:
        """
        Handler for the Edit->Undo menu command. Triggering it undoes the most recent operation on the current
        session's set of curated reaches.
        """
        self._data_manager.session_manager.undo_last_reach_op()

    @Slot()
    def _on_update_undo_action(self) -> None:
        """
        Whenever a session is loaded or the current session's set of curated reaches changes, update the state
        of the "Undo" action that, when triggered, undoes the most recent changed to the curated set.
        """
        op_text = self._data_manager.session_manager.undo_reach_op_description
        self._undo_reach_op.setEnabled(len(op_text) > 0)
        self._undo_reach_op.setText(op_text if (len(op_text) > 0) else "Undo")

    @Slot(bool)
    def _toggle_traj_analysis_window(self, checked: bool) -> None:
        """
        Shows or hides the secondary application window housing the trajectory analysis view.
        :param checked: If checked, the trajectory analysis window is shown, else it is closed.
        """
        if checked:
            self._traj_analysis_window.show()
            self._traj_analysis_window.raise_()
            self._traj_analysis_window.activateWindow()
        else:
            self._traj_analysis_window.close()

    def _on_trajectory_analysis_window_closed(self) -> None:
        """
        Callable passed to the trajectory analysis window to inform the ViewManager when the user closes that window.
        The corresponding checked item in the **View** menu is updated accordingly.
        """
        self._traj_analysis_window_toggle_action.setChecked(False)
