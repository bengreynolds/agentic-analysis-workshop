import random
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

from PySide6.QtCore import QObject, Signal, QByteArray, QTimer, Slot
from PySide6.QtWidgets import QMainWindow

from reachx.data.jobs import BlockingJobDialog, NonBlockingJobRunner, RxJob, RxJobType
from reachx.data.modelmgr import ModelManager
from reachx.common import RX, RxSessionID, RxModelID, RxNetworkType, RxPreferences
from reachx.config.app_log import get_application_logger
from reachx.config.confighelper import ConfigHelper, DEF_WORKSPACE_NAME
from reachx.data.sessionmgr import SessionManager
from reachx.config.workspace import Workspace
from reachx.modeling.segmentation.registry import SegmentationModelRecord, discover_segmentation_models, \
    find_segmentation_model, register_external_model

_RX = RX()
""" Application-wide constants. """


class _SessionCache(QObject):
    """
    ``DataManager`` delegate that handles the task of scanning the root session folder for a workspace and finding all
    sessions defined therein.

    In practice, a large number (thousands or more) of experiment sessions may be stored under a workspace's session
    root. By design, there are three directory levels to scan -- date folders, rig folders for a given date, and
    session folders for a given date and rig. Testing has shown this directory scan can take several minutes.

    To address this issue, ``_SessionCache`` runs a background task to scan the session root directory and construct a
    cache of all session folders found. When the task has finished, the cache is persisted in a YAML file in the
    session root directory. The next time the workspace becomes active, the session cache will be loaded from this
    file instead. A background scan is still run in case additional sessions have been added since the YAML file was
    last written.

    IMPORTANT: As of v0.6.0, there are two kinds of sessions -- those run on the new fixed-cam rig, and those run on the
    original legacy-cam rig. A fixed-cam workspace only exposes fixed-cam sessions and models, and likewise for a
    legacy-cam rig. HOWEVER, there's no restriction on the user storing fixed- and legacy-cam sessons under the same
    session root. ``_SessionCache`` will find **all** sessions under a given session root directory, but put the
    two session types in separate lists.
    """
    sessions_cached: Signal = Signal()
    """ A signal to ``DataManager`` that the session cache has been updated. """

    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self._session_root: Optional[Path] = None
        """ The session root folder contents of which are cached. """
        self._sesh_ids_legacy: List[RxSessionID] = list()
        """ The list of legacy-cam sessions found under session root directory.  """
        self._sesh_ids_fixed: List[RxSessionID] = list()
        """ List of fixed-cam sessions found under session root directory."""
        self._noblock_job_runner = NonBlockingJobRunner(main_window.statusBar())
        """ Runs background task to build a session cache and report progress in application status bar. """

    def session_count(self, is_fixed: bool) -> int:
        """
        The number of fixed-cam or legacy-cam sessions cached. May change if a background directory scan is in progress.
        :param is_fixed: ``True/False`` for fixed-cam/legacy-cam session count. """
        return len(self._sesh_ids_fixed if is_fixed else self._sesh_ids_legacy)

    def sessions(self, is_fixed: bool) -> List[RxSessionID]:
        """
        Retrieve a list of session IDs for all ReachX experiment session folders found in the current session root
        directory. This list could grow if a background directory scan is in progress.
        :param is_fixed: ``True/False`` for fixed-cam/legacy-cam sessions found under session root.
        :return: List of session identifiers.
        """
        return (self._sesh_ids_fixed if is_fixed else self._sesh_ids_legacy).copy()

    def reload(self, root: Optional[Path]):
        """
        Reload the session cache to reflect the ReachX experiment sessions found under the session root directory
        specified.

        A background task is launched to load the session cache file (if present) for the given directory, then rescan
        the directory for any changes since that cache file was last written. When the cache file is missing, the
        background job will deliver s partial session ID list after every 100 sessions found. But if the cache file is
        present, the full session list should be available almost immediately.

        :param root: The session root directory.
        """
        if isinstance(self._session_root, Path) and self._session_root.samefile(root):
            return

        # hopefully this is never an issue!
        if not self._noblock_job_runner.cancel_and_wait():
            get_application_logger().warning(
                f"A background job caching sessions under {str(self._session_root)} may have HUNG.")

        self._session_root = root
        self._sesh_ids_fixed.clear()
        self._sesh_ids_legacy.clear()
        if self._session_root is not None:
            job = RxJob(RxJobType.CACHE, session_root=self._session_root)
            job.message_logged.connect(lambda msg: get_application_logger().info(msg))
            job.data_ready.connect(self._on_session_scan_data_ready)
            self._noblock_job_runner.run_job(job)

    @Slot(object)
    def _on_session_scan_data_ready(self, data: object) -> None:
        """ Receives the session ID list from the background job that scans the session root directory."""
        if isinstance(data, list):
            all_ids: List[RxSessionID] = data
            self._sesh_ids_fixed.clear()
            self._sesh_ids_legacy.clear()
            for session_id in all_ids:
                if session_id.is_fixed_cam:
                    self._sesh_ids_fixed.append(session_id)
                else:
                    self._sesh_ids_legacy.append(session_id)
            self.sessions_cached.emit()

    def on_shutdown(self) -> None:
        """
        Prior to application shutdown, cancel the background job that scans the session root directory and wait for it
        to finish (typically, there won't be a running job, but just to be sure).
        """
        self._noblock_job_runner.cancel_and_wait(1000)


class DataManager(QObject):
    """
    The data model manager for ReachX.

    IMPORTANT: Construct the singleton `DataManager` object at application startup and call `DataManager.on_startup()
    to load application configuration settings and initialize state.

    **DEVNOTES**:
     - For now, it encapsulates all data displayed, edited, and generated in the application. Data management tasks are
       divided among several helpers thus far: ConfigHelper, SessionManager, and ModelManager.
     - This is intended as a singleton, so it's OK that the various signals it defines are class attributes.
    """
    active_workspace_switched: Signal = Signal()
    """ A different workspace has become the active workspace. All views should be reset/refreshed accordingly. """
    active_workspace_path_changed: Signal = Signal()
    """ One of the defined data directories for the active workspace may have been changed. """
    session_loaded: Signal = Signal()
    """ Data from a recorded session (video, timestamps, etc) have been loaded and are ready for access by views. """
    model_loaded: Signal = Signal()
    """ A body part detection ML model is loaded and ready for access by views. """
    segmentation_model_loaded: Signal = Signal()
    """ A reach segmentation model is selected and ready for use by views. """
    model_trained: Signal = Signal()
    """ A long-running job has just completed training of the currently active body part detection model. """
    sessions_available: Signal = Signal()
    """ 
    This signal is sent after the session cache has scanned the workspace's session tree and found available 
    sessions. If there are a great many sessions in the tree, this signal will be sent multiple times until all
    detected sessions have been cached.
    """

    def __init__(self):
        super().__init__()
        self._started = False
        """ Flag is set after successful startup. """
        self._cfg = ConfigHelper()
        """ Application configuration helper. """
        self._active_ws: Optional[Workspace] = None
        """ The currently active workspace. """
        self._session_cache: Optional[_SessionCache] = None
        """ 
        Delegate handles scanning a very large session root directory for all available sessions. Created once
        main window is available, as it needs access to the application status bar. 
        """
        self.session_manager = SessionManager()
        """ Loads and manages data, including camera videos, for the current session (if any). """
        self.model_manager = ModelManager()
        """ Loads, updates, and manages the content of the currently loaded body part detection ML model (if any). """
        self._current_segmentation_model: Optional[SegmentationModelRecord] = None
        """ The currently selected externally trained reach segmentation model, if any. """
        self._job_dlg: Optional[BlockingJobDialog] = None
        """ Modal dialog blocks user input while running a time-consuming job on a background thread. """
        self._main_window: Optional[QMainWindow] = None
        """ 
        Reference to main application window. Since DataManager is constructed before the main window, this has to
        be set later.
        """

    def on_startup(self) -> Optional[str]:
        """
        Perform initializations at application startup.

        Application configuration is loaded from the user's settings file. The last used workspace is restored as the
        current active workspace. The most recent session and most recent ML model from that workspace are loaded if
        they exist. Any other application state is initialized accordingly.

        **This method should be called before constructing the user interface, and the application should abort if it
        fails.**
        :return: None if startup tasks were successful; else a brief error description
        """
        if self._started:
            return None
        get_application_logger().info("Starting up.")
        err = self._cfg.load()
        if err is None:
            self._active_ws = self._cfg.get_workspace_by_name(self._cfg.mru_workspace)
            if self._active_ws.session_exists(self._active_ws.mru_session):
                get_application_logger().info(f"Loading MRU session: {str(self._active_ws.mru_session)}")
                self.load_session(self._active_ws.mru_session)
            else:
                get_application_logger().debug(f"MRU session not found: {str(self._active_ws.mru_session)}")
            if self._active_ws.mru_model in self._active_ws.models():
                get_application_logger().info(f"Loading MRU model: {str(self._active_ws.mru_model)}")
                self.load_model(self._active_ws.mru_model)
            else:
                get_application_logger().debug(f"MRU model not found: {str(self._active_ws.mru_model)}")
            self._load_default_segmentation_model()
            self._started = True
        else:
            get_application_logger().critical(f"Failed on startup: {err}")
        return err

    def on_main_window_first_shown(self) -> None:
        """
        When the main application window is first shown after startup, launch a non-blocking background job to scan
        for all sessions under the active workspace's session root folder. When there are a large number of sessions,
        this can take several minutes.

        In addition, emit the ``session_loaded`` and ``model_loaded`` signals if an MRU session and model were
        successfully loaded at startup; this is necessary because, at startup, the UI views do not yet exist. Once the
        main window is raised, all views have been connected to relevant ``DataManager`` signals.
        """
        # noinspection PyUnresolvedReferences
        self._session_cache.sessions_cached.connect(
            lambda: QTimer.singleShot(10, lambda: self.sessions_available.emit()))
        self._session_cache.reload(self._active_ws.session_root)
        if self.session_manager.session_loaded:
            QTimer.singleShot(10, lambda: self.session_loaded.emit())
        if self.model_manager.model_loaded:
            QTimer.singleShot(10, lambda: self.model_loaded.emit())
        if self._current_segmentation_model is not None:
            QTimer.singleShot(10, lambda: self.segmentation_model_loaded.emit())

    def on_exit(self) -> bool:
        """
        Do any necessary cleanup and save application configuration to settings file prior to application exit. Call
        prior to closing the main window, and abort shutdown if this method returns False.
        :return: False if application should not shutdown at this time, else True
        """
        get_application_logger().info("Cleaning up before application shutdown.")
        self._session_cache.on_shutdown()
        self.session_manager.unload()
        self.model_manager.unload()
        self._cfg.save()
        return True

    def set_main_window(self, w: QMainWindow) -> None:
        """
        Save a reference to the main window -- needed to parent the progress dialog that blocks the rest of the
        application while a time-consuming task runs on a background thread.

        **This is a HACK, necessary because the main window is created AFTER the data manager object.**
        :param w: The main application window
        """
        self._main_window = w
        self._session_cache = _SessionCache(w)

    @property
    def gui_state(self) -> Tuple[QByteArray, QByteArray]:
        """
        The main window's geometry and state as last saved in the user's application settings file.
        :return: A 2-tuple containing the window geometry and state, in that order.
        """
        return self._cfg.window_geometry, self._cfg.window_state

    @gui_state.setter
    def gui_state(self, state: Tuple[QByteArray, QByteArray]) -> None:
        """
        Update main window's geometry and state. Be sure to call this prior to application shutdown so that the GUI
        state is persisted to the user's application settings file.
        :param state: A 2-tuple containing the window geometry and state, in that order.
        """
        self._cfg.window_geometry = state[0]
        self._cfg.window_state = state[1]

    @property
    def trajectory_window_settings(self) -> Tuple[QByteArray, QByteArray, bool]:
        """
        Persisted settings for the ReachX trajectory analysis window.
        :return A 3-tuple containing the window geometry, state, and show/hide flag, in that order.
        """
        return self._cfg.traj_geometry, self._cfg.traj_state, self._cfg.traj_window_open

    @trajectory_window_settings.setter
    def trajectory_window_settings(self, settings: Tuple[QByteArray, QByteArray, bool]) -> None:
        """
        Update trajectory analysis window's persisted settings. Be sure to call this prior to application shutdown so
        that these settings are saved to the user's application settings file.
        :param settings: A 3-tuple containing the window geometry, state, and show/hide flag, in that order.
        """
        self._cfg.traj_geometry = settings[0]
        self._cfg.traj_state = settings[1]
        self._cfg.traj_window_open = settings[2]

    @property
    def preferences(self) -> RxPreferences:
        """ User preferences explicitly set through the application Preferences dialog. """
        return self._cfg.user_preferences

    @preferences.setter
    def preferences(self, prefs: RxPreferences) -> None:
        """
        Update user preferences explicity set through the application Preferences dialog.
        :param prefs: The updated preferences.
        """
        self._cfg.user_preferences = prefs

    @property
    def existing_workspaces(self) -> List[str]:
        """ List of the names of all existing workspace configurations, in alphabetical order. """
        return self._cfg.get_workspace_names()

    @property
    def active_workspace(self) -> Workspace:
        """ The workspace currently governing application behavior. """
        return self._active_ws

    @property
    def current_session_id(self) -> Optional[RxSessionID]:
        """ ID for experiment session currently loaded from the active workspace, or None if no session is loaded. """
        return self.session_manager.id

    @property
    def current_model_id(self) -> Optional[RxModelID]:
        """ Identifier for body part detection ML model currently loaded from the active workspace; possibly None. """
        return self.model_manager.id

    @property
    def existing_models(self) -> List[RxModelID]:
        """ List of all body part detection ML models defined in the active workspace; possibly empty. """
        return [] if self._active_ws is None else self._active_ws.models().copy()

    @property
    def existing_segmentation_models(self) -> List[SegmentationModelRecord]:
        """ List of externally trained reach segmentation models defined in the active workspace. """
        if self._active_ws is None:
            return []
        records = discover_segmentation_models(self._active_ws.segmentation_model_root)
        allowed = {"any", "fixed", "fixed_cam"} if self._active_ws.is_fixed_cam else {"any", "legacy", "legacy_cam"}
        return [record for record in records if record.session_type in allowed]

    @property
    def current_segmentation_model(self) -> Optional[SegmentationModelRecord]:
        """ The currently selected externally trained reach segmentation model, if any. """
        return self._current_segmentation_model

    def create_new_workspace(self, copy_active: bool, ws_name: Optional[str], is_fixed: bool) -> bool:
        """
        Create a new workspace and make it the active workspace for the application.

        :param copy_active: If True, the new workspace will be a copy of the current active workspace (albeit with a
            unique name), else all workspace parameters will be set to application defaults.
        :param ws_name: A valid, unique name for the new workspace. If None, duplicates an existing workspace name, or
            is otherwise invalid, the workspace name will be auto-generated.
        :param is_fixed: True for a fixed-cam workspace; False for a legacy-cam workspace. A fixed-cam workspace only
            exposes fixed-cam sessions analyzed with fixed-cam models, and analogously for a legacy-cam workspace.
            Ignored if ``copy_active`` is True. The fixed/legacy mode of the new workspace matches the copied workspace.
        :return: True if workpace created, in which case it is now the active workspace. False if it cannot be created
        because application state currently forbids a workspace switch.
        """
        if self.can_switch_workspaces():
            ws = self._cfg.create_workspace(copy_ws=self._active_ws if copy_active else None,
                                            name=ws_name, is_fixed=is_fixed)
            self.session_manager.unload()
            self.model_manager.unload()
            self._active_ws = ws
            self._cfg.mru_workspace = ws.name
            self._load_default_segmentation_model(emit_signal=False)
            self._session_cache.reload(self._active_ws.session_root)
            QTimer.singleShot(10, lambda: self.active_workspace_switched.emit())
            get_application_logger().info(f"Switched to workspace: {self._active_ws.name}")
            return True
        return False

    def is_valid_and_unique_workspace(self, candidate: str) -> bool:
        """
        :param candidate: Candidate name for a new workspace.
        :return: ``True`` if name is valid and does not match an existing workspace. Else ``False``.
        """
        return self._cfg.is_valid_workspace_name(candidate)

    def can_delete_active_workspace(self) -> bool:
        """
        Can the current active workspace be deleted?

        The default workspace may never be removed. Also, if application state currently forbids a workspace switch,
        then deletion is also forbidden because a different workspace automatically becomes active (the active
        workspace is always defined).

        :return: True if deletion is possible, else False.
        """
        return self.can_switch_workspaces() and (self._active_ws.name != DEF_WORKSPACE_NAME)

    def delete_active_workspace(self) -> bool:
        """
        If possible, delete the current active workspace and activate the default workspace (which may not be deleted).

        No data or analysis results associated with a workspace are destroyed when a workspace is deleted, only the
        workspace configuration itself.
        :return: False if deletion was not possible, else True.
        """
        if self.can_delete_active_workspace():
            self._cfg.delete_workspace(self._active_ws.name)
            self.session_manager.unload()
            self.model_manager.unload()
            self._active_ws = self._cfg.get_workspace_by_name(DEF_WORKSPACE_NAME)
            self._cfg.mru_workspace = DEF_WORKSPACE_NAME
            self._load_default_segmentation_model(emit_signal=False)
            self._session_cache.reload(self._active_ws.session_root)
            QTimer.singleShot(10, lambda: self.active_workspace_switched.emit())
            get_application_logger().info(f"Switched to workspace: {self._active_ws.name}")
            if self._active_ws.session_exists(self._active_ws.mru_session):
                self.load_session(self._active_ws.mru_session)
            models = self._active_ws.models()
            if len(models) > 0:
                self.load_model(self._active_ws.mru_model if (self._active_ws.mru_model in models) else models[0])
            return True
        return False

    def update_active_workspace_path(self, p: Path, root_key: str) -> None:
        """
        Update session or model root directory for the active workspace, then emit the ``active_workspace_path_changed``
        signal. If the session root is changed, a background task is started to rebuild the session cache.

        :param p: The new root directory. (Does not check that directory exists.)
        :param root_key: Which root directory to update. One of: ``Workspace.SESH_ROOT, .MODEL_ROOT,
            .SEGMENTATION_MODEL_ROOT, .CALIB_ROOT``
            **NOTE that the calibration root directory applies only to fixed-cam workspaces.**
        """
        if not isinstance(p, Path):
            return
        if root_key == Workspace.SESH_ROOT:
            self._active_ws.session_root = p
            self.session_manager.unload()
            self._session_cache.reload(self._active_ws.session_root)
        elif root_key == Workspace.MODEL_ROOT:
            self._active_ws.model_root = p
            if len(self.existing_models) > 0:
                self.load_model(self.existing_models[0])
            else:
                self.model_manager.unload()
        elif root_key == Workspace.SEGMENTATION_MODEL_ROOT:
            self._active_ws.segmentation_model_root = p
            self._load_default_segmentation_model()
        elif root_key == Workspace.CALIB_ROOT and self._active_ws.is_fixed_cam:
            self._active_ws.calibration_root = p
        else:
            return
        QTimer.singleShot(10, lambda: self.active_workspace_path_changed.emit())

    def can_switch_workspaces(self):
        """
        Can the user make a different workspace configuration the active workspace at this time?

        The active workspace governs the behavior of DataManager and ReachX, so switching workspaces in certain
        situations could crash the application or lead to unexpected behavior.
        :return: True if an active workspace switch is permissible.
        """
        return self._active_ws is not None

    def switch_to_workspace(self, ws_name: str) -> bool:
        """
        Make the specified workspace the current active workspace governing application behavior. If the workspace
        switch succeeds, load the most recently accessed session and model for the now-active workspace.
        :param ws_name: Name of the target workspace.
        :return: False if a workspace switch is not permitted, if the specified workspace is already the active one, or
        if the workspace does not exist.
        """
        if (self._active_ws is not None) and self._active_ws.name == ws_name:
            return False
        if self.can_switch_workspaces():
            ws = self._cfg.get_workspace_by_name(ws_name)
            if ws is not None:
                self._active_ws = ws
                self._cfg.mru_workspace = ws.name
                self.session_manager.unload()
                self.model_manager.unload()
                self._load_default_segmentation_model(emit_signal=False)
                self.active_workspace_switched.emit()
                self._session_cache.reload(self._active_ws.session_root)
                get_application_logger().info(f"Switched to workspace: {self._active_ws.name}")
                if self._active_ws.session_exists(self._active_ws.mru_session):
                    self.load_session(self._active_ws.mru_session)
                models = self._active_ws.models()
                if len(models) > 0:
                    self.load_model(self._active_ws.mru_model if (self._active_ws.mru_model in models) else models[0])

                return True
            else:
                get_application_logger().debug(f"Attempted switch to unrecognized workspace: {ws_name}")
        return False

    def can_load_session(self) -> bool:
        """
        Can the user load a previously recorded session from the active workspace?
        :return: True as long as there is an active workspace
        """
        return self._active_ws is not None

    @property
    def has_recorded_sessions(self) -> bool:
        """ Are there any recorded sessions in the active workspace? """
        return (self._active_ws is not None) and (self._session_cache.session_count(self._active_ws.is_fixed_cam) > 0)

    def sessions(self) -> List[RxSessionID]:
        """
        Get the entire list of sessions found in the active workspace.

        **NOTES**:
         - A fixed-cam workspace only exposes fixed-cam sessions, and analogously for a legacy-cam workspace, even
           though a user may use the same session root for both workspace types.
         - If the workspace's session directory tree contains a large number of sessions, it can take several
           minutes to do the directory scan (particularly on a remote/networked file system). The scan is done on a
           background thread, and partial results may be delivered by that thread as it proceeds.

        :return: List of session identifiers. Could be an incomplete list if workspace's session directory tree is
            currently being scanned in the background.
        """
        if self._active_ws is None:
            return []
        else:
            return self._session_cache.sessions(self._active_ws.is_fixed_cam)

    def load_session(self, sesh_id: RxSessionID, init_frame: int = 0) -> bool:
        """
        Load the specified session from the active workspace, if possible. If the session is already loaded, but the
        specified initial frame is not the current frame on display, then go to that frame.

        :param sesh_id: The target session identifier.
        :param init_frame: The frame number of the initial frame to load. Default = 0.
        :return: True if successful (or session already loaded). If unable to load the session, an error description is
            posted to the application log.
        """
        if self.can_load_session():
            if sesh_id == self.session_manager.id:
                if self.session_manager.current_frame_num != init_frame:
                    self.session_manager.go_to_frame(init_frame)
                return True
            if self.active_workspace.session_exists(sesh_id):
                err = self.session_manager.load(sesh_id, init_frame)
                if err is None:
                    get_application_logger().info(f"Loaded session: {sesh_id}; initial frame = {init_frame}")
                    self._active_ws.mru_session = sesh_id
                    QTimer.singleShot(10, lambda: self.session_loaded.emit())
                    return True
                else:
                    get_application_logger().error(f"Unable to load session '{sesh_id}': {err}")
            else:
                get_application_logger().debug(f"Attempt to load non-existent session: {sesh_id}")
        return False

    def can_load_model(self) -> bool:
        """
        Can the user load a previously defined body part detection ML model from the active workspace?
        :return: True if load is psosible; False if ``DataManger`` is busy or otherwise unable to load an ML model, OR
            if the workspace contains no defined models whatsoever.
        """
        return (self._active_ws is not None) and (len(self._active_ws.models()) > 0)

    def load_model(self, model_id: RxModelID) -> bool:
        if self.can_load_model():
            if model_id == self.model_manager.id:
                return True   # already loaded!
            if model_id in self._active_ws.models():
                err = self.model_manager.load(model_id, self._active_ws.session_root)
                if err is None:
                    get_application_logger().info(f"Loaded ML model: {model_id}")
                    self._active_ws.mru_model = model_id
                    QTimer.singleShot(10, lambda: self.model_loaded.emit())
                    return True
                else:
                    get_application_logger().error(f"Unable to load ML model '{model_id}': {err}")
            else:
                get_application_logger().debug(f"Attempt to load non-existent ML model: {model_id}")
        return False

    def load_segmentation_model(self, model: SegmentationModelRecord) -> bool:
        """
        Select an externally trained reach segmentation model from the active workspace.
        """
        if self._active_ws is None:
            return False
        record = find_segmentation_model(self._active_ws.segmentation_model_root, model.model_root)
        if record is None:
            get_application_logger().debug(f"Attempt to load non-existent segmentation model: {model.model_root}")
            return False
        self._current_segmentation_model = record
        self._active_ws.mru_segmentation_model = record.model_root
        get_application_logger().info(f"Selected reach segmentation model: {record.display_name}")
        QTimer.singleShot(10, lambda: self.segmentation_model_loaded.emit())
        return True

    def register_segmentation_model(self, model_root: Path) -> bool:
        """
        Register an externally trained reach segmentation model folder in the active workspace.
        """
        if self._active_ws is None:
            return False
        try:
            register_external_model(Path(model_root), self._active_ws.segmentation_model_root)
            record = find_segmentation_model(self._active_ws.segmentation_model_root, Path(model_root))
            if record is None:
                raise RuntimeError("registered model was not found in segmentation model registry")
            return self.load_segmentation_model(record)
        except Exception as exc:
            get_application_logger().info(f"Unable to register segmentation model folder: {exc}")
            return False

    def _load_default_segmentation_model(self, emit_signal: bool = True) -> None:
        self._current_segmentation_model = None
        if self._active_ws is None:
            return
        records = self.existing_segmentation_models
        if len(records) == 0:
            self._active_ws.mru_segmentation_model = None
            if emit_signal:
                QTimer.singleShot(10, lambda: self.segmentation_model_loaded.emit())
            return
        selected = None
        if self._active_ws.mru_segmentation_model is not None:
            selected = next((record for record in records
                             if record.model_root.resolve() == self._active_ws.mru_segmentation_model.resolve()),
                            None)
        self._current_segmentation_model = selected if selected is not None else records[0]
        self._active_ws.mru_segmentation_model = self._current_segmentation_model.model_root
        if emit_signal:
            QTimer.singleShot(10, lambda: self.segmentation_model_loaded.emit())

    def can_create_model(self) -> bool:
        """
        Can the user create a new body part detection ML model in the active workspace?
        :return: False only if there is no active workspace; else True
        """
        return self._active_ws is not None

    def create_and_load_new_model(self, proj_name: str) -> None:
        """
        Create a new body part detection ML model and make it the current model in the active workspace. If
        unsuccessful, an error description is posted to the application log. Otherwise, the `model_loaded` signal
        is emitted.

        If the active workspace is fixed-cam, it only exposes sessions recorded on fixed-cam rigs, and only fixed-cam
        models can analyze those sessions. Likewise for a legacy-cam workspace.

        :param proj_name: Candidate name for the new model project. Cannot match any existing model.
        """
        if self.can_create_model():
            if proj_name in [m.project_name for m in self.existing_models]:
                get_application_logger().error("Model project name '{proj_name}' duplicates an existing model.")
            else:
                model_id = self.model_manager.create_model(self._active_ws.model_root, proj_name,
                                                           self._active_ws.is_fixed_cam)
                if isinstance(model_id, RxModelID):
                    self.load_model(model_id)

    def can_delete_current_model(self) -> bool:
        """
        Can the user permanently delete the body part detection ML model currently loaded in the active workspace?
        :return: False if there is no current model, or if a background task that uses the model is currently in
            progress; else True.
        """
        return (self._active_ws is not None) and self.model_manager.model_loaded

    def delete_current_model(self) -> None:
        """
        Permanently delete the body part detection ML model currently loaeded in the active workspace. After deletion,
        one of the remaining ML models -- if any -- is loaded in its stead. The `model_loaded` signal is emitted
        regardless.

        **Use with caution**: This operation destroys the underlying file system backing store for the model and cannot
        be undone.
        """
        if self.can_delete_current_model():
            model_loc = self.model_manager.id.location
            self.model_manager.unload()

            shutil.rmtree(model_loc, ignore_errors=True)
            if model_loc.exists():
                get_application_logger().error(f"A problem occurred while permanently removing model directory at: "
                                               f"{str(model_loc.absolute())}.")
            else:
                get_application_logger().info(f"Removed ML model project at: {str(model_loc.absolute())}")

            available_models = self._active_ws.models()
            send_sig = True
            if len(available_models) > 0:
                send_sig = not self.load_model(available_models[0])
            if send_sig:
                QTimer.singleShot(10, lambda: self.model_loaded.emit())

    def train_current_model(self, net_type: RxNetworkType) -> None:
        """
        Run a background job to train the active iteration of the current body part detection ML model.

         - A minimum required number of hand- and pellet-related markers must be defined for the model iteration.
         - For a fixed-cam session, warn (via application log) if there are too few of the special-purpose markers that
           are useful for improving reach segmentation performance.
         - If the model iteration has been trained before, the model files generated by the previous training session
           are deleted.
         - A modal progress dialog blocks input to all other application windows while the job is running.
         :param net_type: The pretrained neural net architecture type upon which model will be based.
        """
        if not self.model_manager.can_train():
            get_application_logger().info("Unable to train model (no model loaded, or insufficient data)")
            return

        self.model_manager.warn_if_too_few_reference_markers_for_fixed_cam_model()

        if self.session_manager.playback_in_progress:
            self.session_manager.stop_playback()

        markers_per_session = self.model_manager.get_all_markers()
        self._create_job_dialog_on_first_use()
        job = RxJob(RxJobType.TRAIN, model_id=self.model_manager.id, iter_num=self.model_manager.current_iteration,
                    markers=markers_per_session, net_type=net_type)
        get_application_logger().info(f"Starting background job to train mode {str(self.model_manager.id)},"
                                      f"iteration={self.model_manager.current_iteration}, net_type={str(net_type)}")
        self._job_dlg.run_job(job)

    def _create_job_dialog_on_first_use(self) -> None:
        if self._job_dlg is None:
            self._job_dlg = BlockingJobDialog(self._main_window)
            # noinspection PyUnresolvedReferences
            self._job_dlg.accepted.connect(self._on_background_job_done)

    @Slot()
    def _on_background_job_done(self) -> None:
        """ Handler called after a long-running background job has finished and user has closed the progress dialog. """
        if self._job_dlg.job_failed:
            return
        job_type = self._job_dlg.job_type
        if job_type == RxJobType.TRAIN:
            QTimer.singleShot(10, lambda: self.model_trained.emit())
        elif job_type == RxJobType.ANALYZE:
            self.session_manager.on_session_analyzed(self._job_dlg.sessions_affected)
        elif job_type == RxJobType.REACH:
            self.session_manager.on_reach_segmentation_done(self._job_dlg.sessions_affected)
        elif job_type == RxJobType.ML_REACH:
            self.session_manager.on_reach_segmentation_done(self._job_dlg.sessions_affected, select_latest=True)

    def analyze_sessions(self, sesh_ids: Optional[List[RxSessionID]] = None, batch_size: int = 32,
                         find_reaches: bool = False) -> None:
        """
        Run a background job that uses the active iteration of the current body part detection model to analyze the
        currently loaded session or a specified list of sessions. Analysis output includes the predicted body part
        marker locations on the master and secondary cameras, and the smoothed position and speed trajectories of the
        animal's hand and the food pellet. Optionally, use those trajectories to perform automated reach segmentation
        on each analyzed session.

         - If the current model iteration has not been trained, no action is taken and an informative message is
           posted to the application log.
         - A modal progress dialog blocks input to all other application windows while the job is running.

        :param sesh_ids: List of session identifiers indicating what sessions to process. **If None or empty, the
            currently loaded session is processed.**
        :param batch_size: Number of frames consumed in one "batch". Default = 32. Tests indicated increasing batch
            size did not improve performance on a system with or without GPU suport. Recommend using the default value.
        :param find_reaches: If True, perform automated reach segmentation on each successfully analyzed session using
            the segmentation control parameters defined in the active workspace.
        """
        model_id, iter_num = self.model_manager.id, self.model_manager.current_iteration
        if (model_id is None) or not model_id.has_been_trained(iter_num):
            get_application_logger().info("Unable to continue -- please select a previously trained model iteration!")
            return

        if (sesh_ids is None) or (len(sesh_ids) == 0):
            if not self.session_manager.session_loaded:
                get_application_logger().info("Unable to continue -- no session specified!")
                return
            sesh_ids = [self.session_manager.id]

        self._create_job_dialog_on_first_use()
        job = RxJob(RxJobType.ANALYZE, model_id=model_id, iter_num=iter_num, sessions=sesh_ids, batch_size=batch_size,
                    use_gpu=self.preferences.use_gpu, enable_pp=self.preferences.enable_pp,
                    seg_parms=self._active_ws.reach_seg_ctrls if find_reaches else None,
                    calib_root=self._active_ws.calibration_root)
        get_application_logger().info(f"Starting analysis of {len(sesh_ids)} session(s)...")
        self._job_dlg.run_job(job)

    def can_find_reaches_for_current_session(self) -> bool:
        """
        Can ReachX find reach segments for the current session? True only if a session is loaded and has been
        analyzed by at least one scorer, ie, model iteration.
        """
        return self.session_manager.current_scorer is not None

    def find_reach_segments_for_current_session(self) -> None:
        """
        Run a background job to perform algorithmic reach segmentation on the currently loaded session using the
        session's current scorer. (A session may be analyzed by more than one scorer, so the session manager maintains
        the notion of a current scorer).

         - If no session is loaded, or the current session is yet to be analyzed by a body part detection model, then
           no action is taken and an informative message is posted to the application log.
         - A modal progress dialog blocks input to all other application windows while the job is running.
        """
        scorer = self.session_manager.current_scorer
        if scorer is None:
            get_application_logger().info("Unable to find reaches: no session loaded, or "
                                          "current session has not been analyzed yet.")
            return

        self._create_job_dialog_on_first_use()
        job = RxJob(RxJobType.REACH, sesh_id=scorer.sesh_id, rx_scorer=scorer.prefix,
                    seg_parms=self._active_ws.reach_seg_ctrls)
        get_application_logger().info(f"Starting reach segmentation on {scorer.sesh_id} using scorer "
                                      f"{scorer.short_name}...")
        self._job_dlg.run_job(job)

    def find_ml_reach_segments_for_current_session(self, model_root: Optional[Path] = None) -> None:
        """
        Run externally trained model-based reach segmentation for the current session scorer.
        """
        scorer = self.session_manager.current_scorer
        if scorer is None:
            get_application_logger().info("Unable to find ML reaches: no session loaded, or "
                                          "current session has not been analyzed yet.")
            return
        selected_model = model_root is None
        if selected_model:
            if self._current_segmentation_model is None:
                get_application_logger().info("Unable to find ML reaches: no segmentation model selected.")
                return
            model_root = self._current_segmentation_model.model_root
        if not selected_model:
            try:
                register_external_model(Path(model_root), self._active_ws.segmentation_model_root)
            except Exception as exc:
                get_application_logger().info(f"Unable to register segmentation model folder: {exc}")
                return
        self._create_job_dialog_on_first_use()
        job = RxJob(RxJobType.ML_REACH, sesh_id=scorer.sesh_id, rx_scorer=scorer.prefix,
                    ml_model_root=Path(model_root))
        get_application_logger().info(f"Starting ML reach segmentation on {scorer.sesh_id} using scorer "
                                      f"{scorer.short_name} and model folder {model_root}...")
        self._job_dlg.run_job(job)

    def run_idle_job(self) -> None:
        """
        **For testing purposes only**: Tests the infrastructure for running a job on a background thread WITHOUT
        blocking the UI. A progress bar and progress messages are shown in the application status bar instead.
        with a progress dialog which displays progress messages and includes a button to cancel the task.

        The test job runs for a set duration randonmly chosen between 2 and 20 seconds, posting progress updates and
        messages once per second.
        """
        runner = NonBlockingJobRunner(self._main_window.statusBar())
        job = RxJob(RxJobType.TEST, dur=random.randint(2, 20))
        runner.run_job(job)
