"""
Configuration module for ReachX.

This contains some configuration-related constants and the ``ConfigHelper`` class, which handles the details of
locating, reading and writing all configuration-related files for the application. The configuration helper is a
singleton instantiated and solely accessed by ``DataManager``, which encapsulates all application state, data, and
configuration details for ReachX.

NOTES
 - We're using the standard Python `configparser` library to read/write all relevant configuration in an INI file.
 - New workspace concept replaces the notion of "pseudo users" (not same as the login user) identified by initials in
   the current application. Users can create different workspaces for different experiments/studies, etc.
 - All workspace configurations and general application settings like UI state are persisted in a single INI file in
   the login user's home directory: ``<user_home>/.<app_name>/settings.ini``. Users should not touch this file.
 - NOTE the way main window state and geometry (encapsulated as QByteArray) are saved/restored from ``settings.ini``.
   That was the result of a bit of trial and error!
 - It is essential that ``ConfigHelper.load()`` is called early during application startup. That method will,
   if necessary, create the application home directory ``<user_home>/.<app_name>`` and initial settings file.
"""
from __future__ import annotations

import re
from configparser import ConfigParser
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import QByteArray

from reachx.common import RX, RxSessionID, RxModelID, RxPreferences
from reachx.config.app_log import get_application_logger
from reachx.config.workspace import Workspace

_RX = RX()
""" App-wide constants. """


DEF_WORKSPACE_NAME: str = 'default'
""" Name of the default workspace, which must always exist. """

# default directory names used to create root paths for the default workspace
_DEF_SESSION_DIR: str = 'sessions'
_DEF_MODEL_DIR: str = 'models'
_DEF_SEGMENTATION_MODEL_DIR: str = 'segmentation_models'
_DEF_CALIB_DIR: str = 'calibrations'

_APPSETTINGS_CURR_VERSION = 1
""" Current version number for the settings file. """

_APPCONFIG_HOME: Path = Path(Path.home(), '.' + _RX.APP_NAME)
""" Per-user application configuration directory. """
_APPSETTINGS_FILE: Path = Path(Path.home(), '.' + _RX.APP_NAME, 'settings.ini')
""" The per-user application settings file. General settings as well as all defined workspace configs go here. """

_DEF_WS_DICT_LEGACY = {
    Workspace.SESH_ROOT: str(Path(_APPCONFIG_HOME, _DEF_SESSION_DIR).absolute()),
    Workspace.MODEL_ROOT: str(Path(_APPCONFIG_HOME, _DEF_MODEL_DIR).absolute()),
    Workspace.SEGMENTATION_MODEL_ROOT: str(Path(_APPCONFIG_HOME, _DEF_SEGMENTATION_MODEL_DIR).absolute()),
    Workspace.CALIB_ROOT: "",
    Workspace.MRU_SESSION: "",
    Workspace.MRU_MODEL: "",
    Workspace.MRU_SEGMENTATION_MODEL: ""
}
""" 
Configuration parameters for the default legacy-cam application workspace (reach seg param values set to app defaults). 
"""

_DEF_WS_DICT_FIXED = {
    Workspace.SESH_ROOT: str(Path(_APPCONFIG_HOME, _DEF_SESSION_DIR).absolute()),
    Workspace.MODEL_ROOT: str(Path(_APPCONFIG_HOME, _DEF_MODEL_DIR).absolute()),
    Workspace.SEGMENTATION_MODEL_ROOT: str(Path(_APPCONFIG_HOME, _DEF_SEGMENTATION_MODEL_DIR).absolute()),
    Workspace.CALIB_ROOT: str(Path(_APPCONFIG_HOME, _DEF_CALIB_DIR)),
    Workspace.MRU_SESSION: "",
    Workspace.MRU_MODEL: "",
    Workspace.MRU_SEGMENTATION_MODEL: ""
}
""" 
Configuration parameters for the default fixed-cam application workspace (reach seg param values set to app defaults). 
"""


class ConfigHelper:
    """
    The application configuration helper, a singleton that reads/writes several kinds of configuration settings:
     - GUI layout and state, the set of all defined workspaces, and the most recently active workspace.
     - Workspace configurations. Any number of workspaces may be defined so that a user can partition their research
       data and results into distinct "silos". Workspace parameters include session, model, and calibration root paths
       (the latter is for fixed-cam workspaces only); reach segmentation control parameters; and the identity of the
       most recently loaded session and model for the workspace.

    Usage: Be sure to call ``load()`` early during application startup to load configuration settings from file (and to
    handle the special case of a fresh install, when no settings file exists). Failure to do so will lead to
    unexpected behavior.
    """

    _instance: Optional[ConfigHelper] = None
    """ The singleton instance. """

    def __new__(cls, *args, **kwargs):
        """ Overridden to enforce singleton instance. """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Construct the application configuration helper in an unloaded state (all settings undefined).
        """
        super().__init__()
        self._loaded: bool = False
        """ Have configuration settings been loaded from file? """
        self._window_state: QByteArray = QByteArray()
        """ Main application window state (initially an empty byte array). """
        self._geometry: QByteArray = QByteArray()
        """ Main application window geometry (initially an empty byte array)."""
        self._traj_state: QByteArray = QByteArray()
        """ Trajectory analysis window state (initially an empty byte array). """
        self._traj_geom: QByteArray = QByteArray()
        """ Trajectory analysis window geometry (initially an empty byte array). """
        self._traj_open: bool = False
        """ Whether or not trajectory analysis window should be opened when ReachX next launches. """
        self._workspaces: List[Workspace] = list()
        """ All currently defined workspaces. Initially empty. """
        self._mru_ws: str = DEF_WORKSPACE_NAME
        """ Name of the workspace that was active the last time the application settings were saved. """
        self._preferences: RxPreferences = RxPreferences()
        """ All other ReachX user preferences/settings set explicitly via a preferences dialog. """

    @property
    def window_state(self) -> QByteArray:
        """
        The last-saved main application window state in a byte array format expected/generated by ``QMainWindow``.
        Will be an empty byte array if the window state is unknown.
        """
        return self._window_state

    @window_state.setter
    def window_state(self, new_state: QByteArray) -> None:
        """
        Update the last-saved main application window state.
        :param new_state: The new state.
        """
        if not isinstance(new_state, QByteArray):
            raise ValueError("Invalid window state")
        self._window_state = new_state

    @property
    def window_geometry(self) -> QByteArray:
        """
        The last-saved main application window geometry (size, position, etc.) in a byte array format expected/generated
        by ``QMainWindow``. Will be an empty byte array if the window geometry is unknown.
        """
        return self._geometry

    @window_geometry.setter
    def window_geometry(self, new_geom: QByteArray) -> None:
        """
        Update the last-saved main application window geometry.
        :param new_geom: The updated geometry.
        """
        if not isinstance(new_geom, QByteArray):
            raise ValueError("Invalid window geometry")
        self._geometry = new_geom

    @property
    def traj_state(self) -> QByteArray:
        """
        The last-saved trajectory analysis window state in a byte array format expected/generated by ``QMainWindow``.
        Will be an empty byte array if the window state is unknown.
        """
        return self._traj_state

    @traj_state.setter
    def traj_state(self, new_state: QByteArray) -> None:
        """
        Update the last-saved state of the trajectory analysis window.
        :param new_state: The new state.
        """
        if not isinstance(new_state, QByteArray):
            raise ValueError("Invalid window state")
        self._traj_state = new_state

    @property
    def traj_geometry(self) -> QByteArray:
        """
        The last-saved trajectory analysis window geometry (size, position, etc.) in the byte array format expected and
        generated by ``QMainWindow``. Will be an empty byte array if the window geometry is unknown.
        """
        return self._traj_geom

    @traj_geometry.setter
    def traj_geometry(self, new_geom: QByteArray) -> None:
        """
        Update the last-saved trajectory analysis window geometry.
        :param new_geom: The updated geometry.
        """
        if not isinstance(new_geom, QByteArray):
            raise ValueError("Invalid window geometry")
        self._traj_geom = new_geom

    @property
    def traj_window_open(self) -> bool:
        """ True(False) if the trajectory analysis window should be shown(hidden). """
        return self._traj_open

    @traj_window_open.setter
    def traj_window_open(self, new_open: bool) -> None:
        """
        Update show/hide status for the trajectory analysis window.
        :param new_open: True if the trajectory analysis window should be shown(hidden).
        """
        self._traj_open = new_open

    @property
    def user_preferences(self) -> RxPreferences:
        """ User preferences explicitly set through the application Preferences dialog. """
        return self._preferences

    @user_preferences.setter
    def user_preferences(self, prefs: RxPreferences) -> None:
        """
        Update user preferences explicity set through the application Preferences dialog.
        :param prefs: The updated preferences.
        """
        if isinstance(prefs, RxPreferences):
            self._preferences = prefs

    @property
    def mru_workspace(self) -> str:
        """ Name of the most recently active workspace. """
        return self._mru_ws

    @mru_workspace.setter
    def mru_workspace(self, ws_name: str) -> None:
        """
        Set the name of most recently active workspace.
        :param ws_name: The name of a workspace.
        """
        if ws_name not in [w.name for w in self._workspaces]:
            get_application_logger().warning(f"Attempt to set MRU workspace to a nonexistent workspace: {ws_name}")
            raise ValueError("Workspace configuration does not exist")
        self._mru_ws = ws_name

    def create_workspace(self, copy_ws: Optional[Workspace] = None,
                         name: str = DEF_WORKSPACE_NAME, is_fixed: bool = False) -> Workspace:
        """
        Create a new workspace, optionally a copy of an existing workspace (but with a unique name). Note that only
        workspace parameters are copied -- this is NOT a deep copy of workspace content on the file system!

        :param copy_ws: If None, a new workspace is created with default parameters and a unique name. Else, the new
            workspace is a copy of this one, albeit with a different name.
        :param name: Name of the new workspace. Default: ``DEF_WORKSPACE_NAME``. If name is invalid or already in use,
            a unique valid workspace name is generated.
        :param is_fixed: True to create a fixed-cam workspace, False for a legacy-cam workspace. **Ignored when copying
            an existing workspace.**

        :return The new workspace object.
        """
        # crude generation of unique valid name
        if copy_ws is None:
            if name == DEF_WORKSPACE_NAME or not self.is_valid_workspace_name(name, False):
                candidate = "myWorkspace"
            else:
                candidate = name
        else:
            candidate = "myWorkspace" if copy_ws.name == DEF_WORKSPACE_NAME else copy_ws.name
        existing_names = [w.name for w in self._workspaces]
        if candidate in existing_names:
            if len(candidate) > _RX.MAX_LEN_WS_NAME - 2:
                candidate = candidate[:-2]
            i = 1
            while True:
                name = f"{candidate}{i}"
                if name not in existing_names:
                    candidate = name
                    break
                i += 1

        ws_dict = (_DEF_WS_DICT_FIXED if is_fixed else _DEF_WS_DICT_LEGACY) if copy_ws is None else copy_ws.to_dict()
        ws = Workspace.from_dict(candidate, ws_dict)
        self._workspaces.append(ws)
        get_application_logger().info(f"Created new workspace: {ws.name}")
        return ws

    def is_valid_workspace_name(self, candidate: str, check_unique: bool = True) -> bool:
        """
        Is the specified workspace name valid? It cannot exceed 25 characters, can contain only the characters
        [A-Za-z0-9_], and cannot match the name of any existing workspace configuration.
        :param: The candiaate workspace name.
        :return: True if the candidate name is valid and unique, else False.
        """
        ok = (0 < len(candidate) <= _RX.MAX_LEN_WS_NAME) and (re.fullmatch('[A-Za-z0-9_]+', candidate) is not None)
        if ok and check_unique:
            ok = candidate not in [w.name for w in self._workspaces]
        return ok

    def delete_workspace(self, ws_name: str) -> bool:
        """
        Delete the workspace configuration specified, if possible.

        NOTE: This does not remove any of the data directories defined by the workspace, so no data is lost -- only
        the workspace configuration itself.

        :param ws_name: Name of workspace to be deleted
        :return: True if successful, False if specified workspace does not exist.
        """
        found = next((i for i, w in enumerate(self._workspaces) if w.name == ws_name), -1)
        if found == -1:
            return False
        old = self._workspaces.pop(found)
        get_application_logger().info(f"Deleted workspace: {old.name}")
        return True

    @property
    def num_workspaces(self) -> int:
        """ The number of defined workspace configurations. """
        return len(self._workspaces)

    def get_workspace_names(self) -> List[str]:
        """
        Get the names of all currently defined application workspaces.
        :return: List of workspace names in alphabetical order.
        """
        out: List[str] = [ws.name for ws in self._workspaces]
        out.sort()
        return out

    def get_workspace_by_name(self, name: str) -> Optional[Workspace]:
        """
        Retrieve a workspace configuration.
        :param name: The (unique) name of the workspace.
        :return The workspace, or None if it does not exist.
        """
        return next((ws for ws in self._workspaces if ws.name == name), None)

    def load(self) -> Optional[str]:
        """
        Load application configuration settings.

        This method MUST be invoked EARLY during application startup. It loads all user settings, including all
        existing workspace configurations. It also deals with a missing settings file, which will always be the case
        when the application launches for the first time on a workstation for a particular user.

        Failure to load the application configuration should be considered a fatal error.

        **Version History**:
          - 0: For all ReachX releases through v0.6.6.
          - 1: (As of v0.6.7) Each workspace has its own ``mru_session`` and ``mru_model`` keys. Removed the same-named
            keys from the ``general`` section.
          - 1: (As of v0.7.0) Added ``preferences`` section defining the user preferences object ``RxPreferences``. Has
            only two boolean-valued keys, ``use_gpu`` and ``enable_pp``.

        :return: None if successful (or already loaded); else a short error description.
        """
        if self._loaded:
            return None

        self._workspaces.clear()

        get_application_logger().info("Loading application configuration.")
        try:
            ConfigHelper._ensure_configuration_exists()

            # load/parse the settings file
            parser = ConfigParser()
            if len(parser.read(_APPSETTINGS_FILE)) != 1:
                raise Exception(f"Unable to find/read user settings file: {str(_APPSETTINGS_FILE)}")
            version = int(parser['general']['version'])
            self._window_state = QByteArray.fromBase64(QByteArray.fromStdString(parser['general']['window_state']))
            self._geometry = QByteArray.fromBase64(QByteArray.fromStdString(parser['general']['geometry']))

            # trajectory analysis window state settings introduced in v0.6. This avoids incrementing version...
            self._traj_state = QByteArray.fromBase64(QByteArray.fromStdString(
                parser.get('general', 'traj_state', fallback='')))
            self._traj_geom = QByteArray.fromBase64(QByteArray.fromStdString(
                parser.get('general', 'traj_geom', fallback='')))
            self._traj_open = parser.getboolean('general', 'traj_open', fallback=False)

            ws_list = parser['general']['workspaces'].split(',')
            if len(ws_list) != len(set(ws_list)):
                raise Exception("Duplicate workspaces listed in settings file")
            if not all(self.is_valid_workspace_name(name, check_unique=True) for name in ws_list):
                raise Exception("One or more invalid workspace names!")
            mru = parser['general']['mru']
            found_mru, found_def = False, False
            for ws_name in ws_list:
                params = dict(parser[ws_name].items())
                self._workspaces.append(Workspace.from_dict(ws_name, params))
                found_mru = found_mru or (ws_name == mru)
                found_def = found_def or (ws_name == DEF_WORKSPACE_NAME)

            # the default workspace must always exist
            if not found_def:
                get_application_logger().warning(f"Did not find default workspace; recreating.")
                self._workspaces.append(Workspace.from_dict(DEF_WORKSPACE_NAME, _DEF_WS_DICT_LEGACY))
            if not found_mru:
                mru = DEF_WORKSPACE_NAME
                get_application_logger().warning(f"MRU workspace {mru} not found; reverting to default workpace.")
            self._mru_ws = mru

            # handling version 0 file: MRU session and model of MRU workspace were stored in keys mru_session and
            # mru_model.
            if found_mru and (version < _APPSETTINGS_CURR_VERSION):
                ws = self.get_workspace_by_name(mru)
                if parser.has_option('general', 'mru_session'):
                    sesh_pfx = parser['general']['mru_session']
                    ws.mru_session = RxSessionID.from_file_prefix(ws.session_root, sesh_pfx)
                if parser.has_option('general', 'mru_model'):
                    ws.mru_model = RxModelID.from_folder_name(ws.model_root, parser['general']['mru_model'])

            # added [preferences] section in version 1, release v0.7.0
            use_gpu = parser.getboolean('preferences', 'use_gpu', fallback=True)
            enable_pp = parser.getboolean('preferences', 'enable_pp', fallback=True)
            self._preferences = RxPreferences(use_gpu=use_gpu, enable_pp=enable_pp)

            self._loaded = True
            return None
        except KeyError as e:
            msg = f"Invalid or missing section or key in settings file: {e}"
            get_application_logger().critical(f"Config load failed: {msg}")
            return msg
        except Exception as e:
            msg = f"Unexpected error occurred while reading settings file: {e}"
            get_application_logger().critical(f"Config load failed: {msg}")
            return msg

    def save(self) -> None:
        """
        Save current application configuration settings. Call this prior to application exit.

        All user-specific application settings and workspace configurations are persisted to the dedicated INI file in
        the application configuration directory. It is written in its entirety even if no changes occurred during
        application runtime.

        Failure to write the settings file is not considered a fatal error, but errors will be logged for debugging
        purposes.
        """
        if not self._loaded:
            return
        try:
            ConfigHelper._ensure_configuration_exists()

            parser = ConfigParser()
            parser['general'] = {
                'version': str(_APPSETTINGS_CURR_VERSION),
                'window_state': self._window_state.toBase64().toStdString(),
                'geometry': self._geometry.toBase64().toStdString(),
                'traj_state': self._traj_state.toBase64().toStdString(),
                'traj_geom': self._traj_geom.toBase64().toStdString(),
                'traj_open': 'True' if self._traj_open else 'False',
                'workspaces': ','.join([ws.name for ws in self._workspaces]),
                'mru': self._mru_ws,
            }
            parser['preferences'] = {
                'use_gpu': 'True' if self._preferences.use_gpu else 'False',
                'enable_pp': 'True' if self._preferences.enable_pp else 'False'
            }
            for ws in self._workspaces:
                parser[ws.name] = ws.to_dict()

            with open(_APPSETTINGS_FILE, 'w') as f:
                parser.write(f)
            get_application_logger().info(f"Application settings saved!")
        except Exception as e:
            get_application_logger().warning(f"Application settings not saved: {e}")

    @staticmethod
    def _ensure_configuration_exists() -> None:
        """
        Ensure the application configuration directory exists within the user's home directory, and ensure a minimal
        settings file is present.

        :raises OSError if configuration directory exists but is a file; or unable to create the path.
        :raises Exception if unable to write a default settings file if one is missing.
        """
        # ensure the application configuration directory is there
        _APPCONFIG_HOME.mkdir(parents=True, exist_ok=True)

        # ensure the data directories defined for the default legacy-cam and fixed-cam workspaces exist. These are
        # subfolders within the  application configuration directory
        for dir_name in [_DEF_SESSION_DIR, _DEF_MODEL_DIR, _DEF_SEGMENTATION_MODEL_DIR, _DEF_CALIB_DIR]:
            Path(_APPCONFIG_HOME, dir_name).mkdir(parents=True, exist_ok=True)

        # write initial settings file if not found. In this case, use the default legacy-cam workspace.
        if not _APPSETTINGS_FILE.is_file():
            try:
                config = ConfigParser()
                config['general'] = {
                    'version': str(_APPSETTINGS_CURR_VERSION),
                    'window_state': QByteArray().toStdString(),
                    'geometry': QByteArray().toStdString(),
                    'workspaces': DEF_WORKSPACE_NAME,
                    'mru': DEF_WORKSPACE_NAME,
                }

                default_ws = Workspace.from_dict(DEF_WORKSPACE_NAME, _DEF_WS_DICT_LEGACY)
                config[default_ws.name] = default_ws.to_dict()

                with open(_APPSETTINGS_FILE, 'w') as f:
                    config.write(f)
            except Exception as e:
                # note - Reraise for handling in load(), but with clarifying message
                raise f"Failed to create missing user settings file ({e})"
