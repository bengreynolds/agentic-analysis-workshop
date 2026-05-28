"""
Workpace configuration.

A ReachX workspace is merely a way to let the user "silo" their research data and analysis results; eg, they could
define different workspaces for different studies or different animal subjects.

The workspace is merely a set of configuration parameters that tell ReachX where to find the data attached to the
workspace (file system paths), as well as other parameters that define workspace-specific user preferences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from reachx.common import RX, RxSessionID, RxModelID, RxReachSegControls


class Workspace:
    """
    The user-specified settings that define an application workspace configuration.

    Users can create multiple workspaces to logically separate their research data and results into distinct "silos",
    eg, individual studies, experiments on different animal subjects, etc.

    As of v0.6.0, ReachX supports a newer experiment setup, the **fixed-cam** setup using ``RxCam.LEFT`` and
    ``RxCam.RIGHT`` video cameras only. The **legacy-cam** setup uses ``RxCam.SIDE`` and ``RxCam.FRONT`` cameras, and
    possibly ``RxCam.STIM` and ``RxCam.FAST``, and will ultimately be replaced by the fixed-cam setup going forward.
    The fixed-cam setups also require a calibration file to map body part locations in the two videos to millimeters in
    XYZ-space. By design, a workspace is designated as fixed-cam or legacy-cam. A fixed-cam workspace can only contain
    fixed-cam sessions and models, and analogously for a legacy-cam workspace.

    Workspace settings include:
     - The session root: File system root directory under which all ReachX experiment sessions are stored.
     - The model root: File system root directory under which all ReachX body part detection models are stored.
     - The segmentation model root: File system root directory under which externally trained reach segmentation models
       are stored or registered.
     - The calibration root: File system root directory under which all fixed-cam calibration files are stored. This is
       unique to a fixed-cam workspace. The calibration root for a legacy-cam workspace is ``None``, which is how
       ReachX distinguished a fixed-cam vs legacy-cam workspace.
     - The most recently loaded session from this workspace (as of v0.6.7).
     - The most recently loaded model from this workspace (as of v0.6.7).
     - The most recently selected reach segmentation model.
     - Reach segmentation control parameters: These governs the behavior of the reach segmentation algorithm that
       analyzes predicted hand and pellet trajectories to find and categorize reaches during a session.

    ``DataManager`` manages the set of defined workspaces, one of which is the **active workspace** governing the
    application's behvior. Always use ``DataManager`` to get the names of all workspaces available, to create/delete a
    workspace, to edit any workspace setting, to get the active workspace configuration, or switch the active workspace.
    Some of these actions are unavailable when the application is in certain states.
    """
    SESH_ROOT: str = "sessions"
    """ Key under which a workspace's session root folder is stored in application settings file. """
    MODEL_ROOT: str = "models"
    """ Key under which a workspace's model root folder is stored in application settings file. """
    SEGMENTATION_MODEL_ROOT: str = "segmentation_models"
    """ Key under which a workspace's segmentation model root folder is stored in application settings file. """
    CALIB_ROOT: str = "calibrations"
    """ Key under which a fixed-cam workspace's calibration root folder is stored in application settings file. """
    MRU_SESSION: str = "mru_session"
    """ Key under which the workspace's most recently accessed session is stored in app settings file. """
    MRU_MODEL: str = "mru_model"
    """ Key under which the workspace's most recently accessed model is stored in app settings file. """
    MRU_SEGMENTATION_MODEL: str = "mru_segmentation_model"
    """ Key under which the workspace's most recently selected segmentation model is stored in app settings file. """

    @staticmethod
    def from_dict(name: str, ws_dict: Dict[str, str]) -> Workspace:
        """
        Construct workspace configuration from the dictionary retrieved from the application settings INI file.

        **NOTES**:
        - The session and model roots must be specified in the supplied dict.
        - If calibration root is not specified or is set to "", then a legacy-cam workspace is created.
        - If any reach segmentation control parameter is missing, it is set to a default value.
        - If the ``MRU_SESSION`` key is not specified or is set to "", then the workspace has no MRU session.
        - If the ``MRU_MODEL`` key is not specified or is set to "", then the workspace has no MRU model.

        :param name: The unique name assigned to the workspace configuration.
        :param ws_dict: Workspace settings in dictionary form.
        :return: The workspace configuration.
        :raises KeyError: If session root or model root parameter is missing.
        :raises ValueError: If any parameter value is invalid. File paths are NOT tested for validity or existence.
        """
        ws = Workspace()
        ws._name = name
        ws._session_root = Path(ws_dict[Workspace.SESH_ROOT])
        ws._model_root = Path(ws_dict[Workspace.MODEL_ROOT])
        ws._segmentation_model_root = Path(
            ws_dict.get(Workspace.SEGMENTATION_MODEL_ROOT, str(Path(RX.HOME, "segmentation_models")))
        )

        # need to handle pre v0.6 settings file, which did not have the Workspace.CALIB_ROOT key. A workspace config
        # lacking that key must be a legacy-cam workspace. From v0.6 onward, the key is set to an empty string for
        # each legacy-cam workspace.
        ws._calibration_root = None
        if (Workspace.CALIB_ROOT in ws_dict) and ws_dict[Workspace.CALIB_ROOT] != "":
            ws._calibration_root = Path(ws_dict[Workspace.CALIB_ROOT])

        # most recently used session and model for this workspace -- will be missing from pre-v0.6.7 settings file
        if (Workspace.MRU_SESSION in ws_dict) and ws_dict[Workspace.MRU_SESSION] != "":
            ws._mru_session = RxSessionID.from_file_prefix(ws._session_root, ws_dict[Workspace.MRU_SESSION])
        if (Workspace.MRU_MODEL in ws_dict) and ws_dict[Workspace.MRU_MODEL] != "":
            ws._mru_model = RxModelID.from_folder_name(ws._model_root, ws_dict[Workspace.MRU_MODEL])
        if (Workspace.MRU_SEGMENTATION_MODEL in ws_dict) and ws_dict[Workspace.MRU_SEGMENTATION_MODEL] != "":
            ws._mru_segmentation_model = Path(ws_dict[Workspace.MRU_SEGMENTATION_MODEL])

        ws._reach_seg_ctrls = RxReachSegControls(
            reach_init_speed=float(ws_dict.get("reach_init_speed", -0.025)),
            reach_dirchange_speed=float(ws_dict.get("reach_dirchange_speed", 0.025)),
            pellet_drop_speed=float(ws_dict.get("pellet_drop_speed", 0.275)),
            pellet_drop_distZ=float(ws_dict.get("pellet_drop_distZ", -5)),
            pellet_drop_distY=float(ws_dict.get("pellet_drop_distY", -3)),
            dist_thresh_end=float(ws_dict.get("dist_thresh_end", 5)),
            confidence=float(ws_dict.get("confidence", 0.9)),
            pellet_dist_to_origin=float(ws_dict.get("pellet_dist_to_origin", 2)),
            max_frame=int(ws_dict.get("max_frame", 100)),
            min_frame=int(ws_dict.get("min_frame", 15)),
            distZ_hand_start=float(ws_dict.get("distZ_hand_start", 3))
        )
        return ws

    def __init__(self):
        """ **DO NOT USE**. Use `from_dict()` instead. """
        self._name: str = ""
        """ The workspace name. """
        self._session_root: Path = Path()
        """ 
        Root path for individual session folders. Each session folder contains the (compressed, not raw) videos acquired
        during the experiment session, timestamp files, and any analysis results for the session.
        """
        self._model_root: Path = Path()
        """ 
        Root path for individual body part detection ML models created via training on session videos. Each folder 
        under this root corresponds to a different trained model.
        """
        self._segmentation_model_root: Path = Path(RX.HOME, "segmentation_models")
        """ Root path for externally trained reach segmentation models. """
        self._calibration_root: Optional[Path] = None
        """ 
        Root path where calibration files for fixed-cam experiment setups are stored. As legacy-cam setups do not
        use calibration files, a value of ``None`` marks this as a legacy-cam workspace. 
        """
        self._mru_session: Optional[RxSessionID] = None
        """ The session that was loaded the last time this workspace was active. """
        self._mru_model: Optional[RxModelID] = None
        """ The body part detection ML model that was loaded the last time this workspace was active. """
        self._mru_segmentation_model: Optional[Path] = None
        """ The segmentation model that was selected the last time this workspace was active. """
        self._reach_seg_ctrls: RxReachSegControls = RxReachSegControls()
        """ Control parameters for the reach segmentation algorithm (initialized to default values). """

    def to_dict(self) -> Dict[str, str]:
        """ The workspace configuration in dictionary form, as it is stored in the ReachX app settings INI file. """
        out = dict()
        out[Workspace.SESH_ROOT] = str(self._session_root.absolute())
        out[Workspace.MODEL_ROOT] = str(self._model_root.absolute())
        out[Workspace.SEGMENTATION_MODEL_ROOT] = str(self._segmentation_model_root.absolute())
        out[Workspace.CALIB_ROOT] = \
            "" if (self._calibration_root is None) else str(self._calibration_root.absolute())
        out[Workspace.MRU_SESSION] = "" if (self._mru_session is None) else self._mru_session.session_file_prefix
        out[Workspace.MRU_MODEL] = "" if (self._mru_model is None) else self._mru_model.project_folder_name
        out[Workspace.MRU_SEGMENTATION_MODEL] = "" if (self._mru_segmentation_model is None) \
            else str(self._mru_segmentation_model.absolute())
        ctrls = self._reach_seg_ctrls._asdict()
        for key, value in ctrls.items():
            out[key] = str(value) if isinstance(value, int) else f"{value:.3f}"
        return out

    @property
    def name(self) -> str:
        """ Workspace name (uniquely identifying it among all workspaces defined on the system). """
        return self._name

    @property
    def session_root(self) -> Path:
        """ Base path where processed session data (compressed video, event timestamps) are stored. """
        return self._session_root

    @session_root.setter
    def session_root(self, p: Path) -> None:
        self._session_root = p

    @property
    def model_root(self) -> Path:
        """ Base path for individual body part detection ML models created via training on session videos. """
        return self._model_root

    @model_root.setter
    def model_root(self, p: Path) -> None:
        self._model_root = p

    @property
    def segmentation_model_root(self) -> Path:
        """ Base path for externally trained reach segmentation models. """
        return self._segmentation_model_root

    @segmentation_model_root.setter
    def segmentation_model_root(self, p: Path) -> None:
        self._segmentation_model_root = p

    @property
    def calibration_root(self) -> Optional[Path]:
        """
        For fixed-cam workspace only: the base path for locating fixed-cam rig calibration files. Always ``None``
        for a legacy-cam workspace.
        """
        return self._calibration_root

    @calibration_root.setter
    def calibration_root(self, p: Path) -> None:
        """ For fixed-cam workspace only: set the calibration root path. No effect for a legacy-cam workspace. """
        if self.is_fixed_cam and isinstance(p, Path):
            self._calibration_root = p

    @property
    def is_fixed_cam(self) -> bool:
        """ True if this is a fixed-cam workspace; False for a legacy-cam workspace. """
        return isinstance(self._calibration_root, Path)

    @property
    def mru_session(self) -> Optional[RxSessionID]:
        """ The most recently loaded experiment session, if any, for this workspace. """
        return self._mru_session

    @mru_session.setter
    def mru_session(self, sesh_id: Optional[RxSessionID]) -> None:
        """
        Update the identity of the most recently loaded session for this workspace.
        :param sesh_id: Session identifier, or None if no session loaded.
        """
        self._mru_session = sesh_id

    @property
    def mru_model(self) -> Optional[RxModelID]:
        """ Most recently loaded body part detection ML model, if any, for this workspace. """
        return self._mru_model

    @mru_model.setter
    def mru_model(self, model_id: Optional[RxModelID]) -> None:
        """
        Update the identity of the most recently loaded body part detection ML model for this workspace.
        :param model_id: The model identifier, or None if no model loaded.
        """
        self._mru_model = model_id

    @property
    def mru_segmentation_model(self) -> Optional[Path]:
        """ Most recently selected reach segmentation model folder, if any, for this workspace. """
        return self._mru_segmentation_model

    @mru_segmentation_model.setter
    def mru_segmentation_model(self, model_root: Optional[Path]) -> None:
        self._mru_segmentation_model = model_root

    @property
    def reach_seg_ctrls(self) -> RxReachSegControls:
        return self._reach_seg_ctrls

    @reach_seg_ctrls.setter
    def reach_seg_ctrls(self, ctrls: RxReachSegControls) -> None:
        # TODO: Range_restrict control values
        self._reach_seg_ctrls = ctrls

    def session_exists(self, session: RxSessionID) -> bool:
        """
        Does specified session exist in this workspace? Checks that the session folder exists AND is relative to the
        session root directory for this workspace. Also, a legacy-cam workspace only exposes legacy-cam sessions
        and vice versa.
        :param session: The recorded session.
        :return: True if found in workspace and is of the same type -- legacy vs fixed cam-- as this workspace; else
            False.
        """
        return (isinstance(session, RxSessionID) and session.exists() and
                session.location.is_relative_to(self._session_root) and (session.is_fixed_cam == self.is_fixed_cam))

    def models(self) -> List[RxModelID]:
        """
        The list of body part detection ML models belonging to this workspace (possibly empty). Note that a
        fixed-cam workspace cannot include legacy-cam models, and vice versa.
        """
        out: List[RxModelID] = list()
        if self._model_root.is_dir():
            model_folder_names = [p.name for p in self._model_root.iterdir() if p.is_dir()]
            for folder_name in model_folder_names:
                model_id = RxModelID.from_folder_name(self._model_root, folder_name)
                if (model_id is not None) and model_id.exists() and (model_id.is_fixed_cam == self.is_fixed_cam):
                    out.append(model_id)
        return out
