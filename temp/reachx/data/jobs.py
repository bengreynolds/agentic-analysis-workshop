import random
import shutil
import time
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional, Dict, List, Set, Union

import numpy as np
import scipy
import yaml
from PySide6.QtCore import QThread, Qt, Slot, QObject
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, \
    QStatusBar

from reachx.config.app_log import get_application_logger
from reachx.common import RX, RxModelID, RxSessionID, RxCam, RxBodyPart, RxNetworkType, \
    RxReachSegControls, validate_date_string, session_number_from_folder
from reachx.uicommon import BackgroundTask
from reachx.data.modelmgr import BodyPartMarker
from reachx.data.sessionmgr import SessionManager
from reachx.modeling.analyze import find_reach_segments
from reachx.modeling.taskfuncs import MAT_FILE_FOR_TRAINING, get_default_training_params, train_network, \
    analyze_sessions

_RX = RX()

JOB_LOG_DIR_NAME: str = "job_logs"
""" The log from a completed background job is saved in this directory within the application home directory. """
JOB_LOG_SUFFIX: str = "-job.log"
""" Each job log file starts with date/time in format 'YYYYMMDDD_HHMM', followed by this suffix. """
MAX_JOB_LOGS_SAVED: int = 20
""" Maximum number of job log files saved. Once reached, the oldest job log is removed. """

SESSION_CACHE_FNAME_OLD: str = "cache.yaml"
""" Name of old version of YAML file in which the session root directory content is cached (replaced a/o v0.6.0). """
SESSION_CACHE_FNAME: str = "cache_v2.yaml"
""" 
Name of YAML file in which session root directory content is cached (v0.6.0 and on). An 'L' or 'F' is appended to
each session number string to identify a "legacy-cam" vs the newer "fixed-cam" session.
"""


class RxJobType(IntEnum):
    """ Enumeration of time-consuming background jobs supported in ReachX. """
    TEST = 0
    """ **For test purposes**: Sleep for N seconds, delivering progress updates and messages every 1 second. """
    TRAIN = 1
    """ Train a neural net model to perform body part detection on ReachX session videos. """
    ANALYZE = 2
    """ 
    Use a previously trained neural net model to detect body part locations in the videos for one or more ReachX 
    experiment sessions, generate hand and pellet trajectories for each session, and (optionally) use those 
    trajectories to perform reach segmentation of each session.
    """
    CACHE = 3
    """ Scan a session root directory and build a cache of all sessions found therein. """
    REACH = 4
    """ Perform model-driven reach segmentation on a single ReachX experiment session. """
    ML_REACH = 5
    """ Perform externally trained model-based reach segmentation on a single ReachX experiment session. """


class RxJob(BackgroundTask):
    """
    A job object that performs one of a number of long-running (hours or more) tasks on a background thread, while
    providing the infrastructure to deliver progress updates, log messages for display on the GUI, and cancel the job
    prematurely.

    See ``RxJobType`` and ``__init__()`` for a list of supported jobs and the arguments that must be supplied to
    perform the work.
    """

    def __init__(self, job_type: RxJobType, **kwargs) -> None:
        """
        Construct a time-consuming job to be run on a background thread. Supported job types and the required keyword
        arguments for each:

        ``RxJobType.TEST``:
          - "dur": (int) Test duration in seconds. Range-restricted to [2, 20]. Default=10 if not specified.
        ``RxJobType.TRAIN``:
          - "model_id": (RxModelID) The model identifier.
          - "iter_num": (int) The model iteration to be trained.
          - "markers": (Dict[RxSessionID, List[BodyPartMarker]]) The annotated training data for the model iteration.
            Each key in the dictionary is the session ID for a marked session, and the corresponding value is the list
            of all body part markers attached to that session.
          - "net_type": (RxNetworkType) Pretrained neural net that is the starting point for training the model.
        ``RxJobType.ANALYZE``:
          - "model_id": (RxModelID) The model identifier.
          - "iter_num": (int) The model iteration to use.
          - "sessions": List[RxSessionID] The list of experiment sessions to be processed.
          - "batch_size": (int) # of video frames processed in one "batch". Defaults to 15. Restricted to [15..200].
          - "use_gpu": (Optional[bool]) If ``True``, then GPU is used for inference analysis when GPU support is
            available. If ``False``, GPU inference is disabled. Default = ``True``.
          - "enable_pp": (Optional[bool]) If ``True``, then the two session videos are analyzed in parallel using two
            separate processes; if ``False``, they are analyzed sequentially. Default = ``True``.
          - "seg_parms": (Optional[RxReachSegControls]) If specified, then reach segmentation is performed on each
            session after it is analyzed.
          - "calib_root": (Optional[Path]) For fixed-cam session analysis only, this is the calibration root directoy
            where all fixed-cam rig calibration data is stored. Ignored for legacy-cam analysis.
        ``RxJobType.CACHE``:
          - "session_root": (Path) The session root directory to be scanned
        ``RxJobType.REACH``:
          - "sesh_id": (RxSessionID) The session identifier.
          - "rx_scorer": (str) Identifies model analysis results folder within the session backing store.
          - "seg_parms": (RxReachSegControls) The parameters controlling the reach segmentation algorithm.
        ``RxJobType.ML_REACH``:
          - "sesh_id": (RxSessionID) The session identifier.
          - "rx_scorer": (str) Identifies model analysis results folder within the session backing store.
          - "ml_model_root": (Path) External segmentation model folder containing model.json.

        The keyword arguments are not thoroughly vetted for correctness -- expect bad behavior if you supply invalid
        arguments!

        :param job_type: The type of job to perform.
        :param kwargs: The job parameters are supplied via keyward arguments, as described for each supported job.
        """
        super().__init__()
        assert isinstance(job_type, RxJobType)

        self._job_type = job_type
        self._desc = "Unrecognized job"
        """ A short description of the job performed -- displayed above progress bar in dialog. """
        self._start_time: float = -1.0
        """ Time at which the job started running -- from ``time.time()``. -1 if job has not started. """
        self._total_run_time: float = 0.0
        """ Total elapsed run time for job in seconds. Will be 0 until job has finished."""
        self._failed: bool = False
        """ True if the job ended because an unhandled exception was caught. """
        self._test_dur: int = 0
        """ ``RxJobType.TEST``: Duration of idle test in seconds. Range 2-20. Default = 10. """
        self._model_id: Optional[RxModelID] = None
        """ ``RxJobType.TRAIN, .ANALYZE``: The identifier of a model. """
        self._iter_num: Optional[int] = None
        """ ``RxJobType.TRAIN, .ANALYZE``: The model iteration number. """
        self._markers: Dict[RxSessionID, List[BodyPartMarker]] = dict()
        """ 
        ``RxJobType.TRAIN``: Annotated training data for the model iteration to be trained, ie, list of body part
        markers attached to each marked session in the training data, keyed by session ID.
        """
        self._net_type: Optional[RxNetworkType] = None
        """ ``RxJobType.TRAIN``: Pretrained neural net that is starting point for training model. """
        self._sessions: List[RxSessionID] = list()
        """ ``RxJobType.ANALYZE``: List of experiment sessions to be processed. """
        self._batch_size: int = 15
        """ ``RxJobType.ANALYZE``: Number of frames processed in one 'batch'. """
        self._use_gpu: bool = True
        """ ``RxJobType.ANALYZE``: ``True`` to use GPU for session video analysis; ``False`` to disable GPU use. """
        self._enable_pp: bool = True
        """ ``RxJobType.ANALYZE``: ``True/False`` to analyze the two session videos in parallel/sequentially. """
        self._session_root: Optional[Path] = None
        """ ``RxJobType.CACHE``: The root directory to be scanned for session folders stored IAW ReachX convention. """
        self._sesh_id: Optional[RxSessionID] = None
        """ ``RxJobType.REACH``: Identifies session for which model-driven reach segmentation is performed. """
        self._rx_scorer: Optional[str] = None
        """ ``RxJobType.REACH``: Identifies the scorer (aka, model) used for the reach segmenation task. """
        self._seg_parms: Optional[RxReachSegControls] = None
        """ ``RxJobType.ANALYZE, .REACH``: The reach segmentation control parameter set. """
        self._ml_model_root: Optional[Path] = None
        """ ``RxJobType.ML_REACH``: External segmentation model folder. """
        self._calib_root: Optional[Path] = None
        """ ``RxJobType.ANALYZE``: Root directory containing calibration data for fixed-cam session analyis only. """

        if self._job_type == RxJobType.TEST:
            self._test_dur = 10
            if "dur" in kwargs and isinstance(kwargs["dur"], int):
                self._test_dur = min(max(2, kwargs["dur"]), 20)
            self._desc = f"Idling for {self._test_dur} seconds"
        elif self._job_type == RxJobType.TRAIN:
            if "model_id" in kwargs and isinstance(kwargs["model_id"], RxModelID):
                self._model_id = kwargs["model_id"]
            if "iter_num" in kwargs and isinstance(kwargs["iter_num"], int):
                self._iter_num = kwargs["iter_num"]
            if "markers" in kwargs and isinstance(kwargs["markers"], dict):
                self._markers = kwargs["markers"]
            if "net_type" in kwargs and isinstance(kwargs["net_type"], RxNetworkType):
                self._net_type = kwargs["net_type"]
            if isinstance(self._model_id, RxModelID):
                self._desc = f"Training model {self._model_id}, iteration {self._iter_num}"
        elif self._job_type == RxJobType.ANALYZE:
            if "model_id" in kwargs and isinstance(kwargs["model_id"], RxModelID):
                self._model_id = kwargs["model_id"]
            if "iter_num" in kwargs and isinstance(kwargs["iter_num"], int):
                self._iter_num = kwargs["iter_num"]
            if "sessions" in kwargs and isinstance(kwargs["sessions"], list):
                self._sessions = kwargs["sessions"]
            if "batch_size" in kwargs and isinstance(kwargs["batch_size"], int):
                self._batch_size = min(max(15, kwargs["batch_size"]), 200)
            if "use_gpu" in kwargs and isinstance(kwargs["use_gpu"], bool):
                self._use_gpu = kwargs["use_gpu"]
            if "enable_pp" in kwargs and isinstance(kwargs["enable_pp"], bool):
                self._enable_pp = kwargs["enable_pp"]
            if "seg_parms" in kwargs and isinstance(kwargs["seg_parms"], RxReachSegControls):
                self._seg_parms = kwargs["seg_parms"]
            if "calib_root" in kwargs and isinstance(kwargs["calib_root"], Path):
                self._calib_root = kwargs["calib_root"]
            if isinstance(self._model_id, RxModelID):
                self._desc = f"Session analysis using {self._model_id}, iteration {self._iter_num}"
        elif self._job_type == RxJobType.CACHE:
            if "session_root" in kwargs and isinstance(kwargs["session_root"], Path):
                self._session_root = kwargs["session_root"]
                path_str = str(self._session_root)
                if len(path_str) > 75:
                    path_str = f"...{path_str[-50:]}"
                self._desc = f"Building session cache from: {path_str}"
        elif self._job_type == RxJobType.REACH:
            if "sesh_id" in kwargs and isinstance(kwargs["sesh_id"], RxSessionID):
                self._sesh_id = kwargs["sesh_id"]
            if "rx_scorer" in kwargs and isinstance(kwargs["rx_scorer"], str):
                self._rx_scorer = kwargs["rx_scorer"]
            if "seg_parms" in kwargs and isinstance(kwargs["seg_parms"], RxReachSegControls):
                self._seg_parms = kwargs["seg_parms"]
            if isinstance(self._sesh_id, RxSessionID):
                self._desc = f"Finding reach segments for '{str(self._sesh_id)}' using {self._rx_scorer}"
        elif self._job_type == RxJobType.ML_REACH:
            if "sesh_id" in kwargs and isinstance(kwargs["sesh_id"], RxSessionID):
                self._sesh_id = kwargs["sesh_id"]
            if "rx_scorer" in kwargs and isinstance(kwargs["rx_scorer"], str):
                self._rx_scorer = kwargs["rx_scorer"]
            if "ml_model_root" in kwargs and isinstance(kwargs["ml_model_root"], Path):
                self._ml_model_root = kwargs["ml_model_root"]
            if isinstance(self._sesh_id, RxSessionID):
                self._desc = f"Finding ML reach segments for '{str(self._sesh_id)}' using {self._rx_scorer}"

    @property
    def short_description(self) -> str:
        """ A short human-facing description of the job. """
        return self._desc

    @property
    def job_type(self) -> RxJobType:
        """ The job type. """
        return self._job_type

    @property
    def sessions_affected(self) -> List[RxSessionID]:
        """ The session or sessions affected by the ``RxJobType.ANALYZE`` and ``RxJobType.REACH`` jobs; else empty. """
        if self._job_type in [RxJobType.REACH, RxJobType.ML_REACH]:
            return [self._sesh_id]
        elif self._job_type == RxJobType.ANALYZE:
            return self._sessions.copy()
        else:
            return []

    @property
    def execution_time(self) -> float:
        """
        Execution time in seconds. Will be 0 until job begins, and will reflect total execution time once job has
        finished.
        """
        if self._total_run_time > 0:
            return self._total_run_time
        else:
            return 0 if (self._start_time < 0) else time.time() - self._start_time

    @property
    def failed(self) -> bool:
        """ True if the job ended because an unhandled exception was caught. """
        return self._failed

    def run(self) -> None:
        """ Run the job. This method should be invoked on the background thread when that thread starts. """
        self.message_logged.emit("Starting up...")
        self._start_time = time.time()
        self._failed = False
        failed = False
        try:
            if self._job_type == RxJobType.TEST:
                self._run_idle_test()
            elif self._job_type == RxJobType.TRAIN:
                self._train_model()
            elif self._job_type == RxJobType.ANALYZE:
                self._analyze_sessions()
            elif self._job_type == RxJobType.CACHE:
                self._build_session_cache()
            elif self._job_type == RxJobType.REACH:
                find_reach_segments(self._sesh_id, self._rx_scorer, self._seg_parms, self)
            else:
                from reachx.modeling.segmentation.workflow import run_model_segmentation
                run_model_segmentation(self._sesh_id, self._rx_scorer, self._ml_model_root, self)
        except Exception as exc:
            failed = True
            self._failed = True
            if self.was_canceled():
                self.message_logged.emit("Operation cancelled.")
            else:
                get_application_logger().error(f"Background job failed: {exc}")
                self.message_logged.emit(f"Operation failed: {exc}")
        finally:
            if not failed:
                self.message_logged.emit("Operation cancelled." if self.was_canceled() else "Done!")
            self._total_run_time = time.time() - self._start_time
            self.finished.emit()

    def _run_idle_test(self) -> None:
        """
        Do nothing for a given test duration in seconds, while monitoring the cancel flag and delivering progress
        updates roughly once per second.
        """
        t_start = time.time()
        next_update: int = 1
        while True:
            if self.was_canceled():
                break
            time.sleep(0.1)
            t_elapsed = time.time() - t_start
            self.progress_updated.emit(int(100 * t_elapsed / self._test_dur))
            if t_elapsed > next_update:
                if next_update == self._test_dur:
                    break
                self.message_logged.emit(f"Elapsed time is {t_elapsed:0.3f} seconds.")
                next_update += 1

    def _train_model(self) -> None:
        """
        Train a ReachX body part detection model IAW the supplied job arguments. This job will take several hours
        to complete, even on a modern system with a fast GPU. Without a GPU, it could take several days.
        """
        try:
            if not self._model_id.check_iteration_folder_structure(self._iter_num):
                raise Exception("Invalid model iteration backing store")

            self.message_logged.emit("Clearing out data and model files from any previous training...")
            training_folder = self._model_id.training_set_folder(self._iter_num)
            model_files_folder = self._model_id.model_files_folder(self._iter_num)
            shutil.rmtree(training_folder, ignore_errors=True)
            shutil.rmtree(model_files_folder, ignore_errors=True)
            if training_folder.is_dir() or model_files_folder.is_dir():
                raise Exception("Failed to clear out old training data in model iteration's backing store. Stopping.")

            training_folder.mkdir()
            model_files_folder.mkdir()
            images_folder = self._model_id.training_set_images_folder(self._iter_num)
            images_folder.mkdir()

            # for each unique frame image saved, this dict maps the image file name to the list of body part markers
            # attached to that image. This is needed to prepare the MAT file for training the network model.
            img_2_bpm: Dict[str, List[BodyPartMarker]] = dict()

            # the body parts included in training data. The user may only be interested in a subset of the body parts
            # supported in ReachX. We can save some training time by excluding body parts that are not marked in the
            # training images!
            bp_included: Set[RxBodyPart] = set()

            self.message_logged.emit(f"Preparing training set from {len(self._markers)} marked session(s)...")
            sesh_mgr = SessionManager(suppress_log=True, auto_save=False)
            bp_list: List[BodyPartMarker]
            for sesh_id, bp_list in self._markers.items():
                # a fixed-cam model can only analyze fixed-cam sessions, and analogously for legacy-cam models. This is
                # considered a fatal error and should never happen.
                if self._model_id.is_fixed_cam != sesh_id.is_fixed_cam:
                    fixed = self._model_id.is_fixed_cam
                    raise Exception(f"A {'fixed-cam' if fixed else 'legacy-cam'} model cannot be used to analyze a "
                                    f"{'legacy-cam' if fixed else 'fixed-cam'} session")

                # load the next marked session
                emsg = sesh_mgr.load(sesh_id)
                if isinstance(emsg, str):
                    raise Exception(f"Failed to load session {sesh_id}: {emsg}")
                else:
                    self.message_logged.emit(f"Loaded session: {sesh_id}...")
                if self.was_canceled():
                    self.message_logged.emit("Operation canceled!")
                    break
                self.progress_updated.emit(0)

                # for image file naming
                n_fnum_digits = int(np.ceil(np.log10(sesh_mgr.total_frames)))
                pfx = sesh_id.session_file_prefix

                # the two cameras on which the user places body part markers are different for fixed-cam vs legacy-cam!
                cam_1: RxCam = RxCam.LEFT if self._model_id.is_fixed_cam else RxCam.SIDE
                cam_2: RxCam = RxCam.RIGHT if self._model_id.is_fixed_cam else RxCam.FRONT

                # process body part marker list to get image frame numbers we need to extract from the two cam videos.
                # We use sets since multiple BP markers could be attached to the same camera frame, which we should only
                # load and write once. Also, in some cases, there will be markers attached to the same frame from both
                # cameras.
                cam_1_frames: Set[int] = set()
                cam_2_frames: Set[int] = set()
                for bpm in bp_list:
                    if bpm.cam == cam_1:
                        cam_1_frames.add(bpm.frame)
                    elif bpm.cam == cam_2:
                        cam_2_frames.add(bpm.frame)
                    else:
                        self.message_logged.emit(f"Skipping body part marker defined on unexpected cam: {str(bpm.cam)}")
                        continue

                    # accumulate list of body parts marked on each unique frame image
                    fnum_str = str(bpm.frame).zfill(n_fnum_digits)
                    img_name = f"{pfx}img_{str(bpm.cam)}_{fnum_str}.png"
                    if img_name not in img_2_bpm:
                        img_2_bpm[img_name] = list()
                    img_2_bpm[img_name].append(bpm)

                    # update set of body parts included in training data
                    bp_included.add(bpm.part)

                # the unique frame numbers we need to load, in ascending order
                all_frames: List[int] = sorted(cam_1_frames | cam_2_frames)

                # extract and save all marked image frames from the primary and secondary cams for the loaded session
                self.message_logged.emit(f"Extracting and saving {len(all_frames)} video frames from session...")
                frames_processed = 0
                for frame_num in all_frames:
                    sesh_mgr.go_to_frame(frame_num)
                    fnum_str = str(frame_num).zfill(n_fnum_digits)
                    if frame_num in cam_1_frames:
                        p = Path(images_folder, f"{pfx}img_{str(cam_1)}_{fnum_str}.png")
                        if not sesh_mgr.save_current_frame_to_png(cam_1, p):
                            raise Exception(f"Failed to save {str(cam_1)} frame {frame_num} to {str(p.absolute())}")
                    if frame_num in cam_2_frames:
                        p = Path(images_folder, f"{pfx}img_{str(cam_2)}_{fnum_str}.png")
                        if not sesh_mgr.save_current_frame_to_png(cam_2, p):
                            raise Exception(f"Failed to save {str(cam_2)} frame {frame_num} to {str(p.absolute())}")

                    frames_processed += 1
                    self.progress_updated.emit(int(frames_processed * 100 / len(all_frames)))
                    if self.was_canceled():
                        self.message_logged.emit("Operation canceled!")
                        break

            # write image file names, and body part markers attached to each image to a MAT file in the format
            # prescribed by the imgaug dataset format. NOTE how the image list is shuffled -- so the neural net
            # processes them in no particular order.
            # IMPORTANT: RxBodyPart is an IntEnum, and the int value is what we put in "joints"
            self.message_logged.emit(f"Writing MAT file with neural net training data in 'imgaug' format...")
            img_names: List[str] = [k for k in img_2_bpm.keys()]

            # CRITICAL: The included body parts, sorted by RxBodyPart enum value. The 0-based position of the body
            # part (NOT the enum value) in this list is what is saved with its marked location in the MAT file. The
            # body part names, in the same order, are saved in cfg["all_joints_names"].
            bp_sorted = list(bp_included)
            bp_sorted.sort()

            frame_dims = _RX.camera_frame_dims(self._model_id.is_fixed_cam)   # (Wpix, Hpix)

            random.shuffle(img_names)
            data: List[Dict] = list()
            for img_name in img_names:
                #  size = [n_ch, Hpix, Wpix]. RGB assumed.
                item = {'image': img_name, 'size': np.array([3, frame_dims[1], frame_dims[0]]), 'joints': None}
                bp_markers = img_2_bpm[img_name]
                joints = np.zeros((len(bp_markers), 3), dtype='int64')
                for i, bpm in enumerate(bp_markers):
                    joints[i, 0] = bp_sorted.index(bpm.part)   # BP identified by 0-based pos in "all_joints_names"
                    joints[i, 1] = bpm.x_pix
                    joints[i, 2] = bpm.y_pix

                # converts to 1x1 cell containing a Nx3 matrix when written to MAT file
                outer = np.array([[None]], dtype=object)
                outer[0, 0] = np.array(joints, dtype='int64')
                item['joints'] = outer
                data.append(item)

            data_for_mat_file = np.array(
                [(np.array([data[idx]['image']], dtype='U'),
                  np.array([data[idx]['size']]),
                  data[idx]['joints'])
                 for idx in range(len(data))],
                dtype=[('image', 'O'), ('size', 'O'), ('joints', 'O')]
            )

            scipy.io.savemat(Path(training_folder, MAT_FILE_FOR_TRAINING), {"dataset": data_for_mat_file})

            # prepare training parameters, and save these to a yaml file in the model directory
            training_params = get_default_training_params()
            training_params["train_dir"] = str(training_folder)
            training_params["model_dir"] = str(model_files_folder)
            training_params["num_joints"] = len(bp_sorted)
            training_params["all_joints_names"] = [str(bp) for bp in bp_sorted]
            training_params["net_type"] = str(self._net_type)
            with open(Path(model_files_folder, RxModelID.TRAINING_CFG_FILE), 'w') as f:
                yaml.dump(training_params, f)

            train_network(training_params, self)

            self.progress_updated.emit(100)
        except Exception as e:
            self.message_logged.emit(f"ERROR: {e}")

    def _analyze_sessions(self) -> None:
        """
        Use a previously trained ReachX body part detection model to detect body part locations (with confidence scores)
        in each frame of the camera videos recorded during one or more experiment sessions. Otionally perform reach
        segmentation on each session analyzed.

        Note that the camera videos used for analysis purposes are ``RxCam.SIDE/.FRONT`` for legacy-cam sessions, and
        ``RxCam.LEFT/.RIGHT`` for fixed-cam sessions. Fixed-cam sessions and models can only appear in fixed-cam
        workspaces, and analogously for legacy-cam mode.
        """
        try:
            if not self._model_id.check_iteration_folder_structure(self._iter_num):
                raise Exception("Invalid model iteration backing store")

            model_files_folder = self._model_id.model_files_folder(self._iter_num)
            model_cfg_file = Path(model_files_folder, RxModelID.TRAINING_CFG_FILE)
            if not model_cfg_file.is_file():
                raise Exception(f"Training configuration file not found: {str(model_cfg_file)}")

            analyze_sessions(model_cfg_file, self._sessions, self, self._batch_size, self._use_gpu, self._enable_pp,
                             self._calib_root, self._seg_parms)
            self.progress_updated.emit(100)
        except Exception as e:
            self.message_logged.emit(f"ERROR: {e}")

    def _build_session_cache(self) -> None:
        """
        Build a ``List[RxSessionID]`` identifying all ReachX experiment sessions found in a workspace's session root
        directory.

        In practice, thousands of sessions could be stored under one session root. Scanning the 3-tiered structure
        (date -> rig -> session) of a ReachX session root could take several minutes in this case -- especially on a
        remote/networked file system. Hence the need to build the session cache on a background thread.

        The session list is saved in a cache file in the session root directory. That way, the next time a workspace
        is loaded, the session list can be quickly compiled from that cache file. Even if the cache file is available,
        its contents could be "stale" if some sessions were removed (very rare) or added (very likely) since the cache
        file was last saved.

        When support for the new "fixed-cam" rig was added in v0.6.0, the contents of the cache file had to be updated
        to distinguish fixed-cma sessions ("left" and "right" cams) from the older legacy-cam sessions ("side" and
        "front" cams). This is done by appending "L" or "F" to each session number in the cache file. The cache file
        name is now "cache_v2.yaml" instead of "cache.yaml". If the latter file is found, it is removed.

        **Procedure**:
         - Check for presence of pre-v0.6.0 version of the session cache file. If present, delete it and start
           building the current version of the cache file.
         - If cache file exists, process the file, prepare the session list, and deliver it via the ``data_ready``
           signal. The cached content is "trusted" -- it is assumed that all the session folders still exist. Any
           removed or added sessions will be caught in the subsequent scan. **IMPORTANT: Simply calling is_dir()
           thousands of times to verify session folders causes a significant delay and defeats the purpose of the
           cache file.**
         - Scan entire session root directory tree to find all session folders therein. If no cache file was present,
           deliver the "partial" session list after every 100 additional sessions found.
         - On completion, if the cache file was present and no changes were detected, do nothing. Otherwise, write the
           cache file and deliver the complete session list.
        """
        try:
            if not self._session_root.is_dir():
                raise Exception(f"Session root not found: {str(self._session_root)}")

            self.progress_updated.emit(0)

            # if old version of session cache file present, delete
            old_cache_file = Path(self._session_root, SESSION_CACHE_FNAME_OLD)
            if old_cache_file.is_file():
                old_cache_file.unlink(missing_ok=True)
                self.message_logged.emit("Old version of session cache file was removed! Rebuilding session cache...")

            # first, load from session cache file if it's there. This should be FAST
            got_cache = False
            cache_file = Path(self._session_root, SESSION_CACHE_FNAME)
            cached_session_list: List[RxSessionID] = list()
            if cache_file.is_file():
                self.message_logged.emit(f"Loading session cache file")
                cache_map = RxJob._load_cache_file(cache_file)
                if isinstance(cache_map, str):
                    self.message_logged.emit(cache_map)
                else:
                    cached_session_list = RxJob._prepare_session_ids_from_cache(self._session_root, cache_map)
                    msg = f"Found {len(cached_session_list)} sessions in cache file"
                    self.message_logged.emit(msg)
                    self.data_ready.emit(cached_session_list)
                    got_cache = True
            if self.was_canceled():
                return

            # whether or not cache is present, scan directory tree to discover any newly added sessions
            cache: Dict[str, Dict[str, List[str]]] = dict()
            sesh_ids_found: List[RxSessionID] = list()
            date_folders: List[Path] = [
                entry for entry in self._session_root.iterdir() if entry.is_dir() and validate_date_string(entry.name)
            ]
            if self.was_canceled():
                return
            self.message_logged.emit(f"Scanning {len(date_folders)} subdirectories for sessions...")
            for i, date_folder in enumerate(date_folders):
                # for each date folder, assume each subfolder corresponds to a rig
                rig_folders: List[Path] = [entry for entry in date_folder.iterdir() if entry.is_dir()]
                if self.was_canceled():
                    return
                for rig_folder in rig_folders:
                    # for each rig folder, include a session for each subfolder properly named
                    session_folders: List[Path] = [entry for entry in rig_folder.iterdir() if entry.is_dir()]
                    if self.was_canceled():
                        return
                    for session_folder in session_folders:
                        num_str = session_number_from_folder(session_folder.name)
                        if num_str is not None:
                            try:
                                is_fixed_cam = RxSessionID.check_for_fixed_cam_session_folder(session_folder)
                            except ValueError:
                                continue
                            if date_folder.name not in cache:
                                cache[date_folder.name] = dict()
                            if rig_folder.name not in cache[date_folder.name]:
                                cache[date_folder.name][rig_folder.name] = list()
                            cache[date_folder.name][rig_folder.name].append(num_str + ("F" if is_fixed_cam else "L"))
                            sesh_ids_found.append(
                                RxSessionID(self._session_root, rig_folder.name, date_folder.name, num_str,
                                            is_fixed_cam))
                            if (not got_cache) and (len(sesh_ids_found) % 100 == 0):
                                self.data_ready.emit(sesh_ids_found.copy())
                        if self.was_canceled():
                            return

                pct = int(100 * i / len(date_folders))
                self.progress_updated.emit(pct)

            # if session cache previously did not exist or has changed, write the cache file and deliver the full
            # session ID list found
            cache_stale = (not got_cache) or (set(cached_session_list) != set(sesh_ids_found))
            if cache_stale:
                n_cached, n_found = len(cached_session_list), len(sesh_ids_found)
                if not got_cache:
                    self.message_logged.emit(f"Writing {n_found} session IDs to session cache.")
                elif n_found > n_cached:
                    self.message_logged.emit(f"Updating session cache with {n_found - n_cached} additional sessions.")
                else:
                    self.message_logged.emit(f"Updating session cache: {n_cached - n_found} sessions removed.")

                with open(Path(self._session_root, SESSION_CACHE_FNAME), 'w') as f:
                    yaml.dump(cache, f)
                self.data_ready.emit(sesh_ids_found)
            else:
                self.message_logged.emit(f"Scan found no updates to session cache.")

            self.progress_updated.emit(100)
        except Exception as e:
            self.message_logged.emit(f"ERROR: {e}")

    @staticmethod
    def _load_cache_file(p: Path) -> Union[str, Dict[str, Dict[str, List[str]]]]:
        """
        Load session directory map from specified cache file. This two-level dictionary is keyed by date folder name
        at the first level and rig name at the second level, with the list of session numbers found (for a given date
        and rig) at the third level.

        As of v0.6.0, to distinguish between a fixed-cam and legacy-cam session, an "F" or "L" is appended to the
        3-digit session number to form the session number string. Thus, "013F" represents a fixed-cam session, while
        "005L" is a legacy-cam session.

        :param p: The session cache file, assumed to exist.
        :return: On success, the session directory map as described; else a brief error description.
        """
        try:
            with open(p, 'r') as f:
                cache = yaml.safe_load(f)
            cache = RxJob._validate_cache_file_content(cache)
            return cache
        except Exception as e:
            return f"Bad session cache file at {str(p)}: {e}"

    @staticmethod
    def _validate_cache_file_content(d: dict) -> Dict[str, Dict[str, List[str]]]:
        """
        Validate structure and content of the session cache file. This DOES NOT check that each session folder exists.
        :param d: The dictionary loaded from the cache file.
        :raises Exception: If any invalid content detected.
        :return: d (with specific typing)
        """
        out: Dict[str, Dict[str, List[str]]] = d
        if not all([isinstance(k, str) and validate_date_string(k) for k in d.keys()]):
            raise Exception("Invalid date folder name found")
        for date, rigs in d.items():
            if not (isinstance(rigs, dict) and all([isinstance(k, str) for k in rigs.keys()])):
                raise Exception(f"Invalid rig folder dictionary in date folder '{date}'")
            for rig, sesh_nums in rigs.items():
                if not (isinstance(sesh_nums, list) and (len(sesh_nums) == len(set(sesh_nums)))):
                    raise Exception(f"Invalid session number list: date='{date}', rig='{rig}'")
                for num in sesh_nums:
                    ok = (len(num) == 4) and (num[-1] in ["L", "F"]) and RxSessionID.is_valid_session_num(num[0:3])
                    if not ok:
                        raise Exception(f"Invalid session number list: date='{date}', rig='{rig}'")
        return out

    @staticmethod
    def _prepare_session_ids_from_cache(
            root: Path, cache: Dict[str, Dict[str, List[str]]]) -> List[RxSessionID]:
        """
        Convert a cached session directory map to a list of session identifiers. The directory map reflects the 3-tiered
        structure of a ReachX session root directory, eg: ``{'20241016': {'rig1': ['001F', '003F']}, ... }`` Note the
        letter "F" appended to the session number, identifying it as a fixed-cam session. Legacy-cam session numbers
        have an "L" at the end.

        For performance reasons, all session folders contained in the directory map are assumed to exist -- checking
        the existence of potentially thousands of directories takes a significant amount of time!

        :param root: The session root directory
        :param cache: Cached map of the directory's contents.
        :return: List of session IDs pointing to all session folders in the cached map.
        """
        out: List[RxSessionID] = list()
        for date, rig_dict in cache.items():
            for rig, sesh_nums in rig_dict.items():
                for num in sesh_nums:
                    is_fixed_cam = num.endswith("F")
                    out.append(RxSessionID(root, rig, date, num[0:3], is_fixed_cam))
        return out


class BlockingJobDialog(QDialog):
    """
    A custom modal dialog that runs a single time-consuming, cancellable job on a background thread while blocking the
    main application thread. It includes a progress bar, a read-only text widget to display messages posted by the
    job object, as well as a Cancel button so the user can cancel the job.

    **Usage**: Call ``run_job()`` with the defined job object to raise the modal dialog. This method returns control to
    the caller, but since the dialog is modal it will block all user input to the rest of the app. Do NOT use
    QDialog methods open() or exec().
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._job_label = QLabel("Short job description")
        """ Label describing the job in progress. """
        self._progress_bar = QProgressBar()
        """ Progress bar. May be determinate or not, depending on the job. """
        self._message_log = QTextEdit()
        """ A read-only text view in which messages posted by the running job are displayed. """
        self._elapsed_time_readout = QLabel("000:00:00")
        """ Label displays the job's elapsed execution time in hours, minutes, seconds. """
        self._cancel_btn = QPushButton("Cancel")
        """ If enabled, user can cancel the running job by clicking this button. """
        self._worker: Optional[QThread] = None
        """ The background thread on which the job is executed. Set up immediately before raising dialog. """
        self._job: Optional[RxJob] = None
        """ The job object. """

        # we want a modal dialog with no frame, no close button
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumSize(600, 600)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)

        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._message_log.setReadOnly(True)
        self._elapsed_time_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        self._elapsed_time_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self._cancel_btn.clicked.connect(self._on_cancel_or_close)

        layout = QVBoxLayout()
        layout.addWidget(self._job_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._message_log, stretch=1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Elapsed Time:"))
        row.addWidget(self._elapsed_time_readout, alignment=Qt.AlignmentFlag.AlignLeft)
        row.addStretch(1)
        row.addWidget(self._cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(row)
        self.setLayout(layout)

    @property
    def job_type(self) -> Optional[RxJobType]:
        """ The type of job that is running currently or was last run. None if no job has been run. """
        return None if (self._job is None) else self._job.job_type

    @property
    def sessions_affected(self) -> List[RxSessionID]:
        """
        List of any sessions that were the subject of the job running currently or last run. Empty if the job does
        not affect a session, or if no job has been run.
        """
        return [] if (self._job is None) else self._job.sessions_affected

    @property
    def job_failed(self) -> bool:
        """ True if the job running currently or last run failed. """
        return False if (self._job is None) else self._job.failed

    def run_job(self, job: RxJob) -> None:
        if self.isVisible():
            return

        # reset state
        self._message_log.setText("")
        self._cancel_btn.setText("Cancel")
        self._progress_bar.setValue(0)
        self._job_label.setText(job.short_description)

        self._job = job
        self._worker = QThread()
        self._connect_signals()

        self._worker.start()
        super().open()

    # noinspection PyUnresolvedReferences
    def _connect_signals(self) -> None:
        self._job.progress_updated.connect(self._on_progress_update)
        self._job.message_logged.connect(self._on_message_logged)
        self._job.finished.connect(self._on_job_finished)

        self._job.moveToThread(self._worker)
        self._worker.started.connect(self._job.run)
        self._worker.finished.connect(self._job.deleteLater)
        self._worker.finished.connect(self._worker.deleteLater)
        self._job.finished.connect(self._worker.quit)
        self._job.finished.connect(self._worker.wait)

    @Slot(int)
    def _on_progress_update(self, pct: int) -> None:
        self._progress_bar.setValue(pct)
        self._elapsed_time_readout.setText(_RX.format_elapsed_time(self._job.execution_time))

    @Slot(str)
    def _on_message_logged(self, msg: str) -> None:
        if isinstance(msg, str) and len(msg) > 0:
            now = datetime.now()
            self._message_log.append(f"{now.strftime('%d %b %H:%M:%S : ')}{msg}")
        self._elapsed_time_readout.setText(_RX.format_elapsed_time(self._job.execution_time))

    @Slot()
    def _on_job_finished(self) -> None:
        """
        When background job has finished, the "Cancel" button becomes a "Close" button. The user must press the button
        again to extinguish the dialog. This gives the user a chance to peruse the message history first.
        """
        self._cancel_btn.setText("Close")
        self._cancel_btn.setEnabled(True)   # upon cancel, the button is disabled until job finishes
        if not self._job.was_canceled() and not self._job.failed:
            self._progress_bar.setValue(100)
        self._elapsed_time_readout.setText(_RX.format_elapsed_time(self._job.execution_time))
        self._save_job_log_and_delete_oldest_if_necessary()

    @Slot()
    def _on_cancel_or_close(self):
        """ Handler called when the Cancel/Close button is clicked. """
        if self._cancel_btn.text() == "Cancel":
            self._cancel_btn.setEnabled(False)
            self._job.cancel()
        else:
            self.accept()

    def _save_job_log_and_delete_oldest_if_necessary(self) -> None:
        """
        Helper method called just prior to closing the blocking dialog. It saves the log messages to a job log file
        and ensures the number of saved job logs does not grow indefinitely.
        """
        # we don't bother saving job logs for these jobs
        if self._job.job_type in [RxJobType.TEST, RxJobType.CACHE]:
            return

        log_dir: Path = Path(_RX.HOME, JOB_LOG_DIR_NAME)
        log_dir.mkdir(parents=True, exist_ok=True)

        # remove oldest job log(s) if the job log directory already has the maximum. NOTE that the job log filenames
        # have the form 'YYYYMMDD_HHMM-task.log', so sorting the filenames alphabetically sorts them chronologically!
        try:
            job_log_files = [f for f in log_dir.iterdir() if f.name.endswith(JOB_LOG_SUFFIX)]
            if len(job_log_files) >= MAX_JOB_LOGS_SAVED:
                job_log_files.sort(key=lambda f: f.name[:13])
                while len(job_log_files) >= MAX_JOB_LOGS_SAVED:
                    f = job_log_files.pop()
                    f.unlink(missing_ok=True)
        except Exception:
            pass

        # now write the log for the just completed job.
        log_text: str = self._message_log.toPlainText()
        now = datetime.now()
        log_file: Path = Path(log_dir, f"{now:%Y%m%d_%H%M}{JOB_LOG_SUFFIX}")
        try:
            with open(log_file, 'w') as f:
                f.write(log_text)
        except Exception as e:
            get_application_logger().warning(f"Failed to write background job log: {e}")


class NonBlockingJobRunner(QObject):
    """
    A non-blocking background job runner -- an alternative to ``BlockingJobDialog``.

    Installs a progress bar in the application window's status bar and runs a single background job at a time,
    updating the progress bar as the job executes and displaying any messages in the status bar as well.
    """
    def __init__(self, status_bar: QStatusBar):
        """
        Construct the non-blocking job runner, which will install a progress bar and display progress messages in the
        application status bar.
        :param status_bar: The application status bar widget.
        """
        super().__init__()
        self._status_bar = status_bar
        """ The application main window's status bar. """
        self._progress_bar = QProgressBar()
        """ A progress bar embedded in the status bar ONLY when a background job is running. """
        self._worker: Optional[QThread] = None
        """ The background thread on which the job is executed. Set up immediately before raising dialog. """
        self._job: Optional[RxJob] = None
        """ The job object. """
        self._running: bool = False
        """ True when running a job. """

        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()

    def run_job(self, job: RxJob) -> None:
        if self._running:
            return

        # install progress bar in application status bar
        self._progress_bar.setValue(0)
        self._status_bar.addPermanentWidget(self._progress_bar)
        self._progress_bar.show()
        self._status_bar.showMessage(job.short_description)

        self._running = True
        self._job = job
        self._worker = QThread()
        self._connect_signals()

        self._worker.start()

    def job_in_progress(self) -> bool:
        return self._running

    def cancel_and_wait(self, wait_time: int = 2000) -> bool:
        """
        If a job is still running and has not already been canceled, cancel and wait for the specified time for the
        thread to stop.
        :param wait_time: Maximum wait time
        :return: True if job has stopped within the specified wait time
        """
        if self.job_in_progress() and not self._job.was_canceled():
            self._job.cancel()
            self._worker.wait(wait_time)
        return not self.job_in_progress()

    # noinspection PyUnresolvedReferences
    def _connect_signals(self) -> None:
        self._job.progress_updated.connect(self._on_progress_update)
        self._job.message_logged.connect(self._on_message_logged)
        self._job.finished.connect(self._on_job_finished)

        self._job.moveToThread(self._worker)
        self._worker.started.connect(self._job.run)
        self._worker.finished.connect(self._job.deleteLater)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._on_worker_finished)
        self._job.finished.connect(self._worker.quit)

    @Slot(int)
    def _on_progress_update(self, pct: int) -> None:
        self._progress_bar.setValue(pct)

    @Slot(str)
    def _on_message_logged(self, msg: str) -> None:
        if isinstance(msg, str) and len(msg) > 0:
            self._status_bar.showMessage(msg)

    @Slot()
    def _on_job_finished(self) -> None:
        """ Clean up after background job finished: Hide progress bar and reset status bar message to "Ready". """
        self._progress_bar.hide()
        self._status_bar.showMessage("Ready")

    @Slot()
    def _on_worker_finished(self) -> None:
        self._running = False
        self._worker = None
        self._job = None
        self._status_bar.removeWidget(self._progress_bar)
