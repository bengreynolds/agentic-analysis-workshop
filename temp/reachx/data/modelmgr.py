"""
Neural net model training and video analysis module for ReachX.

This module handles the details of creating, training, and applying a machine-learning model for detecting body parts in
ReachX session videos.

Classes:
 - ``BodyPartMarker`` : A named tuple defining a single body part marker.
 - ``SessionMarkers``: The set of all body part markers attached to any video frame from the master and secondary camera
   videos of an experiment session. The "training data" for a body part detection ML model is a collection of one or
   more "marked" sessions. For each such session, the user has manually marked various body parts on selected frames
   from the master and secondary camera videos. Handles loading/saving marker set to a text file.
 - ``ModelManager``: Encapsulates the machine-learning model project currently loaded into ReachX. Handles model
   creation and loading/unloading an existing model. Manages the "training data" for the model, with methods for
   attaching and detaching marker to selected frames from one or more experiment sessions. Supports creating new model
   iterations, deleting an existing iteration, and switching among available iterations.

NOTE: Support for the new fixed-cam experiment setup was introduced in v0.6.0. ReachX workspaces, sessions, and models
come in two flavors:
 - **Fixed-cam**: Two cameras only, ``RxCam.LEFT`` and ``RxCam.RIGHT``. Camera frame size is 256x26. Calibration files
   required to convert pixels to millimeters.
 - **Legacy-cam**: Two primary view cameras ``RxCam.SIDE`` and ``RxCam.LEFT``. Other cams may or may not be present, but
   are not used for body part detection. Camera frame size is 300x200, and static calibration constants are used to
   convert pixels to millimeters.
A fixed-cam workspace only exposes fixed-cam sessions and models. Analogously for a legacy-cam workspace. In addition,
the collection of body parts (``RxBodyPart``) that may be marked is different for a fixed-cam vs legacy-cam session.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import NamedTuple, List, Tuple, Dict, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

from reachx.config.app_log import get_application_logger
from reachx.common import RxBodyPart, RxCam, RxSessionID, RxModelID


MIN_RIGHT_HAND_MARKERS: int = 50
""" Minimum number of right-hand-related body part markers required before a ML model iteration can be trained. """
MIN_PELLETS: int = 50
""" Minimum number of pellet-related markers required before a ML model iteration can be trained. """
MIN_LEFT_HAND_MARKERS: int = 50
""" Fixed-cam sessions only: min number of left-hand-related markers required to train a ML model iteration. """
MIN_SPECIAL_MARKERS: int = 20
""" 
Fixed-cam sessions only: Mininum recommended marker count for each of these special purpose markers: "Star",
"Triangle", "Tongue_mid", and "Diamond".
"""


class BodyPartMarker(NamedTuple):
    """ A body part marker attached to a video frame recorded on one of the primary cameras during a session. """
    frame: int
    """ Frame number (in the master camera timeline)."""
    cam: RxCam
    """
    The source camera: ``RxCam.SIDE`` or ``RxCam.FRONT`` for a legacy-cam session; ``RxCam.LEFT`` or ``RxCam.RIGHT``
    for a fixed-cam session. """
    part: RxBodyPart
    """ Identifies the body part marked. """
    x_pix: int
    """ X-coordinate of marker location in pixels (assuming origin at lower left corner of frame image). """
    y_pix: int
    """ Y-coordinate of marker location in pixels (assuming origin at lower left corner of frame image). """


class SessionMarkers:
    """
    The "training data" for the network model that performs body part detection on ReachX session videos is a
    collection of body part-marked frames from the two primary videos recorded aacross one or more experiment sessions.

    This container encapsulates the entire set of body part markers for a single session.
    """
    @staticmethod
    def load(src: Path) -> Tuple[str, SessionMarkers]:
        """
        Load all body part markers found in the text file specified.

        Each line in the text file defines a single body part marker as a sequence of five (5) whitespace-separated
        integers: **frame cam bodypart x_pix y_pix**, where **frame** is the frame number, **cam** identifies the
        camera source (legacy cams``RxCam.SIDE/FRONT`` or fixed cams ``RxCam.LEFT/RIGHT``), **bodypart** identifies the
        body part marked (see ``RxBodyPart``), and **(x_pix, y_pix)** is the marked location in pixels within the
        specified video frame. The location is WRT an origin at the top-left corner, X increasing rightward and Y
        increasing downward.

        If an error occurs while loading the markers file, this should not be considered fatal, but the issue should be
        reported to the user.

        :param src: The source file for the marker set to be loaded.
        :return: ("", marker set) if successful. If the marker file is not found or could not be parsed, returns
            (error_msg, empty marker set).
        """
        if not src.is_file():
            return f"File not found", SessionMarkers()

        try:
            raw = np.loadtxt(src, dtype='i4', ndmin=2)   # always want 2D array even if there's only one line!
            if len(raw.shape) != 2 or raw.shape[1] != 5:
                raise Exception("Invalid format - expect five columns of whitepace-separated ints")
            out = SessionMarkers()
            for row in range(raw.shape[0]):
                frame, cam, bp, x, y = raw[row]
                # NOTE: The int() casts are important to get rid of numpy int objects!!!
                marker = BodyPartMarker(int(frame), RxCam(int(cam)), RxBodyPart(int(bp)), int(x), int(y))
                if frame in out._markers:
                    frame_markers = out._markers[frame]
                else:
                    frame_markers: List[BodyPartMarker] = list()
                    out._markers[frame] = frame_markers
                frame_markers.append(marker)
            out._update_frame_num_cache()
            return "", out
        except Exception as e:
            return f"Error loading body part markers: {e}.", SessionMarkers()

    @staticmethod
    def save(dst: Path, mark_set: SessionMarkers) -> str:
        """
        Save a body part marker set to the specified file. An existing file is truncated and overwritten.

        :param dst: The destination file for the body part marker set.
        :param mark_set: The body part marker set to be saved. **If the marker set is empty and the specified file
           exists, that file is removed!**
        :return: "" on success; else a brief error description
        """
        # if marker set empty and the destination file exists, delete it.
        n = mark_set.num_markers()
        if n == 0:
            dst.unlink(missing_ok=True)
            return ""

        try:
            n = mark_set.num_markers()
            raw = np.zeros((n, 5), dtype='i4')
            row_idx = 0
            # when saving, save in chronological order
            frames = [f for f in mark_set._markers]
            frames.sort()
            for f in frames:
                frame_markers = mark_set._markers[f]
                for marker in frame_markers:
                    raw[row_idx] = [marker.frame, marker.cam, marker.part, marker.x_pix, marker.y_pix]
                    row_idx += 1
            np.savetxt(dst, raw, fmt='%d', header='frame  cam  bodypart  x_pix  y_pix')
            return ""
        except Exception as e:
            return f"Unable to save body part markers: {e}"

    def __init__(self):
        """ Construct an empty set of body part markers. """
        self._markers: Dict[int, List[BodyPartMarker]] = dict()
        """ Maps video frame number to list of body part markers defined on that frame across both side/front cams."""
        self._frame_nums = np.array([], dtype='i4')
        """ The marked frame numbers sorted in ascending order. """

    def num_markers(self, cam: Optional[RxCam] = None) -> int:
        """
        Total number of body parts marked across the two primary cam videos for the entire session, or the number of
        body parts marked on either video.
        :param cam: The camera source.
        :return: Total number of markers defined on specified camera video. If ``cam==None``, total number of markers
            across the two primary camera videos.
        """
        out = 0
        if cam is None:
            out = sum([len(markers) for _, markers in self._markers.items()])
        else:
            for _, marks in self._markers.items():
                out += sum([1 for m in marks if m.cam == cam])
        return out

    def num_marked_frames(self, cam: Optional[RxCam] = None) -> int:
        """
        Total number of unique video frames with at least one body part marked across the two primary cam videos, or
        the number of marked frames on either video. (In general there will be more markers than marked frames
        because multiple body parts are marked in any given frame.)
        :param cam: The camera source.
        :return: Total number of marked frames across specified camera video; if ``cam==None``, total number of marked
            frames across the two primary cam videos.
        """
        if cam is None:
            return len(self._markers)
        else:
            count = 0
            for _, frame_markers in self._markers.items():
                if next((m for m in frame_markers if m.cam == cam), None) is not None:
                    count += 1
            return count

    def marked_frames(self, cam: Optional[RxCam] = None) -> List[int]:
        """
        Get the list of video frames with at least one body part marked on the specified camera's video recording.
        :param cam: The camera source.
        :return: List of marked frames on specified camera's video. If cam is None, returns list of all marked frames
            on the two primary cams. Frame numbers are in chronological order.
        """
        if cam is None:
            return sorted([f for f in self._markers])
        else:
            out: List[int] = []
            for f, frame_markers in self._markers.items():
                if next((m for m in frame_markers if m.cam == cam), None) is not None:
                    out.append(f)
            out.sort()
            return out

    def get_markers_for_frame(self, frame: int, cam: Optional[RxCam] = None) -> List[BodyPartMarker]:
        """
        Get the list of all body part markers attached to the specified frame and, optionally, to a particular camera.
        :param frame: A frame number.
        :param cam: The source camera. If ``cam==None``, method returns all body part markers on the specified frame
            across the two primary cams.
        :return: List of body part markers requested, possibly empty.
        """
        out = []
        frame_markers = self._markers.get(frame, None)
        if frame_markers is not None:
            if cam is None:
                out.extend(frame_markers)
            else:
                out.extend([m for m in frame_markers if m.cam == cam])
        return out

    def num_total_markers(self, cam: RxCam, part: RxBodyPart) -> int:
        """
        Get body part marker count for a specified camera and body part
        :param cam: Camera identifier.
        :param part: Body part identifier.
        :return: Number of frames from specified camera on which specified body part is marked.
        """
        count = 0
        for _, frame_markers in self._markers.items():
            if next((m for m in frame_markers if (m.cam == cam and m.part == part)), None) is not None:
                count += 1
        return count

    def marker_counts(self, parts: List[RxBodyPart]) -> Dict[RxBodyPart, int]:
        """
        Get total number of occurrences of specified body parts across this marker set.
        :param parts: A list of body part identifiers (repeats ignored).
        :return: A dictionary of body part counts, keyed by body part identifier.
        """
        p_set = set(parts)
        counts: Dict[RxBodyPart, int] = {p: 0 for p in p_set}
        for _, frame_markers in self._markers.items():
            parts_in_frame = [bpm.part for bpm in frame_markers]
            for bp in p_set:
                counts[bp] += parts_in_frame.count(bp)
        return counts

    def num_total_hand_and_pellet_markers(self) -> Tuple[int, int, int]:
        """
        Total number of hand- and pellet-related markers across this marker set.
        :return: 3-tuple (# right-hand markers, #left-hand markers, # pellet markers). Note that there are no defined
            left-hand markers for a legacy-cam session. For a fixed-cam session, there are 3 distinct "hand poses" that
            may be marked for either hand.
        """
        n_r_hand, n_l_hand, n_pellet = 0, 0, 0
        for _, frame_markers in self._markers.items():
            for m in frame_markers:
                if m.part.is_left_hand():
                    n_l_hand += 1
                elif m.part.is_right_hand():
                    n_r_hand += 1
                elif m.part == RxBodyPart.PELLET:
                    n_pellet += 1

        return n_r_hand, n_l_hand, n_pellet

    def add_mark(self, marker: BodyPartMarker) -> None:
        """
        Add or update a body part marker to this marker set.
        :param marker: The body part marker to add/update.
        """
        valid_cams_for_markers = [RxCam.SIDE, RxCam.FRONT, RxCam.LEFT, RxCam.RIGHT]
        if (not isinstance(marker, BodyPartMarker)) or (marker.cam not in valid_cams_for_markers):
            raise ValueError("Invalid body bart marker")
        if marker.frame in self._markers:
            frame_markers = self._markers[marker.frame]
        else:
            frame_markers: List[BodyPartMarker] = list()
            self._markers[marker.frame] = frame_markers
            self._update_frame_num_cache()
        # if body part marker already defined, replace it -- the location has changed
        found = next((i for i, m in enumerate(frame_markers) if (m.cam == marker.cam and m.part == marker.part)), None)
        if found is None:
            frame_markers.append(marker)
        else:
            frame_markers[found] = marker

    def remove_mark(self, marker: BodyPartMarker) -> bool:
        """
        Remove a body part marker from this marker set.
        :param marker: The body bart marker to remove.
        :return: True if marker was removed, False if not found.
        """
        valid_cams_for_markers = [RxCam.SIDE, RxCam.FRONT, RxCam.LEFT, RxCam.RIGHT]
        if (not isinstance(marker, BodyPartMarker)) or (marker.cam not in valid_cams_for_markers):
            raise ValueError("Invalid body bart marker")
        if marker.frame in self._markers:
            frame_markers = self._markers[marker.frame]
            found = next((i for i, m in enumerate(frame_markers)
                          if (m.cam == marker.cam and m.part == marker.part)), None)
            if found > -1:
                frame_markers.pop(found)
                if len(frame_markers) == 0:
                    _ = self._markers.pop(marker.frame, None)
                    self._update_frame_num_cache()
                return True
        return False

    def clear(self, frame: Optional[int] = None) -> bool:
        """
        Discard all body part markers attached to a single video frame, or discard markers altogether.
        :param frame: A frame number. If None, the body part marker set is cleared entirely! Default = None.
        :return: True if at least one body part marker was removed; False if no changes were made.
        """
        changed = False
        if isinstance(frame, int):
            frame_markers = self._markers.pop(frame, None)
            if frame_markers is not None:
                changed = True
                frame_markers.clear()
                self._update_frame_num_cache()
        elif self.num_marked_frames() > 0:
            for _, frame_markers in self._markers.items():
                frame_markers.clear()
            self._markers.clear()
            self._update_frame_num_cache()
            changed = True
        return changed

    def _update_frame_num_cache(self) -> None:
        self._frame_nums = np.array([frame for frame in self._markers], dtype='i4')
        self._frame_nums.sort()


class ModelManager(QObject):
    """
    Encapsulates and manages the currently loaded body part detection model, including any and all iterations of that
    model. Manages the "training data" for the currently loaded model iteration: a list of ``BodyPartMarker``s attached
    to selected video frames on the two primary cameras recording during one or more sessions.

    All models are located in subdirectories of the active workspace's "model root". The model folder name has the
    form ``<model>-<date><suffix>``, where ``<model>`` is the model project name, ``<date>`` is the model creation
    date in the form "YYYYMMDD", and ``<suffix>`` = "-fix" for a fixed-cam model and "" for a legacy-cam model.

    Under the model folder are zero or more iteration folders named "iterN", where N is the iteration number. Each
    iteration folder contains three folders:
     - "annotations" : All training data for the model iteration are stored here. ``ModelManager`` persists the defined
       body part markers for each marked session here.
     - "training-set" : Before the model iteration is trained, the training data must be put in a form for digestion by
       the neural net training algorithm. ``ModelManager`` merely creates the subfolder; the training procedure
       populates its conteents.
     - "trained-model" : Files generated during training go here, and these are used when analyzing session videos with
       the model iteration. Again, ``ModelManager`` merely creates this subfolder.
    Note that the structure of a model project folder is defined by ``RxModelID``.
    """
    markers_changed: Signal = Signal(RxSessionID)
    """ 
    A marker has been attached or detached from the specified marked session in the training data for the current
    model iteration, or all markers for that session have been discarded. If the session ID is None, then any and all
    marked sessions have been discarded from the training data, which is now empty.
    """
    iter_loaded: Signal = Signal()
    """ 
    A different iteration of the current ML model has been loaded and made "active". Note this signal is also sent
    when the current iteration is deleted or when a new iteration is created and made the active iteration.
    """

    def __init__(self):
        """ Create the body part detection model manager in a startup state, with no model loaded. """
        super().__init__()
        self._model_id: Optional[RxModelID] = None
        """ Identifier for the currently loaded body part detection ML model, or None if no model loaded. """
        self._session_root: Optional[Path] = None
        """ The session root path -- the workspace directory containing all experiment session folders. """
        self._existing_iterations: List[int] = list()
        """ 
        List of all existing model iterations by iteration number in ascending order (highest iteration is always the 
        most recently created). 
        """
        self._curr_iter: Optional[int] = None
        """ The current loaded model iteration, or None if no model loaded. """
        self._marked_sessions: Dict[RxSessionID, SessionMarkers] = dict()
        """ 
        The training data for the currently loaded model iteration: Selected sideCam and frontCam video frames marked 
        with body part locations, culled from at least one but typically multiple experiment sessions. 
        """
        self._save_timer = QTimer()
        """ 
        A 60-sec timer that is started whenever the training data for the currently loaded model iteration is
        altered for the first time since the last save. The timer is stopped each time training data is saved.
        """
        self._save_timer.setInterval(60000)
        self._save_timer.timeout.connect(lambda: self._save_timer_update(0))

    @staticmethod
    def create_model(model_root: Path, proj_name: str, is_fixed_cam: bool) -> Optional[RxModelID]:
        """
        Create a new body part detection model under the specified project name, with today's date. The model project
        folder is created under the model root directory and will contain a single EMPTY iteration -- no annotations,
        no trained model.

        :param model_root: The current workspace's model root. All model projects are stored in this directory.
        :param proj_name: The name assigned to the project (cannot be changed).
        :param is_fixed_cam: True for a fixed-cam model, False for legacy-cam. In ReachX, a fixed-cam workspace only
            works on fixed-cam models and sessions. Sessions recorded on the fixed-cam setup cannot be mixed with
            sessions recorded on the old legacy-cam setup.
        :return: The identifier for the newly created model, or None if model folder could not be created (in which
            case an error message is posted to the application log.
        """
        model_id = RxModelID.create_model_id(model_root, proj_name, is_fixed_cam)
        if model_id is None:
            get_application_logger().error(f"Unable to create model project - invalid project name: {proj_name}")
            return None
        if model_id.exists():
            get_application_logger().error(f"Model project already exists!")
            return None

        try:
            model_id.location.mkdir()
            model_id.iteration_folder(0).mkdir()
            model_id.markers_folder(0).mkdir()
            model_id.training_set_folder(0).mkdir()
            model_id.model_files_folder(0).mkdir()
        except Exception as e:
            get_application_logger().error(f"Failed to create model project directory structure: {e}")
            return None

        get_application_logger().info(f"New model project created at: {str(model_id.location.absolute())}")
        return model_id

    @property
    def id(self) -> Optional[RxModelID]:
        """ Identifier for the current ML body part detection model, or None if no model is currently loaded. """
        return self._model_id

    @property
    def is_fixed_cam(self) -> bool:
        """ Returns ``True`` if current model is fixed-cam; ``False`` if legacy-cam or no model loaded. """
        return (self.id is not None) and self.id.is_fixed_cam

    @property
    def model_loaded(self) -> bool:
        """ Is a ML body part detection model currently loaded? """
        return isinstance(self._model_id, RxModelID)

    @property
    def existing_iterations(self) -> List[int]:
        """ List of existing model iteration numbers for the current model, in ascending order. """
        return self._existing_iterations.copy()

    @property
    def current_iteration(self) -> Optional[int]:
        """ The current iteration for the model loaded, or None if no model is loaded. """
        return self._curr_iter

    def create_and_load_new_iteration(self, copy: bool = False) -> int:
        """
        Create a new model iteration, initially empty OR containing a copy of the annotated body part markers
        defined on the current iteration. The new iteration is assigned an iteration number one greater than the
        highest iteration. If successful, the new iteration becomes the active iteration for the current model, and the
        `iter_loaded` signal is emitted.
        :param copy: If True, any body part markers defined on the current model iteration are copied to the backing
            store for the new iteration. Default = False.
        :return: The number assigned to the new model iteration, or -1 if operation failed (in which case an error
            message is posted to the application log.
        """
        if not self.model_loaded:
            get_application_logger().error("Failed to create a model iteration because no model is loaded!")
            return -1

        # create the folder structure for the new model iteration
        iter_num = self._existing_iterations[-1] + 1
        iter_dir: Optional[Path] = None
        try:
            iter_dir = self._model_id.iteration_folder(iter_num)
            iter_dir.mkdir()
            self._model_id.markers_folder(iter_num).mkdir()
            self._model_id.training_set_folder(iter_num).mkdir()
            self._model_id.model_files_folder(iter_num).mkdir()
        except Exception as e:
            get_application_logger().error(f"Failed to create backing store for model iteration {iter_num}: {e}")
            if isinstance(iter_dir, Path):
                shutil.rmtree(iter_dir, ignore_errors=True)
            return -1

        if copy and (self._curr_iter is not None):
            # IMPORTANT: The current iteration's markers may have changed since the last time they were loaded, so
            # we need to save them before we copy
            self._save_markers()

            copy_ok = True
            marker_files_to_copy = self._model_id.get_all_existing_marker_files(self._curr_iter)
            dst_dir = self._model_id.markers_folder(iter_num)
            for f in marker_files_to_copy:
                try:
                    shutil.copy(f, dst_dir)
                except Exception:
                    copy_ok = False
            if not copy_ok:
                get_application_logger().warning(
                    f"Some or all body part marker files could not be copied to iteration {iter_num}")

        self._existing_iterations.append(iter_num)
        get_application_logger().info(f"Model {str(self._model_id)} - Iteration {iter_num} created.")
        self.load_iteration(iter_num)
        return iter_num

    def load_iteration(self, iter_num: int) -> None:
        """
        Load the specified iteration of the currently loaded body part detection ML model. If successful, the
        `iter_loaded` signal is emitted. On failure, an error message is posted to the application log.

        If the specified iteration is already the current model iteration, no action is taken.

        :param iter_num: The requested iteration number
        """
        if iter_num == self._curr_iter:
            return
        if iter_num not in self._existing_iterations:
            get_application_logger().error(f"Cannot load non-existent model iteration: {iter_num}")
            return

        self._save_markers()
        self._clear_markers()
        self._curr_iter = iter_num
        self._load_markers()
        QTimer.singleShot(10, lambda: self.iter_loaded.emit())

    def remove_current_iteration_permanently(self) -> None:
        """
        Remove the currently active iteration of the loaded body part detection ML model. The highest-numbered remaining
        iteration becomes the active iteration; if no iterations remain after deletion, then a new empty model iteration
        is created with iteration number 0. The `iter_loaded` signal is emitted to indicate that a different model
        iteration has become active.

        Since the deleted iteration's backing store within the model project folder is destroyed -- including all
        training data and trained model files --, confirm the user's intent before invoking this method. No action taken
        if no model is currently loaded.
        """
        if not self.model_loaded:
            return

        self._save_timer_update(-1)  # stop periodic save timer now

        deleted_iter_folder = self._model_id.iteration_folder(self._curr_iter)
        self._existing_iterations.remove(self._curr_iter)
        self._curr_iter = -1
        self._clear_markers()

        shutil.rmtree(deleted_iter_folder, ignore_errors=True)
        if deleted_iter_folder.exists():
            get_application_logger().error(f"A problem occurred while removing model "
                                           f"iteration at: {str(deleted_iter_folder.absolute())}.")

        if len(self._existing_iterations) > 0:
            self._curr_iter = self._existing_iterations[-1]
            self._load_markers()
        else:
            get_application_logger().info(f"No remaining iterations for model {str(self._model_id)}. "
                                          f"Reinitializing with empty iteration 0.")
            self._curr_iter = 0
            self._existing_iterations.append(self._curr_iter)

            # this should always work. If not, then ModelManager will be in a bad state...
            iter_dir: Optional[Path] = None
            try:
                iter_dir = self._model_id.iteration_folder(0)
                iter_dir.mkdir()
                self._model_id.markers_folder(0).mkdir()
                self._model_id.training_set_folder(0).mkdir()
                self._model_id.model_files_folder(0).mkdir()
            except Exception as e:
                get_application_logger().error(f"Failed to create empty backing store for "
                                               f"model iteration 0: {e}. Contact developer.")
                if isinstance(iter_dir, Path):
                    shutil.rmtree(iter_dir, ignore_errors=True)

        QTimer.singleShot(10, lambda: self.iter_loaded.emit())

    def num_markers(self, cam: RxCam, part: RxBodyPart) -> int:
        """
        Get the total marker count across all marked sessions in the training data for the current iteration of the
        model loaded, per camera source and body part type.
        :param cam: The camera source ID. Must be ``RxCam.SIDE/FRONT`` for a legacy-cam model, ``RxCam.LEFT/RIGHT`` for
            a fixed-cam model.
        :param part: The body part ID.
        :return: Total #frames from specified camera on which specified body part is marked, across all marked sessions
           in the current model iteration's training data. Returns 0 if ``cam`` or ``part`` invalid.
        :raises ValueError: If `cam` or `part` invalid.
        """
        if not self.model_loaded:
            return 0
        allowed_cams = [RxCam.LEFT, RxCam.RIGHT] if self.id.is_fixed_cam else [RxCam.SIDE, RxCam.FRONT]
        if not ((cam in allowed_cams) and (part in RxBodyPart.defined_body_parts(self.id.is_fixed_cam))):
            return 0
        count = 0
        for _, markers in self._marked_sessions.items():
            count += markers.num_total_markers(cam, part)
        return count

    @property
    def has_marked_sessions(self) -> bool:
        """ True if model loaded and current model iteration has at least one marked session in its training data. """
        return self.model_loaded and (len(self._marked_sessions) > 0)

    @property
    def marked_sessions(self) -> List[RxSessionID]:
        """
        List of experiment sessions augmented with body part markers on selected video frames -- constituting the
        training data for the current iteration of the model loaded.
        """
        return list(self._marked_sessions.keys())

    def num_markers_for_session(self, sesh_id: RxSessionID) -> int:
        """
        Total number of body part markers defined on specified marked session in the training data for the current
        model iteration.
        :param sesh_id: The session ID.
        :return: # of body part markers for specified session; 0 if session not marked.
        """
        return self._marked_sessions[sesh_id].num_markers() if (sesh_id in self._marked_sessions) else 0

    def marked_frames_for_session(self, sesh_id: RxSessionID) -> List[int]:
        """
        List of marked frames from the specified marked session in the training data for the current model iteration.
        :param sesh_id: The session ID.
        :return: List of frame numbers in chronological order; empty list if session not marked.
        """
        return self._marked_sessions[sesh_id].marked_frames() if (sesh_id in self._marked_sessions) else []

    def get_markers_for_frame(self, sesh_id: RxSessionID, frame: int) -> List[BodyPartMarker]:
        """
        List of all body part markers attached to a frame from the two primary cam videos recorded during
        a marked session in the training data for the current model iteration.
        :param sesh_id: The session ID.
        :param frame: A frame number.
        :return: List of all body part markers defined on the specified frame from the specified session. Will be empty
            if session is not marked, or if there are no markers defined on the specified frame.
        """

        return self._marked_sessions[sesh_id].get_markers_for_frame(frame) if (sesh_id in self._marked_sessions) else []

    def get_next_marked_frame_starting_at(self, sesh: RxSessionID, frame: int) -> Tuple[Optional[RxSessionID], int]:
        """
        Starting at the displayed frame for the current displayed session, get the next marked frame in the training
        data for the current iteration of the loaded ML model. Since the training data can include more than one marked
        session, the "next marked frame" is fully specified by the marked session ID and the frame number of the marked
        frame.

        This method is intended to provide a means by which the UI can "jump" from one marked frame to the next in the
        training data for the current model iteration. The training data are all body part markers attached to select
        frames across multiple experiment sessions.

        Algorithm:
         - If the current displayed session is NOT among the marked sessions in the training data, then the next marked
           frame is always the first marked frame in the first marked session in the training data.
         - If the specified session IS a marked session, the next marked frame is the first marked frame in that
           session AFTER the current displayed frame.
         - If there are no more marked frames after the frame specified, then the next marked frame is the first
           marked frame in the NEXT marked session. But if the specified session is the last marked session in the
           training data, then there is no next marked frame!

        :param sesh: This should be the identifier for the currently displayed session in ReachX.
        :param frame: This should be the frame number for the currently displayed frame.
        :return: (None, -1) if no model loaded, no marked sessions at all, or there is no next marked frame. Else
            returns (session_id, frame_name).
        """
        sesh_id_list = self.marked_sessions
        if len(sesh_id_list) == 0:
            return None, -1
        if sesh not in sesh_id_list:
            next_sesh = sesh_id_list[0]
            next_frame = self.marked_frames_for_session(next_sesh)[0]
        else:
            next_sesh = sesh
            idx = sesh_id_list.index(next_sesh)
            next_frame = next((fnum for fnum in self.marked_frames_for_session(next_sesh) if fnum > frame), -1)
            if next_frame == -1:
                if idx == len(sesh_id_list) - 1:
                    return None, -1
                else:
                    next_sesh = sesh_id_list[idx+1]
                    next_frame = self.marked_frames_for_session(next_sesh)[0]
        return next_sesh, next_frame

    def get_prev_marked_frame_starting_at(self, sesh: RxSessionID, frame: int) -> Tuple[Optional[RxSessionID], int]:
        """
        Starting at the displayed frame for the current displayed session, get the previous marked frame in the training
        data for the current iteration of the loaded ML model. Since the training data can include more than one marked
        session, the "previous marked frame" is fully specified by the marked session ID and the frame number of the
        marked frame.

        This method is intended to provide a means by which the UI can "jump" from one marked frame to the next in the
        training data for the current model iteration. The training data are all body part markers attached to select
        frames across multiple experiment sessions.

        Algorithm:
         - If the current displayed session is NOT among the marked sessions in the training data, then there is no
           previous marked frame, only a next one.
         - If the specified session IS a marked session, the previous marked frame is the marked frame in that session
           immediately BEFORE the current displayed frame.
         - If there are no marked frames before the frame specified, then the previous marked frame is the LAST
           marked frame in the previous marked session. But if the specified session is the first marked session in the
           training data, then there is no previous marked frame!


        :param sesh: This should be the identifier for the currently displayed session in ReachX.
        :param frame: This should be the frame number for the currently displayed frame.
        :return: (None, -1) if no model loaded, no marked sessions at all, or there is no previous marked frame. Else
            returns (session_id, frame_num).
        """
        sesh_id_list = self.marked_sessions
        if (len(sesh_id_list) == 0) or (sesh not in sesh_id_list):
            return None, -1
        else:
            prev_sesh = sesh
            idx = sesh_id_list.index(prev_sesh)
            rev_frames = self.marked_frames_for_session(prev_sesh)
            rev_frames.reverse()
            prev_frame = next((fnum for fnum in rev_frames if fnum < frame), -1)
            if prev_frame == -1:
                if idx == 0:
                    return None, -1
                else:
                    prev_sesh = sesh_id_list[idx - 1]
                    prev_frame = self.marked_frames_for_session(prev_sesh)[-1]
        return prev_sesh, prev_frame

    def attach_session_marker(self, sesh_id: RxSessionID, frame: int, cam: RxCam, part: RxBodyPart,
                              x_pix: int, y_pix: int) -> None:
        """
        Augment the annotated training data for the current model iteration by attaching a body part marker to the
        experiment session specified.

        If a change is made, the `markers_changed` signal is emitted. No action taken if no model is loaded.

        :param sesh_id: The session ID. Both session and model must be fixed-cam or legacy-cam. No mixing!
        :param frame: The frame number. ASSUMED to be a valid frame number for the session specified.
        :param cam: Identifies source camera. Must be ``RxCam.SIDE/FRONT`` for a legacy-cam session, or
            ``RxCam.LEFT/RIGHT`` for a fixed-cam session.
        :param part: Body part to be marked.
        :param x_pix: X-coordinate of marker location in the frame image (origin at TL corner, X increasing rightward).
        :param y_pix: Y-coordinate of marker location in the frame image (origin at TL corner, Y increasing downward).
        """
        if self.model_loaded:
            if sesh_id.is_fixed_cam != self.id.is_fixed_cam:
                get_application_logger().warning("Cannot use legacy-cam session for a fixed-cam model and vice-versa.")
                return
            valid_cams = [RxCam.LEFT, RxCam.RIGHT] if self.id.is_fixed_cam else [RxCam.SIDE, RxCam.FRONT]
            if cam not in valid_cams:
                get_application_logger().warning(f"Invalid cam for {'fixed-' if self.id.is_fixed_cam else 'legacy-'}cam"
                                                 f" session")
            if sesh_id not in self._marked_sessions:
                self._marked_sessions[sesh_id] = SessionMarkers()
            marker = BodyPartMarker(frame, cam, part, x_pix, y_pix)
            self._marked_sessions[sesh_id].add_mark(marker)
            self._save_timer_update(1)
            QTimer.singleShot(10, lambda sesh=sesh_id: self.markers_changed.emit(sesh))

    def detach_session_marker(self, sesh_id: RxSessionID, frame: int,
                              which: Optional[Tuple[RxCam, RxBodyPart]] = None) -> None:
        """
        Modify the annotated training data for the current model iteration by removing a particular body part marker
        attached to a frame of the session specified, or by discarding all markers attached to that camera frame.

        If a change is made, the `markers_changed` signal is emitted. No action taken if no model is loaded.

        :param sesh_id: The session ID.
        :param frame: The frame number. ASSUMED to be a valid frame number for the session specified.
        :param which: If None, all body part markers for the specified frame are deleted. Otherwise, this is a 2-tuple
            (cam, bodypart) identifying the marker by camera source and body part type.
        """
        if self.model_loaded:
            changed = False
            if sesh_id in self._marked_sessions:
                if which is None:
                    if self._marked_sessions[sesh_id].clear(frame):
                        changed = True
                        get_application_logger().info(f"Removed all markers on frame {frame} in session {str(sesh_id)}")
                elif isinstance(which, tuple) and len(which) == 2:
                    marker = BodyPartMarker(frame, which[0], which[1], 0, 0)
                    if self._marked_sessions[sesh_id].remove_mark(marker):
                        changed = True

            if changed:
                if self._marked_sessions[sesh_id].num_markers() == 0:
                    self._marked_sessions.pop(sesh_id)
                self._save_timer_update(1)
                QTimer.singleShot(10, lambda sesh=sesh_id: self.markers_changed.emit(sesh))

    def discard_all_markers(self, sesh_id: Optional[RxSessionID] = None) -> None:
        """
        Modify the annotated training data for the current model iteration by discarding all body part markers attached
        to the specified experiment session, or by discarding all marked sessions entirely.

        If a change is made, the `markers_changed` signal is emitted. No action taken if no model is loaded.

        :param sesh_id: The session ID. If None, all existing annotated training data is discarded!
        """
        if self.model_loaded:
            changed = False
            if sesh_id is None and (len(self._marked_sessions) > 0):
                self._clear_markers()
                changed = True
                get_application_logger().info(
                    f"{str(self._model_id)}, iter {self._curr_iter}: Discarded all marked sessions!")
            elif sesh_id in self._marked_sessions:
                markers = self._marked_sessions.pop(sesh_id)
                markers.clear()
                changed = True
                get_application_logger().info(f"{str(self._model_id)}, iter {self._curr_iter}: "
                                              f"Discarded all markers attached to session {str(sesh_id)}")
            if changed:
                self._save_timer_update(1)
                QTimer.singleShot(10, lambda sesh=sesh_id: self.markers_changed.emit(sesh))

    def _save_timer_update(self, which: int) -> None:
        """
        Periodic save timer timeout handler and configuration. Usage:
         - Training data is changed: Call method with ``which==1``. The save timer is started unless already running.
         - Training data just saved: Call method with ``which==-1``. The save timer is stopped.
         - Save timer timeout: Call method with ``which`` set to any other value. The training data is saved.
        :param which: Arg determines action taken, as described.
        """
        if which == 1:
            if not self._save_timer.isActive():
                self._save_timer.start()
        elif which == -1:
            self._save_timer.stop()
        else:
            self._save_markers()

    def load(self, model_id: RxModelID, session_root: Path) -> Optional[str]:
        """
        Unload the current body part detection ML model, then load the specified model. The most recent model iteration
        found is loaded initially.

        :param model_id: Identifies the ML model to load.
        :param session_root: The workspace's session root folder. All experiment sessions used to train a model
            iteration or that are analyzed by a model iteration are stored here.
        :return: None if load was successful (or specified model is already loaded!); else a description of the first
            error encountered. In the event the load fails, the model manager remains in the "no model loaded" state.
        """
        if model_id == self._model_id:
            return None

        self.unload()

        if not model_id.exists():
            return f"Model project folder not found: {str(model_id.location.absolute())}"
        if not session_root.is_dir():
            return f"Session root folder not found: {str(session_root.absolute())}"

        iters = model_id.find_all_iterations()
        if len(iters) == 0:
            return f"Invalid model project folder -- no model iterations found."

        self._model_id = model_id
        self._session_root = session_root
        self._existing_iterations = iters
        self._curr_iter = self._existing_iterations[-1]

        self._load_markers()
        return None

    def unload(self) -> None:
        """
        Persist any changes to the definition of the current iteration of the loaded model, then unload the
        model.
        """
        if self.model_loaded:
            self._save_markers()

            self._clear_markers()
            self._model_id = None
            self._session_root = None
            self._curr_iter = None
            self._existing_iterations.clear()

    def _clear_markers(self) -> None:
        for _, markers in self._marked_sessions.items():
            markers.clear()
        self._marked_sessions.clear()

    def _load_markers(self) -> None:
        """
        Load the body part markers for the current model iteration from the iteration's backing store.

        All defined body part markers (on the two primary cams) for a marked session with the unique session
        prefix ``SESH_PFX`` are stored in a single file: ``<MODEL>/iter-N/annotations/<SESH_PFX>markers.txt``, where
        ``<MODEL>`` is the absolute path to the model project folder and ``N`` is the iteration number. This method
        loads all such marker files found in the ``annotations`` directory.

        If SESH_PFX identifies a session not found in the active workspace, a warning is posted to the application log,
        **but the markers file corresponding to that session is NOT removed**. This could happen if the
        user changed the session root for a workspace, or used the same model root but different session roots for two
        workspaces. In this scenario, model training will fail, but no previously accumulated training data (which is
        a tedious undertaking) is destroyed.

        If an error occurs while reading any markers file, the error is posted to the application log. Badly formatted
        markers files are removed, as is any empty markers file.
        """
        if not self.model_loaded:
            return

        marked_session_files = self._model_id.get_all_existing_marker_files(self._curr_iter)
        for f in marked_session_files:
            sesh_id = RxModelID.session_id_from_marker_filename(self._session_root, f)
            if isinstance(sesh_id, RxSessionID):
                err_msg, markers = SessionMarkers.load(f)

                if (len(err_msg) > 0) or (markers.num_markers() == 0) or not sesh_id.exists():
                    remove = True
                    if len(err_msg) > 0:
                        msg = f"{err_msg} - Invalid markers file removed."
                    elif not sesh_id.exists():
                        msg = (f"Found markers for a missing session: {str(sesh_id)}. Markers file kept, but model "
                               f"training will fail in this state!")
                        remove = False
                    else:
                        msg = f"Removed empty marker set for session: {str(sesh_id)}"
                    get_application_logger().warning(msg)
                    if remove:
                        f.unlink(missing_ok=True)
                else:
                    self._marked_sessions[sesh_id] = markers
            else:
                get_application_logger().info(f"Skipped file {str(f.absolute())} - does not correspond to a "
                                              f"valid session.")

    def _save_markers(self) -> None:
        """
        Save all body part markers for the current model iteration to the iteration's backing store.

        See _load_markers() for details. All existing marker files are overwritten, and any stale marker files
        (corresponding to previously marked sessions that were removed) are deleted. Any errors encountered are posted
        to the application log.
        """
        if not self.model_loaded:
            return

        # phase 1: remove existing markers files for any marked sessions that have been removed from the training data
        marked_session_files = self._model_id.get_all_existing_marker_files(self._curr_iter)
        for f in marked_session_files:
            sesh_id = RxModelID.session_id_from_marker_filename(self._session_root, f)
            if sesh_id not in self._marked_sessions:
                get_application_logger().info(
                    f"Previously marked session removed from training data - removing {str(f.absolute())}")
                f.unlink(missing_ok=True)

        # phase 2: save all markers, overwriting existing markers files
        for sesh_id, markers in self._marked_sessions.items():
            f = self._model_id.markers_file_for_session(self._curr_iter, sesh_id)
            if (markers.num_markers() == 0) and f.is_file():
                get_application_logger().info(f"Empty marker set - removing {str(f.absolute())}")
                f.unlink(missing_ok=True)
                continue
            err_msg = SessionMarkers.save(f, markers)
            if len(err_msg) > 0:
                get_application_logger().error(err_msg)
            else:
                get_application_logger().info(f"Saved markers for session {str(sesh_id)} to {str(f.name)}")

        # stop periodic save timer. It won't be restarted until the next change in training data
        self._save_timer_update(-1)

    def can_train(self) -> bool:
        """
        Is there sufficient annotated training data to initiate training of the active iteration of the currently
        loaded ML model?
        :return True if there are enough hand- and pellet-related body part markers defined; else False.
        """
        n_r_hand, n_l_hand, n_pellet = 0, 0, 0
        for _, markers in self._marked_sessions.items():
            rh, lh, p = markers.num_total_hand_and_pellet_markers()
            n_r_hand += rh
            n_l_hand += lh
            n_pellet += p
        ok = (n_r_hand >= MIN_RIGHT_HAND_MARKERS) and (n_pellet >= MIN_PELLETS)
        return (ok and (n_l_hand >= MIN_LEFT_HAND_MARKERS)) if self.is_fixed_cam else ok

    def warn_if_too_few_reference_markers_for_fixed_cam_model(self) -> None:
        """
        Fixed-cam rigs employ some new special-purpose markers not available in the older legacy-cam rigs:
         - `RxBodyPart.STAR` - A star symbol permanently affixed to the pellet cover arm.
         - `RxBodyPart.TRIANGLE` - Triangle-shaped symbol affixed to the pellet holder arm.
         - `RxBodyPart.TONGUE_MID` - Midpoint of animal's tongue.
         - `RxBodyPart.DIAMOND` - A diamond-shaped symbol that is in a fixed location visible on both left and right
           cameras.
        The first 3 parts are used to improve reach segmentation performance for a fixed-cam session, while the last
        one serves as the preferred "origin" of the 3D coordinate space to which all computed body part trajectories
        are transformed during analysis. None are absolutely essential, but this method posts a warning message to
        the application log if there is insufficient numbers of these markers in the training data for the current
        model iteration.

        No action taken for a legacy-cam session.
        """
        if self.is_fixed_cam:
            n_star, n_triangle, n_tongmid, n_diamond = 0, 0, 0, 0
            for _, markers in self._marked_sessions.items():
                counts = markers.marker_counts([RxBodyPart.STAR, RxBodyPart.TRIANGLE, RxBodyPart.TONGUE_MID,
                                                RxBodyPart.DIAMOND])
                n_star += counts[RxBodyPart.STAR]
                n_triangle += counts[RxBodyPart.TRIANGLE]
                n_tongmid += counts[RxBodyPart.TONGUE_MID]
                n_diamond += counts[RxBodyPart.DIAMOND]
            if any([n < MIN_SPECIAL_MARKERS for n in [n_star, n_triangle, n_tongmid, n_diamond]]):
                get_application_logger().warning(
                    f"At least {MIN_SPECIAL_MARKERS} instances of these body parts should be annotated when training a "
                    f"fixed-cma model: '{str(RxBodyPart.STAR)}', '{str(RxBodyPart.TRIANGLE)}', "
                    f"'{str(RxBodyPart.TONGUE_MID)}' and '{str(RxBodyPart.DIAMOND)}'.")

    def is_trained(self) -> bool:
        """
        Has the active iteration of the currently loaded ML model been trained previously?
        :return: True if model iteration was previously trained; False otherwise.
        """
        return self.model_loaded and self._model_id.has_been_trained(self._curr_iter)

    def get_all_markers(self) -> Dict[RxSessionID, List[BodyPartMarker]]:
        """
        Get all body part markers defined for the active iteation of the current ML model.
        :return: A dictionary holding the full list of defined body part markers for each marked session in the
            active model iteration's training data, keyed by session ID.
        """
        out: Dict[RxSessionID, List[BodyPartMarker]] = dict()
        for sesh_id in self._marked_sessions:
            markers: List[BodyPartMarker] = list()
            for f_num in self.marked_frames_for_session(sesh_id):
                markers.extend(self.get_markers_for_frame(sesh_id, f_num))
            out[sesh_id] = markers
        return out
