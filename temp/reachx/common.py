"""
common.py: ReachX constants and various utility classes and functions.

NOTE: By design, this module can be imported standalone into user scripts. Its only dependencies are on Numpy and the
Python standard library. Certain UI-related classes formerly defined here were moved to ``uicommon.py``.
"""

from __future__ import annotations

import bisect
import re
import warnings
from datetime import datetime
from enum import IntEnum


try:
    from enum import StrEnum
except ImportError:
    # If StrEnum is not available (e.g., Python 3.10), use the backport
    # noinspection PyPackageRequirements
    from backports.strenum import StrEnum
from pathlib import Path
from typing import List, Optional, NamedTuple, Tuple, Dict

import numpy as np


class RX:
    """
    Constants and static utility methods for ReachX.

    NOT an exhaustive collection -- only those that are used across multiple modules/packages in the  application.

    Usage:
     - import constants
     - rx = RX()
     - rx.RAW_VIDEO_ROOT
    """
    __slots__ = ()

    APP_NAME: str = 'ReachX'
    """ The application name. """
    APP_VERSION: str = 'v0.7.1'
    """ The application version string."""
    HOME: Path = Path(Path.home(), '.' + APP_NAME)
    """ Per-user home directory for ReachX. """

    MAX_LEN_WS_NAME: int = 25
    """ Maximum number of characters in a workspace name. """
    SUPPORTED_FRAME_RATES: List[int] = [150, 900]
    """ Supported nominal camera frame rates in Hz. """
    MIN_REACHSEG_DUR: int = 5
    """ The minimum allowed duration for a curated reach segment (# video frames). **No maximum enforced.** """

    @staticmethod
    def camera_frame_dims(is_fixed_cam: bool) -> Tuple[int, int]:
        """
        Return the camera frame dimensions for the fixed-cam or the legacy-cam experiment setup.
        :param is_fixed_cam: True for fixed-cam frame dimension, False for legacy-cam frame dimensions.
        :return: A 2-tuple (W, H) specifying camera frame width and height in pixels.
        """
        return (256, 256) if is_fixed_cam else (300, 200)

    @staticmethod
    def format_elapsed_time(t_secs: float, in_hours: bool = True) -> str:
        """
        Convert elapsed time to a specified display format in hours/min/sec or min/sec/millisecs.
        :param t_secs: The elapsed time in seconds.
        :param in_hours: If True, return time as ``HHH:MM:SS``; else as ``MMM:SS:mmm``. Default = True.
        :return: The formatted time. Will be '00:00:00' if ``t_secs <= 0``.
        """
        if t_secs <= 0:
            return "00:00:00"
        elif in_hours:
            m, s = divmod(t_secs, 60)
            h, m = divmod(m, 60)
            return f"{int(h):03d}:{int(m):02d}:{int(s):02d}"
        else:
            minutes = int(t_secs / 60)
            seconds = int(t_secs - 60 * int(t_secs / 60))
            msecs = int(1000 * (t_secs - int(t_secs)))
            return f"{minutes:03d}:{seconds:02d}.{msecs:03d}"


class RxCam(IntEnum):
    """
    Enumeration of the different video cams defined in ReachX.

    **DEVNOTES**:
     - For legacy-cam rigs, the SIDE and FRONT cameras are most important and should be universally present. The
       The ``STIM`` and ``FAST`` cams may or may not be present, and are only used while recording during a ReachX
       experiment on a legacy-cam setup.
     - Using an IntEnum for convenient and compact persistence: Body part markers are fully identified by a sequence of
       5 integers: frame#, cam, bodypart, x, y (the last two being the marker location on the frame image, in pixels.
     - __str__() override and from_nickname() added so that we can easily convert between existing camera nicknames
       and their corresponding RxCam enumerants.
     - 10mar2026: Introduced enumerants unique to the new fixed-cam experiment setup: ``LEFT``, ``RIGHT``. None of the
       other camera types apply to fixed-cam setups.
    """
    SIDE = 0   # starting at 0 for convenient indexing into _nicknames()
    """ (Legacy-cam setup only) Cam facing right side of subject. """
    FRONT = 1
    """ (Legacy-cam setup only) Cam facing front of subject. """
    STIM = 2
    """ (Legacy-cam setup only) Special-purpose legacy cam, not used for reach trajectory analysis. """
    FAST = 3
    """ (Legacy-cam setup only) Special-purpose legacy cam, not used for reach trajectory analysis. """
    LEFT = 4
    """ (Fixed-cam setup only) Camera recording left-hand view of fixed-cam setup. """
    RIGHT = 5
    """ (Fixed-cam setup only) Camera recording right-hand view of fixed-cam setup. """

    @property
    def is_fixed_cam(self) -> bool:
        """ ``True`` for a camera source in a fixed-cam rig, ``False`` otherwise (legacy-cam). """
        return self.value in [RxCam.LEFT, RxCam.RIGHT]

    @staticmethod
    def _nicknames() -> List[str]:
        """ List of all rig camera nicknames, in enumerant order. Not for external use. """
        return ['sideCam', 'frontCam', 'stimCam', 'fastCam', 'left', 'right']

    @staticmethod
    def from_nickname(nickname: str) -> Optional[RxCam]:
        """
        Convert specified ReachX camera nickname to the corresponding `RxCam` enumerant.
        :param nickname: Camera nickname.
        :return: Corresponding enumerant, or None if nickname not recognized.
        """
        try:
            idx = RxCam._nicknames().index(nickname)
            return RxCam(idx)
        except ValueError:
            return None

    @staticmethod
    def available_cams(is_fixed_cam: bool) -> List[RxCam]:
        """
        Cameras present in the fixed-cam vs legacy-cam experiment rigs.
        :param is_fixed_cam: ``True`` (``False``) if experiment rig employs the fixed-cam (legacy-cam) setup.
        """
        return [RxCam.LEFT, RxCam.RIGHT] if is_fixed_cam else [RxCam.SIDE, RxCam.FRONT, RxCam.STIM, RxCam.FAST]

    @staticmethod
    def required_cams(is_fixed_cam: bool) -> List[RxCam]:
        """
        Cameras that MUST be present in the fixed-cam vs legacy-cam experiment rigs.
        :param is_fixed_cam: ``True`` (``False``) if experiment rig employs the fixed-cam (legacy-cam) setup.
        :return: A 2-element list containing the identifiers of the two required cameras.
        """
        return [RxCam.LEFT, RxCam.RIGHT] if is_fixed_cam else [RxCam.SIDE, RxCam.FRONT]

    def __str__(self) -> str:
        """ Overridden to return associated camera nickname. """
        return RxCam._nicknames()[self.value]


class RxBodyPart(IntEnum):
    """
    Enumeration of all defined body parts and other features that may be marked on a video frame for the purposes
    of training a ML model that analyzes recorded session videos.

    **Composite markers** ``RxBodyPart.R_HAND`` and ``RxBodyPart.L_HAND``:

    These are NOT used for marking hand locations for model training. Instead, they refer to the location of the
    animal's right hand and left hand, respectively, when processing the model outputs for a session.
     - For fixed-cam sessions, both hands are tracked. The per-frame location of ``R_HAND`` is the location of the
       ``RIGHT_HAND_*`` body part with the highest confidence score for that frame, and analogously for ``L_HAND``.
       The ``RIGHT_HAND_*`` and ``LEFT_HAND_*`` body parts are tracked on BOTH ``RxCam.LEFT`` and ``.RIGHT``.
     - For a legacy-cam session, only the animal's right-hand is tracked. ``R_HAND`` is the location of the
       ``SIDE_HAND_*`` body part with the highest confidence score on an ``RxCam.SIDE`` frame, and the location of the
       ``FRONT_HAND_*`` body part with the highest confidence score on an ``RxCam.FRONT`` frame.

    **DEVNOTES**:
     - Using an IntEnum for convenient and compact persistence: Body part markers are fully identified by a sequence of
       5 integers: frame#, cam, bodypart, x, y (the last two being the marker location on the frame image, in pixels.
     - __str__() override and from_nickname() added so that we can easily convert between existing body part nicknames
       and their corresponding RxBodyPart enumerants.
     - 24oct2025: Added two generic "exclude" markers that a researcher can use to mark features in session videos that
       the ML model might confuse with an important feature like the food pellet.
     - 14nov2025: Added the "nose" marker.
     - 10mar2026: Added markers unique to fixed-cam setup: ``RIGHT_HAND_FLAT`` to ``DIAMOND``.
     - 25mar2026: Added composite markers ``R_HAND`` and ``L_HAND``.

    """
    TONGUE = 0    # starting at 0 for convenient indexing into _nicknames()
    MOUTH = 1
    PELLET = 2
    SIDE_HAND_FLAT = 3
    """ Lateral view of hand, flattened. """
    SIDE_HAND_SPREAD = 4
    """ Lateral view of hand, spread out. """
    SIDE_HAND_GRAB = 5
    """ Lateral view of hand, grabbing pellet (or air). """
    FRONT_HAND_REACH = 6
    """ Frontal view of hand, reaching out toward camera. """
    FRONT_HAND_GRASP = 7
    """ Frontal view of hand, grabbing pellet (or something else). """
    EXCLUDE1 = 8
    """ Generic marker labeling a feature that might be confused with a desired feature. """
    EXCLUDE2 = 9
    """ Generic marker labeling a feature that might be confused with a desired feature. """
    NOSE = 10
    """ Subject's nose. """
    RIGHT_HAND_FLAT = 11
    """ (Fixed-cam only) Subject's right hand, flattened. """
    RIGHT_HAND_SPREAD = 12
    """ (Fixed-cam only) Subject's right hand, spread out. """
    RIGHT_HAND_GRAB = 13
    """ (Fixed-cam only) Subject's right hand, grabbing pellet (or air). """
    LEFT_HAND_FLAT = 14
    """ (Fixed-cam only) Subject's left hand, flattened. """
    LEFT_HAND_SPREAD = 15
    """ (Fixed-cam only) Subject's left hand, spread out. """
    LEFT_HAND_GRAB = 16
    """ (Fixed-cam only) Subject's left hand, grabbing pellet (or air). """
    STAR = 17
    """ (Fixed-cam only) Special-purpose marker. """
    TONGUE_MID = 18
    """ (Fixed-cam only) Midpoint of subject's tongue. """
    TONGUE_TIP = 19
    """ (Fixed-cam only) Tip of subject's tongue. """
    TRIANGLE = 20
    """ (Fixed-cam only) Special-purpose marker. """
    DIAMOND = 21
    """ (Fixed-cam only) Special-purpose marker."""
    R_HAND = 22
    """ Composite marker representing the location of animal's right-hand in any given frame. NOT used for training. """
    L_HAND = 23
    """ Composite marker representing the location of animal's left-hand in any given frame. NOT used for training. """

    @staticmethod
    def _nicknames() -> List[str]:
        """ List of all body part nicknames, in enumerant order. Not for external use. """
        return ['tongue', 'mouth', 'pellet', 'SdH_Flat', 'SdH_Spread', 'SdH_Grab', 'FtH_Reach', 'FtH_Grasp',
                'exclude1', 'exclude2', 'nose', 'RH_flat', 'RH_spread', 'RH_grab', 'LH_flat', 'LH_spread', 'LH_grab',
                'Star', 'Tongue_mid', 'Tongue_tip', 'Triangle', 'Diamond', 'R_Hand', 'L_Hand']

    @staticmethod
    def from_nickname(nickname: str) -> Optional[RxBodyPart]:
        """
        Convert specified ReachX body part nickname to the corresponding `RxBodyPart` enumerant.
        :param nickname: Body part nickname.
        :return: Corresponding enumerant, or None if nickname not recognized.
        """
        try:
            idx = RxBodyPart._nicknames().index(nickname)
            return RxBodyPart(idx)
        except ValueError:
            return None

    def is_hand(self) -> bool:
        """
        Returns ``True`` if this body part is a hand "pose". False otherwise. Returns ``False`` for the composite
        markers ``R_HAND`` and ``L_HAND``, which are not for use in marking hand poses.
        """
        return self in {
            RxBodyPart.SIDE_HAND_FLAT,
            RxBodyPart.SIDE_HAND_SPREAD,
            RxBodyPart.SIDE_HAND_GRAB,
            RxBodyPart.FRONT_HAND_REACH,
            RxBodyPart.FRONT_HAND_GRASP,
            RxBodyPart.RIGHT_HAND_FLAT,
            RxBodyPart.RIGHT_HAND_SPREAD,
            RxBodyPart.RIGHT_HAND_GRAB,
            RxBodyPart.LEFT_HAND_FLAT,
            RxBodyPart.LEFT_HAND_SPREAD,
            RxBodyPart.LEFT_HAND_GRAB,
        }

    def is_right_hand(self) -> bool:
        """
        Returns ``True`` if this body part is a right-hand "pose". False otherwise. Returns False for the composite
        marker R_HAND, which is not for use in marking hand poses.
        """
        return self.is_hand() and not self.is_left_hand()

    def is_left_hand(self) -> bool:
        """
        Returns ``True`` if this body part is a left-hand "pose". False otherwise. Returns False for the composite
        marker L_HAND, which is not for use in marking hand poses. Also note that no left-hand poses are defined for
        use with a legacy-cam session.
        """
        return self in {RxBodyPart.LEFT_HAND_FLAT, RxBodyPart.LEFT_HAND_SPREAD, RxBodyPart.LEFT_HAND_GRAB}

    @staticmethod
    def hand_body_parts_on(cam: RxCam) -> List[RxBodyPart]:
        """
        The list of alternative hand pose markers for the specified cam. The composite markers ``R_HAND`` and ``L_HAND``
        are excluded; they are not used to mark hand poses.
        :param cam: The camera identifier.
        :return: List of valid hand markers for specified cam. Will be empty list for ``RxCam.STIM``, `RxCam.FAST`.
        """
        if cam == RxCam.SIDE:
            return [RxBodyPart.SIDE_HAND_FLAT, RxBodyPart.SIDE_HAND_SPREAD, RxBodyPart.SIDE_HAND_GRAB]
        elif cam == RxCam.FRONT:
            return [RxBodyPart.FRONT_HAND_REACH, RxBodyPart.FRONT_HAND_GRASP]
        elif cam.is_fixed_cam:
            return [RxBodyPart.RIGHT_HAND_FLAT, RxBodyPart.RIGHT_HAND_SPREAD, RxBodyPart.RIGHT_HAND_GRAB,
                    RxBodyPart.LEFT_HAND_FLAT, RxBodyPart.LEFT_HAND_SPREAD, RxBodyPart.LEFT_HAND_GRAB]
        else:
            return []

    @staticmethod
    def defined_body_parts(is_fixed_cam: bool) -> List[RxBodyPart]:
        """
        The list of all defined body part markers for a legacy-cam or a fixed-cam session. Note that the composite
        markers ``R_HAND`` and ``L_HAND`` are excluded; they are not used to mark hand poses for model training.

        :param is_fixed_cam: ``True`` (``False``) for the defined body parts for a fixed-cam (legacy-cam) session.
        :return: The list of body part markers applicable to a fixed-cam or legacy-cam session.
        """
        if not is_fixed_cam:
            return [p for p in RxBodyPart if p <= RxBodyPart.NOSE]
        else:
            out = [RxBodyPart.MOUTH, RxBodyPart.PELLET]
            out.extend([p for p in RxBodyPart if RxBodyPart.EXCLUDE1 <= p <= RxBodyPart.DIAMOND])
            return out

    @staticmethod
    def fixed_cam_left_hand_poses() -> List[RxBodyPart]:
        """ Subset of body parts representing different poses of animal's left hand in a fixed-cam session. """
        return [RxBodyPart.LEFT_HAND_FLAT, RxBodyPart.LEFT_HAND_SPREAD, RxBodyPart.LEFT_HAND_GRAB]

    @staticmethod
    def fixed_cam_right_hand_poses() -> List[RxBodyPart]:
        """ Subset of body parts representing different poses of animal's right hand in a fixed-cam session. """
        return [RxBodyPart.RIGHT_HAND_FLAT, RxBodyPart.RIGHT_HAND_SPREAD, RxBodyPart.RIGHT_HAND_GRAB]

    @staticmethod
    def markable_body_parts_on(cam: RxCam) -> List[RxBodyPart]:
        """
        List of all body parts that may be marked on frame images from the specified camera source. These are the
        body part markers used to train a model; the composite markers ``R_HAND`` and ``L_HAND`` are excluded.
        :param cam: Camera source ID.
        :return: List of body parts that the user may mark on a frame image from that camera.
        """
        all_parts = RxBodyPart.defined_body_parts(cam.is_fixed_cam)
        relevant_hands = RxBodyPart.hand_body_parts_on(cam)
        out: List[RxBodyPart] = list()
        for bp in all_parts:
            if (not bp.is_hand()) or (bp in relevant_hands):
                out.append(bp)
        return out

    def __str__(self) -> str:
        """ Overridden to return associated body part nickname. """
        return RxBodyPart._nicknames()[self.value]

    @staticmethod
    def marker_for(bp: RxBodyPart) -> Tuple[str, str]:
        """
        Get the shape and color of the marker that represents that body part in ReachX.

        :param bp: A body part.
        :return: A 2-tuple (shape, color) with the shape and color strings. These values are accepted as the
            ``symbol`` and ``brush`` attributes of a PyQtGraph ``TargetItem``.
        """
        return _BODYPART2MARKER[bp]


_BODYPART2MARKER: Dict[RxBodyPart, Tuple[str, str]] = {
        RxBodyPart.TONGUE: ('t', 'tomato'),
        RxBodyPart.MOUTH: ('s', 'red'),
        RxBodyPart.PELLET: ('h', 'navy'),
        RxBodyPart.SIDE_HAND_FLAT: ('o', 'blue'),
        RxBodyPart.SIDE_HAND_SPREAD: ('o', 'cornflowerblue'),
        RxBodyPart.SIDE_HAND_GRAB: ('o', 'deepskyblue'),
        RxBodyPart.FRONT_HAND_REACH: ('o', 'limegreen'),
        RxBodyPart.FRONT_HAND_GRASP: ('o', 'greenyellow'),
        RxBodyPart.EXCLUDE1: ('p', 'slategray'),
        RxBodyPart.EXCLUDE2: ('p', 'silver'),
        RxBodyPart.NOSE: ('d', 'orange'),
        RxBodyPart.RIGHT_HAND_FLAT: ('t2', 'blue'),
        RxBodyPart.RIGHT_HAND_SPREAD: ('t2', 'cornflowerblue'),
        RxBodyPart.RIGHT_HAND_GRAB: ('t2', 'deepskyblue'),
        RxBodyPart.LEFT_HAND_FLAT: ('t3', 'green'),
        RxBodyPart.LEFT_HAND_SPREAD: ('t3', 'limegreen'),
        RxBodyPart.LEFT_HAND_GRAB: ('t3', 'greenyellow'),
        RxBodyPart.STAR: ('star', 'white'),
        RxBodyPart.TONGUE_MID: ('t', 'tomato'),
        RxBodyPart.TONGUE_TIP: ('t1', 'tomato'),
        RxBodyPart.TRIANGLE: ('t', 'white'),
        RxBodyPart.DIAMOND: ('d', 'white'),
        RxBodyPart.R_HAND: ('o', 'red'),      # NOT USED for body part marking
        RxBodyPart.L_HAND: ('o', 'blue')       # NOT USED for body part marking
    }
""" 
Maps each body part to the (shape, color) of the marker that represents that body part in ReachX. The shape and 
color strings must be values accepted as the 'symbol' and 'brush' attributes of a PyQtGraph ``TargetItem``.
"""


class RxEvent(StrEnum):
    """
    Enumeration of defined event types in a ReachX behavioral experiment session.
    DEVNOTE: For now this only includes the events generated during a recording, not those added during analysis.
    """
    T6000 = 'T6000_played'
    """ Tone played to indicate pellet is ready to be delivered. """
    DETECTED = 'pellet_detected'
    """ Pellet detected after deliver arm has swung into position. """
    DELIVERY = 'pellet_delivery'
    """ Pellet delivered -- ie, cover displaced to reveal pellet to animal. """
    T5000 = 'T5000_played'
    """ 'Ready' tone played (usually occurs very shortly after pellet delivery. """
    REACHED = 'reach_detected'
    """ A reachx motion detected by acquisition system. """


class RxSessionID:
    """
    Immutable container for the elements that identify one experiment session within a workspace. This implements
    the ReachX convention for the session folder tree: ``<root>/<date>/<rig>/session<num>``.

    **Support for the new "fixed-cam" experiment setup**: ReachX does not allow mixing of experiment sessions run on
    the newer fixed-cam setup and those run on the legacy-cam setup. ``RxSessionID`` includes a method for identifying a
    valid session as fixed-cam or legacy-cam, based on the convention that the only camera videos found in a fixed-cam
    session folder will start with ``<session_file_prefix>-cam``, where ``cam == "left" or "right"``.
    """
    def __init__(self, root: Path, rig: str, date: str, num: str, is_fixed_cam: bool):
        """
        Construct a session path container: **<root>/<date>/<rig>/session<num>**.
        :param root: The workspace's session root.
        :param rig: The rig name.
        :param date: The session date string in 'YYYYMMDD' format.
        :param num: A 3-digit numeric string indicating the seesion number, eg: '013'.
        :param is_fixed_cam: A boolean indicating whether this is a fixed-cam or legacy-cam session. See
            ``check_for_fixed_cam_session_folder()``.
        """
        assert validate_date_string(date) and RxSessionID.is_valid_session_num(num)
        self._root = root
        self._rig = rig
        self._date = date
        self._num = num
        self._loc: Path = Path(self._root, self.date, self._rig, f"session{self._num}")
        self._is_fixed_cam = is_fixed_cam

    @property
    def root(self) -> Path:
        """ The root path for session recordings. """
        return self._root

    @property
    def rig(self) -> str:
        """ Name of rig on which this sesson was recorded. """
        return self._rig

    @property
    def date(self) -> str:
        """ The session date in the format 'YYYYMMDD'. """
        return self._date

    @property
    def num(self) -> str:
        """ The session number as a 3-digit string (0-padded if necessary). """
        return self._num

    @property
    def location(self) -> Path:
        """ File system location for the session folder. """
        return self._loc

    @property
    def is_fixed_cam(self) -> bool:
        """ True if session was recorded on a modern fixed-cam rig, False if recorded on a legacy-cam rig. """
        return self._is_fixed_cam

    @property
    def session_file_prefix(self) -> str:
        """
        By convention, data and other files for a session start with a prefix that includes the session date,
        rig nickname, and session number -- eg: '20250715_rigA_session001_'.
        """
        return f"{self._date}_{self._rig}_session{self._num}_"

    def exists(self) -> bool:
        """ Does the specified session folder exist in the file system? """
        return self._loc.is_dir()

    def __eq__(self, other) -> bool:
        if isinstance(other, RxSessionID):
            return ((self._root == other._root) and (self._rig == other._rig) and (self._date == other._date) and
                    (self._num == other._num))
        return False

    def __hash__(self) -> int:
        return hash((str(self._root.absolute()), self._rig, self._date, self._num))

    def __str__(self) -> str:
        return f"{datetime.strptime(self._date, '%Y%m%d').strftime('%d%b%Y')} #{self._num}, on {self._rig}"

    @staticmethod
    def from_file_prefix(root: Path, prefix: str) -> Optional[RxSessionID]:
        """
        (For settings file support) Construct a session ID for an experiment session under the specified session root
        having the specified session file prefix (formatted as 'YYYYMMDD_<rig_name>_sessionNNN').
        :param root: The session folder path.
        :param prefix: The file prefix in the form returned by RxSessionID.session_file_prefix.
        :return: The session ID, or None if `prefix` is ill-formed.
        """
        if isinstance(root, Path) and isinstance(prefix, str):
            tokens = prefix.split('_')
            if len(tokens) >= 3:
                date_str, rig, num = tokens[0], tokens[1], session_number_from_folder(tokens[2])
                if validate_date_string(date_str) and (len(tokens[1]) > 0) and (num is not None):
                    sid = RxSessionID(root, rig, date_str, num, False)
                    try:
                        is_fixed = RxSessionID.check_for_fixed_cam_session_folder(sid.location)
                        return RxSessionID(root, rig, date_str, num, True) if is_fixed else sid
                    except ValueError:
                        return None
        return None

    @staticmethod
    def from_session_folder(folder: Path) -> Optional[RxSessionID]:
        """
        Construct a session ID from the session folder's full file system path.
        :param folder: The session folder path.
        :return The session ID, or None if specified folder is not consistent with how ReachX experiment sessions
            are stored in the file system.
        """
        session_path = folder.resolve()
        num = session_path.name.replace("session", "")
        rig = session_path.parent.name
        date = session_path.parent.parent.name
        root = session_path.parent.parent.parent
        try:
            is_fixed = RxSessionID.check_for_fixed_cam_session_folder(session_path)
        except ValueError:
            return None
        if not (RxSessionID.is_valid_session_num(num) and validate_date_string(date)):
            return None
        return RxSessionID(root=root, rig=rig, date=date, num=num, is_fixed_cam=is_fixed)

    @staticmethod
    def is_valid_session_num(num_str: str) -> bool:
        """
        The session number in string form must be a zero-padded 3-digit integer: '003', '102', etc.
        :param num_str: The string to test.
        :return: ``True`` iff the string is a zero-padded 3-digit integer.
        """
        return re.fullmatch('[0-9]{3}', num_str) is not None

    @staticmethod
    def check_for_fixed_cam_session_folder(sesh_folder: Path) -> bool:
        """
        Examine the camera video files (MP4) in the session folder specified. If the folder contains files for the
        RxCam.LEFT and RxCam.RIGHT video cameras, then it is fixed-cam session, else it is a legacy-cam session.
        For performance reasons, method returns as soon as a fixed-cam or legacy-cam video is found. It does NOT
        check, eg, for a session folder missing one of the required cams, or a folder containing both fixed-cam and
        legacy-cam sessions.

        :param sesh_folder: Session folder path.
        :return: True if contents are consistent with a fixed-cam session, else False.
        :raise ValueError: If sesh_folder is not a valid directory or contains no MP4 files, or contains no MP4 files
            for the required cams in either the fixed-cam or legacy-cam setup.
        """
        if not sesh_folder.is_dir():
            raise ValueError("Session folder is not a valid directory!")
        mp4_files = list(sesh_folder.glob('*.mp4'))
        if len(mp4_files) == 0:
            raise ValueError("Session folder contains no MP4 videos!")

        for f in mp4_files:
            if f.name.find(str(RxCam.LEFT)) >= 0 or f.name.find(str(RxCam.RIGHT)) >= 0:
                return True
            if f.name.find(str(RxCam.SIDE)) >= 0 or f.name.find(str(RxCam.FRONT)) >= 0:
                return False

        raise ValueError("Session folder lacked any of the required camera videos!")

    @staticmethod
    def calibration_folder_for(sesh_id: RxSessionID, calib_root: Path) -> Optional[Path]:
        """
        Find the folder containing FLIR camera calibration data for a **fixed-cam** ReachX experiment session.

        A valid calibration folder for the session has path ``<calib_root>/<rig_name>/<date>/<S>mm_<R>_r_<C>c``, where:
         - ``<rig_name>`` is the name of the rig on which the session was recorded.
         - ``<date>`` is the calibration date in "YYYYMMDD" format.
         - ``<S>`` is the size of each square in the calibration checkerboard grid, in millimeters (integer string).
         - ``<R>`` is the number of rows in the calibration grid (integer string).
         - ``<C>`` is the number of columns in the calibration grid (integer string).

        Rigs are calibrateed regularly (at least annually), so the ``<rig_name>`` directory could contain multiple date
        folders. The method returns the "most recent" folder dated prior to the session date, or if that does
        not exist, the oldest folder dated after the session date. It does NOT check the contents of the calibration
        folder ``<S>mm_<R>_r_<C>c`` itself, and it expects to find only one such calibration folder within any given
        date folder.

        :param sesh_id: The session identifier.
        :param calib_root: Source directory containing all calibration data folders.
        :return: The full path to the calibration folder, as described. If no calibration folders are found, returns
            ``None``. Always returns ``None`` for a legacy-cam session, to which calibration data does not apply.
        """
        if not (sesh_id.is_fixed_cam and isinstance(calib_root, Path) and calib_root.is_dir()):
            return None

        # find directory corresponding to this session's rig
        rig_dir = next((path for path in calib_root.rglob(sesh_id.rig) if path.is_dir()), None)
        if rig_dir is None:
            return None

        # get path to each correctly named FLIR calibration folder under a 'YYYYMMDD` date subfolder of the rig folder
        valid_calib_folders: List[Path] = []
        for date_dir in [f for f in rig_dir.iterdir() if f.is_dir() and validate_date_string(f.name)]:
            for f in date_dir.iterdir():
                if f.is_dir() and valid_flir_calib_folder_name(f.name):
                    valid_calib_folders.append(f)
                    break   # we only expect one calib folder within each date folder

        if len(valid_calib_folders) == 0:
            return None
        valid_calib_folders.sort(key=lambda f: f.parent.name)
        idx = bisect.bisect_left(valid_calib_folders, sesh_id.date, key=lambda f: f.parent.name)
        return valid_calib_folders[0 if idx == 0 else idx - 1]


def validate_date_string(s: str) -> bool:
    """
    Verify that specified string is a date string in the 'YYYYMMDD' format used throughout ReachX.
    :param s: The string to test
    :return: True if string is a valid date string as described; else False.
    """
    try:
        datetime.strptime(s, '%Y%m%d')
        return True
    except ValueError as _:
        return False


def session_number_from_folder(s: str) -> Optional[str]:
    """
    Get the 3-digit number of a ReachX experiment session from the session folder name: 'sessionNNN'.
    :param s: Session folder name.
    :return: The 3-digit session number in string form, or None if session folder name is invalid.
    """
    return None if (re.fullmatch('session[0-9]{3}', s) is None) else s[7:]


def valid_flir_calib_folder_name(s: str) -> bool:
    """
    The name of a valid FLIR calibration folder has the format ``<S>mm_<R>_r_<C>c``, where ``<S>, <R>, <C>`` are
    integer strings.
    :param s: Candidate folder name.
    """
    return bool(re.fullmatch(r"^\d+mm_\d+r_\d+c$", s))


class RxModelID:

    """
    Immutable container for the elements that identify the project folder for a body part detection model within a
    workspace, implementing the ReachX convention:
     - ``<model_root>/<proj_name>-<date>`` for a legacy-cam model, or
     - ``<model_root>/<proj_name-<date>-fix`` for a fixed-cam model.

    By design, ReachX separates experiments run on the modern fixed-cam setup from those run on a legacy-cam setup at
    the workspace level. A fixed-cam workspace can contain only fixed-cam sessions and models, and analogously for a
    legacy-cam workspace. Since a user could use the same model root for a fixed-cam and a legacy-cam workspace, there
    must be a way to distinguish a fixed-cam model from a legacy-cam model. Hence the ``-fix`` suffix in the name
    of a fixed-cam model folder.

    ``RxModelID`` also defines the directory structure for the project folder and provides various convenience methods
    related to that structure.
    """
    ITER_PREFIX: str = "iter-"
    """ Prefix for all model iteration directories. Directory name is completed by appending the iteration number.  """
    MARKERS_DIR: str = "annotations"
    """ 
    Subfolder in a model iteration directory in which annotated training data (body part markers attached to selected 
    frames from the sideCam and frontCam videos recorded during one or more experiment sessions) is stored.
    """
    MARKERS_SUFFIX: str = "markers.txt"
    """ 
    Suffix of text file containing all body part markers defined for a given marked session in the training data. 
    The full filename includes the session's prefix prepended to this string.
    """
    TRAINING_DIR: str = "training-set"
    """ 
    Subfolder in model iteration directory in which the collected training data is stored in the format required
    for training the network model IAW the `imgaug` dataset type. 
    """
    IMAGES_DIR: str = "images"
    """ Subfolder in model iteration's training set folder where all body part-marked video frame images are stored. """
    MODEL_DIR: str = "trained-model"
    """ Subfolder in model iteration directory in which the files defining the trained model are stored. """
    TRAINING_CFG_FILE: str = "train_cfg.yaml"
    """ 
    The file name for the YAML file containing the training parameters for a body part detection model. This is
    placed in the model files folder. Notably, it includes the underlying NN type for the model, which is needed when 
    using the model to detect body part locations in session videos.
    """

    def __init__(self, model_root: Path, proj_name: str, date: str, is_fixed: bool):
        """
        Construct a body part detection model identifier: ``<model_root>/<proj_name>-<date>`` for a legacy-cam model,
        or ``<model_root>/<proj_name>-<date>-fix`` for a fixed-cam model.

        :param model_root: The workspace's ML model projects root.
        :param proj_name: The model project name (assigned by user). Must be a non-empty alphanumeric string.
        :param date: The date the model project was created in 'YYYYMMDD' format.
        :param is_fixed: True (False) for a fixed-cam (legacy-cam) model.
        """
        assert (validate_date_string(date) and proj_name.isalnum())
        self._root = model_root
        self._name = proj_name
        self._date = date
        self._is_fixed_cam = is_fixed
        self._loc: Path = Path(self._root, self.project_folder_name)

    @property
    def root(self) -> Path:
        """ Root path for all ReachX ML model project folders. Each subfolder therein represents a distinct model. """
        return self._root

    @property
    def project_name(self) -> str:
        """ User-assigned name for the ML model project. """
        return self._name

    @property
    def date(self) -> str:
        """ The date the model was created, in the format 'YYYYMMDD'. """
        return self._date

    @property
    def is_fixed_cam(self) -> bool:
        """ True for a fixed-cam model (can only operate on fixed-cam sessions); False for a legacy-cam model. """
        return self._is_fixed_cam

    @property
    def project_folder_name(self) -> str:
        """ Name of the subfolder within the workspace's model root in which this model's definition is stored. """
        return f"{self._name}-{self._date}{'-fix' if self._is_fixed_cam else ''}"

    @property
    def location(self) -> Path:
        """ File system location for the model's project folder. """
        return self._loc

    def exists(self) -> bool:
        """ Does the model's project folder exist in the file system? """
        return self._loc.is_dir()

    def iteration_folder(self, iter_num: int) -> Path:
        """
        File system path for the directory containing all information about the specified iteration of this model.
        :param iter_num: The iteration number; must be nonnegative.
        :return: The iteration folder path.
        """
        assert iter_num >= 0
        return Path(self._loc, f"{self.ITER_PREFIX}{iter_num}")

    def find_all_iterations(self) -> List[int]:
        """
        Find all iterations defined for this body part detection ML model by checking all immediate subfolders of the
        model's project directory that conform to the expected structure for a model iteration backing store.
        :return: List of existing iteration numbers, in ascending order.
        """
        out: List[int] = list()
        if self.exists():
            iter_dirs = [p for p in self.location.iterdir() if p.name.startswith(self.ITER_PREFIX)]
            for p in iter_dirs:
                iter_num = -1
                try:
                    iter_num = int(p.name[len(self.ITER_PREFIX):])
                except (ValueError, Exception):
                    pass
                if (iter_num >= 0) and self.check_iteration_folder_structure(iter_num):
                    out.append(iter_num)
        out.sort()
        return out

    def check_iteration_folder_structure(self, iter_num: int) -> bool:
        """
        Verify that the specified model iteration directory contains the 3 required subfolders.
        :param iter_num: The iteration number.
        :return: True if the iteration directory and its 3 required subfolders are present, else False. Subfolder
            content is not examined.
        """
        p = self.iteration_folder(iter_num)
        return all([Path(p, sub).is_dir() for sub in [self.MARKERS_DIR, self.TRAINING_DIR, self.MODEL_DIR]])

    def markers_folder(self, iter_num: int) -> Path:
        """
        File system path for the directory where the body part markers files are stored for each marked session in the
        specified model iteration's annotated training data.
        :param iter_num: The iteration number; must be nonnegative.
        :return: The markers folder path.
        """
        return Path(self.iteration_folder(iter_num), self.MARKERS_DIR)

    def markers_file_for_session(self, iter_num: int, sesh_id: RxSessionID) -> Path:
        """
        Full path to the body part markers file for a marked session included in the specified model iteration's
        annotated training data.
        :param iter_num: The iteration number; must be nonnegative.
        :param sesh_id: The marked session identifier. The session prefix is used to form the markers file name.
        :return: The markers file path.
        :raises AssertionError: If the ``sesh_id`` is not valid, or if this model is fixed-cam and the session is not,
            or vice versa.
        """
        assert isinstance(sesh_id, RxSessionID)
        assert sesh_id.is_fixed_cam == self.is_fixed_cam
        return Path(self.markers_folder(iter_num), f"{sesh_id.session_file_prefix}{self.MARKERS_SUFFIX}")

    def get_all_existing_marker_files(self, iter_num: int) -> List[Path]:
        """
        Get the full paths for all existing body part markers files for the specified iteration of this model.
        :param iter_num: The iteration number; must be nonnegative.
        :return: List of markers files found, possibly empty.
        """
        src_dir = self.markers_folder(iter_num)
        if src_dir.is_dir():
            return [f for f in src_dir.iterdir() if (f.is_file() and f.name.endswith(self.MARKERS_SUFFIX))]
        return []

    @staticmethod
    def session_id_from_marker_filename(session_root: Path, marker_file: Path) -> Optional[RxSessionID]:
        """
        Get the experiment session ID for a body part markers file found in the markers folder within a model's
        project directory. By design, the markers file name begins with the session file prefix.
        :param session_root: The file system location for all experiment sessions. Required to form the session
            identifier.
        :param marker_file: A body part markers file.
        :return: The session identifier, or None if markers file name does not begin with a valid session file prefix,
            the session root directory is invalid.
        """
        out: Optional[RxSessionID] = None
        if isinstance(session_root, Path) and marker_file.name.endswith(RxModelID.MARKERS_SUFFIX):
            last = marker_file.name.index(RxModelID.MARKERS_SUFFIX)
            out = RxSessionID.from_file_prefix(session_root, marker_file.name[0:last])
        return out

    def training_set_folder(self, iter_num: int) -> Path:
        """
        File system path for the training set folder for the specified iteration of this model.
        :param iter_num: The iteration number; must be nonnegative.
        :return: The training set folder path.
        """
        return Path(self.iteration_folder(iter_num), self.TRAINING_DIR)

    def training_set_images_folder(self, iter_num: int) -> Path:
        """
        File system path for the training set subfolder in which are stored all marked image frames extracted from the
        RxCam.SIDE and RxCam.FRONT videos recorded during each marked session included in the training data for the
        specified iteration of this model.
        :param iter_num: The iteration number; must be nonnegative.
        :return: The images folder.
        """
        return Path(self.training_set_folder(iter_num), self.IMAGES_DIR)

    def model_files_folder(self, iter_num: int) -> Path:
        """
        File system path for the directory containing all model definition files generated while training the specified
        iteration of this model.
        :param iter_num: The iteration number; must be nonnegative.
        :return: The model files folder.
        """
        return Path(self.iteration_folder(iter_num), self.MODEL_DIR)

    def has_been_trained(self, iter_num) -> bool:
        """
        Has the specified iteration of this model been trained? This method checks for the existence of key files
        within the model files folder for that iteration.

        :param iter_num: The iteration number; must be nonnegative.
        :return: True if model iteration has been trained, else False.
        """
        mf = self.model_files_folder(iter_num)
        ok = Path(mf, self.TRAINING_CFG_FILE).is_file() and Path(mf, "learning_stats.csv").is_file()
        if ok:
            # is there at least one model snapshot index
            snapshot = next((f for f in mf.iterdir() if (f.name.startswith("snapshot") and f.name.endswith("index"))),
                            None)
            ok = snapshot is not None
        return ok

    def __eq__(self, other) -> bool:
        if isinstance(other, RxModelID):
            return ((self._root == other._root) and (self._name == other._name) and (self._date == other._date) and
                    (self._is_fixed_cam == other._is_fixed_cam))
        return False

    def __str__(self) -> str:
        return self.project_folder_name

    @staticmethod
    def create_model_id(root: Path, name: str, is_fixed: bool) -> Optional[RxModelID]:
        """
        Create a body part detection model identifier using today's date.
        :param root: The workspace's ML model projects root.
        :param name: The model project name (assigned by user). Must be a nonempty alphanumeric string.
        :param is_fixed: True to create a fixed-cam model, False for a legacy-cam model
        :return: The model ID, or None if `proj_name` is invalid.
        """
        if isinstance(root, Path) and name.isalnum():
            return RxModelID(root, name, datetime.today().strftime("%Y%m%d"), is_fixed)
        return None

    @staticmethod
    def from_folder_name(root: Path, folder: str) -> Optional[RxModelID]:
        """
        (For settings file support) Construct the model project ID for a ML model project stored in the specified
        subfolder under the specified model root.
        :param root: Full path to a workspace's model root.
        :param folder: The ML model project subfolder name in the format '<proj_name>-YYYYMMDD' for a legacy-cam model,
            and '<proj_name>-YYYYMMDD-fix' for a fixed cam model.
        :return: The model ID, or None if `folder` is ill-formed.
        """
        if isinstance(root, Path) and isinstance(folder, str):
            tokens = folder.split('-')
            if len(tokens) == 2:
                proj_name, date_str = tokens[0], tokens[1]
                if validate_date_string(date_str) and proj_name.isalnum():
                    return RxModelID(root, proj_name, date_str, False)
            if len(tokens) == 3:
                proj_name, date_str, fix_suffix = tokens[0], tokens[1], tokens[2]
                if validate_date_string(date_str) and proj_name.isalnum() and fix_suffix == 'fix':
                    return RxModelID(root, proj_name, date_str, True)
        return None


class RxNetworkType(StrEnum):
    """
    Enumeration of the neural network architecture types that ReachX supports for training an object detection
    model used to find selected body parts, the food pellet, and other features in ReachX session videos.
    """
    RESNET_50 = "resnet_50"
    """ Residual Network architecture with 50 layers. """
    RESNET_152 = "resnet_152"
    """ Residual Network architecture with 152 layers (more complex, more accurate, and slower performance). """


class RxReach(IntEnum):
    """
    Enumeration of all defined events marking a single reach segment in the recorded session timeline.

    **DEVNOTES**:
     - Designed analogously to RxBodyPart and RxCam for convenient and compact persistence, and so that we can easily
     convert beteen existing reach event names and their corresponding RxReach enumerants.
    """
    INIT = 0    # starting at 0 for convenient indexing into _nicknames()
    """ Start of a reach segment. """
    MAX = 1
    """ When max hand velocity is reached in the reach segment. """
    GRABBED = 2
    """ Reach segment ended successfully. """
    MISSED = 3
    """ Reach segment ended unsuccessfully: pellet missed. """
    DROPPED = 4
    """ Reach segment ended unsuccessfully: pellet dropped. """
    STALLED = 5
    """ Reach segment ended unsuccessfully: stalled. """
    NONE = 6
    """ Reach segment detected but not classified as a grabbed/missed/dropped/stalled outcome. """

    @staticmethod
    def _nicknames() -> List[str]:
        """ List of all reach segment event nicknames, in enumerant order. Not for external use. """
        return ['reachInit', 'reachMax', 'reachEnd_grabbed', 'reachEnd_missed', 'reachEnd_dropped',
                'reachEnd_stalled', 'reachEnd_none']

    @staticmethod
    def from_nickname(nickname: str) -> Optional[RxReach]:
        """
        Convert specified reach segment event nickname to the corresponding `RxReach` enumerant.
        :param nickname: Reach segment event nickname.
        :return: Corresponding enumerant, or None if nickname not recognized.
        """
        try:
            idx = RxReach._nicknames().index(nickname)
            return RxReach(idx)
        except ValueError:
            return None

    @property
    def is_result_code(self) -> bool:
        """ True if this enumerant identifies the end result for a reach. """
        return self.value > RxReach.MAX

    @property
    def short_name(self) -> str:
        out = self.__str__()
        return out[9:] if self.is_result_code else out

    @staticmethod
    def _ui_colors() -> List[str]:
        return ['gray', 'white', 'green', 'darkgoldenrod', 'orange', 'red', 'gray']

    @property
    def ui_color(self) -> str:
        """
        UI color associated with this reach segment enumerant. Must be a color string accepted by PyQtGrqph's
        ``mkBrush()`` utility.
        """
        return RxReach._ui_colors()[self.value]

    def __str__(self) -> str:
        """ Overridden to return associated reach segment event nickname. """
        return RxReach._nicknames()[self.value]


class RxHandPos(IntEnum):
    """ Enumeration of values for pellet-relative hand position at the ``RxReach.MAX`` keyframe of reach segment. """
    UNSPECIFIED = 0
    """ Hand position relative to pellet unspecified. """
    LEFT = 1
    """ Hand left of pellet at reachMax. """
    RIGHT = 2
    """ Hand right of pellet at reachMax. """
    ABOVE = 3
    """ Hand above pellet at reachMax. """
    BELOW = 4
    """ Hand below pellet at reachMax. """

    def __str__(self) -> str:
        """ Overridden to return first letter of enumerant name. """
        return ["U", "L", "R", "A", "B"][self.value]


class RxReachSegment(NamedTuple):
    """
    A single reach segment within an experiment session.

    **Versioning Note**: The pellet-relative hand position at reachMax was added as a parameter a/o v0.5.5. See
    ``load_reach_segments()``.
    """
    frame: int
    """ Frame number at which reach segment starts in master camera timeline. """
    max_delta: int
    """ # of frames after start when hand reaches max velocity: ``RxReach.MAX``. """
    dur: int
    """ Segment duration in frames, ie, segment start - segment end. """
    result: RxReach
    """ The reach segment result: grabbed, missed, stalled, dropped. """
    hand_pos: RxHandPos
    """ Pellet-relative position of hand at ``RxReach.MAX`` keyframe. """

    @property
    def max_frame(self) -> int:
        """ Frame number at which hand reaches max velocity. """
        return self.frame + self.max_delta

    @property
    def end_frame(self) -> int:
        """ Frame number at which reach segment ends. """
        return self.frame + self.dur

    @staticmethod
    def save_reach_segments(reaches: List[RxReachSegment], out_file: Path) -> str:
        """
        Save a list of reach segments to a text file using ``np.savetxt()``. The file starts with a column header
        ("frame max_delta dur result hand_pos"), and each subsequent line is a whitespace-separated list of 5 integers
        -- starting frame, delta to "reach max", duration, a ``RxReach`` integer enumerant value, and a ``RxHandPos``
        enumerant value -- defining one reach segment.

        :param reaches: The list of reach segments.
        :param out_file: The file destination. Any existing file is overwritten.
        :return: "" on success, else an error description.
        """
        try:
            raw = np.zeros((len(reaches), 5), dtype='i4')
            row_idx = 0
            for reach in reaches:
                raw[row_idx] = [reach.frame, reach.max_delta, reach.dur, reach.result, reach.hand_pos]
                row_idx += 1
            np.savetxt(out_file, raw, fmt='%d', header='frame  max_delta  dur  result  hand_pos')
            return ""
        except Exception as e:
            return f"Unable to save reach segments to {out_file.name}: {e}"

    @staticmethod
    def load_reach_segments(p: Path) -> Tuple[str, List[RxReachSegment]]:
        """
        Load a list of reach segments from a text file formatted as described in ``save_reach_segments()``. Any invalid
        reach segments are removed silently, and the list is culled as needed to ensure no two reach segments overlap.

        Prior to v0.5.5, each reach segment was persisted as a line of four whitespace-separated integers. As of v0.5.5,
        the pellet-relative hand position parameter, ``RxHandPos``, was added -- so now each reach segment is saved
        as a line of five integers. This method handles both possibilities. For the older format, the hand position
        parameter is set to ``RxHandPos.UNSPECIFIED`` for all defined reaches.

        :param p: The source file.
        :return:  ("", list of reach segments) if successful; else (error message, []). The list of reach segments are
            sorted in ascending order by starting frame.
        """
        # IMPORTANT: The int() casts are important below to get rid of numpy int objects!!!
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                raw = np.loadtxt(p, dtype='i4', ndmin=2)  # always want 2D array even if there's only one line!
            if raw.size == 0:
                return "", []
            if len(raw.shape) != 2 or not (4 <= raw.shape[1] <= 5):
                raise Exception(f"Invalid format - expect 4 or 5 columns of whitepace-separated ints")
            out: List[RxReachSegment] = list()
            if raw.shape[1] == 4:
                for row in range(raw.shape[0]):
                    frame, max_delta, dur, result = raw[row]
                    out.append(RxReachSegment(int(frame), int(max_delta), int(dur), RxReach(int(result)),
                                              RxHandPos.UNSPECIFIED))
            else:
                for row in range(raw.shape[0]):
                    frame, max_delta, dur, result, hand_pos = raw[row]
                    out.append(RxReachSegment(int(frame), int(max_delta), int(dur), RxReach(int(result)),
                                              RxHandPos(int(hand_pos))))
            RxReachSegment._cull_and_sort(out)
            return "", out
        except Exception as e:
            return f"Error loading reach segments from {p.name}: {e}.", []

    @staticmethod
    def overlaps(test_seg: RxReachSegment, existing: List[RxReachSegment]) -> bool:
        """
        Check if specified reach segment overlaps (strictly - cannot share an endpoint!) any one of an existing list of
        reach segments.
        :param test_seg: The reach segment to test.
        :param existing: A list of existing reach segments.
        :return: True if an overlap is detected, else False.
        """
        for seg in existing:
            if (test_seg.frame <= seg.end_frame) and (test_seg.end_frame >= seg.frame):
                return True
        return False

    @staticmethod
    def is_valid(seg: RxReachSegment) -> bool:
        """
        Is the specified reach segment valid? Verifies the following:
         - Result code and hand position are valid instances of ``RxReach`` and ``RxHandPos``, respectively.
         - Starting frame is non-negative.
         - Segment duration  >= ``RX.MIN_REACHSEG_DUR``. No maximum duration enforced.
         - The "reachMax" frame is between starting and ending frame.
        :param seg: The reach segment to test.
        :return: True if valid, else False.
        """
        return (isinstance(seg.result, RxReach) and seg.result.is_result_code and isinstance(seg.hand_pos, RxHandPos)
                and (seg.frame >= 0) and (0 < seg.max_delta < seg.dur) and (RX.MIN_REACHSEG_DUR <= seg.dur))

    @staticmethod
    def _cull_and_sort(segs: List[RxReachSegment]) -> None:
        """
        Helper mthod for ``load_reach_segments()``. Removes any invalid segments, ensures no overlapping segments,
        then sorts the remaining segments in ascending order by starting frame.
        :param segs: The reach segment list. Upon return, it contains only valid, non-overlapping reach segments
            sorted in ascending order.
        """
        copy_segs = segs.copy()
        segs.clear()
        for seg in copy_segs:
            if RxReachSegment.is_valid(seg) and not RxReachSegment.overlaps(seg, segs):
                segs.append(seg)
        segs.sort(key=lambda s: s.frame)


class RxReachSegControls(NamedTuple):
    """
    User-specified parametric controls for the automated reach segmentation algorithm in ReachX. Note the
    application default value specified for each parameter.
    """
    reach_init_speed: float = -0.025
    """ Hand-to-pellet approach speed threshold in mm/s. Range: [-0.1 .. -0.01]. """
    reach_dirchange_speed: float = 0.025
    """ Hand speed threshold indicating direction change since reach started, in mm/s. Range: [0.01 .. 0.1]. """
    pellet_drop_speed: float = 0.275
    """ Pellet speed exceeding this threshold in mm/s indicates pellet was dropped. Range: [0.1 .. 0.5]. """
    pellet_drop_distZ: float = -5
    """ If pellet Z-coordinate (mm) WRT pellet origin is < this threshold, pellet was dropped. Range: [-2 .. -10]. """
    pellet_drop_distY: float = -3
    """ If pellet Y-coordinate (mm) WRT pellet origin is < this threshold, pellet was dropped. Range: [-2 .. -10]. """
    dist_thresh_end: float = 5
    """ Minimum distance of hand from pellet origin at reach's end, in mm. Range: [2..10]. """
    confidence: float = 0.9
    """ Minimum confidence value required for 'detected' hand and pellet locations, in [0..1]. """
    pellet_dist_to_origin: float = 2
    """ Minimum distance of pellet from pellet origin to indiate pellet was grabbed, in mm. Range: [1..5]. """
    max_frame: int = 100
    """ Maximum allowed length of a reach segment (# video frames). Range: >= 5. """
    min_frame: int = 15
    """ Minimum allowed length of a reach segment (# video frames). Range: >= 5. """
    distZ_hand_start: float = 3
    """ Minimum separation between hand and pellet origin at reach initiation, in mm. Range: [1..9] """


class TrajCol(IntEnum):
    """
    Enumeration that defines the column indices of hand/pellet trajectory data saved in Numpy array.
    These refer to the trajectory data saved in the files ``hand.npy`` and ``pellet.npy`` in each scorer folder
    within the session folder. **For legacy-cam sessions only.** Use ``FixedCamTrajCol`` for fixed-cam sessions.
    """
    Y = 0
    Y_FILT = 1
    Z = 2
    Z_FILT = 3
    YZ_LHOOD = 4
    X = 5
    X_FILT = 6
    X_LHOOD = 7
    DIST = 8
    SPEED = 9
    SPEED_FILT = 10


class FixedCamTrajCol(IntEnum):
    """
    Enumeration that defines the column indices of body part trajectory data saved as ``(N, M)`` Numpy arrays, where
    ``N`` = session length in video frames and ``M = len(FixedCamTrajCol)``. All body part trajectory arrays computed
    for a session are saved in ``trajectories.npz`` in the scorer folder within the session folder. Each trajectory
    array saved is assigned a name matching the corresponding body part's nickname.

    **For fixed-cam sessions only.** Use ``TrajCol`` for legacy-cam sessions.
    """
    X = 0
    """ X-coordinate of body part trajectory (mm), after filtering, interpolation, triangulation, and recentering. """
    Y = 1
    """ Y-coordinate of body part trajectory (mm), after filtering, interpolation, triangulation, and recentering. """
    Z = 2
    """ Z-coordinate of body part trajectory (mm0, after filtering, interpolation, triangulation, and recentering. """
    P = 3
    """ Confidence flag per body part location: 1 for high-confidence, otherwise 0. """
    DIST = 4
    """ Computed 3D distance of body part from the origin (after recentering), in mm. """
    SPEED = 5
    """ Computed 3D speed of body part wrt origin (after recentering), in mm/ms. """
    SPEED_FILT = 6
    """ Filtered 3D speed fo body part wrt origin (after recentering), in mm/ms. """


class RxPreferences:
    """
    Immutable container for user application preferences that are exposed only in the standard "Preferences" dialog and
    are persisted in the user's ReachX settings file.  **This is NOT for settings like window geometry, MRU workspace,
    workspace parameters, and so on.**

    For now, this only includes a couple of performance flags affecting the session analysis task. Additional
    preferences may be added in the future -- again, only for those settings that are not exposed anywhere on the UI
    except in the Preferences dialog.
    """
    def __init__(self, use_gpu: bool = True, enable_pp: bool = True) -> None:
        """
        onstruct a ReachX user preferences object.
        :param use_gpu: Whether to enable GPU inference support for session video analysis. Default = ``True``.
        :param enable_pp: Whether to enable parallel vs sequential analysis of session videos. Default = ``True``.
        """
        self._use_gpu = use_gpu
        self._enable_pp = enable_pp

    @property
    def use_gpu(self) -> bool:
        """ Enable (or disable) use of GPU inference support for session video analysis. """
        return self._use_gpu

    @property
    def enable_pp(self) -> bool:
        """ Enable (or disable) parallel processing of videos during session video analysis. """
        return self._enable_pp
