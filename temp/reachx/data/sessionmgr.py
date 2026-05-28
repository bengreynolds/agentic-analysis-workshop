"""
Session module for ReachX.

This module encapsulates the details of reading and writing content files associated with a single experiment session.

**Fixed-cam vs legacy-cam sessions**:

A new "fixed-cam" rig will eventually replace each of the current "legacy-cam" rigs. Key differences:
 - Required camera videos used for body part detection model training and computation of hand/pellet trajectories:
   ``RxCam.LEFT/RIGHT`` for fixed-cam; ``RxCam.SIDE/FRONT`` for legacy-cam.
 - Additional optional cams: ``RxCam.STIM/FAST`` for legacy-cam; None for fixed-cam.
 - Calibration: To convert pixels to scene millimeters for legacy-cam videos, hard-coded calibration constants are
   used. For the new fixed-cam system, each setup has its own calibration file, and the conversion is more complex.
 - Frame size: 300x200 for legacy-cam videos, 256x256 for fixed-cam.

Support for the fixed-cam setup was introduced in ReachX v0.6.0. It is not appropriate to mix legacy-cam and fixed-cam
experiment sessions, so ReachX workspaces are created as either fixed-cam or legacy-cam workspaces. A fixed-cam
workspace only exposes fixed-cam sessions and models, and fixed-cam models can only process fixed-cam sessions.

**Classes**:
 - ``SessionManager``: Encapsulates the session currently loaded into ReachX. Handles loading/unloading a session,
   navigating through or playing back the videos recorded during the session, providing access to the recorded events
   list, curated reach events, and results from model-driven analysis (predicted hand and pellet trajectories,
   auto-generated reach segments).
 - ``CamSource``: A video camera source recorded during the session. Finds and loads the MP4 video and corresponding
   frame interval timestamps file for one of the recognized rig cams. Uses cv2.VideoCapture() to load the video file,
   and parses the timestamps file to identify dropped frames. Maintains the notion of a current frame; reads and
   processes the image for the current frame, ready for display in a PyQtGraph `ImageWidget'.
 - ``SessionEventList``: Ordered list of events recorded during the session. The events are corrected for any frame
   drops in the timeline of the "master cam".
 - ``SessionScorer``: Identifies the ReachX body part detection model used to analyze an experiment session. Note that
   any given session may be analyzed by more than one such "scorer".

By convention, all session files begin with a prefix FP that preserves the recording date, rig name, and session number:
``FP=YYYYMMDD_<rig>_session<num>``. The minimum contents of a valid session folder include:
 - The video source file ``<FP><cam_name>-NNNN.mp4`` and the frame interval timestamps file
   ``<FP><cam_name>_timestamps.txt`` for each of the two **required** cameras recording the experiment. In a legacy-cam
   session, data files for other cams may also be present but are not essential. It is considered a session load error
   if a required cam's MP4 file is present but the corresponding timestamps file is missing. The timestamps file must be
   present to identify any dropped frames in the cam's timeline.
 - The number of frame intervals in a cam's timestamps file must match the number of captured frames in the cam's
   video file -- else it is a load error.
 - The recorded events file ``<FP>events.txt``, a two-column text file (event_type, event_frame_num), with events in
   ascending chronological order.

Additional files that may be added by ReachX:
 - The list of CURATED reach segments is persisted in a four-column text file (frame, max_delta, dur, result),
   ``<FP>reaches.txt``.
 - If the session has been analyzed by the body part detection model instance, or "scorer", all analysis results are
   stored in the subfolder ``<session_folder>/<rx_scorer>``, where ``<rx_scorer>`` identifies the scorer. Predicted
   per-frame body part locations are in ``<cam_name>.npy`` for each of the two required cameras. Those body part markers
   located with high confidence (> 0.9) are found in ``detected_markers.npy``. Predicted and filtered pellet and hand
   trajectory data derived from the predicted body part locations are stored in ``pellet.npy`` and ``hand.npy``.
   Finally, the results of automated reach segmentation based on the trajectories is found in ``detected_reaches.txt``.

Typically, a copy of the hardware config file **<FP>systemdata_copy.yaml** is present in the session folder. We can
We can use this to identiy the master cam and find the frame rate for any cam. However, user entry errors make that file
somewhat suspect, so it is not required. If missing, then ``RxCam.SIDE`` is assumed to be the master cam for a
legacy-cam session; ``RxCam.LEFT`` for a fixed-cam session. And we find a cam's frame rate by inverting the average of
all the frame intervals listed in the cam's timestamps file.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, NamedTuple

# noinspection PyPackageRequirements
import cv2

import numpy as np
import yaml
from PySide6.QtCore import QObject, Signal, QTimer

from reachx.common import RxSessionID, RxCam, RX, RxEvent, RxReachSegment, RxBodyPart, RxNetworkType, TrajCol, \
    FixedCamTrajCol
from reachx.config.app_log import get_application_logger, get_null_logger
from reachx.modeling.segmentation.result_store import RESULTS_FOLDER_NAME

_RX = RX()
""" Application-wide constants. """


def _determine_camera_master_from_systemdata(session_id: RxSessionID, logger: logging.Logger) -> RxCam:
    """
    Determine which camera source is the master camera for the specified session. If the session folder contains a copy
    of the hardware config file at ``<session_folder>/<session_prefix>systemdata_copy.yaml``, read it to determine which
    camera was "master". The frame numbers listed in the session events file correspond to frames on this camera.

    If the file is not found or cannot be read, assume ``RxCam.SIDE`` is the master camera for a legacy-cam session,
    while ``RxCam.LEFT`` is the default master camera for a fixed-cam session.

    DEVNOTES:
     - This can be important if the frame counts are not the same for the available cameras, and particularly
       if there were one or more dropped frames on any or all of the cameras.
     - Method chooses the first cam for which the ``ismaster`` field is True. The cam's name is in the ``nickname``
       field and should correspond to an ``RxCam`` enumerant.
     - The master cam should always be either ``RxCam.SIDE/FRONT`` for the legacy-cam setup camera setup, or
       ``RxCam.LEFT/RIGHT`` for the fixed-cam setup.

    :param session_id: The session ID, which also locates the session folder in the file system and identifies the
        session as fixed-cam or legacy-cam.
    :param logger: A logger object for logging error message.
    :return: Identity of the master camera.
    """
    yaml_path = Path(session_id.location, f"{session_id.session_file_prefix}systemdata_copy.yaml")
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            for cam in ['cam1', 'cam2', 'cam3', 'cam4']:
                if (cam in config) and config[cam]['ismaster']:  # 'ismaster' field is boolean-valued
                    out = RxCam.from_nickname(config[cam]['nickname'])
                    if out is None:
                        return RxCam.LEFT if session_id.is_fixed_cam else RxCam.SIDE
                    else:
                        return out
    except Exception as e:
        logger.error(f"Error parsing {yaml_path.name}: {e}")

    return RxCam.LEFT if session_id.is_fixed_cam else RxCam.SIDE


class CamSource:
    """
    Video recorded from any one of the ReachX rig cameras.

    `CamSource` reads two files in the session folder for a given camera source:
     - The MP4 file containing the video recorded during the session for the identified camera.
     - A text file containing frame-to-frame intervals in nanoseconds. The number of frame intervals in this file
       should match the number of frames in the video file. Any dropped frames in the video timeline are found by
       comparing these frame intervals against the nominal frame interval. See `_find_dropped_frames()` for details.

    Since `CamSource` accounts for dropped frames, the total frame count N for a recorded video is the total number of
    captured frames plus the number of dropped frames. If one of the dropped frames is requested, the resulting image
    frame is None, indicating that frame was not captured during recording.

    IMPORTANT: We have encountered sessions in which the non-master camera(s) have started recording 1+ frames AFTER
    the master camera. See `CamSource.load_session_cams()`` for how we handle this scenario.

    NOTES:
     - The nominal frame rate for a camera can be read from the copy of ``systemdata.yaml`` present in session folder.
       However, there may be user-entry errors in that file that could cause issues. Instead, average the frame
       intervals in the timstamps file, invert and round to an integer to get the frame rate in Hz. The two acceptable
       values (for now) are 150Hz and 900Hz, but that could change in the future.
     - Sometimes there will be more than one MP4 file present for a given cam (user had to compress the raw file again
       due to failed compression or a corrupted file). The MP4 file name is <FP><cam_name>-NNNN.mp4, where FP is the
       session file prefix and NNNN is a zero-padded 4-digit index. If multiple files are present, we use the one with
       the largest index.
    """
    @staticmethod
    def load_session_cams(session_id: RxSessionID, which: List[RxCam],
                          logger: logging.Logger) -> Dict[RxCam, CamSource]:
        """
        Load one or more rig camera videos for the specified experiment session.

        For each requested camera source, the session folder must contain the MP4 video file and the relevant frame
        interval timestamps file. The latter is essential for detecting any dropped frames in the captured video.

        IMPORTANT: We have encountered sessions in which the non-master camera(s) have started recording 1+ frames AFTER
        the master camera. This method was introduced to handle such scenarios as efficiently as possible.
        - We ASSUME the master camera always starts first. The total frame count (captured + dropped) on that camera is
          the total number of frames in the session.
        - This method always loads the master camera first so that the master frame count is known before loading other
          camera videos. Even if master camera is NOT among the camera sources listed in the ``which`` argument, the
          method will nevertheless load the master camera video in order to pass on the master frame count when loading
          the camera sources requested.
        - For all other cameras: Let M = master cam frame count and N = any other camera's frame count. If N > M
          (unlikely), adjust the camera's captured frame count and dropped frame list so that M == N. If N < M, insert
          M-N additional dropped frames at the BEGINNING of that camera's video stream so that the frame counts match.
          This happens in ``Cam._load()``.

        :param session_id: The session identifier, which locates the session folder on the file system.
        :param which: List of camera videos to be loaded.
        :param logger: Logger object.
        :return: A dictionary ``D`` mapping the ID of each camera loaded to the corresponding ``CamSource`` instance.
            Note that ``D`` will always include the master cam, even if it was not requested via the ``which'' arg. It
            is a fatal error if the master cam cannot be loaded, in which case D is empty and an error message is
            logged. If any other requested cam could not be loaded, an error message is delivered via the logger, and
            that cam is excluded from ``D``.
        """
        if not session_id.location.is_dir():
            logger.error("Unable to load session cams: session folder not found!")
            return {}

        # consistency check: A fixed-cam session cannot have any legacy-cams, and vice versa
        available = set(RxCam.available_cams(session_id.is_fixed_cam))
        if any(cam not in available for cam in which):
            logger.error(f"Requested a cam that is not available in a "
                         f"{'fixed' if session_id.is_fixed_cam else 'legacy'}-cam session!")
            return {}

        # always load the master camera first and include it in output, even if not requested. Need master frame count
        # to detect dropped frames at the start of the video streams of any other cam sources.
        master_cam = _determine_camera_master_from_systemdata(session_id, logger)
        emsg, src = CamSource._load_cam(session_id, master_cam, logger)
        if len(emsg) > 0:
            logger.error(f"Unable to load master cam: {emsg}")
            return {}
        n_master = src.frame_count

        out: Dict[RxCam, CamSource] = dict()
        out[master_cam] = src

        for cam in [c for c in which if c != master_cam]:
            emsg, src = CamSource._load_cam(session_id, cam, logger, n_master)
            if src is not None:
                out[cam] = src
                if src.frame_count_adjusted_to_master:
                    logger.warning(f"Had to adjust {str(cam)} timeline to match master - may not be in sync!")
            elif len(emsg) > 0:
                logger.error(emsg)

        return out

    @staticmethod
    def _load_cam(session_id: RxSessionID, cam: RxCam, logger: logging.Logger,
                  n_master: int = 0) -> Tuple[str, Optional[CamSource]]:
        """
        Load the specified rig camera source from the specified experiment session folder, if the requisite video file
        and timestamps file are present for that camera.

        :param session_id: Session ID and location on file system.
        :param cam: The desired camera.
        :param logger: Logger object.
        :param n_master: If > 0, this is the total number of frames (captured + dropped) in the master camera's
            video stream. Otherwise, the method assumes that ``cam`` is the master cam for the session.
        :return: ("", src) if camera source was successfully loaded. ("", None) if session folder lacks a video file
          for the camera specified. (err_msg, None) if an error occurred while reading the video and timestamps files
          for the camera specified.
        """
        # the video file(s) for the cam, if any
        video_files = [p for p in Path(session_id.location).glob('*.mp4') if p.is_file() and (str(cam) in p.name)]
        if len(video_files) == 0:
            # no video file for this cam. If there's a timestamps file, post a warning to the log
            if Path(session_id.location, f"{session_id.session_file_prefix}{str(cam)}_timestamps.txt").is_file():
                logger.warning(f"Warning: Found timestamps but no video for {str(cam)}")
            return "", None

        try:
            video_path: Optional[Path] = None
            if len(video_files) == 1:
                video_path = video_files[0]
            else:
                # choose the file with the largest index. Expect file name format <prefix><cam>-NNNN.mp4'
                largest_idx = -1
                for p in video_files:
                    index_str = p.name[-8:-4]
                    if index_str.isdigit():
                        idx = int(index_str)
                        if idx > largest_idx:
                            largest_idx = idx
                            video_path = p
                if video_path is None:
                    raise Exception(f"Multiple video files for {cam}, but file names lack expected 4-digit index!")

            src = CamSource(session_id, cam, video_path)
            src._load(n_master)
            return "", src
        except Exception as e:
            msg = f"Unable to load video for {cam}: {e}"
            # logger.error(msg)
            return msg, None

    def __init__(self, session_id: RxSessionID, cam: RxCam, video_path: Path):
        """
        **DO NOT USE DIRECTLY**. Call `load_cam()` to create and load a video camera source from a session folder.

        :param session_id: Session ID and location.
        :param cam: The rig camera ID.
        :param video_path: The video source file for that camera.
        """
        self._session_id = session_id
        """ Identifies the recording session, including file system location. """
        self._video_path = video_path
        """ The video source file. """
        self._cam = cam
        """ Camera id/nickname. """
        self._vid_cap: Optional[cv2.VideoCapture] = None
        """ The video. """
        self._curr_frame_num: int = -1
        """ The current frame number in [0, N-1), where N is the total # of captured frames AND any dropped frames. """
        self._total_frames_captured: int = 0
        """ Total number of frames captured -- EXCLUDING any dropped frames. """
        self._dropped_frame_nums: List[int] = list()
        """ Frame numbers corresponding to dropped frames (not captured by video), in ascending order. """
        self._curr_frame: Optional[np.ndarray] = None
        """ The current frame. Will be None if current frame number is invalid OR corresponds to a dropped frame. """
        self._frame_size: Tuple[int, int] = (0, 0)
        """ The frame size in pixels. """
        self._frame_rate: int = 0
        """ The camera's nominal frame rate in Hz. """
        self._frame_count_adjusted_to_master: bool = False
        """ 
        If set, this secondary camera's frame count had to be adjusted to match the master's frame count. Always
        false for the maaster. See ``_load().`` 
        """

    @property
    def id(self) -> RxCam:
        """ The ID/nickname for this camera source. """
        return self._cam

    @property
    def session_id(self) -> RxSessionID:
        """ ID of the session to which this camera source belongs. """
        return self._session_id

    @property
    def frame_size(self) -> Tuple[int, int]:
        """ The video frame size (W, H) in pixels. (0, 0) if source not loaded. """
        return self._frame_size

    @property
    def frame_rate(self) -> int:
        """ The nominal video frame rate in Hz. """
        return self._frame_rate

    @property
    def frame_count(self) -> int:
        """ Total number of frames in the video = total frames captured + number dropped. 0 if source not loaded. """
        return self._total_frames_captured + len(self._dropped_frame_nums)

    @property
    def frame_count_adjusted_to_master(self) -> bool:
        """
        For a non-master camera only: Returns ``True`` if, after adjusting for dropped frames based on the camera's
        frame interval timestamps file alone, the total frame count still does not match the master frame count. Further
        adjustments are made at load time to ensure a match, but it's an indication that it is more likely than not that
        the secondary camera's timeline could be out of sync with the master. Returns ``False`` always for the
        designated master cam. """
        return self._frame_count_adjusted_to_master

    @property
    def num_dropped_frames(self) -> int:
        """ The number of frames dropped on this camera source during the recording session. """
        return len(self._dropped_frame_nums)

    @property
    def dropped_frames(self) -> List[int]:
        """ The list of dropped frame numbers for this camera source, in ascending order. """
        return self._dropped_frame_nums.copy()

    @property
    def curr_frame_number(self) -> int:
        """
        Index (0-based) of the current frame from this video camera source. If not a valid frame number, or if it
        corresponds to a dropped frame, then the current video frame image will be None. -1 if source not loaded.
        """
        return self._curr_frame_num

    @property
    def curr_frame(self) -> Optional[np.ndarray]:
        """
        The current video frame image, preprocessed for display in a PyQtGraph `ImageItem`.

        :return: A 2d Numpy array with dtype=uint8 and shape=(height, width) in RGB ordering. For an "upright" image, Y
            increases downward, X increases rightward, and the origin lies at the top-left corner. Returns None if
            source not loaded, the current frame number is invalid or corresponds to a dropped frame, or an error
            occurred while trying to read the frame.
        """
        return self._curr_frame

    def go_to_frame(self, frame_num) -> Optional[str]:
        """
        Set the current frame number, then read in and pre-process the corresponding video frame image. If the frame
        number specified is invalid or corresponds to a dropped frame, or if an error occurs, the current frame image is
        set to None.
        :param frame_num: Desired frame number
        :return: None if successful. An error description if an error occurred while reading a valid frame.
        """
        if (self._vid_cap is None) or (frame_num == self._curr_frame_num):
            return None
        try:
            self._curr_frame_num = frame_num
            self._curr_frame = None
            captured_frame_num = self._captured_frame_num(frame_num)
            if captured_frame_num >= 0:
                self._vid_cap.set(cv2.CAP_PROP_POS_FRAMES, captured_frame_num)
                ok, frame = self._vid_cap.read()
                if not ok:
                    raise Exception(f"Unable to read frame {captured_frame_num}: {self._video_path.name}")
                try:
                    self._curr_frame = self._process_frame_image(frame)
                except Exception as _:
                    raise Exception(f"Failed to process frame {captured_frame_num}: {self._video_path.name}")
            return None
        except Exception as e:
            return f"Failed to load frame {frame_num} on {self._cam}: {e}"

    def grab_frame_block(self, m: int, n: int, out: np.ndarray) -> int:
        """
        Grab a block of ``n`` consecutive image frames from this video source, starting at frame ``m``. Upon return, the
        current frame will be ``min(n+m-1, f-1)``, where ``f`` is the total number of frames.

        This method is intended for use when processing videos to, eg, predict body part locations in each frame using
        a trained object detection model. As such, any dropped frame in the batch of images retrieved is represented as
        an "all black" frame rather than ``None``.

        DEVNOTE: Using go_to_frame() is very wasteful when retrieving a block of sequential frames, bc that method
        sets the frame number on each call. This forces the video decoder to perform a costly, random-access "seek"
        operation for every single frame, breaking the efficiency of sequential video reading. Instead, the strategy
        here is:
         - Prefill the output with all black frames. That will take care of any "dropped frames" in the block.
         - Skip past any dropped frames at the beginning of the block, then set the frame number of the first
           "captured" frame in the block.
         - Sequentially read in the remaining captured frames, being sure to skip over dropped frames.

        :param m: Starting frame number of block.
        :param n: The number of frames in block
        :param out: A Numpy array to hold the grabbed frames. Use ``np.empty((n, H, W, 3), dtype='ubyte')``, where H is
            the frame height in pixels and W is the frame width. All images are stored in the array in RGB format.
        :return: The number of frames retrieved. Could be less than ``n`` upon reaching file's end. Will be 0 if video
            source is not loaded or the frame block is ill-specified.
        """
        if (self._vid_cap is None) or (n <= 0) or (m >= self.frame_count):
            return 0

        n_retrieved = min(n, self.frame_count - m)
        out[:n_retrieved].fill(0)
        dropped = set(self._dropped_frame_nums)
        block_end = m + n_retrieved

        first_captured = m
        while first_captured < block_end and first_captured in dropped:
            first_captured += 1

        last_frame = None
        if first_captured < block_end:
            captured_frame_num = self._captured_frame_num(first_captured)
            if captured_frame_num < 0:
                raise Exception(f"Failed to map frame {first_captured} into captured video timeline.")
            self._vid_cap.set(cv2.CAP_PROP_POS_FRAMES, captured_frame_num)

            for frame_num in range(first_captured, block_end):
                if frame_num in dropped:
                    continue

                ok, frame = self._vid_cap.read()
                if not ok:
                    raise Exception(
                        f"Unable to read frame {self._captured_frame_num(frame_num)}: {self._video_path.name}"
                    )
                try:
                    out[frame_num - m] = self._process_frame_image(frame)
                    last_frame = out[frame_num - m].copy()
                except Exception as _:
                    raise Exception(f"Failed to process frame {frame_num}: {self._video_path.name}")

        self._curr_frame_num = block_end - 1
        self._curr_frame = last_frame
        return n_retrieved

    @staticmethod
    def _process_frame_image(frame: np.ndarray) -> np.ndarray:
        """
        Convert a captured OpenCV frame from BGR to RGB without extra ``dtype`` conversion.

        DEVNOTE: The Numpy array returned by ``cv2.VideoCapture.read()`` is already ``np.uint8`` ("ubyte") and is
        unchanged by cv2.cvtColor(), so it is redundant to pass it thorough ``skimage.util.img_as_ubyte``, as was
        done previously.
        :param frame: A captured OpenCV frame image in BGR ordering.
        :return The frame converted to RGB ordering required for GUI display, model training, and video analysis. """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _captured_frame_num(self, n: int) -> int:
        """
        Find the frame number in the captured video timeline that corresponds to the specified frame number in the full
        timeline (captured frames + dropped frames).
        :param n: A frame number in [0, N-1) where N = C (total captured) + D (total dropped) frames.
        :return: The corresponding frame index in the captured video timeline [0, C-1), or -1 if the argument is not
        in [0, N-1) OR is a dropped frame.
        """
        frame = -1
        if 0 <= n < self.frame_count:
            if (len(self._dropped_frame_nums) == 0) or (n < self._dropped_frame_nums[0]):
                frame = n
            elif n > self._dropped_frame_nums[-1]:
                frame = n - len(self._dropped_frame_nums)
            elif n not in self._dropped_frame_nums:
                # adjust by subtracting the number of dropped frames that occurred before the requested frame
                for i, val in enumerate(self._dropped_frame_nums):
                    if val > n:
                        frame = n - i
                        break
        return frame

    def _load(self, n_master: int) -> None:
        """
        Open the source file for this camera video source, then read in and pre-process the first video frame. In
        addition, process the associated frame interval timestamps file in order to detect and account for any **dropped
        frames** -- ie, frames that were not captured during the recording session.

        If this is NOT the maseter camera, then ``n_master`` holds the master frame count, ``N``. For any non-master
        camera, after processing the timestamps file to detect dropped frames, its frame count ``M`` (#captured +
        #dropped) must equal ``N``.
          - If ``M < N``, assume that the secondary camera started recording ``(N-M)`` frames late, and adjust by
            inserting ``(N-M)`` additional dropped frames at the beginning of its timeline.
          - If ``M > N``, assume that both cameras started at the same time, but the secondary camera stopped later
            than the master. Adjust the dropped frame list and captured frame count so that, again, M == N.
          - In either case, it is impossible (with the information available to ReachX) to be certain that these
            adjustments will ensure the secondary camera's timeline is in sync with the master's.

        :param n_master: The master camera frame count, ``N``. If 0, assume we're loading the master cam. Otherwise, if
            if the  total frame count (captured + dropped frames detected via the timestamps file) for this camera does
            not equal ``N``, adjust as described above.
        :raises Exception: If unable to open/read video file or associated timestamps file, or process the first frame.
        """
        try:
            self._vid_cap = cv2.VideoCapture(str(self._video_path))
            if not self._vid_cap.isOpened():
                raise Exception(f"Unable to open video file: {self._video_path.name}")
            self._frame_size = (int(self._vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(self._vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._total_frames_captured = int(self._vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._find_dropped_frames()

            # for non-master cameras only: ensure total frame count matches the master
            if n_master > 0:
                self._frame_count_adjusted_to_master = (self.frame_count != n_master)
                if self.frame_count < n_master:
                    # if the total frame count N is less than the master's M, then we need to insert M-N dropped frames
                    # at the beginning. We assume the master cam ALWAYS starts first.
                    n_extra = n_master - self.frame_count
                    for i in range(len(self._dropped_frame_nums)):
                        self._dropped_frame_nums[i] += n_extra
                    while n_extra > 0:
                        n_extra -= 1
                        self._dropped_frame_nums.insert(0, n_extra)
                elif self.frame_count > n_master:
                    # if N > M, then we remove any dropped frames > M-1, and if necessary adjust the captured frame
                    # count so that N == M.
                    while (self.frame_count > n_master and (len(self._dropped_frame_nums) > 0) and
                           (self._dropped_frame_nums[-1] > n_master - 1)):
                        self._dropped_frame_nums.pop(-1)
                    if self.frame_count > n_master:
                        delta = self.frame_count - n_master
                        self._total_frames_captured -= delta

            emsg = self.go_to_frame(0)
            if isinstance(emsg, str):
                raise Exception(f"Failed to read or process frame 0: {self._video_path.name}; error={emsg}")
        except Exception as e:
            if isinstance(self._vid_cap, cv2.VideoCapture):
                self.unload()
            raise e   # error will be logged in load_cam()

    def _find_dropped_frames(self) -> None:
        """
        Helper method for load(). It processes the camera source's timestamps file to determine if any frames were
        dropped during recording.

        The timestamps file, at **<session_folder>/session_file_prefix><cam_name>_timetamps.txt**, is a text file
        containing a list of integers - the frame intervals in nanonseconds over the course of the recording. If there
        were no dropped frames, we'd expect each frame interval ~= 10^9 / R ns, where R is the camera frame rate in Hz.
        Any interval significantly different from this nominal value indicates a dropped frame.

        **Strategy**: Let ARR be the array of frame intervals. ASSUME that ARR[N] is the frame interval separating
        camera frames N and (N+1). Calculate average frame interval mean(ARR). Given the nominal frame rates supported
        in ReachX, choose the corresponding nominal frame interval P that is closest to mean(ARR). Next, find all
        indices N where abs(ARR[N] - P) > P/2. For each such index, the # of consecutive drops would be
        int(round((ARR[N] - P)/P)). Then, at each index N, the corresponding dropped frame number is N + 1 + (the number
        of drops so far); and when more than one frame is dropped consecutively, we have to account for each before
        proceeding to the next index.

        **DEV NOTE**: Note that we use the frame intervals in timestamps.txt to figure out what the nominal frame rate
        is for the camera, rather than rely on a copy of the systemdata.yaml file in the session folder.

        :raises Exception: If timestamps file could not be found/read, or if the number of frame intervals found
          therein is not equal to the number of captured frames.
        """
        p = Path(self._session_id.location,
                 f"{self._session_id.session_file_prefix}{str(self._cam)}_timestamps.txt")
        if not p.is_file():
            raise FileNotFoundError(f"Missing frame interval timestamps file for: {self._cam}")
        try:
            # load frame interval and use to calculate the nominal frame rate
            intervals_ns = np.loadtxt(p, dtype='i4', usecols=(0,))
            if intervals_ns.size != self._total_frames_captured:
                raise Exception(f"timestamps vs captured frames count mismatch: {intervals_ns.size} vs "
                                f"{self._total_frames_captured}")

            rate = int(np.round(1.0e9 / float(np.median(intervals_ns))))
            if rate not in _RX.SUPPORTED_FRAME_RATES:
                raise Exception(f"Unsupported camera frame rate: {rate}")
            self._frame_rate = rate

            # find dropped frames
            nominal_intv = int(0.5 + 1.0e9 / rate)   # in nanosecs
            interval_diffs = np.abs(intervals_ns - nominal_intv)
            drop_indices = np.nonzero(interval_diffs > int(nominal_intv / 2))[0]
            n_consec = np.round(interval_diffs[drop_indices] / nominal_intv).astype(int)

            self._dropped_frame_nums.clear()
            n_drops = 0
            for i in range(drop_indices.size):
                while n_consec[i] > 0:
                    self._dropped_frame_nums.append(int(drop_indices[i] + 1 + n_drops))
                    n_drops += 1
                    n_consec[i] -= 1
        except Exception as e:
            raise Exception(f"Error loading timestamps file {p.name}: {e}")

    def unload(self) -> None:
        """
        Unload this video camera source. **Be sure to call this method to release the underlying resources for
        reading the video file.**
        """
        if isinstance(self._vid_cap, cv2.VideoCapture):
            self._vid_cap.release()
        self._curr_frame_num, self._total_frames_captured, self._curr_frame = -1, 0, None
        self._dropped_frame_nums.clear()


class SessionEventList:
    """ Immutable container encapsulating the list of events recorded during a ReachX experiment session. """
    @staticmethod
    def load_session_events(session_id: RxSessionID, master_cam: CamSource) -> Tuple[str, Optional[SessionEventList]]:
        """
        Load all events recorded during an experiment session from the session events file at:
        ``<session_folder>/<session_file_prefix>events.txt``.

        If the specified "master" camera had any frame drops, correct the event frame numbers accordingly.

        NOTES:
         - The method will now "skip" over any line that does not conform to the format ``event_name frame_num``. Here
           ``event_name`` is a string matching one of the ``RxEvent`` enumerants and ``frame_num`` is the event
           frame number, an integer string.
         - The method ASSUMES events are listed in the text file in chronological order!

        :param session_id: Session ID, which locates session data folder on the file system.
        :param master_cam: The master camera. Event frame numbers recorded in the text file correspond to **captured**
           frames on this camera. If the camera had any frame drops, the frame numbers are corrected accordingly.
        :return: ("", evt_list) if successful, else (error_msg, None)
        """
        evt_path = Path(session_id.location, f"{session_id.session_file_prefix}events.txt")
        if not evt_path.is_file():
            return f"Events file {evt_path.name} not found", None

        events: List[RxEvent] = []
        frames: List[int] = []
        with open(evt_path) as f:
            line_no = 0
            for line in f:
                tokens = line.split()
                if len(tokens) > 0:
                    try:
                        assert len(tokens) == 2
                        events.append(RxEvent(tokens[0]))
                        frames.append(int(tokens[1]))
                    except Exception:
                        get_application_logger().warning(f"Unexpected content at line {line_no} in session events "
                                                         f"file: '{line.strip()}'")
                line_no += 1
        return "", SessionEventList(events, frames, master_cam.dropped_frames)

    def __init__(self, events: List[RxEvent], frames: List[int], drops: List[int]) -> None:
        """
        **Do not use**. Instead load events list from the relevant text file using `load_session_events()`.

        :param events: The events array, assumed to be in chronological order by frame number.
        :param frames: Array containing the corresponding event frame numbers.
        :param drops: List of dropped frame numbers on the master cam, in chronological order. It is used to correct
            the event frame numbers found in the session events file, which are based on the master cam timeline.
        """
        self._events = events
        """ List of events in chronological order. """
        self._frames = frames
        """ The corresponding event frame numbers. """

        self._correct_for_dropped_frames(drops)

    def _correct_for_dropped_frames(self, drops: List[int]) -> None:
        """
        Helper method for __init__(). Corrects the event frame numbers.
        :param drops: List of dropped frame numbers on the master cam, in chronological order.
        """
        if len(drops) == 0:
            return

        i_start = 0
        frame_array = np.array(self._frames)
        for drop_idx, drop_frame_num in enumerate(drops):
            n = np.count_nonzero(frame_array <= drop_frame_num)
            if n == 0:
                break
            frame_array[i_start:n] += drop_idx
            i_start = n
            if i_start >= frame_array.size:
                break
        self._frames.clear()
        self._frames = frame_array.tolist()

    @property
    def count(self) -> int:
        """ The total number of events recorded. """
        return len(self._events)

    def get(self, index: int) -> Tuple[RxEvent, int]:
        """
        Get the i-th event in this session event list.
        :param index: 0-based index into list
        :return: (event type, frame number)
        :raises IndexError if specified index is invalid
        """
        return self._events[index], self._frames[index]

    def index_of_event_frame_near(self, frame: int) -> int:
        """
        Find the index position of an event that occurred on or immediately after the frame number specified
        :param frame: Frame number.
        :return: 0-based index position of an event; -1 if event list empty.
        """
        out = -1
        if len(self._frames) > 0:
            if frame >= self._frames[-1]:
                out = self.count - 1
            else:
                out = int(np.searchsorted(self._frames, frame, side='right'))
                if out > 0 and self._frames[out - 1] == frame:
                    out -= 1
        return out

    def find_next_occurrence(self, evt: RxEvent, starting_frame: int) -> int:
        """
        Find next occurrence of the specified event type, starting from the specified frame number.
        :param evt: The event type
        :param starting_frame: Find next occurrence AFTER this frame.
        :return: Frame number for next occurrence of the specified event type; -1 if there are no more.
        """
        idx = np.searchsorted(self._frames, starting_frame, side='right')
        try:
            # idx = (idx + 1) if self._frames[idx] == starting_frame else idx
            found = self._events.index(evt, idx)
        except (ValueError, IndexError):
            found = -1
        return self._frames[found] if found > -1 else -1

    def find_prev_occurrence(self, evt: RxEvent, starting_frame: int) -> int:
        """
        Find previous occurrence of the specified event type, starting from the specified frame number.

        :param evt: The event type
        :param starting_frame: Find previous occurrence BEFORE this frame.
        :return: Frame number for previous occurrence of the specified event type; -1 if there are no more.
        """
        idx = int(np.searchsorted(self._frames, starting_frame, side='left'))
        if idx == 0:
            return -1
        try:
            idx = (idx - 1) if self._frames[idx] == starting_frame else idx
            found = len(self._events) - 1 - self._events[::-1].index(evt, len(self._events) - 1 - idx)
        except (ValueError, IndexError):
            found = -1
        return self._frames[found] if found > -1 else -1

    def event_count(self, evt: Optional[RxEvent]) -> int:
        """
        Get the total number of occurrences of the specified event.
        :param evt: The event type.
        :return: Observed number of occurrences, possibly 0. If `evt` is None, returns total # of all events recorded.
        """
        return self.count if evt is None else sum(1 for t in self._events if t == evt)

    def get_all_occurrences_of(self, evt: RxEvent) -> List[int]:
        """
        Get all occurrences of the specified event
        :param evt: The event type.
        :return: List of frame numbers (in chrono order) when the specified event occurred during the session.
        """
        return [self._frames[i] for i in range(len(self._events)) if (self._events[i] == evt)]

    def get_all_events_between(self, start: int, end: int) -> List[Tuple[RxEvent, int]]:
        """
        Get all session events between the specified frame numbers (inclusive).
        :param start: The starting frame number.
        :param end: The ending frame number
        :return: List of all events that occurred in the specified time frame as 2-tuples (event, frame num). Empty
            if there were no events in that time frame (or time frame invalid).
        """
        idx0 = np.searchsorted(self._frames, max(0, start), side='left')
        idx1 = np.searchsorted(self._frames, min(end, self._frames[-1]), side='right')
        if (idx0 == idx1) and (start <= self._frames[idx0] <= end):
            return [(self._events[idx0], self._frames[idx0])]
        elif idx0 < idx1:
            return [(self._events[i], self._frames[i]) for i in range(idx0, idx1)]
        return []


class SessionScorer(NamedTuple):
    """
    A session 'scorer', ie, a particular trained ReachX model iteration that was used to predict body part locations
    during a session and, from those predictions, find and categorize reach segments during the session.

    The session scorer name is derived from the model instance used to analyze the session and serves as the name of
    the session subfolder containing the analysis results. It has the following format:
        Rx_<network name>_<model>_<N>_snapshot-<S>
    where:
     - ``<network name>`` = "resnet50" or "resnet152"
     - ``<model>`` = model project name (until v0.6.0), or full model project folder name (v0.6.0 and later).
     - ``<N>`` = integer indicating which model iteration was used.
     - ``<S>`` = integer indication which model snapshot index was used.
    """
    sesh_id: RxSessionID
    """ The session that was 'scored'. """
    model: str
    """ 
    Model name. Prior to v0.6.0, this was set to the model project name only. As of v0.6.0, it is set to the full model
    project folder name. The folder name has the form ``<project name>-YYYYMMDD`` for legacy-cam models, where 
    <project name> is the model project name. and YYYYMMDD is the model creation date. For fixed-cam models, the
    folder name is ``<project name>-YYYYMMDD-fix``.
    """
    iter_num: int
    """ The model iteration number. A model project may have any number of model iterations. """
    net_type: RxNetworkType
    """ Pretrained residual neural net architecture on which trained model iteration is based. """
    snapshot: int
    """ The particular model iteration snapshot used to analyze the session. """
    prefix: str
    """ The scorer prefix, which serves as the name of the session subfolder containing analysis results. """

    @property
    def short_name(self) -> str:
        """
        A short name for the scorer, reflecting only the model name and iteration number. For use in GUI.

        Note that scorer results generated with v0.6.0 or later use the full model project folder name in the scorer
        name; prior to v0.6.0, only the model project name was included.
        """
        return f"{self.model}_{str(self.iter_num)}"

    @property
    def results_folder(self) -> Path:
        """ Full path to the folder containing this scorer's analysis results for the session. """
        return Path(self.sesh_id.location, self.prefix)

    @staticmethod
    def find_scorers(sesh_id: RxSessionID) -> List[SessionScorer]:
        """
        Traverse specified session's backing store for analysis results from one or more "scorers", and return the
        list of all scorers found.

        Scorer name format is ``Rx_<nnet>_<model_name>_<iter_num>_snapshot-<snapshot_idx>``, where:
         - ``<nnet>`` = "resnet50" or "resnet152".
         - ``<model_name>`` = The model name. **See below.**
         - ``<iter_num>`` = integer string identifying which model iteration was used.
         - ``<snapshot_idx>`` = integer string identifying which snapshot index was used.

        The format of ``<model_name>`` changed in v0.6.0. For backwards compatibility, both forms are accepted:
         - Prior to v0.6.0: Just the model project name, assigned by user when it was created.
         - v0.6.0 and on: The full model folder name: ``<model_project>-YYYYMMDD`` for legacy-cam models and
           ``<model_project>-YYYYMMDD-fix`` for fixed-cam models, where ``YYYYMMDD`` is the model creation date.

        :param sesh_id: The session identifier.
        :return: List of scorers for the session; empty if none were found.
        """
        out = list()
        for p in sesh_id.location.iterdir():
            if p.is_dir() and p.name.startswith("Rx_"):
                try:
                    _, nnet, model, iter_str, snapshot = p.name.split("_")
                    if nnet not in ["resnet50", "resnet152"]:
                        raise Exception()
                    # two forms of model name accepted: <proj> or <proj>-<YYYYMMDD>[-fix]
                    if model.find("-") > -1:
                        tokens = model.split("-")
                        if len(tokens) < 2 or len(tokens) > 3 or not all([t.isalnum() for t in tokens]):
                            raise Exception()
                    elif not model.isalnum():
                        raise Exception()
                    net_type = RxNetworkType.RESNET_50 if nnet == "resnet50" else RxNetworkType.RESNET_152
                    iter_num = int(iter_str)
                    snapshot_idx = int(snapshot.split('-')[1])
                    out.append(SessionScorer(sesh_id, model, iter_num, net_type, snapshot_idx, p.name))
                except Exception:
                    continue
        return out


class SegmentationResultRecord(NamedTuple):
    """A selectable scored-reach result for a session scorer."""
    result_id: str
    display_name: str
    kind: str
    detected_reaches_path: Path
    result_folder: Path
    modified_time: float


class _ReachOp:
    """
    Encapsulation of an operation on a session's set of curated reaches -- for the purpose of undoing the last
    curated reach operation on that session.
    """
    ADD: int = 0
    DEL: int = 1
    REPLACE: int = 2
    DESC: List[str] = ["add reach", "delete reach", "modify reach"]

    def __init__(self, op_type: int, reach_1: RxReachSegment, reach_2: Optional[RxReachSegment] = None) -> None:
        """
        Construct an undoable operation on a session's set of curated reaches.
        :param op_type: Op type -- add/delete/replace.
        :param reach_1: The reach segment added, deleted, or replaced.
        :param reach_2: For a replacemeent op, this is the replacement reach segment; else ignored.
        """
        assert op_type in [self.ADD, self.DEL, self.REPLACE]
        assert isinstance(reach_1, RxReachSegment)
        if op_type == self.REPLACE:
            assert isinstance(reach_2, RxReachSegment)
        self.op_type = op_type
        """ The operation performed. """
        self.reach_1 = reach_1
        """ The reach segment that was added, deleted, or replaced. """
        self.reach_2 = reach_2
        """ For a replace op only, this is the replacement reach. """

    @property
    def undo_description(self) -> str:
        """ Very simple "undo description". """
        return f"Undo: {self.DESC[self.op_type]}"


class SessionManager(QObject):
    _TIMER_INTV_MS: int = 100
    """ Fixed interval for frame updates during playback. """
    PLAYBACK_SPEEDS: List[int] = [-1500, -750, -150, -75, -30, -10, 0, 10, 30, 75, 150, 750, 1500]
    """ Camera video playback speeds in #frames per second. Negative = reverse playbback. 0 = stopped. """
    ZERO_SPEED_IDX: int = 6
    """ Index of zero speed in supported list of video playback speeds. """
    frame_ready: Signal = Signal()
    """ A new video frame is ready for display from all available cameras. """
    playback_started: Signal = Signal()
    """ Video playback has started. """
    playback_stopped: Signal = Signal()
    """ Video playback has stopped. """
    reaches_changed: Signal = Signal()
    """ The set of user-curated reach segments defined on the current session has changed. """
    scorer_changed: Signal = Signal()
    """ 
    The current scorer for analysis results (predicted body part locations, model-driven reach segmentation) has
    changed, or its results have changed (re-analyzed with same scorer). GUI components that display analysis results 
    for the current scorer should be updated accordingly.
    """
    scorer_list_changed: Signal = Signal()
    """ 
    The list of scorers for the current session has changed. Any given session may be analyzed by multiple
    different models (aka, scorers). 
    """
    segmentation_result_changed: Signal = Signal()
    """
    The selected scored-reach result for the current scorer has changed.
    """

    def __init__(self, suppress_log: bool = False, auto_save: bool = True):
        """
         Create the session manager in a startup state, with no session loaded.
        :param suppress_log: If True, the session manager suppresses messages to application log. Default is False.
        :param auto_save: If True, the session manager enables a periodic auto-save feature to persist the
            current session's curated reach set if changes have been made. Default is True.
        """
        super().__init__()

        self._sesh_id: Optional[RxSessionID] = None
        """ Identifier for the currently loaded experiment session, or None if no session loaded"""
        self._videos: Dict[RxCam, CamSource] = dict()
        """ Available rig camera videos, keyed by cam nickname. Empty if no session loaded. """
        self._master_cam: Optional[RxCam] = None
        """ The master cam. Frame numbers in the session events file correspond to this camera source. """
        self._total_frames = 0
        """ 
        The session duration in frames. By convention, this is the frame count for the master cam. **Includes any
        detected frame drops.**
        """

        self._logger: logging.Logger = get_null_logger() if suppress_log else get_application_logger()
        """ The logger for the session manager. If logging is suppressed, this is set to a null logger. """

        self._events: Optional[SessionEventList] = None
        """ Events recorded during the session, corrected for frame drops on master cam; None if no session loaded. """

        self._scorers: List[SessionScorer] = list()
        """ 
        List of "scorers" -- ie, a particular trained iteration of a body part detection model -- that have been used to
        predict body part locations and find reach segments during this session. Each scorer is identified by the
        subfolder within the session folder containing the analysis results from that scorer. Will be empty if the
        session has not been analyzed.
        """
        self._curr_scorer_idx: int = -1
        """ 
        Index locating the scorer for which analysis results (detected hand and pellet locations, autogenerated
        reach segments) are available. Will be -1 if the session has not been analyzed.
        """
        self._scored_markers = np.empty((0, 5))
        """
        If session has been analyzed by a body part detection model, this array contains all detected body part
        markers with a confidence score of 0.9 or better. Each row in the array represents a distinct marker on either
        the front or side cam: ``[frame_num, cam, bp_index, x, y, likelihood * 10000]``. Array is sorted by frame 
        number, and data type is ``int32`` (hence the scaling of ``likelihood`` parameter).
        """
        self._scored_reaches: List[RxReachSegment] = list()
        """
        If session has been analyzed by a body part detection model AND algorithmic reach segmentation performed, this
        is the list of reach segments detected. List is sorted in ascending order by frame number.
        """
        self._segmentation_results: List[SegmentationResultRecord] = list()
        """ Selectable scored-reach result sets for the current scorer. """
        self._curr_segmentation_result_idx: int = -1
        """ Index of the selected scored-reach result in ``_segmentation_results``. """

        self._curated_reaches: List[RxReachSegment] = list()
        """ 
        The set of user-curated reach segments for the current session; empty if no session loaded or if there are no
        user-curated reach segments defined.
        """
        self._undo_history: List[_ReachOp] = list()
        """ 
        Undo history for the set of recent operations on the user-curated set of reaches for the current session -- 
        most recent operation is at the end of the list. The history is not persisted; it is reset whenever the current 
        session is unloaded, or when a wholesale operation (discard all, add all scored reaches) is performed.
        """
        self._auto_save_timer: Optional[QTimer] = None
        """ 
        A 60-sec timer that is started whenever the set of curated reaches for the currently loaded session is
        altered for the first time since the last save. The timer is stopped each time the curated reach set is saved.
        Will be None if auto save feature is not enabled
        """

        self._curr_frame_num = -1
        """ The current frame number available for display. -1 if no session loaded. """
        self._playback_speed_index: int = self.ZERO_SPEED_IDX
        """ Current playback speed: Index into list of available speeds. Playback stopped (zero speed) initially. """
        self._playback_delta: int = 0
        """ The #frames to advance (positive) or rewind (negative) per playback timer update interval. """
        self._playback_timer: QTimer = QTimer()
        """ Playback timer. """

        self._playback_timer.setInterval(SessionManager._TIMER_INTV_MS)
        self._playback_timer.timeout.connect(self._on_playback_timer_timeout)

        if auto_save:
            self._auto_save_timer = QTimer()
            self._auto_save_timer.setInterval(60000)
            # noinspection PyUnresolvedReferences
            self._auto_save_timer.timeout.connect(lambda: self._auto_saver_update(0))

    @property
    def id(self) -> Optional[RxSessionID]:
        """ Identifier for current session, or None if no session is currently loaded. """
        return self._sesh_id

    @property
    def is_fixed_cam(self) -> Optional[bool]:
        """ ``True`` for a fixed-cam, ``False`` for a legacy-cam session, or None if no session loaded. """
        return None if (self.id is None) else self.id.is_fixed_cam

    @property
    def session_loaded(self) -> bool:
        """ Is a session currently loaded? """
        return isinstance(self._sesh_id, RxSessionID)

    @property
    def available_camera_ids(self) -> List[RxCam]:
        """ The IDs/nicknames of all rig cameras recorded during this session. """
        return [cam for cam in self._videos]

    def get_camera(self, cam: RxCam) -> Optional[CamSource]:
        """
        Get the specified camera source.
        :param cam Camera ID/nickname.
        :return: The corresponding camera source, or None if that camera was not recorded during this session.
        """
        return self._videos[cam] if cam in self._videos else None

    @property
    def master_camera_id(self) -> Optional[RxCam]:
        """ ID of the master camera for the session. Event frame numbers correspond to this camera. """
        return self._master_cam

    @property
    def total_frames(self) -> int:
        """
        Total number of frames recorded during the session. By conventions, this is the frame count for the master cam,
        **and it includes any dropped frames**.
        :return: Frame count for the loaded session. Returns 0 if session not loaded.
        """
        return self._total_frames

    @property
    def master_frame_rate(self) -> int:
        """ The nominal recorded frame rate on the master camera for this session. Returns 0 if no session loaded. """
        if self._master_cam is None:
            return 0
        return self._videos[self._master_cam].frame_rate

    @property
    def events(self) -> Optional[SessionEventList]:
        """ The list of events recorded during the session. None if no session is loaded. """
        return self._events

    @property
    def scorers(self) -> List[SessionScorer]:
        """ List of scorers (model iterations) for which analysis results are available for this session. """
        return self._scorers.copy()

    @property
    def current_scorer(self) -> Optional[SessionScorer]:
        """ The current scorer (model iteration) for this session. None if this session has yet to be analyzed. """
        return None if self._curr_scorer_idx < 0 else self._scorers[self._curr_scorer_idx]

    @property
    def current_scorer_index(self) -> int:
        """ Index of the current scorer in the scorers list for this session. Will be -1 if no scorers available. """
        return self._curr_scorer_idx

    @property
    def segmentation_results(self) -> List[SegmentationResultRecord]:
        """ Available scored-reach results for the current scorer. """
        return self._segmentation_results.copy()

    @property
    def current_segmentation_result_index(self) -> int:
        """ Index of the selected scored-reach result, or -1 if none are available. """
        return self._curr_segmentation_result_idx

    @property
    def current_segmentation_result(self) -> Optional[SegmentationResultRecord]:
        """ The selected scored-reach result for the current scorer. """
        if 0 <= self._curr_segmentation_result_idx < len(self._segmentation_results):
            return self._segmentation_results[self._curr_segmentation_result_idx]
        return None

    def change_current_scorer(self, idx: int) -> None:
        """
        Change this session's current scorer (model iteration). ``SessionManager`` serves analysis results (predicted
        body part locations, model-driven reach segmentation results) only for the current scorer. Call this method to
        view results for a different scorer.

        If the current scorer is changed, the ``scorer_changed`` signal is emitted.
        :param idx: The 0-based index selecting a scorer from the list of available scorers for this session. If there
            are no scorers, or if the index is invalid or points to the current scorer already, no action is taken.
        """
        if (idx == self._curr_scorer_idx) or not (0 <= idx < len(self._scorers)):
            return

        self._curr_scorer_idx = idx
        self._scored_markers = SessionManager._load_detected_markers_for(self._scorers[idx])
        self._reload_segmentation_results_for_current_scorer()
        QTimer.singleShot(10, lambda: self.scorer_changed.emit())
        QTimer.singleShot(10, lambda: self.segmentation_result_changed.emit())

    def change_current_segmentation_result(self, idx: int) -> None:
        """
        Change which scored-reach result is exposed for the current scorer.
        :param idx: The 0-based index selecting a result from ``segmentation_results``.
        """
        if (idx == self._curr_segmentation_result_idx) or not (0 <= idx < len(self._segmentation_results)):
            return
        self._curr_segmentation_result_idx = idx
        self._scored_reaches = SessionManager._load_detected_reaches_from_path(
            self._segmentation_results[idx].detected_reaches_path)
        QTimer.singleShot(10, lambda: self.segmentation_result_changed.emit())

    def on_session_analyzed(self, affected_sessions: List[RxSessionID]) -> None:
        """
        Invoked when a long-running job has just finished analyzing a session with a body part detection model. If the
        current session was the subject of the analysis:
         - Check if the list of available scorers has changed. If so, reload the scorer list, preserving the identity
           of the current scorer (if there was one). Notify the GUI with the ``scorer_list_changed`` signal.
         - Then, regardless, for simplicity's sake we go ahead and reload the detected markers and reach segments for
           the current scorer (they may not have changed if session was analyzed with a different scorer) and emit the
           ``scorer_changed`` signal.

        :param affected_sessions: List of session(s) analyzed.
        """
        if self.id in affected_sessions:
            scorers = SessionScorer.find_scorers(self.id)
            scorers_changed = False
            if len(scorers) != len(self._scorers):
                scorers_changed = True
                prev_scorer = self.current_scorer
                self._scorers.clear()
                self._scorers.extend(scorers)

                self._curr_scorer_idx = -1
                if len(scorers) > 0:
                    if prev_scorer is None:
                        self._curr_scorer_idx = 0
                    else:
                        self._curr_scorer_idx = -1
                        for i, scorer in enumerate(self._scorers):
                            if scorer == prev_scorer:
                                self._curr_scorer_idx = i
                                break
                        if self._curr_scorer_idx < 0:
                            self._curr_scorer_idx = 0

            if scorers_changed:
                QTimer.singleShot(10, lambda: self.scorer_list_changed.emit())

            self._scored_reaches.clear()
            self._segmentation_results.clear()
            self._curr_segmentation_result_idx = -1
            self._scored_markers = np.empty((0, 5), dtype=np.int32)
            if self.current_scorer is not None:
                self._reload_segmentation_results_for_current_scorer()
                self._scored_markers = SessionManager._load_detected_markers_for(self.current_scorer)
            QTimer.singleShot(10, lambda: self.scorer_changed.emit())
            QTimer.singleShot(10, lambda: self.segmentation_result_changed.emit())

    def on_reach_segmentation_done(self, affected_sessions: List[RxSessionID], select_latest: bool = False) -> None:
        """
        Invoked when a long-running reach segmentation job has finished. If the current session is among the sessions
        processed, reload the session's model-driven reach segments using the current scorer. If there's a change, the
        ``scorer_changed`` signal is emitted to ensure relevant GUI elements are updated appropiately.
        :param affected_sessions: List of session() for which reach segmentation was performed.
        """
        if (self.id in affected_sessions) and (self.current_scorer is not None):
            old_result_ids = [r.result_id for r in self._segmentation_results]
            old_reaches = self._scored_reaches.copy()
            self._reload_segmentation_results_for_current_scorer(select_latest=select_latest)
            new_result_ids = [r.result_id for r in self._segmentation_results]
            if old_result_ids != new_result_ids or old_reaches != self._scored_reaches:
                QTimer.singleShot(10, lambda: self.segmentation_result_changed.emit())

    @property
    def current_frame_num(self) -> int:
        """ The current frame number for video playback purposes. -1 if no session is loaded. """
        return self._curr_frame_num

    @property
    def current_elapsed_time(self) -> float:
        """ The elapsed time in seconds corresponding to the current frame number. """
        t = 0.0
        if self._curr_frame_num > 0:
            t = self._curr_frame_num / self._videos[self._master_cam].frame_rate
        return t

    def current_frame(self, cam: RxCam) -> Optional[np.ndarray]:
        """
        Get the current frame for the specified video camera.
        :param cam: The rig camera source
        :return: A uint8 Numpy array holding the image for the current frame, or None if camera source unavailable,
        current frame correponds to a dropped frame on the specified cam, or session not loaded.
        """
        return self._videos[cam].curr_frame if cam in self._videos else None

    def predicted_marker_locations_for_current_frame(
            self, cam: RxCam, hand_pellet_only: bool = True) -> Dict[RxBodyPart, Tuple[int, int]]:
        """
        Get the predicted locations of detected body parts on the current frame of the specified camera video based on
        prior analysis of the session's videos with a trained body part detection model. For the animal's left and
        right hands (only right hand applies for a legacy-cam session), the predicted location is returned for the hand
        pose with this highest confidence score; this location is keyed by the special composite marker
        ``RxBodyPart.R_HAND`` or ``RxBodyPart.L_HAND``.

        :param cam: The rig camera source.
        :param hand_pellet_only: If True, only returns the predicated locations of the right-hand, left-hand (fixed-cam
            session only), and pellet (if available)
        :return: Dictionary containing the predicted body part locations (x_pix, y_pix), keyed by body part identifier.
            Only includes body parts for which a prediction is available. Will be empty if no session is loaded, or the
            current session is yet to be analyzed.
        """
        out = dict()
        if self.session_loaded and (self.current_scorer_index > -1):
            fixed = cam.is_fixed_cam
            m: np.ndarray = self._scored_markers
            m = m[m[:, 0] == self._curr_frame_num]
            m = m[m[:, 1] == cam.value]
            pellet_row = m[m[:, 2] == RxBodyPart.PELLET.value]
            if pellet_row.shape[0] == 1:
                out[RxBodyPart.PELLET] = (int(pellet_row[0, 3]), int(pellet_row[0, 4]))

            r_hand_values = [part.value for part in RxBodyPart.fixed_cam_right_hand_poses()] \
                if fixed else [part.value for part in RxBodyPart.hand_body_parts_on(cam)]
            r_hand_rows = m[np.isin(m[:, 2], r_hand_values)]
            if r_hand_rows.shape[0] == 1:
                out[RxBodyPart.R_HAND] = (int(r_hand_rows[0, 3]), int(r_hand_rows[0, 4]))
            elif r_hand_rows.shape[0] > 1:
                likeliest_pose = r_hand_rows[np.argmax(r_hand_rows[:, 5])]
                out[RxBodyPart.R_HAND] = (int(likeliest_pose[3]), int(likeliest_pose[4]))

            if fixed:
                l_hand_values = [part.value for part in RxBodyPart.fixed_cam_left_hand_poses()]
                l_hand_rows = m[np.isin(m[:, 2], l_hand_values)]
                if l_hand_rows.shape[0] == 1:
                    out[RxBodyPart.L_HAND] = (int(l_hand_rows[0, 3]), int(l_hand_rows[0, 4]))
                elif l_hand_rows.shape[0] > 1:
                    likeliest_pose = l_hand_rows[np.argmax(l_hand_rows[:, 5])]
                    out[RxBodyPart.L_HAND] = (int(likeliest_pose[3]), int(likeliest_pose[4]))

            if not hand_pellet_only:
                other_parts = [part for part in RxBodyPart.markable_body_parts_on(cam)
                               if (part != RxBodyPart.PELLET) and not part.is_hand()]
                for part in other_parts:
                    part_row = m[m[:, 2] == part.value]
                    if part_row.shape[0] == 1:
                        out[part] = (int(part_row[0, 3]), int(part_row[0, 4]))

        return out

    def load_hand_pellet_trajectories_for_current_scorer(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the hand and pellet trajectories as computed via the session's current scorer (if any)

        Each of the two trajectories is a Numpy array np.array((N, 4), dtype=np.float32), where N is the number of
        frames in the recorded session, and each row = (x, y, z, S). Here (x,y,z) is the 3D location and S is the
        instantaneous speed of the hand or pellet at the frame corresponding to that row.

        Each column vector is the low-pass filtered version of the raw x/y/z/speed vectors. For both fixed- and
        legacy-cam sessions, the "hand" refers to the animal's right hand.

        :return: A 2-tuple (hand, pellet), where each element is a Numpy array as described. If either trajectory
            is unavailable or could not be loaded, it is set to None. If there is no current scorer, returns
            (None, None).
        """
        scorer = self.current_scorer
        if scorer is None:
            return None, None

        hand_traj: Optional[np.ndarray] = None
        pellet_traj: Optional[np.ndarray] = None
        if self.is_fixed_cam:
            traj_path = Path(scorer.results_folder, "trajectories.npz")
            try:
                with np.load(traj_path) as data:
                    hand_traj = data[str(RxBodyPart.R_HAND)]
                    hand_traj = hand_traj[:, [FixedCamTrajCol.X, FixedCamTrajCol.Y, FixedCamTrajCol.Z,
                                              FixedCamTrajCol.SPEED_FILT]]
                    pellet_traj = data[str(RxBodyPart.PELLET)]
                    pellet_traj = pellet_traj[:, [FixedCamTrajCol.X, FixedCamTrajCol.Y, FixedCamTrajCol.Z,
                                                  FixedCamTrajCol.SPEED_FILT]]
            except Exception as exc:
                get_application_logger().warning(f"Failed to load trajectories from {traj_path}: {exc}")
        else:
            hand_path = Path(scorer.results_folder, "hand.npy")
            if hand_path.is_file():
                try:
                    hand_traj = np.load(hand_path)
                    hand_traj = hand_traj[:, [TrajCol.X_FILT, TrajCol.Y_FILT, TrajCol.Z_FILT, TrajCol.SPEED_FILT]]
                except Exception as exc:
                    get_application_logger().warning(f"Failed to load {hand_path}: {exc}")

            pellet_path = Path(scorer.results_folder, "pellet.npy")
            if pellet_path.is_file():
                try:
                    pellet_traj = np.load(pellet_path)
                    pellet_traj = pellet_traj[:, [TrajCol.X_FILT, TrajCol.Y_FILT, TrajCol.Z_FILT, TrajCol.SPEED_FILT]]
                except Exception as exc:
                    get_application_logger().warning(f"Failed to load {pellet_path}: {exc}")

        return hand_traj, pellet_traj

    def save_current_frame_to_png(self, cam: RxCam, dst: Path) -> bool:
        """
         Save the current frame image from the specified camera source to the specified file destination.
        :param cam: The rig camera source
        :param dst: The destination file name.
        :return: True if successful; False if write failed or frame image not available for specified camera (eg, the
            frame was dropped on that camera).
        """
        if (cam in self._videos) and (self._videos[cam].curr_frame is not None):
            return cv2.imwrite(str(dst.absolute()), cv2.cvtColor(self._videos[cam].curr_frame, cv2.COLOR_RGB2BGR))
        return False

    def was_current_frame_dropped_on(self, cam: RxCam) -> bool:
        """
        Was the current frame dropped on the specified video camera?
        :param cam: The rig camera source
        :return: True if video is available from the specified source, but the current frame was dropped on that cam.
            Returns False if no session loaded or video was not recorded on that cam.
        """
        return (self._curr_frame_num in self._videos[cam].dropped_frames) if (cam in self._videos) else False

    def go_to_frame(self, frame_num: int) -> None:
        """
        Update all available camera sources to move to the frame number specified. No action taken if no session is
        loaded.
        :param frame_num: Desired frame index. Corrected if out of range: [0 .. n_frames).
        """
        if self.session_loaded:
            n = self.total_frames
            frame_num = 0 if frame_num < 0 else (n - 1 if frame_num >= n else frame_num)
            self._curr_frame_num = frame_num
            for cam in self._videos:
                emsg = self._videos[cam].go_to_frame(frame_num)
                if isinstance(emsg, str):
                    self._logger.warning(f"Failed to load frame {frame_num} on {cam}: {emsg}")
            QTimer.singleShot(10, lambda: self.frame_ready.emit())

    def get_scored_reaches(self) -> List[RxReachSegment]:
        """
        Get the list of reach segments auto-detected by the current session scorer for the current session. These are
        NOT the user-curated list of reaches!
        :return: The list of scorer-generated reach segments. Will be empty if no session loaded, current session has
            not been analyzed yet (no scorer defined), or if scorer found no reach segments.
        """
        return self._scored_reaches.copy()

    def get_scored_reach_containing_frame(self, frame: int) -> Optional[RxReachSegment]:
        """
        Get the scorer-generated reach segment, if any, that contains the specified video frame.
        :param frame: A frame number in the master camera timeline.
        :return: The scored reach segment containing the frame specified, or None if not found.
        """
        found: int = (
            next((i for i, reach in enumerate(self._scored_reaches) if 0 <= frame - reach.frame <= reach.dur), -1))
        return None if found < 0 else self._scored_reaches[found]

    def get_all_scored_reaches_in(self, t0: int, t1: int) -> List[RxReachSegment]:
        """
        Get all reach segments auto-detected by the current session scorer that fully or partially overlap the specified
        frame interval in the current session.
        :param t0: Starting frame number for interval of interest.
        :param t1: Ending frame number for interval of interest.
        :return: List of all scorer-generated reach segments inside or overlapping [t0..t1]. Will be empty if no
            session loaded, if session is yet to be analyzed (no scorer defined), or if scorer found no reach segments.
        """
        return SessionManager.get_all_reaches_in_span(self._scored_reaches, max(0, t0), min(self._total_frames-1, t1))

    def get_curated_reaches(self) -> List[RxReachSegment]:
        """
        Get the list of user-curated reach segments for the current session.
        :return: A copy of the curated reach segment list, in chronological order.
        """
        return self._curated_reaches.copy()

    def get_curated_reach_containing_frame(self, frame: int) -> Optional[RxReachSegment]:
        """
        Get the curated reach segment, if any, that contains the specified video frame.
        :param frame: A frame number in the master camera timeline.
        :return: The curated reach segment containing the frame specified, or None if not found.
        """
        found: int = (
            next((i for i, reach in enumerate(self._curated_reaches) if 0 <= frame - reach.frame <= reach.dur), -1))
        return None if found < 0 else self._curated_reaches[found]

    def get_all_curated_reaches_in(self, t0: int, t1: int) -> List[RxReachSegment]:
        """
        Get all curated reach segments that overlap fully or partially with the specified frame interval in the
        current session.
        :param t0: Starting frame number for interval of interest.
        :param t1: Ending frame number for interval of interest.
        :return: List of all reach segments inside or overlapping [t0..t1].
        """
        return SessionManager.get_all_reaches_in_span(self._curated_reaches, max(0, t0), min(self._total_frames-1, t1))

    @staticmethod
    def get_all_reaches_in_span(segments: List[RxReachSegment], t0: int, t1: int) -> List[RxReachSegment]:
        """
        Extract all reach segments in the specified list that overap fully or partially with the specified interval.

        :param segments: List of reach segments **IN CHRONOLOGICAL ORDER**.
        :param t0: Starting frame number for interval of interest.
        :param t1: Ending frame number for interval of interest.
        :return: Returns all segments overlapping the specified time interval. If interval is empty (``t0 > t1``), list
            will be empty. If ``t0==t1``, returns all segments that contain ``t0``.
        """
        out: List[RxReachSegment] = list()
        if len(segments) == 0 or (t0 > t1):
            return out

        if t0 == t1:
            return [seg for seg in segments if seg.frame <= t0 <= seg.end_frame]

        # start at index of first segment that starts before T0. If no such segment, start with the first one
        after = next((i for i, r in enumerate(segments) if r.frame > t0), -1)
        idx = after - 1 if after > 0 else 0

        # append each segment that overlaps [T0 T1] until we reach a segment that starts after T1. THIS WORKS
        # BECAUSE the segments are in chronological order!
        while idx < len(segments):
            start = segments[idx].frame
            end = segments[idx].dur + start
            if start >= t1:
                break
            elif end > t0:
                out.append(segments[idx])
            idx += 1
        return out

    def add_scored_reaches_to_curated_set(self) -> None:
        """
        Add the current set of scored reaches to the session's curated set -- but skip any scored reach that overlaps
        an existing reach in the curated set. If the curated set is changed as a result, the ``reaches_changed`` signal
        is emitted.
        """
        n_before = len(self._curated_reaches)
        if len(self._curated_reaches) == 0:
            self._curated_reaches.extend(self._scored_reaches)
        else:
            for r in self._scored_reaches:
                if not RxReachSegment.overlaps(r, self._curated_reaches):
                    self._curated_reaches.append(r)
        if n_before != len(self._curated_reaches):
            self._curated_reaches.sort(key=lambda seg: seg.frame)
            self._undo_history.clear()  # wholesale operation cannot be undone.
            self._auto_saver_update(1)
            QTimer.singleShot(10, lambda: self.reaches_changed.emit())

    def add_scored_reach_containing_frame_to_curated_set(self, frame: int) -> None:
        """
        Add the scored reach segment containing the frame specified to the session's curated set. If no such segment is
        found in the set of scored reaches, or if the segment found overlaps an existing segment in the curated set, no
        action taken (in the latter case, an info message is logged). Otherwise, the scored reach is added to the
        curated set and the ``reaches_changed`` signal emitted.

        :param frame: A frame number in the master camera timeline.
        """
        found: int = (
            next((i for i, reach in enumerate(self._scored_reaches) if 0 <= frame - reach.frame <= reach.dur), -1))
        if found >= 0:
            self.add_curated_reach(self._scored_reaches[found])

    def add_curated_reach(self, reach: RxReachSegment) -> None:
        """
        Add a reach segment to the set of curated reaches for the current session. By design, a valid reach segment
        must meet certain criteria, must not overlap an existing reach, and must lie within the session timeline.

        :param reach: The reach segment to add. If this duplicates an existing segment, no action is taken. If its
            definition is invalid, it is not added and a message is posted to the application log.
        """
        ok = RxReachSegment.is_valid(reach) and (reach.end_frame < self.total_frames)
        if not ok:
            self._logger.info(f"Curated reach segment not added -- invalid definition.")
        elif RxReachSegment.overlaps(reach, self._curated_reaches):
            self._logger.info(f"Curated each segment not added -- overlaps an existing segment!")
        else:
            self._curated_reaches.append(reach)
            self._curated_reaches.sort(key=lambda seg: seg.frame)
            self._undo_history.append(_ReachOp(_ReachOp.ADD, reach))
            self._auto_saver_update(1)
            QTimer.singleShot(10, lambda: self.reaches_changed.emit())

    def replace_curated_reach(self, old_reach: RxReachSegment, replacement: RxReachSegment) -> bool:
        """
        Replace an existing reach segment in the curated set with the replacement segment specified, but only if the
        replacement is valid and does not overlap with any of the other curated reaches in the set.

        :param old_reach: The reach segment to replace.
        :param replacement: The replacement.
        :return: True if replacement succeeded. False if replacement is invalid or overlaps any other curated reach.
        """
        if old_reach not in self._curated_reaches:
            return False
        if (not RxReachSegment.is_valid(replacement)) or (replacement.end_frame >= self.total_frames):
            get_application_logger().info("Curated reach not replaced -- invalid replacement.")
            return False
        ok = False
        self._curated_reaches.remove(old_reach)
        if RxReachSegment.overlaps(replacement, self._curated_reaches):
            get_application_logger().info("Curated reach not replaced -- replacement overlaps another reach.")
            self._curated_reaches.append(old_reach)
        else:
            self._curated_reaches.append(replacement)
            ok = True
        self._curated_reaches.sort(key=lambda seg: seg.frame)
        if ok:
            self._undo_history.append(_ReachOp(_ReachOp.REPLACE, old_reach, replacement))
            self._auto_saver_update(1)
            QTimer.singleShot(10, lambda: self.reaches_changed.emit())
        return ok

    def delete_curated_reach_containing_frame(self, frame: int) -> None:
        """
        Delete the curated reach segment, if any, that contains the specified frame number.

        NOTE: Overlapping reach segments are allowed (though user would presumably eliminate these). This method only
        removes the first curated reach found that includes the specified frame.
        :param frame: A frame number in the master camera timeline.
        """
        found: int = (
            next((i for i, reach in enumerate(self._curated_reaches) if 0 <= frame - reach.frame <= reach.dur), -1))
        if found >= 0:
            old = self._curated_reaches.pop(found)
            self._undo_history.append(_ReachOp(_ReachOp.DEL, old))
            self._auto_saver_update(1)
            QTimer.singleShot(10, lambda: self.reaches_changed.emit())

    def delete_all_curated_reaches(self) -> None:
        """
        If any curated reach segements are defined for the current loaded session, remove all of them and emit the
        reaches_changed signal to notify GUI elements. No action taken if curated set is empty or no session loaded.
        """
        if len(self._curated_reaches) > 0:
            self._curated_reaches.clear()
            self._undo_history.clear()   # wholesale operation cannot be undone
            self._auto_saver_update(1)
            QTimer.singleShot(10, lambda: self.reaches_changed.emit())

    @property
    def undo_reach_op_description(self) -> str:
        """
        Description of next undo operation on the current session's curated set of reaches.
        :return: "" if undo history is empty, else a simple description of the next undo operation. """
        return self._undo_history[-1].undo_description if (len(self._undo_history) > 0) else ""

    def undo_last_reach_op(self) -> None:
        """
        If the undo history for the current session's curated reaches is not empty, undo the most recent operation.
        No action is taken if the undo history is empty. Otherwise, the most recent operation is undone, and the
        ``reaches_changed`` signal is emitted.
        """
        if len(self._undo_history) > 0:
            op = self._undo_history.pop(-1)
            if op.op_type == _ReachOp.ADD:
                assert (op.reach_1 in self._curated_reaches)
                self._curated_reaches.remove(op.reach_1)
            elif op.op_type == _ReachOp.DEL:
                assert (RxReachSegment.is_valid(op.reach_1) and (op.reach_1.end_frame < self.total_frames)
                        and not RxReachSegment.overlaps(op.reach_1, self._curated_reaches))
                self._curated_reaches.append(op.reach_1)
                self._curated_reaches.sort(key=lambda seg: seg.frame)
            else:
                assert (op.reach_2 in self._curated_reaches)
                self._curated_reaches.remove(op.reach_2)
                assert (RxReachSegment.is_valid(op.reach_1) and (op.reach_1.end_frame < self.total_frames)
                        and not RxReachSegment.overlaps(op.reach_1, self._curated_reaches))
                self._curated_reaches.append(op.reach_1)
                self._curated_reaches.sort(key=lambda seg: seg.frame)
            self._auto_saver_update(1)
            QTimer.singleShot(10, lambda: self.reaches_changed.emit())

    def _auto_saver_update(self, which: int) -> None:
        """
        Periodic auto-save timer timeout handler and configuration. Usage:
         - Set of curated reaches changed: Call method with ``which==1``. The auto save timer is started unless already
           running.
         - Curated reaches just saved: Call method with ``which==-1``. The auto save timer is stopped.
         - Auto-save timer timeout: Call method with ``which`` set to any other value. The curated reaches are saved.
        :param which: Arg determines action taken, as described.
        """
        if self._auto_save_timer is None:
            return
        if which == 1:
            if not self._auto_save_timer.isActive():
                self._auto_save_timer.start()
        else:
            self._auto_save_timer.stop()
            if which != -1:
                self._save_curated_reaches_for_current_session()

    @property
    def current_playback_speed(self) -> int:
        """
        The current playback speed in frames per second. Negative if reverse playback in progress; 0 if playback is
        stopped.
        """
        return self.PLAYBACK_SPEEDS[self._playback_speed_index]

    @property
    def playback_in_progress(self) -> bool:
        """ Current video playback state: running (True) or stopped (False). Always False if no video is available. """
        return self._playback_timer.isActive()

    def start_or_adjust_playback(self, reduce_speed=False) -> None:
        """
        Start playback, or increase/reduce current playback speed.
         - If playback is stopped, begin forward playback (or reverse if ``reduce_speed==True``) at the lowest
           supported playback speed. The ``playback_started`` signal is emitted after starting playback.
         - If playback is already in progress, increase or reduce playback speed IAW the ``reduce_speed`` argument. The
           ``playback_stopped`` signal if the resulting playback speed is zero.

        :param reduce_speed: True (False) to reduce (increase) current playback speed, or to start forward (reverse)
            playback at the lowest playback speed.
        """
        if not self.playback_in_progress:
            if (reduce_speed and (self.current_frame_num == 0) or
                    (self.current_frame_num >= self.total_frames and not reduce_speed)):
                return
            self._playback_speed_index = self.ZERO_SPEED_IDX + (-1 if reduce_speed else 1)
            self._playback_delta = int(self.PLAYBACK_SPEEDS[self._playback_speed_index] * self._TIMER_INTV_MS / 1000.0)
            self._playback_timer.start()
            QTimer.singleShot(10, lambda: self.playback_started.emit())
        elif ((reduce_speed and self._playback_speed_index == 0) or
              ((not reduce_speed) and (self._playback_speed_index == len(self.PLAYBACK_SPEEDS) - 1))):
            return
        else:
            self._playback_speed_index += (-1 if reduce_speed else 1)
            if self._playback_speed_index == self.ZERO_SPEED_IDX:
                self.stop_playback()
            else:
                self._playback_delta = int(
                    self.PLAYBACK_SPEEDS[self._playback_speed_index] * self._TIMER_INTV_MS / 1000.0)

    def stop_playback(self) -> None:
        """
        Stop video playback. If playback was active at the time of invocation, signal `playback_stopped` is emitted.
        No action taken if playback is already stopped.
        """
        if self.playback_in_progress:
            self._playback_timer.stop()
            self._playback_speed_index, self._playback_delta = self.ZERO_SPEED_IDX, 0
            QTimer.singleShot(10, lambda: self.playback_stopped.emit())

    def _on_playback_timer_timeout(self) -> None:
        """
        Update playback state whenever the playback timer expires.
        :return:
        """
        next_frame = self._curr_frame_num + self._playback_delta
        if next_frame <= 0 or next_frame >= self._total_frames:
            next_frame = 0 if self._playback_delta < 0 else self._total_frames - 1
            self._playback_timer.stop()
        self.go_to_frame(next_frame)
        if not self.playback_in_progress:
            self.playback_stopped.emit()

    def load(self, session_id: RxSessionID, init_frame: int = 0) -> Optional[str]:
        """
        Unload the current session contents (if any), then load all relevant content for the specified session from the
        session store.
        :param session_id: Identifies the experiment session to load.
        :param init_frame: The initial video frame number to load. If not a valid frame number, then the first frame is
            loaded. Default = 0.
        :return: None if load was successful (or specified session is already loaded!); else a description of the first
            error encountered. In the event the load fails, session manager remains in the "no session loaded" state.
        """
        if session_id == self._sesh_id:
            return None

        self.unload()

        if not session_id.exists():
            return f"Session folder not found: {str(session_id.location.absolute())}"

        # load all camera videos found and read in frame 0 for each. MUST be able to load master cam.
        master_cam = _determine_camera_master_from_systemdata(session_id, self._logger)
        available_cams = RxCam.available_cams(session_id.is_fixed_cam)
        videos = CamSource.load_session_cams(session_id, [c for c in available_cams], self._logger)
        if len(videos) == 0:
            return f"Failed to load identified master cam: {master_cam}"

        # ensure required camera videos are there
        if not all([cam in videos for cam in RxCam.required_cams(session_id.is_fixed_cam)]):
            return "Session is missing one or more required camera videos!"

        # load events list and correct event frame numbers if master cam had any frame drops
        err, events = SessionEventList.load_session_events(session_id, videos[master_cam])
        if events is None:
            return err

        # load CURATED reach segments for session, if any. On failure, log error but continue
        err, curated_reaches = SessionManager._load_curated_reaches_for(session_id)
        if len(err) > 0:
            self._logger.warning(f"Error while loading reach segments from file: {err}")

        # initially, frame 0 is loaded. If a different frame requested, load it now
        if 0 < init_frame < videos[master_cam].frame_count:
            for cam in videos:
                emsg = videos[cam].go_to_frame(init_frame)
                if isinstance(emsg, str):
                    self._logger.warning(f"Failed to load frame {init_frame} on {cam}: {emsg}")
        else:
            init_frame = 0

        # find all scorers and load analysis results for the first scorer (if any)
        scorers = SessionScorer.find_scorers(session_id)
        markers = np.empty((0, 5), dtype=np.int32)
        scored_reaches: List[RxReachSegment] = list()
        segmentation_results: List[SegmentationResultRecord] = list()
        curr_segmentation_result = -1
        curr_scorer = -1
        if len(scorers) > 0:
            markers = SessionManager._load_detected_markers_for(scorers[0])
            segmentation_results = SessionManager._discover_segmentation_results_for(scorers[0])
            if len(segmentation_results) > 0:
                curr_segmentation_result = 0
                scored_reaches = SessionManager._load_detected_reaches_from_path(
                    segmentation_results[curr_segmentation_result].detected_reaches_path)
            curr_scorer = 0

        # success!
        self._videos = videos
        self._master_cam = master_cam
        self._total_frames = self._videos[self._master_cam].frame_count
        self._events = events
        self._scorers = scorers
        self._curr_scorer_idx = curr_scorer
        self._scored_markers = markers
        self._scored_reaches = scored_reaches
        self._segmentation_results = segmentation_results
        self._curr_segmentation_result_idx = curr_segmentation_result
        self._curated_reaches = curated_reaches
        self._curr_frame_num = init_frame
        self._playback_speed_index, self._playback_delta = self.ZERO_SPEED_IDX, 0
        self._sesh_id = session_id
        return None

    @staticmethod
    def _load_detected_markers_for(scorer: SessionScorer) -> np.ndarray:
        """
        Load the detected body part marker locations (confidence score > 0.9) found by specified session scorer. These
        are stored in ``detected_markers.npy`` within the session subfolder containing all analysis results for that
        scorer.

        :param scorer: Identifies the body part detection model that was used to analyze the session and detect body
            part marker locations with a confidence score > 0.9.
        :return: A np.int32 Numpy array of shape (N*P*2, 5), where N is the session frame count and P is the number of
            body parts for which predictions were made (this could be less than the number of supported body part
            markers in ReachX). Each row contains [frame_num cam_index bp_index x y likelihood*10000], where cam_index
            and bp_index identify the source RxCam and particular RxBodyPart. If file not found or an error occurs,
            an empty np.int32 array is returned
        """
        markers_file = Path(scorer.results_folder, "detected_markers.npy")
        try:
            return np.load(markers_file)
        except Exception as e:
            get_application_logger().warning(f"Failed to load {markers_file}: {e}")
            return np.empty((0, 5), dtype=np.int32)

    @staticmethod
    def _load_curated_reaches_for(session_id: RxSessionID) -> Tuple[str, List[RxReachSegment]]:
        """
        Load the set of curated reach segments for the specified session. By convention in ReachX, these are stored in
        the session folder in the file ``<session_file_prefix>reaches.txt``. If the file is not found, then it is
        assumed no curated reach segments have been defined for the session.

        :param session_id: The session identifier
        :return: A 2-tuple: ("", reaches list) on succeses; (error description, []) if operation fails; ("", []) if
            source file not found.
        """
        p = Path(session_id.location, f"{session_id.session_file_prefix}reaches.txt")
        if not p.is_file():
            return "", []
        else:
            return RxReachSegment.load_reach_segments(p)

    @staticmethod
    def save_curated_reaches_for(session_id: RxSessionID, reaches: List[RxReachSegment]) -> str:
        """
        Save a session's curated reach segments to the dedicated file ``<session_file_prefix>reaches.txt`` in the
        session folder. Existing file is truncated and overwritten.

        :param session_id: Session ID, which locates session data folder on the file system.
        :param reaches: The session's curated reach segments. **If the reach segment list is empty and the dedicated
            file exists, that file is removed!
        :return: "" on success; else a brief error description
        """
        # if no reaches defined but a reach segments file exists in session folder, delete the existing file.
        p = Path(session_id.location, f"{session_id.session_file_prefix}reaches.txt")
        if len(reaches) == 0:
            p.unlink(missing_ok=True)
            return ""
        else:
            return RxReachSegment.save_reach_segments(reaches, p)

    @staticmethod
    def _load_detected_reaches_for(scorer: SessionScorer) -> List[RxReachSegment]:
        """
        Load reach segments found by running reach segmentation algorithm over hand and pellet trajectory data derived
        rom the specified session scorer. They are stored in ``detected_reaches.txt`` within the session subfolder
        containing all analysis results for that scorer.

        :param scorer: Identifies the body part detection model from which hand and pellet trajectory data was derived
            and then used to perform reach segmentation on this session.
        :return: List of detected reach segments. An empty list is returned if reach segmentation has not been done
            using the specified scorer, or if an error occurs while loading.
        """
        reaches_file = Path(scorer.results_folder, "detected_reaches.txt")
        return SessionManager._load_detected_reaches_from_path(reaches_file)

    @staticmethod
    def _load_detected_reaches_from_path(reaches_file: Path) -> List[RxReachSegment]:
        """
        Load reach segments from an explicit detected_reaches.txt path.
        """
        if not reaches_file.is_file():
            return []
        emsg, reaches = RxReachSegment.load_reach_segments(reaches_file)
        if len(emsg) > 0:
            get_application_logger().warning(f"Failed to load {reaches_file}: {emsg}")
        return reaches

    @staticmethod
    def _discover_segmentation_results_for(scorer: SessionScorer) -> List[SegmentationResultRecord]:
        """
        Discover all selectable scored-reach results for a scorer.
        """
        out: List[SegmentationResultRecord] = list()
        direct = Path(scorer.results_folder, "detected_reaches.txt")
        if direct.is_file():
            out.append(SegmentationResultRecord("algorithmic", "Algorithmic", "algorithmic", direct,
                                                scorer.results_folder, direct.stat().st_mtime))

        nested_root = Path(scorer.results_folder, RESULTS_FOLDER_NAME)
        if nested_root.is_dir():
            for folder in sorted([p for p in nested_root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
                reaches_file = Path(folder, "detected_reaches.txt")
                if not reaches_file.is_file():
                    continue
                display_name = SessionManager._segmentation_result_display_name(folder)
                out.append(SegmentationResultRecord(folder.name, display_name, "ml", reaches_file, folder,
                                                    reaches_file.stat().st_mtime))
        return out

    @staticmethod
    def _segmentation_result_display_name(folder: Path) -> str:
        summary_path = Path(folder, "segmentation_summary.json")
        if summary_path.is_file():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                model = summary.get("model", {}) if isinstance(summary, dict) else {}
                model_name = (model.get("display_name") or model.get("model_id")) if isinstance(model, dict) else None
                if model_name:
                    return f"ML: {model_name} ({folder.name})"
            except Exception:
                pass
        return f"ML: {folder.name}"

    def _reload_segmentation_results_for_current_scorer(
            self,
            preferred_result_id: Optional[str] = None,
            select_latest: bool = False) -> None:
        selected = preferred_result_id
        if selected is None and self.current_segmentation_result is not None:
            selected = self.current_segmentation_result.result_id
        self._segmentation_results.clear()
        self._curr_segmentation_result_idx = -1
        self._scored_reaches.clear()
        if self.current_scorer is None:
            return

        self._segmentation_results = SessionManager._discover_segmentation_results_for(self.current_scorer)
        if len(self._segmentation_results) == 0:
            return

        idx = -1
        if select_latest:
            latest = max(self._segmentation_results, key=lambda r: r.modified_time)
            idx = self._segmentation_results.index(latest)
        elif selected is not None:
            idx = next((i for i, record in enumerate(self._segmentation_results) if record.result_id == selected), -1)
        if idx < 0:
            idx = 0
        self._curr_segmentation_result_idx = idx
        self._scored_reaches = SessionManager._load_detected_reaches_from_path(
            self._segmentation_results[idx].detected_reaches_path)

    def unload(self) -> None:
        if self.session_loaded:
            self.stop_playback()
            for cam in self._videos:
                self._videos[cam].unload()
            self._videos.clear()

            # save any curated reach segments
            self._save_curated_reaches_for_current_session()
            self._curated_reaches.clear()

            self._sesh_id, self._total_frames, self._curr_frame_num = None, 0, -1
            self._events = None

            self._scorers.clear()
            self._curr_scorer_idx = -1
            self._scored_markers = np.empty((0, 5), dtype=np.int32)
            self._scored_reaches.clear()
            self._segmentation_results.clear()
            self._curr_segmentation_result_idx = -1

            # the undo history for the set of curated reaches is never persisted
            self._undo_history.clear()

    def _save_curated_reaches_for_current_session(self) -> None:
        if self.session_loaded:
            err_msg = SessionManager.save_curated_reaches_for(self._sesh_id, self._curated_reaches)
            if len(err_msg) > 0:
                self._logger.error(err_msg)
            elif len(self._curated_reaches) > 0:
                self._logger.info(f"Saved curated reach segments for {self._sesh_id}")
            self._auto_saver_update(-1)
