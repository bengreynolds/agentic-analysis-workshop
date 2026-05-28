"""
calibration.py - Read and use content from a FLIR camera calibration folder.

**CREDIT**: Much of the code here was extracted/adapted from the Calibration.calibration_FLIR.py module in the
Christie lab Github repo: https://github.com/Cerebellum-Lab/reach-training-fixed-cam/tree/main.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# noinspection PyPackageRequirements
import cv2
import numpy as np
import yaml

from reachx.common import valid_flir_calib_folder_name


class FlirCalibration:
    """
    Encapsulation of the contents of a FLIR camera calibration folder generated during calibration of a fixed-cam
    experiment rig in the Christie lab.

    This class does not support creation of the calibration data, only reading those files needed in order to transform
    the 2D object trajectories on the ``RxCam.LEFT`` and ``RxCam.RIGHT`` cameras in a fixed-cam setup into 3D
    trajectories centered around a common origin.
    """
    _CAL_VIDEO_DIR: str = "source_videos"
    """ Subfolder within calibration folder containing the two calibration videos. """
    _OFFSETS_FILE: str = "camera_offsets.pkl"
    """ File within calibration folder containing camera offsets. Presence required. """
    _METADATA_FILE: str = "calibration_userset.yaml"
    """ Metadata file within calibration folder containing camera positions and other info. Presence required. """
    _CAM_MATRIX_DIR: str = "camera_matrix"
    """ Subfolder containing stereo parameters file. Presence required. """
    _STEREO_PARAMS_FILE: str = "stereo_params.pickle"
    """ Stereo calibration parameters file. Presence required. """

    @staticmethod
    def validate_calibration_folder_contents(p: Path) -> str:
        """
        Verify the expected contents/structure of a FLIR camera calibration folder. This is not an exhaustive
        validation; it only checks the directory structure and looks for those files required to transform the 2D
        object trajectories on the left and right cameras in a fixed-cam setup into 3D trajectories centered around
        a common origin:
         - Validates format of calibration folder name: ``<S>mm_<R>r_<C>c``. Here ``<S>`` is the size of each square in
           the calibration checkerboard grid, in millimeters. ``<R>`` and ``<C>`` are the number of grid rows and
           columns. All are integer strings.
         - Verifies presence of the camera offsets file, the calibration metadata file, and the stereo parameters file.
           The file contents are NOT read in.
         - Verifies presence of the two calibration video files, ending in ``left.mp4`` and ``right.mp4``, for the left
           and right cameras.

        :param p: Full path to the calibration folder.
        :return: Empty string if calibration folder passes all checks described, or a brief error message describing the
            first issue found.
        """
        try:
            if not (isinstance(p, Path) and p.is_dir()):
                raise Exception("Invalid or nonexistent path")
            if not valid_flir_calib_folder_name(p.name):
                raise Exception("Folder name incorrect format (should be, eg, '4mm_8r_8c')")
            video_dir = Path(p, FlirCalibration._CAL_VIDEO_DIR)
            if not video_dir.is_dir():
                raise Exception(f"Missing calibration videos subfolder")
            mp4files = [f for f in video_dir.iterdir() if f.name.lower().endswith(".mp4")]
            if next((f for f in mp4files if f.name.lower().endswith('left.mp4')), None) is None:
                raise Exception(f"Missing calibration video for left camera")
            if next((f for f in mp4files if f.name.lower().endswith('right.mp4')), None) is None:
                raise Exception(f"Missing calibration video for right camera")
            if not Path(p, FlirCalibration._OFFSETS_FILE).is_file():
                raise Exception(f"Missing camera offsets file")
            if not Path(p, FlirCalibration._METADATA_FILE).is_file():
                raise Exception(f"Missing calibration metadata file")
            if not Path(p, FlirCalibration._CAM_MATRIX_DIR, FlirCalibration._STEREO_PARAMS_FILE).is_file():
                raise Exception(f"Missing stereo parameters file")
            return ""
        except Exception as e:
            return str(e)

    @staticmethod
    def create_flir_calibration(p: Path) -> Tuple[str, Optional[FlirCalibration]]:
        emsg = FlirCalibration.validate_calibration_folder_contents(p)
        if len(emsg) > 0:
            return emsg, None
        try:
            # load camera position from calibration metadata file
            with open(Path(p, FlirCalibration._METADATA_FILE), 'r') as metadata_file:
                metadata = yaml.safe_load(metadata_file)
            cam_pos = metadata['camera_pos']
            if not (isinstance(cam_pos, dict) and all(k in cam_pos
                                                      for k in ['camLele', 'camRele', 'camLazi', 'camRazi'])):
                raise Exception(f"Bad calibration metadata file: {FlirCalibration._METADATA_FILE}")

            # get stems of left and right cam video files, use to form key to access corresponding stereo params
            video_dir = Path(p, FlirCalibration._CAL_VIDEO_DIR)
            mp4files = [f for f in video_dir.iterdir() if f.name.lower().endswith(".mp4")]
            left_path: Path = next((f for f in mp4files if f.name.lower().endswith('left.mp4')))
            right_path: Path = next((f for f in mp4files if f.name.lower().endswith('right.mp4')))
            stereo_key = left_path.stem + "-" + right_path.stem

            # load stereo params and extract the right one (typically there will be only one)
            with open(Path(p, FlirCalibration._CAM_MATRIX_DIR, FlirCalibration._STEREO_PARAMS_FILE), 'rb') as f:
                stereo = pickle.load(f)
            if not (isinstance(stereo, dict) and (stereo_key in stereo) and isinstance(stereo[stereo_key], dict)):
                raise Exception(f"Bad stereo calibration parameters file, or missing parameters.")
            return "", FlirCalibration(p, stereo[stereo_key], cam_pos)
        except Exception as e:
            return str(e), None

    def __init__(self, p: Path, stereo_params: Dict[str, Any], cam_pos: Dict[str, float]) -> None:
        """ DO NOT USE DIRECTLY. Use ``create_flir_calibration()`` instead."""
        self._path = p
        """ File system path to the calibration folder. """
        self._cam_left_elevation = cam_pos['camLele']
        """ Left camera elevation in degrees. """
        self._cam_right_elevation = cam_pos['camRele']
        """ Right camera elevation in degrees. """
        self._cam_left_azimuth = cam_pos['camLazi']
        """ Left camera azimuth in degrees. """
        self._cam_right_azimuth = cam_pos['camRazi']
        """ Right camera azimuth in degrees. """
        self._stereo_params = stereo_params
        """ Stereo calibration parameters. """

    @property
    def location(self) -> Path:
        """ Directory containing the FLIR camera calibration information"""
        return self._path

    @property
    def square_size_mm(self) -> int:
        """ The size of each individual square in the checkerboard calibration grid, in mm. For scaling purposes. """
        idx = self._path.name.index("mm_")
        return int(self._path.name[:idx])

    def undistort_2d_trajectory(self, traj_2d: np.ndarray, is_left: bool) -> None:
        """
        Undistort the points in a 2D trajectory on the left or right camera view, using stereo calibration parameters.

        :param traj_2d: A Numpy array with shape ``(N, M)``, where ``N`` is the # of camera frames, and ``M >= 2``. It
             is assumed the first two columns contain the ``(X,Y)`` coordinates of each trajectory point. **The X and Y
             columns are updated in place; all other columns are left unchanged.**
        :param is_left: ``True`` if trajectory is from the left camera view, ``False`` for the right camera view.
        :raise Exception: If ``traj_2d`` array shape is invalid.
        """
        if traj_2d.ndim != 2 or traj_2d.shape[1] < 2:
            raise Exception(f"Invalid input for 2D trajectory array: {str(traj_2d.shape)}")
        if traj_2d.size == 0:
            return

        points = FlirCalibration.rotate_2d_points(traj_2d[:, :2], self._stereo_params["rot_cor"])
        cam_idx = 1 if is_left else 2
        pts_undist = cv2.undistortPoints(
            src=points,
            cameraMatrix=self._stereo_params[f"cameraMatrix{cam_idx}"],
            distCoeffs=self._stereo_params[f"distCoeffs{cam_idx}"],
            R=self._stereo_params[f"R{cam_idx}"],
            P=self._stereo_params[f"P{cam_idx}"],
        )
        traj_2d[:, :2] = pts_undist.squeeze()

    @staticmethod
    def rotate_2d_points(pts: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate the set of points IAW the specified angle in degrees.

        :param pts: Array of points. Require an ``(N, 2)`` Numpy array. If empty, no action is taken and the original
            array is returned.
        :param angle_deg: Rotation angle in degrees.
        :return: The array of rotated points, also ``(N, 2)`` and of the same dtype as original array.
        :raise Exception: If array shape is invalid.
        """
        if pts.size == 0:
            return pts
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise Exception(f"Invalid input for rotating 2D points. Expected a 2D array of shape (N, 2)")
        original_dtype = pts.dtype

        # apply rotation, using rotation matrix as the same dtype as the input
        rad = np.radians(angle_deg)
        rotation_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ], dtype=original_dtype)
        rotated_points = np.dot(pts, rotation_matrix.T)
        return rotated_points.astype(original_dtype)

    def triangulate_2d_trajectories(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Use stereo calibration to triangulate the 2D trajectories of an object on the left and right camera views,
        forming the object's 3D trajectory in XYZ coordinate.

        :param left: Trajectory points on the left camera view. Require and ``(N, 2)`` Numpy array.
        :param right: Trajectory points on the right camera, also an ``(N, 2)`` Numpy array.
        :return: An ``(N, 3)`` ``np.float32`` array of the 3D points in standard Euclidean coordinates (X, Y, Z).
            If N = 0, returned array is empty.
        :raise Exception: If array shape is invalid.
        """
        if (left.ndim != right.ndim) or (left.ndim != 2):
            raise Exception(f"Expected 2D arrays of shape (N, 2) for left and right trajectories.")
        if left.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        # note that cv2 wants 2xN and returns 4xN. Normalize by homogeneous coordinate W, transpose back to Nx4, then
        # drop the homogenous coordinate (4th column).
        homogenous_3d = cv2.triangulatePoints(self._stereo_params["P1"], self._stereo_params["P2"], left.T, right.T)
        euclidean_3d = homogenous_3d / homogenous_3d[3]
        out = euclidean_3d.T[:, :3]
        return out.astype(np.float32)

    def transform_3d_trajectory(self, traj_3d: np.ndarray, x_off: float, y_off: float, z_off: float) -> None:
        """
        Transform a 3D object trajectory in the XYZ space covered by the left and right camera views represented by
        this FLIR calibration. All points are translated IAW the specified X/Y/Z offsets, then rotated IAW the
        camera azimuth and elevation (averages over both cameras), and the rotation correction for Z (from stereo
        calibration parameters). This is followed by a reflection over the YZ plane followed by a 90-deg CW rotation
        about the new x-axis: so (X, Y, Z) becomes (-X, -Z, Y). This converts the coordinate system from a right- to
        a left-handed system. Finally, coordinates are scaled by the size of an individual square in the "checkerboard"
        calibration grid.

        :param traj_3d: An ``(N, 3)`` array of 3D points in standard Euclidean coordinates (X, Y, Z). The array is
            updated in place. No action taken if array is empty. It is assumed X, Y, Z is in column 0, 1, 2.
        :param x_off: Offset along X-axis.
        :param y_off: Offset along Y-axis.
        :param z_off: Offset along Z-axis.
        :raise Exception: If array shape is invalid.
        """
        if traj_3d.ndim != 2 or traj_3d.shape[1] != 3:
            raise Exception(f"Expected array of shape (N, 3) containing 3D object trajectory.")
        if traj_3d.size == 0:
            return

        # translation
        for i, ofs in enumerate([x_off, y_off, z_off]):
            if ofs != 0:
                traj_3d[:, i] = traj_3d[:, i] - ofs

        # rotation
        avg_elev_deg = float(np.mean((self._cam_left_elevation, self._cam_right_elevation)))
        avg_azi_deg = float(np.mean((self._cam_left_azimuth, self._cam_right_azimuth)))
        rot_deg = self._stereo_params["rot_cor"]
        rotated = FlirCalibration.rotate_3d_points(traj_3d, avg_azi_deg, avg_elev_deg, rot_deg)
        x, y, z = rotated[:, 0], rotated[:, 1], rotated[:, 2]

        # final result: 90deg CCW rotation about X, followed by reflection over YZ plane. And scale to mm units.
        traj_3d[:, 0] = -x * self.square_size_mm
        traj_3d[:, 1] = -z * self.square_size_mm
        traj_3d[:, 2] = y * self.square_size_mm
        return

    @staticmethod
    def rotate_3d_points(pts: np.ndarray, x_deg: float, y_deg: float, z_deg: float):
        """
        Rotate the set of 3D points IAW the specified rotation angles.

        :param pts: Array of 3D points. Require an ``(N, 3)`` Numpy array. If empty, no action is taken and the original
            array is returned.
        :param x_deg: Rotation angle about X-axis in degrees.
        :param y_deg: Rotation angle about Y-axis in degrees.
        :param z_deg: Rotation angle about Z-axis in degrees.
        :return: The array of rotated points, also ``(N, 3)`` and of the same dtype as original array.
        :raise Exception: If array shape is invalid.
        """
        if pts.size == 0:
            return pts
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise Exception(f"Invalid input for rotating 3D points. Expected a 2D array of shape (N, 3)")
        original_dtype = pts.dtype

        # define rotation matrices, using same dtype as original array
        x_rad = np.radians(x_deg)
        y_rad = np.radians(y_deg)
        z_rad = np.radians(z_deg)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(x_rad), -np.sin(x_rad)],
            [0, np.sin(x_rad), np.cos(x_rad)]
        ], dtype=original_dtype)
        rot_y = np.array([
            [np.cos(y_rad), 0, np.sin(y_rad)],
            [0, 1, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad)]
        ], dtype=original_dtype)
        rot_z = np.array([
            [np.cos(z_rad), -np.sin(z_rad), 0],
            [np.sin(z_rad), np.cos(z_rad), 0],
            [0, 0, 1]
        ], dtype=original_dtype)

        # apply rotations sequentially: X, then Y, then Z
        rotated_points = pts @ rot_x.T @ rot_y.T @ rot_z.T
        return rotated_points.astype(original_dtype)
