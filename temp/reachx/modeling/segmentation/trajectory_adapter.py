from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from reachx.common import RxBodyPart, RxSessionID, TrajCol, FixedCamTrajCol
from reachx.modeling.segmentation.metadata import SegmentationModelMetadata


@dataclass(frozen=True)
class SegmentationInputTable:
    frames: np.ndarray
    feature_matrix: np.ndarray
    feature_names: List[str]


_AXIS_TO_TRAJCOL = {
    "x": TrajCol.X_FILT.value,
    "y": TrajCol.Y_FILT.value,
    "z": TrajCol.Z_FILT.value,
}

_AXIS_TO_FIXEDCOL = {
    "x": FixedCamTrajCol.X.value,
    "y": FixedCamTrajCol.Y.value,
    "z": FixedCamTrajCol.Z.value,
}

_RAW_CAMERA_FILES = {
    "front": "frontCam.npy",
    "side": "sideCam.npy",
    "left": "left.npy",
    "right": "right.npy",
}

_RAW_BODYPART_ALIASES = {
    "tongue": RxBodyPart.TONGUE,
    "mouth": RxBodyPart.MOUTH,
    "pellet": RxBodyPart.PELLET,
    "sdh_flat": RxBodyPart.SIDE_HAND_FLAT,
    "sdh_spread": RxBodyPart.SIDE_HAND_SPREAD,
    "sdh_grab": RxBodyPart.SIDE_HAND_GRAB,
    "fth_reach": RxBodyPart.FRONT_HAND_REACH,
    "fth_grasp": RxBodyPart.FRONT_HAND_GRASP,
    "exclude1": RxBodyPart.EXCLUDE1,
    "exclude2": RxBodyPart.EXCLUDE2,
    "nose": RxBodyPart.NOSE,
    "rh_flat": RxBodyPart.RIGHT_HAND_FLAT,
    "rh_spread": RxBodyPart.RIGHT_HAND_SPREAD,
    "rh_grab": RxBodyPart.RIGHT_HAND_GRAB,
    "r_hand_flat": RxBodyPart.RIGHT_HAND_FLAT,
    "r_hand_spread": RxBodyPart.RIGHT_HAND_SPREAD,
    "r_hand_grab": RxBodyPart.RIGHT_HAND_GRAB,
    "right_hand_flat": RxBodyPart.RIGHT_HAND_FLAT,
    "right_hand_spread": RxBodyPart.RIGHT_HAND_SPREAD,
    "right_hand_grab": RxBodyPart.RIGHT_HAND_GRAB,
    "lh_flat": RxBodyPart.LEFT_HAND_FLAT,
    "lh_spread": RxBodyPart.LEFT_HAND_SPREAD,
    "lh_grab": RxBodyPart.LEFT_HAND_GRAB,
    "l_hand_flat": RxBodyPart.LEFT_HAND_FLAT,
    "l_hand_spread": RxBodyPart.LEFT_HAND_SPREAD,
    "l_hand_grab": RxBodyPart.LEFT_HAND_GRAB,
    "left_hand_flat": RxBodyPart.LEFT_HAND_FLAT,
    "left_hand_spread": RxBodyPart.LEFT_HAND_SPREAD,
    "left_hand_grab": RxBodyPart.LEFT_HAND_GRAB,
    "star": RxBodyPart.STAR,
    "tongue_mid": RxBodyPart.TONGUE_MID,
    "tongue_tip": RxBodyPart.TONGUE_TIP,
    "triangle": RxBodyPart.TRIANGLE,
    "diamond": RxBodyPart.DIAMOND,
}

_FIXED_TRAJECTORY_SOURCES = {
    "r_hand": str(RxBodyPart.R_HAND),
    "right_hand": str(RxBodyPart.R_HAND),
    "l_hand": str(RxBodyPart.L_HAND),
    "left_hand": str(RxBodyPart.L_HAND),
    "pellet": str(RxBodyPart.PELLET),
    "tongue": str(RxBodyPart.TONGUE),
    "mouth": str(RxBodyPart.MOUTH),
    "nose": str(RxBodyPart.NOSE),
    "rh_flat": str(RxBodyPart.RIGHT_HAND_FLAT),
    "rh_spread": str(RxBodyPart.RIGHT_HAND_SPREAD),
    "rh_grab": str(RxBodyPart.RIGHT_HAND_GRAB),
    "r_hand_flat": str(RxBodyPart.RIGHT_HAND_FLAT),
    "r_hand_spread": str(RxBodyPart.RIGHT_HAND_SPREAD),
    "r_hand_grab": str(RxBodyPart.RIGHT_HAND_GRAB),
    "right_hand_flat": str(RxBodyPart.RIGHT_HAND_FLAT),
    "right_hand_spread": str(RxBodyPart.RIGHT_HAND_SPREAD),
    "right_hand_grab": str(RxBodyPart.RIGHT_HAND_GRAB),
    "lh_flat": str(RxBodyPart.LEFT_HAND_FLAT),
    "lh_spread": str(RxBodyPart.LEFT_HAND_SPREAD),
    "lh_grab": str(RxBodyPart.LEFT_HAND_GRAB),
    "l_hand_flat": str(RxBodyPart.LEFT_HAND_FLAT),
    "l_hand_spread": str(RxBodyPart.LEFT_HAND_SPREAD),
    "l_hand_grab": str(RxBodyPart.LEFT_HAND_GRAB),
    "left_hand_flat": str(RxBodyPart.LEFT_HAND_FLAT),
    "left_hand_spread": str(RxBodyPart.LEFT_HAND_SPREAD),
    "left_hand_grab": str(RxBodyPart.LEFT_HAND_GRAB),
    "star": str(RxBodyPart.STAR),
    "tongue_mid": str(RxBodyPart.TONGUE_MID),
    "tongue_tip": str(RxBodyPart.TONGUE_TIP),
    "triangle": str(RxBodyPart.TRIANGLE),
    "diamond": str(RxBodyPart.DIAMOND),
}

_LEFT_HAND_TOKENS = (
    "l_hand",
    "left_hand",
    "l_hand_flat",
    "l_hand_spread",
    "l_hand_grab",
    "lh_flat",
    "lh_spread",
    "lh_grab",
    "left_hand_flat",
    "left_hand_spread",
    "left_hand_grab",
)


def load_segmentation_input(
        sesh_id: RxSessionID,
        scorer_folder: Path,
        metadata: SegmentationModelMetadata) -> SegmentationInputTable:
    if metadata.session_type not in {"any", "fixed", "fixed_cam", "legacy", "legacy_cam"}:
        raise ValueError(f"Unsupported segmentation session_type: {metadata.session_type}")
    if sesh_id.is_fixed_cam:
        _validate_fixed_model_left_hand_features(metadata)

    trajectory_keypoints: Dict[str, np.ndarray] | None = None
    raw_keypoints: Dict[str, Dict[str, np.ndarray]] | None = None
    if sesh_id.is_fixed_cam:
        if metadata.session_type in {"legacy", "legacy_cam"}:
            raise ValueError("Selected segmentation model is for legacy-cam sessions, not fixed-cam sessions")
    else:
        if metadata.session_type in {"fixed", "fixed_cam"}:
            raise ValueError("Selected segmentation model is for fixed-cam sessions, not legacy-cam sessions")

    features: List[np.ndarray] = []
    frame_count = -1
    for feature_name in metadata.features:
        raw_feature = _parse_raw_camera_feature(_normalize_feature_name(feature_name))
        if raw_feature is not None:
            if raw_keypoints is None:
                raw_keypoints = _load_raw_camera_keypoints(scorer_folder, sesh_id.is_fixed_cam)
            vector = _raw_camera_feature_vector(feature_name, raw_feature, raw_keypoints)
        else:
            if trajectory_keypoints is None:
                trajectory_keypoints = _load_fixed_cam_keypoints(scorer_folder) if sesh_id.is_fixed_cam \
                    else _load_legacy_keypoints(scorer_folder)
            vector = _feature_vector(feature_name, trajectory_keypoints, sesh_id.is_fixed_cam)
        if frame_count < 0:
            frame_count = int(vector.shape[0])
        elif int(vector.shape[0]) != frame_count:
            raise ValueError(f"Feature {feature_name} has {vector.shape[0]} frames; expected {frame_count}")
        features.append(vector)

    if frame_count < 0:
        raise ValueError("Segmentation model metadata did not define any features")
    matrix = np.column_stack(features).astype(np.float32)
    if matrix.shape != (frame_count, len(metadata.features)):
        raise ValueError("Generated segmentation feature matrix has unexpected shape")
    return SegmentationInputTable(
        frames=np.arange(frame_count, dtype=np.int32),
        feature_matrix=matrix,
        feature_names=metadata.features.copy(),
    )


def _load_fixed_cam_keypoints(scorer_folder: Path) -> Dict[str, np.ndarray]:
    path = Path(scorer_folder, "trajectories.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed-cam trajectory file: {path}")
    with np.load(path) as data:
        required = [str(RxBodyPart.R_HAND), str(RxBodyPart.L_HAND), str(RxBodyPart.PELLET)]
        missing = [source for source in required if source not in data]
        if missing:
            raise KeyError(f"Fixed-cam trajectory file is missing required body part(s): {', '.join(missing)}")
        out: Dict[str, np.ndarray] = {}
        for name, source in _FIXED_TRAJECTORY_SOURCES.items():
            if source in data:
                out[name] = np.asarray(data[source], dtype=np.float32)
        return out


def _load_legacy_keypoints(scorer_folder: Path) -> Dict[str, np.ndarray]:
    hand_path = Path(scorer_folder, "hand.npy")
    pellet_path = Path(scorer_folder, "pellet.npy")
    if not hand_path.is_file() or not pellet_path.is_file():
        raise FileNotFoundError(f"Missing legacy trajectory files under {scorer_folder}")
    hand = np.asarray(np.load(hand_path), dtype=np.float32)
    pellet = np.asarray(np.load(pellet_path), dtype=np.float32)
    return {"hand": hand, "r_hand": hand, "right_hand": hand, "pellet": pellet}


def _load_raw_camera_keypoints(scorer_folder: Path, is_fixed_cam: bool) -> Dict[str, Dict[str, np.ndarray]]:
    camera_names = ("left", "right") if is_fixed_cam else ("front", "side")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for camera_name in camera_names:
        path = Path(scorer_folder, _RAW_CAMERA_FILES[camera_name])
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw camera prediction file: {path}")
        out[camera_name] = _load_raw_camera_file(path)
    return out


def _load_raw_camera_file(path: Path) -> Dict[str, np.ndarray]:
    rows = np.asarray(np.load(path), dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 5:
        raise ValueError(f"Invalid raw camera prediction file shape for {path}: {rows.shape}")
    if rows.shape[0] == 0:
        raise ValueError(f"Raw camera prediction file is empty: {path}")
    if not np.all(np.isfinite(rows[:, :5])):
        raise ValueError(f"Raw camera prediction file contains non-finite values: {path}")
    if np.any(rows[:, 0] < 0):
        raise ValueError(f"Raw camera prediction file contains negative frame numbers: {path}")
    frame_numbers = rows[:, 0].astype(np.int32)
    bodyparts = rows[:, 1].astype(np.int32)
    if not np.array_equal(rows[:, 0], frame_numbers.astype(rows.dtype)):
        raise ValueError(f"Raw camera prediction file contains non-integer frame numbers: {path}")
    if not np.array_equal(rows[:, 1], bodyparts.astype(rows.dtype)):
        raise ValueError(f"Raw camera prediction file contains non-integer body part IDs: {path}")
    frame_count = int(np.max(frame_numbers)) + 1
    out: Dict[str, np.ndarray] = {}
    for alias, part in _RAW_BODYPART_ALIASES.items():
        part_rows = rows[bodyparts == int(part.value)]
        if part_rows.shape[0] == 0:
            continue
        values = np.full((frame_count, 3), np.nan, dtype=np.float32)
        found = np.zeros(frame_count, dtype=bool)
        for row in part_rows:
            frame = int(row[0])
            if 0 <= frame < frame_count and (not found[frame] or row[4] >= values[frame, 2]):
                values[frame] = row[2:5]
                found[frame] = True
        out[alias] = values
    return out


def _raw_camera_feature_vector(
        original_name: str,
        raw_feature: Tuple[str, str, str],
        raw_keypoints: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    camera, keypoint, attr = raw_feature
    if camera not in raw_keypoints:
        raise ValueError(f"Raw camera feature is not available for this session type: {original_name}")
    camera_keypoints = raw_keypoints[camera]
    if keypoint not in camera_keypoints:
        raise ValueError(f"Raw camera prediction file is missing body part for feature: {original_name}")
    if attr in {"x", "y"}:
        vector = camera_keypoints[keypoint][:, 0 if attr == "x" else 1]
    elif attr in {"p", "confidence", "likelihood"}:
        vector = camera_keypoints[keypoint][:, 2]
    else:
        raise ValueError(f"Unsupported raw camera segmentation feature: {original_name}")
    if not np.all(np.isfinite(vector)):
        missing = np.flatnonzero(~np.isfinite(vector))
        raise ValueError(
            f"Raw camera prediction file is missing values for feature {original_name}; "
            f"first missing frame is {int(missing[0])}"
        )
    return vector


def _feature_vector(name: str, keypoints: Dict[str, np.ndarray], is_fixed_cam: bool) -> np.ndarray:
    normalized = _normalize_feature_name(name)
    distance_pair = _parse_distance_feature(normalized, tuple(keypoints.keys()))
    if distance_pair is not None:
        left, right = distance_pair
        return _distance(keypoints[left], keypoints[right], is_fixed_cam)

    keypoint, attr = _split_keypoint_attr(normalized, tuple(keypoints.keys()))
    if keypoint is None or attr is None:
        raise ValueError(f"Unsupported segmentation feature: {name}")
    if attr in {"x", "y", "z"}:
        col = _AXIS_TO_FIXEDCOL[attr] if is_fixed_cam else _AXIS_TO_TRAJCOL[attr]
        return keypoints[keypoint][:, col]
    if attr in {"speed", "speed_filt"}:
        col = FixedCamTrajCol.SPEED_FILT.value if is_fixed_cam else TrajCol.SPEED_FILT.value
        return keypoints[keypoint][:, col]
    if attr in {"p", "confidence"}:
        if is_fixed_cam:
            return keypoints[keypoint][:, FixedCamTrajCol.P.value]
        x_lhood = keypoints[keypoint][:, TrajCol.X_LHOOD.value]
        yz_lhood = keypoints[keypoint][:, TrajCol.YZ_LHOOD.value]
        return np.minimum(x_lhood, yz_lhood)
    if attr == "x_lhood" and not is_fixed_cam:
        return keypoints[keypoint][:, TrajCol.X_LHOOD.value]
    if attr == "yz_lhood" and not is_fixed_cam:
        return keypoints[keypoint][:, TrajCol.YZ_LHOOD.value]
    raise ValueError(f"Unsupported segmentation feature: {name}")


def _normalize_feature_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("right_hand", "r_hand").replace("left_hand", "l_hand")
    return normalized


def _parse_raw_camera_feature(name: str) -> Tuple[str, str, str] | None:
    for camera in _RAW_CAMERA_FILES:
        prefix = f"{camera}_"
        if not name.startswith(prefix):
            continue
        keypoint, attr = _split_keypoint_attr(name[len(prefix):], tuple(_RAW_BODYPART_ALIASES.keys()))
        if keypoint is not None and attr is not None:
            return camera, keypoint, attr
    return None


def _validate_fixed_model_left_hand_features(metadata: SegmentationModelMetadata) -> None:
    names = list(metadata.features) + list(metadata.required_body_parts)
    for name in names:
        normalized = _normalize_feature_name(name)
        if any(token in normalized for token in _LEFT_HAND_TOKENS):
            return
    raise ValueError(
        "Fixed-cam segmentation models must include left-hand features "
        "(L_Hand trajectory or left-hand pose keypoints)"
    )


def _split_keypoint_attr(name: str, keypoints: Tuple[str, ...]) -> Tuple[str | None, str | None]:
    for keypoint in sorted(keypoints, key=len, reverse=True):
        prefix = f"{keypoint}_"
        if name.startswith(prefix):
            return keypoint, name[len(prefix):]
    return None, None


def _parse_distance_feature(name: str, keypoints: Tuple[str, ...]) -> Tuple[str, str] | None:
    if name.startswith("dist_"):
        return _parse_keypoint_pair(name[5:], keypoints)
    if name.endswith("_distance"):
        return _parse_keypoint_pair(name[:-9], keypoints)
    return None


def _parse_keypoint_pair(value: str, keypoints: Tuple[str, ...]) -> Tuple[str, str] | None:
    for left in sorted(keypoints, key=len, reverse=True):
        prefix = f"{left}_"
        if value.startswith(prefix):
            right = value[len(prefix):]
            if right in keypoints:
                return left, right
    return None


def _distance(left: np.ndarray, right: np.ndarray, is_fixed_cam: bool) -> np.ndarray:
    cols = [FixedCamTrajCol.X.value, FixedCamTrajCol.Y.value, FixedCamTrajCol.Z.value] if is_fixed_cam else [
        TrajCol.X_FILT.value, TrajCol.Y_FILT.value, TrajCol.Z_FILT.value]
    delta = left[:, cols] - right[:, cols]
    return np.sqrt(np.sum(delta * delta, axis=1))
