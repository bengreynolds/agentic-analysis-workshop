from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from reachx.common import RX


@dataclass(frozen=True)
class DecoderConfig:
    threshold: float
    min_duration_frames: int
    merge_gap_frames: int
    result_for_binary_events: str = "none"


@dataclass(frozen=True)
class SegmentationModelMetadata:
    model_root: Path
    model_id: str
    display_name: str
    engine: str
    artifact: Optional[str]
    session_type: str
    features: List[str]
    required_body_parts: List[str]
    decoder: DecoderConfig
    raw: Dict[str, Any]

    @property
    def artifact_path(self) -> Optional[Path]:
        return None if self.artifact is None else Path(self.model_root, self.artifact)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "engine": self.engine,
            "artifact": self.artifact,
            "session_type": self.session_type,
            "features": self.features,
            "required_body_parts": self.required_body_parts,
            "decoder": {
                "threshold": self.decoder.threshold,
                "min_duration_frames": self.decoder.min_duration_frames,
                "merge_gap_frames": self.decoder.merge_gap_frames,
                "result_for_binary_events": self.decoder.result_for_binary_events,
            },
        }


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_model_metadata(model_root: Path) -> SegmentationModelMetadata:
    model_root = Path(model_root)
    metadata_path = Path(model_root, "model.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing segmentation model metadata file: {metadata_path}")

    raw = read_json(metadata_path)
    nested_metadata = dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {}
    feature_builder = dict(raw.get("feature_builder", {})) if isinstance(raw.get("feature_builder"), dict) else {}
    feature_spec = _coerce_feature_spec(raw, nested_metadata, feature_builder)

    engine = _normalize_engine(
        raw.get("engine")
        or raw.get("backend")
        or raw.get("family")
        or nested_metadata.get("backend")
        or nested_metadata.get("method_id")
    )
    if not engine:
        raise ValueError("Segmentation model metadata is missing engine/backend/method_id")

    model_id = str(raw.get("model_id") or nested_metadata.get("model_name") or model_root.name)
    display_name = str(raw.get("display_name") or nested_metadata.get("model_name") or model_id)
    artifact = _find_artifact(raw, engine)
    features = [str(item) for item in feature_spec.get("feature_names", feature_builder.get("features", []))]
    if not features:
        raise ValueError("Segmentation model metadata is missing feature names")
    session_type = str(
        raw.get("session_type")
        or nested_metadata.get("session_type")
        or feature_builder.get("session_type")
        or _infer_session_type(features)
    ).lower()

    required_body_parts = [
        str(item) for item in feature_builder.get("required_body_parts", feature_spec.get("keypoints", []))
    ]
    if not required_body_parts:
        required_body_parts = _infer_required_body_parts(features)

    decoder_raw = dict(raw.get("decoder", {})) if isinstance(raw.get("decoder"), dict) else {}
    threshold = float(decoder_raw.get("threshold", nested_metadata.get("decision_threshold", 0.5)))
    min_duration = int(
        decoder_raw.get(
            "min_duration_frames",
            decoder_raw.get("min_event_frames", nested_metadata.get("min_event_frames", RX.MIN_REACHSEG_DUR)),
        )
    )
    merge_gap = int(decoder_raw.get("merge_gap_frames", decoder_raw.get("max_gap_frames",
                                                                        nested_metadata.get("max_gap_frames", 0))))
    result_for_binary = str(decoder_raw.get("result_for_binary_events", "none")).lower()
    decoder = DecoderConfig(threshold, max(RX.MIN_REACHSEG_DUR, min_duration), max(0, merge_gap), result_for_binary)

    if artifact is not None and not Path(model_root, artifact).is_file():
        raise FileNotFoundError(f"Segmentation model artifact not found: {Path(model_root, artifact)}")

    return SegmentationModelMetadata(
        model_root=model_root,
        model_id=model_id,
        display_name=display_name,
        engine=engine,
        artifact=artifact,
        session_type=session_type,
        features=features,
        required_body_parts=required_body_parts,
        decoder=decoder,
        raw=raw,
    )


def _coerce_feature_spec(
        raw: Dict[str, Any],
        nested_metadata: Dict[str, Any],
        feature_builder: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw.get("feature_spec"), dict):
        return dict(raw["feature_spec"])
    if isinstance(nested_metadata.get("feature_spec"), dict):
        return dict(nested_metadata["feature_spec"])
    out: Dict[str, Any] = {}
    if "features" in feature_builder:
        out["feature_names"] = list(feature_builder["features"])
    if "required_body_parts" in feature_builder:
        out["keypoints"] = list(feature_builder["required_body_parts"])
    return out


def _normalize_engine(value: object) -> str:
    engine = str(value or "").lower().replace("-", "_")
    if engine in {"rf", "randomforest"}:
        return "random_forest"
    if engine in {"sklearn_random_forest", "sklearn"}:
        return "random_forest"
    if engine in {"xgb", "xgboost_classifier"}:
        return "xgboost"
    if engine in {"torch", "pytorch", "pytorch_temporal"}:
        return "torch_temporal"
    return engine


def _find_artifact(raw: Dict[str, Any], engine: str) -> Optional[str]:
    for key in ["artifact", "classifier_artifact", "keras_model_path", "weights_path", "model_path"]:
        if raw.get(key):
            return str(raw[key])
    defaults = {
        "xgboost": "xgboost_model.json",
        "random_forest": "random_forest.joblib",
        "tensorflow": "tf_model.keras",
        "torch_temporal": "torch_temporal_model.pt",
    }
    return defaults.get(engine)


def _infer_session_type(features: List[str]) -> str:
    lower = [feature.lower() for feature in features]
    if any(feature.startswith(("front_", "side_")) for feature in lower):
        return "legacy_cam"
    if any(feature.startswith(("left_", "right_")) for feature in lower):
        return "fixed_cam"
    return "any"


def _infer_required_body_parts(features: List[str]) -> List[str]:
    out: List[str] = []
    lower = [feature.lower() for feature in features]
    if any("r_hand" in feature or "right_hand" in feature for feature in lower):
        out.append("R_Hand")
    if any("l_hand" in feature or "left_hand" in feature for feature in lower):
        out.append("L_Hand")
    if any("hand" in feature for feature in lower) and "R_Hand" not in out:
        out.append("R_Hand")
    if any("pellet" in feature for feature in lower):
        out.append("pellet")
    return out
