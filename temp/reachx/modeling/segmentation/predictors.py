from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np

from reachx.modeling.segmentation.metadata import SegmentationModelMetadata


def predict_probabilities(metadata: SegmentationModelMetadata, feature_matrix: np.ndarray) -> np.ndarray:
    engine = metadata.engine
    if engine in {"random_forest", "sklearn"}:
        return _predict_sklearn(metadata, feature_matrix)
    if engine == "xgboost":
        return _predict_xgboost(metadata, feature_matrix)
    if engine == "tensorflow":
        return _predict_tensorflow(metadata, feature_matrix)
    if engine == "torch_temporal":
        return _predict_torch_temporal(metadata, feature_matrix)
    raise ValueError(f"Unsupported segmentation model engine: {engine}")


def _predict_sklearn(metadata: SegmentationModelMetadata, feature_matrix: np.ndarray) -> np.ndarray:
    if metadata.artifact_path is None:
        raise ValueError("Sklearn segmentation model metadata is missing classifier artifact")
    model = _load_joblib_model(metadata.artifact_path)
    return _probabilities_from_model(model, feature_matrix)


def _predict_xgboost(metadata: SegmentationModelMetadata, feature_matrix: np.ndarray) -> np.ndarray:
    if metadata.artifact_path is None:
        raise ValueError("XGBoost segmentation model metadata is missing classifier artifact")
    if metadata.artifact_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        model = _load_joblib_model(metadata.artifact_path)
        return _probabilities_from_model(model, feature_matrix)
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(metadata.artifact_path)
        return _probabilities_from_model(model, feature_matrix)
    except Exception:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(metadata.artifact_path))
        return np.asarray(booster.predict(xgb.DMatrix(feature_matrix)), dtype=np.float32).reshape(-1)


def _predict_tensorflow(metadata: SegmentationModelMetadata, feature_matrix: np.ndarray) -> np.ndarray:
    if metadata.artifact_path is None:
        raise ValueError("TensorFlow segmentation model metadata is missing Keras artifact")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for the selected segmentation model") from exc
    model = tf.keras.models.load_model(metadata.artifact_path)
    probabilities = model.predict(feature_matrix, verbose=0)
    return _coerce_probability_array(probabilities)


def _predict_torch_temporal(metadata: SegmentationModelMetadata, feature_matrix: np.ndarray) -> np.ndarray:
    if metadata.artifact_path is None:
        raise ValueError("Torch segmentation model metadata is missing weights artifact")
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for the selected segmentation model") from exc

    payload = metadata.raw
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(payload.get("artifact_type", "")).lower() in {"torchscript", "jit"}:
        model = torch.jit.load(str(metadata.artifact_path), map_location=device)
    else:
        model = _build_temporal_conv_model(
            torch,
            nn,
            feature_count=len(metadata.features),
            channels=tuple(int(item) for item in payload.get("channels", (64, 64, 64))),
            kernel_size=int(payload.get("kernel_size", 5)),
            dropout=float(payload.get("dropout", 0.1)),
            model_family=str(payload.get("model_family", "temporal_conv")),
            dilations=tuple(int(item) for item in payload.get("dilations", (1, 2, 4))),
        )
        try:
            state_dict = torch.load(metadata.artifact_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(metadata.artifact_path, map_location=device)
        model.load_state_dict(state_dict)

    feature_matrix = _standardize(feature_matrix, payload)
    context_frames = int(payload.get("context_frames", 32))
    batch_size = int(payload.get("batch_size", 512))
    features = torch.as_tensor(feature_matrix, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), max(1, batch_size)):
            batch = _torch_window_batch(
                torch, features, context_frames, start, min(start + max(1, batch_size), int(features.shape[0]))
            ).to(device)
            logits = model(batch)
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return np.asarray(probabilities, dtype=np.float32).reshape(-1)


def _import_joblib() -> Any:
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required for the selected segmentation model") from exc
    return joblib


def _load_joblib_model(path: Path) -> Any:
    joblib = _import_joblib()
    _install_numpy_pickle_compatibility_aliases()
    try:
        return joblib.load(path)
    except ValueError as exc:
        if "incompatible dtype" in str(exc) and "missing_go_to_left" in str(exc):
            raise RuntimeError(
                "This sklearn segmentation model was saved with scikit-learn 1.7+; "
                "install the backend requirements before loading it."
            ) from exc
        raise


def _install_numpy_pickle_compatibility_aliases() -> None:
    import sys

    # NumPy 2.x pickles may reference numpy._core. ReachX currently pins NumPy 1.x
    # for the existing TensorFlow stack, where the equivalent package is numpy.core.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    for name in ("multiarray", "numeric", "numerictypes", "_multiarray_umath", "fromnumeric", "shape_base"):
        module = getattr(np.core, name, None)
        if module is not None:
            sys.modules.setdefault(f"numpy._core.{name}", module)


def _probabilities_from_model(model: Any, feature_matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return _coerce_probability_array(model.predict_proba(feature_matrix))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(feature_matrix), dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-scores))
    if hasattr(model, "predict"):
        return _coerce_probability_array(model.predict(feature_matrix))
    raise RuntimeError("Selected segmentation model does not expose predict_proba, decision_function, or predict")


def _coerce_probability_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr[:, -1]
    elif arr.ndim == 2:
        arr = arr[:, 0]
    return arr.reshape(-1)


def _standardize(feature_matrix: np.ndarray, payload: dict) -> np.ndarray:
    mean = payload.get("feature_mean", ())
    scale = payload.get("feature_scale", ())
    if not mean or not scale:
        return np.asarray(feature_matrix, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    return (np.asarray(feature_matrix, dtype=np.float32) - mean_arr) / scale_arr


def _torch_window_batch(torch: Any, features: Any, context_frames: int, first_frame: int, stop_frame: int) -> Any:
    windows = []
    for frame_index in range(first_frame, stop_frame):
        start = frame_index - context_frames
        end = frame_index + context_frames + 1
        left_pad = max(0, -start)
        right_pad = max(0, end - int(features.shape[0]))
        start = max(0, start)
        end = min(int(features.shape[0]), end)
        window = features[start:end]
        if left_pad or right_pad:
            window = torch.nn.functional.pad(window, (0, 0, left_pad, right_pad))
        windows.append(window)
    return torch.stack(windows)


def _build_temporal_conv_model(
        torch: Any,
        nn: Any,
        feature_count: int,
        channels: Tuple[int, ...],
        kernel_size: int,
        dropout: float,
        model_family: str,
        dilations: Tuple[int, ...]) -> Any:
    class ResidualTemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (kernel_size // 2)
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

        def forward(self, inputs: Any) -> Any:
            return self.net(inputs) + self.residual(inputs)

    class TemporalConvModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_channels = feature_count
            if model_family == "dilated_tcn":
                for index, out_channels in enumerate(channels):
                    dilation = dilations[index % len(dilations)] if dilations else 1
                    layers.append(ResidualTemporalBlock(in_channels, out_channels, dilation=dilation))
                    in_channels = out_channels
            elif model_family == "temporal_conv":
                for out_channels in channels:
                    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=kernel_size // 2))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    in_channels = out_channels
            else:
                raise ValueError(f"Unsupported torch temporal model_family: {model_family}")
            self.encoder = nn.Sequential(*layers)
            self.classifier = nn.Linear(in_channels, 1)

        def forward(self, inputs: Any) -> Any:
            encoded = self.encoder(inputs.transpose(1, 2))
            center = encoded[:, :, encoded.shape[-1] // 2]
            return self.classifier(center).squeeze(-1)

    return TemporalConvModel()
