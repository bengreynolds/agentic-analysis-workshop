from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from reachx.common import RxSessionID
from reachx.modeling.segmentation.decoder import decode_reaches
from reachx.modeling.segmentation.metadata import load_model_metadata
from reachx.modeling.segmentation.predictors import predict_probabilities
from reachx.modeling.segmentation.result_store import write_segmentation_result
from reachx.modeling.segmentation.trajectory_adapter import load_segmentation_input


def run_model_segmentation(
        sesh_id: RxSessionID,
        rx_scorer: str,
        model_root: Path,
        task: Optional[Any] = None,
        result_id: Optional[str] = None) -> Path:
    def emit(progress: int, message: str) -> None:
        if task is not None:
            task.progress_updated.emit(progress)
            task.message_logged.emit(message)

    def check_cancel() -> None:
        if task is not None and task.was_canceled():
            raise RuntimeError("Model-based reach segmentation canceled")

    scorer_folder = Path(sesh_id.location, rx_scorer)
    if not scorer_folder.is_dir():
        raise FileNotFoundError(f"Scorer folder not found: {scorer_folder}")

    emit(5, f"Loading segmentation model metadata from {model_root}...")
    metadata = load_model_metadata(Path(model_root))
    check_cancel()

    emit(20, "Preparing segmentation features from scorer outputs...")
    input_table = load_segmentation_input(sesh_id, scorer_folder, metadata)
    check_cancel()

    emit(45, f"Running {metadata.engine} segmentation model...")
    probabilities = predict_probabilities(metadata, input_table.feature_matrix)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if len(probabilities) != len(input_table.frames):
        raise ValueError("Segmentation model returned a probability count that does not match the frame count")
    check_cancel()

    emit(70, "Decoding model probabilities into ReachX segments...")
    reaches, report = decode_reaches(input_table.frames, probabilities, metadata.decoder)
    check_cancel()

    emit(85, "Writing segmentation result folder...")
    result_folder = write_segmentation_result(
        scorer_folder,
        result_id,
        reaches,
        metadata,
        input_table.frames,
        probabilities,
        report,
    )
    emit(100, f"Wrote model segmentation result: {result_folder.name}")
    return result_folder
