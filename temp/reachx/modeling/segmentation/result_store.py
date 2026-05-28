from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from reachx.common import RxReachSegment
from reachx.modeling.segmentation.decoder import DecodeReport
from reachx.modeling.segmentation.metadata import SegmentationModelMetadata, write_json
from reachx.modeling.segmentation.prediction_store import write_predictions


RESULTS_FOLDER_NAME = "reach_segmentations"


def make_result_id(model_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _safe_path_name(f"{model_id}_{stamp}")


def write_segmentation_result(
        scorer_folder: Path,
        result_id: Optional[str],
        reaches: List[RxReachSegment],
        metadata: SegmentationModelMetadata,
        frames: np.ndarray,
        probabilities: np.ndarray,
        report: DecodeReport) -> Path:
    result_id = _safe_path_name(result_id or make_result_id(metadata.model_id))
    results_root = Path(scorer_folder, RESULTS_FOLDER_NAME)
    results_root.mkdir(parents=True, exist_ok=True)
    final_root = Path(results_root, result_id)
    if final_root.exists():
        raise FileExistsError(f"Segmentation result already exists: {final_root}")
    temp_root = Path(results_root, f".tmp_{result_id}_{uuid.uuid4().hex}")
    temp_root.mkdir(parents=True)
    try:
        reaches_path = Path(temp_root, "detected_reaches.txt")
        emsg = RxReachSegment.save_reach_segments(reaches, reaches_path)
        if len(emsg) > 0:
            raise RuntimeError(emsg)
        readback_msg, readback_reaches = RxReachSegment.load_reach_segments(reaches_path)
        if len(readback_msg) > 0:
            raise RuntimeError(readback_msg)
        if readback_reaches != reaches:
            raise RuntimeError("Saved reach segments failed ReachX readback validation")
        labels = (probabilities >= metadata.decoder.threshold).astype(np.int32)
        prediction_result = write_predictions(temp_root, frames, probabilities, labels)
        write_json(Path(temp_root, "model_metadata.json"), metadata.raw)
        summary: Dict[str, object] = {
            "result_id": result_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": metadata.summary_dict(),
            "source": "ml_segmentation",
            "frame_count": int(len(frames)),
            "decoded_count": report.decoded_count,
            "valid_count": report.valid_count,
            "readback_count": len(readback_reaches),
            "dropped": report.dropped,
            "prediction_formats": prediction_result.formats_written,
            "prediction_store_errors": prediction_result.errors,
            "detected_reaches_path": "detected_reaches.txt",
        }
        with open(Path(temp_root, "segmentation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        temp_root.rename(final_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return final_root


def _safe_path_name(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(value)).strip("._")
    if not safe:
        raise ValueError("Result ID cannot be empty")
    if safe in {".", ".."}:
        raise ValueError("Invalid result ID")
    return safe
