from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class PredictionStoreResult:
    formats_written: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def write_predictions(out_dir: Path, frames: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> PredictionStoreResult:
    result = PredictionStoreResult()
    columns = ["frame", "reach_probability", "predicted_label"]
    matrix = np.column_stack(
        (
            frames.astype(np.int32),
            probabilities.astype(np.float32),
            labels.astype(np.int32),
        )
    )

    np.save(Path(out_dir, "framewise_predictions.npy"), matrix)
    with open(Path(out_dir, "framewise_predictions_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"columns": columns, "dtype": str(matrix.dtype), "shape": list(matrix.shape)}, f, indent=2)
        f.write("\n")
    result.formats_written.append("npy")

    try:
        import pandas as pd
        frame = pd.DataFrame({
            "frame": frames.astype(np.int32),
            "reach_probability": probabilities.astype(np.float32),
            "predicted_label": labels.astype(np.int32),
        })
        try:
            frame.to_hdf(Path(out_dir, "framewise_predictions.h5"), key="predictions", mode="w")
            result.formats_written.append("hdf")
        except Exception as exc:
            result.errors.append(f"hdf: {exc}")
        try:
            frame.to_parquet(Path(out_dir, "framewise_predictions.parquet"), index=False)
            result.formats_written.append("parquet")
        except Exception as exc:
            result.errors.append(f"parquet: {exc}")
    except Exception as exc:
        result.errors.append(f"pandas: {exc}")

    return result
