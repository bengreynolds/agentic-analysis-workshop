from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from reachx.common import RX
from reachx.modeling.segmentation.metadata import load_model_metadata, write_json


@dataclass(frozen=True)
class SegmentationModelRecord:
    model_id: str
    display_name: str
    model_root: Path
    session_type: str
    engine: str
    storage_mode: str
    registry_path: Optional[Path] = None


def default_segmentation_model_root() -> Path:
    root = Path(RX.HOME, "segmentation_models")
    root.mkdir(parents=True, exist_ok=True)
    return root


def register_external_model(model_root: Path, registry_root: Optional[Path] = None) -> Path:
    metadata = load_model_metadata(model_root)
    registry_root = Path(registry_root) if registry_root is not None else default_segmentation_model_root()
    registry_root.mkdir(parents=True, exist_ok=True)
    record: Dict[str, object] = {
        "model_id": metadata.model_id,
        "storage_mode": "external_reference",
        "artifact_root": str(Path(model_root).absolute()),
        "metadata_path": str(Path(model_root, "model.json").absolute()),
    }
    out = Path(registry_root, f"{_safe_name(metadata.model_id)}.ref.json")
    write_json(out, record)
    return out


def discover_segmentation_models(root: Path) -> List[SegmentationModelRecord]:
    root = Path(root)
    if not root.is_dir():
        return []

    records_by_root: Dict[str, SegmentationModelRecord] = {}
    for model_dir in _find_model_dirs(root):
        try:
            record = _record_from_model_dir(model_dir, "local_folder")
            records_by_root[str(record.model_root.resolve())] = record
        except Exception:
            pass

    for ref_path in sorted(root.glob("*.ref.json")):
        try:
            record = _record_from_reference(ref_path)
            records_by_root[str(record.model_root.resolve())] = record
        except Exception:
            pass

    out = list(records_by_root.values())
    out.sort(key=lambda record: record.display_name.lower())
    return out


def find_segmentation_model(root: Path, model_root: Path) -> Optional[SegmentationModelRecord]:
    try:
        target = Path(model_root).resolve()
    except Exception:
        return None
    return next((record for record in discover_segmentation_models(root)
                 if record.model_root.resolve() == target), None)


def _find_model_dirs(root: Path) -> List[Path]:
    found: List[Path] = []
    if Path(root, "model.json").is_file():
        found.append(root)
    for metadata_file in sorted(root.rglob("model.json")):
        model_dir = metadata_file.parent
        if model_dir not in found:
            found.append(model_dir)
    return found


def _record_from_model_dir(model_dir: Path, storage_mode: str,
                           registry_path: Optional[Path] = None) -> SegmentationModelRecord:
    metadata = load_model_metadata(model_dir)
    display_name = metadata.display_name or metadata.model_id
    if model_dir.name not in {metadata.model_id, display_name}:
        display_name = f"{display_name}/{model_dir.name}"
    display_name = f"{display_name} ({metadata.engine})"
    return SegmentationModelRecord(
        model_id=metadata.model_id,
        display_name=display_name,
        model_root=Path(model_dir),
        session_type=metadata.session_type,
        engine=metadata.engine,
        storage_mode=storage_mode,
        registry_path=registry_path,
    )


def _record_from_reference(ref_path: Path) -> SegmentationModelRecord:
    with open(ref_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    model_root = Path(raw["artifact_root"])
    return _record_from_model_dir(model_root, str(raw.get("storage_mode", "external_reference")), ref_path)


def _safe_name(value: str) -> str:
    chars = [c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value]
    safe = "".join(chars).strip("._")
    return safe or "segmentation_model"
