from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from reachx.common import RxHandPos, RxReach, RxReachSegment
from reachx.modeling.segmentation.metadata import DecoderConfig


@dataclass
class DecodeReport:
    decoded_count: int = 0
    valid_count: int = 0
    dropped: List[dict] = field(default_factory=list)


def decode_reaches(frames: np.ndarray, probabilities: np.ndarray, decoder: DecoderConfig) -> tuple[List[RxReachSegment], DecodeReport]:
    if len(frames) != len(probabilities):
        raise ValueError("Frame and probability arrays have different lengths")
    report = DecodeReport()
    active = [int(frame) for frame, prob in zip(frames, probabilities) if float(prob) >= decoder.threshold]
    groups = _merge_active_frames(active, decoder.merge_gap_frames)
    reaches: List[RxReachSegment] = []
    score_by_frame = {int(frame): float(prob) for frame, prob in zip(frames, probabilities)}
    for group in groups:
        start = int(group[0])
        end = int(group[-1])
        duration = end - start
        report.decoded_count += 1
        if duration < decoder.min_duration_frames:
            report.dropped.append({"start": start, "end": end, "reason": "too_short"})
            continue
        interior = [frame for frame in group if start < frame < end]
        if len(interior) == 0:
            report.dropped.append({"start": start, "end": end, "reason": "no_interior_reach_max"})
            continue
        max_frame = max(interior, key=lambda frame: score_by_frame.get(frame, 0.0))
        seg = RxReachSegment(start, max_frame - start, duration, RxReach.NONE, RxHandPos.UNSPECIFIED)
        if not RxReachSegment.is_valid(seg):
            report.dropped.append({"start": start, "end": end, "reason": "invalid_reachx_segment"})
            continue
        reaches.append(seg)
    report.valid_count = len(reaches)
    return reaches, report


def _merge_active_frames(active_frames: List[int], max_gap_frames: int) -> List[List[int]]:
    if len(active_frames) == 0:
        return []
    active_frames = sorted(active_frames)
    groups: List[List[int]] = [[active_frames[0]]]
    for frame in active_frames[1:]:
        if frame - groups[-1][-1] <= max(1, max_gap_frames + 1):
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups
