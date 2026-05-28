"""
Inference-only reach segmentation support.

This package consumes externally trained segmentation models. It intentionally does not
train models inside ReachX.
"""

from reachx.modeling.segmentation.registry import default_segmentation_model_root

__all__ = ["default_segmentation_model_root", "run_model_segmentation"]


def run_model_segmentation(*args, **kwargs):
    from reachx.modeling.segmentation.workflow import run_model_segmentation as _run_model_segmentation
    return _run_model_segmentation(*args, **kwargs)
