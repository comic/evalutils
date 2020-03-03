from .__version__ import __version__ as _version
from .evalutils import (
    ClassificationAlgorithm,
    ClassificationEvaluation,
    DetectionAlgorithm,
    DetectionEvaluation,
    Evaluation,
    SegmentationAlgorithm,
)

__author__ = """James Meakin"""
__email__ = "jamesmeakin@gmail.com"
__version__ = _version
__all__ = [
    "ClassificationAlgorithm",
    "ClassificationEvaluation",
    "DetectionAlgorithm",
    "DetectionEvaluation",
    "Evaluation",
    "SegmentationAlgorithm",
]
