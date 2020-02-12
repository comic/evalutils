from .__version__ import __version__ as _version
from .evalutils import (
    BaseAlgorithm,
    ClassificationEvaluation,
    DetectionEvaluation,
    Evaluation,
)

__author__ = """James Meakin"""
__email__ = "jamesmeakin@gmail.com"
__version__ = _version
__all__ = [
    "ClassificationEvaluation",
    "DetectionEvaluation",
    "Evaluation",
    "BaseAlgorithm",
]
