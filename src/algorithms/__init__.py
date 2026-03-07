"""
src.algorithms — unified community detection algorithm interface.
"""
from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.factory import load_algorithms
from src.algorithms.wrappers import DynamicMethodWrapper, StaticMethodWrapper

__all__ = [
    "CommunityDetectionAlgorithm",
    "StaticMethodWrapper",
    "DynamicMethodWrapper",
    "load_algorithms",
]
