"""
src.algorithms — unified community detection algorithm interface.
"""
from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.factory import load_algorithms
from src.algorithms.registry import ALGORITHM_REGISTRY, register
from src.algorithms.wrappers import DynamicMethodWrapper, StaticMethodWrapper

__all__ = [
    "CommunityDetectionAlgorithm",
    "StaticMethodWrapper",
    "DynamicMethodWrapper",
    "load_algorithms",
    "register",
    "ALGORITHM_REGISTRY",
]
