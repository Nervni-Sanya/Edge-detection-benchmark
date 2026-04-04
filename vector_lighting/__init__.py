"""
Vector Lighting Edge Detector Library

Библиотека для выделения границ на цветных изображениях методом векторного освещения.
Содержит классические методы (Sobel, Prewitt, Canny) и авторский метод vector_lighting.

Author: Панченко Александр Алексеевич
Version: 1.0.0
Repository: https://github.com/Nervni-Sanya/Edge-detection-benchmark
"""

from .core import (
    sobel,
    prewitt,
    canny,
    vector_lighting,
    EdgeDetector,
    __version__,
    __author__
)


__all__ = [
    "sobel",
    "prewitt",
    "canny",
    "vector_lighting",
    "EdgeDetector",
    "__version__",
    "__author__"
]