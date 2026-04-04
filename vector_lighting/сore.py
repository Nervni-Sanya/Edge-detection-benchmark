# vector_lighting/core.py
"""
Vector Lighting Edge Detector Library

Библиотека для выделения границ на изображениях.
Содержит классические методы (Sobel, Prewitt, Canny) и авторский метод
векторного освещения для цветных изображений.
"""

from typing import List, Tuple, Optional, Literal, Union
from itertools import permutations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_opening, binary_dilation
from scipy.signal import correlate2d

__version__ = "1.0.0"
__author__ = "Панченко Александр Алексеевич"


def _normalize_image(image: np.ndarray, target_range: Tuple[float, float] = (0, 255)) -> np.ndarray:
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-10:
        return np.full_like(image, target_range[0], dtype=np.float64)
    normalized = (image - img_min) / (img_max - img_min)
    return normalized * (target_range[1] - target_range[0]) + target_range[0]


def _rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return image


###############################################################################
#                              SOBEL OPERATOR                                 #
###############################################################################

def sobel(image: np.ndarray) -> np.ndarray:
    """
    Оператор Собеля для выделения границ.
    
    Args:
        image: RGB или grayscale изображение.
    
    Returns:
        Магнитуда градиента (uint8).
    """
    gray = _rgb_to_grayscale(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gx = correlate2d(gray.astype(float), sobel_x, mode='same', boundary='symm')
    gy = correlate2d(gray.astype(float), sobel_y, mode='same', boundary='symm')
    magnitude = np.sqrt(gx**2 + gy**2)
    if magnitude.max() > 0:
        magnitude = _normalize_image(magnitude, (0, 255))
    return magnitude.astype(np.uint8)

###############################################################################
#                             PREWITT OPERATOR                                #
###############################################################################

def prewitt(image: np.ndarray) -> np.ndarray:
    """
    Оператор Превитта для выделения границ.
    
    Args:
        image: RGB или grayscale изображение.
    
    Returns:
        Магнитуда градиента (uint8).
    """
    gray = _rgb_to_grayscale(image)
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
    gx = correlate2d(gray.astype(float), prewitt_x, mode='same', boundary='symm')
    gy = correlate2d(gray.astype(float), prewitt_y, mode='same', boundary='symm')
    magnitude = np.sqrt(gx**2 + gy**2)
    if magnitude.max() > 0:
        magnitude = _normalize_image(magnitude, (0, 255))
    return magnitude.astype(np.uint8)


###############################################################################
#                              CANNY DETECTOR                                 #
###############################################################################

def canny(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 100, sigma: float = 1.0) -> np.ndarray:
    """
    Детектор границ Кэнни.
    
    Args:
        image: RGB или grayscale изображение.
        low_threshold: Нижний порог гистерезиса.
        high_threshold: Верхний порог гистерезиса.
        sigma: Параметр Гауссова размытия.

    Returns:
        Бинарная карта границ (uint8).
    """
    gray = _rgb_to_grayscale(image)
    smoothed = gaussian_filter(gray.astype(float), sigma=sigma)
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gx = correlate2d(smoothed, sobel_x, mode='same', boundary='symm')
    gy = correlate2d(smoothed, sobel_y, mode='same', boundary='symm')
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    # Non-maximum suppression
    suppressed = np.zeros_like(magnitude)
    angle = np.rad2deg(direction) % 180
    mask_h = (angle < 22.5) | (angle >= 157.5)
    mask_d1 = (angle >= 22.5) & (angle < 67.5)
    mask_v = (angle >= 67.5) & (angle < 112.5)
    mask_d2 = (angle >= 112.5) & (angle < 157.5)
    
    mag_up = np.roll(magnitude, -1, axis=0)
    mag_down = np.roll(magnitude, 1, axis=0)
    mag_left = np.roll(magnitude, -1, axis=1)
    mag_right = np.roll(magnitude, 1, axis=1)
    mag_ul = np.roll(np.roll(magnitude, -1, axis=0), -1, axis=1)
    mag_ur = np.roll(np.roll(magnitude, -1, axis=0), 1, axis=1)
    mag_dl = np.roll(np.roll(magnitude, 1, axis=0), -1, axis=1)
    mag_dr = np.roll(np.roll(magnitude, 1, axis=0), 1, axis=1)
    
    suppressed[mask_h] = magnitude[mask_h] * ((magnitude[mask_h] >= mag_left[mask_h]) & (magnitude[mask_h] >= mag_right[mask_h]))
    suppressed[mask_d1] = magnitude[mask_d1] * ((magnitude[mask_d1] >= mag_ul[mask_d1]) & (magnitude[mask_d1] >= mag_dr[mask_d1]))
    suppressed[mask_v] = magnitude[mask_v] * ((magnitude[mask_v] >= mag_up[mask_v]) & (magnitude[mask_v] >= mag_down[mask_v]))
    suppressed[mask_d2] = magnitude[mask_d2] * ((magnitude[mask_d2] >= mag_ur[mask_d2]) & (magnitude[mask_d2] >= mag_dl[mask_d2]))
    suppressed[0, :] = suppressed[-1, :] = suppressed[:, 0] = suppressed[:, -1] = 0
    
    # Hysteresis thresholding
    strong = suppressed >= high_threshold
    weak = (suppressed >= low_threshold) & (suppressed < high_threshold)
    edges = strong.copy()
    
    for _ in range(100):
        dilated = binary_dilation(edges, structure=np.ones((3, 3)))
        new_edges = dilated & weak & ~edges
        if not np.any(new_edges):
            break
        edges |= new_edges
    
    return edges.astype(np.uint8) * 255


###############################################################################
#                    CUSTOM VECTOR LIGHTING EDGE DETECTOR                     #
###############################################################################

def _get_light_vectors(mode: int) -> List[Tuple[float, float]]:
    vectors_map = [
        [(1, 1)],
        [(1, 1), (1, -1)],
        [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1)]
    ]
    if not (0 <= mode < len(vectors_map)):
        raise ValueError(f"mode должен быть от 0 до {len(vectors_map)-1}, получено {mode}")
    vectors = []
    for dx, dy in vectors_map[mode]:
        norm = np.sqrt(dx**2 + dy**2)
        vectors.append((dx / norm, dy / norm))
    return vectors


def _calculate_global_threshold(channels: List[np.ndarray],
                                method: Literal['mean_std', 'median', 'percentile'] = 'percentile',
                                threshold_factor: float = 0.25) -> float:
    match method:
        case 'mean_std':
            means = [np.mean(ch) for ch in channels]
            stds = [np.std(ch) for ch in channels]
            zero_mag = np.sqrt(sum(m**2 for m in means))
            threshold = zero_mag + threshold_factor * np.mean(stds)
            return float(np.clip(threshold, 0, 255))
        case 'median':
            all_values = np.concatenate([ch.ravel() for ch in channels])
            return float(np.median(all_values))
        case 'percentile':
            all_values = np.concatenate([ch.ravel() for ch in channels])
            return float(np.percentile(all_values, 75))
        case _:
            raise ValueError(f"Неизвестный метод: {method}")


def _vector_lighting_vectorized(channel_x: np.ndarray, channel_y: np.ndarray, channel_z: np.ndarray,
                               light_vectors: List[Tuple[float, float]], threshold: float, height_weight: float = 1.0) -> np.ndarray:
    grad_x_x, grad_x_y = np.gradient(channel_x.astype(float))
    grad_y_x, grad_y_y = np.gradient(channel_y.astype(float))
    height_factor = 1.0 + height_weight * (channel_z - threshold) / 255.0
    height_factor = np.clip(height_factor, 0.1, 3.0)
    accumulated = np.zeros_like(channel_x, dtype=float)
    for dx, dy in light_vectors:
        resp_x = grad_x_x * dx + grad_x_y * dy
        resp_y = grad_y_x * dx + grad_y_y * dy
        combined = np.sqrt(resp_x**2 + resp_y**2)
        accumulated += combined * height_factor
    accumulated /= len(light_vectors)
    return accumulated


def _merge_edge_maps(edge_maps: List[np.ndarray],
                     method: Literal['mean', 'max', 'weighted', 'adaptive_mean'] = 'max',
                     weights: Optional[np.ndarray] = None) -> np.ndarray:
    if not edge_maps:
        raise ValueError("edge_maps не может быть пустым")
    maps_float = [m.astype(float) for m in edge_maps]
    match method:
        case 'mean':
            merged = np.mean(maps_float, axis=0)
        case 'max':
            merged = np.maximum.reduce(maps_float)
        case 'weighted':
            if weights is None or len(weights) != len(edge_maps):
                raise ValueError("Для метода 'weighted' необходимо задать weights")
            weights = np.array(weights) / np.sum(weights)
            merged = sum(m * w for m, w in zip(maps_float, weights))
        case 'adaptive_mean':
            variances = np.array([np.var(m) for m in maps_float])
            w = variances / (variances + 1e-6)
            w /= w.sum()
            merged = sum(m * wi for m, wi in zip(maps_float, w))
        case _:
            raise ValueError(f"Неизвестный метод слияния: {method}")
    if merged.max() > merged.min():
        merged = _normalize_image(merged, (0, 255))
    return merged


def vector_lighting(image: np.ndarray,
                   mode: int = 3,
                   use_permutations: bool = True,
                   merge_method: Literal['mean', 'max', 'weighted', 'adaptive_mean'] = 'max',
                   threshold_method: Literal['mean_std', 'median', 'percentile'] = 'percentile',
                   threshold_factor: float = 0.25,
                   height_weight: float = 1.0,
                   binary: bool = True,
                   binary_percentile: float = 0.05,
                   sigma: float = 1.0,
                   clean_noise: bool = False,
                   channel_roles: Optional[Tuple[int, int, int]] = None,
                   return_debug: bool = False) -> Union[np.ndarray, dict]:
    """
    Выделение границ методом векторного освещения RGB-каналов.
    
    Концепция:
        RGB-каналы интерпретируются как трёхмерное пространство поверхностей.
        Два канала задают градиенты, третий модулирует отклик по высоте.
        Виртуальное освещение с разных направлений выявляет границы,
        включая изолюминантные переходы (цвет без изменения яркости).
    
    Args:
        image: RGB изображение (H, W, 3).
        mode: Режим освещения (0-3). 0=1 вектор, 1=2, 2=4, 3=8 векторов.
        use_permutations: Перебирать все перестановки каналов.
        merge_method: Метод слияния карт ('mean', 'max', 'adaptive_mean', 'weighted').
        threshold_method: Метод порога модуляции ('mean_std', 'median', 'percentile').
        threshold_factor: Множитель для mean_std.
        height_weight: Вес влияния высоты на отклик.
        binary: Применять бинаризацию.
        binary_percentile: Процент пикселей для бинаризации. 
            0.05 = оставить 5% самых ярких. 0.0 = порог по среднему.
            ⚠️ На градиентах используйте 0.0 для избежания ложных срабатываний.
        sigma: Гауссово сглаживание перед вычислением градиентов.
        clean_noise: Морфологическая очистка (не рекомендуется, удаляет тонкие границы).
        channel_roles: Явные роли каналов (x, y, z). Игнорирует use_permutations.
        return_debug: Вернуть отладочную информацию.
    
    Returns:
        Бинарная карта границ (uint8) или dict с отладкой.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("image должен быть RGB изображением формы (H, W, 3)")
    
    R = image[:, :, 0].astype(float)
    G = image[:, :, 1].astype(float)
    B = image[:, :, 2].astype(float)
    
    if sigma > 0:
        R = gaussian_filter(R, sigma=sigma)
        G = gaussian_filter(G, sigma=sigma)
        B = gaussian_filter(B, sigma=sigma)
    
    channels = [R, G, B]
    threshold = _calculate_global_threshold(channels, method=threshold_method, threshold_factor=threshold_factor)
    light_vectors = _get_light_vectors(mode)
    
    if channel_roles is not None:
        if len(channel_roles) != 3 or not all(0 <= i < 3 for i in channel_roles):
            raise ValueError("channel_roles должен быть кортежем из 3 индексов 0-2")
        configs = [channel_roles]
    elif use_permutations:
        configs = list(permutations([0, 1, 2]))
    else:
        configs = [(0, 1, 2)]
    
    all_edge_maps = []
    debug_info = {
        'configs': configs,
        'threshold': threshold,
        'light_vectors': light_vectors,
        'num_maps': len(configs),
        'binary_threshold': None
    }
    
    for x_idx, y_idx, z_idx in configs:
        edge_map = _vector_lighting_vectorized(
            channels[x_idx], channels[y_idx], channels[z_idx],
            light_vectors, threshold, height_weight
        )
        all_edge_maps.append(edge_map)
    
    merged = _merge_edge_maps(all_edge_maps, method=merge_method)
    
    if binary:
        if binary_percentile > 0:
            thresh = np.percentile(merged, (1 - binary_percentile) * 100)
        else:
            thresh = np.mean(merged)
        result = (merged > thresh).astype(np.uint8) * 255
        debug_info['binary_threshold'] = thresh
    else:
        result = merged.astype(np.uint8)
    
    if clean_noise and binary:
        result = binary_opening(result > 0, structure=np.ones((3, 3))).astype(np.uint8) * 255
    
    if return_debug:
        return {'result': result, 'merged': merged, 'debug': debug_info}
    return result


class EdgeDetector:
    """
    Универсальный класс для детекции границ.
    
    Methods:
        sobel: Оператор Собеля.
        prewitt: Оператор Превитта.
        canny: Детектор Кэнни.
        vector_lighting: Векторное освещение (авторский метод).
    """
    
    @staticmethod
    def sobel(image: np.ndarray) -> np.ndarray:
        return sobel(image)
    
    @staticmethod
    def prewitt(image: np.ndarray) -> np.ndarray:
        return prewitt(image)
    
    @staticmethod
    def canny(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 100, sigma: float = 1.0) -> np.ndarray:
        return canny(image, low_threshold, high_threshold, sigma)
    
    @staticmethod
    def vector_lighting(image: np.ndarray, **kwargs) -> Union[np.ndarray, dict]:
        return vector_lighting(image, **kwargs)
