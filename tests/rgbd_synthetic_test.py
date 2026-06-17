#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контролируемый тест RGB-D: проверка гипотезы о height-члене.

Гипотеза. Если канал Z интерпретировать как физическую глубину
(RGB-D), то канально-асимметричный height-член vector_lighting должен
давать измеримый выигрыш над СИММЕТРИЧНЫМ цветовым Di Zenzo,
который трактует все каналы равноправно.

Проверка на 4 типа изображений 256x256:
  1. color_only       — цветовой край, depth константа.
  2. depth_only       — depth-step, RGB константа (нет ничего, что Di Zenzo
                        на RGB вообще видит).
  3. coincident       — color edge И depth-step В ОДНОМ МЕСТЕ.
                        Главный тест: усиливает ли height-член
                        совпадающие границы?
  4. orthogonal       — color edge (вертикальный) и depth-step
                        (горизонтальный) в РАЗНЫХ местах.
                        Проверка: не запутается ли height-член?

Сравниваемые детекторы:
  • DiZenzo-RGB       — симметричный тензор 3 каналов (без depth).
  • DiZenzo-RGBD      — симметричный тензор 4 каналов (depth равноправно).
  • VL-RGBD           — vector_lighting с depth, явно зафиксированным
                        как канал Z (X,Y ∈ {R,G,B}, Z = depth, α = 1).

Запуск: python tests/rgbd_synthetic_test.py
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_lighting import vector_lighting        # noqa: E402
from color_baselines import _channel_gradients, _normalize_u8  # noqa: E402
from testing import compute_metrics                 # noqa: E402


###############################################################################
#         4-канальный Di Zenzo (честный RGB-D baseline)                       #
###############################################################################

def dizenzo_nchannel(image: np.ndarray, sigma: float = 1.0, percentile: float = 95.0) -> np.ndarray:
    """Многоканальный Di Zenzo для произвольного числа каналов.
    Симметричный по каналам — все участвуют равноправно."""
    if image.ndim != 3:
        raise ValueError("image must be (H, W, C)")
    gx, gy = _channel_gradients(image, sigma)
    E = np.sum(gx * gx, axis=-1)
    F = np.sum(gx * gy, axis=-1)
    G = np.sum(gy * gy, axis=-1)
    tmp = np.sqrt(np.maximum((E - G) ** 2 + 4.0 * F * F, 0.0))
    lam_max = 0.5 * (E + G + tmp)
    mag = _normalize_u8(np.sqrt(np.maximum(lam_max, 0.0)))
    return (mag > np.percentile(mag, percentile)).astype(np.uint8) * 255


###############################################################################
#         vector_lighting на RGB-D (depth жёстко как Z)                       #
###############################################################################

def vl_rgbd(rgb: np.ndarray, depth: np.ndarray, height_weight: float = 1.0) -> np.ndarray:
    """vector_lighting с depth в роли Z. Перебираем все 3 пары (X, Y)
    из {R, G, B}, Z всегда depth. Max-слияние."""
    if depth.dtype != np.uint8:
        d = (depth.astype(float) - depth.min())
        d = (d / max(d.max(), 1.0) * 255.0).astype(np.uint8)
    else:
        d = depth
    pairs = [(0, 1), (0, 2), (1, 2)]  # (R,G), (R,B), (G,B)
    maps = []
    for x_idx, y_idx in pairs:
        img3 = np.stack([rgb[..., x_idx], rgb[..., y_idx], d], axis=-1)
        maps.append(
            vector_lighting(
                img3, mode=3, sigma=1.0, binary_percentile=0.05,
                use_permutations=False,                 # каналы зафиксированы
                channel_roles=(0, 1, 2),                # X, Y, Z = X_chan, Y_chan, depth
                merge_method='max',
                threshold_method='percentile', threshold_factor=0.25,
                height_weight=height_weight, clean_noise=False, binary=True
            )
        )
    return np.maximum.reduce(maps).astype(np.uint8)


###############################################################################
#         Контролируемые синтетические RGB-D изображения                       #
###############################################################################

def _make_canvas(size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.full((size, size, 3), 128, dtype=np.uint8)
    depth = np.full((size, size), 128, dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    return rgb, depth, gt


def make_color_only(size: int = 256) -> Dict[str, np.ndarray]:
    """Изолюминантная цветовая граница вертикально, depth = константа."""
    rgb, depth, gt = _make_canvas(size)
    half = size // 2
    rgb[:, :half] = [255, 0, 255]   # пурпурный
    rgb[:, half:] = [0, 180, 0]     # зелёный, та же яркость
    gt[:, half] = 255
    return {'rgb': rgb, 'depth': depth, 'gt': gt}


def make_depth_only(size: int = 256) -> Dict[str, np.ndarray]:
    """Depth-step вертикально, RGB однородный."""
    rgb, depth, gt = _make_canvas(size)
    half = size // 2
    depth[:, :half] = 60
    depth[:, half:] = 200
    gt[:, half] = 255
    return {'rgb': rgb, 'depth': depth, 'gt': gt}


def make_coincident(size: int = 256) -> Dict[str, np.ndarray]:
    """Color edge И depth step в одном месте — главный тест."""
    rgb, depth, gt = _make_canvas(size)
    half = size // 2
    rgb[:, :half] = [255, 0, 255]
    rgb[:, half:] = [0, 180, 0]
    depth[:, :half] = 60
    depth[:, half:] = 200
    gt[:, half] = 255
    return {'rgb': rgb, 'depth': depth, 'gt': gt}


def make_orthogonal(size: int = 256) -> Dict[str, np.ndarray]:
    """Color edge вертикально, depth step горизонтально (другое место).
    Проверка: ловит ли метод обе границы и не путает ли их?"""
    rgb, depth, gt = _make_canvas(size)
    half = size // 2
    rgb[:, :half] = [255, 0, 255]
    rgb[:, half:] = [0, 180, 0]
    depth[:half, :] = 60
    depth[half:, :] = 200
    gt[:, half] = 255
    gt[half, :] = 255
    return {'rgb': rgb, 'depth': depth, 'gt': gt}


###############################################################################
#                              ЭКСПЕРИМЕНТ                                    #
###############################################################################

DATA = {
    'color_only': make_color_only(),
    'depth_only': make_depth_only(),
    'coincident': make_coincident(),
    'orthogonal': make_orthogonal(),
}


def run(tolerance: int = 2) -> pd.DataFrame:
    rows = []
    for case, d in DATA.items():
        rgb, depth, gt = d['rgb'], d['depth'], d['gt']
        rgbd = np.dstack([rgb, depth])  # (H, W, 4) for the 4-channel DZ

        out = {
            'DiZenzo-RGB':  dizenzo_nchannel(rgb),
            'DiZenzo-RGBD': dizenzo_nchannel(rgbd),
            'VL-RGB':       vector_lighting(rgb),
            'VL-RGBD':      vl_rgbd(rgb, depth, height_weight=1.0),
            'VL-RGBD-aoff': vl_rgbd(rgb, depth, height_weight=0.0),  # height отключён
        }
        for name, edge in out.items():
            m = compute_metrics(edge, gt, tolerance=tolerance)
            rows.append({'case': case, 'detector': name,
                         'f1': round(m['f1'], 3),
                         'precision': round(m['precision'], 3),
                         'recall': round(m['recall'], 3)})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='detector', columns='case', values='f1')
    pivot = pivot[['color_only', 'depth_only', 'coincident', 'orthogonal']]
    pivot.loc['(mean)'] = pivot.mean()
    print("\n📊 Контролируемый RGB-D тест (F1, допуск 2 px)\n")
    print(pivot.to_string())
    print()
    print("Что смотрим:")
    print("  - depth_only: должен показать, что DZ-RGB слеп (как Canny на изолюминантности)")
    print("  - coincident: усиливает ли height-член совпадающие границы?")
    print("  - orthogonal: не путаются ли разнесённые color/depth границы?")
    df.to_csv('rgbd_results.csv', index=False)
    print("\n📁 Результаты сохранены в rgbd_results.csv")
    return df


if __name__ == '__main__':
    run()
