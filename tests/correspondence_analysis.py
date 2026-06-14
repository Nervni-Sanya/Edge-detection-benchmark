#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ математического соответствия vector_lighting ↔ Di Zenzo.

Вопрос: является ли vector_lighting самостоятельным оператором или это
переоткрытие структурного тензора Ди Дзензо (1986)?

Метод vector_lighting вычисляет для каналов (X, Y) и направления света L:
    E_L = sqrt((∇C_X·L)² + (∇C_Y·L)²) · (1 + α·(C_Z − T)/255),
усредняет по 8 симметричным направлениям L и 6 перестановкам каналов
(слияние по максимуму). Di Zenzo берёт sqrt наибольшего собственного
значения структурного тензора [[ΣCk_x², ΣCk_x·Ck_y],[·, ΣCk_y²]].

Сравниваем НЕПРЕРЫВНЫЕ карты отклика (до бинаризации) по Пирсону
(линейная связь) и Спирмену (ранговая = одинаковый порядок границ ⇒
одинаковые карты при любом пороге). Отдельно изолируем height-член,
сравнивая α=1 (дефолт) с α=0 (height отключён).

Запуск:  python tests/correspondence_analysis.py [--bsds /path/to/BSDS500]
"""
import os
import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_lighting import vector_lighting          # noqa: E402
from color_baselines import dizenzo_tensor           # noqa: E402
from scipy.stats import pearsonr, spearmanr          # noqa: E402


def vl_response(img: np.ndarray, height_weight: float = 1.0) -> np.ndarray:
    d = vector_lighting(img, binary=False, return_debug=True, height_weight=height_weight)
    return d['merged'].astype(float)


def dizenzo_response(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    mag, _ = dizenzo_tensor(img, sigma=sigma)
    return mag.astype(float)


def correlate(a: np.ndarray, b: np.ndarray, spearman_n: int = 20000) -> tuple:
    a, b = a.ravel(), b.ravel()
    pear = pearsonr(a, b)[0]
    idx = np.random.default_rng(0).choice(a.size, min(spearman_n, a.size), replace=False)
    spear = spearmanr(a[idx], b[idx])[0]
    return pear, spear


def build_synthetic() -> dict:
    imgs = {}
    iso = np.zeros((128, 128, 3), np.uint8)
    iso[:, :64] = [255, 0, 255]
    iso[:, 64:] = [0, 180, 0]
    imgs['iso_patch'] = iso
    y, x = np.mgrid[0:128, 0:128].astype(float)
    s = x * np.cos(np.deg2rad(30)) + y * np.sin(np.deg2rad(30))
    st = (np.floor(s / 16).astype(int) % 2)
    imgs['iso_stripes30'] = np.where(st[..., None] == 0, [255, 0, 255], [0, 180, 0]).astype(np.uint8)
    imgs['random_rgb'] = np.random.default_rng(1).integers(0, 256, (128, 128, 3), np.uint8)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bsds', type=str, default=None, help='Путь к BSDS500 для реальных изображений')
    ap.add_argument('--bsds-limit', type=int, default=5)
    args = ap.parse_args()

    imgs = build_synthetic()
    if args.bsds:
        import glob
        from PIL import Image
        for cand in ('data/images/test', 'BSDS500/data/images/test', 'BSR/BSDS500/data/images/test'):
            d = os.path.join(args.bsds, cand)
            if os.path.isdir(d):
                for f in sorted(glob.glob(os.path.join(d, '*.jpg')))[:args.bsds_limit]:
                    imgs['bsds_' + os.path.basename(f)[:-4]] = np.array(Image.open(f).convert('RGB'))
                break

    print(f"{'image':22} {'VL(α=1)~DZ':14} {'VL(α=0)~DZ':14} {'VL(α=1)~VL(α=0)':16}")
    print(f"{'':22} {'Pear/Spear':14} {'Pear/Spear':14} {'Pear/Spear':16}")
    print('-' * 70)
    agg = {'a1_dz': [], 'a0_dz': [], 'a1_a0': []}
    for name, img in imgs.items():
        v1, v0, dz = vl_response(img, 1.0), vl_response(img, 0.0), dizenzo_response(img)
        p1, s1 = correlate(v1, dz)
        p2, s2 = correlate(v0, dz)
        p3, s3 = correlate(v1, v0)
        agg['a1_dz'].append(s1); agg['a0_dz'].append(s2); agg['a1_a0'].append(s3)
        print(f"{name:22} {p1:.3f}/{s1:.3f}    {p2:.3f}/{s2:.3f}    {p3:.3f}/{s3:.3f}")
    print('-' * 70)
    print(f"{'СРЕДНИЙ Spearman':22} {np.mean(agg['a1_dz']):.3f}         "
          f"{np.mean(agg['a0_dz']):.3f}         {np.mean(agg['a1_a0']):.3f}")
    print()
    print("Интерпретация:")
    print(f"  VL(α=0) ~ Di Zenzo, Spearman = {np.mean(agg['a0_dz']):.3f}: "
          f"{'ранг-идентичны ⇒ тот же оператор' if np.mean(agg['a0_dz'])>0.98 else 'сильно связаны'}.")
    print(f"  height-член (α=1 vs α=0), Spearman = {np.mean(agg['a1_a0']):.3f}: "
          f"вносит {'малое' if np.mean(agg['a1_a0'])>0.95 else 'заметное'} возмущение.")


if __name__ == '__main__':
    main()
