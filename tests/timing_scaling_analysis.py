#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ масштабирования времени по числу векторов освещения.

Вопрос: почему конфигурации с малым числом векторов работают почти с той же
скоростью (или быстрее), что и Собель/Превитт? Является ли выигрыш следствием
уменьшения числа векторов, или он объясняется другой причиной?

Измеряем:
  1. Время vector_lighting в зависимости от режима (1, 2, 4, 8 векторов)
     при прочих равных -> модель "фиксированная часть + N * цена вектора".
  2. Поэлементную декомпозицию затрат (градиенты, сглаживание, цикл по
     векторам, перцентиль, нормировка).
  3. Честность сравнения с Собелем: correlate2d (общая 2D-корреляция)
     против сепарабельной реализации scipy.ndimage.sobel.

Запуск: python tests/timing_scaling_analysis.py
"""

import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import scipy.ndimage as ndi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_lighting import vector_lighting, sobel, prewitt          # noqa: E402
from vector_lighting.core import _get_light_vectors                   # noqa: E402


def med_ms(fn, img, warmup=3, reps=15):
    for _ in range(warmup):
        fn(img)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(img)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def bar(v, vmax, width=28):
    return '#' * max(1, int(round(width * v / vmax)))


def main():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    # ---------------------------------------------------------------- 1
    print("=" * 74)
    print("1. ВРЕМЯ В ЗАВИСИМОСТИ ОТ ЧИСЛА ВЕКТОРОВ (256x256, без перестановок)")
    print("=" * 74)
    print(f"{'режим':>6} {'векторов':>9} {'sigma=0':>10} {'sigma=1':>10}   "
          f"{'мс/вектор (sigma=0)':>20}")
    rows = []
    for mode in (0, 1, 2, 3):
        n = len(_get_light_vectors(mode))
        t0 = med_ms(lambda i, m=mode: vector_lighting(
            i, mode=m, sigma=0.0, use_permutations=False), img)
        t1 = med_ms(lambda i, m=mode: vector_lighting(
            i, mode=m, sigma=1.0, use_permutations=False), img)
        rows.append((mode, n, t0, t1))
    # линейная регрессия t = a + b*N по sigma=0
    N = np.array([r[1] for r in rows], float)
    T = np.array([r[2] for r in rows], float)
    b, a = np.polyfit(N, T, 1)
    for mode, n, t0, t1 in rows:
        print(f"{mode:>6} {n:>9} {t0:>9.2f}м {t1:>9.2f}м   {b:>19.3f}")
    print(f"\n  Модель: T(N) = {a:.2f} мс (фиксированная часть) "
          f"+ {b:.3f} мс * N (векторы)")
    print(f"  Доля фиксированной части при N=8: {100*a/(a+8*b):.0f}%; "
          f"при N=1: {100*a/(a+b):.0f}%")
    speedup = rows[3][2] / rows[0][2]
    print(f"  Ускорение 8 -> 1 вектор: {speedup:.2f}x "
          f"(при линейном масштабировании ожидалось бы 8x)")

    # ---------------------------------------------------------------- 2
    print()
    print("=" * 74)
    print("2. ДЕКОМПОЗИЦИЯ ЗАТРАТ vector_lighting (mode=1, sigma=0, без перест.)")
    print("=" * 74)
    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 2].astype(float)
    lv = _get_light_vectors(1)

    def t_cast(_):
        img.astype(float)

    def t_grad(_):
        np.gradient(R), np.gradient(G)

    def t_gauss(_):
        gaussian_filter(R, 1.0), gaussian_filter(G, 1.0), gaussian_filter(B, 1.0)

    gx_r, gy_r = np.gradient(R)
    gx_g, gy_g = np.gradient(G)
    hf = np.clip(1.0 + (B - 128.0) / 255.0, 0.1, 3.0)

    def t_loop(_):
        acc = np.zeros_like(R)
        for dx, dy in lv:
            rx = gx_r * dx + gy_r * dy
            ry = gx_g * dx + gy_g * dy
            acc += np.sqrt(rx * rx + ry * ry) * hf
        acc /= len(lv)

    def t_pct(_):
        np.percentile(R, 75)

    def t_norm(_):
        (R - R.min()) / (R.max() - R.min()) * 255.0

    parts = [
        ('приведение к float', t_cast),
        ('np.gradient (2 канала)', t_grad),
        ('цикл по 2 векторам', t_loop),
        ('percentile (порог)', t_pct),
        ('нормировка', t_norm),
        ('gaussian_filter (3 кан., sigma=1)', t_gauss),
    ]
    meas = [(name, med_ms(fn, None, 3, 21)) for name, fn in parts]
    vmax = max(v for _, v in meas)
    for name, v in meas:
        print(f"  {name:<36} {v:6.3f} мс  {bar(v, vmax)}")

    # ---------------------------------------------------------------- 3
    print()
    print("=" * 74)
    print("3. ЧЕСТНОСТЬ СРАВНЕНИЯ С СОБЕЛЕМ: реализация решает")
    print("=" * 74)
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)

    def sobel_correlate2d(_):
        gx = correlate2d(gray, kx, mode='same', boundary='symm')
        gy = correlate2d(gray, ky, mode='same', boundary='symm')
        return np.sqrt(gx * gx + gy * gy)

    def sobel_ndimage(_):
        gx = ndi.sobel(gray, axis=1, mode='reflect')
        gy = ndi.sobel(gray, axis=0, mode='reflect')
        return np.sqrt(gx * gx + gy * gy)

    def sobel_separable(_):
        gx = ndi.correlate1d(ndi.correlate1d(gray, [-1, 0, 1], axis=1, mode='reflect'),
                             [1, 2, 1], axis=0, mode='reflect')
        gy = ndi.correlate1d(ndi.correlate1d(gray, [-1, 0, 1], axis=0, mode='reflect'),
                             [1, 2, 1], axis=1, mode='reflect')
        return np.sqrt(gx * gx + gy * gy)

    t_c2d = med_ms(sobel_correlate2d, None)
    t_ndi = med_ms(sobel_ndimage, None)
    t_sep = med_ms(sobel_separable, None)
    t_vl_turbo = med_ms(lambda i: vector_lighting(
        i, mode=1, sigma=0.0, binary_percentile=0.0, use_permutations=False), img)
    t_vl_repo = med_ms(lambda i: sobel(i), img)

    print(f"  Собель, correlate2d (в репозитории)   {t_c2d:7.2f} мс  {bar(t_c2d, t_c2d)}")
    print(f"  Собель, scipy.ndimage.sobel           {t_ndi:7.2f} мс  {bar(t_ndi, t_c2d)}")
    print(f"  Собель, сепарабельные свёртки 1D      {t_sep:7.2f} мс  {bar(t_sep, t_c2d)}")
    print(f"  vector_lighting (turbo, 2 вектора)    {t_vl_turbo:7.2f} мс  {bar(t_vl_turbo, t_c2d)}")
    print()
    print(f"  Замедление correlate2d против ndimage: {t_c2d/t_ndi:.1f}x")
    print(f"  VL-turbo быстрее нашего Собеля в {t_c2d/t_vl_turbo:.2f}x, "
          f"но МЕДЛЕННЕЕ оптимального Собеля в {t_vl_turbo/t_ndi:.1f}x")

    # ---------------------------------------------------------------- 4
    print()
    print("=" * 74)
    print("4. ВЫВОД")
    print("=" * 74)
    print(f"  * Масштабирование по числу векторов СУБЛИНЕЙНО: 8 -> 1 вектор даёт")
    print(f"    лишь {speedup:.2f}x, а не 8x, т.к. {100*a/(a+8*b):.0f}% времени при N=8 —")
    print(f"    фиксированные операции (градиенты, перцентиль, нормировка).")
    print(f"  * Гиперболического прироста нет: T(N) = a + b*N — прямая линия,")
    print(f"    ускорение при уменьшении N ограничено сверху величиной "
          f"1 + {8*b/a:.2f} = {1 + 8*b/a:.2f}x.")
    print(f"  * Близость к Собелю/Превитту объясняется НЕ числом векторов, а тем,")
    print(f"    что в репозитории Собель использует correlate2d — общую 2D-корреляцию,")
    print(f"    которая в {t_c2d/t_ndi:.0f}x медленнее сепарабельной реализации.")
    print(f"    При честной реализации Собель быстрее vector_lighting в "
          f"{t_vl_turbo/t_ndi:.1f}x.")


if __name__ == '__main__':
    main()
