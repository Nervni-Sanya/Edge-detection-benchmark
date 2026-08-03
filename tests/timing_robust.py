#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Устойчивое измерение времени выполнения детекторов границ.

Мотивация. Наивная схема замера (последовательно: все прогоны детектора A,
затем все прогоны детектора B) даёт систематическую ошибку до 2x: время
зависит от состояния аллокатора памяти и порядка измерения. Один и тот же
вызов vector_lighting в холодном состоянии занимает ~8.0 мс, а после
интенсивных аллокаций — ~4.0 мс. Эта ошибка превышает различия между
сравниваемыми методами, поэтому выводы о скорости, полученные наивной
схемой, недостоверны.

Данная реализация устраняет артефакт:
  * чередование детекторов внутри раунда (round-robin), а не блоками;
  * несколько независимых раундов, между раундами — перемешивание порядка;
  * итог — медиана по раундам, дополнительно приводится разброс;
  * прогрев всей группы детекторов до начала измерений.

Дополнительно сравниваются реализации оператора Собеля: общая двумерная
корреляция scipy.signal.correlate2d (использована в учебной реализации
репозитория) и сепарабельная реализация scipy.ndimage, чтобы сравнение
с классическими операторами было честным по реализации.

Запуск: python tests/timing_robust.py
"""

import os
import random
import sys
import time

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_lighting import vector_lighting, sobel, prewitt, canny   # noqa: E402
from vector_lighting.core import _rgb_to_grayscale                    # noqa: E402
from color_baselines import dizenzo, _normalize_u8                    # noqa: E402


def dizenzo_matched(image: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Di Zenzo в конфигурации, согласованной с быстрыми пресетами
    vector_lighting: без гауссова сглаживания и с центральными разностями
    (np.gradient) вместо градиентов Собеля.

    Нужен для честного сравнения: стандартный Di Zenzo использует
    сглаживание и градиенты Собеля, поэтому «выигрыш» быстрых пресетов
    vector_lighting над ним отражает разницу конфигураций, а не операторов.
    """
    ch = [image[:, :, k].astype(float) for k in range(image.shape[2])]
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        ch = [gaussian_filter(c, sigma) for c in ch]
    grads = [np.gradient(c) for c in ch]
    gx = [g[1] for g in grads]
    gy = [g[0] for g in grads]
    E = sum(a * a for a in gx)
    F = sum(a * b for a, b in zip(gx, gy))
    G = sum(a * a for a in gy)
    tmp = np.sqrt(np.maximum((E - G) ** 2 + 4.0 * F * F, 0.0))
    mag = _normalize_u8(np.sqrt(np.maximum(0.5 * (E + G + tmp), 0.0)))
    return (mag > np.mean(mag)).astype(np.uint8) * 255


# --------------------------------------------------------------------------
#            Честная (сепарабельная) реализация Собеля и Превитта
# --------------------------------------------------------------------------

def sobel_fast(image: np.ndarray) -> np.ndarray:
    """Собель через сепарабельные одномерные свёртки (scipy.ndimage)."""
    gray = _rgb_to_grayscale(image)
    gx = ndi.sobel(gray, axis=1, mode='reflect')
    gy = ndi.sobel(gray, axis=0, mode='reflect')
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    return (mag / m * 255).astype(np.uint8) if m > 0 else mag.astype(np.uint8)


def prewitt_fast(image: np.ndarray) -> np.ndarray:
    """Превитт через сепарабельные одномерные свёртки."""
    gray = _rgb_to_grayscale(image)
    gx = ndi.correlate1d(ndi.correlate1d(gray, [-1, 0, 1], axis=1, mode='reflect'),
                         [1, 1, 1], axis=0, mode='reflect')
    gy = ndi.correlate1d(ndi.correlate1d(gray, [-1, 0, 1], axis=0, mode='reflect'),
                         [1, 1, 1], axis=1, mode='reflect')
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    return (mag / m * 255).astype(np.uint8) if m > 0 else mag.astype(np.uint8)


# --------------------------------------------------------------------------
#                        Устойчивый измеритель
# --------------------------------------------------------------------------

def _once(fn, img):
    t0 = time.perf_counter()
    fn(img)
    return (time.perf_counter() - t0) * 1000.0


def robust_times(detectors, images, rounds=7, reps=5, warmup=2, seed=0):
    """Возвращает {имя: (медиана, минимум, максимум)} в миллисекундах.

    Чередует детекторы внутри раунда и перемешивает их порядок между
    раундами, поэтому состояние аллокатора одинаково влияет на всех.
    """
    rnd = random.Random(seed)
    names = list(detectors)
    for _ in range(warmup):                       # прогрев всей группы
        for n in names:
            for img in images:
                detectors[n](img)
    per_round = {n: [] for n in names}
    for _ in range(rounds):
        order = names[:]
        rnd.shuffle(order)
        samples = {n: [] for n in names}
        for _ in range(reps):
            for n in order:                       # round-robin, не блоками
                for img in images:
                    samples[n].append(_once(detectors[n], img))
        for n in names:
            per_round[n].append(float(np.median(samples[n])))
    return {n: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
            for n, v in per_round.items()}


def make_images(size=256, seed=0):
    """Три типа контента одинакового размера."""
    imgs = []
    cb = np.zeros((size, size, 3), np.uint8)
    step = size // 8
    for i in range(0, size, step):
        for j in range(0, size, step):
            if ((i // step) + (j // step)) % 2 == 0:
                cb[i:i + step, j:j + step] = 255
    imgs.append(cb)
    imgs.append(np.random.default_rng(seed).integers(0, 256, (size, size, 3), np.uint8))
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    grad = np.stack([(x / size * 255), (y / size * 255),
                     ((x + y) / (2 * size) * 255)], axis=-1).astype(np.uint8)
    imgs.append(grad)
    return imgs


def main():
    images = make_images()

    detectors = {
        'Sobel (correlate2d, репозиторий)': sobel,
        'Sobel (сепарабельный, ndimage)':   sobel_fast,
        'Prewitt (correlate2d)':            prewitt,
        'Prewitt (сепарабельный)':          prewitt_fast,
        'Canny (репозиторий)':              canny,
        'Di Zenzo (Собель + sigma=1)':      dizenzo,
        'Di Zenzo (согласованная конфиг.)': dizenzo_matched,
        'VL mode=1 (2 вектора, sigma=0)':
            lambda i: vector_lighting(i, mode=1, sigma=0.0,
                                      binary_percentile=0.0, use_permutations=False),
        'VL mode=3 (8 векторов, sigma=0)':
            lambda i: vector_lighting(i, mode=3, sigma=0.0,
                                      binary_percentile=0.0, use_permutations=False),
        'VL по умолчанию (8 вект. + перест.)':
            lambda i: vector_lighting(i),
    }

    res = robust_times(detectors, images)
    base = res['Sobel (сепарабельный, ndimage)'][0]

    print("=" * 78)
    print("УСТОЙЧИВОЕ ИЗМЕРЕНИЕ (256x256, 3 типа контента, 7 раундов, чередование)")
    print("=" * 78)
    print(f"{'детектор':<38}{'медиана':>10}{'разброс':>16}{'к Собелю':>12}")
    print("-" * 78)
    for name, (med, lo, hi) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"{name:<38}{med:>9.2f}м{lo:>7.2f}-{hi:<7.2f}{med/base:>10.1f}x")
    print("-" * 78)
    print("Столбец «к Собелю» — во сколько раз медленнее честного "
          "сепарабельного Собеля.")

    print()
    print("Ключевые соотношения:")
    s_fast = res['Sobel (сепарабельный, ndimage)'][0]
    v1 = res['VL mode=1 (2 вектора, sigma=0)'][0]
    v3 = res['VL mode=3 (8 векторов, sigma=0)'][0]
    dz_std = res['Di Zenzo (Собель + sigma=1)'][0]
    dz_m = res['Di Zenzo (согласованная конфиг.)'][0]
    print(f"  VL(8 векторов) медленнее VL(2 векторов):             {v3/v1:5.2f}x "
          f"(при линейном масштабировании ожидалось бы 4x)")
    print(f"  VL(2 вектора) относительно честного Собеля:          {v1/s_fast:5.1f}x медленнее")
    print(f"  VL(2 вектора) против Di Zenzo в СТАНДАРТНОЙ конфиг.: "
          f"{dz_std/v1:5.2f}x в пользу VL")
    print(f"  VL(2 вектора) против Di Zenzo в СОГЛАСОВАННОЙ конфиг.: "
          f"{v1/dz_m:5.2f}x в пользу Di Zenzo")
    print()
    print("  Вывод: преимущество быстрых пресетов над Di Zenzo существует только")
    print("  при сравнении с его стандартной конфигурацией (сглаживание + Собель).")
    print("  При одинаковой конфигурации тензорная форма быстрее: она вычисляет")
    print("  ту же величину в замкнутом виде, тогда как виртуальное освещение")
    print("  восстанавливает её выборкой по N направлениям.")


if __name__ == '__main__':
    main()
