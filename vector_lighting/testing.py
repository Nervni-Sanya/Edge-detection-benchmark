#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный бенчмарк детектора границ с перебором параметров и экспортом в Excel.
Требует Python 3.10+ и установленной библиотеки vector_lighting.
"""

import os
import sys
import time
import warnings
from itertools import product, permutations
from typing import List, Tuple, Dict, Optional, Literal, Union

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from tqdm.auto import tqdm

# Импорт библиотеки
try:
    from vector_lighting import EdgeDetector, vector_lighting
except ImportError:
    print("❌ Ошибка: библиотека vector_lighting не найдена.")
    print("Установите её командой: pip install -e .")
    sys.exit(1)

warnings.filterwarnings('ignore')

###############################################################################
#                              МЕТРИКИ                                        #
###############################################################################

def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != bool:
        mask = mask > 0
    if not np.any(mask):
        return np.full_like(mask, np.inf, dtype=float)
    return distance_transform_edt(~mask)

def compute_metrics(detected: np.ndarray, ground_truth: np.ndarray, tolerance: int = 2) -> dict:
    det_mask = detected > 0
    gt_mask = ground_truth > 0 if ground_truth is not None else np.zeros_like(detected, dtype=bool)

    if not np.any(gt_mask):
        if not np.any(det_mask):
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': np.sum(det_mask), 'fn': 0}

    dist_to_gt = compute_distance_map(gt_mask)
    dist_to_det = compute_distance_map(det_mask)
    tp_map = det_mask & (dist_to_gt <= tolerance)
    fn_map = gt_mask & (dist_to_det > tolerance)
    tp = np.sum(tp_map)
    fp = np.sum(det_mask) - tp
    fn = np.sum(fn_map)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

###############################################################################
#                         ГЕНЕРАЦИЯ ДАННЫХ                                     #
###############################################################################

def generate_synthetic_dataset() -> Dict[str, dict]:
    dataset = {}
    size = 256

    # Checkerboard
    name = 'checkerboard.png'
    cell = 32
    img = np.zeros((size, size, 3), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, cell):
        for j in range(0, size, cell):
            color = 255 if (i // cell + j // cell) % 2 == 0 else 0
            img[i:i+cell, j:j+cell, :] = color
    for i in range(0, size, cell): gt[i, :] = 255
    for j in range(0, size, cell): gt[:, j] = 255
    dataset[name] = {'image': img, 'gt': gt}

    # Concentric Circles
    name = 'concentric_circles.png'
    img = np.zeros((size, size, 3), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r_map = np.sqrt((x - center)**2 + (y - center)**2)
    radii = [30, 60, 90]
    for idx, r in enumerate(radii):
        gt[np.abs(r_map - r) <= 1] = 255
        mask = r_map <= r if idx == 0 else (r_map > radii[idx-1]) & (r_map <= r)
        img[mask] = [255, 255, 255] if idx % 2 == 0 else [0, 0, 0]
    img[r_map > radii[-1]] = [255, 255, 255] if len(radii) % 2 == 0 else [0, 0, 0]
    dataset[name] = {'image': img, 'gt': gt}

    # Color Patches
    name = 'color_patches_equal_brightness.png'
    h, w = size, size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    gt = np.zeros((h, w), dtype=np.uint8)
    color_a = np.array([255, 0, 255], dtype=np.uint8)
    color_b = np.array([0, 180, 0], dtype=np.uint8)
    ch, cw = h // 2, w // 2
    img[:ch, :cw] = color_a
    img[:ch, cw:] = color_b
    img[ch:, :cw] = color_b
    img[ch:, cw:] = color_a
    gt[ch, :] = 255
    gt[:, cw] = 255
    dataset[name] = {'image': img, 'gt': gt}

    # Color Wheel
    name = 'color_wheel.png'
    img = np.zeros((size, size, 3), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r_map = np.sqrt((x - center)**2 + (y - center)**2)
    angle = np.arctan2(y - center, x - center)
    num_sectors = 12
    sector_width = 2 * np.pi / num_sectors
    sector_idx = np.floor((angle + np.pi) / sector_width).astype(int) % num_sectors
    import matplotlib.colors as mcolors
    hsv = np.zeros((size, size, 3))
    hsv[:, :, 0] = sector_idx / num_sectors
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = 1.0
    rgb = mcolors.hsv_to_rgb(hsv)
    img = (rgb * 255).astype(np.uint8)
    mask = r_map <= center - 2
    img[~mask] = 0
    for k in range(num_sectors):
        boundary_angle = -np.pi + k * sector_width
        diff = np.abs(angle - boundary_angle)
        diff = np.minimum(diff, 2*np.pi - diff)
        gt[diff < 0.03] = 255
    gt[~mask] = 0
    dataset[name] = {'image': img, 'gt': gt}

    # Blurred Disk
    name = 'blurred_disk.png'
    img = np.zeros((size, size, 3), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    r = size // 3
    disk_mask = r_map <= r
    img[disk_mask] = [200, 100, 50]
    img_float = img.astype(float)
    for c in range(3):
        img_float[:, :, c] = gaussian_filter(img_float[:, :, c], sigma=3)
    img = img_float.astype(np.uint8)
    gt[np.abs(r_map - r) <= 2] = 255
    dataset[name] = {'image': img, 'gt': gt}

    # Gray Scale
    name = 'gray_scale.png'
    img = np.zeros((size, size, 3), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    grad = np.linspace(0, 255, size, dtype=np.uint8)
    img[:] = grad[np.newaxis, :, np.newaxis]
    dataset[name] = {'image': img, 'gt': gt}

    return dataset

def load_real_images() -> Dict[str, dict]:
    files = ['lena.png', 'cameraman.png', 'texture.png', 'lena_gaussian_noise.png']
    images = {}
    for fname in files:
        if os.path.exists(fname):
            try:
                img = np.array(Image.open(fname))
                if len(img.shape) == 2:
                    img = np.stack([img]*3, axis=-1)
                images[fname] = {'image': img, 'gt': None}
                print(f"✅ {fname}: {img.shape}")
            except Exception as e:
                print(f"❌ {fname}: {e}")
        else:
            print(f"⚠️  {fname}: файл не найден")
    return images

###############################################################################
#                              БЕНЧМАРК                                       #
###############################################################################

def run_full_sweep(images: Dict[str, np.ndarray], ground_truths: Dict[str, Optional[np.ndarray]]) -> pd.DataFrame:
    param_grid = {
        'mode': [0, 1, 2, 3],
        'sigma': [0.0, 0.5, 1.0, 1.5],
        'binary_percentile': [0.0, 0.02, 0.05, 0.10],
        'use_permutations': [True, False],
        'merge_method': ['mean', 'max', 'adaptive_mean'],
        'threshold_method': ['mean_std', 'median', 'percentile'],
        'threshold_factor': [0.25, 0.5, 0.75],
        'height_weight': [0.5, 1.0, 1.5],
        'clean_noise': [True, False]
    }
    keys = param_grid.keys()
    configs = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    n_configs = len(configs)
    n_images = sum(1 for gt in ground_truths.values() if gt is not None)
    print(f"🔬 Запуск перебора {n_configs} конфигураций на {n_images} изображениях...")
    results = []
    for cfg in tqdm(configs, desc="Конфигурации"):
        for img_name, img in images.items():
            gt = ground_truths.get(img_name)
            if gt is None:
                continue
            try:
                t0 = time.time()
                result = EdgeDetector.custom(img, **cfg, binary=True, return_debug=False)
                elapsed_ms = (time.time() - t0) * 1000
                metrics = compute_metrics(result, gt, tolerance=2)
                results.append({
                    'image': img_name,
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'time_ms': elapsed_ms,
                    **cfg
                })
            except Exception as e:
                print(f"❌ Ошибка {img_name} cfg={cfg}: {e}")
    return pd.DataFrame(results)

def analyze_and_export(df: pd.DataFrame, output_file: str = 'sweep_results.xlsx'):
    if df.empty:
        print("⚠️  Нет данных")
        return
    summary = df.groupby([
        'mode', 'sigma', 'binary_percentile', 'use_permutations',
        'merge_method', 'threshold_method', 'threshold_factor',
        'height_weight', 'clean_noise'
    ]).agg(
        mean_f1=('f1', 'mean'),
        std_f1=('f1', 'std'),
        mean_time_ms=('time_ms', 'mean'),
        min_f1=('f1', 'min')
    ).reset_index()
    summary['efficiency'] = summary['mean_f1'] / np.log1p(summary['mean_time_ms'])
    summary = summary.sort_values('mean_f1', ascending=False)
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary.to_excel(writer, sheet_name='Summary', index=False)
            df.to_excel(writer, sheet_name='All_Results', index=False)
        print(f"📁 Результаты сохранены в {output_file}")
    except ImportError:
        print("⚠️  openpyxl не установлен. Сохранение в CSV.")
        summary.to_csv('sweep_results_summary.csv', index=False)
        df.to_csv('sweep_results_all.csv', index=False)
    print("\n🏆 ТОП-10 КОНФИГУРАЦИЙ:")
    print(summary.head(10).to_string(index=False))
    best = summary.iloc[0]
    print("\n✅ РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:")
    print(f"mode={best['mode']}, sigma={best['sigma']}, binary_percentile={best['binary_percentile']}")
    print(f"use_permutations={best['use_permutations']}, merge_method='{best['merge_method']}'")
    print(f"threshold_method='{best['threshold_method']}', threshold_factor={best['threshold_factor']}")
    print(f"height_weight={best['height_weight']}, clean_noise={best['clean_noise']}")
    print(f"Mean F1 = {best['mean_f1']:.4f}, Time = {best['mean_time_ms']:.1f} мс")
    return summary

###############################################################################
#                                    MAIN                                     #
###############################################################################

if __name__ == "__main__":
    print("🚀 Запуск бенчмарка...")
    print("🔬 Генерация синтетики...")
    synthetic_data = generate_synthetic_dataset()
    print(f"✅ Сгенерировано {len(synthetic_data)} изображений")
    print("\n📷 Загрузка реальных изображений...")
    real_data = load_real_images()
    all_data = {**synthetic_data, **real_data}
    images = {k: v['image'] for k, v in all_data.items()}
    ground_truths = {k: v['gt'] for k, v in all_data.items()}
    print(f"\n📊 Итого: {len(images)} изображений, {sum(1 for gt in ground_truths.values() if gt is not None)} с GT")
    df_results = run_full_sweep(images, ground_truths)
    analyze_and_export(df_results)
    print("\n🎉 Бенчмарк завершён!")