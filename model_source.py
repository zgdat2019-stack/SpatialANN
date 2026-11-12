# -*- coding: utf-8 -*-
"""

This exploratory version removes the Stage-1 background fit. The neural model
is supervised directly by station-observed daily maximum wind speed, and its
output is the final wind-speed prediction. No fitted background or predicted
residual is added during training, validation, station inference, or grid
inference.

Neural inputs retain ERA5-Land variables, terrain, nighttime light,
time, causal temporal and climate features, and station-identity embeddings. Coordinates are used only for spatial
relationships and data/output alignment. Climate history is 1095 days.
Neither target-day observations from neighbouring stations nor historical
observation-residual statistics are supplied as model inputs. Gaussian soft
inverse distance weighting and the external baseline suite are disabled.

The predictive distribution is split at a fold-specific threshold estimated
from training-station observations. Non-exceedances are represented by the
bulk wind head. Exceedance occurrence is learned by a binary head, while the
positive excess above the threshold follows a generalized Pareto distribution
(GPD). The GPD shape parameter is shared within each fold to avoid unstable
day- and station-specific estimates.

The script reuses aligned-data and station-geometry caches from the original
direct-target experiment when available. Legacy ``base``
and ``residual`` fields in DatasetBundle are kept only for compatibility with
shared utilities; they are set to zero and the observed daily maximum wind
speed, respectively, and are not used by the direct-target neural workflow.
"""

from __future__ import annotations

import gc
import hashlib
from collections import OrderedDict
import json
import math
import pickle
import random
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')



# =============================================================================
# Inlined v20 utilities required by Phase A
# =============================================================================

EARTH_R_KM = 6371.0088

EPS = 1e-8

CODE_VERSION = 'v20_0p1_1979_2019_era5land_btree_align_drop100_v1'

LEGACY_VERSIONS: List[str] = []

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def format_elapsed_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    if minutes > 0:
        return f'{minutes}m {secs}s'
    return f'{secs}s'


def print_runtime_environment() -> None:
    device = get_device()
    print('=' * 80)
    print('[runtime] execution environment')
    print(f'[runtime] torch version      : {torch.__version__}')
    print(f'[runtime] selected device   : {device}')
    print(f'[runtime] cuda available    : {torch.cuda.is_available()}')
    if device.type == 'cuda':
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        total_gb = props.total_memory / (1024 ** 3)
        print(f'[runtime] cuda device count : {torch.cuda.device_count()}')
        print(f'[runtime] cuda current id   : {device_idx}')
        print(f'[runtime] cuda device name  : {torch.cuda.get_device_name(device_idx)}')
        print(f'[runtime] cuda capability   : {props.major}.{props.minor}')
        print(f'[runtime] cuda memory       : {total_gb:.2f} GB')
        print(f'[runtime] torch cuda build  : {torch.version.cuda}')
    else:
        print('[runtime] cuda device info  : not available, using CPU')
    print('=' * 80)


def resolve_input_directory(input_dir: str) -> Path:
    """Resolve data input path robustly for local and server runs.

    The intended server layout for this edition is:
      parent/
        基础数据2/
        ERA5LandBasic/
        MWS06/cnp_phaseI_no_bake_final_single_0p1_1979_2019_ERA5LandBasic_fixed.py

    A relative path is first resolved from the current working directory; if it
    does not exist, it is resolved relative to this script file.
    """
    raw = Path(input_dir)
    if raw.is_absolute():
        return raw

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    script_candidate = (Path(__file__).resolve().parent / raw).resolve()
    return script_candidate


@dataclass
class ExperimentConfig:
    input_dir: str = '../ERA5LandBasic/'
    work_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/'
    cache_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/cache/'
    fig_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/figs/'
    pred_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/preds/'
    ckpt_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/ckpt/'
    sensitivity_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/sens/'
    legacy_cache_dirs: Tuple[str, ...] = ()
    years: Tuple[int, ...] = tuple(range(1979, 2020))
    # ERA5-Land 0.1° CSV files expected in input_dir:
    # ERA5Land_WS_YYYY.csv / ERA5Land_TEM_YYYY.csv / ERA5Land_RH_YYYY.csv / ERA5Land_SP_YYYY.csv
    dynamic_vars: Tuple[str, ...] = ('ERA5Land_WS', 'ERA5Land_TEM', 'ERA5Land_RH', 'ERA5Land_SP')
    # Coordinate matching tolerance for nearest-neighbor alignment of station/static/dynamic grids.
    # Stations/nodes whose nearest match exceeds this threshold are discarded
    # before train/val/test splitting, so they do not participate in any later step.
    coord_match_max_km: float = 100.0
    bh_reference_year: int = 2019
    bh_year_feature_name: str = 'BH_year'
    seq_len: int = 21
    climate_hist_len: int = 365 * 3
    graph_k: int = 12
    graph_sigma_km: float = 180.0
    # v20-expertfix: 恢复为与 v19 兼容的多专家设置（模型参数沿用 0.1° 旧实验）
    region_k_max: int = 8
    expert_topk: int = 3
    gate_temperature: float = 0.35
    split_seed: int = 2025
    split_grid_deg: float = 10.0
    # New data source and cache layout; recompute by default to avoid stale CMFD caches.
    force_recompute: bool = True
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    calib_ratio_within_val: float = 0.5
    hidden_dim: int = 128
    climate_dim: int = 64
    num_graph_layers: int = 2
    graph_batch_days: int = 8
    seq_batch_size: int = 1024
    graph_max_epochs: int = 100
    seq_max_epochs: int = 32
    lr: float = 8.0e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 16
    grad_clip: float = 3.0
    tail_quantile: float = 0.95
    exceedance_loss_weight: float = 0.08
    tail_loss_weight: float = 0.06
    huber_delta: float = 1.5
    # v20: 调整正则权重，移除冲突的 corr_fit_loss
    region_smooth_weight: float = 0.00
    expert_diversity_weight: float = 0.0015
    expert_loadbalance_weight: float = 0.01   # NEW: Switch-Transformer style
    expert_compactness_weight: float = 0.0000
    active_expert_threshold: float = 0.05
    correction_scale: float = 0.75
    correction_reg_weight: float = 0.0000
    expert_correction_loss_weight: float = 0.00   # v20: disabled (conflicts with final_loss)
    expert_direction_loss_weight: float = 0.08
    moran_reg_weight: float = 0.05   # NEW: spatial residual decorrelation
    bh_alpha: float = 0.60
    bh_wmin: float = 0.75
    conformal_alpha: float = 0.10
    rf_max_train_samples: int = 300000
    gbrt_max_train_samples: int = 300000
    rf_n_estimators: int = 300
    rf_max_depth: int = 20
    rf_min_samples_leaf: int = 5
    gbrt_n_estimators: int = 400
    gbrt_learning_rate: float = 0.04
    gbrt_max_depth: int = 4
    gbrt_min_samples_leaf: int = 20
    gbrt_subsample: float = 0.8
    num_workers: int = 0
    seed: int = 42
    n_seeds: int = 1          # NEW: multi-seed runs for statistical robustness
    seed_list: Tuple[int, ...] = (42,)   # used if n_seeds > 1
    run_sensitivity: bool = False
    sensitivity_hidden_dims: Tuple[int, ...] = (128, 160, 192)
    sensitivity_seq_lens: Tuple[int, ...] = (14, 21, 28)
    sensitivity_region_k_max: Tuple[int, ...] = (8, 10, 12)
    sensitivity_gate_temperatures: Tuple[float, ...] = (0.35, 0.50, 0.65)
    sensitivity_tail_quantiles: Tuple[float, ...] = (0.90, 0.95, 0.97)
    sensitivity_seq_epochs: int = 18
    sensitivity_graph_epochs: int = 40
    rk_max_train_samples: int = 300000
    save_prediction_csvs: bool = False
    run_external_baselines: bool = False
    enabled_baselines: Tuple[str, ...] = tuple()
    region_static_prior_weight: float = 0.30
    use_pseudo_missing_train: bool = True
    pseudo_drop_rate: float = 0.25
    pseudo_min_masked: int = 2
    pseudo_context_resid_scale: float = 2.5
    # v20: evaluation extras
    report_crps: bool = False
    report_csi: bool = False
    report_moran: bool = False
    # Dedicated ablation: the model target and output are daily maximum wind
    # speed itself. A Stage-1 background fit is forbidden in this script.
    direct_dailymax_target: bool = True

    def prepare_dirs(self) -> None:
        for p in [self.work_dir, self.cache_dir, self.fig_dir, self.pred_dir, self.ckpt_dir, self.sensitivity_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetBundle:
    node_df: pd.DataFrame
    dates: pd.DataFrame
    static_const: np.ndarray
    static_const_scaled: np.ndarray
    yearly_feats: np.ndarray
    daily_dyn: np.ndarray
    daily_dyn_scaled: np.ndarray
    day_features: np.ndarray
    day_features_scaled: np.ndarray
    obs: np.ndarray
    obs_mask: np.ndarray
    base: np.ndarray
    residual: np.ndarray
    train_node_mask: np.ndarray
    val_node_mask: np.ndarray
    calib_node_mask: np.ndarray
    test_node_mask: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    bh_conf: np.ndarray
    tail_thresholds: Dict[str, float]
    train_tail_threshold: float
    feature_names: List[str]
    const_feature_names: List[str]
    year_feature_names: List[str]
    dyn_feature_names: List[str]
    era5_reg_coef: float
    era5_reg_intercept: float
    era5_reg_feature_coef: Optional[np.ndarray]
    split_summary: Dict[str, int]
    scaler_stats: Dict[str, Dict[str, List[float]]]
    climate_vectors: np.ndarray  # [D, N, climate_raw_dim] precomputed causal climate

    @property
    def n_days(self) -> int:
        return int(self.day_features.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.day_features.shape[1])

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_pickle(obj, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def prediction_cache_valid(arr: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
    """Check if cached prediction has expected shape and is mostly finite."""
    if arr.shape != expected_shape:
        return False
    finite_ratio = np.isfinite(arr).mean()
    return finite_ratio > 0.5

def suspicious_exact_test_match(pred: np.ndarray, bundle: 'DatasetBundle', tol: float = 1e-6,
                                min_count: int = 10) -> bool:
    """Guard against stale/leaky caches: perfect test match is almost certainly invalid here."""
    mask = bundle.obs_mask & bundle.test_node_mask[None, :]
    if mask.sum() < min_count:
        return False
    diff = np.abs(np.asarray(pred, np.float64)[mask] - np.asarray(bundle.obs, np.float64)[mask])
    diff = diff[np.isfinite(diff)]
    if len(diff) < min_count:
        return False
    return float((diff <= tol).mean()) > 0.999

def load_reusable_prediction_cache(cache_dir: Path, prefix: str, expected_shape: Tuple[int, ...],
                                   bundle: 'DatasetBundle',
                                   legacy_cache_dirs: Optional[Sequence[str]] = None) -> Optional[np.ndarray]:
    """Load newest reusable prediction cache from current/legacy cache dirs."""
    search_dirs = [Path(cache_dir)]
    if legacy_cache_dirs:
        for ld in legacy_cache_dirs:
            ldp = Path(ld)
            if ldp.exists() and ldp.resolve() != Path(cache_dir).resolve():
                search_dirs.append(ldp)
    patterns = [f"{prefix}_*.npy", f"{prefix}.npy"]
    candidates = []
    for sdir in search_dirs:
        for pat in patterns:
            candidates.extend(sdir.glob(pat))
    candidates = sorted({p.resolve() for p in candidates}, key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            arr = np.load(str(path), allow_pickle=False)
        except Exception:
            continue
        if not prediction_cache_valid(arr, expected_shape):
            continue
        if suspicious_exact_test_match(arr, bundle):
            continue
        return arr.astype(np.float32, copy=False)
    return None

def load_reusable_pickle(cache_dir: Path, prefix: str,
                         legacy_cache_dirs: Optional[Sequence[str]] = None):
    """Load newest reusable pickle from current/legacy cache dirs."""
    search_dirs = [Path(cache_dir)]
    if legacy_cache_dirs:
        for ld in legacy_cache_dirs:
            ldp = Path(ld)
            if ldp.exists() and ldp.resolve() != Path(cache_dir).resolve():
                search_dirs.append(ldp)
    patterns = [f"{prefix}_*.pkl", f"{prefix}.pkl"]
    candidates = []
    for sdir in search_dirs:
        for pat in patterns:
            candidates.extend(sdir.glob(pat))
    candidates = sorted({p.resolve() for p in candidates}, key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            return load_pickle(str(path))
        except Exception:
            continue
    return None

def find_bh_year_columns(df: pd.DataFrame) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for col in df.columns:
        name = str(col)
        if len(name) == 6 and name.startswith('BH') and name[2:].isdigit():
            out[int(name[2:])] = name
    return dict(sorted(out.items()))

def build_bh_yearly_features(node_df: pd.DataFrame,
                             years: Sequence[int],
                             feature_name: str = 'BH_year') -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    """Build one annual BH feature from the next available BHYYYY snapshot."""
    bh_cols = find_bh_year_columns(node_df)
    if not bh_cols:
        raise KeyError('Static data needs BHYYYY columns, e.g. BH1984/BH1990/BH2000/BH2010/BH2019.')

    src_years = np.asarray(sorted(bh_cols), dtype=np.float64)
    src_values = node_df[[bh_cols[int(y)] for y in src_years]].to_numpy(np.float32)
    src_values = np.where(np.isfinite(src_values), src_values, np.nan)
    col_median = np.nanmedian(src_values, axis=0)
    col_median = np.where(np.isfinite(col_median), col_median, 0.0)
    bad = ~np.isfinite(src_values)
    if np.any(bad):
        src_values[bad] = np.take(col_median, np.where(bad)[1])

    target_years = np.asarray(list(years), dtype=np.float64)
    source_idx = np.searchsorted(src_years, target_years, side='left')
    source_idx = np.clip(source_idx, 0, len(src_years) - 1).astype(np.int64)
    out = np.empty((len(target_years), len(node_df), 1), dtype=np.float32)
    out[:, :, 0] = src_values[:, source_idx].T
    out = np.clip(out, 0.0, 1.0)

    meta = {
        'bh_source_columns': [bh_cols[int(y)] for y in src_years],
        'bh_source_years': [int(y) for y in src_years],
        'bh_target_years': [int(y) for y in target_years],
        'bh_assignment_policy': 'stepwise_next_available_year',
        'bh_pre_first_year_policy': f'use_{bh_cols[int(src_years[0])]}',
        'bh_year_assignment': {
            int(year): bh_cols[int(src_years[int(idx)])]
            for year, idx in zip(target_years.astype(int), source_idx)
        },
        'bh_feature_name': feature_name,
    }
    return out, [feature_name], meta

def sanitize_name(name: str) -> str:
    return (
        name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')
        .replace('/', '_').replace(':', '_').replace(',', '_')
    )


def bh_confidence(bh: np.ndarray, alpha: float = 0.6, wmin: float = 0.75) -> np.ndarray:
    bh = np.clip(np.asarray(bh, np.float64), 0.0, 1.0)
    return (wmin + (1.0 - wmin) * np.power(1.0 - bh, alpha)).astype(np.float32)

def split_allocate(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    ratios_np = np.asarray(ratios, dtype=float)
    ratios_np /= ratios_np.sum()
    raw = ratios_np * n
    cnt = np.floor(raw).astype(int)
    rem = int(n - cnt.sum())
    if rem > 0:
        frac = raw - cnt
        for idx in np.argsort(-frac)[:rem]:
            cnt[idx] += 1
    while cnt.sum() > n:
        cnt[np.argmax(cnt)] -= 1
    while cnt.sum() < n:
        cnt[np.argmin(cnt)] += 1
    return int(cnt[0]), int(cnt[1]), int(cnt[2])

def fit_scale_3d(arr: np.ndarray, train_node_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    ref = arr[:, train_node_mask, :].reshape(-1, arr.shape[-1])
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where((np.isfinite(sd)) & (sd > 0), sd, 1.0).astype(np.float32)
    scaled = ((arr - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, {'mean': mu.tolist(), 'std': sd.tolist()}

def fit_scale_2d(arr: np.ndarray, train_node_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    ref = arr[train_node_mask]
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where((np.isfinite(sd)) & (sd > 0), sd, 1.0).astype(np.float32)
    scaled = ((arr - mu[None, :]) / sd[None, :]).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, {'mean': mu.tolist(), 'std': sd.tolist()}

def precompute_climate_vectors(daily_dyn_scaled: np.ndarray,
                               static_const_scaled: np.ndarray,
                               hist_len: int) -> np.ndarray:
    """Causal mean/std/amplitude over hist_len days, recent <=30-day max, static fields.

    Uses the same inclusive day windows and float32 cumulative-sum convention
    as the original, vectorized in node blocks to bound intermediate memory.
    """
    from scipy.ndimage import maximum_filter1d

    D, N, F_dyn = daily_dyn_scaled.shape
    hist_len = int(hist_len)
    if hist_len < 1:
        raise ValueError('climate_hist_len must be positive.')
    F_static = static_const_scaled.shape[1]
    result = np.empty((D, N, F_dyn * 4 + F_static), dtype=np.float32)
    result[..., 4 * F_dyn:] = static_const_scaled[None, :, :]
    days = np.arange(D)
    starts = np.maximum(0, days - hist_len + 1)
    lengths = (days - starts + 1).astype(np.float32)[:, None, None]
    phase = np.arange(D, dtype=np.float32)
    sine = np.sin(2.0 * np.pi * phase / 366.0)[:, None, None]
    cosine = np.cos(2.0 * np.pi * phase / 366.0)[:, None, None]

    def rolling_sum(x: np.ndarray) -> np.ndarray:
        cs = np.cumsum(x, axis=0)
        valid = starts > 0
        # Subtract only after the cumulative sum is fully computed.
        cs[valid] = cs[valid] - cs[starts[valid] - 1]
        return cs

    width = min(30, hist_len)
    for left in range(0, N, 64):
        right = min(N, left + 64)
        x = daily_dyn_scaled[:, left:right, :]
        mean = rolling_sum(x) / lengths
        result[:, left:right, :F_dyn] = mean
        var = rolling_sum(x ** 2) / lengths - mean ** 2
        result[:, left:right, F_dyn:2 * F_dyn] = np.sqrt(np.maximum(var, 1e-6))
        result[:, left:right, 2 * F_dyn:3 * F_dyn] = maximum_filter1d(
            x, size=width, axis=0, origin=(width - 1) // 2, mode='nearest'
        )
        sin_proj = rolling_sum(sine * x) / lengths
        cos_proj = rolling_sum(cosine * x) / lengths
        result[:, left:right, 3 * F_dyn:4 * F_dyn] = np.sqrt(
            sin_proj ** 2 + cos_proj ** 2 + 1e-8
        )
    return result


def split_calibration_from_val(val_mask: np.ndarray, seed: int, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    val_idx = np.where(val_mask)[0]
    rng = np.random.default_rng(seed)
    idx = val_idx.copy()
    rng.shuffle(idx)
    n_cal = max(1, int(round(len(idx) * ratio))) if len(idx) >= 2 else 0
    calib_idx = idx[:n_cal]
    val_fit_idx = idx[n_cal:]
    if len(val_fit_idx) == 0 and len(idx) > 0:
        val_fit_idx = idx[-1:]
        calib_idx = idx[:-1]
    val_fit_mask = np.zeros_like(val_mask, dtype=bool)
    calib_mask = np.zeros_like(val_mask, dtype=bool)
    val_fit_mask[val_fit_idx] = True
    calib_mask[calib_idx] = True
    return val_fit_mask, calib_mask

def assign_station_grid(node_df: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    """Attach fixed-degree geographic grid information to each station."""
    if grid_deg <= 0:
        raise ValueError('split_grid_deg 必须为正数。')
    required_cols = {'Lat', 'Lon'}
    missing = required_cols - set(node_df.columns)
    if missing:
        raise KeyError(f'站点坐标缺少列: {sorted(missing)}')

    out = node_df.copy()
    lat = out['Lat'].to_numpy(np.float64)
    lon = out['Lon'].to_numpy(np.float64)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('站点坐标中存在非有限值，无法进行网格划分。')

    lat_bin = np.floor(lat / grid_deg).astype(np.int64)
    lon_bin = np.floor(lon / grid_deg).astype(np.int64)
    out['grid_lat_bin'] = lat_bin
    out['grid_lon_bin'] = lon_bin
    out['grid_lat_min'] = (lat_bin * grid_deg).astype(np.float32)
    out['grid_lat_max'] = ((lat_bin + 1) * grid_deg).astype(np.float32)
    out['grid_lon_min'] = (lon_bin * grid_deg).astype(np.float32)
    out['grid_lon_max'] = ((lon_bin + 1) * grid_deg).astype(np.float32)
    out['grid_id'] = [f'{a}_{b}' for a, b in zip(lat_bin.tolist(), lon_bin.tolist())]
    return out

def allocate_grid_split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Allocate train/val/test counts inside a single grid cell.

    尽量遵循 7:2:1；当网格内站点数较少时，优先保证该网格至少覆盖验证/测试样本，
    避免空间上严重失衡。
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    cnt = np.array(split_allocate(n, ratios), dtype=np.int64)
    for target_idx in (1, 2):
        if cnt[target_idx] == 0:
            donor_idx = int(np.argmax(cnt))
            if cnt[donor_idx] > 1:
                cnt[donor_idx] -= 1
                cnt[target_idx] += 1
    if cnt[0] <= 0:
        donor_idx = 1 if cnt[1] >= cnt[2] else 2
        if cnt[donor_idx] > 1:
            cnt[donor_idx] -= 1
            cnt[0] += 1
    return int(cnt[0]), int(cnt[1]), int(cnt[2])

def grid_aware_spatial_split_nodes(node_df: pd.DataFrame,
                                   cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split stations inside each fixed-degree grid cell into train/val/test."""
    required_cols = {'Lat', 'Lon', 'grid_id'}
    missing = required_cols - set(node_df.columns)
    if missing:
        raise KeyError(f'网格空间划分缺少列: {sorted(missing)}')

    n_nodes = len(node_df)
    if n_nodes < 3:
        raise ValueError(f'站点数量过少，无法执行 train/val/test 三划分: {n_nodes}')

    ratios = np.asarray([cfg.train_ratio, cfg.val_ratio, cfg.test_ratio], dtype=np.float64)
    if np.any(ratios <= 0):
        raise ValueError('train_ratio / val_ratio / test_ratio 必须均为正数。')
    ratios = ratios / ratios.sum()

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    rng = np.random.default_rng(cfg.split_seed)

    grid_ids = node_df['grid_id'].astype(str).to_numpy()
    for grid_id in pd.Index(pd.unique(grid_ids)).sort_values():
        idx = np.flatnonzero(grid_ids == grid_id)
        idx = idx.copy()
        rng.shuffle(idx)
        n_train, n_val, n_test = allocate_grid_split_counts(len(idx), tuple(float(x) for x in ratios))
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:n_train + n_val + n_test]] = True

    def _rebalance_empty_subset(target_mask: np.ndarray, subset_name: str) -> None:
        if target_mask.sum() > 0:
            return
        donor_candidates = np.flatnonzero(train_mask)
        if len(donor_candidates) <= 1:
            raise ValueError(f'10度网格划分后 {subset_name} 为空，且无法从训练集安全挪出站点。')
        pick = donor_candidates[int(rng.integers(0, len(donor_candidates)))]
        train_mask[pick] = False
        target_mask[pick] = True

    _rebalance_empty_subset(val_mask, '验证集')
    _rebalance_empty_subset(test_mask, '测试集')

    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
        raise RuntimeError('网格空间划分失败：子集之间存在重叠站点。')
    if int(train_mask.sum() + val_mask.sum() + test_mask.sum()) != n_nodes:
        raise RuntimeError('网格空间划分失败：存在未分配站点。')
    return train_mask, val_mask, test_mask


def initialize_direct_target_state(obs: np.ndarray,
                                   dynamic_feature_count: int
                                   ) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Create compatibility fields without fitting a Stage-1 model.

    ``base`` is an all-zero array. ``residual`` is a compatibility alias of the
    observed daily maximum wind speed, not an observation-minus-background
    residual. The coefficient placeholders are also zero and are never used by
    the direct-target neural workflow.
    """
    obs = np.asarray(obs, dtype=np.float32)
    base = np.zeros_like(obs, dtype=np.float32)
    direct_target = obs.copy()
    coefficient_placeholder = np.zeros(int(dynamic_feature_count) + 2, dtype=np.float32)
    return base, direct_target, 0.0, 0.0, coefficient_placeholder

def train_tail_thresholds(train_obs: np.ndarray, trigger_quantile: float = 0.95) -> Tuple[Dict[str, float], float]:
    arr = np.asarray(train_obs, np.float64)
    arr = arr[np.isfinite(arr)]
    thresholds = {f'P{int(q * 100)}': float(np.quantile(arr, q)) for q in [0.85, 0.90, 0.95]}
    trigger_quantile = float(np.clip(trigger_quantile, 0.85, 0.99))
    return thresholds, float(np.quantile(arr, trigger_quantile))

def build_knn_graph(node_df: pd.DataFrame, k: int, sigma_km: float) -> Tuple[np.ndarray, np.ndarray]:
    coords_rad = np.radians(node_df[['Lat', 'Lon']].to_numpy(np.float64))
    tree = BallTree(coords_rad, metric='haversine')
    kq = min(k + 1, len(node_df))
    dist_rad, idx = tree.query(coords_rad, k=kq)
    dist_km = dist_rad * EARTH_R_KM
    edge_map: Dict[Tuple[int, int], float] = {}
    for i in range(len(node_df)):
        js = idx[i, 1:]
        ds = dist_km[i, 1:]
        for j, d in zip(js, ds):
            w = float(np.exp(-(d ** 2) / max(2.0 * sigma_km ** 2, EPS)))
            edge_map[(i, int(j))] = max(edge_map.get((i, int(j)), 0.0), w)
            edge_map[(int(j), i)] = max(edge_map.get((int(j), i), 0.0), w)
    if not edge_map:
        raise RuntimeError('图构建失败：无边。')
    edges = np.array(list(edge_map.keys()), dtype=np.int64).T
    weights = np.array(list(edge_map.values()), dtype=np.float32)
    return edges, weights


def _canonical_station_id(value) -> str:
    """Normalize a station id for robust column matching."""
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text


def find_met_station_columns(met: pd.DataFrame, station_ids: Sequence) -> List:
    """Find observation columns by StationID instead of positional row order.

    Accepted column styles include ``Station_50136``, ``station_50136``, ``50136``
    and integer-like column names after CSV parsing.  The returned items are the
    original pandas column keys, so this works even if pandas keeps a non-string
    column label.
    """
    lookup: Dict[str, object] = {}
    for col in met.columns:
        raw = str(col).strip()
        canon_raw = _canonical_station_id(raw)
        keys = {raw, raw.lower(), canon_raw, canon_raw.lower()}
        if raw.lower().startswith('station_'):
            sid = _canonical_station_id(raw.split('_', 1)[1])
            keys.update({sid, sid.lower(), f'Station_{sid}', f'station_{sid}', f'STATION_{sid}'})
        for key in keys:
            lookup.setdefault(str(key), col)

    selected = []
    missing = []
    for sid_raw in station_ids:
        sid = _canonical_station_id(sid_raw)
        candidates = [
            f'Station_{sid}', f'station_{sid}', f'STATION_{sid}',
            sid, sid.lower(), sid.upper(),
        ]
        found = None
        for cand in candidates:
            if cand in lookup:
                found = lookup[cand]
                break
            if cand.lower() in lookup:
                found = lookup[cand.lower()]
                break
        if found is None:
            missing.append(sid)
        else:
            selected.append(found)

    if missing:
        preview = ', '.join(missing[:20])
        raise KeyError(
            f'max_metdata.csv 中找不到 {len(missing)} 个站点观测列，示例: {preview}. '
            f'请确认列名是否为 Station_站号 或纯站号。'
        )
    if len(set(map(str, selected))) != len(selected):
        raise ValueError('按 StationID 匹配 metdata 列后出现重复列，请检查站点编号或列名。')
    return selected


def read_dynamic_csv_robust(path: Path) -> pd.DataFrame:
    """Read ERA5-Land dynamic CSV with a fallback for malformed rows.

    Some exported yearly grids can contain a small number of malformed lines
    such as a row with more comma-separated fields than the header.  Pandas'
    default C parser stops immediately in that case.  For dynamic grids we can
    safely fall back to the Python parser and skip only the malformed line(s),
    because station/grid matching uses the remaining Lat/Lon rows by nearest
    neighbor and still enforces coord_match_max_km.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        print(f'  [csv parse warn] {path.name}: pandas C parser failed: {exc}')
        skipped = {'count': 0}

        def _skip_bad_line(fields: List[str]):
            skipped['count'] += 1
            if skipped['count'] <= 5:
                print(
                    f'  [csv parse warn] {path.name}: skip malformed line #{skipped["count"]} '
                    f'with {len(fields)} fields'
                )
            return None

        try:
            df = pd.read_csv(path, engine='python', on_bad_lines=_skip_bad_line)
            print(
                f'  [csv parse recover] {path.name}: loaded with python parser; '
                f'skipped malformed lines = {skipped["count"]}; rows kept = {len(df)}'
            )
        except Exception as exc2:
            print(
                f'  [csv parse warn] {path.name}: callable bad-line fallback failed: {exc2}; '
                f'trying on_bad_lines="skip".'
            )
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
            print(f'  [csv parse recover] {path.name}: loaded with on_bad_lines=skip; rows kept = {len(df)}')

    df = df.replace(-999, np.nan)
    missing = {'Lat', 'Lon'} - set(df.columns)
    if missing:
        raise KeyError(f'{path.name} 缺少坐标列: {sorted(missing)}')
    pcols = [c for c in df.columns if str(c).startswith('p')]
    if not pcols:
        raise ValueError(f'{path.name} 缺少 p* 日列。')
    return df


def nearest_match_indices(ref_coords: np.ndarray,
                          query_coords: np.ndarray,
                          max_dist_km: float,
                          label: str,
                          raise_on_exceed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor coordinate alignment with an optional hard distance tolerance."""
    ref_coords = np.asarray(ref_coords, dtype=np.float64)
    query_coords = np.asarray(query_coords, dtype=np.float64)
    if ref_coords.ndim != 2 or ref_coords.shape[1] != 2:
        raise ValueError(f'{label}: ref_coords 必须为 [M,2] Lat/Lon。')
    if query_coords.ndim != 2 or query_coords.shape[1] != 2:
        raise ValueError(f'{label}: query_coords 必须为 [N,2] Lat/Lon。')
    if len(ref_coords) == 0 or len(query_coords) == 0:
        raise ValueError(f'{label}: 坐标为空，无法匹配。')
    if not np.isfinite(ref_coords).all() or not np.isfinite(query_coords).all():
        raise ValueError(f'{label}: 坐标中存在 NaN/Inf。')

    cache_key = (array_digest(ref_coords), array_digest(query_coords))
    if cache_key in _MATCH_INDEX_CACHE:
        idx, dist_km = _MATCH_INDEX_CACHE.pop(cache_key)
        _MATCH_INDEX_CACHE[cache_key] = (idx, dist_km)
    else:
        tree = BallTree(np.radians(ref_coords), metric='haversine')
        dist_rad, idx = tree.query(np.radians(query_coords), k=1)
        dist_km = (dist_rad[:, 0] * EARTH_R_KM).astype(np.float64)
        idx = idx[:, 0].astype(np.int64)
        _MATCH_INDEX_CACHE[cache_key] = (idx, dist_km)
        while len(_MATCH_INDEX_CACHE) > 8:
            _MATCH_INDEX_CACHE.popitem(last=False)
    max_dist_km = float(max_dist_km)
    if max_dist_km > 0:
        bad = np.where(dist_km > max_dist_km)[0]
        if bad.size > 0 and raise_on_exceed:
            worst = bad[np.argsort(-dist_km[bad])[:10]]
            examples = '; '.join(
                f'query#{int(i)} dist={dist_km[i]:.2f}km lat={query_coords[i,0]:.4f} lon={query_coords[i,1]:.4f}'
                for i in worst
            )
            raise ValueError(
                f'{label}: 最近邻匹配距离超过阈值 {max_dist_km:.2f} km，'
                f'共 {bad.size}/{len(query_coords)} 个点不合格。示例: {examples}'
            )
    if len(query_coords) > 0:
        print(f'  [coord match] {label}: max={dist_km.max():.3f} km, p95={np.quantile(dist_km, 0.95):.3f} km')
    return idx, dist_km


def align_year_dynamic(year_df: pd.DataFrame, node_coords: pd.DataFrame,
                       max_dist_km: float = 100.0,
                       var_name: str = 'dynamic',
                       year: Optional[int] = None,
                       raise_on_exceed: bool = True,
                       return_dist: bool = False):
    """Align one yearly dynamic CSV to target nodes via BallTree nearest neighbor.

    This replaces exact ``Lat/Lon`` merge, so small floating-point or rounding
    differences no longer silently produce NaN. A hard distance threshold still
    prevents gross coordinate mismatches.
    """
    doy_cols = sorted(
        [c for c in year_df.columns if str(c).startswith('p')],
        key=lambda c: int(str(c)[1:]) if str(c)[1:].isdigit() else 10**9,
    )
    if year is not None:
        expected_days = int((pd.Timestamp(int(year) + 1, 1, 1) - pd.Timestamp(int(year), 1, 1)).days)
        expected_cols = [f'p{day}' for day in range(1, expected_days + 1)]
        if set(doy_cols) != set(expected_cols) or len(doy_cols) != len(expected_cols):
            missing = sorted(set(expected_cols) - set(doy_cols))
            extra = sorted(set(doy_cols) - set(expected_cols))
            raise ValueError(f'{var_name}_{year}: daily columns do not match calendar; '
                             f'missing={missing[:5]}, extra={extra[:5]}')
        doy_cols = expected_cols
    if len(doy_cols) == 0:
        raise ValueError('动态文件缺少 p* 日列。')
    if not {'Lat', 'Lon'}.issubset(set(year_df.columns)):
        raise KeyError('动态文件缺少 Lat/Lon 列。')
    if not {'Lat', 'Lon'}.issubset(set(node_coords.columns)):
        raise KeyError('目标节点坐标缺少 Lat/Lon 列。')

    label = f'{var_name}_{year}' if year is not None else str(var_name)
    idx, dist_km = nearest_match_indices(
        year_df[['Lat', 'Lon']].to_numpy(np.float64),
        node_coords[['Lat', 'Lon']].to_numpy(np.float64),
        max_dist_km=max_dist_km,
        label=label,
        raise_on_exceed=raise_on_exceed,
    )
    aligned = year_df.iloc[idx][doy_cols].to_numpy(np.float32).T
    if return_dist:
        return aligned, dist_km
    return aligned


_MATCH_INDEX_CACHE: OrderedDict = OrderedDict()


def array_digest(array: np.ndarray) -> str:
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((x.shape, x.dtype.str)).encode())
    h.update(memoryview(x).cast('B'))
    return h.hexdigest()


def source_file_signature(cfg: ExperimentConfig) -> str:
    """Invalidate aligned/event caches when source files or alignment settings change."""
    root = resolve_input_directory(cfg.input_dir)
    names = ['静态数据汇总.xlsx', 'max_sitelist.csv', 'max_date_data.csv', 'max_metdata.csv']
    names += [f'{var}_{year}.csv' for year in cfg.years for var in cfg.dynamic_vars]
    records = []
    for name in names:
        path = root / name
        if path.exists():
            stat = path.stat()
            records.append((name, stat.st_size, stat.st_mtime_ns))
        else:
            records.append((name, None, None))
    payload = {'root': str(root.resolve()), 'files': records, 'years': list(cfg.years),
               'dynamic_vars': list(cfg.dynamic_vars), 'coord_match_max_km': cfg.coord_match_max_km,
               'bh_reference_year': cfg.bh_reference_year, 'schema': 'raw_no_coord_calendar_checked_v2'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_date_axis(dates: pd.DataFrame, years: Sequence[int]) -> None:
    years = [int(year) for year in years]
    if not years or years != sorted(set(years)):
        raise ValueError('years must be a nonempty ascending sequence without duplicates.')
    actual = pd.DatetimeIndex(pd.to_datetime(dates[['Year', 'Month', 'Day']]))
    expected = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f'{year}-01-01', f'{year}-12-31').to_numpy() for year in years
    ]))
    if actual.has_duplicates or not actual.equals(expected):
        raise ValueError('Date rows must contain every requested calendar day exactly once in ascending order.')
    if 'date' in dates and not pd.DatetimeIndex(pd.to_datetime(dates['date'])).equals(actual):
        raise ValueError('Cached date labels disagree with Year/Month/Day.')


def legacy_source_cache_is_current(cfg: ExperimentConfig, old_cfg: Dict,
                                   old_npz: Path, old_pkl: Path) -> bool:
    """Legacy caches lack fingerprints: verify source directory and file freshness."""
    if not old_cfg.get('input_dir'):
        return False
    root = resolve_input_directory(cfg.input_dir)
    try:
        old_root = resolve_input_directory(old_cfg['input_dir'])
    except FileNotFoundError:
        return False
    if old_root.resolve() != root.resolve():
        return False
    cache_time = min(old_npz.stat().st_mtime_ns, old_pkl.stat().st_mtime_ns)
    names = ['静态数据汇总.xlsx', 'max_sitelist.csv', 'max_date_data.csv', 'max_metdata.csv']
    names += [f'{var}_{year}.csv' for year in cfg.years for var in cfg.dynamic_vars]
    return all((root / name).exists() and (root / name).stat().st_mtime_ns <= cache_time
               for name in names)


def source_bundle_from_raw(meta: Dict, arrays, cfg: ExperimentConfig) -> DatasetBundle:
    node_df = meta['node_df'].copy().reset_index(drop=True)
    dates = meta['dates'].copy().reset_index(drop=True)
    validate_date_axis(dates, cfg.years)
    if list(meta['dyn_feature_names']) != list(cfg.dynamic_vars):
        raise ValueError('Cached dynamic feature order differs from configuration.')
    names = list(meta['const_feature_names'])
    keep = [i for i, name in enumerate(names) if name not in ('Lat', 'Lon')]
    static_const = np.asarray(arrays['static_const'], dtype=np.float32)[:, keep]
    const_names = [names[i] for i in keep]
    daily_dyn = np.asarray(arrays['daily_dyn'], dtype=np.float32)
    yearly_feats = np.asarray(arrays['yearly_feats'], dtype=np.float32)
    obs = np.asarray(arrays['obs'], dtype=np.float32)
    obs_mask = np.asarray(arrays['obs_mask'], dtype=bool)
    year_index = dates['Year'].map({year: i for i, year in enumerate(cfg.years)}).to_numpy(np.int64)
    doy = pd.to_datetime(dates['date']).dt.dayofyear.to_numpy(np.float32)
    date_feat = np.stack([np.sin(2 * np.pi * doy / 366.0),
                          np.cos(2 * np.pi * doy / 366.0)], axis=-1)
    if obs.ndim != 2:
        raise ValueError('Observation cache must be a [date, station] matrix.')
    D, N = obs.shape
    expected_shapes = {
        'obs_mask': (D, N), 'daily_dyn': (D, N, len(cfg.dynamic_vars)),
        'static_const': (N, len(const_names)),
        'yearly_feats': (len(cfg.years), N, len(meta['year_feature_names'])),
    }
    actual_arrays = {'obs_mask': obs_mask, 'daily_dyn': daily_dyn,
                     'static_const': static_const, 'yearly_feats': yearly_feats}
    if len(node_df) != N or len(dates) != D:
        raise ValueError('Cached station/date metadata do not match raw array dimensions.')
    for name, shape in expected_shapes.items():
        if actual_arrays[name].shape != shape:
            raise ValueError(f'Cached {name} shape {actual_arrays[name].shape} != {shape}.')
    if np.any(obs_mask & ~np.isfinite(obs)):
        raise ValueError('Observation mask marks nonfinite labels as valid.')
    day_features = np.concatenate([
        np.broadcast_to(static_const[None], (D, N, len(const_names))),
        yearly_feats[year_index], daily_dyn,
        np.broadcast_to(date_feat[:, None, :], (D, N, 2)),
    ], axis=-1).astype(np.float32, copy=False)
    empty = np.empty((D, N, 0), dtype=np.float32)
    false_mask = np.zeros(N, dtype=bool)
    direct = initialize_direct_target_state(obs, daily_dyn.shape[-1])
    return DatasetBundle(
        node_df=node_df, dates=dates, static_const=static_const,
        static_const_scaled=np.empty((N, 0), dtype=np.float32),
        yearly_feats=yearly_feats, daily_dyn=daily_dyn, daily_dyn_scaled=empty,
        day_features=day_features, day_features_scaled=empty,
        obs=obs, obs_mask=obs_mask, base=direct[0], residual=direct[1],
        train_node_mask=false_mask.copy(), val_node_mask=false_mask.copy(),
        calib_node_mask=false_mask.copy(), test_node_mask=false_mask.copy(),
        edge_index=np.asarray(arrays['edge_index'], dtype=np.int64),
        edge_weight=np.asarray(arrays['edge_weight'], dtype=np.float32),
        bh_conf=np.asarray(arrays['bh_conf'], dtype=np.float32),
        tail_thresholds={}, train_tail_threshold=0.0,
        feature_names=const_names + list(meta['year_feature_names']) + list(cfg.dynamic_vars) + ['DOY_sin', 'DOY_cos'],
        const_feature_names=const_names, year_feature_names=list(meta['year_feature_names']),
        dyn_feature_names=list(cfg.dynamic_vars), era5_reg_coef=direct[2],
        era5_reg_intercept=direct[3], era5_reg_feature_coef=direct[4],
        split_summary={'split_mode': 'aligned_source_only'}, scaler_stats={}, climate_vectors=empty,
    )


def save_aligned_raw_cache(npz_path: Path, pkl_path: Path,
                           bundle: DatasetBundle, signature: str) -> None:
    # Uncompressed NPZ avoids spending startup time compressing derived arrays;
    # only aligned raw inputs are retained, once for all folds.
    tmp = npz_path.with_suffix('.tmp.npz')
    np.savez(tmp, static_const=bundle.static_const, yearly_feats=bundle.yearly_feats,
             daily_dyn=bundle.daily_dyn, obs=bundle.obs, obs_mask=bundle.obs_mask,
             edge_index=bundle.edge_index, edge_weight=bundle.edge_weight, bh_conf=bundle.bh_conf)
    tmp.replace(npz_path)
    save_pickle({'node_df': bundle.node_df, 'dates': bundle.dates,
                 'const_feature_names': bundle.const_feature_names,
                 'year_feature_names': bundle.year_feature_names,
                 'dyn_feature_names': bundle.dyn_feature_names, 'source_signature': signature}, str(pkl_path))


def build_dataset_cache(cfg: ExperimentConfig) -> DatasetBundle:
    """Read/align once. Fold-specific scaling and climate features are built later."""
    cfg.prepare_dirs()
    signature = source_file_signature(cfg)
    raw_npz = Path(cfg.cache_dir) / 'aligned_raw_no_coord_v1.npz'
    raw_pkl = Path(cfg.cache_dir) / 'aligned_raw_no_coord_v1.pkl'
    if not cfg.force_recompute and raw_npz.exists() and raw_pkl.exists():
        meta = load_pickle(str(raw_pkl))
        if meta.get('source_signature') == signature:
            try:
                with np.load(raw_npz, allow_pickle=False) as arrays:
                    bundle = source_bundle_from_raw(meta, arrays, cfg)
            except (ValueError, KeyError) as exc:
                print(f'  [source cache] rebuilding mismatched cache: {exc}')
            else:
                print('  [source cache] reused aligned raw data; fold features are built per fold.')
                return bundle

    # Existing aligned caches can supply raw arrays without decoding their large
    # scaled/climate arrays. Names and dimensions are rebuilt under this schema.
    if not cfg.force_recompute:
        for directory in cfg.legacy_cache_dirs:
            directory = Path(directory)
            old_npz = directory / f'dataset_cache_{CODE_VERSION}.npz'
            old_pkl = directory / f'dataset_cache_{CODE_VERSION}.pkl'
            old_config = directory / f'dataset_config_{CODE_VERSION}.json'
            if not all(path.exists() for path in (old_npz, old_pkl, old_config)):
                continue
            with old_config.open(encoding='utf-8') as f:
                old_cfg = json.load(f)
            if (list(old_cfg.get('years', [])) != list(cfg.years)
                    or list(old_cfg.get('dynamic_vars', [])) != list(cfg.dynamic_vars)
                    or float(old_cfg.get('coord_match_max_km', -1)) != cfg.coord_match_max_km
                    or int(old_cfg.get('bh_reference_year', -1)) != cfg.bh_reference_year):
                continue
            if not legacy_source_cache_is_current(cfg, old_cfg, old_npz, old_pkl):
                print(f'  [source cache] skipping outdated legacy cache: {directory}')
                continue
            old_meta = load_pickle(str(old_pkl))
            try:
                with np.load(old_npz, allow_pickle=False) as arrays:
                    bundle = source_bundle_from_raw(old_meta, arrays, cfg)
            except (ValueError, KeyError) as exc:
                print(f'  [source cache] skipping mismatched legacy cache: {exc}')
                continue
            save_aligned_raw_cache(raw_npz, raw_pkl, bundle, signature)
            print(f'  [source cache] migrated raw aligned arrays from {directory}')
            return bundle

    input_dir = resolve_input_directory(cfg.input_dir)
    print(f'  [data] input_dir = {input_dir}')
    static_path = input_dir / '静态数据汇总.xlsx'
    site_path = input_dir / 'max_sitelist.csv'
    date_path = input_dir / 'max_date_data.csv'
    met_path = input_dir / 'max_metdata.csv'
    if not static_path.exists() or not site_path.exists() or not date_path.exists() or not met_path.exists():
        raise FileNotFoundError(
            f'缺少基础输入文件。需要: {static_path.name}, {site_path.name}, {date_path.name}, {met_path.name}; '
            f'当前 input_dir={input_dir}'
        )

    static = pd.read_excel(static_path).replace(-999, np.nan)
    sites = pd.read_csv(site_path).replace(-999, np.nan)
    dates = pd.read_csv(date_path)
    met = pd.read_csv(met_path).replace(-999, np.nan)

    if len(dates) != len(met):
        raise ValueError('max_date_data.csv and max_metdata.csv have different row counts.')
    dates = dates[['Year', 'Month', 'Day']].copy()
    dates['date'] = pd.to_datetime(dates[['Year', 'Month', 'Day']])
    if dates['date'].duplicated().any():
        raise ValueError('max_date_data.csv contains duplicate dates.')
    row_order = np.argsort(dates['date'].to_numpy(), kind='stable')
    dates = dates.iloc[row_order].reset_index(drop=True)
    met = met.iloc[row_order].reset_index(drop=True)
    keep_mask = dates['Year'].isin(cfg.years).to_numpy()
    dates = dates.loc[keep_mask].reset_index(drop=True)
    met = met.loc[keep_mask].reset_index(drop=True)
    validate_date_axis(dates, cfg.years)

    site_lat_col = 'Latitude' if 'Latitude' in sites.columns else 'Lat'
    site_lon_col = 'Longitude' if 'Longitude' in sites.columns else 'Lon'
    if site_lat_col not in sites.columns or site_lon_col not in sites.columns:
        raise KeyError('max_sitelist.csv 缺少 Latitude/Longitude 或 Lat/Lon 坐标列。')
    if 'StationID' not in sites.columns:
        raise KeyError('max_sitelist.csv 缺少 StationID 列。')

    grid_coords = static[['Lat', 'Lon']].copy().reset_index(drop=True)
    station_coords = sites[[site_lat_col, site_lon_col]].to_numpy(np.float64)
    grid_coords_rad = np.radians(grid_coords[['Lat', 'Lon']].to_numpy(np.float64))
    station_coords_rad = np.radians(station_coords)
    tree = BallTree(grid_coords_rad, metric='haversine')
    dists_rad, nearest_idx = tree.query(station_coords_rad, k=1)
    station_static_dist_km = (dists_rad.ravel() * EARTH_R_KM).astype(np.float64)

    n_sites_raw = int(len(sites))
    static_keep_mask = station_static_dist_km <= float(cfg.coord_match_max_km)
    static_discard_count = int((~static_keep_mask).sum())
    if static_discard_count > 0:
        bad_static = np.where(~static_keep_mask)[0]
        worst = bad_static[np.argsort(-station_static_dist_km[bad_static])[:10]]
        examples = '; '.join(
            f'station_row={int(i)} StationID={sites.iloc[i].get("StationID", "NA")} dist={station_static_dist_km[i]:.2f}km'
            for i in worst
        )
        print(
            f'  [coord match drop] station->static: discarded {static_discard_count}/{n_sites_raw} '
            f'stations with nearest distance > {cfg.coord_match_max_km:.2f} km. Examples: {examples}'
        )
    if not static_keep_mask.any():
        raise ValueError(
            f'所有站点到静态网格最近邻距离均超过 {cfg.coord_match_max_km:.2f} km，无法构建数据集。'
        )
    print(f'  [coord match] station->static all: max={station_static_dist_km.max():.3f} km, '
          f'p95={np.quantile(station_static_dist_km, 0.95):.3f} km')
    print(f'  [coord match] station->static kept: max={station_static_dist_km[static_keep_mask].max():.3f} km, '
          f'p95={np.quantile(station_static_dist_km[static_keep_mask], 0.95):.3f} km')

    mapping_all = pd.DataFrame({
        'station_row': np.arange(len(sites), dtype=int),
        'grid_row': nearest_idx.ravel().astype(int),
        'dist_km': station_static_dist_km,
        'StationID': sites['StationID'].to_numpy(),
    })
    mapping = (
        mapping_all.loc[static_keep_mask]
        .sort_values('dist_km')
        .drop_duplicates('grid_row', keep='first')
        .sort_values('grid_row')
        .reset_index(drop=True)
    )
    duplicate_grid_discard_count = int(static_keep_mask.sum() - len(mapping))
    if duplicate_grid_discard_count > 0:
        print(
            f'  [coord match drop] station->static: discarded {duplicate_grid_discard_count} additional '
            f'stations because multiple stations matched the same static grid cell; kept the closest one.'
        )
    if len(mapping) < 3:
        raise ValueError(f'距离过滤和重复网格过滤后可用站点过少: {len(mapping)}')

    obs_grid_rows = mapping['grid_row'].to_numpy(dtype=int)
    station_rows = mapping['station_row'].to_numpy(dtype=int)
    node_df = static.iloc[obs_grid_rows].copy().reset_index(drop=True)
    node_df['source_station_row'] = station_rows
    node_df['source_station_id'] = mapping['StationID'].to_numpy()
    node_df['station_static_match_km'] = mapping['dist_km'].to_numpy(np.float32)
    node_df = node_df.apply(lambda s: s.fillna(s.median()) if np.issubdtype(s.dtype, np.number) else s)
    node_df = assign_station_grid(node_df, cfg.split_grid_deg)

    met_cols = find_met_station_columns(met, mapping['StationID'].to_numpy())
    obs = met.loc[:, met_cols].to_numpy(np.float32)
    node_df['source_met_col'] = [str(c) for c in met_cols]
    print(f'  [obs align] metdata columns matched by StationID: {len(met_cols)} columns')
    obs_mask = np.isfinite(obs)

    bh_cols = find_bh_year_columns(node_df)
    if not bh_cols:
        raise KeyError('静态数据缺少 BHYYYY 列；请提供 BH1984/BH1990/BH2000/BH2010/BH2019 等列。')
    if cfg.bh_reference_year in bh_cols:
        bh_static_col = bh_cols[cfg.bh_reference_year]
    else:
        bh_static_col = bh_cols[max(bh_cols)]
        print(f'  [BH warn] 未找到 BH{cfg.bh_reference_year}，静态 BH 改用 {bh_static_col}。')
    node_df['BH'] = node_df[bh_static_col]

    const_feature_names = ['Alt', 'Slo', bh_static_col]
    for c in const_feature_names:
        if c not in node_df.columns:
            raise KeyError(f'静态数据缺少列: {c}')
    static_const = node_df[const_feature_names].to_numpy(np.float32)

    yearly_feats_np, year_feature_names, bh_year_meta = build_bh_yearly_features(
        node_df, cfg.years, cfg.bh_year_feature_name
    )

    if len(cfg.dynamic_vars) == 0:
        raise ValueError('dynamic_vars 不能为空。')
    if cfg.dynamic_vars[0] != 'ERA5Land_WS':
        raise ValueError('当前基场拟合默认使用 dynamic_vars[0] 作为风速通道，请将 ERA5Land_WS 放在第一位。')

    dyn_arrays = []
    dynamic_keep_mask = np.ones(len(node_df), dtype=bool)
    dynamic_max_dist_km = np.zeros(len(node_df), dtype=np.float64)
    for year in cfg.years:
        arr_vars = []
        for var in cfg.dynamic_vars:
            path = input_dir / f'{var}_{year}.csv'
            if not path.exists():
                raise FileNotFoundError(f'缺少动态变量文件: {path}')
            year_df = read_dynamic_csv_robust(path)
            aligned, dist_km = align_year_dynamic(
                year_df, node_df[['Lat', 'Lon']],
                max_dist_km=cfg.coord_match_max_km, var_name=var, year=year,
                raise_on_exceed=False, return_dist=True,
            )
            arr_vars.append(aligned)
            dynamic_keep_mask &= dist_km <= float(cfg.coord_match_max_km)
            dynamic_max_dist_km = np.maximum(dynamic_max_dist_km, dist_km)
        arr_year = np.stack(arr_vars, axis=-1)
        year_dates = dates[dates['Year'] == year]
        if len(year_dates) != arr_year.shape[0]:
            raise ValueError(f'年份 {year} 日期数与动态数组天数不一致')
        dyn_arrays.append(arr_year)
    daily_dyn = np.concatenate(dyn_arrays, axis=0).astype(np.float32)

    dynamic_discard_count = int((~dynamic_keep_mask).sum())
    if dynamic_discard_count > 0:
        bad_dyn = np.where(~dynamic_keep_mask)[0]
        worst = bad_dyn[np.argsort(-dynamic_max_dist_km[bad_dyn])[:10]]
        examples = '; '.join(
            f'node={int(i)} StationID={node_df.iloc[i].get("source_station_id", "NA")} max_dyn_dist={dynamic_max_dist_km[i]:.2f}km '
            f'lat={node_df.iloc[i]["Lat"]:.4f} lon={node_df.iloc[i]["Lon"]:.4f}'
            for i in worst
        )
        print(
            f'  [coord match drop] node->dynamic: discarded {dynamic_discard_count}/{len(node_df)} '
            f'stations/nodes with any dynamic nearest distance > {cfg.coord_match_max_km:.2f} km. Examples: {examples}'
        )
        if not dynamic_keep_mask.any():
            raise ValueError(
                f'所有站点/节点到动态变量网格最近邻距离均超过 {cfg.coord_match_max_km:.2f} km，无法构建数据集。'
            )
        node_df = node_df.loc[dynamic_keep_mask].reset_index(drop=True)
        obs = obs[:, dynamic_keep_mask]
        obs_mask = obs_mask[:, dynamic_keep_mask]
        static_const = static_const[dynamic_keep_mask]
        yearly_feats_np = yearly_feats_np[:, dynamic_keep_mask, :]
        daily_dyn = daily_dyn[:, dynamic_keep_mask, :]
        dynamic_max_dist_km = dynamic_max_dist_km[dynamic_keep_mask]
        if len(node_df) < 3:
            raise ValueError(f'动态变量距离过滤后可用站点过少: {len(node_df)}')
    node_df['dynamic_max_match_km'] = dynamic_max_dist_km.astype(np.float32)
    if not np.isfinite(daily_dyn).all():
        med = np.nanmedian(daily_dyn.reshape(-1, daily_dyn.shape[-1]), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(~np.isfinite(daily_dyn))
        daily_dyn[inds] = med[inds[-1]]

    # Source preparation has no fold-specific feature calculation.
    # Keep the original observation/coordinate filtering and graph utilities.
    edge_index, edge_weight = build_knn_graph(node_df[['Lat', 'Lon']], cfg.graph_k, cfg.graph_sigma_km)
    bh_conf = bh_confidence(node_df['BH'].to_numpy(np.float32), alpha=cfg.bh_alpha, wmin=cfg.bh_wmin)
    raw = {
        'static_const': static_const, 'yearly_feats': yearly_feats_np,
        'daily_dyn': daily_dyn, 'obs': obs, 'obs_mask': obs_mask,
        'edge_index': edge_index, 'edge_weight': edge_weight, 'bh_conf': bh_conf,
    }
    meta = {'node_df': node_df, 'dates': dates, 'const_feature_names': const_feature_names,
            'year_feature_names': year_feature_names, 'dyn_feature_names': list(cfg.dynamic_vars)}
    bundle = source_bundle_from_raw(meta, raw, cfg)
    save_aligned_raw_cache(raw_npz, raw_pkl, bundle, signature)
    return bundle


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan)
    return dict(
        RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
        MAE=float(mean_absolute_error(yt, yp)),
        R2=float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
    )

def compute_p90_rmse(y_true: np.ndarray, y_pred: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    thr = float(thresholds.get('P90', np.nan))
    if not np.isfinite(thr):
        return {'RMSE90': np.nan}
    sel = yt >= thr
    if int(sel.sum()) < 2:
        return {'RMSE90': np.nan}
    return {'RMSE90': float(np.sqrt(mean_squared_error(yt[sel], yp[sel])))}

def flatten_split_targets(bundle: DatasetBundle, split: str) -> Tuple[np.ndarray, np.ndarray]:
    node_mask = {'Train': bundle.train_node_mask, 'Val': bundle.val_node_mask,
                 'Calib': bundle.calib_node_mask, 'Test': bundle.test_node_mask}[split]
    obs_mask = bundle.obs_mask & node_mask[None, :]
    return bundle.obs[obs_mask], obs_mask

def evaluate_prediction_matrix(bundle: DatasetBundle, pred_matrix: np.ndarray, split: str) -> Dict[str, float]:
    y_true, mask = flatten_split_targets(bundle, split)
    y_pred = np.asarray(pred_matrix, np.float64)[mask]
    res = compute_metrics(y_true, y_pred)
    res.update(compute_p90_rmse(y_true, y_pred, bundle.tail_thresholds))
    return res

def compute_paper_4p1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Overall statistics needed by ESSD Section 4.1 Taylor and metric tables."""
    y_true = np.asarray(y_true, np.float64).ravel()
    y_pred = np.asarray(y_pred, np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {
            'N': 0, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'Bias': np.nan,
            'Corr': np.nan, 'Obs_std': np.nan, 'Pred_std': np.nan,
            'Std_ratio': np.nan, 'Centered_RMSE': np.nan,
        }

    err = yp - yt
    obs_std = float(np.std(yt, ddof=1)) if len(yt) > 1 else np.nan
    pred_std = float(np.std(yp, ddof=1)) if len(yp) > 1 else np.nan
    if len(yt) > 1 and obs_std > 0 and pred_std > 0:
        corr = float(np.corrcoef(yt, yp)[0, 1])
        std_ratio = float(pred_std / obs_std)
        centered_rmse = float(np.sqrt(np.mean(((yp - yp.mean()) - (yt - yt.mean())) ** 2)))
    else:
        corr = np.nan
        std_ratio = np.nan
        centered_rmse = np.nan

    return {
        'N': int(len(yt)),
        'RMSE': float(np.sqrt(np.mean(err ** 2))),
        'MAE': float(np.mean(np.abs(err))),
        'R2': float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        'Bias': float(np.mean(err)),
        'Corr': corr,
        'Obs_std': obs_std,
        'Pred_std': pred_std,
        'Std_ratio': std_ratio,
        'Centered_RMSE': centered_rmse,
    }

def save_paper_4p1_outputs(bundle: DatasetBundle,
                           all_predictions: Dict[str, np.ndarray],
                           work_dir: Path,
                           sample_per_method_split: int = 200000) -> None:
    """Save compact, paper-oriented outputs for Section 4.1 plotting.

    Full metrics are computed from all valid Val/Test samples.  Error samples are
    only for distribution and hexbin plotting, keeping output size controlled.
    """
    out_dir = work_dir / 'paper_4p1_overall_accuracy'
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20250708)

    metric_rows = []
    sample_frames = []
    split_masks = {
        'Val': bundle.val_node_mask,
        'Test': bundle.test_node_mask,
    }
    for split, node_mask in split_masks.items():
        obs_mask = bundle.obs_mask & node_mask[None, :]
        day_idx_all, node_idx_all = np.where(obs_mask)
        y_true_all = np.asarray(bundle.obs, np.float64)[obs_mask]
        for method_name, pred_matrix in all_predictions.items():
            pred_arr = np.asarray(pred_matrix, np.float64)
            y_pred_all = pred_arr[obs_mask]
            row = {'Split': split, 'Method': method_name}
            row.update(compute_paper_4p1_metrics(y_true_all, y_pred_all))
            row.update(compute_p90_rmse(y_true_all, y_pred_all, bundle.tail_thresholds))
            row['N_nodes'] = int(node_mask.sum())
            metric_rows.append(row)

            valid = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
            valid_idx = np.flatnonzero(valid)
            if len(valid_idx) == 0:
                continue
            if len(valid_idx) > sample_per_method_split:
                pick = rng.choice(valid_idx, size=sample_per_method_split, replace=False)
                pick.sort()
            else:
                pick = valid_idx
            sample_frames.append(pd.DataFrame({
                'Split': split,
                'Method': method_name,
                'day_index': day_idx_all[pick].astype(np.int32),
                'node_index': node_idx_all[pick].astype(np.int32),
                'y_obs': y_true_all[pick].astype(np.float32),
                'y_pred': y_pred_all[pick].astype(np.float32),
                'error': (y_pred_all[pick] - y_true_all[pick]).astype(np.float32),
                'abs_error': np.abs(y_pred_all[pick] - y_true_all[pick]).astype(np.float32),
            }))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / 'Table_4p1_overall_metrics_for_taylor.csv', index=False)
    if sample_frames:
        pd.concat(sample_frames, ignore_index=True).to_csv(
            out_dir / 'Table_4p1_error_sample_for_box_hexbin.csv',
            index=False,
            encoding='utf-8-sig',
        )

    node_out = bundle.node_df.copy()
    node_out.to_csv(out_dir / 'Table_4p1_node_split_metadata.csv', index=False, encoding='utf-8-sig')

    note = {
        'purpose': 'ESSD Section 4.1 overall accuracy plotting outputs',
        'metrics_table': 'Table_4p1_overall_metrics_for_taylor.csv',
        'error_sample_table': 'Table_4p1_error_sample_for_box_hexbin.csv',
        'node_metadata_table': 'Table_4p1_node_split_metadata.csv',
        'sample_per_method_split': int(sample_per_method_split),
        'methods': list(all_predictions.keys()),
        'splits': list(split_masks.keys()),
        'note': (
            'Metrics use all valid Val/Test samples. Error sample table is for '
            'box/violin/hexbin plotting only. ERA5 hourly-derived daily maximum '
            'is not generated in this script and should be added as a separate baseline.'
        ),
    }
    with open(out_dir / 'CHECK_4p1_output_description.json', 'w', encoding='utf-8') as f:
        json.dump(note, f, ensure_ascii=False, indent=2)

# =============================================================================
# Spatial 5-fold cross-validation helpers
# =============================================================================

@dataclass
class Spatial5FoldCVConfig:
    work_dir: str = 'cnp_main_spatial5fold_no_coord_gpd_tail/'
    n_folds: int = 5
    lat_band_deg: float = 3.0
    run_folds: Tuple[int, ...] = tuple(range(1, 6))
    sample_per_method_fold: int = 200000
    save_test_npz: bool = True
    save_train_val_npz: bool = True
    save_test_sample_csv: bool = True
    enabled_baselines: Tuple[str, ...] = tuple()

    def prepare_dirs(self) -> None:
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def assign_interleaved_spatial_folds_for_nodes(node_df: pd.DataFrame,
                                               n_folds: int,
                                               lat_band_deg: float) -> np.ndarray:
    """Assign spatially dispersed folds for interpolation-style CV."""
    order_items = []
    for idx, row in node_df.reset_index(drop=True).iterrows():
        lat = float(row['Lat'])
        lon = float(row['Lon'])
        lat_band = math.floor(lat / lat_band_deg)
        lon_key = lon if lat_band % 2 == 0 else -lon
        order_items.append((lat_band, lon_key, lat, lon, int(idx)))
    order_items.sort()

    folds = np.zeros(len(node_df), dtype=np.int64)
    spatial_order = np.zeros(len(node_df), dtype=np.int64)
    for order_pos, item in enumerate(order_items):
        idx = item[-1]
        folds[idx] = order_pos % n_folds + 1
        spatial_order[idx] = order_pos + 1
    return folds, spatial_order


def cv_masks_from_folds(folds: np.ndarray, test_fold: int, n_folds: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = np.asarray(folds, dtype=np.int64)
    if n_folds != 5 or not 1 <= int(test_fold) <= n_folds:
        raise ValueError('Five-fold CV requires n_folds=5 and test_fold in 1..5.')
    val_folds = {test_fold % n_folds + 1}
    test_mask = folds == int(test_fold)
    val_mask = np.isin(folds, list(val_folds))
    train_mask = ~(test_mask | val_mask)
    return train_mask.astype(bool), val_mask.astype(bool), test_mask.astype(bool)


def rebuild_bundle_for_cv_fold(base_bundle: DatasetBundle,
                               data_cfg: ExperimentConfig,
                               train_node_mask: np.ndarray,
                               val_node_mask: np.ndarray,
                               test_node_mask: np.ndarray,
                               fold_id: int,
                               cv_folds: np.ndarray,
                               cv_spatial_order: np.ndarray) -> DatasetBundle:
    """Recompute fold-dependent scalers, climate vectors and split masks.

    No Stage-1 model is fitted. The compatibility ``residual`` field stores the
    observed daily maximum wind-speed target, while ``base`` remains zero.
    """
    train_node_mask = np.asarray(train_node_mask, dtype=bool)
    val_node_mask = np.asarray(val_node_mask, dtype=bool)
    test_node_mask = np.asarray(test_node_mask, dtype=bool)
    calib_node_mask = np.zeros_like(train_node_mask, dtype=bool)

    node_df = base_bundle.node_df.copy().reset_index(drop=True)
    node_df['cv_fold'] = np.asarray(cv_folds, dtype=np.int64)
    node_df['cv_spatial_order'] = np.asarray(cv_spatial_order, dtype=np.int64)
    node_df['split_precalib'] = 'unused'
    node_df.loc[train_node_mask, 'split_precalib'] = 'train'
    node_df.loc[val_node_mask, 'split_precalib'] = 'val'
    node_df.loc[test_node_mask, 'split_precalib'] = 'test'
    node_df['split_final'] = node_df['split_precalib']

    if not bool(data_cfg.direct_dailymax_target):
        raise ValueError('This script requires direct_dailymax_target=True.')
    base, residual, coef, intercept, coef_full = initialize_direct_target_state(
        base_bundle.obs, base_bundle.daily_dyn.shape[-1]
    )
    tail_thresholds, train_tail_threshold = train_tail_thresholds_from_obs(
        base_bundle.obs[:, train_node_mask], data_cfg.tail_quantile
    )
    # Node coordinates and the associated graph are fixed across folds.
    edge_index = base_bundle.edge_index
    edge_weight = base_bundle.edge_weight
    bh_conf = base_bundle.bh_conf

    static_const_scaled, static_stats = fit_scale_2d(base_bundle.static_const, train_node_mask)
    daily_dyn_scaled, dyn_stats = fit_scale_3d(base_bundle.daily_dyn, train_node_mask)
    day_features_scaled, day_stats = fit_scale_3d(base_bundle.day_features, train_node_mask)
    scaler_stats = {'static_const': static_stats, 'daily_dyn': dyn_stats, 'day_features': day_stats}

    print(f'  [cv fold {fold_id:02d}] precompute causal climate vectors...')
    climate_vectors = precompute_climate_vectors(
        daily_dyn_scaled, static_const_scaled, data_cfg.climate_hist_len
    )
    cv_ref = climate_vectors[:, train_node_mask, :].reshape(-1, climate_vectors.shape[-1])
    cv_mu = np.nanmean(cv_ref, axis=0).astype(np.float32)
    cv_sd = np.nanstd(cv_ref, axis=0).astype(np.float32)
    del cv_ref
    cv_mu = np.where(np.isfinite(cv_mu), cv_mu, 0.0)
    cv_sd = np.where((np.isfinite(cv_sd)) & (cv_sd > 0), cv_sd, 1.0)
    scaler_stats['climate_vectors'] = {'mean': cv_mu.tolist(), 'std': cv_sd.tolist()}
    climate_vectors -= cv_mu[None, None, :]
    climate_vectors /= cv_sd[None, None, :]
    np.nan_to_num(climate_vectors, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    split_summary = {
        'split_mode': 'spatial_interleaved_5fold_cv_train3_val1_test1',
        'fold_id': int(fold_id),
        'total_nodes': int(len(node_df)),
        'train_nodes': int(train_node_mask.sum()),
        'val_nodes': int(val_node_mask.sum()),
        'calib_nodes': int(calib_node_mask.sum()),
        'test_nodes': int(test_node_mask.sum()),
        'train_ratio_actual': float(train_node_mask.mean()),
        'val_ratio_actual': float(val_node_mask.mean()),
        'test_ratio_actual': float(test_node_mask.mean()),
    }

    return DatasetBundle(
        node_df=node_df,
        dates=base_bundle.dates.copy(),
        static_const=np.asarray(base_bundle.static_const, dtype=np.float32),
        static_const_scaled=np.asarray(static_const_scaled, dtype=np.float32),
        yearly_feats=np.asarray(base_bundle.yearly_feats, dtype=np.float32),
        daily_dyn=np.asarray(base_bundle.daily_dyn, dtype=np.float32),
        daily_dyn_scaled=np.asarray(daily_dyn_scaled, dtype=np.float32),
        day_features=np.asarray(base_bundle.day_features, dtype=np.float32),
        day_features_scaled=np.asarray(day_features_scaled, dtype=np.float32),
        obs=np.asarray(base_bundle.obs, dtype=np.float32),
        obs_mask=np.asarray(base_bundle.obs_mask, dtype=bool),
        base=base.astype(np.float32),
        residual=residual.astype(np.float32),
        train_node_mask=train_node_mask,
        val_node_mask=val_node_mask,
        calib_node_mask=calib_node_mask,
        test_node_mask=test_node_mask,
        edge_index=edge_index.astype(np.int64),
        edge_weight=edge_weight.astype(np.float32),
        bh_conf=bh_conf.astype(np.float32),
        tail_thresholds=tail_thresholds,
        train_tail_threshold=train_tail_threshold,
        feature_names=base_bundle.feature_names,
        const_feature_names=base_bundle.const_feature_names,
        year_feature_names=base_bundle.year_feature_names,
        dyn_feature_names=base_bundle.dyn_feature_names,
        era5_reg_coef=coef,
        era5_reg_intercept=intercept,
        era5_reg_feature_coef=np.asarray(coef_full, dtype=np.float32),
        split_summary=split_summary,
        scaler_stats=scaler_stats,
        climate_vectors=climate_vectors,
    )


def train_tail_thresholds_from_obs(obs_train: np.ndarray, tail_quantile: float) -> Tuple[Dict[str, float], float]:
    return train_tail_thresholds(obs_train, tail_quantile)


def cv_safe_method_name(method_name: str) -> str:
    return sanitize_name(method_name).replace('-', '_')


def export_fold_outputs(fold_dir: Path,
                        fold_id: int,
                        bundle: DatasetBundle,
                        all_predictions: Dict[str, np.ndarray],
                        cv_cfg: Spatial5FoldCVConfig,
                        tail_outputs: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Save all per-fold outputs required for downstream 4.1-4.4 figures."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / 'preds').mkdir(parents=True, exist_ok=True)
    test_idx = np.where(bundle.test_node_mask)[0].astype(np.int64)
    obs_test = bundle.obs[:, test_idx].astype(np.float32)
    obs_mask_test = bundle.obs_mask[:, test_idx].astype(bool)

    expected_shape = bundle.obs.shape
    for method_name, prediction in all_predictions.items():
        if np.asarray(prediction).shape != expected_shape:
            raise ValueError(f'{method_name}: prediction shape must match [date, station] {expected_shape}.')
    if tail_outputs is not None:
        for name in ('exceedance_probability', 'gpd_scale', 'tail_conditional_mean'):
            if np.asarray(tail_outputs[name]).shape != expected_shape:
                raise ValueError(f'{name}: predictive field shape must match {expected_shape}.')

    # 1) Three independent, complete split archives; no duplicate all-station archive.
    # All [day, station] matrices retain missing observations and their mask.
    split_masks = {'train': bundle.train_node_mask, 'val': bundle.val_node_mask,
                   'test': bundle.test_node_mask}
    for split, node_mask in split_masks.items():
        if (split == 'test' and not cv_cfg.save_test_npz
                or split != 'test' and not cv_cfg.save_train_val_npz):
            continue
        idx = np.flatnonzero(node_mask).astype(np.int32)
        nodes = bundle.node_df.iloc[idx]
        npz_payload = {
            'fold_id': np.asarray([fold_id], dtype=np.int16),
            f'{split}_node_indices': idx,
            'day_index': np.arange(bundle.n_days, dtype=np.int32),
            'date': pd.to_datetime(bundle.dates['date']).dt.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
            'Year': bundle.dates['Year'].to_numpy(np.int16),
            'Month': bundle.dates['Month'].to_numpy(np.int8),
            'Day': bundle.dates['Day'].to_numpy(np.int8),
            'StationID': nodes['source_station_id'].astype(str).to_numpy(dtype=str),
            'Lat': nodes['Lat'].to_numpy(np.float32),
            'Lon': nodes['Lon'].to_numpy(np.float32),
            'split': np.full(len(idx), split, dtype='U5'),
            f'obs_{split}': bundle.obs[:, idx],
            f'obs_mask_{split}': bundle.obs_mask[:, idx].astype(np.uint8),
        }
        for method_name, pred_matrix in all_predictions.items():
            npz_payload[f'pred_{cv_safe_method_name(method_name)}'] = np.asarray(pred_matrix, np.float32)[:, idx]
        if tail_outputs is not None:
            for name in ('exceedance_probability', 'gpd_scale', 'tail_conditional_mean'):
                npz_payload[name] = np.asarray(tail_outputs[name], np.float32)[:, idx]
            for name in ('gpd_shape', 'tail_threshold'):
                npz_payload[name] = np.asarray(tail_outputs[name], np.float32)
            npz_payload['tail_quantile'] = np.asarray(tail_outputs['tail_quantile'], np.float32).reshape(1)
        np.savez_compressed(
            fold_dir / f'NPZ_fold{fold_id:02d}_{split}_predictions_all_methods.npz', **npz_payload
        )

    # 2) Node metadata with split labels.
    node_meta = bundle.node_df.copy()
    node_meta['node_index'] = np.arange(len(node_meta), dtype=np.int32)
    node_meta.to_csv(fold_dir / f'Table_fold{fold_id:02d}_node_split_metadata.csv',
                     index=False, encoding='utf-8-sig')

    # 3) Overall metrics and station metrics.
    overall_rows = []
    station_rows = []
    for method_name, pred_matrix in all_predictions.items():
        pred_arr = np.asarray(pred_matrix, np.float64)
        basic = evaluate_prediction_matrix(bundle, pred_arr, 'Test')
        y_true, obs_mask = flatten_split_targets(bundle, 'Test')
        y_pred = pred_arr[obs_mask]
        paper_basic = compute_paper_4p1_metrics(y_true, y_pred)
        row = {
            'fold': int(fold_id),
            'Split': 'Test',
            'Method': method_name,
        }
        row.update(basic)
        row.update({
            'N': paper_basic['N'],
            'Bias': paper_basic['Bias'],
            'Corr': paper_basic['Corr'],
            'Obs_std': paper_basic['Obs_std'],
            'Pred_std': paper_basic['Pred_std'],
            'Std_ratio': paper_basic['Std_ratio'],
            'Centered_RMSE': paper_basic['Centered_RMSE'],
        })
        overall_rows.append(row)

        for node_idx in test_idx:
            m = bundle.obs_mask[:, node_idx] & np.isfinite(pred_arr[:, node_idx])
            if not np.any(m):
                continue
            err = pred_arr[m, node_idx] - bundle.obs[m, node_idx].astype(np.float64)
            meta = bundle.node_df.iloc[int(node_idx)]
            station_rows.append({
                'fold': int(fold_id),
                'Method': method_name,
                'node_index': int(node_idx),
                'StationID': str(meta.get('source_station_id', meta.get('StationID', node_idx))),
                'Lat': float(meta['Lat']),
                'Lon': float(meta['Lon']),
                'N': int(len(err)),
                'RMSE': float(np.sqrt(np.mean(err ** 2))),
                'MAE': float(np.mean(np.abs(err))),
                'Bias': float(np.mean(err)),
            })

    pd.DataFrame(overall_rows).to_csv(
        fold_dir / f'Table_fold{fold_id:02d}_overall_metrics.csv',
        index=False, encoding='utf-8-sig',
    )
    pd.DataFrame(station_rows).to_csv(
        fold_dir / f'Table_fold{fold_id:02d}_station_metrics.csv',
        index=False, encoding='utf-8-sig',
    )

    # 4) Small long-format sample for quick diagnostics.
    if cv_cfg.save_test_sample_csv:
        rng = np.random.default_rng(20250710 + fold_id)
        sample_frames = []
        day_grid, local_node_grid = np.where(obs_mask_test)
        if len(day_grid) > 0:
            max_n = int(cv_cfg.sample_per_method_fold)
            if len(day_grid) > max_n:
                pick = rng.choice(np.arange(len(day_grid)), size=max_n, replace=False)
                pick.sort()
            else:
                pick = np.arange(len(day_grid))
            day_pick = day_grid[pick].astype(np.int32)
            local_pick = local_node_grid[pick].astype(np.int32)
            global_pick = test_idx[local_pick].astype(np.int32)
            y_obs = obs_test[day_pick, local_pick].astype(np.float32)
            date_rows = bundle.dates.iloc[day_pick].reset_index(drop=True)
            for method_name, pred_matrix in all_predictions.items():
                pred_vals = np.asarray(pred_matrix, np.float32)[day_pick, global_pick]
                sample_frames.append(pd.DataFrame({
                    'fold': int(fold_id),
                    'Method': method_name,
                    'day_index': day_pick,
                    'Year': date_rows['Year'].to_numpy(np.int32),
                    'Month': date_rows['Month'].to_numpy(np.int16),
                    'Day': date_rows['Day'].to_numpy(np.int16),
                    'node_index': global_pick,
                    'StationID': bundle.node_df.iloc[global_pick]['source_station_id'].astype(str).to_numpy(),
                    'Lat': bundle.node_df.iloc[global_pick]['Lat'].to_numpy(np.float32),
                    'Lon': bundle.node_df.iloc[global_pick]['Lon'].to_numpy(np.float32),
                    'obs': y_obs,
                    'pred': pred_vals.astype(np.float32),
                    'error': (pred_vals - y_obs).astype(np.float32),
                    'abs_error': np.abs(pred_vals - y_obs).astype(np.float32),
                }))
        if sample_frames:
            pd.concat(sample_frames, ignore_index=True).to_csv(
                fold_dir / f'Table_fold{fold_id:02d}_test_prediction_sample.csv',
                index=False, encoding='utf-8-sig',
            )

    # 5) Fold configuration note.
    with open(fold_dir / f'CHECK_fold{fold_id:02d}_outputs.json', 'w', encoding='utf-8') as f:
        json.dump({
            'fold': int(fold_id),
            'purpose': 'Per-fold outputs for spatial 5-fold cross-validation.',
            'methods': list(all_predictions.keys()),
            'test_nodes': int(bundle.test_node_mask.sum()),
            'val_nodes': int(bundle.val_node_mask.sum()),
            'train_nodes': int(bundle.train_node_mask.sum()),
            'npz': f'NPZ_fold{fold_id:02d}_test_predictions_all_methods.npz',
            'split_npz': {split: f'NPZ_fold{fold_id:02d}_{split}_predictions_all_methods.npz'
                          for split in ('train', 'val', 'test')},
            'matrix_order': '[date, station], matching date/StationID/node indices',
            'body_sigma_saved': False,
            'overall_metrics': f'Table_fold{fold_id:02d}_overall_metrics.csv',
            'station_metrics': f'Table_fold{fold_id:02d}_station_metrics.csv',
            'node_metadata': f'Table_fold{fold_id:02d}_node_split_metadata.csv',
            'sample_csv': f'Table_fold{fold_id:02d}_test_prediction_sample.csv',
        }, f, ensure_ascii=False, indent=2)

def compute_crps_gaussian_point(y_true: np.ndarray, y_pred: np.ndarray,
                                sigma: Optional[np.ndarray] = None) -> float:
    """Proper CRPS for a Gaussian distribution centered at y_pred with std sigma.

    If sigma is None, falls back to a residual-based sigma estimated from
    all (y_true, y_pred) pairs (a common trick when a deterministic point
    prediction is the only thing available: CRPS becomes a proper scoring
    rule for the implied Gaussian ensemble and preserves the ranking).
    """
    y_true = np.asarray(y_true, np.float64).ravel()
    y_pred = np.asarray(y_pred, np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float('nan')
    yt, yp = y_true[mask], y_pred[mask]
    if sigma is None:
        # Residual-based sigma (MAE-ish scale)
        s = max(float(np.sqrt(np.mean((yt - yp) ** 2))), 1e-3)
        s_arr = np.full_like(yt, s)
    else:
        s_arr = np.asarray(sigma, np.float64).ravel()[mask].clip(1e-3)
    z = (yt - yp) / s_arr
    # CRPS(Normal(mu, sig), x) = sig * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) )
    from math import sqrt, pi
    phi = np.exp(-0.5 * z ** 2) / sqrt(2.0 * pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / sqrt(2.0)))
    crps = s_arr * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / sqrt(pi))
    return float(np.mean(crps))

def compute_csi_far_hit(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    """Contingency-table metrics for extreme event detection at a given threshold.

    Returns CSI (Critical Success Index = Hits / (Hits + Misses + FalseAlarms)),
    POD (Prob. of Detection = Hits / Obs-positive),
    FAR (False Alarm Ratio   = FalseAlarms / Pred-positive),
    and the raw counts.
    """
    y_true = np.asarray(y_true, np.float64).ravel()
    y_pred = np.asarray(y_pred, np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    pos_true = yt >= threshold
    pos_pred = yp >= threshold
    hits = int(np.sum(pos_true & pos_pred))
    misses = int(np.sum(pos_true & ~pos_pred))
    false_alarms = int(np.sum(~pos_true & pos_pred))
    denom_csi = hits + misses + false_alarms
    denom_pod = hits + misses
    denom_far = hits + false_alarms
    return {
        'CSI': float(hits / denom_csi) if denom_csi > 0 else float('nan'),
        'POD': float(hits / denom_pod) if denom_pod > 0 else float('nan'),
        'FAR': float(false_alarms / denom_far) if denom_far > 0 else float('nan'),
        'Hits': hits, 'Misses': misses, 'FalseAlarms': false_alarms,
    }

def compute_moran_i_residual(node_df: pd.DataFrame, residual_per_node: np.ndarray,
                             k: int = 8) -> float:
    """Global Moran's I of a per-node statistic using inverse-distance neighbor weights.

    residual_per_node: [N] aggregated residual magnitude per node (e.g. mean |y-yhat|
        over valid (day, node) pairs). Non-finite values are excluded from the stat.
    Returns: Moran's I in [-1, 1]. Positive => spatial clustering in residuals.
    """
    x = np.asarray(residual_per_node, np.float64).ravel()
    valid = np.isfinite(x)
    if valid.sum() < 3:
        return float('nan')
    idx_all = np.where(valid)[0]
    coords = np.radians(node_df[['Lat', 'Lon']].to_numpy(np.float64))[idx_all]
    xv = x[idx_all]
    xv_c = xv - xv.mean()
    tree = BallTree(coords, metric='haversine')
    kk = int(min(k + 1, len(xv)))
    d, nn_idx = tree.query(coords, k=kk)
    d_km = d * EARTH_R_KM
    # Inverse-distance weights (drop self)
    W = 1.0 / np.clip(d_km[:, 1:], 1.0, None)
    rows = np.repeat(np.arange(len(xv)), W.shape[1])
    cols = nn_idx[:, 1:].ravel()
    w = W.ravel()
    num = float(np.sum(w * xv_c[rows] * xv_c[cols]))
    w_sum = float(np.sum(w))
    denom = float(np.sum(xv_c ** 2))
    if w_sum <= 0 or denom <= 0:
        return float('nan')
    N = float(len(xv))
    return (N / w_sum) * (num / denom)

def compute_extended_metrics(bundle: DatasetBundle, pred_matrix: np.ndarray, split: str,
                             report_crps: bool = True, report_csi: bool = True,
                             report_moran: bool = True) -> Dict[str, float]:
    """v20 extended metric pack: CRPS + CSI@P85/P90/P95 + Moran's I on |residual|."""
    y_true, mask = flatten_split_targets(bundle, split)
    y_pred = np.asarray(pred_matrix, np.float64)[mask]
    out: Dict[str, float] = {}
    if report_crps:
        out['CRPS'] = compute_crps_gaussian_point(y_true, y_pred)
    if report_csi:
        for tag, thr in bundle.tail_thresholds.items():
            csi = compute_csi_far_hit(y_true, y_pred, thr)
            out[f'{tag}_CSI'] = csi['CSI']
            out[f'{tag}_POD'] = csi['POD']
            out[f'{tag}_FAR'] = csi['FAR']
    if report_moran:
        # Per-node mean |residual| across split observations
        node_mask = {'Train': bundle.train_node_mask, 'Val': bundle.val_node_mask,
                     'Calib': bundle.calib_node_mask, 'Test': bundle.test_node_mask}[split]
        per_node = np.full(bundle.n_nodes, np.nan, dtype=np.float64)
        obs_mask = bundle.obs_mask & node_mask[None, :]
        abs_res_full = np.full_like(bundle.obs, np.nan, dtype=np.float64)
        idx = obs_mask
        abs_res_full[idx] = np.abs(
            np.asarray(bundle.obs, np.float64)[idx] - np.asarray(pred_matrix, np.float64)[idx]
        )
        for n in np.where(node_mask)[0]:
            vals = abs_res_full[:, n]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                per_node[n] = float(np.mean(vals))
        out['Moran_I_resid'] = compute_moran_i_residual(bundle.node_df, per_node, k=8)
    return out

# =============================================================================
# Baseline methods retained from the Phase-C template
# =============================================================================
def haversine_dm(lat_ref: np.ndarray, lon_ref: np.ndarray,
                 lat_query: np.ndarray, lon_query: np.ndarray) -> np.ndarray:
    lat_ref_r, lon_ref_r = np.radians(lat_ref), np.radians(lon_ref)
    lat_query_r, lon_query_r = np.radians(lat_query), np.radians(lon_query)
    dlat = lat_query_r[:, None] - lat_ref_r[None, :]
    dlon = lon_query_r[:, None] - lon_ref_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_query_r[:, None]) * np.cos(lat_ref_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def residual_idw(lat_train: np.ndarray, lon_train: np.ndarray, res_train: np.ndarray,
                 lat_pred: np.ndarray, lon_pred: np.ndarray,
                 power: float = 2.0, min_dist: float = 0.1, chunk: int = 4096) -> np.ndarray:
    pred = np.zeros(len(lat_pred), dtype=np.float64)
    for start in range(0, len(lat_pred), chunk):
        end = min(start + chunk, len(lat_pred))
        d = haversine_dm(lat_train, lon_train, lat_pred[start:end], lon_pred[start:end])
        d = np.maximum(d, min_dist)
        w = 1.0 / np.power(d, power)
        w /= np.maximum(w.sum(axis=1, keepdims=True), EPS)
        pred[start:end] = w @ res_train
    return pred

def fit_spherical_variogram(dist_matrix: np.ndarray, values: np.ndarray):
    gamma = 0.5 * (values[:, None] - values[None, :]) ** 2
    max_h = dist_matrix.max() * 0.5
    if max_h <= 0:
        return None
    bins = np.linspace(0, max_h, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    tri = np.triu_indices(len(values), 1)
    d_flat, g_flat = dist_matrix[tri], gamma[tri]
    b_gamma = np.zeros(20, dtype=np.float64)
    b_count = np.zeros(20, dtype=np.int64)
    for b in range(20):
        m = (d_flat >= bins[b]) & (d_flat < bins[b + 1])
        if m.any():
            b_gamma[b] = g_flat[m].mean()
            b_count[b] = int(m.sum())
    valid = b_count > 10
    if valid.sum() < 3:
        valid = b_count > 0
    if valid.sum() == 0:
        return None
    nugget = max(float(b_gamma[valid][0]) * 0.5, 0.0)
    sill = float(b_gamma[valid].max())
    if sill <= 0:
        return None
    sill_part = max(sill - nugget, 0.01)
    idx = np.where(b_gamma[valid] > 0.95 * sill)[0]
    reach = float(centers[valid][idx[0]]) if len(idx) > 0 else float(max_h)
    a = min(max(reach, 10.0), 500.0)
    return nugget, sill_part, a

def spherical_variogram(h: np.ndarray, nugget: float, sill_part: float, a: float) -> np.ndarray:
    out = np.full_like(h, nugget + sill_part, dtype=np.float64)
    mask = h < a
    hr = h[mask] / a
    out[mask] = nugget + sill_part * (1.5 * hr - 0.5 * hr ** 3)
    out[h == 0] = 0.0
    return out

def residual_ok(lat_train: np.ndarray, lon_train: np.ndarray, res_train: np.ndarray,
                lat_pred: np.ndarray, lon_pred: np.ndarray) -> np.ndarray:
    dtt = haversine_dm(lat_train, lon_train, lat_train, lon_train)
    params = fit_spherical_variogram(dtt, res_train)
    if params is None:
        return residual_idw(lat_train, lon_train, res_train, lat_pred, lon_pred)
    nugget, sill_part, a = params
    n = len(lat_train)
    G = spherical_variogram(dtt, nugget, sill_part, a)
    K = np.zeros((n + 1, n + 1), dtype=np.float64)
    K[:n, :n] = G + np.eye(n, dtype=np.float64) * 1e-6
    K[n, :n] = 1.0
    K[:n, n] = 1.0
    try:
        Kinv = np.linalg.solve(K, np.eye(n + 1, dtype=np.float64))
    except np.linalg.LinAlgError:
        Kinv = np.linalg.lstsq(K, np.eye(n + 1, dtype=np.float64), rcond=None)[0]
    dp = haversine_dm(lat_train, lon_train, lat_pred, lon_pred)
    gp = spherical_variogram(dp, nugget, sill_part, a)
    A = np.hstack([gp, np.ones((len(lat_pred), 1), dtype=np.float64)])
    lam = A @ Kinv
    return lam[:, :n] @ res_train

def maybe_subsample(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return X[idx], y[idx]

def get_enabled_external_baselines(cfg: ExperimentConfig) -> List[str]:
    allowed = ('IDW', 'OK', 'RK', 'RF', 'GBRT', 'LSTM', 'TCN')
    if not cfg.run_external_baselines:
        return []
    selected: List[str] = []
    for name in cfg.enabled_baselines:
        if name in allowed and name not in selected:
            selected.append(name)
    return selected

def run_spatial_baselines(bundle: DatasetBundle, cfg: ExperimentConfig,
                          methods: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    cache_dir = Path(cfg.cache_dir)
    lat = bundle.node_df['Lat'].to_numpy(np.float64)
    lon = bundle.node_df['Lon'].to_numpy(np.float64)
    preds: Dict[str, np.ndarray] = {}
    requested = set(methods or ('IDW', 'OK', 'RK', 'RF', 'GBRT'))
    if not requested:
        return preds

    def _load_or_fit_model(model, model_path: Path, legacy_prefix: str,
                           xtr: np.ndarray, ytr: np.ndarray):
        legacy_model = None if cfg.force_recompute else load_reusable_pickle(
            cache_dir, legacy_prefix, legacy_cache_dirs=cfg.legacy_cache_dirs)
        if model_path.exists() and not cfg.force_recompute:
            try:
                return load_pickle(str(model_path))
            except Exception:
                pass
        if legacy_model is not None:
            save_pickle(legacy_model, str(model_path))
            return legacy_model
        model.fit(xtr, ytr)
        save_pickle(model, str(model_path))
        return model

    for method in [m for m in ('IDW', 'OK') if m in requested]:
        cache_path = cache_dir / f'baseline_{method}_{CODE_VERSION}.npy'
        if not cfg.force_recompute:
            cached = load_reusable_prediction_cache(
                cache_dir, f'baseline_{method}', bundle.base.shape, bundle,
                legacy_cache_dirs=cfg.legacy_cache_dirs)
            if cached is not None:
                preds[method] = cached
                if not cache_path.exists():
                    np.save(cache_path, cached)
                continue
        out = np.full_like(bundle.base, np.nan, dtype=np.float32)
        for d in range(bundle.n_days):
            known = bundle.train_node_mask & bundle.obs_mask[d]
            if known.sum() < 3:
                continue
            res_train = bundle.residual[d, known].astype(np.float64)
            res_pred = residual_idw(lat[known], lon[known], res_train, lat, lon) if method == 'IDW' else residual_ok(
                lat[known], lon[known], res_train, lat, lon)
            out[d] = bundle.base[d] + res_pred.astype(np.float32)
        np.save(cache_path, out)
        preds[method] = out

    need_tabular = any(m in requested for m in ('RF', 'GBRT', 'RK'))
    if not need_tabular:
        return preds

    tab_X = bundle.day_features.reshape(-1, bundle.day_features.shape[-1]).astype(np.float32)
    tab_y = bundle.residual.reshape(-1).astype(np.float32)
    tab_mask = (bundle.obs_mask & bundle.train_node_mask[None, :]).reshape(-1)
    X_train, y_train = tab_X[tab_mask], tab_y[tab_mask]
    X_rf, y_rf = maybe_subsample(X_train, y_train, cfg.rf_max_train_samples, cfg.seed)
    X_gb, y_gb = maybe_subsample(X_train, y_train, cfg.gbrt_max_train_samples, cfg.seed)
    X_rk, y_rk = maybe_subsample(X_train, y_train, cfg.rk_max_train_samples, cfg.seed)

    models = {
        'RF': (RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            random_state=cfg.seed,
            n_jobs=-1,
        ), X_rf, y_rf),
        'GBRT': (GradientBoostingRegressor(
            n_estimators=cfg.gbrt_n_estimators,
            learning_rate=cfg.gbrt_learning_rate,
            max_depth=cfg.gbrt_max_depth,
            subsample=cfg.gbrt_subsample,
            min_samples_leaf=cfg.gbrt_min_samples_leaf,
            random_state=cfg.seed,
        ), X_gb, y_gb),
    }
    for name, (model, xtr, ytr) in models.items():
        if name not in requested:
            continue
        cache_path = cache_dir / f'baseline_{name}_{CODE_VERSION}.npy'
        model_path = cache_dir / f'baseline_{name}_{CODE_VERSION}.pkl'
        if not cfg.force_recompute:
            cached = load_reusable_prediction_cache(
                cache_dir, f'baseline_{name}', bundle.base.shape, bundle,
                legacy_cache_dirs=cfg.legacy_cache_dirs)
            if cached is not None:
                preds[name] = cached
                if not cache_path.exists():
                    np.save(cache_path, cached)
                continue
        model = _load_or_fit_model(model, model_path, f'baseline_{name}', xtr, ytr)
        residual_pred = model.predict(tab_X).reshape(bundle.base.shape).astype(np.float32)
        pred = (bundle.base + residual_pred).astype(np.float32)
        np.save(cache_path, pred)
        preds[name] = pred

    if 'RK' in requested:
        rk_name = 'RK'
        rk_cache_path = cache_dir / f'baseline_{rk_name}_{CODE_VERSION}.npy'
        cached = None
        if not cfg.force_recompute:
            cached = load_reusable_prediction_cache(
                cache_dir, f'baseline_{rk_name}', bundle.base.shape, bundle,
                legacy_cache_dirs=cfg.legacy_cache_dirs)
            if cached is not None:
                preds[rk_name] = cached
        if cached is None:
            rk_model_path = cache_dir / f'baseline_{rk_name}_reg_{CODE_VERSION}.pkl'
            rk_model = _load_or_fit_model(LinearRegression(), rk_model_path, f'baseline_{rk_name}_reg', X_rk, y_rk)
            rk_reg_pred = rk_model.predict(tab_X).reshape(bundle.base.shape).astype(np.float32)

            rk_out = np.full_like(bundle.base, np.nan, dtype=np.float32)
            for d in range(bundle.n_days):
                known = bundle.train_node_mask & bundle.obs_mask[d]
                base_day = bundle.base[d].astype(np.float32)
                reg_day = rk_reg_pred[d].astype(np.float32)
                if known.sum() < 3:
                    rk_out[d] = base_day + reg_day
                    continue
                reg_res_train = bundle.residual[d, known].astype(np.float64) - reg_day[known].astype(np.float64)
                if np.allclose(reg_res_train, reg_res_train[0], atol=1e-8, rtol=0.0):
                    krig_corr = np.full(bundle.n_nodes, float(reg_res_train[0]), dtype=np.float64)
                else:
                    krig_corr = residual_ok(lat[known], lon[known], reg_res_train, lat, lon)
                rk_out[d] = base_day + reg_day + krig_corr.astype(np.float32)

            np.save(rk_cache_path, rk_out)
            preds[rk_name] = rk_out
    return preds

class SequenceSampleDataset(Dataset):
    def __init__(self, daily_dyn: np.ndarray, day_features: np.ndarray,
                 residual: np.ndarray, obs_mask: np.ndarray,
                 node_mask: np.ndarray, seq_len: int):
        self.daily_dyn = daily_dyn.astype(np.float32)
        self.day_features = day_features.astype(np.float32)
        self.residual = residual.astype(np.float32)
        valid = np.argwhere(obs_mask & node_mask[None, :])
        valid = valid[valid[:, 0] >= seq_len - 1]
        self.samples = valid.astype(np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        day_idx, node_idx = self.samples[idx]
        seq = self.daily_dyn[day_idx - self.seq_len + 1:day_idx + 1, node_idx, :]
        cur = self.day_features[day_idx, node_idx, :]
        y = self.residual[day_idx, node_idx]
        return {'seq': torch.from_numpy(seq), 'cur': torch.from_numpy(cur),
                'y': torch.tensor(float(y), dtype=torch.float32),
                'day': int(day_idx), 'node': int(node_idx)}

class LSTMResidualModel(nn.Module):
    def __init__(self, seq_in: int, cur_in: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_in, hidden_size=hidden, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + cur_in, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(seq)
        feat = torch.cat([h[-1], cur], dim=-1)
        return self.head(feat).squeeze(-1)

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)

class TemporalConvEncoder(nn.Module):
    def __init__(self, in_ch: int, hid: int):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, hid, kernel_size=3, dilation=1), nn.GELU(),
            CausalConv1d(hid, hid, kernel_size=3, dilation=2), nn.GELU(),
            CausalConv1d(hid, hid, kernel_size=3, dilation=4), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[..., -1]

class TCNResidualModel(nn.Module):
    def __init__(self, seq_in: int, cur_in: int, hidden: int):
        super().__init__()
        self.enc = TemporalConvEncoder(seq_in, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + cur_in, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        seq_feat = self.enc(seq.transpose(1, 2))
        return self.head(torch.cat([seq_feat, cur], dim=-1)).squeeze(-1)

def train_sequence_model(bundle: DatasetBundle, cfg: ExperimentConfig,
                         model_name: str, model: nn.Module) -> np.ndarray:
    device = get_device()
    safe_name = sanitize_name(model_name)
    pred_cache = Path(cfg.cache_dir) / f'{safe_name}_{CODE_VERSION}.npy'
    ckpt = Path(cfg.ckpt_dir) / f'{safe_name}_{CODE_VERSION}.pt'
    if not cfg.force_recompute:
        cached = load_reusable_prediction_cache(
            Path(cfg.cache_dir), safe_name, bundle.base.shape, bundle,
            legacy_cache_dirs=cfg.legacy_cache_dirs)
        if cached is not None:
            return cached

    model = model.to(device)
    legacy_ckpts = [ckpt] + sorted(Path(cfg.ckpt_dir).glob(f'{safe_name}_*.pt'), reverse=True)
    resumed_from_ckpt = False
    if not cfg.force_recompute:
        for cand in legacy_ckpts:
            if not cand.exists():
                continue
            try:
                state = torch.load(cand, map_location='cpu')
                model.load_state_dict(state, strict=False)
                resumed_from_ckpt = True
                break
            except Exception:
                continue
    train_ds = SequenceSampleDataset(bundle.daily_dyn_scaled, bundle.day_features_scaled,
                                     bundle.residual, bundle.obs_mask, bundle.train_node_mask, cfg.seq_len)
    val_ds = SequenceSampleDataset(bundle.daily_dyn_scaled, bundle.day_features_scaled,
                                   bundle.residual, bundle.obs_mask, bundle.val_node_mask, cfg.seq_len)
    all_ds = SequenceSampleDataset(bundle.daily_dyn_scaled, bundle.day_features_scaled,
                                   bundle.residual, bundle.obs_mask, np.ones(bundle.n_nodes, dtype=bool), cfg.seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.seq_batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.seq_batch_size, shuffle=False, num_workers=cfg.num_workers)
    all_loader = DataLoader(all_ds, batch_size=cfg.seq_batch_size, shuffle=False, num_workers=cfg.num_workers)

    if resumed_from_ckpt and not pred_cache.exists():
        pred_matrix = np.full_like(bundle.base, np.nan, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for batch in all_loader:
                pred_res = model(batch['seq'].to(device), batch['cur'].to(device)).cpu().numpy().astype(np.float32)
                days, nodes = batch['day'].numpy().astype(int), batch['node'].numpy().astype(int)
                pred_matrix[days, nodes] = bundle.base[days, nodes] + pred_res
        np.save(pred_cache, pred_matrix)
        return pred_matrix

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state, best_val, bad_epochs = None, float('inf'), 0

    for _epoch in range(1, cfg.seq_max_epochs + 1):
        model.train()
        for batch in train_loader:
            seq, cur, y = batch['seq'].to(device), batch['cur'].to(device), batch['y'].to(device)
            optim.zero_grad()
            pred = model(seq, cur)
            loss = F.huber_loss(pred, y, delta=cfg.huber_delta)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['seq'].to(device), batch['cur'].to(device)).cpu().numpy()
                val_pred.append(pred)
                val_true.append(batch['y'].numpy())
        if len(val_true) == 0:
            continue
        yt, yp = np.concatenate(val_true), np.concatenate(val_pred)
        val_rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        if np.isfinite(val_rmse) and val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt)

    pred_matrix = np.full_like(bundle.base, np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in all_loader:
            pred_res = model(batch['seq'].to(device), batch['cur'].to(device)).cpu().numpy().astype(np.float32)
            days, nodes = batch['day'].numpy().astype(int), batch['node'].numpy().astype(int)
            pred_matrix[days, nodes] = bundle.base[days, nodes] + pred_res
    np.save(pred_cache, pred_matrix)
    return pred_matrix

def run_external_baseline_suite(bundle: DatasetBundle, cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    enabled_external = get_enabled_external_baselines(cfg)
    if not enabled_external:
        return preds

    spatial_requested = [m for m in enabled_external if m in ('IDW', 'OK', 'RK', 'RF', 'GBRT')]
    if spatial_requested:
        preds.update(run_spatial_baselines(bundle, cfg, spatial_requested))
    if 'LSTM' in enabled_external:
        preds['LSTM'] = train_sequence_model(
            bundle, cfg, 'LSTM',
            LSTMResidualModel(
                seq_in=bundle.daily_dyn_scaled.shape[-1],
                cur_in=bundle.day_features_scaled.shape[-1],
                hidden=cfg.hidden_dim,
            ),
        )
    if 'TCN' in enabled_external:
        preds['TCN'] = train_sequence_model(
            bundle, cfg, 'TCN',
            TCNResidualModel(
                seq_in=bundle.daily_dyn_scaled.shape[-1],
                cur_in=bundle.day_features_scaled.shape[-1],
                hidden=cfg.hidden_dim,
            ),
        )
    return preds

# =============================================================================
# Phase C implementation
# =============================================================================


CODE_VERSION_ANP = 'cnp_gpd_tail_no_coordinate_features_5fold_3yr_era5land'

# =============================================================================
# Config
# =============================================================================
@dataclass
class ANPConfig:
    """Hyperparameters for direct daily-maximum wind speed with a GPD tail."""
    work_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/'
    cache_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/cache/'
    ckpt_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/ckpt/'
    pred_dir: str = 'res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land/preds/'
    # Architecture
    hidden_dim: int = 192
    ctx_enc_layers: int = 4
    dec_layers: int = 4
    n_heads: int = 6
    n_ctx_self_layers: int = 3
    use_dir_bias: bool = True
    dir_bias_hidden: int = 48
    dropout: float = 0.12
    sigma_min: float = 0.05
    sigma_max: float = 6.0
    # Bulk-and-tail distribution. tail_threshold is overwritten in each fold
    # using training-station observations only.
    tail_quantile: float = 0.95
    tail_threshold: float = 0.0
    exceedance_loss_weight: float = 0.25
    gpd_tail_loss_weight: float = 0.25
    deterministic_loss_weight: float = 0.25
    tail_loss_warmup_epochs: int = 5
    early_stop_tail_weight: float = 0.25
    gpd_scale_min: float = 0.05
    gpd_scale_max: float = 20.0
    gpd_scale_init: float = 2.0
    gpd_shape_limit: float = 0.20
    gpd_shape_init: float = 0.0
    gpd_support_penalty_weight: float = 10.0
    # Selected final setting: no Gaussian soft-IDW prior
    # In this selected model, Gaussian soft-IDW is not computed, not fed to the
    # decoder, and not used as a warm-start or regularization teacher.
    # Daily maximum wind speed is predicted directly from target/context
    # exogenous features, attention, temporal/climate features, and optional
    # station embeddings.
    use_anchor: bool = False
    anchor_mode: str = 'no_anchor_prior'   # {'no_anchor_prior', 'nn_init_only', 'legacy_additive'}
    anchor_k: int = 0
    anchor_sigma_km: float = 150.0
    anchor_warm_start_weight: float = 0.0
    anchor_warm_start_epochs: int = 0
    # Local cross-attention
    local_attn_k: int = 32
    # Climate feature projector
    use_climate: bool = True
    climate_proj_dim: int = 32
    # Phase-F2 causal short-term temporal features from exogenous dynamic fields only
    use_temporal_features: bool = True
    temporal_roll_windows: Tuple[int, ...] = (3, 7)
    temporal_include_lag1: bool = True
    temporal_include_delta1: bool = True
    temporal_include_anomaly_to_roll: bool = True
    # Phase-C conservative correction and spatial residual regularization
    # G3: gate released. correction_max_scale 0.75 -> 1.5, gate_init -2.5 -> 0.0
    # (sigmoid(0)=0.5, so initial correction strength is 50% instead of 6%),
    # correction_reg_weight 0.002 -> 0.0005 (4x weaker penalty).
    correction_max_scale: float = 1.5
    correction_gate_init: float = 0.0
    correction_reg_weight: float = 0.0
    # G5: Moran decorrelation penalty removed (positive residual
    # autocorrelation is the Kriging signal, not noise to suppress).
    moran_reg_weight: float = 0.00
    moran_sigma_km: float = 220.0
    moran_max_points: int = 96
    # G2: per-station latent embedding, fed only into the decoder.
    # Train stations get gradients; val/test stations are baked at inference
    # by kNN-averaging neighboring train embeddings (see SpatialANP.bake_unseen).
    use_station_embedding: bool = True
    station_emb_dim: int = 16
    station_emb_init_std: float = 0.01
    station_emb_neighbor_k: int = 8
    # This experimental variant forbids observations and observation-derived
    # residuals in model inputs. The direct supervised target is daily maximum
    # wind speed.
    use_residual_memory: bool = False
    # Direct wind speed is trained with a censored Gaussian likelihood at the
    # physical lower bound of zero metres per second.
    train_in_obs_space: bool = True
    obs_floor: float = 0.0
    # Training
    batch_days: int = 6
    inference_batch_days: int = 8
    val_eval_max_days: int = 1024
    val_eval_batch_days: int = 64
    grid_chunk_size: int = 4096
    max_epochs: int = 80
    patience: int = 14
    lr: float = 6.0e-4
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 5
    grad_clip: float = 3.0
    # Context/target sampling
    drop_rate_min: float = 0.10
    drop_rate_max: float = 0.60
    min_context_nodes: int = 6
    use_spatial_block_mask: bool = True
    spatial_block_prob: float = 0.80
    block_radius_km_min: float = 60.0
    block_radius_km_max: float = 180.0
    min_block_target_nodes: int = 8
    seed: int = 42
    force_recompute: bool = False
    # Final selected setting / reproducibility
    no_bake_unseen_embeddings: bool = True
    load_existing_checkpoint: bool = False
    legacy_checkpoint_paths: Tuple[str, ...] = ()
    # Evaluation
    report_crps: bool = False
    report_csi: bool = False
    report_moran: bool = False

    def prepare_dirs(self) -> None:
        for p in [self.work_dir, self.cache_dir, self.ckpt_dir, self.pred_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)


class DirectionalBias(nn.Module):
    """Map relative geometry (Δlat, Δlon, dist, cosθ, sinθ) to per-head logits bias."""
    def __init__(self, n_heads: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads),
        )

    def forward(self, tgt_latlon: torch.Tensor, ctx_latlon: torch.Tensor,
                relative_geometry: Optional[torch.Tensor] = None) -> torch.Tensor:
        if relative_geometry is None:
            diff = tgt_latlon.unsqueeze(2) - ctx_latlon.unsqueeze(1)
            dlat = diff[..., 0]
            dlon = diff[..., 1]
            dist = torch.sqrt(dlat * dlat + dlon * dlon + 1e-8)
            theta = torch.atan2(dlat, dlon + 1e-8)
            feats = torch.stack([dlat, dlon, dist, torch.cos(theta), torch.sin(theta)], dim=-1)
        else:
            feats = relative_geometry
        bias = self.mlp(feats)                                     # [B, Nt, Nc, H]
        return bias.permute(0, 3, 1, 2).contiguous()              # [B, H, Nt, Nc]


class MaskedMHA(nn.Module):
    """Multi-head attention with additive bias and safe key padding mask handling."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by n_heads={n_heads}.')
        self.dim = int(dim)
        self.nh = int(n_heads)
        self.dh = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.dh)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        Q = self.q_proj(q).view(B, Nq, self.nh, self.dh).transpose(1, 2)
        K = self.k_proj(k).view(B, Nk, self.nh, self.dh).transpose(1, 2)
        V = self.v_proj(v).view(B, Nk, self.nh, self.dh).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            logits = logits + attn_bias

        pad = None
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :].to(torch.bool)
            logits = logits.masked_fill(pad, -1.0e9)

        attn = torch.softmax(logits, dim=-1)

        if pad is not None:
            attn = attn.masked_fill(pad, 0.0)
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            attn = attn / denom

        attn = self.drop(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.dim)
        return self.o_proj(out), attn


# =============================================================================
# SpatialANP model
# =============================================================================
class SpatialANP(nn.Module):
    """Attentive Neural Process for the daily maximum wind-speed field.

    In anchor_mode='no_anchor_prior', Gaussian soft-IDW is removed completely:
    it is not computed, not fed to the decoder, and not used by the loss.
    The final wind-speed mean is the neural-network head output.
    """
    def __init__(self, day_feat_dim: int, climate_feat_dim: int, cfg: ANPConfig,
                 n_nodes: int = 0, latlon_np: Optional[np.ndarray] = None,
                 train_node_mask_np: Optional[np.ndarray] = None,
                 pairwise_km_np: Optional[np.ndarray] = None,
                 relative_geometry_np: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        self.day_feat_dim = int(day_feat_dim)
        self.climate_feat_dim = int(climate_feat_dim)
        self.hidden = int(cfg.hidden_dim)
        self.sigma_min = float(cfg.sigma_min)
        self.sigma_max = float(cfg.sigma_max)
        if not np.isfinite(float(cfg.tail_threshold)) or float(cfg.tail_threshold) <= 0.0:
            raise ValueError('tail_threshold must be a positive training-only wind-speed threshold.')
        self.register_buffer(
            'tail_threshold', torch.tensor(float(cfg.tail_threshold), dtype=torch.float32)
        )
        self.gpd_scale_min = float(cfg.gpd_scale_min)
        self.gpd_scale_max = float(cfg.gpd_scale_max)
        self.gpd_shape_limit = float(cfg.gpd_shape_limit)
        if not (0.0 < self.gpd_shape_limit < 1.0):
            raise ValueError('gpd_shape_limit must be between 0 and 1 so the GPD mean exists.')
        self.anchor_mode = str(getattr(cfg, 'anchor_mode', 'legacy_additive'))
        self.use_anchor_prior = bool(getattr(cfg, 'use_anchor', True)) and self.anchor_mode != 'no_anchor_prior'
        if self.use_anchor_prior:
            raise ValueError('本实验禁止使用由观测残差构造的 soft-IDW anchor。')

        self.model_feat_dim = self.day_feat_dim
        if self.climate_feat_dim > 0 and bool(cfg.use_climate):
            self.climate_proj = nn.Sequential(
                nn.Linear(self.climate_feat_dim, self.hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(self.hidden, cfg.climate_proj_dim),
                nn.LayerNorm(cfg.climate_proj_dim),
            )
            self.model_feat_dim += int(cfg.climate_proj_dim)
        else:
            self.climate_proj = None
        enc_in = self.model_feat_dim

        if pairwise_km_np is not None and relative_geometry_np is not None:
            pairwise_km_np = np.asarray(pairwise_km_np, dtype=np.float32)
            relative_geometry_np = np.asarray(relative_geometry_np, dtype=np.float32)
            if pairwise_km_np.shape != (n_nodes, n_nodes):
                raise ValueError('pairwise_km_np shape does not match n_nodes.')
            if relative_geometry_np.shape != (n_nodes, n_nodes, 5):
                raise ValueError('relative_geometry_np shape must be [N, N, 5].')
            self.register_buffer(
                '_pairwise_km_cache', torch.from_numpy(pairwise_km_np), persistent=False
            )
            self.register_buffer(
                '_relative_geometry_cache', torch.from_numpy(relative_geometry_np), persistent=False
            )
            self.has_station_geometry_cache = True
        else:
            self._pairwise_km_cache = None
            self._relative_geometry_cache = None
            self.has_station_geometry_cache = False

        # G2: per-station embedding (decoder-only, doesn't pollute IDW geometry).
        self.use_station_emb = bool(cfg.use_station_embedding) and n_nodes > 0
        if self.use_station_emb:
            self.station_emb_dim = int(cfg.station_emb_dim)
            self.station_emb = nn.Embedding(int(n_nodes), self.station_emb_dim)
            nn.init.normal_(self.station_emb.weight, mean=0.0, std=float(cfg.station_emb_init_std))
            self.station_emb_ln = nn.LayerNorm(self.station_emb_dim)
            # Cache geographic kNN for unseen-station baking (computed once on CPU).
            if (latlon_np is not None and train_node_mask_np is not None
                    and not bool(getattr(cfg, 'no_bake_unseen_embeddings', True))):
                self._cache_unseen_bake_index(latlon_np, train_node_mask_np,
                                              k=int(cfg.station_emb_neighbor_k))
            else:
                self._unseen_idx = None
                self._unseen_train_neighbors = None
        else:
            self.station_emb_dim = 0
            self.station_emb = None
            self.station_emb_ln = None
            self._unseen_idx = None
            self._unseen_train_neighbors = None

        anchor_feat_dim = 2 if self.use_anchor_prior else 0
        dec_in = self.model_feat_dim + self.hidden + anchor_feat_dim + self.station_emb_dim

        self.ctx_encoder = self._make_mlp(enc_in, self.hidden, cfg.ctx_enc_layers, cfg.dropout)
        self.ctx_self_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MaskedMHA(self.hidden, cfg.n_heads, cfg.dropout),
                'ln1': nn.LayerNorm(self.hidden),
                'ff': nn.Sequential(
                    nn.Linear(self.hidden, 4 * self.hidden),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(4 * self.hidden, self.hidden),
                ),
                'ln2': nn.LayerNorm(self.hidden),
            })
            for _ in range(cfg.n_ctx_self_layers)
        ])
        self.tgt_q_proj = self._make_mlp(self.model_feat_dim, self.hidden, 2, cfg.dropout)
        self.cross_attn = MaskedMHA(self.hidden, cfg.n_heads, cfg.dropout)
        self.cross_ln1 = nn.LayerNorm(self.hidden)
        self.cross_ff = nn.Sequential(
            nn.Linear(self.hidden, 4 * self.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * self.hidden, self.hidden),
        )
        self.cross_ln2 = nn.LayerNorm(self.hidden)
        self.use_dir_bias = bool(cfg.use_dir_bias)
        if self.use_dir_bias:
            self.dir_bias = DirectionalBias(cfg.n_heads, cfg.dir_bias_hidden)

        self.decoder_trunk = self._make_mlp(dec_in, self.hidden, cfg.dec_layers, cfg.dropout)
        self.mu_head = nn.Linear(self.hidden, 1)
        self.correction_gate_head = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.correction_gate_head.weight)
        nn.init.constant_(self.correction_gate_head.bias, float(cfg.correction_gate_init))
        self.log_sigma_head = nn.Linear(self.hidden, 1)
        self.exceedance_logit_head = nn.Linear(self.hidden, 1)
        self.gpd_log_scale_head = nn.Linear(self.hidden, 1)

        event_rate = float(np.clip(1.0 - cfg.tail_quantile, 1.0e-4, 1.0 - 1.0e-4))
        nn.init.zeros_(self.exceedance_logit_head.weight)
        nn.init.constant_(self.exceedance_logit_head.bias, math.log(event_rate / (1.0 - event_rate)))
        scale_offset = max(float(cfg.gpd_scale_init) - self.gpd_scale_min, 1.0e-4)
        nn.init.zeros_(self.gpd_log_scale_head.weight)
        nn.init.constant_(self.gpd_log_scale_head.bias, math.log(math.expm1(scale_offset)))
        shape_ratio = float(np.clip(cfg.gpd_shape_init / self.gpd_shape_limit, -0.999, 0.999))
        self.gpd_raw_shape = nn.Parameter(torch.tensor(float(np.arctanh(shape_ratio))))


    # ----- G2 helpers: per-station embedding bake for unseen stations -----
    def _cache_unseen_bake_index(self, latlon_np: np.ndarray, train_mask_np: np.ndarray, k: int = 8):
        """Precompute, for every unseen (val/test) station, the indices of
        the geographically k-nearest TRAIN stations and their inverse-distance
        weights. Stored as long/float buffers so bake() runs without numpy.
        """
        train_idx = np.where(train_mask_np.astype(bool))[0]
        if train_idx.size == 0:
            self._unseen_train_neighbors = None
            return
        coords_rad = np.radians(latlon_np.astype(np.float64))
        tree = BallTree(coords_rad[train_idx], metric='haversine')
        kk = int(min(k, train_idx.size))
        unseen_mask = ~train_mask_np.astype(bool)
        unseen_idx = np.where(unseen_mask)[0]
        if unseen_idx.size == 0:
            self._unseen_train_neighbors = None
            return
        d_rad, nbr_local = tree.query(coords_rad[unseen_idx], k=kk)
        nbr_global = train_idx[nbr_local]                                # [n_unseen, kk]
        d_km = d_rad * EARTH_R_KM
        w = 1.0 / np.clip(d_km, 1.0, None)
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1.0e-12, None)     # row-normalized
        self.register_buffer('_unseen_idx',
                             torch.from_numpy(unseen_idx.astype(np.int64)),
                             persistent=False)
        self.register_buffer('_unseen_neighbor_idx',
                             torch.from_numpy(nbr_global.astype(np.int64)),
                             persistent=False)
        self.register_buffer('_unseen_neighbor_w',
                             torch.from_numpy(w.astype(np.float32)),
                             persistent=False)
        self._unseen_train_neighbors = True

    @torch.no_grad()
    def bake_unseen_embeddings(self) -> None:
        """Replace val/test station embedding rows with the kNN-weighted
        average of their geographically nearest train embeddings. Run this
        once after training finishes, before inference. Idempotent.
        """
        if not self.use_station_emb or self._unseen_train_neighbors is None:
            return
        unseen_idx = self._unseen_idx                              # [U]
        nbr_idx = self._unseen_neighbor_idx                        # [U, k]
        w = self._unseen_neighbor_w                                # [U, k]
        if unseen_idx.numel() == 0:
            return
        nbr_vec = self.station_emb.weight[nbr_idx]                 # [U, k, D]
        baked = (nbr_vec * w.unsqueeze(-1)).sum(dim=1)             # [U, D]
        self.station_emb.weight.data[unseen_idx] = baked

    @staticmethod
    def _make_mlp(in_dim: int, hidden: int, n_layers: int, dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.LayerNorm(hidden))
        return nn.Sequential(*layers)

    def _encode_features(self, feat: torch.Tensor) -> torch.Tensor:
        if self.climate_proj is None or self.climate_feat_dim <= 0:
            return feat[..., :self.day_feat_dim]
        day_feat = feat[..., :self.day_feat_dim]
        climate_feat = feat[..., self.day_feat_dim:self.day_feat_dim + self.climate_feat_dim]
        climate_proj = self.climate_proj(climate_feat)
        return torch.cat([day_feat, climate_proj], dim=-1)

    def _build_local_attn_bias(self, tgt_latlon: torch.Tensor, ctx_latlon: torch.Tensor,
                               ctx_valid: torch.Tensor,
                               d_km: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        k = int(getattr(self.cfg, 'local_attn_k', 0))
        Nc = ctx_latlon.shape[1]
        if k <= 0 or Nc <= k:
            return None
        if d_km is None:
            d_km = self._pairwise_haversine_km(tgt_latlon, ctx_latlon)
        valid3 = ctx_valid.unsqueeze(1)
        d_masked = torch.where(valid3, d_km, torch.full_like(d_km, 1.0e9))
        topk_idx = torch.topk(d_masked, k=k, dim=-1, largest=False).indices
        keep = torch.zeros_like(d_km, dtype=torch.bool)
        keep.scatter_(-1, topk_idx, True)
        keep = keep & valid3
        local_bias = torch.where(keep, torch.zeros_like(d_km), torch.full_like(d_km, -1.0e9))
        return local_bias.unsqueeze(1)

    @staticmethod
    def _pairwise_haversine_km(tgt_latlon: torch.Tensor, ctx_latlon: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Haversine distance [B, Nt, Nc] from raw degree lat/lon."""
        lat1 = torch.deg2rad(tgt_latlon[..., 0]).unsqueeze(-1)
        lon1 = torch.deg2rad(tgt_latlon[..., 1]).unsqueeze(-1)
        lat2 = torch.deg2rad(ctx_latlon[..., 0]).unsqueeze(1)
        lon2 = torch.deg2rad(ctx_latlon[..., 1]).unsqueeze(1)
        dlat = lat1 - lat2
        dlon = lon1 - lon2
        a = torch.sin(dlat * 0.5).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon * 0.5).pow(2)
        c = 2.0 * torch.arcsin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
        return c * EARTH_R_KM

    def _cached_station_geometry(self,
                                 tgt_node_idx: Optional[torch.Tensor],
                                 ctx_node_idx: Optional[torch.Tensor]
                                 ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Gather fixed station geometry for one padded batch."""
        if (not self.has_station_geometry_cache
                or tgt_node_idx is None or ctx_node_idx is None):
            return None, None
        tgt_safe = tgt_node_idx.clamp(min=0)
        ctx_safe = ctx_node_idx.clamp(min=0)
        tgt_expand = tgt_safe[:, :, None]
        ctx_expand = ctx_safe[:, None, :]
        d_km = self._pairwise_km_cache[tgt_expand, ctx_expand]
        relative_geometry = self._relative_geometry_cache[tgt_expand, ctx_expand]
        return d_km, relative_geometry

    def encode_context(self, ctx_feat: torch.Tensor, ctx_mask: torch.Tensor,
                       encoded_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode reference stations once per day for chunked grid inference."""
        ctx_feat_enc = self._encode_features(ctx_feat) if encoded_features is None else encoded_features
        r_ctx = self.ctx_encoder(ctx_feat_enc)

        ctx_valid = (ctx_mask > 0.5)
        ctx_pad = ~ctx_valid
        valid_f = ctx_valid.unsqueeze(-1).float()
        r_ctx = r_ctx * valid_f

        for layer in self.ctx_self_layers:
            h, _ = layer['attn'](r_ctx, r_ctx, r_ctx, key_padding_mask=ctx_pad)
            r_ctx = layer['ln1'](r_ctx + h)
            r_ctx = layer['ln2'](r_ctx + layer['ff'](r_ctx))
            r_ctx = r_ctx * valid_f

        return r_ctx

    def forward(self, ctx_latlon: torch.Tensor, ctx_feat: torch.Tensor,
                ctx_mask: torch.Tensor,
                tgt_latlon: torch.Tensor, tgt_feat: torch.Tensor,
                tgt_node_idx: Optional[torch.Tensor] = None,
                ctx_node_idx: Optional[torch.Tensor] = None,
                return_aux: bool = False,
                encoded_context: Optional[torch.Tensor] = None
                ):
        """
        ctx_mask uses the Phase-A convention:
          1.0 -> valid context point
          0.0 -> padding / missing

        G2: tgt_node_idx [B, Nt] long is required when use_station_emb=True.
        Padded target slots can use any value within [0, N); they are zeroed
        out by station_emb_ln before contributing (no, padded slots simply
        produce a vector that gets discarded by the target mask in the loss).
        """
        ctx_feat_enc = self._encode_features(ctx_feat) if encoded_context is None else None
        tgt_feat_enc = self._encode_features(tgt_feat)
        ctx_valid = (ctx_mask > 0.5)
        ctx_pad = ~ctx_valid
        r_ctx = (self.encode_context(ctx_feat, ctx_mask, ctx_feat_enc)
                 if encoded_context is None else encoded_context)

        q_tgt = self.tgt_q_proj(tgt_feat_enc)
        cached_km, cached_geometry = self._cached_station_geometry(tgt_node_idx, ctx_node_idx)
        attn_bias = (
            self.dir_bias(tgt_latlon, ctx_latlon, cached_geometry)
            if self.use_dir_bias else None
        )
        local_bias = self._build_local_attn_bias(
            tgt_latlon, ctx_latlon, ctx_valid, d_km=cached_km
        )
        if local_bias is not None:
            attn_bias = local_bias if attn_bias is None else (attn_bias + local_bias)
        r_tgt, _ = self.cross_attn(q_tgt, r_ctx, r_ctx, key_padding_mask=ctx_pad, attn_bias=attn_bias)
        r_tgt = self.cross_ln1(q_tgt + r_tgt)
        r_tgt = self.cross_ln2(r_tgt + self.cross_ff(r_tgt))

        # This experiment has no observed-residual anchor or residual-derived input.
        B0, Nt0 = tgt_latlon.shape[0], tgt_latlon.shape[1]
        anchor_mean = tgt_latlon.new_zeros((B0, Nt0))
        anchor_log_mass = tgt_latlon.new_zeros((B0, Nt0))

        dec_parts = [tgt_feat_enc, r_tgt]
        if self.use_anchor_prior:
            dec_parts.extend([
                anchor_mean.unsqueeze(-1),
                anchor_log_mass.unsqueeze(-1),
            ])
        if self.use_station_emb:
            if tgt_node_idx is None:
                # Fallback: zero embedding (keeps shape consistent if caller
                # forgot to wire indices; degrades gracefully).
                B, Nt = tgt_latlon.shape[:2]
                emb = torch.zeros(B, Nt, self.station_emb_dim,
                                  device=tgt_latlon.device, dtype=tgt_latlon.dtype)
            else:
                emb = self.station_emb(tgt_node_idx)
                emb = self.station_emb_ln(emb)
            dec_parts.append(emb)

        dec_in = torch.cat(dec_parts, dim=-1)
        h = self.decoder_trunk(dec_in)

        # The bulk mean is constrained to the physically meaningful interval
        # from zero to the fold-specific tail threshold.
        raw_nn_mu = self.mu_head(h).squeeze(-1)
        bulk_mean = self.tail_threshold * torch.sigmoid(raw_nn_mu)
        correction_gate = torch.sigmoid(self.correction_gate_head(h).squeeze(-1))
        gated_delta_for_legacy = raw_nn_mu.new_zeros(raw_nn_mu.shape)
        anchor_mode = str(getattr(self.cfg, 'anchor_mode', 'no_anchor_prior'))
        if anchor_mode != 'no_anchor_prior':
            raise ValueError('The GPD-tail experiment requires anchor_mode=no_anchor_prior.')
        nn_minus_anchor = raw_nn_mu.new_zeros(raw_nn_mu.shape)

        exceedance_logit = self.exceedance_logit_head(h).squeeze(-1)
        exceedance_probability = torch.sigmoid(exceedance_logit)
        gpd_scale = F.softplus(self.gpd_log_scale_head(h).squeeze(-1)) + self.gpd_scale_min
        gpd_scale = gpd_scale.clamp(self.gpd_scale_min, self.gpd_scale_max)
        gpd_shape = self.gpd_shape_limit * torch.tanh(self.gpd_raw_shape)
        tail_conditional_mean = self.tail_threshold + gpd_scale / (1.0 - gpd_shape)

        # This is the expected daily maximum wind speed under the two-part
        # model and is the deterministic product evaluated by RMSE and bias.
        mu = ((1.0 - exceedance_probability) * bulk_mean
              + exceedance_probability * tail_conditional_mean)

        sigma = F.softplus(self.log_sigma_head(h).squeeze(-1)) + self.sigma_min
        sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        if return_aux:
            aux = {
                'anchor_mean': anchor_mean,
                'anchor_log_mass': anchor_log_mass,
                'raw_nn_mu': raw_nn_mu,
                'bulk_mean': bulk_mean,
                'exceedance_logit': exceedance_logit,
                'exceedance_probability': exceedance_probability,
                'gpd_scale': gpd_scale,
                'gpd_shape': gpd_shape,
                'tail_conditional_mean': tail_conditional_mean,
                # Backward-compatible diagnostic names:
                'raw_delta_mu': raw_nn_mu,
                'correction_gate': correction_gate,
                # In no_anchor_prior mode this is zero; in nn_init_only mode this is the neural output deviation from
                # the soft-IDW reference, not the final additive correction.
                'scaled_correction': nn_minus_anchor,
                'gated_delta_for_legacy': gated_delta_for_legacy,
                'anchor_mode': anchor_mode,
            }
            return mu, sigma, aux
        return mu, sigma

# =============================================================================
# Per-point feature construction
# =============================================================================
def build_causal_temporal_features(bundle: DatasetBundle, cfg: ANPConfig) -> np.ndarray:
    """Build causal short-term temporal features from exogenous dynamic variables.

    Uses only dynamic fields available for every node/day (ERA5Land_WS,
    ERA5Land_TEM, ERA5Land_RH, ERA5Land_SP after train-node scaling). It never uses
    target station observations/residuals, so Val/Test station labels are not leaked.

    Returned features are [D, N, F_temp]. For day 0, lag features fall back to
    day-0 values; rolling means are causal and include the current day.
    """
    dyn = np.asarray(bundle.daily_dyn_scaled, np.float32)
    dyn = np.nan_to_num(dyn, nan=0.0, posinf=0.0, neginf=0.0)
    feats: List[np.ndarray] = []

    if bool(getattr(cfg, 'temporal_include_lag1', True)):
        lag1 = np.concatenate([dyn[:1], dyn[:-1]], axis=0)
        feats.append(lag1)

    csum = np.cumsum(dyn, axis=0, dtype=np.float64)
    roll_means: Dict[int, np.ndarray] = {}
    for w in tuple(getattr(cfg, 'temporal_roll_windows', (3, 7))):
        w = int(w)
        if w <= 1:
            continue
        days = np.arange(dyn.shape[0])
        starts = np.maximum(0, days - w + 1)
        total = csum.copy()
        valid = starts > 0
        total[valid] -= csum[starts[valid] - 1]
        out = (total / (days - starts + 1)[:, None, None]).astype(np.float32)
        roll_means[w] = out
        feats.append(out)

    if bool(getattr(cfg, 'temporal_include_delta1', True)):
        lag1 = np.concatenate([dyn[:1], dyn[:-1]], axis=0)
        feats.append((dyn - lag1).astype(np.float32))

    if bool(getattr(cfg, 'temporal_include_anomaly_to_roll', True)):
        for _, rm in roll_means.items():
            feats.append((dyn - rm).astype(np.float32))

    if not feats:
        return np.zeros((dyn.shape[0], dyn.shape[1], 0), dtype=np.float32)
    out = np.concatenate(feats, axis=-1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_point_feature_bank(bundle: DatasetBundle, cfg: ANPConfig) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return per-day per-node features and split dims.

    day_feat_dim includes original day_features_scaled plus optional causal
    temporal features derived only from ERA5-Land fields.
    climate_feat_dim is appended after day_feat_dim and is projected inside the model.
    """
    if bool(getattr(cfg, 'use_residual_memory', False)):
        raise ValueError('本实验禁止将观测残差或历史残差统计量作为模型输入。')
    latlon = bundle.node_df[['Lat', 'Lon']].to_numpy(dtype=np.float32)
    base_day_feat = np.asarray(bundle.day_features_scaled, dtype=np.float32)
    day_parts = [base_day_feat]
    temporal_dim = 0
    if bool(getattr(cfg, 'use_temporal_features', False)):
        temp = build_causal_temporal_features(bundle, cfg)
        if temp.shape[-1] > 0:
            day_parts.append(temp)
            temporal_dim = int(temp.shape[-1])

    day_feat = np.concatenate(day_parts, axis=-1).astype(np.float32)

    climate_feat_dim = 0
    if bool(cfg.use_climate) and getattr(bundle, 'climate_vectors', None) is not None:
        climate = np.asarray(bundle.climate_vectors, np.float32)
        climate = np.nan_to_num(climate, nan=0.0, posinf=0.0, neginf=0.0)
        feat_bank = np.concatenate([day_feat, climate], axis=-1).astype(np.float32)
        climate_feat_dim = int(climate.shape[-1])
    else:
        feat_bank = day_feat
    if temporal_dim > 0:
        print(f'[F2] causal temporal feature dim = {temporal_dim}; '
              f'day_feat_dim {base_day_feat.shape[-1]} -> {day_feat.shape[-1]}')
    assert not {'Lat', 'Lon'} & set(bundle.feature_names), 'Coordinates must not enter feature tensors.'
    assert not {'Lat', 'Lon'} & set(bundle.const_feature_names), 'Coordinates must not enter climate features.'
    return feat_bank, latlon, int(day_feat.shape[-1]), climate_feat_dim


def compute_pairwise_haversine_km(latlon: np.ndarray) -> np.ndarray:
    """Full pairwise Haversine distance matrix for node coordinates in degrees."""
    lat = np.radians(latlon[:, 0].astype(np.float64))
    lon = np.radians(latlon[:, 1].astype(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon * 0.5) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (c * EARTH_R_KM).astype(np.float32)


def load_or_build_station_geometry(latlon: np.ndarray,
                                   cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load or compute fixed station distances and directional geometry once.

    The first array stores Haversine distance in kilometres. The second stores,
    for every target-station and context-station pair, relative latitude,
    relative longitude, planar coordinate difference, cosine of direction, and
    sine of direction. These quantities depend only on station coordinates and
    are shared by all folds, epochs, and batches.
    """
    latlon = np.asarray(latlon, dtype=np.float32)
    if latlon.ndim != 2 or latlon.shape[1] != 2 or not np.isfinite(latlon).all():
        raise ValueError('station geometry requires a finite [N, 2] latitude/longitude array.')

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / 'station_geometry_no_residual_inputs.npz'
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                saved_latlon = saved['latlon'].astype(np.float32)
                pairwise_km = saved['pairwise_km'].astype(np.float32)
                relative_geometry = saved['relative_geometry'].astype(np.float32)
            expected_n = latlon.shape[0]
            valid = (
                saved_latlon.shape == latlon.shape
                and np.allclose(saved_latlon, latlon, rtol=0.0, atol=1.0e-6)
                and pairwise_km.shape == (expected_n, expected_n)
                and relative_geometry.shape == (expected_n, expected_n, 5)
                and np.isfinite(pairwise_km).all()
                and np.isfinite(relative_geometry).all()
            )
            if valid:
                print(f'[geometry cache] reused: {cache_path}')
                return pairwise_km, relative_geometry
            print(f'[geometry cache] stale shape or coordinates; rebuilding: {cache_path}')
        except Exception as exc:
            print(f'[geometry cache] failed to load; rebuilding: {exc}')

    pairwise_km = compute_pairwise_haversine_km(latlon)
    dlat = latlon[:, None, 0] - latlon[None, :, 0]
    dlon = latlon[:, None, 1] - latlon[None, :, 1]
    coord_dist = np.sqrt(dlat * dlat + dlon * dlon + 1.0e-8)
    theta = np.arctan2(dlat, dlon + 1.0e-8)
    relative_geometry = np.stack(
        [dlat, dlon, coord_dist, np.cos(theta), np.sin(theta)], axis=-1
    ).astype(np.float32)
    np.savez(
        cache_path,
        latlon=latlon,
        pairwise_km=pairwise_km,
        relative_geometry=relative_geometry,
    )
    print(f'[geometry cache] built once for {latlon.shape[0]} stations: {cache_path}')
    return pairwise_km, relative_geometry


def select_spatial_block_target(cand: np.ndarray, pairwise_km: np.ndarray,
                                rng: np.random.Generator, cfg: ANPConfig) -> np.ndarray:
    """Choose a spatially coherent target subset from observed train nodes."""
    if cand.size == 0:
        return cand
    center = int(cand[rng.integers(0, cand.size)])
    radius = float(rng.uniform(cfg.block_radius_km_min, cfg.block_radius_km_max))
    dist = pairwise_km[center, cand]
    tgt_idx = cand[dist <= radius]

    min_tgt = max(1, int(cfg.min_block_target_nodes))
    max_tgt = max(1, cand.size - cfg.min_context_nodes)

    if tgt_idx.size < min_tgt:
        target_n = max(min_tgt, int(round(cand.size * float(rng.uniform(cfg.drop_rate_min, cfg.drop_rate_max)))))
        target_n = min(target_n, max_tgt)
        order = np.argsort(dist)
        tgt_idx = cand[order[:target_n]]
    elif tgt_idx.size > max_tgt:
        order = np.argsort(dist)
        tgt_idx = cand[order[:max_tgt]]

    if tgt_idx.size == 0:
        tgt_idx = cand[:1]
    return np.unique(tgt_idx.astype(np.int64))


# =============================================================================
# Batch sampler: build padded (context, target) pairs per batch
# =============================================================================
def _pad_to(arr_list: List[np.ndarray], max_len: int, pad_val=0.0, dtype=None) -> np.ndarray:
    """Pad a list of 1- or 2-D arrays along axis=0 to length max_len.
    Returns [B, max_len, ...] stacked array.
    """
    B = len(arr_list)
    if max_len == 0:
        if arr_list and arr_list[0].ndim >= 2:
            rest = arr_list[0].shape[1:]
            out = np.zeros((B, 0) + rest, dtype=dtype or np.float32)
        else:
            out = np.zeros((B, 0), dtype=dtype or np.float32)
        return out
    sample = arr_list[0]
    if sample.ndim == 1:
        out = np.full((B, max_len), pad_val, dtype=dtype or sample.dtype)
        for i, a in enumerate(arr_list):
            L = len(a)
            if L > 0:
                out[i, :L] = a
    else:
        rest = sample.shape[1:]
        out = np.full((B, max_len) + rest, pad_val, dtype=dtype or sample.dtype)
        for i, a in enumerate(arr_list):
            L = a.shape[0]
            if L > 0:
                out[i, :L] = a
    return out


def sample_cnp_batch_train(bundle: DatasetBundle,
                           feat_bank: np.ndarray, latlon: np.ndarray,
                           day_indices: Sequence[int],
                           rng: np.random.Generator,
                           drop_rate_min: float, drop_rate_max: float,
                           min_context: int,
                           pairwise_km: Optional[np.ndarray] = None,
                           cfg: Optional[ANPConfig] = None,
                           include_features: bool = True,
                           ) -> Optional[Dict[str, np.ndarray]]:
    """Build a training batch with direct daily-maximum wind-speed targets.

    If cfg.use_spatial_block_mask is enabled, most days sample a spatially
    coherent target patch rather than iid random points. This better matches
    the held-out-station test setting.
    """
    obs_mask = bundle.obs_mask
    obs = bundle.obs
    train_mask = bundle.train_node_mask

    sampled_days = []
    ctx_pos_list, ctx_feat_list, ctx_idx_list = [], [], []
    tgt_pos_list, tgt_feat_list, tgt_y_list = [], [], []
    tgt_idx_list: List[np.ndarray] = []

    train_idx_all = np.where(train_mask)[0].astype(np.int64)
    for d in day_indices:
        d = int(d)
        cand = np.where(obs_mask[d] & train_mask)[0]
        if cand.size == 0 or train_idx_all.size < min_context + 1:
            continue

        use_block = bool(
            cfg is not None and cfg.use_spatial_block_mask and pairwise_km is not None
            and rng.random() < cfg.spatial_block_prob
        )

        if use_block:
            tgt_idx = select_spatial_block_target(cand, pairwise_km, rng, cfg)
            if tgt_idx.size == 0:
                use_block = False

        if not use_block:
            r = float(rng.uniform(drop_rate_min, drop_rate_max))
            n_tgt = max(1, int(round(cand.size * r)))
            n_tgt = min(n_tgt, cand.size)
            perm = rng.permutation(cand)
            tgt_idx = perm[:n_tgt]

        # Context contains exogenous features from all other training stations.
        # Observation values and observation-availability masks are not inputs.
        ctx_idx = np.setdiff1d(train_idx_all, tgt_idx, assume_unique=False)

        if ctx_idx.size < min_context or tgt_idx.size == 0:
            continue

        sampled_days.append(d)
        ctx_pos_list.append(latlon[ctx_idx])
        if include_features:
            ctx_feat_list.append(feat_bank[d, ctx_idx])
        ctx_idx_list.append(ctx_idx.astype(np.int64))

        tgt_pos_list.append(latlon[tgt_idx])
        if include_features:
            tgt_feat_list.append(feat_bank[d, tgt_idx])
        tgt_y_list.append(obs[d, tgt_idx].astype(np.float32))
        tgt_idx_list.append(tgt_idx.astype(np.int64))

    if len(ctx_pos_list) == 0:
        return None

    Nc_max = max(a.shape[0] for a in ctx_pos_list)
    Nt_max = max(a.shape[0] for a in tgt_pos_list)

    ctx_pos = _pad_to(ctx_pos_list, Nc_max, pad_val=0.0, dtype=np.float32)
    ctx_feat = _pad_to(ctx_feat_list, Nc_max, pad_val=0.0, dtype=np.float32) if include_features else None
    ctx_node_idx = _pad_to(ctx_idx_list, Nc_max, pad_val=-1, dtype=np.int64)
    ctx_mask = np.zeros((len(ctx_pos_list), Nc_max), dtype=np.float32)
    for i, a in enumerate(ctx_pos_list):
        ctx_mask[i, :a.shape[0]] = 1.0

    tgt_pos = _pad_to(tgt_pos_list, Nt_max, pad_val=0.0, dtype=np.float32)
    tgt_feat = _pad_to(tgt_feat_list, Nt_max, pad_val=0.0, dtype=np.float32) if include_features else None
    tgt_y = _pad_to(tgt_y_list, Nt_max, pad_val=0.0, dtype=np.float32)
    # G2: padded slots get index 0; the loss mask zeroes them out anyway.
    tgt_node_idx = _pad_to(tgt_idx_list, Nt_max, pad_val=0, dtype=np.int64)
    tgt_mask = np.zeros((len(tgt_pos_list), Nt_max), dtype=np.float32)
    for i, a in enumerate(tgt_pos_list):
        tgt_mask[i, :a.shape[0]] = 1.0

    return dict(
        ctx_pos=ctx_pos, ctx_feat=ctx_feat, ctx_node_idx=ctx_node_idx, ctx_mask=ctx_mask,
        tgt_pos=tgt_pos, tgt_feat=tgt_feat, tgt_y=tgt_y, tgt_mask=tgt_mask,
        tgt_node_idx=tgt_node_idx, day_indices=np.asarray(sampled_days, dtype=np.int64),
    )


def build_inference_batch(bundle: DatasetBundle,
                           feat_bank: np.ndarray, latlon: np.ndarray,
                           day_indices: Sequence[int],
                           target_idx: Optional[np.ndarray] = None,
                           include_features: bool = True) -> Dict[str, np.ndarray]:
    """Inference-time batch:
        context = ALL TRAIN stations using exogenous features only
        target  = selected stations, or ALL N stations when target_idx is None.
    """
    train_mask = bundle.train_node_mask
    if target_idx is None:
        target_idx_arr = np.arange(bundle.n_nodes, dtype=np.int64)
    else:
        target_idx_arr = np.asarray(target_idx, dtype=np.int64)
    Nt = int(len(target_idx_arr))
    if Nt == 0:
        raise ValueError('build_inference_batch got an empty target_idx.')

    train_idx = np.where(train_mask)[0].astype(np.int64)
    if train_idx.size == 0:
        raise ValueError('build_inference_batch requires at least one training station.')
    day_idx_arr = np.asarray(day_indices, dtype=np.int64)
    B = len(day_idx_arr)
    Nc = int(train_idx.size)
    ctx_pos = np.broadcast_to(latlon[train_idx][None, :, :], (B, Nc, 2)).copy()
    ctx_feat = feat_bank[day_idx_arr[:, None], train_idx[None, :], :] if include_features else None
    ctx_node_idx = np.broadcast_to(train_idx[None, :], (B, Nc)).copy()
    ctx_mask = np.ones((B, Nc), dtype=np.float32)

    # Target = selected nodes for every day in the batch.
    tgt_pos = np.broadcast_to(latlon[target_idx_arr][None, :, :], (B, Nt, 2)).copy()
    # Day-varying per-node features
    tgt_feat = feat_bank[day_idx_arr[:, None], target_idx_arr[None, :], :] if include_features else None
    # G2: target node global indices, broadcast to all batch days.
    tgt_node_idx = np.broadcast_to(
        target_idx_arr[None, :], (B, Nt)
    ).copy()

    return dict(
        ctx_pos=ctx_pos, ctx_feat=ctx_feat, ctx_node_idx=ctx_node_idx, ctx_mask=ctx_mask,
        tgt_pos=tgt_pos, tgt_feat=tgt_feat,
        tgt_node_idx=tgt_node_idx, day_indices=day_idx_arr,
    )


# =============================================================================
# Loss & utilities
# =============================================================================
def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Per-point Gaussian NLL, averaged over valid entries.
      NLL = 0.5 * log(2*pi*sigma^2) + 0.5*(y - mu)^2 / sigma^2
    """
    var = sigma.pow(2)
    # Element-wise NLL
    nll = 0.5 * torch.log(2 * math.pi * var) + 0.5 * (y - mu).pow(2) / var
    m = mask
    denom = m.sum().clamp_min(1.0)
    return (nll * m).sum() / denom


def censored_gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                               mask: torch.Tensor, floor: float = 0.0,
                               cens_eps: float = 1.0e-3) -> torch.Tensor:
    """G5: censored Gaussian NLL for variables with a hard physical floor.

    For y strictly above the floor: ordinary Gaussian NLL.
    For y at the floor (within cens_eps): -log P(Y <= floor) under the
    Gaussian, which uses the standard normal CDF.

    This is the standard Tobit-style likelihood used for non-negative
    measurements (wind speed, precipitation). It prevents the model from
    being penalized for predicting mu just below the floor when the
    observation is "0 or below sensor limit".

    Args:
        mu, sigma, y: [B, Nt] floats in obs-space.
        mask: [B, Nt] valid mask (1 valid, 0 padding).
        floor: physical lower bound.
        cens_eps: tolerance for treating y as exactly on the floor.
    """
    var = sigma.pow(2).clamp_min(1.0e-6)
    # Standard normal CDF via erf.
    sqrt2 = math.sqrt(2.0)
    z_floor = (floor - mu) / sigma.clamp_min(1.0e-3)
    log_cdf_floor = torch.log(0.5 * (1.0 + torch.erf(z_floor / sqrt2)).clamp_min(1.0e-12))

    nll_uncensored = 0.5 * torch.log(2.0 * math.pi * var) + 0.5 * (y - mu).pow(2) / var
    nll_censored = -log_cdf_floor

    on_floor = (y <= floor + cens_eps)
    nll = torch.where(on_floor, nll_censored, nll_uncensored)
    m = mask
    denom = m.sum().clamp_min(1.0)
    return (nll * m).sum() / denom


def masked_binary_cross_entropy_with_logits(logit: torch.Tensor,
                                            target: torch.Tensor,
                                            mask: torch.Tensor) -> torch.Tensor:
    """Binary exceedance loss averaged over valid target entries."""
    point_loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    valid = mask.float()
    return (point_loss * valid).sum() / valid.sum().clamp_min(1.0)


def generalized_pareto_exceedance_loss(excess: torch.Tensor,
                                       scale: torch.Tensor,
                                       shape: torch.Tensor,
                                       tail_mask: torch.Tensor,
                                       support_penalty_weight: float = 10.0
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPD negative log-likelihood for positive threshold excesses.

    ``excess`` is observed wind speed minus the training-only threshold,
    ``scale`` is the positive GPD scale, and ``shape`` is one shared shape
    parameter for the fold. The support condition is
    1 + shape * excess / scale > 0.
    """
    valid = tail_mask.float()
    denom = valid.sum().clamp_min(1.0)
    scale_safe = scale.clamp_min(1.0e-5)
    excess_safe = excess.clamp_min(0.0)
    support = 1.0 + shape * excess_safe / scale_safe
    support_safe = support.clamp_min(1.0e-6)

    shape_abs = torch.abs(shape)
    shape_safe = torch.where(
        shape_abs < 1.0e-4,
        torch.full_like(shape, 1.0e-4),
        shape,
    )
    exponential_nll = torch.log(scale_safe) + excess_safe / scale_safe
    general_nll = (
        torch.log(scale_safe)
        + (1.0 / shape_safe + 1.0) * torch.log(support_safe)
    )
    point_nll = torch.where(shape_abs < 1.0e-4, exponential_nll, general_nll)

    support_violation = F.relu(1.0e-4 - support)
    support_penalty = float(support_penalty_weight) * (
        support_violation.pow(2) * valid
    ).sum() / denom
    nll = (point_nll * valid).sum() / denom
    return nll, support_penalty


def masked_smooth_l1_loss(prediction: torch.Tensor,
                          target: torch.Tensor,
                          mask: torch.Tensor,
                          beta: float = 1.0) -> torch.Tensor:
    """Smooth-L1 loss for the exported deterministic wind-speed estimate."""
    point_loss = F.smooth_l1_loss(prediction, target, reduction='none', beta=float(beta))
    valid = mask.float()
    return (point_loss * valid).sum() / valid.sum().clamp_min(1.0)


def correction_magnitude_loss(aux: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """Optional magnitude penalty.

    In no_anchor_prior mode aux['scaled_correction'] is deliberately zero, so
    this term contributes nothing.  In nn_init_only mode it would be the neural
    output deviation from the soft-IDW reference.
    """
    corr = aux['scaled_correction']
    m = mask.float()
    return ((corr.pow(2) * m).sum() / m.sum().clamp_min(1.0))


def anchor_warm_start_loss(aux: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """Early teacher loss for nn_init_only; disabled in no_anchor_prior.

    The selected no-softIDW-prior setting sets anchor_warm_start_weight=0 and
    anchor_mode='no_anchor_prior', so this function is never active there.
    """
    if 'raw_nn_mu' not in aux:
        return mask.new_tensor(0.0)
    diff = aux['raw_nn_mu'] - aux['anchor_mean']
    m = mask.float()
    return ((diff.pow(2) * m).sum() / m.sum().clamp_min(1.0))


def get_anchor_warm_start_weight(cfg: ANPConfig, epoch: int) -> float:
    """Linear decay of soft-IDW teacher weight for nn_init_only mode."""
    if str(getattr(cfg, 'anchor_mode', 'legacy_additive')) != 'nn_init_only':
        return 0.0
    base_w = float(getattr(cfg, 'anchor_warm_start_weight', 0.0))
    warm_epochs = int(getattr(cfg, 'anchor_warm_start_epochs', 0))
    if base_w <= 0.0 or warm_epochs <= 0:
        return 0.0
    # epoch starts from 1.  Weight is base_w at epoch 1 and linearly reaches 0
    # just after warm_epochs.
    frac = max(0.0, 1.0 - float(epoch - 1) / float(max(1, warm_epochs)))
    return base_w * frac


def spatial_residual_decorrelation_loss(errors: torch.Tensor,
                                        tgt_latlon: torch.Tensor,
                                        mask: torch.Tensor,
                                        sigma_km: float = 220.0,
                                        max_points: int = 96) -> torch.Tensor:
    """Moran-style residual autocorrelation penalty on target patches.

    errors: [B, Nt] residual prediction errors in residual-space.
    tgt_latlon: [B, Nt, 2] raw degree coordinates.
    mask: [B, Nt] valid target mask (1 valid, 0 padding).

    The loss is intentionally lightweight: each batch item is optionally
    subsampled to max_points to avoid O(N^2) blow-up on large target patches.
    """
    losses: List[torch.Tensor] = []
    B = errors.shape[0]
    sigma = max(float(sigma_km), 1e-3)
    for b in range(B):
        valid = mask[b] > 0.5
        n = int(valid.sum().item())
        if n < 4:
            continue
        idx = torch.nonzero(valid, as_tuple=False).flatten()
        if n > max_points:
            pick = torch.linspace(0, n - 1, steps=max_points, device=idx.device).round().long()
            idx = idx[pick]
        e = errors[b, idx]
        xy = tgt_latlon[b, idx, :]
        e = e - e.mean()
        var = e.pow(2).mean().clamp_min(1.0e-8)
        lat = torch.deg2rad(xy[:, 0])
        lon = torch.deg2rad(xy[:, 1])
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = torch.sin(dlat * 0.5).pow(2) + torch.cos(lat[:, None]) * torch.cos(lat[None, :]) * torch.sin(dlon * 0.5).pow(2)
        d_km = 2.0 * EARTH_R_KM * torch.arcsin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
        w = torch.exp(-(d_km.pow(2)) / (2.0 * sigma * sigma))
        w.fill_diagonal_(0.0)
        denom = w.sum().clamp_min(1.0e-8)
        moran_like = (w * (e[:, None] * e[None, :])).sum() / denom / var
        losses.append(moran_like.abs())
    if not losses:
        return errors.new_tensor(0.0)
    return torch.stack(losses).mean()


def get_lr_lambda(warmup_epochs: int, max_epochs: int):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return fn


# =============================================================================
# Training
# =============================================================================

@torch.no_grad()
def zero_unseen_station_embeddings(model: nn.Module, train_node_mask_np: np.ndarray) -> None:
    """Final A_no_bake setting: force val/test station embeddings to zero.

    The station embedding is trained only for train stations.  For held-out
    stations, Phase H used geographic-IDW baking; Phase I3 showed that the best
    final choice is to not bake and instead use the neutral zero embedding.
    """
    if not bool(getattr(model, 'use_station_emb', False)):
        return
    if getattr(model, 'station_emb', None) is None:
        return
    train_mask = np.asarray(train_node_mask_np).astype(bool)
    unseen_idx = np.where(~train_mask)[0]
    if unseen_idx.size == 0:
        return
    unseen_idx_t = torch.from_numpy(unseen_idx.astype(np.int64)).to(model.station_emb.weight.device)
    model.station_emb.weight.data[unseen_idx_t] = 0.0

def select_eval_days(n_days: int, max_days: int, seed: int) -> np.ndarray:
    if max_days <= 0 or max_days >= n_days:
        return np.arange(n_days, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_days, size=int(max_days), replace=False).astype(np.int64))

class StationBatchTransfer:
    """Keep geometry on the device; cache features only when memory permits.

    Sampling stays on CPU with the original RNG. GPU caching changes only
    where index-based gathers happen, not samples, model, losses or precision.
    """
    def __init__(self, feat_bank: np.ndarray, latlon: np.ndarray, device: torch.device):
        self.device = device
        self.latlon = torch.from_numpy(np.ascontiguousarray(latlon)).to(device)
        self.features = None
        if device.type == 'cuda':
            free_bytes, _ = torch.cuda.mem_get_info(device)
            # Leave memory for full station self-attention, validation and gradients.
            if feat_bank.nbytes <= int(0.25 * free_bytes):
                try:
                    self.features = torch.from_numpy(feat_bank).to(device)
                except torch.cuda.OutOfMemoryError:
                    self.features = None
                    torch.cuda.empty_cache()
        self.feature_on_device = self.features is not None
        print(f'[transfer] geometry resident; feature cache={"GPU" if self.feature_on_device else "CPU batches"}')

    def tensor(self, array: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(array))
        if self.device.type == 'cuda':
            t = t.pin_memory()
        return t.to(self.device, non_blocking=self.device.type == 'cuda')

    def __call__(self, batch: Dict) -> Dict[str, torch.Tensor]:
        out = {name: self.tensor(batch[name]) for name in
               ('ctx_node_idx', 'tgt_node_idx', 'ctx_mask')}
        ctx_idx = out['ctx_node_idx'].clamp(min=0)
        tgt_idx = out['tgt_node_idx'].clamp(min=0)
        out['ctx_pos'] = self.latlon[ctx_idx]
        out['ctx_pos'] = out['ctx_pos'].masked_fill((out['ctx_mask'] <= 0.5).unsqueeze(-1), 0.0)
        out['tgt_pos'] = self.latlon[tgt_idx]
        if 'tgt_mask' in batch:
            out['tgt_mask'] = self.tensor(batch['tgt_mask'])
            out['tgt_pos'] = out['tgt_pos'].masked_fill((out['tgt_mask'] <= 0.5).unsqueeze(-1), 0.0)
            out['tgt_y'] = self.tensor(batch['tgt_y'])
        if self.feature_on_device:
            days = self.tensor(batch['day_indices'])[:, None]
            out['ctx_feat'] = self.features[days, ctx_idx]
            out['tgt_feat'] = self.features[days, tgt_idx]
            out['ctx_feat'] = out['ctx_feat'].masked_fill((out['ctx_mask'] <= 0.5).unsqueeze(-1), 0.0)
            if 'tgt_mask' in out:
                out['tgt_feat'] = out['tgt_feat'].masked_fill((out['tgt_mask'] <= 0.5).unsqueeze(-1), 0.0)
        else:
            out['ctx_feat'] = self.tensor(batch['ctx_feat'])
            out['tgt_feat'] = self.tensor(batch['tgt_feat'])
        return out


@torch.no_grad()
def evaluate_anp_split_fast(model: SpatialANP, bundle: DatasetBundle,
                            feat_bank: np.ndarray, latlon: np.ndarray,
                            cfg: ANPConfig, device: torch.device,
                            split: str,
                            day_indices: np.ndarray,
                            batch_days: Optional[int] = None,
                            transfer: Optional[StationBatchTransfer] = None) -> Dict[str, float]:
    """Lightweight validation: predict only selected split nodes and days."""
    model.eval()
    node_mask = {'Train': bundle.train_node_mask, 'Val': bundle.val_node_mask,
                 'Calib': bundle.calib_node_mask, 'Test': bundle.test_node_mask}[split]
    target_idx = np.where(node_mask)[0].astype(np.int64)
    if target_idx.size == 0:
        out = compute_metrics(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
        out.update({'RMSE90': np.nan})
        return out

    days_all = np.asarray(day_indices, dtype=np.int64)
    bs = int(batch_days or getattr(cfg, 'val_eval_batch_days', cfg.inference_batch_days))
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    for start in range(0, len(days_all), bs):
        days = days_all[start:start + bs]
        if days.size == 0:
            continue
        bt = build_inference_batch(bundle, feat_bank, latlon, days, target_idx=target_idx,
                                   include_features=not (transfer and transfer.feature_on_device))

        tb = transfer(bt) if transfer else {
            name: torch.from_numpy(bt[name]).to(device) for name in
            ('ctx_pos', 'ctx_feat', 'ctx_node_idx', 'ctx_mask', 'tgt_pos', 'tgt_feat', 'tgt_node_idx')
        }
        ctx_pos, ctx_feat = tb['ctx_pos'], tb['ctx_feat']
        ctx_node_idx, ctx_mask = tb['ctx_node_idx'], tb['ctx_mask']
        tgt_pos, tgt_feat, tgt_node_idx = tb['tgt_pos'], tb['tgt_feat'], tb['tgt_node_idx']

        mu, _ = model(
            ctx_pos, ctx_feat, ctx_mask, tgt_pos, tgt_feat,
            tgt_node_idx=tgt_node_idx, ctx_node_idx=ctx_node_idx,
        )
        # The neural mean is the direct daily maximum wind-speed prediction.
        pred_obs = np.clip(mu.cpu().numpy().astype(np.float32), 0.0, None)
        obs = bundle.obs[np.ix_(days, target_idx)]
        valid = bundle.obs_mask[np.ix_(days, target_idx)]
        if np.any(valid):
            y_true_parts.append(obs[valid].astype(np.float32))
            y_pred_parts.append(pred_obs[valid].astype(np.float32))

    if not y_true_parts:
        out = compute_metrics(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
        out.update({'RMSE90': np.nan})
        return out

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out = compute_metrics(y_true, y_pred)
    out.update(compute_p90_rmse(y_true, y_pred, bundle.tail_thresholds))
    return out

def train_anp(bundle: DatasetBundle, cfg: ANPConfig,
              pairwise_km: np.ndarray,
              relative_geometry: np.ndarray
              ) -> Tuple[SpatialANP, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """Train the bulk-and-GPD-tail SpatialANP and return predictions."""
    device = get_device()
    set_seed(cfg.seed)
    cfg.prepare_dirs()

    cfg.tail_threshold = float(bundle.train_tail_threshold)
    if not np.isfinite(cfg.tail_threshold) or cfg.tail_threshold <= 0.0:
        raise ValueError('Fold training observations did not produce a valid tail threshold.')
    feat_bank, latlon, day_feat_dim, climate_feat_dim = build_point_feature_bank(bundle, cfg)
    pairwise_km = np.asarray(pairwise_km, dtype=np.float32)
    relative_geometry = np.asarray(relative_geometry, dtype=np.float32)
    if pairwise_km.shape != (bundle.n_nodes, bundle.n_nodes):
        raise ValueError('Shared pairwise distance cache does not match fold node count.')
    if relative_geometry.shape != (bundle.n_nodes, bundle.n_nodes, 5):
        raise ValueError('Shared directional-geometry cache must have shape [N, N, 5].')

    model = SpatialANP(
        day_feat_dim=day_feat_dim, climate_feat_dim=climate_feat_dim,
        cfg=cfg,
        n_nodes=int(bundle.n_nodes),
        latlon_np=latlon,
        train_node_mask_np=bundle.train_node_mask,
        pairwise_km_np=pairwise_km,
        relative_geometry_np=relative_geometry,
    ).to(device)

    # A_no_bake: keep held-out station embedding rows neutral.
    if bool(getattr(cfg, 'no_bake_unseen_embeddings', True)):
        zero_unseen_station_embeddings(model, bundle.train_node_mask)

    n_params = sum(p.numel() for p in model.parameters())
    transfer = StationBatchTransfer(feat_bank, latlon, device)
    total_feat_dim = int(feat_bank.shape[-1])
    print(f'[tnp] SpatialPhysicsTNP params: {n_params/1e3:.1f}K  '
          f'(day_feat_dim={day_feat_dim}, climate_feat_dim={climate_feat_dim}, total_feat_dim={total_feat_dim}, '
          f'coordinate_features=off, H={cfg.hidden_dim}, heads={cfg.n_heads}, '
          f'local_k={cfg.local_attn_k}, anchor_k={cfg.anchor_k}, '
          f'use_anchor_prior={getattr(model, "use_anchor_prior", False)}, '
          f'anchor_mode={getattr(cfg, "anchor_mode", "legacy_additive")}, '
          f'warm_start_w={getattr(cfg, "anchor_warm_start_weight", 0.0)}, '
          f'warm_start_epochs={getattr(cfg, "anchor_warm_start_epochs", 0)}, '
          f'corr_max={cfg.correction_max_scale}, moran_w={cfg.moran_reg_weight}, '
          f'station_emb={"on" if model.use_station_emb else "off"}'
          f'{"(d=" + str(model.station_emb_dim) + ")" if model.use_station_emb else ""}, '
          f'mem={"on" if cfg.use_residual_memory else "off"}, '
          f'direct_wind_NLL={"on" if cfg.train_in_obs_space else "off"}, '
          f'tail_q={cfg.tail_quantile:.2f}, tail_threshold={cfg.tail_threshold:.3f} m/s, '
          f'gpd_shape_shared=on)')

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, get_lr_lambda(cfg.warmup_epochs, cfg.max_epochs))

    n_days = bundle.n_days
    day_order = np.arange(n_days, dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)
    val_eval_days = select_eval_days(
        n_days, int(getattr(cfg, 'val_eval_max_days', 0)), int(cfg.seed) + 2027
    )

    ckpt_path = Path(cfg.ckpt_dir) / f'{CODE_VERSION_ANP}.pt'

    # Fast reproduction path: load an existing compatible checkpoint (typically
    # the Phase-H checkpoint), then evaluate with A_no_bake. Set
    # cfg.load_existing_checkpoint=False to force a fresh training run.
    if bool(getattr(cfg, 'load_existing_checkpoint', True)) and not bool(getattr(cfg, 'force_recompute', False)):
        script_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
        ckpt_candidates = [ckpt_path]
        for item in getattr(cfg, 'legacy_checkpoint_paths', tuple()):
            p = Path(item)
            ckpt_candidates.append(p)
            if not p.is_absolute():
                ckpt_candidates.append(script_dir / p)
        seen_ckpts = []
        for cand in ckpt_candidates:
            if cand in seen_ckpts:
                continue
            seen_ckpts.append(cand)
            if not cand.exists():
                continue
            try:
                print(f'[A_no_bake] loading existing checkpoint: {cand}')
                state = torch.load(str(cand), map_location=device)
                model.load_state_dict(state, strict=True)
                if bool(getattr(cfg, 'no_bake_unseen_embeddings', True)):
                    zero_unseen_station_embeddings(model, bundle.train_node_mask)
                    print('[A_no_bake] val/test station embeddings forced to zero (no bake).')
                pred_mu, pred_sigma, tail_outputs = predict_anp(
                    model, bundle, feat_bank, latlon, cfg, device, transfer=transfer
                )
                val_metrics = evaluate_prediction_matrix(bundle, pred_mu, 'Val')
                best_val = float(val_metrics.get('RMSE', float('nan')))
                return model, feat_bank, pred_mu, pred_sigma, tail_outputs, {
                    'best_val_RMSE': best_val,
                    'n_params': n_params,
                    'day_feat_dim': day_feat_dim,
                    'climate_feat_dim': climate_feat_dim,
                    'total_feat_dim': int(feat_bank.shape[-1]),
                    'loaded_checkpoint': str(cand),
                    'no_bake_unseen_embeddings': True,
                    'prediction_target': 'station_observed_daily_max_wind_speed',
                    'stage1_background_fit': False,
                    'residual_supervision': False,
                    'observed_residual_input': False,
                    'historical_residual_input': False,
                    'station_embedding': bool(model.use_station_emb),
                    'shared_station_geometry': bool(model.has_station_geometry_cache),
                    'tail_quantile': float(cfg.tail_quantile),
                    'tail_threshold_m_per_s': float(cfg.tail_threshold),
                    'gpd_shape': float(tail_outputs['gpd_shape'][0]),
                }
            except Exception as exc:
                print(f'  [warn] failed to load checkpoint {cand}: {exc}')

    best_val_score, best_val_rmse, best_val_rmse90 = float('inf'), float('inf'), float('inf')
    bad_epochs, best_state = 0, None

    print(f'[tnp] training for up to {cfg.max_epochs} epochs (patience={cfg.patience})')
    print(f'[tnp] fast Val evaluation: {len(val_eval_days)}/{n_days} days, '
          f'{int(bundle.val_node_mask.sum())} Val nodes, batch_days={cfg.val_eval_batch_days}')
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        rng.shuffle(day_order)
        loss_totals = torch.zeros(5, device=device, dtype=torch.float64)
        loss_batches = 0
        for start in range(0, n_days, cfg.batch_days):
            batch = day_order[start:start + cfg.batch_days]
            if batch.size == 0:
                continue
            bt = sample_cnp_batch_train(
                bundle, feat_bank, latlon, batch, rng,
                cfg.drop_rate_min, cfg.drop_rate_max, cfg.min_context_nodes,
                pairwise_km=pairwise_km, cfg=cfg,
                include_features=not transfer.feature_on_device,
            )
            if bt is None:
                continue

            tb = transfer(bt)
            ctx_pos, ctx_feat = tb['ctx_pos'], tb['ctx_feat']
            ctx_node_idx, ctx_mask = tb['ctx_node_idx'], tb['ctx_mask']
            tgt_pos, tgt_feat = tb['tgt_pos'], tb['tgt_feat']
            tgt_y, tgt_mask, tgt_node_idx = tb['tgt_y'], tb['tgt_mask'], tb['tgt_node_idx']

            mu, sigma, aux = model(
                ctx_pos, ctx_feat, ctx_mask, tgt_pos, tgt_feat,
                tgt_node_idx=tgt_node_idx, ctx_node_idx=ctx_node_idx,
                return_aux=True,
            )
            if not bool(getattr(cfg, 'train_in_obs_space', True)):
                raise ValueError('Direct-target training requires train_in_obs_space=True.')
            threshold = float(cfg.tail_threshold)
            tail_target = (tgt_y > threshold).float()
            body_mask = tgt_mask * (1.0 - tail_target)
            tail_mask = tgt_mask * tail_target

            body_nll = censored_gaussian_nll_loss(
                aux['bulk_mean'], sigma, tgt_y, body_mask,
                floor=float(cfg.obs_floor),
            )
            exceedance_loss = masked_binary_cross_entropy_with_logits(
                aux['exceedance_logit'], tail_target, tgt_mask
            )
            gpd_nll, gpd_support_penalty = generalized_pareto_exceedance_loss(
                excess=tgt_y - threshold,
                scale=aux['gpd_scale'],
                shape=aux['gpd_shape'],
                tail_mask=tail_mask,
                support_penalty_weight=cfg.gpd_support_penalty_weight,
            )
            deterministic_loss = masked_smooth_l1_loss(mu, tgt_y, tgt_mask, beta=1.0)
            tail_ramp = min(
                1.0,
                float(epoch) / max(1.0, float(cfg.tail_loss_warmup_epochs)),
            )
            distribution_loss = (
                body_nll
                + tail_ramp * float(cfg.exceedance_loss_weight) * exceedance_loss
                + tail_ramp * float(cfg.gpd_tail_loss_weight) * (gpd_nll + gpd_support_penalty)
                + float(cfg.deterministic_loss_weight) * deterministic_loss
            )
            corr_reg = correction_magnitude_loss(aux, tgt_mask)
            warm_w = get_anchor_warm_start_weight(cfg, epoch)
            warm_reg = anchor_warm_start_loss(aux, tgt_mask) if warm_w > 0.0 else mu.new_tensor(0.0)
            # G5: Moran skipped when weight is 0 (saves compute).
            moran_w = float(cfg.moran_reg_weight)
            if moran_w > 0.0:
                moran_reg = spatial_residual_decorrelation_loss(
                    tgt_y - mu, tgt_pos, tgt_mask,
                    sigma_km=cfg.moran_sigma_km,
                    max_points=cfg.moran_max_points,
                )
                loss = (distribution_loss
                        + float(cfg.correction_reg_weight) * corr_reg
                        + warm_w * warm_reg
                        + moran_w * moran_reg)
            else:
                loss = distribution_loss + float(cfg.correction_reg_weight) * corr_reg + warm_w * warm_reg
            if not torch.isfinite(loss):
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            loss_totals += torch.stack([
                loss.detach(), body_nll.detach(), exceedance_loss.detach(),
                gpd_nll.detach(), deterministic_loss.detach(),
            ]).to(torch.float64)
            loss_batches += 1
        scheduler.step()

        # A_no_bake validation: do NOT bake held-out station embeddings.
        # Force unseen rows to a neutral zero vector before each validation pass.
        if bool(getattr(cfg, 'no_bake_unseen_embeddings', True)):
            zero_unseen_station_embeddings(model, bundle.train_node_mask)
        val_metrics = evaluate_anp_split_fast(
            model, bundle, feat_bank, latlon, cfg, device,
            split='Val', day_indices=val_eval_days,
            batch_days=int(getattr(cfg, 'val_eval_batch_days', cfg.inference_batch_days)), transfer=transfer,
        )
        val_rmse = val_metrics.get('RMSE', float('inf'))
        val_rmse90 = val_metrics.get('RMSE90', float('inf'))
        if not np.isfinite(val_rmse90):
            val_rmse90 = val_rmse
        val_score = val_rmse + float(cfg.early_stop_tail_weight) * val_rmse90

        if np.isfinite(val_score) and val_score < best_val_score:
            best_val_score = val_score
            best_val_rmse = val_rmse
            best_val_rmse90 = val_rmse90
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            bad_epochs = 0
            tag = '*'
        else:
            bad_epochs += 1
            tag = ' '

        averages = (loss_totals / max(1, loss_batches)).cpu().numpy()
        if not loss_batches:
            averages[:] = np.nan
        avg_loss, avg_body, avg_event, avg_gpd, avg_point = averages.tolist()
        shape_now = float(model.gpd_shape_limit * torch.tanh(model.gpd_raw_shape).detach().cpu())
        print(f'  [epoch {epoch:03d}]{tag} loss={avg_loss:.4f} body={avg_body:.4f} '
              f'event={avg_event:.4f} gpd={avg_gpd:.4f} point={avg_point:.4f} '
              f'val_RMSE={val_rmse:.4f} val_RMSE90={val_rmse90:.4f} '
              f'best_score={best_val_score:.4f} xi={shape_now:.4f} '
              f'lr={optim.param_groups[0]["lr"]:.2e}  '
              f'{time.time()-t0:.0f}s')

        if bad_epochs >= cfg.patience:
            print(f'  [early-stop] no val improvement for {cfg.patience} epochs.')
            break

    if best_state is None:
        raise RuntimeError('No valid validation checkpoint was produced; station predictions were not exported.')
    model.load_state_dict(best_state)
    # Final selected setting: A_no_bake.  Do not call bake_unseen_embeddings().
    if bool(getattr(cfg, 'no_bake_unseen_embeddings', True)):
        zero_unseen_station_embeddings(model, bundle.train_node_mask)
        print('[A_no_bake] final val/test station embeddings forced to zero (no bake).')
    pred_mu, pred_sigma, tail_outputs = predict_anp(
        model, bundle, feat_bank, latlon, cfg, device, transfer=transfer
    )
    final_val_metrics = evaluate_prediction_matrix(bundle, pred_mu, 'Val')

    return model, feat_bank, pred_mu, pred_sigma, tail_outputs, {
        'best_val_RMSE': float(final_val_metrics.get('RMSE', float('nan'))),
        'best_sampled_val_RMSE': best_val_rmse,
        'best_sampled_val_RMSE90': best_val_rmse90,
        'best_sampled_val_score': best_val_score,
        'val_eval_days': int(len(val_eval_days)),
        'n_params': n_params,
        'day_feat_dim': day_feat_dim,
        'climate_feat_dim': climate_feat_dim,
        'total_feat_dim': int(feat_bank.shape[-1]),
        'prediction_target': 'station_observed_daily_max_wind_speed',
        'stage1_background_fit': False,
        'residual_supervision': False,
        'observed_residual_input': False,
        'historical_residual_input': False,
        'station_embedding': bool(model.use_station_emb),
        'shared_station_geometry': bool(model.has_station_geometry_cache),
        'tail_quantile': float(cfg.tail_quantile),
        'tail_threshold_m_per_s': float(cfg.tail_threshold),
        'gpd_shape': float(tail_outputs['gpd_shape'][0]),
    }


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def predict_anp(model: SpatialANP, bundle: DatasetBundle,
                feat_bank: np.ndarray, latlon: np.ndarray,
                cfg: ANPConfig, device: torch.device,
                batch_days: Optional[int] = None,
                transfer: Optional[StationBatchTransfer] = None
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Predict deterministic wind speed and bulk/GPD tail diagnostics."""
    model.eval()
    n_days = bundle.n_days
    n_nodes = bundle.n_nodes
    bs = batch_days or cfg.inference_batch_days

    # The matrix stores the neural model's direct wind-speed mean.
    pred_mu_full = np.zeros((n_days, n_nodes), dtype=np.float32)
    pred_sigma_full = np.zeros((n_days, n_nodes), dtype=np.float32)
    pred_exceedance_probability = np.zeros((n_days, n_nodes), dtype=np.float32)
    pred_gpd_scale = np.zeros((n_days, n_nodes), dtype=np.float32)
    pred_tail_conditional_mean = np.zeros((n_days, n_nodes), dtype=np.float32)

    for start in range(0, n_days, bs):
        end = min(start + bs, n_days)
        days = np.arange(start, end, dtype=np.int64)
        bt = build_inference_batch(bundle, feat_bank, latlon, days,
                                   include_features=not (transfer and transfer.feature_on_device))

        tb = transfer(bt) if transfer else {
            name: torch.from_numpy(bt[name]).to(device) for name in
            ('ctx_pos', 'ctx_feat', 'ctx_node_idx', 'ctx_mask', 'tgt_pos', 'tgt_feat', 'tgt_node_idx')
        }
        ctx_pos, ctx_feat = tb['ctx_pos'], tb['ctx_feat']
        ctx_node_idx, ctx_mask = tb['ctx_node_idx'], tb['ctx_mask']
        tgt_pos, tgt_feat, tgt_node_idx = tb['tgt_pos'], tb['tgt_feat'], tb['tgt_node_idx']

        mu, sigma, aux = model(
            ctx_pos, ctx_feat, ctx_mask, tgt_pos, tgt_feat,
            tgt_node_idx=tgt_node_idx, ctx_node_idx=ctx_node_idx,
            return_aux=True,
        )
        values = torch.stack([
            mu, sigma, aux['exceedance_probability'], aux['gpd_scale'], aux['tail_conditional_mean'],
        ], dim=-1).cpu().numpy().astype(np.float32, copy=False)
        pred_mu_full[start:end] = values[..., 0]
        pred_sigma_full[start:end] = values[..., 1]
        pred_exceedance_probability[start:end] = values[..., 2]
        pred_gpd_scale[start:end] = values[..., 3]
        pred_tail_conditional_mean[start:end] = values[..., 4]

    pred_obs = np.clip(pred_mu_full, 0.0, None)
    gpd_shape = float(model.gpd_shape_limit * torch.tanh(model.gpd_raw_shape).detach().cpu())
    tail_outputs = {
        'exceedance_probability': pred_exceedance_probability,
        'gpd_scale': pred_gpd_scale,
        'tail_conditional_mean': pred_tail_conditional_mean,
        'gpd_shape': np.asarray([gpd_shape], dtype=np.float32),
        'tail_threshold': np.asarray([float(cfg.tail_threshold)], dtype=np.float32),
        'tail_quantile': np.asarray([float(cfg.tail_quantile)], dtype=np.float32),
    }
    return pred_obs, pred_sigma_full, tail_outputs


# =============================================================================
# PICP @ 90% coverage using learned Gaussian
# =============================================================================
def compute_picp_gaussian(bundle: DatasetBundle, pred_mu: np.ndarray,
                          pred_sigma: np.ndarray, split: str,
                          coverage: float = 0.90) -> Dict[str, float]:
    z = float(NormalDist().inv_cdf(0.5 + coverage / 2.0))
    y_true, mask = flatten_split_targets(bundle, split)
    mu = pred_mu[mask].astype(np.float64)
    sg = pred_sigma[mask].astype(np.float64)
    lo = mu - z * sg
    hi = mu + z * sg
    covered = (y_true >= lo) & (y_true <= hi)
    width = hi - lo
    return {
        f'PICP_{int(coverage*100)}': float(np.mean(covered)),
        f'Width_{int(coverage*100)}': float(np.mean(width)),
    }


# =============================================================================
# Continuous-field demo
# =============================================================================

def scale_2d_with_stats(arr: np.ndarray, stats: Dict[str, List[float]], label: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mu = np.asarray(stats['mean'], dtype=np.float32)
    sd = np.asarray(stats['std'], dtype=np.float32)
    if arr.shape[-1] != mu.shape[0]:
        raise ValueError(f'{label}: feature dim {arr.shape[-1]} != scaler dim {mu.shape[0]}')
    out = (arr - mu[None, :]) / np.where(sd[None, :] > 0, sd[None, :], 1.0)
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def scale_3d_with_stats(arr: np.ndarray, stats: Dict[str, List[float]], label: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mu = np.asarray(stats['mean'], dtype=np.float32)
    sd = np.asarray(stats['std'], dtype=np.float32)
    if arr.shape[-1] != mu.shape[0]:
        raise ValueError(f'{label}: feature dim {arr.shape[-1]} != scaler dim {mu.shape[0]}')
    out = (arr - mu[None, None, :]) / np.where(sd[None, None, :] > 0, sd[None, None, :], 1.0)
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_static_table_for_grid_points(input_dir: Path, grid_ll: np.ndarray,
                                      required_feature_names: Sequence[str],
                                      max_dist_km: float) -> pd.DataFrame:
    """Load real static-grid rows for arbitrary prediction points."""
    static_path = input_dir / '静态数据汇总.xlsx'
    if not static_path.exists():
        raise FileNotFoundError(f'连续格点预测缺少静态文件: {static_path}')
    static = pd.read_excel(static_path).replace(-999, np.nan)
    required = set(['Lat', 'Lon']) | set(required_feature_names)
    # Also keep BHYYYY columns for the yearly BH feature.
    for col in static.columns:
        name = str(col)
        if len(name) == 6 and name.startswith('BH') and name[2:].isdigit():
            required.add(col)
    missing = required - set(static.columns)
    if missing:
        raise KeyError(f'静态文件缺少列: {sorted(missing)}')
    static = static.apply(lambda col: col.fillna(col.median()) if np.issubdtype(col.dtype, np.number) else col)
    idx, _ = nearest_match_indices(
        static[['Lat', 'Lon']].to_numpy(np.float64),
        np.asarray(grid_ll, dtype=np.float64),
        max_dist_km=max_dist_km,
        label='grid->static',
    )
    out = static.iloc[idx].copy().reset_index(drop=True)
    # Lat/Lon should describe the prediction point itself; terrain/BH columns
    # come from the nearest static 0.1° grid cell.
    out['Lat'] = grid_ll[:, 0].astype(np.float32)
    out['Lon'] = grid_ll[:, 1].astype(np.float32)
    return out


def load_dynamic_series_for_grid_points(input_dir: Path,
                                        data_cfg: ExperimentConfig,
                                        bundle: DatasetBundle,
                                        grid_ll: np.ndarray,
                                        day_indices: np.ndarray) -> np.ndarray:
    """Load dynamic variables at grid points for selected global day indices.

    Returned shape: [len(day_indices), G, K]. Each grid point is matched to the
    nearest ERA5-Land CSV row by BallTree and rejected if the distance exceeds
    data_cfg.coord_match_max_km.
    """
    day_indices = np.asarray(day_indices, dtype=np.int64)
    if day_indices.ndim != 1 or day_indices.size == 0:
        raise ValueError('day_indices 不能为空。')
    dates_sel = bundle.dates.iloc[day_indices].copy()
    if 'date' in dates_sel.columns:
        dt = pd.to_datetime(dates_sel['date'])
    else:
        dt = pd.to_datetime(dates_sel[['Year', 'Month', 'Day']])
    dates_sel = dates_sel.assign(_doy=dt.dt.dayofyear.to_numpy(np.int64))

    G = int(grid_ll.shape[0])
    K = len(data_cfg.dynamic_vars)
    out = np.full((len(day_indices), G, K), np.nan, dtype=np.float32)
    dyn_mean = np.asarray(bundle.scaler_stats['daily_dyn']['mean'], dtype=np.float32)

    for year in sorted(pd.unique(dates_sel['Year'])):
        loc = np.where(dates_sel['Year'].to_numpy() == year)[0]
        doys = dates_sel.iloc[loc]['_doy'].to_numpy(np.int64)
        pcols = [f'p{int(d)}' for d in doys]
        for k, var in enumerate(data_cfg.dynamic_vars):
            path = input_dir / f'{var}_{int(year)}.csv'
            if not path.exists():
                raise FileNotFoundError(f'连续格点预测缺少动态变量文件: {path}')
            year_df = read_dynamic_csv_robust(path)
            if not {'Lat', 'Lon'}.issubset(set(year_df.columns)):
                raise KeyError(f'{path.name} 缺少 Lat/Lon 列。')
            missing_cols = [c for c in pcols if c not in year_df.columns]
            if missing_cols:
                raise KeyError(f'{path.name} 缺少日期列: {missing_cols[:10]}')
            idx, _ = nearest_match_indices(
                year_df[['Lat', 'Lon']].to_numpy(np.float64),
                np.asarray(grid_ll, dtype=np.float64),
                max_dist_km=float(data_cfg.coord_match_max_km),
                label=f'grid->{var}_{int(year)}',
            )
            vals = year_df.iloc[idx][pcols].to_numpy(np.float32).T
            out[loc, :, k] = vals

    if not np.isfinite(out).all():
        for k in range(K):
            fill = float(dyn_mean[k]) if k < dyn_mean.size and np.isfinite(dyn_mean[k]) else 0.0
            bad = ~np.isfinite(out[:, :, k])
            if bad.any():
                out[:, :, k][bad] = fill
    return out.astype(np.float32)


def build_single_day_temporal_grid_features(dyn_scaled_series: np.ndarray,
                                            model_cfg: ANPConfig) -> np.ndarray:
    """Match build_causal_temporal_features() ordering for one target day."""
    dyn_scaled_series = np.asarray(dyn_scaled_series, dtype=np.float32)
    cur = dyn_scaled_series[-1]
    lag1 = dyn_scaled_series[-2] if dyn_scaled_series.shape[0] >= 2 else cur
    feats: List[np.ndarray] = []

    if bool(getattr(model_cfg, 'temporal_include_lag1', True)):
        feats.append(lag1)

    roll_means: List[np.ndarray] = []
    for w in tuple(getattr(model_cfg, 'temporal_roll_windows', (3, 7))):
        w = int(w)
        if w <= 1:
            continue
        rm = dyn_scaled_series[-min(w, dyn_scaled_series.shape[0]):].mean(axis=0).astype(np.float32)
        roll_means.append(rm)
        feats.append(rm)

    if bool(getattr(model_cfg, 'temporal_include_delta1', True)):
        feats.append((cur - lag1).astype(np.float32))

    if bool(getattr(model_cfg, 'temporal_include_anomaly_to_roll', True)):
        for rm in roll_means:
            feats.append((cur - rm).astype(np.float32))

    if not feats:
        return np.zeros((cur.shape[0], 0), dtype=np.float32)
    return np.concatenate(feats, axis=-1).astype(np.float32)


def build_single_day_climate_grid_features(dyn_scaled_series: np.ndarray,
                                           static_const_scaled: np.ndarray,
                                           global_day_indices: np.ndarray,
                                           bundle: DatasetBundle) -> np.ndarray:
    """Rebuild one-day climate vector for grid cells, then apply train-only scaling."""
    if 'climate_vectors' not in bundle.scaler_stats:
        raise ValueError(
            '缓存缺少 climate_vectors 标准化参数。请用本修改版代码重新构建数据缓存，'
            '或设置 ExperimentConfig.force_recompute=True 后重新运行。'
        )
    x = np.asarray(dyn_scaled_series, dtype=np.float32)
    L = x.shape[0]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    soft_hi = x[-min(30, L):].max(axis=0)

    gidx = np.asarray(global_day_indices, dtype=np.float32)
    sin_w = np.sin(2.0 * np.pi * gidx / 366.0)[:, None, None] * x
    cos_w = np.cos(2.0 * np.pi * gidx / 366.0)[:, None, None] * x
    sin_proj = sin_w.mean(axis=0)
    cos_proj = cos_w.mean(axis=0)
    amp = np.sqrt(sin_proj ** 2 + cos_proj ** 2 + 1.0e-8)

    raw = np.concatenate([mean, std, soft_hi, amp, static_const_scaled], axis=-1).astype(np.float32)
    return scale_2d_with_stats(raw, bundle.scaler_stats['climate_vectors'], 'grid climate vectors')


def build_grid_features_for_day(bundle: DatasetBundle,
                                model_cfg: ANPConfig,
                                data_cfg: ExperimentConfig,
                                grid_ll: np.ndarray,
                                day_idx: int,
                                grid_store=None,
                                grid_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Rebuild the target-grid feature row from true grid-cell inputs."""
    if bool(getattr(model_cfg, 'use_residual_memory', False)):
        raise ValueError('本实验禁止为连续网格构造观测残差或历史残差特征。')
    input_dir = resolve_input_directory(data_cfg.input_dir)
    if day_idx < 0 or day_idx >= bundle.n_days:
        raise IndexError(f'day_idx={day_idx} 超出范围 [0, {bundle.n_days})')

    grid_ll = np.asarray(grid_ll, dtype=np.float32)
    if grid_store is not None:
        static_table = grid_store.static.iloc[grid_indices].copy().reset_index(drop=True)
    else:
        static_table = load_static_table_for_grid_points(
            input_dir, grid_ll, bundle.const_feature_names,
            max_dist_km=float(data_cfg.coord_match_max_km),
        )
    static_raw = static_table[list(bundle.const_feature_names)].to_numpy(np.float32)
    static_scaled = scale_2d_with_stats(static_raw, bundle.scaler_stats['static_const'], 'grid static_const')

    cur_row = bundle.dates.iloc[int(day_idx)]
    cur_year = int(cur_row['Year'])
    cur_date = pd.to_datetime(cur_row['date']) if 'date' in bundle.dates.columns else pd.to_datetime({
        'year': [int(cur_row['Year'])], 'month': [int(cur_row['Month'])], 'day': [int(cur_row['Day'])]
    })[0]
    doy = float(cur_date.dayofyear)

    if len(bundle.year_feature_names) > 0:
        if bundle.year_feature_names != ['BH_year']:
            raise NotImplementedError(f'当前连续格点预测只支持 BH_year 年度特征，收到: {bundle.year_feature_names}')
        year_feats_3d, _, _ = build_bh_yearly_features(static_table, [cur_year], 'BH_year')
        year_feats = year_feats_3d[0].astype(np.float32)
    else:
        year_feats = np.zeros((grid_ll.shape[0], 0), dtype=np.float32)

    max_temporal_w = 1
    if bool(getattr(model_cfg, 'use_temporal_features', False)):
        ws = [int(w) for w in tuple(getattr(model_cfg, 'temporal_roll_windows', (3, 7))) if int(w) > 1]
        max_temporal_w = max([1] + ws)
        if bool(getattr(model_cfg, 'temporal_include_lag1', True)) or bool(getattr(model_cfg, 'temporal_include_delta1', True)):
            max_temporal_w = max(max_temporal_w, 2)
    hist_needed = max(int(getattr(data_cfg, 'climate_hist_len', 1)) if bool(getattr(model_cfg, 'use_climate', False)) else 1,
                      max_temporal_w)
    start_day = max(0, int(day_idx) - hist_needed + 1)
    global_days = np.arange(start_day, int(day_idx) + 1, dtype=np.int64)

    if grid_store is not None:
        dyn_raw_series = grid_store.raw_series(global_days, grid_indices)
        # Missing grid inputs use this fold's training mean, exactly as before.
        means = np.asarray(bundle.scaler_stats['daily_dyn']['mean'], dtype=np.float32)
        for k in range(dyn_raw_series.shape[-1]):
            channel = dyn_raw_series[..., k]
            channel[~np.isfinite(channel)] = means[k]
    else:
        dyn_raw_series = load_dynamic_series_for_grid_points(input_dir, data_cfg, bundle, grid_ll, global_days)
    dyn_scaled_series = scale_3d_with_stats(dyn_raw_series, bundle.scaler_stats['daily_dyn'], 'grid daily_dyn')
    dyn_current_raw = dyn_raw_series[-1]

    doy_sin_dayfeat = np.float32(np.sin(2.0 * np.pi * doy / 366.0))
    doy_cos_dayfeat = np.float32(np.cos(2.0 * np.pi * doy / 366.0))
    date_feat = np.stack([
        np.full(grid_ll.shape[0], doy_sin_dayfeat, dtype=np.float32),
        np.full(grid_ll.shape[0], doy_cos_dayfeat, dtype=np.float32),
    ], axis=-1)
    day_raw = np.concatenate([static_raw, year_feats, dyn_current_raw, date_feat], axis=-1).astype(np.float32)
    day_scaled = scale_2d_with_stats(day_raw, bundle.scaler_stats['day_features'], 'grid day_features')

    day_parts = [day_scaled]
    if bool(getattr(model_cfg, 'use_temporal_features', False)):
        day_parts.append(build_single_day_temporal_grid_features(dyn_scaled_series, model_cfg))
    day_feat = np.concatenate(day_parts, axis=-1).astype(np.float32)

    if bool(getattr(model_cfg, 'use_climate', False)):
        climate_feat = build_single_day_climate_grid_features(dyn_scaled_series, static_scaled, global_days, bundle)
        grid_feat = np.concatenate([day_feat, climate_feat], axis=-1).astype(np.float32)
    else:
        grid_feat = day_feat

    return grid_feat


@torch.no_grad()
def predict_on_grid(model: SpatialANP, bundle: DatasetBundle,
                    feat_bank: np.ndarray, latlon: np.ndarray,
                    day_idx: int, cfg: ANPConfig, device: torch.device,
                    grid_res_deg: float = 0.1,
                    data_cfg: Optional[ExperimentConfig] = None,
                    grid_latlon: Optional[np.ndarray] = None,
                    grid_store=None) -> Dict[str, np.ndarray]:
    """Predict actual irregular static-grid cells; feature construction is chunked.

    grid_res_deg is retained for call compatibility. No rectangular domain or
    interpolation from predicted stations is used. Raw coordinates enter only
    directional geometry and nearest-reference selection.
    """
    model.eval()
    data_cfg = data_cfg or ExperimentConfig()
    use_static_rows = grid_latlon is None
    if grid_store is None and use_static_rows:
        grid_store = EventGridStore(data_cfg, bundle, Path(cfg.cache_dir) / 'event_grid')
    grid_ll = grid_store.latlon if use_static_rows else np.asarray(grid_latlon, dtype=np.float32)
    if grid_ll.ndim != 2 or grid_ll.shape[1] != 2 or not np.isfinite(grid_ll).all():
        raise ValueError('Grid coordinates must be finite [G, 2] latitude/longitude.')
    if len(grid_ll) == 0:
        raise ValueError('Static grid contains no prediction points.')
    if use_static_rows:
        grid_store.prepare_day(int(day_idx), cfg)

    cand = np.flatnonzero(bundle.train_node_mask)
    if cand.size == 0:
        raise ValueError('Grid inference requires training reference stations.')
    ctx_pos_t = torch.from_numpy(latlon[cand][None]).to(device)
    ctx_feat_t = torch.from_numpy(np.ascontiguousarray(feat_bank[day_idx, cand][None])).to(device)
    ctx_node_idx_t = torch.from_numpy(cand.astype(np.int64)[None]).to(device)
    ctx_mask_t = torch.ones((1, len(cand)), device=device)
    encoded_context = model.encode_context(ctx_feat_t, ctx_mask_t)
    G = len(grid_ll)
    names = ('wind', 'sigma', 'exceedance_probability', 'gpd_scale', 'tail_conditional_mean')
    output = {name: np.empty(G, dtype=np.float32) for name in names}
    # History features are built BEFORE transfer, only for the current chunk.
    chunk = max(1, int(cfg.grid_chunk_size))
    for start in range(0, G, chunk):
        end = min(start + chunk, G)
        indices = np.arange(start, end, dtype=np.int64)
        grid_feat = build_grid_features_for_day(
            bundle, cfg, data_cfg, grid_ll[start:end], int(day_idx),
            grid_store=grid_store if use_static_rows else None,
            grid_indices=indices if use_static_rows else None,
        )
        if grid_feat.shape[-1] != feat_bank.shape[-1]:
            raise ValueError('Station and grid feature dimensions differ.')
        tgt_pos_t = torch.from_numpy(grid_ll[None, start:end]).to(device)
        tgt_feat_t = torch.from_numpy(grid_feat[None]).to(device)
        mu, sigma, aux = model(
            ctx_pos_t, ctx_feat_t, ctx_mask_t, tgt_pos_t, tgt_feat_t,
            tgt_node_idx=None, ctx_node_idx=ctx_node_idx_t,
            return_aux=True, encoded_context=encoded_context,
        )
        values = torch.stack([mu, sigma, aux['exceedance_probability'],
                              aux['gpd_scale'], aux['tail_conditional_mean']], dim=-1)
        values = values.squeeze(0).cpu().numpy()
        for k, name in enumerate(names):
            output[name][start:end] = values[:, k]
    output['wind'] = np.maximum(output['wind'], 0.0)
    output['latlon'] = grid_ll
    output['gpd_shape'] = np.asarray([
        float((model.gpd_shape_limit * torch.tanh(model.gpd_raw_shape)).cpu())
    ], dtype=np.float32)
    output['tail_threshold'] = np.asarray([float(cfg.tail_threshold)], dtype=np.float32)
    if use_static_rows:
        output['static_grid_row_index'] = np.arange(G, dtype=np.int32)
    return output


# Requested representative strong-wind days. These are outputs, not extra scores.
EVENT_DAYS: Tuple[Tuple[str, str], ...] = (
    ('2015-03-30', 'Alashankou gale'),
    ('2015-04-15', 'Northern China gale'),
    ('2015-10-04', 'Typhoon Mujigae'),
    ('2018-09-16', 'Typhoon Mangkhut'),
)


class EventGridStore:
    """Actual irregular static grid and shared, disk-backed raw meteorology.

    CSV parsing and grid matching occur once per required variable/year, not
    once per fold/event/chunk. On-disk arrays contain raw grid inputs; each fold
    still applies ONLY its training-fitted scalers when making features.
    """
    def __init__(self, data_cfg: ExperimentConfig, bundle: DatasetBundle, cache_dir: Path):
        self.cfg = data_cfg
        self.dates = bundle.dates
        self.root = resolve_input_directory(data_cfg.input_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.static = pd.read_excel(self.root / '静态数据汇总.xlsx').replace(-999, np.nan)
        self.static = self.static.reset_index(drop=True)
        self.latlon = self.static[['Lat', 'Lon']].to_numpy(np.float32)
        if not len(self.latlon) or not np.isfinite(self.latlon).all():
            raise ValueError('静态数据汇总.xlsx 必须包含有效的网格经纬度。')
        # Keep every input grid row in its original order, including the irregular boundary.
        for name in self.static.columns:
            if name not in ('Lat', 'Lon') and pd.api.types.is_numeric_dtype(self.static[name]):
                self.static[name] = self.static[name].fillna(self.static[name].median())
        self.grid_hash = array_digest(self.latlon)
        self.signature = source_file_signature(data_cfg)
        self._ready_years = set()
        print(f'[event grid] {len(self.latlon)} actual static-grid rows; irregular domain retained.')

    def _paths(self, year: int):
        prefix = self.cache_dir / f'raw_grid_{year}_{self.grid_hash[:12]}'
        return prefix.with_suffix('.json'), [
            Path(str(prefix) + f'_{var}.npy') for var in self.cfg.dynamic_vars
        ]

    def ensure_year(self, year: int) -> None:
        if year in self._ready_years:
            return
        meta_path, paths = self._paths(year)
        expected_days = int((pd.Timestamp(year + 1, 1, 1) - pd.Timestamp(year, 1, 1)).days)
        expected = {'source_signature': self.signature, 'grid_hash': self.grid_hash,
                    'shape': [expected_days, len(self.latlon)], 'vars': list(self.cfg.dynamic_vars)}
        valid = meta_path.exists() and all(path.exists() for path in paths)
        if valid:
            with meta_path.open(encoding='utf-8') as f:
                valid = json.load(f) == expected
            if valid:
                for path in paths:
                    arr = np.load(path, mmap_mode='r', allow_pickle=False)
                    valid = valid and arr.shape == tuple(expected['shape']) and arr.dtype == np.float32
                    del arr
        if valid:
            self._ready_years.add(year)
            return
        pcols = [f'p{d}' for d in range(1, expected_days + 1)]
        for var, path in zip(self.cfg.dynamic_vars, paths):
            csv_path = self.root / f'{var}_{year}.csv'
            print(f'[event grid cache] aligning {csv_path.name}')
            df = read_dynamic_csv_robust(csv_path)
            missing = set(pcols) - set(df.columns)
            if missing:
                raise KeyError(f'{csv_path.name} 缺少日期列: {sorted(missing)[:5]}')
            idx, _ = nearest_match_indices(
                df[['Lat', 'Lon']].to_numpy(np.float64), self.latlon,
                self.cfg.coord_match_max_km, f'event grid->{var}_{year}',
            )
            tmp = Path(str(path) + '.tmp.npy')
            target = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32,
                                                shape=(expected_days, len(self.latlon)))
            column_indices = df.columns.get_indexer(pcols)
            for left in range(0, len(idx), 2048):
                right = min(len(idx), left + 2048)
                target[:, left:right] = df.iloc[idx[left:right], column_indices].to_numpy(np.float32).T
            target.flush()
            del target, df
            tmp.replace(path)
        tmp_meta = meta_path.with_suffix('.tmp.json')
        with tmp_meta.open('w', encoding='utf-8') as f:
            json.dump(expected, f, ensure_ascii=False)
        tmp_meta.replace(meta_path)
        self._ready_years.add(year)

    def prepare_day(self, day_idx: int, model_cfg: ANPConfig) -> None:
        windows = [int(w) for w in model_cfg.temporal_roll_windows]
        history = max([2] + windows + ([int(self.cfg.climate_hist_len)] if model_cfg.use_climate else []))
        rows = self.dates.iloc[max(0, day_idx - history + 1):day_idx + 1]
        for year in sorted(pd.unique(rows['Year'])):
            self.ensure_year(int(year))

    def raw_series(self, global_days: np.ndarray, grid_indices: np.ndarray) -> np.ndarray:
        rows = self.dates.iloc[np.asarray(global_days, np.int64)]
        years = rows['Year'].to_numpy()
        doy = pd.to_datetime(rows['date']).dt.dayofyear.to_numpy(np.int64) - 1
        grid_indices = np.asarray(grid_indices, np.int64)
        result = np.empty((len(rows), len(grid_indices), len(self.cfg.dynamic_vars)), np.float32)
        for year in sorted(pd.unique(years)):
            self.ensure_year(int(year))
            loc = np.flatnonzero(years == year)
            _, paths = self._paths(int(year))
            for k, path in enumerate(paths):
                raw = np.load(path, mmap_mode='r', allow_pickle=False)
                result[loc, :, k] = raw[np.ix_(doy[loc], grid_indices)]
                del raw
        return result


def plotting_pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_test_scatter(fold_dir: Path, fold_id: int,
                      bundle: DatasetBundle, prediction: np.ndarray) -> None:
    """All valid test day/station pairs, shown as a density scatter without subsampling."""
    plt = plotting_pyplot()
    mask = bundle.obs_mask & bundle.test_node_mask[None, :] & np.isfinite(prediction)
    x, y = bundle.obs[mask], np.asarray(prediction)[mask]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    upper = max(1.0, float(np.nanmax(x)) if x.size else 1.0,
                float(np.nanmax(y)) if y.size else 1.0)
    upper = float(np.ceil(upper))
    if x.size:
        density = ax.hexbin(x, y, gridsize=95, bins='log', mincnt=1, cmap='viridis',
                            extent=(0, upper, 0, upper), linewidths=0, rasterized=True)
        fig.colorbar(density, ax=ax, label='Number of samples')
    ax.plot([0, upper], [0, upper], '--', color='0.25', lw=1)
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel='Observed DMWS (m/s)',
           ylabel='Predicted DMWS (m/s)', title=f'Fold {fold_id}: test stations')
    ax.set_aspect('equal', adjustable='box')
    (fold_dir / 'figs').mkdir(parents=True, exist_ok=True)
    fig.savefig(fold_dir / 'figs' / f'fold{fold_id:02d}_test_scatter.png', dpi=300)
    plt.close(fig)


def save_fold_inference_metadata(fold_dir: Path, bundle: DatasetBundle,
                                 data_cfg: ExperimentConfig, model_cfg: ANPConfig) -> None:
    """Keep scalers and feature order alongside the already-saved best weights."""
    directory = fold_dir / 'ckpt'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'inference_metadata.json').open('w', encoding='utf-8') as f:
        json.dump({
            'model_config': asdict(model_cfg), 'data_config': asdict(data_cfg),
            'scaler_stats': bundle.scaler_stats, 'feature_names': bundle.feature_names,
            'const_feature_names': bundle.const_feature_names,
            'year_feature_names': bundle.year_feature_names, 'dyn_feature_names': bundle.dyn_feature_names,
            'day_feat_dim': int(bundle.day_features.shape[-1]) + (
                len(bundle.dyn_feature_names) * (int(model_cfg.temporal_include_lag1)
                + sum(int(w) > 1 for w in model_cfg.temporal_roll_windows)
                + int(model_cfg.temporal_include_delta1)
                + int(model_cfg.temporal_include_anomaly_to_roll)
                  * sum(int(w) > 1 for w in model_cfg.temporal_roll_windows))
                if model_cfg.use_temporal_features else 0),
            'climate_feat_dim': int(bundle.climate_vectors.shape[-1]) if model_cfg.use_climate else 0,
            'station_metadata': f'../Table_fold{int(bundle.split_summary["fold_id"]):02d}_node_split_metadata.csv',
            'coordinate_features': False, 'grid_domain': 'all rows of 静态数据汇总.xlsx',
            'matrix_order': '[date, station] or [static grid row] for event output',
        }, f, ensure_ascii=False, indent=2)


def export_event_grid_outputs(fold_dir: Path, fold_id: int, model: SpatialANP,
                              bundle: DatasetBundle, feat_bank: np.ndarray,
                              model_cfg: ANPConfig, data_cfg: ExperimentConfig,
                              grid_store: EventGridStore) -> None:
    plt = plotting_pyplot()
    device = next(model.parameters()).device
    latlon = bundle.node_df[['Lat', 'Lon']].to_numpy(np.float32)
    dates = pd.to_datetime(bundle.dates['date'])
    output_dir = fold_dir / 'preds' / 'event_grids'
    figure_dir = fold_dir / 'figs'
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    products = []
    for date, label in EVENT_DAYS:
        idx = np.flatnonzero(dates.eq(pd.Timestamp(date)).to_numpy())
        if idx.size != 1:
            raise ValueError(f'事件日期 {date} 不在输入日期范围内或存在重复。')
        print(f'[fold {fold_id:02d}] predicting full irregular grid: {date} ({label})')
        result = predict_on_grid(model, bundle, feat_bank, latlon, int(idx[0]), model_cfg, device,
                                 data_cfg=data_cfg, grid_store=grid_store)
        # sigma is an internal bulk spread, deliberately excluded from saved results.
        np.savez_compressed(
            output_dir / f'NPZ_fold{fold_id:02d}_grid_{date}.npz',
            fold_id=np.asarray([fold_id], np.int16), date=np.asarray([date]),
            event=np.asarray([label]), static_grid_row_index=result['static_grid_row_index'],
            Lat=result['latlon'][:, 0], Lon=result['latlon'][:, 1], pred=result['wind'],
            exceedance_probability=result['exceedance_probability'], gpd_scale=result['gpd_scale'],
            gpd_shape=result['gpd_shape'], tail_threshold=result['tail_threshold'],
            tail_conditional_mean=result['tail_conditional_mean'],
        )
        products.append((date, label, result['wind']))
        del result
    vmax = max(1.0, max(float(wind.max()) for _, _, wind in products))
    for date, label, wind in products:
        fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
        color = ax.scatter(grid_store.latlon[:, 1], grid_store.latlon[:, 0], c=wind,
                           s=3, marker='s', linewidths=0, cmap='viridis', vmin=0, vmax=vmax,
                           rasterized=True)
        fig.colorbar(color, ax=ax, label='Predicted DMWS (m/s)', shrink=0.88)
        ax.set(xlabel='Longitude (°E)', ylabel='Latitude (°N)',
               title=f'{date} · {label} · Fold {fold_id}')
        ax.set_aspect(1.0 / np.cos(np.deg2rad(float(grid_store.latlon[:, 0].mean()))))
        fig.savefig(figure_dir / f'fold{fold_id:02d}_grid_{date}.png', dpi=300)
        plt.close(fig)



# =============================================================================
# Extreme-risk calibration utilities (E0)
# =============================================================================
def _normal_sf(z: np.ndarray) -> np.ndarray:
    """Vectorized Normal survival function 1-Phi(z), implemented without scipy."""
    z = np.asarray(z, dtype=np.float64)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 - erf_vec(z / math.sqrt(2.0)))


def _logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(p / (1.0 - p))


def _split_obs_mask(bundle: DatasetBundle, split: str) -> np.ndarray:
    node_mask = {
        'Train': bundle.train_node_mask,
        'Val': bundle.val_node_mask,
        'Calib': bundle.calib_node_mask,
        'Test': bundle.test_node_mask,
    }[split]
    return bundle.obs_mask & node_mask[None, :]


def _fit_platt(raw_prob: np.ndarray, y_bin: np.ndarray):
    y_bin = np.asarray(y_bin, dtype=np.int64)
    if y_bin.size == 0 or len(np.unique(y_bin)) < 2:
        return None
    X = _logit_np(raw_prob).reshape(-1, 1)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X, y_bin)
    return clf


def _apply_platt(model, raw_prob: np.ndarray) -> np.ndarray:
    raw_prob = np.asarray(raw_prob, dtype=np.float64)
    if model is None:
        return raw_prob
    return model.predict_proba(_logit_np(raw_prob).reshape(-1, 1))[:, 1]


def _contingency(y_true_bin: np.ndarray, prob: np.ndarray, prob_thr: float) -> Dict[str, float]:
    yt = np.asarray(y_true_bin).astype(bool)
    yp = np.asarray(prob, dtype=np.float64) >= float(prob_thr)
    hits = int(np.sum(yt & yp))
    misses = int(np.sum(yt & ~yp))
    false_alarms = int(np.sum(~yt & yp))
    denom_csi = hits + misses + false_alarms
    denom_pod = hits + misses
    denom_far = hits + false_alarms
    return {
        'CSI': float(hits / denom_csi) if denom_csi > 0 else float('nan'),
        'POD': float(hits / denom_pod) if denom_pod > 0 else float('nan'),
        'FAR': float(false_alarms / denom_far) if denom_far > 0 else float('nan'),
        'Hits': hits,
        'Misses': misses,
        'FalseAlarms': false_alarms,
    }


def _best_prob_threshold(y_bin: np.ndarray, prob: np.ndarray, objective: str = 'CSI') -> float:
    y_bin = np.asarray(y_bin).astype(bool)
    prob = np.asarray(prob, dtype=np.float64)
    if y_bin.size == 0 or len(np.unique(y_bin)) < 2:
        return 0.5
    grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 99)))
    best_t, best_score = 0.5, -np.inf
    for t in grid:
        c = _contingency(y_bin, prob, float(t))
        score = c.get(objective, float('nan'))
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t


def _risk_metric_pack(y_bin: np.ndarray, prob: np.ndarray, decision_thr: float) -> Dict[str, float]:
    y_bin = np.asarray(y_bin).astype(np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    out: Dict[str, float] = {}
    out['Brier'] = float(brier_score_loss(y_bin, prob)) if y_bin.size else float('nan')
    if y_bin.size and len(np.unique(y_bin)) >= 2:
        out['AP'] = float(average_precision_score(y_bin, prob))
        try:
            out['ROC_AUC'] = float(roc_auc_score(y_bin, prob))
        except Exception:
            out['ROC_AUC'] = float('nan')
    else:
        out['AP'] = float('nan')
        out['ROC_AUC'] = float('nan')
    pred = (prob >= float(decision_thr)).astype(np.int64)
    out['Precision'] = float(precision_score(y_bin, pred, zero_division=0))
    out['Recall'] = float(recall_score(y_bin, pred, zero_division=0))
    out['F1'] = float(f1_score(y_bin, pred, zero_division=0))
    out.update(_contingency(y_bin, prob, float(decision_thr)))
    out['DecisionThr'] = float(decision_thr)
    out['EventRate'] = float(np.mean(y_bin)) if y_bin.size else float('nan')
    return out


def evaluate_calibrated_extreme_risk(bundle: DatasetBundle, pred_mu: np.ndarray, pred_sigma: np.ndarray,
                                     out_dir: Path, tag: str = 'E0_gaussian_phaseC') -> pd.DataFrame:
    """Fit Platt calibrators on Calib only and evaluate Val/Test extreme probabilities.

    Risk probability is derived from the Gaussian predictive distribution:
      P(Y > q | context, target) = 1 - Phi((q - mu) / sigma)
    The calibrator is fit only on Calib stations and never sees Test labels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {k: v for k, v in bundle.tail_thresholds.items() if k in ('P90', 'P95')}
    rows: List[Dict[str, float]] = []
    sigma_safe = np.clip(np.asarray(pred_sigma, dtype=np.float64), 1.0e-4, None)
    mu = np.asarray(pred_mu, dtype=np.float64)
    obs = np.asarray(bundle.obs, dtype=np.float64)

    for thr_name, thr in thresholds.items():
        raw_prob_full = _normal_sf((float(thr) - mu) / sigma_safe)
        calib_mask = _split_obs_mask(bundle, 'Calib')
        y_cal = obs[calib_mask]
        p_cal_raw = raw_prob_full[calib_mask]
        y_cal_bin = (y_cal >= float(thr)).astype(np.int64)
        platt = _fit_platt(p_cal_raw, y_cal_bin)
        p_cal = _apply_platt(platt, p_cal_raw)
        t_raw = _best_prob_threshold(y_cal_bin, p_cal_raw, objective='CSI')
        t_cal = _best_prob_threshold(y_cal_bin, p_cal, objective='CSI')

        for split in ['Val', 'Test']:
            m = _split_obs_mask(bundle, split)
            y = obs[m]
            y_bin = (y >= float(thr)).astype(np.int64)
            p_raw = raw_prob_full[m]
            p_calib = _apply_platt(platt, p_raw)
            for mode, prob, dec_thr in [('raw', p_raw, t_raw), ('platt', p_calib, t_cal)]:
                row = {
                    'Tag': tag,
                    'Split': split,
                    'ThresholdName': thr_name,
                    'ThresholdValue': float(thr),
                    'Calibration': mode,
                }
                row.update(_risk_metric_pack(y_bin, prob, dec_thr))
                rows.append(row)
    df = pd.DataFrame(rows)
    path = out_dir / 'cnp_phaseH_calibrated_extreme_risk_metrics.csv'
    df.to_csv(path, index=False)
    print(f'[E0] calibrated extreme-risk metrics written to {path}')
    for _, r in df.iterrows():
        if r['Split'] == 'Test':
            print(f"  [E0 {r['Calibration']} {r['ThresholdName']} Test] "
                  f"Brier={r['Brier']:.4f} AP={r['AP']:.4f} CSI={r['CSI']:.3f} "
                  f"POD={r['POD']:.3f} FAR={r['FAR']:.3f} F1={r['F1']:.3f}")
    return df

# =============================================================================
# Main
# =============================================================================
def main_spatial5fold_cv(cv_cfg: Spatial5FoldCVConfig) -> None:
    print_runtime_environment()
    cv_root = Path(cv_cfg.work_dir)
    cv_cfg.prepare_dirs()
    print('=' * 80)
    print('[spatial5fold] building source aligned dataset bundle')
    print(f'[spatial5fold] output root: {cv_root}')
    print(f'[spatial5fold] enabled baselines: {cv_cfg.enabled_baselines}')
    print('=' * 80)

    source_cfg = ExperimentConfig()
    source_cfg.work_dir = str(cv_root / 'source_dataset') + '/'
    new_source_cache = cv_root / 'source_dataset' / 'cache'
    original_direct_cv_root = Path(
        'res_cnp_direct_dailymax_target_0p1_1979_2019_era5land_spatial10fold_cv'
    )
    original_phase1_cv_root = Path(
        'res_cnp_phaseI_final_no_softidw_prior_0p1_1979_2019_era5land_spatial10fold_cv'
    )
    source_cfg.cache_dir = str(new_source_cache) + '/'
    source_cfg.legacy_cache_dirs = (
        str(original_direct_cv_root / 'source_dataset' / 'cache'),
        str(original_phase1_cv_root / 'source_dataset' / 'cache'),
        str(Path('res_cnp_direct_dailymax_target_gpd_tail_0p1_1979_2019_era5land_spatial10fold_cv') / 'source_dataset' / 'cache'),
    )
    source_cfg.fig_dir = str(cv_root / 'source_dataset' / 'figs') + '/'
    source_cfg.pred_dir = str(cv_root / 'source_dataset' / 'preds') + '/'
    source_cfg.ckpt_dir = str(cv_root / 'source_dataset' / 'ckpt') + '/'
    source_cfg.sensitivity_dir = str(cv_root / 'source_dataset' / 'sens') + '/'
    source_cfg.force_recompute = False
    source_cfg.run_external_baselines = False
    source_cfg.enabled_baselines = tuple()
    source_cfg.prepare_dirs()
    base_bundle = build_dataset_cache(source_cfg)
    print(f'[spatial5fold] source bundle: D={base_bundle.n_days}, N={base_bundle.n_nodes}')

    shared_latlon = base_bundle.node_df[['Lat', 'Lon']].to_numpy(np.float32)
    geometry_cache_candidates = [
        original_direct_cv_root / 'shared_cache',
        original_phase1_cv_root / 'shared_cache',
        cv_root / 'shared_cache',
    ]
    geometry_cache_dir = cv_root / 'shared_cache'
    for candidate in geometry_cache_candidates:
        if (candidate / 'station_geometry_no_residual_inputs.npz').exists():
            geometry_cache_dir = candidate
            break
    shared_pairwise_km, shared_relative_geometry = load_or_build_station_geometry(
        shared_latlon, geometry_cache_dir
    )

    cv_folds, cv_spatial_order = assign_interleaved_spatial_folds_for_nodes(
        base_bundle.node_df, cv_cfg.n_folds, cv_cfg.lat_band_deg
    )
    fold_table = base_bundle.node_df[['source_station_id', 'Lat', 'Lon']].copy()
    fold_table['node_index'] = np.arange(len(fold_table), dtype=np.int32)
    fold_table['cv_fold'] = cv_folds.astype(np.int16)
    fold_table['cv_spatial_order'] = cv_spatial_order.astype(np.int32)
    for test_fold in range(1, cv_cfg.n_folds + 1):
        _, val_mask, test_mask = cv_masks_from_folds(cv_folds, test_fold, cv_cfg.n_folds)
        role = np.full(len(fold_table), 'train', dtype=object)
        role[val_mask] = 'validation'
        role[test_mask] = 'test'
        fold_table[f'role_when_fold{test_fold:02d}_test'] = role
    fold_table_path = cv_root / 'Table_spatial5fold_node_fold_assignment.csv'
    fold_table.to_csv(fold_table_path, index=False, encoding='utf-8-sig')
    print(f'[spatial5fold] fold assignment written: {fold_table_path}')
    print(f'[spatial5fold] fold counts: {dict(pd.Series(cv_folds).value_counts().sort_index())}')

    with open(cv_root / 'spatial5fold_config.json', 'w', encoding='utf-8') as f:
        json.dump({
            'cv_config': asdict(cv_cfg),
            'source_data_config': asdict(source_cfg),
            'fold_assignment_table': str(fold_table_path),
            'note': (
                'Each fold uses one dispersed fold as test, the next fold as validation, '
                'and the remaining three folds as train. Every fold is trained from scratch. '
                'The neural model directly predicts station-observed daily maximum wind speed. '
                'No Stage-1 background is fitted and no background-residual sum is used. '
                'Observations and observation-derived residuals are not supplied as target-day '
                'context values or historical input features. Station embeddings are retained. '
                'Latitude/longitude are used only for geometry, data matching and output coordinates; '
                'raw, standardized and Fourier coordinate features are removed. '
                'The predictive distribution uses a bulk wind component, an exceedance '
                'probability, and a generalized Pareto distribution for threshold excesses. '
                'External baselines are disabled.'
            ),
        }, f, ensure_ascii=False, indent=2)

    for fold_id in cv_cfg.run_folds:
        fold_start = time.time()
        print('\n' + '=' * 80)
        print(f'[spatial5fold] START fold {fold_id:02d}/{cv_cfg.n_folds}')
        print('=' * 80)

        fold_dir = cv_root / f'fold{fold_id:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_mask, val_mask, test_mask = cv_masks_from_folds(cv_folds, int(fold_id), cv_cfg.n_folds)
        print(
            f'[spatial5fold fold {fold_id:02d}] nodes: '
            f'train={int(train_mask.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}'
        )

        fold_data_cfg = ExperimentConfig()
        fold_data_cfg.work_dir = str(fold_dir) + '/'
        fold_data_cfg.cache_dir = str(fold_dir / 'cache') + '/'
        fold_data_cfg.fig_dir = str(fold_dir / 'figs') + '/'
        fold_data_cfg.pred_dir = str(fold_dir / 'preds_baselines') + '/'
        fold_data_cfg.ckpt_dir = str(fold_dir / 'ckpt_baselines') + '/'
        fold_data_cfg.sensitivity_dir = str(fold_dir / 'sens') + '/'
        fold_data_cfg.force_recompute = True
        fold_data_cfg.run_external_baselines = False
        fold_data_cfg.enabled_baselines = tuple()
        fold_data_cfg.prepare_dirs()

        fold_bundle = rebuild_bundle_for_cv_fold(
            base_bundle=base_bundle,
            data_cfg=fold_data_cfg,
            train_node_mask=train_mask,
            val_node_mask=val_mask,
            test_node_mask=test_mask,
            fold_id=int(fold_id),
            cv_folds=cv_folds,
            cv_spatial_order=cv_spatial_order,
        )

        model_cfg = ANPConfig()
        model_cfg.work_dir = str(fold_dir) + '/'
        model_cfg.cache_dir = str(fold_dir / 'cache_anp') + '/'
        model_cfg.ckpt_dir = str(fold_dir / 'ckpt') + '/'
        model_cfg.pred_dir = str(fold_dir / 'preds') + '/'
        model_cfg.force_recompute = True
        model_cfg.load_existing_checkpoint = False
        model_cfg.legacy_checkpoint_paths = tuple()
        model_cfg.tail_quantile = float(fold_data_cfg.tail_quantile)
        model_cfg.tail_threshold = float(fold_bundle.train_tail_threshold)
        model_cfg.seed = int(model_cfg.seed + int(fold_id) - 1)
        assert model_cfg.anchor_mode == 'no_anchor_prior'
        assert model_cfg.use_anchor is False
        assert float(model_cfg.anchor_warm_start_weight) == 0.0
        assert int(model_cfg.anchor_warm_start_epochs) == 0
        assert float(model_cfg.correction_reg_weight) == 0.0
        assert model_cfg.use_residual_memory is False
        assert model_cfg.use_station_embedding is True
        assert model_cfg.train_in_obs_space is True
        assert np.isfinite(model_cfg.tail_threshold) and model_cfg.tail_threshold > 0.0
        assert fold_data_cfg.direct_dailymax_target is True
        model_cfg.prepare_dirs()
        with open(fold_dir / f'fold{fold_id:02d}_anp_config.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(model_cfg), f, ensure_ascii=False, indent=2)

        model, feat_bank, pred_mu, pred_sigma, tail_outputs, diag = train_anp(
            fold_bundle, model_cfg,
            pairwise_km=shared_pairwise_km,
            relative_geometry=shared_relative_geometry,
        )
        with open(fold_dir / f'fold{fold_id:02d}_anp_diagnostics.json', 'w', encoding='utf-8') as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)

        all_predictions: Dict[str, np.ndarray] = {
            'Direct_daily_max_wind_GPD_tail_NN': pred_mu,
        }
        baseline_preds: Dict[str, np.ndarray] = {}
        print(f'[spatial5fold fold {fold_id:02d}] external baselines disabled for this experiment.')

        test_idx = np.where(fold_bundle.test_node_mask)[0].astype(np.int32)
        valid_test = fold_bundle.obs_mask[:, test_idx]
        observed_test = fold_bundle.obs[:, test_idx][valid_test].astype(np.float64)
        probability_test = (
            tail_outputs['exceedance_probability'][:, test_idx][valid_test].astype(np.float64)
        )
        tail_mean_test = (
            tail_outputs['tail_conditional_mean'][:, test_idx][valid_test].astype(np.float64)
        )
        event_test = observed_test > float(model_cfg.tail_threshold)
        tail_error = tail_mean_test[event_test] - observed_test[event_test]
        risk_row = {
            'fold': int(fold_id),
            'tail_quantile': float(model_cfg.tail_quantile),
            'tail_threshold_m_per_s': float(model_cfg.tail_threshold),
            'gpd_shape': float(tail_outputs['gpd_shape'][0]),
            'valid_test_count': int(valid_test.sum()),
            'observed_exceedance_count': int(event_test.sum()),
            'observed_exceedance_rate': float(event_test.mean()),
            'mean_predicted_exceedance_probability': float(probability_test.mean()),
            'brier_score': float(brier_score_loss(event_test.astype(np.int64), probability_test)),
            'average_precision': float(average_precision_score(event_test.astype(np.int64), probability_test)),
            'roc_auc': (
                float(roc_auc_score(event_test.astype(np.int64), probability_test))
                if len(np.unique(event_test)) == 2 else float('nan')
            ),
            'tail_conditional_mean_rmse_m_per_s': (
                float(np.sqrt(np.mean(tail_error ** 2))) if tail_error.size else float('nan')
            ),
            'tail_conditional_mean_bias_m_per_s': (
                float(np.mean(tail_error)) if tail_error.size else float('nan')
            ),
        }
        pd.DataFrame([risk_row]).to_csv(
            fold_dir / f'Table_fold{fold_id:02d}_gpd_tail_diagnostics.csv',
            index=False,
            encoding='utf-8-sig',
        )

        export_fold_outputs(fold_dir, int(fold_id), fold_bundle, all_predictions, cv_cfg, tail_outputs)
        plot_test_scatter(fold_dir, int(fold_id), fold_bundle, pred_mu)
        save_fold_inference_metadata(fold_dir, fold_bundle, fold_data_cfg, model_cfg)
        fold_elapsed = time.time() - fold_start
        print(f'[spatial5fold] DONE fold {fold_id:02d}, elapsed {format_elapsed_time(fold_elapsed)}')

        del model, feat_bank, fold_bundle, pred_mu, pred_sigma, tail_outputs, baseline_preds, all_predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\n' + '=' * 80)
    print('[spatial5fold] all requested folds completed')
    print(f'[spatial5fold] outputs are under: {cv_root}')
    print('=' * 80)


def main() -> None:
    cv_cfg = Spatial5FoldCVConfig()
    cv_cfg.prepare_dirs()
    log_path = Path(cv_cfg.work_dir) / 'spatial5fold_run_log.txt'
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with open(log_path, 'w', encoding='utf-8') as log_f:
        sys.stdout = TeeLogger(old_stdout, log_f)
        sys.stderr = TeeLogger(old_stderr, log_f)
        try:
            main_spatial5fold_cv(cv_cfg)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    print(f'[spatial5fold] run log written to {log_path}')


if __name__ == '__main__':
    _program_start_time = time.time()
    _completed_normally = False
    try:
        main()
        _completed_normally = True
    finally:
        _elapsed = time.time() - _program_start_time
        _status = 'completed' if _completed_normally else 'stopped before normal completion'
        print('=' * 80)
        print(f'[runtime] total elapsed time ({_status}): {format_elapsed_time(_elapsed)} ({_elapsed:.1f} seconds)')
        print('=' * 80)
