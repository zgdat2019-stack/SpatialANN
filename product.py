#!/usr/bin/env python3
"""fold01 inference only: one shared preparation job, ten year-group workers."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.ndimage import maximum_filter1d
import torch

import model_source as core

YEARS = list(range(1979, 2020))
GROUPS = [list(map(int, g)) for g in np.array_split(YEARS, 10)]
DATES = pd.date_range('1979-01-01', '2019-12-31', freq='D')
ORIGIN = pd.Timestamp('1979-01-01')
PHASE = np.arange(len(DATES), dtype=np.float32)
SINE = np.sin(2.0 * np.pi * PHASE / 366.0)
COSINE = np.cos(2.0 * np.pi * PHASE / 366.0)
HERE = Path(__file__).resolve().parent
LOG = logging.getLogger('s56')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stat_record(path):
    path = Path(path).resolve()
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns)


@contextmanager
def exclusive_lock(path):
    # Kernel locks are released on exit, including process/node failure.
    with Path(path).open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process owns this task/cache: {path}') from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Progress:
    def __init__(self, label, total=0):
        self.label, self.total, self.done = label, total, 0
        self.start = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def heartbeat(self):
        while not self.stop.wait(20 * 60):
            self.report()

    def report(self):
        elapsed = time.monotonic() - self.start
        eta = elapsed * (self.total - self.done) / self.done if self.done else float('nan')
        LOG.info('%s: %d/%d; elapsed=%.1f min; ETA=%.1f min',
                 self.label, self.done, self.total, elapsed / 60, eta / 60)

    def __enter__(self):
        LOG.info('Start: %s', self.label)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.report()


def setup(args):
    root = Path(args.main_result).expanduser().resolve()
    fold = root / 'fold01'
    meta_path = fold / 'ckpt/inference_metadata.json'
    meta = read_json(meta_path)
    cfg = core.ANPConfig(**meta['model_config'])
    data_cfg = core.ExperimentConfig(**meta['data_config'])
    require(list(data_cfg.years) == YEARS, 'Expected saved 1979-2019 configuration')
    require(meta.get('coordinate_features') is False and not cfg.use_residual_memory,
            'Expected no-coordinate/no-residual-memory model')
    require(not cfg.use_anchor and cfg.anchor_mode == 'no_anchor_prior' and
            cfg.no_bake_unseen_embeddings, 'Expected original no-anchor/no-bake configuration')
    require(int(data_cfg.climate_hist_len) == 1095, 'Expected saved 1095-day history')
    require(not {'Lat', 'Lon'} & set(meta['const_feature_names']), 'Coordinates in feature names')
    # Relative saved paths belong to the project working directory, not to
    # this bundled model-source module's location under code/.
    data_root = Path(args.input_dir or data_cfg.input_dir).expanduser().resolve()
    data_cfg.input_dir = str(data_root)
    checkpoint = fold / 'ckpt' / (core.CODE_VERSION_ANP + '.pt')
    source_npz = Path(args.station_cache_dir).expanduser().resolve() / 'aligned_raw_no_coord_v1.npz'
    source_pkl = source_npz.with_suffix('.pkl')
    station_path = fold / 'Table_fold01_node_split_metadata.csv'
    sources = [data_root / '静态数据汇总.xlsx', source_npz, source_pkl]
    sources += [data_root / f'{v}_{y}.csv' for y in YEARS for v in data_cfg.dynamic_vars]
    identity = dict(version=1, fold=1, years=YEARS, source_records=[stat_record(p) for p in sources],
        sha256={str(p.resolve()): sha256(p) for p in
                [HERE / 'model_source.py', HERE / 'product.py', checkpoint, meta_path, station_path]},
        numpy=np.__version__, torch=torch.__version__, pandas=pd.__version__)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_root = (Path(args.cache_dir).expanduser().resolve() if args.cache_dir else
                  HERE / 'cache')
    cache = cache_root / signature[:16]
    return SimpleNamespace(root=root, fold=fold, meta=meta, cfg=cfg, data_cfg=data_cfg,
        data_root=data_root, checkpoint=checkpoint, source_npz=source_npz, source_pkl=source_pkl,
        station_path=station_path, identity=identity, signature=signature, cache=cache,
        output=Path(args.output_dir).expanduser().resolve(), tile=args.tile_size)


def check_source_identity(s):
    require([stat_record(p['path']) for p in s.identity['source_records']] == s.identity['source_records'],
            'Input files changed during execution')
    require(all(sha256(p) == digest for p, digest in s.identity['sha256'].items()),
            'Model/code/metadata changed during execution')


def grid_axes(latlon):
    """Store exact source coordinates, without snapping, rounding or resampling.

    A compact Cartesian layout uses the actual unique axes, regardless of
    their spacing. General point grids (or duplicate rows) use time/point
    storage to avoid an enormous sparse Cartesian product or dropped rows.
    """
    ll = np.asarray(latlon, np.float64)
    require(ll.ndim == 2 and ll.shape[1] == 2 and len(ll) and np.isfinite(ll).all(),
            'Invalid static grid coordinates')
    lat, iy = np.unique(ll[:, 0], return_inverse=True)
    lon, ix = np.unique(ll[:, 1], return_inverse=True)
    cells = int(len(lat)) * int(len(lon))
    unique = len(np.unique(iy * len(lon) + ix)) == len(ll)
    compact = cells <= max(16, 8 * len(ll)) and cells <= 4_000_000
    if unique and compact:
        LOG.info('Grid storage=rectilinear; %d actual points, %d lat x %d lon; exact source coordinates',
                 len(ll), len(lat), len(lon))
        return dict(layout=np.asarray('rectilinear'), lat=lat, lon=lon, iy=iy, ix=ix,
                    point_lat=ll[:, 0], point_lon=ll[:, 1])
    LOG.info('Grid storage=point; %d actual rows retained in original order; exact source coordinates', len(ll))
    return dict(layout=np.asarray('point'), lat=ll[:, 0], lon=ll[:, 1],
                point_lat=ll[:, 0], point_lon=ll[:, 1])


def point_layout(static):
    return str(static.get('layout', 'rectilinear')) == 'point'


def build_static(s):
    static = pd.read_excel(s.data_root / '静态数据汇总.xlsx').replace(-999, np.nan).reset_index(drop=True)
    ll_original = static[['Lat', 'Lon']].to_numpy(np.float64)
    ll = ll_original.astype(np.float32)
    require(len(ll) > 0 and np.isfinite(ll).all(), 'Invalid static grid coordinates')
    for name in static:
        if name not in ('Lat', 'Lon') and pd.api.types.is_numeric_dtype(static[name]):
            static[name] = static[name].fillna(static[name].median())
    raw = static[s.meta['const_feature_names']].to_numpy(np.float32)
    scaled = core.scale_2d_with_stats(raw, s.meta['scaler_stats']['static_const'], 'grid static')
    require(s.meta['year_feature_names'] in ([], ['BH_year']), 'Unknown annual feature layout')
    if s.meta['year_feature_names']:
        yearly = core.build_bh_yearly_features(static, YEARS, 'BH_year')[0]
    else:
        yearly = np.empty((len(YEARS), len(ll), 0), np.float32)
    axes = grid_axes(ll_original)
    np.savez(s.cache / 'static.npz', latlon=ll, static_raw=raw, static_scaled=scaled,
             yearly=yearly, **axes)
    return ll


def build_references(s):
    """Rebuild only training-reference features with SAVED scalers, no fitting."""
    with s.source_pkl.open('rb') as f:
        raw_meta = pickle.load(f)
    dates = raw_meta['dates'].reset_index(drop=True)
    require(pd.DatetimeIndex(pd.to_datetime(dates['date'])).equals(DATES), 'Source-cache dates do not match full calendar')
    node = raw_meta['node_df'].reset_index(drop=True)
    station = pd.read_csv(s.station_path, dtype={'source_station_id': str})
    require(np.array_equal(station.node_index, np.arange(len(node))) and
            np.array_equal(station.source_station_id.to_numpy(), node.source_station_id.astype(str).to_numpy()),
            'Saved fold station IDs/order differ from raw cache')
    require(np.allclose(station[['Lat', 'Lon']], node[['Lat', 'Lon']], atol=1e-5, rtol=0), 'Station coordinates differ')
    require(station.split_final.isin(['train', 'val', 'test']).all(), 'Unknown saved station roles')
    refs = np.flatnonzero(station.split_final.eq('train')).astype(np.int64)
    require(len(refs) > 0, 'No training reference stations')
    require(list(raw_meta['dyn_feature_names']) == list(s.meta['dyn_feature_names']), 'Dynamic variable order differs')
    raw_const_names = list(raw_meta['const_feature_names'])
    keep = [raw_const_names.index(n) for n in s.meta['const_feature_names']]
    require(list(raw_meta['year_feature_names']) == list(s.meta['year_feature_names']), 'Annual variable order differs')
    with np.load(s.source_npz, allow_pickle=False) as z:
        dyn = z['daily_dyn'][:, refs].astype(np.float32)
        const = z['static_const'][refs][:, keep].astype(np.float32)
        yearly = z['yearly_feats'][:, refs].astype(np.float32)
    nfeat = s.meta['day_feat_dim'] + s.meta['climate_feat_dim']
    bank = np.lib.format.open_memmap(s.cache / 'reference_features.tmp.npy', mode='w+',
                                    dtype=np.float32, shape=(len(DATES), len(refs), nfeat))
    doy = DATES.dayofyear.to_numpy(np.float32)
    date_feats = np.stack([np.sin(2 * np.pi * doy / 366.), np.cos(2 * np.pi * doy / 366.)], axis=-1)
    yi = DATES.year.to_numpy() - YEARS[0]
    with Progress('reference features', len(refs)) as progress:
        for left in range(0, len(refs), 32):
            right = min(left + 32, len(refs))
            C = right - left
            stats = s.meta['scaler_stats']
            ds = core.scale_3d_with_stats(dyn[:, left:right], stats['daily_dyn'], 'reference dynamics')
            st = core.scale_2d_with_stats(const[left:right], stats['static_const'], 'reference static')
            raw = np.concatenate([np.broadcast_to(const[None, left:right], (len(DATES), C, const.shape[1])),
                yearly[yi, left:right], dyn[:, left:right], np.broadcast_to(date_feats[:, None], (len(DATES), C, 2))], axis=-1)
            day = core.scale_3d_with_stats(raw, stats['day_features'], 'reference day')
            b = SimpleNamespace(node_df=node.iloc[refs[left:right]], feature_names=s.meta['feature_names'],
                const_feature_names=s.meta['const_feature_names'], daily_dyn_scaled=ds, day_features_scaled=day,
                climate_vectors=None)
            if s.cfg.use_climate:
                climate = core.precompute_climate_vectors(ds, st, s.data_cfg.climate_hist_len)
                b.climate_vectors = core.scale_3d_with_stats(climate, stats['climate_vectors'], 'reference climate')
            feat, _, df, cf = core.build_point_feature_bank(b, s.cfg)
            require((df, cf) == (s.meta['day_feat_dim'], s.meta['climate_feat_dim']), 'Reference feature dimensions differ')
            bank[:, left:right] = feat
            progress.done = right
    bank.flush()
    del bank
    os.replace(s.cache / 'reference_features.tmp.npy', s.cache / 'reference_features.npy')
    np.savez(s.cache / 'references.npz', indices=refs, latlon=node[['Lat', 'Lon']].to_numpy(np.float32), n_nodes=len(node))


def seeded_cumsum(values, seed, dtype):
    # Starting with the exact previous-year accumulator preserves the training
    # code's global sequential float32/float64 cumulative-sum convention.
    values = np.asarray(values, dtype=dtype)
    return np.cumsum(np.concatenate([np.asarray(seed, dtype=dtype)[None], values], axis=0), axis=0, dtype=dtype)


def moments(x, start):
    return (x, x ** 2, SINE[start:start + len(x), None, None] * x,
            COSINE[start:start + len(x), None, None] * x)


def build_year_cache(s, year, ll):
    marker = s.cache / f'ready_{year}.json'
    if marker.exists():
        LOG.info('Reuse prepared year: %d', year)
        return
    G, K = len(ll), len(s.data_cfg.dynamic_vars)
    date_rows = np.flatnonzero(DATES.year == year)
    D, start = len(date_rows), int(date_rows[0])
    target = np.lib.format.open_memmap(s.cache / f'dyn_{year}.tmp.npy', mode='w+',
                                     dtype=np.float32, shape=(D, G, K))
    stats = s.meta['scaler_stats']['daily_dyn']
    mean, sd = np.array(stats['mean'], np.float32), np.array(stats['std'], np.float32)
    for k, var in enumerate(s.data_cfg.dynamic_vars):
        path = s.data_root / f'{var}_{year}.csv'
        LOG.info('Align once: %s', path.name)
        df = core.read_dynamic_csv_robust(path)
        columns = df.columns.get_indexer([f'p{i}' for i in range(1, D + 1)])
        require(np.all(columns >= 0), f'Missing daily columns in {path}')
        idx, _ = core.nearest_match_indices(df[['Lat', 'Lon']].to_numpy(np.float64), ll,
            s.data_cfg.coord_match_max_km, f'product grid->{var}_{year}')
        for a in range(0, G, 2048):
            b = min(a + 2048, G)
            raw = df.iloc[idx[a:b], columns].to_numpy(np.float32).T
            raw[~np.isfinite(raw)] = mean[k]
            target[:, a:b, k] = raw
        del df
    target.flush()
    os.replace(s.cache / f'dyn_{year}.tmp.npy', s.cache / f'dyn_{year}.npy')
    seeds = np.load(s.cache / f'prefix32_{year}.npy', mmap_mode='r')
    seed64 = np.load(s.cache / f'prefix64_{year}.npy', mmap_mode='r')
    next32 = np.lib.format.open_memmap(s.cache / f'prefix32_{year + 1}.tmp.npy', mode='w+',
                                      dtype=np.float32, shape=(4, G, K))
    next64 = np.lib.format.open_memmap(s.cache / f'prefix64_{year + 1}.tmp.npy', mode='w+',
                                      dtype=np.float64, shape=(G, K))
    for a in range(0, G, s.tile):
        b = min(a + s.tile, G)
        x = core.scale_3d_with_stats(target[:, a:b], stats, 'prefix dynamics')
        for j, value in enumerate(moments(x, start)):
            next32[j, a:b] = seeded_cumsum(value, seeds[j, a:b], np.float32)[-1]
        next64[a:b] = seeded_cumsum(x, seed64[a:b], np.float64)[-1]
    next32.flush()
    next64.flush()
    del target, next32, next64, seeds, seed64
    for kind in ('32', '64'):
        os.replace(s.cache / f'prefix{kind}_{year + 1}.tmp.npy', s.cache / f'prefix{kind}_{year + 1}.npy')
    write_json(marker, dict(signature=s.signature, year=year, shape=[D, G, K]))


def prepare(s):
    s.cache.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(s.cache / 'prepare.lock'):
        if (s.cache / 'READY.json').exists():
            LOG.info('Shared cache is complete: %s', s.cache)
            return
        write_json(s.cache / 'identity.json', s.identity)
        with Progress('prepare all input years', len(YEARS)) as progress:
            ll = build_static(s)
            build_references(s)
            K = len(s.data_cfg.dynamic_vars)
            for kind, shape, dtype in [('32', (4, len(ll), K), np.float32), ('64', (len(ll), K), np.float64)]:
                p = s.cache / f'prefix{kind}_{YEARS[0]}.npy'
                if not p.exists():
                    tmp = p.with_suffix('.tmp.npy')
                    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=dtype, shape=shape)
                    out[:] = 0
                    out.flush()
                    del out
                    os.replace(tmp, p)
            for year in YEARS:
                build_year_cache(s, year, ll)
                progress.done += 1
            check_source_identity(s)
            write_json(s.cache / 'READY.json', dict(signature=s.signature, years=YEARS, n_grid=len(ll)))
        LOG.info('Shared preparation complete: %s', s.cache)


def year_features(s, static, year, left, right):
    """All days of one year / one grid tile; no repeated 1095-day windows.

    Rolling sums use the same global float32 prefixes as model training.
    Short-term sums use float64. Work is linear in loaded days per grid tile.
    """
    rows = np.flatnonzero(DATES.year == year)
    start, stop = int(rows[0]), int(rows[-1]) + 1
    windows = [int(w) for w in s.cfg.temporal_roll_windows if int(w) > 1]
    hist = max([2] + windows + ([s.data_cfg.climate_hist_len] if s.cfg.use_climate else []))
    first = max(0, start - hist + 1)
    first_year = int(DATES[first].year)
    ext_start = int(np.flatnonzero(DATES.year == first_year)[0])
    blocks = [np.asarray(np.load(s.cache / f'dyn_{y}.npy', mmap_mode='r')[:, left:right])
              for y in range(first_year, year + 1)]
    raw_series = np.concatenate(blocks, axis=0)
    del blocks
    x = core.scale_3d_with_stats(raw_series, s.meta['scaler_stats']['daily_dyn'], 'product dynamics')
    take = np.arange(start, stop) - ext_start
    current = x[take]
    past = x[np.maximum(0, np.arange(start, stop) - 1) - ext_start]
    st_raw = static['static_raw'][left:right]
    st_scaled = static['static_scaled'][left:right]
    yf = static['yearly'][year - YEARS[0], left:right]
    doy = DATES[rows].dayofyear.to_numpy(np.float32)
    date_feats = np.stack([np.sin(2 * np.pi * doy / 366.), np.cos(2 * np.pi * doy / 366.)], axis=-1)
    stats = s.meta['scaler_stats']
    day_stats = stats['day_features']
    D, G, K = current.shape
    raw = np.concatenate([np.broadcast_to(st_raw, (D,) + st_raw.shape), np.broadcast_to(yf, (D,) + yf.shape),
        raw_series[take], np.broadcast_to(date_feats[:, None], (D, G, 2))], axis=-1)
    day = core.scale_3d_with_stats(raw, day_stats, 'product day')
    parts = [day]
    if s.cfg.use_temporal_features:
        seeds = np.load(s.cache / f'prefix64_{first_year}.npy', mmap_mode='r')[left:right]
        prefix = seeded_cumsum(x, seeds, np.float64)
        if s.cfg.temporal_include_lag1:
            parts.append(past)
        rolling = {}
        for w in windows:
            lo = np.maximum(0, np.arange(start, stop) - w + 1)
            avg = ((prefix[take + 1] - prefix[lo - ext_start]) /
                   (np.arange(start, stop) - lo + 1)[:, None, None]).astype(np.float32)
            rolling[w] = avg
            parts.append(avg)
        if s.cfg.temporal_include_delta1:
            parts.append(current - past)
        if s.cfg.temporal_include_anomaly_to_roll:
            parts.extend(current - avg for avg in rolling.values())
        del prefix
    if s.cfg.use_climate:
        seeds = np.load(s.cache / f'prefix32_{first_year}.npy', mmap_mode='r')[:, left:right]
        lo = np.maximum(0, np.arange(start, stop) - s.data_cfg.climate_hist_len + 1)
        lengths = (np.arange(start, stop) - lo + 1).astype(np.float32)[:, None, None]
        values = []
        for j, value in enumerate(moments(x, ext_start)):
            prefix = seeded_cumsum(value, seeds[j], np.float32)
            # Training's rolling_sum avoids subtracting zero for initial windows.
            total = prefix[take + 1].copy()
            valid = lo > 0
            total[valid] -= prefix[(lo - ext_start)[valid]]
            values.append(total / lengths)
        mean, mean2, sin_proj, cos_proj = values
        width = min(30, s.data_cfg.climate_hist_len)
        recent = maximum_filter1d(x, size=width, axis=0, origin=(width - 1) // 2, mode='nearest')[take]
        climate = np.concatenate([mean, np.sqrt(np.maximum(mean2 - mean ** 2, 1e-6)), recent,
            np.sqrt(sin_proj ** 2 + cos_proj ** 2 + 1e-8), np.broadcast_to(st_scaled, (D,) + st_scaled.shape)], axis=-1)
        parts.append(core.scale_3d_with_stats(climate, stats['climate_vectors'], 'product climate'))
    feat = np.concatenate(parts, axis=-1).astype(np.float32)
    require(feat.shape == (D, G, s.meta['day_feat_dim'] + s.meta['climate_feat_dim']) and np.isfinite(feat).all(),
            'Invalid product feature shape/values')
    return feat


def month_path(s, year, month):
    return s.output / f'MWS03_dailymax_wind_0p1deg_{year}{month:02d}.nc'


def completed_month(s, static, year, month, deep=False):
    path = month_path(s, year, month)
    if not path.exists():
        return False
    days = DATES[(DATES.year == year) & (DATES.month == month)]
    try:
        with Dataset(path) as nc:
            require(getattr(nc, 'product_signature', None) == s.signature and
                    getattr(nc, 'complete', 0) == 1, f'Different or incomplete product: {path}')
            is_point = point_layout(static)
            expected = ((len(days), len(static['latlon'])) if is_point else
                        (len(days), len(static['lat']), len(static['lon'])))
            require(nc['wind'].shape == expected and getattr(nc, 'grid_layout', None) == str(static['layout']),
                    f'Wrong shape/layout: {path}')
            require(np.array_equal(nc['time'][:], (days - ORIGIN).days) and
                    np.array_equal(nc['lat'][:], static['lat']) and
                    np.array_equal(nc['lon'][:], static['lon']), f'Wrong coordinates: {path}')
            if is_point:
                require(np.array_equal(nc['point'][:], np.arange(len(static['latlon']))), f'Wrong grid row order: {path}')
            if deep:
                for d in range(len(days)):
                    field = nc['wind'][d].filled(np.nan)
                    point = field if is_point else field[static['iy'], static['ix']]
                    require(np.isfinite(point).all() and (point >= 0).all(), f'Invalid values: {path}')
    except Exception as exc:
        raise RuntimeError(f'Cannot reuse existing file {path}: {exc}') from exc
    return True


def write_month(s, static, year, month, values):
    path = month_path(s, year, month)
    tmp = s.cache / (path.name + '.tmp.nc')
    dates = DATES[(DATES.year == year) & (DATES.month == month)]
    day_of_year = dates.dayofyear.to_numpy() - 1
    is_point = point_layout(static)
    ny, nx = len(static['lat']), len(static['lon'])
    fill = np.float32(-9999.)
    with Dataset(tmp, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', len(dates))
        if is_point:
            nc.createDimension('point', len(static['latlon']))
            row = nc.createVariable('point', 'i4', ('point',))
            row.long_name = 'Zero-based row index in static input table'
            row[:] = np.arange(len(static['latlon']), dtype=np.int32)
        else:
            nc.createDimension('lat', ny)
            nc.createDimension('lon', nx)
        t = nc.createVariable('time', 'i4', ('time',))
        t.units = 'days since 1979-01-01 00:00:00'
        t.calendar = 'proleptic_gregorian'
        t.standard_name, t.axis = 'time', 'T'
        t[:] = (dates - ORIGIN).days
        for name, units, axis in [('lat', 'degrees_north', 'Y'), ('lon', 'degrees_east', 'X')]:
            v = nc.createVariable(name, 'f8', ('point',) if is_point else (name,))
            v.units = units
            if not is_point:
                v.axis = axis
            v.standard_name = 'latitude' if name == 'lat' else 'longitude'
            v[:] = static[name]
        dims = ('time', 'point') if is_point else ('time', 'lat', 'lon')
        chunks = (1, min(8192, len(static['latlon']))) if is_point else (1, min(128, ny), min(128, nx))
        wind = nc.createVariable('wind', 'f4', dims, zlib=True,
            complevel=4, shuffle=True, fill_value=fill, chunksizes=chunks)
        wind.units = 'm s-1'
        wind.long_name = 'Daily maximum wind speed at 10 m'
        wind.standard_name = 'wind_speed'
        wind.cell_methods = 'time: maximum'
        wind.coordinates = 'time lat lon'
        for i, d in enumerate(day_of_year):
            point = np.asarray(values[d], np.float32)
            require(np.isfinite(point).all() and (point >= 0).all(), f'Invalid prediction for {dates[i]}')
            if is_point:
                wind[i] = point
            else:
                field = np.full((ny, nx), fill, np.float32)
                field[static['iy'], static['ix']] = point
                wind[i] = field
        nc.Conventions = 'CF-1.8'
        nc.title = 'China daily maximum wind speed, fold01 SpatialANN reconstruction'
        nc.model_fold = 1
        nc.model_name = 'Direct_daily_max_wind_GPD_tail_NN'
        nc.product_signature = s.signature
        nc.model_checkpoint_sha256 = s.identity['sha256'][str(s.checkpoint.resolve())]
        nc.grid_layout = str(static['layout'])
        nc.grid_definition = ('All rows of 静态数据汇总.xlsx, exact original coordinates; '
            + ('point dimension follows original row order' if is_point else 'unlisted coordinate pairs are missing'))
        nc.target_station_embedding = 'zero, same as original continuous-grid inference'
        nc.reference_stations = 'fold01 training stations only; exogenous features, no observations as inputs'
        nc.history = datetime.now(timezone.utc).isoformat() + ' fold01 checkpoint inference'
        nc.complete = 1
    os.replace(tmp, path)
    LOG.info('Saved: %s', path)


def load_model(s, device):
    with np.load(s.cache / 'references.npz', allow_pickle=False) as z:
        refs, ll, N = z['indices'], z['latlon'], int(z['n_nodes'])
    train = np.zeros(N, bool)
    train[refs] = True
    model = core.SpatialANP(s.meta['day_feat_dim'], s.meta['climate_feat_dim'], s.cfg,
        n_nodes=N, latlon_np=ll, train_node_mask_np=train)
    state = torch.load(s.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    core.zero_unseen_station_embeddings(model, train)
    require(np.isclose(float(model.tail_threshold), s.cfg.tail_threshold), 'Checkpoint tail threshold differs')
    return model.to(device).eval(), refs, ll


def reuse_grid_geometry(model):
    """Reuse the original geometry calculations across dates of one tile.

    These two biases depend only on fixed coordinates and an all-valid
    reference mask. Their original functions are called once per spatial
    subchunk. No weights, feature values, attention formula or outputs change.
    """
    directional, local = {}, {}
    if model.use_dir_bias:
        original_dir = model.dir_bias.forward

        def cached_dir(target, context, geometry=None):
            key = (target.data_ptr(), tuple(target.shape))
            if key not in directional:
                directional[key] = original_dir(target, context, geometry)
            return directional[key]
        model.dir_bias.forward = cached_dir
    original_local = model._build_local_attn_bias

    def cached_local(target, context, valid, d_km=None):
        key = (target.data_ptr(), tuple(target.shape))
        if key not in local:
            local[key] = original_local(target, context, valid, d_km=d_km)
        return local[key]
    model._build_local_attn_bias = cached_local

    def clear():
        directional.clear()
        local.clear()
    return clear


def run(s, args):
    require((s.cache / 'READY.json').is_file(), f'Preparation is not complete: {s.cache}')
    years = GROUPS[args.task_id - 1]
    static = dict(np.load(s.cache / 'static.npz', allow_pickle=False))
    s.output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(s.cache / f'task{args.task_id:02d}.lock'):
        pending = {y: [m for m in range(1, 13) if not completed_month(s, static, y, m)] for y in years}
        if not any(pending.values()):
            LOG.info('Task %d already complete: %s', args.task_id, years)
            return
        device = torch.device(args.device)
        require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable in this job')
        model, refs, ll = load_model(s, device)
        clear_geometry = reuse_grid_geometry(model)
        bank = np.load(s.cache / 'reference_features.npy', mmap_mode='r')
        ctx_pos = torch.from_numpy(ll[refs][None]).to(device)
        ctx_idx = torch.from_numpy(refs[None]).to(device)
        ctx_mask = torch.ones((1, len(refs)), device=device)
        G = len(static['latlon'])
        total = sum(int(np.isin(DATES[DATES.year == y].month, months).sum()) * G for y, months in pending.items())
        with torch.inference_mode(), Progress(f'task {args.task_id}: {years}, device={device}', total) as progress:
            for year in years:
                months = pending[year]
                if not months:
                    continue
                rows = np.flatnonzero(DATES.year == year)
                needed = np.flatnonzero(np.isin(DATES[rows].month, months))
                LOG.info('Year %d, months=%s, grid points=%d', year, months, G)
                # Encode reference stations once/day; keep this year's small
                # encodings on the inference device and reuse for every tile.
                encoded = []
                for d in needed:
                    feat = torch.from_numpy(np.array(bank[rows[d]][None], copy=True)).to(device)
                    encoded.append(model.encode_context(feat, ctx_mask))
                predicted_path = s.cache / f'prediction_work_task{args.task_id:02d}_{year}.npy'
                work_marker = predicted_path.with_suffix('.json')
                resume = 0
                if work_marker.exists() and predicted_path.exists():
                    saved = read_json(work_marker)
                    if (saved.get('signature') == s.signature and saved.get('shape') == [len(rows), G]
                            and set(months) <= set(saved.get('months', []))):
                        resume = int(saved['completed_points'])
                require(0 <= resume <= G, 'Invalid unfinished-year checkpoint')
                predicted = np.lib.format.open_memmap(predicted_path, mode='r+' if resume else 'w+',
                    dtype=np.float32, shape=(len(rows), G))
                require(predicted.shape == (len(rows), G), 'Invalid unfinished-year array shape')
                if resume:
                    LOG.info('Resume year %d from grid point %d/%d', year, resume, G)
                    progress.done += resume * len(needed)
                for left in range(resume, G, s.tile):
                    right = min(left + s.tile, G)
                    clear_geometry()
                    features = year_features(s, static, year, left, right)
                    # Transfer this tile's full year once, avoiding daily history
                    # transfers. Only the prediction values return to CPU.
                    features_t = torch.from_numpy(features).to(device)
                    pos = torch.from_numpy(static['latlon'][None, left:right]).to(device)
                    for i, d in enumerate(needed):
                        context = encoded[i].to(device)
                        for a in range(0, right - left, args.inference_chunk):
                            b = min(a + args.inference_chunk, right - left)
                            mu, _ = model(ctx_pos, None, ctx_mask, pos[:, a:b], features_t[d:d + 1, a:b],
                                tgt_node_idx=None, ctx_node_idx=ctx_idx, encoded_context=context)
                            predicted[d, left + a:left + b] = mu[0].cpu().numpy().clip(min=0)
                            progress.done += b - a
                    del features, features_t
                    predicted.flush()
                    write_json(work_marker, dict(signature=s.signature, shape=[len(rows), G],
                               months=months, completed_points=right))
                predicted.flush()
                for month in months:
                    write_month(s, static, year, month, predicted)
                del predicted, encoded
                predicted_path.unlink()
                work_marker.unlink(missing_ok=True)
                check_source_identity(s)
                LOG.info('Year %d completed', year)
        LOG.info('Task %d complete', args.task_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'prepare', 'run', 'check'])
    parser.add_argument('--main-result', default=str(HERE / 'data'))
    parser.add_argument('--input-dir', default=str(HERE / 'inputs/grid'))
    parser.add_argument('--station-cache-dir', default=str(HERE / 'inputs/station_cache'))
    parser.add_argument('--output-dir', default=str(HERE / 'result'))
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--task-id', type=int, default=1, choices=range(1, 11))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--tile-size', type=int, default=2048)
    parser.add_argument('--inference-chunk', type=int, default=512)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--deep-check', action='store_true')
    args = parser.parse_args()
    require(min(args.tile_size, args.inference_chunk, args.threads) > 0, 'Chunk sizes/threads must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    try:
        s = setup(args)
        LOG.info('fold01 checkpoint=%s; source=%s; cache=%s', s.checkpoint, s.data_root, s.cache)
        if args.command == 'preflight':
            LOG.info('Preflight passed. 492 monthly products; ten task groups: %s', GROUPS)
        elif args.command == 'prepare':
            prepare(s)
        elif args.command == 'run':
            run(s, args)
        else:
            require((s.cache / 'READY.json').exists(), 'Preparation has not completed')
            static = dict(np.load(s.cache / 'static.npz', allow_pickle=False))
            missing = []
            for i, years in enumerate(GROUPS, 1):
                holes = [(y, m) for y in years for m in range(1, 13)
                         if not completed_month(s, static, y, m, args.deep_check)]
                missing.extend(holes)
                print(f'Task {i:02d} years={years}: {len(holes)} missing; {holes}', flush=True)
            require(not missing, f'{len(missing)} monthly files remain unfinished')
            LOG.info('All 492 monthly products are complete')
    except Exception:
        LOG.exception('Task failed; complete traceback follows')
        raise


if __name__ == '__main__':
    main()
