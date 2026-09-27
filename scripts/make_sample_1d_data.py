# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""1次元プロットの動作確認用の小さな netCDF を 2 つ生成する。

- `sample_1d_time.nc`: 時間次元 (length 30, 日次)
- `sample_1d_x.nc`: 標準役割を持たない汎用 dim `x` (length 30)

それぞれ 2 変数を含むので fill_between (line fill) の変数モードも試せる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def create_time_dataset(path: str) -> xr.Dataset:
    """時間次元を持つ 1 次元データ。"""
    rng = np.random.default_rng(7)
    n = 30
    times = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    # 年周期の一部を再現したような気温風の信号 (K) と、観測のばらつき
    base = 278.0 + 6.0 * np.sin(2 * np.pi * t / 60.0)
    model = base + 1.5 * np.sin(2 * np.pi * t / 7.0)
    obs = model + rng.normal(0.0, 1.2, n)

    ds = xr.Dataset(
        {
            "obs": (("time",), obs.astype(np.float32),
                    {"units": "K", "long_name": "observed temperature"}),
            "model": (("time",), model.astype(np.float32),
                      {"units": "K", "long_name": "model temperature"}),
        },
        coords={
            "time": times,
        },
        attrs={"title": "ClimCanvas synthetic 1D time series"},
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return ds


def create_x_dataset(path: str) -> xr.Dataset:
    """標準役割 (lat/lon/vertical/time) を持たない汎用 dim `x` の 1 次元データ。

    座標 `x` には単位や standard_name を付けず、`detect_coord_roles` がどの役割にも
    分類しない状態にしてある (非標準 dim のテスト用)。
    """
    n = 30
    x = np.arange(n, dtype=np.float32)
    # 2 つの曲線: 1 つはランプ + サイン、もう 1 つは指数減衰風
    y1 = (0.5 * x + 3.0 * np.sin(0.6 * x))
    y2 = 15.0 * np.exp(-0.05 * x) * np.cos(0.4 * x)

    # 3 つ目: NaN を含む系列。線プロットでの欠損挙動の確認用
    #   - インデックス 5 に単独 NaN
    #   - インデックス 12-16 に連続 NaN (5 点ブロック)
    #   - インデックス 25 にもう 1 つ単独 NaN
    y_gap = (0.4 * x + 2.5 * np.cos(0.5 * x)).astype(np.float32)
    y_gap[5] = np.nan
    y_gap[12:17] = np.nan
    y_gap[25] = np.nan

    ds = xr.Dataset(
        {
            "y1": (("x",), y1.astype(np.float32),
                   {"long_name": "ramp + sine"}),
            "y2": (("x",), y2.astype(np.float32),
                   {"long_name": "damped oscillation"}),
            "y_gap": (("x",), y_gap,
                       {"long_name": "ramp + cosine with NaN gaps"}),
        },
        coords={
            "x": ("x", x, {"long_name": "sample index"}),
        },
        attrs={"title": "ClimCanvas synthetic 1D generic-axis data"},
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return ds


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    time_path = os.path.join(out_dir, "sample_1d_time.nc")
    x_path = os.path.join(out_dir, "sample_1d_x.nc")
    create_time_dataset(time_path)
    create_x_dataset(x_path)
    print(f"生成しました: {time_path}")
    print(f"生成しました: {x_path}")
