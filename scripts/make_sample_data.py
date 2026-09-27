# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""テスト・デモ用の合成netCDFデータを生成する。

使い方:
    python scripts/make_sample_data.py [出力パス]

デフォルト出力: data/sample/sample_atmos.nc

含まれる変数:
    t, z, u, v  — (time, level, lat, lon) の4次元
    precip, sst — (time, lat, lon) の3次元 (sst は高緯度が欠損)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def create_sample_dataset(path: str) -> xr.Dataset:
    rng = np.random.default_rng(42)
    times = pd.date_range("2024-01-01", periods=8, freq="6h")
    levels = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0])
    lats = np.arange(-90.0, 90.01, 2.5)
    lons = np.arange(0.0, 360.0, 2.5)
    nt, ny, nx = len(times), len(lats), len(lons)

    latr = np.deg2rad(lats)
    lonr = np.deg2rad(lons)
    steps = np.arange(nt, dtype=float)

    # 4次元場 (time, level, lat, lon)
    lat4 = latr[None, None, :, None]
    lon4 = lonr[None, None, None, :]
    lev4 = levels[None, :, None, None]
    t4 = steps[:, None, None, None]
    wave = np.sin(3.0 * lon4 - 0.4 * t4) * np.cos(lat4) ** 2

    t_air = 288.0 * (lev4 / 1000.0) ** 0.19 - 35.0 * np.sin(lat4) ** 2 + 4.0 * wave
    z = 7000.0 * np.log(1000.0 / lev4) - 600.0 * np.sin(lat4) ** 2 + 120.0 * wave
    u = 25.0 * np.sin(2.0 * lat4) ** 2 * (1.3 - lev4 / 1000.0) + 5.0 * wave
    v = (8.0 * np.sin(3.0 * lon4 - 0.4 * t4) * np.cos(lat4)) * np.ones_like(lev4)

    # 3次元場 (time, lat, lon)
    lat3 = latr[None, :, None]
    lon3 = lonr[None, None, :]
    t3 = steps[:, None, None]
    precip = np.maximum(
        0.0,
        10.0 * np.cos(lat3) ** 6 * (1.0 + np.sin(2.0 * lon3 - 0.5 * t3)) - 2.0
        + 0.5 * rng.gamma(2.0, 1.0, (nt, ny, nx)),
    )
    sst = 271.0 + 29.0 * np.cos(lat3) ** 3 + 0.8 * np.sin(lon3 + 0.2 * t3)
    sst = np.where(np.abs(lats)[None, :, None] > 60.0, np.nan, sst)

    ds = xr.Dataset(
        {
            "t": (("time", "level", "lat", "lon"), t_air.astype(np.float32),
                  {"units": "K", "long_name": "air temperature"}),
            "z": (("time", "level", "lat", "lon"), z.astype(np.float32),
                  {"units": "m", "long_name": "geopotential height"}),
            "u": (("time", "level", "lat", "lon"), u.astype(np.float32),
                  {"units": "m s-1", "long_name": "eastward wind"}),
            "v": (("time", "level", "lat", "lon"), v.astype(np.float32),
                  {"units": "m s-1", "long_name": "northward wind"}),
            "precip": (("time", "lat", "lon"), precip.astype(np.float32),
                       {"units": "mm day-1", "long_name": "precipitation rate"}),
            "sst": (("time", "lat", "lon"), sst.astype(np.float32),
                    {"units": "K", "long_name": "sea surface temperature"}),
        },
        coords={
            "time": times,
            "level": ("level", levels,
                      {"units": "hPa", "long_name": "pressure level",
                       "positive": "down", "axis": "Z"}),
            "lat": ("lat", lats,
                    {"units": "degrees_north", "long_name": "latitude",
                     "standard_name": "latitude", "axis": "Y"}),
            "lon": ("lon", lons,
                    {"units": "degrees_east", "long_name": "longitude",
                     "standard_name": "longitude", "axis": "X"}),
        },
        attrs={"title": "ClimCanvas synthetic sample data"},
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return ds


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample/sample_atmos.nc"
    ds = create_sample_dataset(out)
    print(f"生成しました: {out}")
    print(ds)
