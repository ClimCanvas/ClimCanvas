# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""2 次元座標 (curvilinear 格子) の合成サンプルデータを生成する。

ランベルト正角円錐図法 (標準緯線 30°N / 60°N、中心経度 140°E) の等間隔格子
(60 × 50、100 km) 上の場を作り、経緯度を格子番号 (y, x) の 2 次元変数として持つ。
ClimCORE の領域モデル出力 (JMA MSM 系と同じ格子構成) の縮小版で、WRF や海洋モデルの
lat(y,x) 形式の代表でもある。3 ファイルを書く:

- `sample_curvilinear.nc`        : lon(y,x) / lat(y,x) を 2 次元座標として埋め込んだもの
- `sample_curvilinear_bare.nc`   : 経緯度なし (座標は格子番号 x / y だけ。ClimCORE 形式)
- `sample_curvilinear_lonlat.nc` : FLON(y,x) / FLAT(y,x) だけ (座標ファイル)

値は位置と時刻から一意に決まる滑らかな関数 (アーティスト検査で期待値を独立計算
できるように)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

# 格子の定義 (テスト・文書から参照できるようモジュール定数にする)
LCC_CENTRAL_LON = 140.0
LCC_CENTRAL_LAT = 30.0
LCC_STANDARD_PARALLELS = (30.0, 60.0)
NX, NY = 60, 50
DX = 100.0e3          # 格子間隔 [m]
Y_OFFSET = 500.0e3    # 領域の中心を中心緯度より北へずらす [m]


def lcc_lonlat() -> tuple[np.ndarray, np.ndarray]:
    """格子点の (lon, lat) を (NY, NX) の 2 次元配列で返す (cartopy の逆変換)。"""
    import cartopy.crs as ccrs

    proj = ccrs.LambertConformal(central_longitude=LCC_CENTRAL_LON,
                                 central_latitude=LCC_CENTRAL_LAT,
                                 standard_parallels=LCC_STANDARD_PARALLELS)
    x = (np.arange(NX) - (NX - 1) / 2.0) * DX
    y = (np.arange(NY) - (NY - 1) / 2.0) * DX + Y_OFFSET
    xx, yy = np.meshgrid(x, y)
    ll = ccrs.PlateCarree().transform_points(proj, xx, yy)
    return ll[..., 0], ll[..., 1]


def _fields(lon2d, lat2d):
    # 時刻は sample_atmos.nc と同じ起点 (別ファイルの重ね描きで時刻を共有できるように)
    times = pd.date_range("2024-01-01", periods=4, freq="6h")
    levels = np.array([1000.0, 850.0, 500.0, 300.0])
    nt = len(times)
    latr = np.deg2rad(lat2d)[None, None]
    lonr = np.deg2rad(lon2d)[None, None]
    lev4 = levels[None, :, None, None]
    t4 = np.arange(nt, dtype=float)[:, None, None, None]
    wave = np.sin(3.0 * lonr - 0.4 * t4) * np.cos(latr) ** 2
    t_air = 288.0 * (lev4 / 1000.0) ** 0.19 - 35.0 * np.sin(latr) ** 2 + 4.0 * wave
    z = 7000.0 * np.log(1000.0 / lev4) - 600.0 * np.sin(latr) ** 2 + 120.0 * wave
    u = 25.0 * np.sin(2.0 * latr) ** 2 * (1.3 - lev4 / 1000.0) + 5.0 * wave
    v = 8.0 * np.sin(3.0 * lonr - 0.4 * t4) * np.cos(latr) * np.ones_like(lev4)
    t2 = (300.0 - 25.0 * np.sin(latr[:, 0]) ** 2
          + 3.0 * np.sin(2.0 * lonr[:, 0] - 0.5 * t4[:, 0]))
    return times, levels, t_air, z, u, v, t2


def create_curvilinear_dataset(path: str, bare_path: str | None = None,
                               lonlat_path: str | None = None) -> xr.Dataset:
    """埋め込み版を path に書き、bare_path / lonlat_path があれば分離版も書く。"""
    lon2d, lat2d = lcc_lonlat()
    times, levels, t_air, z, u, v, t2 = _fields(lon2d, lat2d)
    grid = ("time", "lev", "y", "x")
    data_vars = {
        "t": (grid, t_air.astype(np.float32),
              {"units": "K", "long_name": "air temperature"}),
        "z": (grid, z.astype(np.float32),
              {"units": "m", "long_name": "geopotential height"}),
        "u": (grid, u.astype(np.float32),
              {"units": "m s-1", "long_name": "eastward wind"}),
        "v": (grid, v.astype(np.float32),
              {"units": "m s-1", "long_name": "northward wind"}),
        "t2": (("time", "y", "x"), t2.astype(np.float32),
               {"units": "K", "long_name": "2 m temperature"}),
    }
    coords = {
        "time": times,
        "lev": ("lev", levels, {"units": "hPa", "long_name": "pressure level"}),
        "y": ("y", np.arange(1, NY + 1, dtype=np.float32),
              {"long_name": "y grid number", "units": ""}),
        "x": ("x", np.arange(1, NX + 1, dtype=np.float32),
              {"long_name": "x grid number", "units": ""}),
    }
    attrs = {"title": "ClimCanvas synthetic curvilinear (Lambert conformal) sample",
             "grid": (f"Lambert conformal, standard parallels "
                      f"{LCC_STANDARD_PARALLELS[0]:g}N/{LCC_STANDARD_PARALLELS[1]:g}N, "
                      f"central longitude {LCC_CENTRAL_LON:g}E, dx = {DX / 1e3:g} km")}
    lon_attrs = {"units": "degrees_east", "long_name": "longitude",
                 "standard_name": "longitude"}
    lat_attrs = {"units": "degrees_north", "long_name": "latitude",
                 "standard_name": "latitude"}
    bare = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds = bare.assign_coords(lon=(("y", "x"), lon2d.astype(np.float32), lon_attrs),
                            lat=(("y", "x"), lat2d.astype(np.float32), lat_attrs))
    enc = {name: {"zlib": True, "complevel": 4} for name in data_vars}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds.to_netcdf(path, encoding=enc)
    if bare_path:
        bare.to_netcdf(bare_path, encoding=enc)
    if lonlat_path:
        ll = xr.Dataset(
            {"FLON": (("y", "x"), lon2d.astype(np.float32), lon_attrs),
             "FLAT": (("y", "x"), lat2d.astype(np.float32), lat_attrs)},
            coords={"y": coords["y"], "x": coords["x"]})
        ll.to_netcdf(lonlat_path)
    return ds


def create_curvilinear_terrain(path: str) -> xr.Dataset:
    """同じランベルト格子の地表の変数 (地形マスクの確認用): 地形高度 zs [m] と地上気圧 ps [Pa]。

    山は 2 つ (138°E 36°N に 1800 m、128°E 33°N に 600 m のガウス型)。ps は
    1000 hPa × exp(−zs / 8000 m) を Pa で持ち、単位の換算 (Pa → hPa) の確認に使う。
    経緯度は本体と同じく 2 次元座標として埋め込む。
    """
    lon2d, lat2d = lcc_lonlat()
    zs = (1800.0 * np.exp(-((lon2d - 138.0) ** 2 / 32.0 + (lat2d - 36.0) ** 2 / 18.0))
          + 600.0 * np.exp(-((lon2d - 128.0) ** 2 / 18.0 + (lat2d - 33.0) ** 2 / 8.0)))
    ps = 1000.0e2 * np.exp(-zs / 8000.0)
    coords = {
        "y": ("y", np.arange(1, NY + 1, dtype=np.float32),
              {"long_name": "y grid number", "units": ""}),
        "x": ("x", np.arange(1, NX + 1, dtype=np.float32),
              {"long_name": "x grid number", "units": ""}),
        "lon": (("y", "x"), lon2d.astype(np.float32),
                {"units": "degrees_east", "long_name": "longitude"}),
        "lat": (("y", "x"), lat2d.astype(np.float32),
                {"units": "degrees_north", "long_name": "latitude"}),
    }
    ds = xr.Dataset(
        {"zs": (("y", "x"), zs.astype(np.float32),
                {"units": "m", "long_name": "terrain height"}),
         "ps": (("y", "x"), ps.astype(np.float32),
                {"units": "Pa", "long_name": "surface pressure"})},
        coords=coords,
        attrs={"title": "ClimCanvas synthetic curvilinear terrain (zs, ps)"})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds.to_netcdf(path)
    return ds


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    ds = create_curvilinear_dataset(
        os.path.join(out_dir, "sample_curvilinear.nc"),
        os.path.join(out_dir, "sample_curvilinear_bare.nc"),
        os.path.join(out_dir, "sample_curvilinear_lonlat.nc"))
    print(ds)
    print(create_curvilinear_terrain(os.path.join(out_dir, "sample_curvilinear_terrain.nc")))
