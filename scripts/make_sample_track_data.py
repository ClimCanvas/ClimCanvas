# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""IBTrACS 風の合成ベストトラック netCDF を生成する。

storm × date_time の構造で、lat / lon (degrees_north / degrees_east)、
wind (kts)、pres (hPa)、time (datetime64、発生年月は GENESIS 表参照)、
name を持つ。実データ同様にトラック長はまちまちで末尾は NaN/NaT パディング。
日付変更線 (経度 ±180) をまたぐトラックを含む。経度は IBTrACS と同じ
-180〜180 規約。

usage: $PY scripts/make_sample_track_data.py [出力パス]
       (省略時 data/sample/sample_besttrack.nc)
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

N_STORM = 8
N_TIME = 40  # 6時間毎で最大10日

# 各ストームの発生 (年, 月)。年・月フィルタのテスト用に複数年・複数月に分散
GENESIS = [(2023, 6), (2023, 9), (2024, 7), (2024, 8),
           (2024, 10), (2025, 7), (2025, 9), (2025, 11)]


def create_track_dataset(path: str) -> None:
    rng = np.random.default_rng(42)
    lat = np.full((N_STORM, N_TIME), np.nan, dtype=np.float32)
    lon = np.full((N_STORM, N_TIME), np.nan, dtype=np.float32)
    wind = np.full((N_STORM, N_TIME), np.nan, dtype=np.float32)
    pres = np.full((N_STORM, N_TIME), np.nan, dtype=np.float32)
    time = np.full((N_STORM, N_TIME), np.datetime64("NaT", "ns"),
                   dtype="datetime64[ns]")
    names = []

    for s in range(N_STORM):
        n = int(rng.integers(16, N_TIME + 1))       # トラック長 (16〜40 点)
        t = np.arange(n, dtype=np.float64)
        # 発生位置: 北西太平洋。最後の2つは日付変更線付近で発生させて
        # 経度 180 またぎを必ず含める
        lon0 = 135.0 + 8.0 * s if s < N_STORM - 2 else 172.0 + 4.0 * (s - N_STORM + 2)
        lat0 = 8.0 + 2.0 * rng.random()
        # 転向 (recurve) するパラボラ型のトラック + 小さな揺らぎ
        la = lat0 + 0.9 * t + 0.012 * t ** 2 + rng.normal(0.0, 0.25, n)
        lo = lon0 + (-0.55 * t + 0.028 * t ** 2) + rng.normal(0.0, 0.25, n)
        # -180〜180 に正規化 (IBTrACS の経度規約)
        lo = (lo + 180.0) % 360.0 - 180.0
        # 強度: 発達 → 最盛期 → 衰弱 (kts)
        peak = float(rng.integers(55, 115))
        w = peak * np.sin(np.pi * (t + 1) / (n + 1)) ** 0.8 + rng.normal(0, 2, n)
        lat[s, :n] = la
        lon[s, :n] = lo
        wind[s, :n] = np.clip(w, 20.0, None)
        pres[s, :n] = 1010.0 - 5.5 * (np.clip(w, 20.0, None) - 20.0) ** 0.9 / 2.0
        # 発生時刻 (GENESIS の年月 + ストーム毎に日をずらす) から 6時間毎
        gy, gm = GENESIS[s]
        t0 = np.datetime64(f"{gy:04d}-{gm:02d}-{1 + 2 * s:02d}T00", "ns")
        time[s, :n] = t0 + np.arange(n) * np.timedelta64(6, "h")
        names.append(f"STORM_{s:02d}")

    ds = xr.Dataset(
        {
            "lat": (("storm", "date_time"), lat,
                    {"long_name": "latitude", "units": "degrees_north"}),
            "lon": (("storm", "date_time"), lon,
                    {"long_name": "longitude", "units": "degrees_east"}),
            "wind": (("storm", "date_time"), wind,
                     {"long_name": "maximum sustained wind", "units": "kts"}),
            "pres": (("storm", "date_time"), pres,
                     {"long_name": "central pressure", "units": "hPa"}),
            "time": (("storm", "date_time"), time,
                     {"long_name": "time of observation"}),
            "name": (("storm",), np.array(names)),
        },
        attrs={"title": "ClimCanvas synthetic best track (IBTrACS-like)"},
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    print(f"wrote {path}  (storms={N_STORM}, max_len={N_TIME})")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample/sample_besttrack.nc"
    create_track_dataset(out)
