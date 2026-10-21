# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""sample_atmos.nc を時刻ごとに分割した「複数ファイル」のサンプルを生成する。

使い方:
    python scripts/make_sample_multifile_data.py [出力ディレクトリ]

デフォルト出力: data/sample/multi/sample_atmos_{YYYYMMDDhh}.nc (8 ファイル、1 時刻ずつ)。
元の data/sample/sample_atmos.nc が無ければ先に生成する。
アプリのパス欄に `data/sample/multi/sample_atmos_*.nc` と書けば 1 つのデータセットとして
開ける (docs/multifile_glob_plan.md)。
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.make_sample_data import create_sample_dataset  # noqa: E402


def split_by_time(src_path: str, out_dir: str, stem: str = "sample_atmos") -> list[str]:
    """src_path を time の 1 要素ごとに out_dir/{stem}_{YYYYMMDDhh}.nc へ書き出し、
    書いたパスのリストを返す (時刻順)。time の encoding (units / calendar) は元のまま。"""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with xr.open_dataset(src_path, decode_timedelta=False) as ds:
        for i in range(ds.sizes["time"]):
            part = ds.isel(time=[i])
            stamp = pd.Timestamp(part["time"].values[0]).strftime("%Y%m%d%H")
            path = os.path.join(out_dir, f"{stem}_{stamp}.nc")
            encoding = {name: {"zlib": True, "complevel": 4} for name in part.data_vars}
            part.to_netcdf(path, encoding=encoding)
            paths.append(path)
    return paths


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample/multi"
    src = "data/sample/sample_atmos.nc"
    if not os.path.isfile(src):
        create_sample_dataset(src)
        print(f"生成しました: {src}")
    for p in split_by_time(src, out_dir):
        print(f"生成しました: {p}")
