# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""無次元の x / y を軸とする小さな2次元行列データを生成する。

2次元プロット (散布図) や categorical heatmap の動作確認用。lat/lon などの
標準座標役割を持たないので、`detect_coord_roles` はどの役割にも分類せず、
水平断面図モードは出ない (無次元の小行列として扱われる)。

- `sample_matrix.nc`      : A(x, y) 10×10。x / y は無次元の**座標変数あり** (0..9)。
                            2 変数 (A, B) を持つので散布図 (A vs B) も試せる
- `sample_matrix_baredims.nc` : 同じ A / B だが x / y に**座標変数を持たない**
                            (bare dims)。現状どのモードにも該当しない (case d) の確認用
"""

from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr


def _fields(n: int = 10):
    """再現可能な 10×10 の A, B を返す (dims 順は (x, y))。"""
    rng = np.random.default_rng(2024)
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    # A: なめらかな山谷パターン + わずかなノイズ
    a = (np.sin(x[:, None] / 1.5) * np.cos(y[None, :] / 1.5)
         + 0.05 * rng.standard_normal((n, n)))
    # B: 中心からの距離に依存する別パターン (散布図で A と関係づけて見られる)
    b = ((x[:, None] - 4.5) * (y[None, :] - 4.5)) / 10.0
    return x, y, a.astype(np.float32), b.astype(np.float32)


def create_matrix_dataset(path: str) -> xr.Dataset:
    """無次元 x/y の座標変数を持つ 10×10 行列 (A, B)。"""
    x, y, a, b = _fields()
    ds = xr.Dataset(
        {
            "A": (("x", "y"), a, {"long_name": "field A"}),
            "B": (("x", "y"), b, {"long_name": "field B"}),
        },
        coords={
            # 無次元: units / standard_name / axis を付けない
            # (detect_coord_roles がどの役割にも分類しないようにする)
            "x": ("x", x, {"long_name": "x index"}),
            "y": ("y", y, {"long_name": "y index"}),
        },
        attrs={"title": "ClimCanvas synthetic 2D matrix (dimensionless x/y)"},
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return ds


def create_baredims_dataset(path: str) -> xr.Dataset:
    """座標変数を持たない (bare dims) 10×10 行列 (A, B)。"""
    _, _, a, b = _fields()
    ds = xr.Dataset(
        {
            "A": (("x", "y"), a, {"long_name": "field A"}),
            "B": (("x", "y"), b, {"long_name": "field B"}),
        },
        # coords を与えない → x / y は座標変数を持たない bare dims になる
        attrs={"title": "ClimCanvas synthetic 2D matrix (bare dims)"},
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return ds


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    matrix_path = os.path.join(out_dir, "sample_matrix.nc")
    bare_path = os.path.join(out_dir, "sample_matrix_baredims.nc")
    create_matrix_dataset(matrix_path)
    create_baredims_dataset(bare_path)
    print(f"生成しました: {matrix_path}")
    print(f"生成しました: {bare_path}")
