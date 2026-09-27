# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""異常データ堅牢性 (観点⑧): エッジデータで「描ける」か「原因の分かるエラー」か。

通常のテスト・ファズは行儀の良いサンプルデータしか流していない。ここでは
実データでユーザーが最初に踏みがちな退化ケース (全 NaN・定数場・1点次元・
重複/非単調座標) を流し、挙動を仕様として固定する
(docs/semantic_test_plan.md 観点⑧)。

方針: matplotlib 深部の不可解なエラーは許容しない (require_increasing_levels
と同じ流儀で render 側に明示ガードを置く)。「描ける」ケースは描けること
だけを確認する (非単調座標などの見た目の正しさはここでは問わない)。
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from climcanvas.core import config as mc_config
from climcanvas.core import render as mc_render


def _map_ds(values, lat=None, lon=None) -> xr.Dataset:
    lat = np.arange(0.0, 50.0, 10.0) if lat is None else np.asarray(lat)
    lon = np.arange(100.0, 190.0, 10.0) if lon is None else np.asarray(lon)
    return xr.Dataset(
        {"f": (("lat", "lon"), np.asarray(values)[:lat.size, :lon.size])},
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})})


def _render_fill(ds, method, levels=None):
    panel = mc_config.default_panel()
    panel["selection"] = {}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = method
    if levels is not None:
        fill["style"]["levels"] = levels
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    fig = mc_render.render_figure(cfg, {"ds0": ds})
    fig.savefig(io.BytesIO(), format="png")  # 描画完了 (colorbar 含む) まで通す
    plt.close(fig)


_VALS = (np.arange(45, dtype=float).reshape(5, 9) + 1.0)
_ALL_NAN = np.full((5, 9), np.nan)
_CONST = np.full((5, 9), 7.5)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --- 全 NaN 変数 ---

def test_all_nan_contourf_renders_blank():
    _render_fill(_map_ds(_ALL_NAN), "contourf")  # 空の図として描ける


def test_all_nan_pcolormesh_raises_clear_error():
    with pytest.raises(mc_render.RenderError, match="missing"):
        _render_fill(_map_ds(_ALL_NAN), "pcolormesh")


# --- 定数場 (min == max で自動レベルが退化) ---

def test_constant_field_contourf_renders():
    _render_fill(_map_ds(_CONST), "contourf")


def test_constant_field_pcolormesh_raises_clear_error():
    with pytest.raises(mc_render.RenderError, match="constant"):
        _render_fill(_map_ds(_CONST), "pcolormesh")


def test_constant_field_pcolormesh_with_explicit_levels_renders():
    """エラーメッセージの案内どおり、レベル直接指定なら定数場も描ける。"""
    _render_fill(_map_ds(_CONST), "pcolormesh", levels=[0.0, 5.0, 10.0])


# --- 1点しかない次元 ---

def test_single_lat_contourf_raises_matplotlib_error():
    """contourf は 2×2 未満を描けない (matplotlib の明示エラーをそのまま許容)。"""
    with pytest.raises(TypeError, match="at least"):
        _render_fill(_map_ds(_VALS, lat=[20.0]), "contourf")


def test_single_lat_pcolormesh_renders():
    _render_fill(_map_ds(_VALS, lat=[20.0]), "pcolormesh")


# --- 重複・非単調座標 (描けることのみ確認。見た目の正しさは対象外) ---

def test_duplicate_lon_renders():
    lon = np.array([100., 110., 110., 120., 130., 140., 150., 160., 170.])
    for method in ("contourf", "pcolormesh"):
        _render_fill(_map_ds(_VALS, lon=lon), method)


def test_unsorted_lon_renders():
    lon = np.array([100., 130., 110., 120., 140., 150., 160., 170., 180.])
    for method in ("contourf", "pcolormesh"):
        _render_fill(_map_ds(_VALS, lon=lon), method)


# --- 2 次元座標 (curvilinear) ---

def _curvi_ds(lon2d, lat2d, values=None) -> xr.Dataset:
    lon2d = np.asarray(lon2d, dtype=float)
    ny, nx = lon2d.shape
    vals = np.arange(ny * nx, dtype=float).reshape(ny, nx) if values is None else values
    return xr.Dataset(
        {"f": (("y", "x"), vals)},
        coords={"y": ("y", np.arange(ny, dtype=float)), "x": ("x", np.arange(nx, dtype=float)),
                "lon": (("y", "x"), lon2d, {"units": "degrees_east"}),
                "lat": (("y", "x"), np.asarray(lat2d, dtype=float), {"units": "degrees_north"})})


def _render_curvi(ds, region=None, method="contourf", proj="PlateCarree", clon=180.0):
    panel = mc_config.default_panel()
    panel["selection"] = {}
    panel["region"] = region
    panel["projection"] = {"name": proj, "central_longitude": clon}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = method
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    fig = mc_render.render_figure(cfg, {"ds0": ds})
    fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)


def test_curvilinear_region_outside_grid_raises_clear_error():
    lon2d, lat2d = np.meshgrid(np.arange(100.0, 160.0, 5.0), np.arange(10.0, 50.0, 5.0))
    ds = _curvi_ds(lon2d + 0.1 * lat2d, lat2d)
    with pytest.raises(mc_render.RenderError) as ei:
        _render_curvi(ds, {"lon_min": 0.0, "lon_max": 30.0, "lat_min": -40.0, "lat_max": -10.0})
    assert ei.value.msg_id == "region_outside_grid"
    assert "no grid points" in str(ei.value)


@pytest.mark.parametrize("method", ["contourf", "pcolormesh"])
def test_curvilinear_region_crossing_dateline_renders(method):
    """-180 規約で日付変更線をまたぐ格子 (150〜-150) に 170〜190 の region: 経度は
    (lon - lon_min) % 360 で比較するので切り出せる。"""
    lon = np.arange(150.0, 211.0, 5.0)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lon2d, lat2d = np.meshgrid(lon, np.arange(10.0, 50.0, 5.0))
    ds = _curvi_ds(lon2d, lat2d + 0.05 * (lon2d % 360.0))
    _render_curvi(ds, {"lon_min": 170.0, "lon_max": 190.0, "lat_min": 15.0, "lat_max": 45.0},
                  method=method)


@pytest.mark.parametrize("method", ["contourf", "pcolormesh"])
def test_curvilinear_with_nan_coordinates_raises_clear_error(method):
    """経緯度が NaN の格子点 (海洋モデルの陸面など) は matplotlib がメッシュを描けない
    ため、描画範囲に含まれるなら原因の分かる RenderError。領域指定でその格子点を外せば
    描ける (切り出し後の範囲だけを検査する)。"""
    lon2d, lat2d = np.meshgrid(np.arange(100.0, 160.0, 5.0), np.arange(10.0, 50.0, 5.0))
    lon2d = lon2d.copy()
    lat2d = lat2d.copy()
    lon2d[0, :3] = np.nan
    lat2d[0, :3] = np.nan
    ds = _curvi_ds(lon2d, lat2d)
    with pytest.raises(mc_render.RenderError) as ei:
        _render_curvi(ds, method=method, clon=130.0)
    assert ei.value.msg_id == "curvilinear_coords_nonfinite"
    assert ei.value.params["n"] == 6
    # NaN の格子点 (南西の隅) を外す領域なら描ける
    _render_curvi(ds, region={"lon_min": 125.0, "lon_max": 150.0, "lat_min": 20.0, "lat_max": 45.0},
                  method=method, clon=130.0)


def test_curvilinear_single_row_region_pads_to_two_rows():
    """領域内の格子点が 1 行しか無くても外接矩形は両側 1 格子広がるので描ける
    (contourf は 2 行以上を要求する)。"""
    lon2d, lat2d = np.meshgrid(np.arange(100.0, 160.0, 5.0), np.arange(10.0, 50.0, 5.0))
    ds = _curvi_ds(lon2d, lat2d)
    _render_curvi(ds, {"lon_min": 100.0, "lon_max": 160.0, "lat_min": 24.0, "lat_max": 26.0},
                  clon=130.0)
