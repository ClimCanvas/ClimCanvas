# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""メタモルフィックテスト: 「同じ図になるべき2つの入力」をオラクルにする。

render⇄scriptgen の画像一致は、**両経路に共通する**データ処理ロジックの誤り
(座標の向きの取り違え等は両経路とも同じに間違う) を検出できない。ここでは
入力の**同値変換** (図が変わらないはずの変換) を施した2つの入力をそれぞれ
コールドプロセスで描画し、ピクセル完全一致を検証する
(docs/semantic_test_plan.md 観点⑥)。期待画像の用意が不要な点が特長。
"""

from __future__ import annotations

import json

import matplotlib.image as mpimg
import numpy as np
import pytest
import xarray as xr

from climcanvas.core import config as mc_config
from test_consistency import _RENDER_RUNNER, _ROOT, _run_cold


def _render_cold(cfg: dict, ds_path: str, tmp_path, tag: str) -> np.ndarray:
    runner = tmp_path / f"runner_{tag}.py"
    runner.write_text(_RENDER_RUNNER, encoding="utf-8")
    cfg_path = tmp_path / f"cfg_{tag}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / f"{tag}.png"
    result = _run_cold([str(runner), str(cfg_path), str(ds_path), str(out),
                        _ROOT], tmp_path)
    assert result.returncode == 0, f"{tag} の描画に失敗:\n{result.stderr}"
    return mpimg.imread(out)


def _assert_identical(cfg_a, path_a, cfg_b, path_b, tmp_path, what: str,
                      *, exact: bool = True):
    """2つの入力の描画が同じ図になることを検証する。

    exact=False は**並べ替え系**の同値変換用: 同一ジオメトリを違う順で描く
    ため、セル共有辺のスナップ/AA が点状に揺れる (実測: 降順 lat で 114 px、
    経度規約で 215 px、いずれも最大連結 <200 px・散在)。構造的な乖離
    (向きの取り違え・列のずれ等) は数千 px 規模の連結領域になるため、
    「不一致 <0.5% かつ 最大連結成分 <300 px」で判定する
    (test_consistency の warm 判定と同じ発想)。
    """
    a = _render_cold(cfg_a, path_a, tmp_path, "a")
    b = _render_cold(cfg_b, path_b, tmp_path, "b")
    assert a.shape == b.shape, f"{what}: 画像サイズ不一致 {a.shape} vs {b.shape}"
    if exact:
        assert np.array_equal(a, b), f"{what}: ピクセルが一致しない"
        return
    from scipy import ndimage
    diff = np.any(a != b, axis=2)
    labels, n_blobs = ndimage.label(diff)
    max_blob = np.bincount(labels.ravel())[1:].max() if n_blobs else 0
    assert diff.mean() < 0.005 and max_blob < 300, \
        (f"{what}: 構造的に乖離 (不一致 {diff.mean():.2%}, "
         f"最大差分領域 {max_blob} px)")


def _map_fill_cfg(levels, **style) -> dict:
    panel = mc_config.default_panel()
    panel["selection"] = {}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"method": "pcolormesh", "levels": levels, **style})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _geo_coords(name, values, units):
    return (name, np.asarray(values, dtype=float), {"units": units})


# --- 緯度の格納順 (昇順 ⇄ 降順) は図を変えない ---

def test_lat_storage_order_equivalence(tmp_path):
    lon = np.arange(0.0, 360.0, 22.5)
    lat = np.arange(-40.0, 50.0, 10.0)
    f = lat[:, None] * 1.0 + np.cos(np.deg2rad(lon))[None, :] * 20.0
    ds_asc = xr.Dataset(
        {"f": (("lat", "lon"), f)},
        coords={"lat": _geo_coords("lat", lat, "degrees_north"),
                "lon": _geo_coords("lon", lon, "degrees_east")})
    ds_desc = xr.Dataset(
        {"f": (("lat", "lon"), f[::-1])},
        coords={"lat": _geo_coords("lat", lat[::-1], "degrees_north"),
                "lon": _geo_coords("lon", lon, "degrees_east")})
    pa, pb = str(tmp_path / "asc.nc"), str(tmp_path / "desc.nc")
    ds_asc.to_netcdf(pa)
    ds_desc.to_netcdf(pb)
    cfg = _map_fill_cfg([-40.0, -20.0, 0.0, 20.0, 40.0])
    _assert_identical(cfg, pa, cfg, pb, tmp_path, "lat 昇順⇄降順",
                      exact=False)  # 描画順が変わる並べ替え系


# --- 経度規約 (0–360 ⇄ −180–180) は図を変えない ---

def test_lon_convention_equivalence(tmp_path):
    lon360 = np.arange(0.0, 360.0, 22.5)
    lat = np.arange(-40.0, 50.0, 10.0)
    # 物理的な経度の関数として場を定義する (規約に依らない)
    f = (np.sin(np.deg2rad(lat))[:, None] * 30.0
         + np.cos(np.deg2rad(lon360))[None, :] * 20.0)
    ds_360 = xr.Dataset(
        {"f": (("lat", "lon"), f)},
        coords={"lat": _geo_coords("lat", lat, "degrees_north"),
                "lon": _geo_coords("lon", lon360, "degrees_east")})
    lon_pm = ((lon360 + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_pm)
    ds_pm = xr.Dataset(
        {"f": (("lat", "lon"), f[:, order])},
        coords={"lat": _geo_coords("lat", lat, "degrees_north"),
                "lon": _geo_coords("lon", lon_pm[order], "degrees_east")})
    pa, pb = str(tmp_path / "l360.nc"), str(tmp_path / "lpm.nc")
    ds_360.to_netcdf(pa)
    ds_pm.to_netcdf(pb)
    cfg = _map_fill_cfg([-40.0, -20.0, 0.0, 20.0, 40.0])
    _assert_identical(cfg, pa, cfg, pb, tmp_path, "経度 0–360⇄−180–180",
                      exact=False)  # 描画順が変わる並べ替え系


# --- 変数の次元順 ((lat, lon) ⇄ (lon, lat)) は図を変えない ---

def test_dim_order_equivalence(tmp_path):
    lon = np.arange(100.0, 190.0, 10.0)  # 非全球 (cyclic なし)
    lat = np.arange(0.0, 50.0, 10.0)
    f = lat[:, None] * 10.0 + lon[None, :] * 0.1
    coords = {"lat": _geo_coords("lat", lat, "degrees_north"),
              "lon": _geo_coords("lon", lon, "degrees_east")}
    ds_latlon = xr.Dataset({"f": (("lat", "lon"), f)}, coords=coords)
    ds_lonlat = xr.Dataset({"f": (("lon", "lat"), f.T)}, coords=coords)
    pa, pb = str(tmp_path / "latlon.nc"), str(tmp_path / "lonlat.nc")
    ds_latlon.to_netcdf(pa)
    ds_lonlat.to_netcdf(pb)
    cfg = _map_fill_cfg([100.0, 200.0, 300.0, 400.0])
    _assert_identical(cfg, pa, cfg, pb, tmp_path, "次元順 (lat,lon)⇄(lon,lat)")


# --- 単位変換の可換性: データ側で変換済み ⇄ config の値変換 ---

def test_value_transform_commutes_with_data(tmp_path):
    lon = np.arange(100.0, 190.0, 10.0)
    lat = np.arange(0.0, 50.0, 10.0)
    f_k = 273.15 + lat[:, None] * 0.5 + lon[None, :] * 0.05  # float64 (丸め回避)
    coords = {"lat": _geo_coords("lat", lat, "degrees_north"),
              "lon": _geo_coords("lon", lon, "degrees_east")}
    ds_k = xr.Dataset({"f": (("lat", "lon"), f_k)}, coords=coords)
    ds_c = xr.Dataset({"f": (("lat", "lon"), f_k - 273.15)}, coords=coords)
    pa, pb = str(tmp_path / "kelvin.nc"), str(tmp_path / "celsius.nc")
    ds_k.to_netcdf(pa)
    ds_c.to_netcdf(pb)
    levels_c = [0.0, 5.0, 10.0, 20.0, 30.0]  # どちらも °C の値で指定
    cfg_k = _map_fill_cfg(levels_c, value_offset=-273.15)
    cfg_c = _map_fill_cfg(levels_c)
    _assert_identical(cfg_k, pa, cfg_c, pb, tmp_path,
                      "値変換 (config) ⇄ 変換済みデータ")


# --- レイアウト指定の同値性: mosaic 文字リネーム / grid⇄mosaic の2経路 ---
# 注: 「パネル順の入れ替え」は mosaic では表現できない (文字は初出順で
# panels に割り当てられるため panels[0] が常に最初のセルに入る)。

def test_layout_spec_equivalence(tmp_path):
    level = np.array([1000.0, 850.0, 700.0, 500.0])
    ds = xr.Dataset(
        {"p": (("level",), np.array([4.0, 6.0, 2.0, 8.0])),
         "q": (("level",), np.array([1.0, 2.0, 3.0, 4.0]))},
        coords={"level": ("level", level, {"units": "hPa"})})
    path = str(tmp_path / "bars.nc")
    ds.to_netcdf(path)

    def _line_panel(var, title):
        panel = mc_config.default_line_panel()
        panel["x_dim"] = "level"
        panel["selection"] = {}
        panel["title"] = title
        panel["layers"] = [mc_config.default_line_layer("ds0", var)]
        return panel

    def _cfg(mosaic=None, nrows=1, ncols=2):
        cfg = mc_config.default_figure_config()
        cfg["panels"] = [_line_panel("p", "P1"), _line_panel("q", "P2")]
        cfg["figure"]["layout"].update({"nrows": nrows, "ncols": ncols,
                                        "mosaic": mosaic})
        return cfg

    # mosaic の文字は任意 (リネームしても同じ配置)
    _assert_identical(_cfg(mosaic="AB"), path, _cfg(mosaic="XY"), path,
                      tmp_path, "mosaic 文字リネーム")
    # 従来の grid 指定と mosaic 指定は同じ配置になる (2つのコード経路)
    _assert_identical(_cfg(mosaic=None), path, _cfg(mosaic="AB"), path,
                      tmp_path, "grid⇄mosaic の2経路")


def _regional_1d_and_2d(sample_path, tmp_path):
    """sample_atmos の一部領域を、(a) 1 次元 lat/lon のまま、(b) dim を y/x (格子番号) に
    改名し lon(y,x) / lat(y,x) を meshgrid で付けた curvilinear 形式、の 2 通りで書く。"""
    ds = xr.open_dataset(sample_path).sel(lon=slice(100.0, 200.0), lat=slice(0.0, 60.0))
    p1 = tmp_path / "regional_1d.nc"
    ds.to_netcdf(p1)
    lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    ds2 = ds.rename({"lat": "y", "lon": "x"})
    ds2 = ds2.assign_coords(
        y=("y", np.arange(1, ds.sizes["lat"] + 1, dtype=np.float32),
           {"long_name": "y grid number", "units": ""}),
        x=("x", np.arange(1, ds.sizes["lon"] + 1, dtype=np.float32),
           {"long_name": "x grid number", "units": ""}),
        lon=(("y", "x"), lon2d, dict(ds["lon"].attrs)),
        lat=(("y", "x"), lat2d, dict(ds["lat"].attrs)))
    p2 = tmp_path / "regional_2d.nc"
    ds2.to_netcdf(p2)
    return p1, p2


@pytest.mark.parametrize("method", ["contourf", "pcolormesh"])
def test_curvilinear_expansion_equivalence(method, sample_path, tmp_path):
    """1 次元格子を 2 次元座標 (meshgrid) に展開しても図は変わらない。

    2 次元座標の描画経路 (transpose(y, x)・2 次元 X/Y の contourf / pcolormesh /
    contour / quiver) が 1 次元経路と同じ絵を出すことの検証。matplotlib は 1 次元
    X/Y を内部で meshgrid するので厳密一致を要求する。
    """
    p1, p2 = _regional_1d_and_2d(sample_path, tmp_path)
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 150.0}
    # 領域は両方に明示する (2 次元座標は領域未指定だと表示範囲が投影座標の範囲になり、
    # 1 次元格子の全球表示と一致しない)。1 次元は padded_slice、2 次元は外接矩形 + 1 格子
    # の切り出しで、この規則格子では同じ範囲になる
    panel["region"] = {"lon_min": 100.0, "lon_max": 200.0, "lat_min": 0.0, "lat_max": 60.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["selection"] = {"level": 500.0}
    fill["style"]["method"] = method
    cont = mc_config.default_contour_layer("ds0", "z")
    cont["selection"] = {"level": 500.0}
    vec = mc_config.default_vector_layer("ds0", "u", "v")
    vec["selection"] = {"level": 500.0}
    vec["style"]["stride_x"] = 2
    vec["style"]["stride_y"] = 2
    panel["layers"] = [fill, cont, vec]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    _assert_identical(cfg, p1, cfg, p2, tmp_path, f"1次元格子 vs 2次元座標展開 ({method})")


@pytest.mark.parametrize("method", ["contourf", "pcolormesh"])
def test_curvilinear_y_storage_order_equivalence(method, curvilinear_sample_path, tmp_path):
    """2 次元座標格子の y (行) の格納順を反転しても図は変わらない (経緯度が各点に付いて
    いるので順序に依存しない)。並べ替え系なので構造的一致で判定 (desc-lat と同じ)。"""
    ds = xr.open_dataset(curvilinear_sample_path)
    p_flip = tmp_path / "curvi_yflip.nc"
    ds.isel(y=slice(None, None, -1)).to_netcdf(p_flip)
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["projection"] = {"name": "LambertConformal", "central_longitude": 140.0,
                           "central_latitude": 30.0, "standard_parallels": [30.0, 60.0]}
    panel["region"] = None
    fill = mc_config.default_fill_layer("ds0", "z")
    fill["selection"] = {"lev": 500.0}
    fill["style"]["method"] = method
    cont = mc_config.default_contour_layer("ds0", "t")
    cont["selection"] = {"lev": 500.0}
    # 等値線ラベルの置き場所は経路の順序 (格納順) で変わるので外す
    cont["style"]["labels"] = {"show": False}
    panel["layers"] = [fill, cont]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    _assert_identical(cfg, curvilinear_sample_path, cfg, p_flip, tmp_path,
                      f"curvilinear y 反転 ({method})", exact=False)
