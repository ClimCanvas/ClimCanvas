# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""データ多様性コーパス: 現実の netCDF で遭遇する座標・値の変種への耐性検証。

全テストが理想的な合成データ (昇順座標・0-360経度・datetime64・欠損は NaN) しか
使っていない穴を塞ぐ。base サンプルを変換した変種 netCDF を生成し、それぞれで

    座標役割の検出 → render_figure → generate_script → 生成スクリプトの実行

が例外なく通ることを検証する (legacy corpus と同じ骨格)。変種:

- **desc-lat**   : 緯度が降順 (90 → -90)。ERA5 等の再解析で標準的
- **lon-180**    : 経度が -180 〜 177.5。0-360 前提の経路 (周期点付加など) の検証
- **pa-level**   : 気圧が Pa 単位 (100000 → 10000)。hPa 前提の判定の検証
- **raw-fill**   : _FillValue 属性なしの生の 1e20 を含む (クラッシュしないこと)
- **edge-values**: 先頭時刻が全 NaN の変数 + 定数場の変数
- **cftime-noleap**: noleap 暦 (CMIP6 標準)。時間軸モードの検証
- **cftime-360day**: 360_day 暦 (年をまたぐ daily)。開始年からの通し日数の
  数値軸 (1年目 1..360、2年目 361..720) への変換を検証
- **cftime-year0**: datetime64[ns] の範囲外 (0000年、proleptic_gregorian 暦)。
  変換可能な暦でも日付の範囲で datetime64 化に失敗するケース
  (EOT 解析出力等の静的ダミー時刻・古気候データ)。先頭時刻からの
  経過日数の数値軸へのフォールバックを検証
- **timedelta-lag**: timedelta 風 units ("days") の lag 次元座標 (ラグ相関解析の
  出力等)。xarray 既定のデコードで timedelta64 になると .sel も matplotlib 軸も
  通らない — decode_timedelta=False で生の数値のまま読むことを検証 (2026-09-07)

新しい「現実データで壊れた」事例が出たら、その特徴を再現する変種を
ここに追加すること。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_sample_data import create_sample_dataset  # noqa: E402

from climcanvas.core import config as mc_config  # noqa: E402
from climcanvas.core import dataset as mc_dataset  # noqa: E402
from climcanvas.core import render as mc_render  # noqa: E402
from climcanvas.core import scriptgen as mc_scriptgen  # noqa: E402


# ---------------------------------------------------------------- 変種の生成

def _make_desc_lat(base: xr.Dataset) -> xr.Dataset:
    return base.isel(lat=slice(None, None, -1))


def _make_lon180(base: xr.Dataset) -> xr.Dataset:
    lon180 = ((base["lon"].values + 180.0) % 360.0) - 180.0
    ds = base.assign_coords(lon=lon180).sortby("lon")
    ds["lon"].attrs.update(base["lon"].attrs)
    return ds


def _make_pa_level(base: xr.Dataset) -> xr.Dataset:
    ds = base.assign_coords(level=base["level"].values * 100.0)
    ds["level"].attrs.update(base["level"].attrs)
    ds["level"].attrs["units"] = "Pa"
    return ds


def _make_raw_fill(base: xr.Dataset) -> xr.Dataset:
    ds = base.copy(deep=True)
    t = ds["t"].values
    t[:, :, :8, :] = 1.0e20  # 南端の緯度帯を生の埋め値に (属性は付けない)
    ds["t"].values = t
    ds["t"].attrs.pop("_FillValue", None)
    ds["t"].attrs.pop("missing_value", None)
    return ds


def _make_edge_values(base: xr.Dataset) -> xr.Dataset:
    ds = base.copy(deep=True)
    t = ds["t"].values
    t[0] = np.nan  # 先頭時刻を全欠損に
    ds["t"].values = t
    ds["constv"] = xr.full_like(ds["precip"], 5.0)
    ds["constv"].attrs.update({"units": "1", "long_name": "constant field"})
    return ds


def _make_cftime_noleap(base: xr.Dataset) -> xr.Dataset:
    times = xr.date_range("2024-01-01", periods=base.sizes["time"],
                          freq="6h", calendar="noleap", use_cftime=True)
    return base.assign_coords(time=times)


def _make_cftime_360day(base: xr.Dataset) -> xr.Dataset:
    # 年またぎの daily (12月27日〜翌年1月4日、360_day 暦)
    times = xr.date_range("2024-12-27", periods=base.sizes["time"],
                          freq="D", calendar="360_day", use_cftime=True)
    return base.assign_coords(time=times)


def _make_cftime_year0(base: xr.Dataset) -> xr.Dataset:
    # 0000年起点の 6-hourly。実データと同じく「数値 + units/calendar 属性」で
    # 格納し、読み込み時の CF デコードで cftime (year 0) にする
    ds = base.assign_coords(time=np.arange(base.sizes["time"]) * 0.25)
    ds["time"].attrs.update({"units": "days since 0000-01-01 00:00:00",
                             "calendar": "proleptic_gregorian"})
    return ds


def _make_str_dim(base: xr.Dataset) -> xr.Dataset:
    # 文字列の次元座標 (IBTrACS 派生プロダクトの basin 等)。数値決め打ちの
    # UI/コアが float('NIO') で落ちた回帰 (2026-08-22)
    ds = base.copy()
    basins = np.array(["NIO", "WP", "SP"])
    n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
    vals = (np.arange(basins.size * n_lat * n_lon, dtype=np.float32)
            .reshape(basins.size, n_lat, n_lon))
    ds["basin_frac"] = (("basin", "lat", "lon"), vals,
                        {"long_name": "per-basin fraction"})
    return ds.assign_coords(basin=("basin", basins))


def _make_timedelta_lag(base: xr.Dataset) -> xr.Dataset:
    # timedelta 風 units ("days") の lag 次元座標。旧 xarray 既定ではこの座標が
    # timedelta64 にデコードされ、UI の .sel と matplotlib 軸の両方で扱えなかった
    # (「not all values found in index 'lag'」で描画失敗、2026-09-07)
    ds = base.copy()
    lags = np.arange(-2, 3)
    n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
    vals = (np.linspace(-1.0, 1.0, lags.size * n_lat * n_lon, dtype=np.float32)
            .reshape(lags.size, n_lat, n_lon))
    ds["lagcorr"] = (("lag", "lat", "lon"), vals,
                     {"long_name": "lag correlation", "units": "1"})
    return ds.assign_coords(
        lag=("lag", lags, {"units": "days", "long_name": "lag"}))


def _make_curvilinear(base: xr.Dataset) -> xr.Dataset:
    # 2 次元座標 (curvilinear): lat/lon の dim を格子番号 y/x に改名し、少し歪ませた
    # lon(y,x) / lat(y,x) を補助座標として付ける (海洋モデル / WRF / ClimCORE 形式)。
    # 歪みはランベルト等の既知の投影に当てはまらない形にする
    sub = base.sel(lon=slice(60.0, 200.0), lat=slice(-20.0, 70.0))
    lon2d, lat2d = np.meshgrid(sub["lon"].values, sub["lat"].values)
    lon2d = lon2d + 3.0 * np.sin(np.deg2rad(lat2d))
    lat2d = np.clip(lat2d + 2.0 * np.cos(np.deg2rad(lon2d)), -89.9, 89.9)
    ds = sub.rename({"lat": "y", "lon": "x"})
    return ds.assign_coords(
        y=("y", np.arange(1, sub.sizes["lat"] + 1, dtype=np.float32),
           {"long_name": "y grid number", "units": ""}),
        x=("x", np.arange(1, sub.sizes["lon"] + 1, dtype=np.float32),
           {"long_name": "x grid number", "units": ""}),
        lon=(("y", "x"), lon2d.astype(np.float32), dict(sub["lon"].attrs)),
        lat=(("y", "x"), lat2d.astype(np.float32), dict(sub["lat"].attrs)))


_VARIANTS = {
    "curvilinear": _make_curvilinear,
    "desc-lat": _make_desc_lat,
    "lon-180": _make_lon180,
    "pa-level": _make_pa_level,
    "raw-fill": _make_raw_fill,
    "edge-values": _make_edge_values,
    "cftime-noleap": _make_cftime_noleap,
    "cftime-360day": _make_cftime_360day,
    "cftime-year0": _make_cftime_year0,
    "str-dim": _make_str_dim,
    "timedelta-lag": _make_timedelta_lag,
}


@pytest.fixture(scope="session")
def variant_paths(tmp_path_factory) -> dict[str, str]:
    """base サンプルの変種 netCDF を生成してパスを返す (セッションで1回)。"""
    root = tmp_path_factory.mktemp("variants")
    base_path = root / "base.nc"
    create_sample_dataset(str(base_path))
    base = xr.open_dataset(str(base_path))
    paths = {}
    for name, maker in _VARIANTS.items():
        ds = maker(base)
        path = root / f"{name}.nc"
        ds.to_netcdf(str(path))
        ds.close()
        paths[name] = str(path)
    base.close()
    return paths


# ------------------------------------------------------------ 設定ビルダー

def _wrap(panel) -> dict:
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _map_cfg(var="t", level=500.0, time_sel="2024-01-01T06:00:00"):
    panel = mc_config.default_panel()
    panel["selection"] = {"time": time_sel}
    if level is not None:
        panel["selection"]["level"] = level
    fill = mc_config.default_fill_layer("ds0", var)
    panel["layers"] = [fill]
    panel["title"] = f"map {var}"
    return _wrap(panel)


def _vsec_cfg(x_dim="lon", fixed=None, log_y=False):
    panel = mc_config.default_section_panel()
    panel["x_dim"], panel["y_dim"] = x_dim, "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00", **(fixed or {})}
    panel["axis"].update({"invert_y": True, "log_y": log_y})
    panel["layers"] = [mc_config.default_fill_layer("ds0", "t")]
    panel["title"] = f"vsec {x_dim}-level"
    return _wrap(panel)


def _line_time_cfg(var="t", level=500.0):
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"level": level, "lat": 35.0, "lon": 140.0}
    layer = mc_config.default_line_layer("ds0", var)
    layer["style"].update({"color": "tab:red", "label": var})
    panel["layers"] = [layer]
    panel["title"] = f"time series {var}"
    return _wrap(panel)


def _hovmoller_cfg():
    panel = mc_config.default_section_panel()
    panel["x_dim"], panel["y_dim"] = "lat", "time"
    panel["selection"] = {"lon": 140.0}
    panel["axis"]["invert_y"] = True
    panel["layers"] = [mc_config.default_fill_layer("ds0", "precip")]
    panel["title"] = "hovmoller"
    return _wrap(panel)


# 変種 × 代表モードの組合せ。変種ごとに「壊れそうな経路」を優先して選ぶ
def _curvilinear_map_cfg():
    """2 次元座標の水平面図: fill + contour + region (isel の外接矩形) + 格子散布。"""
    cfg = _map_cfg()
    panel = cfg["panels"][0]
    panel["projection"] = {"name": "LambertConformal", "central_longitude": 130.0,
                           "central_latitude": 25.0, "standard_parallels": [20.0, 50.0]}
    panel["region"] = {"lon_min": 90.0, "lon_max": 170.0, "lat_min": 0.0, "lat_max": 55.0}
    cont = mc_config.default_contour_layer("ds0", "z")
    cont["selection"] = {"level": 500.0}
    ms = mc_config.default_map_scatter_layer("ds0", "sst")
    ms["style"]["use_cmap"] = False
    ms["style"]["size"] = 3
    panel["layers"] += [cont, ms]
    return cfg


def _curvilinear_no_region_cfg():
    """2 次元座標 + 領域未指定 (表示範囲 = 全格子点の投影座標の範囲) + ベクトル。"""
    cfg = _map_cfg()
    panel = cfg["panels"][0]
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 130.0}
    panel["region"] = None
    vec = mc_config.default_vector_layer("ds0", "u", "v")
    vec["selection"] = {"level": 500.0}
    vec["style"]["stride_x"] = 3
    vec["style"]["stride_y"] = 3
    panel["layers"].append(vec)
    return cfg


_CASES = [
    # curvilinear: lat/lon が 2 次元の補助座標 (dim は y/x)。region は isel の外接矩形
    ("curvilinear", "map", _curvilinear_map_cfg),
    ("curvilinear", "map-no-region", _curvilinear_no_region_cfg),
    ("desc-lat", "map", lambda: _map_cfg()),
    ("desc-lat", "vsec-lat", lambda: _vsec_cfg("lat", {"lon": 140.0})),
    ("desc-lat", "hovmoller", _hovmoller_cfg),
    ("lon-180", "map", lambda: _map_cfg()),
    ("lon-180", "vsec-lon", lambda: _vsec_cfg("lon", {"lat": 35.0})),
    ("pa-level", "vsec-log", lambda: _vsec_cfg("lon", {"lat": 35.0},
                                               log_y=True)),
    ("pa-level", "map", lambda: _map_cfg(level=50000.0)),
    ("raw-fill", "map", lambda: _map_cfg()),
    ("edge-values", "map-allnan", lambda: _map_cfg()),  # 先頭時刻は全NaN
    ("edge-values", "map-const", lambda: _map_cfg(var="constv", level=None)),
    ("cftime-noleap", "line-time", lambda: _line_time_cfg()),
    ("cftime-noleap", "hovmoller", _hovmoller_cfg),
    ("cftime-noleap", "map", lambda: _map_cfg()),
    # 360_day: 通し日数の数値軸。map は day 358 (=2024-12-28) を数値で選択
    ("cftime-360day", "line-time", lambda: _line_time_cfg()),
    ("cftime-360day", "hovmoller", _hovmoller_cfg),
    ("cftime-360day", "map", lambda: _map_cfg(time_sel=358.0)),
    # year0: 経過日数の数値軸。map は 0.25 日目 (2番目の時刻) を数値で選択
    ("cftime-year0", "line-time", lambda: _line_time_cfg()),
    ("cftime-year0", "map", lambda: _map_cfg(time_sel=0.25)),
    # str-dim: 文字列座標 basin を .sel(basin='NIO') で固定して水平面図
    ("str-dim", "map-basin", lambda: _basin_map_cfg()),
    # timedelta-lag: units="days" の lag 座標を数値のまま .sel(lag=-2) で固定
    ("timedelta-lag", "map-lag", lambda: _lag_map_cfg()),
]


def _basin_map_cfg():
    cfg = _map_cfg(var="basin_frac", level=None)
    cfg["panels"][0]["selection"]["basin"] = "NIO"
    return cfg


def _lag_map_cfg():
    cfg = _map_cfg(var="lagcorr", level=None)
    # UI (dim_choices) は数値座標を float で selection に入れるので同じ形にする
    cfg["panels"][0]["selection"]["lag"] = -2.0
    return cfg


def _allnan_map_cfg():
    cfg = _map_cfg()
    cfg["panels"][0]["selection"]["time"] = "2024-01-01T00:00:00"  # 全NaN時刻
    return cfg


# edge-values の map-allnan は先頭時刻 (全NaN) を明示的に選ぶ
_CASES = [(v, m, (_allnan_map_cfg if (v, m) == ("edge-values", "map-allnan")
                  else b)) for v, m, b in _CASES]


@pytest.mark.parametrize("variant,mode,builder", _CASES,
                         ids=[f"{v}-{m}" for v, m, _ in _CASES])
def test_variant_renders_and_scripts(variant, mode, builder,
                                     variant_paths, tmp_path):
    """変種データで render → scriptgen → スクリプト実行が通ること。"""
    path = variant_paths[variant]
    ds = mc_dataset.open_dataset(path)

    # 座標役割が全て検出できること
    roles = mc_dataset.detect_coord_roles(ds)
    for role in ("lat", "lon", "vertical", "time"):
        assert roles[role] is not None, f"{variant}: {role} 役割が未検出"

    cfg = builder()
    datasets = {"ds0": ds}
    fig = mc_render.render_figure(cfg, datasets)
    plt.close(fig)

    out_png = tmp_path / "variant.png"
    script = mc_scriptgen.generate_script(
        cfg, datasets, {"ds0": path},
        figure_output=str(out_png), figure_dpi=100, include_show=False)
    script_path = tmp_path / "repro.py"
    script_path.write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script_path)],
                            capture_output=True, text=True,
                            cwd=tmp_path, env=env, timeout=180)
    assert result.returncode == 0, result.stderr[-1500:]
    assert out_png.exists()


@pytest.mark.parametrize("variant,builder", [
    ("curvilinear", _curvilinear_map_cfg),
    ("desc-lat", _map_cfg),
    ("cftime-noleap", _line_time_cfg),
    ("cftime-360day", _line_time_cfg),
    ("cftime-year0", _line_time_cfg),
    ("timedelta-lag", _lag_map_cfg),
], ids=["curvilinear-map", "desc-lat-map", "cftime-line", "cftime-360day-line",
        "cftime-year0-line", "timedelta-lag-map"])
def test_variant_pixel_match(variant, builder, variant_paths, tmp_path):
    """代表変種でアプリ描画と生成スクリプトのピクセル一致 (ルール1) も確認。

    cftime は読み込み時の暦変換 (dataset._convert_cftime_calendar) を
    生成スクリプト側の convert_calendar 行が正確に再現することの検証になる。
    """
    from test_consistency import _assert_app_and_script_match
    _assert_app_and_script_match(builder(), variant_paths[variant], tmp_path)


def test_timedelta_lag_as_time_role(variant_paths):
    """timedelta-lag 変種で時刻役割を lag に上書きすると時刻送り経路が使える。

    UI の「時刻として扱う次元」が付ける ROLE_TIME_OVERRIDE_ATTR を core が読み、
    アニメーションの時刻次元決定と時刻送りフレーム描画が lag で動くこと。
    """
    ds = mc_dataset.open_dataset(variant_paths["timedelta-lag"]).assign_attrs(
        {mc_dataset.ROLE_TIME_OVERRIDE_ATTR: "lag"})
    assert mc_dataset.detect_coord_roles(ds)["time"] == "lag"
    cfg = _lag_map_cfg()
    datasets = {"ds0": ds}
    assert mc_render.animation_time_dim(cfg, datasets) == "lag"
    n = 0
    for fig in mc_render.iter_time_frames(cfg, datasets, [-2, -1, 0]):
        plt.close(fig)
        n += 1
    assert n == 3


def test_station_map_scatter_renders_and_matches(tmp_path):
    """単一観測点の時系列 (lon/lat がスカラー補助座標、変数の次元は time のみ、
    AMEDAS 型) を地図散布図で描く — render → scriptgen → 実行 → ピクセル一致。

    格子変数ゼロのデータで map_scatter の地点分岐を通す回帰テスト
    (時刻を panel.selection で固定すれば1点が描けること)。
    """
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(4) * np.timedelta64(1, "D"))
    ds = xr.Dataset(
        {"tmean": (("time",), np.array([1.0, 2.5, 3.0, 4.0]))},
        coords={"time": time,
                "lat": ((), 33.94833, {"units": "degrees_north"}),
                "lon": ((), 130.925, {"units": "degrees_east"})})
    path = str(tmp_path / "station.nc")
    ds.to_netcdf(path)

    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-02"}
    # min == max の退化領域 (UI の既定値が1点データで作る形)。
    # extent_args のガードで ±1° に広がる — scriptgen 側も同じ値になること
    panel["region"] = {"lon_min": 130.925, "lon_max": 130.925,
                       "lat_min": 33.94833, "lat_max": 33.94833}
    panel["projection"] = {"name": "PlateCarree",
                           "central_longitude": 130.925,
                           "central_latitude": 0.0}
    panel["layers"] = [mc_config.default_map_scatter_layer("ds0", "tmean")]
    panel["title"] = "station scatter"
    cfg = _wrap(panel)

    from test_consistency import _assert_app_and_script_match
    _assert_app_and_script_match(cfg, path, tmp_path)


def test_360day_daynum_values(variant_paths):
    """360_day 暦は開始年からの通し日数になる (年またぎで 360 → 361)。"""
    ds = mc_dataset.open_dataset(variant_paths["cftime-360day"])
    vals = [float(v) for v in ds["time"].values]
    # 2024-12-27..30 は day 357..360、翌年 1-01..04 は day 361..364
    assert vals == [357.0, 358.0, 359.0, 360.0, 361.0, 362.0, 363.0, 364.0]
    assert ds["time"].attrs["units"] == "day"
    assert ds.attrs[mc_dataset.CALENDAR_CONVERTED_ATTR] == "time:360_day:daynum"
    # 役割検出は名前ベースなので数値軸でも time のまま
    roles = mc_dataset.detect_coord_roles(ds)
    assert roles["time"] == "time"
    ds.close()


def test_year0_elapsed_values(variant_paths):
    """datetime64[ns] 範囲外の日付は先頭時刻からの経過日数になる。"""
    ds = mc_dataset.open_dataset(variant_paths["cftime-year0"])
    vals = [float(v) for v in ds["time"].values]
    assert vals == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    assert ds["time"].attrs["units"] == "day"
    assert ds["time"].attrs["long_name"] == (
        "time (days since 0000-01-01T00:00:00, proleptic_gregorian calendar)")
    assert (ds.attrs[mc_dataset.CALENDAR_CONVERTED_ATTR]
            == "time:proleptic_gregorian:elapsed")
    roles = mc_dataset.detect_coord_roles(ds)
    assert roles["time"] == "time"
    ds.close()
