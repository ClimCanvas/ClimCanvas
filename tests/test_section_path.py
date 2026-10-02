# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""経路断面の純関数のユニットテスト。

render.great_circle_points / grid_fractional_indices / grid_fractional_indices_1d /
sample_bilinear を検証する (docs/section_extension_plan.md Step 1)。2 次元座標格子の
格子番号は、cartopy の投影で「真の小数格子番号 → 経緯度」を作り、そこから逆算した値と
比べる (関数自体は投影法を使わないので独立な検算になる)。
"""

import inspect

import cartopy.crs as ccrs
import numpy as np
import pytest
import xarray as xr

from climcanvas.core import render as mc_render

_R_KM = 6371.0


def _unit(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
                    axis=-1)


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * _R_KM * np.arcsin(np.sqrt(h))


def _projected_grid(proj, nx, ny, dx, x_offset=0.0, y_offset=0.0):
    """投影面で等間隔な格子 (ny, nx) の経緯度と、小数格子番号 → 経緯度の関数を返す。"""
    x0 = -(nx - 1) / 2 * dx + x_offset
    y0 = -(ny - 1) / 2 * dx + y_offset
    xx, yy = np.meshgrid(x0 + np.arange(nx) * dx, y0 + np.arange(ny) * dx)
    ll = ccrs.PlateCarree().transform_points(proj, xx, yy)

    def lonlat_of(fj, fi):
        q = ccrs.PlateCarree().transform_points(
            proj, x0 + np.asarray(fi) * dx, y0 + np.asarray(fj) * dx)
        return q[:, 0], q[:, 1]

    return ll[..., 0], ll[..., 1], lonlat_of


def _lambert(central_longitude=140.0):
    return ccrs.LambertConformal(central_longitude=central_longitude,
                                 central_latitude=30.0, standard_parallels=(30.0, 60.0))


# --- 大円 ---

def test_great_circle_endpoints_spacing_and_plane():
    start, end = (125.0, 30.0), (145.0, 42.0)
    lon, lat, dist = mc_render.great_circle_points(start, end, 41)
    assert lon.shape == lat.shape == dist.shape == (41,)
    assert (lon[0], lat[0]) == pytest.approx(start)
    assert (lon[-1], lat[-1]) == pytest.approx(end)
    # 距離 = 始点からの大円距離 (独立に haversine で計算)
    assert dist[0] == 0.0
    assert dist == pytest.approx(_haversine_km(start[0], start[1], lon, lat), rel=1e-9)
    assert dist[-1] == pytest.approx(_haversine_km(*start, *end), rel=1e-12)
    # 等間隔
    step = _haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
    assert step == pytest.approx(np.full(40, dist[-1] / 40), rel=1e-9)
    # 全点が始点・終点を通る大円の面上にある
    normal = np.cross(_unit(*start), _unit(*end))
    assert np.abs(_unit(lon, lat) @ normal).max() < 1e-12


def test_great_circle_longitudes_are_continuous_across_dateline():
    lon, lat, _ = mc_render.great_circle_points((170.0, 10.0), (-170.0, 20.0), 50)
    assert np.all(np.diff(lon) > 0)            # 東向きに単調 (179 → -179 と跳ばない)
    assert lon[-1] == pytest.approx(190.0)
    assert lat[-1] == pytest.approx(20.0)
    # 西向き: 始点の経度の規約のまま連続
    lon, _, _ = mc_render.great_circle_points((190.0, 0.0), (160.0, 0.0), 7)
    assert lon == pytest.approx(np.linspace(190.0, 160.0, 7))


def test_great_circle_over_the_pole_and_degenerate_paths():
    lon, lat, dist = mc_render.great_circle_points((0.0, 80.0), (180.0, 80.0), 21)
    assert lat.max() == pytest.approx(90.0)
    assert dist[-1] == pytest.approx(np.deg2rad(20.0) * _R_KM)
    # 始点 = 終点: 全点が同じ点、距離 0
    lon, lat, dist = mc_render.great_circle_points((135.0, 35.0), (135.0, 35.0), 5)
    assert np.allclose(lon, 135.0) and np.allclose(lat, 35.0) and np.all(dist == 0.0)
    # 対蹠点: 大円が一意に決まらない
    with pytest.raises(ValueError):
        mc_render.great_circle_points((0.0, 0.0), (180.0, 0.0), 5)


# --- 2 次元座標格子の小数格子番号 ---

def _random_indices(rng, nx, ny, n=200):
    return rng.uniform(0, ny - 1, n), rng.uniform(0, nx - 1, n)


@pytest.mark.parametrize("dx, tol", [(100.0e3, 2e-3), (5.0e3, 1e-4)],
                         ids=["100km", "5km"])
def test_fractional_indices_match_projection(dx, tol):
    """ランベルト格子 (合成サンプルと同じ投影): 投影から作った真の格子番号を復元する。
    手法は接平面での双一次の逆写像なので、誤差は格子間隔に比例して小さくなる
    (実測: 100 km で 8e-4、5 km で 3e-5 格子)。"""
    nx, ny = (60, 50) if dx > 50e3 else (817, 661)
    lon2d, lat2d, lonlat_of = _projected_grid(_lambert(), nx, ny, dx, y_offset=500e3)
    rng = np.random.default_rng(1)
    fj, fi = _random_indices(rng, nx, ny)
    lon, lat = lonlat_of(fj, fi)
    gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, lon, lat)
    assert np.abs(gj - fj).max() < tol
    assert np.abs(gi - fi).max() < tol
    # 格子点そのもの (角・辺を含む) はほぼ整数で返る
    corner = np.array([0, 0, ny - 1, ny - 1]), np.array([0, nx - 1, 0, nx - 1])
    gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, lon2d[corner], lat2d[corner])
    assert gj == pytest.approx(corner[0], abs=1e-6)
    assert gi == pytest.approx(corner[1], abs=1e-6)


def test_fractional_indices_across_dateline_any_lon_convention():
    """格子の経度が -180 規約で 180° をまたぐ (隣の列で 179 → -179) 格子。
    点の経度は 0–360 規約で与えても同じ格子番号になる。"""
    lon2d, lat2d, lonlat_of = _projected_grid(_lambert(180.0), 60, 50, 100e3,
                                              y_offset=500e3)
    lon2d = (lon2d + 180.0) % 360.0 - 180.0
    assert lon2d.max() > 170 and lon2d.min() < -170         # 継ぎ目が格子の中にある
    rng = np.random.default_rng(2)
    fj, fi = _random_indices(rng, 60, 50)
    lon, lat = lonlat_of(fj, fi)
    for lon_in in (lon, lon % 360.0):
        gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, lon_in, lat)
        assert np.abs(gj - fj).max() < 2e-3
        assert np.abs(gi - fi).max() < 2e-3


def test_fractional_indices_near_and_at_the_pole():
    """極ステレオ格子 (極が格子の中心)。極そのもの・極のすぐ近くでも格子番号が求まる。"""
    lon2d, lat2d, lonlat_of = _projected_grid(ccrs.NorthPolarStereo(), 41, 41, 100e3)
    rng = np.random.default_rng(3)
    fj = np.concatenate([rng.uniform(18, 22, 50), rng.uniform(0, 40, 50)])
    fi = np.concatenate([rng.uniform(18, 22, 50), rng.uniform(0, 40, 50)])
    lon, lat = lonlat_of(fj, fi)
    gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, lon, lat)
    assert np.abs(gj - fj).max() < 2e-3
    assert np.abs(gi - fi).max() < 2e-3
    gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, [0.0, 123.0], [90.0, 90.0])
    assert gj == pytest.approx([20.0, 20.0], abs=1e-6)
    assert gi == pytest.approx([20.0, 20.0], abs=1e-6)


def test_points_outside_the_grid_are_nan():
    lon2d, lat2d, lonlat_of = _projected_grid(_lambert(), 60, 50, 100e3, y_offset=500e3)
    # 境界のわずかに外 (格子番号 -0.3 / ny - 0.7) と、遠く離れた点
    lon, lat = lonlat_of(np.array([-0.3, 49.3, 20.0, 20.0]),
                         np.array([30.0, 30.0, -0.3, 59.3]))
    lon = np.append(lon, [60.0, -40.0])
    lat = np.append(lat, [0.0, -60.0])
    gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, lon, lat)
    assert np.all(np.isnan(gj)) and np.all(np.isnan(gi))
    # 境界ちょうどの点は格子の中として扱う
    lon, lat = lonlat_of(np.array([0.0, 49.0]), np.array([30.5, 12.25]))
    gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, lon, lat)
    assert gj == pytest.approx([0.0, 49.0], abs=1e-3)


def test_nan_coordinates_only_affect_neighbouring_cells():
    lon2d, lat2d, lonlat_of = _projected_grid(_lambert(), 60, 50, 100e3, y_offset=500e3)
    lon2d, lat2d = lon2d.copy(), lat2d.copy()
    lon2d[10:15, 10:15] = np.nan
    lat2d[10:15, 10:15] = np.nan
    fj = np.array([12.0, 12.5, 30.0, 40.2])
    fi = np.array([12.0, 13.5, 40.0, 5.7])
    lon, lat = lonlat_of(fj, fi)
    gj, gi = mc_render.grid_fractional_indices(lon2d, lat2d, lon, lat)
    assert np.isnan(gj[:2]).all() and np.isnan(gi[:2]).all()   # 欠けた経緯度の格子
    assert gj[2:] == pytest.approx(fj[2:], abs=2e-3)            # 離れた点は影響なし
    assert gi[2:] == pytest.approx(fi[2:], abs=2e-3)


def test_grid_with_one_row_is_rejected():
    with pytest.raises(ValueError):
        mc_render.grid_fractional_indices(np.zeros((1, 5)), np.zeros((1, 5)), [0.0], [0.0])


# --- 1 次元座標格子の小数格子番号 ---

_GLOBAL_LON = np.arange(0.0, 360.0, 2.5)          # 144 列、最後の列の次が 360 = 0
_GLOBAL_LAT = np.arange(90.0, -90.1, -2.5)        # 73 行、降順


def test_1d_indices_on_a_global_grid():
    fj, fi = mc_render.grid_fractional_indices_1d(
        _GLOBAL_LON, _GLOBAL_LAT, [10.0, -1.0, 358.75, 721.25, 0.0],
        [45.0, 89.0, -90.0, 0.0, 90.0])
    assert fj == pytest.approx([18.0, 0.4, 72.0, 36.0, 0.0])
    # -1° は 359° (最後の列 357.5° と 360° = 0° 列の間)。721.25° は 1.25°
    assert fi == pytest.approx([4.0, 143.6, 143.5, 0.5, 0.0])


def test_1d_indices_on_a_descending_global_lon_grid():
    lon = _GLOBAL_LON[::-1]                        # 357.5, 355, ..., 0
    _, fi = mc_render.grid_fractional_indices_1d(lon, _GLOBAL_LAT, [359.0, 10.0], [0.0, 0.0])
    # 359° は 357.5° (列 0) と 360° = 0° (列 143 ≡ -1) の間 → -0.6
    assert fi == pytest.approx([-0.6, 139.0])


def test_1d_indices_on_a_regional_grid():
    lon = np.arange(100.0, 160.1, 2.5)
    lat = np.arange(10.0, 50.1, 2.5)
    fj, fi = mc_render.grid_fractional_indices_1d(
        lon, lat, [-200.0, 170.0, 130.0, 130.0], [30.0, 30.0, 5.0, 50.0])
    assert fi[0] == pytest.approx(24.0)           # -200° = 160° (東端)
    assert np.isnan(fi[1])                        # 領域の外
    assert np.isnan(fj[2])
    assert fj[3] == pytest.approx(16.0)


def test_1d_and_2d_methods_agree_on_a_lonlat_grid():
    """経緯度格子を meshgrid で 2 次元にしても、ほぼ同じ格子番号になる (2 次元の手法は
    接平面での双一次なので完全一致ではない。実測 2.7e-3 格子 (2.5° 格子))。"""
    lon = np.arange(100.0, 160.1, 2.5)
    lat = np.arange(10.0, 50.1, 2.5)
    lon2d, lat2d = np.meshgrid(lon, lat)
    plon, plat, _ = mc_render.great_circle_points((105.0, 15.0), (155.0, 45.0), 60)
    fj1, fi1 = mc_render.grid_fractional_indices_1d(lon, lat, plon, plat)
    fj2, fi2 = mc_render.grid_fractional_indices(lon2d, lat2d, plon, plat)
    assert np.abs(fj2 - fj1).max() < 5e-3
    assert np.abs(fi2 - fi1).max() < 5e-3


# --- 双一次内挿 ---

def _bilinear_field(ny=8, nx=10):
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lev = np.array([1000.0, 850.0, 500.0])
    base = 2.0 + 3.0 * j - 1.0 * i + 0.5 * i * j
    data = base[None] + lev[:, None, None] * 0.01
    return xr.DataArray(
        data.astype(np.float32), dims=("lev", "y", "x"),
        coords={"lev": lev, "y": np.arange(ny), "x": np.arange(nx),
                "lon": (("y", "x"), np.zeros((ny, nx))),
                "lat": (("y", "x"), np.zeros((ny, nx)))},
        attrs={"units": "m"})


def test_sample_bilinear_reproduces_a_bilinear_field_exactly():
    da = _bilinear_field()
    rng = np.random.default_rng(4)
    fj, fi = rng.uniform(0, 7, 30), rng.uniform(0, 9, 30)
    fj[:2], fi[:2] = [0.0, 7.0], [0.0, 9.0]                  # 角 (端の格子点)
    out = mc_render.sample_bilinear(da, "y", "x", fj, fi, "path")
    assert out.dims == ("lev", "path")
    assert set(out.coords) == {"lev"}                         # 水平の座標は落とす
    assert out.attrs == {"units": "m"}
    expected = (2.0 + 3.0 * fj - fi + 0.5 * fi * fj)[None] + da["lev"].values[:, None] * 0.01
    np.testing.assert_allclose(out.values, expected, rtol=1e-6)


def test_sample_bilinear_nan_rule():
    """重みが正の角に NaN があれば NaN、重み 0 の角の NaN は影響しない。"""
    da = _bilinear_field().isel(lev=0).astype(float)
    da[2, 3] = np.nan
    fj = np.array([2.0, 2.0, 1.5, np.nan, 2.0])
    fi = np.array([2.0, 2.5, 2.0, 1.0, 3.0])
    out = mc_render.sample_bilinear(da, "y", "x", fj, fi, "path").values
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_sample_bilinear_wraps_on_a_global_grid():
    lon = _GLOBAL_LON
    da = xr.DataArray(np.cos(np.deg2rad(lon))[None, :] * np.ones((_GLOBAL_LAT.size, 1)),
                      dims=("lat", "lon"), coords={"lat": _GLOBAL_LAT, "lon": lon})
    fj, fi = mc_render.grid_fractional_indices_1d(lon, _GLOBAL_LAT, [359.0], [0.0])
    out = mc_render.sample_bilinear(da, "lat", "lon", fj, fi, "path", wrap_x=True)
    expected = 0.4 * np.cos(np.deg2rad(357.5)) + 0.6 * np.cos(0.0)
    assert float(out[0]) == pytest.approx(expected)
    # 降順の経度でも同じ値
    lon_d = lon[::-1]
    fj, fi = mc_render.grid_fractional_indices_1d(lon_d, _GLOBAL_LAT, [359.0], [0.0])
    out = mc_render.sample_bilinear(da.isel(lon=slice(None, None, -1)), "lat", "lon",
                                    fj, fi, "path", wrap_x=True)
    assert float(out[0]) == pytest.approx(expected)


# --- 再現スクリプトへの埋め込み (np / xr と引数だけで動くこと) ---

_EMBEDDED = ["great_circle_points", "grid_fractional_indices",
             "grid_fractional_indices_1d", "sample_bilinear"]


@pytest.mark.parametrize("name", _EMBEDDED)
def test_embedded_functions_are_self_contained(name):
    src = inspect.getsource(getattr(mc_render, name))
    assert src.isascii(), "埋め込む関数の docstring・コメントは英語 (ASCII) にする"
    ns = {"np": np, "xr": xr}
    exec(compile(src, f"<{name}>", "exec"), ns)
    lon2d, lat2d, lonlat_of = _projected_grid(_lambert(), 20, 15, 100e3, y_offset=500e3)
    lon, lat = lonlat_of(np.array([3.3, 7.1]), np.array([4.4, 12.9]))
    if name == "great_circle_points":
        args = ((125.0, 30.0), (145.0, 42.0), 9)
    elif name == "grid_fractional_indices":
        args = (lon2d, lat2d, lon, lat)
    elif name == "grid_fractional_indices_1d":
        args = (_GLOBAL_LON, _GLOBAL_LAT, [10.0, 359.0], [45.0, 0.0])
    else:
        args = (_bilinear_field(), "y", "x", np.array([1.5, 3.2]), np.array([2.5, 0.1]),
                "path")
    got = ns[name](*args)
    want = getattr(mc_render, name)(*args)
    if isinstance(want, tuple):
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)
    else:
        xr.testing.assert_identical(got, want)
