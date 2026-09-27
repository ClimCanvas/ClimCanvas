# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
from climcanvas.core import dataset as mc_dataset


def test_detect_coord_roles(sample_path):
    ds = mc_dataset.open_dataset(sample_path)
    roles = mc_dataset.detect_coord_roles(ds)
    assert roles == {"lat": "lat", "lon": "lon", "vertical": "level", "time": "time"}


def test_role_time_override(sample_path):
    """時刻役割の手動上書き (ROLE_TIME_OVERRIDE_ATTR)。

    - 上書きは自動判定 (datetime64 の time) より優先される
    - 上書きされた座標は他の役割の自動判定から除外される
    - 存在しない座標名の上書きは無視して自動判定に戻る
    """
    ds = mc_dataset.open_dataset(sample_path)

    # level (自動では vertical) を時刻扱いに → time = level、vertical は空く
    over = ds.assign_attrs({mc_dataset.ROLE_TIME_OVERRIDE_ATTR: "level"})
    roles = mc_dataset.detect_coord_roles(over)
    assert roles["time"] == "level"
    assert roles["vertical"] is None
    # 自動の時刻 (datetime64 の time 座標) は上書きに勝てない
    assert roles["lat"] == "lat" and roles["lon"] == "lon"

    # 存在しない座標名 → 無視して自動判定
    bad = ds.assign_attrs({mc_dataset.ROLE_TIME_OVERRIDE_ATTR: "nope"})
    assert mc_dataset.detect_coord_roles(bad)["time"] == "time"


def test_variable_summary(sample_path):
    ds = mc_dataset.open_dataset(sample_path)
    rows = {row["変数名"]: row for row in mc_dataset.variable_summary(ds)}
    assert rows["t"]["単位"] == "K"
    assert rows["t"]["long_name"] == "air temperature"
    assert rows["t"]["次元"] == "time × level × lat × lon"
    assert rows["precip"]["次元"] == "time × lat × lon"


def test_horizontal_map_variables(sample_path):
    ds = mc_dataset.open_dataset(sample_path)
    assert set(mc_dataset.horizontal_map_variables(ds)) == {"t", "z", "u", "v", "precip", "sst"}


def test_coord_summary_and_bare_dims():
    """座標変数あり / bare dims で coord_summary・bare_dims が区別できる。"""
    import numpy as np
    import xarray as xr

    arr = np.zeros((4, 3), dtype=np.float32)
    # 座標変数あり
    with_coords = xr.Dataset(
        {"A": (("x", "y"), arr)},
        coords={"x": ("x", np.arange(4), {"units": "1", "long_name": "x idx"}),
                "y": ("y", np.arange(3))})
    rows = {r["座標名"]: r for r in mc_dataset.coord_summary(with_coords)}
    assert set(rows) == {"x", "y"}
    assert rows["x"]["長さ"] == "4"
    assert rows["x"]["範囲"] == "0 .. 3"
    assert rows["x"]["long_name"] == "x idx"
    assert mc_dataset.bare_dims(with_coords) == []

    # bare dims (座標変数なし)
    bare = xr.Dataset({"A": (("x", "y"), arr)})
    assert mc_dataset.coord_summary(bare) == []
    assert set(mc_dataset.bare_dims(bare)) == {"x", "y"}


def test_curvilinear_roles_and_horizontal_dims(curvilinear_sample_path, sample_path):
    """2 次元座標 (lon(y,x) / lat(y,x)) の役割検出と、座標名 / 水平 dim 名の区別。"""
    ds = mc_dataset.open_dataset(curvilinear_sample_path)
    roles = mc_dataset.detect_coord_roles(ds)
    assert roles == {"lat": "lat", "lon": "lon", "vertical": "lev", "time": "time"}
    assert mc_dataset.is_curvilinear(ds, roles)
    assert mc_dataset.horizontal_dims(ds, roles) == ("y", "x")
    # 水平面図の変数 = y/x 両方を持つ変数 (座標名 lat/lon は dim ではない)
    assert mc_dataset.horizontal_map_variables(ds, roles) == ["t", "z", "u", "v", "t2"]
    assert mc_dataset.map_scatter_variables(ds, roles) == ["t", "z", "u", "v", "t2"]
    # 非地理データ (集計モード用) には水平 dim を持つ変数を出さない
    assert mc_dataset.nongeo_variables(ds, roles) == []

    # 1 次元格子では従来どおり (座標名 = dim 名)
    ds1 = mc_dataset.open_dataset(sample_path)
    roles1 = mc_dataset.detect_coord_roles(ds1)
    assert not mc_dataset.is_curvilinear(ds1, roles1)
    assert mc_dataset.horizontal_dims(ds1, roles1) == ("lat", "lon")


def test_horizontal_dims_requires_both_roles_and_matching_dims(curvilinear_sample_path):
    ds = mc_dataset.open_dataset(curvilinear_sample_path)
    # lon 座標が無ければ None / 非 curvilinear
    ds_nolon = ds.drop_vars("lon")
    roles = mc_dataset.detect_coord_roles(ds_nolon)
    assert roles["lon"] is None
    assert mc_dataset.horizontal_dims(ds_nolon, roles) is None
    assert not mc_dataset.is_curvilinear(ds_nolon, roles)
    # lat と lon の dims が食い違う (片方だけ転置) ものは curvilinear とみなさない
    ds_mixed = ds.assign_coords(lon=(("x", "y"), ds["lon"].values.T, dict(ds["lon"].attrs)))
    assert tuple(ds_mixed["lon"].dims) == ("x", "y")
    assert not mc_dataset.is_curvilinear(ds_mixed)


def _split_lonlat(lonlat_path, tmp_path):
    """FLON / FLAT を別ファイルに分ける (ClimCORE の const/FLON.nc, FLAT.nc 形式)。"""
    import xarray as xr
    ll = xr.open_dataset(lonlat_path)
    p_lon, p_lat = tmp_path / "FLON.nc", tmp_path / "FLAT.nc"
    ll[["FLON"]].to_netcdf(p_lon)
    ll[["FLAT"]].to_netcdf(p_lat)
    return str(p_lon), str(p_lat)


def test_attach_coord_files(curvilinear_bare_paths, curvilinear_sample_path, tmp_path):
    """座標ファイルの結び付け: 1 ファイル / 2 ファイル分離、attrs の記録、値は埋め込み版と同じ。"""
    import json
    import numpy as np
    bare_path, lonlat_path = curvilinear_bare_paths
    bare = mc_dataset.open_dataset(bare_path)
    assert mc_dataset.detect_coord_roles(bare)["lat"] is None
    assert mc_dataset.horizontal_map_variables(bare) == []

    # 1 ファイル (FLON と FLAT が同居)
    ds = mc_dataset.attach_coord_files(
        bare, [(lonlat_path, mc_dataset.open_coord_file(lonlat_path))])
    roles = mc_dataset.detect_coord_roles(ds)
    assert roles["lon"] == "FLON" and roles["lat"] == "FLAT"
    assert mc_dataset.is_curvilinear(ds, roles)
    assert mc_dataset.horizontal_dims(ds, roles) == ("y", "x")
    assert mc_dataset.horizontal_map_variables(ds, roles) == ["t", "z", "u", "v", "t2"]
    rec = json.loads(ds.attrs[mc_dataset.COORD_FILES_ATTR])
    assert rec == [[lonlat_path, "FLON"], [lonlat_path, "FLAT"]]
    emb = mc_dataset.open_dataset(curvilinear_sample_path)
    assert np.array_equal(ds["FLON"].values, emb["lon"].values)
    assert np.array_equal(ds["FLAT"].values, emb["lat"].values)
    assert ds["FLON"].attrs["units"] == "degrees_east"
    # 元の dataset は変更されない (キャッシュ共有のため)
    assert "FLON" not in bare.coords and mc_dataset.COORD_FILES_ATTR not in bare.attrs

    # 2 ファイル分離 (経度と緯度が別ファイル)
    p_lon, p_lat = _split_lonlat(lonlat_path, tmp_path)
    ds2 = mc_dataset.attach_coord_files(
        bare, [(p_lon, mc_dataset.open_coord_file(p_lon)),
               (p_lat, mc_dataset.open_coord_file(p_lat))])
    assert json.loads(ds2.attrs[mc_dataset.COORD_FILES_ATTR]) == [[p_lon, "FLON"], [p_lat, "FLAT"]]
    assert np.array_equal(ds2["FLAT"].values, emb["lat"].values)

    # 既に経緯度がある dataset には何もしない (attrs も付かない)
    same = mc_dataset.attach_coord_files(
        emb, [(lonlat_path, mc_dataset.open_coord_file(lonlat_path))])
    assert same is emb


def test_attach_coord_files_errors(curvilinear_bare_paths, tmp_path):
    """lon/lat が見つからない・次元が無い・長さが違う座標ファイルは ValueError。"""
    import pytest
    import xarray as xr
    bare_path, lonlat_path = curvilinear_bare_paths
    bare = mc_dataset.open_dataset(bare_path)
    ll = mc_dataset.open_coord_file(lonlat_path)
    # 経度だけ (緯度が無い)
    with pytest.raises(ValueError, match="lat"):
        mc_dataset.attach_coord_files(bare, [(lonlat_path, ll[["FLON"]])])
    # 長さ不一致 (x を半分に切った座標ファイル)
    with pytest.raises(ValueError, match="points along"):
        mc_dataset.attach_coord_files(bare, [(lonlat_path, ll.isel(x=slice(0, 30)))])
    # dataset に無い次元
    bad = xr.Dataset({"FLON": (("j", "i"), ll["FLON"].values, ll["FLON"].attrs),
                      "FLAT": (("j", "i"), ll["FLAT"].values, ll["FLAT"].attrs)})
    with pytest.raises(ValueError, match="dimension"):
        mc_dataset.attach_coord_files(bare, [(lonlat_path, bad)])


def test_lonlat_bounds(sample_path, curvilinear_sample_path):
    """経緯度範囲: 全球格子は全周 + 全球判定、領域データ (1 次元の切り出し / 2 次元座標) は
    データの min/max、経度が全周で緯度が半球なら lon_global だけ True。"""
    import numpy as np
    ds = mc_dataset.open_dataset(sample_path)
    b = mc_dataset.lonlat_bounds(ds)
    assert b["global"] and b["lon_global"]
    assert (b["lon_min"], b["lon_max"]) == (0.0, 360.0)  # 0..357.5 → 規約の全周
    assert (b["lat_min"], b["lat_max"]) == (-90.0, 90.0)

    # 北半球だけ (経度は全周)
    nh = mc_dataset.lonlat_bounds(ds.sel(lat=slice(0.0, 90.0)))
    assert nh["lon_global"] and not nh["global"]
    assert (nh["lat_min"], nh["lat_max"]) == (0.0, 90.0)

    # 領域の切り出し (経度も部分)
    reg = mc_dataset.lonlat_bounds(ds.sel(lon=slice(100.0, 200.0), lat=slice(0.0, 60.0)))
    assert not reg["lon_global"] and not reg["global"]
    assert (reg["lon_min"], reg["lon_max"]) == (100.0, 200.0)

    # -180 規約の全球
    lon180 = ((ds["lon"].values + 180.0) % 360.0) - 180.0
    ds180 = ds.assign_coords(lon=lon180).sortby("lon")
    ds180["lon"].attrs.update(ds["lon"].attrs)
    b180 = mc_dataset.lonlat_bounds(ds180)
    assert b180["global"] and (b180["lon_min"], b180["lon_max"]) == (-180.0, 180.0)

    # 2 次元座標: 全格子点を含む矩形。合成サンプルの高緯度側は 180° を越えて -180 規約
    # に折り返している (min/max だと全周に化ける) ので、経度は日付変更線をまたぐ弧
    cv = mc_dataset.open_dataset(curvilinear_sample_path)
    cb = mc_dataset.lonlat_bounds(cv)
    assert not cb["lon_global"] and not cb["global"]
    assert float(np.nanmin(cv["lon"].values)) < -179.0  # 折り返しがあること (前提の確認)
    assert 90.0 < cb["lon_min"] < 100.0 and 180.0 < cb["lon_max"] < 190.0
    lon_mod = np.mod(cv["lon"].values, 360.0)
    assert ((lon_mod >= cb["lon_min"]) & (lon_mod <= cb["lon_max"])).all()
    assert cb["lat_max"] == float(np.nanmax(cv["lat"].values))

    # 1 次元格子で日付変更線をまたぐ領域 (-180 規約で 140〜180 と -180〜-140)
    pac = ds.sel(lon=slice(140.0, 220.0))
    pac_lon = ((pac["lon"].values + 180.0) % 360.0) - 180.0
    pac = pac.assign_coords(lon=pac_lon).sortby("lon")
    pac["lon"].attrs.update(ds["lon"].attrs)
    pb = mc_dataset.lonlat_bounds(pac)
    assert (pb["lon_min"], pb["lon_max"]) == (140.0, 220.0) and not pb["lon_global"]

    # lat/lon 役割が無ければ None
    assert mc_dataset.lonlat_bounds(ds.drop_vars(["lat", "lon"])) is None


def test_detect_grid_projection(curvilinear_sample_path, sample_path, tmp_path):
    """2 次元座標格子の投影法の推定: LCC 格子 (30/60°N、140°E) を当て、経緯度格子は棄却、
    南半球 LCC と極ステレオも判別、1 次元格子は None。"""
    import numpy as np
    import xarray as xr
    import cartopy.crs as ccrs
    cv = mc_dataset.open_dataset(curvilinear_sample_path)
    gp = mc_dataset.detect_grid_projection(cv)
    assert gp is not None and gp["name"] == "LambertConformal"
    assert gp["standard_parallels"] == [30.0, 60.0]
    assert abs(gp["central_longitude"] - 140.0) < 0.05
    assert gp["misfit"] < 1e-3

    # 1 次元格子は対象外
    assert mc_dataset.detect_grid_projection(mc_dataset.open_dataset(sample_path)) is None

    # 経緯度格子を 2 次元座標に展開したもの (どの LCC でも直交等間隔にならない) → None
    ds = xr.open_dataset(sample_path).sel(lon=slice(100.0, 200.0), lat=slice(0.0, 60.0))
    lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    flat = ds.rename({"lat": "y", "lon": "x"}).assign_coords(
        y=("y", np.arange(ds.sizes["lat"], dtype=float)),
        x=("x", np.arange(ds.sizes["lon"], dtype=float)),
        lon=(("y", "x"), lon2d, dict(ds["lon"].attrs)),
        lat=(("y", "x"), lat2d, dict(ds["lat"].attrs)))
    assert mc_dataset.detect_grid_projection(flat) is None

    def _grid_from(proj, nx=50, ny=40, dx=100e3, y_off=0.0):
        xx, yy = np.meshgrid((np.arange(nx) - (nx - 1) / 2) * dx,
                             (np.arange(ny) - (ny - 1) / 2) * dx + y_off)
        ll = ccrs.PlateCarree().transform_points(proj, xx, yy)
        return xr.Dataset(
            {"v": (("y", "x"), np.zeros((ny, nx)))},
            coords={"lon": (("y", "x"), ll[..., 0], {"units": "degrees_east"}),
                    "lat": (("y", "x"), ll[..., 1], {"units": "degrees_north"})})

    # 南半球の LCC (標準緯線 -30 / -60、中心経度 150)
    sh = _grid_from(ccrs.LambertConformal(central_longitude=150.0, central_latitude=-30.0,
                                          standard_parallels=(-30.0, -60.0), cutoff=30),
                    y_off=-500e3)
    gp = mc_dataset.detect_grid_projection(sh)
    assert gp["name"] == "LambertConformal" and gp["standard_parallels"] == [-30.0, -60.0]
    assert abs(gp["central_longitude"] - 150.0) < 0.05

    # 極ステレオ (n = 1 の極限)
    ps = _grid_from(ccrs.NorthPolarStereo(central_longitude=0.0), y_off=-3000e3)
    gp = mc_dataset.detect_grid_projection(ps)
    assert gp["name"] == "NorthPolarStereo" and abs(gp["central_longitude"]) < 0.05
