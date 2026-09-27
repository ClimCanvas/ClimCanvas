# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""座標整合性チェック (`align_dim_names`) の挙動。

複数 dataset を使うレイヤー構成で、軸座標の値・dim 名が一致するかを検査し、
値が一致し dim 名だけ違うときは canonical の dim 名に rename する。
"""

from __future__ import annotations


from climcanvas.core import dataset as mc_dataset
from climcanvas.core import config as mc_config
from climcanvas.core import scriptgen as mc_scriptgen


def test_align_same_names_same_values(sample_path):
    """同じ dataset を2つ → rename 不要、エラーなし。"""
    ds = mc_dataset.open_dataset(sample_path)
    datasets = {"ds0": ds, "ds1": ds}
    out, renames, info, errors = mc_dataset.align_dim_names(
        datasets, ["ds0", "ds1"], axis_roles=["lat", "lon"])
    assert renames == {}
    assert errors == []
    assert info == []
    assert "lat" in out["ds1"].dims  # 変わらない


def test_align_diff_names_same_values(sample_path):
    """dim 名を rename して名前違いを作成 → rename 提案、info メッセージ、エラーなし。"""
    ds = mc_dataset.open_dataset(sample_path)
    ds1 = ds.rename({"lat": "latitude"})
    datasets = {"ds0": ds, "ds1": ds1}
    out, renames, info, errors = mc_dataset.align_dim_names(
        datasets, ["ds0", "ds1"], axis_roles=["lat", "lon"])
    assert renames == {"ds1": {"latitude": "lat"}}
    assert errors == []
    assert len(info) == 1 and "latitude" in info[0] and "lat" in info[0]
    # 結果として ds1 は "lat" を持つ
    assert "lat" in out["ds1"].dims
    assert "latitude" not in out["ds1"].dims


def test_align_diff_values(sample_path):
    """lat の値をシフト → 値不一致は info (非ブロック)。errors には出さない。

    解像度・範囲が異なるデータは各レイヤーを各自の格子で重ね描きできるため、
    値不一致は描画をブロックしない (info として通知するだけ)。
    """
    ds = mc_dataset.open_dataset(sample_path)
    ds_shifted = ds.assign_coords(lat=ds["lat"] + 10.0)
    datasets = {"ds0": ds, "ds1": ds_shifted}
    out, renames, info, errors = mc_dataset.align_dim_names(
        datasets, ["ds0", "ds1"], axis_roles=["lat", "lon"])
    assert renames == {}
    assert errors == []
    assert any("格子が異なります" in m for m in info)


def test_align_diff_resolution(sample_path):
    """異なる水平解像度 (lat/lon の点数が違う) → 値不一致 info、エラーなし。

    内挿なしにレイヤーごとに異なる解像度を重ねられることの回帰テスト。
    """
    ds = mc_dataset.open_dataset(sample_path)
    ds_coarse = ds.isel(lat=slice(None, None, 2), lon=slice(None, None, 2))
    datasets = {"ds0": ds, "ds1": ds_coarse}
    out, renames, info, errors = mc_dataset.align_dim_names(
        datasets, ["ds0", "ds1"], axis_roles=["lat", "lon"])
    assert errors == []
    assert any("格子が異なります" in m for m in info)
    # 両 dataset はそのまま (内挿・リサンプルしない)
    assert out["ds0"]["lat"].size == ds["lat"].size
    assert out["ds1"]["lat"].size == ds_coarse["lat"].size


def test_align_missing_role(sample_path):
    """ds1 の lat を role 認識できない名前 + 属性削除 → 欠落エラー。"""
    ds = mc_dataset.open_dataset(sample_path)
    ds1 = ds.rename({"lat": "y_index"})
    # role 認識用属性をクリア
    ds1["y_index"].attrs = {}
    datasets = {"ds0": ds, "ds1": ds1}
    out, renames, info, errors = mc_dataset.align_dim_names(
        datasets, ["ds0", "ds1"], axis_roles=["lat", "lon"])
    assert any("lat" in e and "ありません" in e for e in errors)


def test_align_single_dataset(sample_path):
    """used_ids が1つだけ → 何もしない。"""
    ds = mc_dataset.open_dataset(sample_path)
    datasets = {"ds0": ds}
    out, renames, info, errors = mc_dataset.align_dim_names(
        datasets, ["ds0"], axis_roles=["lat", "lon"])
    assert renames == {}
    assert errors == []
    assert info == []


def test_align_renames_non_axis_role(sample_path):
    """axis_roles 外の役割でも、dim 名が違えば自動 rename する (selection 用)。

    回帰テスト: 経度-高度断面 (axis = lon/vertical) で ds1 の lat を latitude に
    リネームしたファイルを使うと、以前は lat が axis 外なので rename されず、
    selection の `.sel(lat=...)` が silent にスキップされ、transpose が失敗していた。
    """
    ds = mc_dataset.open_dataset(sample_path)
    ds1 = ds.rename({"lat": "latitude"})
    datasets = {"ds0": ds, "ds1": ds1}
    # 経度-高度 section の axis: lon と vertical のみ
    out, renames, info, errors = mc_dataset.align_dim_names(
        datasets, ["ds0", "ds1"], axis_roles=["lon", "vertical"])
    # lat も自動で rename される
    assert renames == {"ds1": {"latitude": "lat"}}
    assert "lat" in out["ds1"].dims
    assert errors == []


def test_script_emits_rename_call(sample_path):
    """scriptgen が rename map を受け取って `.rename(...)` 行を出力する。"""
    ds = mc_dataset.open_dataset(sample_path)
    datasets = {"ds0": ds, "ds1": ds}  # 中身は問わない
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    contour = mc_config.default_contour_layer("ds1", "t")
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    script = mc_scriptgen.generate_script(
        cfg, datasets, {"ds0": "a.nc", "ds1": "b.nc"},
        dataset_renames={"ds1": {"latitude": "lat"}})
    # rename は open の次の行に出る (cftime 変換行が間に入りうるため分離形式)
    assert "ds1 = xr.open_dataset('b.nc', decode_timedelta=False)" in script
    assert "ds1 = ds1.rename({'latitude': 'lat'})" in script
    assert "ds0 = xr.open_dataset('a.nc', decode_timedelta=False)" in script
    # rename 指定が無い ds0 には .rename() は付かない
    assert "ds0 = xr.open_dataset('a.nc').rename" not in script
