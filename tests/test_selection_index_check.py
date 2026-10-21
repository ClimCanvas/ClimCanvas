# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""描画前の selection 照合 (render.check_selections_in_index)。

別ファイルの変数をレイヤーに割り当てたとき、パネルで選んだ時刻 (ds0 の軸) が
そのファイルに無いと xarray は "not all values found in index 'time'" しか言わない
(2026-10-21 の実機報告)。描画の入口で各レイヤーの固定値を参照先 dataset の座標と
突き合わせ、どのパネル・レイヤー・ファイルに何が無いかを RenderError で示すことを固定する。
あわせて、レイヤー側の selection に時刻を入れれば別ファイルの層を別の時刻で描ける
(データモデルは既にレイヤー側優先) ことも確認する (docs/layer_time_fixed_plan.md Step 1)。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from climcanvas.core import config as mc_config
from climcanvas.core import render as mc_render


def _iso(value) -> str:
    import pandas as pd
    return pd.Timestamp(value).isoformat()


@pytest.fixture(scope="module")
def datasets(sample_path):
    """ds0 = サンプル (6 時間毎 8 時刻)、ds1 = 時刻を 3 時間ずらした複製 (ds0 と 1 つも一致しない)、
    ds2 = 鉛直レベルを先頭 2 つに絞った複製 (時刻軸は ds0 と同じ)。"""
    ds0 = xr.open_dataset(sample_path).load()
    ds1 = ds0.assign_coords(time=ds0["time"] + np.timedelta64(3, "h"))
    ds2 = ds0.isel(level=slice(0, 2))
    return {"ds0": ds0, "ds1": ds1, "ds2": ds2}


def _map_config(datasets, *layers, time=None, level=500.0):
    panel = mc_config.default_panel()
    panel["selection"] = {"time": time if time is not None else _iso(datasets["ds0"]["time"].values[1]),
                          "level": level}
    panel["layers"] = list(layers)
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_follower_layer_time_missing_in_second_file(datasets):
    """ds1 の等値線がパネルの時刻 (ds0 の軸) に従うと、ds1 にその時刻が無い → 層・ファイル・値つきのエラー。"""
    cfg = _map_config(datasets, mc_config.default_fill_layer("ds0", "t"),
                      mc_config.default_contour_layer("ds1", "z"))
    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.render_figure(cfg, datasets)
    e = ei.value
    assert e.msg_id == "selection_not_in_index"
    assert e.params["panel"] == 1 and e.params["layer"] == 2
    assert e.params["dataset_id"] == "ds1" and e.params["variable"] == "z"
    assert e.params["dim"] == "time" and e.params["value"] == cfg["panels"][0]["selection"]["time"]
    assert e.params["n"] == 8
    assert e.params["first"] == _iso(datasets["ds1"]["time"].values[0])
    assert e.params["last"] == _iso(datasets["ds1"]["time"].values[-1])
    # str() は英語で、層とファイルが読める
    assert "layer 2" in str(e) and "ds1" in str(e) and "time=" in str(e)


def test_layer_with_own_time_in_second_file_renders(datasets):
    """レイヤー側の selection に ds1 の時刻を入れれば、パネルの時刻と無関係に描ける (層側優先)。"""
    contour = mc_config.default_contour_layer("ds1", "z")
    contour["selection"] = {"time": _iso(datasets["ds1"]["time"].values[2])}
    cfg = _map_config(datasets, mc_config.default_fill_layer("ds0", "t"), contour)
    mc_render.check_selections_in_index(cfg, datasets)
    fig = mc_render.render_figure(cfg, datasets)
    plt.close(fig)


def test_vector_v_component_from_second_file_is_checked(datasets):
    """ベクトルの y 成分だけ別ファイル (v_dataset_id) のときも、その dataset の軸と照合する。"""
    vec = mc_config.default_vector_layer("ds0", "u", "v", v_dataset_id="ds1")
    cfg = _map_config(datasets, vec)
    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.check_selections_in_index(cfg, datasets)
    assert ei.value.params["variable"] == "v"
    assert ei.value.params["dataset_id"] == "ds1"
    assert ei.value.params["dim"] == "time"


def test_layer_level_missing_reports_level(datasets):
    """レイヤー側で固定した鉛直レベルがそのファイルに無いときは dim = level で報告する。"""
    contour = mc_config.default_contour_layer("ds2", "z")
    contour["selection"] = {"level": 500.0}   # ds2 は 1000 / 850 hPa のみ
    cfg = _map_config(datasets, mc_config.default_fill_layer("ds0", "t"), contour, level=1000.0)
    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.check_selections_in_index(cfg, datasets)
    p = ei.value.params
    assert p["dim"] == "level" and p["value"] == 500.0 and p["dataset_id"] == "ds2"
    assert p["n"] == 2 and p["first"] == "1000" and p["last"] == "850"


def test_partial_time_string_is_accepted(datasets):
    """xarray / pandas の部分文字列 (日付だけ) は描画側で通るので、照合でも弾かない。"""
    cfg = _map_config(datasets, mc_config.default_fill_layer("ds0", "t"), time="2024-01-02")
    assert mc_render.check_selections_in_index(cfg, datasets) is None


def test_same_time_axis_passes(datasets):
    """時刻軸が同じ 2 ファイル (ds0 / ds2) なら照合は何もしない。"""
    cfg = _map_config(datasets, mc_config.default_fill_layer("ds0", "t"),
                      mc_config.default_contour_layer("ds2", "z"), level=850.0)
    assert mc_render.check_selections_in_index(cfg, datasets) is None
    fig = mc_render.render_figure(cfg, datasets)
    plt.close(fig)


def test_animation_frame_reports_frame_time(datasets):
    """時刻送りでは各コマで照合し、別ファイルに無いコマの時刻をそのまま報告する。"""
    cfg = _map_config(datasets, mc_config.default_fill_layer("ds0", "t"),
                      mc_config.default_contour_layer("ds1", "z"))
    time_values = [_iso(v) for v in datasets["ds0"]["time"].values[:2]]
    frames = mc_render.iter_frames(cfg, datasets, time_values=time_values)
    with pytest.raises(mc_render.RenderError) as ei:
        next(frames)
    assert ei.value.msg_id == "selection_not_in_index"
    assert ei.value.params["value"] == time_values[0]


def test_heatmap_panel_selection_is_checked(datasets):
    """heatmap はレイヤーを持たないので panel 側の variable / selection を照合する (層番号は "-")。"""
    cfg = {"panels": [{"plot_type": "heatmap", "dataset_id": "ds1", "variable": "t",
                       "selection": {"time": _iso(datasets["ds0"]["time"].values[0])}}]}
    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.check_selections_in_index(cfg, datasets)
    assert ei.value.params["layer"] == "-" and ei.value.params["dataset_id"] == "ds1"


def test_missing_dataset_or_variable_is_left_to_other_errors(datasets):
    """dataset や変数が無い・index の無い次元・None の値は、この照合では見ない (別のエラーに任せる)。"""
    fill = mc_config.default_fill_layer("ds9", "t")
    cfg = _map_config(datasets, fill)
    assert mc_render.check_selections_in_index(cfg, datasets) is None
    fill2 = mc_config.default_fill_layer("ds0", "nonexistent")
    fill2["selection"] = {"time": None, "bogus_dim": 3}
    cfg = _map_config(datasets, fill2)
    assert mc_render.check_selections_in_index(cfg, datasets) is None


def test_panels_with_all_layers_time_fixed(datasets):
    """時刻を持つ全レイヤーが time_fixed のパネルだけを返す (時刻送りで変わらないパネルの警告用)。
    追従層が 1 つでもあれば対象外。時刻が描画軸のパネル (panel_animates_time が False) も対象外。"""
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["time_fixed"] = True
    fill["selection"] = {"time": _iso(datasets["ds0"]["time"].values[0])}
    contour = mc_config.default_contour_layer("ds1", "z")
    cfg = _map_config(datasets, fill, contour)
    assert mc_render.panels_with_all_layers_time_fixed(cfg, datasets, "time") == []
    contour["time_fixed"] = True
    contour["selection"] = {"time": _iso(datasets["ds1"]["time"].values[0])}
    assert mc_render.panels_with_all_layers_time_fixed(cfg, datasets, "time") == [1]
    # 2 パネル目は追従層だけ → 1 だけ返る
    cfg["panels"].append({**cfg["panels"][0], "layers": [mc_config.default_fill_layer("ds0", "t")]})
    assert mc_render.panels_with_all_layers_time_fixed(cfg, datasets, "time") == [1]
    # 時刻が軸のパネルは対象外
    cfg["panels"][0]["x_dim"] = "time"
    assert mc_render.panels_with_all_layers_time_fixed(cfg, datasets, "time") == []
