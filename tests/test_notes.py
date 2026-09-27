# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""図の直下の通知 (core.notes.collect_notes / ui.notes_ui.format_notes) のテスト。

docs/scientific_safeguard_plan.md A-1 の第 1 段階。通知は描画結果ではなく
設定から集めるので、ここでは

- 値の変換 (a, b) が config と一致し、恒等変換 (a=1, b=0) は出ないこと
- 範囲平均の格子数が numpy の独立計算 (render のヘルパーを使わない) と一致すること
- render が平均を適用しない場所 (水平面図・1 次元プロットの x 軸・束の次元・
  変数に無い次元) では通知も出ないこと
- 通知の a, b と平均の行が生成スクリプトにも同じ値で現れること (scriptgen との整合)
- 表示整形 (全角マイナス・パネル/レイヤーのまとめ方)

を固定する。
"""

import numpy as np
import pytest

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import notes as mc_notes
from climcanvas.core import scriptgen as mc_scriptgen
from climcanvas.ui.notes_ui import fmt_num, format_notes

_T0 = "2024-01-01T06:00:00"


@pytest.fixture(scope="module")
def datasets(sample_path):
    return {"ds0": mc_dataset.open_dataset(sample_path)}


def _figure(*panels):
    cfg = mc_config.default_figure_config()
    cfg["panels"] = list(panels)
    return cfg


def _map_panel(*layers):
    panel = mc_config.default_panel()
    panel["selection"] = {"time": _T0, "level": 500.0}
    panel["layers"] = list(layers)
    return panel


def _vsec_panel(*layers):
    """緯度-高度断面 (経度方向の平均を取る典型)。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": _T0}
    panel["layers"] = list(layers)
    return panel


def _line_panel(x_dim, *layers):
    panel = mc_config.default_line_panel()
    panel["x_dim"] = x_dim
    panel["selection"] = {"level": 500.0} if x_dim != "level" else {}
    if x_dim != "time":
        panel["selection"]["time"] = _T0
    panel["layers"] = list(layers)
    return panel


def _by_type(notes, kind):
    return [n for n in notes if n["type"] == kind]


# --- 値の変換 ---

def test_default_config_has_no_notes(datasets):
    fill = mc_config.default_fill_layer("ds0", "t")
    contour = mc_config.default_contour_layer("ds0", "z")
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    assert mc_notes.collect_notes(_figure(_map_panel(fill, contour, vector)), datasets) == []


def test_value_transform_entries_match_config(datasets):
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"value_scale": 1.0, "value_offset": -273.15})
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"].update({"value_scale": 0.1})   # b = 0 のまま
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"]["value_scale"] = 0.01            # 倍率だけ (加算なし)
    hatch = mc_config.default_hatch_layer("ds0", "t")  # 恒等 → 出ない
    notes = mc_notes.collect_notes(_figure(_map_panel(fill, contour, vector, hatch)), datasets)
    assert [n["type"] for n in notes] == ["value_transform"] * 3
    assert [(n["layer_index"], n["kind"], n["variables"], n["scale"], n["offset"])
            for n in notes] == [
        (0, "fill", ["t"], 1.0, -273.15),
        (1, "contour", ["z"], 0.1, 0.0),
        (2, "vector", ["u", "v"], 0.01, None),
    ]
    assert all(n["panel_index"] == 0 and n["axis"] is None for n in notes)


def test_scatter_axes_and_bubble_z(datasets):
    panel = mc_config.default_scatter_panel()
    panel.update({"x_variable": "t", "y_variable": "z", "z_variable": "u"})
    sc = mc_config.default_scatter_layer("ds0")
    sc["style"].update({"y_value_scale": 0.001})
    bb = mc_config.default_bubble_layer("ds0")
    bb["style"].update({"x_value_offset": -273.15, "z_value_scale": 2.0})
    panel["layers"] = [sc, bb]
    notes = mc_notes.collect_notes(_figure(panel), datasets)
    assert [(n["layer_index"], n["axis"], n["variables"], n["scale"], n["offset"])
            for n in notes] == [
        (0, "y", ["z"], 0.001, 0.0),
        (1, "x", ["t"], 1.0, -273.15),
        (1, "z", ["u"], 2.0, 0.0),
    ]


def test_fill_between_and_stackplot_list_all_variables(datasets):
    fb = mc_config.default_fill_between_layer("ds0", "t", variable_upper="z")
    fb["style"]["value_offset"] = 1.0
    sp = mc_config.default_stackplot_layer("ds0", ["t", "u", "v"])
    sp["style"]["value_scale"] = 3.0
    notes = mc_notes.collect_notes(_figure(_line_panel("lon", fb, sp)), datasets)
    assert [(n["kind"], n["variables"]) for n in notes] == [
        ("fill_between", ["t", "z"]), ("stackplot", ["t", "u", "v"])]


def test_panel_index_and_label_follow_figure_order(datasets):
    p1 = _map_panel(mc_config.default_fill_layer("ds0", "t"))
    fill2 = mc_config.default_fill_layer("ds0", "t")
    fill2["style"]["value_offset"] = -273.15
    p2 = _map_panel(fill2)
    p2["label"].update({"show": True, "text": "(b)"})
    notes = mc_notes.collect_notes(_figure(p1, p2), datasets)
    assert len(notes) == 1
    assert (notes[0]["panel_index"], notes[0]["panel_label"]) == (1, "(b)")
    # ラベル非表示ならラベル文字列は付けない
    p2["label"]["show"] = False
    assert mc_notes.collect_notes(_figure(p1, p2), datasets)[0]["panel_label"] is None


# --- 範囲平均 ---

def test_section_averages_count_grid_points_independently(datasets):
    ds = datasets["ds0"]
    lon = ds["lon"].values
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["averages"] = {"lon": {"op": "mean", "range": [120.0, 150.0]}}
    contour = mc_config.default_contour_layer("ds0", "z")
    # 30W〜30E 相当の wrap-around (サンプルは 0–360 系)
    contour["averages"] = {"lon": {"op": "mean", "range": [330.0, 30.0]}}
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["averages"] = {"lon": {"op": "mean", "range": [60.0, 200.0]}}
    notes = mc_notes.collect_notes(_figure(_vsec_panel(fill, contour, vector)), datasets)
    avg = _by_type(notes, "average")
    assert len(avg) == 3 and not _by_type(notes, "value_transform")
    n_plain = int(((lon >= 120.0) & (lon <= 150.0)).sum())
    n_wrap = int(((lon >= 330.0) | (lon <= 30.0)).sum())
    n_vec = int(((lon >= 60.0) & (lon <= 200.0)).sum())
    assert [(n["layer_index"], n["variables"], n["dim"], n["role"], n["op"],
             n["range"], n["n_points"]) for n in avg] == [
        (0, ["t"], "lon", "lon", "mean", [120.0, 150.0], n_plain),
        (1, ["z"], "lon", "lon", "mean", [330.0, 30.0], n_wrap),
        (2, ["u", "v"], "lon", "lon", "mean", [60.0, 200.0], n_vec),
    ]


def test_line_averages_weighted_and_x_dim_excluded(datasets):
    ds = datasets["ds0"]
    lat, lon = ds["lat"].values, ds["lon"].values
    line = mc_config.default_line_layer("ds0", "t")
    line["averages"] = {
        "lat": {"op": "weighted_mean", "range": [20.0, 40.0]},
        "lon": {"op": "mean", "range": [120.0, 150.0]},
        # x 軸 (time) の平均は render が無視する → 通知にも出ない
        "time": {"op": "mean", "range": ["2024-01-01", "2024-01-02"]},
    }
    notes = mc_notes.collect_notes(_figure(_line_panel("time", line)), datasets)
    assert [(n["dim"], n["role"], n["op"], n["n_points"]) for n in notes] == [
        ("lat", "lat", "weighted_mean", int(((lat >= 20.0) & (lat <= 40.0)).sum())),
        ("lon", "lon", "mean", int(((lon >= 120.0) & (lon <= 150.0)).sum())),
    ]


def test_line_bundle_excludes_bundle_dim(datasets):
    bundle = mc_config.default_line_bundle_layer("ds0", "t", bundle_dim="level")
    bundle["averages"] = {"level": {"op": "mean", "range": [1000.0, 200.0]},
                          "lat": {"op": "mean", "range": [0.0, 10.0]}}
    panel = _line_panel("lon", bundle)
    panel["selection"] = {"time": _T0}
    notes = mc_notes.collect_notes(_figure(panel), datasets)
    assert [n["dim"] for n in notes] == ["lat"]


def test_averages_ignored_where_render_ignores_them(datasets):
    # 水平面図の fill: render (_layer_data) は averages を読まない
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["averages"] = {"lon": {"op": "mean", "range": [120.0, 150.0]}}
    assert mc_notes.collect_notes(_figure(_map_panel(fill)), datasets) == []
    # 変数に無い次元の平均は読み飛ばされる
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["averages"] = {"no_such_dim": {"op": "mean", "range": [0.0, 1.0]}}
    assert mc_notes.collect_notes(_figure(_vsec_panel(contour)), datasets) == []


def test_fill_between_upper_uses_averages_upper(datasets):
    ds = datasets["ds0"]
    lat = ds["lat"].values
    fb = mc_config.default_fill_between_layer("ds0", "t", variable_upper="z")
    fb["averages"] = {"lat": {"op": "mean", "range": [0.0, 20.0]}}
    fb["averages_upper"] = {"lat": {"op": "weighted_mean", "range": [0.0, 20.0]}}
    notes = mc_notes.collect_notes(_figure(_line_panel("lon", fb)), datasets)
    n = int(((lat >= 0.0) & (lat <= 20.0)).sum())
    assert [(tuple(x["variables"]), x["op"], x["n_points"]) for x in notes] == [
        (("t",), "mean", n), (("z",), "weighted_mean", n)]
    # 同じ設定なら 1 項目にまとまる
    fb["averages_upper"] = fb["averages"]
    notes = mc_notes.collect_notes(_figure(_line_panel("lon", fb)), datasets)
    assert [(tuple(x["variables"]), x["op"]) for x in notes] == [(("t", "z"), "mean")]


# --- 生成スクリプトとの整合 ---

def test_notes_agree_with_generated_script(datasets, sample_path):
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"value_scale": 0.5, "value_offset": -273.15})
    fill["averages"] = {"lon": {"op": "mean", "range": [120.0, 150.0]}}
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["averages"] = {"lon": {"op": "mean", "range": [330.0, 30.0]}}
    cfg = _figure(_vsec_panel(fill, contour))
    notes = mc_notes.collect_notes(cfg, datasets)
    script = mc_scriptgen.generate_script(cfg, datasets, {"ds0": sample_path})
    for n in _by_type(notes, "value_transform"):
        assert f"* {n['scale']!r} + {n['offset']!r}" in script
    for n in _by_type(notes, "average"):
        assert f".mean(dim={n['dim']!r})" in script
    # 経度平均の切り出し行に現れる格子数 = 通知の格子数 (wrap-around は 2 片の合計)
    lon = datasets["ds0"]["lon"].values
    for n in _by_type(notes, "average"):
        lo, hi = n["range"]
        mask = ((lon >= lo) & (lon <= hi)) if lo <= hi else ((lon >= lo) | (lon <= hi))
        assert n["n_points"] == int(mask.sum())


# --- 表示整形 ---

def test_fmt_num_uses_fullwidth_minus():
    assert fmt_num(-273.15) == "−273.15"
    assert fmt_num(1.0) == "1"
    assert fmt_num(0.01) == "0.01"
    assert fmt_num("2024-01-01") == "2024-01-01"


def test_format_notes_groups_by_panel_and_layer(datasets):
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"value_offset": -273.15})
    fill["averages"] = {"lon": {"op": "mean", "range": [120.0, 150.0]}}
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["averages"] = {"lon": {"op": "mean", "range": [120.0, 150.0]}}
    p1 = _vsec_panel(fill, contour)
    p1["label"].update({"show": True, "text": "(a)"})
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"]["value_scale"] = 0.01
    p2 = _map_panel(vector)
    md = format_notes(mc_notes.collect_notes(_figure(p1, p2), datasets))
    lines = md.splitlines()
    assert lines[0] == "**この図に適用した処理**"
    assert "**パネル 1 (a)**" in lines and "**パネル 2**" in lines
    # 2 項目のレイヤーは見出し + 入れ子、1 項目のレイヤーは 1 行
    i = lines.index("- 塗りつぶし `t` [K]")
    assert lines[i + 1] == "    - 値の変換 a = 1, b = −273.15"
    assert lines[i + 2].startswith("    - 経度平均 lon 120〜150 (算術平均、")
    assert lines[i + 2].endswith(" 格子点)")
    assert any(line.startswith("- 等値線 `z` [m]: 経度平均 lon 120〜150") for line in lines)
    assert "- ベクトル `u` [m s-1], `v` [m s-1]: 倍率 a = 0.01 (加算なし)" in lines
    assert "-273" not in md   # 負号は全角マイナス


# --- マスクアウト・単位・ベクトル固有の通知 ---

def test_maskout_entries_one_per_condition(datasets):
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"]["maskout"].update({"below": 250.0, "above": 300.0})
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"]["maskout"].update({"variable": "t", "var_below": 260.0})
    hatch = mc_config.default_hatch_layer("ds0", "t")
    hatch["style"]["maskout"].update({"variable": "t"})  # 閾値なし → 無効 (出ない)
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"]["mask_below"] = 2.0
    notes = mc_notes.collect_notes(_figure(_map_panel(fill, contour, hatch, vector)), datasets)
    assert [(n["type"], n["layer_index"], n.get("cond"), n.get("value"), n.get("mask_variable"))
            for n in notes] == [
        ("maskout", 0, "below", 250.0, None),
        ("maskout", 0, "above", 300.0, None),
        ("maskout", 1, "var_below", 260.0, "t"),
        ("vector_mask", 3, None, 2.0, None),
    ]


def test_track_maskout_uses_raw_mask_variable(datasets):
    tr = mc_config.default_track_layer("ds0")
    tr.update({"lon_var": "lon", "lat_var": "lat"})
    tr["style"]["maskout"] = {"variable": "wind", "below": 30.0, "above": None}
    tr["style"]["points"].update({"variable": "wind", "value_scale": 0.514444})
    notes = mc_notes.collect_notes(_figure(_map_panel(tr)), datasets)
    assert [(n["type"], n["variables"]) for n in notes] == [
        ("value_transform", ["wind"]), ("maskout", ["wind"])]
    assert (notes[1]["cond"], notes[1]["value"], notes[1]["mask_variable"]) == \
        ("var_below", 30.0, "wind")


def test_heatmap_panel_transform_and_maskout(datasets):
    panel = mc_config.default_heatmap_panel()
    panel.update({"dataset_id": "ds0", "variable": "t", "x_dim": "lon", "y_dim": "lat"})
    panel["style"].update({"value_offset": -273.15})
    panel["style"]["maskout"]["below"] = 0.0
    notes = mc_notes.collect_notes(_figure(panel), datasets)
    assert [(n["type"], n["kind"], n["layer_index"], n["variables"]) for n in notes] == [
        ("value_transform", "heatmap", None, ["t"]), ("maskout", "heatmap", None, ["t"])]


def test_units_attached_from_variable_attrs(datasets):
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"]["value_offset"] = -273.15
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"]["value_scale"] = 0.01
    notes = mc_notes.collect_notes(_figure(_map_panel(fill, vector)), datasets)
    assert notes[0]["units"] == {"t": "K"}
    assert notes[1]["units"] == {"u": "m s-1", "v": "m s-1"}
    md = format_notes(notes)
    assert "塗りつぶし `t` [K]: 値の変換 a = 1, b = −273.15" in md
    assert "ベクトル `u` [m s-1], `v` [m s-1]: 倍率 a = 0.01 (加算なし)" in md


def test_curvilinear_vector_reports_no_rotation(datasets, curvilinear_sample_path):
    dsets = {**datasets, "dsc": mc_dataset.open_dataset(curvilinear_sample_path)}
    vec_c = mc_config.default_vector_layer("dsc", "u", "v")
    strm_c = mc_config.default_stream_layer("dsc", "u", "v")
    vec_r = mc_config.default_vector_layer("ds0", "u", "v")   # 1 次元格子 → 出ない
    panel = mc_config.default_panel()
    panel["layers"] = [vec_c, strm_c, vec_r]
    notes = mc_notes.collect_notes(_figure(panel), dsets)
    assert [(n["type"], n["layer_index"], n["kind"]) for n in notes] == [
        ("vector_no_rotation", 0, "vector"), ("vector_no_rotation", 1, "stream")]
    assert "格子相対風の回転なし" in format_notes(notes)


# --- 平均範囲内の欠損 (描画と同じ経路の記録 = render.layer_averaging_stats) ---

def test_average_missing_counts_nan_in_averaged_block(datasets):
    ds = datasets["ds0"]
    lat, lon = ds["lat"].values, ds["lon"].values
    # sst は |lat| > 60 が NaN。1 次元プロット (x = time) で lat 50–70 × lon 120–150 を平均
    line = mc_config.default_line_layer("ds0", "sst")
    line["averages"] = {"lat": {"op": "weighted_mean", "range": [50.0, 70.0]},
                        "lon": {"op": "mean", "range": [120.0, 150.0]}}
    panel = mc_config.default_line_panel()
    panel.update({"x_dim": "time", "selection": {}, "layers": [line]})
    notes = mc_notes.collect_notes(_figure(panel), datasets)
    missing = [n for n in notes if n["type"] == "average_missing"]
    assert len(missing) == 1
    block = ds["sst"].values[:, (lat >= 50.0) & (lat <= 70.0), :][:, :, (lon >= 120.0) & (lon <= 150.0)]
    assert missing[0]["n_total"] == block.size
    assert missing[0]["n_nan"] == int(np.isnan(block).sum()) > 0
    # 記録の dims は通知の平均項目と同じ dim 集合
    assert missing[0]["dims"] == [n["dim"] for n in notes if n["type"] == "average"]
    # 欠損の無い変数 (t) では出ない
    line_t = mc_config.default_line_layer("ds0", "t")
    line_t["averages"] = dict(line["averages"])
    panel_t = mc_config.default_line_panel()
    panel_t.update({"x_dim": "time", "selection": {"level": 500.0}, "layers": [line_t]})
    assert not [n for n in mc_notes.collect_notes(_figure(panel_t), datasets)
                if n["type"] == "average_missing"]


def test_average_missing_section_path(datasets):
    ds = datasets["ds0"]
    lat = ds["lat"].values
    # 時間–経度断面 (ホフメラー) で緯度 50–70 を平均: _section_data の経路
    fill = mc_config.default_fill_layer("ds0", "sst")
    fill["averages"] = {"lat": {"op": "mean", "range": [50.0, 70.0]}}
    panel = mc_config.default_section_panel()
    panel.update({"x_dim": "lon", "y_dim": "time", "selection": {}, "layers": [fill]})
    notes = mc_notes.collect_notes(_figure(panel), datasets)
    missing = [n for n in notes if n["type"] == "average_missing"]
    block = ds["sst"].values[:, (lat >= 50.0) & (lat <= 70.0), :]
    assert [(m["dims"], m["n_total"], m["n_nan"]) for m in missing] == [
        (["lat"], block.size, int(np.isnan(block).sum()))]
    md = format_notes(notes)
    assert f"平均範囲内の欠損 {int(np.isnan(block).sum())} / {block.size} 要素 (平均から除外)" in md
