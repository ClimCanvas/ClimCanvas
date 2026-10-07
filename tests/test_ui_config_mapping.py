# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""UI 操作 → figure_config の対応と、WIP のラウンドトリップの検証。

スモークテスト (test_app_smoke.py) は「例外なく動く」ことしか見ないので、
ここでは widget を操作した結果が panel_cfg_{pid} (アプリが毎 run 構築して
キャッシュする panel dict) に正しく反映されることを確認する。
widget の書き込み先キーの間違い・設定キャッシュの更新漏れを検出する。

あわせて「作業状態 (WIP) をサーバ側スロットに保存 → 新しいセッションで
復元 → 同じ panel 設定が再構築される」ラウンドトリップも検証する。
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render

# app.py の絶対パス。streamlit 1.63 以降の AppTest.from_file は相対パスを「呼び出し元の
# テストファイルの場所」基準で解決する (1.58 までは cwd 基準) ので、絶対パスで渡す
_APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def _button(at, key):
    return next(b for b in at.get("button") if b.key == key)


def _load_app(sample_path):
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    at.text_input(key="_new_file_path").set_value(sample_path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    assert not at.exception
    return at


def test_line_axis_widgets_map_to_config(sample_path):
    """1次元プロット: 軸・ラベル・目盛・凡例・図枠・背景の widget が
    panel 設定の正しいキーに反映されること。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    # 既定の x 軸は time なので lon に切り替える (xlab キーは x_dim を含む)
    at.selectbox(key="line_x_dim_line0").set_value("lon")
    at.run()
    assert not at.exception

    # 軸 expander
    at.checkbox(key="invy_line0").set_value(True)
    at.checkbox(key="tightx_line0").set_value(True)
    # ラベル expander
    at.text_input(key="xlab_line0_lon").set_value("LON")
    at.checkbox(key="labbold_line0").set_value(True)
    # 目盛 expander (y 等間隔 5)
    at.radio(key="ytmode_line0").set_value("interval")
    # 凡例 expander
    at.checkbox(key="legend_show_line0").set_value(False)
    # 図枠 expander
    at.checkbox(key="frame_top_line0").set_value(False)
    at.checkbox(key="frame_wuse_line0").set_value(True)
    at.run()
    assert not at.exception
    at.number_input(key="ytick_int_line0").set_value(5.0)
    at.slider(key="frame_w_line0").set_value(2.0)
    at.run()
    assert not at.exception

    cfg = at.session_state["panel_cfg_0"]
    axis = cfg["axis"]
    assert axis["invert_y"] is True
    assert axis["tight_x"] is True
    assert axis["x_label"] == "LON"
    assert axis["label_weight"] == "bold"
    assert axis["y_tick_interval"] == 5.0
    assert cfg["legend"]["show"] is False
    assert cfg["frame"]["show_top"] is False
    assert cfg["frame"]["width"] == 2.0


def test_line_secondary_axis_widgets_map_to_config(sample_path):
    """1次元プロット: レイヤーの「第2軸に描く」と「第2軸 (右)」expander の widget が
    style.secondary_y / axis.y2_* / log_y2 / invert_y2 / show_y2_* / grid.show_y2 に
    反映されること。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    lid = at.session_state["layers_line0"][0]["id"]
    at.checkbox(key=f"line_y2_line0_{lid}").set_value(True)
    at.text_input(key="y2lab_line0").set_value("U [m/s]")
    at.checkbox(key="y2lim_manual_line0").set_value(True)
    at.checkbox(key="invy2_line0").set_value(True)
    at.checkbox(key="showy2t_line0").set_value(False)
    at.radio(key="y2tmode_line0").set_value("positions")
    at.checkbox(key="y2minor_line0").set_value(True)
    at.checkbox(key="gridy2_line0").set_value(True)
    at.run()
    assert not at.exception
    at.number_input(key="y2lim_lo_line0").set_value(-10.0)
    at.number_input(key="y2lim_hi_line0").set_value(30.0)
    at.text_input(key="y2tick_pos_line0").set_value("-5, 0, 5")
    at.checkbox(key="y2tlab_use_line0").set_value(True)
    at.run()
    assert not at.exception
    at.text_input(key="y2tlab_line0").set_value("lo, zero, hi")
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    axis = cfg["axis"]
    assert cfg["layers"][0]["style"]["secondary_y"] is True
    assert axis["y2_label"] == "U [m/s]"
    assert axis["y2_lim"] == [-10.0, 30.0]
    assert axis["invert_y2"] is True and axis["log_y2"] is False
    assert axis["show_y2_ticklabels"] is False
    assert axis["y2_tick_positions"] == [-5.0, 0.0, 5.0]
    assert axis["y2_tick_labels"] == ["lo", "zero", "hi"]
    assert axis["show_y2_minor_ticks"] is True
    assert axis["grid"]["show_y2"] is True
    # 第2軸の目盛線だけ ON でも「目盛」に目盛線の色・太さ widget が出る
    assert any(w.key == "gridw_line0" for w in at.get("slider"))
    # 描画も通る (第2軸が作られ、範囲は y2_lim。手動範囲は反転より優先 = 第1軸と同じ)
    fig_cfg = mc_config.default_figure_config()
    fig_cfg["panels"] = [cfg]
    dsid = cfg["layers"][0]["dataset_id"]
    fig = mc_render.render_figure(
        fig_cfg, {dsid: mc_dataset.open_dataset(sample_path)})
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylim() == (-10.0, 30.0)
    assert list(fig.axes[1].get_yticks()) == [-5.0, 0.0, 5.0]


def test_line_twin_align_widgets_map_to_config(sample_path):
    """1次元プロット: 「指定した値の高さを左右で揃える」+「揃える値」が
    axis.y2_align_value に反映され、描画で左右の軸の高さが揃うこと。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    lid = at.session_state["layers_line0"][0]["id"]
    at.checkbox(key=f"line_y2_line0_{lid}").set_value(True)
    at.checkbox(key="y2align_line0").set_value(True)
    at.run()
    assert not at.exception
    at.number_input(key="y2align_v_line0").set_value(15.0)
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["axis"]["y2_align_value"] == 15.0
    fig_cfg = mc_config.default_figure_config()
    fig_cfg["panels"] = [cfg]
    dsid = cfg["layers"][0]["dataset_id"]
    fig = mc_render.render_figure(
        fig_cfg, {dsid: mc_dataset.open_dataset(sample_path)})
    fig.canvas.draw()
    ax, ax2 = fig.axes

    def frac(a):
        lo, hi = a.get_ylim()
        return (15.0 - lo) / (hi - lo)
    assert abs(frac(ax) - frac(ax2)) < 1e-9


def test_layer_dim_roles_independent_of_load_order(sample_path, tmp_path):
    """レイヤーの次元固定 UI の種別 (経度 = 数値入力で 0° またぎ可、鉛直 = 固定のみ、
    緯度 = 重み付き平均) が、先頭ファイル (ds0) ではなくレイヤー自身の dataset の
    役割で決まること。経度を持たない時系列ファイルを先に読み込んでも変わらない。"""
    import numpy as np
    import xarray as xr
    ts_path = tmp_path / "ts_only.nc"
    xr.Dataset(
        {"gm": (("time",), np.arange(4.0))},
        coords={"time": np.array(["2024-01-01", "2024-01-02", "2024-01-03",
                                  "2024-01-04"], dtype="datetime64[ns]")},
    ).to_netcdf(ts_path)
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    for path in (str(ts_path), sample_path):   # 経度なしのファイルが ds0
        at.text_input(key="_new_file_path").set_value(path)
        at.run()
        _button(at, "add_file").set_value(True)
        at.run()
        assert not at.exception
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    lid = at.session_state["layers_line0"][0]["id"]
    prefix = f"line_line0_{lid}"
    atmos_id = next(d["id"] for d in at.session_state["datasets"]
                    if d["path"] == sample_path)
    at.selectbox(key=f"{prefix}_ds").set_value(atmos_id)
    at.run()
    assert not at.exception
    radio_dims = {w.key.split("_avgmode_")[1] for w in at.get("radio")
                  if w.key.startswith(prefix + "_avgmode_")}
    # 鉛直 (level) は固定のみ (範囲平均 radio なし)、lat / lon は範囲平均可
    assert radio_dims == {"lat", "lon"}, radio_dims
    assert any(w.key == f"{prefix}_sel_level" for w in at.get("selectbox"))
    at.radio(key=f"{prefix}_avgmode_lon").set_value("range")
    at.radio(key=f"{prefix}_avgmode_lat").set_value("range")
    at.run()
    assert not at.exception
    # 経度は min/max の数値入力 (スライダーではない)、緯度は cos(lat) 重みチェック
    ni = {w.key for w in at.get("number_input")}
    assert f"{prefix}_avgrnglo_lon" in ni and f"{prefix}_avgrnghi_lon" in ni
    assert not any(w.key == f"{prefix}_avgrng_lon" for w in at.get("select_slider"))
    assert any(w.key == f"{prefix}_wght_lat" for w in at.get("checkbox"))
    # 0° またぎ (330→30) が config に入り描画も通る
    at.number_input(key=f"{prefix}_avgrnglo_lon").set_value(330.0)
    at.number_input(key=f"{prefix}_avgrnghi_lon").set_value(30.0)
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["layers"][0]["averages"]["lon"]["range"] == [330.0, 30.0]
    fig_cfg = mc_config.default_figure_config()
    fig_cfg["panels"] = [cfg]
    fig = mc_render.render_figure(
        fig_cfg, {atmos_id: mc_dataset.open_dataset(sample_path)})
    assert len(fig.axes[0].lines) == 1


def test_reflines_widgets_map_to_config(sample_path):
    """「図枠・背景」の「直線」: 追加ボタン → 向き・値 (時間軸は時刻選択、数値軸は
    数値入力)・体裁・凡例ラベル・端の文字が background.reflines に反映される。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    _button(at, "refline_add_line0").set_value(True)
    at.run()
    assert not at.exception
    sid = at.session_state["reflines_line0"][0]["id"]
    # 既定の x 軸は時間なので、縦線は時刻セレクタで指定する
    at.radio(key=f"refline_ori_line0_{sid}").set_value("x")
    at.run()
    assert not at.exception
    tsel = at.selectbox(key=f"refline_t_line0_{sid}_time")
    tsel.set_value(tsel.options[2])
    at.text_input(key=f"refline_label_line0_{sid}").set_value("event")
    at.text_input(key=f"refline_text_line0_{sid}").set_value("onset")
    at.slider(key=f"refline_lw_line0_{sid}").set_value(2.0)
    at.selectbox(key=f"refline_ls_line0_{sid}").set_value("dotted")
    at.run()
    assert not at.exception
    rl = at.session_state["panel_cfg_0"]["background"]["reflines"][0]
    assert rl["orientation"] == "x"
    assert isinstance(rl["value"], str) and rl["value"].startswith("2024-01-01T12")
    assert rl["label"] == "event" and rl["text"] == "onset"
    assert rl["linewidth"] == 2.0 and rl["linestyle"] == "dotted"
    # 横線に切り替えると数値入力になる
    at.radio(key=f"refline_ori_line0_{sid}").set_value("y")
    at.run()
    assert not at.exception
    at.number_input(key=f"refline_v_line0_{sid}").set_value(240.0)
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    rl = cfg["background"]["reflines"][0]
    assert rl["orientation"] == "y" and rl["value"] == 240.0
    # 描画も通る (基準線 1 本 + 端の文字 1 つ)
    fig_cfg = mc_config.default_figure_config()
    fig_cfg["panels"] = [cfg]
    dsid = cfg["layers"][0]["dataset_id"]
    fig = mc_render.render_figure(
        fig_cfg, {dsid: mc_dataset.open_dataset(sample_path)})
    ax = fig.axes[0]
    assert len(ax.lines) == 2 and [t.get_text() for t in ax.texts] == ["onset"]


def test_line_background_span_maps_to_config(sample_path):
    """背景の塗り範囲: 追加ボタン → 方向・範囲・透過度が config に反映される。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    _button(at, "bgspan_add_line0").set_value(True)
    at.run()
    assert not at.exception
    sid = at.session_state["bgspans_line0"][0]["id"]
    at.radio(key=f"bgspan_ori_line0_{sid}").set_value("y")
    at.run()
    at.number_input(key=f"bgspan_lo_line0_{sid}").set_value(1.0)
    at.number_input(key=f"bgspan_hi_line0_{sid}").set_value(3.0)
    at.slider(key=f"bgspan_a_line0_{sid}").set_value(0.5)
    at.run()
    assert not at.exception

    spans = at.session_state["panel_cfg_0"]["background"]["spans"]
    assert len(spans) == 1
    assert spans[0]["orientation"] == "y"
    assert spans[0]["lo"] == 1.0
    assert spans[0]["hi"] == 3.0
    assert spans[0]["alpha"] == 0.5


def test_vsection_axis_widgets_map_to_config(sample_path):
    """鉛直断面図: 東経・西経表記と目盛線 (grid) の widget 反映。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("vsec")
    at.run()
    assert not at.exception
    at.checkbox(key="lonew_vsec0_lon").set_value(True)
    at.checkbox(key="gridx_vsec0").set_value(True)
    at.run()
    assert not at.exception

    axis = at.session_state["panel_cfg_0"]["axis"]
    assert axis["x_lon_east_west"] is True
    assert axis["grid"]["show_x"] is True
    assert axis["grid"]["show_y"] is False


# モードごとの「既定値から動かす widget」と、復元後に確認する設定値。
# 鉛直・時間断面図と 1次元プロット (時間軸) は範囲 select_slider (tuple 状態)
# を持ち、WIP の JSON 保存 → 復元で tuple が list 化して壊れた実績があるため
# 全モードでラウンドトリップを検証する。
_ROUNDTRIP_MODES = {
    "map": {
        "plot_type": "horizontal_map",
        "tweaks": [("checkbox", "rev_map0_0", True),
                    ("text_input", "title_map0", "ROUNDTRIP")],
        "checks": [(("layers", 0, "style", "reverse_cmap"), True),
                    (("title",), "ROUNDTRIP")],
    },
    "vsec": {
        "plot_type": "section_2d",
        "tweaks": [("checkbox", "gridx_vsec0", True),
                    ("text_input", "title_vsec0", "ROUNDTRIP")],
        "checks": [(("axis", "grid", "show_x"), True),
                    (("title",), "ROUNDTRIP")],
    },
    "tsec": {
        "plot_type": "section_2d",
        "tweaks": [("checkbox", "gridy_tsec0", True),
                    ("text_input", "title_tsec0", "ROUNDTRIP")],
        "checks": [(("axis", "grid", "show_y"), True),
                    (("title",), "ROUNDTRIP")],
    },
    "line": {
        "plot_type": "line_1d",
        "tweaks": [("checkbox", "invy_line0", True),
                    ("text_input", "title_line0", "ROUNDTRIP")],
        "checks": [(("axis", "invert_y"), True),
                    (("title",), "ROUNDTRIP")],
    },
    "scatter": {
        "plot_type": "scatter_2d",
        "tweaks": [("checkbox", "invx_scatter0", True),
                    ("text_input", "title_scatter0", "ROUNDTRIP")],
        "checks": [(("axis", "invert_x"), True),
                    (("title",), "ROUNDTRIP")],
    },
}


def _dig(cfg, path):
    cur = cfg
    for p in path:
        cur = cur[p]
    return cur


def test_station_map_mode_offers_time_selector(tmp_path):
    """格子変数ゼロの地点データ (lon/lat がスカラー補助座標) でも、水平断面図
    モードで時刻セレクタが出て、map_scatter の panel 設定に時刻が入ること。

    回帰テスト: 時刻セレクタは格子変数の次元からしか組み立てられておらず、
    地点データでは時刻を固定する UI が消えていた (2026-07-25)。
    """
    import numpy as np
    import xarray as xr

    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(4) * np.timedelta64(1, "D"))
    ds = xr.Dataset(
        {"tmean": (("time",), np.array([1.0, 2.0, 3.0, 4.0]))},
        coords={"time": time,
                "lat": ((), 33.9, {"units": "degrees_north"}),
                "lon": ((), 130.9, {"units": "degrees_east"})})
    path = str(tmp_path / "station.nc")
    ds.to_netcdf(path)

    at = _load_app(path)
    at.selectbox(key="plot_mode_0").set_value("map")
    at.run()
    assert not at.exception
    # 時刻セレクタが存在する (修正前はここで落ちる)
    assert any(sb.key == "sel_map0_time" for sb in at.get("selectbox"))
    # レイヤーを散布図 (点) に切替えると、選択時刻が panel 設定に反映される
    at.selectbox(key="kind_map0_0").set_value("map_scatter")
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["layers"][0]["kind"] == "map_scatter"
    assert cfg["layers"][0]["variable"] == "tmean"
    assert "time" in cfg["selection"]


@pytest.mark.parametrize("mode", list(_ROUNDTRIP_MODES),
                         ids=["map", "vsec", "tsec", "line", "scatter"])
def test_session_roundtrip_restores_same_config(mode, sample_path, tmp_path,
                                            monkeypatch):
    """WIP 保存 → 新セッションで復元 → 同じ panel 設定が再構築される (全モード)。"""
    spec = _ROUNDTRIP_MODES[mode]
    monkeypatch.setenv("CC_SESSION_DIRS", str(tmp_path))

    # セッション A: モードを切り替え、設定を既定値から動かして保存
    at = _load_app(sample_path)
    if mode != "map":
        at.selectbox(key="plot_mode_0").set_value(mode)
        at.run()
        assert not at.exception
    for kind, key, value in spec["tweaks"]:
        getattr(at, kind)(key=key).set_value(value)
    at.run()
    assert not at.exception
    at.text_input(key="_session_save_name").set_value("slot1")
    at.run()
    _button(at, "_session_save_btn").set_value(True)
    at.run()
    assert not at.exception
    assert (tmp_path / "slot1.json").is_file()
    cfg_before = at.session_state["panel_cfg_0"]
    assert cfg_before["plot_type"] == spec["plot_type"]

    # セッション B: まっさらな状態から復元
    at2 = _load_app(sample_path)
    at2.selectbox(key="_session_restore_slot").set_value("slot1")
    at2.run()
    _button(at2, "_session_restore_btn").set_value(True)
    at2.run()
    assert not at2.exception
    at2.run()  # 復元後の再構築
    assert not at2.exception

    cfg_after = at2.session_state["panel_cfg_0"]
    assert cfg_after["plot_type"] == spec["plot_type"]
    for path, expected in spec["checks"]:
        assert _dig(cfg_after, path) == expected, f"{path} が復元されていない"
    # 図全体として同一の設定に戻ること
    assert cfg_after == cfg_before


def test_wip_roundtrip_multipanel(sample_path, tmp_path, monkeypatch):
    """複数パネル構成 (2パネル・モード混在) の WIP ラウンドトリップ。

    パネル一覧 (並び順)・グリッド設定・選択中パネル・各パネルの設定
    (選択中 = widget 状態から再構築 / 非選択 = panel_cfg キャッシュの復元)
    がすべて元に戻ることを確認する。
    """
    monkeypatch.setenv("CC_SESSION_DIRS", str(tmp_path))

    # セッション A: 1x2 グリッドでパネル追加、モード混在にして保存
    at = _load_app(sample_path)
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    _button(at, "add_panel").set_value(True)
    at.run()
    assert not at.exception
    # パネル2 (pid=1、選択中) を鉛直断面図に
    at.selectbox(key="plot_mode_1").set_value("vsec")
    at.run()
    at.text_input(key="title_vsec1").set_value("P2")
    at.run()
    assert not at.exception
    # パネル1 (pid=0) に戻って水平断面図の設定を変更
    _button(at, "_panel_sel_0").set_value(True)
    at.run()
    at.checkbox(key="rev_map0_0").set_value(True)
    at.text_input(key="title_map0").set_value("P1")
    at.run()
    assert not at.exception
    at.text_input(key="_session_save_name").set_value("multi")
    at.run()
    _button(at, "_session_save_btn").set_value(True)
    at.run()
    assert not at.exception
    assert (tmp_path / "multi.json").is_file()
    cfg0_before = at.session_state["panel_cfg_0"]
    cfg1_before = at.session_state["panel_cfg_1"]
    assert cfg0_before["title"] == "P1"
    assert cfg1_before["title"] == "P2"

    # セッション B: まっさらな状態から復元
    at2 = _load_app(sample_path)
    assert len(at2.session_state["panels"]) == 1
    at2.selectbox(key="_session_restore_slot").set_value("multi")
    at2.run()
    _button(at2, "_session_restore_btn").set_value(True)
    at2.run()
    assert not at2.exception
    at2.run()  # 復元後の再構築
    assert not at2.exception

    assert [p["id"] for p in at2.session_state["panels"]] == [0, 1]
    assert at2.session_state["grid_ncols"] == 2
    assert at2.session_state["panel_edit"] == 0
    cfg0_after = at2.session_state["panel_cfg_0"]
    cfg1_after = at2.session_state["panel_cfg_1"]
    assert cfg0_after["plot_type"] == "horizontal_map"
    assert cfg1_after["plot_type"] == "section_2d"
    assert cfg0_after["layers"][0]["style"]["reverse_cmap"] is True
    assert cfg0_after == cfg0_before
    assert cfg1_after == cfg1_before


def test_maskout_by_variable_maps_to_config(sample_path):
    """マスクアウトの「別の変数の値でマスク」: 変数と閾値が fill レイヤーの
    style.maskout (variable / var_below / var_above) に反映されること。"""
    at = _load_app(sample_path)
    # 既定モードは map、レイヤー0 は fill (lid=map0_0、描画変数 t)
    at.checkbox(key="maskv_on_fill_map0_0").set_value(True)
    at.run()
    assert not at.exception
    at.selectbox(key="maskv_var_fill_map0_0").set_value("u")
    at.checkbox(key="maskvb_on_fill_map0_0").set_value(True)
    at.run()
    assert not at.exception
    at.number_input(key="maskvb_fill_map0_0").set_value(3.0)
    at.run()
    assert not at.exception

    mo = at.session_state["panel_cfg_0"]["layers"][0]["style"]["maskout"]
    assert mo["variable"] == "u"
    assert mo["var_below"] == 3.0
    assert mo["var_above"] is None


def test_string_dim_coordinate_does_not_crash_layer_picker(tmp_path):
    """文字列の次元座標 (IBTrACS の basin='NIO' 等) を持つ変数でもレイヤー UI が
    落ちず、固定値として文字列が selection に入ること (v0.91 で実データの
    float('NIO') ValueError)。"""
    import numpy as np
    import xarray as xr

    path = str(tmp_path / "basin.nc")
    xr.Dataset(
        {"density": (("basin", "lat", "lon"),
                     np.arange(24, dtype=np.float32).reshape(2, 3, 4))},
        coords={"basin": ("basin", np.array(["NIO", "WP"])),
                "lat": ("lat", [0.0, 10.0, 20.0], {"units": "degrees_north"}),
                "lon": ("lon", [40.0, 50.0, 60.0, 70.0],
                        {"units": "degrees_east"})},
    ).to_netcdf(path)

    at = _load_app(path)  # 既定モードは map、レイヤー0 は fill (var=density)
    assert not at.exception
    sel = at.session_state["panel_cfg_0"]["layers"][0]["selection"]
    assert sel["basin"] == "NIO"  # 先頭値が既定で選ばれる


def test_central_lon_hidden_and_auto_for_platecarree_region(sample_path):
    """PlateCarree + 緯度経度範囲指定では中心経度入力を出さず範囲中央を自動使用。

    全経度未満では PlateCarree は経度方向に線形で図が中心経度によらず同一に
    なるため。全経度幅 (>=360°、例 -60〜300) でも範囲の中央を自動使用する —
    extent は全球扱いになり図の左右端は中心経度だけが決めるので、中央を使えば
    左右端 = 指定範囲の端になる。他投影 (極投影等) では従来どおり入力を表示する。
    """
    at = _load_app(sample_path)  # 既定モード = map、投影 = PlateCarree

    def _has_clon_input(proj):
        return any(n.key == f"central_lon_map0_{proj}"
                   for n in at.get("number_input"))

    # 範囲未指定: 中心経度入力あり
    assert _has_clon_input("PlateCarree")

    # 範囲指定 ON → 入力が消え、範囲の中央が config に入る
    at.checkbox(key="reg_check_map0_PlateCarree").set_value(True)
    at.run()
    assert not at.exception
    at.number_input(key="reg_lonmin_map0_PlateCarree").set_value(120.0)
    at.number_input(key="reg_lonmax_map0_PlateCarree").set_value(160.0)
    at.run()
    assert not at.exception
    assert not _has_clon_input("PlateCarree")
    proj = at.session_state["panel_cfg_0"]["projection"]
    assert proj["central_longitude"] == 140.0

    # 範囲を変えると中央へ自動追従する
    at.number_input(key="reg_lonmin_map0_PlateCarree").set_value(100.0)
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["projection"]["central_longitude"] == 130.0

    # 全経度幅 (-60〜300 = 360°) でも入力は出ず、範囲の中央 (120°) を自動使用
    # → 図の左右端が -60 / 300 になる (v0.95 までは手動のままで範囲が効かなかった)
    at.number_input(key="reg_lonmin_map0_PlateCarree").set_value(-60.0)
    at.number_input(key="reg_lonmax_map0_PlateCarree").set_value(300.0)
    at.run()
    assert not at.exception
    assert not _has_clon_input("PlateCarree")
    assert at.session_state["panel_cfg_0"]["projection"]["central_longitude"] == 120.0

    # 範囲指定 OFF に戻すと入力が再表示される
    at.checkbox(key="reg_check_map0_PlateCarree").set_value(False)
    at.run()
    assert not at.exception
    assert _has_clon_input("PlateCarree")

    # 極投影は範囲が常に指定される (回転に効く) ため入力を表示し続ける
    at.selectbox(key="proj_name_map0").set_value("NorthPolarStereo")
    at.run()
    assert not at.exception
    assert _has_clon_input("NorthPolarStereo")


def test_central_lon_follows_region_for_robinson(sample_path):
    """Robinson / EqualEarth + 緯度経度範囲指定では「中心経度を範囲の中央に合わせる」
    (既定 ON) で中心経度が範囲の中央に追従し、外すと入力欄に戻る (ユーザー要望 2026-10-02)。
    極投影などは従来どおり常に入力欄。
    """
    at = _load_app(sample_path)
    at.selectbox(key="proj_name_map0").set_value("Robinson")
    at.run()
    assert not at.exception

    def _has_clon_input(proj):
        return any(n.key == f"central_lon_map0_{proj}" for n in at.get("number_input"))

    # 範囲未指定: チェックは出ず入力あり
    assert _has_clon_input("Robinson")
    assert not any(c.key == "clon_follow_map0_Robinson" for c in at.get("checkbox"))

    # 範囲指定 ON → チェック (既定 ON) が出て入力が消え、範囲の中央が config に入る
    at.checkbox(key="reg_check_map0_Robinson").set_value(True)
    at.run()
    at.number_input(key="reg_lonmin_map0_Robinson").set_value(100.0)
    at.number_input(key="reg_lonmax_map0_Robinson").set_value(200.0)
    at.run()
    assert not at.exception
    assert at.checkbox(key="clon_follow_map0_Robinson").value is True
    assert not _has_clon_input("Robinson")
    assert at.session_state["panel_cfg_0"]["projection"]["central_longitude"] == 150.0

    # 範囲を変えると追従する
    at.number_input(key="reg_lonmax_map0_Robinson").set_value(160.0)
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["projection"]["central_longitude"] == 130.0

    # チェックを外すと入力欄 (初期値 = 範囲の中央) が出て、手動の値が使われる
    at.checkbox(key="clon_follow_map0_Robinson").set_value(False)
    at.run()
    assert not at.exception
    assert _has_clon_input("Robinson")
    assert at.number_input(key="central_lon_map0_Robinson").value == 130.0
    at.number_input(key="central_lon_map0_Robinson").set_value(180.0)
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["projection"]["central_longitude"] == 180.0

    # EqualEarth も同じ (範囲指定は投影法ごとなので入れ直す)。極投影は従来どおり入力欄のみ
    at.selectbox(key="proj_name_map0").set_value("EqualEarth")
    at.run()
    assert not at.exception
    assert _has_clon_input("EqualEarth")                # 範囲未指定
    at.checkbox(key="reg_check_map0_EqualEarth").set_value(True)
    at.run()
    assert not at.exception
    assert at.checkbox(key="clon_follow_map0_EqualEarth").value is True
    assert not _has_clon_input("EqualEarth")
    at.selectbox(key="proj_name_map0").set_value("NorthPolarStereo")
    at.run()
    assert not at.exception
    assert _has_clon_input("NorthPolarStereo")
    assert not any(c.key == "clon_follow_map0_NorthPolarStereo" for c in at.get("checkbox"))


def test_region_inputs_tab_order(sample_path):
    """描画範囲の入力の DOM 順 (= TAB 移動順) が 経度最小→最大→緯度最小→最大。

    st.columns は DOM がカラム毎 (縦方向が先) にまとまり TAB もその順になる
    ため、経度の行と緯度の行で columns を分けている (2026-08-24 ユーザー要望)。
    """
    at = _load_app(sample_path)
    at.checkbox(key="reg_check_map0_PlateCarree").set_value(True)
    at.run()
    assert not at.exception
    keys = [n.key for n in at.get("number_input")
            if n.key and n.key.startswith("reg_")]
    assert keys == ["reg_lonmin_map0_PlateCarree", "reg_lonmax_map0_PlateCarree",
                    "reg_latmin_map0_PlateCarree", "reg_latmax_map0_PlateCarree"]


def test_panel_selector_follows_mosaic(sample_path):
    """mosaic (空きセルつき) でも「編集するパネル」ボタンが全パネル分描画される。

    セル位置とパネル番号の対応は tests/test_mosaic_layout.py の
    panel_selector_cells ユニットテストが検証する。ここでは app.py の配線が
    例外なく動き、ボタンが欠けないことを見る。
    """
    at = _load_app(sample_path)
    _button(at, "add_panel").set_value(True)   # 2パネル目を追加
    at.run()
    assert not at.exception
    at.text_input(key="grid_mosaic").set_value(".A;B.")
    at.run()
    assert not at.exception
    sel_keys = {b.key for b in at.get("button")
                if b.key and b.key.startswith("_panel_sel_")}
    assert {"_panel_sel_0", "_panel_sel_1"} <= sel_keys


def test_manual_range_defaults_follow_value_transform(sample_path):
    """「値の変換」の倍率・加算を入れてから「値の範囲を自動」を外すと、最小値・
    最大値の既定値と「データ範囲」表示が**変換後**のデータ範囲になること
    (vmin/vmax は変換後の単位で指定する仕様。以前は生データの範囲が入り、
    そのまま描くと全面が範囲外になって「倍率が効いていない」ように見えた)。
    倍率を後から変えても手入力済みの値は入れ直さない (ユーザー判断)。"""
    import xarray as xr

    at = _load_app(sample_path)
    lid = "map0_0"  # 既定モード map、レイヤー0 は fill (描画変数 t)
    scale, offset = 2.0, -500.0
    at.number_input(key=f"vscale_fill_{lid}").set_value(scale)
    at.number_input(key=f"voff_fill_{lid}").set_value(offset)
    at.run()
    assert not at.exception
    at.checkbox(key=f"auto_{lid}").set_value(False)
    at.run()
    assert not at.exception

    # 選択中の断面 (既定 = 時刻 index 0・最下層 1000 hPa = index 0) の生データ範囲
    raw = xr.open_dataset(sample_path)["t"].isel(time=0, level=0)
    exp_lo = scale * float(raw.min()) + offset
    exp_hi = scale * float(raw.max()) + offset
    assert at.number_input(key=f"vmin_{lid}_t").value == pytest.approx(exp_lo)
    assert at.number_input(key=f"vmax_{lid}_t").value == pytest.approx(exp_hi)
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert style["vmin"] == pytest.approx(exp_lo)
    assert style["vmax"] == pytest.approx(exp_hi)
    assert style["value_scale"] == scale and style["value_offset"] == offset
    # 「データ範囲: lo 〜 hi」のキャプションも変換後の値 (文言は言語非依存に数値で照合)
    caps = [c.value for c in at.caption]
    assert any(f"{exp_lo:g}" in c and f"{exp_hi:g}" in c for c in caps), caps

    # 倍率を後から変えても手入力した vmin はそのまま (widget を入れ直さない)
    at.number_input(key=f"vmin_{lid}_t").set_value(-10.0)
    at.run()
    at.number_input(key=f"vscale_fill_{lid}").set_value(3.0)
    at.run()
    assert not at.exception
    assert at.number_input(key=f"vmin_{lid}_t").value == -10.0
    assert at.session_state["panel_cfg_0"]["layers"][0]["style"]["vmin"] == -10.0
    # 表示側のデータ範囲だけは新しい倍率に追従する
    new_lo = 3.0 * float(raw.min()) + offset
    new_hi = 3.0 * float(raw.max()) + offset
    caps = [c.value for c in at.caption]
    assert any(f"{new_lo:g}" in c and f"{new_hi:g}" in c for c in caps), caps


def test_animation_value_range_caption_follows_value_transform(sample_path):
    """アニメーション expander の「時間範囲内の値域」が、レイヤー側で固定した
    level と「値の変換」(倍率・加算) を当てた変換後の値で出ること。固定次元
    (level=…) と恒等でない変換 (×a +b) を変数名に添える。level を切り替えると
    値域も追従する。"""
    import xarray as xr

    at = _load_app(sample_path)
    lid = "map0_0"  # fill レイヤー (描画変数 t)
    scale, offset = 2.0, -500.0
    at.number_input(key=f"vscale_fill_{lid}").set_value(scale)
    at.number_input(key=f"voff_fill_{lid}").set_value(offset)
    at.run()
    assert not at.exception

    def _expect(level):
        # _variable_minmax_in_time_range と同じ切り出し: 全時刻 (既定の開始〜終了)
        # × レイヤー側で固定した level (`fill_map0_0_sel_level`)。緯度経度は切らない
        raw = xr.open_dataset(sample_path)["t"].sel(level=level)
        return scale * float(raw.min()) + offset, scale * float(raw.max()) + offset

    def _line():
        caps = [c.value for c in at.caption if "`t`" in c.value]
        assert caps, [c.value for c in at.caption]
        return caps[0]

    level = at.session_state[f"fill_{lid}_sel_level"]
    exp_lo, exp_hi = _expect(level)
    line = _line()
    assert f"`t` (level={level:g}, ×{scale:g} {offset:+g})" in line, line
    assert f"[{exp_lo:g}, {exp_hi:g}]" in line, line

    # level を切り替えると値域も追従する (全 level の範囲ではない)。
    # サンプルは隣接 level で min/max が一致することがあるので値域の違う level を選ぶ
    levels = [float(v) for v in at.selectbox(key=f"fill_{lid}_sel_level").options]
    other = next(v for v in levels if _expect(v) != (exp_lo, exp_hi))
    at.selectbox(key=f"fill_{lid}_sel_level").set_value(other)
    at.run()
    assert not at.exception
    exp_lo2, exp_hi2 = _expect(other)
    line = _line()
    assert f"`t` (level={other:g}, ×{scale:g} {offset:+g})" in line, line
    assert f"[{exp_lo2:g}, {exp_hi2:g}]" in line, line


def test_rotation_animation_ui_builds_request(sample_path):
    """地球回転アニメーション: Orthographic + 単一パネルのときだけ「動かすもの」が
    出て、経由点・終点・総フレーム数から rotation_path の centers が
    アニメーション再現スクリプトに入る。時刻送り + 回転では時刻数でサンプル。
    パネルを増やすと出なくなる。"""
    at = _load_app(sample_path)
    # PlateCarree (既定) では出ない
    assert not [r for r in at.radio if r.key == "anim_motion_map0"]
    at.selectbox(key="proj_name_map0").set_value("Orthographic")
    at.run()
    assert not at.exception
    at.radio(key="anim_motion_map0").set_value("rotate")
    at.run()
    _button(at, "rotadd_map0").set_value(True)
    at.run()
    assert not at.exception
    at.number_input(key="anim_rot_wplon_map0_0").set_value(200.0)
    at.number_input(key="anim_rot_wplat_map0_0").set_value(50.0)
    at.number_input(key="anim_rot_endlon_map0").set_value(260.0)
    at.number_input(key="anim_rot_endlat_map0").set_value(0.0)
    at.number_input(key="anim_rot_frames_map0").set_value(5)
    at.run()
    _button(at, "anim_script_map0").set_value(True)
    at.run()
    assert not at.exception
    script = at.session_state["_anim_script"]
    # 開始点 = 投影法の中心 (Orthographic の既定 180°E, 20°N)
    start = (at.session_state["central_lon_map0_Orthographic"],
             at.session_state["central_lat_map0_Orthographic"])
    expected = mc_render.rotation_path([start, (200.0, 50.0), (260.0, 0.0)], 5)
    assert f"centers = {expected!r}" in script
    assert "for (center_lon, center_lat) in centers:" in script
    assert ("ccrs.Orthographic(central_longitude=center_lon, "
            "central_latitude=center_lat)") in script
    assert "time_values" not in script  # 時刻固定
    assert "bbox_inches='tight'" not in script  # 回転時は固定サイズ

    # 時刻送り + 回転: フレーム数 = 時刻数 (開始〜終了の全時刻)。解像度も反映
    at.radio(key="anim_motion_map0").set_value("both")
    at.number_input(key="anim_dpi_map0").set_value(150)
    at.run()
    _button(at, "anim_script_map0").set_value(True)
    at.run()
    assert not at.exception
    script = at.session_state["_anim_script"]
    # フレーム保存 (savefig) の dpi。plt.figure(..., dpi=100) は図の dpi なので別
    assert "fig.savefig(_buf, format='png', dpi=150)" in script
    n_times = len(at.selectbox(key="anim_start_map0").options)
    expected = mc_render.rotation_path([start, (200.0, 50.0), (260.0, 0.0)], n_times)
    assert f"centers = {expected!r}" in script
    assert "for time_value, (center_lon, center_lat) in zip(time_values, centers):" in script
    assert "for _ in range(" not in script  # 1時刻 1 コマ (既定) では展開しない

    # 1時刻あたり 3 コマ: 経路は時刻数 × 3 で補間、時刻は 3 コマずつ保持
    at.number_input(key="anim_rot_hold_map0").set_value(3)
    at.run()
    _button(at, "anim_script_map0").set_value(True)
    at.run()
    assert not at.exception
    script = at.session_state["_anim_script"]
    expected = mc_render.rotation_path([start, (200.0, 50.0), (260.0, 0.0)], n_times * 3)
    assert f"centers = {expected!r}" in script
    assert "time_values = [t for t in time_values for _ in range(3)]" in script
    at.number_input(key="anim_rot_hold_map0").set_value(1)
    at.run()

    # 経由点を削除すると経路から消える
    _button(at, "rotdel_map0_0").set_value(True)
    at.run()
    assert not at.exception
    assert not [w for w in at.number_input if w.key == "anim_rot_wplon_map0_0"]

    # 複数パネルでは回転アニメーションなし (時刻送りのみ)
    _button(at, "add_panel").set_value(True)
    at.run()
    assert not at.exception
    assert not [r for r in at.radio if r.key.startswith("anim_motion_")]


def test_grid_ratio_inputs_validate_and_render(sample_path):
    """行・列の比率入力: 正しい入力は例外なく描け、不正入力はエラー + 均等で続行。"""
    at = _load_app(sample_path)
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    assert not at.exception
    at.text_input(key="grid_wratios").set_value("3, 1")
    at.run()
    assert not at.exception
    assert not at.error

    # 個数不一致 (列数2 に3個) → エラー表示するが均等のまま描画は続く
    at.text_input(key="grid_wratios").set_value("3, 1, 1")
    at.run()
    assert not at.exception
    assert any("均等" in (e.value or "") for e in at.error)

    # 読めない入力 → エラー表示 (描画は均等のまま続行)
    at.text_input(key="grid_wratios").set_value("abc")
    at.run()
    assert not at.exception
    assert at.error


def test_vector_reference_scale_maps_to_config(sample_path):
    """基準ベクトルの統一入力: 物理量 → key.length、物理量 + 軸幅% → style.scale。"""
    at = _load_app(sample_path)
    at.selectbox(key="kind_map0_0").set_value("vector")
    at.run()
    assert not at.exception
    at.number_input(key="vec_keylen_map0_0").set_value(20.0)
    at.checkbox(key="vec_autoscale_map0_0").set_value(False)
    at.run()
    assert not at.exception
    at.number_input(key="vec_refpct_map0_0").set_value(10.0)
    at.run()
    assert not at.exception
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    # 物理量 20 を軸幅の 10% で描く → quiver scale = 20 / 0.10 = 200
    assert style["key"]["length"] == 20.0
    assert style["scale"] == pytest.approx(200.0)


def test_vector_legacy_scale_migrates_to_percent(sample_path):
    """旧セッションの「スケール」直接値 (vec_scale_) が % に換算されて引き継がれ、
    config の scale は旧値のまま (図が変わらない)。"""
    at = _load_app(sample_path)
    at.selectbox(key="kind_map0_0").set_value("vector")
    at.run()
    assert not at.exception
    # 旧セッション復元を模す: スケール手動 250 の widget 状態が残っている
    at.session_state["vec_scale_map0_0"] = 250.0
    at.session_state["vec_autoscale_map0_0"] = False
    at.run()
    assert not at.exception
    # % = 物理量 10 ÷ 250 × 100 = 4.0 に換算され、scale は 250 に戻る
    assert at.session_state["vec_refpct_map0_0"] == pytest.approx(4.0)
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert style["scale"] == pytest.approx(250.0)


def test_vector_second_file_maps_to_config(sample_path):
    """ベクトルの y 成分を別ファイルから: 「y成分のファイル」selectbox
    (vec_{lid}_vds) → layer.v_dataset_id。既定の「x成分と同じファイル」("") では
    None、ds1 を選べば "ds1" が入り、y 成分の変数は ds1 の変数一覧から選ばれる。"""
    at = _load_app(sample_path)
    # 2 つ目のファイル (同じパス) を追加 → ファイル selectbox が現れる
    at.text_input(key="_new_file_path").set_value(sample_path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.session_state["datasets"]) == 2
    at.selectbox(key="kind_map0_0").set_value("vector")
    at.run()
    assert not at.exception
    layer = at.session_state["panel_cfg_0"]["layers"][0]
    assert layer["kind"] == "vector"
    assert layer["dataset_id"] == "ds0"
    assert layer["v_dataset_id"] is None
    assert at.selectbox(key="vec_map0_0_vds").value == ""

    at.selectbox(key="vec_map0_0_vds").set_value("ds1")
    at.run()
    assert not at.exception
    assert not at.error
    layer = at.session_state["panel_cfg_0"]["layers"][0]
    assert layer["dataset_id"] == "ds0"
    assert layer["v_dataset_id"] == "ds1"
    assert layer["u_variable"] == "u"
    assert layer["v_variable"] == "v"

    # 「x成分と同じファイル」に戻すと None に戻る
    at.selectbox(key="vec_map0_0_vds").set_value("")
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["layers"][0]["v_dataset_id"] is None


def test_range_and_level_captions_follow_selection(sample_path):
    """「データ範囲」は選択中の時刻・レベルの断面から作られ (以前は先頭断面 —
    lag/time の先頭・最下層 — のままで、時刻を送っても変わらなかった)、
    等値線レイヤーには matplotlib が実際に引く等値線の範囲・間隔・本数が出ること
    (「値の範囲を自動」ではレベルが図を見るまで分からなかった。2026-09-12)。"""
    import numpy as np
    import xarray as xr
    from matplotlib.contour import ContourSet

    at = _load_app(sample_path)
    lid = "map0_0"  # 既定モード map、レイヤー0 は fill (描画変数 t)
    at.selectbox(key="sel_map0_time").set_value(3)
    at.selectbox(key=f"fill_{lid}_sel_level").set_value(500.0)
    at.run()
    assert not at.exception

    raw = xr.open_dataset(sample_path)["t"]
    cur = raw.isel(time=3).sel(level=500.0)
    first = raw.isel(time=0, level=0)
    lo, hi = float(cur.min()), float(cur.max())
    assert (f"{lo:g}", f"{hi:g}") != (f"{float(first.min()):g}",
                                      f"{float(first.max()):g}")
    caps = [c.value for c in at.caption]
    assert any(f"{lo:g}" in c and f"{hi:g}" in c for c in caps), caps
    assert not any(f"{float(first.min()):g}" in c and f"{float(first.max()):g}" in c
                   for c in caps), caps

    # 等値線レイヤーを追加 (同じ 500 hPa) → 「等値線: lo 〜 hi、間隔 d (n 本)」が
    # 実際に描いた ContourSet.levels のうちデータ範囲内のものと一致する
    at.selectbox(key="addkind_map0_ja").set_value("contour")
    at.run()
    _button(at, "add_map0").set_value(True)
    at.run()
    assert not at.exception
    at.selectbox(key="cont_map0_1_sel_level").set_value(500.0)
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["layers"][1]["kind"] == "contour"
    assert cfg["layers"][1]["style"].get("vmin") is None  # 値の範囲は自動
    fig_cfg = mc_config.default_figure_config()
    fig_cfg["panels"] = [cfg]
    dsid = cfg["layers"][0]["dataset_id"]
    fig = mc_render.render_figure(fig_cfg, {dsid: mc_dataset.open_dataset(sample_path)})
    cs = [c for c in fig.axes[0].collections
          if isinstance(c, ContourSet) and not c.filled]
    assert len(cs) == 1
    lv = np.asarray(cs[0].levels, dtype=float)
    step = float(lv[1] - lv[0])
    drawn = lv[(lv >= lo) & (lv <= hi)]
    assert drawn.size >= 2
    caps = [c.value for c in at.caption]
    hits = [c for c in caps
            if f"{float(drawn[0]):g}" in c and f"{float(drawn[-1]):g}" in c
            and f"{step:g}" in c and f"({drawn.size} " in c]
    assert hits, caps


def test_line_xrange_inputs_link_with_slider(sample_path):
    """1次元プロット「プロット軸」の範囲: スライダーと最小/最大の数値入力
    (時間軸は開始/終了の文字列入力) が連動し、panel.ranges には数値なら入力値
    そのまま・時刻なら最寄りの時刻が入る。旧セッション (スライダーの値だけ) からは
    入力欄が種付けされる。"""
    import pandas as pd
    import xarray as xr
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    at.selectbox(key="line_x_dim_line0").set_value("lon")
    at.run()
    assert not at.exception
    # 初期値: 全範囲 (ranges なし)
    assert at.number_input(key="line_xrange_lo_line0_lon").value == 0.0
    assert at.number_input(key="line_xrange_hi_line0_lon").value == 357.5
    assert "lon" not in (at.session_state["panel_cfg_0"].get("ranges") or {})
    # スライダー → 数値入力に格子点の値が入る
    at.select_slider(key="line_xrange_line0_lon").set_value((100.0, 180.0))
    at.run()
    assert not at.exception
    assert at.number_input(key="line_xrange_lo_line0_lon").value == 100.0
    assert at.number_input(key="line_xrange_hi_line0_lon").value == 180.0
    assert at.session_state["panel_cfg_0"]["ranges"]["lon"] == [100.0, 180.0]
    # 数値入力 → スライダーは最寄りの格子点 (101 → 100)、範囲は入力値そのまま
    at.number_input(key="line_xrange_lo_line0_lon").set_value(101.0)
    at.run()
    assert not at.exception
    assert tuple(at.select_slider(key="line_xrange_line0_lon").value) == (100.0, 180.0)
    assert at.session_state["panel_cfg_0"]["ranges"]["lon"] == [101.0, 180.0]

    # 時間軸: 文字列入力は最寄りの時刻 (ISO ラベル) に丸めてスライダーと揃う
    tlabels = [pd.Timestamp(v).isoformat()
               for v in xr.open_dataset(sample_path).time.values]
    at.selectbox(key="line_x_dim_line0").set_value("time")
    at.run()
    assert not at.exception
    at.text_input(key="line_xrange_lo_line0_time").set_value("2024-01-01 07:00")
    at.run()
    assert not at.exception
    assert tuple(at.select_slider(key="line_xrange_line0_time").value) == (tlabels[1], tlabels[-1])
    assert at.text_input(key="line_xrange_lo_line0_time").value == tlabels[1]
    assert at.session_state["panel_cfg_0"]["ranges"]["time"] == [tlabels[1], tlabels[-1]]
    # 解釈できない文字列はスライダーの現在値に戻す (範囲は変わらない)
    at.text_input(key="line_xrange_hi_line0_time").set_value("not a date")
    at.run()
    assert not at.exception
    assert at.text_input(key="line_xrange_hi_line0_time").value == tlabels[-1]
    assert at.session_state["panel_cfg_0"]["ranges"]["time"] == [tlabels[1], tlabels[-1]]

    # 旧セッション (スライダーの値だけ保存) からの復元: 数値入力が種付けされる
    at2 = _load_app(sample_path)
    at2.session_state["line_xrange_line0_lon"] = (50.0, 200.0)
    at2.selectbox(key="plot_mode_0").set_value("line")
    at2.run()
    at2.selectbox(key="line_x_dim_line0").set_value("lon")
    at2.run()
    assert not at2.exception
    assert at2.number_input(key="line_xrange_lo_line0_lon").value == 50.0
    assert at2.number_input(key="line_xrange_hi_line0_lon").value == 200.0
    assert at2.session_state["panel_cfg_0"]["ranges"]["lon"] == [50.0, 200.0]


def test_line_legend_label_defaults_to_variable(sample_path, tmp_path):
    """1次元プロットの line レイヤーの凡例ラベルの既定は変数名。変数を変えると
    既定が追従し、空欄なら凡例に出さない (None)。旧セッションの変数名を含まない
    key (line_label_{lid}) に入っていたカスタムラベルは初回だけ引き継ぐ。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    var = at.selectbox(key="line_var_line0_0").value
    assert at.text_input(key=f"line_label_line0_0_{var}").value == var
    assert at.session_state["panel_cfg_0"]["layers"][0]["style"]["label"] == var
    # x 軸ラベルの入力欄初期値も次元名 [units] (long_name "longitude" ではない)
    at.selectbox(key="line_x_dim_line0").set_value("lon")
    at.run()
    assert not at.exception
    assert at.text_input(key="xlab_line0_lon").value == "lon [degrees_east]"
    assert at.session_state["panel_cfg_0"]["axis"]["x_label"] == "lon [degrees_east]"
    other = "u" if var != "u" else "v"
    at.selectbox(key="line_var_line0_0").set_value(other)
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["layers"][0]["style"]["label"] == other
    at.text_input(key=f"line_label_line0_0_{other}").set_value("")
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["layers"][0]["style"]["label"] is None
    # 元の変数に戻せば既定 (変数名) に戻る
    at.selectbox(key="line_var_line0_0").set_value(var)
    at.run()
    assert at.session_state["panel_cfg_0"]["layers"][0]["style"]["label"] == var

    at2 = _load_app(sample_path)
    at2.session_state["line_label_line0_0"] = "custom"
    at2.selectbox(key="plot_mode_0").set_value("line")
    at2.run()
    assert not at2.exception
    var2 = at2.selectbox(key="line_var_line0_0").value
    assert at2.text_input(key=f"line_label_line0_0_{var2}").value == "custom"
    assert at2.session_state["panel_cfg_0"]["layers"][0]["style"]["label"] == "custom"
    assert "line_label_line0_0" not in at2.session_state

    # 1次元プロット(集計) の hist も既定 = 変数名 (long_name [units] ではない)。
    # dist モードは緯度経度次元を持たない変数があるファイルでだけ選べる
    import numpy as np
    import xarray as xr
    dist_path = str(tmp_path / "dist.nc")
    rng = np.random.default_rng(0)
    xr.Dataset({"ts": ("time", rng.normal(size=40),
                       {"long_name": "surface temperature", "units": "K"})},
               coords={"time": np.arange(40)}).to_netcdf(dist_path)
    at3 = _load_app(dist_path)
    at3.selectbox(key="plot_mode_0").set_value("dist")
    at3.run()
    assert not at3.exception
    var3 = at3.selectbox(key="hist_var_dist0_0").value
    assert at3.text_input(key=f"hist_label_dist0_0_{var3}").value == var3
    assert at3.session_state["panel_cfg_0"]["layers"][0]["style"]["label"] == var3
    # 集計の自動軸ラベル (render / scriptgen 共用) も変数名 [units]
    cfg3 = at3.session_state["panel_cfg_0"]
    assert cfg3["axis"]["x_label"] is None
    dsid = cfg3["layers"][0]["dataset_id"]
    xlab, ylab = mc_render.dist_axis_labels(cfg3, {dsid: xr.open_dataset(dist_path)})
    assert (xlab, ylab) == ("ts [K]", "count")


def test_line_bundle_widgets_map_to_config(sample_path):
    """1次元プロットの「ライン (束)」: 束ねる次元・残りの次元の固定・統計線の
    本数と分位が panel_cfg に入り、束ねる次元は selection に混ざらない。
    描画すると束の本数 + 統計線の本数だけ Line2D ができる。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    at.selectbox(key="addkind_line0_ja").set_value("line_bundle")
    at.run()
    _button(at, "add_line0").set_value(True)
    at.run()
    assert not at.exception
    lid = "line0_1"
    at.selectbox(key=f"bundle_var_{lid}").set_value("t")
    at.run()
    assert not at.exception
    # 候補は描画軸 (time) 以外の座標付き次元。表示は「次元 (本数)」
    opts = at.selectbox(key=f"bundle_dim_{lid}").options
    assert [o.split(" ")[0] for o in opts] == ["level", "lat", "lon"]
    assert opts[0].startswith("level (6 ")
    at.selectbox(key=f"bundle_dim_{lid}").set_value("lat")
    at.run()
    assert not at.exception
    at.selectbox(key=f"bundle_{lid}_sel_level").set_value(500.0)
    at.run()
    assert not at.exception
    # 統計線は OFF が既定 (summaries 空、本数の入力欄も出ない)
    assert at.session_state["panel_cfg_0"]["layers"][1]["summaries"] == []
    assert not [w for w in at.number_input if w.key == f"bundle_nstat_{lid}"]
    at.checkbox(key=f"bundle_stats_{lid}").set_value(True)
    at.run()
    assert not at.exception
    # ON にすると本数 1 (平均) から始まる
    assert at.number_input(key=f"bundle_nstat_{lid}").value == 1
    assert [smm["stat"] for smm in
            at.session_state["panel_cfg_0"]["layers"][1]["summaries"]] == ["mean"]
    at.number_input(key=f"bundle_nstat_{lid}").set_value(2)
    at.run()
    assert not at.exception
    at.selectbox(key=f"bundle_stat_{lid}_1").set_value("pct_range")
    at.run()
    assert not at.exception
    at.number_input(key=f"bundle_qlo_{lid}_1").set_value(10.0)
    at.radio(key=f"bundle_draw_{lid}_1").set_value("band_lines")
    at.run()
    assert not at.exception
    at.slider(key=f"bundle_salpha_{lid}_1").set_value(0.2)
    at.slider(key=f"bundle_lw_{lid}").set_value(0.0)   # 束の線を描かない (帯だけ)
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    lyr = cfg["layers"][1]
    assert lyr["kind"] == "line_bundle"
    assert lyr["bundle_dim"] == "lat"
    assert "lat" not in lyr["selection"] and "lat" not in lyr["averages"]
    assert lyr["selection"]["level"] == 500.0
    assert lyr["style"]["label"] == "t"
    assert [smm["stat"] for smm in lyr["summaries"]] == ["mean", "pct_range"]
    assert (lyr["summaries"][1]["q_low"], lyr["summaries"][1]["q_high"]) == (10.0, 95.0)
    assert lyr["summaries"][0]["draw"] == "lines"        # 1 本の統計量に描き方は無い
    assert lyr["summaries"][1]["draw"] == "band_lines"
    assert lyr["summaries"][1]["style"]["alpha"] == 0.2
    assert lyr["style"]["linewidth"] == 0.0
    # 既定ラベルは統計量・分位から (分位を変えると既定に追従する)
    assert lyr["summaries"][0]["style"]["label"] == "mean"
    assert lyr["summaries"][1]["style"]["label"] == "p10-p95"
    # 平均 ± 標準偏差: 倍率 k が入り、既定ラベルは k に追従、描き方も選べる
    at.selectbox(key=f"bundle_stat_{lid}_0").set_value("std_range")
    at.run()
    assert not at.exception
    at.number_input(key=f"bundle_kstd_{lid}_0").set_value(2.0)
    at.run()
    assert not at.exception
    cfg = at.session_state["panel_cfg_0"]
    lyr = cfg["layers"][1]
    assert lyr["summaries"][0]["stat"] == "std_range"
    assert lyr["summaries"][0]["k_std"] == 2.0
    assert lyr["summaries"][0]["draw"] == "lines"
    assert lyr["summaries"][0]["style"]["label"] == "mean±2σ"
    fig_cfg = mc_config.default_figure_config()
    fig_cfg["panels"] = [cfg]
    ds = mc_dataset.open_dataset(sample_path)
    fig = mc_render.render_figure(fig_cfg, {lyr["dataset_id"]: ds})
    # 先頭の line レイヤー 1 本 + 束 (線幅 0 なので 0 本) + 平均 ± 2σ の 2 本 + 帯の縁 2 本、
    # 帯 1 つ
    assert len(fig.axes[0].lines) == 1 + 0 + 2 + 2
    from matplotlib.collections import PolyCollection
    assert len([c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]) == 1


def test_line_marker_size_widget_maps_to_config(sample_path):
    """1次元プロットのライン: マーカーを選ぶと「マーカーサイズ (pt)」のスライダーが
    出て style.marker_size に入る。「なし」ならスライダーは出ず None。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    lid = "line0_0"
    assert not [w for w in at.slider if w.key == f"line_msize_{lid}"]
    assert at.session_state["panel_cfg_0"]["layers"][0]["style"]["marker_size"] is None
    at.selectbox(key=f"line_marker_{lid}").set_value("o")
    at.run()
    assert not at.exception
    assert at.slider(key=f"line_msize_{lid}").value == 6.0
    at.slider(key=f"line_msize_{lid}").set_value(9.5)
    at.run()
    assert not at.exception
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert (style["marker"], style["marker_size"]) == ("o", 9.5)
    at.selectbox(key=f"line_marker_{lid}").set_value("none")
    at.run()
    assert not at.exception
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert (style["marker"], style["marker_size"]) == (None, None)
    assert not [w for w in at.slider if w.key == f"line_msize_{lid}"]


def test_hist_bin_width_caption_follows_bins_and_range(tmp_path):
    """1次元プロット(集計) のヒストグラム: 「ビン数」の横に出るビンの幅が、ビン数と
    「値の範囲を指定」の最小値・最大値 (無指定ならデータの min/max) に連動する。
    ビン境界の直接指定では出ない。"""
    import numpy as np
    import xarray as xr
    dist_path = str(tmp_path / "dist.nc")
    vals = np.linspace(-2.0, 8.0, 41)          # min -2 / max 8 → 範囲 10
    xr.Dataset({"ts": ("time", vals)}, coords={"time": np.arange(41)}).to_netcdf(dist_path)
    at = _load_app(dist_path)
    at.selectbox(key="plot_mode_0").set_value("dist")
    at.run()
    assert not at.exception

    def caption():
        caps = [c.value for c in at.caption if c.value.startswith("ビンの幅")]
        assert len(caps) == 1, caps
        return caps[0]

    assert caption() == "ビンの幅: 0.5 (データ範囲 -2 〜 8)"      # 10 / 20
    at.number_input(key="hist_nbin_dist0_0").set_value(10)
    at.run()
    assert not at.exception
    assert caption() == "ビンの幅: 1 (データ範囲 -2 〜 8)"
    at.checkbox(key="hist_rangeon_dist0_0").set_value(True)
    at.run()
    assert not at.exception
    assert caption() == "ビンの幅: 0.1"                             # 既定 0〜1 を 10 分割
    at.number_input(key="hist_rmax_dist0_0").set_value(5.0)
    at.run()
    assert not at.exception
    assert caption() == "ビンの幅: 0.5"
    at.number_input(key="hist_rmin_dist0_0").set_value(5.0)        # 最大 ≤ 最小
    at.run()
    assert not at.exception
    assert caption().startswith("ビンの幅: —")
    at.checkbox(key="hist_bineq_dist0_0").set_value(False)         # ビン境界の直接指定
    at.run()
    assert not at.exception
    assert not [c for c in at.caption if c.value.startswith("ビンの幅")]


def test_hist_orientation_and_width_widgets_map_to_config(tmp_path):
    """ヒストグラム: 「向き」と「棒の幅を手動指定」が style.orientation / rwidth に
    入る。幅の指定は描き方が塗り (bar) のときだけ出て、step に変えると None に戻る。"""
    import numpy as np
    import xarray as xr
    dist_path = str(tmp_path / "dist.nc")
    xr.Dataset({"ts": ("time", np.linspace(-2.0, 8.0, 41))},
               coords={"time": np.arange(41)}).to_netcdf(dist_path)
    at = _load_app(dist_path)
    at.selectbox(key="plot_mode_0").set_value("dist")
    at.run()
    assert not at.exception
    lid = "dist0_0"
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert (style["orientation"], style["rwidth"]) == ("vertical", None)
    at.radio(key=f"hist_orient_{lid}").set_value("horizontal")
    at.checkbox(key=f"hist_rw_manual_{lid}").set_value(True)
    at.run()
    assert not at.exception
    at.slider(key=f"hist_rw_{lid}").set_value(0.5)
    at.run()
    assert not at.exception
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert (style["orientation"], style["rwidth"]) == ("horizontal", 0.5)
    at.selectbox(key=f"hist_httype_{lid}").set_value("step")
    at.run()
    assert not at.exception
    assert not [w for w in at.checkbox if w.key == f"hist_rw_manual_{lid}"]
    style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert (style["orientation"], style["rwidth"]) == ("horizontal", None)


def test_plot_size_none_leaves_box_aspect_unfixed(sample_path):
    """「プロットサイズ」→「axes 枠の縦横比」の「固定しない (Figure サイズに従う)」で
    panel.box_aspect が None になり、render は set_box_aspect を呼ばない。既定は
    6.4:4.8 (0.75)、2次元プロット系の既定は 1:1 のまま。地図では常に None。"""
    at = _load_app(sample_path)
    assert at.session_state["panel_cfg_0"]["box_aspect"] is None      # 地図
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    assert abs(at.session_state["panel_cfg_0"]["box_aspect"] - 0.75) < 1e-9   # 4.8 / 6.4
    ds = mc_dataset.open_dataset(sample_path)

    def rendered_box_aspect():
        cfg = mc_config.default_figure_config()
        cfg["panels"] = [at.session_state["panel_cfg_0"]]
        dsid = cfg["panels"][0]["layers"][0]["dataset_id"]
        return mc_render.render_figure(cfg, {dsid: ds}).axes[0].get_box_aspect()

    assert abs(rendered_box_aspect() - 0.75) < 1e-9
    at.radio(key="boxaspect_preset_line0").set_value("none")
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["box_aspect"] is None
    assert rendered_box_aspect() is None
    at.selectbox(key="plot_mode_0").set_value("scatter")
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["box_aspect"] == 1.0
    at.radio(key="boxaspect_preset_scatter0").set_value("none")
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["box_aspect"] is None


def test_line_xrange_spans_all_loaded_files(tmp_path):
    """1次元プロット「プロット軸」の範囲: 候補値は読み込んだ全ファイルの x 座標の
    和集合。歴史実験 (1850–2014) の後に将来シナリオ (2015–2100) を読み込んでも
    スライダーが先頭ファイルの範囲に固定されず、最大値の入力欄と連動する
    (実機で発覚 2026-09-20)。"""
    import numpy as np
    import xarray as xr
    rng = np.random.default_rng(1)

    def _write(path, years):
        xr.Dataset({"gmst": (("member", "year"),
                             rng.normal(size=(3, years.size)) + 288.0)},
                   coords={"member": np.arange(1, 4), "year": years}).to_netcdf(path)

    hist = str(tmp_path / "hist.nc")
    ssp = str(tmp_path / "ssp.nc")
    _write(hist, np.arange(1850, 2015))
    _write(ssp, np.arange(2015, 2101))
    at = _load_app(hist)
    at.text_input(key="_new_file_path").set_value(ssp)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.session_state["datasets"]) == 2
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    at.selectbox(key="line_x_dim_line0").set_value("year")
    at.run()
    assert not at.exception
    opts = [float(v) for v in at.select_slider(key="line_xrange_line0_year").options]
    assert (opts[0], opts[-1], len(opts)) == (1850.0, 2100.0, 251)
    assert at.select_slider(key="line_xrange_line0_year").value == (1850.0, 2100.0)
    # 全範囲のままなら ranges には入らない
    assert "year" not in (at.session_state["panel_cfg_0"].get("ranges") or {})
    # 最大値の入力欄 → スライダーが 2 つ目のファイルの範囲まで動き、config にも入る
    at.number_input(key="line_xrange_hi_line0_year").set_value(2050.0)
    at.run()
    assert not at.exception
    assert at.select_slider(key="line_xrange_line0_year").value == (1850.0, 2050.0)
    assert at.session_state["panel_cfg_0"]["ranges"]["year"] == [1850.0, 2050.0]
    # スライダー → 入力欄 (2 つ目のファイルの値)
    at.select_slider(key="line_xrange_line0_year").set_value((1900.0, 2100.0))
    at.run()
    assert not at.exception
    assert at.number_input(key="line_xrange_hi_line0_year").value == 2100.0
    assert at.session_state["panel_cfg_0"]["ranges"]["year"] == [1900.0, 2100.0]


def test_map_resolution_selectbox_maps_to_config(sample_path):
    """「地理データの解像度」の選択が map.resolution に入る (既定は auto)。"""
    at = _load_app(sample_path)
    assert at.session_state["panel_cfg_0"]["map"]["resolution"] == "auto"
    at.selectbox(key="ne_res_map0").set_value("110m")
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["map"]["resolution"] == "110m"


def test_land_above_data_checkbox_maps_to_config(sample_path):
    """「陸域を塗りつぶす → データの上に描く」が map.land.above_data に入る (既定 False)。"""
    at = _load_app(sample_path)
    at.checkbox(key="land_show_map0").set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["map"]["land"]["above_data"] is False
    at.checkbox(key="land_fg_map0").set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["map"]["land"]["above_data"] is True


def test_animation_script_follows_path_style(sample_path):
    """アニメーションの再現スクリプトも「netCDFパスの形式」(絶対 / 相対) に従う。

    回帰テスト (2026-09-29): 静止図のスクリプトは設定に従うのに、アニメーションの
    スクリプトだけ常に絶対パスで書いていた (app._script_dataset_paths で共用に)。
    """
    import os
    at = _load_app(sample_path)
    rel = os.path.relpath(sample_path)
    absp = os.path.abspath(sample_path)
    assert rel != absp
    for style, expected, unexpected in (("relative", rel, absp), ("absolute", absp, None)):
        at.radio(key="script_path_style").set_value(style)
        at.run()
        _button(at, "anim_script_map0").set_value(True)
        at.run()
        assert not at.exception
        script = at.session_state["_anim_script"]
        assert repr(expected) in script, style
        if unexpected:
            assert repr(unexpected) not in script, style
