# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""UI層 (app.py) のスモークテスト。

Streamlit の AppTest でアプリを実行し、ファイル読み込みから描画・スクリプト生成
までが例外なく動くことを確認する。ウィジェットの細かい挙動は対象外。
"""

from streamlit.testing.v1 import AppTest
from pathlib import Path

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
    datasets = at.session_state["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["path"] == sample_path
    assert datasets[0]["id"] == "ds0"
    return at


def test_horizontal_map_mode(sample_path):
    at = _load_app(sample_path)
    assert len(at.selectbox) > 0
    assert not at.error

    # レイヤーを追加しても例外なく再描画される
    _button(at, "add_map0").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.session_state["layers_map0"]) == 2
    assert not at.error

    # 時刻送り (次の時刻 ▶)
    assert at.session_state["sel_map0_time"] == 0
    _button(at, "next_map0_time").set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["sel_map0_time"] == 1


def test_polar_label_sides_ui(sample_path):
    """極投影ではラベルの辺 UI が「上下 + 緯度ラベルのラジオ + 円周沿い経度」に切り替わる。

    左右チェックは極投影では対応するラベルが存在しないため出さない
    (cartopy は緯度ラベルを図中 inline、経度ラベルを境界沿い geo に分類する)。
    geo・極チェックは円形枠のときだけ出す。緯度ラベルはラジオ
    (図中 / 枠沿い / 非表示)。
    """
    at = _load_app(sample_path)

    def keys():
        return {cb.key for cb in at.checkbox}

    def radio_keys():
        return {r.key for r in at.radio}

    assert {"gl_side_left_map0", "gl_side_right_map0"} <= keys()
    assert "gl_side_geo_map0" not in keys()
    assert "gl_lat_lab_map0" not in radio_keys()

    at.selectbox(key="proj_name_map0").set_value("NorthPolarStereo")
    at.run()
    assert not at.exception
    assert not at.error
    assert {"gl_side_geo_map0", "gl_side_pole_map0",
            "gl_side_top_map0", "gl_side_bottom_map0"} <= keys()
    assert "gl_side_left_map0" not in keys()
    assert "gl_lat_lab_map0" in radio_keys()

    # 緯度ラベルのラジオを切り替えても例外なく描画される。
    # 「枠沿い」を選ぶと「表示する縁」のサブラジオが現れる
    assert "gl_lat_edge_side_map0" not in radio_keys()
    at.radio(key="gl_lat_lab_map0").set_value("edge")
    at.run()
    assert not at.exception
    assert not at.error
    assert "gl_lat_edge_side_map0" in radio_keys()
    at.radio(key="gl_lat_edge_side_map0").set_value("left")
    at.run()
    assert not at.exception
    assert not at.error

    # 緯度ラベルの回転指定 (チェック → 角度入力が現れる) も例外なく描画される
    at.checkbox(key="gl_lat_rot_on_map0").set_value(True)
    at.run()
    assert not at.exception
    assert not at.error
    at.number_input(key="gl_lat_rot_map0").set_value(0.0)
    at.run()
    assert not at.exception
    assert not at.error

    # 円形枠を外すと geo・極チェックは消え、緯度ラベルのラジオは残る
    at.checkbox(key="circular_boundary_map0").set_value(False)
    at.run()
    assert not at.exception
    assert "gl_side_geo_map0" not in keys()
    assert "gl_side_pole_map0" not in keys()
    assert "gl_lat_lab_map0" in radio_keys()


def test_notes_box_only_when_processing_is_used(sample_path):
    """図の直下の「この図に適用した処理」は、値の変換や範囲平均を使ったときだけ出る。"""
    at = _load_app(sample_path)
    assert not any("この図に適用した処理" in m.value for m in at.markdown)

    # 塗りつぶしレイヤーの「値の変換」倍率を変えると枠が出て、変数名と a, b が載る
    at.number_input(key="vscale_fill_map0_0").set_value(0.5)
    at.run()
    assert not at.exception
    box = [m.value for m in at.markdown if "この図に適用した処理" in m.value]
    assert len(box) == 1
    assert "`t`" in box[0] and "a = 0.5, b = 0" in box[0]

    # 恒等変換に戻すと枠ごと消える
    at.number_input(key="vscale_fill_map0_0").set_value(1.0)
    at.run()
    assert not at.exception
    assert not any("この図に適用した処理" in m.value for m in at.markdown)


def test_vertical_and_time_section_modes(sample_path):
    at = _load_app(sample_path)

    at.selectbox(key="plot_mode_0").set_value("vsec")
    at.run()
    assert not at.exception
    assert not at.error

    at.selectbox(key="plot_mode_0").set_value("tsec")
    at.run()
    assert not at.exception
    assert not at.error


def test_line_1d_mode(sample_path):
    """1次元プロットモードに切り替えても例外なく描画されること。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    assert not at.error
    # 初期レイヤーは line になっている
    assert at.session_state["layers_line0"][0]["kind"] == "line"


def test_multi_file_layer_switch(sample_path):
    """同じファイルを2つ読み込み、レイヤーが ds1 を参照する状態でも描画できること。

    回帰テスト: 以前は描画関数に `datasets = {"ds0": ds}` の決め打ちが残っていて、
    レイヤーが ds1 を参照すると `KeyError: 'ds1'` で描画が落ちた。
    """
    at = _load_app(sample_path)
    # 2 つ目のファイル (同じパス) を追加
    at.text_input(key="_new_file_path").set_value(sample_path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.session_state["datasets"]) == 2

    # レイヤー1 のファイル selectbox は (fill, lid=map0_0) → key="fill_map0_0_ds"
    # ds0 → ds1 に切替
    at.selectbox(key="fill_map0_0_ds").set_value("ds1")
    at.run()
    assert not at.exception
    assert not at.error


def test_multipanel_add_and_duplicate(sample_path):
    """パネルを追加・複製しても例外なく描画されること (第6段階)。"""
    at = _load_app(sample_path)
    # 2x1 グリッドにしてからパネルを追加
    at.number_input(key="grid_nrows").set_value(2)
    at.run()
    _button(at, "add_panel").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.session_state["panels"]) == 2
    # 新パネル (pid=1) が編集対象になっている
    assert at.session_state["panel_edit"] == 1
    # 新パネルは未設定でも警告のみで、既存パネルは描画される
    assert not at.exception
    # 新パネルのモードを設定して描画 → panel_cfg_1 がキャッシュされる
    at.selectbox(key="plot_mode_1").set_value("vsec")
    at.run()
    assert not at.exception
    assert not at.error
    assert at.session_state["panel_cfg_1"]["plot_type"] == "section_2d"

    # パネル1 (pid=0, 水平断面図) を複製 → 3 枚目 (pid=2) ができる
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    _button(at, "dup_panel_0").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.session_state["panels"]) == 3
    # 複製されたパネルの状態がコピーされている (レイヤー構成)
    assert at.session_state["layers_map2"] == at.session_state["layers_map0"]
    at.run()
    assert not at.exception
    assert not at.error

def test_multipanel_reorder(sample_path):
    """パネルの並べ替えボタン (← / →) でリスト順 = 配置順が入れ替わること。"""
    at = _load_app(sample_path)
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    _button(at, "add_panel").set_value(True)
    at.run()
    assert [p["id"] for p in at.session_state["panels"]] == [0, 1]
    # パネル1 (pid=0) を後ろへ → [1, 0]
    _button(at, "_mv_next_0").set_value(True)
    at.run()
    assert not at.exception
    assert [p["id"] for p in at.session_state["panels"]] == [1, 0]
    # 先頭になった pid=1 の「←」は disabled
    assert next(b for b in at.get("button") if b.key == "_mv_prev_1").disabled
    at.run()
    assert not at.exception

def test_layer_reorder(sample_path):
    """レイヤーの並べ替えボタン (↑ / ↓) でリスト順 = 描画順が入れ替わり、
    各レイヤーの設定 (変数選択) が入れ替え後もそのレイヤーに付いて回ること。"""
    at = _load_app(sample_path)
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    # 1枚のときは並べ替えボタンを出さない
    assert not [b for b in at.get("button") if b.key.startswith("_mvup_")]
    _button(at, "add_line0").set_value(True)
    at.run()
    at.run()
    ids = [it["id"] for it in at.session_state["layers_line0"]]
    assert len(ids) == 2
    id0, id1 = ids
    at.selectbox(key=f"line_var_line0_{id0}").set_value("t")
    at.selectbox(key=f"line_var_line0_{id1}").set_value("u")
    at.run()
    assert not at.exception
    layers = at.session_state["panel_cfg_0"]["layers"]
    assert [ly["variable"] for ly in layers] == ["t", "u"]
    # 先頭の「↑」と末尾の「↓」は disabled
    assert _button(at, f"_mvup_line0_{id0}").disabled
    assert _button(at, f"_mvdn_line0_{id1}").disabled
    assert not _button(at, f"_mvdn_line0_{id0}").disabled
    # レイヤー2 (u) を上へ → 構成リストも cfg の layers も [u, t] の順になる
    _button(at, f"_mvup_line0_{id1}").set_value(True)
    at.run()
    at.run()  # st.rerun() を含む run の後は素の run を1回挟む (stale 要素の掃除)
    assert not at.exception
    assert [it["id"] for it in at.session_state["layers_line0"]] == [id1, id0]
    layers = at.session_state["panel_cfg_0"]["layers"]
    assert [ly["variable"] for ly in layers] == ["u", "t"]
    # 入れ替え後も widget の値はそれぞれのレイヤーに残っている
    assert at.selectbox(key=f"line_var_line0_{id1}").value == "u"
    assert at.selectbox(key=f"line_var_line0_{id0}").value == "t"
    assert _button(at, f"_mvup_line0_{id1}").disabled
    # 下へ戻す → 元の順
    _button(at, f"_mvdn_line0_{id1}").set_value(True)
    at.run()
    assert not at.exception
    assert [it["id"] for it in at.session_state["layers_line0"]] == [id0, id1]
    assert [ly["variable"]
            for ly in at.session_state["panel_cfg_0"]["layers"]] == ["t", "u"]


def test_panel_state_survives_duplicate_and_switch(sample_path):
    """複製・パネル切替・削除で他パネルの widget 状態が破棄されないこと。

    回帰テスト: パネル編集 UI は選択中パネルの widget しか描画しないため、
    以前は複製 (st.rerun) の run で元パネルの widget 状態が Streamlit の
    stale-widget 掃除で消え、「パネル1 (未設定)」になっていた。
    app.py のパネル構成セクションが毎 run パネルスコープ key を自己代入して
    保持することで防いでいる。
    """
    at = _load_app(sample_path)
    at.text_input(key="title_map0").set_value("KEEP-ME")
    at.run()
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    _button(at, "dup_panel_0").set_value(True)
    at.run()
    assert not at.exception
    # 元パネル (pid=0) の状態が残っている
    assert at.session_state["plot_mode_0"] == "map"
    assert at.session_state["title_map0"] == "KEEP-ME"
    # 複製先 (pid=1) にコピーされている
    assert at.session_state["plot_mode_1"] == "map"
    assert at.session_state["title_map1"] == "KEEP-ME"
    assert at.session_state["panel_edit"] == 1
    # 選択ボタンでパネル1 (pid=0) に切り替えても状態が残る
    _button(at, "_panel_sel_0").set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["panel_edit"] == 0
    assert at.session_state["title_map0"] == "KEEP-ME"
    # パネル2 (pid=1) を削除してもパネル1の設定は消えない
    _button(at, "del_panel_1").set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["plot_mode_0"] == "map"
    assert at.session_state["title_map0"] == "KEEP-ME"

def test_global_settings_survive_panel_switch(sample_path):
    """パネル切替で「図全体の書式」「出力」の widget 状態が破棄されないこと。

    回帰テスト: パネル切替ボタンの st.rerun() はスクリプトを「パネル構成」の
    途中で打ち切るため、それより後に描画される共通カラーバー・出力形式などの
    widget が stale 扱いで破棄されていた (ユーザー報告: 共通カラーバーが消える)。
    app.py 冒頭の全 key 自己代入ループで防いでいる。
    """
    at = _load_app(sample_path)
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    _button(at, "add_panel").set_value(True)
    at.run()
    # 共通カラーバー ON + 出力形式 PDF
    at.checkbox(key="scbar_show").set_value(True)
    at.run()
    at.selectbox(key="out_format").set_value("PDF")
    at.run()
    assert at.session_state["scbar_show"] is True
    # パネル1 (pid=0) へ切替 → 設定が残っている
    _button(at, "_panel_sel_0").set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["scbar_show"] is True
    assert at.session_state["out_format"] == "PDF"

def test_common_edit_mode_propagates_changes(sample_path):
    """全パネル共通モード: 代表パネルでの変更が全パネルへ反映されること。

    - 変更した設定は widget 状態と panel_cfg キャッシュの両方に伝播する
    - タイトル等の個別設定は同期されない
    - モード解除後は伝播しない
    """
    at = _load_app(sample_path)
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    _button(at, "dup_panel_0").set_value(True)
    at.run()
    # パネル2 のタイトルを個別化してからパネル1 + 共通モードへ
    at.text_input(key="title_map1").set_value("P2")
    at.run()
    _button(at, "_panel_sel_0").set_value(True)
    at.run()
    _button(at, "_panel_sel_all").set_value(True)
    at.run()
    assert at.session_state["panel_edit_all"] is True

    # 代表 (パネル1) でカラーレベル数を変更 → パネル2 に伝播
    at.number_input(key="nlev_map0_0").set_value(11)
    at.run()
    assert not at.exception
    assert at.session_state["nlev_map1_0"] == 11
    assert at.session_state["panel_cfg_1"]["layers"][0]["style"]["levels"] == 11

    # タイトルは個別のまま
    at.text_input(key="title_map0").set_value("P1")
    at.run()
    assert at.session_state["title_map1"] == "P2"
    assert at.session_state["panel_cfg_1"]["title"] == "P2"

    # 共通モード解除後は伝播しない
    _button(at, "_panel_sel_all").set_value(True)
    at.run()
    assert at.session_state["panel_edit_all"] is False
    at.number_input(key="nlev_map0_0").set_value(7)
    at.run()
    assert at.session_state["nlev_map1_0"] == 11


def test_allowed_dirs_blocks_injected_path(sample_path, tmp_path, monkeypatch):
    """WIP 復元・アップロードで session_state に注入された許可外パスも読み込み時に弾く。

    「追加」ボタンの _is_path_allowed 検証は WIP 経由の注入には効かないため、
    app.py の読み込みループが第二の検証点になっていることを確認する。
    """
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("CC_ALLOWED_DIRS", str(allowed))

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    # WIP 復元をシミュレート: 許可ディレクトリ外の実在ファイルを直接注入
    at.session_state["datasets"] = [{"id": "ds0", "path": sample_path}]
    at.run()
    assert not at.exception
    assert at.error, "許可ディレクトリ外のパスがエラーなしで読み込まれた"
    assert "許可ディレクトリの外" in at.error[0].value


def test_allowed_dirs_accepts_inside_path(sample_path, monkeypatch):
    """許可ディレクトリ内のパスは注入経由でも通常どおり読み込める。"""
    import os as _os
    monkeypatch.setenv("CC_ALLOWED_DIRS", _os.path.dirname(sample_path))

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    at.session_state["datasets"] = [{"id": "ds0", "path": sample_path}]
    at.run()
    assert not at.exception
    assert not at.error


def test_session_path_rejects_traversal(tmp_path):
    """スロット名のパストラバーサル (wip_dirs 外への書き込み・削除) を拒否する。"""
    import pytest
    from climcanvas.ui.state_io import _session_path

    # 正常系
    p = _session_path(str(tmp_path), "el-nino-2015")
    assert p == tmp_path / "el-nino-2015.json"

    # 異常系
    for bad in ("../evil", "a/b", "/etc/x", "..", ".", ""):
        with pytest.raises(ValueError):
            _session_path(str(tmp_path), bad)


def test_single_variable_dimensionless_offers_2d_plot(tmp_path):
    """単一変数・無次元データでも「2次元プロット」が選べ、x=y で散布図が描ける。

    2次元プロットモードの出現条件を「変数2つ以上」から「1つ以上」へ緩和した
    回帰テスト (x軸とy軸に同じ変数を選べるため)。
    """
    import numpy as np
    import xarray as xr

    a = xr.DataArray(np.random.rand(8, 8), dims=("y", "x"),
                     coords={"y": np.arange(8), "x": np.arange(8)}, name="A")
    path = str(tmp_path / "mono.nc")
    a.to_dataset().to_netcdf(path)

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    at.text_input(key="_new_file_path").set_value(path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    assert not at.exception

    # AppTest の .options は format_func 適用後の表示ラベルなので、
    # 選択可否は set_value → session_state で確認する (ラベル非依存)
    mode_sb = at.selectbox(key="plot_mode_0")
    mode_sb.set_value("scatter")
    at.run()
    assert not at.exception
    assert not at.error
    assert at.session_state["plot_mode_0"] == "scatter"
    # 単一変数なので x/y とも同じ変数が選ばれる
    assert at.selectbox(key="panel_xvar_scatter0").value == "A"
    assert at.selectbox(key="panel_yvar_scatter0").value == "A"


def test_map_scatter_pure_station_data(tmp_path):
    """格子変数の無い純・地点データでも水平断面図で散布図 (map_scatter) を描ける。

    地図モードのガード緩和 (map_scatter_variables があれば入れる) の回帰テスト。
    """
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(2)
    n = 25
    path = str(tmp_path / "station.nc")
    xr.Dataset(
        {"temp": (("station",), (rng.standard_normal(n) * 4 + 12).astype("float32"),
                  {"units": "degC"})},
        coords={"lon": ("station", rng.uniform(100, 180, n).astype("float32")),
                "lat": ("station", rng.uniform(0, 60, n).astype("float32")),
                "station": np.arange(n)}).to_netcdf(path)

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    at.text_input(key="_new_file_path").set_value(path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    at.selectbox(key="plot_mode_0").set_value("map")
    at.run()
    assert not at.exception  # 格子変数ゼロでもモードに入れる
    at.selectbox(key="kind_map0_0").set_value("map_scatter")
    at.run()
    assert not at.exception
    assert not at.error
    assert at.selectbox(key="ms_var_map0_0").value == "temp"


def test_dist_1d_hist_mode(tmp_path):
    """非地理の時系列データで「1次元プロット(集計)」に入り hist を描ける。

    lat/lon 次元を持つ変数の除外 (nongeo_variables) と dist_1d モードの
    スモークテスト。
    """
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(5)
    n = 200
    path = str(tmp_path / "series.nc")
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(n) * np.timedelta64(1, "D"))
    bins = np.linspace(0.0, 20.0, 30)
    xr.Dataset(
        {"ts": (("time",), rng.standard_normal(n).astype("float32") * 3 + 10,
                {"units": "degC"}),
         "pdf_ref": (("bin",),
                     np.exp(-0.5 * ((bins - 10) / 3.0) ** 2).astype("float32"))},
        coords={"time": time, "bin": bins}).to_netcdf(path)

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    at.text_input(key="_new_file_path").set_value(path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    at.selectbox(key="plot_mode_0").set_value("dist")
    at.run()
    assert not at.exception
    assert not at.error
    assert at.session_state["layers_dist0"][0]["kind"] == "hist"
    assert at.selectbox(key="hist_var_dist0_0").value == "ts"
    # ECDF に切替 (第2軸チェック ON でも例外なし)
    at.selectbox(key="kind_dist0_0").set_value("ecdf")
    at.run()
    assert not at.exception
    at.checkbox(key="ecdf_y2_dist0_0").set_value(True)
    at.run()
    assert not at.exception
    assert not at.error
    # ライン (参照 PDF) に切替: x軸にする次元は数値座標 bin だけが候補
    at.selectbox(key="kind_dist0_0").set_value("line")
    at.run()
    assert not at.exception
    at.selectbox(key="line_var_dist0_0").set_value("pdf_ref")
    at.run()
    assert not at.exception
    assert not at.error
    # 描画タイプを箱ひげ・バイオリンに切替 → 既存レイヤーの kind は box に落ちる
    at.radio(key="dist_family_dist0").set_value("box")
    at.run()
    assert not at.exception
    assert at.session_state["layers_dist0"][0]["kind"] == "box"
    at.selectbox(key="kind_dist0_0").set_value("violin")
    at.run()
    assert not at.exception
    assert not at.error


def test_agg_2d_mode(tmp_path):
    """「2次元プロット(集計)」で hist2d / hexbin (移設後) を描ける。"""
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(6)
    n = 300
    path = str(tmp_path / "series_agg.nc")
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(n) * np.timedelta64(6, "h"))
    xr.Dataset(
        {"a": (("time",), rng.standard_normal(n).astype("float32")),
         "b": (("time",), (rng.standard_normal(n) * 2 + 1).astype("float32"))},
        coords={"time": time}).to_netcdf(path)

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    at.text_input(key="_new_file_path").set_value(path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    at.selectbox(key="plot_mode_0").set_value("agg")
    at.run()
    assert not at.exception
    assert not at.error
    assert at.session_state["layers_agg0"][0]["kind"] == "hist2d"

    # 色の離散化 ON → 対数スケール checkbox が消える (LogNorm と排他)
    at.checkbox(key="disc_h2d_agg0_0").set_value(True)
    at.run()
    assert not at.exception
    assert all(cb.key != "h2d_log_agg0_0" for cb in at.checkbox)
    _style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert _style["levels"] == 11
    at.checkbox(key="disc_h2d_agg0_0").set_value(False)
    at.run()

    # 対数スケール ON → 離散化 checkbox が消える
    at.checkbox(key="h2d_log_agg0_0").set_value(True)
    at.run()
    assert not at.exception
    assert all(cb.key != "disc_h2d_agg0_0" for cb in at.checkbox)
    _style = at.session_state["panel_cfg_0"]["layers"][0]["style"]
    assert _style["levels"] is None and _style["log_counts"] is True
    at.checkbox(key="h2d_log_agg0_0").set_value(False)
    at.run()

    at.selectbox(key="kind_agg0_0").set_value("hexbin")
    at.run()
    assert not at.exception
    assert not at.error


def test_track_only_dataset_map_mode(track_sample_path):
    """トラック専用データ (lat/lon 役割なし) でも地図モードでトラックを描ける。

    地図モードの入場ガード緩和 (track_capable) と track_layer_ui の回帰テスト。
    """
    at = _load_app(track_sample_path)
    # lat/lon 役割・座標が無いので選べるモードは水平断面図だけ
    assert at.session_state["plot_mode_0"] == "map"
    at.selectbox(key="kind_map0_0").set_value("track")
    at.run()
    assert not at.exception
    assert not at.error
    # lon/lat 変数が自動候補から選ばれる
    assert at.selectbox(key="tr_lon_map0_0").value == "lon"
    assert at.selectbox(key="tr_lat_map0_0").value == "lat"
    # 点の色付けの既定は単色 (radio は 単色/カラーマップ の順)。
    # カラーマップに切り替えると色付け変数の既定は wind
    at.radio(key="cmode_tr_map0_0").set_value("cmap")
    at.run()
    assert not at.exception
    assert at.selectbox(key="tr_pvar_map0_0").value in ("wind", "pres")


def test_common_edit_mode_layer_structure_sync(sample_path):
    """全パネル共通モード: レイヤーの追加・削除・種類変更も他パネルへ反映される。

    同じ描画モードのパネルにのみ反映され、id と kind が引き継がれたレイヤーの
    個別設定 (対象パネル側の変数選択など) は保持される (_sync_common_layers)。
    """
    at = _load_app(sample_path)
    at.number_input(key="grid_ncols").set_value(2)
    at.run()
    _button(at, "dup_panel_0").set_value(True)
    at.run()
    # パネル2 の既存レイヤーの変数を個別化 (保持されることを後で確認)
    at.selectbox(key="fill_var_map1_0").set_value("t")
    at.run()
    _button(at, "_panel_sel_0").set_value(True)
    at.run()
    _button(at, "_panel_sel_all").set_value(True)
    at.run()
    assert at.session_state["panel_edit_all"] is True

    # レイヤー追加 → パネル2 の構成と cfg にも追加される
    _button(at, "add_map0").set_value(True)
    at.run()
    at.run()  # 追加直後の rerun で新レイヤーの widget と cfg が確定
    assert not at.exception
    assert len(at.session_state["layers_map0"]) == 2
    assert at.session_state["layers_map1"] == at.session_state["layers_map0"]
    assert len(at.session_state["panel_cfg_1"]["layers"]) == 2
    # 引き継がれたレイヤーのパネル2 個別設定は保持されている
    assert at.session_state["panel_cfg_1"]["layers"][0]["variable"] == "t"

    # 種類変更 (レイヤー2 を等値線に) → パネル2 にも反映
    at.selectbox(key="kind_map0_1").set_value("contour")
    at.run()
    assert not at.exception
    assert at.session_state["layers_map1"][1]["kind"] == "contour"
    assert at.session_state["panel_cfg_1"]["layers"][1]["kind"] == "contour"

    # 並べ替え (レイヤー2 を上へ) → パネル2 の構成も同じ順になり、
    # 引き継がれたレイヤーの個別設定は id に付いて回る
    _button(at, "_mvup_map0_1").set_value(True)
    at.run()
    at.run()  # st.rerun() を含む run の後は素の run を1回挟む (stale 要素の掃除)
    assert not at.exception
    assert [it["id"] for it in at.session_state["layers_map0"]] == [1, 0]
    assert at.session_state["layers_map1"] == at.session_state["layers_map0"]
    assert at.session_state["panel_cfg_1"]["layers"][0]["kind"] == "contour"
    assert at.session_state["panel_cfg_1"]["layers"][1]["variable"] == "t"

    # レイヤー削除 (id=1 の等値線) → パネル2 も 1 枚に戻る
    # (残ったレイヤーの個別設定は保持)
    _button(at, "del_map0_1").set_value(True)
    at.run()
    assert not at.exception
    assert len(at.session_state["layers_map0"]) == 1
    assert at.session_state["layers_map1"] == at.session_state["layers_map0"]
    assert len(at.session_state["panel_cfg_1"]["layers"]) == 1
    assert at.session_state["panel_cfg_1"]["layers"][0]["variable"] == "t"


def test_role_time_override_ui(tmp_path):
    """時刻役割の手動上書き: lag 次元を時刻扱いにすると時刻セクションで送れる。

    units="days" の lag 座標 (decode_timedelta=False で数値のまま読まれる) を
    持つデータで「時刻として扱う次元」を lag にすると、レイヤー側の固定
    selectbox からパネルの「時刻」セクション (前後ボタン + インデックス選択) に
    移る。自動へ戻しても widget 状態の残留で落ちないこと。
    """
    import numpy as np
    import xarray as xr

    path = str(tmp_path / "lagcorr.nc")
    xr.Dataset(
        {"corr": (("lag", "lat", "lon"),
                  np.linspace(-1.0, 1.0, 5 * 4 * 6)
                  .astype("float32").reshape(5, 4, 6))},
        coords={
            "lag": ("lag", np.arange(-2, 3), {"units": "days"}),
            "lat": ("lat", np.linspace(-30.0, 30.0, 4),
                    {"units": "degrees_north"}),
            "lon": ("lon", np.linspace(0.0, 300.0, 6),
                    {"units": "degrees_east"}),
        }).to_netcdf(path)

    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    at.text_input(key="_new_file_path").set_value(path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    assert not at.exception
    assert not at.error
    # 上書き前: lag はレイヤー側の固定 selectbox で選ぶ
    assert any(sb.key == "fill_map0_0_sel_lag" for sb in at.selectbox)

    at.selectbox(key="role_time_override").set_value("lag")
    at.run()
    assert not at.exception
    assert not at.error
    # 上書き後: 時刻セクション (インデックス選択) に移り、前後ボタンで送れる
    assert at.session_state["sel_map0_lag"] == 0
    _button(at, "next_map0_lag").set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["sel_map0_lag"] == 1

    # 自動認識へ戻す → レイヤー側の固定 selectbox に戻り、例外なし
    at.selectbox(key="role_time_override").set_value("")
    at.run()
    assert not at.exception
    assert not at.error
    assert any(sb.key == "fill_map0_0_sel_lag" for sb in at.selectbox)


def test_curvilinear_map_mode(curvilinear_sample_path):
    """2 次元座標 (lon(y,x) / lat(y,x)) のデータ: 地図モードが開き、断面モードは出ない。

    レイヤーの「固定する次元」には鉛直 (lev) だけが出て、水平面の dim (y / x) は
    固定対象にならないこと。ランベルト図法 + 領域指定 (2 次元座標の min/max が
    既定値) でも描けること。
    """
    at = _load_app(curvilinear_sample_path)
    assert not at.exception
    assert not at.error
    # 使えるモード = 水平断面図 / 1次元プロット / 2次元プロット (vsec / tsec / 集計系は出ない)
    assert len(at.selectbox(key="plot_mode_0").options) == 3
    assert at.session_state["plot_mode_0"] == "map"
    keys = {w.key for w in at.selectbox}
    assert "fill_map0_0_sel_lev" in keys
    assert "fill_map0_0_sel_y" not in keys and "fill_map0_0_sel_x" not in keys
    assert "sel_map0_y" not in keys and "sel_map0_x" not in keys
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["layers"][0]["selection"] == {"lev": 1000.0}
    assert "time" in cfg["selection"]

    # 等値線レイヤーを追加 + 時刻送り
    _button(at, "add_map0").set_value(True)
    at.run()
    assert not at.exception
    _button(at, "next_map0_time").set_value(True)
    at.run()
    assert not at.exception and not at.error

    # 2 次元座標格子では範囲指定は既定 OFF (render が投影座標の範囲を表示範囲にする)。
    # チェックすると初期値はデータの経緯度範囲
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["projection"]["name"] == "LambertConformal"  # 格子から推定
    assert cfg["region"] is None
    assert any("余白なし" in c.value for c in at.caption)
    at.checkbox(key="reg_check_map0_LambertConformal").set_value(True)
    at.run()
    assert not at.exception and not at.error
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["region"] is not None
    assert 90.0 < cfg["region"]["lon_min"] < 100.0


def test_curvilinear_and_regular_grid_mixed(sample_path, curvilinear_sample_path):
    """1 次元格子 (ds0) と curvilinear (ds1) の混在: ds1 のレイヤーでも水平面の dim (y / x)
    が「固定する次元」に出ないこと (レイヤーの dataset 自身の水平 dim で判定)。逆順も同じ。"""
    for first, second in ((sample_path, curvilinear_sample_path),
                          (curvilinear_sample_path, sample_path)):
        at = _load_app(first)
        at.text_input(key="_new_file_path").set_value(second)
        at.run()
        _button(at, "add_file").set_value(True)
        at.run()
        assert not at.exception
        assert len(at.session_state["datasets"]) == 2
        at.selectbox(key="fill_map0_0_ds").set_value("ds1")
        at.run()
        assert not at.exception
        assert not at.error
        keys = {w.key for w in at.selectbox}
        for d in ("y", "x", "lat", "lon"):
            assert f"fill_map0_0_sel_{d}" not in keys, d
        sel = at.session_state["panel_cfg_0"]["layers"][0]["selection"]
        assert set(sel) <= {"lev", "level"}, sel


def test_coord_files_ui(curvilinear_bare_paths):
    """経緯度が別ファイルのデータ (ClimCORE 形式): 座標ファイルを「適用」すると地図モードが
    開き、再現スクリプトに座標ファイルの open + assign_coords が入る。「解除」で元に戻る。"""
    bare_path, lonlat_path = curvilinear_bare_paths
    at = _load_app(bare_path)
    assert not at.exception
    # 経緯度が無いので地図モードは出ない (時間断面 (time × lev)・1次元・集計系は出る)
    assert "水平断面図" not in at.selectbox(key="plot_mode_0").options
    assert at.session_state["datasets"][0].get("coord_paths") is None

    at.text_input(key="_coord_lon_0").set_value(lonlat_path)
    at.run()
    _button(at, "_coord_apply_0").set_value(True)
    at.run()
    assert not at.exception
    at.run()  # st.rerun() 後の stale 要素を掃除
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.session_state["datasets"][0]["coord_paths"] == [lonlat_path]
    # 経緯度が付いたので地図モードが開き、断面・集計系は消える
    assert at.selectbox(key="plot_mode_0").options == ["水平断面図", "1次元プロット", "2次元プロット"]
    at.selectbox(key="plot_mode_0").set_value("map")
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    cfg = at.session_state["panel_cfg_0"]
    assert cfg["layers"][0]["selection"] == {"lev": 1000.0}
    keys = {w.key for w in at.selectbox}
    assert "fill_map0_0_sel_y" not in keys and "fill_map0_0_sel_x" not in keys
    # 再現スクリプト (メインエリアの st.code) に座標ファイルの行が入る
    codes = [c.value for c in at.code if "xr.open_dataset(" in c.value]
    assert codes, "再現スクリプトが表示されていない"
    assert "ds0.assign_coords({'FLON':" in codes[0]
    assert repr(lonlat_path) in codes[0]

    # 解除 → 地図モードが消える
    _button(at, "_coord_clear_0").set_value(True)
    at.run()
    at.run()
    assert not at.exception
    assert at.session_state["datasets"][0].get("coord_paths") is None
    assert "水平断面図" not in at.selectbox(key="plot_mode_0").options


def test_coord_files_ignored_when_lonlat_present(curvilinear_sample_path,
                                                 curvilinear_bare_paths):
    """本体に経緯度がある dataset に座標ファイルを指定しても無視され、通知だけ出る。"""
    _, lonlat_path = curvilinear_bare_paths
    at = _load_app(curvilinear_sample_path)
    at.text_input(key="_coord_lon_0").set_value(lonlat_path)
    at.run()
    _button(at, "_coord_apply_0").set_value(True)
    at.run()
    at.run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert any("座標ファイルは無視" in i.value for i in at.info)
    assert at.session_state["panel_cfg_0"]["layers"][0]["selection"] == {"lev": 1000.0}


def test_coord_files_browser_in_allowed_dirs_mode(curvilinear_bare_paths, monkeypatch):
    """許可ディレクトリ運用: 座標ファイル欄にも「許可ディレクトリから選ぶ」ブラウザが出て、
    選んだファイルが入れ先 (経度 / 緯度) の欄に入り、「適用」で地図モードが開く。"""
    import os as _os
    bare_path, lonlat_path = curvilinear_bare_paths
    monkeypatch.setenv("CC_ALLOWED_DIRS", _os.path.dirname(bare_path))
    at = AppTest.from_file(_APP_PATH, default_timeout=60)
    at.run()
    at.text_input(key="_new_file_path").set_value(bare_path)
    at.run()
    _button(at, "add_file").set_value(True)
    at.run()
    assert not at.exception
    # 参照 (tkinter) ボタンは無く、ブラウザのトグルがある
    keys = {b.key for b in at.get("button")}
    assert "_coord_browse_lon_0" not in keys
    at.toggle(key="_coord_showbrowse_0").set_value(True)
    at.run()
    assert not at.exception
    fname = _os.path.basename(lonlat_path)
    _button(at, f"_coordbrowse_0_file_{fname}").set_value(True)
    at.run()
    at.run()
    assert not at.exception
    assert at.session_state["_coord_lon_0"] == lonlat_path
    # 入れ先を緯度に切り替えて同じファイルを選ぶ → 緯度の欄に入る
    at.radio(key="_coord_target_0").set_value("lat")
    at.run()
    _button(at, f"_coordbrowse_0_file_{fname}").set_value(True)
    at.run()
    at.run()
    assert at.session_state["_coord_lat_0"] == lonlat_path
    _button(at, "_coord_apply_0").set_value(True)
    at.run()
    at.run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    # 同じファイルは 1 つにまとまる
    assert at.session_state["datasets"][0]["coord_paths"] == [lonlat_path]
    assert "水平断面図" in at.selectbox(key="plot_mode_0").options


def test_map_region_defaults_to_data_bounds(sample_path, curvilinear_sample_path, tmp_path):
    """水平面図の領域指定の初期値: 領域データ (2 次元座標 / 切り出した 1 次元格子) では
    データの経緯度範囲が既定で ON、全球データは従来どおり OFF (全球表示)。"""
    import numpy as np
    import xarray as xr
    # 全球データ: 変わらず範囲指定 OFF
    at = _load_app(sample_path)
    assert at.session_state["panel_cfg_0"]["region"] is None

    # 2 次元座標: 全格子点を含む矩形 (経度は日付変更線をまたぐ弧) が既定で ON。
    # 投影法は格子から推定した LambertConformal で、中心経度は格子の中心経線 140
    from climcanvas.core import dataset as mc_dataset
    at = _load_app(curvilinear_sample_path)
    cfg = at.session_state["panel_cfg_0"]
    cv = xr.open_dataset(curvilinear_sample_path)
    # 2 次元座標格子は範囲指定が既定 OFF (投影座標の範囲で表示)。チェックすると
    # 初期値は lonlat_bounds (経度は日付変更線をまたぐ弧)
    assert cfg["region"] is None
    assert cfg["projection"]["name"] == "LambertConformal"
    assert cfg["projection"]["central_longitude"] == 140.0
    at.checkbox(key="reg_check_map0_LambertConformal").set_value(True)
    at.run()
    reg = at.session_state["panel_cfg_0"]["region"]
    b = mc_dataset.lonlat_bounds(cv)
    assert reg == {k: b[k] for k in ("lon_min", "lon_max", "lat_min", "lat_max")}
    assert reg["lat_min"] == float(np.nanmin(cv["lat"].values))
    assert not at.error

    # 1 次元格子の領域切り出し (経度 100–200、緯度 0–60)
    sub = tmp_path / "regional.nc"
    xr.open_dataset(sample_path).sel(lon=slice(100.0, 200.0), lat=slice(0.0, 60.0)).to_netcdf(sub)
    at = _load_app(str(sub))
    reg = at.session_state["panel_cfg_0"]["region"]
    assert reg == {"lon_min": 100.0, "lon_max": 200.0, "lat_min": 0.0, "lat_max": 60.0}
    assert not at.error

    # 経度が全周・緯度が北半球だけ: 経度は規約の全周 (0–360)、緯度はデータ範囲
    nh = tmp_path / "nh.nc"
    xr.open_dataset(sample_path).sel(lat=slice(0.0, 90.0)).to_netcdf(nh)
    at = _load_app(str(nh))
    reg = at.session_state["panel_cfg_0"]["region"]
    assert reg == {"lon_min": 0.0, "lon_max": 360.0, "lat_min": 0.0, "lat_max": 90.0}
    assert not at.error
    # 極投影でも領域データはデータ範囲 (半球固定ではなく)
    at.selectbox(key="proj_name_map0").set_value("NorthPolarStereo")
    at.run()
    assert not at.exception and not at.error
    assert at.session_state["panel_cfg_0"]["region"]["lat_min"] == 0.0


def test_grid_projection_sets_initial_projection(curvilinear_sample_path,
                                                 curvilinear_bare_paths, sample_path):
    """2 次元座標格子の投影法を推定できたら、初期の投影法・中心経度・標準緯線・中心緯度に
    使う (埋め込み版でも座標ファイル経由でも)。1 次元格子は従来どおり PlateCarree。"""
    at = _load_app(sample_path)
    assert at.session_state["panel_cfg_0"]["projection"]["name"] == "PlateCarree"

    at = _load_app(curvilinear_sample_path)
    proj = at.session_state["panel_cfg_0"]["projection"]
    assert proj["name"] == "LambertConformal"
    assert proj["standard_parallels"] == [30.0, 60.0]
    assert proj["central_longitude"] == 140.0
    assert not at.error, [e.value for e in at.error]
    assert any("自動認識" in c.value for c in at.caption)

    # 座標ファイルで後から経緯度を付けた場合も同じ
    bare_path, lonlat_path = curvilinear_bare_paths
    at = _load_app(bare_path)
    at.text_input(key="_coord_lon_0").set_value(lonlat_path)
    at.run()
    _button(at, "_coord_apply_0").set_value(True)
    at.run()
    at.run()
    at.selectbox(key="plot_mode_0").set_value("map")
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    proj = at.session_state["panel_cfg_0"]["projection"]
    assert proj["name"] == "LambertConformal" and proj["central_longitude"] == 140.0


def test_coord_files_survive_session_roundtrip(curvilinear_bare_paths):
    """座標ファイルの指定 (datasets エントリの coord_paths) がセッションのスロット保存 →
    別セッションでの復元で戻り、復元直後から地図モードが使える。"""
    from test_session_roundtrip_fuzz import _save_wip, _restore_wip
    bare_path, lonlat_path = curvilinear_bare_paths
    at = _load_app(bare_path)
    at.text_input(key="_coord_lon_0").set_value(lonlat_path)
    at.run()
    _button(at, "_coord_apply_0").set_value(True)
    at.run()
    at.run()
    at.selectbox(key="plot_mode_0").set_value("map")
    at.run()
    assert not at.exception
    _save_wip(at, "coordtest")

    at2 = AppTest.from_file(_APP_PATH, default_timeout=60)
    at2.run()
    _restore_wip(at2, "coordtest")
    assert at2.session_state["datasets"][0]["coord_paths"] == [lonlat_path]
    assert at2.session_state["plot_mode_0"] == "map"
    assert not at2.error, [e.value for e in at2.error]
    cfg = at2.session_state["panel_cfg_0"]
    assert cfg["projection"]["name"] == "LambertConformal"
    assert cfg["layers"][0]["selection"] == {"lev": 1000.0}
