# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""mosaic 配置文字列 (figure.layout.mosaic) と行・列比率の解決ロジックのユニットテスト。

render.parse_mosaic / panel_subplot_specs / grid_ratio_kwargs / cell_grid_bounds を
検証する。render↔scriptgen の画像一致は tests/test_consistency.py の
"multipanel-mosaic" / "multipanel-grid-ratios" が担当。
"""

import pytest

from climcanvas.core.render import (RenderError, cell_grid_bounds,
                                    grid_ratio_kwargs, panel_subplot_specs,
                                    parse_mosaic)


def test_parse_horizontal_merge():
    """ユーザーケース: 2行3列で下段の2〜3列目を結合。"""
    assert parse_mosaic("ABC;DEE", 5) == (2, 3, [1, 2, 3, 4, (5, 6)])


def test_parse_vertical_merge_and_empty_cell():
    """横結合 + 右列の上下貫通 + 空きセル "."。"""
    assert parse_mosaic("AAB;C.B", 3) == (2, 3, [(1, 2), (3, 6), 4])


def test_parse_block_merge():
    """2×2 のブロック結合 (左上と右下のセル番号になる)。"""
    assert parse_mosaic("AAB;AAC", 3) == (2, 3, [(1, 5), 3, 6])


def test_parse_single_cell_and_row_major_order():
    """結合なしは従来の行優先と同じ int 列。ラベルは初出順に対応する。"""
    assert parse_mosaic("AB;CD", 4) == (2, 2, [1, 2, 3, 4])
    # 初出順なのでラベル文字自体は何でもよい (アルファベット順ではない)
    assert parse_mosaic("ZX;YW", 4) == (2, 2, [1, 2, 3, 4])


def test_parse_ignores_whitespace_and_newlines():
    assert parse_mosaic(" A B C \n D E E ", 5) == parse_mosaic("ABC;DEE", 5)


# エラーは msg_id で検証する (メッセージ本文は i18n 第4段階で英語化され、
# UI 側で翻訳されるため本文への match はしない)

def test_parse_rejects_non_rectangle():
    with pytest.raises(RenderError) as ei:
        parse_mosaic("AA;AB", 2)  # A が L字
    assert ei.value.msg_id == "mosaic_not_rect"


def test_parse_rejects_split_label():
    with pytest.raises(RenderError) as ei:
        parse_mosaic("A.A", 1)  # 同じラベルが離れている
    assert ei.value.msg_id == "mosaic_not_rect"


def test_parse_rejects_ragged_rows():
    with pytest.raises(RenderError) as ei:
        parse_mosaic("AB;CDE", 5)
    assert ei.value.msg_id == "mosaic_ragged"


def test_parse_rejects_panel_count_mismatch():
    with pytest.raises(RenderError) as ei:
        parse_mosaic("AB;CD", 3)
    assert ei.value.msg_id == "mosaic_count_mismatch"


def test_parse_rejects_empty():
    # 呼び出し側 (app.py) は ValueError で捕捉する — subclass 関係も検証
    with pytest.raises(ValueError):
        parse_mosaic(" ; ", 0)


def test_specs_row_major_without_mosaic():
    """mosaic なし (旧 config 含む) は従来の行優先 (i+1)。"""
    cfg = {"figure": {"layout": {"nrows": 2, "ncols": 2}},
           "panels": [{}, {}, {}]}
    assert panel_subplot_specs(cfg) == (2, 2, [1, 2, 3])


def test_specs_mosaic_overrides_grid_dims():
    """mosaic 指定時は nrows/ncols より mosaic の寸法が優先される。"""
    cfg = {"figure": {"layout": {"nrows": 1, "ncols": 1, "mosaic": "AB;CC"}},
           "panels": [{}, {}, {}]}
    assert panel_subplot_specs(cfg) == (2, 2, [1, 2, (3, 4)])


# --- 「編集するパネル」ボタングリッドのセル配置 (ui.panel_ui.panel_selector_cells) ---
# ボタンの並びが図のパネル配置 (parse_mosaic) と一致すること (2026-08-25 報告:
# "ABC.;DEFG" でボタンが 1-4 / 5-7 の行優先になり図とずれていた)

def test_selector_cells_mosaic_with_empty_cell():
    from climcanvas.ui.panel_ui import panel_selector_cells
    # 報告の再現ケース: 2行4列、上段 A B C (空)、下段 D E F G
    cells = panel_selector_cells(2, 4, 7, "ABC.;DEFG")
    assert cells == {0: 0, 1: 1, 2: 2,          # 上段 = パネル1〜3、セル3は空き
                     4: 3, 5: 4, 6: 5, 7: 6}    # 下段 = パネル4〜7


def test_selector_cells_merged_cell_uses_top_left():
    from climcanvas.ui.panel_ui import panel_selector_cells
    # 結合セル (DEE の E) はその左上のセルに置く
    cells = panel_selector_cells(2, 3, 5, "ABC;DEE")
    assert cells == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    # 縦結合 + 空きセル: A が左列2行を結合
    cells = panel_selector_cells(2, 2, 2, "AB;A.")
    assert cells == {0: 0, 1: 1}


def test_selector_cells_fallback_row_major():
    from climcanvas.ui.panel_ui import panel_selector_cells
    # mosaic 無し / 不正 (パネル数不一致) は従来の行優先
    assert panel_selector_cells(2, 2, 3, None) == {0: 0, 1: 1, 2: 2}
    assert panel_selector_cells(2, 2, 3, "AB;CD") == {0: 0, 1: 1, 2: 2}


# --- 行・列の大きさの比率 (layout.width_ratios / height_ratios) ---

def _ratio_cfg(w=None, h=None):
    return {"figure": {"layout": {"width_ratios": w, "height_ratios": h}},
            "panels": []}


def test_grid_ratio_kwargs_resolution():
    """両方 None は空 dict (従来経路)。指定時は float 化して返す。"""
    assert grid_ratio_kwargs(_ratio_cfg(), 2, 3) == {}
    assert grid_ratio_kwargs(_ratio_cfg([2, 1, 1], [2, 1]), 2, 3) == {
        "width_ratios": [2.0, 1.0, 1.0], "height_ratios": [2.0, 1.0]}


def test_grid_ratio_kwargs_rejects_len_mismatch():
    with pytest.raises(RenderError) as ei:
        grid_ratio_kwargs(_ratio_cfg([2, 1]), 2, 3)  # 列数3 に対して2個
    assert ei.value.msg_id == "ratio_len_mismatch"


def test_grid_ratio_kwargs_rejects_invalid_values():
    for bad in ([1.0, -1.0], [1.0, 0.0], ["x", "y"], 3):
        with pytest.raises(ValueError) as ei:  # app.py は ValueError で捕捉
            grid_ratio_kwargs(_ratio_cfg(None, bad), 2, 3)
        assert ei.value.msg_id == "ratio_invalid"


def test_cell_grid_bounds():
    """セル番号 (1始まり行優先) → 0始まり (r0, r1, c0, c1) の変換。"""
    assert cell_grid_bounds(1, 3) == (0, 0, 0, 0)
    assert cell_grid_bounds(5, 3) == (1, 1, 1, 1)
    assert cell_grid_bounds((1, 5), 3) == (0, 1, 0, 1)   # 2×2 ブロック結合
    assert cell_grid_bounds((3, 6), 3) == (0, 1, 2, 2)   # 右列の縦結合
    assert cell_grid_bounds((4, 6), 3) == (1, 1, 0, 2)   # 下段の横結合


def test_width_ratios_axes_sizes(sample_path):
    """width_ratios=[3,1] の 1×2 で描いた axes の実幅が 3:1 になる (意味論検査)。"""
    import matplotlib.pyplot as plt

    from climcanvas.core import config as mc_config
    from climcanvas.core import dataset as mc_dataset
    from climcanvas.core import render as mc_render

    ds = mc_dataset.open_dataset(sample_path)
    panels = []
    for _ in range(2):
        p = mc_config.default_line_panel()
        p["x_dim"] = "time"
        p["selection"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
        p["layers"] = [mc_config.default_line_layer("ds0", "t")]
        panels.append(p)
    cfg = mc_config.default_figure_config()
    cfg["figure"]["layout"] = {"nrows": 1, "ncols": 2,
                               "width_ratios": [3.0, 1.0]}
    cfg["panels"] = panels
    fig = mc_render.render_figure(cfg, {"ds0": ds})
    try:
        w0 = fig.axes[0].get_position().width
        w1 = fig.axes[1].get_position().width
        assert w0 / w1 == pytest.approx(3.0, rel=1e-6)
    finally:
        plt.close(fig)
