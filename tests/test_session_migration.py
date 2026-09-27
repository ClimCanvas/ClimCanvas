# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""i18n 第1段階: 旧形式 WIP (日本語ラベルが保存値) のマイグレーション検査。

2026-07-11 の中立キー化 (docs/i18n_plan.md 第1段階) 以前に保存された
WIP/プリセット JSON は selectbox/radio の表示ラベルをそのまま値に持つ。
復元経路で `_migrate_legacy_value` が新しい中立キーへ変換することを確認する。
"""

import pytest

from climcanvas.ui.state_io import _migrate_legacy_value


@pytest.mark.parametrize("key,old,new", [
    # 描画モード (mode_key 略号へ)
    ("plot_mode_0", "水平断面図", "map"),
    ("plot_mode_3", "2次元プロット(集計)", "agg"),
    # 線種 (ラベル → matplotlib 値)
    ("cont_ls_map0_0", "実線", "solid"),
    ("cont_ls_neg_map0_0", "正と同じ", "same"),
    ("cont_ls_neg_map0_0", "破線", "dashed"),
    ("line_ls_line0_1", "なし", "None"),
    ("gridls_line0", "点線", ":"),
    ("gl_linestyle_map0", "実線", "-"),
    # マーカー
    ("line_marker_line0_0", "○ (circle)", "o"),
    ("mk_map0_2", "☆ (star)", "*"),
    # 色付けモード (「(大きさ)」付きも同じ中立キーに落ちる)
    ("vec_cmode_map0_1", "カラーマップ (大きさ)", "cmap"),
    ("cmode_tr_map0_0", "単色", "single"),
    # 番兵
    ("panel_zvar_scatter0", "(なし)", "(none)"),
    ("sc_ebx_scatter0_0", "(なし)", "(none)"),
    # 各種 radio
    ("fill_avgmode_lat", "範囲平均", "range"),
    ("xtmode_line0", "等間隔 (0 で非表示)", "interval"),
    ("tick_dir_map0", "内側", "in"),
    ("tl_fmt_map0_choice", "年-月 (例 1984-01)", "%Y-%m"),
    ("figsize_preset", "任意の数字", "custom"),
    ("script_path_style", "相対パス", "relative"),
    ("vsec_orient_vsec0", "経度–高度 (緯度を固定)", "lon_height"),
    ("tsec_kind_tsec0", "時間–緯度", "time_lat"),
])
def test_legacy_japanese_labels_migrate(key, old, new):
    assert _migrate_legacy_value(key, old) == new


@pytest.mark.parametrize("key,value", [
    # 既に中立キーの値はそのまま (新形式 WIP の再復元)
    ("plot_mode_0", "map"),
    ("cont_ls_map0_0", "solid"),
    ("cont_ls_neg_map0_0", "same"),
    ("panel_zvar_scatter0", "(none)"),
    ("xtmode_line0", "auto"),
    # 対象外キーの値は日本語でも触らない (自由入力のタイトル等)
    ("title_map0", "実線"),
    ("png_name", "点線.png"),
])
def test_non_legacy_values_pass_through(key, value):
    assert _migrate_legacy_value(key, value) == value


def test_non_string_values_pass_through():
    assert _migrate_legacy_value("plot_mode_0", 3) == 3
    assert _migrate_legacy_value("xtmode_line0", None) is None
    assert _migrate_legacy_value("vsec_range_vsec0", (1000.0, 200.0)) \
        == (1000.0, 200.0)
