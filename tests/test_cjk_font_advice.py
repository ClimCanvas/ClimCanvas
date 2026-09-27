# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""図中 CJK 文字とフォントの整合チェック (i18n 第6段階) のユニットテスト。

cjk_font_advice は UI 層の助言のみ (config・描画は変えない) なので、
render⇄scriptgen の画像一致には影響しない。
"""

from climcanvas.core import config as mc_config
from climcanvas.ui.widgets import cjk_font_advice


def _cfg(title=None, font=None):
    cfg = mc_config.default_figure_config()
    panel = mc_config.default_panel()
    if title is not None:
        panel["title"] = title
    cfg["panels"] = [panel]
    if font is not None:
        cfg["figure"]["font_family"] = font
    return cfg


def test_ascii_only_returns_none():
    assert cjk_font_advice(_cfg(title="Temperature at 500 hPa")) is None


def test_japanese_without_font_warns_with_candidates():
    msg = cjk_font_advice(_cfg(title="500hPa 気温"))
    assert msg is not None
    # bare モード (テスト) では UI 言語 = ja
    assert "フォントが未指定" in msg


def test_hangul_without_font_warns():
    assert cjk_font_advice(_cfg(title="온도 분포")) is not None


def test_dejavu_lacks_cjk_glyphs():
    """DejaVu Sans は matplotlib 同梱で必ず存在し、CJK グリフを持たない。"""
    msg = cjk_font_advice(_cfg(title="気温", font="DejaVu Sans"))
    assert msg is not None
    assert "グリフ" in msg


def test_meta_color_labels_are_ignored():
    """_color_labels 等の図に描かれないメタは CJK 判定の対象外。"""
    cfg = _cfg(title="plain")
    cfg["panels"][0]["_color_labels"] = {"color": "日本語ラベル"}
    assert cjk_font_advice(cfg) is None
