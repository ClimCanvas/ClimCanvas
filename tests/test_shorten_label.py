# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""セッション復元・保存の selectbox で使う表示短縮 (shorten_label) と
caption 用エスケープ (md_plain) のユニットテスト。

どちらも表示専用で、選択値 (セッションのディレクトリ・ファイル名) は変えない。
"""

import os

import pytest

from climcanvas.ui.widgets import _display_width, md_plain, shorten_label


def test_short_string_is_unchanged():
    assert shorten_label("short") == "short"
    assert shorten_label("a" * 30, max_width=30) == "a" * 30


def test_home_is_abbreviated_to_tilde():
    home = os.path.expanduser("~")
    assert shorten_label(os.path.join(home, ".climcanvas", "sessions")) == \
        "~/.climcanvas/sessions"
    assert shorten_label(home) == "~"
    # 前方一致だけでは別ユーザーのホーム (例 /home/alice に対する /home/alice2) まで潰れる
    assert not shorten_label(home + "2/x").startswith("~")


def test_middle_is_elided_keeping_head_and_tail():
    name = "monsoon-precip-anomaly-composite-elnino-2015-jja-fig3-v2"
    out = shorten_label(name, max_width=30)
    assert "…" in out
    assert _display_width(out) <= 30
    head, tail = out.split("…")
    assert name.startswith(head) and name.endswith(tail)
    assert tail.endswith("fig3-v2")  # 末尾 (日付・版) が残る


def test_cjk_counts_as_double_width():
    name = "エルニーニョ2015年冬の降水偏差の合成図_第3図_改訂2"
    out = shorten_label(name, max_width=30)
    assert "…" in out
    assert _display_width(out) <= 30
    head, tail = out.split("…")
    assert name.startswith(head) and name.endswith(tail)


@pytest.mark.parametrize("width", [3, 10, 31, 100])
def test_result_never_exceeds_max_width(width):
    name = "x" * 50 + "全角も混ぜる" + "y" * 50
    out = shorten_label(name, max_width=width)
    assert _display_width(out) <= width


def test_md_plain_escapes_markdown_syntax():
    src = "el*nino_2015 :blue[x] <b> $ #1 `c` ~s~ |t| \\"
    out = md_plain(src)
    for ch in "*_:[]<>$#`~|\\":
        assert "\\" + ch in out
    # 記号でない文字はそのまま
    assert md_plain("plain-name.2015") == "plain-name.2015"
    assert md_plain("日本語の名前") == "日本語の名前"
