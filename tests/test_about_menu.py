# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""右上メニュー About の中身 (climcanvas/ui/about.py) の検査。

UI 言語ごとに正しいフィードバックフォーム (ja → 日本語版、それ以外 → 英語版) へ
版を事前入力したリンクが出ること、訳し漏れが無いこと、`set_page_config` を 2 回
呼べない古い Streamlit でも落ちないことを確かめる (docs/feedback_channels_plan.md 4.2 節)。
"""

from __future__ import annotations

import pathlib
import re

import pytest
from streamlit.errors import StreamlitAPIException

from climcanvas.ui import about, i18n

# 訳し漏れ (日本語原文のまま) の検出。zh_CN は漢字を使うので仮名だけで見る
_KANA = re.compile(r"[ぁ-んァ-ヶ]")
_HAN = re.compile(r"[一-龠]")


@pytest.mark.parametrize("version, shown", [
    ("0.98.0", "0.98"), ("1.0.0", "1.0"), ("0.90.1", "0.90.1"), ("1.00", "1.00"), ("1.00.1", "1.00.1")])
def test_display_version(version, shown):
    assert about.display_version(version) == shown


@pytest.mark.parametrize("lang", sorted(i18n.LANG_LABELS))
def test_feedback_form_url_follows_ui_language(lang):
    url = about.feedback_form_url(lang, "0.98")
    expected = about.FEEDBACK_FORMS["ja" if lang == "ja" else "en"]
    assert url.startswith(expected + "?")
    assert "usp=pp_url" in url
    assert f"{about.FEEDBACK_VERSION_ENTRY}=0.98" in url


def test_new_issue_url_prefills_version():
    """選択画面 (issues/new/choose) は版を引き継がないので、テンプレートを直接指定する。"""
    assert about.new_issue_url("0.98", "bug.yml") == (
        "https://github.com/ClimCanvas/ClimCanvas/issues/new?template=bug.yml&version=0.98")
    assert about.new_issue_url("1.00", "feature.yml") == (
        "https://github.com/ClimCanvas/ClimCanvas/issues/new?template=feature.yml&version=1.00")


@pytest.mark.parametrize("lang", sorted(i18n.LANG_LABELS))
def test_site_url_follows_ui_language(lang):
    """サイトは ja → 日本語版 (CITATION.cff と同じトップ)、それ以外 → 英語版 /en/。"""
    url = about.site_url(lang)
    assert url == (about.SITE_URL if lang == "ja" else about.SITE_URL_EN)
    assert about.SITE_URL_EN.startswith(about.SITE_URL)
    # リンク文字列は scheme と末尾の / を除いた URL
    assert "https://" + about.site_label(lang) + "/" == url


def test_issue_templates_exist():
    """About が指すテンプレート名が .github/ISSUE_TEMPLATE/ に実在する。"""
    root = pathlib.Path(__file__).resolve().parent.parent / ".github" / "ISSUE_TEMPLATE"
    for name in about.ISSUE_TEMPLATES.values():
        assert (root / name).is_file(), name


@pytest.mark.parametrize("lang", sorted(i18n.LANG_LABELS))
def test_about_markdown(lang, monkeypatch):
    monkeypatch.setattr(i18n, "current_lang", lambda: lang)
    monkeypatch.setattr(about, "current_lang", lambda: lang)
    md = about.about_markdown()
    ver = about.display_version()
    assert f"ver {ver}" in md
    assert about.feedback_form_url(lang, ver) in md
    assert about.new_issue_url(ver, "bug.yml") in md
    assert about.new_issue_url(ver, "feature.yml") in md
    assert "AGPL-3.0-only" in md and "/blob/main/LICENSE" in md
    # サイトは UI 言語に応じて日本語版 / 英語版 (リンク文字列も同じ URL を示す)
    assert f"[{about.site_label(lang)}]({about.site_url(lang)})" in md
    assert (about.SITE_URL_EN in md) == (lang != "ja")
    # Markdown のリンクが 5 本とも崩れていない
    assert len(re.findall(r"\[[^\]]+\]\(https://[^)\s]+\)", md)) == 5
    if lang != "ja":
        assert not _KANA.search(md), md
    if lang in ("en", "ko"):
        assert not _HAN.search(md), md


def test_set_about_menu_tolerates_old_streamlit(monkeypatch):
    """Streamlit 1.46 未満は set_page_config の 2 回目で例外 → About を出さずに進む。"""
    def _raise(**kwargs):
        raise StreamlitAPIException("set_page_config() can only be called once")
    monkeypatch.setattr(about.st, "set_page_config", _raise)
    about.set_about_menu()


def test_set_about_menu_passes_only_about(monkeypatch):
    """メニューは About だけ (Get help / Report a bug は渡さない = 出さない)。"""
    calls = []
    monkeypatch.setattr(about.st, "set_page_config", lambda **kw: calls.append(kw))
    about.set_about_menu()
    assert len(calls) == 1
    assert set(calls[0]) == {"menu_items"}
    assert set(calls[0]["menu_items"]) == {"About"}
