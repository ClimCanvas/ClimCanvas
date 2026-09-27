# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""右上メニュー「About」の中身 — 版と、フィードバック・バグ報告の入口。

docs/feedback_channels_plan.md 4.2 節。メニューのうち About だけを使う (Get help /
Report a bug は出さない)。リンクを開くだけで、アプリからは何も送信しない。

フォームの URL は `.github/ISSUE_TEMPLATE/config.yml` と同じもの
(tests/test_issue_templates.py が一致を検査)。
"""

from urllib.parse import quote

import streamlit as st
from streamlit.errors import StreamlitAPIException

from climcanvas import __version__
from climcanvas.ui.i18n import current_lang, t

# Google フォーム (climcanvas@gmail.com 所有)。ja 以外の UI 言語は英語版へ
FEEDBACK_FORMS = {
    "ja": "https://docs.google.com/forms/d/e/1FAIpQLSe-zXza-DXr8eGdOIgLKBH98luuLQdCooD6fcjCO3fA3gvuKw/viewform",
    "en": "https://docs.google.com/forms/d/e/1FAIpQLSc-tnfYzsqcctESephPCJqbht_TQRl6TPw8NE7sgkmlcAZ68w/viewform",
}
# 版の質問の entry 番号 (英語版を日本語版のコピーで作ったので両版で同じ)。
# 版の質問を作り直すと番号が変わり、版が事前入力されなくなる
FEEDBACK_VERSION_ENTRY = "entry.2010447076"
REPO_URL = "https://github.com/ClimCanvas/ClimCanvas"
# Issue フォーム (bug.yml / feature.yml) の版の欄の id。選択画面に渡したクエリは
# 選んだフォームに引き継がれる
ISSUE_VERSION_FIELD = "version"


def display_version(version: str = __version__) -> str:
    """表示用の版。`__version__` はリリースタグから v を外した文字列 (1.00 / 1.00.1) なのでそのまま。
    旧形式 (0.98.0) の末尾のパッチ .0 だけ落とす (0.90.1 等はそのまま)。"""
    return version.removesuffix(".0")


def feedback_form_url(lang: str, version: str) -> str:
    base = FEEDBACK_FORMS.get(lang, FEEDBACK_FORMS["en"])
    return f"{base}?usp=pp_url&{FEEDBACK_VERSION_ENTRY}={quote(version)}"


def new_issue_url(version: str) -> str:
    return f"{REPO_URL}/issues/new/choose?{ISSUE_VERSION_FIELD}={quote(version)}"


def about_markdown() -> str:
    ver = display_version()
    return "\n".join([
        "**ClimCanvas** — " + t("netCDF 大気・海洋データ可視化 — ver {ver}", ver=ver),
        "",
        "- " + t("ご意見・ご要望 (GitHub アカウント不要): [Google フォーム]({url})",
                 url=feedback_form_url(current_lang(), ver)),
        "- " + t("バグ報告・機能の要望: [GitHub Issues]({url})",
                 url=new_issue_url(ver)),
        "- " + t("ライセンス: [{license}]({url})", license="AGPL-3.0-only",
                 url=f"{REPO_URL}/blob/main/LICENSE"),
    ])


def set_about_menu() -> None:
    """UI 言語が決まった後に呼び、About の中身をその言語で設定する。

    `st.set_page_config` の 2 回目の呼び出しは Streamlit 1.46 以降でだけ許される
    (それ以前は例外)。古い版では About を出さずに進む。
    """
    try:
        st.set_page_config(menu_items={"About": about_markdown()})
    except StreamlitAPIException:
        pass
