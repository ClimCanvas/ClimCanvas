# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""UI 文字列の多言語化 (i18n) — 翻訳辞書のロードと `t()`。

docs/i18n_plan.md 第2〜3段階。翻訳は UI 層に閉じる (コア層からは import しない)。

**キー = 日本語原文 (gettext 流)**。コードは表示文字列をそのまま `t("線の太さ")`
のように包む。ja では locale にキーが無くてもキー自身 (= 原文) が返るので
ja.json は空でよい。en.json は「原文 → 英訳」の辞書で、コード中の全キーを
網羅する (tests/test_i18n.py が AST 抽出で検査する)。

使い方の規約:
- ラベル・help・placeholder・caption 等の **表示専用文字列** だけを t() で包む。
  widget の保存値 (options の中立キー) や config に入る値は包まない
- format_func には t() や t を含む lambda を渡さず、run 中に作った翻訳済み
  dict (`tr_labels(...)` の返り値) の `.get` を渡す (AppTest の直列化が
  セッション文脈の外で format_func を呼ぶため。implementation_checklist.md D 節)
- プレースホルダは `t("`{name}` が見つかりません", name=x)` のように
  str.format 形式で。プレースホルダ名は原文と訳文で一致させる
"""

import json
from functools import lru_cache
from pathlib import Path

import streamlit as st

DEFAULT_LANG = "ja"
# 言語選択肢 (中立キー → その言語自身での表示名)。表示名自体は翻訳しない
LANG_LABELS = {"ja": "日本語", "en": "English",
               "zh_CN": "简体中文", "ko": "한국어"}
_LOCALES_DIR = Path(__file__).parent / "locales"


@lru_cache(maxsize=None)
def load_locale(lang: str) -> dict:
    """`locales/<lang>.json` を読み込む (原文 → 訳文)。無ければ空 dict。"""
    path = _LOCALES_DIR / f"{lang}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def current_lang() -> str:
    """session_state の UI 言語 (key="ui_lang")。未設定・不正値は DEFAULT_LANG。"""
    try:
        lang = st.session_state.get("ui_lang", DEFAULT_LANG)
    except Exception:
        # Streamlit の実行文脈の外 (ユニットテスト等) では既定言語
        return DEFAULT_LANG
    return lang if lang in LANG_LABELS else DEFAULT_LANG


def t(key: str, **fmt) -> str:
    """原文 (日本語) を現在の UI 言語の文字列にする。

    訳が無ければ原文をそのまま返す (ja はこの経路)。fmt を渡すと
    str.format() で埋め込む (例 t("{n} 件", n=3))。
    """
    text = load_locale(current_lang()).get(key)
    if text is None:
        text = key
    return text.format(**fmt) if fmt else text


def tr_labels(labels: dict) -> dict:
    """「中立キー → 原文ラベル」辞書を現在の言語の表示辞書に変換する。

    selectbox/radio の format_func 用: `format_func=tr_labels(X).get` と
    **widget を作る式の中 (= run 中)** で呼ぶこと。返り値は素の dict なので
    直列化時に翻訳が再実行されず、描画時の言語と食い違わない。
    """
    return {k: t(v) for k, v in labels.items()}
