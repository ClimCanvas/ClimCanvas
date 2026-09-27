# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""i18n (第2〜3段階) の検証。

キー = 日本語原文の gettext 流 (docs/i18n_plan.md 3節):
- コードから AST で抽出した全キー (t() の引数 + *_LABELS 辞書の値 +
  header_label) を en.json が網羅していること (欠落・余剰の検出)
- placeholder ({name}) が原文と訳文で一致すること
- t() のフォールバック (訳が無ければ原文 = キーを返す)
- UI 言語キー (ui_lang) の WIP 除外 / プリセット包含
- 言語切替の metamorphic 検査: 言語を変えても panel 設定 (figure_config の素) が
  不変であること。render は言語を一切参照しないため、panel 設定の一致は
  出力画像のピクセル一致を意味する (画像一致は test_consistency.py が担保)
"""

import ast
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


from climcanvas.ui import i18n
from climcanvas.ui.state_io import _is_preset_key, _is_session_excluded

_ROOT = Path(__file__).resolve().parent.parent
# app.py の絶対パス。streamlit 1.63 以降の AppTest.from_file は相対パスを「呼び出し元の
# テストファイルの場所」基準で解決する (1.58 までは cwd 基準) ので、絶対パスで渡す
_APP_PATH = str(_ROOT / "app.py")
_LOCALES_DIR = Path(i18n.__file__).parent / "locales"
_LANGS = sorted(p.stem for p in _LOCALES_DIR.glob("*.json"))
_JA = re.compile(r"[ぁ-んァ-ヶ一-龠]")
_UI_FILES = [_ROOT / "app.py"] + sorted((_ROOT / "climcanvas" / "ui").glob("*.py"))


def _load(lang):
    return json.loads((_LOCALES_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def _source_keys():
    """コード中の翻訳キーを抽出する (t() 引数 + *_LABELS 値 + header_label)。"""
    keys = set()
    for path in _UI_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "t" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                names = [tg.id for tg in node.targets
                         if isinstance(tg, ast.Name)]
                if any(n.endswith("_LABELS") for n in names) \
                        and "LANG_LABELS" not in names:
                    for v in node.value.values:
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            keys.add(v.value)
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "header_label" and \
                            isinstance(kw.value, ast.Constant):
                        keys.add(kw.value.value)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                names = [a.arg for a in args.args][-len(args.defaults):] \
                    if args.defaults else []
                for name, d in zip(names, args.defaults):
                    if name == "header_label" and isinstance(d, ast.Constant):
                        keys.add(d.value)
    return keys


def _placeholders(text):
    return set(re.findall(r"{(\w+)(?::[^}]*)?}", text))


def test_locale_files_exist_for_all_langs():
    """LANG_LABELS の全言語に locale ファイルがあり、逆に余剰ファイルも無い。"""
    assert set(_LANGS) == set(i18n.LANG_LABELS)


@pytest.mark.parametrize("lang", [lg for lg in _LANGS if lg != "ja"])
def test_locales_cover_all_source_keys(lang):
    """非 ja の全 locale がコード中の日本語キーを網羅し、未使用キーも無い。

    キー = 日本語原文なので、非日本語の原文 (例 "wiggle") は翻訳不要
    (t() が素通しする)。locale の対象は日本語を含むキーのみ。
    """
    src = {k for k in _source_keys() if _JA.search(k)}
    table = set(_load(lang))
    missing = src - table
    orphan = table - src
    assert not missing, \
        f"{lang}.json に欠落 ({len(missing)}件): {sorted(missing)[:5]}"
    assert not orphan, \
        f"{lang}.json に未使用キー ({len(orphan)}件): {sorted(orphan)[:5]}"


def test_locale_values_are_nonempty_strings():
    for lang in _LANGS:
        for k, v in _load(lang).items():
            assert isinstance(v, str) and v.strip(), f"{lang}.json: {k} が空"


def test_format_placeholders_match_key():
    """訳文の {name} プレースホルダは原文 (= キー) と同じ集合であること。"""
    for lang in _LANGS:
        for k, v in _load(lang).items():
            assert _placeholders(v) == _placeholders(k), \
                f"{lang}.json: {k!r} の placeholder が原文と不一致"


def test_t_unknown_key_returns_key():
    """訳が無いキーは原文 (キー) がそのまま返る = ja の通常経路。"""
    assert i18n.t("未知のキー 123") == "未知のキー 123"


def test_t_uses_translation_when_available(monkeypatch):
    tables = {"en": {"色": "Color"}}
    monkeypatch.setattr(i18n, "load_locale", lambda lang: tables.get(lang, {}))
    monkeypatch.setattr(i18n, "current_lang", lambda: "en")
    assert i18n.t("色") == "Color"
    assert i18n.t("訳の無いキー") == "訳の無いキー"  # 原文フォールバック


def test_t_formats_placeholders(monkeypatch):
    tables = {"en": {"{n} 件": "{n} items"}}
    monkeypatch.setattr(i18n, "load_locale", lambda lang: tables.get(lang, {}))
    monkeypatch.setattr(i18n, "current_lang", lambda: "en")
    assert i18n.t("{n} 件", n=3) == "3 items"
    monkeypatch.setattr(i18n, "current_lang", lambda: "ja")
    assert i18n.t("{n} 件", n=3) == "3 件"


def test_tr_labels_translates_values(monkeypatch):
    tables = {"en": {"実線": "solid (label)"}}
    monkeypatch.setattr(i18n, "load_locale", lambda lang: tables.get(lang, {}))
    monkeypatch.setattr(i18n, "current_lang", lambda: "en")
    out = i18n.tr_labels({"solid": "実線", "dashed": "破線"})
    assert out == {"solid": "solid (label)", "dashed": "破線"}


def test_current_lang_defaults_to_ja_outside_session():
    assert i18n.current_lang() == i18n.DEFAULT_LANG == "ja"


# 表示文字列を受け取る呼び出し (t() ラップ必須の位置)。
# 属性呼び出し (st.X / col.X): ラベル (第1位置引数) と help= / placeholder=
_WRAP_WIDGETS = {"selectbox", "multiselect", "radio", "checkbox", "toggle",
                 "slider", "select_slider", "number_input", "text_input",
                 "text_area", "button", "download_button", "file_uploader",
                 "expander", "spinner", "popover"}
# 属性呼び出しで本文 (第1位置引数) を表示する系
_WRAP_DISPLAYS = {"caption", "markdown", "warning", "error", "success", "info",
                  "title", "header", "subheader", "write", "toast", "progress"}
# 名前呼び出しの自作ヘルパー (第1引数がラベル)
_WRAP_HELPERS = {"section_header", "fontsize_input", "color_selector",
                 "strftime_format_ui", "cmap_selector"}
_WRAP_KWARGS = {"help", "placeholder", "text"}


def test_no_unwrapped_japanese_display_strings():
    """表示位置の日本語リテラルはすべて t() で包まれていること。

    包み忘れは網羅検査 (locale とのキー照合) では検出できない —
    t() の引数にならない文字列は「翻訳対象」と認識されず、全言語で
    日本語のまま表示されてしまう。ここで AST 検査して塞ぐ。
    f-string も対象 (placeholder 形式 t("... {x}", x=...) に直すこと)。
    """
    violations = []
    for path in _UI_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # t("...") の第1引数になっている Constant は適法
        wrapped = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "t" and node.args):
                wrapped.add(id(node.args[0]))
                # t の f-string 引数は無い前提だが、中の定数も適法扱い
                for sub in ast.walk(node.args[0]):
                    wrapped.add(id(sub))

        def check(arg, where):
            for sub in ast.walk(arg):
                if id(sub) in wrapped:
                    continue
                if (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                        and _JA.search(sub.value)):
                    violations.append(
                        f"  {path.name}:{sub.lineno} {where}: "
                        f"{sub.value[:40]!r} → t() で包み、各 locale に訳を追加")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in (
                    _WRAP_WIDGETS | _WRAP_DISPLAYS):
                if node.args:
                    check(node.args[0], f"{func.attr} ラベル")
                for kw in node.keywords:
                    if kw.arg in _WRAP_KWARGS:
                        check(kw.value, f"{func.attr} {kw.arg}=")
            elif isinstance(func, ast.Name) and func.id in _WRAP_HELPERS:
                if node.args:
                    check(node.args[0], f"{func.id} ラベル")
                for kw in node.keywords:
                    if kw.arg in _WRAP_KWARGS:
                        check(kw.value, f"{func.id} {kw.arg}=")

    assert not violations, \
        "t() で包まれていない表示文字列があります:\n" + "\n".join(violations)


def test_render_error_ids_match_ui_mapping():
    """RenderError.MESSAGES と app.py の _RENDER_ERROR_LABELS の msg_id が一致。

    app.py は import すると Streamlit スクリプトが走るため AST で読む。
    """
    from climcanvas.core.render import RenderError

    tree = ast.parse((_ROOT / "app.py").read_text(encoding="utf-8"))
    ui_ids = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            names = [tg.id for tg in node.targets if isinstance(tg, ast.Name)]
            if "_RENDER_ERROR_LABELS" in names:
                ui_ids = {k.value for k in node.value.keys
                          if isinstance(k, ast.Constant)}
    assert ui_ids is not None, "_RENDER_ERROR_LABELS が app.py に見つからない"
    core_ids = set(RenderError.MESSAGES)
    assert ui_ids == core_ids, \
        f"UI のみ: {ui_ids - core_ids} / core のみ: {core_ids - ui_ids}"


def test_render_error_str_is_english():
    """RenderError の str() は英語 (スタンドアロン利用・コア単体で読めること)。"""
    from climcanvas.core.render import RenderError

    e = RenderError("layout_too_small", nrows=1, ncols=2, n_panels=3)
    assert e.msg_id == "layout_too_small"
    assert e.params == {"nrows": 1, "ncols": 2, "n_panels": 3}
    assert str(e) == ("Cannot place 3 panels in a 1x2 layout. "
                      "Increase rows/columns.")
    assert not _JA.search(str(e))
    # ValueError の subclass (parse_mosaic の呼び出し側は ValueError で捕捉)
    assert isinstance(e, ValueError)


def test_ui_lang_key_is_session_excluded_but_preset_included():
    """ui_lang は WIP (作業状態) に入らず、起動時プリセットには入る。"""
    assert _is_session_excluded("ui_lang")
    assert _is_preset_key("ui_lang")


# --- AppTest: 言語切替の metamorphic 検査 ---

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


def test_language_switch_keeps_panel_config(sample_path):
    """UI 言語を ja → en に切り替えても panel 設定が変わらない。"""
    import copy

    at = _load_app(sample_path)
    assert at.session_state["ui_lang"] == "ja"  # 既定は日本語
    cfg_ja = copy.deepcopy(at.session_state["panel_cfg_0"])

    at.selectbox(key="ui_lang").set_value("en")
    at.run()
    assert not at.exception
    assert at.session_state["ui_lang"] == "en"
    cfg_en = at.session_state["panel_cfg_0"]
    assert cfg_en == cfg_ja, "言語切替で panel 設定が変わった (保存値にラベル依存が残っている)"


@pytest.mark.parametrize("lang", ["en", "zh_CN", "ko"])
def test_non_ja_ui_renders_and_works(lang, sample_path):
    """各言語に切り替えた状態でモード変更・再構築が例外なく動く。"""
    at = _load_app(sample_path)
    at.selectbox(key="ui_lang").set_value(lang)
    at.run()
    assert not at.exception
    at.selectbox(key="plot_mode_0").set_value("line")
    at.run()
    assert not at.exception
    assert at.session_state["panel_cfg_0"]["plot_type"] == "line_1d"


def test_language_not_saved_into_wip(sample_path, tmp_path, monkeypatch):
    """WIP スロット保存に ui_lang が含まれず、復元しても言語が変わらない。"""
    monkeypatch.setenv("CC_SESSION_DIRS", str(tmp_path))
    at = _load_app(sample_path)
    at.selectbox(key="ui_lang").set_value("en")
    at.run()
    at.text_input(key="_session_save_name").set_value("slot1")
    at.run()
    _button(at, "_session_save_btn").set_value(True)
    at.run()
    assert not at.exception
    saved = json.loads((tmp_path / "slot1.json").read_text(encoding="utf-8"))
    assert "ui_lang" not in saved

    # 新セッション (既定 ja) で復元しても言語は ja のまま
    at2 = _load_app(sample_path)
    at2.selectbox(key="_session_restore_slot").set_value("slot1")
    at2.run()
    _button(at2, "_session_restore_btn").set_value(True)
    at2.run()
    assert not at2.exception
    assert at2.session_state["ui_lang"] == "ja"
