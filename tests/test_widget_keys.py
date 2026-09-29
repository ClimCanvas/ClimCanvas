# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""UI層 (app.py + climcanvas/ui/*.py) の全 Streamlit 入力 widget に明示的な
`key=` があることを保証する静的解析テスト。

key= の無い widget は Streamlit が auto-generated key を割り当てるが、その key は
ラベル文字列・位置に依存して変わるため、JSON 保存・復元 (セッション/プリセット機能) で
状態が再現できない。新しい widget を追加するときは必ず key= を付けること。

これは AST レベルの解析なので UI コードを実行しない。
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _ui_source_files() -> list[pathlib.Path]:
    """UI 層のソースファイル一覧 (app.py + climcanvas/ui/*.py)。"""
    return [_ROOT / "app.py"] + sorted((_ROOT / "climcanvas" / "ui").glob("*.py"))

# 状態を持つ入力 widget。これらは session_state に値を残すので key= が必要
INPUT_WIDGETS = {
    "selectbox", "multiselect",
    "number_input", "text_input", "text_area",
    "date_input", "time_input",
    "slider", "select_slider",
    "checkbox", "toggle", "radio",
    "color_picker",
}
# ボタン系・アップローダーは session_state を programmatic に設定できないため
# 復元対象外 (セッション/プリセットからも除外している)。key= の有無は復元には影響しない
EXEMPT_WIDGETS = {"button", "download_button", "file_uploader", "form_submit_button"}


def _find_widgets_without_keys(filepath: pathlib.Path) -> list[tuple[int, str]]:
    src = filepath.read_text(encoding="utf-8")
    tree = ast.parse(src)
    missing: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        # st.X(...) や col1.X(...) など、属性呼び出しを対象
        if func.attr not in INPUT_WIDGETS:
            continue
        has_key = any(kw.arg == "key" for kw in node.keywords)
        if not has_key:
            missing.append((node.lineno, func.attr))
    return missing


def test_all_input_widgets_have_explicit_keys():
    """UI 層の入力 widget はすべて key= 付きで呼ばれていること。"""
    lines = []
    for path in _ui_source_files():
        rel = path.relative_to(_ROOT)
        for lineno, name in _find_widgets_without_keys(path):
            lines.append(f"  {rel}:{lineno}  st.{name}(...)")
    if lines:
        raise AssertionError(
            "key= が無い入力 widget があります (セッション/プリセット復元が壊れる原因):\n"
            + "\n".join(lines))


def _read_exclude_lists(tree: ast.AST) -> tuple[set[str], tuple[str, ...]]:
    """AST から `_SESSION_EXCLUDE_KEYS` と `_SESSION_EXCLUDE_PREFIXES` を抽出する。

    リストの実体は climcanvas/ui/state_io.py にある (複数ファイルから集めてマージする)。
    """
    keys: set[str] = set()
    prefixes: tuple[str, ...] = ()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                if target.id == "_SESSION_EXCLUDE_KEYS":
                    keys = set(ast.literal_eval(node.value))
                elif target.id == "_SESSION_EXCLUDE_PREFIXES":
                    prefixes = tuple(ast.literal_eval(node.value))
            except (ValueError, SyntaxError):
                pass
    return keys, prefixes


def _extract_button_key_literals(tree: ast.AST):
    """ボタン系 widget の `key=` の値を `(lineno, widget_name, key_text, is_full)` で抽出。

    - 文字列リテラルなら `key_text` がそのキー全体、`is_full=True`
    - f-string なら先頭リテラル部分、`is_full=False` (残りは変数なので prefix で判定)
    - それ以外 (動的な変数等) は除外 (静的に検査不能なため)
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in EXEMPT_WIDGETS:
            continue
        for kw in node.keywords:
            if kw.arg != "key":
                continue
            val = kw.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                out.append((node.lineno, func.attr, val.value, True))
            elif isinstance(val, ast.JoinedStr):
                prefix_parts = []
                for part in val.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        prefix_parts.append(part.value)
                    else:
                        break  # 最初の式パートで終了 → そこまでが固定 prefix
                prefix = "".join(prefix_parts)
                if prefix:
                    out.append((node.lineno, func.attr, prefix, False))
            # 動的な変数のみのキー (例: key=item_id) は静的検査外
    return out


def test_button_keys_are_excluded_from_session_save():
    """ボタン・アップローダーの key は `_SESSION_EXCLUDE_KEYS` か `_SESSION_EXCLUDE_PREFIXES` で
    必ず除外されていること。

    `st.button` / `st.download_button` / `st.file_uploader` / `st.form_submit_button` の
    session_state は programmatic に設定不可。WIP 保存に含まれてしまうと、復元時に
    `StreamlitValueAssignmentNotAllowedError` が出る。
    新しいボタンを追加して exclude リストへの登録を忘れた瞬間に検出する。
    """
    trees = {path: ast.parse(path.read_text(encoding="utf-8"))
             for path in _ui_source_files()}
    exclude_keys: set[str] = set()
    exclude_prefixes: tuple[str, ...] = ()
    for tree in trees.values():
        keys, prefixes = _read_exclude_lists(tree)
        exclude_keys |= keys
        exclude_prefixes += prefixes
    assert exclude_keys and exclude_prefixes, \
        "_SESSION_EXCLUDE_KEYS / _SESSION_EXCLUDE_PREFIXES が UI 層のどのファイルにも見つかりません"

    violations = []
    for path, tree in trees.items():
        rel = path.relative_to(_ROOT)
        for lineno, widget_name, key_text, is_full in _extract_button_key_literals(tree):
            if is_full:
                # 完全リテラル: EXCLUDE_KEYS か EXCLUDE_PREFIXES のどちらかにマッチ
                covered = (key_text in exclude_keys
                           or any(key_text.startswith(p) for p in exclude_prefixes))
                if not covered:
                    violations.append(
                        f"  {rel}:{lineno}  st.{widget_name}(key={key_text!r}) → "
                        f"`_SESSION_EXCLUDE_KEYS` への追加が必要")
            else:
                # f-string の固定 prefix: いずれかの EXCLUDE_PREFIXES と一致するか、
                # prefix が EXCLUDE_PREFIXES の一つで始まる (より長いプレフィックス) なら OK
                covered = any(p == key_text or key_text.startswith(p)
                              for p in exclude_prefixes)
                if not covered:
                    violations.append(
                        f"  {rel}:{lineno}  st.{widget_name}(key=f'{key_text}...') → "
                        f"`_SESSION_EXCLUDE_PREFIXES` への追加が必要")

    if violations:
        raise AssertionError(
            "セッション保存から除外されていないボタン系 widget があります:\n" + "\n".join(violations))


# --- プリセットのキー一覧 ⇄ 実在する widget key ---

def _key_text(node: ast.AST) -> tuple[str, bool] | None:
    """key 引数の式から (固定部分, 完全リテラルか) を返す。f-string は先頭の固定部分。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    if isinstance(node, ast.JoinedStr):
        head = ""
        for part in node.values:
            if isinstance(part, ast.Constant):
                head += part.value
            else:
                return head, False
        return head, True
    return None


def _collect_widget_key_texts() -> set[tuple[str, bool]]:
    """UI 層で widget の key になる文字列 (固定部分, 完全リテラルか) の集合。

    key= キーワード引数に加え、UI 層で定義した関数の引数名が key / key_prefix の
    位置引数 (fontsize_input(label, key)・strftime_format_ui(label, key_prefix) 等、
    関数の中で key を組み立てる部品) も拾う。state_io.py (一覧そのもの) は除く。
    """
    files = [f for f in _ui_source_files() if f.name != "state_io.py"]
    trees = [ast.parse(f.read_text(encoding="utf-8")) for f in files]
    params: dict[str, list[str]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params[node.name] = [a.arg for a in node.args.args]
    texts: set[tuple[str, bool]] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg in ("key", "key_prefix"):
                    kt = _key_text(kw.value)
                    if kt:
                        texts.add(kt)
            name = (node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute) else None)
            for pname, arg in zip(params.get(name, []), node.args):
                if pname in ("key", "key_prefix"):
                    kt = _key_text(arg)
                    if kt:
                        texts.add(kt)
    return texts


def test_preset_keys_exist_as_widget_keys():
    """プリセットに保存するキー・プレフィックスが、実在する widget key を指していること。

    widget の key を変えたのに state_io の一覧を直し忘れると、その設定は黙って
    プリセットに入らなくなる (実例: 画像出力の背景を 3 択のラジオ out_bg_mode に
    変えたとき一覧が旧 out_transparent のままで、背景がプリセットに入らなかった。
    2026-09-29 修正)。完全一致キーは完全リテラルの key と、プレフィックスは
    key の固定部分の先頭と照合する。
    """
    from climcanvas.ui import state_io
    texts = _collect_widget_key_texts()
    exact = {s for s, full in texts if full}
    missing = sorted(k for k in state_io._PRESET_INCLUDE_KEYS | state_io._PRESET_ONLY_KEYS
                     if k not in exact)
    missing += [p + "*" for p in state_io._PRESET_INCLUDE_PREFIXES
                if not any(s.startswith(p) for s, _ in texts)]
    assert not missing, ("プリセットの一覧に、どの widget key にも当たらない項目があります "
                         f"(state_io.py を widget の今の key に合わせる): {missing}")
