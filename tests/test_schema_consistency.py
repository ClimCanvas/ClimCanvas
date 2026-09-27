# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""コア層が読むキーと config スキーマ・UI が書くキーの整合を AST で検証する。

figure_config は素の dict なので、キー名のタイポは実行時に黙って既定値に
フォールバックしてしまう (バグとして最頻の類型)。ここでは

- render.py / scriptgen.py が `axis_cfg.get("k")` / `axis_cfg["k"]` 等で読む
  リテラルキーが、config の default_*() スキーマに必ず存在すること
- UI (panel_ui.py / app.py) が `axis["k"] = ...` 等で書くキーも同様

を機械チェックする。対象は変数名で対応スキーマが特定できるグループのみ
(axis / grid / frame / background / span / legend)。layer の style は
kind ごとにスキーマが異なり変数名から特定できないため対象外。
"""

from __future__ import annotations

import ast
from pathlib import Path

from climcanvas.core import config as C

ROOT = Path(__file__).resolve().parent.parent

# 変数名 -> 許容キー集合。同名変数が複数モードで使われるため union を取る
def _schema_map() -> dict[str, set[str]]:
    axis_keys = (set(C.default_section_panel()["axis"])
                 | set(C.default_line_panel()["axis"])
                 | set(C.default_scatter_panel()["axis"])
                 | set(C.default_dist_panel()["axis"]))
    grid_keys = (set(C.default_section_panel()["axis"]["grid"])
                 | set(C.default_line_panel()["axis"]["grid"])
                 | {"show"})  # 旧スキーマ (render/scriptgen が互換処理で読む)
    legend_keys = set(C.default_line_panel()["legend"])
    return {
        "axis_cfg": axis_keys,
        "axis": axis_keys,
        "grid": grid_keys,
        "frame": set(C.default_frame()),
        "background": set(C.default_axes_background()),
        "span": set(C.default_background_span()),
        "refline": set(C.default_refline()),
        "legend": legend_keys,
        "legend_cfg": legend_keys,
    }


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _scan_accesses(path: Path):
    """(変数名, キー, 行番号, 種別) を列挙する。種別は read / write。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets = set(_schema_map())
    for node in ast.walk(tree):
        # var.get("key"[, default])
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in targets and node.args):
            key = _literal(node.args[0])
            if key:
                yield node.func.value.id, key, node.lineno, "read"
        # var["key"] (read / write)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id in targets):
            key = _literal(node.slice)
            if key:
                kind = "write" if isinstance(node.ctx, ast.Store) else "read"
                yield node.value.id, key, node.lineno, kind


def _collect_violations(paths):
    schema = _schema_map()
    violations = []
    for path in paths:
        for var, key, lineno, kind in _scan_accesses(path):
            if key not in schema[var]:
                violations.append(
                    f"{path.name}:{lineno} {var}[{key!r}] ({kind}) "
                    "がスキーマに存在しません")
    return violations


def test_core_reads_match_schema():
    """render / scriptgen が読むキーは config スキーマに必ず存在する。"""
    paths = [ROOT / "climcanvas/core/render.py",
             ROOT / "climcanvas/core/scriptgen.py"]
    violations = _collect_violations(paths)
    assert not violations, "\n" + "\n".join(violations)


def test_ui_writes_match_schema():
    """UI が書き込むキーは config スキーマに必ず存在する。"""
    paths = [ROOT / "climcanvas/ui/panel_ui.py", ROOT / "app.py"]
    violations = _collect_violations(paths)
    assert not violations, "\n" + "\n".join(violations)
