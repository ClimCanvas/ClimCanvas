# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""図の直下に出す「この図に適用した処理」の整形 (core.notes の項目 → Markdown)。

core.notes.collect_notes が返す中立キーの項目を、パネルごと・レイヤーごとに
まとめた Markdown 1 本にする。表示用の文字列はここで t() で包む (core 層は
i18n を持たない。docs/i18n_guide.md 4 節 7)。負号は全角マイナス (−) で出す
(2026-09-27 ユーザー判断)。app.py は該当項目が 1 つでもあるときだけ枠
(st.container(border=True)) を作ってこの Markdown を置く。
"""

from __future__ import annotations

from climcanvas.ui.constants import KIND_LABELS
from climcanvas.ui.i18n import t

# レイヤーを持たない plot_type の見出し (KIND_LABELS に無いもの)
_NOTE_HEAD_LABELS = {"heatmap": "ヒートマップ"}
# 平均した次元の座標役割 → 見出し。役割の無い dim は "other"
_AVG_ROLE_LABELS = {"lat": "緯度平均", "lon": "経度平均", "time": "時間平均",
                    "vertical": "鉛直平均", "other": "範囲平均"}
_AVG_OP_LABELS = {"mean": "算術平均", "weighted_mean": "cos(lat) 重み付き平均"}


def fmt_num(v) -> str:
    """数値を %g 書式にし、負号を全角マイナス (−) にして返す。数値以外は str のまま。"""
    try:
        s = f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)
    return s.replace("-", "−")


def _head(note: dict) -> str:
    """項目の見出し: レイヤー種別 (+ 散布系の軸) と変数名。"""
    kind = note.get("kind")
    if kind in KIND_LABELS:
        label = t(KIND_LABELS[kind])
    else:
        label = t(_NOTE_HEAD_LABELS.get(kind, kind))
    axis = f" {note['axis']}" if note.get("axis") else ""
    units = note.get("units") or {}
    names = ", ".join(f"`{v}`" + (f" [{units[v]}]" if units.get(v) else "")
                      for v in note.get("variables") or [])
    return f"{label}{axis} {names}".rstrip()


def _body(note: dict) -> str:
    kind = note["type"]
    if kind == "value_transform":
        if note.get("offset") is None:
            return t("倍率 a = {a} (加算なし)", a=fmt_num(note["scale"]))
        return t("値の変換 a = {a}, b = {b}",
                 a=fmt_num(note["scale"]), b=fmt_num(note["offset"]))
    if kind == "maskout":
        v = fmt_num(note["value"])
        cond = note.get("cond")
        if cond == "below":
            return t("マスクアウト: {v} 以下を描かない", v=v)
        if cond == "above":
            return t("マスクアウト: {v} 以上を描かない", v=v)
        if cond == "var_below":
            return t("マスクアウト: `{var}` が {v} 以下の位置を描かない",
                     var=note.get("mask_variable"), v=v)
        return t("マスクアウト: `{var}` が {v} 以上の位置を描かない",
                 var=note.get("mask_variable"), v=v)
    if kind == "vector_mask":
        return t("|V| が {v} 以下のベクトルを描かない", v=fmt_num(note["value"]))
    if kind == "vector_no_rotation":
        return t("2 次元座標格子: 成分を東西・南北とみなして描く (格子相対風の回転なし)")
    if kind == "average_missing":
        return t("平均範囲内の欠損 {n} / {total} 要素 (平均から除外)",
                 n=note["n_nan"], total=note["n_total"])
    role = note.get("role") if note.get("role") in _AVG_ROLE_LABELS else "other"
    lo, hi = note["range"]
    op = note.get("op", "mean")
    return t("{what} {dim} {lo}〜{hi} ({op}、{n} 格子点)",
             what=t(_AVG_ROLE_LABELS[role]), dim=note["dim"],
             lo=fmt_num(lo), hi=fmt_num(hi),
             op=t(_AVG_OP_LABELS.get(op, op)), n=note["n_points"])


def format_notes(notes: list[dict]) -> str:
    """項目をパネル → レイヤー (変数・軸) の順にまとめた Markdown を返す。

    1 レイヤーに項目が 1 つなら 1 行、複数なら見出しの下に入れ子の箇条書き。
    項目の順序 (= collect_notes の走査順 = 図の配置順・レイヤー順) は保つ。
    """
    lines = [f"**{t('この図に適用した処理')}**"]
    by_panel: dict[int, list[dict]] = {}
    for note in notes:
        by_panel.setdefault(note["panel_index"], []).append(note)
    for pidx, items in by_panel.items():
        head = t("パネル {n}", n=pidx + 1)
        if items[0].get("panel_label"):
            head += f" {items[0]['panel_label']}"
        lines.append("")
        lines.append(f"**{head}**")
        groups: dict[tuple, list[dict]] = {}
        for note in items:
            key = (note.get("layer_index"), tuple(note.get("variables") or ()),
                   note.get("axis"))
            groups.setdefault(key, []).append(note)
        for group in groups.values():
            if len(group) == 1:
                lines.append(f"- {_head(group[0])}: {_body(group[0])}")
            else:
                lines.append(f"- {_head(group[0])}")
                lines.extend(f"    - {_body(n)}" for n in group)
    return "\n".join(lines)
