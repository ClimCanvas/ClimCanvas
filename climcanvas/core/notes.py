# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""図に適用した処理 (値の変換・範囲平均) の通知を figure_config から集める。

docs/scientific_safeguard_plan.md A-1 (「実際に描いた値」の明細) の第 1 段階。
値の変換 (a, b) と範囲平均 (次元・範囲・方法・含まれた格子数) は設定に宣言
されている情報なので、描画結果ではなく figure_config と座標値から集める
(描画には触らない)。どの dim を実際に平均するかの除外規則と範囲の切り出しは
render.effective_averages / render.averaging_slice を呼び、解決ロジックを
二重に持たない。返す dict は中立キーだけで UI 文字列を持たない (表示点で
ui/notes_ui.py が訳す)。

項目 (dict) の形:

- ``{"type": "value_transform", "panel_index", "panel_id", "panel_label",
  "layer_index", "kind", "variables": [...], "axis": None | "x" | "y" | "z",
  "scale": a, "offset": b | None}``
  offset None = 加算の無い種別 (vector / stream は倍率だけ)。axis は散布系
  (scatter / hist2d / hexbin / bubble) の軸ごとの変換。
- ``{"type": "average", "panel_index", "panel_id", "panel_label", "layer_index",
  "kind", "variables": [...], "dim", "role": "lat"|"lon"|"time"|"vertical"|None,
  "op": "mean"|"weighted_mean", "range": [lo, hi], "n_points": int}``
  n_points は render と同じ切り出し (averaging_slice) に含まれた格子点の数。
- ``{"type": "average_missing", ..., "variables", "dims": [...], "n_nan", "n_total"}``
  平均に使ったブロック (全対象 dim を切り出した平均前の配列) の要素数と欠損数
  (render.layer_averaging_stats = 描画と同じ経路の記録)。欠損が 1 つ以上あるときだけ。
- ``{"type": "maskout", ..., "cond": "below"|"above"|"var_below"|"var_above",
  "value": float, "mask_variable": str | None}``
  値で描かない設定 (fill / hatch / contour / heatmap の maskout、track の maskout)。
  below / above は描画変数の値 (値の変換後)、var_* は別変数 mask_variable の生の値。
- ``{"type": "vector_mask", ..., "value": float}`` ベクトルの |V| がこの値以下の
  矢印を描かない (style.mask_below)。
- ``{"type": "vector_no_rotation", ...}`` 2 次元座標格子 (curvilinear) の
  ベクトル・流線は成分を東西・南北とみなして描く (格子相対風を回転しない)。

全項目に ``"dataset_id"`` と ``"units": {変数名: units 属性 | None}`` を付ける
(表示は変数名の横に単位を添える。二重の単位換算に気づくため)。

panel_index は figure_config["panels"] 内の位置 (= 図の配置順)。layer_index は
panel["layers"] 内の位置 (heatmap のようにレイヤーを持たない plot_type は None)。
同じレイヤーの複数変数 (stackplot、ベクトルの u/v、fill_between の 2 本) は
結果が同じなら 1 項目にまとめる (variables に列挙)。
"""

from __future__ import annotations

from .dataset import detect_coord_roles, is_curvilinear
from .render import (averaging_slice, effective_averages, layer_averaging_stats,
                     maskout_var_config, vector_v_dataset_id)

# 加算 (value_offset) を持たない種別 (両成分に同じ倍率だけ掛ける)
_SCALE_ONLY_KINDS = ("vector", "stream")
# 軸ごとの変換 ({axis}_value_scale / {axis}_value_offset) を持つ種別と、その軸
_AXIS_KINDS = {"scatter": ("x", "y"), "hist2d": ("x", "y"), "hexbin": ("x", "y"),
               "bubble": ("x", "y", "z")}
# render が layer.averages を適用する plot_type (水平面図・集計・散布は適用しない)
_AVERAGING_PLOT_TYPES = ("section_2d", "line_1d")
_ROLE_ORDER = ("lat", "lon", "time", "vertical")


def _is_identity(scale, offset) -> bool:
    """render.apply_value_transform が何もしない条件と同じ。"""
    return float(scale) == 1.0 and float(offset) == 0.0


def _panel_base(index: int, panel: dict) -> dict:
    label = panel.get("label")
    text = (label.get("text") if isinstance(label, dict) and label.get("show")
            else None)
    return {"panel_index": index, "panel_id": panel.get("panel_id"),
            "panel_label": text or None}


def _layer_variables(layer: dict) -> list[str]:
    """値の変換が掛かる変数名 (種別ごとの在処)。未設定 (None) は除く。"""
    kind = layer.get("kind")
    if kind in _SCALE_ONLY_KINDS:
        names = [layer.get("u_variable"), layer.get("v_variable")]
    elif kind == "stackplot":
        names = list(layer.get("variables") or [])
    elif kind == "fill_between":
        names = [layer.get("variable"), layer.get("variable_upper")]
    elif kind == "track":
        pts = (layer.get("style") or {}).get("points") or {}
        names = [pts.get("variable")] if pts.get("show", True) else []
    else:
        names = [layer.get("variable")]
    return [n for n in names if n]


def _transform_entry(base: dict, variables: list[str], style: dict,
                     axis: str | None, scale_only: bool) -> dict | None:
    prefix = f"{axis}_" if axis else ""
    scale = float(style.get(f"{prefix}value_scale", 1.0))
    offset = float(style.get(f"{prefix}value_offset", 0.0))
    if not variables or _is_identity(scale, offset):
        return None
    return {**base, "type": "value_transform", "variables": list(variables),
            "axis": axis, "scale": scale,
            "offset": None if (scale_only and offset == 0.0) else offset}


def _transform_entries(panel: dict, layer: dict, base: dict) -> list[dict]:
    kind = layer.get("kind")
    style = layer.get("style") or {}
    if kind in _AXIS_KINDS:
        out = []
        for axis in _AXIS_KINDS[kind]:
            var = panel.get(f"{axis}_variable")
            e = _transform_entry(base, [var] if var else [], style, axis, False)
            if e:
                out.append(e)
        return out
    if kind == "track":
        style = style.get("points") or {}
    e = _transform_entry(base, _layer_variables(layer), style, None,
                         kind in _SCALE_ONLY_KINDS)
    return [e] if e else []


def _heatmap_entries(panel: dict, base: dict) -> list[dict]:
    """heatmap はレイヤーを持たず panel 直下の variable / style を使う。"""
    var = panel.get("variable")
    hbase = {**base, "layer_index": None, "kind": "heatmap",
             "dataset_id": panel.get("dataset_id")}
    style = panel.get("style") or {}
    out = []
    e = _transform_entry(hbase, [var] if var else [], style, None, False)
    if e:
        out.append(e)
    out.extend(_maskout_entries(style, hbase, [var] if var else []))
    return out


def _maskout_entries(style: dict, base: dict, variables: list[str]) -> list[dict]:
    """maskout (値で描かない) の条件を 1 条件 1 項目で返す。

    below / above は描画変数の値 (render.apply_maskout と同じ「以下 / 以上」)、
    var_below / var_above は別変数 (maskout_var_config が有効と判定したときだけ)。
    """
    m = style.get("maskout") or {}
    out = []
    for cond in ("below", "above"):
        if m.get(cond) is not None:
            out.append({**base, "type": "maskout", "variables": list(variables),
                        "cond": cond, "value": float(m[cond]), "mask_variable": None})
    cfg = maskout_var_config(style)
    if cfg:
        var, vb, va = cfg
        for cond, val in (("var_below", vb), ("var_above", va)):
            if val is not None:
                out.append({**base, "type": "maskout", "variables": list(variables),
                            "cond": cond, "value": float(val), "mask_variable": var})
    return out


def _track_maskout_entries(layer: dict, base: dict) -> list[dict]:
    """track の maskout {variable, below, above} は別変数の生の値による条件。"""
    m = (layer.get("style") or {}).get("maskout") or {}
    var = m.get("variable")
    if not var:
        return []
    out = []
    for cond, key in (("var_below", "below"), ("var_above", "above")):
        if m.get(key) is not None:
            out.append({**base, "type": "maskout", "variables": _layer_variables(layer),
                        "cond": cond, "value": float(m[key]), "mask_variable": var})
    return out


def _vector_entries(layer: dict, base: dict, datasets: dict) -> list[dict]:
    """ベクトル・流線に固有の通知: 弱風の省略 (mask_below) と 2 次元座標格子の非回転。"""
    kind = layer.get("kind")
    if kind not in _SCALE_ONLY_KINDS:
        return []
    out = []
    style = layer.get("style") or {}
    variables = _layer_variables(layer)
    if style.get("mask_below") is not None:
        out.append({**base, "type": "vector_mask", "variables": variables,
                    "value": float(style["mask_below"])})
    ds = datasets.get(layer.get("dataset_id"))
    if ds is not None and is_curvilinear(ds):
        out.append({**base, "type": "vector_no_rotation", "variables": variables})
    return out


def _average_specs(layer: dict) -> list[tuple[str | None, str, dict | None]]:
    """(変数名, dataset_id, averages) の組。fill_between の上側は averages_upper。"""
    kind = layer.get("kind")
    dsid = layer.get("dataset_id")
    avg = layer.get("averages")
    if kind == "fill_between":
        specs = [(layer.get("variable"), dsid, avg)]
        if layer.get("variable_upper"):
            specs.append((layer["variable_upper"], dsid, layer.get("averages_upper")))
        return specs
    if kind == "stackplot":
        return [(v, dsid, avg) for v in (layer.get("variables") or [])]
    if kind in _SCALE_ONLY_KINDS:
        return [(layer.get("u_variable"), dsid, avg),
                (layer.get("v_variable"), vector_v_dataset_id(layer), avg)]
    return [(layer.get("variable"), dsid, avg)]


def _average_items(ds, var: str, averages: dict, exclude) -> list[dict]:
    """1 変数に実際に適用される平均を dim ごとに解決する (render と同じ規則)。"""
    roles = detect_coord_roles(ds)
    items = []
    for dim, cfg in effective_averages(ds[var].dims, averages, exclude=exclude).items():
        lo, hi = cfg["range"]
        if dim not in ds.coords:
            # 座標の無い dim は render も切り出せない (UI は提供しない)
            continue
        n = int(averaging_slice(ds[dim], dim, lo, hi, ds).sizes[dim])
        role = next((r for r in _ROLE_ORDER if roles.get(r) == dim), None)
        items.append({"dim": dim, "role": role, "op": cfg.get("op", "mean"),
                      "range": [lo, hi], "n_points": n})
    return items


def _average_entries(panel: dict, layer: dict, base: dict, datasets: dict) -> list[dict]:
    if panel.get("plot_type") not in _AVERAGING_PLOT_TYPES:
        return []
    exclude: tuple = ()
    if panel.get("plot_type") == "line_1d":
        # 1 次元プロットは x 軸を平均しない。ライン (束) は束の次元も平均しない
        exclude = (panel.get("x_dim"),)
        if layer.get("kind") == "line_bundle":
            exclude += (layer.get("bundle_dim"),)
    per_var = []
    for var, dsid, averages in _average_specs(layer):
        if not var or not averages or dsid not in datasets or var not in datasets[dsid]:
            continue
        items = _average_items(datasets[dsid], var, averages, exclude)
        if items:
            per_var.append((var, items))
    if not per_var:
        return []
    # 平均ブロックの要素数・欠損数は描画と同じ経路 (render.layer_averaging_stats)
    # の記録から取る (変数名 → 記録)。実際に平均する変数があるときだけ走らせる
    stats = {st["variable"]: st for st in layer_averaging_stats(panel, layer, datasets)}
    groups: list[tuple[str, list[str], list[dict], dict | None]] = []
    for var, items in per_var:
        st = stats.get(var)
        missing = (None if not st or st["n_nan"] <= 0
                   else {"dims": list(st["dims"]), "n_nan": int(st["n_nan"]),
                         "n_total": int(st["n_total"])})
        key = repr((items, missing))
        for g in groups:
            if g[0] == key:
                g[1].append(var)
                break
        else:
            groups.append((key, [var], items, missing))
    out = []
    for _, variables, items, missing in groups:
        out.extend({**base, "type": "average", "variables": list(variables), **item}
                   for item in items)
        if missing:
            out.append({**base, "type": "average_missing", "variables": list(variables),
                        **missing})
    return out


def _attach_units(notes: list[dict], datasets: dict) -> None:
    """各項目の variables に units 属性を添える ({変数名: units | None})。"""
    for note in notes:
        ds = datasets.get(note.get("dataset_id"))
        note["units"] = {
            v: (str(ds[v].attrs.get("units")) if ds is not None and v in ds
                and ds[v].attrs.get("units") not in (None, "") else None)
            for v in note.get("variables") or []}


def collect_notes(figure_config: dict, datasets: dict) -> list[dict]:
    """figure_config を走査し、値の変換と範囲平均の項目を図の配置順に返す。

    該当が無ければ空リスト (UI は枠ごと出さない)。datasets は dataset_id →
    xr.Dataset (render_figure に渡すものと同じ)。
    """
    notes: list[dict] = []
    for i, panel in enumerate(figure_config.get("panels") or []):
        base = _panel_base(i, panel)
        if panel.get("plot_type") == "heatmap":
            notes.extend(_heatmap_entries(panel, base))
            continue
        for j, layer in enumerate(panel.get("layers") or []):
            lbase = {**base, "layer_index": j, "kind": layer.get("kind"),
                     "dataset_id": layer.get("dataset_id")}
            notes.extend(_transform_entries(panel, layer, lbase))
            if layer.get("kind") == "track":
                notes.extend(_track_maskout_entries(layer, lbase))
            else:
                notes.extend(_maskout_entries(layer.get("style") or {}, lbase,
                                              _layer_variables(layer)))
            notes.extend(_vector_entries(layer, lbase, datasets))
            notes.extend(_average_entries(panel, layer, lbase, datasets))
    _attach_units(notes, datasets)
    return notes
