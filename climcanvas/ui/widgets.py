# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""汎用の小さな UI 部品 (ラベル整形・カラーマップ/色セレクタ・フォントサイズ入力など)。"""

import functools as _functools
import re as _re

import pandas as pd
import streamlit as st

from climcanvas.core import render as mc_render
from climcanvas.ui.constants import (CMAP_GROUPS, COLOR_GROUPS,
                                     CUSTOM_COLOR_LABEL, DEFAULT_CMAP_GROUP,
                                     _EMOJI_OVERRIDES, _closest_emoji_square)
from climcanvas.ui.i18n import t


def var_label(ds, v):
    long_name = ds[v].attrs.get("long_name", "")
    return f"{v} — {long_name}" if long_name else v


def coord_label(ds, roles, dim):
    """軸ラベルの既定値: 次元名 [units] (時間軸は "time")。

    long_name は使わない (図中では変数名・次元名で統一。2026-09-17)。
    """
    if dim == roles["time"]:
        return "time"
    units = ds[dim].attrs.get("units", "")
    return f"{dim} [{units}]" if units else str(dim)


def preview_slice(ds, var, keep_dims):
    """描画軸以外の次元を先頭 index で固定した断面を返す (layer_preview_data の予備)。

    選択中の時刻・レベルを反映しないので、既定値・参考表示には
    layer_preview_data() を使う (panel が無いときの後退先としてだけ残す)。
    """
    da = ds[var]
    fixed = {d: 0 for d in da.dims if d not in keep_dims}
    return da.isel(fixed) if fixed else da


def layer_preview_data(panel, layer, datasets, keep_dims, *, variable=None,
                       dataset_id=None, drawn=False):
    """既定値・参考表示用のデータ断面 —「いま選んでいる断面」を render と同じ手順で切り出す。

    panel は app.py がモード毎に layers_ui へ渡す部分的な panel dict
    (selection / region / ranges / x_dim / y_dim)。切り出しは
    render.layer_selected_data (選択中の時刻・レベル・領域・平均を反映)、
    drawn=True なら render.layer_drawn_data (さらに値の変換・maskout を適用。
    layer が完成した後の「データ範囲」「描画レベル」表示用)。
    variable / dataset_id は描画変数以外を見るとき (ベクトルの成分) に指定する。
    panel が None (モード側が渡さない) か切り出しに失敗したとき (設定途中で
    render も落ちる状態) は従来の preview_slice (描画軸以外の次元を先頭 index
    で固定) に戻し、サイドバー構築中の例外で UI 全体を止めない
    (drawn=True ではそのうえで値の変換と閾値 maskout だけ style から掛ける)。
    以前は常に preview_slice だったため、時刻 (lag) やレベルを切り替えても
    「データ範囲」が先頭断面の値のままだった (2026-09-12)。
    """
    if panel is not None:
        try:
            if drawn:
                return mc_render.layer_drawn_data(panel, layer, datasets)
            return mc_render.layer_selected_data(panel, layer, datasets,
                                                 variable=variable,
                                                 dataset_id=dataset_id)
        except Exception:
            pass
    ds = datasets[dataset_id or layer["dataset_id"]]
    da = preview_slice(ds, layer["variable"] if variable is None else variable,
                       keep_dims)
    if drawn:
        style = layer["style"]
        da = mc_render.apply_maskout(mc_render.apply_value_transform(da, style),
                                     style)
    return da


def section_header(label):
    """サイドバーの大セクション見出し (前に薄い区切り線を入れる)。"""
    st.divider()
    st.header(label)


def dim_choices(values) -> list:
    """次元座標を selectbox の選択肢リストにする。

    数値 dtype は float、それ以外 (IBTrACS の basin='NIO' のような文字列座標等)
    は文字列のまま返す。次元座標は数値とは限らない — float() 決め打ちにすると
    実データで落ちる (implementation_checklist.md D節)。
    .sel() は文字列座標も受けるので、値はそのまま selection に入れてよい。
    """
    import numpy as np
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return [float(v) for v in arr]
    return [str(v) for v in arr]


def dim_choice_label(v) -> str:
    """dim_choices の要素の表示ラベル (数値は %g、文字列はそのまま)。"""
    return f"{v:g}" if isinstance(v, float) else str(v)


def union_coord_values(datasets, dim):
    """読み込んだ全 dataset の dim 座標値の和集合 (昇順・重複なし) を返す。

    パネル側の軸範囲 UI (スライダー・最小/最大の入力欄) の候補値に使う。先頭の
    dataset だけから作ると、期間の続く別ファイル (歴史実験 + 将来シナリオ等) を
    同じ軸に重ねたときにスライダーが先頭ファイルの範囲に固定され、入力欄と
    連動できない (2026-09-20)。dim を持たない dataset は無視し、1 つしか無ければ
    その座標をそのまま (並び順を保って) 返す。
    """
    import numpy as np
    arrays = [d[dim].values for d in datasets.values() if dim in d.coords]
    if not arrays:
        return np.asarray([])
    if len(arrays) == 1:
        return arrays[0]
    return np.unique(np.concatenate([np.asarray(a).ravel() for a in arrays]))


def time_labels_and_values(values):
    """時間座標の (表示用ラベルリスト, sel/range用の値リスト) を返す。

    - datetime64: 両方とも ISO 文字列 (xarray が文字列を時刻として解釈する)
    - 整数 (year など): ラベルは str(年)、値は Python int
    - 浮動小数: ラベルは "%g" 整形、値は Python float

    selection / ranges に「ラベル文字列」を入れると整数座標の .sel が
    KeyError を出すため、表示と値を分ける。
    """
    import numpy as np
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        iso = [pd.Timestamp(t).isoformat() for t in values]
        return iso, iso
    labels, vals = [], []
    for v in arr:
        py = v.item()
        labels.append(f"{py:g}" if isinstance(py, float) else str(py))
        vals.append(py)
    return labels, vals


def _nearest_index(values, target):
    """values (数値リスト) の中で target に最も近い要素の index。"""
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def linked_range_ui(label, options, slider_key, lo_key, hi_key, *,
                    lo_label, hi_label, help=None):
    """数値座標の範囲を「select_slider + 最小/最大の数値入力」で選ぶ (両者は連動)。

    - スライダーを動かすと、数値入力にその格子点の値が入る
    - 数値を入力すると、スライダーが最寄りの格子点へ動く。範囲に使うのは入力値
      そのもの (格子点に丸めない)。降順座標でも入力欄は 最小 ≤ 最大 で持ち、
      slice の向きは render.coord_slice が座標に合わせる

    連動は on_change コールバックで相手の key に書く (コールバックは script 本体
    より先に走るので「描画済み widget の key への代入」にはならない。programmatic な
    代入は相手の on_change を起こさないので無限ループにもならない)。
    旧セッション (スライダーの値だけが保存されている) は、数値入力の key が無い
    初回にスライダーの値から種付けして食い違いを防ぐ。
    返り値は (最小, 最大) の float。
    """
    opts = [float(v) for v in options]
    lo_full, hi_full = min(opts), max(opts)

    def _from_slider():
        a, b = (float(v) for v in st.session_state[slider_key])
        st.session_state[lo_key], st.session_state[hi_key] = min(a, b), max(a, b)

    def _from_inputs():
        i = _nearest_index(opts, float(st.session_state[lo_key]))
        j = _nearest_index(opts, float(st.session_state[hi_key]))
        st.session_state[slider_key] = (opts[min(i, j)], opts[max(i, j)])

    if lo_key not in st.session_state and slider_key in st.session_state:
        try:
            _from_slider()
        except (TypeError, ValueError):
            pass
    st.select_slider(label, options=opts, value=(opts[0], opts[-1]),
                     key=slider_key, on_change=_from_slider, help=help)
    # ± ボタンの刻みは格子間隔 (中央値)
    diffs = sorted(abs(opts[i + 1] - opts[i]) for i in range(len(opts) - 1))
    step = diffs[len(diffs) // 2] if diffs else 1.0
    if not step > 0:
        step = 1.0
    c1, c2 = st.columns(2)
    lo = c1.number_input(lo_label, value=lo_full, step=step, format="%g",
                         key=lo_key, on_change=_from_inputs)
    hi = c2.number_input(hi_label, value=hi_full, step=step, format="%g",
                         key=hi_key, on_change=_from_inputs)
    return float(lo), float(hi)


def linked_time_range_ui(label, labels, slider_key, lo_key, hi_key, *,
                         lo_label, hi_label, help=None):
    """時刻座標の範囲を「select_slider + 開始/終了の文字列入力」で選ぶ (両者は連動)。

    labels は time_labels_and_values() の ISO 文字列。入力欄の文字列は
    pd.Timestamp で解釈して最寄りの時刻に丸め、スライダーと入力欄の両方を
    その時刻のラベルに揃える (解釈できない文字列はスライダーの現在値に戻す)。
    連動・種付けの仕組みは linked_range_ui と同じ。返り値は (開始, 終了) の index。
    """
    import numpy as np
    labels = [str(v) for v in labels]
    times = pd.to_datetime(labels)

    def _nearest(text):
        ts = pd.Timestamp(str(text).strip())
        if pd.isna(ts):
            raise ValueError(text)
        return int(np.argmin(np.abs((times - ts).values)))

    def _from_slider():
        a, b = st.session_state[slider_key]
        st.session_state[lo_key], st.session_state[hi_key] = str(a), str(b)

    def _from_inputs():
        try:
            i = _nearest(st.session_state[lo_key])
            j = _nearest(st.session_state[hi_key])
        except (TypeError, ValueError):
            a, b = st.session_state.get(slider_key, (labels[0], labels[-1]))
            st.session_state[lo_key], st.session_state[hi_key] = str(a), str(b)
            return
        i, j = min(i, j), max(i, j)
        st.session_state[slider_key] = (labels[i], labels[j])
        st.session_state[lo_key], st.session_state[hi_key] = labels[i], labels[j]

    if lo_key not in st.session_state and slider_key in st.session_state:
        try:
            _from_slider()
        except (TypeError, ValueError):
            pass
    a, b = st.select_slider(label, options=labels, value=(labels[0], labels[-1]),
                            key=slider_key, on_change=_from_slider, help=help)
    c1, c2 = st.columns(2)
    c1.text_input(lo_label, value=labels[0], key=lo_key, on_change=_from_inputs)
    c2.text_input(hi_label, value=labels[-1], key=hi_key, on_change=_from_inputs)
    return labels.index(a), labels.index(b)


def cmap_selector(lid):
    """カラーマップを「分類 → カラーマップ名」の2段プルダウンで選ぶ。

    グループを切り替えると selectbox の key にグループ名を含めて新しい widget
    として扱い、初期値が新しいリストの先頭になるようにする (古い値が残って
    エラーになるのを避ける)。
    """
    group_keys = list(CMAP_GROUPS)
    group = st.selectbox(
        t("カラーマップ分類"), group_keys,
        index=group_keys.index(DEFAULT_CMAP_GROUP),
        key=f"cmap_group_{lid}")
    options = CMAP_GROUPS[group]
    return st.selectbox(t("カラーマップ"), options, key=f"cmap_{lid}_{group}")


def color_selector(label, default, key, *, meta_store=None, meta_key=None):
    """カラーグループのラジオ + グループ内 selectbox + カスタム時のみ color_picker。

    グループ: basecolor / tab10 / tab20 / other / IPCC (other に d_gray, l_gray, custom;
    IPCC は AR6 WGI Visual Style Guide 2022 p.9 の RCP / SSP 公式色)。
    default の HEX をどのグループが含むかは tab10 → basecolor → other → tab20 → IPCC の
    優先順で探す (tab10/tab20 は重複する HEX があるので、より語彙が明確な tab10 を優先)。

    ラベル追跡 (meta_store / meta_key):
        両方指定すると、選択したエントリのラベル (例: "SSP5-8.5") を
        `meta_store["_color_labels"][meta_key]` に保存する。IPCC の RCP 8.5 と
        SSP5-8.5 のように同一 HEX で意味論が異なるケースを figure_config 上でも
        厳密に区別できる。UI 復元時にはこのラベルを優先して初期選択を復元する。
        IPCC 以外のカテゴリでは `_color_labels` からエントリを消す (夾雑を残さない)。
        render / scriptgen は `_color_labels` を参照しないので描画に影響しない。
    """
    d = default.lower()
    saved_label = None
    if meta_store is not None and meta_key is not None:
        _labels = meta_store.get("_color_labels")
        if isinstance(_labels, dict):
            saved_label = _labels.get(meta_key)

    cat_priority = ["tab10", "basecolor", "other", "tab20", "IPCC"]
    found_cat = "other"
    # IPCC ラベルが figure_config 側に保存されているならそれを優先
    if saved_label and saved_label in COLOR_GROUPS.get("IPCC", {}):
        found_cat = "IPCC"
    else:
        for cat in cat_priority:
            if any(h.lower() == d for h in COLOR_GROUPS[cat].values()):
                found_cat = cat
                break
    cats = list(COLOR_GROUPS.keys())  # basecolor, tab10, tab20, other, IPCC
    category = st.radio(label, cats, horizontal=True,
                         index=cats.index(found_cat), key=f"{key}_cat")
    # オプションは色の名前のみ (HEX はスウォッチの右側に出す)
    if category == "other":
        options = list(COLOR_GROUPS["other"].keys()) + [CUSTOM_COLOR_LABEL]
    else:
        options = list(COLOR_GROUPS[category].keys())
    # default のグループと一致するときだけ初期選択を default に合わせる
    init_idx = 0
    if category == found_cat:
        if category == "IPCC" and saved_label and saved_label in options:
            init_idx = options.index(saved_label)
        else:
            for i, h in enumerate(COLOR_GROUPS[category].values()):
                if h.lower() == d:
                    init_idx = i
                    break

    def _fmt(opt):
        if opt == CUSTOM_COLOR_LABEL:
            return opt
        emoji = _EMOJI_OVERRIDES.get(
            (category, opt),
            _closest_emoji_square(COLOR_GROUPS[category][opt]))
        return f"{emoji} {opt}"

    choice = st.selectbox(label, options, index=init_idx,
                           format_func=_fmt,
                           key=f"{key}_sel_{category}",
                           label_visibility="collapsed")
    if category == "other" and choice == CUSTOM_COLOR_LABEL:
        picked = st.color_picker(label, default, key=f"{key}_picker",
                                  label_visibility="collapsed")
        _render_color_swatch(picked)
        hex_value = picked
    else:
        hex_value = COLOR_GROUPS[category][choice]
        _render_color_swatch(hex_value)

    _write_color_label(meta_store, meta_key, category, choice)
    return hex_value


def _write_color_label(meta_store, meta_key, category, choice):
    """IPCC を選んだときはラベルを保存、それ以外はエントリを削除。

    `_color_labels` が空になったら丸ごと消す (figure_config を汚さない)。
    """
    if meta_store is None or meta_key is None:
        return
    labels = meta_store.get("_color_labels")
    if not isinstance(labels, dict):
        labels = {}
    if category == "IPCC":
        labels[meta_key] = choice
    else:
        labels.pop(meta_key, None)
    if labels:
        meta_store["_color_labels"] = labels
    else:
        meta_store.pop("_color_labels", None)


def _render_color_swatch(hex_color: str) -> None:
    """selectbox の直下に、色のスウォッチ + その右に HEX 値を出す。"""
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;"
        "margin:-2px 0 4px 0;'>"
        f"<div style='width:40px;height:14px;background-color:{hex_color};"
        "border:1px solid #888;'></div>"
        f"<span style='font-family:monospace;font-size:0.85em;color:#666;'>"
        f"{hex_color}</span></div>",
        unsafe_allow_html=True,
    )


def value_transform_state(key_prefix):
    """「値の変換」widget の現在値 (scale, offset) を session_state から読む。

    value_transform_ui() は範囲 UI (fill_levels_ui / cmap_section_ui 等) より
    **後**に描画されるので、範囲 UI の時点では style にまだ入っていない。
    widget 値はスクリプト開始前に session_state に載るため、ここから読めば
    同じ run の入力が取れる。未描画 (初回・vector の加算なし) は 1.0 / 0.0。
    """
    scale = st.session_state.get(f"vscale_{key_prefix}")
    offset = st.session_state.get(f"voff_{key_prefix}")
    return (1.0 if scale is None else float(scale),
            0.0 if offset is None else float(offset))


def transformed_range(key_prefix, dmin, dmax, magnitude=False):
    """生データの (min, max) に「値の変換」を掛けた既定範囲を返す。

    vmin/vmax・レベル・ハッチ閾値は**変換後**の単位で指定する仕様
    (render.apply_value_transform) なので、UI の既定値と「データ範囲」表示も
    変換後の値で出す。倍率が負なら大小が入れ替わるので昇順に直す。
    magnitude=True (vector / stream の |V|) は |倍率| だけ掛ける
    (両成分に同じ倍率 → |V| は |a| 倍。加算は無い)。
    """
    scale, offset = value_transform_state(key_prefix)
    if magnitude:
        scale, offset = abs(scale), 0.0
    lo, hi = scale * dmin + offset, scale * dmax + offset
    return (lo, hi) if lo <= hi else (hi, lo)


def value_transform_ui(style, key_prefix, with_offset=True):
    """値の線形変換 y = a*x + b を expander 内で設定する。

    fill/contour/hatch は scale + offset。vector は scale のみ
    (加算は方向が変わるため非対応)。範囲 UI 側の既定値・データ範囲表示は
    transformed_range() で同じ key_prefix の widget 値を読んで変換後にする。
    """
    with st.expander(t("値の変換 (単位換算など)"), expanded=False):
        style["value_scale"] = float(st.number_input(
            t("倍率 a"), value=1.0, format="%g", key=f"vscale_{key_prefix}",
            help=t("例: Pa→hPa は 0.01")))
        if with_offset:
            style["value_offset"] = float(st.number_input(
                t("加算 b"), value=0.0, format="%g", key=f"voff_{key_prefix}",
                help=t("例: K→°C は -273.15")))
            st.caption(t("変換後 = a × 元の値 + b。"
                       "vmin/vmax・レベル・ハッチの閾値などは変換後の単位で入力してください。"))
        else:
            st.caption(t("両成分に同じ倍率を掛けます (加算は方向が変わるため非対応)。"
                       "ベクトルキーのラベルも変換後の単位で。"))


def maskout_ui(style, key_prefix, ds=None, variable=None, required_dims=()):
    """maskout (GrADS 相当): 閾値以下・以上の値を描画しない設定を expander 内で行う。

    fill / hatch / contour (水平面図・断面図共通) 用。閾値は「値の変換」
    適用後の単位で指定する。

    ds と variable (描画変数) を渡すと「別の変数の値でマスク」も出す。
    候補は同じデータセットの変数のうち、描画軸 (required_dims) を持ち、
    かつ描画変数に無い次元を持たないもの (同じ selection で固定できる範囲)。
    こちらの閾値はマスク変数の生の値で指定する (「値の変換」は掛からない)。
    """
    with st.expander(t("マスクアウト (値で非表示)"), expanded=False):
        m = {"below": None, "above": None}
        c1, c2 = st.columns(2)
        if c1.checkbox(t("この値以下を描かない"), value=False,
                       key=f"maskb_on_{key_prefix}"):
            m["below"] = float(c1.number_input(
                t("閾値 (以下)"), value=0.0, format="%g",
                key=f"maskb_{key_prefix}"))
        if c2.checkbox(t("この値以上を描かない"), value=False,
                       key=f"maska_on_{key_prefix}"):
            m["above"] = float(c2.number_input(
                t("閾値 (以上)"), value=0.0, format="%g",
                key=f"maska_{key_prefix}"))
        st.caption(t("閾値は「値の変換」適用後の単位で入力してください。"
                   "マスクされた範囲は塗り・ハッチ・等値線が描かれません。"))
        if ds is not None and variable is not None:
            m.update({"variable": None, "var_below": None, "var_above": None})
            var_dims = set(ds[variable].dims)
            cand = [v for v in ds.data_vars
                    if v != variable
                    and set(required_dims) <= set(ds[v].dims) <= var_dims]
            if cand and st.checkbox(t("別の変数の値でマスク"), value=False,
                                    key=f"maskv_on_{key_prefix}"):
                vkey = f"maskv_var_{key_prefix}"
                if vkey in st.session_state and st.session_state[vkey] not in cand:
                    st.session_state[vkey] = cand[0]
                m["variable"] = st.selectbox(t("マスク変数"), cand, key=vkey)
                c3, c4 = st.columns(2)
                if c3.checkbox(t("この値以下を描かない"), value=False,
                               key=f"maskvb_on_{key_prefix}"):
                    m["var_below"] = float(c3.number_input(
                        t("閾値 (以下)"), value=0.0, format="%g",
                        key=f"maskvb_{key_prefix}"))
                if c4.checkbox(t("この値以上を描かない"), value=False,
                               key=f"maskva_on_{key_prefix}"):
                    m["var_above"] = float(c4.number_input(
                        t("閾値 (以上)"), value=0.0, format="%g",
                        key=f"maskva_{key_prefix}"))
                st.caption(t("マスク変数の閾値はその変数の生の値で指定します"
                           "(描画変数の「値の変換」は適用されません)。"
                           "マスク変数には描画変数と同じ次元固定が適用されます。"))
        style["maskout"] = m


def fontsize_input(label, key, default=0, show_auto_hint=True):
    """0を「自動」とするフォントサイズ入力。default は初期表示値 (0=自動)。"""
    suffix = t(" (0=自動)") if show_auto_hint else ""
    val = st.number_input(f"{label}{suffix}", min_value=0, max_value=40, value=default, key=key)
    return int(val) or None


def parse_float_list(text: str) -> list[float] | None:
    """カンマ/空白区切りの数値文字列を float リストに変換。失敗・空なら None。

    カラーレベルの直接指定 (fill / vector / bubble) で使う。境界値は
    contourf / BoundaryNorm が単調増加を要求するため、昇順に整列し
    重複を除去して返す (UI の help 文の記述と対応)。
    """
    if not text or not text.strip():
        return None
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    try:
        return sorted({float(p) for p in parts}) or None
    except ValueError:
        return None


def parse_ratio_list(text: str) -> list[float] | None:
    """カンマ/空白区切りの正数リストに変換 (順序保持)。空・不正・非正値は None。

    グリッドの行・列の比率 (layout.width_ratios / height_ratios) で使う。
    parse_float_list はカラーレベル用に昇順整列・重複除去するため
    比率には使えない (「2, 1, 2」のような並びを保つ必要がある)。
    """
    import math
    if not text or not text.strip():
        return None
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if any(not math.isfinite(v) or v <= 0 for v in vals):
        return None
    return vals


# --- フォント選択 (図全体の書式) ---

# 「よく使うフォント」の候補。インストールされているものだけを UI に出す
# (外部カラーマップと同じ「あるものだけ見せる」方式)
_COMMON_FONT_CANDIDATES = [
    "DejaVu Sans",           # matplotlib 既定 (同梱)
    "DejaVu Serif",          # matplotlib 同梱のセリフ体
    "Arial",
    "Helvetica",
    "Times New Roman",
    "Georgia",
    "Courier New",
    # 日本語 (文字化け対策)
    "Hiragino Sans",
    "Hiragino Mincho ProN",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Yu Gothic",
    "Meiryo",
    "IPAexGothic",
]


@_functools.lru_cache(maxsize=1)
def available_font_families() -> tuple[str, ...]:
    """matplotlib が検出した全フォントファミリー名 (昇順・重複なし)。

    font_manager の走査は初回のみ (プロセス内キャッシュ)。
    """
    from matplotlib import font_manager as _fm
    return tuple(sorted({f.name for f in _fm.fontManager.ttflist}))


@_functools.lru_cache(maxsize=1)
def common_fonts_available() -> tuple[str, ...]:
    """「よく使うフォント」候補のうちインストール済みのものを返す。"""
    avail = set(available_font_families())
    return tuple(f for f in _COMMON_FONT_CANDIDATES if f in avail)


# --- 図中 CJK 文字とフォントの整合チェック (i18n 第6段階) ---
# タイトル・軸ラベル等のユーザー入力に CJK 文字があるのに対応フォントが
# 無いと matplotlib は □ (豆腐) を描く。config を走査して事前に警告する

_CJK_KANA = _re.compile("[\u3040-\u30ff]")
_CJK_HANGUL = _re.compile("[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_CJK_HAN = _re.compile("[\u4e00-\u9fff\u3400-\u4dbf]")

# スクリプト別の代表フォント候補 (インストール済みのものだけ提示する)
_CJK_FONT_CANDIDATES = {
    "ja": ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic",
           "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "MS Gothic"),
    "zh": ("PingFang SC", "Noto Sans CJK SC", "Noto Sans SC",
           "Microsoft YaHei", "SimHei", "Songti SC"),
    "ko": ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR",
           "Malgun Gothic", "AppleGothic", "NanumGothic"),
}


def _collect_cjk_chars(node, out):
    """figure_config を再帰走査し CJK 文字を集める。"_" 始まりのキー
    (_color_labels 等の図に描かれないメタ) は飛ばす。"""
    if isinstance(node, str):
        for ch in node:
            if (_CJK_KANA.match(ch) or _CJK_HANGUL.match(ch)
                    or _CJK_HAN.match(ch)):
                out.setdefault(ch, None)
    elif isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            _collect_cjk_chars(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _collect_cjk_chars(v, out)


@_functools.lru_cache(maxsize=8)
def _load_ft2font(path):
    from matplotlib.ft2font import FT2Font
    return FT2Font(path)


def cjk_font_advice(figure_config) -> str | None:
    """図中の CJK 文字とフォント設定の不整合を検知し、警告文を返す。

    - フォント未指定 + CJK 文字あり → 環境にある対応フォント候補を挙げて案内
    - フォント指定済み → そのフォントのグリフ被覆を検査し、欠落があれば警告
    問題なし (または判定不能) なら None。UI 言語で翻訳された文字列を返す。
    """
    chars: dict = {}
    _collect_cjk_chars(figure_config, chars)
    if not chars:
        return None
    sample = "".join(list(chars)[:40])
    font = (figure_config.get("figure") or {}).get("font_family")
    if not font:
        scripts = set()
        if _CJK_KANA.search(sample):
            scripts.add("ja")
        if _CJK_HANGUL.search(sample):
            scripts.add("ko")
        if _CJK_HAN.search(sample):
            from climcanvas.ui.i18n import current_lang
            scripts.add("zh" if current_lang() == "zh_CN" else "ja")
        installed = set(available_font_families())
        cands = [f for s in ("ja", "zh", "ko") if s in scripts
                 for f in _CJK_FONT_CANDIDATES[s] if f in installed]
        fonts = ", ".join(list(dict.fromkeys(cands))[:6]) or             t("候補が見つかりません — Noto Sans CJK 等のインストールを検討してください")
        return t("図中に日本語・中国語・韓国語の文字がありますが、フォントが"
                 "未指定です。matplotlib 既定 (DejaVu Sans) では □ (豆腐) に"
                 "なります。「図全体の書式 > フォント」で対応フォントを指定して"
                 "ください。この環境で使える候補: {fonts}", fonts=fonts)
    try:
        from matplotlib import font_manager
        path = font_manager.findfont(
            font_manager.FontProperties(family=font),
            fallback_to_default=False)
        ft = _load_ft2font(path)
        missing = [c for c in sample if ft.get_char_index(ord(c)) == 0]
    except Exception:
        return None  # フォント解決不能などは matplotlib の挙動に任せる
    if missing:
        return t("選択中のフォント ({font}) には図中の一部の文字 ({chars}) の"
                 "グリフがありません。□ (豆腐) になる可能性があります。"
                 "対応フォントへの変更を検討してください。",
                 font=font, chars="".join(missing[:8]))
    return None
