# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""パネル単位の設定 UI (レイヤー一覧・軸・凡例・アニメーション・文字列・box)。"""

import numpy as np
import streamlit as st

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render
from climcanvas.ui.constants import (GL_LINESTYLE_LABELS, KIND_LABELS,
                                     LINESTYLE_LABELS, PRESSURE_UNITS)
from climcanvas.ui.i18n import current_lang, t, tr_labels
from climcanvas.ui.layer_ui import LAYER_UI, _default_dim_index
from climcanvas.ui.widgets import (color_selector, coord_label,
                                   dim_choice_label, dim_choices,
                                   fontsize_input, section_header,
                                   time_labels_and_values, transformed_range,
                                   value_transform_state)

# 選択肢の表示ラベル (中立キー → 日本語)。保存値は中立キー側 (i18n 第1段階)
_COORD_LABELS = {"axes": "図に対する相対位置 (axes)",
                 "data": "データ座標 (地図は経度・緯度)"}
_TL_LOC_LABELS = {"left": "左", "center": "中央", "right": "右"}
_TICK_MODE_LABELS = {"auto": "自動", "interval": "等間隔 (0 で非表示)",
                     "positions": "位置を直接指定"}
_REFLINE_ORI_LABELS = {"y": "横線 (y 一定)", "x": "縦線 (x 一定)"}
_SPAN_ORI_LABELS = {"x": "x軸の範囲 (縦帯)", "y": "y軸の範囲 (横帯)"}


def panel_selector_cells(nrows: int, ncols: int, n_panels: int,
                         mosaic: str | None = None) -> dict[int, int]:
    """「編集するパネル」ボタングリッドの セル番号 → パネル番号 の対応を返す。

    キーは 0 始まり行優先のセル番号、値は 0 始まりのパネル番号。図のパネル
    配置とボタンの並びを一致させるため、mosaic 指定時は parse_mosaic の
    セル位置に置く (結合セルは左上のセル、空きセルには置かない)。無指定時は
    従来の行優先 (セル i = パネル i)。mosaic が不正な場合も行優先へ
    フォールバックする (呼び出し側の figure 配置の検証と同じ扱い)。
    """
    if mosaic:
        try:
            _, _, specs = mc_render.parse_mosaic(mosaic, n_panels)
        except ValueError:
            pass
        else:
            return {(spec[0] if isinstance(spec, tuple) else spec) - 1: i
                    for i, spec in enumerate(specs)}
    return {i: i for i in range(n_panels)}


def boxes_ui(mode_key):
    """任意の矩形領域を追加・削除する。各 box は緯度経度・色・線太さ・線種を持つ。

    mode_key はパネルスコープ (例 "map0")。widget key をパネル毎に分離する。
    """
    state_key = f"boxes_{mode_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = []
        st.session_state[f"{state_key}_next"] = 0
    items = st.session_state[state_key]
    boxes_cfg = []
    for i, item in enumerate(list(items)):
        bid = item["id"]
        st.markdown(f"**box {i + 1}**")
        # TAB の移動順 (DOM 順 = カラム毎) を 経度min→max→緯度min→max に
        # するため、経度の行と緯度の行で columns を分ける (描画範囲指定と同じ)
        c1, c2 = st.columns(2)
        lon_min = c1.number_input(t("経度 min"), -360.0, 360.0, 160.0, 1.0,
                                   key=f"box_lonmin_{mode_key}_{bid}",
                                   help=t("lon_max < lon_min なら日付変更線をまたぐ矩形"))
        lon_max = c2.number_input(t("経度 max"), -360.0, 360.0, 200.0, 1.0,
                                   key=f"box_lonmax_{mode_key}_{bid}")
        c1, c2 = st.columns(2)
        lat_min = c1.number_input(t("緯度 min"), -90.0, 90.0, -10.0, 1.0,
                                   key=f"box_latmin_{mode_key}_{bid}")
        lat_max = c2.number_input(t("緯度 max"), -90.0, 90.0, 10.0, 1.0,
                                   key=f"box_latmax_{mode_key}_{bid}")
        box_meta = {}
        color = color_selector(t("線の色"), "#ff0000", key=f"box_color_{mode_key}_{bid}",
                                meta_store=box_meta, meta_key="color")
        linewidth = st.slider(t("線の太さ"), 0.2, 4.0, 1.5, 0.1, key=f"box_lw_{mode_key}_{bid}")
        linestyle = st.selectbox(t("線種"), list(LINESTYLE_LABELS),
                                 format_func=tr_labels(LINESTYLE_LABELS).get,
                                 key=f"box_ls_{mode_key}_{bid}")
        box_dict = {"lon_min": float(lon_min), "lon_max": float(lon_max),
                    "lat_min": float(lat_min), "lat_max": float(lat_max),
                    "color": color, "linewidth": float(linewidth),
                    "linestyle": linestyle}
        if "_color_labels" in box_meta:
            box_dict["_color_labels"] = box_meta["_color_labels"]
        boxes_cfg.append(box_dict)
        if st.button(t("この box を削除"), key=f"box_del_{mode_key}_{bid}"):
            items.remove(item)
            st.rerun()
        st.divider()
    if st.button(t("box を追加"), key=f"box_add_{mode_key}"):
        items.append({"id": st.session_state[f"{state_key}_next"]})
        st.session_state[f"{state_key}_next"] += 1
        st.rerun()
    return boxes_cfg


def texts_ui(mode_key):
    """任意位置の文字列を追加・編集・削除する UI。axes 座標 (0..1)。

    モードごとに独立した state を持つ (`texts_{mode_key}`)。
    各文字列は default_text_annotation() のスキーマと一致した dict を返す。
    """
    state_key = f"texts_{mode_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = []
        st.session_state[f"{state_key}_next"] = 0
    items = st.session_state[state_key]
    texts_cfg = []
    pending_deletes = []
    HA_OPTS = ["left", "center", "right"]
    VA_OPTS = ["top", "center", "bottom"]
    for i, item in enumerate(list(items)):
        tid = item["id"]
        with st.expander(t("文字列{n}", n=i + 1), expanded=(i == len(items) - 1)):
            text = st.text_input(t("テキスト"), value="", key=f"txt_{mode_key}_{tid}",
                                  placeholder=t("(空のとき非表示)"))
            coord = st.radio(t("挿入位置の座標系"), list(_COORD_LABELS),
                             format_func=tr_labels(_COORD_LABELS).get,
                             horizontal=True, key=f"txtcoord_{mode_key}_{tid}",
                             help=t("axes: プロット枠の左下 (0,0)〜右上 (1,1) の相対位置。"
                                  "データ座標: 水平断面図では経度・緯度、"
                                  "その他のプロットでは軸の値で指定"))
            c1, c2 = st.columns(2)
            if coord == "data":
                x = c1.number_input(t("x (地図では経度)"), value=180.0, step=1.0,
                                     format="%g", key=f"txtxd_{mode_key}_{tid}")
                y = c2.number_input(t("y (地図では緯度)"), value=0.0, step=1.0,
                                     format="%g", key=f"txtyd_{mode_key}_{tid}")
            else:
                x = c1.number_input(t("x (axes 座標, 0=左 1=右)"), -0.5, 1.5, 0.05, 0.01,
                                     key=f"txtx_{mode_key}_{tid}")
                y = c2.number_input(t("y (axes 座標, 0=下 1=上)"), -0.5, 1.5, 0.95, 0.01,
                                     key=f"txty_{mode_key}_{tid}")
            fontsize = st.number_input(t("文字サイズ"), 1, 40, 12, 1,
                                        key=f"txtfs_{mode_key}_{tid}")
            txt_meta = {}
            color = color_selector(t("文字色"), "#000000", key=f"txtc_{mode_key}_{tid}",
                                    meta_store=txt_meta, meta_key="color")
            c3, c4 = st.columns(2)
            ha = c3.selectbox(t("横揃え (ha)"), HA_OPTS, index=0,
                               key=f"txtha_{mode_key}_{tid}",
                               help=t("x にテキストのどこを合わせるか "
                                    "(left=左端 / center=中央 / right=右端)"))
            va = c4.selectbox(t("縦揃え (va)"), VA_OPTS, index=0,
                               key=f"txtva_{mode_key}_{tid}",
                               help=t("y にテキストのどこを合わせるか "
                                    "(top=上端 / center=中央 / bottom=下端)"))
            rotation = st.slider(t("回転角 (度)"), -180.0, 180.0, 0.0, 1.0,
                                  key=f"txtr_{mode_key}_{tid}")
            txt_dict = {"text": text, "coord": coord,
                        "x": float(x), "y": float(y),
                        "fontsize": int(fontsize), "color": color,
                        "ha": ha, "va": va, "rotation": float(rotation)}
            if "_color_labels" in txt_meta:
                txt_dict["_color_labels"] = txt_meta["_color_labels"]
            texts_cfg.append(txt_dict)
            if st.button(t("この文字列を削除"), key=f"txtdel_{mode_key}_{tid}"):
                pending_deletes.append(item)
    if st.button(t("文字列を追加"), key=f"txtadd_{mode_key}"):
        items.append({"id": st.session_state[f"{state_key}_next"]})
        st.session_state[f"{state_key}_next"] += 1
        st.rerun()
    if pending_deletes:
        for it in pending_deletes:
            items.remove(it)
        st.rerun()
    return texts_cfg


_ANNOTATION_MARKER_LABELS = {
    "o": "○ (circle)",
    "s": "■ (square)",
    "^": "▲ (triangle up)",
    "v": "▼ (triangle down)",
    "<": "◀ (triangle left)",
    ">": "▶ (triangle right)",
    "x": "× (x)",
    "+": "+ (plus)",
    "D": "◇ (diamond)",
    "*": "☆ (star)",
    ".": "• (dot)",
}


def markers_ui(mode_key):
    """任意位置の記号 (マーカー) を追加・編集・削除する UI。

    texts_ui と同じ構造で、モードごとに独立した state を持つ (`markers_{mode_key}`)。
    各記号は default_marker_annotation() のスキーマと一致した dict を返す。
    """
    state_key = f"markers_{mode_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = []
        st.session_state[f"{state_key}_next"] = 0
    items = st.session_state[state_key]
    markers_cfg = []
    pending_deletes = []
    for i, item in enumerate(list(items)):
        mid = item["id"]
        with st.expander(t("記号{n}", n=i + 1), expanded=(i == len(items) - 1)):
            marker = st.selectbox(
                t("記号の種類"), list(_ANNOTATION_MARKER_LABELS),
                format_func=tr_labels(_ANNOTATION_MARKER_LABELS).get,
                key=f"mk_{mode_key}_{mid}")
            coord = st.radio(t("挿入位置の座標系"), list(_COORD_LABELS),
                             format_func=tr_labels(_COORD_LABELS).get,
                             horizontal=True, key=f"mkcoord_{mode_key}_{mid}",
                             help=t("axes: プロット枠の左下 (0,0)〜右上 (1,1) の相対位置。"
                                  "データ座標: 水平断面図では経度・緯度、"
                                  "その他のプロットでは軸の値で指定"))
            c1, c2 = st.columns(2)
            if coord == "data":
                x = c1.number_input(t("x (地図では経度)"), value=180.0, step=1.0,
                                     format="%g", key=f"mkxd_{mode_key}_{mid}")
                y = c2.number_input(t("y (地図では緯度)"), value=0.0, step=1.0,
                                     format="%g", key=f"mkyd_{mode_key}_{mid}")
            else:
                x = c1.number_input(t("x (axes 座標, 0=左 1=右)"), -0.5, 1.5, 0.5, 0.01,
                                     key=f"mkx_{mode_key}_{mid}")
                y = c2.number_input(t("y (axes 座標, 0=下 1=上)"), -0.5, 1.5, 0.5, 0.01,
                                     key=f"mky_{mode_key}_{mid}")
            size = float(st.slider(t("大きさ (pt)"), 1.0, 60.0, 10.0, 0.5,
                                    key=f"mks_{mode_key}_{mid}"))
            mk_meta = {}
            color = color_selector(t("記号の色"), "#d62728",
                                    key=f"mkc_{mode_key}_{mid}",
                                    meta_store=mk_meta, meta_key="color")
            edge_width = 0.0
            edge_color = "#000000"
            if st.checkbox(t("枠線を付ける"), value=False,
                            key=f"mke_{mode_key}_{mid}"):
                edge_width = float(st.slider(
                    t("枠線の太さ"), 0.1, 3.0, 0.8, 0.1,
                    key=f"mkelw_{mode_key}_{mid}"))
                edge_color = color_selector(t("枠線の色"), "#000000",
                                             key=f"mkec_{mode_key}_{mid}",
                                             meta_store=mk_meta,
                                             meta_key="edge_color")
            mk_dict = {"marker": marker, "coord": coord,
                       "x": float(x), "y": float(y),
                       "size": size, "color": color,
                       "edge_width": edge_width, "edge_color": edge_color}
            if "_color_labels" in mk_meta:
                mk_dict["_color_labels"] = mk_meta["_color_labels"]
            markers_cfg.append(mk_dict)
            if st.button(t("この記号を削除"), key=f"mkdel_{mode_key}_{mid}"):
                pending_deletes.append(item)
    if st.button(t("記号を追加"), key=f"mkadd_{mode_key}"):
        items.append({"id": st.session_state[f"{state_key}_next"]})
        st.session_state[f"{state_key}_next"] += 1
        st.rerun()
    if pending_deletes:
        for it in pending_deletes:
            items.remove(it)
        st.rerun()
    return markers_cfg


def layers_ui(datasets, variables_by_ds, keep_dims, allowed_kinds, mode_key,
              roles=None, allow_averaging=False, panel=None):
    """レイヤーの追加・削除・設定。モードごとに独立した状態を持つ。

    roles を渡すと各レイヤー UI で時刻・描画軸以外の次元 (鉛直レベル等) を
    レイヤー毎に固定できる。水平断面図でのみ使う。
    allow_averaging=True で固定 dim を「固定値」or「範囲平均」に切替可能にする
    (鉛直断面図で経度/緯度方向の平均を取りたい用途)。
    panel はここまでに決まったパネル側の設定 (selection / region / ranges /
    x_dim / y_dim) を持つ部分的な panel dict。各レイヤー UI が既定値・参考表示
    (データ範囲・描画レベル) を「選択中の断面」から作るのに使う
    (widgets.layer_preview_data)。渡さないモードは先頭断面で代用される。
    """
    state_key = f"layers_{mode_key}"
    if state_key not in st.session_state:
        initial_kind = allowed_kinds[0] if allowed_kinds else "fill"
        st.session_state[state_key] = [{"id": 0, "kind": initial_kind}]
        st.session_state[f"{state_key}_next"] = 1
    items = st.session_state[state_key]
    layers_cfg = []
    # ループ中の削除・並べ替えは溜めておいて、全 widget render 後にまとめて反映
    # (途中で st.rerun() すると未 render の後続レイヤーの widget state が消える)
    pending_deletes = []
    pending_moves = []  # (現在位置, 移動先位置)
    section_header(t("レイヤー"))
    for i, item in enumerate(list(items)):
        with st.expander(t("レイヤー{n}: {kind}", n=i + 1,
                           kind=t(KIND_LABELS[item["kind"]])), expanded=(i == 0)):
            lid = f"{mode_key}_{item['id']}"
            # 並べ替え (リスト順 = 描画順 = 凡例の並び)。重なりは matplotlib の既定 zorder
            # (塗り・ハッチ・ベクトル・散布 1 / 海岸線 1.5 / 等値線・流線・トラック 2) が
            # 先に効き、リスト順は同じ zorder の中でしか効かない (2026-09-20 確認)。
            # widget key は id ベースなので、リストを入れ替えるだけで各レイヤーの
            # 設定はそのまま付いて回る。key は "_" 始まりでセッション保存から除外
            if len(items) > 1:
                c_up, c_dn, _sp = st.columns([1, 1, 3])
                if c_up.button("↑", key=f"_mvup_{lid}", disabled=(i == 0),
                               help=t("1つ上へ (先に描く。凡例の並びも同じ。地図では同じ種類のレイヤー同士の重なりだけが変わる — 塗り・ハッチ・ベクトル・散布は常に等値線・流線・トラックの下)")):
                    pending_moves.append((i, i - 1))
                if c_dn.button("↓", key=f"_mvdn_{lid}",
                               disabled=(i == len(items) - 1),
                               help=t("1つ下へ (後に描く。凡例の並びも同じ。地図では同じ種類のレイヤー同士の重なりだけが変わる — 塗り・ハッチ・ベクトル・散布は常に等値線・流線・トラックの下)")):
                    pending_moves.append((i, i + 1))
            kind = st.selectbox(t("種類"), allowed_kinds,
                                index=allowed_kinds.index(item["kind"])
                                if item["kind"] in allowed_kinds else 0,
                                format_func=tr_labels(KIND_LABELS).get,
                                key=f"kind_{lid}")
            # 種類変更時は item を更新するだけ。st.rerun() を呼ぶと未 render の
            # 後続レイヤーの widget state が消えるので呼ばない (expander 見出しは
            # 次の自然な rerun で更新される)
            if kind != item["kind"]:
                item["kind"] = kind
            cfg = LAYER_UI[item["kind"]](datasets, variables_by_ds, keep_dims,
                                          lid, roles=roles,
                                          allow_averaging=allow_averaging,
                                          panel=panel)
            if cfg is not None:
                layers_cfg.append(cfg)
            if len(items) > 1 and st.button(t("このレイヤーを削除"), key=f"del_{lid}"):
                pending_deletes.append(item)
    # key に UI 言語を含める: expander の外にある selectbox は、同じ key・同じ index の
    # まま選択肢の文字列だけ変わっても (言語切替)、ブラウザ側が表示中の文字列を
    # 保持して新しい言語に更新されない (2026-09-19 実機で確認。サーバー側は翻訳済み)。
    # 言語ごとに別 widget にして再生成させる。選択は先頭に戻るが、追加種別を選ぶ
    # だけの widget なので設定には影響しない (セッション除外キー)
    new_kind = st.selectbox(t("追加するレイヤー種別"), allowed_kinds,
                            format_func=tr_labels(KIND_LABELS).get,
                            key=f"addkind_{mode_key}_{current_lang()}")
    if st.button(t("レイヤーを追加"), key=f"add_{mode_key}"):
        items.append({"id": st.session_state[f"{state_key}_next"], "kind": new_kind})
        st.session_state[f"{state_key}_next"] += 1
        st.rerun()
    # 削除・並べ替えはここで適用 (全レイヤーの widget が render 済みなので
    # state は保たれる)
    if pending_deletes:
        for it in pending_deletes:
            items.remove(it)
        st.rerun()
    if pending_moves:
        for src, dst in pending_moves:
            if 0 <= src < len(items) and 0 <= dst < len(items):
                items[src], items[dst] = items[dst], items[src]
        st.rerun()
    return layers_cfg


def _used_variables_from_state(mode_key):
    """layers_ui が前回の rerun で保存した state を覗いて、使用中の
    (変数名, 「値の変換」の key_prefix) を集める。

    key_prefix はそのレイヤーの value_transform_ui() に渡しているもの
    (`fill_{lid}` 等) で、値域表示を変換後の単位にするために使う
    (widgets.transformed_range)。初回起動時は空になり得る (UI 構築前)。
    重複は許す (呼び出し側で dedup)。
    """
    items = st.session_state.get(f"layers_{mode_key}", [])
    used = []
    for item in items:
        lid = f"{mode_key}_{item['id']}"
        kind = item.get("kind")
        if kind == "fill":
            v = st.session_state.get(f"fill_var_{lid}")
            if v:
                used.append((v, f"fill_{lid}"))
        elif kind == "contour":
            v = st.session_state.get(f"cont_var_{lid}")
            if v:
                used.append((v, f"cont_{lid}"))
        elif kind == "hatch":
            v = st.session_state.get(f"hatch_var_{lid}")
            if v:
                used.append((v, f"hatch_{lid}"))
        elif kind in ("vector", "stream"):
            prefix = "vec" if kind == "vector" else "strm"
            for k in (f"{prefix}_u_{lid}", f"{prefix}_v_{lid}"):
                v = st.session_state.get(k)
                if v:
                    used.append((v, f"{prefix}_{lid}"))
    return used


def _transform_parts(scale, offset):
    """値域表示に添える変換の表記 (言語非依存)。恒等なら空リスト。

    例: ×0.01 / -273.15 / ×2 -500。同じ変数を別の変換で使う行を区別する。
    """
    parts = []
    if scale != 1.0:
        parts.append(f"×{scale:g}")
    if offset != 0.0:
        parts.append(f"{offset:+g}")
    return parts


def _layer_fixed_dims_from_state(ds, var, time_dim, prefix):
    """レイヤー側 (_layer_dim_picker / _with_avg) が固定・範囲平均している次元を
    session_state から集める。{dim: 固定値 | (lo, hi)}。

    水平断面図の鉛直レベル等はパネル側 (`sel_{mode_key}_{dim}`) ではなく
    レイヤー側 (`{prefix}_sel_{dim}`) で固定されるので、値域の集計はまず
    こちらを見る。範囲平均は「平均後の値域」が出せないので範囲の生データを
    そのまま使う (平均よりワイド)。経度の wrap-around (min > max) は
    切り出さない (全周のまま)。
    """
    fixed = {}
    for dim in ds[var].dims:
        if dim == time_dim or dim not in ds.coords:
            continue
        if st.session_state.get(f"{prefix}_avgmode_{dim}") == "range":
            lo = st.session_state.get(f"{prefix}_avgrnglo_{dim}")
            hi = st.session_state.get(f"{prefix}_avgrnghi_{dim}")
            if lo is None or hi is None:
                rng = st.session_state.get(f"{prefix}_avgrng_{dim}")
                if rng is not None:
                    lo, hi = rng
            if lo is not None and hi is not None and float(lo) <= float(hi):
                fixed[dim] = (float(lo), float(hi))
            continue
        v = st.session_state.get(f"{prefix}_sel_{dim}")
        if v is not None:
            fixed[dim] = v
    return fixed


def _fixed_dim_label(dim, v):
    """値域表示に添える固定次元の表記 (言語非依存)。例: level=500 / lat=20–40。"""
    if isinstance(v, tuple):
        return f"{dim}={v[0]:g}–{v[1]:g}"
    return f"{dim}={dim_choice_label(v)}"


def _variable_minmax_in_time_range(ds, var, time_dim, t_start, t_end, roles,
                                   mode_key, prefix=None):
    """変数の時間範囲内での最小値・最大値と、適用した次元固定を返す
    (失敗時は (None, None, {}))。

    時刻以外の次元は、まずレイヤー側の固定・範囲平均 (`{prefix}_sel_{dim}` 等、
    _layer_fixed_dims_from_state) を適用し、それが無い次元のうち経度・緯度
    以外はパネル側の固定値 (`sel_{mode_key}_{dim}`) を適用する。経度・緯度の
    region 切り出しは適用しないため、min/max はややワイドめになる。
    戻り値の 3 つ目は {dim: 固定値 | (lo, hi)} (表示用)。
    """
    try:
        da = ds[var].sel({time_dim: slice(t_start, t_end)})
        layer_fixed = (_layer_fixed_dims_from_state(ds, var, time_dim, prefix)
                       if prefix else {})
        applied = {}
        for dim in da.dims:
            if dim == time_dim:
                continue
            if dim in layer_fixed:
                v = layer_fixed[dim]
                if isinstance(v, tuple):
                    da = da.sel({dim: mc_render.coord_slice(da[dim], v[0], v[1])})
                else:
                    da = da.sel({dim: v})
                applied[dim] = v
                continue
            # 水平面の dim (1 次元格子 = lat/lon、curvilinear = y/x) は固定しない
            if dim in (mc_dataset.horizontal_dims(ds, roles)
                       or (roles["lat"], roles["lon"])):
                continue
            fixed = st.session_state.get(f"sel_{mode_key}_{dim}")
            if fixed is not None:
                da = da.sel({dim: fixed})
                applied[dim] = fixed
        return float(da.min()), float(da.max()), applied
    except Exception:
        return None, None, {}


# アニメーションで動かすもの (中立キー → 日本語原文)
_ANIM_MOTION_LABELS = {
    "time": "時刻送り",
    "rotate": "地球の回転 (時刻固定)",
    "both": "時刻送り + 回転",
}


def rotation_animation_available(projection) -> bool:
    """地球回転アニメーションを出せるか。

    投影法が Orthographic で、図が単一パネルのときだけ (2026-08-28 決定。複数
    パネルでは回転アニメーションなし。時刻送りは従来どおり複数パネルでも可)。
    """
    return (bool(projection) and projection.get("name") == "Orthographic"
            and len(st.session_state.get("panels", [{"id": 0}])) == 1)


def _rotation_path_ui(mode_key, projection, n_frames=None):
    """回転経路 (開始点 = 投影中心 → 経由点 (0個以上) → 終点) の UI。

    render.rotation_path で補間した各フレームの投影中心 [(lon, lat), ...] を返す。
    n_frames が None なら総フレーム数を入力させる (回転のみ)。時刻送りと同時の
    ときは時刻フレーム数を渡す。経由点リストは texts_ui と同じ動的リスト
    (`anim_rot_wps_{mode_key}` + 追加/削除ボタン)。
    """
    start = (float(projection.get("central_longitude", 0.0)),
             float(projection.get("central_latitude", 0.0)))
    st.caption(t("開始点は投影法の中心経度・中心緯度 ({lon:g}°, {lat:g}°)",
                 lon=start[0], lat=start[1]))
    state_key = f"anim_rot_wps_{mode_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = []
        st.session_state[f"{state_key}_next"] = 0
    items = st.session_state[state_key]
    points = [start]
    pending_deletes = []
    for i, item in enumerate(list(items)):
        wid = item["id"]
        c1, c2 = st.columns(2)
        lon = c1.number_input(t("経由点{n} 経度", n=i + 1), -180.0, 360.0, start[0], 10.0,
                              key=f"anim_rot_wplon_{mode_key}_{wid}")
        lat = c2.number_input(t("経由点{n} 緯度", n=i + 1), -90.0, 90.0, start[1], 10.0,
                              key=f"anim_rot_wplat_{mode_key}_{wid}")
        points.append((float(lon), float(lat)))
        if st.button(t("この経由点を削除"), key=f"rotdel_{mode_key}_{wid}"):
            pending_deletes.append(item)
    if st.button(t("経由点を追加"), key=f"rotadd_{mode_key}"):
        items.append({"id": st.session_state[f"{state_key}_next"]})
        st.session_state[f"{state_key}_next"] += 1
        st.rerun()
    if pending_deletes:
        for it in pending_deletes:
            items.remove(it)
        st.rerun()
    # 終点の既定は開始点から東へ 90° (入力範囲 -180〜360 に収める)
    end_lon_default = start[0] + 90.0
    if end_lon_default > 360.0:
        end_lon_default -= 360.0
    c1, c2 = st.columns(2)
    end_lon = c1.number_input(t("終点 経度"), -180.0, 360.0, end_lon_default, 10.0,
                              key=f"anim_rot_endlon_{mode_key}")
    end_lat = c2.number_input(t("終点 緯度"), -90.0, 90.0, start[1], 10.0,
                              key=f"anim_rot_endlat_{mode_key}")
    points.append((float(end_lon), float(end_lat)))
    st.caption(t("経路は緯度経度の線形補間。経度は最短方向に回る "
                 "(同じ向きに回し続けるには 180° 未満の間隔で経由点を置く)"))
    if n_frames is None:
        n_frames = int(st.number_input(
            t("総フレーム数"), 2, 1000, 36, 1, key=f"anim_rot_frames_{mode_key}",
            help=t("開始点から終点までを等速で動かす。各フレームで図全体を"
                   "描き直すため、フレーム数に比例して時間がかかる")))
    return mc_render.rotation_path(points, n_frames)


def animation_ui(ds, mode_key, labels, sel_vals, time_dim, roles, projection=None):
    """「時刻」セクション内に出すアニメーション操作 UI。

    生成は重い処理 (各フレームを描画) なので、ボタン押下で session_state にリクエストを
    入れ、メインの描画パス側で処理する。リクエストは
    ("play" | "file" | "script", time_values, fps, fmt, centers, dpi, frames_per_time)。
    time_values は時刻送りしないとき None、centers は地球を回転しないとき None。
    dpi は GIF / MP4 のフレーム解像度 (再生には使わない)。frames_per_time は
    各時刻のコマ数 (時刻送り + 回転のみ 1 以外になる)。

    projection (水平断面図の投影法設定) が Orthographic で単一パネルなら
    「動かすもの」で地球の回転を選べる (rotation_animation_available)。
    time_dim が None (時刻次元のないデータ) のときは回転のみ。
    """
    with st.expander(t("アニメーション表示"), expanded=False):
        rotation_ok = rotation_animation_available(projection)
        if time_dim is None:
            if not rotation_ok:
                return
            motion = "rotate"
            st.caption(t("時刻次元がないため地球の回転のみ"))
        elif rotation_ok:
            motion = st.radio(t("動かすもの"), list(_ANIM_MOTION_LABELS),
                              format_func=tr_labels(_ANIM_MOTION_LABELS).get,
                              horizontal=True, key=f"anim_motion_{mode_key}")
        else:
            motion = "time"

        time_values = None
        start_idx = end_idx = None
        if motion in ("time", "both"):
            if len(labels) < 2:
                st.caption(t("時刻が1つしかないためアニメーションできません"))
                return
            c1, c2 = st.columns(2)
            start_idx = c1.selectbox(t("開始時刻"), range(len(labels)),
                                      format_func=lambda i, L=labels: L[i],
                                      index=0, key=f"anim_start_{mode_key}")
            end_idx = c2.selectbox(t("終了時刻"), range(len(labels)),
                                    format_func=lambda i, L=labels: L[i],
                                    index=len(labels) - 1, key=f"anim_end_{mode_key}")
            step = int(st.number_input(t("ステップ (n個ごと)"), 1, 100, 1, key=f"anim_step_{mode_key}"))
            if end_idx < start_idx:
                st.warning(t("開始時刻 ≤ 終了時刻 にしてください"))
                return
            frame_indices = list(range(start_idx, end_idx + 1, step))
            time_values = [sel_vals[i] for i in frame_indices]

        centers = None
        frames_per_time = 1
        if motion == "both":
            # 各時刻を何コマ保持するか。回転はコマ毎に進むので、時刻の進みに
            # 対して回転を滑らかにできる (render.expand_time_values)
            frames_per_time = int(st.number_input(
                t("1時刻あたりのコマ数"), 1, 50, 1, 1, key=f"anim_rot_hold_{mode_key}",
                help=t("同じ時刻を何コマ続けるか。地球の回転はコマ毎に進むので、"
                       "大きくすると時刻の進みに対して回転が滑らかになる "
                       "(総フレーム数 = 時刻数 × コマ数)")))
        if motion in ("rotate", "both"):
            centers = _rotation_path_ui(
                mode_key, projection,
                n_frames=(len(time_values) * frames_per_time
                          if time_values is not None else None))

        fps = int(st.slider(t("fps (frame/秒)"), 1, 10, 4, key=f"anim_fps_{mode_key}"))
        dpi = int(st.number_input(
            t("解像度 (dpi)"), 50, 400, 100, 10, key=f"anim_dpi_{mode_key}",
            help=t("GIF / MP4 の各フレームの解像度。画素数 = 図の大きさ (inch) × dpi。"
                   "大きいほど生成に時間がかかりファイルも大きくなる")))
        n_frames = mc_render.animation_frame_count(time_values, centers, frames_per_time)
        fig_w = float(st.session_state.get("figsize_w", 10.0))
        fig_h = float(st.session_state.get("figsize_h", 6.0))
        st.caption(t("{n} フレーム、フレームの大きさの目安 {w} × {h} px",
                     n=n_frames, w=int(round(fig_w * dpi)), h=int(round(fig_h * dpi))))

        # 時間範囲内の変数の min/max (前回 rerun の layer state を読む)。
        # レイヤー毎に固定した次元 (level 等) を当て、vmin/vmax は「値の変換」後の
        # 単位で指定するので値域も変換後で出す。固定次元と変換は変数名に添えて
        # 区別し、同じ行になるものは1行にまとめる
        used = list(dict.fromkeys(_used_variables_from_state(mode_key)))
        if used and time_dim and time_values is not None:
            t_start = sel_vals[start_idx]
            t_end = sel_vals[end_idx]
            lines = []
            for var, prefix in used:
                vmin, vmax, applied = _variable_minmax_in_time_range(
                    ds, var, time_dim, t_start, t_end, roles, mode_key, prefix)
                if vmin is None:
                    continue
                scale, offset = value_transform_state(prefix)
                vmin, vmax = transformed_range(prefix, vmin, vmax)
                tparts = _transform_parts(scale, offset)
                parts = ([_fixed_dim_label(d, v) for d, v in applied.items()]
                         + ([" ".join(tparts)] if tparts else []))
                suffix = f" ({', '.join(parts)})" if parts else ""
                line = f"`{var}`{suffix}: [{vmin:g}, {vmax:g}]"
                if line not in lines:
                    lines.append(line)
            if lines:
                st.caption(t("時間範囲内の値域 — {ranges}", ranges=" / ".join(lines)))

        # 形式選択 (MP4 は ffmpeg があるときだけ)
        has_ffmpeg = mc_render.ffmpeg_available()
        fmt_opts = ["GIF"] + (["MP4"] if has_ffmpeg else [])
        fmt = st.radio(t("ファイル形式"), fmt_opts, horizontal=True, key=f"anim_format_{mode_key}")
        if not has_ffmpeg:
            st.caption(t("MP4 を出力するには ffmpeg のインストールが必要です"))
        fmt_key = fmt.lower()

        if st.button(t("再生"), key=f"anim_play_{mode_key}"):
            st.session_state["_anim_request"] = ("play", time_values, fps, fmt_key, centers, dpi, frames_per_time)
        if st.button(t("ファイル生成"), key=f"anim_file_{mode_key}"):
            st.session_state["_anim_request"] = ("file", time_values, fps, fmt_key, centers, dpi, frames_per_time)
        if st.button(t("再現スクリプト"), key=f"anim_script_{mode_key}"):
            st.session_state["_anim_request"] = ("script", time_values, fps, fmt_key, centers, dpi, frames_per_time)

# strftime プリセット (中立キー = strftime 書式そのもの、auto/custom のみ特別)
_STRFTIME_PRESET_LABELS = {
    "auto": "自動",
    "%Y": "年 (例 1984)",
    "%Y-%m": "年-月 (例 1984-01)",
    "%Y-%m-%d": "年-月-日 (例 1984-01-15)",
    "%m/%d": "月/日 (例 01/15)",
    "%H:%M": "時:分 (例 12:30)",
    "custom": "カスタム…",
}


def strftime_format_ui(label, key_prefix):
    """strftime 書式のプリセット selectbox + 「カスタム…」時のみテキスト入力。

    `%D` は標準では MM/DD/YY だが、ユーザーの直観 (D=day) に合わせて `%d` に置換する。
    """
    choice = st.selectbox(label, list(_STRFTIME_PRESET_LABELS),
                          format_func=tr_labels(_STRFTIME_PRESET_LABELS).get,
                          key=f"{key_prefix}_choice")
    if choice == "custom":
        raw = st.text_input(
            t("strftime 書式"), value="%Y", key=f"{key_prefix}_custom",
            help=t("例: %Y=年、%m=月、%D または %d=日、%H=時、%M=分。"
                 "%D は標準では MM/DD/YY ですがこのアプリでは「日」に固定")) or None
        return raw.replace("%D", "%d") if raw else None
    return None if choice == "auto" else choice


def selection_widgets(ds, roles, used_vars, keep_dims, mode_key,
                      header_label="断面の選択", skip_dims=(),
                      show_animation=False, projection=None):
    """描画軸以外の次元を固定する。時刻は前後ボタンで送れる。

    header_label は日本語原文のまま渡す (表示時に t() で翻訳される)。

    mode_key はパネルスコープ (例 "map0")。widget key をパネル毎に分離する。
    skip_dims に渡した次元は (別の場所で既に選択済みとみなして) 飛ばす。
    show_animation=True で「時刻」セクション内にアニメーション UI も出す
    (水平断面・鉛直断面・1次元プロット。時間断面では時刻が軸そのものなので不要)。
    projection は水平断面図の投影法設定 (Orthographic なら地球回転アニメーション
    を選べる。時刻次元が無いデータでも回転のみの UI を出す)。

    戻り値: (selection, time_label_settings)。
      time_label_settings = {"show": bool, "loc": str, "format": str | None}
    """
    selection = {}
    time_label_settings = {"show": False, "loc": "left", "format": None}
    seen = list(skip_dims)
    section_header(t(header_label))
    for var in used_vars:
        for dim in ds[var].dims:
            if dim in keep_dims or dim in seen or dim not in ds.coords:
                continue
            seen.append(dim)
            values = ds[dim].values
            if dim == roles["time"]:
                labels, sel_vals = time_labels_and_values(values)
                key = f"sel_{mode_key}_{dim}"
                if key not in st.session_state:
                    st.session_state[key] = 0
                c1, c2 = st.columns(2)
                if c1.button(t("◀ 前の時刻"), key=f"prev_{mode_key}_{dim}"):
                    st.session_state[key] = max(0, st.session_state[key] - 1)
                if c2.button(t("次の時刻 ▶"), key=f"next_{mode_key}_{dim}"):
                    st.session_state[key] = min(len(labels) - 1, st.session_state[key] + 1)
                idx = st.selectbox(t("時刻 ({dim})", dim=dim), range(len(labels)),
                                   format_func=lambda i, L=labels: L[i], key=key)
                selection[dim] = sel_vals[idx]
                with st.expander(t("時刻を表示"), expanded=False):
                    show = st.checkbox(t("プロット上部に時刻を表示"), value=False,
                                       key=f"show_time_label_{mode_key}")
                    if show:
                        loc = st.selectbox(t("位置"), list(_TL_LOC_LABELS),
                                           format_func=tr_labels(_TL_LOC_LABELS).get,
                                           key=f"tl_loc_{mode_key}")
                        # strftime 書式は datetime64 のときだけ (数値時刻 —
                        # cftime 由来の通し日数軸や時刻役割の上書き — では無意味)
                        fmt = (strftime_format_ui(t("時刻の書式"),
                                                  f"tl_fmt_{mode_key}")
                               if np.issubdtype(np.asarray(values).dtype,
                                                np.datetime64) else None)
                        time_label_settings = {
                            "show": True,
                            "loc": loc,
                            "format": fmt,
                        }
                if show_animation:
                    animation_ui(ds, mode_key, labels, sel_vals, dim, roles,
                                 projection=projection)
            else:
                units = ds[dim].attrs.get("units", "")
                choices = dim_choices(values)
                selection[dim] = st.selectbox(
                    f"{dim}" + (f" [{units}]" if units else ""), choices,
                    index=_default_dim_index(ds, dim, roles, choices),
                    format_func=dim_choice_label, key=f"sel_{mode_key}_{dim}")
    if (show_animation and roles.get("time") not in seen
            and rotation_animation_available(projection)):
        # 時刻次元のないデータ (気候値など) でも Orthographic なら地球回転だけは出す
        animation_ui(ds, mode_key, [], [], None, roles, projection=projection)
    return selection, time_label_settings


def axis_settings_ui(ds, roles, x_dim, y_dim, mode_key, header_label="軸・ラベル"):
    """断面図の軸設定 (反転・対数・ラベル・目盛)。header_label は原文のまま渡す。"""
    section_header(t(header_label))
    axis = mc_config.default_section_panel()["axis"]
    # 「時間軸」の扱いは datetime64 のときだけ (cftime 由来の数値時間軸では
    # strftime 書式を隠し、目盛位置の手動指定を解禁する)
    x_is_time = (x_dim == roles["time"]
                 and np.issubdtype(ds[x_dim].dtype, np.datetime64))
    y_role_time = (y_dim == roles["time"])
    y_is_time = y_role_time and np.issubdtype(ds[y_dim].dtype, np.datetime64)
    y_attrs = ds[y_dim].attrs
    pressure_like = (str(y_attrs.get("units", "")).lower() in PRESSURE_UNITS
                     or y_attrs.get("positive") == "down")
    with st.expander(t("軸"), expanded=False):
        axis["invert_x"] = st.checkbox(t("横軸を反転"), value=False,
                                       key=f"invx_{mode_key}_{x_dim}")
        # 気圧軸は上が低圧、Hovmöller図は時間が下向きになるよう、デフォルトで反転
        axis["invert_y"] = st.checkbox(t("縦軸を反転"),
                                       value=bool(pressure_like or y_role_time),
                                       key=f"invy_{mode_key}_{y_dim}")
        axis["swap_y_sides"] = st.checkbox(
            t("縦軸を右側に表示"), value=False, key=f"swapy_{mode_key}",
            help=t("y 軸の目盛・ラベルを右側に表示する"))
        if not y_is_time:
            axis["log_y"] = st.checkbox(t("縦軸を対数軸にする"), value=False,
                                        key=f"logy_{mode_key}_{y_dim}")
        if x_dim == roles.get("lon"):
            axis["x_lon_east_west"] = st.checkbox(
                t("x軸 (経度) を東経・西経表記にする"), value=False,
                key=f"lonew_{mode_key}_{x_dim}",
                help=t("180° を中心に 120°E … 180° … 120°W のように表示する。"
                     "off では数値 (度、東経 0–360) のまま。「目盛」で"
                     "カスタムラベルを指定した場合はそちらが優先される"))
        if x_is_time or y_is_time:
            axis["time_axis_format"] = strftime_format_ui(t("時間軸の書式"),
                                                           f"tfmt_{mode_key}")
    with st.expander(t("ラベル"), expanded=False):
        axis["x_label"] = st.text_input(t("x軸ラベル"),
                                        value=coord_label(ds, roles, x_dim),
                                        key=f"xlab_{mode_key}_{x_dim}") or None
        axis["y_label"] = st.text_input(t("y軸ラベル"),
                                        value=coord_label(ds, roles, y_dim),
                                        key=f"ylab_{mode_key}_{y_dim}") or None
        axis["label_fontsize"] = fontsize_input(t("軸ラベル文字サイズ"),
                                                f"labfs_{mode_key}",
                                                default=10, show_auto_hint=False)
        label_style_ui(axis, mode_key)
    axis_ticks_ui(axis, mode_key, x_is_time=x_is_time, y_is_time=y_is_time)
    return axis


def _parse_tick_positions(text: str) -> list[float] | None:
    """カンマ/空白区切りの数値文字列を float のリストに変換。失敗したら None。"""
    if not text or not text.strip():
        return None
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    try:
        return [float(p) for p in parts] or None
    except ValueError:
        return None


def _parse_tick_labels(text: str) -> list[str] | None:
    """カンマ区切りの文字列をラベルのリストに変換。空文字や None は None。

    位置と違ってラベル内の空白や記号 (°/E など) を保持したいので、カンマでのみ分割し
    各要素を strip する。空欄エントリは空文字列として残す (位置との対応のため)。
    """
    if not text or not text.strip():
        return None
    return [p.strip() for p in text.split(",")] or None


def line_axis_settings_ui(ds, roles, x_dim, mode_key):
    """1次元プロットの軸設定 (x/y ラベル・対数・反転・グリッド・時間軸書式)。"""
    section_header(t("軸・ラベル・凡例"))
    axis = mc_config.default_line_panel()["axis"]
    # 既定は次元名 [units] (long_name は使わない)
    x_units = ds[x_dim].attrs.get("units", "") if x_dim in ds.coords else ""
    x_default = f"{x_dim} [{x_units}]" if x_units else str(x_dim)
    # datetime64 のときだけ「時間軸」扱い (数値時間軸では対数・範囲手動
    # 指定・目盛位置を解禁し、strftime 書式を隠す)
    x_is_time = (x_dim == roles.get("time")
                 and np.issubdtype(ds[x_dim].dtype, np.datetime64))
    axis_basic_ui(axis, mode_key, show_tight=True, x_is_time=x_is_time,
                  show_swap_y=True)
    with st.expander(t("ラベル"), expanded=False):
        axis["x_label"] = st.text_input(t("x軸ラベル"), value=x_default,
                                         key=f"xlab_{mode_key}_{x_dim}") or None
        axis["y_label"] = st.text_input(t("y軸ラベル"), value="", placeholder=t("(なし)"),
                                         key=f"ylab_{mode_key}") or None
        axis["label_fontsize"] = fontsize_input(t("文字サイズ"), f"labfs_{mode_key}",
                                                 default=10, show_auto_hint=False)
        label_style_ui(axis, mode_key)
    # 第2軸は「目盛」より前に描く (目盛線の色・太さ UI が grid.show_y2 を見るため)
    secondary_axis_ui(axis, mode_key)
    axis_ticks_ui(axis, mode_key, x_is_time=x_is_time)
    return axis


def secondary_axis_ui(axis: dict, mode_key: str):
    """「第2軸 (右)」expander (1次元プロット): 第2軸 (twinx) のラベル・範囲・
    対数・反転・値揃え・目盛文字の表示・目盛位置・補助目盛・目盛線。

    axis の y2 系キー (config.default_line_panel) を直接更新する。第2軸は
    レイヤーの「第2軸 (右の縦軸) に描く」が ON のときだけ作られ、無ければ
    ここでの設定は無視される。文字サイズ・線の太さ・回転・ラベル体裁・目盛線の
    色/太さ/線種は第1軸 (「ラベル」「目盛」) と共有 (render.secondary_axis_cfg)。
    """
    with st.expander(t("第2軸 (右)"), expanded=False):
        st.caption(t("第2軸はレイヤーの「第2軸 (右の縦軸) に描く」が ON の"
                   "ときだけ作られる"))
        axis["y2_label"] = st.text_input(
            t("第2軸 (右) ラベル"), value="", placeholder=t("(なし)"),
            key=f"y2lab_{mode_key}") or None
        c1, c2 = st.columns(2)
        axis["log_y2"] = c1.checkbox(t("第2軸を対数"), value=False,
                                      key=f"logy2_{mode_key}")
        axis["invert_y2"] = c2.checkbox(t("第2軸を反転"), value=False,
                                         key=f"invy2_{mode_key}")
        if st.checkbox(t("第2軸 (右) の範囲を手動指定"), value=False,
                        key=f"y2lim_manual_{mode_key}"):
            c1, c2 = st.columns(2)
            y2_min = c1.number_input(t("第2軸 最小"), value=0.0, step=1.0,
                                      format="%g", key=f"y2lim_lo_{mode_key}")
            y2_max = c2.number_input(t("第2軸 最大"), value=1.0, step=1.0,
                                      format="%g", key=f"y2lim_hi_{mode_key}")
            axis["y2_lim"] = [float(y2_min), float(y2_max)]
        if st.checkbox(t("指定した値の高さを左右で揃える"), value=False,
                        key=f"y2align_{mode_key}",
                        help=t("両軸の範囲を必要な側へ広げて、この値 (例: 0) が"
                             "左右の軸で同じ高さになるようにする。軸は縮めない。"
                             "どちらかが対数軸のときは無効")):
            axis["y2_align_value"] = float(st.number_input(
                t("揃える値"), value=0.0, step=1.0, format="%g",
                key=f"y2align_v_{mode_key}"))
        axis["show_y2_ticklabels"] = st.checkbox(
            t("第2軸の目盛文字を表示"), value=True, key=f"showy2t_{mode_key}")
        axis["y2_tick_interval"] = None
        axis["y2_tick_positions"] = None
        _tick_mode_ui(
            axis, mode_key, "y2",
            mode_label=t("第2軸の目盛位置"),
            interval_label=t("第2軸の目盛間隔 (0 で非表示)"),
            positions_label=t("第2軸の目盛位置 (カンマ区切り)"),
            default_positions="0, 10, 20, 50",
            positions_help=t("例: 0, 1, 2, 5, 10. 不等間隔可"),
            labels_label=t("第2軸の目盛ラベル (位置と同数、カンマ区切り)"),
            labels_help=t("位置の数と一致しないと無視される"))
        c1, c2 = st.columns(2)
        axis["show_y2_minor_ticks"] = c1.checkbox(
            t("第2軸に補助目盛を表示"), value=False, key=f"y2minor_{mode_key}")
        axis["grid"]["show_y2"] = c2.checkbox(
            t("第2軸に目盛線を表示"), value=False, key=f"gridy2_{mode_key}",
            help=t("色・太さ・線種は「目盛」の目盛線設定と共通。第2軸は上層に"
                 "描かれるため、第1軸の線の上に目盛線が乗る"))


def axis_basic_ui(axis: dict, mode_key: str, *, show_tight: bool = False,
                  x_is_time: bool = False, show_log: bool = True,
                  show_swap_y: bool = False, show_y2: bool = False):
    """「軸」expander (1次元・2次元プロット・heatmap・集計 共通): 対数・反転・範囲。

    axis を直接更新する。show_tight=True (1次元プロット) のとき
    「範囲ぴったりに揃える」も描く。x_is_time=True のとき log_x と
    x範囲の手動指定を出さず、時間軸の書式を描く (datetime 軸では
    対数軸・数値での範囲指定が未対応のため)。show_log=False (heatmap) の
    とき対数チェックを出さない (カテゴリ軸に対数は無意味なため)。
    show_swap_y / show_y2 (1次元プロット・dist_1d) で縦軸の左右入れ替えと
    第2軸 (twinx) の範囲指定を出す。
    """
    with st.expander(t("軸"), expanded=False):
        if show_log:
            c1, c2 = st.columns(2)
            if not x_is_time:
                axis["log_x"] = c1.checkbox(t("横軸を対数"), value=False,
                                              key=f"logx_{mode_key}")
            axis["log_y"] = c2.checkbox(t("縦軸を対数"), value=False,
                                          key=f"logy_{mode_key}")
        c1, c2 = st.columns(2)
        axis["invert_x"] = c1.checkbox(t("横軸を反転"), value=False,
                                        key=f"invx_{mode_key}")
        axis["invert_y"] = c2.checkbox(t("縦軸を反転"), value=False,
                                        key=f"invy_{mode_key}")
        if show_swap_y:
            axis["swap_y_sides"] = st.checkbox(
                t("縦軸の左右を入れ替える"), value=False, key=f"swapy_{mode_key}",
                help=t("y 軸の目盛・ラベルを右側に表示する。第2軸 (twinx) が"
                     "あるときは第1軸→右・第2軸→左の入れ替えになる"))
        if show_tight:
            axis["tight_x"] = st.checkbox(
                t("x軸を範囲ぴったりに揃える (余白なし)"), value=False,
                key=f"tightx_{mode_key}",
                help=t("off だと matplotlib が両端に小さな余白を取る。"
                     "on で図の左右端をデータ範囲に揃える"))
            axis["tight_y"] = st.checkbox(
                t("y軸を範囲ぴったりに揃える (余白なし)"), value=False,
                key=f"tighty_{mode_key}",
                help=t("off だと matplotlib が両端に小さな余白を取る。"
                     "on で図の上下端をデータ範囲に揃える"))
        # x軸範囲の手動指定 (時間軸では非対応)
        if not x_is_time and st.checkbox(
                t("x軸の範囲を手動指定"), value=False, key=f"xlim_manual_{mode_key}"):
            c1, c2 = st.columns(2)
            x_min_val = c1.number_input(t("x軸 最小"), value=0.0, step=1.0,
                                          format="%g", key=f"xlim_lo_{mode_key}")
            x_max_val = c2.number_input(t("x軸 最大"), value=1.0, step=1.0,
                                          format="%g", key=f"xlim_hi_{mode_key}")
            axis["x_lim"] = [float(x_min_val), float(x_max_val)]
        if st.checkbox(t("y軸の範囲を手動指定"), value=False,
                        key=f"ylim_manual_{mode_key}"):
            c1, c2 = st.columns(2)
            y_min_val = c1.number_input(t("y軸 最小"), value=0.0, step=1.0,
                                          format="%g", key=f"ylim_lo_{mode_key}")
            y_max_val = c2.number_input(t("y軸 最大"), value=1.0, step=1.0,
                                          format="%g", key=f"ylim_hi_{mode_key}")
            axis["y_lim"] = [float(y_min_val), float(y_max_val)]
        if show_y2 and st.checkbox(
                t("第2軸 (右) の範囲を手動指定"), value=False,
                key=f"y2lim_manual_{mode_key}",
                help=t("第2軸はレイヤーの「第2軸 (右の縦軸) に描く」が ON の"
                     "ときだけ作られる")):
            c1, c2 = st.columns(2)
            y2_min = c1.number_input(t("第2軸 最小"), value=0.0, step=1.0,
                                      format="%g", key=f"y2lim_lo_{mode_key}")
            y2_max = c2.number_input(t("第2軸 最大"), value=1.0, step=1.0,
                                      format="%g", key=f"y2lim_hi_{mode_key}")
            axis["y2_lim"] = [float(y2_min), float(y2_max)]
        if x_is_time:
            axis["time_axis_format"] = strftime_format_ui(t("時間軸の書式"),
                                                           f"tfmt_{mode_key}")


def axis_labels_ui(axis: dict, mode_key: str, *, x_placeholder: str,
                   y_placeholder: str, show_y2: bool = False) -> None:
    """「ラベル」expander (x/y 軸ラベル・文字サイズ・体裁)。axis を直接更新する。

    1次元プロット(集計)・2次元プロット(集計)・散布図・heatmap 共通
    (1次元プロット・断面図は line_axis_settings_ui / axis_settings_ui 側)。
    x_placeholder / y_placeholder は空欄時の案内 (翻訳済みの文字列を渡す)。
    show_y2=True で第2軸 (右) のラベル欄も出す (dist_1d)。
    """
    with st.expander(t("ラベル"), expanded=False):
        axis["x_label"] = st.text_input(
            t("x軸ラベル"), value="", placeholder=x_placeholder,
            key=f"xlab_{mode_key}") or None
        axis["y_label"] = st.text_input(
            t("y軸ラベル"), value="", placeholder=y_placeholder,
            key=f"ylab_{mode_key}") or None
        if show_y2:
            axis["y2_label"] = st.text_input(
                t("第2軸 (右) ラベル"), value="", placeholder=t("(なし)"),
                key=f"y2lab_{mode_key}",
                help=t("レイヤーの「第2軸 (右の縦軸) に描く」が ON のときだけ"
                     "表示される軸のラベル")) or None
        axis["label_fontsize"] = fontsize_input(
            t("文字サイズ"), f"labfs_{mode_key}", default=10,
            show_auto_hint=False)
        label_style_ui(axis, mode_key)


def label_style_ui(axis: dict, mode_key: str):
    """「ラベル」expander 内の体裁 widgets (色・太字・斜体・距離・回転)。

    1次元・2次元プロット・断面図共通。axis を直接更新する。
    """
    # 色 (既定: matplotlib デフォルトで rcParam に従う)
    if st.checkbox(t("文字の色を指定"), value=False, key=f"labcoluse_{mode_key}"):
        axis["label_color"] = color_selector(t("文字の色"), "#000000",
                                               key=f"labcol_{mode_key}",
                                               meta_store=axis,
                                               meta_key="label_color")
    c1, c2 = st.columns(2)
    axis["label_weight"] = "bold" if c1.checkbox(
        t("太字"), value=False, key=f"labbold_{mode_key}") else "normal"
    axis["label_italic"] = c2.checkbox(
        t("斜体"), value=False, key=f"labital_{mode_key}")
    # 軸からの距離 (labelpad)
    if st.checkbox(t("軸からの距離を指定"), value=False, key=f"labpaduse_{mode_key}",
                     help=t("off で matplotlib 既定 (おおむね 4 pt)")):
        axis["label_pad"] = float(st.number_input(
            t("距離 (pt)"), value=4.0, step=0.5, format="%g",
            key=f"labpad_{mode_key}"))
    # 回転 (軸ごと)。既定: matplotlib (x=0°, y=90°)
    if st.checkbox(t("ラベルの回転を指定"), value=False, key=f"labrotuse_{mode_key}",
                     help=t("off で matplotlib 既定 (x=0°, y=90°)")):
        c3, c4 = st.columns(2)
        axis["x_label_rotation"] = float(c3.number_input(
            t("x軸の回転 (度)"), value=0.0, step=15.0, format="%g",
            key=f"xlabrot_{mode_key}"))
        axis["y_label_rotation"] = float(c4.number_input(
            t("y軸の回転 (度)"), value=90.0, step=15.0, format="%g",
            key=f"ylabrot_{mode_key}"))


def _tick_mode_ui(axis: dict, mode_key: str, ax: str, *, mode_label: str,
                  interval_label: str, positions_label: str,
                  default_positions: str, positions_help: str,
                  labels_label: str, labels_help: str) -> None:
    """目盛位置 (自動 / 等間隔 / 位置を直接指定 + カスタムラベル) の共通部品。

    ax = "x" / "y" / "y2"。書き込み先は axis[f"{ax}_tick_interval"] /
    axis[f"{ax}_tick_positions"] / axis[f"{ax}_tick_labels"] (呼び出し側で
    interval / positions を None に初期化しておく)。widget key は x / y の
    既存体系 ({ax}tmode_ / {ax}tick_int_ / {ax}tick_pos_ / {ax}tlab_use_ /
    {ax}tlab_) をそのまま使う (WIP 互換)。表示文字列は呼び出し側で t() 済みの
    ものを受け取る (i18n の網羅テストが t() のリテラル引数を数えるため)。
    """
    mode = st.radio(
        mode_label, list(_TICK_MODE_LABELS),
        format_func=tr_labels(_TICK_MODE_LABELS).get,
        horizontal=True, key=f"{ax}tmode_{mode_key}",
        help=t("位置を直接指定: 不等間隔も可。カンマ区切りで入力"))
    if mode == "interval":
        axis[f"{ax}_tick_interval"] = float(st.number_input(
            interval_label, value=1.0, min_value=0.0, step=0.1, format="%g",
            key=f"{ax}tick_int_{mode_key}"))
    elif mode == "positions":
        positions = _parse_tick_positions(st.text_input(
            positions_label, value=default_positions,
            key=f"{ax}tick_pos_{mode_key}", help=positions_help))
        if positions:
            axis[f"{ax}_tick_positions"] = positions
            if st.checkbox(t("目盛位置にカスタムラベルを使う"), value=False,
                            key=f"{ax}tlab_use_{mode_key}"):
                default_text = ", ".join(f"{p:g}" for p in positions)
                labels = _parse_tick_labels(st.text_input(
                    labels_label, value=default_text,
                    key=f"{ax}tlab_{mode_key}", help=labels_help))
                if labels and len(labels) == len(positions):
                    axis[f"{ax}_tick_labels"] = labels


def axis_ticks_ui(axis: dict, mode_key: str, x_is_time: bool = False,
                  y_is_time: bool = False, *, show_minor: bool = True,
                  show_grid: bool = True,
                  rotation_panel: dict | None = None):
    """「目盛」expander (1次元・2次元プロット・断面図・heatmap 共通)。
    axis を直接更新する。

    x_is_time / y_is_time が True のとき、その軸の目盛位置の手動指定 UI を
    出さない (datetime 軸では MultipleLocator / FixedLocator が未対応のため)。
    show_minor / show_grid を False (heatmap) にすると補助目盛・目盛線 (grid)
    のチェックを出さない (カテゴリ軸には不要なため)。
    rotation_panel (heatmap 専用): panel dict を渡すと x/y 目盛りラベルの
    回転 widget を expander 内に描き、panel["xtick_rotation"] /
    panel["ytick_rotation"] に書き込む (panel 直下のキーなので axis とは
    別に受け取る)。
    """
    with st.expander(t("目盛"), expanded=False):
        axis["tick_fontsize"] = fontsize_input(
            t("目盛り文字サイズ"), f"tickfs_{mode_key}",
            default=10, show_auto_hint=False)
        c1, c2 = st.columns(2)
        axis["show_x_ticklabels"] = c1.checkbox(
            t("x軸の目盛文字を表示"), value=True, key=f"showxt_{mode_key}")
        axis["show_y_ticklabels"] = c2.checkbox(
            t("y軸の目盛文字を表示"), value=True, key=f"showyt_{mode_key}")
        if rotation_panel is None:
            # heatmap はカテゴリ軸用の回転 (rotation_panel) を別に持つので出さない
            cr1, cr2 = st.columns(2)
            axis["x_tick_rotation"] = float(cr1.number_input(
                t("x目盛文字の回転 (度)"), -90.0, 90.0, 0.0, 5.0,
                key=f"xtrot_{mode_key}"))
            axis["y_tick_rotation"] = float(cr2.number_input(
                t("y目盛文字の回転 (度)"), -90.0, 90.0, 0.0, 5.0,
                key=f"ytrot_{mode_key}"))
        # 目盛位置: 自動 / 等間隔 / 位置を直接指定
        axis["x_tick_interval"] = None
        axis["x_tick_positions"] = None
        axis["y_tick_interval"] = None
        axis["y_tick_positions"] = None
        if not x_is_time:
            _tick_mode_ui(
                axis, mode_key, "x",
                mode_label=t("x軸の目盛位置"),
                interval_label=t("x軸の目盛間隔 (0 で非表示)"),
                positions_label=t("x軸の目盛位置 (カンマ区切り)"),
                default_positions="0, 60, 120, 180, 240, 300",
                positions_help=t("例: 0, 60, 120, 180. 不等間隔可。空欄/解析失敗で自動"),
                labels_label=t("x軸の目盛ラベル (位置と同数、カンマ区切り)"),
                labels_help=t("位置の数と一致しないと無視される。"
                            "例: 0°, 90°E, 180°, 90°W"))
        else:
            st.caption(t("x軸は時間軸のため、目盛位置の手動指定は未対応です"))
        if rotation_panel is not None:
            rotation_panel["xtick_rotation"] = float(st.number_input(
                t("x 目盛りラベルの回転 (度)"), -90.0, 90.0, 0.0, 5.0,
                key=f"hm_xrot_{mode_key}",
                help=t("カテゴリラベル (x軸の目盛位置が「自動」のとき) に適用される。"
                     "位置を手動指定した数値軸には効かない")))
        if not y_is_time:
            _tick_mode_ui(
                axis, mode_key, "y",
                mode_label=t("y軸の目盛位置"),
                interval_label=t("y軸の目盛間隔 (0 で非表示)"),
                positions_label=t("y軸の目盛位置 (カンマ区切り)"),
                default_positions="0, 10, 20, 50",
                positions_help=t("例: 0, 1, 2, 5, 10. 不等間隔可"),
                labels_label=t("y軸の目盛ラベル (位置と同数、カンマ区切り)"),
                labels_help=t("位置の数と一致しないと無視される"))
        else:
            st.caption(t("y軸は時間軸のため、目盛位置の手動指定は未対応です"))
        if rotation_panel is not None:
            rotation_panel["ytick_rotation"] = float(st.number_input(
                t("y 目盛りラベルの回転 (度)"), -90.0, 90.0, 0.0, 5.0,
                key=f"hm_yrot_{mode_key}",
                help=t("カテゴリラベル (y軸の目盛位置が「自動」のとき) に適用される。"
                     "位置を手動指定した数値軸には効かない")))
        # 補助目盛 (minor ticks) — x/y 個別
        if show_minor:
            c_mx, c_my = st.columns(2)
            axis["show_x_minor_ticks"] = c_mx.checkbox(
                t("x軸に補助目盛を表示"), value=False, key=f"xminor_{mode_key}",
                help=t("AutoMinorLocator で主目盛の間に細かい目盛線を追加"))
            axis["show_y_minor_ticks"] = c_my.checkbox(
                t("y軸に補助目盛を表示"), value=False, key=f"yminor_{mode_key}")
        if st.checkbox(t("目盛の線の太さを指定"), value=False,
                        key=f"tickwuse_{mode_key}",
                        help=t("軸から突き出る目盛 (tick) の線の太さ (pt)。"
                             "off で matplotlib 既定 (主目盛 0.8 / 補助目盛 0.6)。"
                             "on にすると主目盛・補助目盛の両方が同じ指定値になり、"
                             "補助目盛だけ細いという既定の関係は失われる")):
            axis["tick_width"] = float(st.slider(
                t("目盛の線の太さ"), 0.1, 5.0, 0.8, 0.1, key=f"tickw_{mode_key}"))
        # 目盛線 (grid) — x/y 個別 ON/OFF + 共通色・太さ・線種
        if show_grid:
            grid = axis["grid"]
            c_gx, c_gy = st.columns(2)
            grid["show_x"] = c_gx.checkbox(
                t("x軸に目盛線を表示"), value=False, key=f"gridx_{mode_key}")
            grid["show_y"] = c_gy.checkbox(
                t("y軸に目盛線を表示"), value=False, key=f"gridy_{mode_key}")
            # 第2軸の目盛線 (grid.show_y2、1次元プロットの「第2軸 (右)」で
            # 先に決まる) だけ ON のときも色・太さ・線種を出す
            if grid["show_x"] or grid["show_y"] or grid.get("show_y2"):
                grid["color"] = color_selector(
                    t("目盛線の色"), "#808080", key=f"gridc_{mode_key}",
                    meta_store=grid, meta_key="color")
                grid["width"] = float(st.slider(
                    t("目盛線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"gridw_{mode_key}"))
                grid["linestyle"] = st.selectbox(
                    t("目盛線の線種"), list(GL_LINESTYLE_LABELS),
                    format_func=tr_labels(GL_LINESTYLE_LABELS).get,
                    key=f"gridls_{mode_key}")
    return axis


def frame_background_ui(mode_key, x_time=None, y_time=None, *,
                        frame_width_only=False):
    """「図枠・背景」セクション (1次元・2次元プロット・断面図共通)。
    expander は「図枠」「背景」「直線」の 3 つ。

    (frame, background) の dict ペアを返す (config.default_frame /
    default_axes_background のスキーマ。基準線は background["reflines"])。

    x_time / y_time: その軸が時間軸のとき (dim名, 時刻ラベル list,
    ISO 文字列 list) を渡す。該当方向の塗り範囲を数値入力の代わりに
    時刻スライダーで選ばせ、lo/hi に ISO 文字列を入れる。
    frame_width_only=True (断面図) のとき「図枠」は枠線の太さのみ指定できる。
    """
    section_header(t("図枠・背景"))
    frame = mc_config.default_frame()
    background = mc_config.default_axes_background()
    with st.expander(t("図枠"), expanded=False):
        if not frame_width_only:
            st.caption(t("表示する枠線"))
            c1, c2, c3, c4 = st.columns(4)
            frame["show_top"] = c1.checkbox(t("上"), value=True,
                                            key=f"frame_top_{mode_key}")
            frame["show_bottom"] = c2.checkbox(t("下"), value=True,
                                               key=f"frame_bottom_{mode_key}")
            frame["show_left"] = c3.checkbox(t("左"), value=True,
                                             key=f"frame_left_{mode_key}")
            frame["show_right"] = c4.checkbox(t("右"), value=True,
                                              key=f"frame_right_{mode_key}")
        if st.checkbox(t("枠線の太さを指定"), value=False,
                        key=f"frame_wuse_{mode_key}",
                        help=t("off で matplotlib 既定 (おおむね 0.8)")):
            frame["width"] = float(st.slider(
                t("枠線の太さ"), 0.1, 5.0, 0.8, 0.1, key=f"frame_w_{mode_key}"))
        if not frame_width_only:
            if st.checkbox(t("枠線の色を指定"), value=False,
                            key=f"frame_cuse_{mode_key}",
                            help=t("off で matplotlib 既定 (黒)")):
                frame["color"] = color_selector(
                    t("枠線の色"), "#000000", key=f"frame_c_{mode_key}",
                    meta_store=frame, meta_key="color")
    with st.expander(t("背景"), expanded=False):
        if st.checkbox(t("背景色を指定"), value=False, key=f"bg_cuse_{mode_key}",
                        help=t("off で matplotlib 既定 (白)")):
            background["color"] = color_selector(
                t("背景色"), "#ffffff", key=f"bg_c_{mode_key}",
                meta_store=background, meta_key="color")
        # 範囲を指定した塗り (axvspan / axhspan)。追加・削除できるリスト
        state_key = f"bgspans_{mode_key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = []
            st.session_state[f"{state_key}_next"] = 0
        items = st.session_state[state_key]
        spans_cfg = []
        for i, item in enumerate(list(items)):
            sid = item["id"]
            st.markdown(t("**塗り範囲{n}**", n=i + 1))
            ori = st.radio(
                t("塗る方向"), list(_SPAN_ORI_LABELS),
                format_func=tr_labels(_SPAN_ORI_LABELS).get, horizontal=True,
                key=f"bgspan_ori_{mode_key}_{sid}",
                help=t("x軸の範囲: 指定した x の区間を上下いっぱいに塗る (axvspan)。"
                     "y軸の範囲: 指定した y の区間を左右いっぱいに塗る (axhspan)"))
            axis_time = x_time if ori == "x" else y_time
            if axis_time is not None:
                # 時間軸: 時刻ラベルのスライダーで範囲を選び ISO 文字列を格納
                dim, tlabels, tvalues = axis_time
                trange_key = (f"bgspan_trange_{mode_key}_{sid}_{dim}"
                              if ori == "x"
                              else f"bgspan_trangey_{mode_key}_{sid}_{dim}")
                lo_lbl, hi_lbl = st.select_slider(
                    t("範囲 (時刻)"), options=tlabels,
                    value=(tlabels[0], tlabels[-1]), key=trange_key)
                lo = tvalues[tlabels.index(lo_lbl)]
                hi = tvalues[tlabels.index(hi_lbl)]
            else:
                c1, c2 = st.columns(2)
                lo = float(c1.number_input(
                    t("範囲 最小"), value=0.0, step=1.0, format="%g",
                    key=f"bgspan_lo_{mode_key}_{sid}",
                    help=t("軸の値 (数値) で指定")))
                hi = float(c2.number_input(
                    t("範囲 最大"), value=1.0, step=1.0, format="%g",
                    key=f"bgspan_hi_{mode_key}_{sid}"))
            span_meta = {}
            color = color_selector(t("塗り色"), "#ffd7d7",
                                    key=f"bgspan_c_{mode_key}_{sid}",
                                    meta_store=span_meta, meta_key="color")
            alpha = st.slider(t("透過度 (alpha)"), 0.0, 1.0, 0.3, 0.05,
                               key=f"bgspan_a_{mode_key}_{sid}",
                               help=t("0 で完全に透明、1 で不透明"))
            span_dict = {"orientation": ori, "lo": lo, "hi": hi,
                         "color": color, "alpha": float(alpha)}
            if "_color_labels" in span_meta:
                span_dict["_color_labels"] = span_meta["_color_labels"]
            spans_cfg.append(span_dict)
            if st.button(t("この塗り範囲を削除"), key=f"bgspan_del_{mode_key}_{sid}"):
                items.remove(item)
                st.rerun()
            st.divider()
        if st.button(t("塗り範囲を追加"), key=f"bgspan_add_{mode_key}"):
            items.append({"id": st.session_state[f"{state_key}_next"]})
            st.session_state[f"{state_key}_next"] += 1
            st.rerun()
        background["spans"] = spans_cfg
    # 基準線 (axvline / axhline)。塗り範囲と同じ「追加・削除できるリスト」で
    # 1 本ずつ体裁・凡例ラベル・端の文字を指定する (config.default_refline)
    with st.expander(t("直線"), expanded=False):
        st.caption(t("x 一定の縦線 / y 一定の横線を引く (値は軸の単位)。"
                     "線はデータの上に描かれる"))
        state_key = f"reflines_{mode_key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = []
            st.session_state[f"{state_key}_next"] = 0
        items = st.session_state[state_key]
        reflines_cfg = []
        for i, item in enumerate(list(items)):
            sid = item["id"]
            st.markdown(t("**直線{n}**", n=i + 1))
            refline = mc_config.default_refline()
            ori = st.radio(
                t("向き"), list(_REFLINE_ORI_LABELS),
                format_func=tr_labels(_REFLINE_ORI_LABELS).get, horizontal=True,
                key=f"refline_ori_{mode_key}_{sid}")
            refline["orientation"] = ori
            axis_time = x_time if ori == "x" else y_time
            if axis_time is not None:
                # 時間軸: 時刻ラベルから選び ISO 文字列を格納 (塗り範囲と同じ規約)
                dim, tlabels, tvalues = axis_time
                tkey = (f"refline_t_{mode_key}_{sid}_{dim}" if ori == "x"
                        else f"refline_ty_{mode_key}_{sid}_{dim}")
                sel = st.selectbox(t("時刻"), tlabels, key=tkey)
                refline["value"] = tvalues[tlabels.index(sel)]
            else:
                refline["value"] = float(st.number_input(
                    t("値"), value=0.0, step=1.0, format="%g",
                    key=f"refline_v_{mode_key}_{sid}",
                    help=t("軸の値 (数値) で指定")))
            refline["color"] = color_selector(
                t("線の色"), "#000000", key=f"refline_c_{mode_key}_{sid}",
                meta_store=refline, meta_key="color")
            c1, c2 = st.columns(2)
            refline["linewidth"] = float(c1.slider(
                t("線の太さ"), 0.1, 4.0, 1.0, 0.1,
                key=f"refline_lw_{mode_key}_{sid}"))
            refline["alpha"] = float(c2.slider(
                t("透過度 (alpha)"), 0.0, 1.0, 1.0, 0.05,
                key=f"refline_a_{mode_key}_{sid}"))
            refline["linestyle"] = st.selectbox(
                t("線種"), list(LINESTYLE_LABELS), index=1,
                format_func=tr_labels(LINESTYLE_LABELS).get,
                key=f"refline_ls_{mode_key}_{sid}")
            refline["label"] = st.text_input(
                t("凡例ラベル"), value="", key=f"refline_label_{mode_key}_{sid}",
                placeholder=t("(空欄なら凡例に出さない)")) or None
            refline["text"] = st.text_input(
                t("線の端の文字"), value="", key=f"refline_text_{mode_key}_{sid}",
                placeholder=t("(なし)"),
                help=t("横線は右端の上、縦線は上端の左に添える")) or None
            if refline["text"]:
                refline["text_fontsize"] = fontsize_input(
                    t("文字サイズ"), f"refline_tfs_{mode_key}_{sid}")
                if st.checkbox(t("文字の色を指定"), value=False,
                                key=f"refline_tcuse_{mode_key}_{sid}",
                                help=t("off で線と同じ色")):
                    refline["text_color"] = color_selector(
                        t("文字の色"), "#000000",
                        key=f"refline_tc_{mode_key}_{sid}",
                        meta_store=refline, meta_key="text_color")
            reflines_cfg.append(refline)
            if st.button(t("この直線を削除"), key=f"refline_del_{mode_key}_{sid}"):
                items.remove(item)
                st.rerun()
            st.divider()
        if st.button(t("直線を追加"), key=f"refline_add_{mode_key}"):
            items.append({"id": st.session_state[f"{state_key}_next"]})
            st.session_state[f"{state_key}_next"] += 1
            st.rerun()
        background["reflines"] = reflines_cfg
    return frame, background


def legend_settings_ui(mode_key):
    """「凡例」expander (「軸・ラベル・凡例」セクション内、1次元・2次元プロット共通)。"""
    legend = {"show": True, "loc": "best", "fontsize": None, "x": None, "y": None}
    with st.expander(t("凡例"), expanded=False):
        legend["show"] = st.checkbox(t("凡例を表示"), value=True,
                                     key=f"legend_show_{mode_key}")
        if legend["show"]:
            loc_options = ["best", "upper left", "upper right", "lower left",
                           "lower right", "center", "upper center", "lower center"]
            legend["loc"] = st.selectbox(t("位置"), loc_options,
                                         key=f"legend_loc_{mode_key}")
            if st.checkbox(t("位置を座標で指定 (枠外も可)"), value=False,
                            key=f"legend_xy_{mode_key}",
                            help=t("axes 座標 (プロット枠の左下 0,0〜右上 1,1) で"
                                 "**凡例の左下角**の位置を直接指定する。1 を超える値・"
                                 "負の値で枠の外に出せる。指定中は上の「位置」"
                                 "プリセットは使われない")):
                c1, c2 = st.columns(2)
                legend["x"] = float(c1.number_input(
                    t("x (axes 座標, 0=左 1=右)"), -1.0, 2.0, 1.02, 0.01,
                    key=f"legend_x_{mode_key}"))
                legend["y"] = float(c2.number_input(
                    t("y (axes 座標, 0=下 1=上)"), -1.0, 2.0, 0.0, 0.01,
                    key=f"legend_y_{mode_key}"))
                st.caption(t("例: x=1.02, y=0.0 → 枠の右外側 (下揃え) / "
                           "x=0.0, y=-0.25 → 枠の下外側 (左揃え)"))
            legend["fontsize"] = fontsize_input(t("文字サイズ"), f"legend_fs_{mode_key}",
                                                 default=10, show_auto_hint=False)
    return legend
