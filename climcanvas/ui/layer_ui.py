# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""レイヤーごとの設定 UI。layer.kind ごとの *_layer_ui 関数と LAYER_UI レジストリ。"""

import numpy as np
import streamlit as st

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render
from climcanvas.ui.constants import (HATCH_PATTERNS, LINESTYLE_LABELS,
                                     LINE_LAYER_LINESTYLE_LABELS,
                                     PRESSURE_UNITS)
from climcanvas.ui.i18n import t, tr_labels
from climcanvas.ui.widgets import (cmap_selector, color_selector,
                                   dim_choice_label, dim_choices,
                                   fontsize_input, layer_preview_data,
                                   parse_float_list, time_labels_and_values,
                                   maskout_ui, transformed_range,
                                   value_transform_ui, var_label)

# --- 選択肢の表示ラベル (中立キー → 日本語) ---
# session_state / WIP に入るのは中立キー側。表示だけ format_func で変換する
# (i18n 第1段階。docs/i18n_plan.md)
_AVGMODE_LABELS = {"fixed": "固定値", "range": "範囲平均"}
# ライン (束) の統計線 (config.BUNDLE_STAT_KINDS と同じ中立キー)
_BUNDLE_STAT_LABELS = {"mean": "平均", "median": "中央値",
                       "min": "最小", "max": "最大",
                       "minmax": "最小と最大 (2本)",
                       "percentile": "パーセンタイル (1本)",
                       "pct_range": "パーセンタイル範囲 (2本)",
                       "std_range": "平均 ± 標準偏差 (2本)"}
# 2 本組の統計線の描き方 (config.default_bundle_summary の draw)
_BUNDLE_DRAW_LABELS = {"lines": "線 (2本)", "band": "帯 (塗りつぶし)",
                       "band_lines": "帯 + 縁の線"}
_LEVEL_MODE_LABELS = {"count": "レベル数 (等間隔)",
                      "explicit": "レベルを直接指定"}
_FILL_METHOD_LABELS = {"contourf": "contourf (滑らか)",
                       "pcolormesh": "pcolormesh (格子・離散)"}
_CMODE_LABELS = {"single": "単色", "cmap": "カラーマップ"}
_CMODE_MAG_LABELS = {"single": "単色", "cmap": "カラーマップ (大きさ)"}
_FB_MODE_LABELS = {"variable": "変数で指定", "constant": "定数 (ベースライン)"}
_ERR_SRC_LABELS = {"variable": "変数で指定", "constant": "定数"}
_BAR_ORIENT_LABELS = {"vertical": "縦棒 (ax.bar)", "horizontal": "横棒 (ax.barh)"}
_DENSITY_LABELS = {"count": "度数", "density": "確率密度 (density)"}
_WHIS_MODE_LABELS = {"iqr": "IQR 倍率", "percentile": "パーセンタイル範囲"}
_BW_MODE_LABELS = {"scott": "自動 (scott)", "silverman": "silverman",
                   "value": "数値指定"}
_KEYLPOS_LABELS = {"E": "矢印の右", "W": "矢印の左"}
_SIDE_LABELS = {"both": "両側", "low": "左半分", "high": "右半分"}
_TRACK_SELECT_LABELS = {"by_time": "発生年・月で選ぶ",
                        "by_index": "番号 (index) で選ぶ",
                        "all": "全トラック"}
# 負の線種の「正と同じ」/ 変数選択の「(なし)」の中立キー (番兵)
_SAME_AS_POSITIVE = "same"
_NONE_OPTION = "(none)"


def _none_option_labels(options):
    """「(なし)」番兵つき変数選択肢の表示辞書 (format_func 用。run 中に呼ぶこと)。"""
    none_label = t("(なし)")
    return {v: (none_label if v == _NONE_OPTION else v) for v in options}

# --- レイヤーごとの設定UI ---

def _layer_dataset_selector(datasets, key_prefix, label=None):
    """レイヤーが参照するデータセットを選ぶ。1つしか無ければ自動でそれを返す。

    保存済み (session_state) の dsid が現存していない場合は最初の dataset にフォールバックする。
    label は selectbox の表示名 (既定「ファイル」。ベクトル系は「x成分のファイル」)。
    """
    dsids = list(datasets.keys())
    ds_key = f"{key_prefix}_ds"
    if len(dsids) <= 1:
        return dsids[0]
    if st.session_state.get(ds_key) not in dsids:
        st.session_state[ds_key] = dsids[0]
    return st.selectbox(label or t("ファイル"), dsids, key=ds_key)


def _second_dataset_selector(datasets, primary, key_prefix):
    """ベクトル系レイヤーの y 成分を持つファイルを選ぶ (key {key_prefix}_vds)。

    先頭の選択肢 "" は「x成分と同じファイル」で、x 成分側のファイル切替に追従する
    (明示的に同じ id を選んだ場合とは区別しない)。1 ファイルしか無ければ widget を
    出さずに primary を返す。保存済みの id が現存しなければ "" に戻す。
    """
    dsids = list(datasets.keys())
    if len(dsids) <= 1:
        return primary
    key = f"{key_prefix}_vds"
    options = [""] + dsids
    if st.session_state.get(key) not in options:
        st.session_state[key] = ""
    chosen = st.selectbox(
        t("y成分のファイル"), options, key=key,
        format_func=lambda d: d or t("x成分と同じファイル"),
        help=t("x成分と y成分が別々のファイルに入っているときに y成分側の"
             "ファイルを選ぶ。両成分は同じ格子 (同じ次元名・同じ形状) であること"))
    return chosen or primary


def _vector_components_ui(datasets, variables_by_ds, short, lid, few_vars_warning):
    """ベクトル系レイヤー (vector / stream) の成分選択 UI (ファイル + 変数 × 2 成分)。

    x成分のファイル ({short}_{lid}_ds) → x成分の変数 ({short}_u_{lid}) →
    y成分のファイル ({short}_{lid}_vds、複数ファイル時のみ。"" = x と同じ) →
    y成分の変数 ({short}_v_{lid}) の順。y 成分を別ファイルから読めるようにした
    (2026-09-10)。描ける変数が足りなければ警告して None を返す。
    few_vars_warning(dsid) は「同じファイルに変数が2つ未満」の警告文
    (vector / stream で文言が違う)。
    返り値: (dsid, ds, u_var, v_dsid, v_ds, v_var)。v_dsid は x成分と同じ
    ファイルなら None (config の v_dataset_id にそのまま入れる)。
    """
    prefix = f"{short}_{lid}"
    multi = len(datasets) > 1
    dsid = _layer_dataset_selector(datasets, prefix,
                                   label=t("x成分のファイル") if multi else None)
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には描画可能な変数がありません", dsid=dsid))
        return None
    if not multi and len(variables) < 2:
        st.warning(few_vars_warning(dsid))
        return None
    _reset_invalid_var_key(f"{short}_u_{lid}", variables)
    u_idx = variables.index("u") if "u" in variables else 0
    u_var = st.selectbox(t("x成分の変数"), variables, index=u_idx,
                         key=f"{short}_u_{lid}",
                         format_func=lambda v: var_label(ds, v))
    v_dsid = _second_dataset_selector(datasets, dsid, prefix)
    v_ds = datasets[v_dsid]
    v_variables = variables_by_ds[v_dsid]
    if not v_variables:
        st.warning(t("`{dsid}` には描画可能な変数がありません", dsid=v_dsid))
        return None
    if v_dsid == dsid and len(v_variables) < 2:
        st.warning(few_vars_warning(dsid))
        return None
    _reset_invalid_var_key(f"{short}_v_{lid}", v_variables)
    if "v" in v_variables:
        v_idx = v_variables.index("v")
    elif v_dsid == dsid:
        v_idx = min(1, len(v_variables) - 1)   # x と同じ変数を避ける
    else:
        v_idx = 0
    v_var = st.selectbox(t("y成分の変数"), v_variables, index=v_idx,
                         key=f"{short}_v_{lid}",
                         format_func=lambda v: var_label(v_ds, v))
    return dsid, ds, u_var, (v_dsid if v_dsid != dsid else None), v_ds, v_var


def _preview_magnitude(panel, layer, datasets, keep_dims):
    """cmap 範囲の既定値用に |V| = √(u²+v²) の選択中断面を返す。

    別ファイルの成分で格子が違うと hypot が broadcast できないので、その場合は
    |u| で代用する (描画側は render.check_vector_shapes が原因を示して止める)。
    """
    au = np.asarray(layer_preview_data(
        panel, layer, datasets, keep_dims,
        variable=layer["u_variable"]).values, dtype=float)
    av = np.asarray(layer_preview_data(
        panel, layer, datasets, keep_dims, variable=layer["v_variable"],
        dataset_id=mc_render.vector_v_dataset_id(layer)).values, dtype=float)
    return np.hypot(au, av) if au.shape == av.shape else np.abs(au)


def _reset_invalid_var_key(var_key: str, variables: list[str]) -> None:
    """セッションに残っている変数選択が現在の variables に無ければ先頭にリセット。

    データセット切替直後にゴミ値が残ると selectbox が壊れるのを防ぐ。
    """
    if not variables:
        return
    if var_key in st.session_state and st.session_state[var_key] not in variables:
        st.session_state[var_key] = variables[0]


def _dataset_roles(ds, roles):
    """レイヤーが参照する dataset 自身の座標役割 (lat / lon / vertical) を返す。

    app.py の roles は先頭ファイル (ds0) から判定したグローバル値で、ds0 に無い
    役割 (経度・鉛直など) は None になる。レイヤーの次元固定 UI はそのレイヤーの
    dataset を扱うので、経度の wrap-around 入力・鉛直の固定のみ・緯度の重み付きの
    判定はその dataset から役割を取り直す (経度を持たないファイルを先に読むと
    全レイヤーの経度がスライダーになる読み込み順依存の修正、2026-09-10)。
    時刻役割はパネル側 (selection_widgets、ds0 基準) と対にするためグローバル値の
    まま (ds0 に時刻が無いときはレイヤー側で固定値として選べる従来動作を保つ)。
    """
    ds_roles = mc_dataset.detect_coord_roles(ds)
    return {**ds_roles, "time": roles.get("time")}


def _layer_keep_dims(ds, keep_dims, panel):
    """水平面図で、レイヤーの dataset 自身の水平 dim を keep_dims に合成して返す。

    水平面図の keep_dims は app.py が先頭ファイル (ds0) から決める — 1 次元格子なら
    (lat, lon)、curvilinear 格子なら lat/lon 座標の dims (y, x)。別ファイルのレイヤーで
    1 次元格子と curvilinear が混在すると dim 名が食い違い、水平面の dim が「固定する
    次元」の selectbox に出てしまう。panel に x_dim が無い (= 水平面図) ときだけ、
    その dataset の horizontal_dims を足す。断面図・1 次元プロットでは keep_dims は
    描画軸そのもので lat/lon は固定対象なので触らない。1 次元格子どうしでは
    (dim 名整列後) 集合が変わらず、従来と同じ tuple を返す。
    """
    if panel is None or "x_dim" in panel:
        return keep_dims
    hd = mc_dataset.horizontal_dims(ds, mc_dataset.detect_coord_roles(ds))
    if not hd or all(d in keep_dims for d in hd):
        return keep_dims
    return tuple(dict.fromkeys((*keep_dims, *hd)))


def _default_dim_index(ds, dim, roles, choices):
    """dim を固定値で選ぶ selectbox の初期値インデックス。

    鉛直なら最下層 (気圧・positive:down なら最大値、高度などは最小値)。
    緯度なら 0°、経度なら 180° に最も近い格子点を選ぶ。
    その他の dim と非数値座標 (文字列等) はリストの先頭 (index 0)。
    """
    if not choices or not isinstance(choices[0], float):
        return 0
    if dim == roles.get("vertical"):
        attrs = ds[dim].attrs
        pressure_like = (str(attrs.get("units", "")).lower() in PRESSURE_UNITS
                         or attrs.get("positive") == "down")
        return choices.index(max(choices) if pressure_like else min(choices))
    target = None
    if dim == roles.get("lat"):
        target = 0.0
    elif dim == roles.get("lon"):
        target = 180.0
    if target is None:
        return 0
    return min(range(len(choices)), key=lambda i: abs(choices[i] - target))


def _layer_dim_picker(ds, var_dims, roles, keep_dims, lid_prefix):
    """変数の次元のうち keep_dims にも時刻にも入らないものを selectbox で固定する。

    水平断面図でレイヤー毎に鉛直レベルを選ばせる用途。{dim: value} を返す。
    時刻次元は panel.selection 側で扱うのでここでは無視する。
    """
    roles = _dataset_roles(ds, roles)
    selection = {}
    for dim in var_dims:
        if dim in keep_dims or dim == roles.get("time") or dim not in ds.coords:
            continue
        units = ds[dim].attrs.get("units", "")
        choices = dim_choices(ds[dim].values)
        selection[dim] = st.selectbox(
            f"{dim}" + (f" [{units}]" if units else ""), choices,
            index=_default_dim_index(ds, dim, roles, choices),
            format_func=dim_choice_label,
            key=f"{lid_prefix}_sel_{dim}")
    return selection


def _layer_dim_picker_with_avg(ds, var_dims, roles, keep_dims, lid_prefix):
    """_layer_dim_picker の範囲平均対応版。

    各 dim について「固定値」か「範囲平均」かをトグルで選ばせる。
    鉛直 (vertical) 次元は当面平均非対応で固定値のみ。
    緯度 dim のときだけ cos(lat) 重み付き平均オプションを出す。
    経度 dim のときだけ min/max を number_input で受け、データ範囲外の値や
    min > max (日付変更線をまたぐ wrap-around) も許す。
    返り値は (selection, averages) のタプル。
    """
    roles = _dataset_roles(ds, roles)
    selection: dict = {}
    averages: dict = {}
    lat_dim = roles.get("lat")
    lon_dim = roles.get("lon")
    vertical_dim = roles.get("vertical")
    for dim in var_dims:
        if dim in keep_dims or dim == roles.get("time") or dim not in ds.coords:
            continue
        units = ds[dim].attrs.get("units", "")
        label = f"{dim}" + (f" [{units}]" if units else "")
        values = dim_choices(ds[dim].values)
        # 鉛直 (高度・気圧) 方向と非数値座標 (文字列等) は範囲平均を提供しない
        # (固定値のみ)
        if dim == vertical_dim or (values and isinstance(values[0], str)):
            selection[dim] = st.selectbox(
                f"{label}", values,
                index=_default_dim_index(ds, dim, roles, values),
                format_func=dim_choice_label,
                key=f"{lid_prefix}_sel_{dim}")
            continue
        mode = st.radio(t("{label} の扱い", label=label), list(_AVGMODE_LABELS),
                         format_func=tr_labels(_AVGMODE_LABELS).get, horizontal=True,
                         key=f"{lid_prefix}_avgmode_{dim}",
                         help=t("範囲平均: xarray の .mean() のデフォルト挙動 "
                              "(skipna=True) に従い、NaN を除外して平均する。"
                              "例: 範囲内 10 点中 2 点が NaN なら、残り 8 点の和を "
                              "8 で割る。"))
        if mode == "fixed":
            selection[dim] = st.selectbox(
                t("値"), values,
                index=_default_dim_index(ds, dim, roles, values),
                format_func=lambda v: f"{v:g}",
                key=f"{lid_prefix}_sel_{dim}")
        else:
            if dim == lon_dim:
                data_lo, data_hi = float(min(values)), float(max(values))
                c1, c2 = st.columns(2)
                lo = c1.number_input(
                    "min", value=data_lo, step=1.0,
                    key=f"{lid_prefix}_avgrnglo_{dim}",
                    help=t("経度の下限。-30 や 330 のようにデータ規約外の値も入力可。"
                         "min > max のとき日付変更線をまたぐ範囲として扱う"
                         "(例: 330→30 で 30W〜30E)。"))
                hi = c2.number_input(
                    "max", value=data_hi, step=1.0,
                    key=f"{lid_prefix}_avgrnghi_{dim}",
                    help=t("経度の上限。詳細は min のヘルプ参照。"))
                st.caption(t(
                    "データの経度範囲は [{lo:g}, {hi:g}]。"
                    "min > max にすると日付変更線をまたぐ範囲として扱う。",
                    lo=data_lo, hi=data_hi))
            else:
                lo, hi = st.select_slider(
                    t("範囲"), options=values, value=(values[0], values[-1]),
                    format_func=lambda v: f"{v:g}",
                    key=f"{lid_prefix}_avgrng_{dim}")
            op = "mean"
            if dim == lat_dim:
                if st.checkbox(t("cos(lat) 重み付き"), value=False,
                                key=f"{lid_prefix}_wght_{dim}",
                                help=t("緯度方向の面積要素 cos(緯度) で重み付き平均する "
                                     "(xarray の .weighted().mean() のデフォルト挙動)。"
                                     "NaN 点はデータ・重み双方から除外され、"
                                     "有効点だけで重み付き平均が計算される。")):
                    op = "weighted_mean"
            averages[dim] = {"op": op, "range": [float(lo), float(hi)]}
    return selection, averages


# --- カラーマップまわりの共通部品 (全レイヤー種別で UI の並び・文言を揃える) ---

def colorbar_ui(cb, p, default_label="", label_suffix=""):
    """カラーバー expander の共通 UI (フル装備、全レイヤー種別で同一)。

    p は widget key の接尾辞 (fill は lid をそのまま渡す = 既存キー互換)。
    label_suffix に変数名 (f"_{var}") を渡すと、変数を切り替えたときに
    ラベル初期値が新しい変数で入り直す (key が変わり新しい widget になる)。
    """
    with st.expander(t("カラーバー"), expanded=False):
        cb["show"] = st.checkbox(t("カラーバーを表示"), value=True, key=f"cb_{p}")
        if not cb["show"]:
            return
        cb["label"] = st.text_input(t("カラーバーラベル"), value=default_label,
                                    key=f"cbl_{p}{label_suffix}") or None
        cb["location"] = st.selectbox(t("位置"), ["right", "bottom", "left", "top"],
                                      key=f"cbloc_{p}")
        cb["flip_ticks"] = st.checkbox(
            t("目盛りを反対側に表示"), value=False, key=f"cbflip_{p}",
            help=t("目盛り線・目盛り文字・ラベルをカラーバーの反対側 "
                 "(プロット側) に出す"))
        cb["label_opposite"] = st.checkbox(
            t("ラベルを目盛りと反対側に表示"), value=False,
            key=f"cbflip_lab_{p}",
            help=t("カラーバーのラベルだけを目盛り文字と逆のサイドに出す (例: 右配置のカラーバーで目盛りは左・ラベルは右、またはその逆)"))
        cb["shrink"] = float(st.slider(
            t("長さ (shrink)"), 0.3, 1.0, 1.0, 0.05,
            help=t("1.0 でプロット領域 (地図の描画範囲) の長さと同じ"),
            key=f"cbshrink_{p}"))
        cb["aspect"] = float(st.slider(
            t("厚み (aspect)"), 5.0, 50.0, 20.0, 1.0,
            help=t("大きいほど細い (matplotlib デフォルト 20)"),
            key=f"cbaspect_{p}"))
        if not st.checkbox(t("枠線の太さを自動"), value=True, key=f"cbolw_auto_{p}"):
            cb["outline_width"] = float(st.slider(
                t("枠線の太さ"), 0.0, 3.0, 0.8, 0.1, key=f"cbolw_{p}",
                help=t("0 で枠線なし (matplotlib 既定 0.8)")))
        if not st.checkbox(t("目盛り線の太さを自動"), value=True,
                            key=f"cbtw_auto_{p}"):
            cb["tick_width"] = float(st.slider(
                t("目盛り線の太さ"), 0.0, 3.0, 0.8, 0.1, key=f"cbtw_{p}",
                help=t("カラーバーの目盛り線 (tick) の太さ。0 で非表示 "
                     "(matplotlib 既定 0.8)")))
        if not st.checkbox(t("図との間隔を自動"), value=True, key=f"cbpad_auto_{p}"):
            # matplotlib のデフォルト: 縦配置 0.05 / 横配置 0.15
            default_pad = 0.15 if cb["location"] in ("bottom", "top") else 0.05
            cb["pad"] = float(st.slider(t("図との間隔 (pad)"), 0.0, 0.5, default_pad,
                                        0.01, key=f"cbpad_{p}"))
        cb["label_fontsize"] = fontsize_input(t("ラベル文字サイズ"), f"cblfs_{p}",
                                              default=10, show_auto_hint=False)
        if not st.checkbox(t("ラベルの距離を自動"), value=True,
                            key=f"cblpad_auto_{p}"):
            cb["label_pad"] = float(st.number_input(
                t("ラベルの距離 (pt)"), -50.0, 50.0, 4.0, 1.0,
                key=f"cblpad_{p}",
                help=t("カラーバーとラベル文字の距離 (matplotlib 既定 ~4)。"
                     "負の値で近づく")))
        cb["tick_fontsize"] = fontsize_input(t("目盛り文字サイズ"), f"cbtfs_{p}",
                                             default=10, show_auto_hint=False)
        if not st.checkbox(t("目盛り文字の距離を自動"), value=True,
                            key=f"cbtpad_auto_{p}"):
            cb["tick_pad"] = float(st.number_input(
                t("目盛り文字の距離 (pt)"), -50.0, 50.0, 3.5, 0.5,
                key=f"cbtpad_{p}",
                help=t("カラーバーと目盛り文字の距離 (matplotlib 既定 3.5)。"
                     "負の値で近づく")))


def cmap_section_ui(s, p, preview_fn, *, default_levels="0, 5, 10, 15, 20",
                    log_exclusive_key=None):
    """cmap 色付けの共通 UI (vector 式の標準形)。

    並び: カラーマップ (分類→名前) → 反転 → ☑ 値の範囲を自動 (OFF で
    最小値・最大値、既定は preview_fn() の (min, max)) → ☑ 色を離散化する
    (レベル数 (等間隔) / レベルを直接指定) → extend (neither 既定)。
    s の cmap / reverse_cmap / vmin / vmax / levels / extend に書き込む。

    preview_fn は既定範囲 (dmin, dmax) を返す遅延評価関数 (範囲を手動指定
    したときだけ呼ばれる)。値は「値の変換」適用後の単位で返すこと
    (transformed_range() を通す — vmin/vmax は変換後の値で指定する仕様)。
    log_exclusive_key は対数スケール checkbox の
    session_state キー。on のときは離散化 UI を出さない
    (LogNorm と BoundaryNorm は norm の枠を取り合うため排他)。
    """
    s["cmap"] = cmap_selector(p)
    s["reverse_cmap"] = st.checkbox(t("カラーマップを反転"), key=f"rev_{p}")
    if not st.checkbox(t("値の範囲を自動"), value=True, key=f"autorange_{p}"):
        dmin, dmax = preview_fn()
        c1, c2 = st.columns(2)
        s["vmin"] = float(c1.number_input(t("最小値"), value=float(dmin),
                                          format="%g", key=f"vmin_{p}"))
        s["vmax"] = float(c2.number_input(t("最大値"), value=float(dmax),
                                          format="%g", key=f"vmax_{p}"))
    if log_exclusive_key and st.session_state.get(log_exclusive_key):
        st.caption(t("色の離散化は対数スケール off のとき選べます"))
    elif st.checkbox(t("色を離散化する"), value=False, key=f"disc_{p}",
                      help=t("off で連続 (グラデーション)。on で段階化し、"
                           "カラーバーも段々になる")):
        _lv_mode = st.radio(
            t("レベルの指定方法"), list(_LEVEL_MODE_LABELS),
            format_func=tr_labels(_LEVEL_MODE_LABELS).get,
            horizontal=True, key=f"lvmode_{p}")
        if _lv_mode == "explicit":
            _lv_list = parse_float_list(st.text_input(
                t("レベル (カンマ区切り)"), value=default_levels,
                key=f"lvls_{p}",
                help=t("色の境界値。昇順に整列・重複除去される。2個以上必要。"
                     "不等間隔も可。指定時は「値の範囲」は使われない。"
                     "対数風の階級 (1, 2, 5, 10, 20, 50 …) もここで")))
            if _lv_list and len(set(_lv_list)) >= 2:
                s["levels"] = _lv_list
            else:
                st.warning(t("レベルは2個以上の数値をカンマ区切りで入力してください"))
        else:
            s["levels"] = int(st.number_input(
                t("カラーレベル数"), 3, 60, 11, key=f"nlev_{p}"))
        s["zero_white"] = st.checkbox(
            t("ゼロ近傍を白にする"), key=f"zerow_{p}",
            help=t("0 を含む色レベル帯を白にする (0 がレベル境界のときは両隣の2帯)。"
                 "正負の偏差を発散カラーマップで描くときに"))
    s["extend"] = st.selectbox(
        t("extend (範囲外の扱い)"), ["neither", "both", "min", "max"],
        key=f"ext_{p}",
        help=t("範囲外の値に割り当てる色をカラーバーの矢印で表す "
             "(neither = 矢印なし)"))


def fill_levels_ui(s, p, preview_fn, var_suffix="", default_nlev=21,
                   zero_white=False):
    """fill 式のレベル指定 UI (fill / contour 共用)。

    ☑ レベルを等間隔にする (既定 ON): ☑ 値の範囲を自動 (横に「データ範囲」)
    → (OFF で) 最小値・最大値 → カラーレベル数 (下に「描画レベル」)。
    OFF: レベル (カンマ区切り) の直接指定。
    s の levels / vmin / vmax に書き込む。preview_fn は既定範囲 (dmin, dmax)
    を返す関数で、「値の範囲を自動」OFF のときだけ呼ばれる (最小値・最大値の
    既定値)。値は「値の変換」適用後の単位で返すこと (transformed_range() を通す)。
    倍率・加算を後から変えても手入力済みの vmin/vmax は入れ直さない
    (widget 値はそのまま。2026-08-25 ユーザー判断)。
    var_suffix (f"_{var}") で変数切替時に範囲の初期値が
    入り直る (fill の既存キー互換のため vmin/vmax キーだけに付く)。
    default_nlev はカラーレベル数の既定値 (fill=21 / contour=11、
    default_*_layer の levels と揃える)。
    zero_white=True (fill のみ) で「ゼロ近傍を白にする」checkbox を出して
    s["zero_white"] に書き込む (contour は線が白=不可視になるため出さない)。

    戻り値は参考表示の placeholder {"range": st.empty, "levels": st.empty}
    (等間隔 OFF では None)。「データ範囲」「描画レベル」の文言は layer が
    完成した後 (値の変換・maskout の widget 描画後) に呼び出し側が
    _write_levels_captions() で埋める — render と同じ前処理を経た値を
    出すため (widget の並びは変えずに後から書ける st.empty を使う)。
    """
    slots = None
    if st.checkbox(t("レベルを等間隔にする"), value=True, key=f"fill_lveq_{p}",
                    help=t("off にすると色の境界値をカンマ区切りで直接指定できる "
                         "(不等間隔可)")):
        c_auto, c_rng = st.columns(2)
        auto_range = c_auto.checkbox(t("値の範囲を自動"), value=True,
                                     key=f"auto_{p}")
        slots = {"range": c_rng.empty(), "levels": None}
        if not auto_range:
            dmin, dmax = preview_fn()
            # 最大値を上に (気圧面の値等は「上 = 大きい」の直感に合わせる)
            s["vmax"] = float(st.number_input(t("最大値"), value=float(dmax),
                                              key=f"vmax_{p}{var_suffix}"))
            s["vmin"] = float(st.number_input(t("最小値"), value=float(dmin),
                                              key=f"vmin_{p}{var_suffix}"))
        s["levels"] = int(st.number_input(t("カラーレベル数"), 3, 60, default_nlev,
                                          key=f"nlev_{p}"))
        slots["levels"] = st.empty()
    else:
        _lv_list = parse_float_list(st.text_input(
            t("レベル (カンマ区切り)"), value="0, 3, 4, 6, 10",
            key=f"fill_lvls_{p}",
            help=t("色の境界値。昇順に整列・重複除去される。2個以上必要。"
                 "不等間隔も可。指定時は値の範囲 (最小値・最大値) は使われない")))
        if _lv_list and len(set(_lv_list)) >= 2:
            s["levels"] = _lv_list
        else:
            st.warning(t("レベルは2個以上の数値をカンマ区切りで入力してください"))
    if zero_white:
        s["zero_white"] = st.checkbox(
            t("ゼロ近傍を白にする"), key=f"zerow_{p}",
            help=t("0 を含む色レベル帯を白にする (0 がレベル境界のときは両隣の2帯)。"
                 "正負の偏差を発散カラーマップで描くときに"))
    return slots


def _write_levels_captions(slots, panel, layer, datasets, keep_dims, kind):
    """fill_levels_ui の placeholder に「データ範囲」と「描画レベル」を書く。

    layer が完成した後 (値の変換・maskout の widget 描画後) に呼ぶ。値は
    render が matplotlib に渡す直前の DataArray (layer_preview_data(drawn=True)
    = 選択中の時刻・レベル・領域・平均・値の変換・maskout 適用後) から作る。
    以前は先頭断面 (描画軸以外を先頭 index で固定) の範囲だったため、
    時刻 (lag) やレベルを切り替えても先頭断面の範囲が出ていた (2026-09-12)。
    描画レベルは「レベルを等間隔にする」のときだけ (render.preview_levels)。
    contour は線が引かれるレベル (データ範囲内のもの) だけを数え、fill は
    色の境界を全部数える。kind は "fill" / "contour"。
    """
    if not slots:
        return
    da = layer_preview_data(panel, layer, datasets, keep_dims, drawn=True)
    vals = np.asarray(da, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        slots["range"].caption(t("データ範囲: 値がありません (すべて欠損)"))
        return
    dmin, dmax = float(finite.min()), float(finite.max())
    slots["range"].caption(t("データ範囲: {lo:g} 〜 {hi:g}", lo=dmin, hi=dmax))
    if slots.get("levels") is None:
        return
    try:
        lv = mc_render.preview_levels(layer["style"], da, kind=kind)
    except Exception:
        # 設定途中で render 側も落ちる状態 (RenderError 等)。表示だけ諦める
        lv = None
    if lv is None or lv.size == 0:
        return
    step = float(lv[1] - lv[0]) if lv.size >= 2 else None
    uniform = step is not None and bool(np.allclose(np.diff(lv), step,
                                                    rtol=1e-6, atol=0.0))
    if kind == "contour":
        lv = lv[(lv >= dmin) & (lv <= dmax)]
        if lv.size == 0:
            slots["levels"].caption(t("等値線: データ範囲内にレベルがありません"))
            return
    lo, hi, n = float(lv[0]), float(lv[-1]), int(lv.size)
    if kind == "contour":
        msg = (t("等値線: {lo:g} 〜 {hi:g}、間隔 {step:g} ({n} 本)",
                 lo=lo, hi=hi, step=step, n=n) if uniform
               else t("等値線: {lo:g} 〜 {hi:g} ({n} 本)", lo=lo, hi=hi, n=n))
    else:
        msg = (t("色の境界: {lo:g} 〜 {hi:g}、間隔 {step:g} ({n} 個)",
                 lo=lo, hi=hi, step=step, n=n) if uniform
               else t("色の境界: {lo:g} 〜 {hi:g} ({n} 個)", lo=lo, hi=hi, n=n))
    slots["levels"].caption(msg)

def fill_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                  allow_averaging=False, panel=None):
    dsid = _layer_dataset_selector(datasets, f"fill_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には描画可能な変数がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"fill_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"fill_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_fill_layer(dsid, var)
    keep_dims = _layer_keep_dims(ds, keep_dims, panel)
    if roles is not None:
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[var].dims, roles, keep_dims, f"fill_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[var].dims, roles, keep_dims, f"fill_{lid}")
    s = layer["style"]
    s["method"] = st.radio(t("描画方法"), list(_FILL_METHOD_LABELS),
                            format_func=tr_labels(_FILL_METHOD_LABELS).get,
                            horizontal=True, key=f"fill_method_{lid}")
    s["cmap"] = cmap_selector(lid)
    s["reverse_cmap"] = st.checkbox(t("カラーマップを反転"), key=f"rev_{lid}")

    def _preview_range():
        pv = layer_preview_data(panel, layer, datasets, keep_dims)
        return transformed_range(f"fill_{lid}", float(pv.min()), float(pv.max()))

    slots = fill_levels_ui(s, lid, _preview_range, var_suffix=f"_{var}",
                           zero_white=True)
    s["extend"] = st.selectbox(t("extend (範囲外の扱い)"), ["both", "neither", "min", "max"],
                               key=f"ext_{lid}")
    s["alpha"] = float(st.slider(
        t("透明度 (alpha)"), 0.0, 1.0, 1.0, 0.05, key=f"fill_alpha_{lid}",
        help=t("1 で不透明。matplotlib の性質上、contourf では 1 未満にすると"
               "レベルの境に細い継ぎ目が見えることがあります (pcolormesh では"
               "起きません)。カラーバーにも同じ透過が掛かります")))
    # カラーバーラベルの既定は変数名 [units] (long_name は使わない)
    units = ds[var].attrs.get("units", "")
    default_label = f"{var} [{units}]" if units else str(var)
    colorbar_ui(s["colorbar"], lid, default_label, label_suffix=f"_{var}")
    value_transform_ui(s, f"fill_{lid}")
    maskout_ui(s, f"fill_{lid}", ds=ds, variable=var, required_dims=keep_dims)
    _write_levels_captions(slots, panel, layer, datasets, keep_dims, "fill")
    return layer


def contour_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                     allow_averaging=False, panel=None):
    dsid = _layer_dataset_selector(datasets, f"cont_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には描画可能な変数がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"cont_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"cont_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_contour_layer(dsid, var)
    keep_dims = _layer_keep_dims(ds, keep_dims, panel)
    if roles is not None:
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[var].dims, roles, keep_dims, f"cont_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[var].dims, roles, keep_dims, f"cont_{lid}")
    s = layer["style"]
    color_mode = st.radio(t("線の色付け"), list(_CMODE_LABELS),
                          format_func=tr_labels(_CMODE_LABELS).get, horizontal=True,
                          key=f"cont_cmode_{lid}")
    if color_mode == "cmap":
        s["use_cmap"] = True
        s["cmap"] = cmap_selector(lid)
        s["reverse_cmap"] = st.checkbox(t("カラーマップを反転"), key=f"cont_rev_{lid}")
    else:
        s["use_cmap"] = False
        s["color"] = color_selector(t("線の色"), "#000000", key=f"cont_color_{lid}",
                                     meta_store=s, meta_key="color")

    def _preview_range():
        pv = layer_preview_data(panel, layer, datasets, keep_dims)
        return transformed_range(f"cont_{lid}", float(pv.min()), float(pv.max()))

    slots = fill_levels_ui(s, f"cont_{lid}", _preview_range, var_suffix=f"_{var}",
                           default_nlev=11)
    if s["use_cmap"]:
        s["extend"] = st.selectbox(
            t("extend (範囲外の扱い)"), ["neither", "both", "min", "max"],
            key=f"ext_cont_{lid}",
            help=t("レベル範囲外の値に割り当てる色をカラーバーの矢印で表す"))
    s["linewidth"] = float(st.slider(t("線の太さ"), 0.2, 4.0, 1.0, 0.1, key=f"cont_lw_{lid}"))
    s["linestyle"] = st.selectbox(t("正の値の線種"), list(LINESTYLE_LABELS),
                                  format_func=tr_labels(LINESTYLE_LABELS).get,
                                  key=f"cont_ls_{lid}")
    neg_options = list(LINESTYLE_LABELS) + [_SAME_AS_POSITIVE]
    neg_choice = st.selectbox(t("負の値の線種"), neg_options,
                              index=neg_options.index("dashed"),
                              format_func={**tr_labels(LINESTYLE_LABELS),
                                           _SAME_AS_POSITIVE: t("正と同じ")}.get,
                              key=f"cont_ls_neg_{lid}")
    s["negative_linestyle"] = None if neg_choice == _SAME_AS_POSITIVE else neg_choice
    labels = s["labels"]
    labels["show"] = st.checkbox(t("等値線ラベルを表示"), value=True, key=f"cont_lab_{lid}")
    if labels["show"]:
        labels["fontsize"] = int(st.number_input(t("ラベル文字サイズ"), 4, 24, 8,
                                                 key=f"cont_labfs_{lid}"))
        labels["fmt"] = st.text_input(
            t("ラベル書式"), value="%g", key=f"cont_fmt_{lid}",
            help=t("printf 系フォーマット書式と同じ。例: %g (自動), %.0f (整数), "
                 "%.2f (小数2桁), %g hPa (単位付き)"))
    with st.expander(t("指定レベルの強調・非表示"), expanded=False):
        emp = {"levels": None, "linewidth": None, "color": None}
        if st.checkbox(t("特定のレベルを太く/色付け/非表示にする"), value=False,
                       key=f"cont_emp_on_{lid}",
                       help=t("0 線や基準線だけ目立たせる、または特定レベルを消す用途。"
                            "値変換適用後のレベル値をカンマ区切りで指定 "
                            "(描画レベルに含まれる値だけ効く)")):
            emp["levels"] = parse_float_list(st.text_input(
                t("対象レベル (カンマ区切り)"), value="0",
                key=f"cont_emp_lv_{lid}")) or None
            if st.checkbox(t("太さを変える"), value=True, key=f"cont_emp_lwon_{lid}"):
                emp["linewidth"] = float(st.slider(
                    t("対象レベルの線の太さ"), 0.0, 6.0, 2.5, 0.1,
                    key=f"cont_emp_lw_{lid}",
                    help=t("0 でその等値線を非表示にする (等値線ラベルも一緒に消える)")))
            if st.checkbox(t("色を変える"), value=False, key=f"cont_emp_colon_{lid}"):
                emp["color"] = color_selector(
                    t("強調線の色"), "#d00000", key=f"cont_emp_col_{lid}",
                    meta_store=emp, meta_key="color")
        s["emphasis"] = emp
    if s["use_cmap"]:
        units = ds[var].attrs.get("units", "")
        default_label = f"{var} [{units}]" if units else str(var)
        colorbar_ui(s["colorbar"], f"cont_{lid}", default_label,
                    label_suffix=f"_{var}")
    value_transform_ui(s, f"cont_{lid}")
    maskout_ui(s, f"cont_{lid}", ds=ds, variable=var, required_dims=keep_dims)
    _write_levels_captions(slots, panel, layer, datasets, keep_dims, "contour")
    return layer


def vector_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                    allow_averaging=False, panel=None):
    src = _vector_components_ui(
        datasets, variables_by_ds, "vec", lid,
        lambda dsid: t("`{dsid}` にはベクトル描画可能な変数が2つ未満です", dsid=dsid))
    if src is None:
        return None
    dsid, ds, u_var, v_dsid, v_ds, v_var = src
    layer = mc_config.default_vector_layer(dsid, u_var, v_var, v_dataset_id=v_dsid)
    keep_dims = _layer_keep_dims(ds, keep_dims, panel)
    if roles is not None:
        # u/v は通常同じ次元を持つので u_var の次元で十分
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[u_var].dims, roles, keep_dims, f"vec_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[u_var].dims, roles, keep_dims, f"vec_{lid}")
    s = layer["style"]
    color_mode = st.radio(t("ベクトルの色付け"), list(_CMODE_MAG_LABELS),
                          format_func=tr_labels(_CMODE_MAG_LABELS).get,
                          horizontal=True, key=f"vec_cmode_{lid}",
                          help=t("カラーマップ: ベクトルの大きさ |V| = √(u²+v²) に"
                               "応じて矢印を色分けする"))
    if color_mode == "cmap":
        s["use_cmap"] = True

        def _preview_mag_range():
            mag = _preview_magnitude(panel, layer, datasets, keep_dims)
            return transformed_range(f"vec_{lid}", float(np.nanmin(mag)),
                                     float(np.nanmax(mag)), magnitude=True)

        cmap_section_ui(s, f"vec_{lid}", _preview_mag_range)
    else:
        s["use_cmap"] = False
        s["color"] = color_selector(t("ベクトルの色"), "#000000", key=f"vec_color_{lid}",
                                      meta_store=s, meta_key="color")
    # UIは「間引く格子数」(0=間引きなし、1=1格子飛ばし…)。
    # 内部設定・生成スクリプトは isel のステップ幅 (UI値+1) で保持する。
    skip_x = int(st.slider(t("間引き格子数 (x方向)"), 0, 9, 2,
                           help=t("0=間引きなし、1=1格子おき、2=2格子おき…"),
                           key=f"vec_skip_x_{lid}"))
    skip_y = int(st.slider(t("間引き格子数 (y方向)"), 0, 9, 2,
                           help=t("0=間引きなし、1=1格子おき、2=2格子おき…"),
                           key=f"vec_skip_y_{lid}"))
    s["stride_x"] = skip_x + 1
    s["stride_y"] = skip_y + 1
    if st.checkbox(t("大きさが閾値以下の矢印を非表示 (maskout)"), value=False,
                    key=f"vec_mask_{lid}",
                    help=t("|V| = √(u²+v²) が閾値以下の格子点に矢印を描かない。"
                         "弱風域を省いて図を見やすくする用途。閾値は値の変換"
                         "適用後の単位で入力")):
        s["mask_below"] = float(st.number_input(
            t("閾値 (|V| がこの値以下を非表示)"), value=1.0, min_value=0.0,
            step=0.5, format="%g", key=f"vec_maskval_{lid}"))
    units = ds[u_var].attrs.get("units", "")
    # 基準ベクトル: ベクトルキー (quiverkey の U) とスケール手動指定の両方が
    # 参照する物理量。2箇所に同じ数字が並んで連動しない混乱を避けるため
    # 1入力に統一 (2026-09-10)
    key_cfg = s["key"]
    key_cfg["length"] = ref_mag = float(st.number_input(
        t("基準ベクトルの物理量") + (f" [{units}]" if units else ""),
        value=10.0, min_value=0.0, format="%g", key=f"vec_keylen_{lid}",
        help=t("ベクトルキーの矢印が代表する値。スケールを手動指定するときの"
             "基準にもなる")))
    # 旧セッション互換: 旧「スケール」直接値 (vec_scale_) が残っていて % が
    # 未設定なら、% = 物理量 ÷ scale × 100 に換算して引き継ぐ (図は変わらない)
    _legacy_scale = st.session_state.get(f"vec_scale_{lid}")
    if (_legacy_scale and ref_mag > 0
            and f"vec_refpct_{lid}" not in st.session_state):
        st.session_state[f"vec_refpct_{lid}"] = min(
            1000.0, max(0.01, ref_mag / float(_legacy_scale) * 100.0))
    if not st.checkbox(t("スケールを自動"), value=True, key=f"vec_autoscale_{lid}"):
        pct = float(st.number_input(
            t("基準ベクトルの長さ (軸幅の %)"), value=5.0, min_value=0.01,
            max_value=1000.0, step=0.5, format="%g", key=f"vec_refpct_{lid}",
            help=t("基準ベクトルの物理量を、プロット領域の幅の何%の長さで描くか。"
                 "ベクトルキーの矢印もこの長さになる")))
        if ref_mag > 0:
            s["scale"] = ref_mag / (pct / 100.0)
            st.caption(t("換算後の quiver scale = {scale:g} "
                       "(再現スクリプトにはこの値が入る)", scale=s["scale"]))
        else:
            st.warning(t("基準ベクトルの物理量が 0 のためスケールは自動になります"))
    if not st.checkbox(t("線幅を自動"), value=True, key=f"vec_autowidth_{lid}"):
        s["width"] = float(st.number_input(t("線幅"), value=0.005, min_value=0.001,
                                           max_value=0.05, step=0.001, format="%.3f",
                                           key=f"vec_width_{lid}"))
    if not st.checkbox(t("頭の長さを自動"), value=True, key=f"vec_autohl_{lid}"):
        s["headlength"] = float(st.number_input(
            t("頭の長さ (headlength)"), value=5.0, min_value=0.0, max_value=20.0,
            step=0.5, format="%g", key=f"vec_headlen_{lid}",
            help=t("矢じりの長さ (軸幅単位、matplotlib 既定 5)。大きいほど頭が"
                 "長くなる。0 で頭なし (線分のみ)。付け根の長さは既定比率で連動")))
    if st.checkbox(t("枠線を表示"), value=False, key=f"vec_edge_{lid}",
                    help=t("矢印の輪郭に線を引く。白い矢印に黒枠を付けて背景から"
                         "浮かせる、といった用途")):
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                          key=f"vec_ecol_{lid}",
                                          meta_store=s, meta_key="edge_color")
        s["edge_width"] = float(st.slider(
            t("枠線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"vec_elw_{lid}"))
    with st.expander(t("ベクトルキー"), expanded=False):
        key_cfg["show"] = st.checkbox(t("ベクトルキーを表示"), value=True,
                                      key=f"vec_key_{lid}")
        if key_cfg["show"]:
            st.caption(t("キーの矢印は上の「基準ベクトルの物理量」を代表する"))
            key_cfg["label"] = st.text_input(
                t("キーのラベル"),
                value=f"{ref_mag:g} {units}".strip(),
                key=f"vec_keylab_{lid}")
            key_cfg["labelpos"] = st.selectbox(
                t("ラベルの配置"), list(_KEYLPOS_LABELS),
                format_func=tr_labels(_KEYLPOS_LABELS).get, key=f"vec_keylpos_{lid}")
            key_cfg["fontsize"] = fontsize_input(t("文字サイズ"),
                                                 f"vec_keyfs_{lid}")
            key_cfg["x"] = float(st.number_input(
                t("キーの x 位置"), -0.5, 1.5, 1.0, 0.05,
                help=t("プロット領域の左端が0、右端が1。ラベルが矢印の右のときは"
                     "矢印の右端、左のときは矢印の左端がこの位置に揃う"),
                key=f"vec_keyx_{lid}"))
            key_cfg["y"] = float(st.number_input(
                t("キーの y 位置"), -0.5, 1.5, -0.07, 0.01,
                help=t("プロット領域の下端が0、上端が1。負の値で領域の下側"),
                key=f"vec_keyy_{lid}"))
    if s["use_cmap"]:
        units = ds[u_var].attrs.get("units", "")
        default_label = f"|V| [{units}]" if units else "|V|"
        colorbar_ui(s["colorbar"], f"vec_{lid}", default_label,
                    label_suffix=f"_{u_var}")
    value_transform_ui(s, f"vec_{lid}", with_offset=False)
    return layer


def stream_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                    allow_averaging=False, panel=None):
    """流線 (streamplot) レイヤーの設定 UI。水平面図・鉛直断面図で使う。"""
    src = _vector_components_ui(
        datasets, variables_by_ds, "strm", lid,
        lambda dsid: t("`{dsid}` には流線描画可能な変数が2つ未満です", dsid=dsid))
    if src is None:
        return None
    dsid, ds, u_var, v_dsid, v_ds, v_var = src
    layer = mc_config.default_stream_layer(dsid, u_var, v_var, v_dataset_id=v_dsid)
    keep_dims = _layer_keep_dims(ds, keep_dims, panel)
    if roles is not None:
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[u_var].dims, roles, keep_dims, f"strm_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[u_var].dims, roles, keep_dims, f"strm_{lid}")
    s = layer["style"]
    c1, c2, c3 = st.columns(3)
    s["density"] = float(c1.number_input(
        t("密度 (density)"), 0.2, 5.0, 1.0, 0.1, key=f"strm_dens_{lid}",
        help=t("流線の本数。大きいほど密")))
    s["linewidth"] = float(c2.number_input(
        t("線の太さ"), 0.2, 5.0, 1.0, 0.1, key=f"strm_lw_{lid}"))
    s["arrowsize"] = float(c3.number_input(
        t("矢じりの大きさ"), 0.0, 5.0, 1.0, 0.1, key=f"strm_asz_{lid}"))
    st.caption(t("流線は等間隔・昇順の格子を要求するため、気圧レベル等の"
               "**不等間隔な軸**をもつ断面では自動で等間隔に線形補間してから描きます"
               "(緯度経度など等間隔の軸はそのまま)。"))
    color_mode = st.radio(t("流線の色付け"), list(_CMODE_MAG_LABELS),
                          format_func=tr_labels(_CMODE_MAG_LABELS).get,
                          horizontal=True, key=f"strm_cmode_{lid}",
                          help=t("カラーマップ: |V| = √(u²+v²) に応じて流線を色分けする"))
    if color_mode == "cmap":
        s["use_cmap"] = True

        def _preview_mag_range():
            mag = _preview_magnitude(panel, layer, datasets, keep_dims)
            return transformed_range(f"strm_{lid}", float(np.nanmin(mag)),
                                     float(np.nanmax(mag)), magnitude=True)

        cmap_section_ui(s, f"strm_{lid}", _preview_mag_range)
    else:
        s["use_cmap"] = False
        s["color"] = color_selector(t("流線の色"), "#000000", key=f"strm_color_{lid}",
                                     meta_store=s, meta_key="color")
    if s["use_cmap"]:
        units = ds[u_var].attrs.get("units", "")
        default_label = f"|V| [{units}]" if units else "|V|"
        colorbar_ui(s["colorbar"], f"strm_{lid}", default_label,
                    label_suffix=f"_{u_var}")
    value_transform_ui(s, f"strm_{lid}", with_offset=False)
    return layer


def hatch_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                   allow_averaging=False, panel=None):
    dsid = _layer_dataset_selector(datasets, f"hatch_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には描画可能な変数がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"hatch_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"hatch_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_hatch_layer(dsid, var)
    keep_dims = _layer_keep_dims(ds, keep_dims, panel)
    if roles is not None:
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[var].dims, roles, keep_dims, f"hatch_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[var].dims, roles, keep_dims, f"hatch_{lid}")
    p = layer_preview_data(panel, layer, datasets, keep_dims)
    # 閾値は「値の変換」後の単位なので既定値も変換後の範囲から作る
    dmin, dmax = transformed_range(f"hatch_{lid}", float(p.min()), float(p.max()))
    lo = st.number_input(t("下限値 (この値以上にハッチ)"), value=float((dmin + dmax) / 2),
                         key=f"hatch_lo_{lid}_{var}")
    hi = st.number_input(t("上限値"), value=dmax, key=f"hatch_hi_{lid}_{var}")
    layer["style"]["levels"] = [float(lo), float(hi)]
    layer["style"]["pattern"] = st.selectbox(t("ハッチの種類"), HATCH_PATTERNS,
                                             key=f"hatch_pat_{lid}")
    layer["style"]["density"] = int(st.slider(
        t("ハッチの密度"), 1, 6, 3, 1,
        help=t("パターン文字の繰り返し回数。大きいほど線が密になる (3 が既定)"),
        key=f"hatch_den_{lid}"))
    layer["style"]["linewidth"] = float(st.slider(
        t("ハッチ線の太さ"), 0.2, 4.0, 1.0, 0.1,
        help=t("rcParams['hatch.linewidth'] を一時的に変更する"),
        key=f"hatch_lw_{lid}"))
    layer["style"]["color"] = color_selector(
        t("ハッチ線の色"), "#000000", key=f"hatch_col_{lid}",
        meta_store=layer["style"], meta_key="color")
    value_transform_ui(layer["style"], f"hatch_{lid}")
    maskout_ui(layer["style"], f"hatch_{lid}", ds=ds, variable=var,
               required_dims=keep_dims)
    return layer


# マーカー選択肢 (中立キー → 表示ラベル)。"none" はマーカーなし (config は None)
_LINE_MARKER_LABELS = {
    "none": "なし",
    "o": "○ (circle)",
    "s": "■ (square)",
    "^": "▲ (triangle)",
    "x": "× (x)",
    ".": "• (dot)",
    "+": "+ (plus)",
}


def _legend_label_ui(key, default, legacy_key=None):
    """1次元プロット系レイヤー (line / line fill / bar / hist / ecdf) の
    「凡例ラベル」text_input。返り値は文字列か None。

    既定値は変数名。key に変数名を含めるので、変数を変えると既定 (新しい変数名)
    に戻り、元の変数に戻せば入力したラベルも戻る。空欄なら None (凡例に出さない)。

    legacy_key: v0.97 以前の変数名を含まない key。旧セッションの保存値があれば
    消費し、カスタムラベル (非空) だけを新 key が無い初回に引き継ぐ
    (旧セッションの空欄は引き継がず、既定 = 変数名になる)。
    """
    if legacy_key and legacy_key in st.session_state:
        old = st.session_state.pop(legacy_key)
        if old and key not in st.session_state:
            st.session_state[key] = old
    return st.text_input(t("凡例ラベル"), value=default, key=key,
                         placeholder=t("(空欄なら凡例に出さない)")) or None


def line_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                  allow_averaging=False, panel=None):
    dsid = _layer_dataset_selector(datasets, f"line_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には選択した軸を持つ変数がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"line_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"line_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_line_layer(dsid, var)
    # 1次元プロット(集計) では x はレイヤー毎の「x軸にする次元」の座標値
    # (理論分布などをヒストグラムに重ねる用途)。lid の mode_key 部で判定する
    is_dist = lid.split("_", 1)[0].startswith("dist")
    if is_dist:
        _numdims = [d for d in ds[var].dims
                    if d in ds.coords
                    and np.issubdtype(ds[d].dtype, np.number)]
        if not _numdims:
            st.warning(t("数値座標を持つ次元が無いためラインを描けません "
                       "(x は値軸なので datetime 座標は使えません)"))
            return None
        if len(_numdims) == 1:
            x_dim = _numdims[0]
            st.caption(t("x軸にする次元: `{x_dim}` (座標値が x になる)", x_dim=x_dim))
        else:
            x_dim = st.selectbox(
                t("x軸にする次元"), _numdims, key=f"line_xdim_{lid}",
                help=t("この次元の座標値が x になる (ヒストグラムの値軸と"
                     "同じ単位にすること)"))
        layer["x_dim"] = x_dim
        layer["selection"] = _fix_dims_ui(ds, list(ds[var].dims), (x_dim,),
                                          f"line_{lid}")
    elif roles is not None:
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[var].dims, roles, keep_dims, f"line_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[var].dims, roles, keep_dims, f"line_{lid}")
    s = layer["style"]
    auto_color = st.checkbox(t("色を自動 (カラーサイクル)"), value=True,
                              key=f"line_autocolor_{lid}")
    if auto_color:
        s["color"] = None
    else:
        s["color"] = color_selector(t("線の色"), "#1f77b4", key=f"line_color_{lid}",
                                     meta_store=s, meta_key="color")
    s["linewidth"] = float(st.slider(t("線の太さ"), 0.5, 4.0, 1.5, 0.1,
                                      key=f"line_lw_{lid}"))
    s["linestyle"] = st.selectbox(
        t("線種"), list(LINE_LAYER_LINESTYLE_LABELS),
        format_func=tr_labels(LINE_LAYER_LINESTYLE_LABELS).get, key=f"line_ls_{lid}",
        help=t("「なし」を選ぶと線を描かず、マーカーだけ表示する"))
    _marker = st.selectbox(t("マーカー"), list(_LINE_MARKER_LABELS),
                           format_func=tr_labels(_LINE_MARKER_LABELS).get,
                           key=f"line_marker_{lid}")
    s["marker"] = None if _marker == "none" else _marker
    # マーカーを選んだときだけ大きさ (pt) を出す。なしなら None (既定のまま)
    s["marker_size"] = (float(st.slider(
        t("マーカーサイズ (pt)"), 1.0, 15.0, 6.0, 0.5, key=f"line_msize_{lid}"))
        if s["marker"] else None)
    if is_dist:
        s["secondary_y"] = st.checkbox(
            t("第2軸 (右の縦軸) に描く"), value=False, key=f"line_y2_{lid}",
            help=t("度数ヒストグラムと単位の違う曲線 (密度など) を重ねるとき用"))
    else:
        s["secondary_y"] = st.checkbox(
            t("第2軸 (右の縦軸) に描く"), value=False, key=f"line_y2_{lid}",
            help=t("単位の違う物理量を重ねるとき用。第2軸のラベル・範囲・目盛などは「第2軸 (右)」で指定する"))
    s["label"] = _legend_label_ui(f"line_label_{lid}_{var}", var,
                                  legacy_key=f"line_label_{lid}")
    value_transform_ui(s, f"line_{lid}")
    return layer


def line_bundle_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                         allow_averaging=False, panel=None):
    """ライン (束): 1 つの次元の全スライス (アンサンブルメンバー等) を同じ体裁で
    重ね描きし、凡例は 1 つ。統計線 (平均・中央値・最小/最大・パーセンタイル) を
    上に重ねられる。1次元プロット専用 (x は panel.x_dim)。"""
    dsid = _layer_dataset_selector(datasets, f"bundle_{lid}")
    ds = datasets[dsid]
    time_dim = _dataset_roles(ds, roles or {}).get("time")

    def _bundle_candidates(v):
        # 束ねる次元の候補: 座標付きで、描画軸 (keep_dims) でも時刻でもない次元
        # (時刻はパネル側で固定・アニメーションするため対象外)
        return [str(d) for d in ds[v].dims
                if d in ds.coords and d not in keep_dims and d != time_dim]

    variables = [v for v in variables_by_ds[dsid] if _bundle_candidates(v)]
    if not variables:
        st.warning(t("`{dsid}` には選択した軸のほかに束ねられる次元を持つ変数が"
                     "ありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"bundle_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"bundle_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    cands = _bundle_candidates(var)
    _reset_invalid_var_key(f"bundle_dim_{lid}", cands)
    dim_labels = {d: t("{dim} ({n} 本)", dim=d, n=int(ds.sizes[d])) for d in cands}
    bdim = st.selectbox(
        t("束ねる次元"), cands, key=f"bundle_dim_{lid}", format_func=dim_labels.get,
        help=t("この次元の全スライスを同じ色・線種・透明度で重ね描きし、凡例は1つに"
               "まとめる (アンサンブルメンバーなど)。残りの次元は固定値か範囲平均"))
    layer = mc_config.default_line_bundle_layer(dsid, var, bdim)
    if roles is not None:
        keep = tuple(keep_dims) + (bdim,)
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[var].dims, roles, keep, f"bundle_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[var].dims, roles, keep, f"bundle_{lid}")
    s = layer["style"]
    s["color"] = color_selector(t("線の色"), "#808080", key=f"bundle_color_{lid}",
                                 meta_store=s, meta_key="color")
    s["linewidth"] = float(st.slider(
        t("線の太さ"), 0.0, 4.0, 0.8, 0.1, key=f"bundle_lw_{lid}",
        help=t("0 にすると束の線を描かない (統計線の帯だけを見せるとき)")))
    s["alpha"] = float(st.slider(
        t("透明度 (alpha)"), 0.05, 1.0, 0.5, 0.05, key=f"bundle_alpha_{lid}",
        help=t("本数が多いときは小さめにすると重なりの濃淡が見える (1 で不透明)")))
    s["linestyle"] = st.selectbox(
        t("線種"), list(LINE_LAYER_LINESTYLE_LABELS),
        format_func=tr_labels(LINE_LAYER_LINESTYLE_LABELS).get, key=f"bundle_ls_{lid}",
        help=t("「なし」を選ぶと線を描かず、マーカーだけ表示する"))
    _marker = st.selectbox(t("マーカー"), list(_LINE_MARKER_LABELS),
                           format_func=tr_labels(_LINE_MARKER_LABELS).get,
                           key=f"bundle_marker_{lid}")
    s["marker"] = None if _marker == "none" else _marker
    s["marker_size"] = (float(st.slider(
        t("マーカーサイズ (pt)"), 1.0, 15.0, 6.0, 0.5, key=f"bundle_msize_{lid}"))
        if s["marker"] else None)
    s["secondary_y"] = st.checkbox(
        t("第2軸 (右の縦軸) に描く"), value=False, key=f"bundle_y2_{lid}",
        help=t("単位の違う物理量を重ねるとき用。第2軸のラベル・範囲・目盛などは「第2軸 (右)」で指定する"))
    s["label"] = _legend_label_ui(f"bundle_label_{lid}_{var}", var)
    value_transform_ui(s, f"bundle_{lid}")

    # 統計線: 束ねた次元方向の集計を別の線 (2 本組は帯も可) として重ねる。
    # ON にしたときだけ本数 (1 以上) と各統計線の設定を出す
    stats_on = st.checkbox(
        t("統計線 (束の集計を重ねる)"), value=False, key=f"bundle_stats_{lid}",
        help=t("束ねた次元方向に集計した線 (平均・中央値・最小/最大・パーセンタイル) を"
               "束の上に重ねる。NaN は除外して集計する。「2本」の統計量は 1 組で"
               "凡例 1 つで、帯 (塗りつぶし) でも描ける"))
    if not stats_on:
        return layer
    # 本数の下限を 0 → 1 に変えた (2026-09-20)。0 が保存された旧 WIP は下限に丸める
    # (widget 描画前の代入なので許される)
    if st.session_state.get(f"bundle_nstat_{lid}", 1) < 1:
        st.session_state[f"bundle_nstat_{lid}"] = 1
    n_stats = int(st.number_input(
        t("統計線の本数"), min_value=1, max_value=6, value=1, step=1,
        key=f"bundle_nstat_{lid}"))
    for i in range(n_stats):
        with st.expander(t("統計線{n}", n=i + 1), expanded=True):
            stat = st.selectbox(t("統計量"), list(_BUNDLE_STAT_LABELS),
                                format_func=tr_labels(_BUNDLE_STAT_LABELS).get,
                                key=f"bundle_stat_{lid}_{i}")
            summary = mc_config.default_bundle_summary(stat)
            if stat == "percentile":
                summary["q_low"] = float(st.number_input(
                    t("パーセンタイル (%)"), min_value=0.0, max_value=100.0,
                    value=95.0, step=1.0, key=f"bundle_q_{lid}_{i}"))
            elif stat == "pct_range":
                c1, c2 = st.columns(2)
                summary["q_low"] = float(c1.number_input(
                    t("下側 (%)"), min_value=0.0, max_value=100.0, value=5.0,
                    step=1.0, key=f"bundle_qlo_{lid}_{i}"))
                summary["q_high"] = float(c2.number_input(
                    t("上側 (%)"), min_value=0.0, max_value=100.0, value=95.0,
                    step=1.0, key=f"bundle_qhi_{lid}_{i}"))
            elif stat == "std_range":
                summary["k_std"] = float(st.number_input(
                    t("標準偏差の倍率 k"), min_value=0.1, max_value=5.0, value=1.0,
                    step=0.5, key=f"bundle_kstd_{lid}_{i}",
                    help=t("平均 ± k × 標準偏差 の 2 本。標準偏差は標本標準偏差 "
                           "(N−1 で割る。numpy の nanstd(ddof=1)、NaN 除外)")))
            if stat in ("minmax", "pct_range", "std_range"):
                summary["draw"] = st.radio(
                    t("描き方"), list(_BUNDLE_DRAW_LABELS),
                    format_func=tr_labels(_BUNDLE_DRAW_LABELS).get, horizontal=True,
                    key=f"bundle_draw_{lid}_{i}",
                    help=t("帯は 2 本の間を塗りつぶす (fill_between)。"
                           "「帯 + 縁の線」は帯の上に 2 本の線も描く"))
            ss = summary["style"]
            ss["color"] = color_selector(t("線の色"), "#000000",
                                          key=f"bundle_scolor_{lid}_{i}",
                                          meta_store=ss, meta_key="color")
            if summary["draw"] != "band":
                ss["linewidth"] = float(st.slider(
                    t("線の太さ"), 0.0, 4.0, 2.0, 0.1, key=f"bundle_slw_{lid}_{i}",
                    help=t("0 にすると線を描かない")))
                ss["linestyle"] = st.selectbox(
                    t("線種"), list(LINESTYLE_LABELS),
                    format_func=tr_labels(LINESTYLE_LABELS).get,
                    key=f"bundle_sls_{lid}_{i}")
            if summary["draw"] != "lines":
                ss["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.05, 1.0, 0.3, 0.05,
                                               key=f"bundle_salpha_{lid}_{i}"))
            # 既定ラベルは統計量と分位から作る (言語に依存しない英字)。key に
            # 含めるので統計量・分位を変えると既定に戻る (line の変数名と同じ流儀)
            default_label = mc_render.bundle_stat_default_label(summary)
            ss["label"] = _legend_label_ui(
                f"bundle_slabel_{lid}_{i}_{default_label}", default_label)
            layer["summaries"].append(summary)
    return layer


def fill_between_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                          allow_averaging=False, panel=None):
    """1次元プロット用: 2線間 (または 1線と定数 baseline の間) を塗りつぶすレイヤー。

    変数を2つ選ぶ場合、各変数で独立に次元を固定できる (例: 1つ目を500hPa、
    2つ目を850hPa にして垂直差の帯を描く)。
    """
    dsid = _layer_dataset_selector(datasets, f"fb_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には選択した軸を持つ変数がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"fb_var_{lid}", variables)
    var = st.selectbox(t("変数 (1つ目)"), variables, key=f"fb_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    # 1つ目の次元固定
    selection_lower: dict = {}
    averages_lower: dict = {}
    if roles is not None:
        if allow_averaging:
            selection_lower, averages_lower = _layer_dim_picker_with_avg(
                ds, ds[var].dims, roles, keep_dims, f"fb_{lid}")
        else:
            selection_lower = _layer_dim_picker(
                ds, ds[var].dims, roles, keep_dims, f"fb_{lid}")

    mode = st.radio(t("2つ目の値"), list(_FB_MODE_LABELS),
                     format_func=tr_labels(_FB_MODE_LABELS).get, horizontal=True,
                     key=f"fb_mode_{lid}")
    var_upper = None
    baseline = 0.0
    selection_upper: dict = {}
    averages_upper: dict = {}
    if mode == "variable":
        # 同じ dataset 内で1つ目と異なる変数を初期選択
        default_idx = 1 if len(variables) >= 2 and variables[0] == var else 0
        _reset_invalid_var_key(f"fb_var2_{lid}", variables)
        var_upper = st.selectbox(t("変数 (2つ目)"), variables, index=default_idx,
                                  key=f"fb_var2_{lid}",
                                  format_func=lambda v: var_label(ds, v))
        # 2つ目の次元固定 (key prefix を分けて1つ目と独立)
        if roles is not None:
            if allow_averaging:
                selection_upper, averages_upper = _layer_dim_picker_with_avg(
                    ds, ds[var_upper].dims, roles, keep_dims, f"fb_upper_{lid}")
            else:
                selection_upper = _layer_dim_picker(
                    ds, ds[var_upper].dims, roles, keep_dims, f"fb_upper_{lid}")
    else:
        baseline = float(st.number_input(t("ベースライン値"), value=0.0, step=1.0,
                                          key=f"fb_baseline_{lid}",
                                          help=t("value_transform 後 (= 描画される単位) の値")))
    layer = mc_config.default_fill_between_layer(dsid, var, var_upper, baseline)
    layer["selection"] = selection_lower
    layer["selection_upper"] = selection_upper
    layer["averages"] = averages_lower
    layer["averages_upper"] = averages_upper
    s = layer["style"]
    # baseline モードのみ「上下で色を変える」オプションを出す
    split_color = False
    if mode == "constant":
        split_color = st.checkbox(
            t("ベースラインの上下で色を変える"), value=False,
            key=f"fb_split_{lid}",
            help=t("ON にすると、ベースラインより上と下で別の色で塗る"))
    s["color"] = color_selector(t("上側の色") if split_color else t("色"),
                                 "#1f77b4", key=f"fb_color_{lid}",
                                 meta_store=s, meta_key="color")
    if split_color:
        s["color_below"] = color_selector(t("下側の色"), "#d62728",
                                            key=f"fb_color_below_{lid}",
                                            meta_store=s, meta_key="color_below")
    else:
        s["color_below"] = None
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 0.3, 0.05,
                                  key=f"fb_alpha_{lid}"))
    s["secondary_y"] = st.checkbox(
        t("第2軸 (右の縦軸) に描く"), value=False, key=f"fb_y2_{lid}",
        help=t("単位の違う物理量を重ねるとき用。第2軸のラベル・範囲・目盛などは「第2軸 (右)」で指定する"))
    # 既定 = 変数名 (2変数モードは選んだ順に「1つ目–2つ目」)
    if var_upper is None:
        s["label"] = _legend_label_ui(f"fb_label_{lid}_{var}", var,
                                      legacy_key=f"fb_label_{lid}")
    else:
        s["label"] = _legend_label_ui(f"fb_label_{lid}_{var}_{var_upper}",
                                      f"{var}–{var_upper}",
                                      legacy_key=f"fb_label_{lid}")
    value_transform_ui(s, f"fb_{lid}")
    return layer


_STACKPLOT_BASELINE_LABELS = {
    "zero": "zero (積み上げ)",
    "sym": "sym (対称)",
    "wiggle": "wiggle",
    "weighted_wiggle": "weighted_wiggle (Streamgraph)",
}


def stackplot_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                       allow_averaging=False, panel=None):
    """1次元プロット用のスタックプロット (ax.stackplot)。

    複数変数を multiselect で順序付きに選び、積み順 (下→上) で塗り分ける。
    """
    dsid = _layer_dataset_selector(datasets, f"sp_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には選択した軸を持つ変数がありません", dsid=dsid))
        return None
    default_vars = variables[:min(2, len(variables))]
    sel_vars = st.multiselect(
        t("変数 (順番が積み順、下→上)"), variables, default=default_vars,
        key=f"sp_vars_{lid}",
        format_func=lambda v: var_label(ds, v))
    if not sel_vars:
        st.warning(t("変数を 1 つ以上選択してください"))
        return None
    layer = mc_config.default_stackplot_layer(dsid, sel_vars)
    if roles is not None:
        # 1 つ目の変数の次元で固定 UI を出す (すべての変数で同じ次元固定を共有)
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[sel_vars[0]].dims, roles, keep_dims, f"sp_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[sel_vars[0]].dims, roles, keep_dims, f"sp_{lid}")
    s = layer["style"]
    s["alpha"] = float(st.slider(
        t("透明度 (alpha)"), 0.0, 1.0, 0.8, 0.05, key=f"sp_alpha_{lid}"))
    s["baseline"] = st.selectbox(
        t("ベースライン"), list(_STACKPLOT_BASELINE_LABELS),
        format_func=tr_labels(_STACKPLOT_BASELINE_LABELS).get, key=f"sp_base_{lid}",
        help=t("zero: 0 を底に積み上げ / sym: 0 を中央に対称配置 / "
             "wiggle: 全体の揺らぎを最小化 / "
             "weighted_wiggle: ストリームグラフ (Byron-Wattenberg)"))
    s["show_labels_in_legend"] = st.checkbox(
        t("変数名を凡例に表示"), value=True, key=f"sp_legend_{lid}")
    if st.checkbox(t("各変数の色を手動指定"), value=False, key=f"sp_color_use_{lid}"):
        colors = []
        for i, v in enumerate(sel_vars):
            colors.append(color_selector(
                t("{v} の色", v=v), "#1f77b4", key=f"sp_color_{lid}_{i}",
                meta_store=s, meta_key=f"colors[{i}]"))
        s["colors"] = colors
    else:
        s["colors"] = None
    s["secondary_y"] = st.checkbox(
        t("第2軸 (右の縦軸) に描く"), value=False, key=f"sp_y2_{lid}",
        help=t("単位の違う物理量を重ねるとき用。第2軸のラベル・範囲・目盛などは「第2軸 (右)」で指定する"))
    value_transform_ui(s, f"sp_{lid}")
    return layer


def bar_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                 allow_averaging=False, panel=None):
    """1次元プロット用の棒グラフレイヤー UI。

    panel-level の bar_mode (overlap/dodge/stack) は別の場所で指定するため、
    ここでは レイヤー固有の orientation, width, color, edge, label のみを扱う。
    """
    dsid = _layer_dataset_selector(datasets, f"bar_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には選択した軸を持つ変数がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"bar_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"bar_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_bar_layer(dsid, var)
    if roles is not None:
        if allow_averaging:
            layer["selection"], layer["averages"] = _layer_dim_picker_with_avg(
                ds, ds[var].dims, roles, keep_dims, f"bar_{lid}")
        else:
            layer["selection"] = _layer_dim_picker(
                ds, ds[var].dims, roles, keep_dims, f"bar_{lid}")
    s = layer["style"]
    s["orientation"] = st.radio(
        t("向き"), list(_BAR_ORIENT_LABELS), format_func=tr_labels(_BAR_ORIENT_LABELS).get,
        horizontal=True, key=f"bar_orient_{lid}")
    s["color"] = color_selector(t("色"), "#1f77b4", key=f"bar_color_{lid}",
                                  meta_store=s, meta_key="color")
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 0.8, 0.05,
                                  key=f"bar_alpha_{lid}"))
    if st.checkbox(t("棒の幅を手動指定"), value=False,
                    key=f"bar_w_manual_{lid}",
                    help=t("off だと隣接 x 差分の中央値の 0.8 倍で自動")):
        s["width"] = float(st.number_input(
            t("幅 (データ単位)"), value=1.0, step=0.1, format="%g",
            key=f"bar_w_{lid}"))
    if st.checkbox(t("枠線を表示"), value=False, key=f"bar_edge_{lid}"):
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                          key=f"bar_edge_color_{lid}",
                                          meta_store=s, meta_key="edge_color")
        s["edge_linewidth"] = float(st.slider(
            t("枠線の太さ"), 0.1, 3.0, 1.0, 0.1, key=f"bar_edge_lw_{lid}"))
    else:
        s["edge_linewidth"] = 0.0
    # ハッチ (枠線の有無と独立)
    if st.checkbox(t("ハッチをかける"), value=False, key=f"bar_hatch_{lid}"):
        s["hatch_pattern"] = st.selectbox(
            t("ハッチの種類"), HATCH_PATTERNS, key=f"bar_hatch_pat_{lid}")
        s["hatch_density"] = int(st.slider(
            t("ハッチの密度"), 1, 6, 3, 1, key=f"bar_hatch_den_{lid}",
            help=t("パターン文字の繰り返し回数。大きいほど線が密")))
        # 枠線 off のときだけ独立した色を指定可能。枠線 on のときは枠線色と共有
        s["hatch_color"] = color_selector(
            t("ハッチの色"), "#000000", key=f"bar_hatch_col_{lid}",
            meta_store=s, meta_key="hatch_color")
        if float(s.get("edge_linewidth", 0.0)) > 0:
            st.caption(t("⚠ 枠線が ON のときは、ハッチの色は枠線の色と同じになります "
                        "(matplotlib の制約)"))
    else:
        s["hatch_pattern"] = None
    # エラーバー (対称、変数 or 定数)
    if st.checkbox(t("エラーバーを表示"), value=False, key=f"bar_err_{lid}",
                    help=t("変数または定数で対称エラーを描く。値は |err| に正規化される。"
                         "値の変換は掛けないので、図に表示する単位で用意する")):
        err = dict(s.get("errorbar", {}))
        src_label = st.radio(
            t("エラー量"), list(_ERR_SRC_LABELS),
            format_func=tr_labels(_ERR_SRC_LABELS).get, horizontal=True,
            key=f"bar_err_src_{lid}")
        if src_label == "variable":
            err["source"] = "variable"
            err["variable"] = st.selectbox(
                t("エラー量の変数"), variables, key=f"bar_err_var_{lid}",
                format_func=lambda v: var_label(ds, v))
        else:
            err["source"] = "constant"
            err["constant"] = float(st.number_input(
                t("エラー量 (定数、対称)"), value=1.0, step=0.1, format="%g",
                key=f"bar_err_const_{lid}"))
        err["color"] = color_selector(
            t("エラーバーの色"), "#000000", key=f"bar_err_col_{lid}",
            meta_store=err, meta_key="color")
        err["linewidth"] = float(st.slider(
            t("エラーバーの太さ"), 0.1, 3.0, 1.0, 0.1, key=f"bar_err_lw_{lid}"))
        err["capsize"] = float(st.slider(
            t("キャップサイズ (pt)"), 0.0, 10.0, 3.0, 0.5, key=f"bar_err_cap_{lid}"))
        s["errorbar"] = err
    else:
        s["errorbar"] = {"source": "none", "variable": None, "constant": 0.0,
                          "color": "#000000", "linewidth": 1.0, "capsize": 3.0}
    if s["orientation"] == "vertical":
        # 横向き棒は値が x 軸に乗るので第2 (右) 軸は無意味 → 縦向きだけ出す。
        # 描画しない run では style は既定 (False) のまま
        s["secondary_y"] = st.checkbox(
            t("第2軸 (右の縦軸) に描く"), value=False, key=f"bar_y2_{lid}",
            help=t("単位の違う物理量を重ねるとき用。第2軸のラベル・範囲・目盛などは「第2軸 (右)」で指定する"))
    s["label"] = _legend_label_ui(f"bar_label_{lid}_{var}", var,
                                  legacy_key=f"bar_label_{lid}")
    value_transform_ui(s, f"bar_{lid}")
    return layer


_SCATTER_MARKER_LABELS = {
    "o": "○ (circle)",
    "s": "■ (square)",
    "^": "▲ (triangle)",
    "x": "× (x)",
    ".": "• (dot)",
    "+": "+ (plus)",
    "D": "◇ (diamond)",
    "*": "* (star)",
}


def _scatter_fixed_picker(ds, var_dims, drawing_dim, lid_prefix):
    """drawing_dim 以外の dim を 1 点に固定する UI。{dim: value} を返す。

    時刻次元のときは ISO 文字列で値を返す (xarray の .sel と互換)。
    初期値は各 dim の **中央のインデックス** を使う (端点の特異値で u/v が 0 に
    なって全点が原点に重なるなどの罠を避けるため)。
    """
    fixed = {}
    for dim in var_dims:
        if dim == drawing_dim or dim not in ds.coords:
            continue
        units = ds[dim].attrs.get("units", "")
        label = f"{dim}" + (f" [{units}]" if units else "")
        values = ds[dim].values
        mid_idx = len(values) // 2
        if np.issubdtype(np.asarray(values).dtype, np.datetime64):
            labels, sel_vals = time_labels_and_values(values)
            idx_key = f"{lid_prefix}_fixt_{dim}"
            if idx_key not in st.session_state:
                st.session_state[idx_key] = mid_idx
            idx = st.selectbox(label, range(len(labels)),
                                 format_func=lambda i, L=labels: L[i],
                                 key=idx_key)
            fixed[dim] = sel_vals[idx]
        else:
            fixed[dim] = st.selectbox(
                label, dim_choices(values), index=mid_idx,
                format_func=dim_choice_label,
                key=f"{lid_prefix}_fix_{dim}")
    return fixed


def _scatter_drawing_range(ds, drawing_dim, lid_prefix):
    """drawing_dim の範囲 (lo, hi) を選ばせる。全範囲なら None を返す。

    datetime64 は select_slider のラベル/値が ISO 文字列。
    """
    values = ds[drawing_dim].values
    if np.issubdtype(np.asarray(values).dtype, np.datetime64):
        labels, sel_vals = time_labels_and_values(values)
        lo_lbl, hi_lbl = st.select_slider(
            t("`{dim}` の範囲", dim=drawing_dim), options=labels,
            value=(labels[0], labels[-1]),
            key=f"{lid_prefix}_drange_t_{drawing_dim}")
        i0, i1 = labels.index(lo_lbl), labels.index(hi_lbl)
        if (i0, i1) == (0, len(labels) - 1):
            return None
        return [sel_vals[i0], sel_vals[i1]]
    try:
        vfloats = [float(v) for v in values]
        lo, hi = st.select_slider(
            t("`{dim}` の範囲", dim=drawing_dim), options=vfloats,
            value=(vfloats[0], vfloats[-1]),
            key=f"{lid_prefix}_drange_n_{drawing_dim}")
        if (lo, hi) == (vfloats[0], vfloats[-1]):
            return None
        return [float(lo), float(hi)]
    except (TypeError, ValueError):
        st.caption(t("`{dim}` は非数値のため範囲指定不可", dim=drawing_dim))
        return None


def scatter_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                     allow_averaging=False, panel=None):
    """2次元プロット用の散布図レイヤー。

    panel.x_variable / panel.y_variable は session_state 経由で取得し、ここでは
    描画する次元 (drawing_dim) と x/y それぞれの固定する次元 (x_fixed/y_fixed) を
    選ばせる。drawing_dim とその範囲は x/y で共有される (同一の dim 沿いに散布)。
    """
    dsid = _layer_dataset_selector(datasets, f"sc_{lid}")
    ds = datasets[dsid]
    mode_key = lid.rsplit("_", 1)[0]  # lid = f"{mode_key}_{n}"
    x_var = st.session_state.get(f"panel_xvar_{mode_key}")
    y_var = st.session_state.get(f"panel_yvar_{mode_key}")
    if not x_var or not y_var:
        st.warning(t("「プロット変数」セクションで x/y 軸の変数を選んでください"))
        return None
    if x_var not in ds.data_vars or y_var not in ds.data_vars:
        st.warning(t("`{dsid}` には変数 `{x_var}` または `{y_var}` がありません",
                   dsid=dsid, x_var=x_var, y_var=y_var))
        return None
    x_dims = [d for d in ds[x_var].dims if d in ds.coords]
    y_dims = [d for d in ds[y_var].dims if d in ds.coords]
    common_dims = [d for d in x_dims if d in y_dims]
    if not common_dims:
        st.warning(t("`{x_var}` と `{y_var}` に共通の座標次元がありません",
                   x_var=x_var, y_var=y_var))
        return None
    # 描画する次元 (x/y で共有)
    drawing_dim = st.selectbox(t("描画する次元"), common_dims,
                                  key=f"sc_drawing_dim_{lid}",
                                  help=t("この次元に沿って散布する。範囲スライダで絞れる"))
    drawing_range = _scatter_drawing_range(ds, drawing_dim, f"sc_{lid}")

    layer = mc_config.default_scatter_layer(dsid)
    layer["drawing_dim"] = drawing_dim
    layer["drawing_range"] = drawing_range
    # x 側の固定する次元
    st.markdown(t("**x 軸の変数 `{var}` で固定する次元**", var=x_var))
    layer["x_fixed"] = _scatter_fixed_picker(
        ds, x_dims, drawing_dim, f"sc_x_{lid}")
    # y 側の固定する次元
    st.markdown(t("**y 軸の変数 `{var}` で固定する次元**", var=y_var))
    layer["y_fixed"] = _scatter_fixed_picker(
        ds, y_dims, drawing_dim, f"sc_y_{lid}")

    s = layer["style"]
    s["color"] = color_selector(t("色"), "#1f77b4", key=f"sc_color_{lid}",
                                  meta_store=s, meta_key="color")
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 0.7, 0.05,
                                   key=f"sc_alpha_{lid}"))
    s["marker"] = st.selectbox(t("マーカー"), list(_SCATTER_MARKER_LABELS),
                               format_func=tr_labels(_SCATTER_MARKER_LABELS).get,
                               key=f"sc_marker_{lid}")
    s["size"] = float(st.slider(t("マーカーサイズ (s, 面積 pt²)"),
                                   1.0, 200.0, 20.0, 1.0, key=f"sc_size_{lid}"))
    if st.checkbox(t("枠線を表示"), value=False, key=f"sc_edge_{lid}"):
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                           key=f"sc_edge_color_{lid}",
                                           meta_store=s, meta_key="edge_color")
        s["edge_linewidth"] = float(st.slider(
            t("枠線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"sc_edge_lw_{lid}"))
    else:
        s["edge_linewidth"] = 0.0
    label = st.text_input(t("凡例ラベル"), value="", key=f"sc_label_{lid}",
                           placeholder=t("(空欄なら凡例に出さない)"))
    s["label"] = label or None
    with st.expander(t("エラーバー"), expanded=False):
        _eb_vars = [_NONE_OPTION] + list(variables_by_ds.get(dsid, []))
        eb = s["errorbar"]
        _ebx = st.selectbox(t("x方向の誤差の変数"), _eb_vars, key=f"sc_ebx_{lid}",
                            format_func=_none_option_labels(_eb_vars).get,
                            help=t("各点の x 方向の誤差 (±値、対称)。本体と同じ"
                                 "次元固定・範囲で切り出す。値の変換は掛けないので、"
                                 "図に表示する単位で用意する"))
        _eby = st.selectbox(t("y方向の誤差の変数"), _eb_vars, key=f"sc_eby_{lid}",
                            format_func=_none_option_labels(_eb_vars).get,
                            help=t("各点の y 方向の誤差 (±値、対称)。値の変換は掛けない"))
        eb["x_variable"] = None if _ebx == _NONE_OPTION else _ebx
        eb["y_variable"] = None if _eby == _NONE_OPTION else _eby
        if eb["x_variable"] or eb["y_variable"]:
            eb["color"] = color_selector(t("エラーバーの色"), "#000000",
                                          key=f"sc_ebc_{lid}",
                                          meta_store=eb, meta_key="color")
            eb["linewidth"] = float(st.slider(
                t("エラーバーの太さ"), 0.2, 3.0, 1.0, 0.1, key=f"sc_ebw_{lid}"))
            eb["capsize"] = float(st.slider(
                t("キャップ幅 (pt)"), 0.0, 10.0, 3.0, 0.5, key=f"sc_ebcap_{lid}"))
    with st.expander(t("値の変換 (x/y 独立、単位換算など)"), expanded=False):
        c1, c2 = st.columns(2)
        s["x_value_scale"] = float(c1.number_input(
            t("x 倍率 a"), value=1.0, format="%g", key=f"sc_xsc_{lid}",
            help=t("例: Pa→hPa は 0.01")))
        s["x_value_offset"] = float(c2.number_input(
            t("x 加算 b"), value=0.0, format="%g", key=f"sc_xoff_{lid}",
            help=t("例: K→°C は -273.15")))
        s["y_value_scale"] = float(c1.number_input(
            t("y 倍率 a"), value=1.0, format="%g", key=f"sc_ysc_{lid}",
            help=t("例: Pa→hPa は 0.01")))
        s["y_value_offset"] = float(c2.number_input(
            t("y 加算 b"), value=0.0, format="%g", key=f"sc_yoff_{lid}",
            help=t("例: K→°C は -273.15")))
        st.caption(t("変換後 = a × 元の値 + b。"
                    "x_lim/y_lim・凡例ラベル等は変換後の単位で入力してください。"))
    return layer


def map_scatter_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                         allow_averaging=False, panel=None):
    """水平断面図の散布図レイヤー (kind="map_scatter")。

    格子データ (lat×lon 次元) と地点データ (lon/lat が補助座標) の両方に対応。
    変数候補はこの UI で独自に集める (fill 等の格子変数リストとは別)。
    """
    dsid = _layer_dataset_selector(datasets, f"ms_{lid}")
    ds = datasets[dsid]
    ds_roles = mc_dataset.detect_coord_roles(ds)
    lat_name, lon_name = ds_roles["lat"], ds_roles["lon"]
    if not lat_name or not lon_name:
        st.warning(t("lon/lat 座標が無いため散布図を描けません"))
        return None
    candidates = mc_dataset.map_scatter_variables(ds, ds_roles)
    if not candidates:
        st.warning(t("`{dsid}` に lon/lat をもつ変数がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"ms_var_{lid}", candidates)
    var = st.selectbox(t("変数"), candidates, key=f"ms_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_map_scatter_layer(dsid, var)
    # 「点を成す次元」(格子: lat/lon、地点: lon/lat が張る次元) 以外を固定する。
    # 時刻は panel.selection 側で扱うので無視 (fill と同じ流儀)。
    hdims = mc_dataset.horizontal_dims(ds, ds_roles)
    if hdims is not None and all(d in ds[var].dims for d in hdims):
        # 格子 (1 次元格子 = lat/lon、curvilinear = y/x。render.map_scatter_points と同じ判定)
        point_dims = set(hdims)
    else:
        point_dims = set(ds[lon_name].dims) | set(ds[lat_name].dims)
    sel = {}
    for d in ds[var].dims:
        if (d in point_dims or d == ds_roles.get("time")
                or d not in ds.coords):
            continue
        vals = [float(v) for v in ds[d].values]
        sel[d] = st.selectbox(
            t("固定: {d}", d=d), vals, index=0, format_func=lambda v: f"{v:g}",
            key=f"ms_sel_{lid}_{d}")
    layer["selection"] = sel

    s = layer["style"]
    color_mode = st.radio(t("点の色付け"), list(_CMODE_LABELS),
                          format_func=tr_labels(_CMODE_LABELS).get, horizontal=True,
                          key=f"cmode_ms_{lid}",
                          help=t("カラーマップ: 変数の値に応じて点を色分けする"))
    if color_mode == "cmap":
        s["use_cmap"] = True
        cmap_section_ui(s, f"ms_{lid}",
                        lambda: transformed_range(f"ms_{lid}", float(ds[var].min()),
                                                  float(ds[var].max())))
    else:
        s["use_cmap"] = False
        s["color"] = color_selector(t("点の色"), "#1f77b4", key=f"ms_color_{lid}",
                                    meta_store=s, meta_key="color")
    s["marker"] = st.selectbox(t("マーカー"), list(_SCATTER_MARKER_LABELS),
                               format_func=tr_labels(_SCATTER_MARKER_LABELS).get,
                               key=f"ms_marker_{lid}")
    if st.checkbox(t("枠線を表示"), value=False, key=f"ms_edge_{lid}"):
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                         key=f"ms_edge_color_{lid}",
                                         meta_store=s, meta_key="edge_color")
        s["edge_linewidth"] = float(st.slider(
            t("枠線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"ms_edge_lw_{lid}"))
    else:
        s["edge_linewidth"] = 0.0
    s["size"] = float(st.slider(t("マーカーサイズ (s, 面積 pt²)"), 1.0, 200.0,
                                20.0, 1.0, key=f"ms_size_{lid}"))
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 1.0, 0.05,
                                 key=f"ms_alpha_{lid}"))
    if s["use_cmap"]:
        colorbar_ui(s["colorbar"], f"ms_{lid}",
                    ds[var].attrs.get("units", ""), label_suffix=f"_{var}")
    value_transform_ui(s, f"ms_{lid}")
    return layer


def _agg_fixed_ui(ds, x_var, y_var, p):
    """agg_2d 共通: x/y で共通でない次元を固定する UI (共通次元は全て集計)。"""
    common = set(ds[x_var].dims) & set(ds[y_var].dims)
    x_fixed, y_fixed = {}, {}
    if [d for d in ds[x_var].dims if d not in common]:
        st.markdown(t("**x 変数 `{var}` の固定** (y と共通でない次元)", var=x_var))
        x_fixed = _fix_dims_ui(ds, list(ds[x_var].dims), tuple(common),
                               f"{p}_x")
    if [d for d in ds[y_var].dims if d not in common]:
        st.markdown(t("**y 変数 `{var}` の固定** (x と共通でない次元)", var=y_var))
        y_fixed = _fix_dims_ui(ds, list(ds[y_var].dims), tuple(common),
                               f"{p}_y")
    if common:
        st.caption(t("集計する次元: {dims} (この次元に沿った全標本ペアを集計)",
                     dims=", ".join(f"`{d}`" for d in ds[x_var].dims
                                    if d in common)))
    return x_fixed, y_fixed


def _agg_xy_vars(ds, dsid, lid):
    """agg_2d レイヤー共通: panel の x/y 変数を取得しガードする。"""
    mode_key = lid.rsplit("_", 1)[0]
    x_var = st.session_state.get(f"panel_xvar_{mode_key}")
    y_var = st.session_state.get(f"panel_yvar_{mode_key}")
    if not x_var or not y_var:
        st.warning(t("「プロット変数」セクションで x/y 軸の変数を選んでください"))
        return None, None
    if x_var not in ds.data_vars or y_var not in ds.data_vars:
        st.warning(t("`{dsid}` には変数 `{x_var}` または `{y_var}` がありません",
                   dsid=dsid, x_var=x_var, y_var=y_var))
        return None, None
    return x_var, y_var


def hexbin_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                    allow_averaging=False, panel=None):
    """2次元プロット(集計) の hexbin (六角ビン密度) レイヤー。

    panel の x/y 変数の全標本ペア (共通次元を全て ravel) を六角ビンに集計する。
    2026-07-07 に散布図モードから移設 — 旧 drawing_dim 方式の設定は
    render/scriptgen が互換描画する (この UI では作られない)。
    """
    dsid = _layer_dataset_selector(datasets, f"hx_{lid}")
    ds = datasets[dsid]
    x_var, y_var = _agg_xy_vars(ds, dsid, lid)
    if not x_var:
        return None
    layer = mc_config.default_hexbin_layer(dsid)
    layer["x_fixed"], layer["y_fixed"] = _agg_fixed_ui(ds, x_var, y_var,
                                                       f"hx_{lid}")
    s = layer["style"]
    s["gridsize"] = int(st.slider(
        t("六角ビン数 (gridsize)"), 5, 60, 20, 1, key=f"hx_grid_{lid}",
        help=t("x 方向のビン数。大きいほど細かい")))
    if st.session_state.get(f"disc_hx_{lid}"):
        st.caption(t("点数の対数スケールは色の離散化と併用できません "
                   "(対数風の階級は離散化のレベル直接指定 1, 2, 5, 10, 20, 50 … で)"))
    else:
        s["log_counts"] = st.checkbox(
            t("点数を対数スケール (bins='log')"), value=False, key=f"hx_log_{lid}",
            help=t("桁の違うビンを同時に見る。色の離散化とは併用できない"))
    if st.checkbox(t("空のビンも塗る (mincnt=None)"), value=False,
                   key=f"hx_showempty_{lid}",
                   help=t("off で点が1つ以上あるビンだけ描く")):
        s["mincnt"] = None
    else:
        s["mincnt"] = int(st.number_input(
            t("最小点数 (mincnt)"), 1, 50, 1, key=f"hx_mincnt_{lid}"))
    # 点数の範囲は集計するまで不明なので既定値は固定 (0〜10)
    cmap_section_ui(s, f"hx_{lid}", lambda: (0.0, 10.0),
                    default_levels="0, 1, 2, 5, 10, 20",
                    log_exclusive_key=f"hx_log_{lid}")
    if st.checkbox(t("ビンの枠線を表示"), value=False, key=f"hx_edge_{lid}",
                   help=t("六角形の輪郭線。点が描かれるビンにだけ付く")):
        s["edge_width"] = float(st.slider(
            t("枠線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"hx_ew_{lid}"))
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                          key=f"hx_ec_{lid}",
                                          meta_store=s, meta_key="edge_color")
    colorbar_ui(s["colorbar"], f"hx_{lid}", "count")
    with st.expander(t("値の変換 (x/y 独立)"), expanded=False):
        c1, c2 = st.columns(2)
        s["x_value_scale"] = float(c1.number_input(
            t("x 倍率 a"), value=1.0, format="%g", key=f"hx_xsc_{lid}"))
        s["x_value_offset"] = float(c2.number_input(
            t("x 加算 b"), value=0.0, format="%g", key=f"hx_xoff_{lid}"))
        c3, c4 = st.columns(2)
        s["y_value_scale"] = float(c3.number_input(
            t("y 倍率 a"), value=1.0, format="%g", key=f"hx_ysc_{lid}"))
        s["y_value_offset"] = float(c4.number_input(
            t("y 加算 b"), value=0.0, format="%g", key=f"hx_yoff_{lid}"))
    return layer


def hist2d_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                    allow_averaging=False, panel=None):
    """2次元プロット(集計) の2次元ヒストグラムレイヤー (kind="hist2d")。"""
    dsid = _layer_dataset_selector(datasets, f"h2d_{lid}")
    ds = datasets[dsid]
    x_var, y_var = _agg_xy_vars(ds, dsid, lid)
    if not x_var:
        return None
    layer = mc_config.default_hist2d_layer(dsid)
    layer["x_fixed"], layer["y_fixed"] = _agg_fixed_ui(ds, x_var, y_var,
                                                       f"h2d_{lid}")
    s = layer["style"]
    cc1, cc2 = st.columns(2)
    s["bins_x"] = int(cc1.number_input(t("ビン数 x"), 2, 200, 20,
                                       key=f"h2d_bx_{lid}"))
    s["bins_y"] = int(cc2.number_input(t("ビン数 y"), 2, 200, 20,
                                       key=f"h2d_by_{lid}"))
    if st.checkbox(t("集計範囲を指定"), value=False, key=f"h2d_rng_{lid}",
                   help=t("off でデータの min/max から等間隔に分割 (x/y まとめて指定)")):
        c1, c2 = st.columns(2)
        s["range_x"] = [
            float(c1.number_input(t("x 最小"), value=0.0, format="%g",
                                  key=f"h2d_rx0_{lid}")),
            float(c2.number_input(t("x 最大"), value=1.0, format="%g",
                                  key=f"h2d_rx1_{lid}"))]
        c3, c4 = st.columns(2)
        s["range_y"] = [
            float(c3.number_input(t("y 最小"), value=0.0, format="%g",
                                  key=f"h2d_ry0_{lid}")),
            float(c4.number_input(t("y 最大"), value=1.0, format="%g",
                                  key=f"h2d_ry1_{lid}"))]
    _cnt = st.radio(t("集計"), list(_DENSITY_LABELS),
                    format_func=tr_labels(_DENSITY_LABELS).get, horizontal=True,
                    key=f"h2d_den_{lid}",
                    help=t("確率密度: 全体の積分が 1 になるよう正規化"))
    s["density"] = _cnt == "density"
    s["hide_zeros"] = st.checkbox(t("0 のビンを塗らない"), value=True,
                                  key=f"h2d_hz_{lid}")
    if st.session_state.get(f"disc_h2d_{lid}"):
        st.caption(t("度数の対数スケールは色の離散化と併用できません "
                   "(対数風の階級は離散化のレベル直接指定 1, 2, 5, 10, 20, 50 … で)"))
    else:
        s["log_counts"] = st.checkbox(
            t("度数を対数スケール (LogNorm)"), value=False, key=f"h2d_log_{lid}",
            help=t("桁の違うビンを同時に見る。0 のビンは表示されない。"
                 "色の離散化とは併用できない"))
    # 度数の範囲は集計するまで不明なので既定値は固定 (0〜10)
    cmap_section_ui(s, f"h2d_{lid}", lambda: (0.0, 10.0),
                    default_levels="0, 1, 2, 5, 10, 20",
                    log_exclusive_key=f"h2d_log_{lid}")
    if st.checkbox(t("ビンの枠線を表示"), value=False, key=f"h2d_edge_{lid}",
                   help=t("ビンの矩形の輪郭線。「0 のビンを塗らない」で"
                        "マスクされたビンには付かない")):
        s["edge_width"] = float(st.slider(
            t("枠線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"h2d_ew_{lid}"))
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                          key=f"h2d_ec_{lid}",
                                          meta_store=s, meta_key="edge_color")
    colorbar_ui(s["colorbar"], f"h2d_{lid}", "count")
    with st.expander(t("値の変換 (x/y 独立)"), expanded=False):
        c1, c2 = st.columns(2)
        s["x_value_scale"] = float(c1.number_input(
            t("x 倍率 a"), value=1.0, format="%g", key=f"h2d_xsc_{lid}"))
        s["x_value_offset"] = float(c2.number_input(
            t("x 加算 b"), value=0.0, format="%g", key=f"h2d_xoff_{lid}"))
        c3, c4 = st.columns(2)
        s["y_value_scale"] = float(c3.number_input(
            t("y 倍率 a"), value=1.0, format="%g", key=f"h2d_ysc_{lid}"))
        s["y_value_offset"] = float(c4.number_input(
            t("y 加算 b"), value=0.0, format="%g", key=f"h2d_yoff_{lid}"))
    return layer


def bubble_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                    allow_averaging=False, panel=None):
    """2次元プロット用のバブルチャート (ax.scatter + 可変サイズ)。

    panel.x_variable / y_variable / z_variable を session_state 経由で読み、
    drawing_dim 沿いに散布する。z は [size_min, size_max] に正規化されてマーカー
    サイズになる。x/y/z それぞれで固定する次元を独立に選べる。
    """
    dsid = _layer_dataset_selector(datasets, f"bb_{lid}")
    ds = datasets[dsid]
    mode_key = lid.rsplit("_", 1)[0]  # lid = f"{mode_key}_{n}"
    x_var = st.session_state.get(f"panel_xvar_{mode_key}")
    y_var = st.session_state.get(f"panel_yvar_{mode_key}")
    z_var = st.session_state.get(f"panel_zvar_{mode_key}")
    if z_var == _NONE_OPTION:
        z_var = None
    if not x_var or not y_var:
        st.warning(t("「プロット変数」セクションで x/y 軸の変数を選んでください"))
        return None
    if not z_var:
        st.warning(t("バブルチャートには z 軸の変数も必要です。"
                    "「プロット変数」セクションで z を選んでください"))
        return None
    if any(v not in ds.data_vars for v in [x_var, y_var, z_var]):
        st.warning(t("`{dsid}` に必要な変数の一部 (x/y/z) がありません", dsid=dsid))
        return None
    x_dims = [d for d in ds[x_var].dims if d in ds.coords]
    y_dims = [d for d in ds[y_var].dims if d in ds.coords]
    z_dims = [d for d in ds[z_var].dims if d in ds.coords]
    common_dims = [d for d in x_dims if d in y_dims and d in z_dims]
    if not common_dims:
        st.warning(t("`{x_var}` / `{y_var}` / `{z_var}` に共通の座標次元がありません",
                   x_var=x_var, y_var=y_var, z_var=z_var))
        return None
    drawing_dim = st.selectbox(t("描画する次元"), common_dims,
                                  key=f"bb_drawing_dim_{lid}",
                                  help=t("この次元に沿って散布する。範囲スライダで絞れる"))
    drawing_range = _scatter_drawing_range(ds, drawing_dim, f"bb_{lid}")
    layer = mc_config.default_bubble_layer(dsid)
    layer["drawing_dim"] = drawing_dim
    layer["drawing_range"] = drawing_range
    st.markdown(t("**x 軸の変数 `{var}` で固定する次元**", var=x_var))
    layer["x_fixed"] = _scatter_fixed_picker(
        ds, x_dims, drawing_dim, f"bb_x_{lid}")
    st.markdown(t("**y 軸の変数 `{var}` で固定する次元**", var=y_var))
    layer["y_fixed"] = _scatter_fixed_picker(
        ds, y_dims, drawing_dim, f"bb_y_{lid}")
    st.markdown(t("**z 軸の変数 `{var}` で固定する次元**", var=z_var))
    layer["z_fixed"] = _scatter_fixed_picker(
        ds, z_dims, drawing_dim, f"bb_z_{lid}")

    s = layer["style"]
    s["marker"] = st.selectbox(t("マーカー"), list(_SCATTER_MARKER_LABELS),
                               format_func=tr_labels(_SCATTER_MARKER_LABELS).get,
                               key=f"bb_marker_{lid}")
    c1, c2 = st.columns(2)
    s["size_min"] = float(c1.number_input(
        t("最小サイズ (pt²)"), value=10.0, min_value=1.0, max_value=2000.0,
        step=1.0, key=f"bb_smin_{lid}",
        help=t("z の最小値に対応するマーカー面積")))
    s["size_max"] = float(c2.number_input(
        t("最大サイズ (pt²)"), value=200.0, min_value=1.0, max_value=5000.0,
        step=10.0, key=f"bb_smax_{lid}",
        help=t("z の最大値に対応するマーカー面積")))
    if st.checkbox(t("枠線を表示"), value=False, key=f"bb_edge_{lid}"):
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                           key=f"bb_ecolor_{lid}",
                                           meta_store=s, meta_key="edge_color")
        s["edge_linewidth"] = float(st.slider(
            t("枠線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"bb_elw_{lid}"))
    else:
        s["edge_linewidth"] = 0.0
    # 色: 単色 or z 値で cmap
    s["use_cmap"] = st.checkbox(
        t("z の値で色を変える (cmap)"), value=False, key=f"bb_usecmap_{lid}",
        help=t("on で各バブルの色を z の値に応じて cmap でマッピングする"))
    if s["use_cmap"]:
        cmap_section_ui(s, f"bb_{lid}",
                        lambda: (float(ds[z_var].min()), float(ds[z_var].max())))
    else:
        s["color"] = color_selector(t("色"), "#1f77b4", key=f"bb_color_{lid}",
                                     meta_store=s, meta_key="color")
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 0.6, 0.05,
                                   key=f"bb_alpha_{lid}"))
    label = st.text_input(t("凡例ラベル"), value="", key=f"bb_label_{lid}",
                           placeholder=t("(空欄なら凡例に出さない)"))
    s["label"] = label or None
    with st.expander(t("エラーバー"), expanded=False):
        _eb_vars = [_NONE_OPTION] + list(variables_by_ds.get(dsid, []))
        eb = s["errorbar"]
        _ebx = st.selectbox(t("x方向の誤差の変数"), _eb_vars, key=f"bb_ebx_{lid}",
                            format_func=_none_option_labels(_eb_vars).get,
                            help=t("各点の x 方向の誤差 (±値、対称)。本体と同じ"
                                 "次元固定・範囲で切り出す。値の変換は掛けないので、"
                                 "図に表示する単位で用意する"))
        _eby = st.selectbox(t("y方向の誤差の変数"), _eb_vars, key=f"bb_eby_{lid}",
                            format_func=_none_option_labels(_eb_vars).get,
                            help=t("各点の y 方向の誤差 (±値、対称)。値の変換は掛けない"))
        eb["x_variable"] = None if _ebx == _NONE_OPTION else _ebx
        eb["y_variable"] = None if _eby == _NONE_OPTION else _eby
        if eb["x_variable"] or eb["y_variable"]:
            eb["color"] = color_selector(t("エラーバーの色"), "#000000",
                                          key=f"bb_ebc_{lid}",
                                          meta_store=eb, meta_key="color")
            eb["linewidth"] = float(st.slider(
                t("エラーバーの太さ"), 0.2, 3.0, 1.0, 0.1, key=f"bb_ebw_{lid}"))
            eb["capsize"] = float(st.slider(
                t("キャップ幅 (pt)"), 0.0, 10.0, 3.0, 0.5, key=f"bb_ebcap_{lid}"))
    if s["use_cmap"]:
        colorbar_ui(s["colorbar"], f"bb_{lid}", "", label_suffix=f"_{z_var}")
    with st.expander(t("値の変換 (x/y/z 独立、単位換算など)"), expanded=False):
        c1, c2 = st.columns(2)
        s["x_value_scale"] = float(c1.number_input(
            t("x 倍率 a"), value=1.0, format="%g", key=f"bb_xsc_{lid}",
            help=t("例: Pa→hPa は 0.01")))
        s["x_value_offset"] = float(c2.number_input(
            t("x 加算 b"), value=0.0, format="%g", key=f"bb_xoff_{lid}",
            help=t("例: K→°C は -273.15")))
        s["y_value_scale"] = float(c1.number_input(
            t("y 倍率 a"), value=1.0, format="%g", key=f"bb_ysc_{lid}",
            help=t("例: Pa→hPa は 0.01")))
        s["y_value_offset"] = float(c2.number_input(
            t("y 加算 b"), value=0.0, format="%g", key=f"bb_yoff_{lid}",
            help=t("例: K→°C は -273.15")))
        s["z_value_scale"] = float(c1.number_input(
            t("z 倍率 a"), value=1.0, format="%g", key=f"bb_zsc_{lid}"))
        s["z_value_offset"] = float(c2.number_input(
            t("z 加算 b"), value=0.0, format="%g", key=f"bb_zoff_{lid}"))
        st.caption(t("変換後 = a × 元の値 + b。"
                    "z は変換後の値を [最小サイズ, 最大サイズ] に線形正規化される。"))
    return layer


_HISTTYPE_LABELS = {"bar": "塗り (bar)", "step": "階段 (step, 線のみ)",
                    "stepfilled": "階段塗り (stepfilled)"}
# ヒストグラムの向き (style.orientation)。bar レイヤーの _BAR_ORIENT_LABELS とは別
# (ax.hist の orientation。横では値が y 軸・度数が x 軸)
_HIST_ORIENT_LABELS = {"vertical": "縦 (値が x 軸)", "horizontal": "横 (値が y 軸)"}


def _fix_dims_ui(ds, dims, skip, p):
    """指定した次元以外を固定する selection を作る (集計系レイヤー共用)。

    時刻座標は ISO 文字列、座標なし (bare dim) は index で固定する。
    p は widget key の接頭辞。
    """
    sel = {}
    for d in dims:
        if d in skip:
            continue
        if d in ds.coords:
            if np.issubdtype(ds[d].dtype, np.datetime64):
                _lbls, _vals = time_labels_and_values(ds[d].values)
                _lbl = st.selectbox(t("固定: {d}", d=d), _lbls, key=f"{p}_fix_{d}")
                sel[d] = _vals[_lbls.index(_lbl)]
            else:
                _vals = [x.item() for x in ds[d].values]
                sel[d] = st.selectbox(
                    t("固定: {d}", d=d), _vals,
                    format_func=lambda v: (f"{v:g}" if isinstance(v, (int, float))
                                            else str(v)),
                    key=f"{p}_fix_{d}")
        else:
            sel[d] = int(st.number_input(
                t("固定: {d} (index)", d=d), 0, int(ds.sizes[d]) - 1, 0,
                key=f"{p}_fix_{d}"))
    return sel


def _dist_sampling_ui(ds, var, layer, p):
    """集計系レイヤー (hist / ecdf) 共通のデータ取り: 集計する次元・範囲・固定。"""
    dims = list(ds[var].dims)
    if len(dims) == 1:
        agg_dim = dims[0]
        st.caption(t("集計する次元: `{dim}` (全 {n} 点)",
                   dim=agg_dim, n=int(ds.sizes[agg_dim])))
    else:
        agg_dim = st.selectbox(t("集計する次元"), dims, key=f"{p}_aggdim",
                               help=t("この次元に沿った全値を集計する。"
                                    "それ以外の次元は固定"))
    layer["agg_dim"] = agg_dim
    if agg_dim in ds.coords and st.checkbox(
            t("集計する範囲を指定"), value=False, key=f"{p}_aggrng_on",
            help=t("off で全範囲。期間や区間を絞って集計できる")):
        layer["agg_range"] = _scatter_drawing_range(ds, agg_dim, p)
    layer["selection"] = _fix_dims_ui(ds, dims, (agg_dim,), p)


def hist_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                  allow_averaging=False, panel=None):
    """1次元プロット(集計) のヒストグラムレイヤー (kind="hist")。

    variables_by_ds には lat/lon 次元を持たない変数だけが渡ってくる
    (dataset.nongeo_variables)。集計する次元以外は selection で固定する。
    """
    dsid = _layer_dataset_selector(datasets, f"hist_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には集計できる変数 (緯度経度次元なし) がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"hist_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"hist_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_hist_layer(dsid)
    layer["variable"] = var
    _dist_sampling_ui(ds, var, layer, f"hist_{lid}")
    s = layer["style"]
    # ビン (fill レイヤーの「レベルを等間隔にする」と同じ流儀)
    if st.checkbox(t("ビンを等間隔にする"), value=True, key=f"hist_bineq_{lid}",
                   help=t("off にするとビン境界をカンマ区切りで直接指定できる "
                        "(不等間隔可)")):
        if st.checkbox(t("値の範囲を指定"), value=False, key=f"hist_rangeon_{lid}",
                       help=t("off でデータの min/max から等間隔に分割")):
            cc1, cc2 = st.columns(2)
            s["range"] = [
                float(cc1.number_input(t("最小値"), value=0.0, format="%g",
                                       key=f"hist_rmin_{lid}")),
                float(cc2.number_input(t("最大値"), value=1.0, format="%g",
                                       key=f"hist_rmax_{lid}"))]
        c_nb, c_bw = st.columns(2)
        s["bins"] = int(c_nb.number_input(t("ビン数"), 2, 200, 20,
                                          key=f"hist_nbin_{lid}"))
        # ビンの幅の参考表示 (number_input の入力欄の高さに合わせて 1 行下げる)。
        # 値はレイヤー完成後に _write_hist_bin_width() が埋める
        c_bw.markdown("&nbsp;")
        bw_slot = c_bw.empty()
    else:
        bw_slot = None
        _bl = parse_float_list(st.text_input(
            t("ビン境界 (カンマ区切り)"), value="0, 1, 2, 5, 10",
            key=f"hist_binedges_{lid}",
            help=t("昇順に整列・重複除去される。2個以上。不等間隔可。"
                 "指定時は「値の範囲」は使われない")))
        if _bl and len(set(_bl)) >= 2:
            s["bins"] = _bl
        else:
            st.warning(t("ビン境界は2個以上の数値をカンマ区切りで入力してください"))
    _ymode = st.radio(t("縦軸"), list(_DENSITY_LABELS),
                      format_func=tr_labels(_DENSITY_LABELS).get, horizontal=True,
                      key=f"hist_den_{lid}",
                      help=t("確率密度: 面積の合計が 1 になるよう正規化"))
    s["density"] = _ymode == "density"
    s["cumulative"] = st.checkbox(t("累積にする"), value=False,
                                  key=f"hist_cum_{lid}")
    s["histtype"] = st.selectbox(
        t("描き方"), list(_HISTTYPE_LABELS), format_func=tr_labels(_HISTTYPE_LABELS).get,
        key=f"hist_httype_{lid}",
        help=t("塗り (bar) = ビン毎の独立した矩形、階段塗り (stepfilled) = 外郭を"
             "なぞった一筆書きの多角形。**枠線なしでは見た目はほぼ同一**で、"
             "枠線を付けると bar はビンの間にも縦線が入り、stepfilled は"
             "輪郭だけになる。階段 (step) は塗りなしの輪郭線のみ"))
    s["orientation"] = st.radio(
        t("向き"), list(_HIST_ORIENT_LABELS),
        format_func=tr_labels(_HIST_ORIENT_LABELS).get, horizontal=True,
        key=f"hist_orient_{lid}",
        help=t("横にすると値が y 軸・度数が x 軸になる (自動の軸ラベルも入れ替わる)。"
             "同じパネルの ECDF・ラインは縦向きのまま"))
    # 棒の幅 (ビン幅に対する比) は塗り (bar) のときだけ (step 系では matplotlib が無視)
    if s["histtype"] == "bar" and st.checkbox(
            t("棒の幅を手動指定"), value=False, key=f"hist_rw_manual_{lid}",
            help=t("off だとビンいっぱいに描く (matplotlib 既定)。"
                 "描き方が「塗り (bar)」のときだけ有効")):
        s["rwidth"] = float(st.slider(t("幅 (ビン幅に対する比)"), 0.1, 1.0, 0.8, 0.05,
                                      key=f"hist_rw_{lid}"))
    if not st.checkbox(t("色を自動 (カラーサイクル)"), value=True,
                       key=f"hist_autocol_{lid}"):
        s["color"] = color_selector(t("色"), "#1f77b4", key=f"hist_col_{lid}",
                                    meta_store=s, meta_key="color")
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 0.7, 0.05,
                                 key=f"hist_alpha_{lid}",
                                 help=t("重ね描きを想定して既定は半透明")))
    if st.checkbox(t("枠線を表示"), value=False, key=f"hist_edge_{lid}"):
        s["edge_color"] = color_selector(t("枠線の色"), "#000000",
                                         key=f"hist_ecol_{lid}",
                                         meta_store=s, meta_key="edge_color")
        s["edge_linewidth"] = float(st.slider(t("枠線の太さ"), 0.1, 3.0, 0.8, 0.1,
                                              key=f"hist_elw_{lid}"))
    else:
        s["edge_linewidth"] = 0.0
    s["label"] = _legend_label_ui(f"hist_label_{lid}_{var}", var)
    value_transform_ui(s, f"hist_{lid}")
    _write_hist_bin_width(bw_slot, layer, datasets)
    return layer


def _write_hist_bin_width(slot, layer, datasets):
    """hist_layer_ui の「ビン数」の横の placeholder に等間隔ビンの幅を書く。

    layer が完成した後 (値の変換の widget 描画後) に呼ぶ。範囲は「値の範囲を指定」
    の最小値・最大値、無指定なら render と同じ値 (dist_values = 固定・範囲制限・
    値の変換後の有限値) の min/max — matplotlib の ax.hist が range=None で使う
    もの (min == max なら numpy.histogram と同じく ±0.5 に広げる)。ビン境界の
    直接指定 (slot が None) では何も出さない。ビン数・範囲・データの選択を変える
    たびに再計算される (2026-10-07)。
    """
    if slot is None:
        return
    s = layer["style"]
    nbins = s.get("bins")
    if not isinstance(nbins, int) or nbins < 1:
        return
    rng = s.get("range")
    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
        if not hi > lo:
            slot.caption(t("ビンの幅: — (最大値は最小値より大きくしてください)"))
            return
        slot.caption(t("ビンの幅: {w:g}", w=(hi - lo) / nbins))
        return
    vals = mc_render.dist_values(layer, datasets)
    if vals.size == 0:
        slot.caption(t("ビンの幅: — (値がありません)"))
        return
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    slot.caption(t("ビンの幅: {w:g} (データ範囲 {lo:g} 〜 {hi:g})",
                   w=(hi - lo) / nbins, lo=lo, hi=hi))


def ecdf_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                  allow_averaging=False, panel=None):
    """1次元プロット(集計) の ECDF レイヤー (kind="ecdf"、matplotlib 3.8+)。

    データの取り方 (集計する次元・範囲・固定) は hist と共通。
    度数ヒストグラムと重ねるときは「第2軸に描く」を ON にする (ECDF は 0〜1)。
    """
    dsid = _layer_dataset_selector(datasets, f"ecdf_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には集計できる変数 (緯度経度次元なし) がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"ecdf_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"ecdf_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_ecdf_layer(dsid)
    layer["variable"] = var
    _dist_sampling_ui(ds, var, layer, f"ecdf_{lid}")
    s = layer["style"]
    s["complementary"] = st.checkbox(
        t("補分布 (1 − CDF) にする"), value=False, key=f"ecdf_comp_{lid}",
        help=t("値がしきい値以上になる割合 (超過確率) を描く"))
    if not st.checkbox(t("色を自動 (カラーサイクル)"), value=True,
                       key=f"ecdf_autocol_{lid}"):
        s["color"] = color_selector(t("線の色"), "#1f77b4", key=f"ecdf_col_{lid}",
                                    meta_store=s, meta_key="color")
    s["linewidth"] = float(st.slider(t("線の太さ"), 0.5, 4.0, 1.5, 0.1,
                                     key=f"ecdf_lw_{lid}"))
    s["linestyle"] = st.selectbox(t("線種"), list(LINESTYLE_LABELS),
                                  format_func=tr_labels(LINESTYLE_LABELS).get,
                                  key=f"ecdf_ls_{lid}")
    s["secondary_y"] = st.checkbox(
        t("第2軸 (右の縦軸) に描く"), value=False, key=f"ecdf_y2_{lid}",
        help=t("度数ヒストグラムと重ねるとき用 (ECDF は 0〜1 なのでスケールが違う)"))
    s["label"] = _legend_label_ui(f"ecdf_label_{lid}_{var}", var)
    value_transform_ui(s, f"ecdf_{lid}")
    return layer


def box_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                 allow_averaging=False, panel=None):
    """1次元プロット(集計) の箱ひげ図レイヤー (kind="box")。1レイヤー = 1系列。"""
    dsid = _layer_dataset_selector(datasets, f"box_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には集計できる変数 (緯度経度次元なし) がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"box_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"box_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_box_layer(dsid)
    layer["variable"] = var
    _dist_sampling_ui(ds, var, layer, f"box_{lid}")
    s = layer["style"]
    s["label"] = st.text_input(t("系列名 (x 目盛に表示)"), value=var,
                               key=f"box_label_{lid}_{var}") or None
    s["in_legend"] = st.checkbox(
        t("凡例に載せる"), value=False, key=f"box_leg_{lid}",
        help=t("系列名を凡例にも表示する (色で系列を区別したいとき用)。"
             "凡例自体の表示・位置は「軸・ラベル・凡例 > 凡例」"))
    s["width"] = float(st.slider(t("箱の幅"), 0.1, 1.0, 0.5, 0.05,
                                 key=f"box_w_{lid}"))
    _whis_mode = st.radio(t("ひげの定義"), list(_WHIS_MODE_LABELS),
                          format_func=tr_labels(_WHIS_MODE_LABELS).get,
                          horizontal=True, key=f"box_whismode_{lid}",
                          help=t("パーセンタイル範囲: ひげを指定パーセンタイル点に"
                               "固定する (例 5–95%)。範囲外は外れ値"))
    if _whis_mode == "iqr":
        s["whis"] = float(st.number_input(
            t("ひげの長さ (whis, IQR 倍率)"), 0.0, 10.0, 1.5, 0.5,
            key=f"box_whis_{lid}",
            help=t("四分位範囲の何倍までをひげにするか (matplotlib 既定 1.5)。"
                 "超える点は外れ値として描かれる")))
    else:
        cc1, cc2 = st.columns(2)
        _plo = float(cc1.number_input(t("下 (%)"), 0.0, 50.0, 5.0, 1.0,
                                      key=f"box_whis_lo_{lid}"))
        _phi = float(cc2.number_input(t("上 (%)"), 50.0, 100.0, 95.0, 1.0,
                                      key=f"box_whis_hi_{lid}"))
        s["whis"] = [_plo, _phi]
    s["notch"] = st.checkbox(
        t("ノッチ (中央値の信頼区間の切れ込み)"), value=False,
        key=f"box_notch_{lid}",
        help=t("切れ込み幅 ≈ 中央値の信頼区間 (1.57×IQR/√N)。2系列のノッチが"
             "重ならなければ中央値の差が有意という視覚的目安。bootstrap 版 CI は"
             "乱数で再現性が壊れるため非対応"))
    s["showfliers"] = st.checkbox(t("外れ値を表示"), value=True,
                                  key=f"box_fliers_{lid}")
    s["showcaps"] = st.checkbox(t("ひげ先端の横棒 (キャップ)"), value=True,
                                key=f"box_caps_{lid}")
    if s["showcaps"] and st.checkbox(t("キャップ幅を指定"), value=False,
                                     key=f"box_capw_on_{lid}"):
        s["capwidths"] = float(st.slider(
            t("キャップ幅"), 0.05, 1.0, 0.25, 0.05, key=f"box_capw_{lid}",
            help=t("箱の幅と同じ単位")))
    s["showbox"] = st.checkbox(t("箱を表示"), value=True,
                               key=f"box_showbox_{lid}",
                               help=t("off でひげと中央値だけの図になる"))
    s["showmeans"] = st.checkbox(t("平均値を表示"), value=False,
                                 key=f"box_means_{lid}")
    if s["showmeans"]:
        s["meanline"] = st.checkbox(
            t("平均を線で描く (meanline)"), value=False,
            key=f"box_meanline_{lid}", help=t("off はマーカー (▲)"))
    if st.checkbox(t("箱を塗る"), value=False, key=f"box_fill_{lid}"):
        s["fill_color"] = color_selector(t("塗り色"), "#1f77b4",
                                         key=f"box_fillcol_{lid}",
                                         meta_store=s, meta_key="fill_color")
    if st.checkbox(t("元データの点を重ねる (strip)"), value=False,
                   key=f"box_strip_{lid}",
                   help=t("集計の元になった個々のデータ点を系列位置に散らして描く。"
                        "散らし (ジッター) は固定シードで再現可能。"
                        "「外れ値を表示」と点が重複するので OFF 推奨")):
        s["show_points"] = True
        s["point_jitter"] = float(st.slider(
            t("散らし幅 (ジッター)"), 0.0, 0.4, 0.2, 0.02, key=f"box_ptj_{lid}",
            help=t("系列軸方向の広がり。0 で一直線")))
        s["point_size"] = float(st.slider(t("点の大きさ (s)"), 1.0, 80.0, 6.0, 1.0,
                                          key=f"box_ptsz_{lid}"))
        s["point_color"] = color_selector(t("点の色"), "#555555",
                                          key=f"box_ptc_{lid}",
                                          meta_store=s, meta_key="point_color")
        s["point_alpha"] = float(st.slider(t("点の透明度"), 0.05, 1.0, 0.4, 0.05,
                                           key=f"box_pta_{lid}"))
    with st.expander(t("体裁の詳細"), expanded=False):
        if st.checkbox(t("線 (箱・ひげ・キャップ) の色・太さを指定"), value=False,
                       key=f"box_lp_on_{lid}"):
            s["line_color"] = color_selector(t("線の色"), "#000000",
                                             key=f"box_lc_{lid}",
                                             meta_store=s, meta_key="line_color")
            s["line_width"] = float(st.slider(t("線の太さ"), 0.5, 4.0, 1.0, 0.1,
                                              key=f"box_lw_{lid}"))
        if st.checkbox(t("中央値線の色・太さを指定"), value=False,
                       key=f"box_mp_on_{lid}"):
            s["median_color"] = color_selector(
                t("中央値線の色"), "#d62728", key=f"box_mc_{lid}",
                meta_store=s, meta_key="median_color")
            s["median_width"] = float(st.slider(
                t("中央値線の太さ"), 0.5, 4.0, 1.5, 0.1, key=f"box_mw_{lid}"))
        if s["showmeans"] and st.checkbox(t("平均の色・太さを指定"), value=False,
                                          key=f"box_meanc_on_{lid}"):
            s["mean_color"] = color_selector(
                t("平均の色"), "#2ca02c", key=f"box_meanc_{lid}",
                meta_store=s, meta_key="mean_color")
            if s["meanline"]:
                s["mean_width"] = float(st.slider(
                    t("平均線の太さ"), 0.5, 4.0, 1.5, 0.1,
                    key=f"box_meanw_{lid}"))
        if s["showfliers"] and st.checkbox(t("外れ値マーカーを指定"), value=False,
                                           key=f"box_fp_on_{lid}"):
            s["flier_marker"] = st.selectbox(
                t("マーカー"), list(_SCATTER_MARKER_LABELS),
                format_func=tr_labels(_SCATTER_MARKER_LABELS).get, key=f"box_fmark_{lid}")
            s["flier_size"] = float(st.slider(
                t("マーカーサイズ (pt)"), 1.0, 15.0, 6.0, 0.5,
                key=f"box_fsize_{lid}"))
            s["flier_color"] = color_selector(
                t("マーカー色"), "#555555", key=f"box_fcol_{lid}",
                meta_store=s, meta_key="flier_color")
    value_transform_ui(s, f"box_{lid}")
    return layer


def violin_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                    allow_averaging=False, panel=None):
    """1次元プロット(集計) のバイオリンレイヤー (kind="violin")。1レイヤー = 1系列。"""
    dsid = _layer_dataset_selector(datasets, f"vio_{lid}")
    ds = datasets[dsid]
    variables = variables_by_ds[dsid]
    if not variables:
        st.warning(t("`{dsid}` には集計できる変数 (緯度経度次元なし) がありません", dsid=dsid))
        return None
    _reset_invalid_var_key(f"vio_var_{lid}", variables)
    var = st.selectbox(t("変数"), variables, key=f"vio_var_{lid}",
                       format_func=lambda v: var_label(ds, v))
    layer = mc_config.default_violin_layer(dsid)
    layer["variable"] = var
    _dist_sampling_ui(ds, var, layer, f"vio_{lid}")
    s = layer["style"]
    s["label"] = st.text_input(t("系列名 (x 目盛に表示)"), value=var,
                               key=f"vio_label_{lid}_{var}") or None
    s["in_legend"] = st.checkbox(
        t("凡例に載せる"), value=False, key=f"vio_leg_{lid}",
        help=t("系列名を凡例にも表示する。凡例自体の表示・位置は"
             "「軸・ラベル・凡例 > 凡例」"))
    s["width"] = float(st.slider(t("幅"), 0.1, 1.0, 0.7, 0.05,
                                 key=f"vio_w_{lid}"))
    c1, c2, c3 = st.columns(3)
    s["showmedians"] = c1.checkbox(t("中央値"), value=True, key=f"vio_med_{lid}")
    s["showmeans"] = c2.checkbox(t("平均値"), value=False, key=f"vio_mean_{lid}")
    s["showextrema"] = c3.checkbox(t("極値 (ひげ)"), value=True,
                                   key=f"vio_ext_{lid}")
    if st.checkbox(t("分位数の線を引く"), value=False, key=f"vio_q_on_{lid}"):
        _qs = parse_float_list(st.text_input(
            t("分位数 (0〜1, カンマ区切り)"), value="0.05, 0.95",
            key=f"vio_q_{lid}",
            help=t("例: 0.05, 0.95 で 5–95% の位置に横線を引く")))
        if _qs and all(0.0 <= q <= 1.0 for q in _qs):
            s["quantiles"] = _qs
        else:
            st.warning(t("0〜1 の数値をカンマ区切りで入力してください"))
    _bw_mode = st.radio(t("KDE バンド幅"), list(_BW_MODE_LABELS),
                        format_func=tr_labels(_BW_MODE_LABELS).get,
                        horizontal=True, key=f"vio_bwmode_{lid}",
                        help=t("分布の形の滑らかさ。数値が小さいほど凹凸が細かい"))
    if _bw_mode == "silverman":
        s["bw_method"] = "silverman"
    elif _bw_mode == "value":
        s["bw_method"] = float(st.number_input(
            t("バンド幅係数"), 0.01, 2.0, 0.3, 0.05, key=f"vio_bw_{lid}"))
    s["points"] = int(st.number_input(
        t("KDE 評価点数"), 20, 1000, 100, 10, key=f"vio_pts_{lid}",
        help=t("形の解像度 (matplotlib 既定 100)")))
    s["side"] = st.radio(
        t("形"), list(_SIDE_LABELS), format_func=tr_labels(_SIDE_LABELS).get,
        horizontal=True, key=f"vio_side_{lid}",
        help=t("半バイオリン: 2系列を同じ位置に向かい合わせて比較する表現に使う"))
    if not st.checkbox(t("色を自動"), value=True, key=f"vio_autocol_{lid}"):
        s["color"] = color_selector(t("色"), "#1f77b4", key=f"vio_col_{lid}",
                                    meta_store=s, meta_key="color")
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 0.5, 0.05,
                                 key=f"vio_alpha_{lid}",
                                 help=t("本体 (KDE 形状) の透明度")))
    if st.checkbox(t("元データの点を重ねる (strip)"), value=False,
                   key=f"vio_strip_{lid}",
                   help=t("集計の元になった個々のデータ点を系列位置に散らして描く。"
                        "散らし (ジッター) は固定シードで再現可能")):
        s["show_points"] = True
        s["point_jitter"] = float(st.slider(
            t("散らし幅 (ジッター)"), 0.0, 0.4, 0.2, 0.02, key=f"vio_ptj_{lid}",
            help=t("系列軸方向の広がり。0 で一直線")))
        s["point_size"] = float(st.slider(t("点の大きさ (s)"), 1.0, 80.0, 6.0, 1.0,
                                          key=f"vio_ptsz_{lid}"))
        s["point_color"] = color_selector(t("点の色"), "#555555",
                                          key=f"vio_ptc_{lid}",
                                          meta_store=s, meta_key="point_color")
        s["point_alpha"] = float(st.slider(t("点の透明度"), 0.05, 1.0, 0.4, 0.05,
                                           key=f"vio_pta_{lid}"))
    value_transform_ui(s, f"vio_{lid}")
    return layer


def track_layer_ui(datasets, variables_by_ds, keep_dims, lid, roles=None,
                   allow_averaging=False, panel=None):
    """水平断面図のトラック (軌跡) レイヤー (kind="track")。

    IBTrACS netCDF のような「storm × 時刻」の lat/lon 変数をもつベストトラックを
    線で結び、観測点を強度 (風速・気圧など) で色付けできる。
    """
    dsid = _layer_dataset_selector(datasets, f"tr_{lid}")
    ds = datasets[dsid]
    lon_c, lat_c = mc_dataset.track_lonlat_candidates(ds)
    if not lon_c or not lat_c:
        st.warning(t("経度・緯度の変数 (units が degrees_east / degrees_north、"
                   "または名前に lon / lat を含む) が見つかりません"))
        return None
    layer = mc_config.default_track_layer(dsid)
    c1, c2 = st.columns(2)
    _reset_invalid_var_key(f"tr_lon_{lid}", lon_c)
    _reset_invalid_var_key(f"tr_lat_{lid}", lat_c)
    layer["lon_var"] = c1.selectbox(t("経度の変数"), lon_c, key=f"tr_lon_{lid}")
    layer["lat_var"] = c2.selectbox(t("緯度の変数"), lat_c, key=f"tr_lat_{lid}")
    dims = list(ds[layer["lon_var"]].dims)
    if len(dims) >= 2:
        sdim = st.selectbox(
            t("トラックを区別する次元"), dims, index=0, key=f"tr_sdim_{lid}",
            help=t("ストーム番号などの次元。残りの次元が軌跡 (時刻) 方向になる"))
        layer["storm_dim"] = sdim
        n = int(ds.sizes[sdim])
        # 発生年・月で選ぶには datetime64 の時刻変数 (経度と同じ次元) が必要
        tcands = [str(name) for name, v in ds.variables.items()
                  if v.dims == ds[layer["lon_var"]].dims
                  and np.issubdtype(v.dtype, np.datetime64)]
        sel_opts = ((["by_time"] if tcands else []) + ["by_index", "all"])
        how = st.radio(t("トラックの選択"), sel_opts,
                       format_func=tr_labels(_TRACK_SELECT_LABELS).get, horizontal=True,
                       key=f"tr_how_{lid}")
        if how == "by_time":
            tv = tcands[0]
            if len(tcands) > 1:
                _reset_invalid_var_key(f"tr_tvar_{lid}", tcands)
                tv = st.selectbox(t("発生日時に使う時刻変数"), tcands,
                                  key=f"tr_tvar_{lid}")
            layer["time_var"] = tv
            _tmin = ds[tv].min(skipna=True).values
            _tmax = ds[tv].max(skipna=True).values
            if np.isnat(_tmin) or np.isnat(_tmax):
                st.warning(t("時刻変数に有効な値がありません"))
            else:
                y_min = int(_tmin.astype("datetime64[Y]").astype(int)) + 1970
                y_max = int(_tmax.astype("datetime64[Y]").astype(int)) + 1970
                cc1, cc2 = st.columns(2)
                y0 = int(cc1.number_input(t("開始年"), y_min, y_max, y_max,
                                          key=f"tr_y0_{lid}"))
                y1 = int(cc2.number_input(t("終了年"), y_min, y_max, y_max,
                                          key=f"tr_y1_{lid}"))
                layer["year_range"] = [min(y0, y1), max(y0, y1)]
                st.caption(t("データの発生年の範囲: {y_min}〜{y_max} 年。"
                           "発生時刻 = 時刻変数の最初の有効時刻で判定",
                           y_min=y_min, y_max=y_max))
                if st.checkbox(t("発生月でも絞り込む"), value=False,
                               key=f"tr_use_mon_{lid}"):
                    cc3, cc4 = st.columns(2)
                    m0 = int(cc3.number_input(t("開始月"), 1, 12, 7,
                                              key=f"tr_m0_{lid}"))
                    m1 = int(cc4.number_input(t("終了月"), 1, 12, 10,
                                              key=f"tr_m1_{lid}"))
                    layer["month_range"] = [m0, m1]
                    st.caption(t("開始月 > 終了月 で年またぎ "
                               "(例 11→3 は 11,12,1,2,3月)"))
        elif how == "by_index":
            cc1, cc2 = st.columns(2)
            lo = int(cc1.number_input(t("開始 index"), 0, n - 1, 0,
                                      key=f"tr_lo_{lid}"))
            hi = int(cc2.number_input(t("終了 index"), 0, n - 1, min(9, n - 1),
                                      key=f"tr_hi_{lid}"))
            layer["storm_range"] = [lo, max(lo, hi)]
        # 選択本数のフィードバック (位置が全欠損で描けないトラックも知らせる)
        _idxs = mc_render.track_storm_indices(layer, {dsid: ds})
        if not _idxs or _idxs == [None]:
            if not _idxs:
                st.warning(t("選択条件に合うトラックがありません"))
        else:
            _others = [d for d in ds[layer["lon_var"]].dims if d != sdim]
            _drawable = int(np.isfinite(
                ds[layer["lon_var"]].isel({sdim: _idxs})).any(
                    dim=_others).sum())
            if _drawable < len(_idxs):
                st.warning(t(
                    "選択中のトラック: {n_sel} 本 — うち位置データが"
                    "あるのは {n_drawable} 本 (残りは選んだ経度・緯度変数が"
                    "全欠損のため描かれません。IBTrACS の速報期間は tokyo_* 等の"
                    "機関別変数が未入力のことがあります — 無印の lat/lon なら"
                    "埋まっています)",
                    n_sel=len(_idxs), n_drawable=_drawable))
            else:
                st.caption(t("選択中のトラック: {n} 本", n=len(_idxs)))
    s = layer["style"]
    if st.checkbox(t("線の色を指定"), value=False, key=f"tr_coluse_{lid}",
                   help=t("off でトラック毎に自動で色が変わる "
                        "(matplotlib カラーサイクル)")):
        s["color"] = color_selector(t("線の色"), "#1f77b4", key=f"tr_col_{lid}",
                                    meta_store=s, meta_key="color")
    s["linewidth"] = float(st.slider(t("線の太さ"), 0.1, 5.0, 1.5, 0.1,
                                     key=f"tr_lw_{lid}"))
    s["linestyle"] = st.selectbox(t("線種"), list(LINESTYLE_LABELS),
                                  format_func=tr_labels(LINESTYLE_LABELS).get,
                                  key=f"tr_ls_{lid}")
    s["alpha"] = float(st.slider(t("透明度 (alpha)"), 0.0, 1.0, 1.0, 0.05,
                                 key=f"tr_alpha_{lid}"))
    # 値変数の候補: lon と同じ次元をもつ数値変数 (風速・気圧など)。
    # maskout と観測点の色付けで共用する
    lon_dims = set(ds[layer["lon_var"]].dims)
    cand = [str(name) for name, v in ds.data_vars.items()
            if set(v.dims) == lon_dims and v.dtype.kind in "fiu"
            and str(name) not in (layer["lon_var"], layer["lat_var"])]
    with st.expander(t("マスクアウト (値で非表示)"), expanded=False):
        if not cand:
            st.caption(t("maskout に使える変数 (経度と同じ次元の数値変数) が"
                       "ありません"))
        else:
            _mo_opts = [_NONE_OPTION] + cand
            _reset_invalid_var_key(f"tr_movar_{lid}", _mo_opts)
            _mo_var = st.selectbox(
                t("maskout に使う変数"), _mo_opts, key=f"tr_movar_{lid}",
                format_func=lambda v, _none=t("(なし)"): (
                    _none if v == _NONE_OPTION else var_label(ds, v)),
                help=t("この変数の値で描く・描かないを決める (例: tokyo_wind)"))
            if _mo_var != _NONE_OPTION:
                mo = s["maskout"]
                mo["variable"] = _mo_var
                c1, c2 = st.columns(2)
                if c1.checkbox(t("この値以下を描かない"), value=False,
                               key=f"tr_mob_on_{lid}"):
                    mo["below"] = float(c1.number_input(
                        t("閾値 (以下)"), value=0.0, format="%g",
                        key=f"tr_mob_{lid}"))
                if c2.checkbox(t("この値以上を描かない"), value=False,
                               key=f"tr_moa_on_{lid}"):
                    mo["above"] = float(c2.number_input(
                        t("閾値 (以上)"), value=0.0, format="%g",
                        key=f"tr_moa_{lid}"))
                st.caption(
                    t("閾値は変数の**生の値**で指定 (例: tokyo_wind なら kt。"
                    "点の色付けの値変換は適用されない)。条件を満たす位置で線が"
                    "切れ、点も消える。変数が欠損の位置も描かれない。"
                    "閾値を変えたレイヤーを重ねると「強度によって線の色が"
                    "変わる」表現ができる (例: 下の層 = 以上 50 で弱い部分、"
                    "上の層 = 以下 50 で強い部分を別色に)"))
    pts = s["points"]
    with st.expander(t("観測点マーカー"), expanded=True):
        pts["show"] = st.checkbox(t("観測点に点を打つ"), value=True,
                                  key=f"tr_pts_{lid}")
        if pts["show"]:
            pts["every"] = int(st.number_input(
                t("表示間隔 (N 点毎)"), 1, 100, 1, key=f"tr_every_{lid}",
                help=t("6時間毎データで 4 にすると1日毎の点になる")))
            pts["size"] = float(st.slider(t("点の大きさ (s, 面積 pt²)"), 1.0, 200.0,
                                          12.0, 1.0, key=f"tr_size_{lid}"))
            pts["marker"] = st.selectbox(
                t("マーカー"), list(_SCATTER_MARKER_LABELS),
                format_func=tr_labels(_SCATTER_MARKER_LABELS).get, key=f"tr_marker_{lid}")
            mode = "single"
            if cand:
                mode = st.radio(t("点の色付け"), list(_CMODE_LABELS),
                                format_func=tr_labels(_CMODE_LABELS).get,
                                horizontal=True, key=f"cmode_tr_{lid}",
                                help=t("カラーマップ: 風速・中心気圧などの変数で"
                                     "色分けする"))
            if mode == "cmap":
                _reset_invalid_var_key(f"tr_pvar_{lid}", cand)
                pts["variable"] = st.selectbox(
                    t("色付けに使う変数"), cand, key=f"tr_pvar_{lid}",
                    format_func=lambda v: var_label(ds, v))
                _tvar = pts["variable"]
                cmap_section_ui(
                    pts, f"tr_{lid}",
                    lambda: transformed_range(f"tr_{lid}", float(ds[_tvar].min()),
                                              float(ds[_tvar].max())))
                colorbar_ui(pts["colorbar"], f"tr_{lid}",
                            ds[_tvar].attrs.get("units", ""),
                            label_suffix=f"_{_tvar}")
                value_transform_ui(pts, f"tr_{lid}")
            else:
                pts["variable"] = None
                pts["color"] = color_selector(t("点の色"), "#333333",
                                              key=f"tr_pcol_{lid}",
                                              meta_store=pts, meta_key="color")
    return layer


LAYER_UI = {"fill": fill_layer_ui, "contour": contour_layer_ui,
            "vector": vector_layer_ui, "stream": stream_layer_ui,
            "map_scatter": map_scatter_layer_ui,
            "track": track_layer_ui,
            "hatch": hatch_layer_ui,
            "line": line_layer_ui, "line_bundle": line_bundle_layer_ui,
            "fill_between": fill_between_layer_ui,
            "stackplot": stackplot_layer_ui,
            "bar": bar_layer_ui,
            "scatter": scatter_layer_ui,
            "bubble": bubble_layer_ui,
            "hexbin": hexbin_layer_ui,
            "hist2d": hist2d_layer_ui,
            "hist": hist_layer_ui,
            "ecdf": ecdf_layer_ui,
            "box": box_layer_ui,
            "violin": violin_layer_ui}
