# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""セッション (作業状態)・プリセットの JSON 保存・復元。

session_state を JSON 化してダウンロード / サーバ側スロット /
起動時プリセット (~/.climcanvas/preset.json) に保存・復元する。

`_SESSION_EXCLUDE_KEYS` / `_SESSION_EXCLUDE_PREFIXES` は tests/test_widget_keys.py が
AST でリテラルとして読むので、リテラル代入のまま維持すること。
"""

import os
import re as _re

import streamlit as st

# --- 設定の保存・復元 ---

# セッション (作業状態) 保存・復元から除外するキー
# - ボタンの session_state は programmatically 設定不可 (Streamlit が拒否) なので必ず除外
# - file_uploader の session_state は UploadedFile オブジェクトで JSON 化できない
# パネルスコープの key (例 anim_play_map0) は _SESSION_EXCLUDE_PREFIXES 側で除外する
_SESSION_EXCLUDE_KEYS = {
    # 操作ボタン
    "browse_file", "load_file",
    "save_startup_preset", "delete_startup_preset",
    # download_button
    "anim_file_dl", "anim_script_dl", "session_dl",
    # file_uploader
    "cfg_upload", "wip_upload", "preset_upload",
}
_SESSION_EXCLUDE_PREFIXES = ("_", "next_", "prev_", "add_", "del_", "dup_",
                         "addkind_", "box_add_", "box_del_",
                         "txtadd_", "txtdel_", "mkadd_", "mkdel_",
                         "bgspan_add_", "bgspan_del_",
                         "refline_add_", "refline_del_",
                         "anim_play_", "anim_file_", "anim_script_",
                         "rotadd_", "rotdel_")

# セッションからは除外するが、起動時プリセットでは保存・復元するキー。
# UI 言語はセッションから独立 — セッションを読み込んでも言語は変わらない
# (i18n 第2段階、docs/i18n_plan.md)。ボタンと違い programmatic 設定は可能
_PRESET_ONLY_KEYS = {"ui_lang"}



def _is_session_excluded(key: str) -> bool:
    """保存・復元どちらでも使う除外判定。"""
    return (key in _SESSION_EXCLUDE_KEYS or key in _PRESET_ONLY_KEYS
            or any(key.startswith(p) for p in _SESSION_EXCLUDE_PREFIXES))

# プリセット (個人デフォルト) に含めるキー (データ非依存の「好み」のみ)。
# パネルに属さないグローバル key のみ exact 一致で列挙する
_PRESET_INCLUDE_KEYS = {
    # 出力
    "png_dpi", "png_name", "script_path_style",
    "script_include_save", "script_include_show",
    "out_format", "out_bg_mode", "out_tight",   # 背景色の色選択は下の prefix "out_bg_color"
    # フォント (図全体)
    "font_use", "font_mode", "font_common", "font_all",
}

# レイヤー・パネルの好み (キーが `_{lid}` や `_{mode_key}` を含むので prefix で一括判定)
_PRESET_INCLUDE_PREFIXES = (
    # 出力: 背景「色を指定」の色 (color_selector の out_bg_color_cat / _sel_* / _picker)
    "out_bg_color",
    # 地図設定 (coast_show_{mode_key} など)
    "coast_", "borders_show", "land_", "ocean_", "frame_",
    # 緯度経度線
    "gl_",
    # ティックマーク
    "ticks_show", "tick_",
    # アニメーション (anim_play_ 等のボタンはセッション除外側で弾かれる)
    "anim_fps", "anim_format", "anim_dpi",
    # 時刻表示
    "show_time_label", "tl_loc", "tl_fmt_",
    # fill 系
    "cmap_",          # cmap_group_{lid} / cmap_{lid}_{group}
    "rev_",           # fill reverse cmap
    "nlev_",          # color level count
    "ext_",           # extend
    # colorbar 詳細 (label text の cbl_ は別プレフィックスなので含まない)
    "cb_", "cbloc_", "cbshrink_", "cbaspect_", "cbpad_", "cblfs_", "cbtfs_",
    "cbolw_", "cbtw_", "cblpad_", "cbtpad_", "cbflip_",
    # 等値線
    "cont_cmode_", "cont_rev_", "cont_color_", "cont_lw_", "cont_ls_",
    "cont_lab_", "cont_labfs_", "cont_fmt_",
    # ベクトル (vec_keylab_ = ラベル文字列、vec_u_ / vec_v_ = 変数選択は含まない)。
    # 色付けのカラーマップとカラーバーは共通部品 (cmap_section_ui / colorbar_ui、
    # p = vec_{lid}) の key なので、上の cmap_ / rev_ / nlev_ / ext_ / cb* に含まれる
    # (旧 key の vec_revcmap_ 等は 2026-09-29 に一覧から削除。tests/test_widget_keys.py
    # の test_preset_keys_exist_as_widget_keys が実在しない項目を検出する)
    "vec_color_", "vec_cmode_",
    "vec_mask",
    "vec_skip_x_", "vec_skip_y_",
    "vec_autoscale_", "vec_refpct_", "vec_autowidth_",
    "vec_width_",
    "vec_autohl_", "vec_headlen_", "vec_edge_", "vec_ecol_", "vec_elw_",
    "vec_key_", "vec_keylen_", "vec_keylpos_", "vec_keyfs_",
    "vec_keyx_", "vec_keyy_",
    # ハッチ (パターン・密度・太さに加え、閾値も含める。閾値は変数依存だがユーザ希望)
    "hatch_pat_", "hatch_den_", "hatch_lw_", "hatch_col_", "hatch_lo_", "hatch_hi_",
    # 軸文字サイズ (per mode)
    "labfs_", "tickfs_", "title_fs_",
)


def _to_json_safe(value):
    """JSON 互換にする。互換不可なら None を返す (呼び出し側で skip)。

    tuple は `{"__tuple__": [...]}` にエンコードし、復元時 (_from_json_safe)
    に tuple へ戻す。範囲 select_slider の widget 状態は tuple で、
    list のまま session_state に書き戻すと Streamlit が範囲モードとして
    認識せずクラッシュするため (単純な list 化では復元が壊れる)。
    """
    if isinstance(value, tuple):
        return {"__tuple__": [_to_json_safe(v) for v in value]}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _from_json_safe(value):
    """_to_json_safe の逆変換。`{"__tuple__": [...]}` を tuple に戻す。

    旧形式の JSON (tuple が素の list で保存されている) はそのまま返す
    (従来挙動と同じ)。すべての復元経路 (スロット復元・アップロード・
    起動時プリセット) で使うこと。
    """
    if isinstance(value, dict):
        if set(value.keys()) == {"__tuple__"} and isinstance(value["__tuple__"], list):
            return tuple(_from_json_safe(v) for v in value["__tuple__"])
        return {k: _from_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_json_safe(v) for v in value]
    return value


# --- i18n 第1段階: 旧形式セッション (旧称 WIP)/プリセットの値マイグレーション ---
# 2026-07-11 より前の保存では selectbox/radio の表示ラベル (日本語) がそのまま
# 保存値だった。中立キー化 (docs/i18n_plan.md 第1段階) に伴い、復元時に旧ラベルを
# 新しい中立キーへ変換する。widget を新設・変更してもここには追記不要
# (中立キーで保存される限り変換は不要)。

_MIG_LS = {"実線": "solid", "破線": "dashed", "点線": "dotted",
           "一点鎖線": "dashdot"}
_MIG_GLLS = {"点線": ":", "実線": "-", "破線": "--"}
_MIG_MARKER = {"なし": "none", "○ (circle)": "o", "■ (square)": "s",
               "▲ (triangle)": "^", "× (x)": "x", "• (dot)": ".",
               "+ (plus)": "+", "◇ (diamond)": "D", "* (star)": "*"}
_MIG_ANNOT_MARKER = {"○ (circle)": "o", "■ (square)": "s",
                     "▲ (triangle up)": "^", "▼ (triangle down)": "v",
                     "◀ (triangle left)": "<", "▶ (triangle right)": ">",
                     "× (x)": "x", "+ (plus)": "+", "◇ (diamond)": "D",
                     "☆ (star)": "*", "• (dot)": "."}
_MIG_CMODE = {"単色": "single", "カラーマップ": "cmap",
              "カラーマップ (大きさ)": "cmap"}
_MIG_COORD = {"図に対する相対位置 (axes)": "axes",
              "データ座標 (地図は経度・緯度)": "data"}
_MIG_NONE = {"(なし)": "(none)"}
_MIG_STRFTIME = {"自動": "auto", "年 (例 1984)": "%Y",
                 "年-月 (例 1984-01)": "%Y-%m",
                 "年-月-日 (例 1984-01-15)": "%Y-%m-%d",
                 "月/日 (例 01/15)": "%m/%d", "時:分 (例 12:30)": "%H:%M",
                 "カスタム…": "custom"}

# (マッチ方法, パターン, {旧値: 新値})。上から順に最初に key がマッチした規則
# だけを適用する — prefix が重なるもの (cont_ls_neg_ と cont_ls_ 等) は
# 長い方を先に置くこと
_SESSION_VALUE_MIGRATIONS = [
    ("prefix", "plot_mode_", {"水平断面図": "map", "鉛直断面図": "vsec",
                              "時間断面図": "tsec", "1次元プロット": "line",
                              "1次元プロット(集計)": "dist",
                              "2次元プロット": "scatter",
                              "2次元プロット(集計)": "agg"}),
    ("prefix", "cont_ls_neg_", {**_MIG_LS, "正と同じ": "same"}),
    ("prefix", "cont_ls_", _MIG_LS),
    ("prefix", "ecdf_ls_", _MIG_LS),
    ("prefix", "tr_ls_", _MIG_LS),
    ("prefix", "box_ls_", _MIG_LS),
    ("prefix", "line_ls_", {**_MIG_LS, "なし": "None"}),
    ("prefix", "gridls_", _MIG_GLLS),
    ("prefix", "gl_linestyle_", _MIG_GLLS),
    ("prefix", "line_marker_", _MIG_MARKER),
    ("prefix", "sc_marker_", _MIG_MARKER),
    ("prefix", "ms_marker_", _MIG_MARKER),
    ("prefix", "bb_marker_", _MIG_MARKER),
    ("prefix", "box_fmark_", _MIG_MARKER),
    ("prefix", "tr_marker_", _MIG_MARKER),
    ("prefix", "mk_", _MIG_ANNOT_MARKER),
    ("prefix", "cont_cmode_", _MIG_CMODE),
    ("prefix", "vec_cmode_", _MIG_CMODE),
    ("prefix", "strm_cmode_", _MIG_CMODE),
    ("prefix", "cmode_ms_", _MIG_CMODE),
    ("prefix", "cmode_tr_", _MIG_CMODE),
    ("prefix", "txtcoord_", _MIG_COORD),
    ("prefix", "mkcoord_", _MIG_COORD),
    ("contains", "_avgmode_", {"固定値": "fixed", "範囲平均": "range"}),
    ("prefix", "lvmode_", {"レベル数 (等間隔)": "count",
                           "レベルを直接指定": "explicit"}),
    ("prefix", "fill_method_", {"contourf (滑らか)": "contourf",
                                "pcolormesh (格子・離散)": "pcolormesh"}),
    ("prefix", "fb_mode_", {"変数で指定": "variable",
                            "定数 (ベースライン)": "constant"}),
    ("prefix", "bar_err_src_", {"変数で指定": "variable", "定数": "constant"}),
    ("prefix", "bar_orient_", {"縦棒 (ax.bar)": "vertical",
                               "横棒 (ax.barh)": "horizontal"}),
    ("prefix", "bar_mode_", {"重ね合わせ": "overlap", "横並び": "dodge",
                             "積み上げ": "stack"}),
    ("prefix", "sp_base_", {"zero (積み上げ)": "zero", "sym (対称)": "sym",
                            "weighted_wiggle (Streamgraph)": "weighted_wiggle"}),
    ("prefix", "h2d_den_", {"度数": "count", "確率密度 (density)": "density"}),
    ("prefix", "hist_den_", {"度数": "count", "確率密度 (density)": "density"}),
    ("prefix", "hist_httype_", {"塗り (bar)": "bar",
                                "階段 (step, 線のみ)": "step",
                                "階段塗り (stepfilled)": "stepfilled"}),
    ("prefix", "box_whismode_", {"IQR 倍率": "iqr",
                                 "パーセンタイル範囲": "percentile"}),
    ("prefix", "vio_bwmode_", {"自動 (scott)": "scott", "数値指定": "value"}),
    ("prefix", "vio_side_", {"両側": "both", "左半分": "low", "右半分": "high"}),
    ("prefix", "vec_keylpos_", {"矢印の右": "E", "矢印の左": "W"}),
    ("prefix", "tr_how_", {"発生年・月で選ぶ": "by_time",
                           "番号 (index) で選ぶ": "by_index",
                           "全トラック": "all"}),
    ("prefix", "sc_ebx_", _MIG_NONE),
    ("prefix", "sc_eby_", _MIG_NONE),
    ("prefix", "bb_ebx_", _MIG_NONE),
    ("prefix", "bb_eby_", _MIG_NONE),
    ("prefix", "tr_movar_", _MIG_NONE),
    ("prefix", "panel_zvar_", _MIG_NONE),
    ("prefix", "tl_fmt_", _MIG_STRFTIME),
    ("prefix", "tfmt_", _MIG_STRFTIME),
    ("prefix", "xtmode_", {"自動": "auto", "等間隔 (0 で非表示)": "interval",
                           "位置を直接指定": "positions"}),
    ("prefix", "ytmode_", {"自動": "auto", "等間隔 (0 で非表示)": "interval",
                           "位置を直接指定": "positions"}),
    ("prefix", "tick_dir_", {"外側": "out", "内側": "in", "両方": "inout"}),
    ("prefix", "gl_lat_lab_", {"図中 (既定)": "inline", "枠沿い": "edge",
                               "非表示": "none"}),
    ("prefix", "gl_lat_edge_side_", {"両方": "both", "左のみ": "left",
                                     "右のみ": "right"}),
    ("prefix", "tl_loc_", {"左": "left", "中央": "center", "右": "right"}),
    ("prefix", "bgspan_ori_", {"x軸の範囲 (縦帯)": "x",
                               "y軸の範囲 (横帯)": "y"}),
    ("prefix", "boxaspect_preset_", {"1:1": "square",
                                     "6.4:4.8 (matplotlib default)": "default",
                                     "任意": "custom"}),
    ("prefix", "dist_family_", {"ヒストグラム・ECDF": "hist",
                                "箱ひげ・バイオリン": "box"}),
    ("prefix", "dist_orient_", {"縦": "vertical", "横": "horizontal"}),
    ("prefix", "draw2d_", {"散布図・バブル": "scatter_bubble"}),
    ("prefix", "vsec_orient_", {"経度–高度 (緯度を固定)": "lon_height",
                                "緯度–高度 (経度を固定)": "lat_height"}),
    ("prefix", "tsec_kind_", {"時間–高度": "time_height",
                              "時間–緯度": "time_lat",
                              "時間–経度": "time_lon"}),
    ("exact", "figsize_preset", {"6.4:4.8 (matplotlib default)": "default",
                                 "1:1": "square", "√2:1 (A4)": "a4",
                                 "16:9": "wide", "任意の数字": "custom"}),
    ("exact", "font_mode", {"よく使うフォント": "common",
                            "全フォントから検索": "all"}),
    ("exact", "out_bg_mode", {"白 (既定)": "white", "色を指定": "color",
                              "透明": "transparent"}),
    ("exact", "script_path_style", {"絶対パス": "absolute",
                                    "相対パス": "relative"}),
]


def _migrate_legacy_value(key, value):
    """旧形式 (日本語ラベルが保存値) のセッション/プリセット値を中立キーへ変換する。

    すべての復元経路 (スロット復元・アップロード・起動時プリセット) で
    session_state へ書き込む直前に通すこと。該当しない値はそのまま返す。
    """
    if not isinstance(value, str):
        return value
    for kind, pat, vmap in _SESSION_VALUE_MIGRATIONS:
        if ((kind == "exact" and key == pat)
                or (kind == "prefix" and key.startswith(pat))
                or (kind == "contains" and pat in key)):
            return vmap.get(value, value)
    return value


_CLON_FOLLOW_KEY = _re.compile(r"^central_lon_(?P<mode>[A-Za-z]+\d+)_(?P<proj>Robinson|EqualEarth)$")


def _migrate_loaded_session(loaded: dict) -> dict:
    """読み込んだセッション/プリセットの辞書に、後から増えた key の互換値を補う。

    すべての復元経路で、個々の値を session_state へ書く前に通すこと (値の変換は
    `_migrate_legacy_value`、ここは「他の key の有無で決まる補い」)。
    - Robinson / EqualEarth の「中心経度を範囲の中央に合わせる」(`clon_follow_*`、
      2026-10-02 追加、既定 ON) が無く、中心経度の入力値 (`central_lon_*`) が保存されて
      いる旧セッションは、追従を OFF にして保存時の中心経度のまま描く (既定の ON を
      適用すると図が変わる)
    """
    out = dict(loaded)
    for key in loaded:
        m = _CLON_FOLLOW_KEY.match(key)
        if m:
            follow_key = f"clon_follow_{m['mode']}_{m['proj']}"
            if follow_key not in loaded:
                out[follow_key] = False
    return out


def _serialize_session() -> dict:
    """作業状態 (session_state 全体) を JSON 互換 dict にする。"""
    out = {}
    for k, v in st.session_state.items():
        if _is_session_excluded(k):
            continue
        safe = _to_json_safe(v)
        try:
            import json as _json
            _json.dumps(safe)
        except (TypeError, ValueError):
            continue
        out[k] = safe
    return out


def _is_preset_key(key: str) -> bool:
    if key in _PRESET_INCLUDE_KEYS or key in _PRESET_ONLY_KEYS:
        return True
    return any(key.startswith(p) for p in _PRESET_INCLUDE_PREFIXES)


def _serialize_preset() -> dict:
    """プリセット (好み) として保存するキーだけを抽出。"""
    out = {}
    for k, v in st.session_state.items():
        if not _is_preset_key(k):
            continue
        safe = _to_json_safe(v)
        try:
            import json as _json
            _json.dumps(safe)
        except (TypeError, ValueError):
            continue
        out[k] = safe
    return out


# 起動時に自動適用するプリセットの保存先
# (テスト時は CLIMCANVAS_PRESET_PATH で差し替え可 — CLIMCANVAS_CMAP_DIR と同じ流儀)
import pathlib as _pathlib_mod
STARTUP_PRESET_PATH = _pathlib_mod.Path(
    os.environ.get("CLIMCANVAS_PRESET_PATH")
    or (_pathlib_mod.Path.home() / ".climcanvas" / "preset.json"))


def _load_startup_preset_once():
    """初回起動時のみ STARTUP_PRESET_PATH を session_state に適用する。

    widget 描画より前に呼ぶこと。`_startup_preset_loaded` フラグで一度だけ実行。
    """
    if st.session_state.get("_startup_preset_loaded"):
        return
    st.session_state["_startup_preset_loaded"] = True
    if not STARTUP_PRESET_PATH.is_file():
        return
    import json as _json
    try:
        loaded = _migrate_loaded_session(
            _json.loads(STARTUP_PRESET_PATH.read_text(encoding="utf-8")))
        applied = 0
        for k, v in loaded.items():
            # _PRESET_ONLY_KEYS (UI 言語) はセッション除外だがプリセットでは復元する
            if _is_session_excluded(k) and k not in _PRESET_ONLY_KEYS:
                continue
            st.session_state[k] = _migrate_legacy_value(k, _from_json_safe(v))
            applied += 1
        st.session_state["_startup_preset_applied"] = applied
    except Exception as e:
        st.session_state["_startup_preset_error"] = str(e)


def _save_startup_preset() -> int:
    """現在のプリセット相当の値を STARTUP_PRESET_PATH に書き出す。書いた項目数を返す。"""
    import json as _json
    STARTUP_PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
    preset = _serialize_preset()
    STARTUP_PRESET_PATH.write_text(
        _json.dumps(preset, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(preset)


def _delete_startup_preset() -> bool:
    """STARTUP_PRESET_PATH を削除する。削除した時 True。"""
    if STARTUP_PRESET_PATH.is_file():
        STARTUP_PRESET_PATH.unlink()
        return True
    return False


# --- セッションの名前付き保存 (dir 選択 + ファイル名) ---

_DEFAULT_SESSION_DIR = _pathlib_mod.Path.home() / ".climcanvas" / "sessions"


def _load_session_dirs() -> list[str]:
    """セッション保存先候補のディレクトリ一覧を返す。

    優先順:
      1. 環境変数 CC_SESSION_DIRS (`:` 区切り)
      2. ~/.climcanvas/config.toml の session_dirs
      3. デフォルト [~/.climcanvas/sessions]
    """
    from climcanvas.ui.data_access import (expand_user_paths, read_user_config,
                                           split_env_paths)
    raw = os.environ.get("CC_SESSION_DIRS", "").strip()
    if raw:
        return split_env_paths(raw)
    items = read_user_config().get("session_dirs") or []
    if items:
        return expand_user_paths(items)
    return [str(_DEFAULT_SESSION_DIR)]


def _list_sessions(directory: str) -> list[str]:
    """directory 内の `.json` ファイル名 (stem) をソートして返す。"""
    try:
        d = _pathlib_mod.Path(directory)
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.json"))
    except OSError:
        return []


def _session_path(directory: str, name: str) -> _pathlib_mod.Path:
    # パス区切りや `..` を含む名前は wip_dirs の外への書き込み・削除に
    # つながる (例 "../../home/userB/x") ため拒否する
    if not name or name in (".", "..") or name != os.path.basename(name):
        raise ValueError(f"スロット名にパス区切りや '..' は使えません: {name!r}")
    return _pathlib_mod.Path(directory) / f"{name}.json"


def _save_session_to_disk(directory: str, name: str) -> int:
    """現在の session_state を `<directory>/<name>.json` に書き出す。書いた項目数を返す。"""
    import json as _json
    target_dir = _pathlib_mod.Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    session = _serialize_session()
    target = _session_path(directory, name)
    target.write_text(
        _json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(session)


def _load_session_from_disk(directory: str, name: str) -> int:
    """`<directory>/<name>.json` を読んで session_state に書き戻す。

    既存の値は上書きされる。読み込んだ項目数を返す。
    """
    import json as _json
    path = _session_path(directory, name)
    if not path.is_file():
        return 0
    loaded = _migrate_loaded_session(_json.loads(path.read_text(encoding="utf-8")))
    applied = 0
    for k, v in loaded.items():
        if _is_session_excluded(k):
            continue
        st.session_state[k] = _migrate_legacy_value(k, _from_json_safe(v))
        applied += 1
    return applied
