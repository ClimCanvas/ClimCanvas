# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""ClimCanvas — Streamlit UI層 (メインフロー)。

ユーザーの選択を figure_config 辞書に詰めてコア層 (climcanvas.core) に渡すだけの
薄い層にする。描画ロジック・スクリプト生成をここに書かないこと。
UI 部品は climcanvas/ui/ に分離してある:
  constants / state_io / data_access / widgets / layer_ui / panel_ui
"""

import copy
import io
import logging
import os
import time as time_mod

# JSON 読み込みで session_state を埋めた後の widget 描画では、Streamlit が
# 「widget に value= もあり session_state にも値がある」という情報 warning を毎回出す。
# 挙動 (session_state 優先) は意図通りなので、この特定ロガーだけ ERROR 以上にして黙らせる。
# Streamlit 1.30+ の policies モジュール
logging.getLogger("streamlit.elements.lib.policies").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# set_page_config は最初の Streamlit コマンドとして呼ぶ必要がある
# (climcanvas.ui の import より前に置く)
st.set_page_config(page_title="ClimCanvas", layout="wide")

# Streamlit 1.64+ の「Install the official Streamlit skills」の案内を次回以降出さない
# (「Don't show again」と同じ目印ファイルを作る。詳細は climcanvas/ui/streamlit_env.py)
from climcanvas.ui.streamlit_env import dismiss_skills_nudge_once  # noqa: E402

dismiss_skills_nudge_once()

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import notes as mc_notes
from climcanvas.core import render as mc_render
from climcanvas.core import scriptgen as mc_scriptgen
from climcanvas.ui.constants import (GL_LINESTYLE_LABELS, KIND_LABELS,
                                     NE_RESOLUTION_LABELS, PROJECTION_LABELS)
from climcanvas.ui.data_access import (_add_dataset_to_state,
                                       _browse_file_callback, _is_path_allowed,
                                       _load_allowed_dirs, app_mode,
                                       _remove_dataset_from_state,
                                       browse_into_callback,
                                       detect_grid_projection_cached,
                                       load_dataset, load_dataset_with_coords)
from climcanvas.ui.panel_ui import (axis_basic_ui, axis_labels_ui,
                                    axis_settings_ui, axis_ticks_ui, boxes_ui,
                                    frame_background_ui,
                                    layers_ui, legend_settings_ui,
                                    line_axis_settings_ui, markers_ui,
                                    panel_selector_cells, selection_widgets,
                                    texts_ui)
from climcanvas.ui.about import display_version, set_about_menu
from climcanvas.ui.i18n import LANG_LABELS, t, tr_labels
from climcanvas.ui.notes_ui import format_notes
from climcanvas.ui.state_io import (STARTUP_PRESET_PATH, _PRESET_ONLY_KEYS,
                                    _delete_startup_preset,
                                    _from_json_safe, _is_session_excluded,
                                    _list_sessions, _migrate_legacy_value,
                                    _migrate_loaded_session,
                                    _load_startup_preset_once, _load_session_dirs,
                                    _load_session_from_disk, _save_startup_preset,
                                    _save_session_to_disk, _serialize_session,
                                    _session_path)
from climcanvas.ui.widgets import (available_font_families,
                                   cjk_font_advice, cmap_selector,
                                   color_selector, common_fonts_available,
                                   coord_label,
                                   fontsize_input, linked_range_ui,
                                   linked_time_range_ui, maskout_ui,
                                   md_plain, parse_float_list,
                                   parse_ratio_list, section_header,
                                   shorten_label,
                                   time_labels_and_values, union_coord_values,
                                   value_transform_ui, var_label)

# --- メイン ---

# st.title 相当の h1 を translate="no" 付きの生 HTML で描く。ブラウザの自動翻訳
# (Chrome の「このページを翻訳」) がアプリ名を音写 (クライムキャンバス) するのを
# 防ぐ。class="notranslate" は Google 翻訳向けの同義指定 (st.title は HTML 属性を
# 付けられないため markdown で代替。見た目は markdown h1 = st.title と同一)
st.markdown('<h1 translate="no" class="notranslate">ClimCanvas</h1>',
            unsafe_allow_html=True)
# 表記はリリースタグに合わせ major.minor (display_version。About の版と同じ)
st.caption(t("netCDF 大気・海洋データ可視化 — ver {ver}", ver=display_version()))

# --- widget 状態の保持 (stale-widget 掃除対策) ---
# パネル切替・追加・複製などのボタンは st.rerun() でスクリプトを途中終了させる。
# その run で描画されなかった widget (選択外パネルの編集 UI、ボタンより後に
# 描画される「図全体の書式」「出力」など) の session_state を Streamlit が
# run 終了時に破棄してしまうため、毎 run 冒頭で全 key を自己代入して通常の
# session 値に昇格させ、破棄を防ぐ。ボタン等 (セッション除外キー) は programmatic
# 設定不可なのでスキップ。**どの widget よりも先に実行すること**
# (描画済み widget の key への代入は Streamlit が拒否する)。
for _k in list(st.session_state.keys()):
    # _PRESET_ONLY_KEYS (UI 言語) はセッション除外だが widget なので自己代入で保持する
    if _is_session_excluded(_k) and _k not in _PRESET_ONLY_KEYS:
        continue
    st.session_state[_k] = st.session_state[_k]


def _allowed_dir_browser(allowed_dirs: list[str], target_key: str,
                         key_prefix: str = "_browse") -> None:
    """許可ディレクトリの中を辿って .nc を選ぶ Streamlit ネイティブのブラウザ。

    リモート運用想定 (tkinter ダイアログが動かないヘッドレス環境用)。選んだファイルの
    絶対パスを session_state[target_key] に入れて rerun する。**target_key の widget
    より前に呼ぶこと** (インスタンス化後の書き換えは Streamlit が拒否する)。
    key_prefix で widget key を分ける: 本体ファイルの欄は "_browse" (従来の key と同じ)、
    座標ファイルの欄は "_coordbrowse_{index}"。
    """
    st.caption(t("許可されているディレクトリ: {dirs}",
                 dirs=", ".join(f"`{d}`" for d in allowed_dirs)))
    root_key = f"{key_prefix}_root"
    cur_key = f"{key_prefix}_cur"
    last_key = f"{key_prefix}_last_root"
    if st.session_state.get(cur_key) is None:
        st.session_state[cur_key] = allowed_dirs[0]
    # ルート (複数の許可 dir があれば selectbox で切替)
    if len(allowed_dirs) > 1:
        cur_root = st.selectbox(t("ルート"), allowed_dirs, key=root_key)
        if (st.session_state.get(last_key) != cur_root):
            st.session_state[cur_key] = cur_root
            st.session_state[last_key] = cur_root
    else:
        cur_root = allowed_dirs[0]
        st.session_state[last_key] = cur_root
    cur = st.session_state[cur_key]
    # 現在地は「許可ディレクトリ名/相対パス」形式で表示する
    # (トップの "/" 表示は FS ルートと紛らわしかった。ディレクトリ名を
    # 頭に付ければ、複数の許可 dir を切替えてもどこにいるか分かる)
    _root_name = os.path.basename(os.path.normpath(cur_root))
    try:
        rel = os.path.relpath(cur, cur_root)
        _loc = _root_name + "/" + (rel if rel != "." else "")
    except ValueError:
        _loc = cur
    st.markdown(t("**現在地**: `{path}`", path=_loc))
    # 上へ (ルートから出ない)
    if os.path.realpath(cur) != os.path.realpath(cur_root):
        if st.button(t("⬆ 上の階層へ"), key=f"{key_prefix}_up"):
            st.session_state[cur_key] = os.path.dirname(cur)
            st.rerun()
    try:
        entries = sorted(os.listdir(cur))
    except OSError as e:
        st.error(t("ディレクトリを開けません: {e}", e=e))
        entries = []
    subdirs = [e for e in entries
               if os.path.isdir(os.path.join(cur, e))
               and not e.startswith(".")]
    ncfiles = [e for e in entries
               if e.lower().endswith(".nc")
               and os.path.isfile(os.path.join(cur, e))]
    if subdirs:
        st.markdown(t("**サブディレクトリ**"))
        for d in subdirs:
            if st.button(f"📂 {d}", key=f"{key_prefix}_dir_{d}"):
                st.session_state[cur_key] = os.path.join(cur, d)
                st.rerun()
    if ncfiles:
        st.markdown(t("**netCDF ファイル**"))
        for f in ncfiles:
            if st.button(f"📄 {f}", key=f"{key_prefix}_file_{f}"):
                st.session_state[target_key] = os.path.join(cur, f)
                st.rerun()
    elif not subdirs:
        st.caption(t("(このディレクトリには .nc ファイルもサブディレクトリもありません)"))


# 座標ファイルのブラウザで選んだファイルの入れ先 (中立キー → 日本語原文)
_COORD_TARGET_LABELS = {"lon": "経度 (lon) の欄", "lat": "緯度 (lat) の欄"}


def _coord_files_ui(index: int, item: dict, allowed_dirs: list[str]) -> None:
    """読み込み済みファイル 1 件の「座標ファイル (任意)」expander。

    経緯度 (lon/lat) がデータ本体に無いとき (ClimCORE のように別ファイルの FLON / FLAT に
    あるとき)、そのファイルの経緯度変数を座標として結び付ける。永続化は datasets
    エントリの `coord_paths` (セッション JSON に入る)。widget key は "_" 始まり
    (セッション除外) で、入力欄の初期値は保存済みの coord_paths から入れる。
    経度と緯度が別ファイルなら 2 つ、同じファイルなら経度側だけ指定する。
    ファイルの選び方は本体ファイルの欄と同じ: 許可ディレクトリ運用ならトグルで
    「許可ディレクトリから選ぶ」ブラウザ (入れ先を経度 / 緯度で選ぶ。expander の
    入れ子は不可なのでトグル)、そうでなければ「参照」の tkinter ダイアログ。
    """
    current = list(item.get("coord_paths") or [])
    label = t("座標ファイル (任意)") + (" ✓" if current else "")
    with st.expander(label, expanded=False):
        st.caption(t("経緯度 (lon/lat) がデータ本体に無いとき、別ファイルの経緯度変数 "
                     "(2 次元の lon(y,x) / lat(y,x) など) を座標として結び付けます。"
                     "経度と緯度が別ファイルなら両方を指定します "
                     "(例: ClimCORE の FLON.nc と FLAT.nc)。"))
        if current:
            st.caption(t("結合中: {paths}",
                         paths=", ".join(f"`{p}`" for p in current)))
        lon_key, lat_key = f"_coord_lon_{index}", f"_coord_lat_{index}"
        if lon_key not in st.session_state:
            st.session_state[lon_key] = current[0] if current else ""
        if lat_key not in st.session_state:
            st.session_state[lat_key] = current[1] if len(current) > 1 else ""
        # 別のファイル (最初に座標ファイルを付けたもの) と同じ座標ファイルを使う
        # チェック (ClimCORE の Z と地表ファイルに同じ FLON / FLAT を付ける手間を省く。
        # 自動では付けない — 格子の取り違えに気づけるよう明示の操作にする。
        # ユーザー要望 2026-09-30)。入力欄より前に置く (欄の session_state を書き換えて
        # から欄をインスタンス化する)
        source = next((it for it in st.session_state.get("datasets", [])
                       if it is not item and it.get("coord_paths")), None)
        if source is not None:
            src_paths = list(source["coord_paths"])
            same = st.checkbox(
                t("{dsid} と同じ座標ファイルを使う ({files})",
                  dsid=source["id"],
                  files=", ".join(os.path.basename(p) for p in src_paths)),
                value=(current == src_paths), key=f"_coord_same_{index}")
            if same and current != src_paths:
                item["coord_paths"] = src_paths
                st.session_state[lon_key] = src_paths[0]
                st.session_state[lat_key] = src_paths[1] if len(src_paths) > 1 else ""
                st.rerun()
        # ブラウザは入力欄より前に置く (選択結果を欄の session_state に書いてから
        # 欄をインスタンス化する)
        if allowed_dirs and st.toggle(t("📁 許可ディレクトリから選ぶ"),
                                      key=f"_coord_showbrowse_{index}"):
            target = st.radio(t("選んだファイルの入れ先"), list(_COORD_TARGET_LABELS),
                              format_func=tr_labels(_COORD_TARGET_LABELS).get,
                              horizontal=True, key=f"_coord_target_{index}")
            _allowed_dir_browser(allowed_dirs,
                                 lon_key if target == "lon" else lat_key,
                                 key_prefix=f"_coordbrowse_{index}")
        st.text_input(t("経度 (lon) のファイル"), key=lon_key)
        st.text_input(t("緯度 (lat) のファイル (経度と同じファイルなら空)"), key=lat_key)
        if not allowed_dirs:
            # 参照ボタン: 許可ディレクトリ運用では非表示 (本体ファイルの欄と同じ)
            b1, b2 = st.columns(2)
            b1.button(t("参照 (経度)"), key=f"_coord_browse_lon_{index}",
                      on_click=browse_into_callback, args=(lon_key,))
            b2.button(t("参照 (緯度)"), key=f"_coord_browse_lat_{index}",
                      on_click=browse_into_callback, args=(lat_key,))
        c1, c2 = st.columns(2)
        if c1.button(t("適用"), key=f"_coord_apply_{index}", type="primary"):
            paths: list[str] = []
            for raw in (st.session_state[lon_key], st.session_state[lat_key]):
                raw = (raw or "").strip()
                if not raw:
                    continue
                abspath = os.path.abspath(os.path.expanduser(raw))
                if not _is_path_allowed(abspath, allowed_dirs):
                    st.error(t("座標ファイル `{path}` は許可ディレクトリの外にあります。"
                               "許可: {dirs}", path=abspath, dirs=allowed_dirs))
                    return
                if not os.path.isfile(abspath):
                    st.error(t("座標ファイルが見つかりません: {path}", path=abspath))
                    return
                if abspath not in paths:
                    paths.append(abspath)
            if not paths:
                st.error(t("座標ファイルのパスを入力してください"))
                return
            item["coord_paths"] = paths
            st.rerun()
        if current and c2.button(t("解除"), key=f"_coord_clear_{index}"):
            item.pop("coord_paths", None)
            st.session_state[lon_key] = ""
            st.session_state[lat_key] = ""
            st.rerun()


with st.sidebar:
    # ~/.climcanvas/preset.json を「最初の widget が描画される前」に適用する
    _load_startup_preset_once()

    # UI 言語 (i18n 第2段階)。セッションには保存されず、「起動時プリセットに保存」で
    # 永続化できる。ラベルは選択前でも読めるよう二言語併記の固定文字列
    st.selectbox(t("言語 / Language"), list(LANG_LABELS),
                 format_func=tr_labels(LANG_LABELS).get, key="ui_lang")
    # 右上メニューの About (版・フィードバックフォーム・Issues) を今の UI 言語で
    set_about_menu()

    st.header(t("データ・セッション読み込み"))

    # 作業の再開 (セッション復元) は、以下のすべての widget より先に評価する。
    # これによって `pending_file_path` を含む値も「widget 描画前の初期値」として
    # 安全に session_state へ書き込める (描画後の書き換えは Streamlit が拒否する)。
    # また st.stop() (データ未読み込み) より前に置くことで、起動直後の画面から
    # スロット復元できる (リモート運用の「昨日の続き」がワンクリックになる)
    import json as _json_mod
    # 運用形態 (明示設定のみ)。青タブの文言 (PC/サーバー) の出し分けに使う
    _remote = app_mode() == "remote"
    st.markdown(t("**作業の再開**"))
    with st.expander(t("セッション復元"), expanded=False):
        # 色の対応: 青 = ClimCanvas を動かすマシン側 (local=PC / remote=サーバー)、
        # オレンジ = 手元 PC とのファイル受け渡し (アップロード/ダウンロード)。
        # 保存側のタブと共通
        _tab_slot, _tab_file = st.tabs([
            f":blue[{t('サーバーから') if _remote else t('PC から')}]",
            f":orange[{t('アップロード')}]"])
        with _tab_slot:
            st.caption(t("サーバーに保存されているセッションを選んで復元する")
                       if _remote else
                       t("この PC に保存されているセッションを選んで復元する"))
            _restore_dirs = _load_session_dirs()
            _restore_dir = _restore_dirs[0]
            # サイドバーの selectbox は 1 行固定で長い名前の末尾が「…」に
            # 落ちるため、表示は shorten_label (ホーム → ~、中央省略) で短くし、
            # 全文は直下の caption (折り返す) に出す。値は変えない
            if len(_restore_dirs) > 1:
                _restore_dir = st.selectbox(t("ディレクトリ"), _restore_dirs,
                                            format_func=shorten_label,
                                            key="_session_restore_dir")
                st.caption(md_plain(_restore_dir))
            _existing = _list_sessions(_restore_dir)
            if _existing:
                _restore_slot = st.selectbox(
                    t("ファイル名"), _existing, format_func=shorten_label,
                    key="_session_restore_slot")
                st.caption(t("選択中: {name}", name=md_plain(_restore_slot)))
                if st.button(t("復元"), key="_session_restore_btn"):
                    try:
                        n = _load_session_from_disk(_restore_dir, _restore_slot)
                        st.session_state["_session_provenance"] = {
                            "kind": "slot", "name": _restore_slot}
                        # 保存側「保存するスロット」に編集中スロットを自動選択
                        # させる (次 run の widget 描画前に反映される)
                        st.session_state["_session_ow_pending"] = {
                            "dir": _restore_dir, "name": _restore_slot}
                        st.success(t("復元しました ({n} 項目)", n=n))
                        st.rerun()
                    except Exception as e:
                        st.error(t("復元失敗: {e}", e=e))
            else:
                st.caption(t("(このディレクトリには保存済みのセッションがありません)"))
        with _tab_file:
            st.caption(t("手元 PC に保存したセッションファイルを読み込む"))
            if st.session_state.get("_cfg_applied_id"):
                # 読み込み済みの間はアップローダを出さない — 別ファイルの
                # 追加アップロードで作業状態が即座に上書きされる事故を防ぐ。
                # 差し替えは明示的にクリアしてから (アップローダは描画しない run で
                # widget 状態ごと破棄される。cfg_upload はセッション除外キーなので
                # 冒頭の自己代入ループにも保持されない)
                _loaded_msg = st.session_state.pop("_cfg_loaded_msg", None)
                if _loaded_msg is not None:
                    st.success(t("セッションを読み込みました ({n} 項目)", n=_loaded_msg))
                st.caption(t("読み込み済み: {name}",
                             name=st.session_state.get("_cfg_applied_name", "")))
                if st.button(t("別のセッションファイルを読み込む"),
                             key="_cfg_clear_btn"):
                    st.session_state.pop("_cfg_applied_id", None)
                    st.session_state.pop("_cfg_applied_name", None)
                    st.rerun()
                cfg_up = None
            else:
                cfg_up = st.file_uploader(
                    t("セッションファイル"),
                    type=["json"], key="cfg_upload")
            if cfg_up is not None:
                # 同じファイルを毎 rerun で再適用しないように file_id を覚えておく
                fid = getattr(cfg_up, "file_id", None) or f"{cfg_up.name}:{cfg_up.size}"
                if st.session_state.get("_cfg_applied_id") != fid:
                    try:
                        loaded = _json_mod.loads(cfg_up.getvalue())
                        if "datasets" not in loaded:
                            # セッション以外の JSON (個人プリセット等) は受け付けない。
                            # _serialize_session の出力は datasets キーを必ず含むことを利用
                            # (適用済み扱いにしないので、ファイルを外すまでエラーが出続ける)
                            st.error(t("セッションの JSON ではないようです (`datasets` が"
                                       "含まれていません)。個人プリセットは `{path}` に置くと"
                                       "起動時に自動適用されます", path=STARTUP_PRESET_PATH))
                        else:
                            # ボタン・アップローダー系キーは Streamlit が programmatic 設定を
                            # 拒否するためスキップ (古い JSON がこれらを含んでいても安全)
                            applied = 0
                            for k, v in _migrate_loaded_session(loaded).items():
                                if _is_session_excluded(k):
                                    continue
                                st.session_state[k] = _migrate_legacy_value(
                                    k, _from_json_safe(v))
                                applied += 1
                            st.session_state["_cfg_applied_id"] = fid
                            st.session_state["_cfg_applied_name"] = cfg_up.name
                            st.session_state["_session_provenance"] = {
                                "kind": "file", "name": cfg_up.name}
                            # このアップローダ自身の表示分岐は上で評価済みのため
                            # st.rerun() で「読み込み済み」表示へ即切替する
                            # (成功メッセージは _cfg_loaded_msg で持ち越す)
                            st.session_state["_cfg_loaded_msg"] = applied
                            st.rerun()
                    except Exception as e:
                        st.error(t("セッション読み込み失敗: {e}", e=e))

    # 読み込み済みファイルの一覧 (削除可)
    _allowed_dirs = _load_allowed_dirs()
    items = st.session_state.setdefault("datasets", [])
    if items:
        st.markdown(t("**読み込み済みファイル**"))
        for i, item in enumerate(list(items)):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"`{item['id']}`: `{item['path']}`")
            # アイコンボタン (狭い列。文字ラベルは streamlit 1.64 以降折り返さず「削…」に切れる)
            if c2.button("", icon=":material/delete:", help=t("削除"),
                         key=f"_del_ds_{i}"):
                _remove_dataset_from_state(i)
                st.rerun()
            _coord_files_ui(i, item, _allowed_dirs)

    # 新規追加 UI (ラベルはデータ読み込み後は「ファイルの追加」に変わる)
    st.markdown(t("**新規作業**")
                if not st.session_state.get("datasets")
                else t("**ファイルの追加**"))
    if "_new_file_path" not in st.session_state:
        st.session_state["_new_file_path"] = "data/sample/sample_atmos.nc"

    # 許可ディレクトリがあれば Streamlit ネイティブのブラウザを出す
    # (リモート運用想定: tkinter ダイアログが動かないヘッドレス環境用)
    if _allowed_dirs:
        with st.expander(t("📁 許可ディレクトリから選ぶ"), expanded=False):
            _allowed_dir_browser(_allowed_dirs, "_new_file_path")

    st.text_input(t("netCDFファイルのパス"), key="_new_file_path",
                   help=t("許可ディレクトリ: {dirs}",
                          dirs=", ".join(_allowed_dirs)) if _allowed_dirs else None)
    _col_browse, _col_add = st.columns(2)
    # 参照ボタン: 許可ディレクトリ運用では非表示 (tkinter は使わない)
    if not _allowed_dirs:
        _col_browse.button(t("参照..."), on_click=_browse_file_callback,
                            key="browse_file",
                            help=t("ネイティブのファイル選択ダイアログを開く"))
    if _col_add.button(t("追加"), type="primary", key="add_file"):
        abspath = os.path.abspath(os.path.expanduser(
            st.session_state["_new_file_path"]))
        if not _is_path_allowed(abspath, _allowed_dirs):
            st.error(t("パス `{path}` は許可ディレクトリの外にあります。許可: {dirs}",
                       path=abspath, dirs=_allowed_dirs))
        elif os.path.isfile(abspath):
            _add_dataset_to_state(abspath)
            st.rerun()
        else:
            st.error(t("ファイルが見つかりません: {path}", path=abspath))

# --- 編集中のセッションの由来バナー (メインエリア: タイトル直下に常時表示) ---
# 色はタブと同じ対応: スロット = 青 (アプリを動かすマシン側) /
# ファイル = オレンジ (手元 PC 側)。由来は _session_provenance で追跡
# (スロット復元・ファイル読み込み・スロット保存/上書きで更新。"_" 始まりなので
# セッション JSON には保存されず、ブラウザセッション内でのみ有効)
_prov = st.session_state.get("_session_provenance")
if _prov:
    if _prov["kind"] == "slot":
        _slot_banner = (t("サーバー内のセッション『{name}』を編集中", name=_prov["name"])
                        if _remote else
                        t("PC 内のセッション『{name}』を編集中", name=_prov["name"]))
        st.markdown(":blue-background[🗄️ " + _slot_banner + "]")
    else:
        st.markdown(":orange-background[📄 "
                    + t("ファイル『{name}』から読み込んだ内容を編集中",
                        name=_prov["name"]) + "]")

if not st.session_state.get("datasets"):
    st.info(t("サイドバーで netCDF ファイルのパスを指定して「追加」を押してください。\n\n"
            "サンプルデータは `python scripts/make_sample_data.py` で生成できます。"))
    st.stop()

# 全データセットを読み込む
datasets = {}
for item in st.session_state["datasets"]:
    # 「追加」ボタンの検証だけでは不十分: セッションアップロード・サーバスロット復元は
    # session_state に直接パスを注入するため、読み込み直前にも必ず検証する
    if not _is_path_allowed(item["path"], _allowed_dirs):
        st.error(t("パス `{path}` ({dsid}) は許可ディレクトリの外にあります。"
                 "サイドバーの「削除」でこのファイルを外してください。"
                 "許可: {dirs}",
                 path=item["path"], dsid=item["id"], dirs=_allowed_dirs))
        st.stop()
    try:
        datasets[item["id"]] = load_dataset(item["path"], os.path.getmtime(item["path"]))
    except Exception as exc:
        st.error(t("netCDFファイルを開けませんでした — {dsid} ({path}): {exc}",
                 dsid=item["id"], path=item["path"], exc=exc))
        st.stop()
    # 座標ファイル (経緯度が別ファイル)。本体に経緯度があれば無視して通知する。
    # 結び付けは時刻役割の上書き・dim 名整列より前 (両者が lat/lon 座標を見るため)
    _coord_paths = tuple(item.get("coord_paths") or [])
    if _coord_paths:
        for _cp in _coord_paths:
            if not _is_path_allowed(_cp, _allowed_dirs):
                st.error(t("座標ファイル `{path}` ({dsid}) は許可ディレクトリの外にあります。"
                           "許可: {dirs}", path=_cp, dsid=item["id"], dirs=_allowed_dirs))
                st.stop()
        _base_roles = mc_dataset.detect_coord_roles(datasets[item["id"]])
        if _base_roles["lat"] and _base_roles["lon"]:
            st.info(t("`{dsid}` には既に経緯度があるため座標ファイルは無視しました",
                      dsid=item["id"]))
        else:
            try:
                datasets[item["id"]] = load_dataset_with_coords(
                    item["path"], os.path.getmtime(item["path"]), _coord_paths,
                    tuple(os.path.getmtime(p) for p in _coord_paths))
            except Exception as exc:
                st.error(t("座標ファイルを結び付けられませんでした — {dsid}: {exc}",
                           dsid=item["id"], exc=exc))
                st.stop()

# 複数 dataset がある時は **読み込み直後に dim 名整列** を行う (Phase 1)。
# 値チェックは描画レイヤー確定後 (Phase 2) に行う。
# こうすることで variables_by_ds 構築や `.sel` の準備に rename が間に合う。
dataset_renames = {}
_all_ids = [item["id"] for item in st.session_state["datasets"]]
if len(_all_ids) > 1:
    datasets, dataset_renames, _info_msgs, _ = mc_dataset.align_dim_names(
        datasets, _all_ids, axis_roles=[])
    for _msg in _info_msgs:
        st.info(_msg)

# --- 時刻役割の手動上書き (「この次元を時刻として扱う」) ---
# widget (`role_time_override`) は「ファイル情報」expander 内。値を align 後の
# 全 dataset に attrs (ROLE_TIME_OVERRIDE_ATTR) で付与し、detect_coord_roles が
# 最優先で読む。render / scriptgen にも同じ ds が渡るので全経路に行き渡る。
# align より後に適用するのは、整列は自動 roles で行うため (先に当てると他 dataset
# の実時刻次元が上書き先の名前へ誤 rename される)
_time_override = st.session_state.get("role_time_override") or ""
_prev_override = st.session_state.get("role_time_applied") or ""
if _time_override != _prev_override:
    # 上書き切替で「時刻セクション (インデックス) ⇄ 通常の固定 selectbox (値)」と
    # widget 値の意味が変わる dim の選択 key と、時刻数に依存する
    # アニメーション key を初期化する (古い値が残ると selectbox の options 外で落ちる)
    _auto_time = mc_dataset.detect_coord_roles(
        datasets[st.session_state["datasets"][0]["id"]])["time"]
    _affected = {_time_override, _prev_override, _auto_time or ""} - {""}
    for _k in list(st.session_state.keys()):
        if (_k.startswith("anim_")
                or any((_k.startswith("sel_") and _k.endswith(f"_{_d}"))
                       or _k.endswith(f"_sel_{_d}") for _d in _affected)):
            del st.session_state[_k]
    st.session_state["role_time_applied"] = _time_override
if _time_override:
    for _item in st.session_state["datasets"]:
        _ds_or = datasets[_item["id"]]
        if _time_override in _ds_or.coords:
            datasets[_item["id"]] = _ds_or.assign_attrs(
                {mc_dataset.ROLE_TIME_OVERRIDE_ATTR: _time_override})

ds = datasets[st.session_state["datasets"][0]["id"]]
path = st.session_state["datasets"][0]["path"]

roles = mc_dataset.detect_coord_roles(ds)

# 2 次元座標格子 (curvilinear) の投影法の推定 (ds0 のみ)。水平面図の投影法・中心経度・
# 標準緯線の初期値に使う (ClimCORE のランベルト格子など、座標ファイルに投影の属性が
# 無くても格子の形から判定する。ユーザー要望 2026-09-24)
_grid_projection = None
if mc_dataset.is_curvilinear(ds, roles):
    _item0 = st.session_state["datasets"][0]
    _cp0 = tuple(_item0.get("coord_paths") or [])
    _grid_projection = detect_grid_projection_cached(
        _item0["path"], os.path.getmtime(_item0["path"]), _cp0,
        tuple(os.path.getmtime(p) for p in _cp0))

with st.expander(t("ファイル情報"), expanded=False):
    # summary の列名は core が日本語キーで返す (core 層は i18n 不可) — 表示点で訳す
    _col_labels = {"変数名": t("変数名"), "座標名": t("座標名"), "次元": t("次元"),
                   "形状": t("形状"), "長さ": t("長さ"), "範囲": t("範囲"),
                   "単位": t("単位")}
    for item in st.session_state["datasets"]:
        _ds = datasets[item["id"]]
        st.markdown(f"**`{item['id']}`**: `{item['path']}`")
        st.caption(t("データ変数"))
        st.dataframe(pd.DataFrame(mc_dataset.variable_summary(_ds))
                     .rename(columns=_col_labels))
        _coord_rows = mc_dataset.coord_summary(_ds)
        _bare = mc_dataset.bare_dims(_ds)
        st.caption(t("座標変数"))
        if _coord_rows:
            st.dataframe(pd.DataFrame(_coord_rows).rename(columns=_col_labels))
        else:
            st.write(t("(座標変数なし)"))
        if _bare:
            st.caption(t("座標変数を持たない次元 (bare dims): {dims}",
                       dims=", ".join(f"`{d}`" for d in _bare)))
    st.write(t("座標の自動認識 (ds0): "),
             {role: name or t("(なし)") for role, name in roles.items()})
    # 時刻役割の手動上書き。適用は本体フロー冒頭 (align 後) — ここは widget だけ
    _dim_opts = [""] + [str(d) for d in ds.dims
                        if d in ds.coords and ds[d].ndim == 1]
    if st.session_state.get("role_time_override") not in _dim_opts:
        # 別ファイルのセッション残留値など、候補にない値は自動へ戻す
        st.session_state["role_time_override"] = ""
    # t() は widget 生成時に評価して束縛する (format_func 内で遅延評価すると
    # AppTest が run 外で評価したときに言語がずれる)
    _auto_label = t("(自動認識に従う)")
    st.selectbox(
        t("時刻として扱う次元"), _dim_opts,
        format_func=lambda v, _a=_auto_label: _a if v == "" else v,
        key="role_time_override",
        help=t("自動で時刻と認識されない数値次元 (ラグ相関の lag 軸など) を"
               "時刻役割として扱う。時刻送り・アニメーション・時間断面モードが"
               "この次元で使えるようになる"))

# --- 描画モード ---
# 保存値は中立 id ("map" 等 = mode_key の略号)。表示は t("mode.<id>") (i18n)
mode_options = []
# lat/lon 役割が無くてもトラック (軌跡) を描けるデータなら地図モードを開く
if (roles["lat"] and roles["lon"]) or mc_dataset.track_capable(ds):
    mode_options.append("map")
# 2 次元座標 (curvilinear 格子) では lat/lon が dim ではないため、経緯度を軸にとる
# 時間断面は出さない (docs/curvilinear_grid_plan.md の制約)
_curvilinear = mc_dataset.is_curvilinear(ds, roles)
# 鉛直断面は 2 次元座標格子でも出す (格子線に沿う断面と、経路 (等緯度線・等経度線・大円)
# に沿って内挿する断面。docs/section_extension_plan.md)。時間断面は従来どおり出さない
if roles["vertical"] and (roles["lon"] or roles["lat"]):
    mode_options.append("vsec")
if (roles["time"] and (roles["vertical"] or roles["lat"] or roles["lon"])
        and not _curvilinear):
    mode_options.append("tsec")
# 1次元プロットは座標を持つ次元が1つ以上あれば常に可能
if any(d in ds.coords for d in ds.dims):
    mode_options.append("line")
# 1次元プロット(集計): lat/lon 次元を持たない数値変数があれば可能
if mc_dataset.nongeo_variables(ds, roles):
    mode_options.append("dist")
# 2次元プロット (散布図): 変数が1つでも可能 (x軸とy軸に同じ変数を選べる)
if len(list(ds.data_vars)) >= 1:
    mode_options.append("scatter")
# 2次元プロット(集計): lat/lon 次元を持たない数値変数があれば可能
if mc_dataset.nongeo_variables(ds, roles):
    mode_options.append("agg")
if not mode_options:
    st.error(t("描画可能な座標の組み合わせが見つかりません。"))
    st.stop()

with st.sidebar:
    # --- 保存 (読み込みは最上部の「データ・セッション読み込み」) ---
    import json as _json_mod
    section_header(t("保存"))
    with st.expander(t("セッション"), expanded=False):
        st.caption(t("ファイル名・変数・時刻・領域・レイヤー構成を含む**全状態**を JSON 保存。"
                   "復元は「データ・セッション読み込み」→「セッション復元」から"))
        session_state_json = _serialize_session()
        # 復元側「作業の再開 → セッション復元」と同じ順・同じ色のタブ構造
        _save_tab_slot, _save_tab_file = st.tabs([
            f":blue[{t('サーバーへ') if _remote else t('PC へ')}]",
            f":orange[{t('ダウンロード')}]"])
        with _save_tab_slot:
            # 名前を付けて保存した直後の成功メッセージ (st.rerun() をまたいで表示)
            _saved_msg = st.session_state.pop("_session_saved_msg", None)
            if _saved_msg:
                st.success(t("保存しました ({n} 項目): {path}",
                             n=_saved_msg[0], path=_saved_msg[1]))
            st.caption(t("サーバー側に保存する (手元 PC には残らない)。"
                         "`session_dirs` から保存先を選び、名前を付けて保存")
                       if _remote else
                       t("この PC の保存領域 (`session_dirs`) に、名前を付けて保存"))
            _session_dirs = _load_session_dirs()
            # 復元・名前を付けて保存の直後は、編集中のスロットを
            # 「保存するスロット」に自動選択する (widget 描画前にここで反映。
            # ハンドラからの直接代入は描画順の制約があるため pending 経由)
            _ow_pending = st.session_state.pop("_session_ow_pending", None)
            if _ow_pending:
                if _ow_pending["dir"] in _session_dirs:
                    st.session_state["_session_save_dir"] = _ow_pending["dir"]
                st.session_state["_session_overwrite_slot"] = _ow_pending["name"]
            # 表示の短縮と全文 caption は「作業の再開」の復元側と同じ流儀
            _session_dir = st.selectbox(
                t("保存先ディレクトリ"), _session_dirs, key="_session_save_dir",
                format_func=shorten_label,
                help=t("`~/.climcanvas/config.toml` の `session_dirs` に追加候補を書ける"))
            st.caption(md_plain(_session_dir))
            # 既存スロットへの上書き (スロット選択 + 上書きボタン)
            _existing_save = _list_sessions(_session_dir)
            # 選択値が一覧に無ければ既定へ戻す (dir 切替・外部削除への防御)
            if st.session_state.get("_session_overwrite_slot") not in _existing_save:
                st.session_state.pop("_session_overwrite_slot", None)
            if _existing_save:
                _c_ow_slot, _c_ow_btn = st.columns(
                    [3, 1], vertical_alignment="bottom")
                # 3/4 幅の列に入るので短縮幅も狭める (全幅は既定 24)
                _ow_slot = _c_ow_slot.selectbox(
                    t("上書きするファイル名"), _existing_save,
                    format_func=lambda v: shorten_label(v, 18),
                    key="_session_overwrite_slot")
                st.caption(t("選択中: {name}", name=md_plain(_ow_slot)))
                if _c_ow_btn.button(t("上書き"), key="_session_overwrite_btn",
                                    help=t("選択したファイルに現在のセッションを"
                                           "上書き保存する")):
                    try:
                        n = _save_session_to_disk(_session_dir, _ow_slot)
                        # メインの由来バナーはこのハンドラより前に描画済みの
                        # ため st.rerun() で即反映し、成功メッセージは
                        # _session_saved_msg 経由で持ち越す
                        st.session_state["_session_provenance"] = {
                            "kind": "slot", "name": _ow_slot}
                        st.session_state["_session_saved_msg"] = (
                            n, str(_session_path(_session_dir, _ow_slot)))
                        st.rerun()
                    except Exception as e:
                        st.error(t("保存失敗: {e}", e=e))
            # 新しいファイル名で保存
            _c_save_name, _c_save_btn = st.columns(
                [3, 1], vertical_alignment="bottom")
            _session_name = _c_save_name.text_input(
                t("新規ファイル名"), value="", placeholder=t("例: el-nino-2015"),
                key="_session_save_name")
            if _c_save_btn.button(t("保存"), key="_session_save_btn",
                                    disabled=not _session_name.strip()):
                try:
                    n = _save_session_to_disk(_session_dir, _session_name.strip())
                    # 「作業の再開」のセッション一覧とメインの由来
                    # バナーはこのハンドラより前に描画済みのため st.rerun() で
                    # 即反映し、成功メッセージは _session_saved_msg 経由で持ち越す
                    st.session_state["_session_provenance"] = {
                        "kind": "slot", "name": _session_name.strip()}
                    st.session_state["_session_ow_pending"] = {
                        "dir": _session_dir, "name": _session_name.strip()}
                    st.session_state["_session_saved_msg"] = (
                        n, str(_session_path(_session_dir, _session_name.strip())))
                    st.rerun()
                except Exception as e:
                    st.error(t("保存失敗: {e}", e=e))
        with _save_tab_file:
            st.caption(t("手元 PC へダウンロード保存する"))
            st.download_button(
                t("セッションの保存"),
                data=_json_mod.dumps(session_state_json, ensure_ascii=False, indent=2),
                file_name="climcanvas_session.json", mime="application/json",
                key="session_dl",
                help=t("保存先は手元ブラウザのダウンロード設定に従う。ファイル名は自由 "
                       "(保存時に付けるか、保存後に変えてよい)"))
    with st.expander(t("プリセット"), expanded=False):
        st.caption(t(
            "データに依存しない好みの設定値 "
            "(グリッド・ティック・線種・カラーマップ・カラーバー など) を "
            "`{path}` に保存。アプリ起動時に自動適用される。",
            path=STARTUP_PRESET_PATH))
        _has_startup = STARTUP_PRESET_PATH.is_file()
        st.caption(t("✓ 設定済み") if _has_startup else t("(未設定)"))
        if st.button(t("プリセットの保存"), key="save_startup_preset"):
            n = _save_startup_preset()
            st.success(t("保存しました ({n} 項目): {path}", n=n, path=STARTUP_PRESET_PATH))
        if _has_startup and st.button(t("起動時プリセットを削除"), key="delete_startup_preset"):
            _delete_startup_preset()
            st.success(t("削除しました"))


# --- パネル構成・図全体の書式 (第6段階: 複数パネル) ---

# 描画モード (中立 id = widget key の mode_key 略号 → 日本語原文ラベル)。
# 表示は tr_labels() で現在言語に変換する (キー = 原文の gettext 流)
_MODE_LABELS = {"map": "水平断面図", "vsec": "鉛直断面図",
                "tsec": "時間断面図", "line": "1次元プロット",
                "dist": "1次元プロット(集計)",
                "scatter": "2次元プロット",
                "agg": "2次元プロット(集計)"}
_MODE_IDS = tuple(_MODE_LABELS)
# プリセット縦横比 -> width/height (= 横/縦) の比率。
# 規定の幅は 6.4 inch 固定、高さは比率から自動計算。
_ASPECT_PRESETS = {
    "default": 6.4 / 4.8,
    "square": 1.0,
    "a4": 2.0 ** 0.5,
    "wide": 16.0 / 9.0,
    "custom": None,
}
_ASPECT_PRESET_LABELS = {
    "default": "6.4:4.8 (matplotlib default)",
    "square": "1:1",
    "a4": "√2:1 (A4)",
    "wide": "16:9",
    "custom": "任意の数字",
}
_PRESET_DEFAULT_WIDTH = 6.4
_FONT_MODE_LABELS = {"common": "よく使うフォント", "all": "全フォントから検索"}
_BOXASPECT_PRESET_LABELS = {"square": "1:1",
                            "default": "6.4:4.8 (matplotlib default)",
                            # None = set_box_aspect を呼ばず、Figure サイズと余白に従う (2026-10-07)
                            "none": "固定しない (Figure サイズに従う)",
                            "custom": "任意"}
_LAT_LABEL_LABELS = {"inline": "図中 (既定)", "edge": "枠沿い", "none": "非表示"}
_LAT_EDGE_SIDE_LABELS = {"both": "両方", "left": "左のみ", "right": "右のみ"}
_TICK_DIR_LABELS = {"out": "外側", "in": "内側", "inout": "両方"}
_VSEC_ORIENT_LABELS = {"lon_height": "経度–高度 (緯度を固定)",
                       "lat_height": "緯度–高度 (経度を固定)",
                       # 2 次元座標格子の格子線断面 (内挿なし) と、経路断面 (双一次内挿。
                       # 大円は 1 次元格子でも選べる)。docs/section_extension_guide.md
                       "grid_row": "格子の行に沿う (x–高度、行を固定)",
                       "grid_col": "格子の列に沿う (y–高度、列を固定)",
                       "parallel": "等緯度線に沿う (経度–高度、内挿)",
                       "meridian": "等経度線に沿う (緯度–高度、内挿)",
                       "great_circle": "2 点間の大円に沿う (距離–高度、内挿)"}
_TSEC_KIND_LABELS = {"time_height": "時間–高度", "time_lat": "時間–緯度",
                     "time_lon": "時間–経度"}
_DIST_FAMILY_LABELS = {"hist": "ヒストグラム・ECDF", "box": "箱ひげ・バイオリン"}
_DIST_ORIENT_LABELS = {"vertical": "縦", "horizontal": "横"}
_DRAW2D_LABELS = {"scatter_bubble": "散布図・バブル",
                  "heatmap": "categorical heatmap"}
# 変数選択の「(なし)」番兵 (layer_ui._NONE_OPTION と同じ値であること)
_NONE_OPTION = "(none)"
_OUT_BG_LABELS = {"white": "白 (既定)", "color": "色を指定",
                  "transparent": "透明"}
_PATH_STYLE_LABELS = {"absolute": "絶対パス", "relative": "相対パス"}


def _script_dataset_paths(path_style: str) -> dict:
    """再現スクリプトに書く netCDF のパス (dataset id → パス)。静止図とアニメーションで共用。

    path_style は「netCDFパスの形式」(absolute / relative)。相対パスはアプリを起動した
    ディレクトリからの相対 (2026-09-29 まではアニメーションのスクリプトだけ常に絶対パスだった)。
    """
    return {item["id"]: (os.path.abspath(item["path"]) if path_style == "absolute"
                         else os.path.relpath(item["path"]))
            for item in st.session_state["datasets"]}

# render.RenderError の msg_id → 日本語原文 (i18n 第4段階)。コア層の str() は
# 英語で、アプリではここから現在の UI 言語に翻訳して表示する。
# msg_id の集合は RenderError.MESSAGES と一致すること (tests/test_i18n.py が検査)
_RENDER_ERROR_LABELS = {
    "levels_invalid":
        "色レベルのリスト (style.levels) は昇順・重複なしの2個以上で"
        "指定してください: {levels!r}",
    "unsupported_projection": "未対応の投影法です: {name}",
    "lon_range_dateline":
        "指定された経度範囲 [{lo:g}, {hi:g}] は日付変更線をまたぎますが、"
        "データの経度範囲 [{data_min:g}, {data_max:g}] は全球を覆っていません。"
        "指定範囲のうちデータに含まれない領域があるため、平均を計算できません。",
    "lon_range_outside":
        "指定された経度範囲 [{lo:g}, {hi:g}] "
        "(データ規約に正規化後 [{lo_n:g}, {hi_n:g}]) は、"
        "データの経度範囲 [{data_min:g}, {data_max:g}] からはみ出しています。"
        "平均を計算できません。",
    "curvilinear_coords_nonfinite":
        "2 次元の経緯度座標 ({lon!r} / {lat!r}) の描画範囲内に非有限値 (NaN / inf) が "
        "{n} 個あります。matplotlib は角の位置が未定義のメッシュを描けません。"
        "その格子点の経緯度を隣の点などで埋めてデータ側をマスクするか、"
        "領域指定でその格子点を外してください。",
    "region_outside_grid":
        "指定された領域 (経度 [{lon_min:g}, {lon_max:g}]、緯度 [{lat_min:g}, "
        "{lat_max:g}]) にこの 2 次元座標格子の格子点がありません "
        "(データの経度 [{data_lon_min:g}, {data_lon_max:g}]、"
        "緯度 [{data_lat_min:g}, {data_lat_max:g}])。",
    "section_path_no_lonlat":
        "経路に沿った断面には経度・緯度の座標が必要ですが、データセットから認識できませんでした。",
    "section_path_dim_conflict":
        "データセットに次元 {dim!r} が既にあります。この名前は断面の経路用に予約されているので、次元名を変えてください。",
    "section_path_unknown_kind":
        "断面の経路の種類が不明です: {kind!r} ('parallel' / 'meridian' / 'great_circle' のいずれか)",
    "section_path_antipodal":
        "断面の始点と終点が地球の反対側 (対蹠点) にあり、2 点を結ぶ大円が一つに決まりません。どちらかの点を動かしてください。",
    "section_path_outside_grid":
        "断面の経路 ({kind}) がデータの格子 (経度 {lon_min:g}〜{lon_max:g}、緯度 {lat_min:g}〜{lat_max:g}) の外にあります。経路を動かすか、経度の規約を確認してください。",
    "terrain_vertical_units_unknown":
        "地形マスクには鉛直座標 {dim!r} が気圧か高度かの判定が必要ですが、units 属性 {units!r} を認識できません (hPa、Pa、m、km など)。",
    "terrain_method_mismatch":
        "地形マスクの方法 {method!r} は、鉛直座標が{kind} ({dim!r} [{units}]) のときは使えません。{expected} を使ってください。",
    "terrain_variable_missing":
        "地形マスクの変数 {var!r} がデータセット {dataset!r} にありません。",
    "terrain_units_unknown":
        "地形の変数 {var!r} の units 属性 {units!r} を鉛直座標の単位 {target!r} に換算できません ({expected} など)。",
    "terrain_profile_dims":
        "地形の変数 {var!r} は、ほかの次元を固定すると断面の横軸 {x_dim!r} だけになる必要がありますが、次元 {dims} が残っています。",
    "terrain_no_lonlat":
        "データセット {dataset!r} の地形の変数 {var!r} に経度・緯度の座標が認識できないため、断面の経路に沿って標本化できません (そのデータセットに座標ファイルを付けてください)。",
    "section_overlay_panel_missing":
        "地図が断面の経路を描くために参照しているパネル (panel_id {panel_id!r}) が図にありません。",
    "section_overlay_not_section":
        "地図が断面の経路として参照しているパネル {panel_id!r} は鉛直断面ではないか、経路を決められません。",
    "layout_too_small":
        "レイアウト {nrows}×{ncols} にパネル {n_panels} 個は配置できません。"
        "行数・列数を増やしてください。",
    "mosaic_empty": "mosaic 文字列が空です",
    "mosaic_ragged": "mosaic の各行の文字数 (列数) を揃えてください: {rows}",
    "mosaic_count_mismatch":
        "mosaic のラベル数 {n_labels} ({labels}) がパネル数 {n_panels} と"
        "一致しません",
    "mosaic_not_rect": "mosaic のラベル '{label}' のセルが矩形になっていません",
    "ratio_len_mismatch":
        "{name} の個数 {n} がグリッドの {unit} 数 {expected} と一致しません",
    "ratio_invalid": "{name} は正の数値のリストで指定してください: {values!r}",
    "no_time_dim": "時刻次元が見つからないためアニメーションできません",
    "no_frames": "生成するフレームがありません",
    "ffmpeg_missing":
        "ffmpeg が見つかりません。MP4 出力には ffmpeg のインストールが必要です。",
    "no_time_dim_script":
        "時刻次元が見つからないためアニメーションスクリプトを生成できません",
    "all_nan": "変数の値がすべて欠損 (NaN) のため、色レベルを自動決定できません",
    "constant_value":
        "値が一定 (最小値 = 最大値 = {vmin:g}) のため、pcolormesh の"
        "等間隔レベルを自動決定できません。レベルを直接指定するか、"
        "値の範囲 (最小値・最大値) を指定してください",
    "map_scatter_dims_unfixed":
        "散布図を描けません: 変数 {var} に固定されていない次元 {dims} が"
        "残っています (点 {n_points} 個に対し値 {n_values} 個)。"
        "残りの次元 (時刻など) を固定してください",
    "maskout_var_missing":
        "マスク変数 {var} がデータセットにありません",
    "maskout_var_extra_dims":
        "マスク変数 {var} は描画変数 {target} に無い次元 {dims} を持つため、"
        "位置を揃えられずマスクに使えません",
    "maskout_var_missing_dims":
        "マスク変数 {var} に描画軸の次元 {dims} が無いため、マスクに使えません",
    "vector_shape_mismatch":
        "ベクトルの x成分 {u_var} (形状 {u_shape}) と y成分 {v_var} (形状 {v_shape}) の"
        "格子が違うため描けません。両成分は同じ次元名・同じ形状の格子にしてください "
        "(別ファイルの成分を組み合わせるときは格子の一致を確認)",
    "degenerate_geometry":
        "cartopy の既知の不具合 (退化ポリゴン) により描画に失敗しました。"
        "色レベルの端がデータの最小値・最大値や定数領域の値とちょうど一致すると"
        "起きることがあります。「値の範囲を自動」に戻す、最小値・最大値を"
        "データ範囲から少しずらす、または塗りつぶしの描画方法を pcolormesh に"
        "すると回避できます",
    "degenerate_gridline":
        "cartopy の既知の不具合により描画に失敗しました (緯度経度線が地図の縁で"
        "1点に退化)。Orthographic など縁が曲線の投影で特定の中心経度・緯度のとき"
        "に起きます。中心経度・緯度を少しずらす (回転アニメーションなら経由点や"
        "フレーム数を変える) か、緯度経度線を非表示にすると回避できます",
}


def _anim_error_text(exc, n_done: int, n_frames: int, centers) -> str:
    """アニメーション失敗の表示文: 失敗したフレーム番号 (+ 回転中なら投影中心) を添える。"""
    i = n_done + 1
    if centers is not None and 0 <= n_done < len(centers):
        lon, lat = centers[n_done]
        return t("フレーム {i}/{n} (中心 {lon:g}°, {lat:g}°) で失敗: {exc}",
                 i=i, n=n_frames, lon=lon, lat=lat, exc=_error_text(exc))
    return t("フレーム {i}/{n} で失敗: {exc}", i=i, n=n_frames, exc=_error_text(exc))


def _error_text(exc) -> str:
    """例外を表示用文字列にする。RenderError は現在の UI 言語に翻訳する。"""
    if isinstance(exc, mc_render.RenderError):
        ja = _RENDER_ERROR_LABELS.get(exc.msg_id)
        if ja is not None:
            return t(ja, **exc.params)
    return str(exc)


def _copy_panel_state(src_pid: int, dst_pid: int,
                      skip_prefixes: tuple[str, ...] = ()) -> None:
    """パネル src の widget 状態をすべて dst にコピーする (複製と共通編集のモード変更)。

    パネルスコープの key は mode_key (例 "map0") を `_` 区切りトークンとして
    含むため、トークン単位で置換した新キーへ値を写す。ボタン等 (セッション除外キー) は
    programmatic 設定不可なのでスキップ。dst は widget 未描画のパネルであること。
    skip_prefixes で始まる key は写さない (dst 自身の値を残す。共通編集の個別設定用)。
    """
    token_map = {f"{m}{src_pid}": f"{m}{dst_pid}" for m in _MODE_IDS}
    for key in list(st.session_state.keys()):
        if _is_session_excluded(key):
            continue
        if skip_prefixes and key.startswith(skip_prefixes):
            continue
        parts = key.split("_")
        if not any(p in token_map for p in parts):
            continue
        new_key = "_".join(token_map.get(p, p) for p in parts)
        st.session_state[new_key] = copy.deepcopy(st.session_state[key])
    for prefix in ("plot_mode", "panel_cfg"):
        src_key = f"{prefix}_{src_pid}"
        if src_key in st.session_state:
            st.session_state[f"{prefix}_{dst_pid}"] = copy.deepcopy(
                st.session_state[src_key])


# --- 全パネル共通編集 (仕様22.4「共通設定」相当の同期編集モード) ---

# 個別設定として同期しない panel 設定のトップレベルキー (仕様22.4 の個別設定例)
_COMMON_SKIP_CFG_KEYS = {"panel_id", "title", "title_fontsize", "texts",
                         "markers", "label"}
# 個別設定として同期しない widget key のプレフィックス (上と対応)。
# indivcfg_{mode_key} は widget ではなく、パネルのモードごとの個別設定の控え
# (_remember_individual_cfg。共通編集でモードを変えたときに各パネル自身の値を使う)
_COMMON_SKIP_WIDGET_PREFIXES = ("title_", "txt", "plabel_", "texts_",
                                "mk", "markers_", "indivcfg_")
# モードごとに控える個別設定 (panel_id は UI が設定しないので除く)
_INDIVIDUAL_CFG_KEYS = ("title", "title_fontsize", "texts", "markers", "label")


def _remember_individual_cfg(pid: int, cfg: dict | None = None) -> None:
    """パネル pid の今のモードの個別設定 (タイトル・文字列・記号・ラベル) を控える。

    cfg を省くと、キャッシュ済みの panel_cfg_{pid} (= 今のモードの設定) から取る。
    """
    mode = st.session_state.get(f"plot_mode_{pid}")
    if cfg is None:
        cfg = st.session_state.get(f"panel_cfg_{pid}")
    if mode in _MODE_IDS and cfg:
        st.session_state[f"indivcfg_{mode}{pid}"] = {
            k: copy.deepcopy(cfg.get(k)) for k in _INDIVIDUAL_CFG_KEYS}


def _individual_cfg(pid: int, mode: str) -> dict:
    """パネル pid のモード mode での個別設定。控えが無ければ UI の初期値と同じ既定
    (_texts_and_labels_ui を何も操作せずに描いたときの値)。

    限界: この仕組みより前のセッションを復元した直後など、そのモードの widget 状態は
    あるのに控えが無いときも既定になる (そのパネルを選んで表示すれば widget から
    作り直される)。
    """
    saved = st.session_state.get(f"indivcfg_{mode}{pid}")
    if saved:
        return copy.deepcopy(saved)
    return {"title": None, "title_fontsize": 12, "texts": [], "markers": [],
            "label": mc_config.default_panel_label()}
# layers_* (構成リスト + next counter) は汎用ミラーではなく
# _sync_common_layers が同モードのパネルに限って同期する
_COMMON_SKIP_WIDGET_EXACT_PREFIXES = ("layers_",)


def _replace_pid_tokens(key: str, src_pid: int, dst_pid: int) -> str:
    """widget key 中の mode_key トークン (例 map0) を別パネルのものに置換する。"""
    if key == f"plot_mode_{src_pid}":
        return f"plot_mode_{dst_pid}"
    token_map = {f"{m}{src_pid}": f"{m}{dst_pid}" for m in _MODE_IDS}
    return "_".join(token_map.get(p, p) for p in key.split("_"))


def _panel_state_snapshot(pid: int) -> dict:
    """パネル pid にスコープされた widget/session 状態のスナップショットを返す。"""
    tokens = {f"{m}{pid}" for m in _MODE_IDS}
    out = {}
    for k, v in st.session_state.items():
        if _is_session_excluded(k):
            continue
        if k == f"plot_mode_{pid}" or any(t in tokens for t in k.split("_")):
            out[k] = copy.deepcopy(v)
    return out


def _cfg_diff_paths(old, new, prefix=()):
    """ネストした panel 設定の葉の差分を (path, new_value) で列挙する。

    list は同じ長さなら要素ごとに再帰し、長さが違えば構造変更とみなして
    スキップする (レイヤーの追加・削除は全パネルへ伝播しない)。
    """
    if isinstance(old, dict) and isinstance(new, dict):
        for k, nv in new.items():
            if not prefix and k in _COMMON_SKIP_CFG_KEYS:
                continue
            if k not in old:
                yield prefix + (k,), nv
            else:
                yield from _cfg_diff_paths(old[k], nv, prefix + (k,))
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) == len(new):
            for i, (ov, nv) in enumerate(zip(old, new)):
                yield from _cfg_diff_paths(ov, nv, prefix + (i,))
        return
    if old != new:
        yield prefix, new


def _apply_cfg_path(cfg, path, value) -> None:
    """path の位置に value を書き込む。途中の構造が一致しなければ何もしない。"""
    node = cfg
    for key in path[:-1]:
        if isinstance(node, dict):
            node = node.get(key)
        elif isinstance(node, list) and isinstance(key, int) and key < len(node):
            node = node[key]
        else:
            return
        if node is None:
            return
    last = path[-1]
    if isinstance(node, dict):
        node[last] = value
    elif isinstance(node, list) and isinstance(last, int) and last < len(node):
        node[last] = value


def _sync_common_layers(master_pid: int, new_cfg: dict, other_pids: list,
                        prev_state: dict) -> None:
    """全パネル共通モード: レイヤーの追加・削除・種類変更を他パネルへ反映する。

    layers_{mode_key} の項目リスト ({id, kind}) を代表パネルと同じ内容に揃え、
    キャッシュ済み panel 設定の layers も同じ構成に組み替える。id と kind が
    引き継がれたレイヤーは対象パネルの既存設定 (変数選択など) を保持し、
    追加・種類変更されたレイヤーは代表パネルの設定のコピーで埋める。
    レイヤー構成は描画モードに属するため、**同じ描画モードのパネルのみ**対象。
    """
    mode = st.session_state.get(f"plot_mode_{master_pid}")
    mk = mode if mode in _MODE_IDS else None
    if not mk:
        return
    lkey = f"layers_{mk}{master_pid}"
    new_items = st.session_state.get(lkey)
    prev_items = prev_state.get(lkey)
    if (not isinstance(new_items, list) or prev_items is None
            or prev_items == new_items):
        return
    master_layers = new_cfg.get("layers") or []
    prev_ids = [it.get("id") for it in prev_items]
    for pid in other_pids:
        if st.session_state.get(f"plot_mode_{pid}") != mode:
            continue
        st.session_state[f"layers_{mk}{pid}"] = copy.deepcopy(new_items)
        if f"{lkey}_next" in st.session_state:
            st.session_state[f"layers_{mk}{pid}_next"] = (
                st.session_state[f"{lkey}_next"])
        tcfg = st.session_state.get(f"panel_cfg_{pid}")
        if not tcfg:
            continue
        target_layers = tcfg.get("layers") or []
        rebuilt = []
        for idx, it in enumerate(new_items):
            kept = None
            if it.get("id") in prev_ids:
                old_pos = prev_ids.index(it.get("id"))
                if (old_pos < len(target_layers)
                        and prev_items[old_pos].get("kind") == it.get("kind")
                        and target_layers[old_pos].get("kind") == it.get("kind")):
                    kept = target_layers[old_pos]
            if kept is None and idx < len(master_layers):
                kept = copy.deepcopy(master_layers[idx])
            if kept is not None:
                rebuilt.append(kept)
        tcfg["layers"] = rebuilt


def _propagate_common_edits(master_pid: int, new_cfg: dict) -> None:
    """全パネル共通モード: 代表パネルで加えた変更 (差分) を全パネルへ反映する。

    - widget 状態の差分を token 置換して各パネルへミラー
      (後でそのパネルを個別編集しても変更が保持される)
    - キャッシュ済み panel 設定 (panel_cfg_{pid}) にも同じ差分を適用
      (図が即時更新される)。layers 配下は同じ index のレイヤー種別が
      一致するときのみ適用
    - レイヤーの追加・削除・種類変更は _sync_common_layers が
      同じ描画モードのパネルへ反映 (2026-07-09 追加)
    - 描画モードの変更は差分ではなく全設定コピー (_copy_panel_state)。ただし
      個別設定 (タイトル・文字列・記号・パネルラベル) は写さず、各パネル自身の
      新しいモードでの設定 (控え indivcfg_、無ければ既定) を使う (2026-09-29 修正:
      以前は全モードの個別設定と panel_cfg が代表パネルのもので上書きされていた)
    - タイトル・任意文字列・パネルラベルは個別設定として同期しない
    """
    other_pids = [p["id"] for p in st.session_state["panels"]
                  if p["id"] != master_pid]
    snap = st.session_state.get("_common_ws_snapshot")
    cur = _panel_state_snapshot(master_pid)
    changed = {}
    prev_state = {}
    if snap and snap.get("pid") == master_pid:
        prev_state = snap["state"]
        for k, v in cur.items():
            if k not in prev_state or prev_state[k] != v:
                changed[k] = v
    st.session_state["_common_ws_snapshot"] = {"pid": master_pid, "state": cur}
    if not changed or not other_pids:
        return

    if f"plot_mode_{master_pid}" in changed:
        # モード変更: 差分同期では追いきれないので全設定コピー。個別設定は写さず、
        # 各パネルの今のモードの個別設定を控えてから、新しいモードでの自分の値を使う
        new_mode = st.session_state.get(f"plot_mode_{master_pid}")
        for pid in other_pids:
            _remember_individual_cfg(pid)
            _copy_panel_state(master_pid, pid,
                              skip_prefixes=_COMMON_SKIP_WIDGET_PREFIXES)
            cfg = copy.deepcopy(new_cfg)
            cfg.update(_individual_cfg(pid, new_mode))
            st.session_state[f"panel_cfg_{pid}"] = cfg
        return

    prev_cfg = st.session_state.get(f"panel_cfg_{master_pid}")
    diffs = list(_cfg_diff_paths(prev_cfg, new_cfg)) if prev_cfg else []
    master_layers = new_cfg.get("layers") or []
    for pid in other_pids:
        # 1) widget 状態のミラー
        for k, v in changed.items():
            if k.startswith(_COMMON_SKIP_WIDGET_PREFIXES):
                continue
            if k.startswith(_COMMON_SKIP_WIDGET_EXACT_PREFIXES):
                continue
            nk = _replace_pid_tokens(k, master_pid, pid)
            if nk != k:
                st.session_state[nk] = copy.deepcopy(v)
        # 2) キャッシュ済み panel 設定への差分適用
        tcfg = st.session_state.get(f"panel_cfg_{pid}")
        if not tcfg:
            continue
        target_layers = tcfg.get("layers") or []
        for cfg_path, value in diffs:
            if cfg_path and cfg_path[0] == "layers" and len(cfg_path) >= 2:
                li = cfg_path[1]
                if (not isinstance(li, int) or li >= len(target_layers)
                        or li >= len(master_layers)
                        or target_layers[li].get("kind")
                        != master_layers[li].get("kind")):
                    continue
            _apply_cfg_path(tcfg, cfg_path, copy.deepcopy(value))

    # 3) レイヤー構成 (追加・削除・種類変更) の同期 — 同じ描画モードのパネルのみ
    _sync_common_layers(master_pid, new_cfg, other_pids, prev_state)


# パネル構成はメインエリア (「ファイル情報」expander の直下) に出す。
# コードの実行位置はこのままで良い — ファイル情報より後、メインエリアには
# まだ何も描画されていないため、この expander がその直下に並ぶ。
with st.expander(t("パネル構成"), expanded=False):
    _panels = st.session_state.setdefault("panels", [{"id": 0}])
    st.session_state.setdefault("panels_next",
                                max(p["id"] for p in _panels) + 1)
    c_r, c_c, _sp_grid = st.columns([1, 1, 3])
    grid_nrows = int(c_r.number_input(t("行数"), 1, 8, 1, key="grid_nrows"))
    grid_ncols = int(c_c.number_input(t("列数"), 1, 8, 1, key="grid_ncols"))
    # mosaic: セル結合・空きセルを使う配置。指定時は行数・列数より優先される
    grid_mosaic = None
    _mosaic_txt = st.text_input(
        t("セルの結合・空きセル (mosaic)"), value="", key="grid_mosaic",
        placeholder=t("例: ABC;DEE (空欄 = 上の行数×列数に行優先で配置)"),
        help=t("行を ; (または改行) で区切り、1文字 = 1セル。同じ文字を矩形に"
             "並べるとそのパネルのセル結合、\".\" は空きセル。文字の初出順 "
             "(左上から行優先) が下のパネル一覧の順に対応する。指定時は"
             "行数・列数より優先される。例: 2行3列で下段の2〜3列目を結合 = "
             "ABC;DEE"))
    if _mosaic_txt.strip():
        try:
            _m_nr, _m_nc, _ = mc_render.parse_mosaic(_mosaic_txt, len(_panels))
            grid_mosaic = _mosaic_txt.strip()
            grid_nrows, grid_ncols = _m_nr, _m_nc
            st.caption(t("mosaic 配置: {nr}行 × {nc}列", nr=_m_nr, nc=_m_nc))
        except ValueError as _e:
            st.error(t("mosaic が不正なため行優先で配置します — {e}", e=_error_text(_e)))
    # 行・列の大きさの比率 (GridSpec の width_ratios / height_ratios)。
    # 空欄 = 均等 (従来どおり)。mosaic 使用時は mosaic のグリッドに適用される。
    # 不正・個数不一致はエラー表示して均等にフォールバック (mosaic と同じ流儀)
    _rc1, _rc2 = st.columns(2)
    _wr_txt = _rc1.text_input(
        t("列の幅の比率"), value="", key="grid_wratios",
        placeholder=t("例: 2, 1, 1 (空欄 = 均等)"),
        help=t("グリッドの各列の幅の比率をカンマ区切りの正の数値 (小数も可) で。"
             "個数は列数と一致させる。mosaic のセル結合と併用でき、結合セルは"
             "またいだ行・列の合計の大きさになる"))
    _hr_txt = _rc2.text_input(
        t("行の高さの比率"), value="", key="grid_hratios",
        placeholder=t("例: 2, 1 (空欄 = 均等)"),
        help=t("グリッドの各行の高さの比率をカンマ区切りの正の数値 (小数も可) で。"
             "個数は行数と一致させる"))
    def _validated_ratios(txt, expected, name):
        if not txt.strip():
            return None
        vals = parse_ratio_list(txt)
        if vals is None:
            st.error(t("{name} が読めません (正の数値をカンマ区切りで入力): {text}",
                     name=name, text=txt))
            return None
        if len(vals) != expected:
            st.error(t("{name} の個数 {n} がグリッドの行・列数 {expected} と"
                     "合いません。均等のまま描画します。",
                     name=name, n=len(vals), expected=expected))
            return None
        return vals

    grid_wratios = _validated_ratios(_wr_txt, grid_ncols, t("列の幅の比率"))
    grid_hratios = _validated_ratios(_hr_txt, grid_nrows, t("行の高さの比率"))
    if grid_mosaic is None and grid_nrows * grid_ncols < len(_panels):
        st.error(t("グリッド {nr}×{nc} にパネル {n} 個は"
                 "配置できません。行数・列数を増やすかパネルを削除してください。",
                 nr=grid_nrows, nc=grid_ncols, n=len(_panels)))
    # パネル一覧 (並べ替え・複製・削除。「編集するパネル」ボタンの描画前に処理する)
    for _i, _p in enumerate(list(_panels)):
        _pid = _p["id"]
        _mode_lbl = tr_labels(_MODE_LABELS).get(
            st.session_state.get(f"plot_mode_{_pid}"), t("(未設定)"))
        # 1 行 = ラベル + ← → 複製 削除。旧 [2, 0.5, 0.5, 1, 1, 3] は既定幅のサイドバーで
        # 0.5 の列 (約 18 px) がボタン (約 35 px) より狭く「←」「→」が重なった。右端の
        # 空き列を無くし列間を 0.5rem に詰めて、既定幅でも 1 列 ≈ 38 px を確保する
        c_lbl, c_prev, c_next, c_dup, c_del = st.columns(
            [2, 1, 1, 1, 1], gap="xsmall", vertical_alignment="center")
        c_lbl.markdown(f"**{_i + 1}**: {_mode_lbl}")
        # 配置順の入れ替え (行優先の並び。パネル番号は配置順に振り直される)
        if c_prev.button("←", key=f"_mv_prev_{_pid}", disabled=(_i == 0),
                         help=t("配置順を1つ前へ (グリッドの左/上方向)")):
            _panels[_i - 1], _panels[_i] = _panels[_i], _panels[_i - 1]
            st.rerun()
        if c_next.button("→", key=f"_mv_next_{_pid}",
                         disabled=(_i == len(_panels) - 1),
                         help=t("配置順を1つ後ろへ (グリッドの右/下方向)")):
            _panels[_i + 1], _panels[_i] = _panels[_i], _panels[_i + 1]
            st.rerun()
        # 複製・削除はアイコンボタン (文字ラベルは 1.64 以降折り返さず「複…」に切れる)
        if c_dup.button("", icon=":material/content_copy:", key=f"dup_panel_{_pid}",
                        help=t("このパネルの設定をコピーした新しいパネルを追加する")):
            _new_pid = st.session_state["panels_next"]
            st.session_state["panels_next"] += 1
            _copy_panel_state(_pid, _new_pid)
            _panels.append({"id": _new_pid})
            st.session_state["panel_edit"] = _new_pid
            st.rerun()
        if len(_panels) > 1 and c_del.button("", icon=":material/delete:", help=t("削除"),
                                             key=f"del_panel_{_pid}"):
            _panels.remove(_p)
            st.session_state.pop(f"panel_cfg_{_pid}", None)
            if st.session_state.get("panel_edit") == _pid:
                st.session_state["panel_edit"] = _panels[0]["id"]
            st.rerun()
    if st.button(t("パネルを追加"), key="add_panel"):
        _new_pid = st.session_state["panels_next"]
        st.session_state["panels_next"] += 1
        _panels.append({"id": _new_pid})
        st.session_state["panel_edit"] = _new_pid
        st.rerun()
    _pids = [p["id"] for p in _panels]
    if st.session_state.get("panel_edit") not in _pids:
        st.session_state["panel_edit"] = _pids[0]
    _panel_labels = {
        p: t("パネル {n} ({mode})", n=i + 1,
             mode=tr_labels(_MODE_LABELS).get(
                 st.session_state.get(f"plot_mode_{p}"), t("(未設定)")))
        for i, p in enumerate(_pids)}
    # 編集するパネルの選択。ボタンを実際のパネル配置 (グリッド行優先) と同じ
    # 並びで置き、選択中のパネルを primary 色で示す
    st.markdown(t("**編集するパネル** (並びは図のパネル配置と同じ。赤 = 編集中)"))
    # 全パネル共通モード: ON の間は編集中パネルでの変更 (差分) を全パネルへ反映
    _all_mode = bool(st.session_state.setdefault("panel_edit_all", False))
    if st.button(t("🔗 全パネル共通"), key="_panel_sel_all",
                 type=("primary" if _all_mode else "secondary"),
                 help=t("選択中は、編集中パネルで変更した設定が全パネルに反映される "
                      "(タイトル・文字列・パネルラベルは個別のまま)。"
                      "レイヤーの追加・削除・種類変更は同じ描画モードのパネルに"
                      "反映され、引き継がれたレイヤーの個別設定 (変数選択など) は"
                      "保持される。もう一度押すか、パネルを選ぶと解除")):
        st.session_state["panel_edit_all"] = not _all_mode
        if _all_mode:
            st.session_state.pop("_common_ws_snapshot", None)
        st.rerun()
    # ボタンのセル位置は図のパネル配置と同じ解決 (parse_mosaic) から作る。
    # mosaic の空きセルにはボタンを置かず、結合セルは左上に置く
    _sel_cells = panel_selector_cells(grid_nrows, grid_ncols, len(_pids),
                                      grid_mosaic)
    for _r in range(grid_nrows):
        _btn_cols = st.columns(grid_ncols)
        for _c in range(grid_ncols):
            _i_panel = _sel_cells.get(_r * grid_ncols + _c)
            if _i_panel is None:
                continue
            _pid_btn = _pids[_i_panel]
            _is_sel = (st.session_state["panel_edit"] == _pid_btn
                       and not _all_mode)
            if _btn_cols[_c].button(
                    _panel_labels[_pid_btn],
                    key=f"_panel_sel_{_pid_btn}",
                    type=("primary" if _is_sel else "secondary")):
                st.session_state["panel_edit"] = _pid_btn
                st.session_state["panel_edit_all"] = False
                st.session_state.pop("_common_ws_snapshot", None)
                st.rerun()
    edit_pid = st.session_state["panel_edit"]
    if _all_mode:
        _master_label = _panel_labels.get(edit_pid, "")
        st.caption(t("🔗 全パネル共通編集中 — {label} を代表として表示し、"
                   "変更は全パネルに反映されます", label=_master_label))

# 図の再描画ボタン (メインエリア、「パネル構成」の直下)。処理は何もしないが、
# 押すと Streamlit がスクリプトを再実行し、ブラウザ側が保持する全 widget 値を
# 送り直した上で現在の設定から図を描き直す。操作が図に反映されなかったとき
# (再実行中の操作の取りこぼし、number_input / text_input の Enter 押し忘れ等)
# に、ブラウザの更新 (= セッションが作り直され設定が全部消える) の代わりに押す。
# key は "_" 始まりでセッション保存の対象外 (ボタンは programmatic 設定不可)
st.button(t("🔄 図を再描画"), key="_redraw_btn",
          help=t("現在の設定で図を描き直す。操作が図に反映されなかったときに押す"))

with st.sidebar:
    section_header(t("図全体の書式"))
    # Figure サイズ: figure (キャンバス) 自体の inch 単位の縦横サイズ (図全体で共通)
    with st.expander(t("Figure サイズ"), expanded=False):
        preset = st.radio(t("縦横比"), list(_ASPECT_PRESETS),
                          format_func=tr_labels(_ASPECT_PRESET_LABELS).get,
                          key="figsize_preset")
        if preset == "custom":
            c_fw, c_fh = st.columns(2)
            fig_w = float(c_fw.number_input(
                t("幅 (inch)"), value=6.4, min_value=1.0, max_value=30.0, step=0.1,
                format="%g", key="figsize_w"))
            fig_h = float(c_fh.number_input(
                t("高さ (inch)"), value=4.8, min_value=1.0, max_value=30.0, step=0.1,
                format="%g", key="figsize_h"))
        else:
            ratio = _ASPECT_PRESETS[preset]
            fig_w = _PRESET_DEFAULT_WIDTH
            fig_h = fig_w / ratio
            st.caption(f"{fig_w:g} × {fig_h:.3g} inch")
    # レイアウト調整 (仕様22.7): fig.subplots_adjust の引数
    # フォント (図全体で1つ。タイトル・軸・目盛り・カラーバー・注記の全テキストに効く)
    font_family = None
    with st.expander(t("フォント"), expanded=False):
        if st.checkbox(t("フォントを指定"), value=False, key="font_use",
                        help=t("off で matplotlib 既定 (DejaVu Sans)。"
                             "日本語を使うときは日本語フォント (Hiragino Sans 等) を"
                             "選ぶと文字化け (□) を防げる")):
            _font_mode = st.radio(
                t("選び方"), list(_FONT_MODE_LABELS),
                format_func=tr_labels(_FONT_MODE_LABELS).get,
                horizontal=True, key="font_mode")
            if _font_mode == "common":
                _commons = list(common_fonts_available())
                if _commons:
                    font_family = st.selectbox(
                        t("フォント"), _commons, key="font_common",
                        help=t("インストール済みの定番フォントのみ表示"))
                else:
                    st.warning(t("定番フォントが見つかりません。"
                               "「全フォントから検索」を使ってください"))
            else:
                font_family = st.selectbox(
                    t("フォント (入力で絞り込み)"), list(available_font_families()),
                    key="font_all",
                    help=t("matplotlib が検出した全フォント。ボックスに文字を"
                         "入力すると絞り込める"))
            st.caption(t("⚠ フォントは環境依存。再現スクリプトを別マシンで実行する"
                       "場合、同じフォントが無いと matplotlib 既定に戻る。"))
    # レイアウト調整 (仕様22.7): fig.subplots_adjust の引数
    layout_cfg = {"nrows": grid_nrows, "ncols": grid_ncols,
                  "mosaic": grid_mosaic,
                  "width_ratios": grid_wratios, "height_ratios": grid_hratios,
                  "hspace": None, "wspace": None,
                  "left": None, "right": None, "bottom": None, "top": None}
    with st.expander(t("レイアウト調整 (余白)"), expanded=False):
        if st.checkbox(t("パネル間の余白を指定"), value=False, key="layout_gap_use",
                        help=t("off で matplotlib 既定 (wspace=0.2, hspace=0.2)")):
            layout_cfg["wspace"] = float(st.slider(
                t("横方向 (wspace)"), -1.0, 1.0, 0.2, 0.01, key="layout_wspace",
                help=t("パネル幅に対する比率。負の値でセル同士を重ねて詰められる")))
            layout_cfg["hspace"] = float(st.slider(
                t("縦方向 (hspace)"), -1.0, 1.0, 0.2, 0.01, key="layout_hspace",
                help=t("パネル高さに対する比率。負の値でセル同士を重ねて詰められる")))
            st.caption(
                t("💡 水平断面図 (地図) は投影の縦横比が固定されるため、割り当て"
                "セルの上下に空白が残り、hspace=0 でも行間が開いて見えることが"
                "ある。その場合は **hspace を負の値** にして詰めるか、「Figure "
                "サイズ」の縦横比を地図に合わせる (例: 全球図の 2×2 は 横:縦 "
                "= 2:1 が目安)。負にしすぎるとタイトル・経緯度ラベルが重なる"
                "ので図を見ながら調整する。"))
        if st.checkbox(t("図の外側の余白を指定"), value=False, key="layout_margin_use",
                        help=t("off で matplotlib 既定 (left=0.125, right=0.9, "
                             "bottom=0.11, top=0.88)")):
            c1, c2 = st.columns(2)
            layout_cfg["left"] = float(c1.number_input(
                t("左 (figure 座標 0-1)"), 0.0, 0.45, 0.125, 0.005,
                format="%g", key="layout_left"))
            layout_cfg["right"] = float(c2.number_input(
                t("右 (0-1)"), 0.55, 1.0, 0.9, 0.005, format="%g", key="layout_right"))
            layout_cfg["bottom"] = float(c1.number_input(
                t("下 (0-1)"), 0.0, 0.45, 0.11, 0.005, format="%g", key="layout_bottom"))
            layout_cfg["top"] = float(c2.number_input(
                t("上 (0-1)"), 0.55, 1.0, 0.88, 0.005, format="%g", key="layout_top"))
    # 共通カラーバー (仕様22.5): 全パネルの fill で cmap・値域を揃えて使う。
    # 図全体の設定なので「図全体の書式」の一番下に置く
    scbar_cfg = {"show": False, "label": None, "location": "right",
                 "shrink": 1.0, "aspect": 30.0, "pad": None,
                 "label_fontsize": None, "tick_fontsize": None}
    with st.expander(t("共通カラーバー (全パネル)"), expanded=False):
        scbar_cfg["show"] = st.checkbox(
            t("全パネル共通のカラーバーを表示"), value=False, key="scbar_show",
            help=t("最初の fill レイヤーの mappable を代表に使う。全パネルの fill で "
                 "cmap・値域・レベル数を揃え、各レイヤーの個別カラーバーを OFF に"
                 "して使うこと"))
        if scbar_cfg["show"]:
            scbar_cfg["label"] = st.text_input(
                t("ラベル"), value="", placeholder=t("(なし)"), key="scbar_label") or None
            scbar_cfg["location"] = st.selectbox(
                t("位置"), ["right", "bottom", "left", "top"], key="scbar_loc")
            scbar_cfg["flip_ticks"] = st.checkbox(
                t("目盛りを反対側に表示"), value=False, key="scbar_flip",
                help=t("目盛り線・目盛り文字・ラベルをカラーバーの反対側に出す"))
            scbar_cfg["label_opposite"] = st.checkbox(
                t("ラベルを目盛りと反対側に表示"), value=False,
                key="scbar_flip_lab",
                help=t("カラーバーのラベルだけを目盛り文字と逆のサイドに出す (例: 右配置のカラーバーで目盛りは左・ラベルは右、またはその逆)"))
            scbar_cfg["shrink"] = float(st.slider(
                t("長さ (shrink)"), 0.3, 1.0, 1.0, 0.05, key="scbar_shrink"))
            scbar_cfg["aspect"] = float(st.slider(
                t("厚み (aspect)"), 5.0, 50.0, 30.0, 1.0, key="scbar_aspect",
                help=t("大きいほど細い")))
            if not st.checkbox(t("枠線の太さを自動"), value=True, key="scbar_olw_auto"):
                scbar_cfg["outline_width"] = float(st.slider(
                    t("枠線の太さ"), 0.0, 3.0, 0.8, 0.1, key="scbar_olw",
                    help=t("0 で枠線なし (matplotlib 既定 0.8)")))
            if not st.checkbox(t("目盛り線の太さを自動"), value=True, key="scbar_tw_auto"):
                scbar_cfg["tick_width"] = float(st.slider(
                    t("目盛り線の太さ"), 0.0, 3.0, 0.8, 0.1, key="scbar_tw"))
            if not st.checkbox(t("図との間隔を自動"), value=True, key="scbar_pad_auto"):
                # matplotlib のデフォルト: 縦配置 0.05 / 横配置 0.15
                _scbar_default_pad = (0.15 if scbar_cfg["location"] in ("bottom", "top")
                                      else 0.05)
                scbar_cfg["pad"] = float(st.slider(
                    t("図との間隔 (pad)"), 0.0, 0.5, _scbar_default_pad, 0.01,
                    key="scbar_pad"))
            scbar_cfg["label_fontsize"] = fontsize_input(
                t("ラベル文字サイズ"), "scbar_lfs", default=10, show_auto_hint=False)
            if not st.checkbox(t("ラベルの距離を自動"), value=True, key="scbar_lpad_auto"):
                scbar_cfg["label_pad"] = float(st.number_input(
                    t("ラベルの距離 (pt)"), -50.0, 50.0, 4.0, 1.0, key="scbar_lpad",
                    help=t("カラーバーとラベル文字の距離 (matplotlib 既定 ~4)")))
            scbar_cfg["tick_fontsize"] = fontsize_input(
                t("目盛り文字サイズ"), "scbar_tfs", default=10, show_auto_hint=False)
            if not st.checkbox(t("目盛り文字の距離を自動"), value=True,
                                key="scbar_tpad_auto"):
                scbar_cfg["tick_pad"] = float(st.number_input(
                    t("目盛り文字の距離 (pt)"), -50.0, 50.0, 3.5, 0.5, key="scbar_tpad",
                    help=t("カラーバーと目盛り文字の距離 (matplotlib 既定 3.5)")))


def _build_heatmap_panel(datasets, roles, mode_key):
    """categorical heatmap パネルを構築して返す。描けない場合は警告して None。

    2次元プロットモードのラジオで「categorical heatmap」を選んだときに呼ぶ。
    lat/lon 次元は軸に使えない (無次元の小行列用) ので候補から除く。
    """
    ids = list(datasets.keys())
    dsid = ids[0] if len(ids) == 1 else st.selectbox(
        t("データセット"), ids, key=f"hm_ds_{mode_key}")
    ds = datasets[dsid]
    # lat/lon をもつ地図データは対象外 (無次元の小行列用)
    ds_roles = mc_dataset.detect_coord_roles(ds)
    if ds_roles["lat"] and ds_roles["lon"]:
        st.warning(t("categorical heatmap は無次元の小行列用です。緯度経度データ "
                   "(lat/lon をもつデータ) には使えません — 水平面図の塗りつぶしを"
                   "使ってください。"))
        return None
    geographic = {ds_roles.get("lat"), ds_roles.get("lon")} - {None}
    # 2次元以上に落とせる変数だけを候補にする
    candidates = [str(n) for n, v in ds.data_vars.items()
                  if len([d for d in v.dims if d not in geographic]) >= 2]
    if not candidates:
        st.warning(t("2次元 (行列) にできる変数がありません。"))
        return None
    var = st.selectbox(t("変数"), candidates, key=f"hm_var_{mode_key}",
                       format_func=lambda v: var_label(ds, v))
    dims = [d for d in ds[var].dims if d not in geographic]
    c1, c2 = st.columns(2)
    x_dim = c1.selectbox(t("x 軸の次元"), dims, index=0,
                         key=f"hm_xdim_{mode_key}")
    y_opts = [d for d in dims if d != x_dim]
    y_dim = c2.selectbox(t("y 軸の次元"), y_opts, key=f"hm_ydim_{mode_key}")
    # x/y 以外の次元は固定して2次元に落とす
    extra = [d for d in ds[var].dims if d not in (x_dim, y_dim)]
    sel = {}
    for d in extra:
        if d in ds.coords:
            _vals = [x.item() for x in ds[d].values]
            sel[d] = st.selectbox(t("固定: {d}", d=d), _vals,
                                  key=f"hm_fix_{mode_key}_{d}")
        else:
            sel[d] = int(st.number_input(
                t("固定: {d} (index)", d=d), 0, int(ds.sizes[d]) - 1, 0,
                key=f"hm_fix_{mode_key}_{d}"))

    panel = mc_config.default_heatmap_panel()
    panel["dataset_id"] = dsid
    panel["variable"] = var
    panel["x_dim"], panel["y_dim"] = x_dim, y_dim
    panel["selection"] = sel
    s = panel["style"]
    s["cmap"] = cmap_selector(f"hm_{mode_key}")
    s["reverse_cmap"] = st.checkbox(t("カラーマップを反転"), key=f"hm_rev_{mode_key}")
    # 色レベル (fill レイヤーと同じ流儀: 等間隔 or 境界値の直接指定 + extend)
    if st.checkbox(t("レベルを等間隔にする"), value=True, key=f"hm_lveq_{mode_key}",
                   help=t("off にすると色の境界値をカンマ区切りで直接指定できる "
                        "(不等間隔可)")):
        if not st.checkbox(t("値の範囲を自動"), value=True,
                           key=f"hm_autorange_{mode_key}"):
            cc1, cc2 = st.columns(2)
            s["vmin"] = float(cc1.number_input(
                t("最小値"), value=float(ds[var].min()), format="%g",
                key=f"hm_vmin_{mode_key}_{var}"))
            s["vmax"] = float(cc2.number_input(
                t("最大値"), value=float(ds[var].max()), format="%g",
                key=f"hm_vmax_{mode_key}_{var}"))
        s["levels"] = int(st.number_input(t("カラーレベル数"), 3, 60, 21,
                                          key=f"hm_nlev_{mode_key}"))
    else:
        _lv = parse_float_list(st.text_input(
            t("レベル (カンマ区切り)"), value="0, 3, 4, 6, 10",
            key=f"hm_lvls_{mode_key}",
            help=t("色の境界値。昇順に整列・重複除去される。2個以上必要。"
                 "不等間隔も可。指定時は値の範囲 (最小値・最大値) は使われない")))
        if _lv and len(set(_lv)) >= 2:
            s["levels"] = _lv
        else:
            st.warning(t("レベルは2個以上の数値をカンマ区切りで入力してください"))
    s["extend"] = st.selectbox(t("extend (範囲外の扱い)"),
                               ["both", "neither", "min", "max"],
                               key=f"hm_ext_{mode_key}")
    s["zero_white"] = st.checkbox(
        t("ゼロ近傍を白にする"), key=f"hm_zerow_{mode_key}",
        help=t("0 を含む色レベル帯を白にする (0 がレベル境界のときは両隣の2帯)。"
             "正負の偏差を発散カラーマップで描くときに"))
    ceq = st.checkbox(t("正方形セル (aspect=equal)"), value=False,
                      key=f"hm_eq_{mode_key}")
    s["aspect"] = "equal" if ceq else "auto"
    # 行の向きは「軸・ラベル > 軸 > 縦軸を反転」で変える (origin は upper 固定。
    # 旧設定の origin=lower は render/scriptgen が従来どおり解釈する)。
    # x 目盛りラベルの回転は「軸・ラベル > 目盛」expander (axis_ticks_ui) に置く
    ann = s["annotate"]
    ann["show"] = st.checkbox(t("セルに数値を表示"), value=False,
                              key=f"hm_ann_{mode_key}")
    if ann["show"]:
        ann["fmt"] = st.text_input(t("数値書式"), value="%.2g",
                                   key=f"hm_annfmt_{mode_key}") or "%.2g"
        ann["fontsize"] = int(st.number_input(
            t("注記文字サイズ"), 4, 30, 8, key=f"hm_annfs_{mode_key}"))
        ann["color"] = color_selector(t("注記の文字色"), "#000000",
                                      key=f"hm_anncol_{mode_key}",
                                      meta_store=ann, meta_key="color")
    # 末尾: カラーバー / 値の変換 / マスクアウト (fill レイヤーと同じ並び)
    cb = s["colorbar"]
    with st.expander(t("カラーバー"), expanded=False):
        cb["show"] = st.checkbox(t("カラーバーを表示"), value=True,
                                 key=f"hmcb_{mode_key}")
        if cb["show"]:
            units = ds[var].attrs.get("units", "")
            default_label = f"{var} [{units}]" if units else str(var)
            cb["label"] = st.text_input(t("カラーバーラベル"), value=default_label,
                                        key=f"hmcbl_{mode_key}_{var}") or None
            cb["location"] = st.selectbox(
                t("位置"), ["right", "bottom", "left", "top"],
                key=f"hmcbloc_{mode_key}")
            cb["flip_ticks"] = st.checkbox(
                t("目盛りを反対側に表示"), value=False, key=f"hmcbflip_{mode_key}",
                help=t("目盛り線・目盛り文字・ラベルをカラーバーの反対側 "
                     "(プロット側) に出す"))
            cb["label_opposite"] = st.checkbox(
                t("ラベルを目盛りと反対側に表示"), value=False,
                key=f"hmcbflip_lab_{mode_key}",
                help=t("カラーバーのラベルだけを目盛り文字と逆のサイドに出す (例: 右配置のカラーバーで目盛りは左・ラベルは右、またはその逆)"))
            cb["shrink"] = float(st.slider(
                t("長さ (shrink)"), 0.3, 1.0, 1.0, 0.05,
                help=t("1.0 でプロット領域の長さと同じ"),
                key=f"hmcbshrink_{mode_key}"))
            cb["aspect"] = float(st.slider(
                t("厚み (aspect)"), 5.0, 50.0, 20.0, 1.0,
                help=t("大きいほど細い (matplotlib デフォルト 20)"),
                key=f"hmcbaspect_{mode_key}"))
            if not st.checkbox(t("枠線の太さを自動"), value=True,
                               key=f"hmcbolw_auto_{mode_key}"):
                cb["outline_width"] = float(st.slider(
                    t("枠線の太さ"), 0.0, 3.0, 0.8, 0.1, key=f"hmcbolw_{mode_key}",
                    help=t("0 で枠線なし (matplotlib 既定 0.8)")))
            if not st.checkbox(t("目盛り線の太さを自動"), value=True,
                               key=f"hmcbtw_auto_{mode_key}"):
                cb["tick_width"] = float(st.slider(
                    t("目盛り線の太さ"), 0.0, 3.0, 0.8, 0.1,
                    key=f"hmcbtw_{mode_key}",
                    help=t("カラーバーの目盛り線 (tick) の太さ。0 で非表示 "
                         "(matplotlib 既定 0.8)")))
            if not st.checkbox(t("図との間隔を自動"), value=True,
                               key=f"hmcbpad_auto_{mode_key}"):
                # matplotlib のデフォルト: 縦配置 0.05 / 横配置 0.15
                default_pad = (0.15 if cb["location"] in ("bottom", "top")
                               else 0.05)
                cb["pad"] = float(st.slider(
                    t("図との間隔 (pad)"), 0.0, 0.5, default_pad, 0.01,
                    key=f"hmcbpad_{mode_key}"))
            cb["label_fontsize"] = fontsize_input(
                t("ラベル文字サイズ"), f"hmcblfs_{mode_key}",
                default=10, show_auto_hint=False)
            if not st.checkbox(t("ラベルの距離を自動"), value=True,
                               key=f"hmcblpad_auto_{mode_key}"):
                cb["label_pad"] = float(st.number_input(
                    t("ラベルの距離 (pt)"), -50.0, 50.0, 4.0, 1.0,
                    key=f"hmcblpad_{mode_key}",
                    help=t("カラーバーとラベル文字の距離 (matplotlib 既定 ~4)。"
                         "負の値で近づく")))
            cb["tick_fontsize"] = fontsize_input(
                t("目盛り文字サイズ"), f"hmcbtfs_{mode_key}",
                default=10, show_auto_hint=False)
            if not st.checkbox(t("目盛り文字の距離を自動"), value=True,
                               key=f"hmcbtpad_auto_{mode_key}"):
                cb["tick_pad"] = float(st.number_input(
                    t("目盛り文字の距離 (pt)"), -50.0, 50.0, 3.5, 0.5,
                    key=f"hmcbtpad_{mode_key}",
                    help=t("カラーバーと目盛り文字の距離 (matplotlib 既定 3.5)。"
                         "負の値で近づく")))
    value_transform_ui(s, f"hm_{mode_key}")
    maskout_ui(s, f"hm_{mode_key}")
    return panel


def _plot_size_ui(mode: str, mode_key: str):
    """「プロットサイズ」expander: axes 枠 (プロット領域) の縦横比 (set_box_aspect)。
    地図では操作不可 (None)。"""
    with st.sidebar:
        # プロットサイズ: axes 枠 (プロット領域) の縦横比を set_box_aspect で固定
        with st.expander(t("プロットサイズ"), expanded=False):
            is_map_mode = mode == "map"
            is_scatter_mode = mode in ("scatter", "agg")
            if is_map_mode:
                st.caption(
                    t("水平断面図ではマップの形は投影法と extent で固定されます "
                    "(cartopy の GeoAxes が aspect=equal を効かせる)。"
                    "set_box_aspect は外側枠と colorbar 位置を動かすだけで"
                    "マップ自体の形は変わらないため、ここでの操作はできません。"))
                box_aspect = None
            else:
                # 先頭が既定 (2次元プロット系は 1:1、他は 6.4:4.8)。"none" は固定しない
                options = (["square", "none", "custom"] if is_scatter_mode
                            else ["default", "none", "custom"])
                plot_preset = st.radio(
                    t("axes 枠の縦横比"), options,
                    format_func=tr_labels(_BOXASPECT_PRESET_LABELS).get,
                    key=f"boxaspect_preset_{mode_key}",
                    help=t("set_box_aspect() で axes 枠 (プロット領域) の高さ/幅を固定する。"
                         "「固定しない」にすると Figure サイズとレイアウト調整の余白に従って伸縮する。"
                         "タイトル・カラーバー等は枠の外に追加で乗るので、"
                         "保存 PNG 全体の縦横比とは一致しない"))
                if plot_preset == "custom":
                    box_aspect = float(st.number_input(
                        t("高さ / 幅"), value=1.0, min_value=0.05, max_value=20.0,
                        step=0.05, format="%g", key=f"boxaspect_val_{mode_key}",
                        help=t("例: 0.5 → 横長 / 2.0 → 縦長")))
                elif plot_preset == "square":
                    box_aspect = 1.0
                elif plot_preset == "none":
                    box_aspect = None
                else:  # "default" (6.4:4.8)
                    box_aspect = 4.8 / 6.4   # = 0.75
    return box_aspect


# 緯度経度範囲を指定したとき、中心経度を範囲の中央に追従させる選択肢 (既定 ON) を出す投影法。
# PlateCarree は常に追従 (入力なし)、極投影・Lambert・Orthographic は常に手動
_CLON_FOLLOW_PROJECTIONS = ("Robinson", "EqualEarth")


def _map_projection_ui(ds, roles, mode_key):
    """「投影法・領域」セクション。(projection, region, proj_name, is_polar) を返す。"""
    section_header(t("投影法・領域"))
    # 2 次元座標格子で投影法を推定できたら、それを初期の投影法にする (widget の状態が
    # 無いときだけ効く index。パラメータの初期値も下で合わせる)
    _gp = _grid_projection if _grid_projection else None
    _proj_names = list(PROJECTION_LABELS)
    _proj_index = (_proj_names.index(_gp["name"])
                   if _gp and _gp["name"] in _proj_names else 0)
    proj_name = st.selectbox(t("投影法"), _proj_names,
                             index=_proj_index,
                             format_func=tr_labels(PROJECTION_LABELS).get,
                             key=f"proj_name_{mode_key}")
    if _gp and _gp["name"] == "LambertConformal":
        st.caption(t("格子の投影法を自動認識: ランベルト正角円錐 (標準緯線 {sp1}° / {sp2}°、"
                     "中心経度 {clon}°)。この値を初期値にしています",
                     sp1=f"{_gp['standard_parallels'][0]:g}",
                     sp2=f"{_gp['standard_parallels'][1]:g}",
                     clon=f"{_gp['central_longitude']:g}"))
    elif _gp:
        st.caption(t("格子の投影法を自動認識: 極ステレオ (中心経度 {clon}°)。"
                     "この値を初期値にしています", clon=f"{_gp['central_longitude']:g}"))
    region = None
    # 極投影では範囲指定を常に有効にする (チェックボックスは出さない)
    is_polar = proj_name in ("NorthPolarStereo", "SouthPolarStereo")
    # データの経緯度範囲 (座標ファイルで後から付けた 2 次元座標も含む)。領域データ
    # (全球でない) では、どの投影法でもこの範囲を初期値にし、範囲指定を既定で ON に
    # する (2026-09-24 ユーザー要望: 初期範囲は全球ではなくデータの範囲。curvilinear
    # では全格子点を含む経緯度の矩形)。全球データは従来どおり (全球表示。極投影は
    # 半球、Lambert は固定の中緯度領域)。保存済みセッションはチェック状態を持つので
    # 影響しない
    _bounds = (mc_dataset.lonlat_bounds(ds, roles)
               if roles["lat"] and roles["lon"] else None)
    data_regional = _bounds is not None and not _bounds["global"]
    # 2 次元座標格子は領域未指定でも render が「全格子点の投影座標の範囲」を表示範囲に
    # する (余白なしの長方形) ので、範囲指定の既定は OFF (ユーザー提案 2026-09-24)。
    # チェックすれば緯度経度で絞れる (初期値はデータの範囲)
    _curvi = mc_dataset.is_curvilinear(ds, roles)
    region_default = ((proj_name == "LambertConformal" or data_regional)
                      and not _curvi)
    if _curvi and not is_polar and proj_name != "Orthographic":
        st.caption(t("2 次元座標格子: 範囲未指定では全格子点の投影座標の範囲 "
                     "(余白なしの長方形) を表示します。緯度経度で絞るには"
                     "「緯度経度範囲を指定」をチェックしてください"))
    if is_polar or st.checkbox(t("緯度経度範囲を指定"), value=region_default,
                                key=f"reg_check_{mode_key}_{proj_name}",
                                help=t("**表示範囲**はこの指定どおり。データの"
                                     "**切り出し**だけは境界で塗りが切れない"
                                     "よう両側に1格子分広めに行う (色の自動"
                                     "範囲はこの広めの切り出しに基づく)。"
                                     "領域**平均** (レイヤーの範囲平均) は"
                                     "指定範囲ちょうどで計算され、この"
                                     "パディングの影響を受けない")):
        if proj_name == "Orthographic":
            st.warning(t("Orthographic図法では表示範囲の制限は適用されません (データの切り出しのみ)。"))
        lons = ds[roles["lon"]].values
        lats = ds[roles["lat"]].values
        # 極投影では対応する半球・全経度を初期値にする (投影を変えるとリセット)
        if data_regional:
            # 領域データ: lonlat_bounds の範囲 (経度は全点を含む最小の弧。日付変更線を
            # またぐ格子でも min/max のように全周に化けない。経度が全周なら規約の
            # 全周 0–360 / -180–180)
            lat_min_default, lat_max_default = _bounds["lat_min"], _bounds["lat_max"]
            lon_min_default, lon_max_default = _bounds["lon_min"], _bounds["lon_max"]
        else:
            lat_min_default, lat_max_default = float(lats.min()), float(lats.max())
            lon_min_default, lon_max_default = float(lons.min()), float(lons.max())
        # 単一観測点などデータが1点しかないと min == max の退化領域に
        # なる (set_extent が特異)。±5° の窓を初期値にする
        if lat_min_default == lat_max_default:
            lat_min_default = max(lat_min_default - 5.0, -90.0)
            lat_max_default = min(lat_max_default + 5.0, 90.0)
        if lon_min_default == lon_max_default:
            lon_min_default -= 5.0
            lon_max_default += 5.0
        if data_regional:
            pass  # 上で初期値を決めた (投影法ごとの固定値は全球データのときだけ)
        elif proj_name == "NorthPolarStereo":
            lat_min_default, lat_max_default = 0.0, 90.0
            lon_min_default, lon_max_default = 0.0, 360.0
        elif proj_name == "SouthPolarStereo":
            lat_min_default, lat_max_default = -90.0, 0.0
            lon_min_default, lon_max_default = 0.0, 360.0
        elif proj_name == "LambertConformal":
            lat_min_default, lat_max_default = 0.0, 65.0
            lon_min_default, lon_max_default = 100.0, 180.0
        # TAB の移動順は DOM 順 = columns のカラム毎 (縦方向が先) に
        # なるため、経度の行と緯度の行で columns を分けて
        # 経度最小→経度最大→緯度最小→緯度最大 の順で移動させる
        c1, c2 = st.columns(2)
        lon_min = c1.number_input(t("経度 最小"), value=lon_min_default,
                                  key=f"reg_lonmin_{mode_key}_{proj_name}")
        lon_max = c2.number_input(t("経度 最大"), value=lon_max_default,
                                  key=f"reg_lonmax_{mode_key}_{proj_name}")
        c1, c2 = st.columns(2)
        lat_min = c1.number_input(t("緯度 最小"), value=lat_min_default,
                                  key=f"reg_latmin_{mode_key}_{proj_name}")
        lat_max = c2.number_input(t("緯度 最大"), value=lat_max_default,
                                  key=f"reg_latmax_{mode_key}_{proj_name}")
        region = {"lon_min": float(lon_min), "lon_max": float(lon_max),
                  "lat_min": float(lat_min), "lat_max": float(lat_max)}

    # 中心経度のデフォルトは領域の中央 (領域未指定時は Lambert=140、その他=180)。
    # キーは reg_* と同じ投影法別 — 投影法ごとに適切な既定値が違うため、
    # 切り替えると既定値が入り直る。同一投影法内では領域を変えても自動追従
    # しない (key を default 値で変える方式はセッション/プリセット復元が効かない
    # ため不採用)
    if _gp and _gp["name"] == proj_name:
        # 推定した格子の投影: 中心経度は格子の中心経線 (範囲の中央にすると格子が
        # 図の軸に対して傾く)
        default_clon = float(_gp["central_longitude"])
    elif region:
        default_clon = round((region["lon_min"] + region["lon_max"]) / 2, 1)
    else:
        default_clon = 140.0 if proj_name == "LambertConformal" else 180.0
    # PlateCarree + 範囲指定では中心経度の入力を出さず「範囲の中央」を
    # 自動使用する (表示のみ)。
    # - 全経度未満: 中心経度は見た目に影響しない (投影が経度方向に線形で、
    #   表示範囲は region が決める)。wrap (範囲が中心±180° に収まる制約) が
    #   常に成立する値として範囲の中央を使う
    # - 全経度幅 (>=360°): extent は全球扱い (extent_args) になり、図の
    #   左右端 (継ぎ目) は中心経度だけが決める。範囲の中央を使えば
    #   左右端 = lon_min / lon_max になり指定どおりの見た目になる
    #   (v0.95 までは手動のままで「範囲指定が効かない」ように見えた)
    # lon_min > lon_max (0°またぎ指定) は中央の式が半周ずれるため対象外
    # (従来どおり手動)。他投影は region があっても中心経度で見た目が変わる
    # (極投影の回転・Lambert の円錐軸・Robinson/EqualEarth の湾曲・
    # Orthographic の可視半球) ので表示する。ただし Robinson / EqualEarth は
    # 「中心経度を範囲の中央に合わせる」(既定 ON) で範囲の中央に追従させ、外したときだけ
    # 入力を出す (ユーザー要望 2026-10-02。範囲を動かすたびに中心経度を打ち直す手間を
    # 省く。極投影・Lambert・Orthographic は回転・円錐軸・可視半球の意図的な指定が
    # 普通なので従来どおり)。旧セッションには追従キーが無いので、復元時に
    # state_io._migrate_loaded_session が OFF を補って図を変えない
    follow_region = (proj_name in _CLON_FOLLOW_PROJECTIONS and region
                     and region["lon_min"] <= region["lon_max"]
                     and st.checkbox(t("中心経度を範囲の中央に合わせる"), value=True,
                                     key=f"clon_follow_{mode_key}_{proj_name}",
                                     help=t("経度範囲を変えると中心経度も範囲の中央に"
                                          "追従します。外すと中心経度を数値で指定"
                                          "できます (範囲の中央から離すと図が湾曲"
                                          "します)")))
    if proj_name in _CLON_FOLLOW_PROJECTIONS:
        # 追従を外した直後は、入力欄を今の範囲の中央から始める (追従前に入力欄が持っていた
        # 古い値を出さない)。"_" 始まりのキーはセッションに保存されないので、旧セッションの
        # 復元 (追従 OFF + 保存した中心経度) ではここを通らず保存値のまま
        _flag = f"_clon_following_{mode_key}_{proj_name}"
        if follow_region:
            st.session_state[_flag] = True
        elif st.session_state.get(_flag):
            st.session_state[f"central_lon_{mode_key}_{proj_name}"] = default_clon
            st.session_state[_flag] = False
    if (proj_name == "PlateCarree" and region
            and region["lon_min"] <= region["lon_max"]):
        central_lon = default_clon
        if mc_render.region_spans_all_longitudes(region):
            st.caption(t("経度幅が 360° (全経度) のため図の左右端は"
                         "中心経度で決まります。範囲の中央 ({clon:g}°) を"
                         "自動使用 (左右端 = 指定した経度範囲の端)",
                         clon=default_clon))
        else:
            st.caption(t("中心経度は範囲の中央 ({clon:g}°) を自動使用",
                         clon=default_clon))
    elif follow_region:
        central_lon = default_clon
        st.caption(t("中心経度は範囲の中央 ({clon:g}°) を自動使用",
                     clon=default_clon))
    else:
        central_lon = float(st.number_input(t("中心経度"), -180.0, 360.0, default_clon, 10.0,
                                            key=f"central_lon_{mode_key}_{proj_name}"))
    central_lat = 0.0
    if proj_name == "Orthographic":
        central_lat = float(st.number_input(t("中心緯度"), -90.0, 90.0, 20.0, 10.0,
                                             key=f"central_lat_{mode_key}_{proj_name}"))
    elif proj_name == "LambertConformal":
        _lcc = _gp if (_gp and _gp["name"] == "LambertConformal") else None
        central_lat = float(st.number_input(
            t("中心緯度"), -90.0, 90.0,
            float(_lcc["central_latitude"]) if _lcc else 35.0, 1.0,
            key=f"central_lat_{mode_key}_{proj_name}",
            help=t("円錐の軸に対応する基準緯度")))
    projection = {"name": proj_name,
                  "central_longitude": central_lon, "central_latitude": central_lat}
    if proj_name == "LambertConformal":
        _sp_default = (_lcc["standard_parallels"] if _lcc else [33.0, 45.0])
        cc1, cc2 = st.columns(2)
        sp1 = cc1.number_input(t("標準緯度1 (standard_parallels)"), -90.0, 90.0,
                                float(_sp_default[0]), 1.0,
                                key=f"sp1_lc_{mode_key}",
                                help=t("cartopy 推奨は中緯度の2本 (例 33° と 45°)。"
                                     "標準緯度間でひずみが小さい"))
        sp2 = cc2.number_input(t("標準緯度2"), -90.0, 90.0, float(_sp_default[1]), 1.0,
                                key=f"sp2_lc_{mode_key}")
        projection["standard_parallels"] = [float(sp1), float(sp2)]
    if proj_name in ("NorthPolarStereo", "SouthPolarStereo"):
        projection["circular_boundary"] = st.checkbox(t("枠を円形にする"), value=True,
                                                      key=f"circular_boundary_{mode_key}")
    if region and abs(((region["lon_min"] + region["lon_max"]) / 2 - central_lon + 180) % 360 - 180) > 90:
        st.warning(t("中心経度が指定範囲から離れています。表示範囲が正しく適用されない場合は"
                   "中心経度を範囲の中央付近にしてください。"))
    return projection, region, proj_name, is_polar


def _map_gridlines_ui(gl, mode_key, is_polar, projection):
    """「緯度経度線の設定」expander。map_cfg["gridlines"] (gl) を直接更新する。"""
    with st.expander(t("緯度経度線の設定"), expanded=False):
        gl["show"] = st.checkbox(t("緯度経度線・ラベルを表示"), value=True,
                                 key=f"gl_show_{mode_key}")
        if gl["show"]:
            # --- 線 (ラベルと独立に ON/OFF・間隔指定できる) ---
            gl["lines"] = st.checkbox(t("線を表示"), value=True,
                                      key=f"gl_lines_{mode_key}")
            if gl["lines"]:
                gl["width"] = float(st.slider(
                    t("線の太さ"), 0.1, 2.0, 0.5, 0.1, key=f"gl_width_{mode_key}"))
                gl["color"] = color_selector(t("線の色"), "#808080",
                                              key=f"gl_color_{mode_key}",
                                              meta_store=gl, meta_key="color")
                gl["linestyle"] = st.selectbox(
                    t("線種"), list(GL_LINESTYLE_LABELS),
                    format_func=tr_labels(GL_LINESTYLE_LABELS).get,
                    key=f"gl_linestyle_{mode_key}")
                lon_int = st.number_input(
                    t("経度線の間隔 (0=自動)"), 0.0, 180.0, 0.0, 5.0,
                    key=f"gl_lon_interval_{mode_key}")
                lat_int = st.number_input(
                    t("緯度線の間隔 (0=自動)"), 0.0, 90.0, 0.0, 5.0,
                    key=f"gl_lat_interval_{mode_key}")
                gl["lon_interval"] = float(lon_int) or None
                gl["lat_interval"] = float(lat_int) or None
            # --- ラベル (線と独立に ON/OFF・間隔指定できる) ---
            gl["labels"] = st.checkbox(t("ラベルを表示"), value=True,
                                       key=f"gl_labels_{mode_key}")
            if gl["labels"]:
                if is_polar:
                    # 極投影では緯度ラベルは図中 (inline)、経度ラベルは
                    # 境界沿い (geo) に描かれ、「左」「右」の辺に分類される
                    # ラベルは存在しない (チェックを出さない)。上・下は
                    # 境界が図枠の上下に接する位置の経度ラベルに効く。
                    # 並びは「経度ラベルの設定 (辺・円周沿い・極) →
                    # 緯度ラベルの設定」
                    st.caption(t("経度ラベルを表示する辺"))
                    cc1, cc2 = st.columns(2)
                    gl["label_sides"] = {
                        "left": True, "right": True,
                        "top": cc1.checkbox(t("上"), value=False, key=f"gl_side_top_{mode_key}"),
                        "bottom": cc2.checkbox(t("下"), value=True, key=f"gl_side_bottom_{mode_key}"),
                    }
                    if projection.get("circular_boundary", True):
                        gl["label_sides"]["geo"] = st.checkbox(
                            t("円周沿いの経度ラベル"), value=True,
                            key=f"gl_side_geo_{mode_key}",
                            help=t("円が図枠の上下に接する位置のラベルは"
                                 "「上」「下」チェックで、それ以外の円周"
                                 "沿いのラベルはこのチェックで制御する"))
                        gl["pole_label"] = st.checkbox(
                            t("極に置かれる経度ラベル"), value=True,
                            key=f"gl_side_pole_{mode_key}",
                            help=t("経度範囲を指定した扇形では経度線が"
                                 "極 (扇の要) で終わり、そこにも経度"
                                 "ラベルが置かれる。off でその極の"
                                 "ラベルだけを消す (弧沿いのラベルは"
                                 "残る)。全経度の円形では極にラベルは"
                                 "置かれないため効果なし"))
                    _lat_lab = st.radio(
                        t("緯度ラベル"), list(_LAT_LABEL_LABELS),
                        format_func=tr_labels(_LAT_LABEL_LABELS).get,
                        horizontal=True, key=f"gl_lat_lab_{mode_key}",
                        help=t("図中 = 放射方向に並べて図の中に描く (cartopy "
                             "既定)。枠沿い = 枠線 (扇の縁) との交点に描く — "
                             "経度範囲を指定した扇形で有効。全経度の円形では"
                             "緯度線が枠と交差しないため表示されない。"
                             "枠沿いのラベルは「円周沿いの経度ラベル」オフで"
                             "一緒に消える"))
                    gl["lat_label_placement"] = (
                        "edge" if _lat_lab == "edge" else "inline")
                    gl["label_sides"]["inline"] = _lat_lab != "none"
                    if _lat_lab == "edge":
                        gl["lat_label_edge_side"] = st.radio(
                            t("表示する縁"), list(_LAT_EDGE_SIDE_LABELS),
                            format_func=tr_labels(_LAT_EDGE_SIDE_LABELS).get,
                            horizontal=True,
                            key=f"gl_lat_edge_side_{mode_key}",
                            help=t("扇の左の縁 / 右の縁だけにラベルを"
                                 "表示する。ラベルがどちらの縁 (経度"
                                 "範囲の端) に属するかで判定するので、"
                                 "扇が傾いていても縁単位で選べる"))
                else:
                    st.caption(t("ラベルを表示する辺"))
                    cc1, cc2 = st.columns(2)
                    gl["label_sides"] = {
                        "left": cc1.checkbox(t("左"), value=True, key=f"gl_side_left_{mode_key}"),
                        "right": cc2.checkbox(t("右"), value=False, key=f"gl_side_right_{mode_key}"),
                        "top": cc1.checkbox(t("上"), value=False, key=f"gl_side_top_{mode_key}"),
                        "bottom": cc2.checkbox(t("下"), value=True, key=f"gl_side_bottom_{mode_key}"),
                    }
                lab_lon = st.number_input(
                    t("ラベルの経度間隔 (0=線と同じ)"), 0.0, 180.0, 0.0, 5.0,
                    key=f"gl_lab_lon_interval_{mode_key}",
                    help=t("線の間隔と異なる値も指定できる (例: 線は10°毎、"
                         "ラベルは30°毎)。線が非表示のときの 0 は自動"))
                lab_lat = st.number_input(
                    t("ラベルの緯度間隔 (0=線と同じ)"), 0.0, 90.0, 0.0, 5.0,
                    key=f"gl_lab_lat_interval_{mode_key}")
                gl["label_lon_interval"] = float(lab_lon) or None
                gl["label_lat_interval"] = float(lab_lat) or None
                lab_lon0 = st.number_input(
                    t("ラベルの開始経度 (0=間隔の倍数)"), -180.0, 360.0, 0.0, 5.0,
                    key=f"gl_lab_lon_start_{mode_key}",
                    help=t("この経度を起点に間隔ごとにラベルを置く"
                         " (起点より小さい側へも延長する)。"
                         "例: 開始15°・間隔30° → …15°, 45°, 75°…。"
                         "ラベルの間隔 (0 のときは線の間隔) が自動の"
                         "ときは効かない"))
                lab_lat0 = st.number_input(
                    t("ラベルの開始緯度 (0=間隔の倍数)"), -90.0, 90.0, 0.0, 5.0,
                    key=f"gl_lab_lat_start_{mode_key}")
                gl["label_lon_start"] = float(lab_lon0) or None
                gl["label_lat_start"] = float(lab_lat0) or None
                st.caption(
                    t("⚠ ラベル同士が重なる場合、cartopy が重なった分を"
                    "自動的に間引く (例: 間隔20°で1つおきの表示になる)。"
                    "回転・文字サイズ縮小・図の幅拡大で全て表示できる。"))
                gl["label_rotation"] = float(st.number_input(
                    t("経度ラベルの回転 (度)"), -90.0, 90.0, 0.0, 5.0,
                    key=f"gl_lab_rot_{mode_key}",
                    help=t("細かい間隔でラベルが間引かれるときは 45° 前後を"
                         "指定すると全て表示される"))) or None
                if st.checkbox(
                        t("緯度ラベルの回転を指定"), value=False,
                        key=f"gl_lat_rot_on_{mode_key}",
                        help=t("off = 自動 (極投影では縁・経度線の向きに"
                             "沿って回転する)。on で角度を固定 — "
                             "0° にすると水平になる")):
                    gl["lat_label_rotation"] = float(st.number_input(
                        t("緯度ラベルの回転 (度)"), -90.0, 90.0, 0.0, 5.0,
                        key=f"gl_lat_rot_{mode_key}"))
                gl["label_fontsize"] = fontsize_input(
                    t("ラベル文字サイズ"), f"gl_fs_{mode_key}",
                    default=10, show_auto_hint=False)
                gl["label_padding"] = int(st.number_input(
                    t("ラベルの距離 (pt)"), -20, 50, 5, 1,
                    help=t("枠線からのラベルまでの距離 (matplotlib 既定 5)"),
                    key=f"gl_pad_{mode_key}"))


def _map_ticks_ui(ticks, mode_key, proj_name):
    """「ティックマークの設定」expander。map_cfg["ticks"] を直接更新する。"""
    with st.expander(t("ティックマークの設定"), expanded=False):
        if proj_name != "PlateCarree":
            st.caption(t("⚠ PlateCarree (正距円筒図法) 以外では描画されません"))
        ticks["show"] = st.checkbox(t("ティックマークを表示"), value=False, key=f"ticks_show_{mode_key}")
        if ticks["show"]:
            ticks["lon_interval"] = float(st.number_input(
                t("経度の間隔"), 1.0, 90.0, 30.0, 1.0, key=f"tick_lon_{mode_key}"))
            ticks["lat_interval"] = float(st.number_input(
                t("緯度の間隔"), 1.0, 45.0, 15.0, 1.0, key=f"tick_lat_{mode_key}"))
            ticks["length"] = float(st.slider(
                t("長さ"), 1.0, 12.0, 4.0, 0.5, key=f"tick_len_{mode_key}"))
            ticks["width"] = float(st.slider(
                t("太さ"), 0.2, 3.0, 0.8, 0.1, key=f"tick_w_{mode_key}"))
            ticks["direction"] = st.radio(
                t("向き"), list(_TICK_DIR_LABELS),
                format_func=tr_labels(_TICK_DIR_LABELS).get, horizontal=True,
                key=f"tick_dir_{mode_key}")
            st.caption(t("ティックを表示する辺"))
            tc1, tc2 = st.columns(2)
            ticks["sides"] = {
                "left": tc1.checkbox(t("左"), value=True, key=f"tick_side_left_{mode_key}"),
                "right": tc2.checkbox(t("右"), value=False, key=f"tick_side_right_{mode_key}"),
                "top": tc1.checkbox(t("上"), value=False, key=f"tick_side_top_{mode_key}"),
                "bottom": tc2.checkbox(t("下"), value=True, key=f"tick_side_bottom_{mode_key}"),
            }
            # 短いティック (補助目盛): 長い線の間に細かい間隔で入れる
            minor = ticks["minor"]
            minor["show"] = st.checkbox(
                t("短いティックを追加 (補助目盛)"), value=False,
                key=f"tick_minor_{mode_key}",
                help=t("上の間隔の線 (長) の間に、細かい間隔の短い線を"
                     "入れる (例: 長10°毎 + 短5°毎)。向きと表示辺は"
                     "長い線と共通"))
            if minor["show"]:
                minor["lon_interval"] = float(st.number_input(
                    t("経度の間隔 (短)"), 0.5, 90.0, 10.0, 0.5,
                    key=f"tick_minor_lon_{mode_key}"))
                minor["lat_interval"] = float(st.number_input(
                    t("緯度の間隔 (短)"), 0.5, 45.0, 5.0, 0.5,
                    key=f"tick_minor_lat_{mode_key}"))
                minor["length"] = float(st.slider(
                    t("長さ (短)"), 0.5, 8.0, 2.5, 0.5,
                    key=f"tick_minor_len_{mode_key}"))
                minor["width"] = float(st.slider(
                    t("太さ (短)"), 0.2, 3.0, 0.8, 0.1,
                    key=f"tick_minor_w_{mode_key}"))


def _map_settings_ui(mode_key, proj_name, is_polar, projection):
    """「地図・グリッド線」セクション (地図・緯度経度線・ティック・box)。map_cfg を返す。"""
    section_header(t("地図・グリッド線"))
    map_cfg = mc_config.default_map_settings()
    with st.expander(t("地図の設定"), expanded=False):
        coast = map_cfg["coastlines"]
        coast["show"] = st.checkbox(t("海岸線を表示"), value=True, key=f"coast_show_{mode_key}")
        if coast["show"]:
            coast["width"] = float(st.slider(t("海岸線の太さ"), 0.2, 3.0, 0.8, 0.1,
                                              key=f"coast_width_{mode_key}"))
            coast["color"] = color_selector(t("海岸線の色"), "#000000", key=f"coast_color_{mode_key}",
                                              meta_store=coast, meta_key="color")
            map_cfg["borders"] = st.checkbox(t("国境線を表示"), value=False, key=f"borders_show_{mode_key}")
        else:
            map_cfg["borders"] = False
        map_cfg["land"]["show"] = st.checkbox(t("陸域を塗りつぶす"), value=False, key=f"land_show_{mode_key}")
        if map_cfg["land"]["show"]:
            map_cfg["land"]["color"] = color_selector(t("陸域の色"), "#d9d2c2", key=f"land_color_{mode_key}",
                                                      meta_store=map_cfg["land"], meta_key="color")
            map_cfg["land"]["above_data"] = st.checkbox(
                t("データの上に描く"), value=False, key=f"land_fg_{mode_key}",
                help=t("陸の上にある塗り・ハッチ・ベクトルを陸で隠す。等値線・流線・"
                       "トラック・海岸線・国境線・グリッド線・box・注記は陸の上に出る。"
                       "散布点・トラックの点・基準ベクトルも陸の上に出す"))
        map_cfg["ocean"]["show"] = st.checkbox(t("海域を塗りつぶす"), value=False, key=f"ocean_show_{mode_key}")
        if map_cfg["ocean"]["show"]:
            map_cfg["ocean"]["color"] = color_selector(t("海域の色"), "#cfe2f3", key=f"ocean_color_{mode_key}",
                                                       meta_store=map_cfg["ocean"], meta_key="color")
        map_cfg["resolution"] = st.selectbox(
            t("地理データの解像度"), list(NE_RESOLUTION_LABELS),
            format_func=tr_labels(NE_RESOLUTION_LABELS).get, key=f"ne_res_{mode_key}",
            help=t("海岸線・国境線・陸域・海域に共通の Natural Earth の解像度。"
                   "自動 = 表示範囲の短辺が 50° 以下で 1:50m、15° 以下で 1:10m、"
                   "それ以外は 1:110m。1:50m / 1:10m のデータは初回に自動で"
                   "ダウンロードされる (ネット接続が必要)"))
        if st.checkbox(t("図の枠線の太さを指定"), value=False,
                        key=f"frame_w_use_{mode_key}",
                        help=t("off で matplotlib 既定。極投影の円形枠にも適用される")):
            map_cfg["frame_width"] = float(st.slider(
                t("枠線の太さ"), 0.2, 5.0, 1.0, 0.1, key=f"frame_w_{mode_key}"))
    _map_gridlines_ui(map_cfg["gridlines"], mode_key, is_polar, projection)
    _map_ticks_ui(map_cfg["ticks"], mode_key, proj_name)
    with st.expander(t("boxの設定 (矩形領域の線)"), expanded=False):
        map_cfg["boxes"] = boxes_ui(mode_key)
    return map_cfg


def _map_mode_ui(datasets, ds, roles, mode_key):
    """水平断面図 (map) モードのパネル設定 UI。"""
    variables = mc_dataset.horizontal_map_variables(ds, roles)
    variables_by_ds = {dsid: mc_dataset.horizontal_map_variables(d)
                        for dsid, d in datasets.items()}
    # 格子変数が無くても、地点データ (lon/lat 補助座標) の散布図
    # (map_scatter) やトラック (track) を描けるならモードを開く。
    if (not variables and not mc_dataset.map_scatter_variables(ds, roles)
            and not mc_dataset.track_capable(ds)):
        st.error(t("緯度・経度をもつ変数が見つかりません。"))
        st.stop()

    with st.sidebar:
        projection, region, proj_name, is_polar = _map_projection_ui(ds, roles, mode_key)

        # 鉛直レベル等の非時刻次元はレイヤー毎に固定 (下の「レイヤー」セクションへ)。
        # ここでは時刻だけを panel.selection に置く。
        # 次元集めは格子変数だけでなく地点散布 (map_scatter) 対象変数も含める —
        # 格子変数ゼロの地点データ (lon/lat がスカラー/補助座標) では
        # これが無いと時刻セレクタがどこにも出ず、時刻を固定できない
        _sel_vars = list(dict.fromkeys(
            variables + mc_dataset.map_scatter_variables(ds, roles)))
        # 水平面を張る dim = 描画軸として固定しない次元。1 次元格子では (lat, lon)、
        # curvilinear 格子では lat/lon 座標の dims (y, x)。トラック専用データ
        # (lat/lon 役割なし) では None なので従来どおり (None, None)
        hdims = (mc_dataset.horizontal_dims(ds, roles)
                 or (roles["lat"], roles["lon"]))
        non_time_dims = {dim for var in _sel_vars for dim in ds[var].dims
                          if dim in ds.coords
                          and dim not in (*hdims, roles["time"])}
        selection, time_label_settings = selection_widgets(
            ds, roles, _sel_vars, hdims,
            header_label="時刻", skip_dims=tuple(non_time_dims),
            mode_key=mode_key, show_animation=True, projection=projection)
        # ライン(line)は 1次元プロット専用なので水平断面図では選択肢から外す
        # panel= はレイヤー UI の既定値・参考表示 (データ範囲・描画レベル) を
        # 選択中の時刻・領域の断面から作るための部分的な panel dict
        layers_cfg = layers_ui(datasets, variables_by_ds,
                                hdims,
                                [k for k in KIND_LABELS if k not in ("line", "line_bundle", "fill_between", "bar", "stackplot", "scatter", "bubble", "hexbin", "hist2d", "hist", "ecdf", "box", "violin")],
                                mode_key, roles=roles,
                                panel={"selection": selection, "region": region})

        map_cfg = _map_settings_ui(mode_key, proj_name, is_polar, projection)
        map_cfg["section_paths"] = _section_overlay_ui(mode_key)

    panel = mc_config.default_panel()
    panel["selection"] = selection
    panel["region"] = region
    panel["projection"] = projection
    panel["map"] = map_cfg
    panel["layers"] = layers_cfg
    return panel, selection, time_label_settings


def _section_overlay_ui(mode_key):
    """地図に重ねる断面の経路 (map.section_paths、docs/section_extension_guide.md 6 節)。

    同じ図の鉛直断面モードのパネル (設定済みのもの) を選ぶ。参照は panel_id
    (= セッションのパネル ID の文字列。図の組み立て時に各 panel に付ける)。
    """
    pid = mode_key[len("map"):]
    panels = st.session_state.get("panels") or []
    index_of = {str(p["id"]): i + 1 for i, p in enumerate(panels)}
    cands = [str(p["id"]) for p in panels
             if str(p["id"]) != pid
             and st.session_state.get(f"plot_mode_{p['id']}") == "vsec"
             and st.session_state.get(f"panel_cfg_{p['id']}") is not None]
    with st.expander(t("断面の経路"), expanded=False):
        if not cands:
            st.caption(t("鉛直断面のパネルを追加して設定すると、その経路をこの地図に線で重ねられます"))
            return []
        key = f"secov_sel_{mode_key}"
        stored = [v for v in (st.session_state.get(key) or []) if v in cands]
        if st.session_state.get(key) != stored:
            st.session_state[key] = stored          # 消えたパネルへの参照は外す
        chosen = st.multiselect(
            t("重ねる断面のパネル"), cands,
            format_func=lambda v: t("パネル {n}", n=index_of.get(v, v)), key=key)
        if not chosen:
            return []
        color = color_selector(t("線の色"), "#d62728", key=f"secov_color_{mode_key}")
        width = float(st.number_input(t("線の太さ"), 0.2, 10.0, 1.5, 0.1,
                                      key=f"secov_width_{mode_key}"))
        linestyle = st.selectbox(t("線種"), list(GL_LINESTYLE_LABELS),
                                 format_func=tr_labels(GL_LINESTYLE_LABELS).get,
                                 index=1, key=f"secov_ls_{mode_key}")
        end_labels = st.checkbox(t("端点に文字を付ける (A, B, …)"), value=True,
                                 key=f"secov_labels_{mode_key}")
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        items = []
        for i, p in enumerate(chosen):
            labels = [letters[(2 * i) % 26], letters[(2 * i + 1) % 26]]
            if end_labels:
                # 断面ごとに始点・終点の文字を自由に (空欄ならその端には出さない)
                c1, c2 = st.columns(2)
                with c1:
                    labels[0] = st.text_input(
                        t("始点の文字 (パネル {n})", n=index_of.get(p, p)), value=labels[0],
                        key=f"secov_lab0_{mode_key}_{p}")
                with c2:
                    labels[1] = st.text_input(
                        t("終点の文字 (パネル {n})", n=index_of.get(p, p)), value=labels[1],
                        key=f"secov_lab1_{mode_key}_{p}")
            items.append({"panel_id": p, "color": color, "width": width,
                          "linestyle": linestyle, "end_labels": bool(end_labels),
                          "labels": [str(labels[0]), str(labels[1])],
                          "label_fontsize": 10.0})
        return items


def _lonlat_input(label, value, key, lo, hi):
    """経度・緯度の number_input (0.1° 刻み、小数 2 桁)。"""
    return float(st.number_input(label, value=float(value), min_value=float(lo),
                                 max_value=float(hi), step=0.1, format="%.2f", key=key))


def _section_path_ui(orient, ds, roles, mode_key):
    """経路断面 (等緯度線・等経度線・大円) の入力欄。panel["section_path"] を返す。

    初期値はデータの経緯度範囲 (lonlat_bounds) から: 等緯度線は中央の緯度と経度の全範囲、
    等経度線は中央の経度と緯度の全範囲、大円は領域の中央を通る東西の線 (両端 15% を空ける)。
    点の数は自動 (経路長 ÷ 格子間隔) か指定。
    """
    b = mc_dataset.lonlat_bounds(ds, roles) or {}
    lon0, lon1 = float(b.get("lon_min", -180.0)), float(b.get("lon_max", 180.0))
    lat0, lat1 = float(b.get("lat_min", -90.0)), float(b.get("lat_max", 90.0))
    lat_c = round((lat0 + lat1) / 2, 1)
    lon_c = round((lon0 + lon1) / 2, 1)
    if orient == "parallel":
        lat = _lonlat_input(t("緯度 (°N)"), lat_c, f"vsec_par_lat_{mode_key}", -90.0, 90.0)
        c1, c2 = st.columns(2)
        with c1:
            lo = _lonlat_input(t("経度の始点"), round(lon0, 1), f"vsec_par_lon0_{mode_key}",
                               -720.0, 720.0)
        with c2:
            hi = _lonlat_input(t("経度の終点"), round(lon1, 1), f"vsec_par_lon1_{mode_key}",
                               -720.0, 720.0)
        spec = {"kind": "parallel", "lat": lat, "lon_range": [lo, hi], "npoints": None}
    elif orient == "meridian":
        lon = _lonlat_input(t("経度 (°E)"), lon_c, f"vsec_mer_lon_{mode_key}", -720.0, 720.0)
        c1, c2 = st.columns(2)
        with c1:
            lo = _lonlat_input(t("緯度の始点"), round(lat0, 1), f"vsec_mer_lat0_{mode_key}",
                               -90.0, 90.0)
        with c2:
            hi = _lonlat_input(t("緯度の終点"), round(lat1, 1), f"vsec_mer_lat1_{mode_key}",
                               -90.0, 90.0)
        spec = {"kind": "meridian", "lon": lon, "lat_range": [lo, hi], "npoints": None}
    else:
        w = lon1 - lon0
        c1, c2 = st.columns(2)
        with c1:
            slon = _lonlat_input(t("始点の経度"), round(lon0 + 0.15 * w, 1),
                                 f"vsec_gc_lon0_{mode_key}", -720.0, 720.0)
            slat = _lonlat_input(t("始点の緯度"), lat_c, f"vsec_gc_lat0_{mode_key}",
                                 -90.0, 90.0)
        with c2:
            elon = _lonlat_input(t("終点の経度"), round(lon1 - 0.15 * w, 1),
                                 f"vsec_gc_lon1_{mode_key}", -720.0, 720.0)
            elat = _lonlat_input(t("終点の緯度"), lat_c, f"vsec_gc_lat1_{mode_key}",
                                 -90.0, 90.0)
        spec = {"kind": "great_circle", "start": [slon, slat], "end": [elon, elat],
                "npoints": None}
    auto = st.checkbox(t("点の数を自動にする (点の間隔 ≈ 格子間隔)"), value=True,
                       key=f"vsec_np_auto_{mode_key}")
    if not auto:
        spec["npoints"] = int(st.number_input(t("点の数"), min_value=2, max_value=5000,
                                              value=200, step=10, key=f"vsec_np_{mode_key}"))
    # 参考表示: 経路長・点の間隔・格子間隔 (render と同じ関数で計算)
    try:
        full = mc_render.section_path_spec(spec, ds)
        length = mc_render.section_path_length_km(full)
        n = full["npoints"]
        lat_ref = {"parallel": spec.get("lat"),
                   "meridian": float(np.mean(spec.get("lat_range", [0.0, 0.0])))}.get(
            orient, float(np.mean([spec.get("start", [0, 0])[1], spec.get("end", [0, 0])[1]])))
        spacing = mc_render.grid_spacing_km(ds, roles, lat_ref)
        st.caption(t("経路長 {length} km、{n} 点、点の間隔 約 {step} km (格子間隔 約 {grid} km)",
                     length=f"{length:.0f}", n=n,
                     step=f"{length / max(n - 1, 1):.1f}", grid=f"{spacing:.1f}"))
        if orient in ("parallel", "meridian"):
            lon_p, lat_p, _ = mc_render.section_path_points(full)
            fj, fi, _, _, _ = mc_render.section_path_indices(ds, lon_p, lat_p)
            ok = np.isfinite(fj) & np.isfinite(fi)
            if ok.any():
                vals = (lon_p if orient == "parallel" else lat_p)[ok]
                st.caption(t("このうち格子の中にある範囲: {lo}〜{hi}",
                             lo=f"{vals.min():.1f}", hi=f"{vals.max():.1f}"))
            else:
                st.warning(t("経路がデータの格子の外にあります"))
    except mc_render.RenderError as exc:
        st.warning(_error_text(exc))
    return spec


_TERRAIN_METHOD_LABELS = {"surface_pressure": "地上気圧の変数 (鉛直座標が気圧)",
                          "surface_height": "地形高度の変数 (鉛直座標が高度)",
                          "height_field": "高度の変数と地形高度の変数 (鉛直座標が気圧)"}


def _terrain_ui(datasets, ds, y_dim, mode_key):
    """地形マスクの設定 (docs/section_extension_guide.md 5 節)。panel["terrain"] を返す。

    方法は鉛直座標の単位 (気圧 / 高度) で絞る。変数は読み込み済みの全ファイルから、
    水平の 2 次元だけ (時刻はあってもよい) の変数を「dsid: 変数名」で選ぶ。
    """
    cfg = mc_config.default_section_panel()["terrain"]
    kind = mc_render.vertical_axis_kind(ds[y_dim].attrs.get("units"))
    with st.expander(t("地形マスク"), expanded=False):
        if kind is None:
            st.caption(t("鉛直座標 {dim} の単位 ({units}) から気圧か高度か判定できないため、"
                         "地形マスクは使えません",
                         dim=y_dim, units=str(ds[y_dim].attrs.get("units", ""))))
            return cfg
        cfg["show"] = st.checkbox(
            t("地面より下を地形で覆う"), value=False, key=f"vsec_ter_show_{mode_key}",
            help=t("全レイヤーをデータの値のまま描いた上に、地面より下を塗った多角形を"
                   "重ねる (地下の値を欠損にはしない。色の自動範囲には地下の値も含まれる)"))
        if not cfg["show"]:
            return cfg
        methods = (["surface_pressure", "height_field"] if kind == "pressure"
                   else ["surface_height"])
        mkey = f"vsec_ter_method_{mode_key}"
        if st.session_state.get(mkey) not in methods:
            st.session_state[mkey] = methods[0]
        cfg["method"] = st.selectbox(t("地面の求め方"), methods,
                                     format_func=tr_labels(_TERRAIN_METHOD_LABELS).get, key=mkey)

        y_units = str(ds[y_dim].attrs.get("units", ""))
        units_of = {}

        def _candidates(need_vertical, target, ukind):
            """「dsid: 変数名」の候補。単位が target に換算できるものを先に並べる。"""
            good, other = [], []
            for dsid, d in datasets.items():
                hd = mc_dataset.horizontal_dims(d, mc_dataset.detect_coord_roles(d))
                if not hd:
                    continue
                for v in d.data_vars:
                    dims = set(d[v].dims)
                    if not set(hd) <= dims or need_vertical != (y_dim in dims):
                        continue
                    extra = dims - set(hd) - {y_dim}
                    if len(extra) > 1 or not all(
                            e in d.coords and np.issubdtype(d[e].dtype, np.datetime64)
                            for e in extra):
                        continue
                    key = f"{dsid}: {v}"
                    units_of[key] = str(d[v].attrs.get("units", ""))
                    (good if mc_render.unit_factor(units_of[key], target, ukind) is not None
                     else other).append(key)
            return good + other

        def _fmt(key):
            return f"{key} [{units_of.get(key) or '?'}]"

        if cfg["method"] == "surface_pressure":
            surf = _candidates(False, y_units, "pressure")
        else:
            surf = _candidates(False, "m", "height")
        if not surf:
            st.warning(t("水平 2 次元の変数 (地上気圧・地形高度) が読み込み済みのファイルにありません。"
                         "ファイルを追加してください"))
            cfg["show"] = False
            return cfg
        label = (t("地上気圧の変数") if cfg["method"] == "surface_pressure"
                 else t("地形高度の変数"))
        # 方法ごとに別の key (地上気圧 ps と地形高度 zs の選択を別々に覚える)
        vkey = f"vsec_ter_var_{cfg['method']}_{mode_key}"
        if st.session_state.get(vkey) not in surf:
            st.session_state[vkey] = surf[0]
        choice = st.selectbox(label, surf, key=vkey, format_func=_fmt)
        cfg["dataset_id"], cfg["variable"] = choice.split(": ", 1)
        if cfg["method"] == "height_field":
            hv = _candidates(True, "m", "height")
            if not hv:
                st.warning(t("鉛直を持つ高度の変数が読み込み済みのファイルにありません"))
                cfg["show"] = False
                return cfg
            hkey = f"vsec_ter_hvar_{mode_key}"
            if st.session_state.get(hkey) not in hv:
                st.session_state[hkey] = hv[0]
            hchoice = st.selectbox(t("高度の変数 (ジオポテンシャル高度など)"), hv, key=hkey,
                                   format_func=_fmt)
            cfg["height_dataset_id"], cfg["height_variable"] = hchoice.split(": ", 1)
        cfg["color"] = color_selector(t("地面の色"), "#7f7f7f", key=f"vsec_ter_color_{mode_key}")
        st.caption(t("変数の単位は units 属性から鉛直座標の単位 [{yunits}] に換算して比べる "
                     "(Pa ⇄ hPa、m ⇄ km、ジオポテンシャル m2 s-2 は g で割る)", yunits=y_units))
    return cfg


def _vsec_mode_ui(datasets, ds, roles, mode_key):
    """鉛直断面 (vsec) モードのパネル設定 UI。

    向き: 1 次元格子は 経度–高度 / 緯度–高度 (従来、内挿なし) と 大円、2 次元座標格子は
    格子の行 / 列 (内挿なし) と 等緯度線 / 等経度線 / 大円 (双一次内挿)。経路断面では
    x_dim = render.SECTION_PATH_DIM ("path")。docs/section_extension_guide.md 4 節。
    """
    curvi = mc_dataset.is_curvilinear(ds, roles)
    hdims = mc_dataset.horizontal_dims(ds, roles) if curvi else None
    with st.sidebar:
        section_header(t("断面"))
        orient_opts = []
        if curvi:
            orient_opts += ["grid_row", "grid_col", "parallel", "meridian", "great_circle"]
        else:
            if roles["lon"]:
                orient_opts.append("lon_height")
            if roles["lat"]:
                orient_opts.append("lat_height")
            if roles["lon"] and roles["lat"]:
                orient_opts.append("great_circle")
        # 別のデータの残留値 (1 次元格子の "lon_height" など) は先頭に戻す
        okey = f"vsec_orient_{mode_key}"
        if st.session_state.get(okey) not in orient_opts:
            st.session_state[okey] = orient_opts[0]
        orient = st.radio(t("向き"), orient_opts,
                          format_func=tr_labels(_VSEC_ORIENT_LABELS).get, key=okey)
        y_dim = roles["vertical"]
        section_path = None
        fixed_dim = None
        lonlat_ticks = None
        if orient in ("lon_height", "lat_height"):
            x_dim = roles["lon"] if orient == "lon_height" else roles["lat"]
            fixed_dim = roles["lat"] if orient == "lon_height" else roles["lon"]
            # 固定する緯度/経度はレイヤー毎に選ぶ (下の「レイヤー」セクションで指定)
            keep_dims = (x_dim, y_dim)
        elif orient in ("grid_row", "grid_col"):
            x_dim = hdims[1] if orient == "grid_row" else hdims[0]
            fixed_dim = hdims[0] if orient == "grid_row" else hdims[1]
            keep_dims = (x_dim, y_dim)
        else:
            x_dim = mc_render.SECTION_PATH_DIM
            keep_dims = (*(hdims or (roles["lat"], roles["lon"])), y_dim)
            section_path = _section_path_ui(orient, ds, roles, mode_key)

        lev_vals = [float(v) for v in ds[y_dim].values]
        lev_lo, lev_hi = st.select_slider(
            t("高度・気圧レベル範囲"), options=lev_vals,
            value=(lev_vals[0], lev_vals[-1]), key=f"vsec_range_{mode_key}")
        ranges = {} if (lev_lo, lev_hi) == (lev_vals[0], lev_vals[-1]) else {y_dim: [lev_lo, lev_hi]}
        if orient in ("lon_height", "lat_height"):
            x_vals = [float(v) for v in ds[x_dim].values]
            x_label = t("経度範囲") if orient == "lon_height" else t("緯度範囲")
            x_lo, x_hi = st.select_slider(
                x_label, options=x_vals,
                value=(x_vals[0], x_vals[-1]), key=f"vsec_xrange_{mode_key}")
            if (x_lo, x_hi) != (x_vals[0], x_vals[-1]):
                ranges[x_dim] = [x_lo, x_hi]
        elif orient in ("grid_row", "grid_col"):
            # 格子番号の範囲 (座標のない dim は 0 始まりの index)
            x_vals = [float(v) for v in ds[x_dim].values]
            x_lo, x_hi = st.select_slider(
                t("格子番号の範囲 ({dim})", dim=x_dim), options=x_vals,
                value=(x_vals[0], x_vals[-1]), key=f"vsec_grange_{mode_key}")
            if (x_lo, x_hi) != (x_vals[0], x_vals[-1]):
                ranges[x_dim] = [x_lo, x_hi]
        if orient in ("grid_row", "grid_col", "great_circle"):
            lonlat_ticks = st.checkbox(
                t("目盛に経度・緯度を併記する"), value=True, key=f"vsec_llticks_{mode_key}",
                help=t("横軸の目盛ごとに、軸の値 (格子番号・距離) とその位置の経度・緯度を "
                       "3 段で表示する"))

    if section_path is None:
        variables = mc_dataset.section_variables(ds, x_dim, y_dim)
        variables_by_ds = {dsid: mc_dataset.section_variables(d, x_dim, y_dim)
                           for dsid, d in datasets.items()}
    else:
        # 経路断面: 水平の 2 次元 (内挿で潰す) と鉛直の 3 つを持つ変数
        def _path_vars(d):
            hd = mc_dataset.horizontal_dims(d, mc_dataset.detect_coord_roles(d))
            if not hd:
                return []
            return [str(v) for v in d.data_vars
                    if all(dim in d[v].dims for dim in (*hd, y_dim))]
        variables = _path_vars(ds)
        variables_by_ds = {dsid: _path_vars(d) for dsid, d in datasets.items()}
    if not variables:
        st.error(t("{x_dim} と {y_dim} の両方の次元を持つ変数が見つかりません。",
                 x_dim=x_dim if section_path is None else "/".join(keep_dims[:-1]),
                 y_dim=y_dim))
        st.stop()

    with st.sidebar:
        # fixed_dim (緯度 or 経度、格子の行 or 列) はレイヤー毎なので panel.selection からは除外
        selection, time_label_settings = selection_widgets(
            ds, roles, variables, keep_dims,
            header_label="時刻", skip_dims=(fixed_dim,) if fixed_dim else (),
            mode_key=mode_key, show_animation=True)
        # ライン(line)は 1次元プロット専用なので鉛直断面図では選択肢から外す
        layers_cfg = layers_ui(datasets, variables_by_ds, keep_dims,
                                [k for k in KIND_LABELS if k not in ("line", "line_bundle", "fill_between", "bar", "stackplot", "scatter", "bubble", "hexbin", "hist2d", "map_scatter", "track", "hist", "ecdf", "box", "violin")],
                                mode_key, roles=roles, allow_averaging=True,
                                panel={"selection": selection, "ranges": ranges,
                                       "x_dim": x_dim, "y_dim": y_dim,
                                       "section_path": section_path})
        if orient in ("grid_row", "grid_col") and layers_cfg:
            # 固定した行・列の経緯度の範囲 (格子の傾きに気づけるように)
            sel = {**selection, **(layers_cfg[0].get("selection") or {})}
            if fixed_dim in sel:
                try:
                    lon_l = ds[roles["lon"]].sel({fixed_dim: sel[fixed_dim]}).values
                    lat_l = ds[roles["lat"]].sel({fixed_dim: sel[fixed_dim]}).values
                    st.caption(t("{dim} = {v} に沿う断面: 緯度 {lat0}〜{lat1}°N、経度 {lon0}〜{lon1}°E (先頭レイヤー)",
                                 dim=fixed_dim, v=f"{sel[fixed_dim]:g}",
                                 lat0=f"{np.nanmin(lat_l):.1f}", lat1=f"{np.nanmax(lat_l):.1f}",
                                 lon0=f"{np.nanmin(lon_l):.1f}", lon1=f"{np.nanmax(lon_l):.1f}"))
                except (KeyError, TypeError, ValueError):
                    pass

        if section_path is None:
            axis = axis_settings_ui(ds, roles, x_dim, y_dim, mode_key)
        else:
            x_label_default = {"parallel": coord_label(ds, roles, roles["lon"]),
                               "meridian": coord_label(ds, roles, roles["lat"]),
                               "great_circle": "distance [km]"}[orient]
            axis = axis_settings_ui(ds, roles, x_dim, y_dim, mode_key,
                                    x_label_default=x_label_default,
                                    x_is_lon=(orient == "parallel"),
                                    x_key=f"{x_dim}_{orient}")
        if lonlat_ticks is not None:
            axis["x_lonlat_ticks"] = bool(lonlat_ticks)
        terrain_cfg = _terrain_ui(datasets, ds, y_dim, mode_key)
        frame_cfg, background_cfg = frame_background_ui(
            mode_key, frame_width_only=True)

    panel = mc_config.default_section_panel()
    panel["x_dim"], panel["y_dim"] = x_dim, y_dim
    panel["section_path"] = section_path
    panel["terrain"] = terrain_cfg
    panel["selection"] = selection
    panel["ranges"] = ranges
    panel["axis"] = axis
    panel["frame"] = frame_cfg
    panel["background"] = background_cfg
    panel["layers"] = layers_cfg
    return panel, selection, time_label_settings


def _tsec_mode_ui(datasets, ds, roles, mode_key):
    """時間断面 (tsec) モードのパネル設定 UI。"""
    with st.sidebar:
        section_header(t("時間断面の種類"))
        kinds = []
        if roles["vertical"]:
            kinds.append("time_height")
        if roles["lat"]:
            kinds.append("time_lat")
        if roles["lon"]:
            kinds.append("time_lon")
        tkind = st.selectbox(t("種類"), kinds,
                             format_func=tr_labels(_TSEC_KIND_LABELS).get,
                             key=f"tsec_kind_{mode_key}")
        if tkind == "time_height":
            x_dim, y_dim = roles["time"], roles["vertical"]
        elif tkind == "time_lat":
            x_dim, y_dim = roles["lat"], roles["time"]
        else:
            x_dim, y_dim = roles["lon"], roles["time"]
        if y_dim == roles["time"]:
            st.caption(t("Hovmöller図の慣例に従い、時間を縦軸 (下向き) にとります。"))

        # 時刻範囲スライダーは「時間断面の種類」セクション内に置く
        tlabels, tvals = time_labels_and_values(ds[roles["time"]].values)
        # ラベルで表示しつつ index で値を引く (整数年などラベル≠値のケースに対応)
        t0_lbl, t1_lbl = st.select_slider(t("時刻範囲"), options=tlabels,
                                          value=(tlabels[0], tlabels[-1]), key=f"tsec_range_{mode_key}")
        i0, i1 = tlabels.index(t0_lbl), tlabels.index(t1_lbl)
        ranges = ({} if (i0, i1) == (0, len(tlabels) - 1)
                  else {roles["time"]: [tvals[i0], tvals[i1]]})

    variables = mc_dataset.section_variables(ds, x_dim, y_dim)
    variables_by_ds = {dsid: mc_dataset.section_variables(d, x_dim, y_dim)
                        for dsid, d in datasets.items()}
    if not variables:
        st.error(t("{x_dim} と {y_dim} の両方の次元を持つ変数が見つかりません。",
                 x_dim=x_dim, y_dim=y_dim))
        st.stop()

    with st.sidebar:
        # 時間断面でのベクトル・流線の2成分は物理的に未定義のため非対応
        # (development_policy.md)。ライン(line)は 1次元プロット専用なので除外
        kinds_allowed = [k for k in KIND_LABELS if k not in ("vector", "stream", "line", "line_bundle", "fill_between", "bar", "stackplot", "scatter", "bubble", "hexbin", "hist2d", "map_scatter", "track", "hist", "ecdf", "box", "violin")]
        # 固定する2次元 (lat / level など) はレイヤー毎に指定するので、
        # panel.selection は空にして「断面の選択」セクションは作らない。
        # 緯度・経度方向は範囲平均も選べる (鉛直方向は固定値のみ)。
        selection = {}
        time_label_settings = {"show": False, "loc": "left", "format": None}
        layers_cfg = layers_ui(datasets, variables_by_ds, (x_dim, y_dim),
                                kinds_allowed, mode_key, roles=roles,
                                allow_averaging=True,
                                panel={"selection": selection, "ranges": ranges,
                                       "x_dim": x_dim, "y_dim": y_dim})

        axis = axis_settings_ui(ds, roles, x_dim, y_dim, mode_key)
        # 時間軸側の背景の塗り範囲は時刻スライダーで選べるようにする
        _tsec_time = (roles["time"], tlabels, tvals)
        frame_cfg, background_cfg = frame_background_ui(
            mode_key,
            x_time=_tsec_time if x_dim == roles["time"] else None,
            y_time=_tsec_time if y_dim == roles["time"] else None,
            frame_width_only=True)

    panel = mc_config.default_section_panel()
    panel["x_dim"], panel["y_dim"] = x_dim, y_dim
    panel["selection"] = selection
    panel["ranges"] = ranges
    panel["axis"] = axis
    panel["frame"] = frame_cfg
    panel["background"] = background_cfg
    panel["layers"] = layers_cfg
    return panel, selection, time_label_settings


def _line_mode_ui(datasets, ds, roles, mode_key):
    """1次元プロット (line) モードのパネル設定 UI。"""
    axis_candidates = [str(d) for d in ds.dims if d in ds.coords]
    if not axis_candidates:
        st.error(t("座標を持つ次元がありません。1次元プロットは描画できません。"))
        st.stop()

    with st.sidebar:
        section_header(t("プロット軸"))
        if st.session_state.get(f"line_x_dim_{mode_key}") not in axis_candidates:
            st.session_state[f"line_x_dim_{mode_key}"] = axis_candidates[0]
        x_dim = st.selectbox(t("軸"), axis_candidates, key=f"line_x_dim_{mode_key}")

        # 経度軸のときだけ周期的展開を許可 (period=360°)
        cyclic_x = False
        if x_dim == roles.get("lon"):
            cyclic_x = st.checkbox(
                t("x軸を周期的に展開 (360°を超えて表示)"), value=False,
                key=f"line_cyclic_{mode_key}_{x_dim}",
                help=t("経度データを 360° 周期で繰り返して描く。"
                     "範囲をデータ範囲を超えて指定できる"))

        # 範囲 (x_dim)。スライダーと最小/最大 (時刻は開始/終了) の入力欄を
        # 連動させる (widgets.linked_range_ui / linked_time_range_ui)
        import numpy as _np_local
        # 候補値は読み込んだ全 dataset の x_dim 座標の和集合 (先頭 dataset だけだと、
        # 期間の続く別ファイルを重ねたときスライダーが先頭の範囲に固定される)
        x_vals_raw = union_coord_values(datasets, x_dim)
        ranges = {}
        _rng_keys = dict(slider_key=f"line_xrange_{mode_key}_{x_dim}",
                         lo_key=f"line_xrange_lo_{mode_key}_{x_dim}",
                         hi_key=f"line_xrange_hi_{mode_key}_{x_dim}")
        if _np_local.issubdtype(_np_local.asarray(x_vals_raw).dtype, _np_local.datetime64):
            xlabels, xv = time_labels_and_values(x_vals_raw)
            i0, i1 = linked_time_range_ui(
                t("範囲"), xlabels, **_rng_keys,
                lo_label=t("範囲 開始"), hi_label=t("範囲 終了"),
                help=t("スライダーと開始・終了の入力欄は連動する。入力欄は ISO 形式 "
                       "(例 2024-01-01 や 2024-01-01T12:00) で、最寄りの時刻に丸める"))
            if (i0, i1) != (0, len(xlabels) - 1):
                ranges[x_dim] = [xv[i0], xv[i1]]
        elif cyclic_x:
            # cyclic モード: 数値入力で任意の範囲 (360°を超える値も指定可)
            xv = [float(v) for v in x_vals_raw]
            period = 360.0
            c1, c2 = st.columns(2)
            x_lo = c1.number_input(t("範囲 最小"), value=float(xv[0]), step=10.0,
                                    key=f"line_xrange_lo_cyc_{mode_key}_{x_dim}")
            x_hi = c2.number_input(t("範囲 最大"), value=float(xv[-1] + period), step=10.0,
                                    key=f"line_xrange_hi_cyc_{mode_key}_{x_dim}")
            ranges[x_dim] = [float(x_lo), float(x_hi)]
        else:
            try:
                xv = [float(v) for v in x_vals_raw]
            except (TypeError, ValueError):
                xv = None  # 非数値軸はスキップ
            if xv:
                x_lo, x_hi = linked_range_ui(
                    t("範囲"), xv, **_rng_keys,
                    lo_label=t("範囲 最小"), hi_label=t("範囲 最大"),
                    help=t("スライダーと最小・最大の入力欄は連動する。入力した数値は"
                           "そのまま範囲に使い (格子点に丸めない)、スライダーは最寄りの"
                           "格子点を示す"))
                if (x_lo, x_hi) != (min(xv), max(xv)):
                    ranges[x_dim] = [x_lo, x_hi]

    variables_by_ds = {dsid: [str(v) for v in d.data_vars if x_dim in d[v].dims]
                        for dsid, d in datasets.items()}
    canonical_id = next(iter(datasets))
    if not variables_by_ds[canonical_id]:
        st.error(t("プロット軸 `{x_dim}` を持つ変数が `{dsid}` にありません。",
                 x_dim=x_dim, dsid=canonical_id))
        st.stop()
    variables = variables_by_ds[canonical_id]

    # 非時刻・非 x_dim の次元 (lat, lon, level など) はレイヤー毎に固定する
    non_time_dims = {dim for var in variables for dim in ds[var].dims
                      if dim in ds.coords
                      and dim != x_dim and dim != roles["time"]}

    has_time_to_fix = (
        roles["time"] is not None
        and roles["time"] != x_dim
        and any(roles["time"] in ds[v].dims for v in variables)
    )

    with st.sidebar:
        if has_time_to_fix:
            selection, time_label_settings = selection_widgets(
                ds, roles, variables, (x_dim,),
                header_label="時刻",
                skip_dims=tuple(non_time_dims),
                mode_key=mode_key, show_animation=True)
        else:
            selection = {}
            time_label_settings = {"show": False, "loc": "left", "format": None}
        layers_cfg = layers_ui(datasets, variables_by_ds, (x_dim,),
                                ["line", "line_bundle", "fill_between", "stackplot", "bar"], mode_key,
                                roles=roles, allow_averaging=True)
        # bar レイヤーが含まれるときだけ「棒グラフの配置」を出す
        bar_mode = "overlap"
        bar_dodge_gap = 0.05
        if any(lyr.get("kind") == "bar" for lyr in layers_cfg):
            _BAR_MODE_LABELS = {"overlap": "重ね合わせ", "dodge": "横並び",
                                 "stack": "積み上げ"}
            bar_mode = st.radio(
                t("棒グラフの配置"), list(_BAR_MODE_LABELS),
                format_func=tr_labels(_BAR_MODE_LABELS).get, horizontal=True,
                key=f"bar_mode_{mode_key}",
                help=t("複数の棒グラフレイヤーがある時の並び。overlap は alpha で重ね、"
                     "dodge は等間隔の横並び、stack は積み上げ (正/負を別々に積む)"))
            if bar_mode == "dodge":
                bar_dodge_gap = float(st.slider(
                    t("棒の隙間"), 0.0, 0.5, 0.05, 0.01,
                    key=f"bar_dodge_gap_{mode_key}",
                    help=t("隣り合う棒の間に空ける隙間 (スロット幅に対する比率)。"
                         "0 だと境界が滲んで見えやすい")))
        axis = line_axis_settings_ui(ds, roles, x_dim, mode_key)
        legend = legend_settings_ui(mode_key)
        # x 軸が時間軸なら、背景の塗り範囲を時刻スライダーで選べるようにする
        x_time = None
        if _np_local.issubdtype(_np_local.asarray(x_vals_raw).dtype,
                                 _np_local.datetime64):
            _tlabels, _tvalues = time_labels_and_values(x_vals_raw)
            x_time = (x_dim, _tlabels, _tvalues)
        frame_cfg, background_cfg = frame_background_ui(mode_key, x_time=x_time)

    panel = mc_config.default_line_panel()
    panel["x_dim"] = x_dim
    panel["selection"] = selection
    panel["ranges"] = ranges
    panel["axis"] = axis
    panel["axis"]["cyclic_x"] = cyclic_x  # 「プロット軸」セクションで決定
    panel["axis"]["bar_mode"] = bar_mode   # 「レイヤー」末尾で決定
    panel["axis"]["bar_dodge_gap"] = bar_dodge_gap
    panel["frame"] = frame_cfg
    panel["background"] = background_cfg
    panel["legend"] = legend
    panel["layers"] = layers_cfg
    return panel, selection, time_label_settings


def _dist_mode_ui(datasets, ds, roles, mode_key):
    """1次元プロット(集計) (dist) モードのパネル設定 UI。"""
    variables_by_ds = {dsid: mc_dataset.nongeo_variables(d)
                        for dsid, d in datasets.items()}
    # 集計系は時刻選択・アニメーション無し (時間は集計する次元として消費)
    selection = {}
    time_label_settings = {"show": False, "loc": "left", "format": None}
    with st.sidebar:
        dist_family = st.radio(
            t("描画タイプ"), list(_DIST_FAMILY_LABELS),
            format_func=tr_labels(_DIST_FAMILY_LABELS).get,
            horizontal=True, key=f"dist_family_{mode_key}",
            help=t("ヒストグラム・ECDF: x = 値、y = 度数・割合。"
                 "箱ひげ・バイオリン: x = 系列の並び (1レイヤー = 1系列)、"
                 "y = 値"))
        _dist_kinds = (["hist", "ecdf", "line"]
                       if dist_family == "hist"
                       else ["box", "violin"])
        dist_orientation = "vertical"
        if dist_family == "box":
            dist_orientation = st.radio(
                t("向き"), list(_DIST_ORIENT_LABELS),
                format_func=tr_labels(_DIST_ORIENT_LABELS).get, horizontal=True,
                key=f"dist_orient_{mode_key}",
                help=t("横: 箱・バイオリンを横倒しにして、系列名を y 軸、"
                     "値を x 軸にする (系列名が長いときに読みやすい)"))
        layers_cfg = layers_ui(datasets, variables_by_ds, (),
                                _dist_kinds, mode_key, roles=roles)
        section_header(t("軸・ラベル・凡例"))
        axis = mc_config.default_dist_panel()["axis"]
        axis_basic_ui(axis, mode_key, show_swap_y=True, show_y2=True)
        axis_labels_ui(axis, mode_key,
                       x_placeholder=t("(自動: 変数名 [units])"),
                       y_placeholder=t("(自動: count / probability density)"),
                       show_y2=True)
        axis_ticks_ui(axis, mode_key)
        legend = legend_settings_ui(mode_key)
        frame_cfg, background_cfg = frame_background_ui(mode_key)
    panel = mc_config.default_dist_panel()
    panel["box_orientation"] = dist_orientation
    panel["axis"] = axis
    panel["frame"] = frame_cfg
    panel["background"] = background_cfg
    panel["legend"] = legend
    panel["layers"] = layers_cfg
    return panel, selection, time_label_settings


def _agg_mode_ui(datasets, ds, roles, mode_key):
    """2次元プロット(集計) (agg) モードのパネル設定 UI。"""
    nongeo = mc_dataset.nongeo_variables(ds, roles)
    variables_by_ds = {dsid: mc_dataset.nongeo_variables(d)
                        for dsid, d in datasets.items()}
    selection = {}
    time_label_settings = {"show": False, "loc": "left", "format": None}
    with st.sidebar:
        section_header(t("プロット変数"))
        if st.session_state.get(f"panel_xvar_{mode_key}") not in nongeo:
            st.session_state[f"panel_xvar_{mode_key}"] = nongeo[0]
        if st.session_state.get(f"panel_yvar_{mode_key}") not in nongeo:
            st.session_state[f"panel_yvar_{mode_key}"] = (
                nongeo[1] if len(nongeo) >= 2 else nongeo[0])
        x_var = st.selectbox(t("x 軸の変数"), nongeo,
                             key=f"panel_xvar_{mode_key}",
                             format_func=lambda v: var_label(ds, v))
        y_var = st.selectbox(t("y 軸の変数"), nongeo,
                             key=f"panel_yvar_{mode_key}",
                             format_func=lambda v: var_label(ds, v))
        layers_cfg = layers_ui(datasets, variables_by_ds, (),
                                ["hist2d", "hexbin"], mode_key,
                                roles=roles)
        section_header(t("軸・ラベル"))
        axis = mc_config.default_agg_panel()["axis"]
        axis_basic_ui(axis, mode_key, show_swap_y=True)
        axis_labels_ui(axis, mode_key,
                       x_placeholder=t("(自動: x 変数 [units])"),
                       y_placeholder=t("(自動: y 変数 [units])"))
        axis_ticks_ui(axis, mode_key)
        frame_cfg, background_cfg = frame_background_ui(mode_key)
    panel = mc_config.default_agg_panel()
    panel["x_variable"] = x_var
    panel["y_variable"] = y_var
    panel["axis"] = axis
    panel["frame"] = frame_cfg
    panel["background"] = background_cfg
    panel["layers"] = layers_cfg
    return panel, selection, time_label_settings


def _scatter_mode_ui(datasets, ds, roles, mode_key):
    """2次元プロット (scatter) モードのパネル設定 UI (描画タイプで散布図 / heatmap)。"""
    variables = [str(v) for v in ds.data_vars]
    variables_by_ds = {dsid: [str(v) for v in d.data_vars]
                        for dsid, d in datasets.items()}
    if len(variables) < 1:
        st.error(t("2次元プロットには変数が1つ以上必要です。"))
        st.stop()

    # 散布モードではアニメーション・時刻ラベルは未対応 (drawing_dim 沿いに散布)
    selection = {}
    time_label_settings = {"show": False, "loc": "left", "format": None}

    with st.sidebar:
        section_header(t("プロット変数"))
        draw2d = st.radio(
            t("描画タイプ"), list(_DRAW2D_LABELS),
            format_func=tr_labels(_DRAW2D_LABELS).get,
            horizontal=True, key=f"draw2d_{mode_key}",
            help=t("categorical heatmap: 1つの2次元変数 (無次元の小行列) を "
                 "imshow で等幅セルの行列として色表示します"))

    if draw2d == "heatmap":
        with st.sidebar:
            panel = _build_heatmap_panel(datasets, roles, mode_key)
            section_header(t("軸・ラベル"))
            if panel is not None:
                axis = panel["axis"]
                # 「軸」= 1次元プロットと同じ部品 (対数・ぴったりは出さない)
                axis_basic_ui(axis, mode_key, show_log=False,
                              show_swap_y=True)
                # 「ラベル」= 1次元プロットと同じ構成
                axis_labels_ui(axis, mode_key, x_placeholder=t("(なし)"),
                               y_placeholder=t("(なし)"))
                # 「目盛」= 1次元プロットと同じ部品 (補助目盛・目盛線は出さない)。
                # 目盛位置を手動指定した軸はカテゴリラベルではなくセル index の
                # 数値軸として扱われる。x 目盛りラベルの回転もこの中
                axis_ticks_ui(axis, mode_key, show_minor=False,
                              show_grid=False, rotation_panel=panel)
            frame_cfg, background_cfg = frame_background_ui(mode_key)
        if panel is None:
            # ガードで描けない (メッセージは _build_heatmap_panel で表示済み)
            st.stop()
        panel["frame"] = frame_cfg
        panel["background"] = background_cfg
    else:
        with st.sidebar:
            x_default_idx = 0
            y_default_idx = 1 if len(variables) >= 2 else 0
            # session_state のキーは scatter_layer_ui から参照される (panel-level)
            if st.session_state.get(f"panel_xvar_{mode_key}") not in variables:
                st.session_state[f"panel_xvar_{mode_key}"] = variables[x_default_idx]
            if st.session_state.get(f"panel_yvar_{mode_key}") not in variables:
                st.session_state[f"panel_yvar_{mode_key}"] = variables[y_default_idx]
            x_var = st.selectbox(
                t("x 軸の変数"), variables, key=f"panel_xvar_{mode_key}",
                format_func=lambda v: var_label(ds, v))
            y_var = st.selectbox(
                t("y 軸の変数"), variables, key=f"panel_yvar_{mode_key}",
                format_func=lambda v: var_label(ds, v))
            # z 変数 (バブルチャート専用、任意)
            z_options = [_NONE_OPTION] + variables
            if st.session_state.get(f"panel_zvar_{mode_key}") not in z_options:
                st.session_state[f"panel_zvar_{mode_key}"] = _NONE_OPTION
            z_var_choice = st.selectbox(
                t("z 軸の変数 (バブルチャート用、任意)"), z_options,
                key=f"panel_zvar_{mode_key}",
                format_func=lambda v, _none=t("(なし)"): (
                    _none if v == _NONE_OPTION else var_label(ds, v)),
                help=t("バブルチャートのマーカーサイズに使う変数"))
            z_var = None if z_var_choice == _NONE_OPTION else z_var_choice
            layers_cfg = layers_ui(datasets, variables_by_ds, (),
                                    ["scatter", "bubble"],
                                    mode_key, roles=roles)
            # 軸 (簡易版)
            section_header(t("軸・ラベル・凡例"))
            axis = mc_config.default_scatter_panel()["axis"]
            axis_basic_ui(axis, mode_key, show_swap_y=True)
            # ラベル・目盛の詳細はセクションの一番下に置く (ユーザー指示)
            axis_labels_ui(axis, mode_key, x_placeholder=t("(なし)"),
                           y_placeholder=t("(なし)"))
            axis_ticks_ui(axis, mode_key)
            legend = legend_settings_ui(mode_key)
            frame_cfg, background_cfg = frame_background_ui(mode_key)

        panel = mc_config.default_scatter_panel()
        panel["x_variable"] = x_var
        panel["y_variable"] = y_var
        panel["z_variable"] = z_var
        panel["axis"] = axis
        panel["frame"] = frame_cfg
        panel["background"] = background_cfg
        panel["legend"] = legend
        panel["layers"] = layers_cfg
    return panel, selection, time_label_settings


def _texts_and_labels_ui(mode_key):
    """「文字・記号」セクション: タイトル・パネルラベル・文字列・記号。"""
    # --- 文字 (タイトル + 任意位置の文字列) ---
    with st.sidebar:
        section_header(t("文字・記号"))
        with st.expander(t("タイトル"), expanded=False):
            title = st.text_input(t("図タイトル"), value="", placeholder=t("(なし)"),
                                   key=f"title_{mode_key}")
            title_fontsize = fontsize_input(t("タイトル文字サイズ"), f"title_fs_{mode_key}",
                                             default=12, show_auto_hint=False)
        # パネルラベル (仕様22.6)
        label_cfg = mc_config.default_panel_label()
        with st.expander(t("パネルラベル"), expanded=False):
            label_cfg["show"] = st.checkbox(
                t("パネルラベルを表示"), value=False, key=f"plabel_show_{mode_key}",
                help=t("論文の複数パネル図で使う (a), (b) などのラベル"))
            if label_cfg["show"]:
                label_cfg["text"] = st.text_input(
                    t("ラベル文字列"), value="(a)", key=f"plabel_text_{mode_key}")
                c1, c2 = st.columns(2)
                label_cfg["x"] = float(c1.number_input(
                    t("x (axes 座標, 0=左 1=右)"), -0.5, 1.5, 0.0, 0.01,
                    key=f"plabel_x_{mode_key}"))
                label_cfg["y"] = float(c2.number_input(
                    t("y (axes 座標, 0=下 1=上)"), -0.5, 1.5, 1.02, 0.01,
                    key=f"plabel_y_{mode_key}"))
                label_cfg["fontsize"] = int(st.number_input(
                    t("文字サイズ"), 4, 40, 12, key=f"plabel_fs_{mode_key}"))
                label_cfg["weight"] = ("bold" if st.checkbox(
                    t("太字"), value=True, key=f"plabel_bold_{mode_key}") else "normal")
                label_cfg["color"] = color_selector(
                    t("ラベル色"), "#000000", key=f"plabel_col_{mode_key}",
                    meta_store=label_cfg, meta_key="color")
        with st.expander(t("文字列"), expanded=False):
            texts_cfg = texts_ui(mode_key)
        with st.expander(t("記号"), expanded=False):
            markers_cfg = markers_ui(mode_key)
    return title, title_fontsize, label_cfg, texts_cfg, markers_cfg


_MODE_PANEL_UI = {"map": _map_mode_ui, "vsec": _vsec_mode_ui, "tsec": _tsec_mode_ui,
                  "line": _line_mode_ui, "dist": _dist_mode_ui, "agg": _agg_mode_ui,
                  "scatter": _scatter_mode_ui}


def _panel_editor_ui(pid: int, datasets, ds, roles, mode_options):
    """選択中パネルの編集 UI をサイドバーに描画し、panel 設定 dict を返す。

    widget key はすべて mode_key (= モード略号 + パネル id、例 "map0") を含み、
    パネル毎に独立した状態を持つ。返り値の panel は JSON 直列化可能な dict
    (呼び出し側が `panel_cfg_{pid}` にキャッシュし、非選択パネルの設定として
    使う)。共通カラーバーの UI は「図全体の書式」の一番下にある (図全体の設定)。

    モード別の本体は `_*_mode_ui(datasets, ds, roles, mode_key)` で、いずれも
    `(panel, selection, time_label_settings)` を返す。widget の並び順 = 関数の
    呼び出し順なので、分割・移動するときは順序を変えないこと (セッション互換)。
    """
    with st.sidebar:
        section_header(t("各パネルの描画モード"))
        mode = st.selectbox(t("描画モード"), mode_options,
                            format_func=tr_labels(_MODE_LABELS).get,
                            key=f"plot_mode_{pid}")
        mode_key = f"{mode}{pid}"
    box_aspect = _plot_size_ui(mode, mode_key)
    panel, selection, time_label_settings = _MODE_PANEL_UI[mode](datasets, ds, roles, mode_key)
    title, title_fontsize, label_cfg, texts_cfg, markers_cfg = _texts_and_labels_ui(mode_key)
    if time_label_settings["show"] and roles["time"] in selection:
        panel["time_label"] = {
            "show": True,
            "text": str(selection[roles["time"]]),
            "loc": time_label_settings["loc"],
            "format": time_label_settings["format"],
        }
    panel["title"] = title or None
    panel["title_fontsize"] = title_fontsize
    panel["texts"] = texts_cfg

    panel["box_aspect"] = box_aspect
    panel["label"] = label_cfg
    panel["markers"] = markers_cfg
    return panel


# --- 選択中パネルの編集 + 全パネルの設定を組み立て ---
panel_edited = _panel_editor_ui(edit_pid, datasets, ds, roles, mode_options)
# 全パネル共通モード: cfg キャッシュを上書きする前に差分を全パネルへ伝播する
# (差分計算に前回の panel_cfg_{edit_pid} を使うため、この順序が必要)
if st.session_state.get("panel_edit_all"):
    _propagate_common_edits(edit_pid, panel_edited)
else:
    st.session_state.pop("_common_ws_snapshot", None)
st.session_state[f"panel_cfg_{edit_pid}"] = panel_edited
# 今のモードの個別設定を控える (共通編集でモードを行き来したときに各パネル自身の値を使う)
_remember_individual_cfg(edit_pid, panel_edited)



panels_cfg = []
for _p in st.session_state["panels"]:
    _pid = _p["id"]
    _cfg = (panel_edited if _pid == edit_pid
            else st.session_state.get(f"panel_cfg_{_pid}"))
    if _cfg is None:
        _idx = [q["id"] for q in st.session_state["panels"]].index(_pid) + 1
        st.warning(t("パネル {n} は未設定のため図から除外します。"
                   "サイドバーの「編集するパネル」で選択して設定してください。",
                   n=_idx))
        continue
    # panel_id = セッションのパネル ID (地図の「断面の経路」が参照する。キャッシュした
    # panel_cfg は触らず、図に渡すコピーにだけ付ける)
    panels_cfg.append({**_cfg, "panel_id": str(_pid)})
if not panels_cfg:
    st.info(t("設定済みのパネルがありません。"))
    st.stop()
if grid_nrows * grid_ncols < len(panels_cfg):
    st.stop()  # エラーはパネル構成セクションで表示済み
# mosaic はパネル全数に対して検証済み。未設定パネルが除外されて数が合わなく
# なった場合はラベル数不一致で描けないため、行優先にフォールバックする
if layout_cfg.get("mosaic") and len(panels_cfg) != len(st.session_state["panels"]):
    st.warning(t("未設定のパネルがあるため mosaic 配置を無視して行優先で配置します。"))
    layout_cfg["mosaic"] = None


# --- 描画 ---
figure_config = mc_config.default_figure_config()
figure_config["figure"]["figsize"] = [fig_w, fig_h]
figure_config["figure"]["font_family"] = font_family
figure_config["figure"]["layout"] = layout_cfg
figure_config["figure"]["shared_colorbar"] = scbar_cfg
figure_config["panels"] = panels_cfg

# --- 軸座標の整合性チェック (4.3 Phase 2) ---
# dim 名は読み込み直後 (Phase 1) に整列済み。ここではパネル毎に描画軸を検査する。
# 座標役割の欠落は描画不可なのでブロック。値の不一致 (= 解像度・範囲が異なる) は
# ブロックせず情報通知のみ: 各レイヤーは各自の格子で独立に描かれ、内挿なしに重ねられる。
_grid_diff_notes = []
for _panel in panels_cfg:
    if not _panel.get("layers"):
        continue
    # ベクトル系は y 成分だけ別 dataset を参照できるので layer["dataset_id"]
    # だけでなく layer_dataset_ids で集める
    used_ids_set = {dsid for layer in _panel["layers"]
                    for dsid in mc_render.layer_dataset_ids(layer)}
    used_ids_ordered = [item["id"] for item in st.session_state["datasets"]
                         if item["id"] in used_ids_set]
    if len(used_ids_ordered) > 1:
        # 描画軸の役割を決定する
        if _panel["plot_type"] == "horizontal_map":
            axis_roles = ["lat", "lon"]
        else:
            canon_ds = datasets[used_ids_ordered[0]]
            canon_roles = mc_dataset.detect_coord_roles(canon_ds)
            dim_to_role = {v: k for k, v in canon_roles.items() if v}
            axis_roles = [r for r in (dim_to_role.get(_panel.get("x_dim")),
                                        dim_to_role.get(_panel.get("y_dim"))) if r]
        _, _, _notes, err_msgs = mc_dataset.align_dim_names(
            datasets, used_ids_ordered, axis_roles=axis_roles)
        if err_msgs:
            for msg in err_msgs:
                st.error(msg)
            st.stop()
        _grid_diff_notes.extend(_notes)
# 格子差の通知は重複排除して一度だけ表示 (順序維持)
for _note in dict.fromkeys(_grid_diff_notes):
    st.info(_note)

# 図中の CJK 文字とフォントの不整合を事前に警告 (i18n 第6段階)
_font_advice = cjk_font_advice(figure_config)
if _font_advice:
    st.warning(_font_advice)

# 描画には冒頭で構築した `datasets` (全ファイル、rename 後) をそのまま使う
try:
    fig = mc_render.render_figure(figure_config, datasets)
except Exception as exc:
    st.error(t("描画に失敗しました: {exc}", exc=_error_text(exc)))
    st.stop()

anim_req = st.session_state.pop("_anim_request", None)
if anim_req and anim_req[0] == "play":
    _, anim_time_values, anim_fps, _, anim_centers, _, anim_hold = anim_req
    _n_frames = mc_render.animation_frame_count(anim_time_values, anim_centers, anim_hold)
    placeholder = st.empty()
    placeholder.pyplot(fig)
    progress = st.progress(0, text=t("アニメーション再生中..."))
    _anim_done = 0
    try:
        # iter_frames が全パネルの時刻を同期して送る (時刻が描画軸のパネルは除く)。
        # centers (地球回転) は Orthographic のパネルの投影中心を送る
        for i, fig_frame in enumerate(mc_render.iter_frames(
                figure_config, datasets, time_values=anim_time_values,
                centers=anim_centers, frames_per_time=anim_hold)):
            placeholder.pyplot(fig_frame)
            plt.close(fig_frame)
            _anim_done = i + 1
            progress.progress(_anim_done / _n_frames)
            time_mod.sleep(1.0 / anim_fps)
    except Exception as exc:
        st.error(t("アニメーション再生に失敗: {detail}",
                   detail=_anim_error_text(exc, _anim_done, _n_frames, anim_centers)))
    progress.empty()
else:
    st.pyplot(fig)

# 図に適用した処理 (値の変換・範囲平均) を図の直下の枠に出す。該当が無ければ
# 枠ごと出さない。項目は設定から集める (描画には触らない) ので、アニメーション
# のコマでも変わらない。docs/scientific_safeguard_plan.md A-1 の第 1 段階
try:
    _notes = mc_notes.collect_notes(figure_config, datasets)
except Exception as exc:  # 通知側の不具合で図の表示を止めない (黙って消さず警告する)
    _notes = []
    st.warning(t("適用した処理の要約を作れませんでした: {exc}", exc=_error_text(exc)))
if _notes:
    with st.container(border=True):
        st.markdown(format_notes(_notes))

# 「生成されるスクリプトを表示」は図の直下に出す。スクリプト本文は下の
# 「再現スクリプト」列の widget 値に依存するため、ここでは場所 (コンテナ)
# だけ確保し、スクリプト生成後に流し込む
_script_view = st.container()

if anim_req and anim_req[0] == "file":
    _, anim_time_values, anim_fps, anim_fmt, anim_centers, anim_dpi, anim_hold = anim_req
    progress = st.progress(0, text=t("{fmt}生成中...", fmt=anim_fmt.upper()))
    _n_frames = mc_render.animation_frame_count(anim_time_values, anim_centers, anim_hold)
    _anim_done = 0

    def _anim_progress(i, n):
        global _anim_done
        _anim_done = i
        progress.progress(i / n)

    try:
        if anim_fmt == "gif":
            buf = io.BytesIO()
            mc_render.save_animation_gif(
                figure_config, datasets, anim_time_values, buf,
                fps=anim_fps, dpi=anim_dpi, centers=anim_centers,
                frames_per_time=anim_hold,
                progress=_anim_progress)
            st.session_state["_anim_file"] = (buf.getvalue(), "animation.gif", "image/gif")
        else:  # mp4
            import tempfile as _tf
            with _tf.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name
            try:
                mc_render.save_animation_mp4(
                    figure_config, datasets, anim_time_values, tmp_path,
                    fps=anim_fps, dpi=anim_dpi, centers=anim_centers,
                    frames_per_time=anim_hold,
                    progress=_anim_progress)
                with open(tmp_path, "rb") as f:
                    st.session_state["_anim_file"] = (f.read(), "animation.mp4", "video/mp4")
            finally:
                os.unlink(tmp_path)
    except Exception as exc:
        st.error(t("アニメーション生成に失敗: {detail}",
                   detail=_anim_error_text(exc, _anim_done, _n_frames, anim_centers)))
    progress.empty()
if "_anim_file" in st.session_state:
    data, fname, mime = st.session_state["_anim_file"]
    st.download_button(t("アニメーション {fmt} をダウンロード",
                         fmt=fname.split(".")[-1].upper()),
                       data=data, file_name=fname, mime=mime, key="anim_file_dl")

if anim_req and anim_req[0] == "script":
    _, anim_time_values, anim_fps, anim_fmt, anim_centers, anim_dpi, anim_hold = anim_req
    try:
        out_name = f"animation.{anim_fmt}"
        # 「netCDFパスの形式」は図の下の「再現スクリプト」にあり、この時点ではまだ描画前
        # なので session_state から読む (既定 = 絶対パス)
        anim_paths = _script_dataset_paths(
            st.session_state.get("script_path_style", "absolute"))
        anim_script = mc_scriptgen.generate_animation_script(
            figure_config, datasets, anim_paths,
            time_values=anim_time_values, centers=anim_centers,
            frames_per_time=anim_hold, fps=anim_fps,
            output_path=out_name, frame_dpi=anim_dpi, format=anim_fmt,
            dataset_renames=dataset_renames)
        st.session_state["_anim_script"] = anim_script
    except Exception as exc:
        st.error(t("アニメーション再現スクリプトの生成に失敗: {exc}", exc=_error_text(exc)))
if "_anim_script" in st.session_state:
    st.download_button(t("アニメーション再現スクリプトをダウンロード"),
                       data=st.session_state["_anim_script"],
                       file_name="animation.py", mime="text/x-python",
                       key="anim_script_dl")

# --- 出力 (仕様23章: PNG / EPS / PDF / SVG) ---
_OUT_FORMATS = {
    "PNG": ("png", "image/png"),
    "JPG": ("jpg", "image/jpeg"),
    "TIFF": ("tiff", "image/tiff"),
    "SVG": ("svg", "image/svg+xml"),
    "PDF": ("pdf", "application/pdf"),
    "EPS": ("eps", "application/postscript"),
}
# 透明背景に非対応な形式 (選択時に注意を出し、透明選択でも背景色扱いにする)
_NO_TRANSPARENCY_FORMATS = {"JPG", "EPS"}
col_png, col_script = st.columns(2)

with col_png:
    st.markdown(f"#### {t('画像出力')}")
    out_fmt_label = st.selectbox(t("出力形式"), list(_OUT_FORMATS), key="out_format")
    out_ext, out_mime = _OUT_FORMATS[out_fmt_label]
    if out_fmt_label == "EPS":
        st.caption(t("⚠ EPS は透明度や一部の描画要素 (半透明の塗りつぶし等) が"
                   "完全には再現されない場合があります。"))
    elif out_fmt_label == "JPG":
        st.caption(t("⚠ JPG は非可逆圧縮 (輪郭がにじむ) で透明背景に非対応。"
                   "論文・印刷には PNG / TIFF / PDF を推奨します。"))
    elif out_fmt_label == "TIFF":
        st.caption(t("TIFF は無圧縮のためファイルが大きくなります (透明背景に対応)。"))
    dpi = st.selectbox(t("解像度 (dpi)"), [150, 300, 600], index=1, key="png_dpi",
                       help=t("ベクター形式 (SVG/PDF/EPS) ではラスタライズされる要素にのみ影響"))
    png_name = st.text_input(t("出力ファイル名"), value="figure.png", key="png_name")
    out_name = os.path.splitext(png_name)[0] + f".{out_ext}"
    # 背景は「白 / 色を指定 / 透明」の三者択一 (プロット枠の外側の余白に効く。
    # 枠の内側 = 地図・軸の領域は各パネルの塗り・地図・背景設定が決める)
    out_bg_mode = st.radio(t("背景"), list(_OUT_BG_LABELS),
                           format_func=tr_labels(_OUT_BG_LABELS).get,
                           horizontal=True, key="out_bg_mode",
                           help=t("画像全体 (プロット枠の外側の余白) の背景。"
                                "この画像出力にのみ効き、プレビューには反映されません。"
                                "JPG / EPS は透明背景に非対応"))
    out_transparent = out_bg_mode == "transparent"
    out_facecolor = None
    if out_bg_mode == "color":
        out_facecolor = color_selector(t("背景色"), "#ffffff", key="out_bg_color")
    # 透明非対応の形式で透明を選んだ場合は白背景で保存する (誤解防止)
    if out_transparent and out_fmt_label in _NO_TRANSPARENCY_FORMATS:
        st.warning(t("{fmt} は透明背景に非対応のため、白背景で出力します。",
                   fmt=out_fmt_label))
        out_transparent = False
    out_tight = st.checkbox(t('余白を切り詰める (bbox_inches="tight")'), value=True,
                            key="out_tight")
    _save_kwargs = {"format": out_ext, "dpi": int(dpi)}
    if out_tight:
        _save_kwargs["bbox_inches"] = "tight"
    if out_transparent:
        _save_kwargs["transparent"] = True
    elif out_facecolor is not None:
        _save_kwargs["facecolor"] = out_facecolor
    buf = io.BytesIO()
    fig.savefig(buf, **_save_kwargs)
    st.download_button(t("{fmt}をダウンロード", fmt=out_fmt_label), data=buf.getvalue(),
                       file_name=out_name, mime=out_mime)

with col_script:
    st.markdown(f"#### {t('再現スクリプト')}")
    path_style = st.radio(t("netCDFパスの形式"), list(_PATH_STYLE_LABELS),
                          format_func=tr_labels(_PATH_STYLE_LABELS).get, horizontal=True,
                          key="script_path_style")
    include_save = st.checkbox(t("savefig を含める"), value=True, key="script_include_save")
    include_show = st.checkbox(t("plt.show() を含める"), value=True, key="script_include_show")
    ds_paths_for_script = _script_dataset_paths(path_style)
    # 描画と同じ設定の検査 (経路断面の点数・地形マスクの単位など) を scriptgen も行う。
    # 通常は描画側で先に止まるが、描画が通ってスクリプトだけ失敗しても落とさない
    try:
        script = mc_scriptgen.generate_script(
            figure_config, datasets, ds_paths_for_script,
            figure_output=out_name, figure_dpi=int(dpi),
            include_save=include_save, include_show=include_show,
            figure_transparent=out_transparent, figure_tight=out_tight,
            figure_facecolor=out_facecolor, dataset_renames=dataset_renames)
    except mc_render.RenderError as exc:
        st.error(t("再現スクリプトを生成できません: {exc}", exc=_error_text(exc)))
        st.stop()
    st.download_button(t("スクリプトをダウンロード"), data=script,
                       file_name=os.path.splitext(out_name)[0] + ".py",
                       mime="text/x-python")

with _script_view.expander(t("生成されるスクリプトを表示"), expanded=False):
    st.code(script, language="python")

