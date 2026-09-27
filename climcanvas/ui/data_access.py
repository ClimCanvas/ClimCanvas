# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""netCDF ファイルの読み込み・ファイル選択ダイアログ・許可ディレクトリ制御。"""

import os
import subprocess
import sys

import streamlit as st

from climcanvas.core import dataset as mc_dataset
from climcanvas.ui.i18n import t


@st.cache_resource(show_spinner="netCDFファイルを読み込んでいます...")
def load_dataset(path: str, mtime: float):
    return mc_dataset.open_dataset(path)


@st.cache_resource(show_spinner="座標ファイルを結び付けています...")
def load_dataset_with_coords(path: str, mtime: float,
                             coord_paths: tuple[str, ...],
                             coord_mtimes: tuple[float, ...]):
    """本体を読み、座標ファイル (経緯度が別ファイル) の lon/lat を座標として結び付ける。

    mtime 群はキャッシュキー (ファイル更新で読み直す)。結び付けの本体は
    dataset.attach_coord_files (scriptgen が同じ行を出す)。
    """
    ds = load_dataset(path, mtime)
    coord_files = [(p, mc_dataset.open_coord_file(p)) for p in coord_paths]
    return mc_dataset.attach_coord_files(ds, coord_files)


_FILE_DIALOG_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', True)
picked = filedialog.askopenfilename(
    parent=root,
    title=sys.argv[1],
    initialdir=sys.argv[2],
    filetypes=[('netCDF', '*.nc *.nc4 *.netcdf *.cdf'), ('All files', '*.*')],
)
root.destroy()
sys.stdout.write(picked or '')
"""


@st.cache_resource(show_spinner=False)
def detect_grid_projection_cached(path: str, mtime: float,
                                  coord_paths: tuple[str, ...],
                                  coord_mtimes: tuple[float, ...]):
    """2 次元座標格子の投影法の推定 (dataset.detect_grid_projection) をファイル単位で
    キャッシュする。座標ファイル付きなら結び付けた dataset で判定する。"""
    if coord_paths:
        ds = load_dataset_with_coords(path, mtime, coord_paths, coord_mtimes)
    else:
        ds = load_dataset(path, mtime)
    return mc_dataset.detect_grid_projection(ds)


def pick_file_dialog(initial_path: str | None = None) -> str | None:
    """別プロセスで tkinter のネイティブダイアログを開いて netCDF ファイルを選ぶ。

    実装上の罠: Streamlit のコールバックは worker スレッドで走るため、tkinter を
    同プロセスで起動すると macOS で `NSWindow should only be instantiated on the
    main thread` でプロセスごと abort する。別プロセスを起動して自分の main
    thread で NSWindow を作らせ、選択結果を標準出力で受け取る方式に倒した。
    Windows/Linux でも同じコードパスで動く。

    OS依存性:
      - 動く: macOS / Windows / デスクトップ Linux (X 越し含む)
      - 動かない: ヘッドレス環境 (X無し SSH・Docker・クラウド配信)。tkinter が
        ウィンドウを作れず subprocess が非ゼロ終了 → ここでは None を返して
        テキスト入力にフォールバックする
      - 要件: Python の tkinter (macOS/Windows は標準同梱。Linux は `python3-tk`
        等を別途入れる必要がある場合あり)
    """
    initial_dir = (os.path.dirname(initial_path) if initial_path else "") or os.getcwd()
    try:
        result = subprocess.run(
            [sys.executable, "-c", _FILE_DIALOG_SCRIPT,
             t("netCDFファイルを選択"), initial_dir],
            capture_output=True, text=True, timeout=600,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def read_user_config() -> dict:
    """ユーザー設定 `~/.climcanvas/config.toml` を dict で返す (無い・壊れていれば {})。

    allowed_dirs / mode (data_access) と session_dirs (state_io) が共用する。
    """
    try:
        cfg_path = os.path.expanduser("~/.climcanvas/config.toml")
        if os.path.isfile(cfg_path):
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            with open(cfg_path, "rb") as f:
                return tomllib.load(f) or {}
    except Exception:
        pass
    return {}


def expand_user_paths(items) -> list[str]:
    """パスの列を `~` 展開 + 絶対パス化して返す。"""
    return [os.path.abspath(os.path.expanduser(p)) for p in items]


def split_env_paths(raw: str) -> list[str]:
    """`:` 区切りの環境変数値をパスのリストにする (空要素は捨てる)。"""
    return expand_user_paths(p.strip() for p in raw.split(":") if p.strip())


def _load_allowed_dirs() -> list[str]:
    """許可ディレクトリの一覧を返す。空リストなら制限なし (ローカル運用)。

    優先順:
      1. 環境変数 CC_ALLOWED_DIRS (`:` 区切りの絶対パス)。
         **設定されていれば空でも優先** — 空文字列は「明示的に制限なし」を
         意味し、config.toml を無視する (テストの隔離にも使う)
      2. ~/.climcanvas/config.toml の allowed_dirs = ["..."]
      3. どちらもなければ空リスト (= 制限なし)
    """
    raw = os.environ.get("CC_ALLOWED_DIRS")
    if raw is not None:
        return split_env_paths(raw)
    return expand_user_paths(read_user_config().get("allowed_dirs") or [])


def app_mode() -> str:
    """アプリの運用形態 "local" / "remote" を返す。**明示設定のみ**で判定する
    (接続元 IP 等からの推定はしない — SSH トンネル構成では原理的に判別できないため)。

    優先順:
      1. 環境変数 CC_MODE ("remote" / それ以外は local。設定されていれば空でも優先 —
         テストの隔離にも使う)
      2. ~/.climcanvas/config.toml の mode = "remote"
      3. どちらも無ければ "local"
    """
    raw = os.environ.get("CC_MODE")
    if raw is not None:
        return "remote" if raw.strip().lower() == "remote" else "local"
    if str(read_user_config().get("mode", "")).strip().lower() == "remote":
        return "remote"
    return "local"


def _is_path_allowed(path: str, allowed_dirs: list[str]) -> bool:
    """path が allowed_dirs のいずれかの配下にあるかを返す。

    allowed_dirs が空ならローカル運用扱いで常に True (制限なし)。
    シンボリックリンクは realpath で解決して比較する。
    """
    if not allowed_dirs:
        return True
    try:
        real = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    except OSError:
        return False
    for d in allowed_dirs:
        try:
            real_d = os.path.realpath(d)
        except OSError:
            continue
        # commonpath で配下判定 (real_d == real or real は real_d の subpath)
        try:
            if os.path.commonpath([real, real_d]) == real_d:
                return True
        except ValueError:
            continue  # drive 違いなど (Windows)
    return False


def _browse_file_callback():
    picked = pick_file_dialog(st.session_state.get("_new_file_path"))
    if picked:
        st.session_state["_new_file_path"] = picked


def browse_into_callback(key: str) -> None:
    """ネイティブダイアログで選んだパスを session_state[key] に入れる (座標ファイル欄用)。"""
    picked = pick_file_dialog(st.session_state.get(key))
    if picked:
        st.session_state[key] = picked


def _add_dataset_to_state(path: str) -> None:
    """session_state["datasets"] に新しいエントリを追加する (id は末尾に連番)。"""
    items = st.session_state.setdefault("datasets", [])
    items.append({"id": f"ds{len(items)}", "path": path})


def _remove_dataset_from_state(index: int) -> None:
    """指定 index のエントリを削除して残りの id を ds0 から振り直す。"""
    items = st.session_state["datasets"]
    items.pop(index)
    for i, item in enumerate(items):
        item["id"] = f"ds{i}"
