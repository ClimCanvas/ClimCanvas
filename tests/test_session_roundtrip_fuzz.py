# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""WIP ラウンドトリップのファズテスト (widget 自動列挙 + パラメタを自動で振る)。

モードを切り替えた直後の UI に**実際に表示されている入力 widget を全て自動列挙**
してプールとし (手書きの一覧は持たない)、固定シードの乱数で選んで
「今と異なる有効値」に変更 → WIP 保存 → 新セッションで復元 → 検証する。

ケースは3種類 (seed で決まる):
- **normal** (seed 0-14): 1パネル。追加ボタン (レイヤー・塗り範囲・文字列・
  記号・box) を 0〜2 回クリックしてから、プールの widget を 4〜12 個操作
- **full** (seed 15-19): 1パネル。追加ボタンを押した上で**プール全数**を操作
  (個々の widget の復元を毎回全数検証する)
- **multipanel** (seed 20-24): 2パネル (モード混在)。両パネルで widget を操作し、
  パネル一覧・グリッド・選択中パネル・両パネルの設定の復元を検証

検証は2段:
1. 操作した各 widget の session_state 値が復元後も一致する
2. パネル設定 panel_cfg_{pid} が保存前と完全一致する

widget を新設するとプールに自動で入るので、一覧の保守は不要。

- 既定はシード固定なので毎回同じ組合せ (再現可能)。環境変数 **CC_FUZZ_SEED**
  を設定すると別の組合せを探索できる:
      CC_FUZZ_SEED=$(date +%s) pytest tests/test_session_roundtrip_fuzz.py
  失敗メッセージに再現用のシード値が出る。探索を常設で広げるときは
  _N_NORMAL / _N_FULL / _N_MULTI を増やす
- 失敗時は「モード + 適用した操作一覧」を stdout に出す (そのまま再現手順)

AppTest の制約: 要素は表示文字列 (format_func 適用後) しか持たないため、
表示名 ≠ 生値の selectbox 等は事前フィルタ (_drivable) とプローブ
(適用 → 実行 → 例外なら差し戻し) の二段で除外する。
"""

from __future__ import annotations

import json
import os
import random
import zlib

import pytest

from climcanvas.ui.state_io import _is_session_excluded
from test_consistency import _make_series_nc
from test_ui_config_mapping import _button, _load_app

_N_NORMAL = 15
_N_FULL = 5
_N_MULTI = 5
_N_CASES = _N_NORMAL + _N_FULL + _N_MULTI
_DEFAULT_SEED_BASE = 3000

_MODES = ["map", "vsec", "tsec", "line",
          "scatter", "dist", "agg"]
# 集計系モードは lat/lon 次元を持たない変数が必要 (sample_atmos には無い) ため、
# 専用の時系列サンプル (_make_series_nc) を読み込んで入る
_SERIES_MODES = {"dist", "agg"}
# multipanel ケースはパネル1が水平断面図固定 (= sample_atmos が必要)。
# モード選択肢は ds0 から決まるため、集計系とは同一セッションで混在できない
_MODES_ATMOS = _MODES[:5]
_INPUT_KINDS = ("checkbox", "radio", "selectbox", "text_input",
                "number_input", "slider", "select_slider")
# テスト側で制御する・触ると前提が崩れる widget
_SKIP_KEYS = {"plot_mode_0", "plot_mode_1"}
# リストに項目を追加する系のボタン (クリックで子 widget が既定値で現れる)。
# add_file (ファイル管理) と add_panel (パネル構成はケース側で制御) は除外
_ADD_BUTTON_PREFIXES = ("add_", "bgspan_add_", "txtadd_", "mkadd_",
                        "box_add_")
_ADD_BUTTON_SKIP = {"add_file", "add_panel"}


def _seed_base() -> int:
    """CC_FUZZ_SEED が設定されていればそれを、なければ既定の固定値を返す。

    整数以外の文字列 (日付など) も CRC32 で決定的に整数化して受け付ける。
    """
    raw = os.environ.get("CC_FUZZ_SEED", "").strip()
    if not raw:
        return _DEFAULT_SEED_BASE
    try:
        return int(raw)
    except ValueError:
        return zlib.crc32(raw.encode("utf-8"))


def _drivable(kind, el) -> bool:
    """AppTest から正しく値を変えられる可能性がある widget か (事前フィルタ)。

    AppTest の要素は表示文字列 (format_func 適用後) しか持たないため、
    表示名 ≠ 生値の selectbox / radio (変数選択・色・時刻インデックス等) は
    外から駆動すると生値でない文字列が session_state に入り壊れる。
    「現在の生値が表示 options に含まれる文字列」であるものだけを対象にする。
    範囲 select_slider も生値が文字列 (時刻ラベル等) のものだけ。

    注意: このチェックは「現在値だけ表示と一致し、他の選択肢は formatted」
    というケース (例: z変数の "(なし)") をすり抜けるため、選択系 widget は
    さらに適用時のプローブ (_tweak_widgets) で例外なら差し戻す。
    """
    if kind in ("selectbox", "radio"):
        return isinstance(el.value, str) and el.value in el.options
    if kind == "select_slider":
        if isinstance(el.value, tuple):
            return (len(el.options) >= 3
                    and all(isinstance(v, str) for v in el.value))
        return isinstance(el.value, str) and el.value in el.options
    return True


def _discover_pool(at) -> list[tuple[str, str]]:
    """現在の UI に表示されている入力 widget を (種別, key) で全列挙する。

    WIP 保存から除外されるキー (先頭 `_` のインフラ系・ボタン系) は
    復元対象外なので入れない。AppTest から駆動できない widget
    (_drivable 参照) も除外する。順序は決定的 (ソート済み)。
    """
    pool = []
    for kind in _INPUT_KINDS:
        for el in at.get(kind):
            key = getattr(el, "key", None)
            if not key or key in _SKIP_KEYS or _is_session_excluded(key):
                continue
            if not _drivable(kind, el):
                continue
            pool.append((kind, key))
    return sorted(set(pool))


def _discover_add_buttons(at) -> list[str]:
    """「追加」系ボタンの key を列挙する (レイヤー・塗り範囲・文字列など)。"""
    keys = set()
    for b in at.get("button"):
        key = getattr(b, "key", None) or ""
        if key in _ADD_BUTTON_SKIP:
            continue
        if any(key.startswith(p) for p in _ADD_BUTTON_PREFIXES):
            keys.add(key)
    return sorted(keys)


def _pick_new_value(kind, el, rng):
    """今と異なる有効値と設定メソッド名を返す。変えようがなければ None。

    駆動可否 (_drivable) は discovery 側で判定済みの前提。
    """
    if kind == "checkbox":
        return (not el.value), "set_value"
    if kind in ("selectbox", "radio"):
        opts = [o for o in el.options if o != el.value]
        if not opts:
            return None
        return rng.choice(opts), "set_value"
    if kind == "text_input":
        return (el.value or "") + "-FZ", "set_value"
    if kind == "number_input":
        step = el.step or 1
        value = el.value if el.value is not None else 0
        cand = value + step
        if el.max is not None and cand > el.max:
            cand = value - step
        if el.min is not None and cand < el.min:
            return None
        if isinstance(value, int) and float(step).is_integer():
            cand = int(cand)
        if cand == value:
            return None
        return cand, "set_value"
    if kind == "slider":
        # 現在値から遠い方の端へ動かす
        if el.min is None or el.max is None or el.min == el.max:
            return None
        target = (el.max if abs(el.value - el.min) <= abs(el.max - el.value)
                  else el.min)
        if target == el.value:
            return None
        return target, "set_value"
    if kind == "select_slider":
        opts = list(el.options)
        if isinstance(el.value, tuple):  # 範囲スライダー
            # 生値が文字列 (時刻ラベル等) のものだけ対象にする。float 生値は
            # proto 上の options (文字列) と対応が取れないため動かさない
            if len(opts) < 3 or not all(isinstance(v, str) for v in el.value):
                return None
            return (opts[1], opts[-1]), "set_range"
        alts = [o for o in opts if o != el.value]
        if not alts:
            return None
        return rng.choice(alts), "set_value"
    return None


def _apply(at, kind, key, value, method):
    el = getattr(at, kind)(key=key)
    if method == "set_range":  # set_range(lower, upper) の2引数
        el.set_range(*value)
    else:
        getattr(el, method)(value)


def _state(at, key):
    """AppTest の session_state は .get() 非対応なので添字アクセスで包む。"""
    try:
        return at.session_state[key]
    except KeyError:
        return None


def _click_add_buttons(at, rng, applied, max_clicks=2):
    """追加系ボタンをランダムに 0〜max_clicks 回クリックする。

    追加された項目 (レイヤー・塗り範囲・文字列など) は既定値のリスト要素と
    して保存・復元されるので、クリックだけで復元経路のカバレッジになる。
    """
    buttons = _discover_add_buttons(at)
    if not buttons:
        return
    n = rng.randint(0, max_clicks)
    for key in rng.sample(buttons, min(n, len(buttons))):
        _button(at, key).set_value(True)
        at.run()
        assert not at.exception, f"{key} クリックで例外"
        applied.append(["button", key, "(クリック)"])


def _tweak_widgets(at, rng, applied, n_tweaks=None):
    """プールから widget を選んで操作する。n_tweaks=None は全数 (full)。

    操作した (スキップ含む) キーのリストを返す (復元検証に使う)。
    """
    pool = _discover_pool(at)
    assert len(pool) >= 20, f"プールが小さすぎる ({len(pool)}): 列挙に失敗?"
    if n_tweaks is None:
        chosen = list(pool)  # 全数 (決定的な順序)
    else:
        chosen = rng.sample(pool, min(n_tweaks, len(pool)))
    # 選択系 (表示名 ≠ 生値の可能性が残る) は1個ずつ適用してプローブし、
    # 例外が出たらハーネス限界とみなして元の値に差し戻す。
    # チェック・数値・テキスト系は生値で駆動できるので一括適用し、
    # そこで出る例外は本物のバグとして扱う (差し戻さない)。
    probe_kinds = {"selectbox", "radio", "select_slider"}
    for kind, key in [c for c in chosen if c[0] in probe_kinds]:
        try:
            el = getattr(at, kind)(key=key)
        except KeyError:
            # 先行の構造変更 (プロット軸・断面の種類など) で消えた widget
            applied.append([kind, key, "(先行の変更で消えたためスキップ)"])
            continue
        picked = _pick_new_value(kind, el, rng)
        if picked is None:
            applied.append([kind, key, "(変更不可のためスキップ)"])
            continue
        original = el.value
        value, method = picked
        _apply(at, kind, key, value, method)
        # 表示名 ≠ 生値の widget に表示文字列を入れると、例外は
        # 「記録」(at.exception) ではなく at.run() から送出されることも
        # あるため両方を失敗として扱う
        try:
            at.run()
            probe_failed = bool(at.exception)
        except Exception:
            probe_failed = True
        if probe_failed:
            _apply(at, kind, key,
                   original if method != "set_range" else tuple(original),
                   method)
            at.run()
            assert not at.exception, \
                f"{key} の差し戻しにも失敗: {at.exception[0].message}"
            applied.append([kind, key, "(表示名≠生値のため差し戻し)"])
        else:
            applied.append([kind, key, repr(value)])
    for kind, key in [c for c in chosen if c[0] not in probe_kinds]:
        try:
            el = getattr(at, kind)(key=key)
        except KeyError:
            applied.append([kind, key, "(先行の変更で消えたためスキップ)"])
            continue
        picked = _pick_new_value(kind, el, rng)
        if picked is None:
            applied.append([kind, key, "(変更不可のためスキップ)"])
            continue
        value, method = picked
        _apply(at, kind, key, value, method)
        applied.append([kind, key, repr(value)])
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    return [key for _, key in chosen]


def _save_wip(at, name="fuzz"):
    at.text_input(key="_session_save_name").set_value(name)
    at.run()
    _button(at, "_session_save_btn").set_value(True)
    at.run()
    assert not at.exception


def _restore_wip(at2, name="fuzz"):
    at2.selectbox(key="_session_restore_slot").set_value(name)
    at2.run()
    _button(at2, "_session_restore_btn").set_value(True)
    at2.run()
    assert not at2.exception, \
        at2.exception[0].message if at2.exception else ""
    at2.run()  # 復元後の再構築
    assert not at2.exception


def _verify_roundtrip(at, at2, touched_keys, cfg_keys):
    """widget 状態と panel 設定 (+パネル構成) の復元一致を検証する。"""
    state_before = {key: _state(at, key) for key in touched_keys}
    for key in touched_keys:
        assert _state(at2, key) == state_before[key], \
            f"widget 状態が復元されていない: {key}"
    for ck in cfg_keys:
        assert _state(at2, ck) == _state(at, ck), \
            f"{ck} が復元後に一致しない"


@pytest.fixture(autouse=True)
def _stub_render(monkeypatch):
    """描画を空図に差し替え、UI 状態の保存・復元だけを高速に検証する。

    このテストの対象は widget 状態と panel_cfg の構築・保存・復元であり、
    figure_config → 描画の正しさは test_consistency / test_fuzz_configs が
    担う。matplotlib 描画 (1 run あたり 1〜2 秒) を省くと全体が数倍速くなる。
    """
    import matplotlib.pyplot as plt

    from climcanvas.core import render as mc_render

    def _tiny_figure(figure_config, datasets, **kwargs):
        plt.close("all")  # スタブ図の蓄積を防ぐ
        return plt.figure(figsize=(1, 1))

    monkeypatch.setattr(mc_render, "render_figure", _tiny_figure)


def _case_kind(seed: int) -> str:
    if seed < _N_NORMAL:
        return "normal"
    if seed < _N_NORMAL + _N_FULL:
        return "full"
    return "multipanel"


@pytest.mark.parametrize("seed", range(_N_CASES),
                         ids=[f"wipfuzz-{i}" for i in range(_N_CASES)])
def test_session_roundtrip_fuzz(seed, sample_path, tmp_path, monkeypatch):
    base = _seed_base()
    rng = random.Random(base + seed)
    kind = _case_kind(seed)
    mode = _MODES[seed % len(_MODES)]
    monkeypatch.setenv("CC_SESSION_DIRS", str(tmp_path))

    applied = []
    try:
        if kind in ("normal", "full") and mode in _SERIES_MODES:
            # 集計系モードは非地理変数が必要なので専用サンプルに切り替える
            series_path = str(tmp_path / "series_fuzz.nc")
            _make_series_nc(series_path)
            at = _load_app(series_path)
        else:
            at = _load_app(sample_path)
        touched = []

        if kind in ("normal", "full"):
            if mode != "map":
                at.selectbox(key="plot_mode_0").set_value(mode)
                at.run()
                assert not at.exception
            _click_add_buttons(at, rng, applied,
                               max_clicks=2 if kind == "normal" else 3)
            n_tweaks = rng.randint(4, 12) if kind == "normal" else None
            touched += _tweak_widgets(at, rng, applied, n_tweaks)
            cfg_keys = ["panel_cfg_0"]
        else:  # multipanel: 2パネル (モード混在) で両方を操作
            mode_b = _MODES_ATMOS[(seed + 2) % len(_MODES_ATMOS)]
            at.number_input(key="grid_ncols").set_value(2)
            at.run()
            _button(at, "add_panel").set_value(True)
            at.run()
            assert not at.exception
            if mode_b != "map":
                at.selectbox(key="plot_mode_1").set_value(mode_b)
                at.run()
                assert not at.exception
            applied.append(["case", "panel1", f"mode={mode_b}"])
            _click_add_buttons(at, rng, applied, max_clicks=1)
            touched += _tweak_widgets(at, rng, applied, rng.randint(3, 8))
            _button(at, "_panel_sel_0").set_value(True)
            at.run()
            assert not at.exception
            applied.append(["case", "panel0", "mode=水平断面図"])
            _click_add_buttons(at, rng, applied, max_clicks=1)
            touched += _tweak_widgets(at, rng, applied, rng.randint(3, 8))
            cfg_keys = ["panel_cfg_0", "panel_cfg_1", "panels",
                        "grid_ncols", "panel_edit"]

        _save_wip(at)
        assert (tmp_path / "fuzz.json").is_file()

        # セッション B: まっさらな状態から復元して検証
        at2 = _load_app(sample_path)
        _restore_wip(at2)
        _verify_roundtrip(at, at2, touched, cfg_keys)
    except Exception:
        print(f"--- failing WIP fuzz case: kind={kind} mode={mode} "
              f"seed={seed} ---")
        print(f"再現するには: CC_FUZZ_SEED={base} pytest "
              f"'tests/test_session_roundtrip_fuzz.py::test_session_roundtrip_fuzz"
              f"[wipfuzz-{seed}]'")
        print(json.dumps(applied, ensure_ascii=False, indent=1))
        raise
