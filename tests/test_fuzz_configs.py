# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""ランダム生成した figure_config での render↔scriptgen 画像一致 (ファズテスト)。

test_consistency.py の手書き設定は「機能を狙い撃ち」だが、機能同士の組み合わせは
網羅できない (例: pcolormesh × レベル直接指定は実際に組み合わせの穴だった)。
ここではスキーマの範囲内でランダムな設定を**固定シード**で生成し、アプリ内描画と
生成スクリプトのピクセル一致 (test_consistency の共通ヘルパ) を検証する。

- 既定はシード固定なので実行のたびに同じ設定 (再現可能)。環境変数
  **CC_FUZZ_SEED** を設定すると別の組合せを探索できる
  (test_session_roundtrip_fuzz.py と同じ仕組み):
      CC_FUZZ_SEED=$(date +%s) pytest tests/test_fuzz_configs.py
  失敗メッセージに再現用のシード値が出る。探索を常設で広げるときは
  _N_CASES を増やす
- 失敗時は落ちた設定 JSON を stdout に出す (pytest が表示する)
- 見た目の妥当性は問わない (目盛位置が描画範囲外でも、両者が一致すればよい)
"""

from __future__ import annotations

import json
import os
import random
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render
from test_consistency import (_assert_app_and_script_match,
                              _assert_multi_dataset_match, _make_series_nc)

_N_CASES = 12
# 集計系 (agg_2d / dist_1d) は非地理変数が必要なので、_make_series_nc の
# 時系列サンプルを使う別パラメタライズで回す
_N_SERIES_CASES = 6
_DEFAULT_SEED_BASE = 1000


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


def _rand_axis_common(rng: random.Random, axis: dict):
    """1D/2D/断面図共通の axis 項目をランダムに埋める。"""
    axis["invert_x"] = rng.random() < 0.4
    axis["invert_y"] = rng.random() < 0.4
    if rng.random() < 0.5:
        axis["label_fontsize"] = rng.choice([8, 12])
    if rng.random() < 0.4:
        axis["label_color"] = rng.choice(["#003366", "#990000"])
        axis["label_weight"] = rng.choice(["normal", "bold"])
        axis["label_italic"] = rng.random() < 0.5
    if rng.random() < 0.3:
        axis["label_pad"] = rng.choice([2.0, 8.0])
    if rng.random() < 0.3:
        axis["x_label_rotation"] = rng.choice([0.0, 20.0])
        axis["y_label_rotation"] = rng.choice([90.0, 45.0])
    if rng.random() < 0.5:
        axis["tick_fontsize"] = rng.choice([7, 11])
    if rng.random() < 0.4:
        axis["tick_width"] = rng.choice([0.5, 1.6])
    if rng.random() < 0.25:
        axis["show_x_ticklabels"] = False
    if rng.random() < 0.25:
        axis["show_y_ticklabels"] = False
    axis["show_x_minor_ticks"] = rng.random() < 0.5
    axis["show_y_minor_ticks"] = rng.random() < 0.5
    axis["grid"].update({
        "show_x": rng.random() < 0.5, "show_y": rng.random() < 0.5,
        "color": rng.choice(["gray", "#88aacc"]),
        "width": rng.choice([0.4, 1.0]),
        "linestyle": rng.choice([":", "--", "-"]),
    })


def _rand_ticks(rng: random.Random, axis: dict, prefix: str,
                lo: float, hi: float):
    """x または y の目盛指定 (自動/等間隔/位置直接) をランダムに選ぶ。"""
    mode = rng.choice(["auto", "interval", "positions"])
    if mode == "interval":
        axis[f"{prefix}_tick_interval"] = round((hi - lo) / rng.choice([3, 5]), 3)
    elif mode == "positions":
        n = rng.choice([3, 4])
        fracs = rng.sample([0.1, 0.3, 0.5, 0.7, 0.9], n)
        pos = sorted(round(lo + (hi - lo) * f, 2) for f in fracs)
        axis[f"{prefix}_tick_positions"] = pos
        if rng.random() < 0.5:
            axis[f"{prefix}_tick_labels"] = [f"L{i}" for i in range(n)]


def _rand_frame_bg(rng: random.Random, panel: dict,
                   x_range: tuple[float, float], y_range: tuple[float, float]):
    frame = panel["frame"]
    for side in ("top", "bottom", "left", "right"):
        if rng.random() < 0.25:
            frame[f"show_{side}"] = False
    if rng.random() < 0.5:
        frame["width"] = rng.choice([0.5, 1.8])
    if rng.random() < 0.4:
        frame["color"] = "#224466"
    bg = panel["background"]
    if rng.random() < 0.4:
        bg["color"] = "#f7f4ec"
    spans = []
    for _ in range(rng.choice([0, 1, 2])):
        ori = rng.choice(["x", "y"])
        lo, hi = x_range if ori == "x" else y_range
        a, b = sorted(rng.uniform(lo, hi) for _ in range(2))
        if a == b:
            continue
        spans.append({"orientation": ori, "lo": round(a, 2), "hi": round(b, 2),
                      "color": rng.choice(["#ffd7d7", "#d7e8ff"]),
                      "alpha": round(rng.uniform(0.2, 0.8), 2)})
    bg["spans"] = spans


def _rand_fill_style(rng: random.Random, style: dict, vmin: float, vmax: float):
    style["method"] = rng.choice(["contourf", "pcolormesh"])
    style["cmap"] = rng.choice(["viridis", "coolwarm", "RdBu_r"])
    style["reverse_cmap"] = rng.random() < 0.3
    style["extend"] = rng.choice(["both", "neither", "min", "max"])
    # 値の変換 (K→°C)。レベル・maskout は変換後の値で指定するので、
    # 以降の乱数は変換後の範囲で振る
    if rng.random() < 0.3:
        style["value_offset"] = -273.15
        vmin, vmax = vmin - 273.15, vmax - 273.15
    kind = rng.choice(["auto", "range", "list"])
    if kind == "range":
        style["vmin"], style["vmax"] = vmin, vmax
        style["levels"] = rng.choice([7, 13, 21])
    elif kind == "list":
        n = rng.choice([4, 6])
        fracs = sorted(rng.sample([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0], n))
        style["levels"] = [round(vmin + (vmax - vmin) * f, 2) for f in fracs]
    # 0 を含むレベル帯の白置換 (値が 0 をまたがない組合せでも no-op で安全)
    style["zero_white"] = rng.random() < 0.25
    # maskout (変換後の値)。範囲の端だけ切ってデータが必ず残るようにする
    if rng.random() < 0.25:
        style["maskout"]["below"] = round(vmin + (vmax - vmin) * 0.15, 2)
    if rng.random() < 0.25:
        style["maskout"]["above"] = round(vmin + (vmax - vmin) * 0.85, 2)
    style["colorbar"]["show"] = rng.random() < 0.8


def _rand_cmap_section_style(rng: random.Random, style: dict,
                             vmin: float, vmax: float):
    """cmap 色付けの共通スタイルをランダムに埋める (UI の cmap_section_ui と対)。

    範囲 (自動/手動) × 離散化 (なし/レベル数/直接指定) × extend × カラーバー。
    vector / stream / map_scatter など use_cmap 系レイヤーで共用する。
    """
    style["use_cmap"] = True
    style["cmap"] = rng.choice(["viridis", "plasma", "coolwarm"])
    style["reverse_cmap"] = rng.random() < 0.3
    if rng.random() < 0.5:
        style["vmin"], style["vmax"] = vmin, vmax
    disc = rng.choice(["none", "nlev", "list"])
    if disc == "nlev":
        style["levels"] = rng.choice([5, 8, 11])
    elif disc == "list":
        n = rng.choice([4, 5])
        fracs = sorted(rng.sample([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n))
        style["levels"] = [round(vmin + (vmax - vmin) * f, 2) for f in fracs]
    if disc != "none":
        style["zero_white"] = rng.random() < 0.25
    style["extend"] = rng.choice(["neither", "both", "min", "max"])
    style["colorbar"]["show"] = rng.random() < 0.8


def _rand_contour_style(rng: random.Random, style: dict,
                        vmin: float, vmax: float):
    """contour スタイル: fill 式レベル指定 + 単色/カラーマップ (+extend/cb)。"""
    kind = rng.choice(["auto", "range", "list"])
    if kind == "range":
        style["vmin"], style["vmax"] = vmin, vmax
        style["levels"] = rng.choice([7, 13, 21])
    elif kind == "list":
        n = rng.choice([4, 6])
        fracs = sorted(rng.sample([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0], n))
        style["levels"] = [round(vmin + (vmax - vmin) * f, 2) for f in fracs]
    # kind == "auto": 既定 (levels=21 の自動範囲) のまま
    style["labels"]["show"] = rng.random() < 0.5
    if rng.random() < 0.5:
        style["use_cmap"] = True
        style["cmap"] = rng.choice(["viridis", "plasma"])
        style["reverse_cmap"] = rng.random() < 0.3
        style["extend"] = rng.choice(["neither", "both", "min", "max"])
        style["colorbar"]["show"] = rng.random() < 0.8
    else:
        style["linewidth"] = rng.choice([0.8, 1.5])
        style["negative_linestyle"] = rng.choice([None, "dashed"])


def _rand_legend(rng: random.Random, panel: dict):
    panel["legend"]["show"] = rng.random() < 0.8
    if rng.random() < 0.3:
        panel["legend"]["fontsize"] = rng.choice([7, 12])
    if rng.random() < 0.3:
        panel["legend"]["x"] = 1.02
        panel["legend"]["y"] = 0.0


def _random_line_panel(rng: random.Random) -> dict:
    panel = mc_config.default_line_panel()
    x_dim, (x_lo, x_hi) = rng.choice([
        ("lon", (0.0, 357.5)), ("lat", (-90.0, 90.0)),
        ("level", (100.0, 1000.0))])
    panel["x_dim"] = x_dim
    sel = {"time": "2024-01-01T06:00:00", "level": 500.0,
           "lat": 35.0, "lon": 140.0}
    sel.pop(x_dim)
    var = rng.choice(["t", "u"])
    panel["selection"] = sel
    panel["title"] = f"fuzz line {x_dim}/{var}"
    axis = panel["axis"]
    _rand_axis_common(rng, axis)
    axis["x_label"] = x_dim
    axis["y_label"] = var
    axis["tight_x"] = rng.random() < 0.5
    axis["tight_y"] = rng.random() < 0.5
    if var == "t" and rng.random() < 0.3:
        axis["log_y"] = True  # t は正値なので対数可
    _rand_ticks(rng, axis, "x", x_lo, x_hi)
    _rand_ticks(rng, axis, "y", 200.0, 300.0)
    _rand_frame_bg(rng, panel, (x_lo, x_hi), (200.0, 300.0))
    _rand_legend(rng, panel)
    layer = mc_config.default_line_layer("ds0", var)
    layer["style"].update({
        "color": rng.choice(["tab:blue", "#cc3300"]),
        "linewidth": rng.choice([1.0, 2.5]),
        "linestyle": rng.choice(["-", "--", ":"]),
        "marker": rng.choice([None, "o", "^"]),
        "marker_size": rng.choice([None, 3.0, 9.0]),
        "label": var,
    })
    panel["layers"] = [layer]
    if rng.random() < 0.4:
        # 基準線 (background.reflines): 横線 (凡例ラベル) + 縦線 (端の文字)
        hline = mc_config.default_refline()
        hline.update({"orientation": "y", "value": 250.0 if var == "t" else 0.0,
                      "label": "ref", "alpha": rng.choice([1.0, 0.5]),
                      "text": rng.choice([None, "ref"])})
        vline = mc_config.default_refline()
        vline.update({"orientation": "x", "value": round((x_lo + x_hi) / 2, 2),
                      "linestyle": rng.choice(["dotted", "dashdot"]),
                      "text": "mid", "text_fontsize": rng.choice([None, 7])})
        panel["background"]["reflines"] = [hline, vline]
    if rng.random() < 0.3:
        # 第2軸 (twinx): もう一方の変数を右軸の線 or 棒で重ねる
        other = "u" if var == "t" else "t"
        if rng.random() < 0.5:
            l2 = mc_config.default_line_layer("ds0", other)
            l2["style"].update({"color": "tab:green", "linestyle": "--",
                                "label": other, "secondary_y": True})
        else:
            l2 = mc_config.default_bar_layer("ds0", other)
            l2["style"].update({"color": "tab:green", "alpha": 0.4,
                                "label": other, "secondary_y": True})
        panel["layers"].append(l2)
        axis["y2_label"] = other
        if rng.random() < 0.5:
            axis["y2_lim"] = ([200.0, 300.0] if other == "t"
                              else [-40.0, 40.0])
        axis["swap_y_sides"] = rng.random() < 0.3
        # 第2軸の軸設定 (secondary_axis_cfg の読み替え経路)
        lo2, hi2 = (200.0, 300.0) if other == "t" else (-40.0, 40.0)
        if other == "t" and rng.random() < 0.3:
            axis["log_y2"] = True  # t は正値なので対数可
        axis["invert_y2"] = rng.random() < 0.3
        _rand_ticks(rng, axis, "y2", lo2, hi2)
        axis["show_y2_minor_ticks"] = rng.random() < 0.3
        axis["show_y2_ticklabels"] = rng.random() < 0.8
        axis["grid"]["show_y2"] = rng.random() < 0.3
        # 第2軸との値揃え (手動範囲・対数と組み合わせない: 不変条件を単純に保つ)
        if (rng.random() < 0.4 and axis.get("y_lim") is None
                and axis.get("y2_lim") is None
                and not axis.get("log_y") and not axis.get("log_y2")):
            axis["y2_align_value"] = rng.choice([0.0, 250.0])
    return panel


def _random_scatter_panel(rng: random.Random) -> dict:
    panel = mc_config.default_scatter_panel()
    panel["x_variable"], panel["y_variable"] = "u", "v"
    panel["title"] = "fuzz scatter"
    axis = panel["axis"]
    _rand_axis_common(rng, axis)
    axis["x_label"], axis["y_label"] = "u", "v"
    _rand_ticks(rng, axis, "x", -20.0, 40.0)
    _rand_ticks(rng, axis, "y", -20.0, 20.0)
    _rand_frame_bg(rng, panel, (-20.0, 40.0), (-20.0, 20.0))
    _rand_legend(rng, panel)
    layer = mc_config.default_scatter_layer("ds0")
    layer["drawing_dim"] = "time"
    fixed = {"level": rng.choice([500.0, 850.0]), "lat": 35.0, "lon": 140.0}
    layer["x_fixed"] = dict(fixed)
    layer["y_fixed"] = dict(fixed)
    layer["style"].update({
        "color": rng.choice(["tab:purple", "#008866"]),
        "alpha": round(rng.uniform(0.4, 1.0), 2),
        "marker": rng.choice(["o", "s", "^"]),
        "size": rng.choice([20.0, 60.0]),
        "label": "u-v",
    })
    panel["layers"] = [layer]
    return panel


def _random_section_panel(rng: random.Random) -> dict:
    panel = mc_config.default_section_panel()
    x_dim, (x_lo, x_hi), fix = rng.choice([
        ("lon", (0.0, 357.5), {"lat": 35.0}),
        ("lat", (-90.0, 90.0), {"lon": 140.0})])
    panel["x_dim"], panel["y_dim"] = x_dim, "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00", **fix}
    panel["title"] = f"fuzz section {x_dim}-level"
    axis = panel["axis"]
    _rand_axis_common(rng, axis)
    axis["x_label"], axis["y_label"] = x_dim, "pressure"
    axis["invert_y"] = True
    axis["log_y"] = rng.random() < 0.5
    if x_dim == "lon":
        axis["x_lon_east_west"] = rng.random() < 0.5
    _rand_ticks(rng, axis, "x", x_lo, x_hi)
    _rand_ticks(rng, axis, "y", 100.0, 1000.0)
    _rand_frame_bg(rng, panel, (x_lo, x_hi), (100.0, 1000.0))
    fill = mc_config.default_fill_layer("ds0", "t")
    _rand_fill_style(rng, fill["style"], 200.0, 300.0)
    panel["layers"] = [fill]
    if rng.random() < 0.5:
        contour = mc_config.default_contour_layer("ds0", "z")
        _rand_contour_style(rng, contour["style"], 0.0, 16000.0)
        panel["layers"].append(contour)
    return panel


def _random_map_panel(rng: random.Random) -> dict:
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "fuzz map"
    # 投影法・領域 (地図の座標変換と組み合わせる)
    proj = rng.choice(["default", "Robinson", "EqualEarth",
                       "NorthPolarStereo", "region"])
    if proj in ("Robinson", "EqualEarth"):
        panel["projection"] = {"name": proj,
                               "central_longitude": rng.choice([0.0, 140.0]),
                               "central_latitude": 0.0}
    elif proj == "NorthPolarStereo":
        panel["projection"] = {"name": "NorthPolarStereo",
                               "central_longitude": 140.0,
                               "central_latitude": 0.0}
        panel["region"] = {"lon_min": 0.0, "lon_max": 360.0,
                           "lat_min": rng.choice([20.0, 40.0]),
                           "lat_max": 90.0}
    elif proj == "region":
        panel["projection"] = {"name": "PlateCarree",
                               "central_longitude": 150.0,
                               "central_latitude": 0.0}
        panel["region"] = {"lon_min": 100.0, "lon_max": 200.0,
                           "lat_min": 0.0, "lat_max": 60.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    _rand_fill_style(rng, fill["style"], 230.0, 290.0)
    # maskout の「別変数の値でマスク」。v = 8 sin(3λ-0.4t) cosφ (±8) なので
    # 閾値 ±6 なら必ず描画セルが残る (全 NaN の自動レベル失敗を避ける)
    if rng.random() < 0.3:
        vb, va = rng.choice([(-6.0, None), (None, 6.0), (-6.0, 6.0)])
        fill["style"]["maskout"].update(
            {"variable": "v", "var_below": vb, "var_above": va})
    panel["layers"] = [fill]
    return panel


def _random_map_layers_panel(rng: random.Random) -> dict:
    """地図の重ね描き: fill + {vector, stream, hatch, map_scatter, contour} 1種。

    cmap 系レイヤーは _rand_cmap_section_style を通して、離散化・extend・
    カラーバーの組合せを地図の座標変換と一緒に振る。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "fuzz map layers"
    fill = mc_config.default_fill_layer("ds0", "t")
    _rand_fill_style(rng, fill["style"], 230.0, 290.0)
    layers = [fill]
    extra = rng.choice(["vector", "stream", "hatch", "map_scatter", "contour"])
    if extra == "vector":
        vec = mc_config.default_vector_layer("ds0", "u", "v")
        vec["style"].update({"stride_x": 4, "stride_y": 4})
        vec["style"]["key"]["show"] = rng.random() < 0.5
        if rng.random() < 0.7:
            _rand_cmap_section_style(rng, vec["style"], 0.0, 30.0)
        layers.append(vec)
    elif extra == "stream":
        sp = mc_config.default_stream_layer("ds0", "u", "v")
        sp["style"].update({"density": 1.0,
                            "linewidth": rng.choice([0.8, 1.2])})
        if rng.random() < 0.7:
            _rand_cmap_section_style(rng, sp["style"], 0.0, 30.0)
        layers.append(sp)
    elif extra == "hatch":
        ha = mc_config.default_hatch_layer("ds0", "t")
        lo = rng.choice([230.0, 250.0])
        ha["style"].update({"levels": [lo, lo + 30.0],
                            "pattern": rng.choice(["/", "x", "."]),
                            "density": rng.choice([2, 4])})
        layers.append(ha)
    elif extra == "map_scatter":
        ms = mc_config.default_map_scatter_layer("ds0", "precip")
        ms["style"].update({"size": rng.choice([8.0, 16.0]),
                            "marker": rng.choice(["o", "s"]),
                            "alpha": round(rng.uniform(0.5, 1.0), 2)})
        if rng.random() < 0.7:
            _rand_cmap_section_style(rng, ms["style"], 0.0, 20.0)
        else:
            ms["style"]["use_cmap"] = False
        layers.append(ms)
    else:  # contour (地図でのカラーバー位置合わせも通る)
        contour = mc_config.default_contour_layer("ds0", "z")
        _rand_contour_style(rng, contour["style"], 4900.0, 5900.0)
        layers.append(contour)
    panel["layers"] = layers
    return panel


_PANEL_BUILDERS = [_random_line_panel, _random_section_panel,
                   _random_scatter_panel, _random_map_layers_panel,
                   _random_section_panel, _random_map_panel,
                   _random_scatter_panel, _random_line_panel,
                   _random_map_layers_panel, _random_section_panel,
                   _random_map_layers_panel, _random_map_panel]


def _random_config(base: int, seed: int) -> dict:
    rng = random.Random(base + seed)
    panel = _PANEL_BUILDERS[seed % len(_PANEL_BUILDERS)](rng)
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _expected_fill_array(ds, panel: dict, layer: dict) -> "np.ndarray":
    """fill (pcolormesh) の期待配列を独立に再計算する (render ヘルパー不使用)。

    選択 → (地図なら region の1格子パディング切り出し + 全球 cyclic 列複製) →
    転置 → 値変換 → maskout。docs/semantic_test_plan.md 段階2 (2-4)。
    """
    style = layer["style"]

    def _pipeline(varname: str) -> "np.ndarray":
        """選択 → (地図なら region パディング + cyclic 列) → 転置の共通部。"""
        da = ds[varname]
        sel = {d: v for d, v in {**(panel.get("selection") or {}),
                                 **(layer.get("selection") or {})}.items()
               if d in da.dims}
        if sel:
            da = da.sel(sel)
        if panel.get("plot_type") == "horizontal_map":
            # dtype は元のまま (float32 データを先に float64 化すると値変換の
            # 丸めが render と変わり、~1e-6 の差で不一致になる)
            arr = np.asarray(da.transpose("lat", "lon").values)
            lat = ds["lat"].values
            lon = ds["lon"].values
            region = panel.get("region")
            if region:
                # 両側1格子パディング (padded_slice 仕様)。経度は全球データでは
                # 切らない (cyclic 処理に任せる render の仕様)
                inside = np.where((lat >= region["lat_min"])
                                  & (lat <= region["lat_max"]))[0]
                i0 = max(inside[0] - 1, 0)
                i1 = min(inside[-1] + 1, lat.size - 1)
                arr = arr[i0:i1 + 1]
            step = abs(float(lon[1] - lon[0]))
            if (lon.max() - lon.min()) + step >= 360.0 - 1e-6:
                arr = np.concatenate([arr, arr[:, :1]], axis=1)  # cyclic 列
        else:  # section_2d
            arr = np.asarray(da.transpose(panel["y_dim"], panel["x_dim"]).values)
        return arr

    arr = _pipeline(layer["variable"])
    arr = arr * style.get("value_scale", 1.0) + style.get("value_offset", 0.0)
    mo = style.get("maskout") or {}
    if mo.get("below") is not None:
        arr = np.where(arr <= float(mo["below"]), np.nan, arr)
    if mo.get("above") is not None:
        arr = np.where(arr >= float(mo["above"]), np.nan, arr)
    # 別変数によるマスク (閾値はマスク変数の生の値。値変換は掛からない)
    if mo.get("variable") and (mo.get("var_below") is not None
                               or mo.get("var_above") is not None):
        marr = _pipeline(mo["variable"]).astype(float)
        if mo.get("var_below") is not None:
            arr = np.where(marr <= float(mo["var_below"]), np.nan, arr)
        if mo.get("var_above") is not None:
            arr = np.where(marr >= float(mo["var_above"]), np.nan, arr)
    return arr.astype(float)


def _check_boundary_norm(norm, style: dict, expected_arr=None):
    """離散化 norm の境界が style と一致するか (list = そのまま、int = linspace)。"""
    from matplotlib.colors import BoundaryNorm
    lv = style.get("levels")
    assert isinstance(norm, BoundaryNorm), f"BoundaryNorm でない: {type(norm)}"
    if isinstance(lv, list):
        np.testing.assert_allclose(norm.boundaries, sorted(set(lv)))
    else:
        vmin, vmax = style.get("vmin"), style.get("vmax")
        if vmin is None and expected_arr is not None:
            vmin = float(np.nanmin(expected_arr))
        if vmax is None and expected_arr is not None:
            vmax = float(np.nanmax(expected_arr))
        if vmin is not None and vmax is not None:
            np.testing.assert_allclose(norm.boundaries,
                                       np.linspace(vmin, vmax, int(lv)))
        else:
            assert len(norm.boundaries) == int(lv)
    ext = style.get("extend")
    if ext and ext != "neither":
        assert norm.extend == ext, f"norm.extend が {norm.extend} (設定 {ext})"


def _check_fill_layer(ax, ds, panel: dict, layer: dict):
    """fill レイヤーのデータ検査 (2-4) とレベル境界の照合 (2-3)。"""
    from matplotlib.collections import QuadMesh
    from matplotlib.contour import ContourSet
    style = layer["style"]
    expected = _expected_fill_array(ds, panel, layer)
    if style.get("method", "contourf") == "pcolormesh":
        qms = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(qms) == 1, "fill (pcolormesh) の QuadMesh が特定できない"
        got = np.asarray(np.ma.filled(qms[0].get_array(), np.nan),
                         dtype=float).reshape(expected.shape)
        np.testing.assert_allclose(got, expected, equal_nan=True,
                                   err_msg="fill の描画配列が期待値と不一致")
        _check_boundary_norm(qms[0].norm, style, expected)
    else:
        # contourf: 配列は取り出せないのでレベル列だけ照合する。
        # hatch も filled な ContourSet を作るので「一致するものがある」で判定
        lv = style.get("levels")
        if isinstance(lv, list):
            target = sorted(set(float(v) for v in lv))
        elif style.get("vmin") is not None and style.get("vmax") is not None:
            target = np.linspace(style["vmin"], style["vmax"], int(lv))
            if style.get("zero_white"):
                # zero_white の contourf は端レベルを切り落とす (render と同規則)
                i0, i1 = mc_render.fill_zero_white_trim(
                    style.get("extend", "both"), int(lv))
                target = target[i0:i1]
        else:
            return  # int + 自動範囲は matplotlib の自動レベル (対象外)
        filled = [c for c in ax.collections
                  if isinstance(c, ContourSet) and c.filled]
        assert any(len(c.levels) == len(target)
                   and np.allclose(c.levels, target) for c in filled), \
            f"contourf のレベル {list(target)} を持つ ContourSet がない"


def _check_cmap_layer_norms(ax, panel: dict):
    """cmap 系レイヤー (vector/stream/map_scatter/contour) の離散化境界照合 (2-3)。

    レイヤー種別ごとにアーティスト型が違うことを利用して対応付ける。
    """
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.contour import ContourSet
    from matplotlib.quiver import Quiver
    for layer in panel.get("layers", []):
        style = layer.get("style", {})
        kind = layer.get("kind")
        lv = style.get("levels")
        if kind == "contour" and isinstance(lv, list):
            lines = [c for c in ax.collections
                     if isinstance(c, ContourSet) and not c.filled]
            target = sorted(set(float(v) for v in lv))
            assert any(len(c.levels) == len(target)
                       and np.allclose(c.levels, target) for c in lines), \
                f"contour のレベル {target} を持つ ContourSet がない"
        if not style.get("use_cmap") or not isinstance(lv, list):
            continue
        if kind == "vector":
            arts = [c for c in ax.collections if isinstance(c, Quiver)]
        elif kind == "stream":
            arts = [c for c in ax.collections
                    if isinstance(c, LineCollection)]
        elif kind == "map_scatter":
            arts = [c for c in ax.collections
                    if isinstance(c, PathCollection)]
        else:
            continue
        assert len(arts) == 1, f"{kind} のアーティストが特定できない"
        _check_boundary_norm(arts[0].norm, style)


def _assert_semantic_invariants(cfg: dict, ds_path: str):
    """設定値がアーティストに反映されているかの不変条件 (セマンティック検査)。

    画像一致は「render と scriptgen が同じ」ことしか保証しないため、
    in-process で描画して config → アーティストの対応を検査する
    (docs/semantic_test_plan.md 段階2)。ピクセル比較ではないので長命プロセスの
    汚染問題 (implementation_checklist E) は無関係。
    """
    ds = xr.open_dataset(ds_path)
    fig = mc_render.render_figure(cfg, {"ds0": ds})
    try:
        panel = cfg["panels"][0]
        ptype = panel.get("plot_type")
        ax = fig.axes[0]
        axis_cfg = panel.get("axis", {})
        # fill のデータ検査 + 離散境界、cmap 系レイヤーの離散境界 (2-3 / 2-4)
        if ptype in ("horizontal_map", "section_2d"):
            for layer in panel.get("layers", []):
                if layer.get("kind") == "fill":
                    _check_fill_layer(ax, ds, panel, layer)
            _check_cmap_layer_norms(ax, panel)
        if ptype == "agg_2d":
            from matplotlib.collections import PolyCollection, QuadMesh
            from matplotlib.colors import LogNorm
            layer = panel["layers"][0]
            style = layer["style"]
            cls = QuadMesh if layer["kind"] == "hist2d" else PolyCollection
            arts = [c for c in ax.collections if isinstance(c, cls)]
            assert len(arts) == 1
            if style.get("log_counts"):
                assert isinstance(arts[0].norm, LogNorm), "LogNorm でない"
            elif style.get("levels") is not None:
                _check_boundary_norm(arts[0].norm, style,
                                     np.ma.filled(arts[0].get_array(), np.nan))
        # 軸の意味論 (共有の軸コードを使うモードのみ。dist は categorical 軸、
        # agg は対象の軸機能が異なるため対象外)
        if ptype in ("line_1d", "scatter_2d", "section_2d"):
            if axis_cfg.get("invert_x"):
                assert ax.get_xlim()[0] > ax.get_xlim()[1], "invert_x 未反映"
            if axis_cfg.get("invert_y"):
                assert ax.get_ylim()[0] > ax.get_ylim()[1], "invert_y 未反映"
            if axis_cfg.get("log_y"):
                assert ax.get_yscale() == "log", "log_y 未反映"
            if axis_cfg.get("log_x"):
                assert ax.get_xscale() == "log", "log_x 未反映"
            for prefix, getter in (("x", ax.get_xticks), ("y", ax.get_yticks)):
                pos = axis_cfg.get(f"{prefix}_tick_positions")
                if pos:
                    np.testing.assert_allclose(np.sort(getter()),
                                               np.sort(pos),
                                               err_msg=f"{prefix}_tick_positions 未反映")
        if ptype in ("line_1d", "scatter_2d"):
            legend_cfg = panel.get("legend", {})
            # 第2軸 (twinx) があるときは両軸の handles を結合した凡例が
            # ax2 (fig.axes[1]) 上に置かれる (render._render_line_1d)
            legend_ax = ax
            if ptype == "line_1d" and mc_render.line_uses_secondary_axis(panel):
                legend_ax = fig.axes[1]
                assert ax.get_legend() is None, "第2軸ありで第1軸に凡例がある"
            assert ((legend_ax.get_legend() is not None)
                    == bool(legend_cfg.get("show"))), "legend.show 未反映"
    finally:
        plt.close(fig)
        ds.close()


@pytest.mark.parametrize("seed", range(_N_CASES),
                         ids=[f"fuzz-{i}" for i in range(_N_CASES)])
def test_fuzz_config_render_and_script_match(seed, sample_path, tmp_path):
    base = _seed_base()
    cfg = _random_config(base, seed)
    try:
        _assert_app_and_script_match(cfg, sample_path, tmp_path)
        _assert_semantic_invariants(cfg, sample_path)
    except Exception:
        # 落ちた設定を再現できるように出力しておく
        print(f"--- failing fuzz config (seed={seed}) ---")
        print(f"再現するには: CC_FUZZ_SEED={base} pytest "
              f"'tests/test_fuzz_configs.py::"
              f"test_fuzz_config_render_and_script_match[fuzz-{seed}]'")
        print(json.dumps(cfg, ensure_ascii=False, indent=1))
        raise


# --- 経路断面 (section_path): 1 次元格子のサンプルで、経路の種類・端点・点数・目盛の併記を
#     ランダムに。既存のファズ (上) とは別の抽選列なので既存の seed の config は変わらない ---

_N_PATH_CASES = 6


def _random_path_section_panel(rng: random.Random) -> dict:
    panel = mc_config.default_section_panel()
    panel["x_dim"], panel["y_dim"] = mc_render.SECTION_PATH_DIM, "level"
    kind = rng.choice(["great_circle", "great_circle", "parallel", "meridian"])
    npoints = rng.choice([None, rng.randint(12, 120)])
    if kind == "great_circle":
        lon0 = rng.uniform(0.0, 360.0)
        spec = {"kind": kind, "start": [round(lon0, 1), round(rng.uniform(-60.0, 60.0), 1)],
                "end": [round(lon0 + rng.uniform(20.0, 150.0), 1),
                        round(rng.uniform(-60.0, 70.0), 1)], "npoints": npoints}
    elif kind == "parallel":
        lo = rng.uniform(-30.0, 300.0)
        spec = {"kind": kind, "lat": round(rng.uniform(-70.0, 70.0), 1),
                "lon_range": [round(lo, 1), round(lo + rng.uniform(15.0, 200.0), 1)],
                "npoints": npoints}
    else:
        lo = rng.uniform(-80.0, 20.0)
        spec = {"kind": kind, "lon": round(rng.uniform(0.0, 360.0), 1),
                "lat_range": [round(lo, 1), round(lo + rng.uniform(15.0, 100.0), 1)],
                "npoints": npoints}
    panel["section_path"] = spec
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["title"] = f"fuzz path section {kind}"
    axis = panel["axis"]
    _rand_axis_common(rng, axis)
    axis["x_label"], axis["y_label"] = "path", "pressure"
    axis["invert_y"] = True
    axis["log_y"] = rng.random() < 0.5
    axis["x_lonlat_ticks"] = rng.random() < 0.5
    if kind == "parallel":
        axis["x_lon_east_west"] = rng.random() < 0.5
    _rand_ticks(rng, axis, "y", 100.0, 1000.0)
    fill = mc_config.default_fill_layer("ds0", "t")
    _rand_fill_style(rng, fill["style"], 200.0, 300.0)
    # 別変数によるマスクは期待値の独立計算に入れていないので外す (閾値のマスクは入る)
    for k in ("variable", "var_below", "var_above"):
        (fill["style"].get("maskout") or {}).pop(k, None)
    panel["layers"] = [fill]
    if rng.random() < 0.5:
        contour = mc_config.default_contour_layer("ds0", "z")
        _rand_contour_style(rng, contour["style"], 0.0, 16000.0)
        panel["layers"].append(contour)
    if rng.random() < 0.4:
        vec = mc_config.default_vector_layer("ds0", "u", "v")
        vec["style"]["stride_x"] = rng.randint(2, 6)
        panel["layers"].append(vec)
    return panel


def _expected_path_fill_array(ds, panel: dict, layer: dict) -> "np.ndarray":
    """経路断面の fill の期待配列を独立に計算する (render の関数を使わない)。

    1 次元格子: 経路の点の格子番号を np.interp で求め (経度は格子の規約に写し、全球なら
    最後の列の次を先頭の列として扱う)、周りの 4 点の双一次内挿。経路の点そのものは
    great_circle_points (球面線形補間) を使う — これは test_section_path で独立に検証済み。
    """
    spec = mc_render.section_path_spec(panel["section_path"], ds)
    n = spec["npoints"]
    if spec["kind"] == "parallel":
        plon = np.linspace(*spec["lon_range"], n)
        plat = np.full(n, spec["lat"])
    elif spec["kind"] == "meridian":
        plat = np.linspace(*spec["lat_range"], n)
        plon = np.full(n, spec["lon"])
    else:
        plon, plat, _ = mc_render.great_circle_points(spec["start"], spec["end"], n)
    da = ds[layer["variable"]].sel(time=panel["selection"]["time"])
    lat = ds["lat"].values.astype(float)
    lon = ds["lon"].values.astype(float)
    fj = np.interp(plat, lat, np.arange(lat.size), left=np.nan, right=np.nan)
    lonw = (plon - lon.min()) % 360.0 + lon.min()
    lon_ext = np.append(lon, lon[0] + 360.0)          # 全球格子: 最後の次は先頭
    fi = np.interp(lonw, lon_ext, np.arange(lon_ext.size), left=np.nan, right=np.nan)
    vals = da.transpose("level", "lat", "lon").values.astype(float)
    out = np.full((vals.shape[0], n), np.nan)
    for k in range(n):
        if not (np.isfinite(fj[k]) and np.isfinite(fi[k])):
            continue
        j0 = min(int(np.floor(fj[k])), lat.size - 2)
        a = fj[k] - j0
        i0 = int(np.floor(fi[k])) % lon.size
        b = fi[k] - np.floor(fi[k])
        i1 = (i0 + 1) % lon.size
        out[:, k] = ((1 - a) * (1 - b) * vals[:, j0, i0] + (1 - a) * b * vals[:, j0, i1]
                     + a * (1 - b) * vals[:, j0 + 1, i0] + a * b * vals[:, j0 + 1, i1])
    style = layer["style"]
    out = out * style.get("value_scale", 1.0) + style.get("value_offset", 0.0)
    mo = style.get("maskout") or {}
    if mo.get("below") is not None:
        out = np.where(out <= float(mo["below"]), np.nan, out)
    if mo.get("above") is not None:
        out = np.where(out >= float(mo["above"]), np.nan, out)
    return out


@pytest.mark.parametrize("seed", range(_N_PATH_CASES),
                         ids=[f"pfuzz-{i}" for i in range(_N_PATH_CASES)])
def test_fuzz_section_path_render_and_script_match(seed, sample_path, tmp_path):
    """経路断面のファズ: render ⇄ scriptgen の画素一致と、fill の描画配列 = 独立計算の
    双一次内挿 (経路の全点が格子の外なら明示エラーで終わる)。"""
    base = _seed_base()
    rng = random.Random(base + 7000 + seed)
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [_random_path_section_panel(rng)]
    ds = mc_dataset.open_dataset(sample_path)
    try:
        try:
            expected = _expected_path_fill_array(ds, cfg["panels"][0], cfg["panels"][0]["layers"][0])
        except Exception:
            expected = None
        if expected is not None and not np.isfinite(expected).any():
            with pytest.raises(mc_render.RenderError) as ei:
                mc_render.render_figure(cfg, {"ds0": ds})
            assert ei.value.msg_id == "section_path_outside_grid"
            return
        _assert_app_and_script_match(cfg, sample_path, tmp_path)
        fig = mc_render.render_figure(cfg, {"ds0": ds})
        layer = cfg["panels"][0]["layers"][0]
        if layer["style"].get("method") == "pcolormesh":
            from matplotlib.collections import QuadMesh
            qm = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)][0]
            got = np.asarray(np.ma.filled(qm.get_array(), np.nan), dtype=float).reshape(expected.shape)
            np.testing.assert_allclose(got, expected, rtol=1e-5, equal_nan=True)
        else:
            da = mc_render.select_section_data(
                ds, layer["variable"], cfg["panels"][0]["selection"], None,
                mc_render.SECTION_PATH_DIM, "level", path=cfg["panels"][0]["section_path"])
            got = mc_render.apply_maskout(mc_render.apply_value_transform(da, layer["style"]),
                                          layer["style"], None).values.astype(float)
            np.testing.assert_allclose(got, expected, rtol=1e-5, equal_nan=True)
        plt.close(fig)
    except Exception:
        print(f"--- failing path-section fuzz config (seed={seed}) ---")
        print(f"再現するには: CC_FUZZ_SEED={base} pytest "
              f"'tests/test_fuzz_configs.py::"
              f"test_fuzz_section_path_render_and_script_match[pfuzz-{seed}]'")
        print(json.dumps(cfg, ensure_ascii=False, indent=1))
        raise


# --- 集計系 (agg_2d / dist_1d): 非地理の時系列サンプルで回すファズ ---

def _random_agg_panel(rng: random.Random) -> dict:
    """agg_2d: hist2d / hexbin を抽選し、度数の色付けは
    「なし / 対数 / 離散化 (レベル数 or 直接指定)」から排他で選ぶ (UI と同じ制約)。
    """
    panel = mc_config.default_agg_panel()
    panel["title"] = "fuzz agg"
    xv, yv = rng.choice([("ts", "ens"), ("ens", "ens"), ("ts", "ts")])
    panel["x_variable"], panel["y_variable"] = xv, yv
    _rand_axis_common(rng, panel["axis"])
    kind = rng.choice(["hist2d", "hexbin"])
    if kind == "hist2d":
        ly = mc_config.default_hist2d_layer("ds0")
        ly["style"].update({"bins_x": rng.choice([12, 24]),
                            "bins_y": rng.choice([10, 30]),
                            "density": rng.random() < 0.3,
                            "hide_zeros": rng.random() < 0.7})
        if rng.random() < 0.4:
            ly["style"]["range_x"] = [5.0, 25.0]
            ly["style"]["range_y"] = [5.0, 25.0]
    else:
        ly = mc_config.default_hexbin_layer("ds0")
        ly["style"]["gridsize"] = rng.choice([12, 20, 30])
        ly["style"]["mincnt"] = rng.choice([None, 1, 2])
    # 度数の色付け: 対数と離散化は排他 (UI 側の制約と同じ組合せだけ作る)
    color = rng.choice(["plain", "log", "nlev", "list"])
    if color == "log":
        ly["style"]["log_counts"] = True
    elif color == "nlev":
        ly["style"]["levels"] = rng.choice([5, 8])
        if rng.random() < 0.5:
            ly["style"]["vmax"] = 10.0  # 手動上限 + 自動下限の linspace
        ly["style"]["extend"] = rng.choice(["neither", "max"])
    elif color == "list":
        ly["style"]["levels"] = [1.0, 2.0, 5.0, 10.0, 20.0]
        ly["style"]["extend"] = rng.choice(["neither", "both"])
    ly["style"]["cmap"] = rng.choice(["viridis", "magma"])
    ly["style"]["reverse_cmap"] = rng.random() < 0.3
    ly["style"]["colorbar"]["show"] = rng.random() < 0.85
    # x/y で共通でない次元は固定が必須 (agg_xy_values が ValueError を出す)
    if xv == "ts" and yv == "ens":
        ly["y_fixed"] = {"member": rng.choice([0, 1, 2])}
    if rng.random() < 0.3:
        ly["style"]["y_value_scale"] = 1.2
        ly["style"]["y_value_offset"] = 1.0
    panel["layers"] = [ly]
    return panel


def _random_dist_panel(rng: random.Random) -> dict:
    """dist_1d: hist / ecdf / box / violin を抽選して各オプションを振る。"""
    panel = mc_config.default_dist_panel()
    panel["title"] = "fuzz dist"
    _rand_axis_common(rng, panel["axis"])
    _rand_legend(rng, panel)
    kind = rng.choice(["hist", "ecdf", "box", "violin"])
    var = rng.choice(["ts", "ens"])
    if kind == "hist":
        ly = mc_config.default_hist_layer("ds0")
        ly["style"].update({"bins": rng.choice([10, 15, 30]),
                            "density": rng.random() < 0.4,
                            "cumulative": rng.random() < 0.2,
                            "histtype": rng.choice(["bar", "step",
                                                    "stepfilled"]),
                            "alpha": round(rng.uniform(0.5, 1.0), 2),
                            "edge_linewidth": rng.choice([0.0, 0.8]),
                            "orientation": rng.choice(["vertical", "vertical",
                                                       "horizontal"]),
                            "rwidth": rng.choice([None, 0.6, 0.9]),
                            "label": var})
    elif kind == "ecdf":
        ly = mc_config.default_ecdf_layer("ds0")
        ly["style"].update({"complementary": rng.random() < 0.3,
                            "linewidth": rng.choice([1.0, 2.0]),
                            "linestyle": rng.choice(["solid", "dashed"]),
                            "label": var})
    elif kind == "box":
        ly = mc_config.default_box_layer("ds0")
        ly["style"].update({"label": var, "notch": rng.random() < 0.3,
                            "whis": rng.choice([1.5, [5.0, 95.0]]),
                            "showmeans": rng.random() < 0.4,
                            "showfliers": rng.random() < 0.7,
                            "show_points": rng.random() < 0.3,
                            "fill_color": rng.choice([None, "#87b8d8"])})
    else:
        ly = mc_config.default_violin_layer("ds0")
        ly["style"].update({"label": var,
                            "showmeans": rng.random() < 0.4,
                            "quantiles": rng.choice([None, [0.05, 0.95]]),
                            "side": rng.choice(["both", "low", "high"]),
                            "alpha": round(rng.uniform(0.3, 0.8), 2)})
    ly["variable"] = var
    ly["agg_dim"] = "time"
    if var == "ens":
        ly["selection"] = {"member": rng.choice([0, 1, 2])}
    if rng.random() < 0.3:
        ly["agg_range"] = ["2024-03-01T00:00:00", "2024-09-30T18:00:00"]
    panel["layers"] = [ly]
    return panel


_SERIES_BUILDERS = [_random_agg_panel, _random_dist_panel]


@pytest.mark.parametrize("seed", range(_N_SERIES_CASES),
                         ids=[f"sfuzz-{i}" for i in range(_N_SERIES_CASES)])
def test_fuzz_series_config_render_and_script_match(seed, tmp_path):
    base = _seed_base()
    rng = random.Random(base + 500 + seed)  # 地図系ケースと系列を分ける
    panel = _SERIES_BUILDERS[seed % len(_SERIES_BUILDERS)](rng)
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    path = str(tmp_path / "series_fuzz.nc")
    _make_series_nc(path)
    try:
        _assert_multi_dataset_match(cfg, dataset_paths={"ds0": path},
                                    dataset_renames={}, tmp_path=tmp_path)
        _assert_semantic_invariants(cfg, path)
    except Exception:
        print(f"--- failing series fuzz config (seed={seed}) ---")
        print(f"再現するには: CC_FUZZ_SEED={base} pytest "
              f"'tests/test_fuzz_configs.py::"
              f"test_fuzz_series_config_render_and_script_match[sfuzz-{seed}]'")
        print(json.dumps(cfg, ensure_ascii=False, indent=1))
        raise
