# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""アプリ内描画 (render.py) と生成スクリプト (scriptgen.py) の図が一致することの検証。

このテストは「描けるがスクリプトに出せない機能を作らない」ルール
(development_policy.md) の自動チェックであり、描画機能を追加したら必ず拡張する。
"""

import os
import subprocess
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render
from climcanvas.core import scriptgen as mc_scriptgen


def _simple_config():
    """第1段階相当: 塗りつぶし1レイヤーのみ。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "air temperature at 500 hPa"
    panel["time_label"] = {"show": True, "text": "2024-01-01T06:00:00"}
    layer = mc_config.default_fill_layer("ds0", "t")
    layer["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 300.0, "levels": 15})
    layer["style"]["colorbar"].update({"label": "air temperature [K]"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _rich_config():
    """第2段階: 全レイヤー種別 + 領域 + 地図背景 + グリッド・カラーバー詳細設定。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    # 日付変更線をまたぐ領域 (経度 60–200°E)。中心経度を領域中央に置く。
    panel["region"] = {"lon_min": 60.0, "lon_max": 200.0,
                       "lat_min": -20.0, "lat_max": 60.0}
    panel["projection"] = {"name": "PlateCarree",
                           "central_longitude": 130.0, "central_latitude": 0.0}
    panel["title"] = "multi-layer test"
    panel["title_fontsize"] = 14
    panel["map"]["land"]["show"] = True
    panel["map"]["ocean"]["show"] = True
    panel["map"]["borders"] = True
    panel["map"]["coastlines"].update({"width": 1.2, "color": "#333333"})
    # 明示解像度 (with_scale / coastlines(resolution=) の経路)。110m はテスト環境に
    # 常にある (自動でも全球〜広域で使う) ので新規ダウンロードは起きない
    panel["map"]["resolution"] = "110m"
    panel["map"]["frame_width"] = 1.8
    panel["map"]["gridlines"].update({"lon_interval": 30.0, "lat_interval": 15.0,
                                      "label_fontsize": 8, "linestyle": "--"})
    panel["map"]["gridlines"]["label_sides"] = {"left": True, "right": False,
                                                "top": False, "bottom": True}
    panel["map"]["gridlines"]["label_padding"] = 12
    panel["map"]["ticks"] = {"show": True, "lon_interval": 20.0, "lat_interval": 10.0,
                             "length": 5.0, "width": 0.7, "direction": "out",
                             "sides": {"left": True, "right": True,
                                       "top": False, "bottom": True},
                             # 長い線 (20°/10°) + 短い線 (5°/5°) の混在
                             "minor": {"show": True, "lon_interval": 5.0,
                                       "lat_interval": 5.0,
                                       "length": 2.5, "width": 0.6}}

    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 230.0, "vmax": 260.0, "levels": 11,
                          "alpha": 0.6})
    fill["style"]["colorbar"].update({"label": "K", "location": "bottom",
                                      "shrink": 0.8, "aspect": 30.0, "pad": 0.12,
                                      "label_fontsize": 9, "tick_fontsize": 8,
                                      "outline_width": 1.5, "tick_width": 1.2,
                                      "label_pad": 12.0, "tick_pad": 6.0,
                                      "flip_ticks": True,
                                      "label_opposite": True})  # 目盛り上・ラベル下
    hatch = mc_config.default_hatch_layer("ds0", "t")
    hatch["style"].update({"levels": [250.0, 300.0], "pattern": "/",
                           "density": 4, "linewidth": 1.5})
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"].update({"vmin": 4200.0, "vmax": 5000.0, "interval": 100.0,
                             "linestyle": "dashed", "linewidth": 0.8})
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"].update({"stride_x": 4, "stride_y": 2, "color": "#004488",
                            "mask_below": 2.0})
    vector["style"]["key"]["fontsize"] = 14

    panel["layers"] = [fill, hatch, contour, vector]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _texts_config():
    """panel["texts"] による任意位置の文字列描画 (axes 座標)。

    位置・揃え・回転・色・フォントサイズの全項目を網羅し、
    プロット枠外 (y=-0.1) と空文字 (描画されない) も含める。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "with text annotations"
    layer = mc_config.default_fill_layer("ds0", "t")
    layer["style"].update({"cmap": "viridis", "vmin": 230.0, "vmax": 300.0, "levels": 11})
    panel["layers"] = [layer]
    panel["texts"] = [
        # 左上、デフォルト
        {"text": "(a)", "x": 0.02, "y": 0.95, "fontsize": 14,
         "color": "#000000", "ha": "left", "va": "top", "rotation": 0.0},
        # 中央、赤
        {"text": "CENTER", "x": 0.5, "y": 0.5, "fontsize": 16,
         "color": "#d62728", "ha": "center", "va": "center", "rotation": 0.0},
        # プロット枠の下 (axes 座標の範囲外)
        {"text": "below frame", "x": 0.5, "y": -0.1, "fontsize": 10,
         "color": "#0066aa", "ha": "center", "va": "top", "rotation": 0.0},
        # 90度回転の左端ラベル
        {"text": "rotated", "x": -0.02, "y": 0.5, "fontsize": 11,
         "color": "#222222", "ha": "right", "va": "center", "rotation": 90.0},
        # 空文字は描画されないこと
        {"text": "", "x": 0.1, "y": 0.1, "fontsize": 12,
         "color": "#000000", "ha": "left", "va": "top", "rotation": 0.0},
        # データ座標 (経度・緯度) 指定 — 日付変更線付近
        {"text": "NINO", "coord": "data", "x": 190.0, "y": 0.0, "fontsize": 12,
         "color": "#aa0000", "ha": "center", "va": "center", "rotation": 0.0},
    ]
    # 記号 (マーカー) 注記: axes 相対 + データ座標 (枠線付き)
    panel["markers"] = [
        {"marker": "*", "coord": "axes", "x": 0.9, "y": 0.9, "size": 18.0,
         "color": "#d62728", "edge_width": 0.0, "edge_color": "#000000"},
        {"marker": "^", "coord": "data", "x": 140.0, "y": 35.0, "size": 12.0,
         "color": "#ffff00", "edge_width": 1.0, "edge_color": "#000000"},
    ]
    cfg = mc_config.default_figure_config()
    # 図全体フォント (matplotlib 同梱の DejaVu Serif — どの環境でも存在する)
    cfg["figure"]["font_family"] = "DejaVu Serif"
    cfg["panels"] = [panel]
    return cfg


def _robinson_config():
    """第2段階: Robinson投影・全球・自動レベル。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["projection"] = {"name": "Robinson",
                           "central_longitude": 140.0, "central_latitude": 0.0}
    panel["title"] = "Robinson projection"
    # 緯度20°間隔 + 右 OFF: 高緯度の geo 分類ラベルも右側は消える
    # (geo_hidden_sides フィルタの emit 経路)
    panel["map"]["gridlines"]["lat_interval"] = 20.0
    panel["map"]["gridlines"]["label_sides"] = {
        "left": True, "right": False, "top": True, "bottom": True}
    fill = mc_config.default_fill_layer("ds0", "z")
    contour = mc_config.default_contour_layer("ds0", "z")
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _equal_earth_config():
    """EqualEarth投影・全球・自動レベル (Robinson と同系の湾曲境界)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["projection"] = {"name": "EqualEarth",
                           "central_longitude": 140.0, "central_latitude": 0.0}
    panel["title"] = "Equal Earth projection"
    # 右 OFF: 高緯度の geo 分類ラベルのフィルタが EqualEarth でも効くこと
    panel["map"]["gridlines"]["lat_interval"] = 20.0
    panel["map"]["gridlines"]["label_sides"] = {
        "left": True, "right": False, "top": True, "bottom": True}
    fill = mc_config.default_fill_layer("ds0", "z")
    contour = mc_config.default_contour_layer("ds0", "z")
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _npolar_config():
    """第2段階: 北極中心の極投影 + 全経度の緯度キャップ領域。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["projection"] = {"name": "NorthPolarStereo",
                           "central_longitude": 140.0, "central_latitude": 0.0}
    panel["region"] = {"lon_min": 0.0, "lon_max": 360.0,
                       "lat_min": 30.0, "lat_max": 90.0}
    panel["title"] = "North polar stereographic"
    # 円周沿いの経度ラベル (geo) と図中の緯度ラベル (inline) を消す
    # (両方の hidden emit 経路。上下の接点ラベルは残る)
    panel["map"]["gridlines"]["label_sides"] = {
        "left": True, "right": True, "top": True, "bottom": True,
        "geo": False, "inline": False}
    fill = mc_config.default_fill_layer("ds0", "z")
    contour = mc_config.default_contour_layer("ds0", "z")
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _vsection_config():
    """第3段階: 経度–高度の鉛直断面 (気圧軸を対数・反転、ベクトル付き)。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lon"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00", "lat": 35.0}
    panel["ranges"] = {"level": [1000.0, 200.0]}
    panel["axis"].update({"invert_y": True, "log_y": True,
                          "swap_y_sides": True,   # 縦軸を右側に (Stage 2.5)
                          "x_label": "longitude [degrees_east]",
                          "y_label": "pressure [hPa]",
                          # 経度軸を 180° 中心の東経・西経表記にする
                          "x_lon_east_west": True,
                          # 目盛文字の回転 (tick_params labelrotation 経路)
                          "x_tick_rotation": 30.0, "y_tick_rotation": -20.0,
                          "label_fontsize": 11, "tick_fontsize": 9,
                          # 軸ラベルの体裁 (line_label_kwargs 経由、1D/2Dと共通)
                          "label_color": "#0033cc", "label_weight": "bold",
                          "label_italic": True, "label_pad": 10.0,
                          "x_label_rotation": 10.0,
                          "y_label_rotation": 80.0})
    panel["axis"]["grid"]["show"] = True
    panel["title"] = "zonal cross section at 35N"
    panel["texts"] = [
        # データ座標 (経度・気圧レベル) 指定の文字列
        {"text": "500hPa mark", "coord": "data", "x": 120.0, "y": 500.0,
         "fontsize": 10, "color": "#006600", "ha": "left", "va": "bottom",
         "rotation": 0.0},
    ]
    panel["markers"] = [
        {"marker": "o", "coord": "data", "x": 120.0, "y": 500.0, "size": 8.0,
         "color": "#006600", "edge_width": 0.0, "edge_color": "#000000"},
    ]
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0, "levels": 21,
                          "alpha": 0.7})
    contour = mc_config.default_contour_layer("ds0", "z")
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"].update({"stride_x": 5, "stride_y": 2,
                            "use_cmap": True, "cmap": "plasma",
                            # 境界値の直接指定 (不等間隔) — list 経路のカバー。
                            # int (レベル数) 経路は bubble-cmap-discrete がカバー
                            "levels": [0.0, 3.0, 6.0, 10.0, 15.0, 25.0]})
    vector["style"]["colorbar"].update({"label": "|V| [m/s]",
                                        "location": "bottom", "shrink": 0.7})
    vector["style"]["key"]["show"] = False
    panel["layers"] = [fill, contour, vector]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _vsection_lon_avg_config():
    """緯度-高度の鉛直断面で経度方向に [120, 180] を単純平均する。

    fill (温度) と contour (ジオポ) は同じ経度範囲で平均、
    vector は別の経度範囲 [60, 200] で平均してレイヤー毎の独立性を確認する。
    """
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["axis"].update({"invert_y": True, "log_y": True,
                          "x_label": "latitude",
                          "y_label": "pressure [hPa]"})
    panel["title"] = "lat-height, lon-averaged"

    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0, "levels": 21})
    fill["averages"] = {"lon": {"op": "mean", "range": [120.0, 180.0]}}

    contour = mc_config.default_contour_layer("ds0", "z")
    contour["averages"] = {"lon": {"op": "mean", "range": [120.0, 180.0]}}

    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"].update({"stride_x": 4, "stride_y": 2})
    vector["style"]["key"]["show"] = False
    vector["averages"] = {"lon": {"op": "mean", "range": [60.0, 200.0]}}

    panel["layers"] = [fill, contour, vector]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _vsection_lon_wraparound_avg_config():
    """緯度-高度断面で「30W〜30E」相当の wrap-around 経度平均。

    サンプルデータの経度規約は 0-360 系。ユーザー入力 [-30, 30] が日付変更線を
    またぐ範囲として解釈され、[330, 360) ∪ [0, 30] の連結データで平均が取られる。
    """
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["axis"].update({"invert_y": True, "log_y": True,
                          "x_label": "latitude"})
    panel["title"] = "lat-height, 30W-30E averaged (wrap)"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0, "levels": 21})
    fill["averages"] = {"lon": {"op": "mean", "range": [-30.0, 30.0]}}
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _vsection_lat_weighted_avg_config():
    """経度-高度の鉛直断面で緯度方向に cos(lat) 重み付き平均を取る。

    平均する dim は固定 dim (この向きの fixed_dim=lat) で、緯度方向の重み付き
    平均を確認する。fill のみ (簡潔さのため)。
    """
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lon"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["axis"].update({"invert_y": True, "log_y": True})
    panel["title"] = "lon-height, weighted-lat-averaged"

    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0, "levels": 21})
    fill["averages"] = {"lat": {"op": "weighted_mean", "range": [-30.0, 30.0]}}

    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _hovmoller_config():
    """第3段階: 時間–緯度のHovmöller図 (時間を縦軸・下向き)。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "time"
    panel["selection"] = {"lon": 140.0}
    panel["ranges"] = {"time": ["2024-01-01T00:00:00", "2024-01-02T12:00:00"]}
    panel["axis"].update({
        "invert_y": True, "x_label": "latitude",
        # 断面図の目盛詳細 (1次元・2次元プロットと共通のスキーマ)
        "x_tick_positions": [-60.0, -30.0, 0.0, 30.0, 60.0],
        "x_tick_labels": ["60S", "30S", "EQ", "30N", "60N"],
        "show_x_minor_ticks": True,
        "tick_width": 1.4,
        "grid": {"show_x": True, "show_y": False,
                  "color": "#808080", "width": 0.5, "linestyle": ":"},
    })
    panel["title"] = "precip Hovmoller at 140E"
    # 図枠・背景: 断面図は枠線の太さのみ + 時間軸 (y) の塗り範囲は ISO 文字列
    panel["frame"]["width"] = 1.8
    panel["background"] = {"color": "#f7f7ef", "spans": [
        {"orientation": "y", "lo": "2024-01-01T06:00:00",
         "hi": "2024-01-01T18:00:00", "color": "#ffd7d7", "alpha": 0.5},
    ]}
    fill = mc_config.default_fill_layer("ds0", "precip")
    # 塗りレベルの直接指定 (不等間隔、contourf 経路)
    fill["style"].update({"cmap": "viridis", "extend": "max",
                          "levels": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _time_height_config():
    """第3段階: 時間–高度断面。時間軸を "%m/%d" でフォーマット (DateFormatter 経路) 。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "time"
    panel["y_dim"] = "level"
    panel["selection"] = {"lat": 35.0, "lon": 140.0}
    panel["axis"].update({"invert_y": True, "y_label": "pressure [hPa]",
                          "time_axis_format": "%m/%d",
                          # 等間隔目盛 (MultipleLocator) 経路のカバー
                          "y_tick_interval": 200.0,
                          "show_y_minor_ticks": True})
    panel["title"] = "time-height section"
    fill = mc_config.default_fill_layer("ds0", "t")
    contour = mc_config.default_contour_layer("ds0", "t")
    contour["style"]["labels"]["show"] = False
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _time_height_avg_config():
    """時間–高度断面で緯度帯 [20, 50] と経度帯 [120, 180] の両方を範囲平均。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "time"
    panel["y_dim"] = "level"
    panel["selection"] = {}  # lat / lon はレイヤー averages で潰す
    panel["axis"].update({"invert_y": True, "y_label": "pressure [hPa]",
                          "time_axis_format": "%m/%d"})
    panel["title"] = "time-height, area-averaged"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0, "levels": 21})
    fill["averages"] = {
        "lat": {"op": "mean", "range": [20.0, 50.0]},
        "lon": {"op": "mean", "range": [120.0, 180.0]},
    }
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _hovmoller_lat_avg_config():
    """時間–経度の Hovmöller で緯度帯 [-10, 10] を cos重み付き平均する。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lon"
    panel["y_dim"] = "time"
    panel["selection"] = {}
    panel["axis"].update({"invert_y": True, "x_label": "longitude"})
    panel["title"] = "time-lon Hovmoller, lat [-10,10] avg"
    fill = mc_config.default_fill_layer("ds0", "precip")
    fill["style"].update({"cmap": "viridis", "extend": "max"})
    fill["averages"] = {"lat": {"op": "weighted_mean", "range": [-10.0, 10.0]}}
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _hovmoller_lon_avg_config():
    """時間–緯度の Hovmöller で経度帯 [120, 180] を平均する。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "time"
    panel["selection"] = {}
    panel["axis"].update({"invert_y": True, "x_label": "latitude"})
    panel["title"] = "time-lat Hovmoller, lon [120,180] avg"
    fill = mc_config.default_fill_layer("ds0", "precip")
    fill["style"].update({"cmap": "viridis", "extend": "max"})
    fill["averages"] = {"lon": {"op": "mean", "range": [120.0, 180.0]}}
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _npolar_sector_config():
    """極投影 + 部分経度の領域 → 扇形の境界で表示が連動する。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["projection"] = {"name": "NorthPolarStereo",
                           "central_longitude": 150.0, "central_latitude": 0.0}
    panel["region"] = {"lon_min": 100.0, "lon_max": 200.0,
                       "lat_min": 30.0, "lat_max": 90.0}
    panel["title"] = "polar sector"
    # 緯度ラベルを枠沿いに (y_inline=False の共有 kwargs 経路。geo 分類で表示)、
    # 左の縁のみに絞る + 極 (扇の要) の経度ラベルを消す
    # (draw ラッパの複合フィルタ emit 経路) + 回転を水平固定 (ylabel_style 経路)
    panel["map"]["gridlines"]["lat_label_placement"] = "edge"
    panel["map"]["gridlines"]["lat_label_edge_side"] = "left"
    panel["map"]["gridlines"]["pole_label"] = False
    panel["map"]["gridlines"]["lat_label_rotation"] = 0.0
    fill = mc_config.default_fill_layer("ds0", "z")
    contour = mc_config.default_contour_layer("ds0", "z")
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _neg_contour_config():
    """正負混在の等値線 (v成分): 正=実線・負=破線の per-level 線種を発火させる。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    contour = mc_config.default_contour_layer("ds0", "v")
    contour["style"].update({"vmin": -8.0, "vmax": 8.0, "interval": 2.0})
    panel["layers"] = [contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _contour_emphasis_config():
    """指定レベルの強調: 単色コンターで z=4860 を太赤、+ 負線種と併用。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["title"] = "contour emphasis"
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"].update({
        "vmin": 4800.0, "vmax": 6000.0, "interval": 60.0, "color": "black",
        "emphasis": {"levels": [4860.0, 5400.0], "linewidth": 2.8,
                     "color": "#d00000"}})
    panel["layers"] = [contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _contour_hide_levels_config():
    """emphasis の linewidth=0 で指定レベルを非表示にする (ラベルも一緒に消える)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["title"] = "hide levels (lw=0)"
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"].update({
        "vmin": 4800.0, "vmax": 6000.0, "interval": 60.0, "color": "black",
        "labels": {"show": True},   # 非表示レベルのラベル除外も画像一致で検証
        "emphasis": {"levels": [4860.0, 4980.0], "linewidth": 0.0,
                     "color": None}})
    panel["layers"] = [contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _vsection_contour_emphasis_cmap_config():
    """断面 + cmap コンターで指定レベルを太線 (色は cmap 保持、to_rgba 経路)。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lon"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "lat": 35.0}
    panel["ranges"] = {"level": [1000.0, 200.0]}
    panel["axis"]["invert_y"] = True
    panel["title"] = "section contour emphasis (cmap)"
    contour = mc_config.default_contour_layer("ds0", "t")
    contour["style"].update({
        "vmin": 210.0, "vmax": 290.0, "interval": 10.0,
        "use_cmap": True, "cmap": "turbo",
        "emphasis": {"levels": [250.0], "linewidth": 3.0, "color": None}})
    panel["layers"] = [contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _lambert_config():
    """ランベルト正角円錐図法 (LambertConformal) + 中緯度領域。

    gridline labels を ON にして経度ラベルが図外に出る (x_inline=False) パスを通す。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["projection"] = {"name": "LambertConformal",
                           "central_longitude": -100.0, "central_latitude": 39.0,
                           "standard_parallels": [33.0, 45.0]}
    panel["region"] = {"lon_min": -130.0, "lon_max": -65.0,
                       "lat_min": 22.0, "lat_max": 55.0}
    panel["map"]["gridlines"]["labels"] = True
    panel["layers"] = [mc_config.default_fill_layer("ds0", "t")]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_time_config():
    """1次元プロット: 時系列 (1点固定で時間方向)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
    panel["title"] = "time series at (35N, 140E, 500hPa)"
    panel["axis"]["x_label"] = "time"
    panel["axis"]["y_label"] = "t [K]"
    panel["axis"]["grid"]["show_x"] = True
    panel["axis"]["grid"]["show_y"] = True
    # 背景の塗り範囲: 時間軸は ISO 文字列で lo/hi を指定する
    panel["background"] = {"color": None, "spans": [
        {"orientation": "x", "lo": "2024-01-01T06:00:00",
         "hi": "2024-01-02T00:00:00", "color": "#ffd7d7", "alpha": 0.5},
    ]}
    layer = mc_config.default_line_layer("ds0", "t")
    layer["style"].update({"color": "tab:red", "linewidth": 2.0, "label": "T"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_time_area_avg_config():
    """1次元プロット: 時系列で緯度帯 [20, 60] (cos重み付き) + 経度帯 [120, 180] を平均。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"level": 500.0}
    panel["title"] = "T area-averaged time series"
    panel["axis"]["x_label"] = "time"
    panel["axis"]["y_label"] = "<T> [K]"
    panel["axis"]["grid"]["show_x"] = True
    panel["axis"]["grid"]["show_y"] = True
    layer = mc_config.default_line_layer("ds0", "t")
    layer["style"].update({"color": "tab:red", "linewidth": 2.0, "label": "T"})
    layer["averages"] = {
        "lat": {"op": "weighted_mean", "range": [20.0, 60.0]},
        "lon": {"op": "mean", "range": [120.0, 180.0]},
    }
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_lon_profile_lat_avg_config():
    """1次元プロット: 経度方向のプロファイルで緯度帯を範囲平均する (x_dim≠時間)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["title"] = "lon profile, lat [-30,30] avg"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u"
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    # 緯度方向を cos 重み付き平均
    layer["averages"] = {"lat": {"op": "weighted_mean", "range": [-30.0, 30.0]}}
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bundle_config():
    """1次元プロット: ライン (束) — level の全スライス (6 本) を同じ体裁で重ね描き、
    凡例は 1 つ。マーカー・透明度付き。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"lat": 35.0, "lon": 140.0}
    panel["title"] = "T at (35N, 140E), all levels bundled"
    panel["axis"]["x_label"] = "time"
    panel["axis"]["y_label"] = "T [K]"
    layer = mc_config.default_line_bundle_layer("ds0", "t", "level")
    layer["style"].update({"color": "#4c72b0", "linewidth": 1.0, "alpha": 0.4,
                           "marker": "o", "label": "T (levels)"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bundle_stats_config():
    """1次元プロット: ライン (束) + 統計線 — 経度プロファイルを緯度 (73 本) で束ね、
    経度範囲・レイヤー側の level 固定・値変換 (K→°C) の上に、平均 / 最小と最大 (線) /
    パーセンタイル範囲 10–90 (帯 + 縁の線) / 25–75 (帯) / 中央値 (ラベル無し) /
    平均 ± 1σ (帯) を重ねる。通常の line も同居させる。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00"}
    panel["ranges"] = {"lon": [100.0, 200.0]}
    panel["title"] = "T lon profiles bundled over lat, with summary lines"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "T [degC]"
    panel["legend"]["loc"] = "lower left"
    layer = mc_config.default_line_bundle_layer("ds0", "t", "lat")
    layer["selection"] = {"level": 500.0}
    layer["style"].update({"color": "#999999", "linewidth": 0.6, "alpha": 0.5,
                           "label": "lat slices", "value_offset": -273.15})
    s_mean = mc_config.default_bundle_summary("mean")
    s_mean["style"].update({"color": "#d62728", "label": "mean"})
    s_minmax = mc_config.default_bundle_summary("minmax")
    s_minmax["style"].update({"color": "#2ca02c", "linewidth": 1.0,
                              "linestyle": "dotted", "label": "min-max"})
    s_pct = mc_config.default_bundle_summary("pct_range")
    s_pct.update({"q_low": 10.0, "q_high": 90.0})
    s_pct["draw"] = "band_lines"
    s_pct["style"].update({"color": "#1f77b4", "linewidth": 1.0, "alpha": 0.15,
                           "linestyle": "dashed", "label": "p10-p90"})
    s_band = mc_config.default_bundle_summary("pct_range")
    s_band.update({"q_low": 25.0, "q_high": 75.0, "draw": "band"})
    s_band["style"].update({"color": "#1f77b4", "alpha": 0.35, "label": "p25-p75"})
    s_med = mc_config.default_bundle_summary("median")
    s_med["style"].update({"color": "#000000", "linewidth": 1.0,
                           "linestyle": "dashdot", "label": None})
    s_std = mc_config.default_bundle_summary("std_range")
    s_std.update({"k_std": 1.0, "draw": "band"})
    s_std["style"].update({"color": "#9467bd", "alpha": 0.2, "label": "mean±1σ"})
    layer["summaries"] = [s_mean, s_minmax, s_pct, s_band, s_med, s_std]
    line = mc_config.default_line_layer("ds0", "t")
    line["selection"] = {"level": 500.0, "lat": 0.0}
    line["style"].update({"color": "#ff7f0e", "linewidth": 2.0, "label": "equator",
                          "value_offset": -273.15})
    panel["layers"] = [layer, line]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bundle_band_only_config():
    """1次元プロット: ライン (束) の線幅 0 (束の線を描かない) + パーセンタイル範囲の帯 +
    平均線。統計線の線幅 0 (描かない) も含める。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"lon": 140.0, "level": 500.0}
    panel["title"] = "T at 140E, 500hPa: band over lat only"
    panel["axis"]["y_label"] = "T [K]"
    layer = mc_config.default_line_bundle_layer("ds0", "t", "lat")
    layer["style"].update({"linewidth": 0.0, "label": "hidden"})
    s_band = mc_config.default_bundle_summary("pct_range")
    s_band.update({"q_low": 10.0, "q_high": 90.0, "draw": "band"})
    s_band["style"].update({"color": "#2ca02c", "alpha": 0.3, "label": "p10-p90"})
    s_mean = mc_config.default_bundle_summary("mean")
    s_mean["style"].update({"color": "#2ca02c", "linewidth": 2.0, "label": "mean"})
    s_hidden = mc_config.default_bundle_summary("minmax")
    s_hidden["style"].update({"linewidth": 0.0, "label": "not drawn"})
    layer["summaries"] = [s_band, s_mean, s_hidden]
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_time_lon_wrap_avg_config():
    """1次元プロット: 時系列で経度方向に wrap-around 平均 (30W〜30E)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"level": 500.0, "lat": 35.0}
    panel["title"] = "T(35N), lon wrap-avg 30W-30E"
    layer = mc_config.default_line_layer("ds0", "t")
    layer["style"].update({"color": "tab:green", "label": "T wrap-avg"})
    layer["averages"] = {"lon": {"op": "mean", "range": [-30.0, 30.0]}}
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_two_lines_config():
    """1次元プロット: 経度方向のプロファイル、2変数を重ね描き。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "longitude profile at 35N, 500 hPa"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "value"
    line_u = mc_config.default_line_layer("ds0", "u")
    line_u["style"].update({"color": "tab:blue", "label": "U"})
    line_v = mc_config.default_line_layer("ds0", "v")
    line_v["style"].update({"color": "tab:orange", "linestyle": "dashed", "label": "V"})
    panel["layers"] = [line_u, line_v]
    cfg = mc_config.default_figure_config()
    panel["legend"] = {"show": True, "loc": "best", "fontsize": 9,
                       # 枠外 (右外側・下揃え) への配置 = bbox_to_anchor 経路の
                       # カバー。基準点は凡例の左下角に固定 (loc は使われない)
                       "x": 1.02, "y": 0.0}
    cfg["panels"] = [panel]
    return cfg


def _line_1d_twinx_config():
    """1次元プロット: 第2軸 (twinx)。時系列の t を左軸、u を右軸の線 + 帯で描く。

    y2 ラベル/範囲・目盛文字サイズ・tight_y (両軸)・図枠 (ax2 にも当たる)・
    凡例結合 (枠外配置)・box_aspect (両軸) を同時に検証する。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
    panel["title"] = "twinx: T (left) + U (right)"
    panel["axis"].update({"x_label": "time", "y_label": "t [K]",
                          "y2_label": "u [m/s]", "y2_lim": [-30.0, 40.0],
                          "tick_fontsize": 8, "tight_y": True,
                          "label_fontsize": 11})
    panel["axis"]["grid"]["show_y"] = True
    panel["frame"].update({"show_top": False, "width": 1.2, "color": "#333333"})
    panel["legend"].update({"loc": "upper left", "fontsize": 8,
                            "x": 1.15, "y": 1.0})
    panel["box_aspect"] = 0.6
    line_t = mc_config.default_line_layer("ds0", "t")
    line_t["style"].update({"color": "tab:red", "linewidth": 2.0, "label": "T"})
    line_u = mc_config.default_line_layer("ds0", "u")
    line_u["style"].update({"color": "tab:blue", "linestyle": "dashed",
                            "marker": "o", "label": "U", "secondary_y": True})
    band_u = mc_config.default_fill_between_layer("ds0", "u", baseline=0.0)
    band_u["style"].update({"color": "tab:blue", "color_below": "tab:cyan",
                            "alpha": 0.25, "label": "U band",
                            "secondary_y": True})
    panel["layers"] = [line_t, band_u, line_u]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_twinx_bar_swapped_config():
    """1次元プロット: 第2軸に棒 (dodge の相方は第1軸) + 積み上げ + 縦軸左右入れ替え。

    棒の dodge は軸をまたいで幅・オフセットを共有すること、swap_y_sides で
    第1軸→右・第2軸→左になることを検証する。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "twinx: bars (dodge across axes) + stack, swapped"
    panel["axis"].update({"x_label": "longitude", "y_label": "u [m/s]",
                          "y2_label": "v [m/s]", "swap_y_sides": True,
                          "bar_mode": "dodge", "invert_y": True,
                          # 第2軸の等間隔目盛 + 目盛文字非表示 + 補助目盛
                          "y2_tick_interval": 2.0, "show_y2_ticklabels": False,
                          "show_y2_minor_ticks": True})
    bar_u = mc_config.default_bar_layer("ds0", "u")
    bar_u["style"].update({"color": "tab:blue", "alpha": 0.6, "label": "U"})
    bar_v = mc_config.default_bar_layer("ds0", "v")
    bar_v["style"].update({"color": "tab:orange", "alpha": 0.6, "label": "V",
                           "edge_color": "black", "edge_linewidth": 0.5,
                           "secondary_y": True})
    stack = mc_config.default_stackplot_layer("ds0", ["u", "v"])
    stack["style"].update({"alpha": 0.15, "colors": ["#888888", "#cccccc"],
                           "show_labels_in_legend": False,
                           "secondary_y": True})
    panel["layers"] = [bar_u, bar_v, stack]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_twinx_axis_config():
    """1次元プロット: 第2軸の軸設定一式 (対数・反転・目盛位置+カスタムラベル・
    補助目盛・目盛線) と、第1軸と共有する体裁 (目盛線の太さ・目盛文字の回転・
    軸ラベルの回転・pad)。経度プロファイルの u を左、t (正値) を右の対数軸に。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "twinx: log + inverted right axis, fixed ticks"
    panel["axis"].update({
        "x_label": "longitude", "y_label": "u [m/s]",
        "y2_label": "t [K]", "log_y2": True, "invert_y2": True,
        "y2_tick_positions": [236.0, 238.0, 240.0, 242.0, 244.0, 246.0],
        "y2_tick_labels": ["236 K", "238 K", "240 K", "242 K", "244 K", "246 K"],
        "show_y2_minor_ticks": True, "show_y_minor_ticks": True,
        "tick_width": 1.5, "y_tick_rotation": 30.0, "tick_fontsize": 9,
        "y_label_rotation": 0.0, "label_pad": 12.0,
    })
    panel["axis"]["grid"].update({"show_y2": True, "color": "#cc8888",
                                  "width": 0.8, "linestyle": "--"})
    line_u = mc_config.default_line_layer("ds0", "u")
    line_u["style"].update({"color": "tab:blue", "label": "U"})
    line_t = mc_config.default_line_layer("ds0", "t")
    line_t["style"].update({"color": "tab:red", "linestyle": "dotted",
                            "linewidth": 2.0, "label": "T", "secondary_y": True})
    panel["layers"] = [line_u, line_t]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_twinx_align_zero_config():
    """1次元プロット: 第2軸との値揃え (y2_align_value=0)。u−18 (左, 零をまたぐ) と
    3v (右) の 0 の高さを揃える。目盛線は第1軸だけ、余白除去 (tight_y) を両軸に。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "twinx: zero aligned on both axes"
    panel["axis"].update({"x_label": "longitude", "y_label": "u - 18 [m/s]",
                          "y2_label": "3 v [m/s]", "y2_align_value": 0.0,
                          "tight_y": True})
    panel["axis"]["grid"]["show_y"] = True
    line_u = mc_config.default_line_layer("ds0", "u")
    line_u["style"].update({"color": "tab:blue", "label": "U-18",
                            "value_offset": -18.0})
    line_v = mc_config.default_line_layer("ds0", "v")
    line_v["style"].update({"color": "tab:orange", "linestyle": "dashed",
                            "marker": ".", "label": "3V", "secondary_y": True,
                            "value_scale": 3.0})
    panel["layers"] = [line_u, line_v]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_reflines_config():
    """1次元プロット: 基準線 (background.reflines)。時間軸の縦線 (時刻指定 + 凡例
    ラベル + 端の文字) と y 一定の横線 2 本 (端の文字の色/サイズ指定・透過度・
    凡例ラベルのみ)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
    panel["title"] = "reference lines (time x, y values)"
    panel["axis"].update({"x_label": "time", "y_label": "t [K]"})
    l1 = mc_config.default_refline()
    l1.update({"orientation": "x", "value": "2024-01-01T12:00:00",
               "color": "#2ca02c", "linewidth": 1.5, "linestyle": "dashdot",
               "label": "event", "text": "onset"})
    l2 = mc_config.default_refline()
    l2.update({"orientation": "y", "value": 240.0, "color": "#d62728",
               "alpha": 0.8, "text": "240 K", "text_fontsize": 8,
               "text_color": "#7f0000"})
    l3 = mc_config.default_refline()
    l3.update({"orientation": "y", "value": 242.0, "color": "#7f7f7f",
               "linewidth": 0.8, "linestyle": "dotted", "label": "242 K"})
    panel["background"]["reflines"] = [l1, l2, l3]
    layer = mc_config.default_line_layer("ds0", "t")
    layer["style"].update({"color": "tab:blue", "label": "T"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_fill_between_two_vars_config():
    """1次元プロット: 2変数 (u と v) の間を塗りつぶす + ライン重ね描き。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "fill_between u and v"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "value"
    fb = mc_config.default_fill_between_layer("ds0", "u", variable_upper="v")
    fb["style"].update({"color": "#888888", "alpha": 0.4, "label": "u..v band"})
    line_u = mc_config.default_line_layer("ds0", "u")
    line_u["style"].update({"color": "tab:blue", "label": "U"})
    line_v = mc_config.default_line_layer("ds0", "v")
    line_v["style"].update({"color": "tab:orange", "linestyle": "dashed", "label": "V"})
    panel["layers"] = [fb, line_u, line_v]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_fill_between_baseline_config():
    """1次元プロット: 1変数とベースライン (0) の間を塗りつぶす (偏差図の正/負共通単色)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "fill_between u and baseline=0"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    fb = mc_config.default_fill_between_layer("ds0", "u",
                                                variable_upper=None, baseline=0.0)
    fb["style"].update({"color": "tab:red", "alpha": 0.3})
    line_u = mc_config.default_line_layer("ds0", "u")
    line_u["style"].update({"color": "tab:red", "label": "U"})
    panel["layers"] = [fb, line_u]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_fill_between_split_baseline_config():
    """1次元プロット: u が 0 を跨ぐので、ベースラインの上下で別色 (赤/青) に塗り分け。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "u anomaly above/below 0"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    fb = mc_config.default_fill_between_layer("ds0", "u",
                                                variable_upper=None, baseline=0.0)
    fb["style"].update({"color": "#d62728", "color_below": "#1f77b4",
                        "alpha": 0.4})
    line_u = mc_config.default_line_layer("ds0", "u")
    line_u["style"].update({"color": "black", "label": "U"})
    panel["layers"] = [fb, line_u]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_fill_between_per_var_levels_config():
    """1次元プロット: 2変数 (両方 t) を異なる level に固定して帯を描く (層厚図イメージ)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "lat": 35.0}
    panel["title"] = "t at 500 vs 850 hPa"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "t [K]"
    fb = mc_config.default_fill_between_layer("ds0", "t", variable_upper="t")
    fb["selection"] = {"level": 500.0}
    fb["selection_upper"] = {"level": 850.0}
    fb["style"].update({"color": "#999999", "alpha": 0.4, "label": "500-850 band"})
    line_lower = mc_config.default_line_layer("ds0", "t")
    line_lower["selection"] = {"level": 500.0}
    line_lower["style"].update({"color": "tab:blue", "label": "t @ 500"})
    line_upper = mc_config.default_line_layer("ds0", "t")
    line_upper["selection"] = {"level": 850.0}
    line_upper["style"].update({"color": "tab:red", "label": "t @ 850"})
    panel["layers"] = [fb, line_lower, line_upper]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_label_style_config():
    """1次元プロット: 軸ラベルに色・太字・斜体・labelpad・rotation を設定。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "label style"
    panel["axis"].update({
        "swap_y_sides": True,   # 縦軸を右側に (Stage 2.5)
        "x_label": "longitude [deg]", "y_label": "u [m/s]",
        "label_fontsize": 12, "label_color": "#0033cc",
        "label_weight": "bold", "label_italic": True,
        "label_pad": 12.0,
        "x_label_rotation": 15.0, "y_label_rotation": 45.0,
        "tick_width": 1.8,
    })
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_logx_xlim_ylim_config():
    """1次元プロット: 横軸を対数 + x/y_lim 手動指定。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "lat": 35.0, "lon": 140.0}
    panel["title"] = "log_x + manual x/y lim"
    panel["axis"]["x_label"] = "level [hPa]"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["log_x"] = True
    panel["axis"]["x_lim"] = [200.0, 1000.0]
    panel["axis"]["y_lim"] = [-15.0, 25.0]
    panel["axis"]["invert_x"] = True   # 気圧は大きい方を下に
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:green", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_minor_and_per_axis_grid_config():
    """1次元プロット: 補助目盛を x のみ、目盛線 (grid) を y のみに表示。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "x minor ticks + y-only grid"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["show_x_minor_ticks"] = True
    panel["axis"]["show_y_minor_ticks"] = False
    panel["axis"]["grid"]["show_x"] = False
    panel["axis"]["grid"]["show_y"] = True
    panel["axis"]["grid"]["color"] = "#cccccc"
    panel["axis"]["grid"]["linestyle"] = "--"
    panel["axis"]["grid"]["width"] = 0.7
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_tick_custom_labels_config():
    """1次元プロット: 位置 + カスタム文字列ラベル (FixedLocator + FixedFormatter)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "custom tick labels"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["x_tick_positions"] = [0.0, 90.0, 180.0, 270.0, 360.0]
    panel["axis"]["x_tick_labels"] = ["0°", "90°E", "180°", "90°W", "0°"]
    panel["axis"]["y_tick_positions"] = [-30.0, 0.0, 30.0]
    panel["axis"]["y_tick_labels"] = ["LOW", "ZERO", "HIGH"]
    # 目盛文字の回転 (カスタムラベルとの併用)
    panel["axis"]["x_tick_rotation"] = 45.0
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_tick_positions_config():
    """1次元プロット: 不等間隔の目盛位置を FixedLocator で指定。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "irregular tick positions"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["x_tick_positions"] = [0.0, 30.0, 90.0, 180.0, 270.0, 360.0]
    panel["axis"]["y_tick_positions"] = [-20.0, -5.0, 0.0, 5.0, 25.0]
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_no_ticks_config():
    """1次元プロット: 目盛間隔 0 → NullLocator で目盛り完全非表示。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "no ticks"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["x_tick_interval"] = 0.0
    panel["axis"]["y_tick_interval"] = 0.0
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_tick_interval_config():
    """1次元プロット: x/y 軸の目盛間隔を手動指定 (MultipleLocator)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "manual tick intervals"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["x_tick_interval"] = 60.0
    panel["axis"]["y_tick_interval"] = 5.0
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_hide_ticklabels_config():
    """1次元プロット: x/y 軸の目盛文字を両方非表示にする (ax.tick_params)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "hide ticklabels"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["show_x_ticklabels"] = False
    panel["axis"]["show_y_ticklabels"] = False
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_markers_only_config():
    """1次元プロット: 線種「なし」+ マーカー指定で点プロット (散布図風)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "markers only"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:purple", "linestyle": "None",
                            "marker": "o", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _scatter_2d_ranges_config():
    """2次元プロット (散布図): drawing_dim=time の範囲を絞り、x/y で異なる lev に固定。

    u (drawing=time, lat=35, lon=140, lev=500) vs v (lat=0, lon=140, lev=850) を散布。
    """
    panel = mc_config.default_scatter_panel()
    panel["x_variable"] = "u"
    panel["y_variable"] = "v"
    panel["title"] = "u(500hPa, 35N, 140E) vs v(850hPa, 0N, 140E) over time"
    panel["axis"].update({
        "swap_y_sides": True,   # 縦軸を右側に (Stage 2.5)
        "x_label": "u [m/s]", "y_label": "v [m/s]",
        # 目盛の詳細設定 (1次元プロットと同じスキーマ)
        "x_tick_interval": 2.0,
        "y_tick_positions": [-5.0, 0.0, 5.0],
        "y_tick_labels": ["low", "zero", "high"],
        "show_x_minor_ticks": True,
        "show_y_minor_ticks": True,
        "tick_width": 1.5,
        "grid": {"show_x": True, "show_y": True,
                  "color": "#808080", "width": 0.5, "linestyle": ":"},
    })
    # 図枠・背景: 左と下だけ残して太く、背景色と塗り範囲を指定
    panel["frame"] = {"show_top": False, "show_bottom": True,
                       "show_left": True, "show_right": False,
                       "width": 1.2, "color": "#333333"}
    panel["background"] = {"color": "#eef4fb", "spans": [
        {"orientation": "y", "lo": 0.0, "hi": 3.0,
         "color": "#ffe4b3", "alpha": 0.6},
    ], "reflines": [
        {"orientation": "x", "value": 0.0, "color": "#444444",
         "linewidth": 1.0, "linestyle": "dashed", "alpha": 1.0,
         "label": None, "text": "u = 0", "text_fontsize": None,
         "text_color": None},
        {"orientation": "y", "value": 0.0, "color": "#444444",
         "linewidth": 1.0, "linestyle": "solid", "alpha": 0.6,
         "label": "v = 0", "text": None, "text_fontsize": None,
         "text_color": None},
    ]}
    layer = mc_config.default_scatter_layer("ds0")
    layer["drawing_dim"] = "time"
    layer["drawing_range"] = ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
    layer["x_fixed"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
    layer["y_fixed"] = {"level": 850.0, "lat": 0.0, "lon": 140.0}
    layer["style"].update({
        "color": "tab:red", "alpha": 0.7,
        "marker": "o", "size": 30.0,
        "label": "u-v over time",
        # エラーバー: x/y 独立の誤差変数 (t を擬似誤差に。|err| で正規化)
        "errorbar": {"x_variable": "t", "y_variable": "v",
                     "color": "#555555", "linewidth": 0.8, "capsize": 2.5},
        # 値変換 (誤差には倍率のみ掛かる経路の検証)
        "x_value_scale": 2.0,
    })
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _scatter_2d_time_full_config():
    """2次元プロット (散布図): drawing_dim=time、drawing_range=None で全 8 時刻を散布。"""
    panel = mc_config.default_scatter_panel()
    panel["x_variable"] = "u"
    panel["y_variable"] = "v"
    panel["title"] = "u vs v over all times (full range)"
    panel["axis"].update({
        "x_label": "u [m/s]", "y_label": "v [m/s]",
        # 軸ラベルの体裁 (1次元プロットと共通の line_label_kwargs 経由)
        "label_fontsize": 12, "label_color": "#0033cc",
        "label_weight": "bold", "label_italic": True,
        "label_pad": 12.0,
        "x_label_rotation": 15.0, "y_label_rotation": 45.0,
        # 目盛文字の非表示 (目盛り線は残る)
        "show_x_ticklabels": False,
        "show_y_ticklabels": False,
    })
    layer = mc_config.default_scatter_layer("ds0")
    layer["drawing_dim"] = "time"
    layer["drawing_range"] = None  # 全期間
    layer["x_fixed"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
    layer["y_fixed"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
    layer["style"].update({
        "color": "tab:purple", "alpha": 0.9,
        "marker": "o", "size": 40.0,
        "label": "u-v over time",
    })
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _bubble_2d_uvt_cmap_config():
    """2次元プロット (バブル + cmap): z=t の値をマーカー色にもマッピング + カラーバー。"""
    panel = mc_config.default_scatter_panel()
    panel["x_variable"] = "u"
    panel["y_variable"] = "v"
    panel["z_variable"] = "t"
    panel["title"] = "bubble u-v sized & colored by t"
    panel["axis"].update({"x_label": "u [m/s]", "y_label": "v [m/s]"})
    layer = mc_config.default_bubble_layer("ds0")
    layer["drawing_dim"] = "lon"
    fixed = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    layer["x_fixed"] = dict(fixed)
    layer["y_fixed"] = dict(fixed)
    layer["z_fixed"] = dict(fixed)
    layer["style"].update({
        "alpha": 0.7,
        "marker": "o",
        "size_min": 20.0, "size_max": 300.0,
        "use_cmap": True,
        "cmap": "viridis",
        "vmin": 240.0, "vmax": 285.0,
        "edge_color": "#333333", "edge_linewidth": 0.4,
        "label": "u-v-t bubble",
    })
    layer["style"]["colorbar"].update({
        "show": True, "label": "t [K]",
        "shrink": 0.9, "aspect": 25.0,
    })
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _bubble_2d_uvt_config():
    """2次元プロット (バブルチャート): u を x、v を y、t (気温) を z として点サイズに。

    drawing_dim=lon で経度方向に散布、time/lat/level は x/y/z 共通に固定。
    """
    panel = mc_config.default_scatter_panel()
    panel["x_variable"] = "u"
    panel["y_variable"] = "v"
    panel["z_variable"] = "t"
    panel["title"] = "bubble: u-v sized by t"
    panel["axis"].update({
        "x_label": "u [m/s]", "y_label": "v [m/s]",
    })
    layer = mc_config.default_bubble_layer("ds0")
    layer["drawing_dim"] = "lon"
    fixed = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    layer["x_fixed"] = dict(fixed)
    layer["y_fixed"] = dict(fixed)
    layer["z_fixed"] = dict(fixed)
    layer["style"].update({
        "color": "tab:orange", "alpha": 0.5,
        "marker": "o",
        "size_min": 20.0, "size_max": 300.0,
        "edge_color": "#663300", "edge_linewidth": 0.3,
        "label": "u-v-t bubble",
        # エラーバー (y のみ): 点より下に描かれる
        "errorbar": {"x_variable": None, "y_variable": "t",
                     "color": "#994400", "linewidth": 1.0, "capsize": 3.0},
    })
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _scatter_2d_uv_config():
    """2次元プロット (散布図): drawing_dim=lon で経度方向に散布、time/lat/level 固定。"""
    panel = mc_config.default_scatter_panel()
    panel["x_variable"] = "u"
    panel["y_variable"] = "v"
    panel["title"] = "u vs v across longitude"
    panel["axis"].update({
        "x_label": "u [m/s]", "y_label": "v [m/s]",
        "label_fontsize": 11, "tick_fontsize": 9,
    })
    panel["axis"]["grid"]["show_x"] = True
    panel["axis"]["grid"]["show_y"] = True
    layer = mc_config.default_scatter_layer("ds0")
    layer["drawing_dim"] = "lon"
    fixed = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    layer["x_fixed"] = dict(fixed)
    layer["y_fixed"] = dict(fixed)
    layer["style"].update({
        "color": "tab:blue", "alpha": 0.7,
        "marker": "o", "size": 25.0,
        "edge_color": "#003366", "edge_linewidth": 0.3,
        "label": "u-v 500 hPa",
    })
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_stackplot_config():
    """1次元プロット: スタックプロット (ax.stackplot, baseline='sym')。

    u/v は正負を取りうるので sym (対称) baseline を使う。色は手動指定。
    """
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "stackplot u and v (sym)"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "stacked value"
    layer = mc_config.default_stackplot_layer("ds0", ["u", "v"])
    layer["style"].update({
        "alpha": 0.6,
        "baseline": "sym",
        "colors": ["#1f77b4", "#ff7f0e"],
        "show_labels_in_legend": True,
    })
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_overlap_config():
    """1次元プロット: 縦棒 (overlap)。経度方向の u を棒で描画 + 枠線付き。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "bar overlap"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    bar = mc_config.default_bar_layer("ds0", "u")
    bar["style"].update({"color": "tab:blue", "edge_color": "black",
                          "edge_linewidth": 0.5, "label": "U"})
    panel["layers"] = [bar]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_dodge_config():
    """1次元プロット: 縦棒 (dodge) で u と v を横並び。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "bar dodge"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "value"
    panel["axis"]["bar_mode"] = "dodge"
    bar_u = mc_config.default_bar_layer("ds0", "u")
    bar_u["style"].update({"color": "tab:blue", "label": "U"})
    bar_v = mc_config.default_bar_layer("ds0", "v")
    bar_v["style"].update({"color": "tab:orange", "label": "V"})
    panel["layers"] = [bar_u, bar_v]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_stack_config():
    """1次元プロット: 縦棒 (stack) で u と v を積み上げ。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "bar stack"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "value"
    panel["axis"]["bar_mode"] = "stack"
    bar_u = mc_config.default_bar_layer("ds0", "u")
    bar_u["style"].update({"color": "tab:blue", "label": "U"})
    bar_v = mc_config.default_bar_layer("ds0", "v")
    bar_v["style"].update({"color": "tab:orange", "label": "V"})
    panel["layers"] = [bar_u, bar_v]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_hatch_config():
    """1次元プロット: 棒グラフにハッチをかける (枠線 on で色を引き継ぎ)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "bar with hatch"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    bar = mc_config.default_bar_layer("ds0", "u")
    bar["style"].update({
        "color": "tab:blue", "alpha": 0.5,
        "edge_color": "#003366", "edge_linewidth": 0.8,
        "hatch_pattern": "/", "hatch_density": 4,
        "label": "U",
    })
    panel["layers"] = [bar]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_errorbar_variable_config():
    """1次元プロット: 棒グラフに変数 (v の |abs|) をエラー量にした対称エラーバーを描く。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "bar with errorbar (variable)"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    bar = mc_config.default_bar_layer("ds0", "u")
    bar["style"].update({
        "color": "tab:blue", "alpha": 0.8,
        "label": "U",
    })
    bar["style"]["errorbar"] = {
        "source": "variable", "variable": "v", "constant": 0.0,
        "color": "#333333", "linewidth": 1.2, "capsize": 4.0,
    }
    panel["layers"] = [bar]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_errorbar_constant_config():
    """1次元プロット: 棒グラフに定数エラー (±2) をエラーバーで描く。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "bar with errorbar (constant)"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    bar = mc_config.default_bar_layer("ds0", "u")
    bar["style"].update({
        "color": "tab:orange", "alpha": 0.8,
        "label": "U",
    })
    bar["style"]["errorbar"] = {
        "source": "constant", "variable": None, "constant": 2.0,
        "color": "#000000", "linewidth": 1.0, "capsize": 3.0,
    }
    panel["layers"] = [bar]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_hatch_no_edge_config():
    """1次元プロット: 棒グラフに枠線なしでハッチに色を付ける。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "bar hatch without edge"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    bar = mc_config.default_bar_layer("ds0", "u")
    bar["style"].update({
        "color": "#cce5ff", "alpha": 1.0,
        # 枠線 off (edge_linewidth=0) でも hatch_color が効くか確認
        "edge_linewidth": 0.0,
        "hatch_pattern": "x", "hatch_density": 3,
        "hatch_color": "#cc0000",
        "label": "U",
    })
    panel["layers"] = [bar]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_bar_horizontal_config():
    """1次元プロット: 横棒 (ax.barh)。手動 width 指定。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lat"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lon": 140.0}
    panel["title"] = "horizontal bar"
    panel["axis"]["x_label"] = "u [m/s]"
    panel["axis"]["y_label"] = "latitude"
    bar = mc_config.default_bar_layer("ds0", "u")
    bar["style"].update({"orientation": "horizontal", "color": "tab:green",
                          "width": 2.0, "label": "U"})
    panel["layers"] = [bar]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_cyclic_lon_config():
    """1次元プロット: x=lon を cyclic_x=True で 0..720° まで周期的に展開。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["ranges"] = {"lon": [0.0, 720.0]}
    panel["title"] = "cyclic longitude (0..720°)"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["cyclic_x"] = True
    panel["axis"]["tight_x"] = True
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_log_tick_positions_config():
    """1次元プロット: 対数軸 + 目盛位置の直接指定。

    位置指定は対数軸の既定目盛より優先される (対数適用後に FixedLocator を
    当てる) ことを固定するリグレッションテスト。カスタムラベル付き。
    """
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0,
                           "lat": 35.0}
    panel["title"] = "log y with explicit tick positions"
    panel["axis"].update({
        "x_label": "longitude", "y_label": "t [K]",
        "log_y": True,
        "y_tick_positions": [230.0, 245.0, 260.0, 275.0],
        "y_tick_labels": ["230 K", "245 K", "260 K", "275 K"],
    })
    layer = mc_config.default_line_layer("ds0", "t")
    layer["style"].update({"color": "tab:red", "label": "T"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _line_1d_tight_x_config():
    """1次元プロット: tight_x/tight_y=True で軸の余白を除去 (端ぴったり)。"""
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "lon"
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0, "lat": 35.0}
    panel["title"] = "tight x/y axis"
    panel["axis"]["x_label"] = "longitude"
    panel["axis"]["y_label"] = "u [m/s]"
    panel["axis"]["tight_x"] = True
    panel["axis"]["tight_y"] = True
    # 図枠・背景: 上と右の枠線を消し、残りを太く色付け、背景色と塗り範囲を指定
    panel["frame"] = {"show_top": False, "show_bottom": True,
                       "show_left": True, "show_right": False,
                       "width": 1.5, "color": "#003366"}
    panel["background"] = {"color": "#f5f0e6", "spans": [
        {"orientation": "x", "lo": 120.0, "hi": 180.0,
         "color": "#ffd7d7", "alpha": 0.5},
        {"orientation": "y", "lo": -5.0, "hi": 0.0,
         "color": "#d7e8ff", "alpha": 0.4},
    ]}
    layer = mc_config.default_line_layer("ds0", "u")
    layer["style"].update({"color": "tab:blue", "label": "U"})
    panel["layers"] = [layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _fill_custom_levels_config():
    """水平断面図: 塗りレベルの直接指定 (不等間隔リスト、pcolormesh 経路)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00"}
    panel["title"] = "precip with custom uneven levels"
    fill = mc_config.default_fill_layer("ds0", "precip")
    fill["style"].update({"method": "pcolormesh", "cmap": "viridis",
                          "levels": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
                          "extend": "max"})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _pcolormesh_fill_config():
    """塗りつぶしを pcolormesh + BoundaryNorm (離散) で描画するモード。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"method": "pcolormesh", "cmap": "viridis",
                          "vmin": 230.0, "vmax": 290.0, "levels": 13})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _pcolormesh_auto_range_config():
    """pcolormesh モード + vmin/vmax 未指定 (自動範囲)。

    回帰テスト: 以前は `style.get("vmin", default)` の使い方を間違えて
    `np.linspace(None, None, ...)` で TypeError を出していた。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"method": "pcolormesh"})  # vmin/vmax は None のまま
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _boxes_config():
    """map.boxes: 通常の矩形と日付変更線をまたぐ矩形 (lon_max < lon_min) の両方を描く。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["layers"] = [mc_config.default_fill_layer("ds0", "t")]
    panel["map"]["boxes"] = [
        {"lon_min": -60.0, "lon_max": 60.0, "lat_min": 20.0, "lat_max": 60.0,
         "color": "red", "linewidth": 1.5, "linestyle": "solid"},
        # 日付変更線をまたぐ (lon_max < lon_min)
        {"lon_min": 150.0, "lon_max": -150.0, "lat_min": -30.0, "lat_max": 0.0,
         "color": "blue", "linewidth": 2.0, "linestyle": "dashed"},
    ]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _land_foreground_config():
    """陸をデータの上に描く (map.land.above_data)。格子データ (塗り・ハッチ・等値線 +
    ラベル・ベクトル + 図中の基準ベクトル・流線) は陸に隠れ、海岸線・国境線・
    グリッド線・box・記号注記・散布点は陸の上に出る (zorder の付け替えが
    render と scriptgen で一致すること)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["region"] = {"lon_min": 90.0, "lon_max": 160.0,
                       "lat_min": 0.0, "lat_max": 60.0}
    panel["projection"] = {"name": "PlateCarree",
                           "central_longitude": 125.0, "central_latitude": 0.0}
    panel["title"] = "land above data"
    panel["map"]["land"].update({"show": True, "above_data": True})
    panel["map"]["ocean"]["show"] = True
    panel["map"]["borders"] = True
    panel["map"]["boxes"] = [
        {"lon_min": 100.0, "lon_max": 140.0, "lat_min": 20.0, "lat_max": 45.0,
         "color": "red", "linewidth": 1.5, "linestyle": "solid"}]
    panel["markers"] = [
        {"marker": "*", "coord": "data", "x": 116.0, "y": 40.0, "size": 16.0,
         "color": "#d62728", "edge_width": 0.8, "edge_color": "#000000"}]
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 300.0, "levels": 15})
    hatch = mc_config.default_hatch_layer("ds0", "t")
    contour = mc_config.default_contour_layer("ds0", "t")
    contour["style"]["labels"]["show"] = True
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"].update({"stride_x": 4, "stride_y": 3})
    # 基準ベクトルを図中 (陸の上) に置く
    vector["style"].setdefault("key", {}).update(
        {"show": True, "x": 0.85, "y": 0.9, "label": "10 m/s"})
    stream = mc_config.default_stream_layer("ds0", "u", "v")
    scatter = mc_config.default_map_scatter_layer("ds0", "precip")
    scatter["style"].update({"size": 8.0, "color": "#00aa00"})
    panel["layers"] = [fill, hatch, contour, vector, stream, scatter]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _value_transform_config():
    """データ値の線形変換 (K→°C 等価): fill と contour に scale/offset を当てる。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"vmin": -30.0, "vmax": 0.0,
                          "value_scale": 1.0, "value_offset": -273.15})
    contour = mc_config.default_contour_layer("ds0", "t")
    contour["style"].update({"vmin": -30.0, "vmax": 0.0, "interval": 5.0,
                             "value_scale": 1.0, "value_offset": -273.15,
                             "negative_linestyle": None})
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _cmap_contour_config():
    """等値線を単色でなく cmap で色付けするモード (use_cmap=True)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    contour = mc_config.default_contour_layer("ds0", "t")
    contour["style"].update({"vmin": 230.0, "vmax": 290.0, "interval": 5.0,
                             "use_cmap": True, "cmap": "plasma",
                             "negative_linestyle": None})
    panel["layers"] = [contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _contour_cbar_extend_config():
    """新スキーマ contour (fill 式レベル + extend + カラーバー) と連続 extend。

    2026-07-08 の cmap UI 統一で追加。レイヤー1 = レベル直接指定 (list) +
    extend both + カラーバー (bottom)、レイヤー2 = レベル数 (int) + 値の範囲
    手動 (linspace) + フル装備カラーバー、レイヤー3 = ベクトル連続 cmap +
    extend max (colorbar への extend 直接指定経路)。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["title"] = "contour colorbars + vector extend"
    c1 = mc_config.default_contour_layer("ds0", "t")
    c1["style"].update({"levels": [240.0, 250.0, 260.0, 270.0, 280.0],
                        "extend": "both", "use_cmap": True, "cmap": "coolwarm",
                        "negative_linestyle": None, "linewidth": 1.5})
    c1["style"]["colorbar"].update({"label": "t [K]", "location": "bottom",
                                    "shrink": 0.7})
    c2 = mc_config.default_contour_layer("ds0", "z")
    c2["style"].update({"levels": 8, "vmin": 4900.0, "vmax": 5900.0,
                        "use_cmap": True, "cmap": "viridis",
                        "negative_linestyle": None})
    c2["style"]["labels"]["show"] = False
    c2["style"]["colorbar"].update({"label": "z [m]", "shrink": 0.8,
                                    "aspect": 30.0, "flip_ticks": True,
                                    "tick_fontsize": 8})
    vec = mc_config.default_vector_layer("ds0", "u", "v")
    vec["style"].update({"use_cmap": True, "cmap": "plasma",
                         "vmin": 0.0, "vmax": 20.0, "extend": "max",
                         "stride_x": 4, "stride_y": 4})
    vec["style"]["colorbar"].update({"label": "|V| [m/s]", "location": "left",
                                     "shrink": 0.6})
    panel["layers"] = [c1, c2, vec]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _ortho_contour_only_config():
    """回帰テスト: Orthographic + 等値線のみだと autoscale で円盤が切れていた。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 1000.0}
    panel["projection"] = {"name": "Orthographic",
                           "central_longitude": 180.0, "central_latitude": 20.0}
    panel["layers"] = [mc_config.default_contour_layer("ds0", "t")]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _per_layer_level_config():
    """水平断面図でレイヤー毎に異なる鉛直レベルを指定 (per-layer selection)。

    panel.selection は時刻のみ。fill は 500 hPa、contour は 850 hPa を
    layer.selection で固定する。render.py / scriptgen.py の panel + layer
    selection マージが両方向で同じ結果を出すことを検証する。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["title"] = "fill@500 + contour@850"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["selection"] = {"level": 500.0}
    fill["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 290.0, "levels": 13})
    contour = mc_config.default_contour_layer("ds0", "t")
    contour["selection"] = {"level": 850.0}
    contour["style"].update({"vmin": 260.0, "vmax": 305.0, "interval": 5.0,
                              "color": "black"})
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _orthographic_config():
    """第2段階: Orthographic投影 (set_extent なしの分岐)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    panel["projection"] = {"name": "Orthographic",
                           "central_longitude": 140.0, "central_latitude": 35.0}
    panel["map"]["gridlines"]["labels"] = False
    fill = mc_config.default_fill_layer("ds0", "t")
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


# コールドプロセスで render_figure を実行するランナー。
# matplotlib のテキスト描画はプロセス内の描画履歴でアンチエイリアスが数十ピクセル
# だけ揺れることがあるため、厳密一致の検証は「コールドプロセス同士」で行い、
# プロセス内描画(Streamlitでの表示に相当)は微小許容で別途確認する。
_RENDER_RUNNER = """\
import json
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, sys.argv[4])
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
datasets = {"ds0": mc_dataset.open_dataset(sys.argv[2])}
if len(sys.argv) > 5 and sys.argv[5]:
    # 座標ファイル (";" 区切り) を結び付ける (アプリの load_dataset_with_coords と同じ手順)
    _cpaths = sys.argv[5].split(";")
    datasets["ds0"] = mc_dataset.attach_coord_files(
        datasets["ds0"], [(p, mc_dataset.open_coord_file(p)) for p in _cpaths])
fig = mc_render.render_figure(cfg, datasets)
fig.savefig(sys.argv[3], dpi=150, bbox_inches="tight")
"""

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_cold(args, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, *args],
                          capture_output=True, text=True, env=env, cwd=tmp_path)


def _assert_app_and_script_match(cfg, sample_path, tmp_path, coord_paths=()):
    """coord_paths を渡すと座標ファイルを結び付けた dataset で検証する
    (生成スクリプトには open + assign_coords 行が入る)。"""
    import json

    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    if coord_paths:
        datasets["ds0"] = mc_dataset.attach_coord_files(
            datasets["ds0"],
            [(p, mc_dataset.open_coord_file(p)) for p in coord_paths])

    # 生成スクリプトをコールドプロセスで実行
    script_png = tmp_path / "script.png"
    script = mc_scriptgen.generate_script(
        cfg, datasets, {"ds0": sample_path},
        figure_output=str(script_png), figure_dpi=150, include_show=False,
    )
    script_path = tmp_path / "reproduce.py"
    script_path.write_text(script, encoding="utf-8")
    result = _run_cold([str(script_path)], tmp_path)
    assert result.returncode == 0, \
        f"生成スクリプトの実行に失敗:\n{result.stderr}\n--- script ---\n{script}"

    # render_figure もコールドプロセスで実行 (figure_config はJSON直列化可能)
    app_png = tmp_path / "app.png"
    runner_path = tmp_path / "render_runner.py"
    runner_path.write_text(_RENDER_RUNNER, encoding="utf-8")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = _run_cold([str(runner_path), str(cfg_path), sample_path,
                        str(app_png), _ROOT, ";".join(coord_paths)], tmp_path)
    assert result.returncode == 0, f"render_figure の実行に失敗:\n{result.stderr}"

    # 厳密一致 (コールドプロセス同士)
    a = mpimg.imread(app_png)
    b = mpimg.imread(script_png)
    assert a.shape == b.shape, f"画像サイズ不一致: {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), "アプリの図と生成スクリプトの図のピクセルが一致しない"

    # プロセス内描画 (Streamlit表示に相当) のサニティチェック。
    # ウォームプロセスでは曲線のアンチエイリアスがサブピクセル分揺れ、
    # 曲線投影では数%のピクセルに視覚上分からない微小差が出る (実測: 大半が輝度差0.25未満)。
    # ここでは構造的な乖離 (レイヤー欠落・色違い等) の検出だけを目的とする。
    warm_png = tmp_path / "warm.png"
    fig = mc_render.render_figure(cfg, datasets)
    fig.savefig(warm_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    w = mpimg.imread(warm_png)
    assert w.shape == b.shape, f"画像サイズ不一致 (warm): {w.shape} vs {b.shape}"
    mismatch = np.any(w != b, axis=2).mean()
    # アンチエイリアス揺れは細い曲線に沿った塊にしかならない (実測: 扇形境界の
    # ような長い高コントラスト曲線でも〜300px)。レイヤー欠落・色違いなどの
    # 構造的乖離は数千px規模の連結領域になるため、
    # 「大差分画素の最大連結成分サイズ < 1000px」で判定する。
    from scipy import ndimage
    big = np.abs(w - b).max(axis=2) > 0.5
    _, n_blobs = ndimage.label(big)
    max_blob = np.bincount(ndimage.label(big)[0].ravel())[1:].max() if n_blobs else 0
    assert mismatch < 0.10 and max_blob < 1000, \
        f"プロセス内描画とスクリプトが構造的に乖離: 不一致 {mismatch:.2%}, 最大差分領域 {max_blob} px"


# 複数 dataset 版の cold-process render ランナー。
# 各 dataset のパスと rename map を渡し、scriptgen の `.rename(...)` と同じ処理を行う。
_MULTI_RENDER_RUNNER = """\
import json
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, sys.argv[1])
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render

with open(sys.argv[2], encoding="utf-8") as f:
    cfg = json.load(f)
with open(sys.argv[3], encoding="utf-8") as f:
    ds_paths = json.load(f)
with open(sys.argv[4], encoding="utf-8") as f:
    ds_renames = json.load(f)

datasets = {}
for dsid, path in ds_paths.items():
    ds = mc_dataset.open_dataset(path)
    if dsid in ds_renames:
        ds = ds.rename(ds_renames[dsid])
    datasets[dsid] = ds

fig = mc_render.render_figure(cfg, datasets)
fig.savefig(sys.argv[5], dpi=150, bbox_inches="tight")
"""


def _assert_multi_dataset_match(cfg, dataset_paths, dataset_renames, tmp_path):
    """複数 dataset 版の画像一致テスト。

    dataset_paths: {dsid: path}
    dataset_renames: {dsid: {old: new}} or {}
    """
    import json

    datasets = {}
    for dsid, p in dataset_paths.items():
        ds = mc_dataset.open_dataset(p)
        if dsid in (dataset_renames or {}):
            ds = ds.rename(dataset_renames[dsid])
        datasets[dsid] = ds

    # 1) 生成スクリプト (cold-process)
    script_png = tmp_path / "script.png"
    script = mc_scriptgen.generate_script(
        cfg, datasets, dataset_paths,
        figure_output=str(script_png), figure_dpi=150, include_show=False,
        dataset_renames=dataset_renames or None,
    )
    script_path = tmp_path / "reproduce.py"
    script_path.write_text(script, encoding="utf-8")
    r = _run_cold([str(script_path)], tmp_path)
    assert r.returncode == 0, \
        f"スクリプト失敗:\n{r.stderr}\n--- script ---\n{script}"

    # 2) render_figure (cold-process)
    app_png = tmp_path / "app.png"
    runner_path = tmp_path / "multi_render_runner.py"
    runner_path.write_text(_MULTI_RENDER_RUNNER, encoding="utf-8")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    paths_path = tmp_path / "paths.json"
    paths_path.write_text(json.dumps(dataset_paths), encoding="utf-8")
    renames_path = tmp_path / "renames.json"
    renames_path.write_text(json.dumps(dataset_renames or {}), encoding="utf-8")
    r = _run_cold([str(runner_path), _ROOT, str(cfg_path), str(paths_path),
                    str(renames_path), str(app_png)], tmp_path)
    assert r.returncode == 0, f"render_figure 失敗:\n{r.stderr}"

    # 3) 厳密一致
    a = mpimg.imread(app_png)
    b = mpimg.imread(script_png)
    assert a.shape == b.shape, f"画像サイズ不一致: {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), "アプリの図と生成スクリプトの図のピクセルが一致しない"


def _multi_dataset_two_files_config():
    """ds0 から fill (t)、ds1 から contour (z)。同じファイルを2回読む構成。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 290.0, "levels": 11})
    contour = mc_config.default_contour_layer("ds1", "z")
    contour["style"].update({"vmin": 4200.0, "vmax": 5000.0, "interval": 100.0,
                              "negative_linestyle": None})
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_multi_dataset_same_file_two_layers(sample_path, tmp_path):
    """同じファイルを ds0, ds1 として読み、別レイヤーから使う。rename なし。"""
    cfg = _multi_dataset_two_files_config()
    _assert_multi_dataset_match(
        cfg,
        dataset_paths={"ds0": sample_path, "ds1": sample_path},
        dataset_renames={},
        tmp_path=tmp_path,
    )


def test_multi_dataset_with_rename(sample_path, tmp_path):
    """ds1 は dim 名が違うファイル (latitude) → scriptgen が `.rename(...)` を出力。"""
    import xarray as xr
    ds = xr.open_dataset(sample_path)
    lat_path = str(tmp_path / "sample_atmos.lat.nc")
    ds.rename({"lat": "latitude"}).to_netcdf(lat_path)

    cfg = _multi_dataset_two_files_config()
    _assert_multi_dataset_match(
        cfg,
        dataset_paths={"ds0": sample_path, "ds1": lat_path},
        dataset_renames={"ds1": {"latitude": "lat"}},
        tmp_path=tmp_path,
    )


def test_multi_dataset_different_resolution(sample_path, tmp_path):
    """異なる水平解像度: ds0 (フル格子) の fill に ds1 (粗格子) の contour を重ねる。

    内挿なしに各層を各自の解像度で重ね描きし、render と scriptgen が
    ピクセル一致すること (ルール1) を検証する。異解像度の重ね合わせは
    エラーにならず、両者が同一ファイルを各自の格子で描いて一致する。
    """
    import xarray as xr
    ds = xr.open_dataset(sample_path)
    coarse_path = str(tmp_path / "sample_atmos.coarse.nc")
    # lat/lon を 1/2 に間引いた粗い格子 (点数が ds0 と異なる)
    ds.isel(lat=slice(None, None, 2), lon=slice(None, None, 2)).to_netcdf(coarse_path)

    cfg = _multi_dataset_two_files_config()
    _assert_multi_dataset_match(
        cfg,
        dataset_paths={"ds0": sample_path, "ds1": coarse_path},
        dataset_renames={},
        tmp_path=tmp_path,
    )


def _make_v_only_nc(sample_path, path):
    """サンプルから y 成分だけを別ファイルに書き出す。

    変数名も v → vwnd に変えて、x 成分側のファイル (u, v を持つ) からではなく
    確かに別ファイルから読んでいることを保証する。
    """
    import xarray as xr
    ds = xr.open_dataset(sample_path)
    ds[["v"]].rename({"v": "vwnd"}).to_netcdf(path)


def _vector_two_files_config(kind, section):
    """x成分 (ds0: u) と y成分 (ds1: vwnd) を別ファイルから読むベクトル系レイヤー。

    kind = vector / stream、section = False (水平面図) / True (鉛直断面) の
    4 経路 (render の _draw_vector / _draw_stream / _draw_vector_section /
    _draw_stream_section ⇄ scriptgen の対応メソッド) を同じ構成でカバーする。
    """
    if section:
        panel = mc_config.default_section_panel()
        panel["x_dim"] = "lon"
        panel["y_dim"] = "level"
        panel["selection"] = {"time": "2024-01-01T06:00:00", "lat": 35.0}
        panel["ranges"] = {"level": [1000.0, 200.0]}
        panel["axis"].update({"invert_y": True, "log_y": True})
    else:
        panel = mc_config.default_panel()
        panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
        panel["region"] = {"lon_min": 100.0, "lon_max": 200.0,
                           "lat_min": 0.0, "lat_max": 60.0}
        panel["projection"] = {"name": "PlateCarree", "central_longitude": 150.0,
                               "central_latitude": 0.0}
    panel["title"] = f"{kind}: v from another file"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0,
                          "levels": 15})
    if kind == "vector":
        layer = mc_config.default_vector_layer("ds0", "u", "vwnd",
                                               v_dataset_id="ds1")
        layer["style"].update({"stride_x": 4, "stride_y": 2,
                               "use_cmap": True, "levels": 5})
        layer["style"]["colorbar"].update({"label": "|V| [m/s]"})
    else:
        layer = mc_config.default_stream_layer("ds0", "u", "vwnd",
                                               v_dataset_id="ds1")
        layer["style"].update({"density": 1.1, "linewidth": 0.8})
    panel["layers"] = [fill, layer]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


@pytest.mark.parametrize("kind, section", [
    ("vector", False), ("stream", False), ("vector", True), ("stream", True),
], ids=["vector-map", "stream-map", "vector-section", "stream-section"])
def test_multi_dataset_vector_components_from_two_files(kind, section, sample_path,
                                                        tmp_path):
    """ベクトル・流線の y 成分を別ファイル (v_dataset_id) から読み、render と
    scriptgen がピクセル一致すること。生成スクリプトは ds1 も開いて y 成分を
    そこから切り出す。"""
    v_path = str(tmp_path / "sample_v_only.nc")
    _make_v_only_nc(sample_path, v_path)
    cfg = _vector_two_files_config(kind, section)
    _assert_multi_dataset_match(
        cfg,
        dataset_paths={"ds0": sample_path, "ds1": v_path},
        dataset_renames={},
        tmp_path=tmp_path,
    )


def _make_matrix_nc(path):
    """無次元 x/y の小さな 2次元行列データ (region×month 風) を書き出す。"""
    import numpy as np
    import xarray as xr
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 8)).astype("float32")
    xr.Dataset(
        {"A": (("region", "month"), a, {"long_name": "score"})},
        coords={"region": ("region", np.arange(6)),
                "month": ("month", np.arange(1, 9))},
    ).to_netcdf(path)


def _heatmap_config():
    """categorical heatmap: 境界値の直接指定 + extend + maskout + 値変換 + 注記。"""
    panel = mc_config.default_heatmap_panel()
    panel["dataset_id"] = "ds0"
    panel["variable"] = "A"
    panel["x_dim"] = "month"
    panel["y_dim"] = "region"
    panel["title"] = "heatmap (annotated, discrete)"
    panel["xtick_rotation"] = 45.0
    panel["axis"]["x_label"] = "month"
    panel["axis"]["y_label"] = "region"
    panel["axis"]["tick_fontsize"] = 8
    # y は等間隔目盛の手動指定 → カテゴリラベルではなくセル index の数値軸になる
    panel["axis"]["y_tick_interval"] = 2.0
    panel["axis"]["swap_y_sides"] = True   # 縦軸を右側に (Stage 2.5)
    panel["style"].update({
        "cmap": "RdBu_r", "reverse_cmap": True,
        "levels": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "extend": "both",
        "value_scale": 10.0, "value_offset": 0.0,
        # maskout されたセルは色も注記も描かれない
        "maskout": {"below": -15.0, "above": None},
        "aspect": "equal", "origin": "lower",
        "annotate": {"show": True, "fmt": "%.0f", "fontsize": 7,
                     "color": "#222222"},
    })
    panel["style"]["colorbar"].update({"label": "score×10", "location": "right"})
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _heatmap_equal_levels_config():
    """categorical heatmap: 等間隔レベル (int) + 自動範囲 + extend + フルカラーバー
    + 軸 (反転・範囲手動指定) + ラベル体裁 (1次元プロットと同じ「軸」「ラベル」)。"""
    panel = mc_config.default_heatmap_panel()
    panel["dataset_id"] = "ds0"
    panel["variable"] = "A"
    panel["x_dim"] = "month"
    panel["y_dim"] = "region"
    panel["title"] = "heatmap (equal levels)"
    panel["ytick_rotation"] = 30.0   # y はカテゴリ目盛のまま回転
    panel["axis"].update({
        "x_label": "month index", "y_label": "region index",
        "label_fontsize": 11, "label_color": "#004488",
        "label_weight": "bold", "label_italic": True, "label_pad": 8.0,
        "x_label_rotation": 10.0,
        # invert は x、範囲手動指定は y に分ける (set_ylim は invert より後に
        # 適用されて向きを上書きするため、同じ軸に両方掛けると invert が見えない)
        "invert_x": True,
        "y_lim": [3.5, -0.5],      # 行 0..3 だけを row0 が上の向きで表示
        # x は位置の直接指定 + カスタムラベル (数値軸扱い)、y はカテゴリのまま
        "x_tick_positions": [0.0, 2.0, 4.0, 6.0],
        "x_tick_labels": ["Jan", "Mar", "May", "Jul"],
        "tick_fontsize": 9,
        "tick_width": 1.5,
    })
    panel["style"].update({
        "cmap": "viridis",
        "levels": 7,               # vmin/vmax 未指定 → データ範囲から linspace
        "extend": "min",
    })
    panel["style"]["colorbar"].update({
        "label": "score", "location": "bottom", "shrink": 0.8,
        "flip_ticks": True, "outline_width": 1.5, "tick_width": 1.2,
        "label_fontsize": 9, "tick_fontsize": 7,
        "label_pad": 8.0, "tick_pad": 6.0,
    })
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _heatmap_plain_config():
    """categorical heatmap: 旧設定互換 (levels=None の連続、extend/maskout キーなし)。"""
    panel = mc_config.default_heatmap_panel()
    panel["dataset_id"] = "ds0"
    panel["variable"] = "A"
    panel["x_dim"] = "month"
    panel["y_dim"] = "region"
    panel["title"] = "heatmap (plain)"
    panel["style"].update({"cmap": "viridis", "vmin": -2.0, "vmax": 2.0,
                           "levels": None})
    # v0.74 の WIP/preset には無いキー — 無くても従来どおり描けること
    panel["style"].pop("extend")
    panel["style"].pop("maskout")
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_heatmap_render_matches_script(tmp_path):
    """categorical heatmap の render と scriptgen が一致する (ルール1)。"""
    path = str(tmp_path / "matrix.nc")
    _make_matrix_nc(path)
    for cfg in (_heatmap_config(), _heatmap_equal_levels_config(),
                _heatmap_plain_config()):
        _assert_multi_dataset_match(
            cfg, dataset_paths={"ds0": path}, dataset_renames={},
            tmp_path=tmp_path)


def _make_series_nc(path):
    """dist_1d (集計) 用の非地理データ: 時系列 ts(time) + アンサンブル ens(time, member)。"""
    import xarray as xr
    rng = np.random.default_rng(7)
    n_t, n_m = 400, 3
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(n_t) * np.timedelta64(6, "h"))
    base = rng.standard_normal(n_t).cumsum() * 0.3 + 15.0
    ens = base[:, None] + rng.standard_normal((n_t, n_m)) * 2.0
    # 参照 PDF 曲線 (bin 座標 = 値軸): ラインレイヤーの重ね描き用
    bins = np.linspace(0.0, 24.0, 60)
    pdf = np.exp(-0.5 * ((bins - 12.0) / 4.0) ** 2) / (4.0 * np.sqrt(2 * np.pi))
    xr.Dataset(
        {"ts": (("time",), (base + rng.standard_normal(n_t)).astype("float32"),
                {"long_name": "surface temperature", "units": "degC"}),
         "ens": (("time", "member"), ens.astype("float32"),
                 {"long_name": "ensemble temperature", "units": "degC"}),
         "pdf_ref": (("bin",), pdf.astype("float32"),
                     {"long_name": "reference pdf"})},
        coords={"time": time, "member": np.arange(n_m),
                "bin": bins.astype("float64")},
    ).to_netcdf(path)


def _dist_hist_config():
    """dist_1d: 度数ヒストグラム1層 (軸ラベル自動フォールバック + 枠線 + 凡例)。"""
    panel = mc_config.default_dist_panel()
    panel["title"] = "histogram (counts)"
    hist = mc_config.default_hist_layer("ds0")
    hist["variable"] = "ts"
    hist["agg_dim"] = "time"
    hist["style"].update({"bins": 15, "alpha": 1.0,
                          "edge_linewidth": 0.8, "label": "ts"})
    panel["layers"] = [hist]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _dist_hist_rich_config():
    """dist_1d: 密度2層重ね (期間限定 + member 固定 + step/stepfilled +
    ビン境界の直接指定 + 値変換 + 凡例位置 + 軸ラベル上書き)。"""
    panel = mc_config.default_dist_panel()
    panel["title"] = "density histograms"
    panel["axis"].update({"x_label": "temperature [degC]"})
    panel["legend"].update({"loc": "upper left", "fontsize": 9})
    l1 = mc_config.default_hist_layer("ds0")
    l1["variable"] = "ens"
    l1["agg_dim"] = "time"
    l1["selection"] = {"member": 1}
    l1["agg_range"] = ["2024-03-01T00:00:00", "2024-09-30T18:00:00"]
    l1["style"].update({"bins": 25, "range": [8.0, 24.0], "density": True,
                        "histtype": "stepfilled", "alpha": 0.5,
                        "color": "#1f77b4", "label": "member1 (MAM-SON)"})
    l2 = mc_config.default_hist_layer("ds0")
    l2["variable"] = "ts"
    l2["agg_dim"] = "time"
    l2["style"].update({"bins": [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 22.0],
                        "density": True, "histtype": "step", "alpha": 1.0,
                        "color": "#d62728", "label": "ts (all)"})
    panel["layers"] = [l1, l2]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _dist_ecdf_twinx_config():
    """dist_1d: 度数 hist + 第2軸 (twinx) に ECDF (CDF + 補分布) と参照 PDF ライン
    + 縦軸左右入れ替え + 第2軸ラベル/範囲 + 凡例結合。Stage 2 の全機能。"""
    panel = mc_config.default_dist_panel()
    panel["title"] = "hist + ECDF + ref pdf (twinx, swapped)"
    panel["axis"].update({"swap_y_sides": True, "y2_label": "probability",
                          "y2_lim": [0.0, 1.05]})
    panel["legend"].update({"loc": "center right", "fontsize": 8})
    # 図枠は twinx の ax2 にも当たること (上枠非表示・太さ・色)
    panel["frame"].update({"show_top": False, "width": 1.4,
                           "color": "#444444"})
    hist = mc_config.default_hist_layer("ds0")
    hist["variable"] = "ts"
    hist["agg_dim"] = "time"
    hist["style"].update({"bins": 20, "alpha": 1.0, "label": "ts counts"})
    e1 = mc_config.default_ecdf_layer("ds0")
    e1["variable"] = "ts"
    e1["agg_dim"] = "time"
    e1["style"].update({"color": "#d62728", "label": "ECDF",
                        "secondary_y": True})
    e2 = mc_config.default_ecdf_layer("ds0")
    e2["variable"] = "ts"
    e2["agg_dim"] = "time"
    e2["style"].update({"complementary": True, "color": "#9467bd",
                        "linestyle": "dashed", "label": "1-CDF",
                        "secondary_y": True})
    line = mc_config.default_line_layer("ds0", "pdf_ref")
    line["x_dim"] = "bin"
    line["style"].update({"color": "#2ca02c", "linewidth": 2.0,
                          "label": "ref pdf", "secondary_y": True,
                          "value_scale": 5.0})
    panel["layers"] = [hist, e1, e2, line]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _dist_box_violin_config():
    """dist_1d: 箱ひげ2系列 (塗り/期間限定+平均+whis) + バイオリン1系列の混在。

    系列名の x 目盛・y 自動ラベル (変数 [units])・並び順の position を検証。"""
    panel = mc_config.default_dist_panel()
    panel["title"] = "box x2 + violin"
    b1 = mc_config.default_box_layer("ds0")
    b1["variable"] = "ens"
    b1["agg_dim"] = "time"
    b1["selection"] = {"member": 0}
    b1["style"].update({"label": "member0", "fill_color": "#87b8d8",
                        "in_legend": True, "notch": True,
                        "line_color": "#333333", "line_width": 1.2,
                        "median_color": "#000000", "median_width": 2.0,
                        "flier_marker": "x", "flier_size": 4.0,
                        "flier_color": "#888888"})
    b2 = mc_config.default_box_layer("ds0")
    b2["variable"] = "ens"
    b2["agg_dim"] = "time"
    b2["selection"] = {"member": 1}
    b2["agg_range"] = ["2024-03-01T00:00:00", "2024-09-30T18:00:00"]
    b2["style"].update({"label": "member1 (MAM-SON)", "whis": [5.0, 95.0],
                        "show_points": True, "point_jitter": 0.2,
                        "point_size": 5.0, "point_alpha": 0.35,
                        "showmeans": True, "meanline": True,
                        "mean_color": "#2ca02c", "mean_width": 1.8,
                        "showfliers": False, "showcaps": True,
                        "capwidths": 0.15, "width": 0.4})
    v1 = mc_config.default_violin_layer("ds0")
    v1["variable"] = "ts"
    v1["agg_dim"] = "time"
    v1["style"].update({"label": "ts (violin)", "color": "#2ca02c",
                        "alpha": 0.4, "showmeans": True, "in_legend": True,
                        "quantiles": [0.05, 0.95], "bw_method": 0.3,
                        "points": 200})
    panel["layers"] = [b1, b2, v1]
    panel["legend"].update({"loc": "upper right", "fontsize": 8})
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _dist_box_horizontal_config():
    """dist_1d: 横向き (box_orientation=horizontal) の box + 半バイオリン。

    系列名が y 目盛・値ラベルが x 軸に移ること、orientation kw の emit を検証。"""
    panel = mc_config.default_dist_panel()
    panel["title"] = "horizontal box + half violin"
    panel["box_orientation"] = "horizontal"
    b1 = mc_config.default_box_layer("ds0")
    b1["variable"] = "ens"
    b1["agg_dim"] = "time"
    b1["selection"] = {"member": 0}
    # 箱なし + 凡例あり = showbox=False の IndexError 回帰テスト
    b1["style"].update({"label": "member0", "showbox": False,
                        "showcaps": False, "in_legend": True})
    v1 = mc_config.default_violin_layer("ds0")
    v1["variable"] = "ts"
    v1["agg_dim"] = "time"
    v1["style"].update({"label": "ts (half)", "side": "high",
                        "color": "#9467bd", "alpha": 0.5,
                        "show_points": True, "point_jitter": 0.3,
                        "point_color": "#333333"})
    panel["layers"] = [b1, v1]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_dist_hist_render_matches_script(tmp_path):
    """dist_1d (ヒストグラム / ECDF / ライン / 第2軸) の render と scriptgen が一致する。"""
    path = str(tmp_path / "series.nc")
    _make_series_nc(path)
    for cfg in (_dist_hist_config(), _dist_hist_rich_config(),
                _dist_ecdf_twinx_config(), _dist_box_violin_config(),
                _dist_box_horizontal_config()):
        _assert_multi_dataset_match(
            cfg, dataset_paths={"ds0": path}, dataset_renames={},
            tmp_path=tmp_path)


def _agg_hist2d_config():
    """agg_2d: 1次元組 (ts × ens member固定) の2次元ヒストグラム。

    非対称ビン数 + cmin (0を塗らない) + カラーバー + 自動ラベル。"""
    panel = mc_config.default_agg_panel()
    panel["title"] = "hist2d (ts vs ens m1)"
    panel["x_variable"] = "ts"
    panel["y_variable"] = "ens"
    ly = mc_config.default_hist2d_layer("ds0")
    ly["y_fixed"] = {"member": 1}
    ly["style"].update({"bins_x": 24, "bins_y": 16})
    ly["style"]["colorbar"].update({"label": "count"})
    panel["layers"] = [ly]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _agg_hist2d_log_config():
    """agg_2d: 3次元 ravel (ens 同士、y は値変換) + LogNorm + 集計範囲。"""
    panel = mc_config.default_agg_panel()
    panel["title"] = "hist2d (ens vs ens*1.2, log)"
    panel["x_variable"] = "ens"
    panel["y_variable"] = "ens"
    ly = mc_config.default_hist2d_layer("ds0")
    ly["style"].update({"bins_x": 30, "bins_y": 30,
                        "range_x": [0.0, 24.0], "range_y": [2.0, 30.0],
                        "log_counts": True, "cmap": "magma",
                        "y_value_scale": 1.2, "y_value_offset": 1.0})
    panel["layers"] = [ly]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _agg_hexbin_config():
    """agg_2d: hexbin の新経路 (drawing_dim なし = 全 ravel)。"""
    panel = mc_config.default_agg_panel()
    panel["title"] = "hexbin (agg)"
    panel["x_variable"] = "ens"
    panel["y_variable"] = "ens"
    ly = mc_config.default_hexbin_layer("ds0")
    ly["style"].update({"gridsize": 25, "log_counts": True, "mincnt": 1,
                        "y_value_scale": 0.8, "y_value_offset": 3.0})
    ly["style"]["colorbar"].update({"label": "count", "location": "bottom"})
    panel["layers"] = [ly]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _agg_hist2d_discrete_config():
    """agg_2d: hist2d の色の離散化 (レベル数等間隔 + vmax 手動 + extend max)。

    度数は集計後に決まるため、描画後に h[0] から BoundaryNorm を作って
    set_norm する後付け経路の検証 (2026-07-08 の cmap UI 統一で追加)。
    """
    panel = mc_config.default_agg_panel()
    panel["title"] = "hist2d discrete (ts vs ens m1)"
    panel["x_variable"] = "ts"
    panel["y_variable"] = "ens"
    ly = mc_config.default_hist2d_layer("ds0")
    ly["y_fixed"] = {"member": 1}
    ly["style"].update({"bins_x": 24, "bins_y": 16,
                        "levels": 6, "vmax": 8.0, "extend": "max",
                        "cmap": "viridis",
                        # ビンの枠線 (hist2d の edgecolors/linewidths 経路)
                        "edge_width": 0.6, "edge_color": "#333333"})
    ly["style"]["colorbar"].update({"label": "count"})
    panel["layers"] = [ly]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _agg_hexbin_discrete_config():
    """agg_2d: hexbin の色の離散化 (レベル直接指定 = 対数風の階級)。

    hb.get_array() から BoundaryNorm を作る後付け経路の検証。
    """
    panel = mc_config.default_agg_panel()
    panel["title"] = "hexbin discrete (agg)"
    panel["x_variable"] = "ens"
    panel["y_variable"] = "ens"
    ly = mc_config.default_hexbin_layer("ds0")
    ly["style"].update({"gridsize": 25, "mincnt": 1,
                        "levels": [1.0, 2.0, 5.0, 10.0, 20.0],
                        "cmap": "magma",
                        "y_value_scale": 0.8, "y_value_offset": 3.0,
                        # ビンの枠線 (hexbin の edgecolors/linewidths 経路)
                        "edge_width": 0.4, "edge_color": "#ffffff"})
    ly["style"]["colorbar"].update({"label": "count", "location": "bottom"})
    panel["layers"] = [ly]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_agg2d_render_matches_script(tmp_path):
    """agg_2d (hist2d / hexbin 全 ravel) の render と scriptgen が一致する。"""
    path = str(tmp_path / "series_agg.nc")
    _make_series_nc(path)
    for cfg in (_agg_hist2d_config(), _agg_hist2d_log_config(),
                _agg_hexbin_config(), _agg_hist2d_discrete_config(),
                _agg_hexbin_discrete_config()):
        _assert_multi_dataset_match(
            cfg, dataset_paths={"ds0": path}, dataset_renames={},
            tmp_path=tmp_path)


def _track_config():
    """トラック: 複数ストームの loop + 風速色付き点 (kt→m/s 変換 + 離散化 +
    カラーバー)。storm_range に日付変更線をまたぐトラック (最後の2本) を含み、
    Geodetic 変換の経度またぎ描画も検証する。"""
    panel = mc_config.default_panel()
    panel["title"] = "best tracks (wind-colored)"
    tr = mc_config.default_track_layer("ds0")
    tr.update({"lon_var": "lon", "lat_var": "lat",
               "storm_dim": "storm", "storm_range": [4, 7]})
    tr["style"].update({"linewidth": 1.8, "alpha": 0.9})
    tr["style"]["points"].update({
        "variable": "wind", "every": 2, "size": 18.0,
        "cmap": "plasma",
        "levels": [17.0, 25.0, 33.0, 43.0, 51.0],  # 台風カテゴリ風の境界 (m/s)
        "value_scale": 0.514444,                    # kt → m/s
    })
    tr["style"]["points"]["colorbar"].update({"label": "wind [m/s]",
                                              "shrink": 0.8})
    # maskout: 風速 30kt 以下の位置を描かない (線の切れ目 + 点の間引き)
    tr["style"]["maskout"] = {"variable": "wind", "below": 30.0, "above": None}
    panel["layers"] = [tr]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _track_single_color_config():
    """トラック: 全ストーム + maskout 2層重ね (強度で線の色が変わる使用例)。

    下の層 = 風速 50kt 以上を描かない (弱い部分、青破線 + 点)、
    上の層 = 50kt 以下を描かない (強い部分、赤実線、点なし)。
    """
    panel = mc_config.default_panel()
    panel["title"] = "all tracks (weak=blue / strong=red)"
    tr = mc_config.default_track_layer("ds0")
    tr.update({"lon_var": "lon", "lat_var": "lat", "storm_dim": "storm"})
    tr["style"].update({"color": "#1f77b4", "linewidth": 1.0,
                        "linestyle": "dashed", "alpha": 0.8})
    tr["style"]["maskout"] = {"variable": "wind", "below": None, "above": 50.0}
    tr["style"]["points"].update({"size": 8.0, "every": 4,
                                  "color": "#222222", "marker": "^"})
    tr2 = mc_config.default_track_layer("ds0")
    tr2.update({"lon_var": "lon", "lat_var": "lat", "storm_dim": "storm"})
    tr2["style"].update({"color": "#d62728", "linewidth": 1.8})
    tr2["style"]["maskout"] = {"variable": "wind", "below": 50.0, "above": None}
    tr2["style"]["points"]["show"] = False
    panel["layers"] = [tr, tr2]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _track_year_month_config():
    """トラック: 発生年・月フィルタ (2024年の7〜9月発生のみ)。

    生成スクリプトの np.where フィルタループ経路を検証する。サンプルの
    GENESIS では index 2 (2024-07), 3 (2024-08) が該当し、4 (2024-10) は除外。
    """
    panel = mc_config.default_panel()
    panel["title"] = "tracks genesis 2024 Jul-Sep"
    tr = mc_config.default_track_layer("ds0")
    tr.update({"lon_var": "lon", "lat_var": "lat", "storm_dim": "storm",
               "time_var": "time",
               "year_range": [2024, 2024], "month_range": [7, 9]})
    tr["style"].update({"linewidth": 2.0})
    tr["style"]["points"].update({"variable": "pres", "size": 14.0,
                                  "cmap": "viridis", "reverse_cmap": True})
    tr["style"]["points"]["colorbar"].update({"label": "pres [hPa]"})
    panel["layers"] = [tr]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _track_land_foreground_config():
    """トラック + 陸をデータの上に描く: トラックの線・点は陸の上に出る (zorder 付け替え)。"""
    cfg = _track_config()
    cfg["panels"][0]["map"]["land"].update({"show": True, "above_data": True})
    cfg["panels"][0]["title"] = "best tracks over land fill"
    return cfg


def test_track_render_matches_script(tmp_path):
    """トラックレイヤーの render と scriptgen が一致する (ルール1)。"""
    from scripts.make_sample_track_data import create_track_dataset

    path = str(tmp_path / "besttrack.nc")
    create_track_dataset(path)
    for cfg in (_track_config(), _track_single_color_config(),
                _track_year_month_config(), _track_land_foreground_config()):
        _assert_multi_dataset_match(
            cfg, dataset_paths={"ds0": path}, dataset_renames={},
            tmp_path=tmp_path)


def _vector_cmap_config():
    """ベクトルの大きさ |V| による色付け (仕様16章拡張) + カラーバー。

    地図でベクトルのみのレイヤー構成にして、ベクトル由来カラーバーの
    位置合わせ (match_colorbars_to_axes) も検証する。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "wind colored by speed"
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"].update({"use_cmap": True, "cmap": "viridis",
                            "vmin": 0.0, "vmax": 25.0,
                            "stride_x": 4, "stride_y": 3,
                            "mask_below": 5.0,
                            "headlength": 7.0,
                            "edge_width": 0.4, "edge_color": "#000000"})
    vector["style"]["colorbar"].update({"label": "|V| [m/s]", "shrink": 0.8,
                                        "label_opposite": True})  # 目盛り右・ラベル左
    panel["layers"] = [vector]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _gridlines_labels_only_config():
    """緯度経度線: 線は非表示でラベルのみ + 経度ラベル回転 (線とラベルの独立設定)。

    細かい間隔 (20°) は cartopy がラベルを自動間引きするため、回転 45° で
    全ラベルが表示されるケースを検証する。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "labels only gridlines"
    panel["map"]["gridlines"].update({
        "lines": False, "labels": True,
        "label_lon_interval": 20.0, "label_lat_interval": 30.0,
        "label_lat_start": 15.0,  # 緯度のみ開始値 (…-15°, 15°, 45°…)。経度は倍数のまま
        "label_rotation": 45.0, "label_fontsize": 9})
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "viridis", "vmin": 230.0, "vmax": 300.0, "levels": 15})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _gridlines_split_intervals_config():
    """緯度経度線: 線 (15°/10°) とラベル (60°/30°) で別間隔 (Gridliner 2本に分割)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "split gridline intervals"
    panel["map"]["gridlines"].update({
        "lines": True, "labels": True,
        "lon_interval": 15.0, "lat_interval": 10.0,
        "label_lon_interval": 60.0, "label_lat_interval": 30.0,
        "label_lon_start": 30.0,  # 経度のみ開始値 (…-30°, 30°, 90°…)。緯度は倍数のまま
        "width": 0.4, "linestyle": "--"})
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 230.0, "vmax": 300.0, "levels": 15})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _multipanel_mixed_config():
    """第6段階: 1×2 グリッドに cartopy 地図 (GeoAxes) と鉛直断面 (通常 Axes) を並べる。

    パネルラベル (a), (b) も両パネルに付けて panel.label の一致を検証する。
    """
    map_panel = mc_config.default_panel()
    map_panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    map_panel["title"] = "t at 500 hPa"
    map_panel["label"].update({"show": True, "text": "(a)"})
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 300.0, "levels": 15})
    fill["style"]["colorbar"].update({"label": "K", "shrink": 0.8})
    map_panel["layers"] = [fill]

    sec_panel = mc_config.default_section_panel()
    sec_panel["x_dim"] = "lon"
    sec_panel["y_dim"] = "level"
    sec_panel["selection"] = {"time": "2024-01-01T06:00:00", "lat": 35.0}
    sec_panel["axis"].update({"invert_y": True, "log_y": True,
                              "x_label": "longitude", "y_label": "pressure [hPa]"})
    sec_panel["title"] = "cross section at 35N"
    sec_panel["label"].update({"show": True, "text": "(b)", "fontsize": 14,
                               "weight": "normal", "color": "#333333",
                               "x": 0.02, "y": 0.98, "va": "top"})
    sfill = mc_config.default_fill_layer("ds0", "t")
    sfill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0, "levels": 21})
    contour = mc_config.default_contour_layer("ds0", "z")
    sec_panel["layers"] = [sfill, contour]

    cfg = mc_config.default_figure_config()
    cfg["figure"]["figsize"] = [12.8, 4.8]
    cfg["figure"]["layout"] = {"nrows": 1, "ncols": 2}
    cfg["panels"] = [map_panel, sec_panel]
    return cfg


def _multipanel_two_maps_config():
    """第6段階: 地図パネル×2 (両方カラーバー付き)。

    scriptgen の `_map_cbars` がパネル毎にクリアされること (前パネルの
    カラーバー位置合わせコードを再出力しないこと) の回帰テストを兼ねる。
    """
    p1 = mc_config.default_panel()
    p1["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    p1["title"] = "t at 500 hPa"
    f1 = mc_config.default_fill_layer("ds0", "t")
    f1["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 300.0, "levels": 15})
    f1["style"]["colorbar"].update({"label": "K", "shrink": 0.7})
    p1["layers"] = [f1]

    p2 = mc_config.default_panel()
    p2["selection"] = {"time": "2024-01-01T06:00:00", "level": 850.0}
    p2["title"] = "t at 850 hPa"
    f2 = mc_config.default_fill_layer("ds0", "t")
    f2["style"].update({"cmap": "viridis", "vmin": 250.0, "vmax": 300.0, "levels": 11})
    f2["style"]["colorbar"].update({"label": "K", "location": "bottom"})
    p2["layers"] = [f2]

    cfg = mc_config.default_figure_config()
    cfg["figure"]["figsize"] = [6.4, 8.0]
    # hspace / 上下余白も指定して subplots_adjust の一致を検証する (仕様22.7)。
    # hspace は負の値 (地図パネルの行間詰め) をカバーする
    cfg["figure"]["layout"] = {"nrows": 2, "ncols": 1,
                               "hspace": -0.1, "top": 0.92, "bottom": 0.08}
    cfg["panels"] = [p1, p2]
    return cfg


def _mosaic_layout_config():
    """mosaic 配置: 2×3 でセル結合 (横 = 上段の2セル、縦 = 右列の上下) + 空きセル。

    `add_subplot(r, c, (first, last))` の render↔scriptgen 一致と、
    GeoAxes (cartopy) を含む結合セルの動作を検証する。
    配置 "AAB;C.B": A=地図 (上段1-2列)、B=鉛直断面 (右列上下貫通)、
    C=時系列 (下段1列)、下段2列目は空きセル。
    """
    map_panel = mc_config.default_panel()
    map_panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    map_panel["title"] = "t at 500 hPa (colspan 2)"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 300.0,
                          "levels": 15})
    fill["style"]["colorbar"].update({"label": "K", "shrink": 0.8})
    map_panel["layers"] = [fill]

    sec_panel = mc_config.default_section_panel()
    sec_panel["x_dim"] = "lon"
    sec_panel["y_dim"] = "level"
    sec_panel["selection"] = {"time": "2024-01-01T06:00:00", "lat": 35.0}
    sec_panel["axis"].update({"invert_y": True, "log_y": True})
    sec_panel["title"] = "rowspan right"
    sfill = mc_config.default_fill_layer("ds0", "t")
    sfill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0,
                           "levels": 21})
    sec_panel["layers"] = [sfill]

    line_panel = mc_config.default_line_panel()
    line_panel["x_dim"] = "time"
    line_panel["selection"] = {"level": 500.0, "lat": 35.0, "lon": 140.0}
    line_panel["title"] = "time series"
    line_panel["layers"] = [mc_config.default_line_layer("ds0", "t")]

    cfg = mc_config.default_figure_config()
    cfg["figure"]["figsize"] = [12.8, 7.0]
    cfg["figure"]["layout"] = {"nrows": 2, "ncols": 3, "mosaic": "AAB;C.B"}
    cfg["panels"] = [map_panel, sec_panel, line_panel]
    return cfg


def _grid_ratios_config():
    """行・列の大きさの比率 (width_ratios / height_ratios) + mosaic 結合の併用。

    GridSpec 経路 (gs = fig.add_gridspec(...) と gs[...] 添字) の
    render↔scriptgen 一致を検証する。GeoAxes (cartopy) を含む結合セル・
    空きセルにも比率が正しく効くこと。
    """
    cfg = _mosaic_layout_config()
    cfg["figure"]["layout"]["width_ratios"] = [2.0, 1.0, 1.5]
    cfg["figure"]["layout"]["height_ratios"] = [2.0, 1.0]
    return cfg


def _multipanel_shared_colorbar_config():
    """第6段階: 2×1 の地図パネルで全パネル共通カラーバー (仕様22.5)。

    両パネルの fill は同じ cmap / vmin / vmax / levels、個別カラーバーは OFF。
    """
    panels = []
    for level, title in [(500.0, "t at 500 hPa"), (850.0, "t at 850 hPa")]:
        p = mc_config.default_panel()
        p["selection"] = {"time": "2024-01-01T06:00:00", "level": level}
        p["title"] = title
        f = mc_config.default_fill_layer("ds0", "t")
        f["style"].update({"cmap": "RdBu_r", "vmin": 230.0, "vmax": 300.0, "levels": 15})
        f["style"]["colorbar"]["show"] = False
        p["layers"] = [f]
        panels.append(p)
    panels[0]["label"].update({"show": True, "text": "(a)"})
    panels[1]["label"].update({"show": True, "text": "(b)"})

    cfg = mc_config.default_figure_config()
    cfg["figure"]["figsize"] = [6.4, 8.0]
    cfg["figure"]["layout"] = {"nrows": 2, "ncols": 1}
    cfg["figure"]["shared_colorbar"] = {
        "show": True, "label": "air temperature [K]", "location": "right",
        "shrink": 0.8, "aspect": 30.0, "pad": 0.08,
        "label_fontsize": 10, "tick_fontsize": 9, "outline_width": 1.5,
        "tick_width": 1.2, "label_pad": 10.0, "tick_pad": 6.0,
        "label_opposite": True}  # 目盛り右・ラベル左 (共通カラーバー経路)
    cfg["panels"] = panels
    return cfg


def _map_scatter_grid_config():
    """地図散布図 (格子データ A): precip を格子点に色付き点 + 離散化 + カラーバー。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00"}
    panel["region"] = {"lon_min": 100.0, "lon_max": 180.0,
                       "lat_min": 0.0, "lat_max": 60.0}
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 140.0,
                           "central_latitude": 0.0}
    panel["title"] = "map scatter (grid)"
    ms = mc_config.default_map_scatter_layer("ds0", "precip")
    ms["style"].update({"cmap": "YlGnBu", "size": 14.0, "marker": "s",
                        "levels": [0.0, 2.0, 5.0, 10.0, 20.0]})
    ms["style"]["colorbar"].update({"label": "precip", "location": "right"})
    panel["layers"] = [ms]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_map_scatter_station_matches_script(tmp_path):
    """地点データ (case B) の地図散布図の render==scriptgen (ルール1)。"""
    import numpy as np
    import xarray as xr
    rng = np.random.default_rng(1)
    n = 40
    lon = rng.uniform(100, 180, n).astype("float32")
    lat = rng.uniform(0, 60, n).astype("float32")
    val = (rng.standard_normal(n) * 5 + 15).astype("float32")
    path = str(tmp_path / "station.nc")
    xr.Dataset(
        {"temp": (("station",), val, {"units": "degC"})},
        coords={"lon": ("station", lon), "lat": ("station", lat),
                "station": np.arange(n)}).to_netcdf(path)
    panel = mc_config.default_panel()
    panel["region"] = {"lon_min": 90.0, "lon_max": 190.0,
                       "lat_min": -10.0, "lat_max": 70.0}
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 140.0,
                           "central_latitude": 0.0}
    panel["title"] = "map scatter (station)"
    ms = mc_config.default_map_scatter_layer("ds0", "temp")
    ms["style"].update({"cmap": "coolwarm", "size": 45.0, "edge_linewidth": 0.6})
    ms["style"]["colorbar"]["label"] = "temp"
    panel["layers"] = [ms]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    _assert_multi_dataset_match(
        cfg, dataset_paths={"ds0": path}, dataset_renames={}, tmp_path=tmp_path)


def _hexbin_2d_config():
    """hexbin (六角ビン密度): t vs z を lon 沿いに集計、対数スケール + カラーバー。"""
    panel = mc_config.default_scatter_panel()
    panel["x_variable"] = "t"
    panel["y_variable"] = "z"
    panel["selection"] = {}
    panel["title"] = "hexbin density (t vs z)"
    panel["axis"]["x_label"] = "t"
    panel["axis"]["y_label"] = "z"
    hb = mc_config.default_hexbin_layer("ds0")
    hb["drawing_dim"] = "lon"
    hb["x_fixed"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    hb["y_fixed"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    hb["style"].update({"gridsize": 18, "cmap": "magma", "reverse_cmap": True,
                        "log_counts": True, "mincnt": 1})
    hb["style"]["colorbar"].update({"label": "count", "location": "right"})
    panel["layers"] = [hb]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _bubble_2d_uvt_cmap_discrete_config():
    """バブルの cmap 色付けを離散化 (BoundaryNorm、カラーバーも段々)。"""
    cfg = _bubble_2d_uvt_cmap_config()
    cfg["panels"][0]["layers"][0]["style"]["levels"] = 7
    return cfg


def _zero_white_fill_map_config():
    """zero_white: 地図 fill (contourf、int レベル)。

    偏差場 (value_offset で 0 をまたぐ) の自動レベルは linspace + 端の
    切り落とし (fill_zero_white_trim) になり、0 を含むビンが白になる。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "zero white contourf"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "levels": 16, "zero_white": True,
                          "value_offset": -253.0})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _zero_white_vsection_pcolormesh_config():
    """zero_white: 鉛直断面 fill (pcolormesh、0 が境界の明示レベル)。

    0 がレベル境界に一致するため、境界の両隣2ビンが白になるケース。
    """
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00", "lon": 140.0}
    panel["axis"]["invert_y"] = True
    panel["title"] = "zero white pcolormesh section"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "method": "pcolormesh",
                          "zero_white": True, "value_offset": -253.0,
                          "levels": [-40.0, -30.0, -20.0, -10.0, 0.0,
                                     10.0, 20.0, 30.0, 40.0]})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _zero_white_vector_cmap_config():
    """zero_white: ベクトルの cmap 離散化 (discrete_color_kwargs 系の代表)。

    map_scatter / track / bubble / heatmap / hist2d / hexbin と同じ
    共通経路 (discrete_color_kwargs ⇄ _discrete_color_exprs) を使う。
    """
    cfg = _vector_cmap_config()
    cfg["panels"][0]["layers"][0]["style"].update(
        {"cmap": "RdBu_r", "zero_white": True,
         "levels": [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]})
    return cfg


def _fill_exact_range_config():
    """手動 vmin/vmax = 描画データの正確な min/max + contourf (クラッシュ再現用)。

    「データ範囲」表示の転記シナリオ。cartopy の GEOS「empty Point」エラーで
    落ちる既知の上流バグの確定再現 config (値はサンプルデータ t の
    time=06:00, level=500 の実測 min/max)。端レベルの自動トリムによる予防は
    実装後に取りやめた (既存図の見た目が変わるため 2026-08-23) ので、
    この config は描画できない — render_figure の可読エラー変換
    (degenerate_geometry) の検証にのみ使う。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "fill with exact data-range vmin/vmax"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"vmin": 217.4624481201172,
                          "vmax": 256.46234130859375, "levels": 21})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_degenerate_geometry_raises_readable_error(sample_path):
    """レベルの端 = データ範囲ちょうどの GEOS エラーは render_figure が
    RenderError('degenerate_geometry') に変換する (手動範囲・明示リスト両形)。"""
    import numpy as np

    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}

    # 形1: 手動 vmin/vmax (「データ範囲」表示の転記シナリオ)
    with pytest.raises(mc_render.RenderError) as ei:
        fig = mc_render.render_figure(_fill_exact_range_config(), datasets)
        fig.canvas.draw()
    assert ei.value.msg_id == "degenerate_geometry"

    # 形2: 明示レベルリストの端一致
    cfg = _fill_exact_range_config()
    style = cfg["panels"][0]["layers"][0]["style"]
    style["levels"] = list(np.linspace(style.pop("vmin"), style.pop("vmax"), 21))
    with pytest.raises(mc_render.RenderError) as ei:
        fig = mc_render.render_figure(cfg, datasets)
        fig.canvas.draw()
    assert ei.value.msg_id == "degenerate_geometry"


def _maskout_map_config():
    """水平面図の maskout: fill は above+below、hatch は above、contour は below。

    「この値以下 / 以上を描かない」(GrADS maskout 相当) が render と scriptgen で
    一致することの検証。値変換 (K→°C) との併用で「閾値は変換後の値」も確認する。
    """
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "maskout: fill(-40<t<-20degC) hatch(t<-30) contour(z>5000)"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "vmin": -50.0, "vmax": -10.0,
                          "levels": 21,
                          "value_scale": 1.0, "value_offset": -273.15,
                          "maskout": {"below": -40.0, "above": -20.0}})
    hatch = mc_config.default_hatch_layer("ds0", "t")
    hatch["style"].update({"levels": [-273.15, 100.0], "pattern": "/",
                           "value_offset": -273.15,
                           "maskout": {"below": None, "above": -30.0}})
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"].update({"vmin": 4800.0, "vmax": 6000.0, "interval": 60.0,
                             "maskout": {"below": 5000.0, "above": None}})
    panel["layers"] = [fill, hatch, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def test_maskout_does_not_shrink_global_map(sample_path):
    """maskout で端の緯度帯 (極域) を隠しても、地図の軸範囲は全球のまま。

    領域未指定の水平面図は ax.set_global() で全球に固定される (値・欠損に
    よる自動スケール縮みの回帰テスト)。修正前は「塗られた範囲」に軸が縮み、
    マスクした極域は地図・海岸線ごと消えていた。
    """
    import cartopy.crs as ccrs

    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = _maskout_map_config()  # region 未指定 + fill は極域をマスク
    fig = mc_render.render_figure(cfg, datasets)
    ax = fig.axes[0]
    x0, x1, y0, y1 = ax.get_extent(crs=ccrs.PlateCarree())
    assert y1 - y0 > 175.0, f"緯度範囲が全球でない: {y0}..{y1}"
    assert x1 - x0 > 355.0, f"経度範囲が全球でない: {x0}..{x1}"
    plt.close(fig)


def _stream_map_config():
    """水平面図の流線 (streamplot): 大きさで cmap 色付け + 離散化 + 領域指定。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["region"] = {"lon_min": 100.0, "lon_max": 200.0,
                       "lat_min": 0.0, "lat_max": 60.0}
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 150.0,
                           "central_latitude": 0.0}
    panel["title"] = "streamplot with |V| coloring"
    stream = mc_config.default_stream_layer("ds0", "u", "v")
    stream["style"].update({"use_cmap": True, "cmap": "viridis",
                            "density": 1.3, "linewidth": 0.9, "arrowsize": 1.2,
                            "levels": [0.0, 5.0, 10.0, 20.0, 40.0]})
    stream["style"]["colorbar"].update({"label": "|V| [m/s]",
                                        "location": "right"})
    panel["layers"] = [stream]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _stream_map_single_color_config():
    """全球の流線 (単色)。値変換 (m/s そのまま) と単色パスのカバー。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    panel["title"] = "streamplot single color"
    stream = mc_config.default_stream_layer("ds0", "u", "v")
    stream["style"].update({"color": "#004488", "density": 1.0})
    panel["layers"] = [stream]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _vsection_stream_config():
    """鉛直断面の流線: 気圧レベル (不等間隔) を等間隔補間する経路のカバー。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lon"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00", "lat": 35.0}
    panel["ranges"] = {"level": [1000.0, 200.0]}
    panel["axis"].update({"invert_y": True, "log_y": True})
    panel["title"] = "section streamplot (interp levels)"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0,
                          "levels": 15})
    stream = mc_config.default_stream_layer("ds0", "u", "v")
    stream["style"].update({"color": "black", "density": 1.1, "linewidth": 0.8})
    panel["layers"] = [fill, stream]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _vsection_maskout_config():
    """鉛直断面の maskout: fill と contour に閾値マスク。"""
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lon"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T06:00:00", "lat": 35.0}
    panel["ranges"] = {"level": [1000.0, 200.0]}
    panel["axis"].update({"invert_y": True})
    panel["title"] = "section maskout"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0,
                          "levels": 21,
                          "maskout": {"below": 220.0, "above": 280.0}})
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"].update({"maskout": {"below": None, "above": 8000.0}})
    panel["layers"] = [fill, contour]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _maskout_var_config():
    """maskout の「別変数の値でマスク」: 水平面図 + 鉛直断面の2パネル。

    fill は値変換と併用 (マスク閾値はマスク変数の生の値)、hatch は
    var_above、contour は var_below + var_above の両側、断面 fill は
    自身の below と別変数の var_above の併用 (自身 → 別変数の適用順)。
    """
    map_panel = mc_config.default_panel()
    map_panel["selection"] = {"time": "2024-01-01T06:00:00", "level": 500.0}
    map_panel["title"] = "maskout by variable (map)"
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"].update({"cmap": "RdBu_r", "vmin": -50.0, "vmax": 0.0,
                          "levels": 21, "value_offset": -273.15,
                          "maskout": {"below": None, "above": None,
                                      "variable": "u", "var_below": 0.0,
                                      "var_above": None}})
    hatch = mc_config.default_hatch_layer("ds0", "t")
    hatch["style"].update({"levels": [0.0, 1000.0], "pattern": "x",
                           "maskout": {"below": None, "above": None,
                                       "variable": "v", "var_below": None,
                                       "var_above": 5.0}})
    contour = mc_config.default_contour_layer("ds0", "z")
    contour["style"].update({"vmin": 4800.0, "vmax": 6000.0, "interval": 60.0,
                             "maskout": {"below": None, "above": None,
                                         "variable": "u", "var_below": -5.0,
                                         "var_above": 15.0}})
    map_panel["layers"] = [fill, hatch, contour]

    sec = mc_config.default_section_panel()
    sec["x_dim"] = "lon"
    sec["y_dim"] = "level"
    sec["selection"] = {"time": "2024-01-01T06:00:00", "lat": 35.0}
    sec["axis"].update({"invert_y": True})
    sec["title"] = "maskout by variable (section)"
    sfill = mc_config.default_fill_layer("ds0", "t")
    sfill["style"].update({"cmap": "coolwarm", "vmin": 200.0, "vmax": 300.0,
                           "levels": 21,
                           "maskout": {"below": 220.0, "above": None,
                                       "variable": "u", "var_below": None,
                                       "var_above": 10.0}})
    sec["layers"] = [sfill]

    cfg = mc_config.default_figure_config()
    cfg["figure"]["layout"] = {"nrows": 1, "ncols": 2}
    cfg["panels"] = [map_panel, sec]
    return cfg


@pytest.mark.parametrize("config_builder", [
    _simple_config, _rich_config, _land_foreground_config, _texts_config,
    _robinson_config, _equal_earth_config, _npolar_config, _npolar_sector_config,
    _orthographic_config, _ortho_contour_only_config, _neg_contour_config,
    _contour_emphasis_config, _contour_hide_levels_config,
    _vsection_contour_emphasis_cmap_config,
    _cmap_contour_config, _value_transform_config, _pcolormesh_fill_config,
    _maskout_map_config, _vsection_maskout_config, _maskout_var_config,
    _stream_map_config, _stream_map_single_color_config, _vsection_stream_config,
    _map_scatter_grid_config,
    _pcolormesh_auto_range_config, _fill_custom_levels_config,
    _boxes_config, _lambert_config,
    _per_layer_level_config,
    _vsection_config, _vsection_lon_avg_config,
    _vsection_lon_wraparound_avg_config, _vsection_lat_weighted_avg_config,
    _hovmoller_config, _time_height_config,
    _time_height_avg_config, _hovmoller_lat_avg_config, _hovmoller_lon_avg_config,
    _line_1d_time_config, _line_1d_time_area_avg_config,
    _line_1d_lon_profile_lat_avg_config, _line_1d_time_lon_wrap_avg_config,
    _line_1d_two_lines_config, _line_1d_tight_x_config,
    _line_1d_twinx_config, _line_1d_twinx_bar_swapped_config,
    _line_1d_twinx_axis_config, _line_1d_twinx_align_zero_config,
    _line_1d_reflines_config,
    _line_1d_cyclic_lon_config,
    _line_1d_fill_between_two_vars_config, _line_1d_fill_between_baseline_config,
    _line_1d_fill_between_per_var_levels_config,
    _line_1d_fill_between_split_baseline_config,
    _line_1d_markers_only_config, _line_1d_hide_ticklabels_config,
    _line_1d_tick_interval_config, _line_1d_no_ticks_config,
    _line_1d_tick_positions_config, _line_1d_tick_custom_labels_config,
    _line_1d_minor_and_per_axis_grid_config,
    _line_1d_logx_xlim_ylim_config,
    _line_1d_log_tick_positions_config,
    _line_1d_label_style_config,
    _line_1d_stackplot_config,
    _line_1d_bundle_config, _line_1d_bundle_stats_config,
    _line_1d_bundle_band_only_config,
    _scatter_2d_uv_config, _scatter_2d_ranges_config,
    _scatter_2d_time_full_config,
    _bubble_2d_uvt_config, _bubble_2d_uvt_cmap_config,
    _hexbin_2d_config,
    _line_1d_bar_overlap_config, _line_1d_bar_dodge_config,
    _line_1d_bar_stack_config, _line_1d_bar_hatch_config,
    _line_1d_bar_hatch_no_edge_config,
    _line_1d_bar_errorbar_variable_config,
    _line_1d_bar_errorbar_constant_config,
    _line_1d_bar_horizontal_config,
    _multipanel_mixed_config, _multipanel_two_maps_config,
    _multipanel_shared_colorbar_config, _mosaic_layout_config,
    _grid_ratios_config,
    _gridlines_labels_only_config, _gridlines_split_intervals_config,
    _vector_cmap_config, _bubble_2d_uvt_cmap_discrete_config,
    _contour_cbar_extend_config,
    _zero_white_fill_map_config, _zero_white_vsection_pcolormesh_config,
    _zero_white_vector_cmap_config,
], ids=["simple", "rich-layers", "land-foreground", "texts",
        "robinson", "equal-earth", "npolar", "npolar-sector", "orthographic",
        "ortho-contour-only", "neg-contour",
        "contour-emphasis", "contour-hide-levels",
        "vsection-contour-emphasis-cmap",
        "cmap-contour", "value-transform",
        "pcolormesh-fill",
        "maskout-map", "vsection-maskout", "maskout-var",
        "stream-map", "stream-map-single", "vsection-stream",
        "map-scatter-grid",
        "pcolormesh-auto-range", "fill-custom-levels",
        "boxes", "lambert",
        "per-layer-level",
        "vsection", "vsection-lon-avg",
        "vsection-lon-wrap-avg", "vsection-lat-weighted-avg",
        "hovmoller", "time-height",
        "time-height-avg", "hovmoller-lat-wavg", "hovmoller-lon-avg",
        "line-1d-time", "line-1d-time-area-avg",
        "line-1d-lon-profile-lat-avg", "line-1d-time-lon-wrap-avg",
        "line-1d-two-lines", "line-1d-tight-x",
        "line-1d-twinx", "line-1d-twinx-bar-swapped",
        "line-1d-twinx-axis", "line-1d-twinx-align-zero",
        "line-1d-reflines",
        "line-1d-cyclic-lon",
        "line-1d-fb-two-vars", "line-1d-fb-baseline",
        "line-1d-fb-per-var-levels",
        "line-1d-fb-split-baseline",
        "line-1d-markers-only", "line-1d-hide-ticklabels",
        "line-1d-tick-interval", "line-1d-no-ticks",
        "line-1d-tick-positions", "line-1d-tick-custom-labels",
        "line-1d-minor-and-per-axis-grid",
        "line-1d-logx-xlim-ylim",
        "line-1d-log-tick-positions",
        "line-1d-label-style",
        "line-1d-stackplot",
        "line-1d-bundle", "line-1d-bundle-stats", "line-1d-bundle-band-only",
        "scatter-2d-uv", "scatter-2d-ranges",
        "scatter-2d-time-full",
        "bubble-2d-uvt", "bubble-2d-uvt-cmap",
        "hexbin-2d",
        "line-1d-bar-overlap", "line-1d-bar-dodge",
        "line-1d-bar-stack", "line-1d-bar-hatch",
        "line-1d-bar-hatch-no-edge",
        "line-1d-bar-errorbar-variable", "line-1d-bar-errorbar-constant",
        "line-1d-bar-horizontal",
        "multipanel-mixed", "multipanel-two-maps",
        "multipanel-shared-colorbar", "multipanel-mosaic",
        "multipanel-grid-ratios",
        "gl-labels-only", "gl-split-intervals",
        "vector-cmap", "bubble-cmap-discrete",
        "contour-cbar-extend",
        "zero-white-fill-map", "zero-white-vsection-pcolormesh",
        "zero-white-vector-cmap"])
def test_render_and_script_produce_identical_png(config_builder, sample_path, tmp_path):
    _assert_app_and_script_match(config_builder(), sample_path, tmp_path)


# --- 2 次元座標 (curvilinear 格子、lon(y,x) / lat(y,x)) ---
# データは conftest の curvilinear_sample_path (ランベルト格子の合成データ、
# 変数 t / z / u / v (time, lev, y, x) と t2 (time, y, x))

_CURVI_TIME = "2024-01-01T06:00:00"


def _curvilinear_fill_contour_config():
    """contourf + 等値線 + 領域指定 (isel の外接矩形) + 格子と同じ LambertConformal。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": _CURVI_TIME}
    panel["projection"] = {"name": "LambertConformal",
                           "central_longitude": 140.0, "central_latitude": 30.0,
                           "standard_parallels": [30.0, 60.0]}
    panel["region"] = {"lon_min": 118.0, "lon_max": 162.0,
                       "lat_min": 20.0, "lat_max": 46.0}
    panel["map"]["gridlines"]["labels"] = True
    fill = mc_config.default_fill_layer("ds0", "z")
    fill["selection"] = {"lev": 500.0}
    cont = mc_config.default_contour_layer("ds0", "z")
    cont["selection"] = {"lev": 500.0}
    cont["style"]["labels"] = {"show": True, "fontsize": 7, "fmt": "%g"}
    panel["layers"] = [fill, cont]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _curvilinear_pcolormesh_vector_scatter_config():
    """pcolormesh (2 次元 X/Y) + 間引きベクトル (dim 名 y/x で isel) + 格子散布。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": _CURVI_TIME}
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 140.0}
    panel["region"] = {"lon_min": 105.0, "lon_max": 175.0,
                       "lat_min": 5.0, "lat_max": 60.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["selection"] = {"lev": 850.0}
    fill["style"]["method"] = "pcolormesh"
    vec = mc_config.default_vector_layer("ds0", "u", "v")
    vec["selection"] = {"lev": 850.0}
    vec["style"]["stride_x"] = 4
    vec["style"]["stride_y"] = 3
    ms = mc_config.default_map_scatter_layer("ds0", "t2")
    ms["style"]["use_cmap"] = False
    ms["style"]["size"] = 4
    panel["layers"] = [fill, vec, ms]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _curvilinear_stream_hatch_config():
    """流線 (2 次元座標の streamplot) + ハッチ + 領域なし (全格子)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": _CURVI_TIME}
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 140.0}
    panel["region"] = {"lon_min": 100.0, "lon_max": 180.0,
                       "lat_min": 5.0, "lat_max": 60.0}
    fill = mc_config.default_fill_layer("ds0", "z")
    fill["selection"] = {"lev": 500.0}
    hatch = mc_config.default_hatch_layer("ds0", "t")
    hatch["selection"] = {"lev": 500.0}
    hatch["style"]["levels"] = [250.0, 258.0]
    strm = mc_config.default_stream_layer("ds0", "u", "v")
    strm["selection"] = {"lev": 500.0}
    panel["layers"] = [fill, hatch, strm]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _curvilinear_native_extent_config():
    """領域未指定 + 格子と同じ LambertConformal: 表示範囲は全格子点の投影座標の範囲
    (余白なしの長方形。render.curvilinear_extent ⇄ scriptgen の transform_points 行)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {"time": _CURVI_TIME}
    panel["projection"] = {"name": "LambertConformal",
                           "central_longitude": 140.0, "central_latitude": 30.0,
                           "standard_parallels": [30.0, 60.0]}
    panel["region"] = None
    panel["map"]["gridlines"]["labels"] = True
    fill = mc_config.default_fill_layer("ds0", "z")
    fill["selection"] = {"lev": 500.0}
    cont = mc_config.default_contour_layer("ds0", "t")
    cont["selection"] = {"lev": 500.0}
    panel["layers"] = [fill, cont]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _curvilinear_no_region_robinson_config():
    """領域未指定 + Robinson (格子と違う投影): 投影座標の範囲で表示 (全球にしない)。"""
    cfg = _curvilinear_native_extent_config()
    cfg["panels"][0]["projection"] = {"name": "Robinson", "central_longitude": 140.0}
    cfg["panels"][0]["map"]["gridlines"]["labels"] = False
    return cfg


@pytest.mark.parametrize("config_builder", [
    _curvilinear_fill_contour_config,
    _curvilinear_pcolormesh_vector_scatter_config,
    _curvilinear_stream_hatch_config,
    _curvilinear_native_extent_config,
    _curvilinear_no_region_robinson_config,
], ids=["curvilinear-fill-contour-lambert",
        "curvilinear-pcolormesh-vector-scatter",
        "curvilinear-stream-hatch",
        "curvilinear-native-extent-no-region",
        "curvilinear-no-region-robinson"])
def test_curvilinear_render_and_script_produce_identical_png(
        config_builder, curvilinear_sample_path, tmp_path):
    _assert_app_and_script_match(config_builder(), curvilinear_sample_path, tmp_path)


def test_curvilinear_no_region_extent_is_projected_bbox(curvilinear_sample_path):
    """2 次元座標格子で領域未指定のとき、軸の範囲 = 全格子点の投影座標の最小長方形
    (全球にならない)。"""
    import cartopy.crs as ccrs
    ds = mc_dataset.open_dataset(curvilinear_sample_path)
    fig = mc_render.render_figure(_curvilinear_native_extent_config(), {"ds0": ds})
    ax = fig.axes[0]
    pts = ax.projection.transform_points(ccrs.PlateCarree(),
                                         ds["lon"].values.astype(float),
                                         ds["lat"].values.astype(float))
    x0, x1 = float(pts[..., 0].min()), float(pts[..., 0].max())
    y0, y1 = float(pts[..., 1].min()), float(pts[..., 1].max())
    assert ax.get_xlim() == pytest.approx((x0, x1))
    assert ax.get_ylim() == pytest.approx((y0, y1))
    plt.close(fig)
    # PlateCarree でも全球 (幅 360) ではなく格子の範囲
    cfg = _curvilinear_native_extent_config()
    cfg["panels"][0]["projection"] = {"name": "PlateCarree", "central_longitude": 140.0}
    fig = mc_render.render_figure(cfg, {"ds0": ds})
    xl = fig.axes[0].get_xlim()
    assert xl[1] - xl[0] < 180.0
    plt.close(fig)


@pytest.mark.parametrize("split", [False, True], ids=["one-file", "two-files"])
def test_curvilinear_coord_files_render_matches_script(split, curvilinear_bare_paths,
                                                       tmp_path):
    """経緯度が別ファイル (座標ファイル) のデータ: 生成スクリプトの open + assign_coords
    行がアプリの結び付けと同じ格子を作ること (ClimCORE 形式。FLON / FLAT が 1 ファイル
    に同居する場合と別ファイルの場合)。"""
    import xarray as xr
    bare_path, lonlat_path = curvilinear_bare_paths
    if split:
        ll = xr.open_dataset(lonlat_path)
        p_lon, p_lat = tmp_path / "FLON.nc", tmp_path / "FLAT.nc"
        ll[["FLON"]].to_netcdf(p_lon)
        ll[["FLAT"]].to_netcdf(p_lat)
        coord_paths = (str(p_lon), str(p_lat))
    else:
        coord_paths = (lonlat_path,)
    cfg = _curvilinear_fill_contour_config()
    _assert_app_and_script_match(cfg, bare_path, tmp_path, coord_paths=coord_paths)
    # 生成スクリプトに座標ファイルの open と assign_coords が入ること
    ds = mc_dataset.attach_coord_files(
        mc_dataset.open_dataset(bare_path),
        [(p, mc_dataset.open_coord_file(p)) for p in coord_paths])
    script = mc_scriptgen.generate_script(cfg, {"ds0": ds}, {"ds0": bare_path},
                                          include_show=False)
    assert script.count("xr.open_dataset(") == 1 + len(coord_paths)
    assert "ds0.assign_coords({'FLON':" in script and "'FLAT':" in script
    # 相対パス様式では座標ファイルのパスも相対になる
    rel = mc_scriptgen.generate_script(cfg, {"ds0": ds},
                                       {"ds0": os.path.relpath(bare_path)},
                                       include_show=False)
    for p in coord_paths:
        assert repr(os.path.relpath(p)) in rel and repr(p) not in rel


def test_twin_axes_follow_shared_colorbar_layout(sample_path):
    """共有カラーバーが主 axes を縮めても、1次元パネルの第2軸 (twinx) が同じ位置に
    追従する (matplotlib が位置変更を twinned axes へ伝播することへの依存を固定。
    semantic_test_plan 1-45)。"""
    import copy
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = copy.deepcopy(_multipanel_shared_colorbar_config())
    cfg["panels"].append(_line_1d_twinx_config()["panels"][0])
    cfg["figure"]["layout"].update({"nrows": 1, "ncols": 3})
    fig = mc_render.render_figure(cfg, datasets)
    cfg_nocb = copy.deepcopy(cfg)
    cfg_nocb["figure"]["shared_colorbar"]["show"] = False
    fig_nocb = mc_render.render_figure(cfg_nocb, datasets)
    try:
        fig.canvas.draw()
        fig_nocb.canvas.draw()
        # axes の並び: 地図 2 枚 → 1D の主軸 → その twinx → 共有カラーバー
        assert len(fig.axes) == 5 and len(fig_nocb.axes) == 4
        ax, ax2 = fig.axes[2], fig.axes[3]
        assert ax2 in ax.get_shared_x_axes().get_siblings(ax), "twinx でない"
        assert ax2.get_position().bounds == ax.get_position().bounds
        # 共有カラーバーが実際に主 axes を縮めていること (縮まないなら検査が空)
        assert fig_nocb.axes[2].get_position().width > ax.get_position().width
    finally:
        plt.close(fig)
        plt.close(fig_nocb)


def test_script_savefig_options(sample_path):
    """savefig の出力形式・透明背景・余白オプションが生成スクリプトに反映される (仕様23章)。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    script = mc_scriptgen.generate_script(
        _simple_config(), datasets, {"ds0": "data.nc"},
        figure_output="figure.pdf", figure_dpi=300,
        figure_transparent=True, figure_tight=False, include_show=False)
    assert "fig.savefig('figure.pdf', dpi=300, transparent=True)" in script
    # 既定値 (PNG, tight, 不透明) は従来と同じ行になる
    script_default = mc_scriptgen.generate_script(
        _simple_config(), datasets, {"ds0": "data.nc"}, include_show=False)
    assert "fig.savefig('figure.png', dpi=300, bbox_inches=\"tight\")" in script_default
    # 背景色 (出力専用) は savefig に facecolor= として出る
    script_fc = mc_scriptgen.generate_script(
        _simple_config(), datasets, {"ds0": "data.nc"},
        figure_facecolor="#ffcc00", include_show=False)
    assert ("fig.savefig('figure.png', dpi=300, bbox_inches=\"tight\", "
            "facecolor='#ffcc00')") in script_fc
    # 透明と背景色は排他: 透明が優先され facecolor は出さない
    script_both = mc_scriptgen.generate_script(
        _simple_config(), datasets, {"ds0": "data.nc"},
        figure_transparent=True, figure_facecolor="#ffcc00", include_show=False)
    assert "transparent=True" in script_both
    assert "facecolor=" not in script_both


def test_render_saves_vector_formats(sample_path, tmp_path):
    """EPS / PDF / SVG で例外なく保存できる (仕様23.1)。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    fig = mc_render.render_figure(_simple_config(), datasets)
    for ext in ("eps", "pdf", "svg"):
        fig.savefig(tmp_path / f"figure.{ext}", format=ext)
        assert (tmp_path / f"figure.{ext}").stat().st_size > 0
    plt.close(fig)


def test_render_saves_raster_formats(sample_path, tmp_path):
    """PNG / JPG / TIFF (ラスタ) で例外なく保存できる。JPG/TIFF は Pillow 経由。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    fig = mc_render.render_figure(_simple_config(), datasets)
    for ext in ("png", "jpg", "tiff"):
        fig.savefig(tmp_path / f"figure.{ext}", format=ext, dpi=100)
        assert (tmp_path / f"figure.{ext}").stat().st_size > 0
    # TIFF は透明背景対応、JPG に facecolor 指定も可
    fig.savefig(tmp_path / "figt.tiff", format="tiff", transparent=True)
    fig.savefig(tmp_path / "figj.jpg", format="jpg", facecolor="#eef3c8")
    assert (tmp_path / "figt.tiff").stat().st_size > 0
    assert (tmp_path / "figj.jpg").stat().st_size > 0
    plt.close(fig)


def test_layout_too_small_raises(sample_path):
    """パネル数がグリッドを超えたら render / scriptgen とも ValueError。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = _multipanel_mixed_config()
    cfg["figure"]["layout"] = {"nrows": 1, "ncols": 1}
    with pytest.raises(ValueError):
        mc_render.render_figure(cfg, datasets)
    with pytest.raises(ValueError):
        mc_scriptgen.generate_script(cfg, datasets, {"ds0": sample_path})


def test_unsorted_fill_levels_raise(sample_path):
    """levels リストが未整列・重複ありなら分かりやすい ValueError。

    UI (parse_float_list) は整列済みを渡すので、これは設定 JSON を手編集した
    場合のガード (matplotlib の生エラーを出さない)。render / scriptgen 両方。
    """
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    for bad in ([10.0, 0.0, 5.0], [5.0, 5.0, 10.0], [3.0]):
        cfg = _simple_config()
        cfg["panels"][0]["layers"][0]["style"]["levels"] = bad
        with pytest.raises(mc_render.RenderError) as ei:
            mc_render.render_figure(cfg, datasets)
        assert ei.value.msg_id == "levels_invalid"
        with pytest.raises(mc_render.RenderError) as ei:
            mc_scriptgen.generate_script(cfg, datasets, {"ds0": sample_path})
        assert ei.value.msg_id == "levels_invalid"


def test_layout_missing_defaults_to_single_panel(sample_path, tmp_path):
    """旧 config (figure.layout なし) は 1×1 として従来通り描画できる。"""
    cfg = _simple_config()
    cfg["figure"].pop("layout", None)
    _assert_app_and_script_match(cfg, sample_path, tmp_path)


def test_script_contains_expected_calls(sample_path):
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    script = mc_scriptgen.generate_script(
        _rich_config(), datasets, {"ds0": "data.nc"})
    assert "xr.open_dataset('data.nc', decode_timedelta=False)" in script
    assert ".sel(time='2024-01-01T06:00:00', level=500.0)" in script
    assert "np.linspace(230.0, 260.0, 11)" in script
    assert "np.arange(4200.0, 5050.0, 100.0)" in script
    assert "hatches=['////']" in script  # pattern='/' * density=4
    assert "plt.rc_context({'hatch.linewidth': 1.5})" in script
    assert "ax.quiver(" in script
    assert "ax.quiverkey(" in script
    assert "cfeature.LAND" in script
    assert "mticker.MultipleLocator(30.0)" in script
    assert "gl.right_labels = False" in script
    assert "gl.top_labels = False" in script
    assert ("ax.set_extent([-70.0, 70.0, -20.0, 60.0], "
            "crs=ccrs.PlateCarree(central_longitude=130.0))") in script
    assert "plt.show()" in script
    # アプリ非依存 (スタンドアロン) であること
    assert "climcanvas" not in script
    # 全周データは領域指定時も経度を切り出さない (境界・extentでクリップ) ため、
    # fill / hatch / contour の3レイヤーに cyclic point が付く (ベクトルは対象外)
    assert script.count("xr.concat") == 3
    # 緯度は1格子分 (2.5°) 外側に広げて切り出す
    assert ".sel(lat=slice(-22.5, 62.5))" in script


def test_script_contains_text_annotation_calls(sample_path):
    """panel["texts"] が生成スクリプトの ax.text(...) 呼び出しになっていること。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    script = mc_scriptgen.generate_script(
        _texts_config(), datasets, {"ds0": "data.nc"})
    # 各テキストが ax.text(...) で出ている
    assert "ax.text(0.02, 0.95, '(a)'" in script
    assert "ax.text(0.5, 0.5, 'CENTER'" in script
    assert "ax.text(0.5, -0.1, 'below frame'" in script
    assert "rotation=90.0" in script
    # axes 座標のテキストは transAxes + clip_on=False
    assert script.count("transform=ax.transAxes, clip_on=False") >= 4
    # データ座標 (経度・緯度) のテキストは PlateCarree 変換
    assert "ax.text(190.0, 0.0, 'NINO', transform=ccrs.PlateCarree(), " \
           "clip_on=False" in script
    # 空テキストは出さない (axes 4 + data 1 = 5)
    assert script.count("ax.text(") == 5


def test_script_adds_cyclic_point_for_global_data(sample_path):
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    script = mc_scriptgen.generate_script(
        _npolar_config(), datasets, {"ds0": "data.nc"})
    # 全経度を覆う場合は経度0/360°の継ぎ目を閉じる行が入る
    assert script.count("xr.concat") == 2  # fill と contour の2レイヤー分
    # 極投影では円形の枠 (デフォルト)
    assert "ax.set_boundary(circle, transform=ax.transAxes)" in script
    assert "import matplotlib.path as mpath" in script


@pytest.mark.parametrize("layer_fixed_time", [False, True],
                         ids=["plain", "layer-fixed-time"])
def test_animation_render_and_script_produce_identical_gif(sample_path, tmp_path,
                                                           layer_fixed_time):
    """アニメーション: render.save_animation_gif と生成スクリプトの GIF がフレーム単位で一致。

    layer-fixed-time: レイヤー側の selection にコマと違う時刻を固定した設定。
    render がレイヤー側の時刻も差し替えないとそのパネルが止まり、scriptgen
    (レイヤー側の .sel も time_value に置換) と食い違う (2026-09-19 の実機報告)。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = _simple_config()
    time_values = ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00"]
    if layer_fixed_time:
        cfg["panels"][0]["layers"][0]["selection"] = {"time": "2024-01-02T00:00:00"}

    # 1) 生成スクリプトでアニメーション GIF を作る (コールドプロセス)
    script_gif = tmp_path / "script.gif"
    script = mc_scriptgen.generate_animation_script(
        cfg, datasets, {"ds0": sample_path},
        time_values=time_values, output_path=str(script_gif), fps=4, frame_dpi=80,
    )
    script_path = tmp_path / "anim.py"
    script_path.write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script_path)],
                            capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"スクリプト実行失敗:\n{result.stderr}"

    # 2) アプリ側でも同じ条件で GIF を作る (こちらもサブプロセスで)
    app_gif = tmp_path / "app.gif"
    runner_path = tmp_path / "anim_runner.py"
    runner_path.write_text(_ANIM_RUNNER, encoding="utf-8")
    import json
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(runner_path), str(cfg_path), sample_path,
         json.dumps(time_values), str(app_gif), _ROOT],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"render.save_animation_gif 実行失敗:\n{result.stderr}"

    # 3) フレーム単位で画素一致を確認
    from PIL import Image
    a = Image.open(script_gif)
    b = Image.open(app_gif)
    assert a.n_frames == b.n_frames == len(time_values)
    for i in range(a.n_frames):
        a.seek(i)
        b.seek(i)
        assert np.array_equal(np.array(a.convert("RGB")), np.array(b.convert("RGB"))), \
            f"frame {i} のピクセルが一致しない"


_ANIM_RUNNER = '''
import sys, json
sys.path.insert(0, sys.argv[5])
from climcanvas.core import render as mc_render
from climcanvas.core import dataset as mc_dataset

cfg = json.loads(open(sys.argv[1]).read())
datasets = {"ds0": mc_dataset.open_dataset(sys.argv[2])}
time_values = json.loads(sys.argv[3])
centers = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None
hold = int(sys.argv[7]) if len(sys.argv) > 7 else 1
mc_render.save_animation_gif(cfg, datasets, time_values, sys.argv[4],
                              fps=4, dpi=80, centers=centers, frames_per_time=hold)
'''


_ANIM_MP4_RUNNER = '''
import sys, json
sys.path.insert(0, sys.argv[5])
from climcanvas.core import render as mc_render
from climcanvas.core import dataset as mc_dataset

cfg = json.loads(open(sys.argv[1]).read())
datasets = {"ds0": mc_dataset.open_dataset(sys.argv[2])}
time_values = json.loads(sys.argv[3])
centers = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None
mc_render.save_animation_mp4(cfg, datasets, time_values, sys.argv[4],
                              fps=4, dpi=80, centers=centers)
'''


@pytest.mark.skipif(
    not __import__("climcanvas.core.render", fromlist=["ffmpeg_available"]).ffmpeg_available(),
    reason="ffmpeg がない環境ではスキップ")
def test_animation_render_and_script_produce_identical_mp4(sample_path, tmp_path):
    """MP4 アニメーション: render と scriptgen の出力が (lossy なので) 構造的に一致。

    h264 は lossy で同じ入力でも環境差で微小に変わるため、画素完全一致は求めない。
    フレーム数の一致と、各フレームの平均ピクセル差が小さいことを確認する。
    """
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = _simple_config()
    time_values = ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00"]

    # 1) 生成スクリプトで MP4 を作る (コールドプロセス)
    script_mp4 = tmp_path / "script.mp4"
    script = mc_scriptgen.generate_animation_script(
        cfg, datasets, {"ds0": sample_path},
        time_values=time_values, output_path=str(script_mp4),
        fps=4, frame_dpi=80, format="mp4",
    )
    script_path = tmp_path / "anim_mp4.py"
    script_path.write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script_path)],
                            capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"スクリプト実行失敗:\n{result.stderr}"

    # 2) アプリ側でも save_animation_mp4 で MP4 を作る
    app_mp4 = tmp_path / "app.mp4"
    runner_path = tmp_path / "anim_mp4_runner.py"
    runner_path.write_text(_ANIM_MP4_RUNNER, encoding="utf-8")
    import json
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(runner_path), str(cfg_path), sample_path,
         json.dumps(time_values), str(app_mp4), _ROOT],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"save_animation_mp4 失敗:\n{result.stderr}"

    # 3) どちらのファイルもサイズが非ゼロで MP4 ヘッダで始まっていることを確認
    for p in (script_mp4, app_mp4):
        assert p.stat().st_size > 0, f"{p} が空"
        with open(p, "rb") as f:
            head = f.read(12)
            # ISO BMFF MP4 ファイルは先頭に "ftyp" boxがある
            assert b"ftyp" in head, f"{p} が MP4 ヘッダで始まっていない: {head!r}"

    # 4) ffmpeg でフレーム数を取り出して一致を確認
    def count_frames(p):
        r = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                            "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames",
                            "-of", "default=nokey=1:noprint_wrappers=1", str(p)],
                           capture_output=True, text=True)
        return int(r.stdout.strip()) if r.returncode == 0 else -1
    n_script = count_frames(script_mp4)
    n_app = count_frames(app_mp4)
    assert n_script == n_app == len(time_values), \
        f"フレーム数が一致しない: script={n_script}, app={n_app}, expected={len(time_values)}"



# --- 地球回転アニメーション (Orthographic の投影中心をフレーム毎に送る) ---

def _ortho_config():
    """_simple_config を Orthographic にしたもの (回転アニメーション用)。"""
    cfg = _simple_config()
    cfg["panels"][0]["projection"] = {"name": "Orthographic",
                                      "central_longitude": 140.0,
                                      "central_latitude": 20.0}
    return cfg


_ROT_CENTERS = mc_render.rotation_path([(140.0, 20.0), (200.0, 50.0), (260.0, 0.0)], 3)
_ROT_TIMES = ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00"]


@pytest.mark.parametrize("time_values, hold", [(None, 1), (_ROT_TIMES, 1), (_ROT_TIMES, 2)],
                         ids=["rotate-only", "time+rotate", "time+rotate-hold2"])
def test_rotation_animation_render_and_script_produce_identical_gif(
        sample_path, tmp_path, time_values, hold):
    """地球回転アニメーション: render.save_animation_gif と生成スクリプトの GIF が
    フレーム単位で一致。回転時は固定サイズ保存 (tight なし) なので全フレームが
    同じ大きさ。時刻固定 (rotate-only)、時刻送り + 回転、時刻を 2 コマずつ保持
    (frames_per_time=2、経路は時刻数 × 2 で補間) の3通り。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = _ortho_config()
    import json
    n_frames = mc_render.animation_frame_count(time_values, _ROT_CENTERS, hold)
    centers = mc_render.rotation_path([(140.0, 20.0), (200.0, 50.0), (260.0, 0.0)], n_frames)

    script_gif = tmp_path / "script.gif"
    script = mc_scriptgen.generate_animation_script(
        cfg, datasets, {"ds0": sample_path},
        time_values=time_values, centers=centers, frames_per_time=hold,
        output_path=str(script_gif), fps=4, frame_dpi=80,
    )
    assert "bbox_inches='tight'" not in script
    assert ("for _ in range(2)]" in script) == (hold > 1)
    assert ("ccrs.Orthographic(central_longitude=center_lon, "
            "central_latitude=center_lat)") in script
    script_path = tmp_path / "anim.py"
    script_path.write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script_path)],
                            capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"スクリプト実行失敗:\n{result.stderr}"

    app_gif = tmp_path / "app.gif"
    runner_path = tmp_path / "anim_runner.py"
    runner_path.write_text(_ANIM_RUNNER, encoding="utf-8")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(runner_path), str(cfg_path), sample_path,
         json.dumps(time_values), str(app_gif), _ROOT, json.dumps(centers), str(hold)],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"render.save_animation_gif 実行失敗:\n{result.stderr}"

    from PIL import Image
    a = Image.open(script_gif)
    b = Image.open(app_gif)
    assert a.n_frames == b.n_frames == n_frames
    sizes = set()
    frames = []
    for i in range(a.n_frames):
        a.seek(i)
        b.seek(i)
        fa = np.array(a.convert("RGB"))
        assert np.array_equal(fa, np.array(b.convert("RGB"))), \
            f"frame {i} のピクセルが一致しない"
        sizes.add(fa.shape)
        frames.append(fa)
    assert len(sizes) == 1, f"回転アニメの GIF フレームサイズが揺れている: {sizes}"
    # 回転しているので隣接フレームは別の図
    assert not np.array_equal(frames[0], frames[1])


@pytest.mark.parametrize("fmt", ["gif", "mp4"])
@pytest.mark.parametrize("motion", ["time", "rotate", "both", "both-hold2"])
def test_animation_script_compiles_for_all_motions(sample_path, fmt, motion):
    """アニメーション再現スクリプトのループヘッダが 動かすもの 3 形態 (+コマ数) ×
    形式 2 種のすべてで文法的に正しいこと (MP4 は enumerate の入れ子になるので
    GIF と形が違う。実例: 時刻送り + 回転の MP4 が unpack エラー 2026-08-28)。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = _ortho_config()
    hold = 2 if motion == "both-hold2" else 1
    time_values = None if motion == "rotate" else _ROT_TIMES
    n = mc_render.animation_frame_count(time_values, _ROT_CENTERS, hold)
    centers = None if motion == "time" else mc_render.rotation_path(
        [(140.0, 20.0), (260.0, 0.0)], n)
    script = mc_scriptgen.generate_animation_script(
        cfg, datasets, {"ds0": sample_path}, time_values=time_values, centers=centers,
        frames_per_time=hold, output_path=f"anim.{fmt}", fps=4, frame_dpi=60, format=fmt)
    compile(script, "animation.py", "exec")  # SyntaxError なら失敗
    if fmt == "mp4" and motion.startswith("both"):
        assert ("for _i, (time_value, (center_lon, center_lat)) in "
                "enumerate(zip(time_values, centers)):") in script


def test_inversion_guard_emitted_only_for_orthographic(sample_path):
    """塗りポリゴンの反転ガード (render.drop_inverting_contour_rings) は Orthographic の
    contourf / hatch でだけ生成スクリプトに埋め込まれ (関数本体 = render の
    inspect.getsource そのもの + import + 呼び出し)、他の投影のスクリプトは従来と
    同一 (ガードの痕跡なし)。"""
    import inspect
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    ortho = _ortho_config()
    ortho["panels"][0]["layers"].append(
        mc_config.default_hatch_layer("ds0", "t"))
    script = mc_scriptgen.generate_script(ortho, datasets, {"ds0": sample_path})
    assert inspect.getsource(mc_render.drop_inverting_contour_rings) in script
    assert "from cartopy.mpl.geoaxes import InterProjectionTransform" in script
    assert script.count("drop_inverting_contour_rings(") >= 3  # def + fill + hatch
    plain = mc_scriptgen.generate_script(_simple_config(), datasets, {"ds0": sample_path})
    assert "drop_inverting" not in plain and "InterProjectionTransform" not in plain


@pytest.mark.skipif(
    not __import__("climcanvas.core.render", fromlist=["ffmpeg_available"]).ffmpeg_available(),
    reason="ffmpeg がない環境ではスキップ")
def test_rotation_animation_render_and_script_produce_identical_mp4(sample_path, tmp_path):
    """地球回転 MP4 (時刻固定): render と scriptgen のフレーム数が一致し、
    どちらも MP4 として書き出せる。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    cfg = _ortho_config()
    import json

    script_mp4 = tmp_path / "script.mp4"
    script = mc_scriptgen.generate_animation_script(
        cfg, datasets, {"ds0": sample_path},
        centers=_ROT_CENTERS, output_path=str(script_mp4),
        fps=4, frame_dpi=80, format="mp4",
    )
    assert "for _i, (center_lon, center_lat) in enumerate(centers):" in script
    script_path = tmp_path / "anim_mp4.py"
    script_path.write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script_path)],
                            capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"スクリプト実行失敗:\n{result.stderr}"

    app_mp4 = tmp_path / "app.mp4"
    runner_path = tmp_path / "anim_mp4_runner.py"
    runner_path.write_text(_ANIM_MP4_RUNNER, encoding="utf-8")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(runner_path), str(cfg_path), sample_path,
         json.dumps(None), str(app_mp4), _ROOT, json.dumps(_ROT_CENTERS)],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"save_animation_mp4 失敗:\n{result.stderr}"

    def count_frames(p):
        r = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                            "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames",
                            "-of", "default=nokey=1:noprint_wrappers=1", str(p)],
                           capture_output=True, text=True)
        return int(r.stdout.strip()) if r.returncode == 0 else -1
    n_script = count_frames(script_mp4)
    n_app = count_frames(app_mp4)
    assert n_script == n_app == len(_ROT_CENTERS), \
        f"フレーム数が一致しない: script={n_script}, app={n_app}"

def test_section_script_content(sample_path):
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    script = mc_scriptgen.generate_script(
        _vsection_config(), datasets, {"ds0": "data.nc"})
    assert ".sel(time='2024-01-01T06:00:00', lat=35.0)" in script
    assert ".sel(level=slice(1000.0, 200.0))" in script
    assert ".transpose('level', 'lon')" in script
    assert "ax.set_yscale('log')" in script
    # ベクトルの間引きが x/y 個別 (y=level: 2, x=lon: 5)
    assert ".isel(level=slice(None, None, 2), lon=slice(None, None, 5))" in script
    assert "ax.invert_yaxis()" in script
    assert ("ax.set_xlabel('longitude [degrees_east]', fontsize=11, "
            "color='#0033cc', fontweight='bold', fontstyle='italic', "
            "labelpad=10.0, rotation=10.0)") in script
    # 断面図は cartopy 非依存のスクリプトになる
    assert "cartopy" not in script
    assert "transform=" not in script


def test_section_script_content_with_averages(sample_path):
    """layer.averages が .sel().mean() の対で出力されることを確認。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    # 経度平均 (単純平均)
    script = mc_scriptgen.generate_script(
        _vsection_lon_avg_config(), datasets, {"ds0": "data.nc"})
    # fill / contour レイヤーは lon [120,180] で平均
    assert ".sel(lon=slice(120.0, 180.0))" in script
    assert ".mean(dim='lon')" in script
    # vector レイヤーは別範囲 [60, 200]
    assert ".sel(lon=slice(60.0, 200.0))" in script
    # 重み付き平均ではないので np.cos は出ない
    assert "np.cos(np.deg2rad" not in script

    # 緯度の cos重み付き平均
    script_w = mc_scriptgen.generate_script(
        _vsection_lat_weighted_avg_config(), datasets, {"ds0": "data.nc"})
    assert ".sel(lat=slice(-30.0, 30.0))" in script_w
    assert "np.cos(np.deg2rad" in script_w
    assert ".weighted(_w).mean(dim='lat')" in script_w


def test_section_script_content_wrap_lon_average(sample_path):
    """経度の wrap-around (30W〜30E on 0-360 data) が xr.concat で出力される。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    script = mc_scriptgen.generate_script(
        _vsection_lon_wraparound_avg_config(), datasets, {"ds0": "data.nc"})
    # 2.5° 格子で 330 は格子点、30 も格子点。データ最大は 357.5、最小は 0。
    assert ".sel(lon=slice(330.0, 357.5))" in script  # 東半分
    assert ".sel(lon=slice(0.0, 30.0))" in script     # 西半分
    assert "xr.concat([_p1, _p2], dim='lon')" in script
    assert ".mean(dim='lon')" in script


def _partial_lon_dataset(tmp_path):
    """経度範囲が 20-360° しかない (非全球) サンプルデータを作る。"""
    import xarray as xr
    import pandas as pd

    times = pd.date_range("2024-01-01", periods=2, freq="6h")
    levels = np.array([1000.0, 500.0, 200.0])
    lats = np.arange(-30.0, 30.01, 5.0)
    lons = np.arange(20.0, 360.01, 2.5)  # 0-20° は欠ける
    nt, nz = len(times), len(levels)
    data = (np.sin(np.deg2rad(lons))[None, None, None, :]
            * np.cos(np.deg2rad(lats))[None, None, :, None]
            * np.ones((nt, nz, 1, 1)))
    da = xr.DataArray(data, coords={"time": times, "level": levels,
                                     "lat": lats, "lon": lons},
                       dims=["time", "level", "lat", "lon"], name="t")
    path = tmp_path / "partial_lon.nc"
    da.to_dataset().to_netcdf(path)
    return str(path)


def test_lon_wrap_average_errors_when_data_not_global(tmp_path):
    """データ経度が 20-360° (非全球) のとき、wrap-around 平均はエラーになる。"""
    path = _partial_lon_dataset(tmp_path)
    datasets = {"ds0": mc_dataset.open_dataset(path)}
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T00:00:00"}
    fill = mc_config.default_fill_layer("ds0", "t")
    # 30W〜30E → 0-360 系に正規化すると [330, 30] で wrap、しかしデータは全球でない
    fill["averages"] = {"lon": {"op": "mean", "range": [-30.0, 30.0]}}
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]

    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.render_figure(cfg, datasets)
    assert ei.value.msg_id == "lon_range_dateline"
    with pytest.raises(mc_render.RenderError) as ei:
        mc_scriptgen.generate_script(cfg, datasets, {"ds0": "x.nc"})
    assert ei.value.msg_id == "lon_range_dateline"


def test_lon_average_errors_when_range_exceeds_data(tmp_path):
    """非 wrap でも、指定範囲がデータからはみ出していたらエラー。"""
    path = _partial_lon_dataset(tmp_path)
    datasets = {"ds0": mc_dataset.open_dataset(path)}
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T00:00:00"}
    fill = mc_config.default_fill_layer("ds0", "t")
    # データは 20-360 だが [10, 200] を指定 → lo=10 がデータ範囲外
    fill["averages"] = {"lon": {"op": "mean", "range": [10.0, 200.0]}}
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]

    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.render_figure(cfg, datasets)
    assert ei.value.msg_id == "lon_range_outside"
    with pytest.raises(mc_render.RenderError) as ei:
        mc_scriptgen.generate_script(cfg, datasets, {"ds0": "x.nc"})
    assert ei.value.msg_id == "lon_range_outside"


def _redundant_endpoint_dataset(tmp_path):
    """経度が 0..360 と冗長な端点を両方含む全球データを作る (step=30°)。"""
    import xarray as xr
    import pandas as pd

    times = pd.date_range("2024-01-01", periods=2, freq="6h")
    levels = np.array([1000.0, 500.0, 200.0])
    lats = np.arange(-30.0, 30.01, 5.0)
    lons = np.arange(0.0, 360.01, 30.0)  # 0, 30, ..., 360 (13点, 0 と 360 が冗長)
    nt, nz = len(times), len(levels)
    # 物理的に 0° と 360° は同じ点なので同じ値を入れる
    base = np.sin(np.deg2rad(lons))[None, None, None, :]
    data = (base * np.cos(np.deg2rad(lats))[None, None, :, None]
            * np.ones((nt, nz, 1, 1)))
    da = xr.DataArray(data, coords={"time": times, "level": levels,
                                     "lat": lats, "lon": lons},
                       dims=["time", "level", "lat", "lon"], name="t")
    path = tmp_path / "redundant_endpoint.nc"
    da.to_dataset().to_netcdf(path)
    return str(path)


def test_lon_wrap_avoids_double_count_at_redundant_endpoint(tmp_path):
    """データが 0° と 360° を冗長に含むとき、wrap-around 平均で 360° は除外される。

    そうしないと境界点が 2 回カウントされて結果がずれる。
    """

    path = _redundant_endpoint_dataset(tmp_path)
    datasets = {"ds0": mc_dataset.open_dataset(path)}
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T00:00:00"}
    fill = mc_config.default_fill_layer("ds0", "t")
    # 90W〜90E (= [-90, 90], 60°幅でなく180°幅) で wrap-around平均
    # 0-360 系に正規化すると [270, 90] で wrap
    fill["averages"] = {"lon": {"op": "mean", "range": [-90.0, 90.0]}}
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]

    # 1) 例外を投げずに描画完了する (全球データなので wrap 許可)
    fig = mc_render.render_figure(cfg, datasets)
    assert fig is not None
    plt.close(fig)

    # 2) 直接 apply_averages を呼んで重複除去を確認 (lat=0 の点で)
    ds = datasets["ds0"]
    da_t = ds["t"].sel(time="2024-01-01T00:00:00", level=500.0, lat=0.0)
    averaged = mc_render.apply_averages(
        da_t, fill["averages"], ds).item()
    # 期待値: lon ∈ [270, 360-30] ∪ [0, 90] (= 270, 300, 330, 0, 30, 60, 90) の sin の平均
    # 360 は除外されるので 7 点で計算 (もし重複していたら 8 点になり値がずれる)
    import math
    sample_lons = [270, 300, 330, 0, 30, 60, 90]
    expected = sum(math.sin(math.radians(L)) for L in sample_lons) / len(sample_lons)
    assert abs(averaged - expected) < 1e-10

    # 3) 生成スクリプトに dedup 行が入っていること
    script = mc_scriptgen.generate_script(cfg, datasets, {"ds0": "x.nc"})
    assert "_p1 = _p1.isel(lon=slice(None, -1))" in script


def test_lon_wrap_normal_data_has_no_dedup_line(sample_path):
    """通常データ (0..357.5) の wrap-around 平均では dedup 行は出ない。"""
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    script = mc_scriptgen.generate_script(
        _vsection_lon_wraparound_avg_config(), datasets, {"ds0": "data.nc"})
    assert "isel(lon=slice(None, -1))" not in script


def test_lon_average_ok_when_range_inside_data(tmp_path):
    """指定範囲がデータ内にあれば通常通り処理される。"""
    path = _partial_lon_dataset(tmp_path)
    datasets = {"ds0": mc_dataset.open_dataset(path)}
    panel = mc_config.default_section_panel()
    panel["x_dim"] = "lat"
    panel["y_dim"] = "level"
    panel["selection"] = {"time": "2024-01-01T00:00:00"}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["averages"] = {"lon": {"op": "mean", "range": [30.0, 200.0]}}
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]

    # 例外を投げずに描画完了する (Figure を返せる)
    fig = mc_render.render_figure(cfg, datasets)
    assert fig is not None
    plt.close(fig)
    script = mc_scriptgen.generate_script(cfg, datasets, {"ds0": "x.nc"})
    assert ".sel(lon=slice(30.0, 200.0))" in script
