# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""UI層で使う定数 (カラーマップ分類・色パレット・投影法ラベルなど)。

app.py から分離した UI 専用の定数群。コア層 (climcanvas.core) には置かない。
CMAP_GROUPS はモジュール読み込み時にカスタム cmap と外部 cmap パッケージを
自動検出して拡張される。
"""

from climcanvas.core import external_cmaps as mc_ext_cmaps
from climcanvas.core import render as mc_render

# 用途別に分類したカラーマップ。Perceptually Uniform をデフォルト分類とする。
# 各リスト先頭が分類のおすすめ初期値。matplotlib の全ビルトイン cmap を
# matplotlib 公式の分類 (https://matplotlib.org/stable/users/explain/colors/colormaps.html)
# に沿って収録。`_r` サフィックス版は「カラーマップを反転」チェックボックスで
# 動的に切り替わるので基本は素の名前のみ。ただし `RdBu_r` / `RdYlBu_r` は
# 気候・海洋分野で慣例的に用いられる向き (低=青→高=赤) を初期選択にする
# 便宜のため素で入れておく (既存プリセットとの互換性も兼ねる)。
CMAP_GROUPS = {
    "Perceptually Uniform": [
        "viridis", "plasma", "inferno", "magma", "cividis"],
    "Sequential": [
        # 単一色相 (ColorBrewer)
        "Blues", "Greens", "Reds", "Oranges", "Purples", "Greys",
        # 多色相 (ColorBrewer)
        "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
        "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"],
    "Sequential (2)": [
        "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
        "spring", "summer", "autumn", "winter", "cool", "Wistia",
        "hot", "afmhot", "gist_heat", "copper"],
    "Diverging": [
        "RdBu_r", "coolwarm", "bwr", "seismic",
        "BrBG", "PiYG", "PRGn", "PuOr", "RdGy",
        "RdYlBu_r", "RdYlGn", "Spectral",
        # matplotlib 3.10+ で追加 (Crameri 由来)
        "berlin", "managua", "vanimo"],
    "Cyclic": [
        "twilight", "twilight_shifted", "hsv"],
    "Qualitative": [
        "tab10", "tab20", "tab20b", "tab20c",
        "Set1", "Set2", "Set3", "Paired", "Dark2", "Accent",
        "Pastel1", "Pastel2"],
    "Others": [
        "terrain", "ocean", "gist_earth", "turbo", "cubehelix",
        "nipy_spectral", "jet", "rainbow", "gist_rainbow",
        "CMRmap", "brg", "flag", "gist_ncar", "gist_stern",
        "gnuplot", "gnuplot2", "prism"],
}
DEFAULT_CMAP_GROUP = "Perceptually Uniform"

# ~/.climcanvas/cmaps/ のカスタムカラーマップを起動時に読み込んで matplotlib に登録、
# UI に「Custom」グループを追加する
_custom_cmap_names = mc_render.load_custom_cmaps()
if _custom_cmap_names:
    CMAP_GROUPS["Custom"] = _custom_cmap_names

# 外部カラーマップパッケージ (cmocean / cmcrameri / cmaps) が
# インストールされていれば自動で分類グループとして追加する。
for _pkg_key, _ext_names in mc_ext_cmaps.discover().items():
    CMAP_GROUPS[mc_ext_cmaps.package_group_label(_pkg_key)] = _ext_names
KIND_LABELS = {"fill": "塗りつぶし", "contour": "等値線",
               "vector": "ベクトル", "stream": "流線",
               "map_scatter": "散布図 (点)",
               "track": "トラック (軌跡)", "hatch": "ハッチ",
               "line": "ライン", "line_bundle": "ライン (束)",
               "fill_between": "line fill",
               "stackplot": "スタックプロット",
               "bar": "棒グラフ",
               "scatter": "散布図",
               "bubble": "バブルチャート",
               "hexbin": "hexbin (六角ビン密度)",
               "hist2d": "2次元ヒストグラム",
               "hist": "ヒストグラム",
               "ecdf": "ECDF (経験累積分布)",
               "box": "箱ひげ図",
               "violin": "バイオリン"}
# 選択肢辞書は「中立キー (session_state / config に入る値) → 表示ラベル」の向きで持ち、
# selectbox には options=list(辞書) + format_func=辞書.get を渡す (i18n 第1段階)。
# 表示ラベルを保存値にしない — WIP/プリセットが UI 言語に依存してしまうため
LINESTYLE_LABELS = {"solid": "実線", "dashed": "破線",
                    "dotted": "点線", "dashdot": "一点鎖線"}
# line レイヤーは「なし」(マーカーのみ表示) も選択可能 (matplotlib の linestyle='None')
LINE_LAYER_LINESTYLE_LABELS = {**LINESTYLE_LABELS, "None": "なし"}
GL_LINESTYLE_LABELS = {":": "点線", "-": "実線", "--": "破線"}
# Natural Earth の解像度 (map.resolution)。海岸線・国境線・陸域・海域に共通
NE_RESOLUTION_LABELS = {"auto": "自動 (表示範囲に応じて)", "110m": "低 (1:110m)",
                        "50m": "中 (1:50m)", "10m": "高 (1:10m)"}
HATCH_PATTERNS = ["/", "\\", "-", "|", "+", "x", "o", "."]


COLOR_GROUPS = {
    "basecolor": {
        "blue": "#0000ff",
        "green": "#008000",
        "red": "#ff0000",
        "cyan": "#00ffff",
        "magenta": "#ff00ff",
        "yellow": "#ffff00",
        "black": "#000000",
        "white": "#ffffff",
    },
    "tab10": {
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
        "green": "#2ca02c",
        "red": "#d62728",
        "purple": "#9467bd",
        "brown": "#8c564b",
        "pink": "#e377c2",
        "gray": "#7f7f7f",
        "olive": "#bcbd22",
        "cyan": "#17becf",
    },
    "tab20": {
        "blue": "#1f77b4",
        "light blue": "#aec7e8",
        "orange": "#ff7f0e",
        "light orange": "#ffbb78",
        "green": "#2ca02c",
        "light green": "#98df8a",
        "red": "#d62728",
        "light red": "#ff9896",
        "purple": "#9467bd",
        "light purple": "#c5b0d5",
        "brown": "#8c564b",
        "light brown": "#c49c94",
        "pink": "#e377c2",
        "light pink": "#f7b6d2",
        "gray": "#7f7f7f",
        "light gray": "#c7c7c7",
        "olive": "#bcbd22",
        "light olive": "#dbdb8d",
        "cyan": "#17becf",
        "light cyan": "#9edae5",
    },
    "other": {
        "dark gray": "#404040",
        "light gray": "#c0c0c0",
    },
    # IPCC AR6 WGI Visual Style Guide (2022) p.9 の Representative Concentration
    # Pathways (RCPs) と Socio-economic Pathways (SSPs) の公式カラー。
    # RCP 8.5 と SSP5-8.5 は同一 HEX、RCP 2.6 と SSP1-1.9 も同一 HEX。
    "IPCC": {
        "RCP 8.5":  "#951b1e",
        "RCP 6.0":  "#c47900",
        "RCP 4.5":  "#5492cd",
        "RCP 2.6":  "#00adcf",
        "SSP5-8.5": "#951b1e",
        "SSP3-7.0": "#e71d25",
        "SSP2-4.5": "#f79420",
        "SSP1-2.6": "#173c66",
        "SSP1-1.9": "#00adcf",
    },
}

# (group, name) ペアごとに絵文字を上書きするマップ。
# 同じ HEX でも近接色エミの自動判定が直感に合わないときの調整に使う
_EMOJI_OVERRIDES = {
    ("tab10", "olive"): "🟩",
    ("tab10", "gray"): "⬜",
    ("tab20", "light blue"): "🟦",
    ("tab20", "light green"): "🟩",
    ("tab20", "light red"): "🟥",
    ("tab20", "light purple"): "🟪",
    ("tab20", "light brown"): "🟫",
    ("tab20", "light pink"): "🟪",
    ("tab20", "gray"): "⬛",
    ("tab20", "light gray"): "⬜",
    ("tab20", "olive"): "🟩",
    ("tab20", "light olive"): "🟩",
    ("tab20", "light cyan"): "🟦",
    ("other", "dark gray"): "⬛",
}
CUSTOM_COLOR_LABEL = "custom"

# selectbox 内で色を直感的に示すための代表 9 色 (Unicode の色付き四角絵文字)。
# HEX を RGB に分解し、ユークリッド距離が最小の絵文字を当てる
_EMOJI_SWATCHES = [
    ("🟥", (255, 0, 0)),
    ("🟧", (255, 140, 0)),
    ("🟨", (255, 230, 0)),
    ("🟩", (40, 180, 60)),
    ("🟦", (30, 110, 220)),
    ("🟪", (160, 60, 200)),
    ("🟫", (140, 85, 55)),
    ("⬛", (0, 0, 0)),
    ("⬜", (240, 240, 240)),
]


def _closest_emoji_square(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "⬜"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    best, best_d = "⬜", float("inf")
    for emoji, (er, eg, eb) in _EMOJI_SWATCHES:
        d = (r - er) ** 2 + (g - eg) ** 2 + (b - eb) ** 2
        if d < best_d:
            best_d, best = d, emoji
    return best
PROJECTION_LABELS = {
    "PlateCarree": "正距円筒図法 (PlateCarree)",
    "Robinson": "ロビンソン図法 (Robinson)",
    "EqualEarth": "イコールアース図法 (EqualEarth)",
    "NorthPolarStereo": "極投影・北極中心 (NorthPolarStereo)",
    "SouthPolarStereo": "極投影・南極中心 (SouthPolarStereo)",
    "Orthographic": "Orthographic図法",
    "LambertConformal": "ランベルト正角円錐図法 (LambertConformal)",
}
PRESSURE_UNITS = ("hpa", "pa", "millibars", "mb", "hectopascals")
