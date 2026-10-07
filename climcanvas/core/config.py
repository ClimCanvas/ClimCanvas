# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""figure_config / plot_config スキーマの定義とデフォルト値。

figure_config = {
    "figure": {"figsize": [幅inch, 高さinch], "dpi": int, "font_family": str|None,
               "layout": {...},            # パネルのグリッド配置 (default_figure_config 参照)
               "shared_colorbar": {...}},  # 全パネル共通カラーバー
    "panels": [panel, ...],   # パネルは layout のグリッドに行優先 (左上→右) で配置される
}

panel は plot_type ごとに default_*_panel() を参照する:
    horizontal_map  default_panel()          地図 (selection / region / projection / map)
    section_2d      default_section_panel()  鉛直断面・時間断面 (x_dim / y_dim)
    line_1d         default_line_panel()     1次元プロット (x_dim、第2軸あり)
    dist_1d         default_dist_panel()     1次元プロット(集計)
    scatter_2d      default_scatter_panel()  散布図・バブル図 (x/y/z_variable)
    agg_2d          default_agg_panel()      2次元集計 (hist2d / hexbin)
    heatmap         default_heatmap_panel()  カテゴリカル行列 (imshow、layers を持たない)
全パネル共通のキー (panel_id / title / time_label / texts / markers / label /
box_aspect) は _panel_common()、地図以外の軸の体裁は _axis_text_defaults() に
まとめてある。

layer は kind ごとに default_*_layer() を参照する (fill / hatch / contour / vector /
stream / map_scatter / track / line / fill_between / stackplot / bar / hist / ecdf /
box / violin / scatter / bubble / hist2d / hexbin)。値は JSON 直列化可能な型に限る
(session_state 保存とスクリプト生成のため)。
"""

from __future__ import annotations


def default_text_annotation() -> dict:
    """任意位置に1つの文字列を描画するための設定。

    coord="axes": x/y は axes 座標 (左下 (0,0), 右上 (1,1))。枠の外 (例 y=-0.1) も可。
    coord="data": x/y はデータ座標 (水平断面図では経度・緯度、その他はプロットの値)。
    全 plot_type で同じスキーマ。panel["texts"] に何個でも積める。
    """
    return {
        "text": "",
        "coord": "axes",     # "axes" | "data"
        "x": 0.05,
        "y": 0.95,
        "fontsize": 12,
        "color": "#000000",
        "ha": "left",        # left / center / right (matplotlib horizontalalignment)
        "va": "top",         # top / center / bottom (matplotlib verticalalignment)
        "rotation": 0.0,     # 度 (反時計回り)
    }


def default_marker_annotation() -> dict:
    """任意位置に1つの記号 (マーカー) を描画するための設定。

    coord は default_text_annotation と同じ ("axes" | "data")。
    size はマーカーの大きさ (pt、matplotlib の markersize)。
    全 plot_type で同じスキーマ。panel["markers"] に何個でも積める。
    """
    return {
        "marker": "o",       # matplotlib マーカー記号 (o, s, ^, x, ., +, D, * など)
        "coord": "axes",     # "axes" | "data"
        "x": 0.5,
        "y": 0.5,
        "size": 10.0,        # markersize (pt)
        "color": "#d62728",
        "edge_width": 0.0,   # 枠線の太さ (0 = 枠線なし)
        "edge_color": "#000000",
    }


def default_panel_label() -> dict:
    """パネルラベル ((a), (b), ...) の設定 (仕様22.6)。

    座標は axes 座標 (プロット枠左下 (0,0)、右上 (1,1))。既定は枠の左上外側。
    全 plot_type で同じスキーマ。
    """
    return {
        "show": False,
        "text": "(a)",
        "x": 0.0,
        "y": 1.02,
        "fontsize": 12,
        "weight": "bold",    # normal / bold
        "color": "#000000",
        "ha": "left",        # left / center / right
        "va": "bottom",      # top / center / bottom
    }


def default_time_label() -> dict:
    """プロット領域上部の時刻表示 (全 plot_type 共通)。"""
    return {
        "show": False,
        "text": None,
        "loc": "left",             # left / center / right (matplotlib set_title)
        "format": None,            # strftime 書式 (例: "%Y-%m-%d"). None は raw text
    }


def default_legend() -> dict:
    """凡例の設定 (line_1d / dist_1d / scatter_2d 共通)。"""
    return {
        "show": True,
        "loc": "best",       # 凡例の位置。x/y 指定時は使わない (基準点は凡例の左下角に固定 = render.legend_kwargs)
        "fontsize": None,
        # 凡例の位置を axes 座標で直接指定 (bbox_to_anchor)。両方 None で
        # loc のみの従来動作。枠外 (例 x=1.02) にも置ける
        "x": None,
        "y": None,
    }


def _panel_common(plot_type: str) -> dict:
    """全 plot_type の panel に共通するキー。各 default_*_panel() の先頭で展開する。"""
    return {
        "panel_id": "a",
        "plot_type": plot_type,
        "title": None,
        "title_fontsize": None,
        "time_label": default_time_label(),
        "texts": [],                   # 任意位置の文字列 (default_text_annotation)
        "markers": [],                 # 任意位置の記号 (default_marker_annotation)
        "label": default_panel_label(),  # パネルラベル (a), (b), ... (仕様22.6)
        # axes 枠 (プロット領域) の縦横比 = 高さ / 幅。None で未指定 (規定動作)。
        # 水平断面図では None のとき cartopy の GeoAxes が aspect="equal" を効かせる
        "box_aspect": None,
    }


def _axis_text_defaults() -> dict:
    """軸ラベルと目盛の体裁 (地図以外の全 plot_type の axis に共通)。"""
    return {
        # 軸ラベルの体裁 (x/y 共通)。各 None / 既定値で matplotlib デフォルト
        "label_fontsize": None,
        "label_color": None,        # None で rcParam 既定
        "label_weight": "normal",   # "normal" | "bold"
        "label_italic": False,
        "label_pad": None,          # None で matplotlib 既定 (~4 pt)
        # 軸ラベルの回転 (度)。None で matplotlib 既定 (x=0, y=90)
        "x_label_rotation": None,
        "y_label_rotation": None,
        "tick_fontsize": None,
        # 目盛り文字 (tick label) の表示。False で文字だけ消す (目盛り線は残る)
        "show_x_ticklabels": True,
        "show_y_ticklabels": True,
        # 目盛り文字の回転 (度)。0 = 回転なし (既定、旧設定不変)
        "x_tick_rotation": 0.0,
        "y_tick_rotation": 0.0,
        # 目盛り間隔 (数値軸用)。None で matplotlib 自動、0 で目盛り非表示
        # (NullLocator)、正の値で MultipleLocator (datetime 軸では未対応)
        "x_tick_interval": None,
        "y_tick_interval": None,
        # 目盛りの位置を直接指定する (数値軸用、FixedLocator)。設定すると
        # x_tick_interval より優先される
        "x_tick_positions": None,
        "y_tick_positions": None,
        # 目盛りの位置に対応するラベル文字列 (FixedFormatter)。
        # positions と同じ長さの list[str] のときだけ適用される
        "x_tick_labels": None,
        "y_tick_labels": None,
        # 補助目盛 (minor tick) の表示。AutoMinorLocator を当てる
        "show_x_minor_ticks": False,
        "show_y_minor_ticks": False,
        # 目盛 (tick) 本体の線の太さ。None で matplotlib 既定 (~0.8)
        "tick_width": None,
    }


def _grid_defaults() -> dict:
    """目盛線 (grid)。x軸/y軸を独立に ON/OFF できる。"""
    return {"show_x": False, "show_y": False,
            "color": "gray", "width": 0.5, "linestyle": ":"}


def _xy_axis_defaults() -> dict:
    """x/y 両軸のラベル・範囲・対数・反転 (line_1d / scatter_2d 系の axis に共通)。"""
    return {
        "x_label": None,          # None = ラベルなし (UI側でデフォルト文字列を入れる)
        "y_label": None,
        "x_lim": None,            # (min, max) or None (自動)
        "y_lim": None,
        "log_x": False,           # 横軸を対数軸 (時間軸では未対応)
        "log_y": False,
        "invert_x": False,
        "invert_y": False,
        "swap_y_sides": False,    # 縦軸の目盛・ラベルを右側に表示
    }


def _scatter_axis_defaults() -> dict:
    """scatter_2d / agg_2d / heatmap / dist_1d の axis (時刻軸なし)。"""
    return {**_xy_axis_defaults(), **_axis_text_defaults(), "grid": _grid_defaults()}


def _xy_errorbar_defaults() -> dict:
    """散布図・バブル図のエラーバー (対称、|err|)。x/y 独立に誤差の変数を指定 (None = なし)。

    誤差は本体と同じ固定 (x_fixed / y_fixed)・drawing_range で切り出し、
    値変換は倍率のみ適用 (オフセットは差分には無関係)。
    """
    return {
        "x_variable": None,
        "y_variable": None,
        "color": "#000000",
        "linewidth": 1.0,
        "capsize": 3.0,
    }


def _strip_points_defaults() -> dict:
    """箱ひげ・バイオリンに元データの点を重ねる (strip)。ジッターは固定シード乱数で決定的。"""
    return {
        "show_points": False,
        "point_jitter": 0.2,    # 系列軸方向の散らし幅 (0 = 一直線)
        "point_size": 6.0,      # 点の大きさ (s, pt^2)
        "point_color": "#555555",
        "point_alpha": 0.4,
    }


def default_figure_config() -> dict:
    return {
        # matplotlib の rcParams["figure.figsize"] / ["figure.dpi"] に合わせた既定値
        "figure": {"figsize": [6.4, 4.8], "dpi": 100,
                   # 図全体のフォントファミリー (rcParams["font.family"])。
                   # None = matplotlib 既定。全テキスト (タイトル・軸・目盛り・
                   # カラーバー・注記) に一括で効く。日本語フォントを選ぶと
                   # 日本語文字列の文字化け (豆腐) を防げる
                   "font_family": None,
                   # パネルのグリッド配置。panels はグリッドに行優先で埋まる
                   # (nrows * ncols >= len(panels) であること)。
                   # hspace/wspace はパネル間の縦/横余白、left/right/bottom/top は
                   # 図全体の余白 (いずれも fig.subplots_adjust の引数、None = matplotlib 既定)。
                   # mosaic: セル結合・空きセルを使う配置文字列 (例 "ABC;DEE"、
                   # "." は空きセル)。指定時は nrows/ncols より優先され、
                   # ラベルの初出順 (行優先) が panels の順に対応する。
                   # None = 従来の行優先配置 (render.parse_mosaic 参照)。
                   # width_ratios / height_ratios: 列の幅・行の高さの比率
                   # (正数のリスト。長さ = 列数 / 行数。mosaic 使用時は mosaic の
                   # グリッドに適用)。None = 均等。指定時は GridSpec 経由で配置する
                   # (render.grid_ratio_kwargs 参照)
                   "layout": {"nrows": 1, "ncols": 1, "mosaic": None,
                              "width_ratios": None, "height_ratios": None,
                              "hspace": None, "wspace": None,
                              "left": None, "right": None,
                              "bottom": None, "top": None},
                   # 全パネル共通カラーバー (仕様22.5)。最初に見つかった fill レイヤーの
                   # mappable を代表に使うため、全パネルの fill で cmap / vmin / vmax /
                   # levels を揃えること。通常は各レイヤーの colorbar.show を OFF にして使う。
                   # スキーマはレイヤーの colorbar 設定と同じ
                   "shared_colorbar": {**default_colorbar(),
                                       "show": False, "aspect": 30.0}},
        "panels": [default_panel()],
    }


def default_panel() -> dict:
    """水平断面図 (地図, plot_type="horizontal_map") のデフォルト panel。"""
    return {
        **_panel_common("horizontal_map"),
        "selection": {},
        "region": None,
        "projection": {"name": "PlateCarree",
                       "central_longitude": 180.0, "central_latitude": 0.0},
        "map": default_map_settings(),
        "layers": [],
    }


def default_section_panel() -> dict:
    """鉛直断面・時間断面 (plot_type="section_2d") のデフォルト panel。"""
    return {
        **_panel_common("section_2d"),
        "x_dim": None,                # 実際の次元名 (例: "lon", "time")
        "y_dim": None,                # 実際の次元名 (例: "level", "time")
        "selection": {},              # x/y 以外の次元の固定値
        "ranges": {},                 # {次元名: [下限, 上限]} 軸方向の範囲制限
        # 経路断面 (docs/section_extension_plan.md)。None = 従来の断面 (格子線に沿う、
        # 内挿なし)。dict なら経路上の点へ双一次内挿し、x_dim は合成次元
        # render.SECTION_PATH_DIM ("path") にする。kind ごとのキー:
        #   {"kind": "parallel", "lat": 35.0, "lon_range": [125.0, 145.0], "npoints": None}
        #   {"kind": "meridian", "lon": 130.0, "lat_range": [25.0, 45.0], "npoints": None}
        #   {"kind": "great_circle", "start": [125.0, 30.0], "end": [140.0, 40.0],
        #    "npoints": None}
        # npoints: None = 自動 (経路長 ÷ 格子間隔を切り上げ + 1)
        "section_path": None,
        # 地形マスク: 全レイヤーをデータの値のまま描いた上に、地面より下を塗った多角形を
        # 重ねる (欠損にはしない。docs/section_extension_guide.md 5 節)。method:
        #   "surface_pressure": 地上気圧の変数 (鉛直座標が気圧のとき)
        #   "surface_height":   地形高度の変数 (鉛直座標が高度のとき)
        #   "height_field":     高度の変数 (height_variable、鉛直を持つ) と地形高度の変数
        #                       から、各列で高度 = 地形高度 になる気圧を求める (鉛直座標が気圧)
        # dataset_id / variable = 地上気圧または地形高度 (水平 2 次元、時刻はあってもよい)。
        # 単位は units 属性から鉛直座標の単位へ換算する (Pa ⇄ hPa、m ⇄ km、ジオポテンシャルは g で割る)
        "terrain": {
            "show": False,
            "method": "surface_pressure",
            "dataset_id": None, "variable": None,
            "height_dataset_id": None, "height_variable": None,
            "color": "#7f7f7f",       # tab10 の gray (UI の色選択に存在する HEX)
        },
        "axis": {
            "invert_x": False,
            "invert_y": False,        # 気圧軸 (上が低圧) や Hovmöller (時間下向き) で True
            "swap_y_sides": False,    # 縦軸の目盛・ラベルを右側に表示
            "log_y": False,           # 気圧の対数軸
            # x軸が経度のとき、180° を中心とした東経・西経表記
            # (120°E … 180° … 120°W) で目盛文字を描く。False で数値のまま
            "x_lon_east_west": False,
            # 横軸の目盛に、その位置の経度・緯度を 2 段で併記する (格子線断面の格子番号軸・
            # 大円断面の距離軸向け。render.section_x_lonlat が経緯度を求められる断面だけ)
            "x_lonlat_ticks": False,
            # 時間軸の strftime 書式 (例: "%Y", "%Y-%m", "%m/%d")。None で matplotlib 既定。
            # 時間軸が x/y のどちらかに来ているときだけ適用される
            "time_axis_format": None,
            "x_label": None,          # None = ラベルなし (UI側でデフォルト文字列を入れる)
            "y_label": None,
            **_axis_text_defaults(),
            # 旧スキーマの {"show": True} も render/scriptgen 側で両軸扱いする
            "grid": _grid_defaults(),
        },
        # axes 枠線と背景 (「図枠・背景」セクション)。断面図の UI では
        # 枠線は太さのみ指定できる (スキーマ自体は 1D/2D と共通)
        "frame": default_frame(),
        "background": default_axes_background(),
        "layers": [],
    }


def default_frame() -> dict:
    """axes 枠線 (spine) の設定 (1次元・2次元プロット共通)。

    上下左右の枠線を個別に表示/非表示でき、太さ・色は4辺共通。
    width / color は None で matplotlib 既定 (rcParam) のまま。
    """
    return {
        "show_top": True, "show_bottom": True,
        "show_left": True, "show_right": True,
        "width": None,   # None で matplotlib 既定 (~0.8)
        "color": None,   # None で matplotlib 既定 (黒)
    }


def default_axes_background() -> dict:
    """axes 背景の設定 (1次元・2次元プロット共通)。

    color は None で既定 (白)。spans は範囲を指定した塗り
    (default_background_span) のリスト。reflines は基準線 (default_refline)
    のリスト (「図枠・背景」セクションの「直線」expander。データの上に描く)。
    """
    return {"color": None, "spans": [], "reflines": []}


def default_refline() -> dict:
    """基準線 (axvline / axhline)。「図枠・背景」の「直線」で 1 本ずつ追加する。

    orientation="x" は x 一定の縦線 (axvline)、"y" は y 一定の横線 (axhline)。
    value は数値軸では float、時間軸では ISO 日時文字列 (spans と同じ規約)。
    label は凡例ラベル (None で凡例に載せない)。text は線の端に添える文字
    (None で無し。横線は右端の上、縦線は上端の左。text_color None で線と同色、
    text_fontsize None で matplotlib 既定)。
    """
    return {
        "orientation": "y",
        "value": 0.0,
        "color": "#000000",
        "linewidth": 1.0,
        "linestyle": "dashed",
        "alpha": 1.0,
        "label": None,
        "text": None,
        "text_fontsize": None,
        "text_color": None,
    }


def default_background_span() -> dict:
    """背景の帯 (axvspan / axhspan)。

    orientation="x" は x軸の範囲 lo..hi を塗る縦帯 (axvspan)、
    "y" は y軸の範囲を塗る横帯 (axhspan)。lo/hi は数値軸では float、
    時間軸では ISO 日時文字列 (例 "2024-01-01T06:00:00")。
    """
    return {
        "orientation": "x",
        "lo": 0.0, "hi": 1.0,
        "color": "#ffd7d7",
        "alpha": 0.3,
    }


def default_line_panel() -> dict:
    """1次元プロット (line_1d) のデフォルト panel。"""
    return {
        **_panel_common("line_1d"),
        "x_dim": None,                # プロット軸 (任意の dim)
        "selection": {},              # x_dim 以外の次元の固定値
        "ranges": {},                 # {次元名: [下限, 上限]} 範囲制限 (主に x_dim 用)
        "axis": {
            **_xy_axis_defaults(),
            # 第2軸 (右, twinx)。線・帯・積み上げ・棒レイヤーの
            # style.secondary_y=True があるときだけ作られる。以下の y2 系キーは
            # render.secondary_axis_cfg が y スロットへ読み替え、第1軸と同じ関数で
            # ax2 に当てる。目盛文字サイズ・目盛線の太さ・目盛文字の回転・
            # 軸ラベルの体裁・目盛線の色/太さ/線種・tight_y は第1軸と共有
            "y2_label": None,
            "y2_lim": None,
            "log_y2": False,
            "invert_y2": False,
            "y2_tick_interval": None,
            "y2_tick_positions": None,
            "y2_tick_labels": None,
            "show_y2_minor_ticks": False,
            "show_y2_ticklabels": True,
            # 第2軸との値揃え: この値 (例: 0) が左右の軸で同じ高さになるよう
            # 両軸の範囲を広げる (縮めない)。None で無効。対数軸では無効
            # (render.twin_align_value / align_twin_ylim)
            "y2_align_value": None,
            "time_axis_format": None,  # x_dim が時刻のとき
            # x/y 軸の余白を 0 にしてデータ範囲を図の端ぴったりに揃える
            # (matplotlib のデフォルトは小さなマージンを取って描画する)
            "tight_x": False,
            "tight_y": False,
            # x_dim が経度のとき、データを 360° 周期で繰り返し描画する
            # (range で 360° を超える範囲を指定したいときに使う)
            "cyclic_x": False,
            # 棒グラフレイヤー (kind="bar") が複数あるときの配置:
            #   "overlap" - 同じ x に重ねる (alpha で半透明にできる)
            #   "dodge"   - 横並び (それぞれの幅を 1/n に縮めて等間隔オフセット)
            #   "stack"   - 積み上げ (正は上に、負は下に積む)
            "bar_mode": "overlap",
            # dodge モードでの隣接棒間の隙間 (各スロット幅に対する比率、0..0.5)。
            # 0 だと棒が接触してエッジ部のピクセルが重なって滲んで見える
            "bar_dodge_gap": 0.05,
            **_axis_text_defaults(),
            # show_y2 は第2軸 (twinx) の y 方向の目盛線 (色・太さ・線種は共通)
            "grid": {**_grid_defaults(), "show_y2": False},
        },
        "legend": default_legend(),
        # axes 枠線と背景 (「図枠・背景」セクション)
        "frame": default_frame(),
        "background": default_axes_background(),
        "layers": [],
    }


def default_scatter_panel() -> dict:
    """2次元プロット (散布図など) 用の panel。

    変数は panel レベルで `x_variable` / `y_variable` を選び、すべての layer で共有する。
    座標 dim の扱い (どれを散布、どれを固定) は layer 側で指定する。
    レイヤーの kind は "scatter" / "bubble" (hexbin は agg_2d へ移設、旧設定互換で残る)。
    """
    return {
        **_panel_common("scatter_2d"),
        "x_variable": None,       # panel レベルで選択する x 軸の変数名
        "y_variable": None,       # panel レベルで選択する y 軸の変数名
        # バブルチャート (kind="bubble") 用の z 軸の変数。点の大きさに使う。
        # 散布図 (kind="scatter") では使われない
        "z_variable": None,
        "axis": _scatter_axis_defaults(),
        "legend": default_legend(),
        # axes 枠線と背景 (「図枠・背景」セクション)
        "frame": default_frame(),
        "background": default_axes_background(),
        "layers": [],
    }


def default_heatmap_panel() -> dict:
    """categorical heatmap 用の panel (2次元プロットモードのラジオで選ぶ)。

    1つの2次元変数を `ax.imshow` で等幅セルの行列として色表示する。
    x_dim / y_dim が軸になり、各次元の座標値 (座標変数が無ければインデックス
    0..n-1) が目盛りラベルになる。散布図と違い x/y は「変数の値」ではなく
    「行列の次元」なので、無次元・小行列 (相関行列や地域×月など) 向け。
    緯度経度データ (lat/lon 次元) には使わない (UI 側で弾く)。
    layers を持たない唯一の plot_type (変数は panel 直下の dataset_id / variable)。
    """
    return {
        **_panel_common("heatmap"),
        "dataset_id": None,        # 表示する変数を持つ dataset
        "variable": None,          # 表示する2次元変数
        "x_dim": None,             # x 軸にする次元
        "y_dim": None,             # y 軸にする次元
        # x/y 軸の目盛りラベルの回転 (度)。カテゴリ名が長いとき用。
        # 目盛位置が「自動」(カテゴリラベル) のときだけ適用される
        "xtick_rotation": 0.0,
        "ytick_rotation": 0.0,
        "selection": {},           # 余分な次元を固定して2次元に落とす
        "style": {
            "cmap": "viridis",
            "reverse_cmap": False,
            "vmin": None,
            "vmax": None,
            # 離散化 (fill と同じ流儀)。int=レベル数 (vmin/vmax 未指定はデータ範囲
            # から linspace)、list=境界値の直接指定 (不等間隔可)、None=連続 (旧設定互換)
            "levels": 21,
            # 範囲外の色の扱い (both / neither / min / max)。levels 指定時のみ有効
            "extend": "both",
            # 0 を含むレベル帯を白に (0 が境界なら両隣の2帯)
            "zero_white": False,
            # データ値の線形変換 y = value_scale * x + value_offset
            "value_scale": 1.0,
            "value_offset": 0.0,
            # maskout: below 以下 / above 以上を描かない (default_fill_layer 参照)
            "maskout": {"below": None, "above": None},
            "aspect": "auto",      # "auto" (軸いっぱい) / "equal" (正方形セル)
            "origin": "upper",     # 行 (y) の向き: "upper"=行0が上 / "lower"=下
            # セル注記: 各セルに数値を描く。既定 off・固定色
            "annotate": {"show": False, "fmt": "%.2g",
                         "fontsize": 8, "color": "#000000"},
            "colorbar": default_colorbar(),
        },
        "axis": _scatter_axis_defaults(),
        "frame": default_frame(),
        "background": default_axes_background(),
    }


def default_dist_panel() -> dict:
    """1次元プロット(集計) 用の panel (plot_type="dist_1d")。

    hist などの「値を集計してから描く」レイヤーを重ねる。x 軸は「値」、
    y 軸は「度数 / 確率密度」(hist の場合)。緯度経度次元を持つ変数は対象外
    (UI 側で候補から除外)。軸スキーマは scatter_2d と共通 (時刻軸なし)。
    axis.x_label / y_label が None のときは描画時に自動ラベル
    (render.dist_axis_labels: 変数名 [units] / 度数・確率密度) が入る。
    """
    return {
        **_panel_common("dist_1d"),
        # box/violin の向き: "vertical" (既定) = x が系列・y が値、
        # "horizontal" = 横倒し (y が系列・x が値)。hist/ecdf には影響しない
        "box_orientation": "vertical",
        "axis": {
            **_scatter_axis_defaults(),
            # 第2軸 (右, twinx)。line/ecdf レイヤーの style.secondary_y=True の
            # レイヤーがあるときだけ作られる
            "y2_label": None,
            "y2_lim": None,
        },
        "legend": default_legend(),
        "frame": default_frame(),
        "background": default_axes_background(),
        "layers": [],
    }


def default_hist_layer(dataset_id: str) -> dict:
    """1次元プロット(集計) のヒストグラムレイヤー (kind="hist")。

    変数の値をビンに区切って度数 (density=True で確率密度) を ax.hist で描く。
    agg_dim = 集計する次元 (それ以外の次元は selection で固定)、
    agg_range = 集計する範囲 ([lo, hi]、座標値。時刻は ISO 文字列。None = 全部)。
    """
    return {
        "kind": "hist",
        "dataset_id": dataset_id,
        "variable": None,
        "agg_dim": None,
        "agg_range": None,
        "selection": {},
        "style": {
            # int = 等間隔ビン数 (range 未指定ならデータの min/max から分割)、
            # list = ビン境界の直接指定 (不等間隔可。range は使われない)
            "bins": 20,
            "range": None,          # [min, max] 等間隔ビンの範囲 (変換後の値)
            "density": False,       # True = 確率密度 (面積合計 1)
            "cumulative": False,    # True = 累積
            "histtype": "bar",      # bar / step (線のみ) / stepfilled
            "color": None,          # None = matplotlib カラーサイクル
            "alpha": 0.7,           # 重ね描き前提で既定は半透明
            "edge_color": "#000000",
            "edge_linewidth": 0.0,  # 0 で枠線なし
            "label": None,          # 凡例ラベル
            "value_scale": 1.0,
            "value_offset": 0.0,
        },
    }


def default_ecdf_layer(dataset_id: str) -> dict:
    """1次元プロット(集計) の ECDF レイヤー (kind="ecdf"、matplotlib 3.8+)。

    値をソートして経験累積分布関数 (その値以下の割合、0〜1) を階段状に描く。
    データの取り方 (agg_dim / agg_range / selection) は hist と共通
    (render.dist_values)。度数ヒストグラムと重ねるときは secondary_y=True で
    第2軸 (右) に描く。
    """
    return {
        "kind": "ecdf",
        "dataset_id": dataset_id,
        "variable": None,
        "agg_dim": None,
        "agg_range": None,
        "selection": {},
        "style": {
            "complementary": False,   # True = 補分布 (1 − CDF、超過確率)
            "color": None,            # None = matplotlib カラーサイクル
            "linewidth": 1.5,
            "linestyle": "solid",
            "alpha": 1.0,
            "label": None,
            "value_scale": 1.0,
            "value_offset": 0.0,
            "secondary_y": False,     # 第2軸 (右, twinx) に描く
        },
    }


def default_box_layer(dataset_id: str) -> dict:
    """1次元プロット(集計) の箱ひげ図レイヤー (kind="box")。

    1レイヤー = 1系列 (1箱)。x 位置はレイヤーの並び順 (1, 2, ...) で、
    系列名 (style.label、None = 変数名) が x 目盛に表示される。
    データの取り方 (agg_dim / agg_range / selection) は hist と共通。
    """
    return {
        "kind": "box",
        "dataset_id": dataset_id,
        "variable": None,
        "agg_dim": None,
        "agg_range": None,
        "selection": {},
        "style": {
            "label": None,          # 系列名 (x 目盛)。None = 変数名
            # 系列名を凡例にも載せる (箱の代表アーティストにラベル付け)。
            # 凡例自体の表示・位置は panel.legend
            "in_legend": False,
            "width": 0.5,           # 箱の幅
            # ひげ: 数値 = IQR 倍率 (matplotlib 既定 1.5)、[lo, hi] =
            # パーセンタイル対 (例 [5, 95] で 5–95% ひげ)
            "whis": 1.5,
            # ノッチ (中央値の信頼区間 1.57×IQR/√N の切れ込み)。bootstrap 版
            # CI は乱数で再現性が壊れるため非対応
            "notch": False,
            "showfliers": True,     # 外れ値の表示
            "showcaps": True,       # ひげ先端の横棒 (キャップ)
            "capwidths": None,      # キャップ幅 (箱の幅と同じ単位)。None = 既定
            "showbox": True,        # 箱自体の表示
            "showmeans": False,     # 平均値
            "meanline": False,      # 平均をマーカーではなく線で描く
            "fill_color": None,     # None = 塗りなし (白箱)。指定色で箱を塗る
            # 体裁の詳細 (None = matplotlib 既定)
            "line_color": None,     # 箱・ひげ・キャップの線色
            "line_width": None,     # 同 太さ
            "median_color": None,   # 中央値線の色
            "median_width": None,   # 同 太さ
            "mean_color": None,     # 平均 (マーカー/線) の色
            "mean_width": None,     # 平均線の太さ (meanline のとき)
            "flier_marker": None,   # 外れ値マーカー (None = 既定 'o')
            "flier_size": None,     # 同 サイズ (pt)
            "flier_color": None,    # 同 色
            **_strip_points_defaults(),
            "value_scale": 1.0,
            "value_offset": 0.0,
        },
    }


def default_violin_layer(dataset_id: str) -> dict:
    """1次元プロット(集計) のバイオリンレイヤー (kind="violin")。

    カーネル密度推定 (KDE) で分布の形を描く。1レイヤー = 1系列で
    x 位置・系列名の扱いは box と同じ。
    """
    return {
        "kind": "violin",
        "dataset_id": dataset_id,
        "variable": None,
        "agg_dim": None,
        "agg_range": None,
        "selection": {},
        "style": {
            "label": None,          # 系列名 (x 目盛)。None = 変数名
            "in_legend": False,     # 系列名を凡例にも載せる (box と同じ)
            "width": 0.7,           # バイオリンの幅
            "showmedians": True,    # 中央値の線
            "showmeans": False,     # 平均値の線
            "showextrema": True,    # 極値 (ひげ)
            # 任意の分位数の位置に横線 (0〜1 のリスト。例 [0.05, 0.95])。None = なし
            "quantiles": None,
            # KDE バンド幅: None = 自動 (scott) / "silverman" / 数値 (小さいほど
            # 形の凹凸が細かい)
            "bw_method": None,
            "points": 100,          # KDE 評価点数 (形の解像度、matplotlib 既定 100)
            # 半バイオリン: "both" = 両側 / "low" = 左半分 / "high" = 右半分
            # (2系列を向かい合わせて比較する表現)
            "side": "both",
            "color": None,          # None = matplotlib 既定色。本体と線に適用
            "alpha": 0.5,           # 本体 (KDE 形状) の透明度
            **_strip_points_defaults(),
            "value_scale": 1.0,
            "value_offset": 0.0,
        },
    }


def default_scatter_layer(dataset_id: str) -> dict:
    """散布図レイヤー。panel の x_variable / y_variable から x/y の値を取り、

    1 つの「描画する次元」(drawing_dim) に沿って散布する。drawing_dim 以外の dim は
    `x_fixed` / `y_fixed` でそれぞれの変数について 1 点に固定する (x と y で異なる値
    を指定してよい)。drawing_dim の範囲は drawing_range = [lo, hi] (None で全範囲)。
    """
    return {
        "kind": "scatter",
        "dataset_id": dataset_id,
        # 散布対象の dim 名 (この dim に沿って flatten)
        "drawing_dim": None,
        # drawing_dim の範囲 [lo, hi]。None で全範囲
        "drawing_range": None,
        # drawing_dim 以外の dim を 1 点に固定する辞書 (x / y 独立)
        "x_fixed": {},
        "y_fixed": {},
        "style": {
            "color": "#1f77b4",
            "alpha": 0.7,
            "marker": "o",
            "size": 20.0,         # ax.scatter の s (= 面積、pt²)
            "edge_color": "#000000",
            "edge_linewidth": 0.0,  # 0 で枠線なし
            "errorbar": _xy_errorbar_defaults(),
            "label": None,
            # 値変換 (x/y 独立)
            "x_value_scale": 1.0,
            "x_value_offset": 0.0,
            "y_value_scale": 1.0,
            "y_value_offset": 0.0,
        },
    }


def default_agg_panel() -> dict:
    """2次元プロット(集計) 用の panel (plot_type="agg_2d")。

    x/y 2変数の全標本ペア (共通次元を全て ravel) を hist2d / hexbin で集計する。
    緯度経度次元を持つ変数は対象外 (UI 側で候補から除外)。凡例は無し
    (カラーバーで語る図)。axis.x_label / y_label が None なら描画時に
    変数名 [units] の自動ラベルが入る (render.agg_axis_labels)。
    """
    return {
        **_panel_common("agg_2d"),
        "x_variable": None,
        "y_variable": None,
        "axis": _scatter_axis_defaults(),
        "frame": default_frame(),
        "background": default_axes_background(),
        "layers": [],
    }


def default_colorbar() -> dict:
    """カラーバー設定の共通デフォルト (全レイヤー種別でフル装備・同一スキーマ)。"""
    return {
        "show": True,
        "label": None,
        "location": "right",  # right / left / top / bottom
        "shrink": 1.0,
        "aspect": 20.0,       # カラーバーの厚み (大きいほど細い、matplotlib デフォルト 20)
        "pad": None,          # None = matplotlib デフォルト
        "label_fontsize": None,
        "tick_fontsize": None,
        # 枠線 (outline) の太さ。None = matplotlib 既定 (axes.linewidth 0.8)
        "outline_width": None,
        # 目盛り線 (tick) の太さ。None = matplotlib 既定 (0.8)
        "tick_width": None,
        # カラーバーとラベル文字の距離 (labelpad, pt)。None = matplotlib 既定 (~4)
        "label_pad": None,
        # カラーバーと目盛り文字の距離 (tick_params の pad, pt)。
        # None = matplotlib 既定 (3.5)
        "tick_pad": None,
        # 目盛り線・目盛り文字・ラベルをカラーバーの反対側に表示する
        # (例: 右配置のカラーバーで左側 = プロット側に目盛りを出す)
        "flip_ticks": False,
        # ラベルだけを目盛り文字と反対側に表示する (flip_ticks と組み合わせて
        # 目盛りとラベルを本体の左右 (上下) に振り分けられる)
        "label_opposite": False,
    }


def default_hist2d_layer(dataset_id: str) -> dict:
    """2次元プロット(集計) の2次元ヒストグラムレイヤー (kind="hist2d")。

    (x, y) 標本ペアを矩形グリッドのビンに区切って度数 (density=True で
    確率密度) を色で表す (ax.hist2d)。x/y で共通でない次元は
    x_fixed / y_fixed で固定し、共通次元は全て集計する。
    """
    return {
        "kind": "hist2d",
        "dataset_id": dataset_id,
        "x_fixed": {},
        "y_fixed": {},
        "style": {
            "bins_x": 20,           # x 方向のビン数
            "bins_y": 20,           # y 方向のビン数
            # 集計範囲 ([xmin, xmax] / [ymin, ymax])。両方指定したときだけ有効
            # (UI は1つのチェックで4値をまとめて指定する)
            "range_x": None,
            "range_y": None,
            "density": False,       # True = 確率密度 (積分 1)
            # 0 のビンを塗らない (counts は cmin=1、density は cmin=1e-300)
            "hide_zeros": True,
            "log_counts": False,    # 度数を対数スケール (LogNorm)
            "cmap": "viridis",
            "reverse_cmap": False,
            "vmin": None,
            "vmax": None,
            # 色の離散化 (None = 連続、int = レベル数 (等間隔)、list = 境界値の
            # 直接指定)。度数はビン集計後に決まるため、render は描画後に
            # 集計結果 (h[0]) から BoundaryNorm を作って set_norm する。
            # 対数スケール (log_counts) とは排他 (UI 側で排他制御)
            "levels": None,
            "extend": "neither",     # 範囲外の色 (カラーバーの矢印)
            "zero_white": False,     # 0 を含むレベル帯を白に (離散化時のみ)
            # ビンの枠線 (edge_width > 0 で表示。0 = なし — 旧設定不変)
            "edge_width": 0.0,
            "edge_color": "#000000",
            "x_value_scale": 1.0,
            "x_value_offset": 0.0,
            "y_value_scale": 1.0,
            "y_value_offset": 0.0,
            "colorbar": default_colorbar(),
        },
    }


def default_hexbin_layer(dataset_id: str) -> dict:
    """hexbin (六角ビン密度) レイヤー (kind="hexbin")。

    x/y データ (panel.x_variable / y_variable) を六角形のビンに集計し、
    ビンごとの**点数 (密度)** を色で表す。2026-07-07 に 2次元プロット(集計)
    (agg_2d) へ移設: **drawing_dim が None なら共通次元を全て ravel する
    新方式** (agg_xy_values)、drawing_dim があれば旧 scatter 方式
    (_scatter_xy_arrays、旧設定互換)。
    """
    return {
        "kind": "hexbin",
        "dataset_id": dataset_id,
        "drawing_dim": None,
        "drawing_range": None,
        "x_fixed": {},
        "y_fixed": {},
        "style": {
            "gridsize": 20,          # x 方向の六角ビン数 (大きいほど細かい)
            "cmap": "viridis",
            "reverse_cmap": False,
            "log_counts": False,     # True で点数を対数スケール (hexbin bins='log')
            # この点数未満のビンは描かない。None で 0 個のビンも塗る
            "mincnt": 1,
            "vmin": None,            # 点数の色範囲。None = 自動
            "vmax": None,
            # 色の離散化 (None = 連続、int = レベル数 (等間隔)、list = 境界値の
            # 直接指定)。点数は集計後に決まるため、render は描画後に
            # hb.get_array() から BoundaryNorm を作って set_norm する。
            # 対数スケール (log_counts) とは排他 (UI 側で排他制御)
            "levels": None,
            "extend": "neither",     # 範囲外の色 (カラーバーの矢印)
            "zero_white": False,     # 0 を含むレベル帯を白に (離散化時のみ)
            # ビンの枠線 (edge_width > 0 で表示。0 = なし — 旧設定不変)
            "edge_width": 0.0,
            "edge_color": "#000000",
            "x_value_scale": 1.0,
            "x_value_offset": 0.0,
            "y_value_scale": 1.0,
            "y_value_offset": 0.0,
            "colorbar": default_colorbar(),
        },
    }


def default_bubble_layer(dataset_id: str) -> dict:
    """バブルチャートレイヤー (kind="bubble")。

    散布図と同様に panel.x_variable / panel.y_variable から x/y を取り、
    panel.z_variable から z を取って各点の **マーカーサイズ** に変換する。
    z の値は [size_min, size_max] (pt²) の範囲に線形正規化される。
    drawing_dim / drawing_range は scatter と共通の意味。
    """
    return {
        "kind": "bubble",
        "dataset_id": dataset_id,
        "drawing_dim": None,
        "drawing_range": None,
        # 各変数で固定する dim (x/y/z 独立)
        "x_fixed": {},
        "y_fixed": {},
        "z_fixed": {},
        "style": {
            "color": "#1f77b4",       # use_cmap=False のときの単色
            "alpha": 0.6,
            "marker": "o",
            # マーカーサイズの正規化レンジ (面積 pt²)
            "size_min": 10.0,
            "size_max": 200.0,
            # 色を z の値で変えるか (True で c=z, cmap で色付け)
            "use_cmap": False,
            "cmap": "viridis",
            "reverse_cmap": False,
            "vmin": None,             # cmap 値域 (None で自動)
            "vmax": None,
            # 色の離散化。None = 連続、int = レベル数 (等間隔)、
            # list = 境界値の直接指定 (BoundaryNorm)
            "levels": None,
            "extend": "neither",      # 範囲外の色 (カラーバーの矢印)
            "zero_white": False,      # 0 を含むレベル帯を白に (離散化時のみ)
            "colorbar": default_colorbar(),
            "edge_color": "#000000",
            "edge_linewidth": 0.0,
            # エラーバー (x/y のみ、z は対象外)
            "errorbar": _xy_errorbar_defaults(),
            "label": None,
            # 値変換 (x/y/z 独立)
            "x_value_scale": 1.0,
            "x_value_offset": 0.0,
            "y_value_scale": 1.0,
            "y_value_offset": 0.0,
            "z_value_scale": 1.0,
            "z_value_offset": 0.0,
        },
    }


def default_fill_between_layer(dataset_id: str, variable: str,
                                 variable_upper: str | None = None,
                                 baseline: float = 0.0) -> dict:
    """ライン2本の間 (or 1変数と定数 baseline の間) を塗りつぶす 1次元プロット用レイヤー。

    variable_upper が None なら baseline (定数) を上側として塗る。
    value_scale / value_offset は両カーブに同じ係数を適用する (baseline は変換しない)。
    """
    return {
        "kind": "fill_between",
        "dataset_id": dataset_id,
        "variable": variable,             # 下側 (または1つ目) のカーブ
        "variable_upper": variable_upper,  # 上側のカーブ。None なら baseline を使う
        "baseline": float(baseline),       # variable_upper が None のときの上側 (定数)
        "selection": {},                   # variable 用の次元固定
        "selection_upper": {},             # variable_upper 用の次元固定 (baseline モードでは無視)
        "averages": {},                    # variable 用の範囲平均 (default_fill_layer 参照)
        "averages_upper": {},              # variable_upper 用の範囲平均
        "style": {
            "color": "#1f77b4",
            # ベースラインモード時の「ベースラインより下」側の色。None なら color と同じ。
            # 変数モード (variable_upper 指定) では使われない
            "color_below": None,
            "alpha": 0.3,
            "label": None,
            "value_scale": 1.0,
            "value_offset": 0.0,
            # 第2軸 (右の縦軸, twinx) に描く (1次元プロット)
            "secondary_y": False,
        },
    }


def default_stackplot_layer(dataset_id: str, variables: list) -> dict:
    """1次元プロット用のスタックプロット (ax.stackplot)。

    複数変数を積み上げ (または対称・wiggle) して塗り分けるレイヤー。
    値変換 (`value_scale` / `value_offset`) は全変数に同じものが適用される。
    """
    return {
        "kind": "stackplot",
        "dataset_id": dataset_id,
        # 積み順 (リスト先頭が下、末尾が上)
        "variables": list(variables),
        "selection": {},
        "averages": {},                    # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "style": {
            "alpha": 0.8,
            # "zero" (積み上げ) / "sym" (対称) / "wiggle" / "weighted_wiggle"
            "baseline": "zero",
            # 各変数の色 (len(variables) と一致しないと無視 → カラーサイクル)
            "colors": None,
            # 凡例ラベルに variables 名を出すか
            "show_labels_in_legend": True,
            "value_scale": 1.0,
            "value_offset": 0.0,
            # 第2軸 (右の縦軸, twinx) に描く (1次元プロット)
            "secondary_y": False,
        },
    }


def default_bar_layer(dataset_id: str, variable: str) -> dict:
    """1次元プロット用の棒グラフレイヤー。

    複数 bar レイヤーがあるときの並び方は panel.axis.bar_mode で決まる:
    "overlap" / "dodge" / "stack"。orientation はレイヤーごとに指定。
    """
    return {
        "kind": "bar",
        "dataset_id": dataset_id,
        "variable": variable,
        "selection": {},
        "averages": {},                  # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "style": {
            "orientation": "vertical",   # "vertical" (ax.bar) | "horizontal" (ax.barh)
            "color": "#1f77b4",
            "alpha": 0.8,
            # 棒の幅 (データ単位)。None = 隣接 x 差分の中央値 × 0.8 で自動
            "width": None,
            # 枠線: 既定 off。on にしたいときは linewidth > 0 と edge_color を指定
            "edge_color": "#000000",
            "edge_linewidth": 0.0,
            # ハッチ: pattern (単一文字) を density 回繰り返す。None で無効
            # hatch_color は枠線が off (edge_linewidth=0) のときのハッチ色。
            # 枠線が on のときはハッチ色も edge_color と共有される (matplotlib の制約)
            "hatch_pattern": None,
            "hatch_density": 3,
            "hatch_color": "#000000",
            # エラーバー (対称、|err| で正規化される)。
            #   source="none" で無効
            #   source="variable" で variable 名の値を err として使う
            #   source="constant" で constant の値を全棒共通の err として使う
            "errorbar": {
                "source": "none",
                "variable": None,
                "constant": 0.0,
                "color": "#000000",
                "linewidth": 1.0,
                "capsize": 3.0,
            },
            "label": None,
            "value_scale": 1.0,
            "value_offset": 0.0,
            # 第2軸 (右の縦軸, twinx) に描く (1次元プロット。横向き棒では無効)
            "secondary_y": False,
        },
    }


def default_line_layer(dataset_id: str, variable: str) -> dict:
    """ライン (1次元プロット) レイヤーのデフォルト。"""
    return {
        "kind": "line",
        "dataset_id": dataset_id,
        "variable": variable,
        # dist_1d (1次元プロット(集計)) 専用: この次元の座標値を x にして描く
        # (理論分布曲線などをヒストグラムに重ねる用途)。1次元プロットでは
        # 使わない (x は panel.x_dim)
        "x_dim": None,
        "selection": {},          # レイヤー固有の次元固定 (default_fill_layer 参照)
        "averages": {},           # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "style": {
            "color": None,            # None = matplotlib のカラーサイクル
            "linewidth": 1.5,
            "linestyle": "solid",
            "marker": None,           # None / "o" / "x" / "s" / "."
            "marker_size": None,      # マーカーの大きさ (pt)。None = matplotlib 既定 (6)
            "label": None,            # None = 変数名を流用
            "value_scale": 1.0,
            "value_offset": 0.0,
            # 第2軸 (右の縦軸, twinx) に描く (1次元プロット・集計の両方)
            "secondary_y": False,
        },
    }


# ライン (束) に重ねる統計線の種類 (line_bundle.summaries[].stat の中立キー)。
# minmax / pct_range は 2 本で 1 組 (凡例は 1 つ)
BUNDLE_STAT_KINDS = ("mean", "median", "min", "max", "minmax",
                     "percentile", "pct_range", "std_range")


def default_bundle_summary(stat: str = "mean") -> dict:
    """ライン (束) に重ねる統計線 1 組 (束ねた次元方向の集計)。

    stat は BUNDLE_STAT_KINDS のいずれか。percentile は q_low の分位を 1 本、
    pct_range は q_low〜q_high の分位を 2 本 (同じ体裁)、minmax は最小と最大を
    2 本、std_range は 平均 ± k_std × 標本標準偏差 (N−1 で割る。numpy の
    nanstd(ddof=1)) を 2 本
    描く。集計は numpy の nan* 関数 (NaN を除外) で行う。
    2 本組 (minmax / pct_range / std_range) は draw で描き方を選べる: "lines"
    (2 本の線) / "band" (2 本の間を塗る帯) / "band_lines" (帯 + 縁の線)。
    1 本の stat では無視。
    """
    return {
        "stat": stat,
        # percentile: 使う分位 (%) / pct_range: 下側の分位。既定は画面の初期値に合わせる
        # (パーセンタイル 1 本 = 95、範囲の下側 = 5。2026-09-29 に percentile を 5 → 95)
        "q_low": 95.0 if stat == "percentile" else 5.0,
        "q_high": 95.0,          # pct_range: 上側の分位 (他の stat では未使用)
        "k_std": 1.0,            # std_range: 標準偏差の倍率 k (平均 ± k σ)
        "draw": "lines",         # 2 本組の描き方: lines / band / band_lines
        "style": {
            "color": "#000000",
            "linewidth": 2.0,    # 線 (band では未使用)。0 なら線を描かない
            "linestyle": "solid",
            "alpha": 0.3,        # 帯の透明度 (lines では未使用)
            "label": None,       # None = 凡例に出さない (2 本組・帯でも凡例は 1 つ)
        },
    }


def default_line_bundle_layer(dataset_id: str, variable: str,
                              bundle_dim: str | None = None) -> dict:
    """ライン (束) レイヤーのデフォルト (1次元プロット)。

    bundle_dim の全スライス (アンサンブルメンバー等) を同じ色・線種・透明度で
    重ね描きし、凡例には 1 本目だけを出す。残りの次元は selection / averages で
    固定・範囲平均する (line と同じ)。summaries に統計線 (default_bundle_summary)
    を並べると束の集計 (平均・中央値・最小/最大・パーセンタイル) を重ねる。
    """
    return {
        "kind": "line_bundle",
        "dataset_id": dataset_id,
        "variable": variable,
        "bundle_dim": bundle_dim,  # この次元の全スライスを重ね描き (x_dim・時刻以外)
        "selection": {},           # レイヤー固有の次元固定 (default_fill_layer 参照)
        "averages": {},            # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "summaries": [],           # 統計線のリスト (default_bundle_summary)
        "style": {
            "color": "#808080",       # 束の色 (カラーサイクルは使わない: 全線同色)
            "linewidth": 0.8,         # 0 なら束の線を描かない (統計線の帯だけを見せる用途)
            "linestyle": "solid",
            "marker": None,           # None / "o" / "x" / "s" / "."
            "marker_size": None,      # マーカーの大きさ (pt)。None = matplotlib 既定 (6)
            "alpha": 0.5,             # 束の透明度 (本数が多いときに重なりを見せる)
            "label": None,            # None = 凡例に出さない (1 本目にだけ付く)
            "value_scale": 1.0,
            "value_offset": 0.0,
            # 第2軸 (右の縦軸, twinx) に描く
            "secondary_y": False,
        },
    }


def default_map_settings() -> dict:
    return {
        "coastlines": {"show": True, "color": "black", "width": 0.8},
        # 図の枠線 (cartopy の "geo" spine) の太さ。None = matplotlib 既定
        "frame_width": None,
        "borders": False,
        # above_data: 陸を面の格子データ (塗り・ハッチ・ベクトル) の上、海岸線・
        # 等値線・流線・トラックの線・グリッド線・box・注記の下に描く。地点を表す点
        # (散布点・トラックの点) と基準ベクトルは陸の上に出す (zorder は
        # render.LAND_FG_ZORDER / MAP_FG_ZORDER)。キーが無い旧設定は False (背景)
        "land": {"show": False, "color": "#d9d2c2", "above_data": False},
        # 同じ図の鉛直断面パネルの経路を線で重ねる (docs/section_extension_guide.md 6 節)。
        # 要素: {"panel_id": 参照する断面パネルの panel_id (figure_config 内)、
        #        "color", "width", "linestyle", "end_labels": 端点に文字を出すか,
        #        "labels": [始点, 終点] の文字, "label_fontsize"}。空 = 表示しない
        "section_paths": [],
        "ocean": {"show": False, "color": "#cfe2f3"},
        # Natural Earth の解像度 (海岸線・国境線・陸域・海域の塗りつぶしに共通)。
        # "auto" = cartopy の自動 (表示範囲の短辺が 50° 以下で 50m、15° 以下で
        # 10m、それ以外は 110m) / "110m" / "50m" / "10m"。キーが無い旧設定は auto
        "resolution": "auto",
        "gridlines": {
            "show": True,             # 全体スイッチ (線・ラベル両方の親)
            "lines": True,            # 線の表示 (labels と独立に ON/OFF できる)
            "labels": True,
            # 辺ごとのラベル表示 (labels が True のときのみ有効)。デフォルトは左・下のみ。
            # geo = 境界沿いのラベル (極投影の円周沿いの経度ラベル等)、
            # inline = 図中のラベル (極投影の緯度ラベル)。cartopy は辺に分類されない
            # ラベルをこの2種に分類する。キーが無い旧設定は表示 (cartopy 既定)
            "label_sides": {"left": True, "right": False, "top": False, "bottom": True,
                            "geo": True, "inline": True},
            # 極に置かれる経度ラベル (極投影で経度範囲を指定した扇形では、経度線が
            # 極 = 扇の要で終わり、そこにもラベルが置かれる)。False でそのラベル
            # だけを消す (弧沿いのラベルは label_sides.geo / top / bottom の管轄)。
            # キーが無い旧設定は表示 (cartopy 既定)
            "pole_label": True,
            # 極投影の緯度ラベルの置き方: "inline" (図中、cartopy 既定) / "edge"
            # (枠線 = 扇の縁との交点。y_inline=False で実現、ラベルは geo 分類に
            # なる)。全経度の円形では緯度線が枠と交差しないため edge は表示なし。
            # 表示/非表示は従来どおり label_sides.inline (placement とは独立)
            "lat_label_placement": "inline",
            # "edge" のとき緯度ラベルを出す縁: "both" (両方、既定) / "left" /
            # "right"。ラベルがどちらの縁 (region の lon_min / lon_max) に
            # 属するかで判定し、どちらの縁が視覚的に「左」かは縁を投影した
            # x 座標で解決する (render.polar_sector_edge_lons) — 扇が傾いて
            # いても縁単位で選べる
            "lat_label_edge_side": "both",
            "label_padding": 5,        # 枠線からのラベルの距離 (pt、matplotlib 既定 5)
            "color": "gray",
            "width": 0.5,
            "linestyle": ":",
            "label_fontsize": None,   # None = matplotlib デフォルト
            # 経度ラベルの回転 (度)。細かい間隔でラベルが重なると cartopy が
            # 自動で間引くため、回転で回避する用途 (None = 回転なし)
            "label_rotation": None,
            # 緯度ラベルの回転 (度)。None = 自動 (極投影では縁・経度線の向きに
            # 沿って自動回転)。0 は「水平に固定」— None と意味が違うので注意
            "lat_label_rotation": None,
            "lon_interval": None,     # 線の間隔。None = 自動
            "lat_interval": None,
            # ラベルの間隔。None = 線の間隔と同じ。線と異なる値を指定すると
            # ラベル用の Gridliner が分離される (render.gridline_plan)
            "label_lon_interval": None,
            "label_lat_interval": None,
            # ラベルの開始経度・緯度 (位相)。None = 間隔の倍数 (従来)。指定すると
            # 「開始値 + n×間隔」の位置に置く (n は負も含む = 開始値より小さい側
            # へも延長)。線の位置とは独立なので、線と合わないときはラベル用の
            # Gridliner が分離される。ラベルの実効間隔が自動 (None) の軸では無効
            "label_lon_start": None,
            "label_lat_start": None,
        },
        # 任意の矩形領域を線で囲む (水平断面図のみ。要素は box dict のリスト)
        # box: {"lon_min", "lon_max", "lat_min", "lat_max",
        #       "color", "linewidth", "linestyle"}
        "boxes": [],
        # 緯度経度のティックマーク (PlateCarree のみ有効。ラベルは付かず線のみ)
        "ticks": {
            "show": False,
            "lon_interval": 30.0,
            "lat_interval": 15.0,
            "length": 4.0,
            "width": 0.8,
            "direction": "out",   # out / in / inout
            # 辺ごとのティック表示 (matplotlib デフォルトに合わせて左・下のみ)
            "sides": {"left": True, "right": False, "top": False, "bottom": True},
            # 短いティック (補助目盛)。長い線 (上記) の間に細かい間隔で入れる
            # (例: 長い線10°毎 + 短い線5°毎)。向きと表示辺は長い線と共通
            "minor": {
                "show": False,
                "lon_interval": 10.0,
                "lat_interval": 5.0,
                "length": 2.5,
                "width": 0.8,
            },
        },
    }


def default_fill_layer(dataset_id: str, variable: str) -> dict:
    return {
        "kind": "fill",
        "dataset_id": dataset_id,
        "variable": variable,
        # レイヤー固有の次元固定 (例: 鉛直レベル)。panel.selection と
        # マージされ、同じ dim については layer 側が優先される
        "selection": {},
        # レイヤー固有の範囲平均 (鉛直断面で経度/緯度方向に平均する等)。
        # {dim: {"op": "mean"|"weighted_mean", "range": [lo, hi]}}。
        # selection と averages は同じ dim について排他 (UI 側で1つに絞る)。
        # weighted_mean は緯度 dim に対する cos(lat) 重み付き平均
        "averages": {},
        "style": {
            "method": "contourf",     # "contourf" (滑らか) または "pcolormesh" (格子・離散)
            "cmap": "viridis",
            "reverse_cmap": False,
            "vmin": None,
            "vmax": None,
            # int: vmin/vmax 指定時は linspace 分割数。
            # list[float]: 境界値の直接指定 (不等間隔可、vmin/vmax は使われない)
            "levels": 21,
            "extend": "both",
            # 塗りの透過度 (0=透明, 1=不透明)。1.0 は alpha を渡さない
            # (cmap 自身の alpha を上書きしないため。render.fill_alpha 参照)
            "alpha": 1.0,
            # 0 を含むレベル帯を白に (0 が境界なら両隣の2帯)
            "zero_white": False,
            # データ値の線形変換 y = value_scale * x + value_offset (vmin/levels/colorbar
            # はすべて変換後の値で指定する)。例: K→°C は scale=1, offset=-273.15
            "value_scale": 1.0,
            "value_offset": 0.0,
            # maskout (GrADS 相当): below = この値以下を描かない、above = この値以上を
            # 描かない。None = マスクなし。閾値は値変換 (scale/offset) 後の値で指定。
            # variable を指定すると「同じデータセットの別変数の値」でもマスクできる
            # (var_below 以下 / var_above 以上の位置を描かない)。マスク変数には
            # 描画変数と同じ selection が適用され、閾値はその変数の生の値で指定する
            # (描画変数の値変換は掛からない)
            "maskout": {"below": None, "above": None,
                        "variable": None, "var_below": None, "var_above": None},
            "colorbar": default_colorbar(),
        },
    }


def default_hatch_layer(dataset_id: str, variable: str) -> dict:
    return {
        "kind": "hatch",
        "dataset_id": dataset_id,
        "variable": variable,
        "selection": {},          # レイヤー固有の次元固定 (default_fill_layer 参照)
        "averages": {},           # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "style": {
            "levels": [0.0, 1.0],     # [下限, 上限] の範囲にハッチを掛ける (変換後の値)
            "pattern": "/",           # ハッチ文字 (matplotlib: / \ | - + x o O . *)
            "density": 3,             # パターン文字の繰り返し回数 (大きいほど線が密)
            "linewidth": 1.0,         # ハッチ線の太さ (rcParams['hatch.linewidth'])
            "color": "#000000",       # ハッチ線の色 (rcParams['hatch.color'])
            "value_scale": 1.0,
            "value_offset": 0.0,
            # maskout: below 以下 / above 以上を描かない (default_fill_layer 参照)
            "maskout": {"below": None, "above": None,
                        "variable": None, "var_below": None, "var_above": None},
        },
    }


def default_contour_layer(dataset_id: str, variable: str) -> dict:
    return {
        "kind": "contour",
        "dataset_id": dataset_id,
        "variable": variable,
        "selection": {},          # レイヤー固有の次元固定 (default_fill_layer 参照)
        "averages": {},           # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "style": {
            # レベル指定 (fill と同じ流儀)。levels: int = レベル数 (vmin/vmax
            # 指定時は linspace 分割数、未指定時は自動範囲の目安数)、
            # list = 境界値の直接指定 (不等間隔可)
            "vmin": None,
            "vmax": None,
            # 旧スキーマ互換: vmin/vmax/interval 全指定なら arange (levels より優先)
            "interval": None,
            "levels": 11,
            "extend": "neither",       # use_cmap 時の範囲外の扱い (カラーバーの矢印)
            "color": "black",          # use_cmap=False のときの線色
            "use_cmap": False,         # True なら cmap で線色を変化させる (color は無視)
            "cmap": "viridis",         # use_cmap=True のときのカラーマップ
            "reverse_cmap": False,
            "colorbar": default_colorbar(),  # use_cmap=True のときだけ使われる
            "linewidth": 1.0,
            "linestyle": "solid",     # 正の値の線種 (solid / dashed / dotted / dashdot)
            "negative_linestyle": "dashed",  # 負の値の線種 (None なら linestyle に統一)
            "value_scale": 1.0,        # データ値の線形変換 y = scale * x + offset
            "value_offset": 0.0,
            # maskout: below 以下 / above 以上を描かない (default_fill_layer 参照)
            "maskout": {"below": None, "above": None,
                        "variable": None, "var_below": None, "var_above": None},
            # 指定レベルの強調・非表示: そのレベルの線だけ太さ・色を変える (0 線の強調、
            # または linewidth=0 で特定レベルを非表示にする等)。
            # levels = 対象レベル値のリスト (値変換後の値。描画レベルに含まれる値だけ効く)。
            # linewidth / color は None で「変えない」(片方だけ指定も可)。linewidth=0 で非表示
            "emphasis": {"levels": None, "linewidth": None, "color": None},
            "labels": {"show": True, "fontsize": 8, "fmt": "%g"},
        },
    }


def default_vector_layer(dataset_id: str, u_variable: str, v_variable: str,
                         v_dataset_id: str | None = None) -> dict:
    """ベクトル (quiver) レイヤー。

    dataset_id は x 成分 (u_variable) を持つ dataset で、レイヤーの次元固定 UI や
    座標役割の基準にもなる。y 成分 (v_variable) が別ファイルにあるときは
    v_dataset_id にその dataset id を入れる (None = x 成分と同じ dataset。
    旧設定にはキー自体が無く、同じ dataset として扱う)。両成分は同じ格子
    (同じ次元名・同じ形状) であること (render が形状不一致を RenderError にする)。
    """
    return {
        "kind": "vector",
        "dataset_id": dataset_id,
        "u_variable": u_variable,
        "v_variable": v_variable,
        "v_dataset_id": v_dataset_id,   # y 成分の dataset (None = dataset_id と同じ)
        "selection": {},          # レイヤー固有の次元固定 (default_fill_layer 参照)
        "averages": {},           # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "style": {
            "color": "black",
            # ベクトルの大きさ |V| = sqrt(u²+v²) で矢印を色付けする (仕様16章拡張)。
            # use_cmap=True のとき color は使わず cmap / vmin / vmax / colorbar が有効
            "use_cmap": False,
            "cmap": "viridis",
            "reverse_cmap": False,
            "vmin": None,             # 大きさの範囲。None = 自動
            "vmax": None,
            # 色の離散化。None = 連続 (グラデーション)、int = レベル数 (等間隔)、
            # list = 境界値の直接指定 (昇順)。BoundaryNorm で段階化され
            # カラーバーも段々になる
            "levels": None,
            "extend": "neither",      # 範囲外の色 (カラーバーの矢印)
            "zero_white": False,      # 0 を含むレベル帯を白に (離散化時のみ)
            # 大きさ |V| がこの値以下の矢印を描かない (maskout)。None = マスクなし。
            # value_scale 適用後・間引き後の値に対する閾値
            "mask_below": None,
            "colorbar": default_colorbar(),
            "width": None,            # None = matplotlib デフォルト
            "scale": None,            # quiver の自動スケール (データ値の変換ではない)
            # 矢印の頭の長さ (軸幅単位、matplotlib 既定 5)。None = 既定。
            # 指定時は headaxislength も既定比率 (4.5/5) を保って連動する
            "headlength": None,
            # 矢印の枠線 (ポリゴンの輪郭)。edge_width=0 で枠線なし。
            # 単色・cmap 色付けのどちらとも併用できる
            "edge_width": 0.0,
            "edge_color": "#000000",
            "stride_x": 3,            # x方向の間引き間隔 (格子点数)
            "stride_y": 3,            # y方向の間引き間隔 (格子点数)
            # ベクトル成分の倍率 (両成分に同じ係数を掛ける。単位変換用)。
            # 加算は方向が変わるため非対応。
            "value_scale": 1.0,
            "key": {
                "show": True,
                "length": 10.0,
                "label": "10 m/s",
                "x": 1.0,             # axes座標。labelpos='E' では矢印の右端が x に揃う
                "y": -0.07,
                "labelpos": "E",      # E = 矢印の右にラベル / W = 左
                "fontsize": None,     # ラベルの文字サイズ (pt)。None = matplotlib 既定
            },
        },
    }


def default_stream_layer(dataset_id: str, u_variable: str, v_variable: str,
                         v_dataset_id: str | None = None) -> dict:
    """流線 (streamplot) レイヤー。dataset_id / v_dataset_id の意味は
    default_vector_layer と同じ (y 成分だけ別ファイルから読める)。"""
    return {
        "kind": "stream",
        "dataset_id": dataset_id,
        "u_variable": u_variable,
        "v_variable": v_variable,
        "v_dataset_id": v_dataset_id,   # y 成分の dataset (None = dataset_id と同じ)
        "selection": {},          # レイヤー固有の次元固定 (default_fill_layer 参照)
        "averages": {},           # レイヤー固有の範囲平均 (default_fill_layer 参照)
        "style": {
            "color": "black",         # use_cmap=False のときの線色
            # 大きさ |V| で流線を色付けする (vector と同じ流儀)
            "use_cmap": False,
            "cmap": "viridis",
            "reverse_cmap": False,
            "vmin": None,             # 大きさの範囲。None = 自動
            "vmax": None,
            # 色の離散化。None = 連続、int = レベル数、list = 境界値の直接指定
            "levels": None,
            "extend": "neither",      # 範囲外の色 (カラーバーの矢印)
            "zero_white": False,      # 0 を含むレベル帯を白に (離散化時のみ)
            "density": 1.0,           # 流線の密度 (matplotlib streamplot の density)
            "linewidth": 1.0,
            "arrowsize": 1.0,         # 矢じりの大きさ (matplotlib 既定 1)
            # 両成分に同じ倍率 (単位変換用。加算は方向が変わるため非対応)
            "value_scale": 1.0,
            "colorbar": default_colorbar(),
        },
    }


def default_map_scatter_layer(dataset_id: str, variable: str) -> dict:
    """水平断面図 (地図) の散布図レイヤー (kind="map_scatter")。

    1つの変数を、その (lon, lat) 位置にマーカーで打つ。データ構造で自動分岐:
      - 格子データ (変数が lat×lon 次元をもつ): 全格子点に打つ
      - 地点データ (変数が地点次元をもち lon/lat が補助座標): 各地点に打つ
    どちらも `ax.scatter(lon, lat, ...)` で描画。use_cmap=True で値による色付け。
    """
    return {
        "kind": "map_scatter",
        "dataset_id": dataset_id,
        "variable": variable,
        "selection": {},          # 余分な次元の固定 (default_fill_layer 参照)
        "style": {
            # 値で色付けする (use_cmap=False なら color の単色)。既定は画面の初期値
            # (「点の色付け」= 単色) に合わせる (2026-09-29。それまでは True で、画面と食い違っていた)
            "use_cmap": False,
            "color": "#1f77b4",
            "cmap": "viridis",
            "reverse_cmap": False,
            "vmin": None,
            "vmax": None,
            "levels": None,        # 離散化 (None=連続, int, list)
            "extend": "neither",   # 範囲外の色 (カラーバーの矢印)
            "zero_white": False,   # 0 を含むレベル帯を白に (離散化時のみ)
            "size": 20.0,          # マーカーサイズ (s, 面積 pt^2)
            "marker": "o",
            "alpha": 1.0,
            "edge_color": "#000000",
            "edge_linewidth": 0.0,  # 0 で枠線なし
            "value_scale": 1.0,
            "value_offset": 0.0,
            "colorbar": default_colorbar(),
        },
    }


def default_track_layer(dataset_id: str) -> dict:
    """水平断面図のトラック (軌跡) レイヤー (kind="track")。

    熱帯低気圧のベストトラック等。IBTrACS netCDF のような「storm × 時刻」
    構造を想定する: lon_var / lat_var が軌跡に沿った経度・緯度を持ち、
    storm_dim (複数トラックを区別する次元) があれば storm_range = [lo, hi]
    (isel、両端含む) で描く範囲を選ぶ (None = 全トラック)。
    線は ccrs.Geodetic() で描くため日付変更線をまたいでも正しくつながる。
    欠損 (NaN) の位置で線は切れる (IBTrACS の末尾 NaN パディングは無視される)。
    """
    return {
        "kind": "track",
        "dataset_id": dataset_id,
        "lon_var": None,          # 経度の変数名 (units=degrees_east)
        "lat_var": None,          # 緯度の変数名 (units=degrees_north)
        "storm_dim": None,        # 複数トラックの次元 (None = 1本もの)
        "storm_range": None,      # [lo, hi] isel 範囲 (両端含む)。None = 全部
        # 発生年・月によるトラック選択 (storm_range より優先)。time_var は
        # datetime64 の時刻変数 (IBTrACS の time 等)。各ストームの発生時刻
        # (最初の有効時刻) が範囲内のトラックだけを描く
        "time_var": None,
        "year_range": None,       # [y0, y1] 両端含む。None = 全年
        # [m0, m1] 両端含む。m0 > m1 で年またぎ (例 [11, 3] = 11,12,1,2,3月)
        "month_range": None,
        "style": {
            # 線。color=None で matplotlib カラーサイクル (トラック毎に色が変わる)
            "color": None,
            "linewidth": 1.5,
            "linestyle": "solid",
            "alpha": 1.0,
            # maskout: 指定変数 (風速など) の値が条件を満たす位置を描かない
            # (線はそこで切れ、点も消える)。below = この値**以下**を描かない、
            # above = この値**以上**を描かない。しきい値は変数の**生の値**
            # (例 tokyo_wind なら kt。points の値変換は適用されない)。
            # 変数が欠損の位置も描かれない。しきい値を変えたレイヤーを重ねると
            # 「強度によって線の色が変わる」表現ができる
            "maskout": {"variable": None, "below": None, "above": None},
            # 軌跡上の観測点マーカー。variable を指定すると値で色付け (強度など)、
            # None なら単色 color。every = N 点毎に間引き
            "points": {
                "show": True,
                "variable": None,
                "color": "#333333",
                "size": 12.0,
                "marker": "o",
                "every": 1,
                "cmap": "viridis",
                "reverse_cmap": False,
                "vmin": None,
                "vmax": None,
                # 離散化 (None=連続, int=レベル数, list=境界値の直接指定)。
                # 台風カテゴリの色分けは境界値の直接指定で
                "levels": None,
                "extend": "neither",  # 範囲外の色 (カラーバーの矢印)
                "zero_white": False,  # 0 を含むレベル帯を白に (離散化時のみ)
                # 値変換 (ノット→m/s は scale=0.514444 など)。閾値・カラーバーは
                # 変換後の値
                "value_scale": 1.0,
                "value_offset": 0.0,
                "colorbar": default_colorbar(),
            },
        },
    }
