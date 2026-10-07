# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""figure_config から matplotlib Figure を生成する。

ここでの描画呼び出しは scriptgen.py が生成するコードと1対1で対応している。
描画を変更するときは必ず scriptgen.py と画像比較テスト (tests/test_consistency.py)
も同時に更新すること。「描けるがスクリプトに出せない」機能を作らない。

*_kwargs() 関数群は render と scriptgen の両方から使い、設定値→描画引数の
解決ロジックを一箇所に保つ。
"""

from __future__ import annotations

import copy
import io
import os
import pathlib
import subprocess
import tempfile
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import InterProjectionTransform
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

from .dataset import detect_coord_roles, horizontal_dims, is_curvilinear, lonlat_bounds

PROJECTIONS = ["PlateCarree", "Robinson", "EqualEarth", "NorthPolarStereo",
               "SouthPolarStereo", "Orthographic", "LambertConformal"]


class RenderError(ValueError):
    """ユーザー操作起因の描画エラー (msg_id + params 付き)。

    str() は英語 — コア層は UI 言語を知らないため (生成スクリプト単体や
    ライブラリ利用でもそのまま読める)。アプリでは app.py が msg_id から
    現在の UI 言語のメッセージに翻訳して表示する
    (app.py の _RENDER_ERROR_LABELS。msg_id の集合はテストで同期を検査)。
    """

    MESSAGES = {
        "levels_invalid":
            "Color levels (style.levels) must be two or more ascending, "
            "non-duplicated values: {levels!r}",
        "unsupported_projection": "Unsupported projection: {name}",
        "lon_range_dateline":
            "The longitude range [{lo:g}, {hi:g}] crosses the dateline, but "
            "the data longitudes [{data_min:g}, {data_max:g}] do not cover "
            "the globe. Part of the range has no data, so the mean cannot "
            "be computed.",
        "lon_range_outside":
            "The longitude range [{lo:g}, {hi:g}] (normalized to the data "
            "convention: [{lo_n:g}, {hi_n:g}]) extends beyond the data "
            "longitudes [{data_min:g}, {data_max:g}]. The mean cannot be "
            "computed.",
        "layout_too_small":
            "Cannot place {n_panels} panels in a {nrows}x{ncols} layout. "
            "Increase rows/columns.",
        "mosaic_empty": "The mosaic string is empty",
        "mosaic_ragged":
            "All mosaic rows must have the same number of characters "
            "(columns): {rows}",
        "mosaic_count_mismatch":
            "The number of mosaic labels {n_labels} ({labels}) does not "
            "match the number of panels {n_panels}",
        "mosaic_not_rect":
            "The cells of mosaic label '{label}' do not form a rectangle",
        "ratio_len_mismatch":
            "{name} has {n} value(s) but the grid has {expected} "
            "{unit}(s); the lengths must match",
        "ratio_invalid":
            "{name} must be a list of positive numbers: {values!r}",
        "no_time_dim": "No time dimension found; cannot animate",
        "no_frames": "No frames to generate",
        "ffmpeg_missing":
            "ffmpeg not found. MP4 output requires ffmpeg to be installed.",
        "no_time_dim_script":
            "No time dimension found; cannot generate an animation script",
        "all_nan":
            "All values are missing (NaN); color levels cannot be "
            "determined automatically",
        "map_scatter_dims_unfixed":
            "Cannot draw the map scatter: variable {var!r} still has "
            "unfixed dimension(s) {dims} ({n_values} values for {n_points} "
            "point(s)). Fix the remaining dimensions (e.g. select a time).",
        "constant_value":
            "The values are constant (min = max = {vmin:g}); equally "
            "spaced pcolormesh levels cannot be determined automatically. "
            "Specify levels directly, or set the value range (min/max).",
        "maskout_var_missing":
            "Maskout variable {var!r} is not in the dataset",
        "maskout_var_extra_dims":
            "Maskout variable {var!r} has dimension(s) {dims} that the "
            "drawn variable {target!r} does not have; it cannot be aligned "
            "for masking",
        "maskout_var_missing_dims":
            "Maskout variable {var!r} lacks the drawing dimension(s) "
            "{dims}; it cannot be used for masking",
        "vector_shape_mismatch":
            "The x component {u_var!r} (shape {u_shape}) and the y component "
            "{v_var!r} (shape {v_shape}) are on different grids after "
            "selection; both components must share the same dimension "
            "names and shape",
        "degenerate_geometry":
            "Drawing failed due to a degenerate polygon inside cartopy "
            "(a known upstream issue: GEOS 'empty Point' error). This can "
            "happen when a color level exactly matches the data "
            "minimum/maximum or the value of a constant region. Use the "
            "automatic value range, shift min/max slightly away from the "
            "data range, or switch the fill method to pcolormesh.",
        "curvilinear_coords_nonfinite":
            "The 2-D longitude/latitude coordinates ({lon!r} / {lat!r}) contain "
            "{n} non-finite values (NaN / inf) inside the drawn area; matplotlib "
            "cannot draw a mesh with undefined corner positions. Fill the "
            "coordinates of those grid points (e.g. with the values of the "
            "neighbouring points) and mask the data instead, or restrict the "
            "region to exclude them",
        "section_path_no_lonlat":
            "A section along a path needs longitude and latitude coordinates, "
            "but none were recognized in the dataset",
        "section_path_dim_conflict":
            "The dataset already has a dimension named {dim!r}, which is "
            "reserved for the section path; rename that dimension",
        "section_path_unknown_kind":
            "Unknown section path kind: {kind!r} (expected 'parallel', "
            "'meridian' or 'great_circle')",
        "section_path_antipodal":
            "The start and end points of the section are antipodal, so the "
            "great circle between them is not unique; move one of the points",
        "section_path_outside_grid":
            "The section path ({kind}) lies entirely outside the data grid "
            "(longitude {lon_min:g} to {lon_max:g}, latitude {lat_min:g} to "
            "{lat_max:g}); move the path or check the longitude convention",
        "terrain_vertical_units_unknown":
            "The terrain mask needs to know whether the vertical coordinate {dim!r} "
            "is a pressure or a height, but its units attribute {units!r} is not "
            "recognized (expected e.g. hPa, Pa, m or km)",
        "terrain_method_mismatch":
            "The terrain method {method!r} does not fit a vertical coordinate that "
            "is a {kind} ({dim!r} [{units}]); use {expected}",
        "terrain_variable_missing":
            "The terrain mask variable {var!r} was not found in dataset {dataset!r}",
        "terrain_units_unknown":
            "The units attribute {units!r} of the terrain variable {var!r} cannot be "
            "converted to the vertical coordinate units {target!r} (expected e.g. "
            "{expected})",
        "terrain_profile_dims":
            "The terrain variable {var!r} must reduce to the section axis {x_dim!r} "
            "after fixing the other dimensions, but has dimensions {dims}",
        "terrain_no_lonlat":
            "The terrain variable {var!r} in dataset {dataset!r} has no recognized "
            "longitude/latitude coordinates, so it cannot be sampled along the "
            "section path (attach coordinate files to that dataset)",
        "section_overlay_panel_missing":
            "The map refers to a section panel with panel_id {panel_id!r} to draw its "
            "path, but no panel in the figure has that id",
        "section_overlay_not_section":
            "The panel {panel_id!r} referred to by the map for a section path is not "
            "a vertical section (or its path cannot be determined)",
        "region_outside_grid":
            "The region lon [{lon_min:g}, {lon_max:g}] / lat [{lat_min:g}, "
            "{lat_max:g}] contains no grid points of this curvilinear grid "
            "(data lon [{data_lon_min:g}, {data_lon_max:g}], lat "
            "[{data_lat_min:g}, {data_lat_max:g}])",
        "degenerate_gridline":
            "Drawing failed because a gridline degenerated to a single point "
            "at the map boundary (a known cartopy issue: GEOS 'point array "
            "must contain 0 or >1 elements'). It happens at particular "
            "central longitudes/latitudes of curved-boundary projections "
            "such as Orthographic. Shift the central longitude/latitude "
            "slightly (for a rotation animation, move a waypoint or change "
            "the number of frames) or turn the gridlines off.",
    }

    def __init__(self, msg_id: str, **params):
        self.msg_id = msg_id
        self.params = params
        super().__init__(self.MESSAGES[msg_id].format(**params))


# --- 設定値 → 描画引数の解決 (render / scriptgen 共通) ---

# --- カスタムカラーマップ (~/.climcanvas/cmaps/) ---

# 各カラーマップの RGB データ (0-1 範囲) を保持。scriptgen が再現スクリプトに
# 埋め込むためにここから読み出す
_CUSTOM_CMAP_DATA: dict[str, list[tuple[float, float, float]]] = {}
_custom_cmaps_loaded_for_dir: str | None = None


def custom_cmap_dir() -> pathlib.Path:
    """カスタムカラーマップの保存ディレクトリ (テスト時は環境変数で上書き可)。"""
    override = os.environ.get("CLIMCANVAS_CMAP_DIR")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".climcanvas" / "cmaps"


def _parse_cmap_file(path: pathlib.Path) -> list[tuple[float, float, float]] | None:
    """RGB テキストファイルを (r,g,b) 0-1 タプルのリストにパース。

    - `#` で始まる行はコメント、空行は無視
    - 数値は空白またはカンマ区切り
    - 値の最大が 1 を超えていれば 0-255 範囲とみなして 1 に正規化
    - パース失敗時は None
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        rows.append((r, g, b))
    if len(rows) < 2:
        return None
    max_val = max(max(t) for t in rows)
    if max_val > 1.0:
        rows = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in rows]
    return rows


def load_custom_cmaps(force_reload: bool = False) -> list[str]:
    """カスタムカラーマップを `custom_cmap_dir()` から読み込んで matplotlib に登録する。

    各ファイル名 (拡張子除く) がカラーマップ名になる。返り値は登録に成功した名前のリスト。
    反転名 `<name>_r` も併せて登録する (matplotlib は組み込み cmap にしか `_r` を
    用意しないため、登録しないと「カラーマップを反転」で解決に失敗する。2026-09-29)。
    UI 層 (ui/constants.py) の import 時と render_figure の入口で呼ぶ。複数回呼んでも
    安全 (同じディレクトリなら最初の結果をキャッシュ)。
    """
    global _custom_cmaps_loaded_for_dir
    dir_path = custom_cmap_dir()
    dir_key = str(dir_path)
    if not force_reload and _custom_cmaps_loaded_for_dir == dir_key:
        return list(_CUSTOM_CMAP_DATA.keys())

    _CUSTOM_CMAP_DATA.clear()
    _custom_cmaps_loaded_for_dir = dir_key
    if not dir_path.is_dir():
        return []

    for path in sorted(dir_path.iterdir()):
        if path.suffix.lower() not in (".rgb", ".txt", ".dat"):
            continue
        rgb = _parse_cmap_file(path)
        if rgb is None:
            continue
        name = path.stem
        _CUSTOM_CMAP_DATA[name] = rgb
        cmap = mcolors.LinearSegmentedColormap.from_list(name, rgb)
        for reg_name, reg_cmap in ((name, cmap), (name + "_r", cmap.reversed())):
            try:
                mpl.colormaps.register(reg_cmap, name=reg_name, force=True)
            except (ValueError, TypeError):
                # 古い matplotlib 等で force= が未対応な場合は unregister 経由
                try:
                    mpl.colormaps.unregister(reg_name)
                except Exception:
                    pass
                mpl.colormaps.register(reg_cmap, name=reg_name)
    return list(_CUSTOM_CMAP_DATA.keys())


def custom_cmap_rgb(name: str) -> list[tuple[float, float, float]] | None:
    """カスタムカラーマップの RGB データを返す。登録されていなければ None。"""
    return _CUSTOM_CMAP_DATA.get(name)


def resolve_cmap(name: str, reverse: bool) -> str:
    if reverse:
        return name[:-2] if name.endswith("_r") else name + "_r"
    return name


def box_polygon(box: dict, n: int = 50):
    """矩形領域の境界 (4辺をサンプリングした閉路) の経度・緯度配列を返す。

    `lon_max < lon_min` のときは経度を東向きに連続させて (lon_max += 360) 日付変更線を
    またぐ矩形に対応する。`n` は1辺あたりのサンプル数 (投影歪みが滑らかに出る)。
    """
    lon_min = float(box["lon_min"])
    lon_max = float(box["lon_max"])
    lat_min = float(box["lat_min"])
    lat_max = float(box["lat_max"])
    if lon_max < lon_min:
        lon_max += 360.0
    lons = np.concatenate([
        np.linspace(lon_min, lon_max, n),
        np.full(n, lon_max),
        np.linspace(lon_max, lon_min, n),
        np.full(n, lon_min),
    ])
    lats = np.concatenate([
        np.full(n, lat_min),
        np.linspace(lat_min, lat_max, n),
        np.full(n, lat_max),
        np.linspace(lat_max, lat_min, n),
    ])
    return lons, lats


def hatch_pattern(style: dict) -> str:
    """ハッチパターン (単一文字) を density 回繰り返した最終文字列を返す。"""
    return str(style.get("pattern", "/")) * int(style.get("density", 3))


# ハッチ線の既定 (matplotlib の rcParams の既定と同じ。scriptgen はこれと違うときだけ出す)
HATCH_DEFAULT_LINEWIDTH = 1.0
HATCH_DEFAULT_COLOR = "#000000"


def hatch_rc_params(style: dict, *, only_changed: bool = False) -> dict:
    """ハッチ線の太さと色 (plt.rc_context に渡す rcParams)。render/scriptgen 共用。

    matplotlib はハッチの色・太さを artist 作成時の rcParams (hatch.color /
    hatch.linewidth) から取る (3.10.8 で実測。docs/cartopy_compat_notes.md)。
    only_changed=True は既定値と違う項目だけ (再現スクリプトを簡潔にする)。
    """
    lw = float(style.get("linewidth", HATCH_DEFAULT_LINEWIDTH))
    color = style.get("color") or HATCH_DEFAULT_COLOR
    params = {}
    if not only_changed or lw != HATCH_DEFAULT_LINEWIDTH:
        params["hatch.linewidth"] = lw
    if not only_changed or color.lower() != HATCH_DEFAULT_COLOR:
        params["hatch.color"] = color
    return params


def apply_value_transform(da, style: dict):
    """データ値の線形変換 `y = value_scale * x + value_offset` を適用。

    デフォルト (scale=1.0, offset=0.0) では何もしない。
    vmin/vmax/levels/colorbar など下流の設定はすべて変換後の値で指定する。
    """
    scale = style.get("value_scale", 1.0)
    offset = style.get("value_offset", 0.0)
    if scale == 1.0 and offset == 0.0:
        return da
    return da * scale + offset


def apply_maskout(da, style: dict, mask_da=None):
    """maskout: 閾値以下・以上の値を NaN にして描画から除外する (GrADS maskout 相当)。

    style["maskout"] = {"below": y, "above": x,
                        "variable": v, "var_below": yb, "var_above": xa}
      - below: y **以下**を描かない (da > y のみ残す)
      - above: x **以上**を描かない (da < x のみ残す)
      - var_below / var_above: 別変数 (variable) の値で同じ条件を掛ける。
        mask_da (変数 v を描画変数と同じ選択で切り出した DataArray) が
        渡されたときだけ適用する
    値変換 (value_scale/offset) の**後**に適用するので、閾値は変換後の値で指定する。
    マスク変数側の閾値は mask_da の生の値で指定する (値変換は掛からない)。
    旧設定 (maskout キーなし) では何もしない。
    """
    m = style.get("maskout") or {}
    below = m.get("below")
    above = m.get("above")
    if below is not None:
        da = da.where(da > float(below))
    if above is not None:
        da = da.where(da < float(above))
    if mask_da is not None:
        if m.get("var_below") is not None:
            da = da.where(mask_da > float(m["var_below"]))
        if m.get("var_above") is not None:
            da = da.where(mask_da < float(m["var_above"]))
    return da


def maskout_var_config(style: dict):
    """maskout の「別変数の値でマスク」設定 (variable, var_below, var_above) を返す。

    variable 未指定、または閾値が両方 None なら None (機能無効)。
    旧設定 (キーなし) も None。render と scriptgen の両方から呼ぶ。
    """
    m = style.get("maskout") or {}
    var = m.get("variable")
    below, above = m.get("var_below"), m.get("var_above")
    if not var or (below is None and above is None):
        return None
    return str(var), below, above


def require_increasing_levels(levels) -> list[float]:
    """境界値リスト (levels) が2個以上かつ狭義単調増加であることを検証して返す。

    UI (parse_float_list) は整列・重複除去済みを渡すので、これは設定 JSON を
    手編集した場合のガード。matplotlib の生エラー (Contour levels must be
    increasing 等) より原因の分かるメッセージを出す。render と scriptgen の
    両方から呼ぶ。
    """
    vals = [float(v) for v in levels]
    if len(vals) < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
        raise RenderError("levels_invalid", levels=list(levels))
    return vals


def fill_levels(style: dict):
    """vmin/vmax が指定されていれば等間隔レベル列、なければ分割数をそのまま返す。

    levels がリスト (境界値の直接指定) のときは検証して返す。
    """
    vmin, vmax = style.get("vmin"), style.get("vmax")
    n = style.get("levels", 21)
    if isinstance(n, int):
        if vmin is not None and vmax is not None:
            return np.linspace(vmin, vmax, n)
        return n
    return require_increasing_levels(n)


def fill_alpha(style: dict):
    """fill の透過度を解決する (render/scriptgen 共用)。

    1.0 (既定) は None を返して matplotlib の既定に任せる — alpha=1.0 を
    明示すると cmap 自身が持つ alpha を上書きしてしまうのと、旧 config の
    生成スクリプトを変えないため。
    """
    a = float(style.get("alpha", 1.0))
    return None if a >= 1.0 else a


def contour_levels(style: dict):
    """contour のレベル列を解決する (render/scriptgen 共用)。

    旧スキーマ (vmin/vmax/interval 全指定の arange) を互換のため優先し、
    無ければ新スキーマ levels (fill と同じ int/list、fill_levels に委譲)。
    どちらも無ければ None (matplotlib の自動レベル)。
    """
    vmin, vmax, interval = style.get("vmin"), style.get("vmax"), style.get("interval")
    if vmin is not None and vmax is not None and interval:
        return np.arange(vmin, vmax + 0.5 * interval, interval)
    if style.get("levels") is not None:
        return fill_levels(style)
    return None


def matplotlib_auto_levels(zmin: float, zmax: float, n: int, *, filled: bool,
                           extend: str = "neither") -> np.ndarray:
    """matplotlib が levels=N (整数) に対して実際に選ぶレベル列を、図を描かずに求める。

    ContourSet のレベル選択 (MaxNLocator(N+1) の目盛りをデータ範囲で切り詰め、
    extend のある側の端を落とす) は z の min/max・N・extend・塗りか線かだけで
    決まる。その私有ロジック (_autolev) を写す代わりに、同じ min/max を持つ
    2×2 のダミー配列を pyplot を使わない Figure 上で contour / contourf して
    cs.levels を読む — matplotlib の版が変わっても実描画と一致する
    (tests/test_levels_preview.py が乱数の実データと突き合わせる。
    cartopy_compat_notes.md「リスク低」節)。UI の参考表示 (描画レベル) 専用で、
    render 自身は従来どおり整数を matplotlib に渡す (図は変えない)。
    """
    from matplotlib.figure import Figure
    ax = Figure().add_subplot()
    z = np.array([[zmin, zmax], [zmax, zmin]], dtype=float)
    draw = ax.contourf if filled else ax.contour
    cs = draw(z, levels=int(n), extend=extend)
    return np.asarray(cs.levels, dtype=float)


def preview_levels(style: dict, da, *, kind: str):
    """UI の参考表示用に、この style で実際に描かれるレベル列を返す (描画はしない)。

    kind は "fill" / "contour"。da は render が matplotlib に渡す直前の
    DataArray (切り出し・値の変換・maskout 適用後 = layer_drawn_data)。
    「レベルを等間隔にする」(style["levels"] が int) のときだけ配列を返し、
    境界値の直接指定 (list)・旧スキーマの interval 指定・全欠損・定数データは
    None (ユーザーが値を知っている / レベルが定義できない)。
    解決順は各 _draw_* と同じ:
      contour — contour_levels() の結果。int (値の範囲を自動) なら
                matplotlib_auto_levels (線。extend は contour_kwargs と同条件)
      fill    — pcolormesh と zero_white は _fill_pcolormesh_kwargs の
                norm.boundaries (linspace + zero_white の端トリム)。それ以外の
                contourf は fill_levels() の結果。int なら matplotlib_auto_levels
                (塗り。extend 既定 both)
    """
    if not isinstance(style.get("levels", 21), int):
        return None
    vals = np.asarray(da, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return None
    zmin, zmax = float(finite.min()), float(finite.max())
    if zmin == zmax:
        return None
    if kind == "contour":
        vmin, vmax, interval = style.get("vmin"), style.get("vmax"), style.get("interval")
        if vmin is not None and vmax is not None and interval:
            return None
        lv = contour_levels(style)
        if isinstance(lv, int):
            extend = contour_kwargs(style).get("extend", "neither")
            lv = matplotlib_auto_levels(zmin, zmax, lv, filled=False, extend=extend)
        return np.asarray(lv, dtype=float)
    if kind != "fill":
        raise ValueError(f"preview_levels: unknown kind {kind!r}")
    if style.get("method", "contourf") == "pcolormesh":
        return np.asarray(_fill_pcolormesh_kwargs(style, da)[1].boundaries, dtype=float)
    if style.get("zero_white"):
        return np.asarray(
            _fill_pcolormesh_kwargs(style, da, contourf_levels=True)[1].boundaries,
            dtype=float)
    lv = fill_levels(style)
    if isinstance(lv, int):
        lv = matplotlib_auto_levels(zmin, zmax, lv, filled=True,
                                    extend=style.get("extend", "both"))
    return np.asarray(lv, dtype=float)


def colorbar_kwargs(cbar_cfg: dict, extend: str | None = None) -> dict:
    kwargs = {"location": cbar_cfg.get("location", "right")}
    if cbar_cfg.get("shrink") is not None:
        kwargs["shrink"] = cbar_cfg["shrink"]
    if cbar_cfg.get("aspect") is not None:
        kwargs["aspect"] = cbar_cfg["aspect"]
    if cbar_cfg.get("pad") is not None:
        kwargs["pad"] = cbar_cfg["pad"]
    if extend and extend != "neither":
        kwargs["extend"] = extend
    return kwargs


def colorbar_side_plan(cbar_cfg: dict) -> tuple[str, str, str] | None:
    """目盛りとラベルの表示サイド (axis 属性名, 目盛り側, ラベル側) を解決する
    (render/scriptgen 共用)。

    location=right/left は縦カラーバー (yaxis)、top/bottom は横 (xaxis)。
    flip_ticks = 目盛り (線・文字) を既定の反対側 (プロット側) に出す。
    label_opposite = ラベルだけを目盛りの反対側に出す (キーの無い旧設定は
    従来どおり目盛りと同じ側)。両方 False なら None (既定のまま切り替え不要)。
    fig.colorbar の ticklocation 引数は location 指定時に上書きされるため、
    作成後に set_ticks_position / set_label_position で切り替える。
    """
    flip_ticks = bool(cbar_cfg.get("flip_ticks"))
    label_opp = bool(cbar_cfg.get("label_opposite"))
    if not flip_ticks and not label_opp:
        return None
    sides = {"right": ("yaxis", "right", "left"),
             "left": ("yaxis", "left", "right"),
             "bottom": ("xaxis", "bottom", "top"),
             "top": ("xaxis", "top", "bottom")}
    if cbar_cfg.get("location", "right") not in sides:
        return None
    axis, default, opposite = sides[cbar_cfg.get("location", "right")]
    ticks_side = opposite if flip_ticks else default
    if not label_opp:
        label_side = ticks_side
    else:
        label_side = default if ticks_side == opposite else opposite
    return axis, ticks_side, label_side


def legend_kwargs(legend_cfg: dict) -> dict:
    """凡例 (ax.legend) の kwargs を解決する (render/scriptgen 共用)。

    x/y (axes 座標) が両方指定されていれば bbox_to_anchor で位置を直接指定する
    (枠外も可)。基準点は凡例の**左下角**に固定 (loc="lower left"、プリセットの
    loc は座標指定時には使わない)。
    """
    kw = {"loc": legend_cfg.get("loc", "best")}
    if legend_cfg.get("fontsize"):
        kw["fontsize"] = legend_cfg["fontsize"]
    if legend_cfg.get("x") is not None and legend_cfg.get("y") is not None:
        kw["bbox_to_anchor"] = (float(legend_cfg["x"]), float(legend_cfg["y"]))
        kw["loc"] = "lower left"
    return kw


def colorbar_label_kwargs(cbar_cfg: dict) -> dict:
    """カラーバー set_label() の kwargs (fontsize / labelpad) を返す (render/scriptgen 共用)。"""
    kw = {}
    if cbar_cfg.get("label_fontsize"):
        kw["fontsize"] = cbar_cfg["label_fontsize"]
    if cbar_cfg.get("label_pad") is not None:
        kw["labelpad"] = float(cbar_cfg["label_pad"])
    return kw


def contour_kwargs(style: dict) -> dict:
    """`use_cmap=True` なら `cmap=`、それ以外は単色 `colors=` を返す。

    `colors` と `cmap` は matplotlib の contour に同時指定できないため
    モードで排他にする。
    """
    base = {
        "linewidths": style.get("linewidth", 1.0),
        "linestyles": style.get("linestyle", "solid"),
    }
    if style.get("use_cmap"):
        base["cmap"] = resolve_cmap(style.get("cmap", "viridis"),
                                    style.get("reverse_cmap", False))
        # extend は色の割り当てとカラーバーの矢印に効く (neither は既定なので省略)
        ext = style.get("extend")
        if ext and ext != "neither":
            base["extend"] = ext
    else:
        base["colors"] = style.get("color", "black")
    return base


def apply_contour_linestyles(cs, style: dict) -> None:
    """正の値と負の値で線種を分ける後処理。

    `linestyle` 引数では全レベル一律にしか指定できないため、描画後に
    `cs.set_linestyle` でレベルごとの線種リストを当てる。負の値の線種が
    未指定 / 正と同じ場合は何もしない。
    """
    neg = style.get("negative_linestyle")
    pos = style.get("linestyle", "solid")
    if not neg or neg == pos:
        return
    cs.set_linestyle([neg if lev < 0 else pos for lev in cs.levels])


def apply_contour_emphasis(cs, style: dict) -> None:
    """指定したコンターレベルだけ太さ・色を変える後処理 (0 線の強調等)。

    style["emphasis"] = {"levels": [...], "linewidth": w, "color": c}。
    levels に含まれるレベルだけ w / c を当て、それ以外は元の値を保つ。
    linewidth / color が None なら該当項目は変えない。
    """
    emp = style.get("emphasis") or {}
    targets = emp.get("levels")
    lw = emp.get("linewidth")
    color = emp.get("color")
    if not targets or (lw is None and color is None):
        return
    targets = [float(v) for v in targets]
    is_emph = [any(abs(float(lev) - t) <= 1e-9 for t in targets)
               for lev in cs.levels]
    if lw is not None:
        base_lw = float(style.get("linewidth", 1.0))
        cs.set_linewidth([float(lw) if e else base_lw for e in is_emph])
    if color is not None:
        if style.get("use_cmap"):
            base_colors = list(cs.to_rgba(np.asarray(cs.levels, dtype=float)))
        else:
            base_colors = [style.get("color", "black")] * len(cs.levels)
        # 線コンターなので edgecolor を使う (set_color は face も塗り潰す)
        cs.set_edgecolor([color if e else base_colors[i]
                          for i, e in enumerate(is_emph)])


def contour_label_levels(style: dict, levels):
    """clabel に渡すレベル列を返す (None = 全レベルにラベル)。

    「指定レベルの強調・非表示」で太さ 0 (非表示) にしたレベルは、線と
    一緒にラベルも消す。levels は描画結果の cs.levels。レベルの一致判定は
    apply_contour_emphasis と同じ許容誤差。scriptgen の
    _contour_label_levels_lines と1対1対応。
    """
    emp = style.get("emphasis") or {}
    targets = emp.get("levels")
    lw = emp.get("linewidth")
    if not targets or lw is None or float(lw) != 0.0:
        return None
    targets = [float(v) for v in targets]
    return [float(lev) for lev in levels
            if not any(abs(float(lev) - t) <= 1e-9 for t in targets)]


def vector_v_dataset_id(layer: dict) -> str:
    """ベクトル系レイヤー (vector / stream) の y 成分を持つ dataset id。

    v_dataset_id が未設定 (None / 旧設定でキー無し) なら x 成分と同じ
    dataset_id を返す。render / scriptgen 共用。
    """
    return layer.get("v_dataset_id") or layer["dataset_id"]


def layer_dataset_ids(layer: dict) -> list[str]:
    """レイヤーが参照する dataset id の一覧 (先頭 = 主 dataset、重複なし)。

    「使用中の dataset」を集める側 (app.py の格子整合チェック等) は
    layer["dataset_id"] だけを見ずにこれを使う。現状 2 つ以上を参照するのは
    ベクトル系の v_dataset_id だけ。
    """
    ids = [layer["dataset_id"]]
    if layer.get("v_dataset_id") and layer["v_dataset_id"] not in ids:
        ids.append(layer["v_dataset_id"])
    return ids


def check_vector_shapes(u, v, layer: dict) -> None:
    """選択後の u / v が同じ格子 (次元名・形状) でなければ RenderError。

    別ファイルの成分を組み合わせたとき、格子が違うと np.hypot / quiver が
    broadcast エラーで不可解に落ちるため、原因の分かるメッセージで先に止める。
    """
    if tuple(u.dims) != tuple(v.dims) or u.shape != v.shape:
        raise RenderError("vector_shape_mismatch",
                          u_var=layer["u_variable"], u_shape=tuple(u.shape),
                          v_var=layer["v_variable"], v_shape=tuple(v.shape))


def vector_strides(style: dict) -> tuple[int, int]:
    """ベクトルの間引き間隔 (x方向, y方向) を返す。旧キー stride にも対応。"""
    fallback = style.get("stride", 1) or 1
    sx = style.get("stride_x", fallback) or 1
    sy = style.get("stride_y", fallback) or 1
    return int(sx), int(sy)


def vector_kwargs(style: dict) -> dict:
    """quiver の kwargs を解決する。

    use_cmap=True のとき矢印はベクトルの大きさ (呼び出し側が C 配列として渡す)
    で色付けされるため color は渡さず cmap を渡す。大きさの範囲 (vmin/vmax) は
    Normalize オブジェクトが生成コードに書けないため、quiver 作成後に
    set_clim で適用する (呼び出し側)。
    """
    if style.get("use_cmap"):
        kwargs = {"cmap": resolve_cmap(style.get("cmap", "viridis"),
                                       style.get("reverse_cmap", False))}
    else:
        kwargs = {"color": style.get("color", "black")}
    if style.get("width") is not None:
        kwargs["width"] = style["width"]
    if style.get("scale") is not None:
        kwargs["scale"] = style["scale"]
    if style.get("headlength") is not None:
        hl = float(style["headlength"])
        kwargs["headlength"] = hl
        # 頭の付け根の長さも既定比率 (4.5/5 = 0.9) を保って連動させ、
        # headlength だけ変えたときに矢じりの形が崩れないようにする
        kwargs["headaxislength"] = round(hl * 0.9, 6)
    if float(style.get("edge_width", 0.0) or 0.0) > 0:
        # 矢印ポリゴンの輪郭線。単色 (color=facecolor) とも cmap とも併用可
        kwargs["edgecolor"] = style.get("edge_color", "#000000")
        kwargs["linewidth"] = float(style["edge_width"])
    return kwargs


def stream_kwargs(style: dict) -> dict:
    """streamplot の kwargs を解決する (render/scriptgen 共用)。

    use_cmap=True のとき流線は大きさ |V| (呼び出し側が color= に2次元配列で
    渡す) で色付けされるため、ここでは color を渡さず cmap を渡す。
    vmin/vmax は Normalize が生成コードに書けないため streamplot 作成後に
    `.lines.set_clim` で適用する (呼び出し側)。
    """
    if style.get("use_cmap"):
        kwargs = {"cmap": resolve_cmap(style.get("cmap", "viridis"),
                                       style.get("reverse_cmap", False))}
    else:
        kwargs = {"color": style.get("color", "black")}
    kwargs["density"] = float(style.get("density", 1.0))
    kwargs["linewidth"] = float(style.get("linewidth", 1.0))
    kwargs["arrowsize"] = float(style.get("arrowsize", 1.0))
    return kwargs


def stream_uniform_coords(values) -> np.ndarray | None:
    """streamplot 用の等間隔・昇順座標を返す (render/scriptgen 共用)。

    matplotlib の streamplot は**昇順かつ等間隔**の座標を要求する
    (気圧レベルなどは不等間隔なのでそのままでは描けない)。既に条件を
    満たしていれば None (補間不要)。そうでなければ同じ端点・同じ点数の
    等間隔グリッド (np.linspace) を返し、呼び出し側が `.interp()` で
    線形補間する。
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return None
    lo = float(min(v[0], v[-1]))
    hi = float(max(v[0], v[-1]))
    uniform = np.linspace(lo, hi, v.size)
    if v[0] < v[-1] and np.allclose(v, uniform, rtol=0.0,
                                    atol=abs(hi - lo) * 1e-9):
        return None
    return uniform


def resolve_discrete_levels(style: dict):
    """style["levels"] を正規化して返す (render/scriptgen 共用)。

    - None / 0 → None (連続)
    - int → そのまま (レベル数。境界は vmin/vmax から linspace)
    - list → 昇順・重複除去した境界値リスト (2個未満なら None = 連続扱い)
    """
    nlev = style.get("levels")
    if not nlev:
        return None
    if isinstance(nlev, (list, tuple)):
        lv = sorted({float(v) for v in nlev})
        return lv if len(lv) >= 2 else None
    return int(nlev)


def _discrete_level_array(style: dict, data_min: float, data_max: float):
    """style["levels"] を具体的なレベル列に解決する (連続なら None)。

    int は vmin/vmax (未指定はデータ範囲) の linspace、list は境界値そのもの。
    """
    nlev = resolve_discrete_levels(style)
    if nlev is None:
        return None
    if isinstance(nlev, list):
        return np.asarray(nlev, dtype=float)
    vmin = style.get("vmin")
    vmax = style.get("vmax")
    if vmin is None:
        vmin = float(data_min)
    if vmax is None:
        vmax = float(data_max)
    return np.linspace(vmin, vmax, nlev)


def zero_white_colors(cmap, lv, extend: str):
    """レベル列 lv の各ビン (+extend の端ビン) の色列を作り、0 を含むビンを白にする。

    色は cmap の等間隔サンプリング (extend の端ビン込み)。0 を含むビンは
    lv[i] <= 0 <= lv[i+1] で判定するため、0 がレベル境界に一致するときは
    境界の両隣2ビンが白になる。scriptgen._zero_white_lines と1対1対応。
    """
    lv = np.asarray(lv, dtype=float)
    nbin = len(lv) - 1
    n_low = 1 if extend in ("min", "both") else 0
    n_extra = n_low + (1 if extend in ("max", "both") else 0)
    colors = cmap(np.linspace(0.0, 1.0, nbin + n_extra))
    for i in range(nbin):
        if lv[i] <= 0.0 <= lv[i + 1]:
            colors[n_low + i] = (1.0, 1.0, 1.0, 1.0)
    return colors


def discrete_cmap_norm(style: dict, data_min: float, data_max: float):
    """style["levels"] から離散化用の BoundaryNorm を作る (連続なら None)。

    ベクトル (大きさ)・バブル (z 値)・heatmap の cmap 離散化で共用する。
    fill の pcolormesh と同じ仕組み。style に "extend" があれば範囲外の色を
    確保する (heatmap。extend キーの無い vector/bubble 等は従来どおり neither)。
    """
    lv = _discrete_level_array(style, data_min, data_max)
    if lv is None:
        return None
    cmap_name = resolve_cmap(style.get("cmap", "viridis"),
                             style.get("reverse_cmap", False))
    return mcolors.BoundaryNorm(lv, ncolors=plt.get_cmap(cmap_name).N,
                                extend=style.get("extend", "neither"))


def discrete_color_kwargs(style: dict, data_min: float, data_max: float) -> dict:
    """離散化の色 kwargs を返す ({} = 連続)。

    通常は {"norm": BoundaryNorm} (cmap は呼び出し側の解決のまま)。
    zero_white 有効時は 0 を含むビンを白に置き換えた
    {"cmap": ListedColormap, "norm": BoundaryNorm} を返すので、呼び出し側は
    通常の cmap 解決より後に merge して上書きすること。
    scriptgen._discrete_color_exprs と1対1対応。
    """
    if not style.get("zero_white"):
        norm = discrete_cmap_norm(style, data_min, data_max)
        return {"norm": norm} if norm is not None else {}
    lv = _discrete_level_array(style, data_min, data_max)
    if lv is None:
        return {}
    cmap_name = resolve_cmap(style.get("cmap", "viridis"),
                             style.get("reverse_cmap", False))
    extend = style.get("extend", "neither")
    colors = zero_white_colors(plt.get_cmap(cmap_name), lv, extend)
    # extend の端色も本体に含む ListedColormap + BoundaryNorm(extend=)。
    # (from_levels_and_colors は norm.extend が neither になり、カラーバーの
    # 矢印が出なくなるため使わない)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(lv, ncolors=len(colors), extend=extend)
    return {"cmap": cmap, "norm": norm}


def continuous_cbar_extend(style: dict) -> str | None:
    """連続カラーマップ時のカラーバー extend を返す (render/scriptgen 共用)。

    離散化 (levels 指定) 時は BoundaryNorm(extend=) が矢印を持つため None
    (colorbar への二重指定を避ける)。extend 未指定・neither も None。
    """
    if resolve_discrete_levels(style) is not None:
        return None
    ext = style.get("extend")
    return ext if ext and ext != "neither" else None


# Natural Earth の解像度 (1:110m / 1:50m / 1:10m)
NE_SCALES = ("110m", "50m", "10m")


def natural_earth_scale(map_cfg: dict) -> str | None:
    """map.resolution を Natural Earth のスケール名に解決する。

    海岸線 (`ax.coastlines(resolution=)`) と LAND / OCEAN / BORDERS
    (`feature.with_scale()`) に共通で使う。None = cartopy の自動
    (`cartopy.feature.auto_scaler`: 描画時の表示範囲の短辺が 50° 以下で 50m、
    15° 以下で 10m、それ以外は 110m。0.25 実測)。旧設定 (キー無し) も自動。
    """
    res = map_cfg.get("resolution", "auto")
    if res in (None, "", "auto"):
        return None
    if res not in NE_SCALES:
        raise ValueError(f"map.resolution は {NE_SCALES} か 'auto': {res!r}")
    return res


# 陸を前景に描く (map.land.above_data) ときの zorder。matplotlib / cartopy の既定は
# 塗り・ハッチ・ベクトル・散布点・トラックの点 = 1、基準ベクトル = 1.1、海岸線・
# 国境線 = 1.5、等値線・流線・トラックの線・グリッド線・box・記号注記 = 2、文字 = 3、
# 等値線ラベル = 4 (cartopy 0.25 / matplotlib 3.10 実測)。陸は面の格子データ (塗り・
# ハッチ・ベクトル) の上、海岸線・等値線・流線の下 (1.4) に置く。地点を表す点
# (散布点・トラックの点) と基準ベクトルは陸に隠れてはいけないので海岸線の高さ (1.5)
# に上げる (海岸線は後から追加されるので点の上に描かれる)。cartopy の Gridliner は
# 子の線を自分の draw (zorder 2) で描くので子の zorder は効かない (実測: 線に 2.6 を
# 渡しても陸 2.5 の下に消えた) → 陸を 2 未満に置くことでグリッド線を隠さない
LAND_FG_ZORDER = 1.4
MAP_FG_ZORDER = 1.5


def land_foreground(map_cfg: dict) -> bool:
    """map.land.above_data (陸をデータの上に描く) を解決する。旧設定 (キー無し) は False。"""
    return bool((map_cfg.get("land") or {}).get("above_data"))


def panel_land_foreground(panel: dict) -> bool:
    """panel が地図で陸を前景に描くとき True (レイヤー・注記の描画側の判定用)。"""
    return (panel.get("plot_type") == "horizontal_map"
            and land_foreground(panel.get("map") or {}))


def gridline_kwargs(gl_cfg: dict, proj_cfg: dict | None = None,
                    draw_labels: bool | None = None) -> dict:
    kwargs = {
        "draw_labels": (gl_cfg.get("labels", True) if draw_labels is None
                        else draw_labels),
        "linewidth": gl_cfg.get("width", 0.5),
        "color": gl_cfg.get("color", "gray"),
        "linestyle": gl_cfg.get("linestyle", ":"),
    }
    # LambertConformal は経度線が曲線なので cartopy の既定では経度ラベルが図中
    # に inline 描画される。図の外 (枠線上) に出すよう強制する
    if proj_cfg and proj_cfg.get("name") == "LambertConformal":
        kwargs["x_inline"] = False
        kwargs["y_inline"] = False
    # 極投影の「緯度ラベルを枠沿いに」: 図中 (inline) をやめ、緯度線が境界
    # (扇の縁) と交差する位置に置く。ラベルは geo 分類になる (円周沿いの
    # 経度ラベルと同じフラグの管轄)。全経度の円形では緯度線が枠と交差しない
    # ためラベルは出ない (UI で案内)。scriptgen も本関数を使うため 1対1
    if (proj_cfg and proj_cfg.get("name") in POLAR_PROJECTIONS
            and gl_cfg.get("lat_label_placement", "inline") == "edge"):
        kwargs["y_inline"] = False
    return kwargs


def gridline_label_styles(gl_cfg: dict) -> tuple[dict, dict]:
    """緯度経度線ラベルの (xlabel_style, ylabel_style) を返す (render/scriptgen 共用)。

    label_rotation は経度ラベル (x) のみに適用する。回転時は回転の向きに応じて
    ha と rotation_mode="anchor" を付け、目盛り位置にラベル端が揃うようにする。
    lat_label_rotation は緯度ラベル (y) のみ。None = 自動 (極投影では cartopy が
    縁・経度線の向きに沿って回転する) で、0 は「水平に固定」の意味を持つため
    None と区別する。style の rotation は cartopy の自動回転で決めた角度より
    後に適用されるので上書きできる (cartopy 0.25 実測、compat_notes 参照)。
    """
    x_style: dict = {}
    y_style: dict = {}
    fs = gl_cfg.get("label_fontsize")
    if fs:
        x_style["size"] = fs
        y_style["size"] = fs
    rot = gl_cfg.get("label_rotation")
    if rot:
        x_style.update({"rotation": rot,
                        "ha": "right" if rot > 0 else "left",
                        "rotation_mode": "anchor"})
    lat_rot = gl_cfg.get("lat_label_rotation")
    if lat_rot is not None:
        y_style.update({"rotation": lat_rot, "rotation_mode": "anchor"})
    return x_style, y_style


def gridline_plan(gl_cfg: dict) -> dict:
    """緯度経度線の「線」と「ラベル」の構成を解決する (render / scriptgen 共用)。

    cartopy の Gridliner は線とラベルで locator を共有するため、ラベルの間隔・
    開始値が線の locator と異なるときだけ「線用」と「ラベル用 (線非表示)」の
    2つに分割する (split=True)。ラベルの間隔が None のときは線の間隔に従う
    (従来と同じ挙動)。

    label_lon_start / label_lat_start はラベルの開始値 (位相): 「開始値 + n×間隔」
    の位置に置く (n は負も含む)。0 は倍数と同位相なので None に正規化し、
    実効間隔が無い (自動 locator の) 軸では効果を持たないため落とす。
    """
    lines_on = bool(gl_cfg.get("lines", True))
    labels_on = bool(gl_cfg.get("labels", True))
    line_lon = gl_cfg.get("lon_interval")
    line_lat = gl_cfg.get("lat_interval")
    label_lon = gl_cfg.get("label_lon_interval") or line_lon
    label_lat = gl_cfg.get("label_lat_interval") or line_lat
    label_lon_start = (gl_cfg.get("label_lon_start") or None) if label_lon else None
    label_lat_start = (gl_cfg.get("label_lat_start") or None) if label_lat else None
    split = (lines_on and labels_on
             and (label_lon != line_lon or label_lat != line_lat
                  or label_lon_start is not None or label_lat_start is not None))
    return {"lines": lines_on, "labels": labels_on, "split": split,
            "line_lon": line_lon, "line_lat": line_lat,
            "label_lon": label_lon, "label_lat": label_lat,
            "label_lon_start": label_lon_start, "label_lat_start": label_lat_start}


def polar_sector_edge_lons(region: dict | None, proj_cfg: dict) -> tuple[float, float] | None:
    """扇形の (左の縁の経度, 右の縁の経度) を返す (render / scriptgen 共用)。

    左右は「縁の中間緯度の点を投影した x 座標」の大小で判定するので、
    投影の向き (北極/南極、中心経度のずれ) によらず視覚的な左右に対応する。
    全経度 (縁が無い) や region 未指定は None。
    """
    if region is None or region_spans_all_longitudes(region):
        return None
    proj = make_projection(proj_cfg)
    latm = 0.5 * (region["lat_min"] + region["lat_max"])
    x0 = proj.transform_point(region["lon_min"], latm, ccrs.PlateCarree())[0]
    x1 = proj.transform_point(region["lon_max"], latm, ccrs.PlateCarree())[0]
    if x0 <= x1:
        return (float(region["lon_min"]), float(region["lon_max"]))
    return (float(region["lon_max"]), float(region["lon_min"]))


def gridline_label_filters(gl_cfg: dict, proj_cfg: dict | None,
                           region: dict | None) -> dict:
    """draw ラッパで適用するラベルフィルタの構成 (render / scriptgen 共用の判定)。

    極投影のみ対象。返り値のキー:
    - "hide_pole": True — 極 (投影原点) 付近の経度ラベルを消す (pole_label=False)
    - "lat_edge_keep_lon" / "lat_edge_other_lon": 枠沿い配置の緯度ラベルを
      扇の片方の縁 (keep_lon 側) だけに絞る。左右→縁の経度の解決は
      polar_sector_edge_lons
    非極投影 (Robinson / EqualEarth / Orthographic = 境界が湾曲する投影) では:
    - "geo_hidden_sides": [...] — 辺から離れて geo に分類されたラベル
      (Robinson の高緯度ラベル等) を位置で最寄りの辺 (緯度=左右、経度=上下)
      に帰属させ、OFF の辺のものを消す。これで「ラベルを表示する辺」の
      チェックが全ラベルに効く
    空 dict ならラッパ不要。
    """
    if proj_cfg is None:
        return {}
    if proj_cfg.get("name") not in POLAR_PROJECTIONS:
        if proj_cfg.get("name") not in ("Robinson", "EqualEarth", "Orthographic"):
            return {}  # 矩形境界の投影は geo ラベルが出ない (ラッパ不要)
        sides = gl_cfg.get("label_sides", {})
        off = [s for s in ("left", "right", "top", "bottom")
               if not sides.get(s, True)]
        return {"geo_hidden_sides": off} if off else {}
    filters: dict = {}
    if not gl_cfg.get("pole_label", True):
        filters["hide_pole"] = True
    side = gl_cfg.get("lat_label_edge_side", "both")
    if (gl_cfg.get("lat_label_placement", "inline") == "edge"
            and side in ("left", "right")):
        lons = polar_sector_edge_lons(region, proj_cfg)
        if lons is not None:
            left_lon, right_lon = lons
            filters["lat_edge_keep_lon"] = (
                left_lon if side == "left" else right_lon)
            filters["lat_edge_other_lon"] = (
                right_lon if side == "left" else left_lon)
    return filters


def apply_gridline_label_filters(gl, ax, filters: dict) -> None:
    """位置に基づく緯度経度ラベルのフィルタを Gridliner に組み込む。

    cartopy の Gridliner は draw のたびにラベルと可視性を再計算し、ラベルは
    gl.draw の内部で描画されるため、「位置計算 (_draw_gridliner) → ラベル描画」
    の間に割り込んで非表示にする。filters は gridline_label_filters の返り値:
    - hide_pole: 極 = 投影原点から図サイズ 5% 以内の経度ラベルを消す
      (実測: 極ラベルは原点から ~1e5 m、弧沿いは ~7e6 m。極そのものは
      transform_point が NaN を返すため原点を直接使う)
    - lat_edge_keep_lon / lat_edge_other_lon: 枠沿い配置の緯度ラベル
      (geo 分類、位置は投影座標) を経緯度に逆変換し、経度の角距離が
      keep_lon 側に近いものだけ残す (扇の縁そのものへの帰属で判定。
      ラベルは縁から少しオフセットして置かれるため経度は数度ずれ得るが、
      縁同士の間隔より十分小さい)
    - geo_hidden_sides: geo 分類のラベルを位置で最寄りの辺 (緯度ラベル =
      xlim 中央より左/右、経度ラベル = ylim 中央より下/上) に帰属させ、
      リストの辺のものを消す (Robinson 等の湾曲境界用)
    scriptgen の emit と1対1。
    """
    hide_pole = filters.get("hide_pole", False)
    keep_lon = filters.get("lat_edge_keep_lon")
    other_lon = filters.get("lat_edge_other_lon")
    geo_hidden = filters.get("geo_hidden_sides") or ()

    def _draw(renderer=None):
        gl._draw_gridliner(renderer=renderer)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        tol = 0.05 * max(x1 - x0, y1 - y0)
        xmid = 0.5 * (x0 + x1)
        ymid = 0.5 * (y0 + y1)
        for lb in gl._labels:
            lx, ly = lb.artist.get_position()
            if (hide_pole and lb.xy == "x"
                    and abs(lx) < tol and abs(ly) < tol):
                lb.artist.set_visible(False)
            if keep_lon is not None and lb.xy == "y":
                plon = ccrs.PlateCarree().transform_point(
                    lx, ly, ax.projection)[0]
                d_keep = abs((plon - keep_lon + 180.0) % 360.0 - 180.0)
                d_other = abs((plon - other_lon + 180.0) % 360.0 - 180.0)
                if d_keep > d_other:
                    lb.artist.set_visible(False)
            if geo_hidden and lb.loc == "geo":
                if lb.xy == "y":
                    side = "left" if lx < xmid else "right"
                else:
                    side = "bottom" if ly < ymid else "top"
                if side in geo_hidden:
                    lb.artist.set_visible(False)
        for c in gl.get_visible_children():
            c.draw(renderer=renderer)
    gl.draw = _draw


def coord_slice(coord: xr.DataArray, vmin, vmax) -> slice:
    """座標の昇順・降順に合わせた slice を返す (数値・時刻どちらの座標にも対応)。

    vmin / vmax はどちらの順で渡してもよい (気圧レベルの [1000, 200] など)。
    """
    lo, hi = (vmin, vmax) if vmin <= vmax else (vmax, vmin)
    if bool(coord[0] > coord[-1]):
        return slice(hi, lo)
    return slice(lo, hi)


def coord_covers_globe(coord: xr.DataArray) -> bool:
    """経度座標が全周 (360°) を覆っているかを判定する。"""
    v = coord.values
    if v.size < 3 or v[1] <= v[0]:
        return False
    return abs((v[-1] - v[0]) + (v[1] - v[0]) - 360.0) < 1e-3


def padded_slice(coord: xr.DataArray, vmin: float, vmax: float) -> slice:
    """領域指定のデータ切り出し用 slice (1格子分外側に広げる)。

    切り出しを表示範囲ちょうどにすると、扇形境界などのクリップとの間に
    白い隙間が出るため、塗りつぶしが境界の外まで届くようにする。
    """
    v = coord.values
    step = abs(float(v[1] - v[0])) if v.size > 1 else 0.0
    return coord_slice(coord, vmin - step, vmax + step)


def region_indexers(ds: xr.Dataset, region: dict, roles: dict) -> dict:
    """領域指定によるデータ切り出しの indexer を返す (render / scriptgen 共通)。

    経度が全周あるデータでは経度を切り出さない (cyclic point 処理に任せ、
    どの経度範囲でも継ぎ目・隙間なく描けるようにする)。
    """
    lat_name, lon_name = roles["lat"], roles["lon"]
    indexers = {lat_name: padded_slice(ds[lat_name], region["lat_min"], region["lat_max"])}
    if not coord_covers_globe(ds[lon_name]):
        indexers[lon_name] = padded_slice(ds[lon_name], region["lon_min"], region["lon_max"])
    return indexers


def unwrap_lon2d(lon2d) -> np.ndarray:
    """2 次元の経度 (y, x) を格子上で連続になるようにほどく (日付変更線・0° の継ぎ目対策)。

    -180〜180 規約の格子が 180° を越えると隣の列で 179 → -179 と飛び、contour /
    contourf / pcolormesh はその継ぎ目をまたいで (経度空間で) 計算されるため、線が
    図を横切る・交差するなどの崩れが出る。各行を period 360 でほどき、行どうしは
    先頭列をほどいて揃える。継ぎ目の無い格子 (ClimCORE 等) では値は変わらない。
    cartopy の transform=PlateCarree() は 180° を越える経度をそのまま扱える。
    scriptgen が出す np.unwrap の行と同じ計算であること。
    """
    lon = np.asarray(lon2d, dtype=float)
    lon_u = np.unwrap(lon, axis=1, period=360.0)
    lon_u = lon_u + (np.unwrap(lon_u[:, 0], period=360.0) - lon_u[:, 0])[:, None]
    return lon_u


def curvilinear_region_indexers(ds: xr.Dataset, region: dict, roles: dict) -> dict:
    """2 次元座標格子 (curvilinear) の領域切り出し — `isel` 用の整数 slice を返す。

    lat/lon が dim ではないので `.sel(lat=slice(...))` は使えない。領域内の格子点
    (中心が範囲内) のマスクを取り、その行・列の外接矩形を 1 格子ずつ外側に広げた
    整数 slice `{y: slice(i0, i1), x: slice(j0, j1)}` を返す (1 次元格子の
    padded_slice と同じ「境界で塗りが切れない」配慮)。整数なので scriptgen は
    同じ `.isel(...)` をそのまま出せる (1 対 1)。

    - 経度は `(lon - lon_min) % 360 ≤ (lon_max - lon_min) % 360` で比較し、0/360 と
      -180/180 の規約差や日付変更線をまたぐ指定にも対応する (幅 360° 以上は全経度)
    - 範囲外の格子点を NaN にはしない (contourf の縁が崩れる。表示範囲は
      set_extent が切る)。色の自動範囲は矩形内の値に基づく
    - 該当格子点が無ければ RenderError("region_outside_grid")
    render / scriptgen 共通。
    """
    lat = ds[roles["lat"]]
    lon = ds[roles["lon"]]
    ydim, xdim = (str(d) for d in lat.dims)
    lat_v = np.asarray(lat.values, dtype=float)
    lon_v = np.asarray(lon.values, dtype=float)
    lon_min, lon_max = float(region["lon_min"]), float(region["lon_max"])
    lat_lo = min(float(region["lat_min"]), float(region["lat_max"]))
    lat_hi = max(float(region["lat_min"]), float(region["lat_max"]))
    width = lon_max - lon_min
    if width >= 360.0:
        lon_ok = np.ones(lon_v.shape, dtype=bool)
    else:
        lon_ok = ((lon_v - lon_min) % 360.0) <= (width % 360.0)
    mask = (lon_ok & (lat_v >= lat_lo) & (lat_v <= lat_hi)
            & np.isfinite(lat_v) & np.isfinite(lon_v))
    if not mask.any():
        raise RenderError("region_outside_grid",
                          lon_min=lon_min, lon_max=lon_max,
                          lat_min=lat_lo, lat_max=lat_hi,
                          data_lon_min=float(np.nanmin(lon_v)),
                          data_lon_max=float(np.nanmax(lon_v)),
                          data_lat_min=float(np.nanmin(lat_v)),
                          data_lat_max=float(np.nanmax(lat_v)))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    i0 = max(int(rows[0]) - 1, 0)
    i1 = min(int(rows[-1]) + 2, lat_v.shape[0])
    j0 = max(int(cols[0]) - 1, 0)
    j1 = min(int(cols[-1]) + 2, lat_v.shape[1])
    return {ydim: slice(i0, i1), xdim: slice(j0, j1)}


def extent_args(region: dict, proj_cfg: dict) -> tuple[list[float], float]:
    """set_extent の引数を返す。

    経度を投影の中心経度基準 (-180〜+180) に変換することで、0–360 規約のデータや
    日付変更線をまたぐ領域 (例: 60–200°E) でも正しく表示範囲を設定できる。
    領域が中心経度から±180°を超える場合は表示できないため、中心経度を領域の
    中央付近に設定する必要がある (PlateCarree + 範囲指定では UI が範囲の中央を
    自動使用。他投影は手動)。経度幅が 360° 以上の領域は全経度扱いになり
    extent では制限しない — 図の左右端 (継ぎ目) は中心経度だけが決める。
    """
    clon = proj_cfg.get("central_longitude", 0.0)
    lon_lo, lon_hi = float(region["lon_min"]), float(region["lon_max"])
    if lon_hi == lon_lo:
        # 退化領域 (単一観測点等)。そのまま wrap 計算に進むと x1 <= x0 の
        # 補正で全経度扱いになってしまうため、点の周囲 ±1° の窓に広げる
        lon_lo -= 1.0
        lon_hi += 1.0
    x0 = ((lon_lo - clon + 180.0) % 360.0) - 180.0
    x1 = ((lon_hi - clon + 180.0) % 360.0) - 180.0
    if x1 <= x0:
        x1 += 360.0
    if x1 - x0 >= 360.0:  # 全経度 (極投影の緯度キャップなど)
        x0, x1 = -180.0, 180.0
    # Robinson 等では極に近い extent が NaN になる (実測で ±89.5 が限界) ため
    # クランプする。全球図で 0.5° の差は視認できない
    y0 = max(float(region["lat_min"]), -89.5)
    y1 = min(float(region["lat_max"]), 89.5)
    if y0 == y1:
        # 高さゼロは set_ylim の特異変換警告 (cartopy が自動拡張) になる
        y0 = max(y0 - 1.0, -89.5)
        y1 = min(y1 + 1.0, 89.5)
    return [x0, x1, y0, y1], clon


def extent_is_global(extent: list[float]) -> bool:
    """extent が実質全球 (set_extent 不要) かを判定する。"""
    return (extent[0] <= -180.0 and extent[1] >= 180.0
            and extent[2] <= -89.5 and extent[3] >= 89.5)


POLAR_PROJECTIONS = ("NorthPolarStereo", "SouthPolarStereo")


def curvilinear_extent_dataset(panel: dict, datasets: dict) -> str | None:
    """「領域未指定の 2 次元座標格子」の表示範囲に使う dataset id を返す (無ければ None)。

    2 次元座標 (curvilinear) のデータは領域未指定でも全球ではなく、全格子点を投影座標に
    変換した最小の長方形 (ランベルト格子をランベルトで描けば余白なしの長方形) を
    表示範囲にする (ユーザー要望 2026-09-24)。対象はパネルのレイヤーが参照する dataset
    のうち最初の curvilinear なもの。region 指定あり / 極投影 (円形・扇形の枠) /
    Orthographic (表示範囲を制限しない) では None を返し、従来の経路 (set_extent /
    set_global) を通す。1 次元格子のデータは従来どおり (領域未指定 = 全球)。
    render._render_horizontal_map と scriptgen.map_panel で同じ条件 (1 対 1)。
    """
    if panel.get("region"):
        return None
    proj_cfg = panel.get("projection", {})
    if (use_circular_boundary(proj_cfg)
            or proj_cfg.get("name", "PlateCarree") == "Orthographic"):
        return None
    for layer in panel.get("layers", []):
        for dsid in layer_dataset_ids(layer):
            ds = datasets.get(dsid)
            if ds is not None and is_curvilinear(ds):
                return dsid
    return None


def curvilinear_extent(ds: xr.Dataset, projection: ccrs.Projection) -> list[float] | None:
    """全格子点を projection に変換した最小の長方形 [x0, x1, y0, y1] (投影座標)。

    投影できない点 (inf / NaN) は除く。有限な点が無ければ None。
    scriptgen が出す transform_points の行と同じ計算であること。
    """
    roles = detect_coord_roles(ds)
    pts = projection.transform_points(ccrs.PlateCarree(),
                                      np.asarray(ds[roles["lon"]].values, dtype=float),
                                      np.asarray(ds[roles["lat"]].values, dtype=float))
    px, py = pts[..., 0], pts[..., 1]
    ok = np.isfinite(px) & np.isfinite(py)
    if not ok.any():
        return None
    return [float(px[ok].min()), float(px[ok].max()),
            float(py[ok].min()), float(py[ok].max())]


def use_circular_boundary(proj_cfg: dict) -> bool:
    """極投影で図の枠を円形にするか (デフォルトは円形)。"""
    return (proj_cfg.get("name") in POLAR_PROJECTIONS
            and proj_cfg.get("circular_boundary", True))


def region_spans_all_longitudes(region: dict) -> bool:
    return region["lon_max"] - region["lon_min"] >= 360.0 - 1e-6


def sector_vertices(region: dict) -> list[list[float]]:
    """領域の輪郭 (緯度経度の矩形 → 極投影では扇形) の頂点を返す。

    極付近 (緯度±89.5°超) は座標変換が NaN になるため ±89.5° にクランプする
    (extent_args と同じ限界値)。
    """
    x0, x1 = float(region["lon_min"]), float(region["lon_max"])
    y0 = max(float(region["lat_min"]), -89.5)
    y1 = min(float(region["lat_max"]), 89.5)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def sector_boundary_path(region: dict, projection: ccrs.Projection) -> mpath.Path:
    """扇形の境界パスを投影座標で返す。

    CRS変換つきの set_boundary は経度180°をまたぐパスを切り刻んで空にする
    ことがあるため、頂点を自前で投影座標に変換してデータ座標のパスにする。
    scriptgen が生成するコードと同じ構成であること。
    """
    verts = mpath.Path(sector_vertices(region)).interpolated(50).vertices
    xy = projection.transform_points(ccrs.PlateCarree(), verts[:, 0], verts[:, 1])[:, :2]
    return mpath.Path(xy)


def circular_boundary_path() -> mpath.Path:
    """軸座標系 (0–1) で中心 (0.5, 0.5)・半径 0.5 の円パスを返す。

    scriptgen が生成するコードと同じ構成であること。
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    return mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + [0.5, 0.5])


def make_projection(proj_cfg: dict) -> ccrs.Projection:
    name = proj_cfg.get("name", "PlateCarree")
    clon = proj_cfg.get("central_longitude", 0.0)
    if name == "PlateCarree":
        return ccrs.PlateCarree(central_longitude=clon)
    if name == "Robinson":
        return ccrs.Robinson(central_longitude=clon)
    if name == "EqualEarth":
        return ccrs.EqualEarth(central_longitude=clon)
    if name == "NorthPolarStereo":
        return ccrs.NorthPolarStereo(central_longitude=clon)
    if name == "SouthPolarStereo":
        return ccrs.SouthPolarStereo(central_longitude=clon)
    if name == "Orthographic":
        return ccrs.Orthographic(
            central_longitude=clon,
            central_latitude=proj_cfg.get("central_latitude", 0.0))
    if name == "LambertConformal":
        sp = proj_cfg.get("standard_parallels", [33.0, 45.0])
        return ccrs.LambertConformal(
            central_longitude=clon,
            central_latitude=proj_cfg.get("central_latitude", 39.0),
            standard_parallels=tuple(sp))
    raise RenderError("unsupported_projection", name=name)


def select_panel_data(ds: xr.Dataset, variable: str, selection: dict | None,
                      region: dict | None, roles: dict) -> xr.DataArray:
    """変数から固定次元・領域を切り出し、(lat, lon) の次元順に揃えて返す。"""
    da = ds[variable]
    sel = {dim: val for dim, val in (selection or {}).items() if dim in da.dims}
    if sel:
        da = da.sel(sel)
    curvi = is_curvilinear(ds, roles)
    if region:
        if curvi:
            # lat/lon が dim でない格子: マスクの外接矩形を整数 isel で切る
            da = da.isel(curvilinear_region_indexers(ds, region, roles))
        else:
            da = da.sel(region_indexers(ds, region, roles))
    if curvi:
        # 経緯度に NaN / inf があると matplotlib の pcolormesh / contourf が不可解に
        # 落ちる (海洋モデルの陸面などで経緯度が未定義の格子) ため、原因の分かる
        # エラーで先に止める (切り出し後の範囲だけを検査)
        lat_v = np.asarray(da[roles["lat"]].values, dtype=float)
        lon_v = np.asarray(da[roles["lon"]].values, dtype=float)
        bad = int((~np.isfinite(lat_v)).sum() + (~np.isfinite(lon_v)).sum())
        if bad:
            raise RenderError("curvilinear_coords_nonfinite",
                              lon=roles["lon"], lat=roles["lat"], n=bad)
    # 1 次元格子では (lat, lon)、curvilinear では lat 座標の dims (y, x)
    da = da.transpose(*horizontal_dims(ds, roles))
    if curvi:
        # 経度の継ぎ目 (179 → -179) をほどいて格子上で連続にする (scriptgen と 1 対 1)
        lon_name = roles["lon"]
        da = da.assign_coords({lon_name: (da[lon_name].dims,
                                          unwrap_lon2d(da[lon_name].values),
                                          dict(da[lon_name].attrs))})
    return da


def canon_lon_range_for_averaging(lo: float, hi: float,
                                  lon_values) -> tuple[float, float, bool, float, float]:
    """経度範囲 [lo, hi] をデータの経度規約に正規化し、wrap 要否を返す。

    - データが 0-360 系 (data_min >= 0) なら [0, 360) に modulo 正規化
    - データが -180-180 系 (data_min < 0) なら [-180, 180) に modulo 正規化
    - 正規化後 lo > hi なら日付変更線をまたぐ wrap-around
    - 元の入力幅が 360°以上のときは全周 (data_min, data_max, wrap=False)

    returns (lo_canon, hi_canon, wrap, data_min, data_max)
    """
    data_min = float(np.min(lon_values))
    data_max = float(np.max(lon_values))
    if (hi - lo) >= 360.0:
        return data_min, data_max, False, data_min, data_max
    convention_lo = 0.0 if data_min >= 0.0 else -180.0

    def norm(v):
        return ((v - convention_lo) % 360.0) + convention_lo

    lo_n = norm(lo)
    hi_n = norm(hi)
    return lo_n, hi_n, lo_n > hi_n, data_min, data_max


def has_redundant_lon_endpoint(lon_values) -> bool:
    """データの先頭と末尾が同じ物理経度を指しているか (例: 0 と 360 が両方ある)。

    気象データでは通常 0 から step 未満を引いた値まで (例: 0..357.5) で
    両端点の冗長を避けるが、ファイルによっては両端を冗長に含むものもある。
    wrap-around 平均でこの冗長を残すとダブルカウントになるため検出して除く。
    """
    if len(lon_values) < 2:
        return False
    s = sorted(float(v) for v in lon_values)
    return abs((s[-1] - s[0]) - 360.0) < 1e-6


def check_lon_range_covered(lo: float, hi: float, lon_values) -> None:
    """ユーザー指定の経度範囲がデータの経度範囲に完全に含まれているか確認する。

    含まれていない場合は ValueError を投げて描画を停止する。
    具体的には:
      - 入力幅が 360°以上なら全周扱いで OK
      - wrap-around (lo_n > hi_n) のとき、データが全球を覆っていないと不可
        (例: データ経度が 20-360 のとき [-30, 30] は wrap が必要だが、0-20° の
        範囲がデータに無いため不可)
      - 非 wrap のとき、[lo_n, hi_n] が [data_min, data_max] に収まる必要あり
    apply_averages (render) と _section_data_lines (scriptgen) の両方から呼ぶ。
    """
    data_min = float(np.min(lon_values))
    data_max = float(np.max(lon_values))
    if (hi - lo) >= 360.0:
        return
    lo_n, hi_n, wrap, _, _ = canon_lon_range_for_averaging(lo, hi, lon_values)
    if wrap:
        if len(lon_values) >= 2:
            sorted_lons = sorted(float(v) for v in lon_values)
            step = sorted_lons[1] - sorted_lons[0]
        else:
            step = 0.0
        is_global = (data_max - data_min + step) >= 360.0 - 1e-6
        if not is_global:
            raise RenderError("lon_range_dateline", lo=lo, hi=hi,
                              data_min=data_min, data_max=data_max)
    else:
        if lo_n < data_min - 1e-6 or hi_n > data_max + 1e-6:
            raise RenderError("lon_range_outside", lo=lo, hi=hi,
                              lo_n=lo_n, hi_n=hi_n,
                              data_min=data_min, data_max=data_max
            )


def effective_averages(dims, averages: dict | None, exclude=()) -> dict:
    """render が実際に適用する範囲平均だけを残す (変数に無い dim と exclude の dim を除く)。

    描画側 (apply_averages / select_line_data / _bundle_layer_data)、scriptgen
    (_emit_averages_lines)、図の直下の通知 (notes.collect_notes) が同じ除外規則を
    共有するための単一実装。exclude は 1 次元プロットの x 軸や束の次元など、
    平均の対象にしない dim。順序は averages のまま。
    """
    if not averages:
        return {}
    return {d: cfg for d, cfg in averages.items()
            if d in dims and d not in exclude}


def averaging_slice(da: xr.DataArray, dim: str, lo, hi,
                    ds: xr.Dataset) -> xr.DataArray:
    """範囲平均の対象範囲 [lo, hi] を dim について切り出す (平均はしない)。

    経度 dim (detect_coord_roles の lon) は canon_lon_range_for_averaging で
    正規化し、wrap-around (lo > hi) なら [lo, data_max] ∪ [data_min, hi] を
    xr.concat で繋ぐ。冗長端点 (0 と 360 が両方ある等) は繋ぎ目で 1 つ落とす。
    apply_averages (平均本体) と notes.collect_notes (含まれた格子数の集計) が
    同じ切り出しを共有する。da が ds[dim] (座標そのもの) でもよい。
    """
    lon_name = detect_coord_roles(ds).get("lon")
    if dim == lon_name:
        check_lon_range_covered(lo, hi, ds[dim].values)
        lo_n, hi_n, wrap, dmin, dmax = canon_lon_range_for_averaging(
            lo, hi, ds[dim].values)
        if wrap:
            p1 = da.sel({dim: coord_slice(ds[dim], lo_n, dmax)})
            p2 = da.sel({dim: coord_slice(ds[dim], dmin, hi_n)})
            # データに冗長端点 (0 と 360 が両方ある等) がある場合、
            # p1 の末尾 (= dmax) と p2 の先頭 (= dmin) は同じ物理点なので
            # 一方を落としてダブルカウントを防ぐ
            if has_redundant_lon_endpoint(ds[dim].values):
                p1 = p1.isel({dim: slice(None, -1)})
            return xr.concat([p1, p2], dim=dim)
        return da.sel({dim: coord_slice(ds[dim], lo_n, hi_n)})
    return da.sel({dim: coord_slice(ds[dim], lo, hi)})


def apply_averages(da: xr.DataArray, averages: dict | None,
                   ds: xr.Dataset, record: list | None = None) -> xr.DataArray:
    """各 dim の範囲を切り出し、平均演算で潰す。

    averages = {dim: {"op": "mean"|"weighted_mean", "range": [lo, hi]}}。
    "weighted_mean" は cos(緯度) 重み付き平均で、当該 dim の値を緯度 (度) として扱う。
    範囲の切り出し (経度の正規化・wrap-around・冗長端点) は averaging_slice、
    対象 dim の絞り込み (変数に無い dim は読み飛ばす) は effective_averages。
    render と scriptgen の両方から呼ぶ単一実装。

    record (list) を渡すと、平均に使ったブロック (全対象 dim を切り出した平均前の
    配列) の要素数と欠損数 {"dims", "n_total", "n_nan"} を 1 件 append する
    (図の直下の通知「平均範囲内の欠損」用。layer_averaging_stats 参照)。
    平均の計算順序 (dim ごとに切り出して潰す) は record の有無で変えない。
    """
    eff = effective_averages(da.dims, averages)
    if record is not None and eff:
        block = da
        for dim, op_cfg in eff.items():
            block = averaging_slice(block, dim, op_cfg["range"][0], op_cfg["range"][1], ds)
        record.append({"dims": list(eff), "n_total": int(block.size),
                       "n_nan": int(block.isnull().sum())})
    for dim, op_cfg in eff.items():
        lo, hi = op_cfg["range"]
        da_sliced = averaging_slice(da, dim, lo, hi, ds)
        op = op_cfg.get("op", "mean")
        if op == "weighted_mean":
            weights = np.cos(np.deg2rad(da_sliced[dim]))
            da = da_sliced.weighted(weights).mean(dim=dim)
        else:
            da = da_sliced.mean(dim=dim)
    return da


def select_section_data(ds: xr.Dataset, variable: str, selection: dict | None,
                        ranges: dict | None, x_dim: str, y_dim: str,
                        averages: dict | None = None,
                        record: list | None = None,
                        path: dict | None = None) -> xr.DataArray:
    """断面用に固定次元・軸範囲を切り出し、(y, x) の次元順に揃えて返す。

    averages があれば selection の後・panel ranges の前に適用する
    (平均で潰された dim は ranges の対象外)。record は apply_averages に渡す
    (平均ブロックの要素数・欠損数の記録)。path (panel["section_path"]) があれば
    平均の後・ranges の前に経路上へ双一次内挿し、水平の 2 次元を x_dim
    (= SECTION_PATH_DIM) に置き換える (scriptgen._section_data_lines と 1 対 1)。
    """
    da = ds[variable]
    sel = {dim: val for dim, val in (selection or {}).items() if dim in da.dims}
    if sel:
        da = da.sel(sel)
    da = apply_averages(da, averages, ds, record=record)
    if path:
        da = sample_section_path(da, ds, section_path_spec(path, ds), x_dim)
    for dim, bounds in (ranges or {}).items():
        if dim in da.dims:
            da = da.sel({dim: coord_slice(ds[dim], bounds[0], bounds[1])})
    return da.transpose(y_dim, x_dim)


def select_line_data(ds: xr.Dataset, variable: str, x_dim: str,
                      selection: dict | None, ranges: dict | None,
                      averages: dict | None = None,
                      record: list | None = None) -> xr.DataArray:
    """1次元プロット用に、x_dim 以外を selection で固定し ranges で範囲制限する。

    averages があれば selection の後・ranges の前に適用する
    (x_dim の averages は意味がないので暗黙に無視)。
    返り値は 1次元 DataArray (x_dim の座標を保持)。
    """
    da = ds[variable]
    sel = {dim: val for dim, val in (selection or {}).items()
            if dim in da.dims and dim != x_dim}
    if sel:
        da = da.sel(sel)
    da = apply_averages(da, effective_averages(da.dims, averages, exclude=(x_dim,)), ds,
                        record=record)
    for dim, bounds in (ranges or {}).items():
        if dim in da.dims:
            da = da.sel({dim: coord_slice(ds[dim], bounds[0], bounds[1])})
    return da


def needs_cyclic_lon(da: xr.DataArray, lon_name: str) -> bool:
    """経度が全周を覆っているか (cyclic point の追加が必要か) を判定する。

    2 次元の経度座標 (curvilinear 格子) は常に False (周期点の概念が無い)。
    """
    lon = da[lon_name].values
    if lon.ndim != 1 or lon.size < 3 or lon[1] <= lon[0]:
        return False
    dlon = lon[1] - lon[0]
    return abs((lon[-1] - lon[0]) + dlon - 360.0) < 1e-3


def add_cyclic_lon(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    """経度の先頭の列を +360° の位置に複製し、0/360°境界の継ぎ目を閉じる。

    scriptgen が生成する xr.concat の行と同じ操作であること。
    """
    return xr.concat(
        [da, da.isel({lon_name: 0}).assign_coords({lon_name: da[lon_name][0] + 360.0})],
        dim=lon_name)


def cyclic_tile_lon(da: xr.DataArray, x_dim: str, x_lo: float, x_hi: float,
                     period: float = 360.0, max_tiles: int = 10) -> xr.DataArray:
    """x_dim の値が [x_lo, x_hi] を覆うよう、period 倍ずつシフトしたコピーを連結する。

    元データは 1 周期分 ([0, 360) 等) を想定。返り値の x_dim は連続な (重複なし)
    座標になる。scriptgen の line_layer もこれと同じ操作を出力する。
    """
    x = da[x_dim].values
    x_min = float(x.min())
    x_max = float(x.max())
    n_right = max(0, int(np.ceil((x_hi - x_max) / period)))
    n_left = max(0, int(np.ceil((x_min - x_lo) / period)))
    n_right = min(n_right, max_tiles)
    n_left = min(n_left, max_tiles)
    if n_right == 0 and n_left == 0:
        return da
    parts = []
    for k in range(-n_left, n_right + 1):
        if k == 0:
            parts.append(da)
        else:
            parts.append(da.assign_coords({x_dim: x + k * period}))
    return xr.concat(parts, dim=x_dim)


# --- 断面の経路 (大円・格子番号の逆算・双一次内挿) ---
# 経路に沿った鉛直断面 (等緯度線・等経度線・大円) の標本化。2 次元座標格子は投影法の
# 推定に頼らず経緯度だけから格子番号を求める (docs/section_extension_plan.md)。
# NOTE: 以下 4 関数の本体は scriptgen が inspect.getsource でそのまま再現スクリプトに
# 埋め込む (render ⇄ scriptgen を文字どおり同一コードにするため)。そのため
# docstring・コメントは英語、参照してよい名前は np / xr と引数だけ
# (生成スクリプトの import と一致させること。tests/test_section_path.py が検査する)。

def great_circle_points(start, end, npoints, radius_km=6371.0):
    """Sample equally spaced points on the great circle from start to end.

    ``start`` and ``end`` are (lon, lat) in degrees. Returns (lon, lat, dist_km)
    arrays of length ``npoints``. Longitudes are continuous along the path
    (no jump at the dateline, so they may leave [-180, 180]) and start at
    ``start[0]``; ``dist_km`` is the distance from ``start``.
    """
    def unit(lon, lat):
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                         np.sin(lat)])

    p1 = unit(float(start[0]), float(start[1]))
    p2 = unit(float(end[0]), float(end[1]))
    # central angle from atan2(|p1 x p2|, p1 . p2): accurate for any angle
    omega = np.arctan2(np.linalg.norm(np.cross(p1, p2)), np.dot(p1, p2))
    t = np.linspace(0.0, 1.0, int(npoints))
    if omega < 1e-12:
        pts = np.repeat(p1[None, :], t.size, axis=0)
    elif np.pi - omega < 1e-9:
        raise ValueError("start and end are antipodal: the great circle is not unique")
    else:
        pts = (np.sin((1.0 - t) * omega)[:, None] * p1
               + np.sin(t * omega)[:, None] * p2) / np.sin(omega)
    lat = np.rad2deg(np.arcsin(np.clip(pts[:, 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0]))
    # continuous longitudes: offsets from start[0] wrapped to [-180, 180), then unwrapped
    offset = (lon - float(start[0]) + 180.0) % 360.0 - 180.0
    lon = float(start[0]) + np.rad2deg(np.unwrap(np.deg2rad(offset)))
    return lon, lat, t * omega * radius_km


def grid_fractional_indices(lon2d, lat2d, lon, lat, coarse=100, max_iter=8):
    """Fractional (row, column) indices of points on a grid with 2-D lon / lat.

    ``lon2d`` / ``lat2d`` are the grid longitudes / latitudes (ny, nx) in
    degrees (any longitude convention, NaN allowed); ``lon`` / ``lat`` are the
    points. The nearest grid point is searched on the unit sphere (a coarse
    search on a subsampled grid, then a local search), and the position in the
    grid cell is refined by Newton's method on the bilinear map of the cell
    corners projected onto the plane tangent at the point (gnomonic
    projection). No map projection of the grid is assumed. Returns (fj, fi);
    points outside the grid, or in a cell with a NaN corner, are NaN.
    """
    def unit(lon, lat):
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                         np.sin(lat)], axis=-1)

    lon2d = np.asarray(lon2d, dtype=float)
    lat2d = np.asarray(lat2d, dtype=float)
    ny, nx = lon2d.shape
    if ny < 2 or nx < 2:
        raise ValueError("the grid needs at least 2 x 2 points")
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    g = unit(lon2d, lat2d)                       # (ny, nx, 3), NaN where coords are NaN
    ok = np.all(np.isfinite(g), axis=-1)
    g0 = np.where(ok[..., None], g, 0.0)
    p = unit(lon, lat)                           # (n, 3)
    n = p.shape[0]
    fj = np.full(n, np.nan)
    fi = np.full(n, np.nan)

    # 1) coarse nearest search on a subsampled grid (chunks of points)
    step = max(1, int(np.ceil(max(ny, nx) / coarse)))
    cj = np.arange(0, ny, step)
    ci = np.arange(0, nx, step)
    gc = g0[np.ix_(cj, ci)].reshape(-1, 3)
    okc = ok[np.ix_(cj, ci)].ravel()
    start = np.full(n, -1)
    for k0 in range(0, n, 512):
        d = gc @ np.nan_to_num(p[k0:k0 + 512]).T      # (m, chunk)
        d[~okc, :] = -np.inf
        start[k0:k0 + 512] = np.argmax(d, axis=0)
    # 2) local search: move a window until its centre is the nearest point in it
    for k in range(n):
        if not (np.all(np.isfinite(p[k])) and okc[start[k]]):
            continue
        j, i = cj[start[k] // ci.size], ci[start[k] % ci.size]
        for _ in range(ny + nx):
            j0, j1 = max(j - step, 0), min(j + step + 1, ny)
            i0, i1 = max(i - step, 0), min(i + step + 1, nx)
            d = g0[j0:j1, i0:i1] @ p[k]
            d[~ok[j0:j1, i0:i1]] = -np.inf
            jj, ii = np.unravel_index(int(np.argmax(d)), d.shape)
            if (j0 + jj, i0 + ii) == (j, i):
                break
            j, i = j0 + jj, i0 + ii
        fj[k], fi[k] = j, i

    # 3) Newton's method on the bilinear map in the tangent (gnomonic) plane
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     axis=-1)

    def plane(v):
        w = v / np.sum(v * p, axis=-1, keepdims=True)
        return np.sum(w * east, axis=-1), np.sum(w * north, axis=-1)

    def cell(fj, fi):
        jc = np.clip(np.floor(np.nan_to_num(fj)), 0, ny - 2).astype(int)
        ic = np.clip(np.floor(np.nan_to_num(fi)), 0, nx - 2).astype(int)
        a, b = fj - jc, fi - ic
        x00, y00 = plane(g[jc, ic])
        x01, y01 = plane(g[jc, ic + 1])
        x10, y10 = plane(g[jc + 1, ic])
        x11, y11 = plane(g[jc + 1, ic + 1])
        x = x00 * (1 - a) * (1 - b) + x01 * (1 - a) * b + x10 * a * (1 - b) + x11 * a * b
        y = y00 * (1 - a) * (1 - b) + y01 * (1 - a) * b + y10 * a * (1 - b) + y11 * a * b
        xa = (x10 - x00) * (1 - b) + (x11 - x01) * b
        xb = (x01 - x00) * (1 - a) + (x11 - x10) * a
        ya = (y10 - y00) * (1 - b) + (y11 - y01) * b
        yb = (y01 - y00) * (1 - a) + (y11 - y10) * a
        return x, y, xa, xb, ya, yb

    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            x, y, xa, xb, ya, yb = cell(fj, fi)
            det = xa * yb - xb * ya
            da = np.clip((xb * y - yb * x) / det, -1.0, 1.0)
            db = np.clip((ya * x - xa * y) / det, -1.0, 1.0)
            fj, fi = fj + da, fi + db
            if not np.any(np.abs(np.nan_to_num(da)) + np.abs(np.nan_to_num(db)) > 1e-10):
                break
        # not converged (residual larger than 1e-6 of the cell size) -> NaN
        x, y, xa, xb, ya, yb = cell(fj, fi)
        size = np.hypot(xa, ya) + np.hypot(xb, yb)
        bad = ~(np.hypot(x, y) <= 1e-6 * size)
    tol = 1e-6
    bad |= (fj < -tol) | (fj > ny - 1 + tol) | (fi < -tol) | (fi > nx - 1 + tol)
    fj = np.where(bad, np.nan, np.clip(fj, 0, ny - 1))
    fi = np.where(bad, np.nan, np.clip(fi, 0, nx - 1))
    return fj, fi


def grid_fractional_indices_1d(lon1d, lat1d, lon, lat):
    """Fractional (row, column) indices of points on a grid with 1-D lat / lon.

    Point longitudes are wrapped into the grid's convention and both axes may
    be ascending or descending. On a global grid whose longitudes wrap around
    (last + spacing = first + 360), a point between the last and the first
    column gets fi outside [0, nx - 1] (nx - 1 .. nx, or -1 .. 0 when the
    longitudes descend); pass ``wrap_x=True`` to sample_bilinear then.
    Returns (fj, fi); points outside the grid are NaN.
    """
    lon1d = np.asarray(lon1d, dtype=float)
    lat1d = np.asarray(lat1d, dtype=float)
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))

    def frac(coord, values, wrap=False):
        order = np.argsort(coord)
        c, idx = coord[order], order.astype(float)
        if wrap:
            # the column after the last one is the first one again, 360 degrees on
            c = np.append(c, c[0] + 360.0)
            idx = np.append(idx, idx[-1] + (1.0 if idx[-1] > idx[0] else -1.0))
        return np.interp(values, c, idx, left=np.nan, right=np.nan)

    nx = lon1d.size
    wrap = bool(nx > 2 and abs(abs(lon1d[-1] - lon1d[0])
                               + abs(lon1d[1] - lon1d[0]) - 360.0) < 1e-3)
    base = np.min(lon1d)
    lonw = (lon - base) % 360.0 + base
    return frac(lat1d, lat), frac(lon1d, lonw, wrap)


def sample_bilinear(da, ydim, xdim, fj, fi, dim, wrap_x=False):
    """Bilinearly interpolate ``da`` at fractional grid indices (fj, fi).

    The two horizontal dimensions ``ydim`` / ``xdim`` are replaced by the new
    dimension ``dim``; coordinates that depend on them are dropped (assign the
    path coordinates afterwards). A point is NaN if fj / fi is NaN or if a
    corner with a positive weight is NaN (no value is made up from missing
    data). With ``wrap_x`` the column after the last one is column 0.
    """
    ny, nx = da.sizes[ydim], da.sizes[xdim]
    attrs = dict(da.attrs)
    da = da.drop_vars([c for c in da.coords if set(da[c].dims) & {ydim, xdim}])
    fj = np.asarray(fj, dtype=float)
    fi = np.asarray(fi, dtype=float)
    valid = np.isfinite(fj) & np.isfinite(fi)
    fj0 = np.where(valid, fj, 0.0)
    fi0 = np.where(valid, fi, 0.0)
    j0 = np.clip(np.floor(fj0), 0, ny - 2).astype(int)
    if wrap_x:
        i0 = np.floor(fi0).astype(int)
        b = fi0 - i0
        i1 = (i0 + 1) % nx
        i0 = i0 % nx
    else:
        i0 = np.clip(np.floor(fi0), 0, nx - 2).astype(int)
        b = fi0 - i0
        i1 = i0 + 1
    a = fj0 - j0
    out = 0.0
    for jj, ii, w in ((j0, i0, (1 - a) * (1 - b)), (j0, i1, (1 - a) * b),
                      (j0 + 1, i0, a * (1 - b)), (j0 + 1, i1, a * b)):
        v = da.isel({ydim: xr.DataArray(jj, dims=dim), xdim: xr.DataArray(ii, dims=dim)})
        w = xr.DataArray(w, dims=dim)
        # zero-weight corners do not contribute (their NaN must not spread);
        # DataArray.where keeps the dimension order of v
        out = out + (v * w).where(w > 0, 0.0)
    out = out.where(xr.DataArray(valid, dims=dim))
    out.attrs = attrs
    return out


def lonlat_tick_label(value, x, lon, lat):
    """Three-line tick label 'value / lon / lat' for the axis value ``value``.

    ``x`` are the axis values along the section (any order) with the
    longitudes / latitudes ``lon`` / ``lat`` of the same points; the position
    is interpolated linearly. Values outside the section get an empty label.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    if value < x[order[0]] - 1e-9 or value > x[order[-1]] + 1e-9:
        return ""
    # unwrap longitudes so that a 179 -> -179 step does not interpolate to 0
    lon = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(lon, dtype=float)[order])))
    lo = float(np.interp(value, x[order], lon))
    la = float(np.interp(value, x[order], np.asarray(lat, dtype=float)[order]))
    lo = (lo + 180.0) % 360.0 - 180.0
    ew = "E" if lo > 0 else ("W" if lo < 0 else "")
    ns = "N" if la > 0 else ("S" if la < 0 else "")
    return f"{value:g}\n{abs(lo):.1f}\u00b0{ew}\n{abs(la):.1f}\u00b0{ns}"


def ground_pressure_from_height(z, lev, zs):
    """Pressure of the ground in each column of a vertical section.

    ``z`` (nlev, n) are the heights of the pressure levels ``lev`` (nlev, any
    order, any pressure unit) and ``zs`` (n) the terrain heights, all in the
    same length unit. Between the two levels that bracket ``zs`` the pressure
    is interpolated linearly in log(p). Columns whose ground lies below the
    lowest level (nothing to cover) and columns with NaN get NaN; columns whose
    ground lies above the highest level get the highest level.
    """
    z = np.asarray(z, dtype=float)
    lev = np.asarray(lev, dtype=float)
    zs = np.asarray(zs, dtype=float)
    order = np.argsort(-lev)                 # from the ground (high pressure) upwards
    z, lev = z[order], lev[order]
    out = np.full(zs.shape, np.nan)
    for k in range(lev.size - 1):
        z0, z1 = z[k], z[k + 1]
        hit = np.isnan(out) & (zs > z0) & (zs <= z1)
        frac = (zs - z0) / (z1 - z0)
        out = np.where(hit, np.exp(np.log(lev[k]) + frac * (np.log(lev[k + 1]) - np.log(lev[k]))),
                       out)
    above_top = np.isnan(out) & (zs > z[-1])
    return np.where(above_top, lev[-1], out)


def draw_section_terrain(ax, x, ground, pressure_like, color, zorder=2.5):
    """Fill the area below the ground line of a vertical section, above the layers.

    ``ground`` is the ground position in the units of the vertical axis (NaN
    where unknown). For a pressure axis (``pressure_like``) the ground side is
    the large values, otherwise the small values. The polygon is clipped to the
    current axis limits so that it does not change the automatic axis range.
    """
    ylim = ax.get_ylim()
    lo, hi = min(ylim), max(ylim)
    g = np.clip(np.asarray(ground, dtype=float), lo, hi)
    bottom = hi if pressure_like else lo
    ax.fill_between(np.asarray(x, dtype=float), g, bottom, color=color, linewidth=0,
                    zorder=zorder)
    ax.set_ylim(ylim)


# 地形マスクの多角形の zorder: 等値線・流線 (2) の上、文字 (3)・等値線ラベル (4) の下
TERRAIN_ZORDER = 2.5
# 単位の換算表 (小文字)。値 = 基準単位 (Pa / m) への倍率
_PRESSURE_UNIT_FACTORS = {"pa": 1.0, "hpa": 100.0, "mb": 100.0, "mbar": 100.0, "millibar": 100.0,
                          "millibars": 100.0, "hectopascal": 100.0, "hectopascals": 100.0,
                          "kpa": 1000.0}
_LENGTH_UNIT_FACTORS = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
                        "km": 1000.0, "gpm": 1.0, "geopotential_meter": 1.0,
                        "geopotential meters": 1.0}
_GEOPOTENTIAL_UNITS = ("m2 s-2", "m2/s2", "m**2 s**-2", "m^2 s^-2", "m^2/s^2", "m2 s^-2")
_GRAVITY = 9.80665


def vertical_axis_kind(units) -> str | None:
    """鉛直座標の units から "pressure" / "height" / None を返す。"""
    u = str(units or "").strip().lower()
    if u in _PRESSURE_UNIT_FACTORS:
        return "pressure"
    if u in _LENGTH_UNIT_FACTORS:
        return "height"
    return None


def unit_factor(units_from, units_to, kind: str) -> float | None:
    """units_from の値を units_to にする倍率 (kind = "pressure" / "height")。不明なら None。
    高度ではジオポテンシャル (m2 s-2) も受け、g で割って m にする。"""
    table = _PRESSURE_UNIT_FACTORS if kind == "pressure" else _LENGTH_UNIT_FACTORS
    a = str(units_from or "").strip().lower()
    b = str(units_to or "").strip().lower()
    if b not in table:
        return None
    if kind == "height" and a in _GEOPOTENTIAL_UNITS:
        return 1.0 / _GRAVITY / table[b]
    if a not in table:
        return None
    return table[a] / table[b]


# 経路断面の合成次元の名前 (panel["x_dim"] に入れる)
SECTION_PATH_DIM = "path"
SECTION_PATH_KINDS = ("parallel", "meridian", "great_circle")
_EARTH_RADIUS_KM = 6371.0
_SECTION_PATH_MAX_POINTS = 5000


def grid_spacing_km(ds: xr.Dataset, roles: dict, lat_ref: float | None = None) -> float:
    """水平格子の代表的な格子間隔 [km] (経路断面の点数の自動決定用)。

    2 次元座標格子は隣り合う格子点の球面距離 (行方向・列方向、間引いて計算) の中央値。
    1 次元格子は緯度間隔と経度間隔 (lat_ref、無ければ格子の緯度の中央値で cos を掛ける)
    の小さい方。
    """
    lon_name, lat_name = roles["lon"], roles["lat"]
    lon = ds[lon_name].values.astype(float)
    lat = ds[lat_name].values.astype(float)

    def dist(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
        h = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    if lon.ndim == 2:
        step = max(1, int(np.ceil(max(lon.shape) / 200)))
        lo, la = lon[::step, ::step], lat[::step, ::step]
        d = np.concatenate([dist(lo[:, :-1], la[:, :-1], lo[:, 1:], la[:, 1:]).ravel(),
                            dist(lo[:-1, :], la[:-1, :], lo[1:, :], la[1:, :]).ravel()])
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) / step if d.size else 1.0
    deg_km = np.deg2rad(1.0) * _EARTH_RADIUS_KM
    ref = float(np.median(lat)) if lat_ref is None else float(lat_ref)
    cands = []
    if lat.size > 1:
        cands.append(float(np.median(np.abs(np.diff(lat)))) * deg_km)
    if lon.size > 1:
        cands.append(float(np.median(np.abs(np.diff(lon)))) * deg_km
                     * max(np.cos(np.deg2rad(ref)), 0.05))
    return min(cands) if cands else 1.0


def section_path_length_km(path: dict) -> float:
    """経路の長さ [km] (等緯度線は緯線に沿った距離、大円は大円距離)。"""
    kind = path.get("kind")
    if kind == "parallel":
        lo, hi = path["lon_range"]
        return abs(np.deg2rad(hi - lo)) * _EARTH_RADIUS_KM * abs(np.cos(np.deg2rad(path["lat"])))
    if kind == "meridian":
        lo, hi = path["lat_range"]
        return abs(np.deg2rad(hi - lo)) * _EARTH_RADIUS_KM
    if kind == "great_circle":
        try:
            return float(great_circle_points(path["start"], path["end"], 2)[2][-1])
        except ValueError:
            raise RenderError("section_path_antipodal") from None
    raise RenderError("section_path_unknown_kind", kind=kind)


def section_path_spec(path: dict, ds: xr.Dataset) -> dict:
    """panel["section_path"] を、点の数まで確定した dict にして返す。

    npoints が None なら「経路長 ÷ 格子間隔 を切り上げ + 1」(2〜5000 点) で決める。
    scriptgen も同じ関数で点数を決め、リテラルで生成スクリプトに出す (1 対 1)。
    """
    kind = path.get("kind")
    if kind not in SECTION_PATH_KINDS:
        raise RenderError("section_path_unknown_kind", kind=kind)
    spec = dict(path)
    n = spec.get("npoints")
    if n is None:
        roles = detect_coord_roles(ds)
        if not (roles.get("lon") and roles.get("lat")):
            raise RenderError("section_path_no_lonlat")
        lat_ref = None
        if kind == "parallel":
            lat_ref = float(spec["lat"])
        elif kind == "meridian":
            lat_ref = float(np.mean(spec["lat_range"]))
        else:
            lat_ref = float(np.mean([spec["start"][1], spec["end"][1]]))
        spacing = grid_spacing_km(ds, roles, lat_ref)
        n = int(np.ceil(section_path_length_km(spec) / max(spacing, 1e-6))) + 1
    spec["npoints"] = int(min(max(n, 2), _SECTION_PATH_MAX_POINTS))
    return spec


def section_path_points(spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """確定した経路 (section_path_spec の返り値) の点 (lon, lat, 横軸の値) を返す。

    横軸の値は 等緯度線 = 経度、等経度線 = 緯度、大円 = 始点からの距離 [km]。
    scriptgen は同じ値を np.linspace / np.full / great_circle_points の行で出す。
    """
    kind, n = spec["kind"], int(spec["npoints"])
    if kind == "parallel":
        lon = np.linspace(float(spec["lon_range"][0]), float(spec["lon_range"][1]), n)
        return lon, np.full(n, float(spec["lat"])), lon
    if kind == "meridian":
        lat = np.linspace(float(spec["lat_range"][0]), float(spec["lat_range"][1]), n)
        return np.full(n, float(spec["lon"])), lat, lat
    if kind == "great_circle":
        try:
            return great_circle_points(tuple(spec["start"]), tuple(spec["end"]), n)
        except ValueError:
            raise RenderError("section_path_antipodal") from None
    raise RenderError("section_path_unknown_kind", kind=kind)


def section_path_grid(ds: xr.Dataset) -> tuple[str, str, bool]:
    """経路断面で内挿する水平格子の (ydim, xdim, 2 次元座標格子か) を返す。

    1 次元格子は全球格子の継ぎ目をまたげるよう sample_bilinear に wrap_x=True を渡す
    (2 次元座標格子は False)。経緯度が認識できなければ RenderError。
    """
    roles = detect_coord_roles(ds)
    if not (roles.get("lon") and roles.get("lat")):
        raise RenderError("section_path_no_lonlat")
    ydim, xdim = horizontal_dims(ds, roles)
    return ydim, xdim, is_curvilinear(ds, roles)


def section_path_indices(ds: xr.Dataset, lon: np.ndarray, lat: np.ndarray):
    """経路の点の小数格子番号 (fj, fi) と、sample_bilinear に渡す (ydim, xdim, wrap_x)。

    2 次元座標格子は grid_fractional_indices、1 次元格子は grid_fractional_indices_1d。
    scriptgen._section_path_vars と 1 対 1。
    """
    roles = detect_coord_roles(ds)
    ydim, xdim, curvi = section_path_grid(ds)
    if curvi:
        fj, fi = grid_fractional_indices(
            ds[roles["lon"]].transpose(ydim, xdim).values,
            ds[roles["lat"]].transpose(ydim, xdim).values, lon, lat)
    else:
        fj, fi = grid_fractional_indices_1d(ds[roles["lon"]].values, ds[roles["lat"]].values,
                                            lon, lat)
    return fj, fi, ydim, xdim, not curvi


def sample_section_path(da: xr.DataArray, ds: xr.Dataset, spec: dict,
                        dim: str = SECTION_PATH_DIM) -> xr.DataArray:
    """変数 da を経路上の点へ双一次内挿し、水平の 2 次元を経路の次元 dim に置き換える。

    dim の座標 = 横軸の値 (section_path_points)、補助座標 {dim}_lon / {dim}_lat =
    各点の経緯度 (目盛の経緯度併記・地図への経路表示用)。
    """
    if dim in ds.dims:
        raise RenderError("section_path_dim_conflict", dim=dim)
    lon, lat, x = section_path_points(spec)
    fj, fi, ydim, xdim, wrap = section_path_indices(ds, lon, lat)
    if not np.any(np.isfinite(fj) & np.isfinite(fi)):
        b = lonlat_bounds(ds) or {}
        raise RenderError("section_path_outside_grid", kind=spec["kind"],
                          lon_min=b.get("lon_min", float("nan")),
                          lon_max=b.get("lon_max", float("nan")),
                          lat_min=b.get("lat_min", float("nan")),
                          lat_max=b.get("lat_max", float("nan")))
    out = sample_bilinear(da, ydim, xdim, fj, fi, dim, wrap_x=wrap)
    return out.assign_coords({dim: x, f"{dim}_lon": (dim, lon), f"{dim}_lat": (dim, lat)})


def section_x_lonlat(panel: dict, datasets: dict):
    """断面の横軸の値と、その位置の経緯度 (x, lon, lat) を返す (目盛の経緯度併記用)。

    経路断面は経路の点そのもの。格子線断面 (2 次元座標格子で x_dim が格子の dim) は、
    先頭レイヤーの固定した行・列に沿った格子点の経緯度。求められない断面
    (1 次元格子の従来の断面など) は None。scriptgen は同じ値を出す行を生成する。
    """
    layers = panel.get("layers") or []
    if not layers:
        return None
    ds = datasets[layers[0]["dataset_id"]]
    spec = panel.get("section_path")
    if spec:
        lon, lat, x = section_path_points(section_path_spec(spec, ds))
        return x, lon, lat
    roles = detect_coord_roles(ds)
    if not is_curvilinear(ds, roles):
        return None
    hdims = horizontal_dims(ds, roles)
    x_dim = panel.get("x_dim")
    if x_dim not in hdims:
        return None
    other = hdims[1] if x_dim == hdims[0] else hdims[0]
    selection = {**(panel.get("selection") or {}), **(layers[0].get("selection") or {})}
    if other not in selection:
        return None
    # 座標のない次元 (bare dim) の .sel は位置 (index) の固定として働く (データの
    # 切り出し select_section_data と同じ書き方)。x_dim が bare なら格子番号 = 0 始まりの index
    lon = ds[roles["lon"]].sel({other: selection[other]}).transpose(x_dim).values
    lat = ds[roles["lat"]].sel({other: selection[other]}).transpose(x_dim).values
    return ds[x_dim].values.astype(float), lon, lat


def section_panel_lonlat(panel: dict, datasets: dict):
    """鉛直断面パネルの経路の経緯度 (lon, lat) — 地図に重ねる用。鉛直断面でなければ None。

    経路断面 = 経路の標本点そのもの。2 次元座標格子の格子線断面 = 先頭レイヤーで固定した
    行・列に沿った格子点 (横軸の範囲内)。1 次元格子の従来の断面 = 先頭レイヤーで固定した
    緯度 (経度) に沿う格子点 (横軸の範囲内)。固定値が無い (平均など) なら None。
    scriptgen._section_overlay_lines と 1 対 1。
    """
    layers = panel.get("layers") or []
    if panel.get("plot_type") != "section_2d" or not layers:
        return None
    ds = datasets.get(layers[0].get("dataset_id"))
    if ds is None:
        return None
    spec = panel.get("section_path")
    if spec:
        lon, lat, _x = section_path_points(section_path_spec(spec, ds))
        return lon, lat
    roles = detect_coord_roles(ds)
    x_dim, y_dim = panel.get("x_dim"), panel.get("y_dim")
    if y_dim != roles.get("vertical") or not (roles.get("lon") and roles.get("lat")):
        return None
    selection = _terrain_selection(panel)
    ranges = panel.get("ranges") or {}
    if is_curvilinear(ds, roles):
        hd = horizontal_dims(ds, roles)
        if x_dim not in hd:
            return None
        other = hd[1] if x_dim == hd[0] else hd[0]
        if other not in selection:
            return None
        lon = ds[roles["lon"]].sel({other: selection[other]}).transpose(x_dim)
        lat = ds[roles["lat"]].sel({other: selection[other]}).transpose(x_dim)
        if x_dim in ranges and x_dim in ds.coords:
            sl = coord_slice(ds[x_dim], ranges[x_dim][0], ranges[x_dim][1])
            lon, lat = lon.sel({x_dim: sl}), lat.sel({x_dim: sl})
        return lon.values.astype(float), lat.values.astype(float)
    if x_dim == roles["lon"]:
        fixed = roles["lat"]
    elif x_dim == roles["lat"]:
        fixed = roles["lon"]
    else:
        return None
    if fixed not in selection:
        return None
    x = ds[x_dim]
    if x_dim in ranges:
        x = x.sel({x_dim: coord_slice(ds[x_dim], ranges[x_dim][0], ranges[x_dim][1])})
    x = x.values.astype(float)
    const = np.full(x.size, float(selection[fixed]))
    return (x, const) if x_dim == roles["lon"] else (const, x)


def section_overlay_items(panel: dict, figure_config: dict | None, datasets: dict):
    """地図パネルの map.section_paths を解決し、[(設定, lon, lat)] を返す。

    参照先は figure_config["panels"] の panel_id で探す。無ければ / 鉛直断面でなければ
    RenderError (黙って描かない、ではなく知らせる)。figure_config が無い (単独描画) なら空。
    """
    items = (panel.get("map") or {}).get("section_paths") or []
    if not items or not figure_config:
        return []
    out = []
    for item in items:
        ref = item.get("panel_id")
        target = next((p for p in figure_config.get("panels") or []
                       if p.get("panel_id") == ref), None)
        if target is None:
            raise RenderError("section_overlay_panel_missing", panel_id=ref)
        ll = section_panel_lonlat(target, datasets)
        if ll is None:
            raise RenderError("section_overlay_not_section", panel_id=ref)
        out.append((item, ll[0], ll[1]))
    return out


def section_overlay_kwargs(item: dict) -> dict:
    """経路の線の ax.plot 引数 (render / scriptgen 共用)。"""
    return {"color": item.get("color") or "black",
            "linewidth": float(item.get("width", 1.5)),
            "linestyle": item.get("linestyle") or "-"}


def section_overlay_label_specs(item: dict, lon, lat) -> list[tuple]:
    """端点の文字 [(lon, lat, text, ha)]。end_labels が False なら空、空文字の端は出さない。
    始点は進行方向と逆側 (右向きの経路なら左) に寄せる。"""
    if not item.get("end_labels"):
        return []
    labels = list(item.get("labels") or ["A", "B"])
    labels = (labels + ["A", "B"])[:2]
    eastward = float(lon[-1]) >= float(lon[0])
    specs = [(float(lon[0]), float(lat[0]), str(labels[0]), "right" if eastward else "left"),
             (float(lon[-1]), float(lat[-1]), str(labels[1]), "left" if eastward else "right")]
    return [sp for sp in specs if sp[2].strip()]


def terrain_enabled(panel: dict) -> bool:
    """地形マスク (panel["terrain"]) が有効か。旧 config (キー無し) は False。"""
    return bool((panel.get("terrain") or {}).get("show"))


def _terrain_selection(panel: dict) -> dict:
    """地形の変数に当てる固定値: パネルの selection + 先頭レイヤーの selection
    (格子線断面・従来の断面の固定した行・緯度は先頭レイヤーが持つ)。"""
    layers = panel.get("layers") or []
    first = (layers[0].get("selection") or {}) if layers else {}
    return {**(panel.get("selection") or {}), **first}


def section_terrain_profile(panel: dict, datasets: dict, dsid: str, var: str,
                            with_vertical: bool = False) -> xr.DataArray:
    """地形マスク用に変数 var を断面の位置で切り出す。

    with_vertical=False: 水平 2 次元の変数 (地上気圧・地形高度) → 断面の横軸だけの 1 次元。
    with_vertical=True: 鉛直を持つ変数 (高度の変数) → (y_dim, x_dim) の 2 次元。
    経路断面は経路上へ双一次内挿、格子線・従来の断面は先頭レイヤーの固定値で切り出す
    (scriptgen._terrain_lines と 1 対 1)。
    """
    ds = datasets.get(dsid)
    if ds is None or var not in ds.data_vars:
        raise RenderError("terrain_variable_missing", var=var, dataset=dsid)
    x_dim, y_dim = panel["x_dim"], panel["y_dim"]
    spec = panel.get("section_path")
    if spec:
        roles = detect_coord_roles(ds)
        if not (roles.get("lon") and roles.get("lat")):
            raise RenderError("terrain_no_lonlat", var=var, dataset=dsid)
    if with_vertical:
        return select_section_data(ds, var, _terrain_selection(panel), panel.get("ranges"),
                                   x_dim, y_dim, path=spec)
    da = ds[var]
    sel = {d: v for d, v in _terrain_selection(panel).items() if d in da.dims}
    if sel:
        da = da.sel(sel)
    if spec:
        da = sample_section_path(da, ds, section_path_spec(spec, ds), x_dim)
    for dim, bounds in (panel.get("ranges") or {}).items():
        if dim in da.dims:
            da = da.sel({dim: coord_slice(ds[dim], bounds[0], bounds[1])})
    if tuple(da.dims) != (x_dim,):
        raise RenderError("terrain_profile_dims", var=var, x_dim=x_dim, dims=list(da.dims))
    return da


def section_terrain_spec(panel: dict, datasets: dict) -> dict:
    """地形マスクの設定を解決し、単位の換算まで確定した dict を返す。

    返り値: {"method", "dsid", "var", "factor", "pressure_like", "hdsid", "hvar",
    "hfactor", "lev_factor", "y_units"}。scriptgen も同じ関数で倍率を決めてリテラルで出す。
    """
    cfg = panel.get("terrain") or {}
    layers = panel.get("layers") or []
    ds0 = datasets[layers[0]["dataset_id"]]
    y_dim = panel["y_dim"]
    y_units = str(ds0[y_dim].attrs.get("units", ""))
    kind = vertical_axis_kind(y_units)
    if kind is None:
        raise RenderError("terrain_vertical_units_unknown", dim=y_dim, units=y_units)
    method = cfg.get("method") or "surface_pressure"
    need = {"surface_pressure": "pressure", "surface_height": "height",
            "height_field": "pressure"}.get(method)
    if need is None or need != kind:
        expected = ("surface_pressure or height_field" if kind == "pressure"
                    else "surface_height")
        raise RenderError("terrain_method_mismatch", method=method, kind=kind, dim=y_dim,
                          units=y_units, expected=expected)
    dsid = cfg.get("dataset_id") or layers[0]["dataset_id"]
    var = cfg.get("variable")
    ds = datasets.get(dsid)
    if ds is None or not var or var not in ds.data_vars:
        raise RenderError("terrain_variable_missing", var=var, dataset=dsid)
    out = {"method": method, "dsid": dsid, "var": var, "pressure_like": kind == "pressure",
           "y_units": y_units, "hdsid": None, "hvar": None, "hfactor": None, "lev_factor": None}
    if method == "height_field":
        # 地形高度は m に、高度の変数も m に (ジオポテンシャルは g で割る)、気圧面は y_units のまま
        zs_units = str(ds[var].attrs.get("units", ""))
        f = unit_factor(zs_units, "m", "height")
        if f is None:
            raise RenderError("terrain_units_unknown", units=zs_units, var=var, target="m",
                              expected="m, km, m2 s-2")
        out["factor"] = f
        hdsid = cfg.get("height_dataset_id") or layers[0]["dataset_id"]
        hvar = cfg.get("height_variable")
        hds = datasets.get(hdsid)
        if hds is None or not hvar or hvar not in hds.data_vars:
            raise RenderError("terrain_variable_missing", var=hvar, dataset=hdsid)
        h_units = str(hds[hvar].attrs.get("units", ""))
        hf = unit_factor(h_units, "m", "height")
        if hf is None:
            raise RenderError("terrain_units_unknown", units=h_units, var=hvar, target="m",
                              expected="m, km, m2 s-2")
        out.update({"hdsid": hdsid, "hvar": hvar, "hfactor": hf, "lev_factor": 1.0})
    else:
        s_units = str(ds[var].attrs.get("units", ""))
        f = unit_factor(s_units, y_units, kind)
        if f is None:
            raise RenderError("terrain_units_unknown", units=s_units, var=var, target=y_units,
                              expected=("Pa, hPa" if kind == "pressure" else "m, km, m2 s-2"))
        out["factor"] = f
    return out


def section_terrain_ground(panel: dict, datasets: dict):
    """地形マスクの地面の線 (x, ground [鉛直座標の単位], pressure_like) を返す。

    scriptgen._terrain_lines と 1 対 1。show が False なら None。
    """
    if not terrain_enabled(panel):
        return None
    spec = section_terrain_spec(panel, datasets)
    surf = section_terrain_profile(panel, datasets, spec["dsid"], spec["var"])
    x = surf[panel["x_dim"]].values
    if spec["method"] == "height_field":
        z = section_terrain_profile(panel, datasets, spec["hdsid"], spec["hvar"],
                                    with_vertical=True)
        ground = ground_pressure_from_height(z.values * spec["hfactor"],
                                             z[panel["y_dim"]].values * spec["lev_factor"],
                                             surf.values * spec["factor"])
    else:
        ground = surf.values * spec["factor"]
    return x, ground, spec["pressure_like"]


# --- 描画 ---

def figure_layout(figure_config: dict) -> tuple[int, int]:
    """figure.layout から (nrows, ncols) を返す。パネル数に対して不足なら ValueError。

    layout 未指定 (旧 config) は 1×1 として扱い、単一パネルの後方互換を保つ。
    パネルはグリッドに行優先 (左上→右) で配置される。
    """
    fig_cfg = figure_config.get("figure", {})
    layout = fig_cfg.get("layout") or {}
    nrows = int(layout.get("nrows", 1) or 1)
    ncols = int(layout.get("ncols", 1) or 1)
    n_panels = len(figure_config["panels"])
    if nrows * ncols < n_panels:
        raise RenderError("layout_too_small", nrows=nrows, ncols=ncols,
                          n_panels=n_panels)
    return nrows, ncols


def parse_mosaic(mosaic: str, n_panels: int) -> tuple[int, int, list]:
    """mosaic 文字列を (nrows, ncols, [パネル毎のセル位置]) に解決する。

    記法は matplotlib subplot_mosaic と同じ発想: 行を ";" (または改行) で
    区切り、1文字 = 1セル。同じ文字を矩形に並べるとセル結合、"." は空きセル。
    空白は無視する。ラベルの初出順 (左上から行優先) が panels の順に対応する。

    返り値のセル位置は add_subplot 用の 1始まり行優先番号で、単一セルは int、
    結合セルは (左上, 右下) のタプル (`fig.add_subplot(r, c, (first, last))`)。
    render / scriptgen / UI (検証・プレビュー) の3者で共用する (ルール4)。
    不正な入力 (行の長さ不揃い・非矩形・パネル数不一致) は ValueError。
    """
    rows = ["".join(r.split())
            for r in str(mosaic).replace("\n", ";").split(";")
            if r.strip()]
    if not rows:
        raise RenderError("mosaic_empty")
    ncols = len(rows[0])
    if any(len(r) != ncols for r in rows):
        raise RenderError("mosaic_ragged", rows=" / ".join(rows))
    nrows = len(rows)
    order: list[str] = []                       # ラベルの初出順
    label_cells: dict[str, list[tuple[int, int]]] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            if ch not in label_cells:
                order.append(ch)
                label_cells[ch] = []
            label_cells[ch].append((r, c))
    if len(order) != n_panels:
        raise RenderError("mosaic_count_mismatch", n_labels=len(order),
                          labels="".join(order), n_panels=n_panels)
    specs = []
    for ch in order:
        rs = [rc[0] for rc in label_cells[ch]]
        cs = [rc[1] for rc in label_cells[ch]]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(label_cells[ch]):
            raise RenderError("mosaic_not_rect", label=ch)
        first = r0 * ncols + c0 + 1
        last = r1 * ncols + c1 + 1
        specs.append(first if first == last else (first, last))
    return nrows, ncols, specs


def panel_subplot_specs(figure_config: dict) -> tuple[int, int, list]:
    """各パネルの add_subplot 位置引数 (nrows, ncols, [セル位置]) を解決する。

    layout.mosaic が指定されていればセル結合・空きセルを含む mosaic 配置
    (nrows/ncols は mosaic から決まる)、無ければ従来の行優先 (i+1)。
    render_figure と scriptgen._emit_panels の両方から使う (ルール4)。
    """
    panels = figure_config["panels"]
    layout = figure_config.get("figure", {}).get("layout") or {}
    mosaic = layout.get("mosaic")
    if mosaic:
        return parse_mosaic(mosaic, len(panels))
    nrows, ncols = figure_layout(figure_config)
    return nrows, ncols, list(range(1, len(panels) + 1))


def grid_ratio_kwargs(figure_config: dict, nrows: int, ncols: int) -> dict:
    """layout の width_ratios / height_ratios を add_gridspec の kwargs に解決する。

    両方 None (既定) なら空 dict — このとき呼び出し側は GridSpec を使わない
    従来の add_subplot(nrows, ncols, n) 経路のまま (既存図・既存生成スクリプトを
    変えない)。長さ不一致・非正値は可読な RenderError。
    render と scriptgen の両方から使う (ルール4)。
    """
    layout = figure_config.get("figure", {}).get("layout") or {}
    kwargs: dict = {}
    for key, name, expected, unit in (
            ("width_ratios", "width_ratios", ncols, "column"),
            ("height_ratios", "height_ratios", nrows, "row")):
        vals = layout.get(key)
        if vals is None:
            continue
        try:
            vals = [float(v) for v in vals]
        except (TypeError, ValueError):
            raise RenderError("ratio_invalid", name=name, values=layout.get(key))
        if any(not np.isfinite(v) or v <= 0 for v in vals):
            raise RenderError("ratio_invalid", name=name, values=layout.get(key))
        if len(vals) != expected:
            raise RenderError("ratio_len_mismatch", name=name, n=len(vals),
                              expected=expected, unit=unit)
        kwargs[key] = vals
    return kwargs


def cell_grid_bounds(cell, ncols: int) -> tuple[int, int, int, int]:
    """セル位置 (int または結合の (左上, 右下)、1始まり行優先番号) を
    0始まりの (r0, r1, c0, c1) に変換する (両端含む)。

    render は gs[r0:r1+1, c0:c1+1]、scriptgen は同じ式のテキストを組む (1対1)。
    """
    first, last = (cell, cell) if isinstance(cell, int) else cell
    r0, c0 = divmod(first - 1, ncols)
    r1, c1 = divmod(last - 1, ncols)
    return r0, r1, c0, c1


def _render_panel(fig, panel: dict, datasets: dict[str, xr.Dataset],
                  subplot: tuple, figure_config: dict | None = None):
    """plot_type に応じたレンダラへ dispatch する (subplot = add_subplot の位置引数)。

    subplot は (nrows, ncols, セル位置) — セル位置は行優先のセル番号 (int) または
    結合セルの (左上, 右下) タプル (parse_mosaic 参照) — か、行・列の比率指定時は
    (SubplotSpec,) の1要素タプル (どちらも fig.add_subplot(*subplot) で通る)。
    """
    if panel["plot_type"] == "horizontal_map":
        return _render_horizontal_map(fig, panel, datasets, subplot=subplot,
                                      figure_config=figure_config)
    elif panel["plot_type"] == "section_2d":
        return _render_section_2d(fig, panel, datasets, subplot=subplot)
    elif panel["plot_type"] == "line_1d":
        return _render_line_1d(fig, panel, datasets, subplot=subplot)
    elif panel["plot_type"] == "scatter_2d":
        return _render_scatter_2d(fig, panel, datasets, subplot=subplot)
    elif panel["plot_type"] == "heatmap":
        return _render_heatmap(fig, panel, datasets, subplot=subplot)
    elif panel["plot_type"] == "dist_1d":
        return _render_dist_1d(fig, panel, datasets, subplot=subplot)
    elif panel["plot_type"] == "agg_2d":
        return _render_agg_2d(fig, panel, datasets, subplot=subplot)
    raise NotImplementedError(f"Unsupported plot_type: {panel['plot_type']}")


def subplots_adjust_kwargs(figure_config: dict) -> dict:
    """figure.layout の余白設定を fig.subplots_adjust() の kwargs に変換する。

    None (未指定) の項目は含めない (matplotlib 既定に任せる)。
    subplot 追加前に適用すること (後から呼ぶと colorbar の配置がずれる)。
    """
    layout = (figure_config.get("figure", {}).get("layout") or {})
    out = {}
    for key in ("left", "right", "bottom", "top", "wspace", "hspace"):
        value = layout.get(key)
        if value is not None:
            out[key] = float(value)
    return out


def shared_colorbar_config(figure_config: dict) -> dict:
    """figure.shared_colorbar (全パネル共通カラーバー、仕様22.5) を返す。"""
    return figure_config.get("figure", {}).get("shared_colorbar") or {}


def _record_shared_mappable(ax, mappable) -> None:
    """共通カラーバー用に、axes ごとの最初の fill mappable を記録する。"""
    if not hasattr(ax, "_cc_shared_mappable"):
        ax._cc_shared_mappable = mappable


def _draw_shared_colorbar(fig, axes: list, figure_config: dict):
    """全パネル共通のカラーバーを描画する (fill レイヤーの mappable を共有)。

    最初に見つかった fill mappable を代表として使う。全パネルで vmin/vmax/levels/
    cmap を揃えるのは呼び出し側 (UI) の責任。fill が1つも無ければ何もしない。
    個別レイヤーのカラーバーとの併用は可能だが、通常は UI 側で個別を OFF にする。
    """
    cfg = shared_colorbar_config(figure_config)
    if not cfg.get("show"):
        return None
    mappable = None
    for ax in axes:
        mappable = getattr(ax, "_cc_shared_mappable", None)
        if mappable is not None:
            break
    if mappable is None:
        return None
    return _add_colorbar(fig, axes, mappable, cfg)


# 起動時 (matplotlibrc 適用後) のフォント設定。font_family 未指定の図で
# 毎回ここへ戻すことで、直前の図のフォント設定が漏れないようにする
_INITIAL_FONT_FAMILY = list(plt.rcParams["font.family"])


def apply_font_family(figure_config: dict) -> None:
    """figure.font_family を rcParams に適用する (None なら起動時の値へリセット)。

    rc_context (一時適用) にしないのは、cartopy の Gridliner ラベルや目盛りが
    savefig 時の再描画で作り直され、そのとき現在の rcParams を参照するため。
    グローバルに設定し、描画・保存まで同じフォント状態を保つ
    (生成スクリプトの冒頭 rcParams 設定と1対1対応)。
    """
    font = figure_config.get("figure", {}).get("font_family")
    plt.rcParams["font.family"] = font if font else _INITIAL_FONT_FAMILY


def render_figure(figure_config: dict, datasets: dict[str, xr.Dataset]):
    """figure_config から Figure を作る (アプリ描画の入口)。

    cartopy 深部の退化ポリゴン起因の GEOS エラー (getX called on empty
    Point — contourf の色レベルがデータの min/max や定数領域の値と一致
    すると起きる) は、対処ヒント付きの RenderError に変換する。
    端レベルの自動トリムによる予防も実装したが、既存図の見た目が変わる
    ため取りやめた (2026-08-23) — 回避はユーザー操作に委ね、この変換だけを
    残す (cartopy_compat_notes.md「既知の上流バグ」節)。
    """
    # カスタムカラーマップの登録を保証する (UI 層を import しない経路 — テストの
    # コールド実行やコア層だけの利用 — でも custom cmap 名を解決できるように。
    # 読み込み済みならキャッシュで即返る)
    load_custom_cmaps()
    try:
        return _render_figure(figure_config, datasets)
    except RenderError:
        raise
    except Exception as exc:
        if "empty Point" in str(exc):
            raise RenderError("degenerate_geometry") from exc
        if "point array must contain 0 or >1 elements" in str(exc):
            # cartopy の gridliner が地平線に1点で接する緯度経度線を退化
            # LineString にする (Orthographic の特定の中心で実測 2026-08-28、
            # cartopy_compat_notes.md「既知の上流バグ」節)
            raise RenderError("degenerate_gridline") from exc
        raise


def _render_figure(figure_config: dict, datasets: dict[str, xr.Dataset]):
    fig_cfg = figure_config.get("figure", {})
    apply_font_family(figure_config)
    fig = plt.figure(figsize=tuple(fig_cfg.get("figsize", (10.0, 6.0))),
                     dpi=fig_cfg.get("dpi", 100))
    adjust = subplots_adjust_kwargs(figure_config)
    if adjust:
        fig.subplots_adjust(**adjust)
    nrows, ncols, cells = panel_subplot_specs(figure_config)
    ratios = grid_ratio_kwargs(figure_config, nrows, ncols)
    if ratios:
        # 比率指定時のみ GridSpec 経由 (余白は figure.subplotpars を参照する
        # ので上の subplots_adjust がそのまま効く)。add_subplot は
        # SubplotSpec 1引数も受けるため、各レンダラの呼び出しは無変更
        gs = fig.add_gridspec(nrows, ncols, **ratios)
        subplots = []
        for cell in cells:
            r0, r1, c0, c1 = cell_grid_bounds(cell, ncols)
            subplots.append((gs[r0:r1 + 1, c0:c1 + 1],))
    else:
        subplots = [(nrows, ncols, cell) for cell in cells]
    axes = []
    for panel, subplot in zip(figure_config["panels"], subplots):
        axes.append(_render_panel(fig, panel, datasets, subplot, figure_config))
    _draw_shared_colorbar(fig, axes, figure_config)
    return fig


def animation_time_dim(figure_config: dict, datasets: dict[str, xr.Dataset]) -> str | None:
    """最初のレイヤーを持つパネルの第1レイヤーの dataset から時刻次元名を判定する。

    複数 dataset は読み込み時に dim 名が canonical に整列される (align_dim_names)
    ため、時刻次元名は全パネルで共通とみなせる。
    """
    for panel in figure_config["panels"]:
        if panel.get("layers"):
            ds = datasets[panel["layers"][0]["dataset_id"]]
            return detect_coord_roles(ds).get("time")
    return None


def panel_animates_time(panel: dict, time_dim: str) -> bool:
    """時刻送りで selection[time_dim] を上書きしてよいパネルかを返す。

    時刻が描画軸そのもの (Hovmöller の x/y 軸、1次元プロットの x 軸) のパネルは
    時刻を固定すると図が壊れるため対象外。scriptgen の時刻変数注入と共通のルール。
    """
    return time_dim not in (panel.get("x_dim"), panel.get("y_dim"))


def rotation_path(waypoints, n_frames: int) -> list[tuple[float, float]]:
    """経由点列 [(lon, lat), ...] を n_frames 個の投影中心に線形補間する。

    Orthographic 図法の地球回転アニメーション用。先頭が開始点、末尾が終点。
    - 経度の区間差は最短方向にとる (差がちょうど ±180° のときは差の符号の向き)。
      同じ向きに回し続けるには 180° 未満の間隔で経由点を置く
    - 区間には角距離 (度空間の hypot) に比例してフレームを配分し、経路上の
      速度を一定にする (経由点で速度が急変しない)
    - 経度は正規化せず連続値のまま返す (cartopy は ±180° を超える中心経度も
      受け付ける。再現スクリプトの centers が単調に読める)
    - 値は小数 4 桁に丸める (render と scriptgen が同じ値を使う前提)
    n_frames == 1 は開始点のみ。長さ 0 の経路 (全点同一) は開始点の繰り返し。
    """
    pts = [(float(lon), float(lat)) for lon, lat in waypoints]
    if not pts:
        raise ValueError("rotation_path: at least one waypoint is required")
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("rotation_path: n_frames must be >= 1")
    lons = [pts[0][0]]
    lats = [pts[0][1]]
    for (lon, lat), (prev_lon, _prev_lat) in zip(pts[1:], pts[:-1]):
        d = lon - prev_lon
        w = ((d + 180.0) % 360.0) - 180.0  # [-180, 180)
        if w == -180.0 and d > 0:
            w = 180.0  # +180° は東回り、-180° は西回り
        lons.append(lons[-1] + w)
        lats.append(lat)
    dist = [0.0]
    for i in range(1, len(lons)):
        dist.append(dist[-1] + float(np.hypot(lons[i] - lons[i - 1], lats[i] - lats[i - 1])))
    total = dist[-1]
    if total <= 0.0 or n_frames == 1:
        return [(round(lons[0], 4) + 0.0, round(lats[0], 4) + 0.0)] * n_frames
    # 長さ 0 の区間 (同一点の連続) は np.interp の単調性のため落とす
    keep = [0] + [i for i in range(1, len(dist)) if dist[i] > dist[i - 1]]
    xs = [dist[i] for i in keep]
    s = np.linspace(0.0, total, n_frames)
    out_lon = np.interp(s, xs, [lons[i] for i in keep])
    out_lat = np.interp(s, xs, [lats[i] for i in keep])
    return [(round(float(a), 4) + 0.0, round(float(b), 4) + 0.0)
            for a, b in zip(out_lon, out_lat)]


def expand_time_values(time_values, frames_per_time: int = 1) -> list:
    """各時刻を frames_per_time コマずつ繰り返したフレーム列にする。

    時刻送り + 地球回転で「時刻はゆっくり (同じ時刻を数コマ) 進め、回転はコマ毎に
    滑らかに進める」ための展開。scriptgen は同じ内包表記を生成スクリプトに書く。
    """
    k = int(frames_per_time)
    if k < 1:
        raise ValueError("frames_per_time must be >= 1")
    return [v for v in time_values for _ in range(k)]


def animation_frame_count(time_values, centers, frames_per_time: int = 1) -> int:
    """アニメーションのフレーム数 (時刻数 × コマ数、時刻固定なら中心の数)。"""
    if time_values is not None:
        return len(time_values) * int(frames_per_time)
    return len(centers)


def iter_frames(figure_config: dict, datasets: dict[str, xr.Dataset],
                time_values=None, centers=None, frames_per_time: int = 1):
    """アニメーションの各フレームの Figure を yield する。

    - time_values: 各フレームで selection[time_dim] に入れる時刻。None なら時刻は
      パネルの selection のまま (時刻固定)
    - centers: 各フレームの投影中心 (lon, lat)。None なら投影はそのまま。
      Orthographic のパネルにだけ適用する (地球回転アニメーション、rotation_path)
    - frames_per_time: 各時刻を何コマ保持するか (expand_time_values)。時刻送り +
      回転で回転を滑らかにする。時刻固定では無視
    時刻 (展開後) と中心の両方を渡すときは同じ長さであること (同時に動かす)。

    複数パネルでは全パネルの時刻を同期して送る。ただし時刻が描画軸のパネル
    (panel_animates_time が False) は上書きしない。selection は変数の次元で
    フィルタされるため、時刻次元を持たない変数でも無視されるだけで安全。
    呼び出し側は yield された Figure を使い終わったら `plt.close(fig)` すること。
    time_label が ON なら text も時刻に追従させる。
    """
    if time_values is None and centers is None:
        raise ValueError("iter_frames: time_values or centers is required")
    if time_values is not None:
        time_values = expand_time_values(time_values, frames_per_time)
    n = len(time_values) if time_values is not None else len(centers)
    if time_values is not None and centers is not None and len(centers) != n:
        raise ValueError("iter_frames: time_values and centers must have the same length")
    time_dim = None
    if time_values is not None:
        time_dim = animation_time_dim(figure_config, datasets)
        if time_dim is None:
            raise RenderError("no_time_dim")
    for i in range(n):
        cfg = copy.deepcopy(figure_config)
        for panel in cfg["panels"]:
            if time_values is not None and panel_animates_time(panel, time_dim):
                time_value = time_values[i]
                panel.setdefault("selection", {})[time_dim] = time_value
                # レイヤー側の selection に時刻が固定値で入っている設定 (最初の
                # dataset に時刻役割が無い読み込み順で作られたレイヤー等) では、
                # 描画時に layer.selection が panel.selection を上書きするため
                # パネル側だけ差し替えてもそのパネルの時刻が進まない (2026-09-19、
                # 複製した 2 パネルの片方だけ止まる実機報告)。scriptgen の _sel_arg は
                # レイヤー側の .sel(time=...) も time_value に置き換えるので、
                # render もレイヤー側を差し替えて一致させる
                for layer in panel.get("layers") or []:
                    layer_sel = layer.get("selection")
                    if layer_sel and time_dim in layer_sel:
                        layer_sel[time_dim] = time_value
                tl = panel.get("time_label")
                if tl and tl.get("show"):
                    tl["text"] = str(time_value)
            if centers is not None and panel_rotates(panel):
                clon, clat = centers[i]
                panel["projection"]["central_longitude"] = float(clon)
                panel["projection"]["central_latitude"] = float(clat)
        yield render_figure(cfg, datasets)


def panel_rotates(panel: dict) -> bool:
    """地球回転アニメーションで投影中心を上書きするパネルか (Orthographic の地図)。"""
    return (panel.get("plot_type") == "horizontal_map"
            and panel.get("projection", {}).get("name") == "Orthographic")


def iter_time_frames(figure_config: dict, datasets: dict[str, xr.Dataset], time_values):
    """time_values の各値について selection[time_dim] を上書きし、Figure を yield する
    (iter_frames の時刻送りのみの形)。"""
    return iter_frames(figure_config, datasets, time_values=time_values)


def animation_gif_tight(centers) -> bool:
    """GIF フレームを bbox_inches="tight" で保存するか。

    地球回転 (centers あり) では緯度経度ラベルの位置がフレーム毎に動いて外接枠が
    揺れ、Pillow が先頭フレームの大きさで後続を切ってしまうため固定サイズにする。
    回転なしは従来どおり tight (見た目互換)。scriptgen と共通のルール。
    """
    return centers is None


def save_animation_gif(figure_config: dict, datasets: dict[str, xr.Dataset],
                       time_values, output_path: str,
                       fps: int = 4, dpi: int = 100,
                       progress=None, centers=None, frames_per_time: int = 1) -> None:
    """time_values / centers で各フレームを描画し、Pillow で GIF として保存する。

    time_values は None 可 (時刻固定で地球回転のみ)。centers は各フレームの
    投影中心、frames_per_time は各時刻のコマ数 (iter_frames 参照)。
    progress は (i, n) を受け取るコールバック (省略可)。
    """
    from PIL import Image
    frames = []
    n = animation_frame_count(time_values, centers, frames_per_time)
    tight = animation_gif_tight(centers)
    for i, fig in enumerate(iter_frames(figure_config, datasets,
                                        time_values=time_values, centers=centers,
                                        frames_per_time=frames_per_time)):
        buf = io.BytesIO()
        if tight:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        if progress and n:
            progress(i + 1, n)
    if not frames:
        raise RenderError("no_frames")
    # output_path はパス文字列でも file-like (BytesIO 等) でもよい
    frames[0].save(output_path, format="GIF", save_all=True,
                   append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)


def ffmpeg_available() -> bool:
    """ffmpeg バイナリが PATH 上で見つかり実行できるか。"""
    try:
        result = subprocess.run(["ffmpeg", "-version"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def save_animation_mp4(figure_config: dict, datasets: dict[str, xr.Dataset],
                       time_values, output_path: str,
                       fps: int = 4, dpi: int = 100,
                       progress=None, centers=None, frames_per_time: int = 1) -> None:
    """各フレームを tempdir に PNG として書き出し、ffmpeg で MP4 に結合する。

    time_values / centers / frames_per_time の扱いは save_animation_gif と同じ。
    `bbox_inches="tight"` は使わない (フレーム間でサイズが揺れて ffmpeg が拒否するため)。
    libx264 + yuv420p で広い再生環境に対応する。`-vf scale=...` で縦横を偶数にクランプする。
    """
    if not ffmpeg_available():
        raise RenderError("ffmpeg_missing")
    n = animation_frame_count(time_values, centers, frames_per_time)
    n_frames = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, fig in enumerate(iter_frames(figure_config, datasets,
                                            time_values=time_values, centers=centers,
                                            frames_per_time=frames_per_time)):
            fig.savefig(os.path.join(tmp, f"frame_{i:05d}.png"), dpi=dpi)
            plt.close(fig)
            n_frames = i + 1
            if progress and n:
                progress(n_frames, n)
        if n_frames == 0:
            raise RenderError("no_frames")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            output_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def _set_title(ax, panel: dict):
    if panel.get("title"):
        if panel.get("title_fontsize"):
            ax.set_title(panel["title"], fontsize=panel["title_fontsize"])
        else:
            ax.set_title(panel["title"])


def text_annotation_kwargs(t: dict) -> dict:
    """default_text_annotation の1要素を ax.text() の kwargs に変換する。

    座標 (x, y) と text 本文は引数で渡すため kwargs には含めない。
    transform は呼び出し側で ax.transAxes をセットする。
    """
    return {
        "fontsize": t.get("fontsize", 12),
        "color": t.get("color", "#000000"),
        "ha": t.get("ha", "left"),
        "va": t.get("va", "top"),
        "rotation": float(t.get("rotation", 0.0)),
    }


def _draw_text_annotations(ax, panel: dict):
    """panel["texts"] にある任意位置の文字列を描画する。

    coord="axes" (既定): (x, y) は axes fraction (0..1, ax.transAxes)。
    coord="data": (x, y) はデータ座標。水平断面図では経度・緯度
    (PlateCarree で投影変換)、その他のプロットでは値そのもの (transData)。
    どちらも範囲外は clip_on=False で許可する。
    """
    is_map = panel.get("plot_type") == "horizontal_map"
    for t in panel.get("texts", []) or []:
        s = t.get("text")
        if not s:
            continue
        x = float(t.get("x", 0.05))
        y = float(t.get("y", 0.95))
        kw = text_annotation_kwargs(t)
        if t.get("coord", "axes") == "data":
            if is_map:
                ax.text(x, y, s, transform=ccrs.PlateCarree(),
                        clip_on=False, **kw)
            else:
                ax.text(x, y, s, clip_on=False, **kw)
        else:
            ax.text(x, y, s, transform=ax.transAxes, clip_on=False, **kw)


def marker_annotation_kwargs(m: dict) -> dict:
    """default_marker_annotation の1要素を ax.plot() の kwargs に変換する。

    座標 (x, y) と transform は呼び出し側で渡す。linestyle="None" で線を描かず
    マーカーのみ表示する。
    """
    kw = {
        "marker": m.get("marker", "o"),
        "markersize": float(m.get("size", 10.0)),
        "color": m.get("color", "#d62728"),
        "linestyle": "None",
    }
    if float(m.get("edge_width", 0.0)) > 0:
        kw["markeredgewidth"] = float(m.get("edge_width", 0.0))
        kw["markeredgecolor"] = m.get("edge_color", "#000000")
    else:
        kw["markeredgewidth"] = 0.0
    return kw


def _draw_marker_annotations(ax, panel: dict):
    """panel["markers"] にある任意位置の記号を描画する。

    coord の解釈は _draw_text_annotations と同じ (axes 相対 / データ座標。
    水平断面図のデータ座標は経度・緯度で PlateCarree 変換)。
    """
    is_map = panel.get("plot_type") == "horizontal_map"
    for m in panel.get("markers", []) or []:
        if not m.get("marker"):
            continue
        x = float(m.get("x", 0.5))
        y = float(m.get("y", 0.5))
        kw = marker_annotation_kwargs(m)
        if m.get("coord", "axes") == "data":
            if is_map:
                ax.plot([x], [y], transform=ccrs.PlateCarree(),
                        clip_on=False, **kw)
            else:
                ax.plot([x], [y], clip_on=False, **kw)
        else:
            ax.plot([x], [y], transform=ax.transAxes, clip_on=False, **kw)


def panel_label_kwargs(label_cfg: dict) -> dict:
    """panel.label を ax.text() の kwargs に変換する (座標と本文は含めない)。

    仕様22.6: パネルラベル (a), (b), ... の書式。既定は太字・左上外側。
    """
    return {
        "fontsize": label_cfg.get("fontsize", 12),
        "fontweight": label_cfg.get("weight", "bold"),
        "color": label_cfg.get("color", "#000000"),
        "ha": label_cfg.get("ha", "left"),
        "va": label_cfg.get("va", "bottom"),
    }


def _draw_panel_label(ax, panel: dict):
    """panel.label ((a), (b) などのパネルラベル) を axes 座標で描画する。"""
    cfg = panel.get("label") or {}
    if not (cfg.get("show") and cfg.get("text")):
        return
    ax.text(float(cfg.get("x", 0.0)), float(cfg.get("y", 1.02)), cfg["text"],
            transform=ax.transAxes, clip_on=False,
            **panel_label_kwargs(cfg))


def format_time_label(text, fmt) -> str:
    """time_label の text に strftime 書式を適用する。

    fmt=None や text が parse できない場合は str(text) のまま返す。
    """
    if not fmt:
        return str(text)
    try:
        import pandas as _pd
        return _pd.Timestamp(text).strftime(fmt)
    except (ValueError, TypeError):
        return str(text)


def _draw_time_label(ax, panel: dict):
    """選択中の時刻をプロット領域の上部に表示する。

    位置は loc ('left'/'center'/'right') で指定。matplotlib のタイトル位置に乗せると
    上側の目盛りラベルを自動回避でき、中央タイトルと共存できる。
    """
    tl = panel.get("time_label", {})
    if tl.get("show") and tl.get("text"):
        text = format_time_label(tl["text"], tl.get("format"))
        ax.set_title(text, loc=tl.get("loc", "left"))


def _add_colorbar(fig, ax, mappable, cbar_cfg: dict, extend: str | None = None):
    if not cbar_cfg.get("show", True):
        return None
    cbar = fig.colorbar(mappable, ax=ax, **colorbar_kwargs(cbar_cfg, extend))
    plan = colorbar_side_plan(cbar_cfg)
    if plan:
        _axis = getattr(cbar.ax, plan[0])
        _axis.set_ticks_position(plan[1])
        _axis.set_label_position(plan[2])
    if cbar_cfg.get("label"):
        cbar.set_label(cbar_cfg["label"], **colorbar_label_kwargs(cbar_cfg))
    if cbar_cfg.get("tick_fontsize"):
        cbar.ax.tick_params(labelsize=cbar_cfg["tick_fontsize"])
    if cbar_cfg.get("outline_width") is not None:
        cbar.outline.set_linewidth(float(cbar_cfg["outline_width"]))
    if cbar_cfg.get("tick_width") is not None:
        cbar.ax.tick_params(width=float(cbar_cfg["tick_width"]))
    if cbar_cfg.get("tick_pad") is not None:
        cbar.ax.tick_params(pad=float(cbar_cfg["tick_pad"]))
    return cbar


def match_colorbars_to_axes(fig, ax, cbars):
    """カラーバーの長さを地図の実描画高さ (アスペクト比適用後) に合わせる。

    投影のアスペクト比固定により、確保された axes 領域より地図の描画域が
    小さくなるため、デフォルトのカラーバーは地図より長くなる。描画後の
    位置を基準に shrink 比で合わせ直す。scriptgen が生成するコードと
    同じ操作であること。
    """
    if not cbars:
        return
    fig.canvas.draw()
    pos = ax.get_position()
    for cbar, cbar_cfg in cbars:
        shrink = cbar_cfg.get("shrink", 1.0) or 1.0
        cax = cbar.ax
        cpos = cax.get_position()
        cax.set_axes_locator(None)
        if cbar_cfg.get("location", "right") in ("right", "left"):
            cax.set_position([cpos.x0, pos.y0 + pos.height * ((1 - shrink) / 2),
                              cpos.width, pos.height * shrink])
        else:
            cax.set_position([pos.x0 + pos.width * ((1 - shrink) / 2), cpos.y0,
                              pos.width * shrink, cpos.height])


def _apply_box_aspect(ax, panel: dict) -> None:
    """panel.box_aspect が指定されていれば axes 枠の縦横比を固定する。"""
    ba = panel.get("box_aspect")
    if ba is not None:
        ax.set_box_aspect(float(ba))


def _render_horizontal_map(fig, panel: dict, datasets: dict[str, xr.Dataset],
                           subplot: tuple[int, int, int] = (1, 1, 1),
                           figure_config: dict | None = None):
    if panel["plot_type"] != "horizontal_map":
        raise NotImplementedError(f"Unsupported plot_type: {panel['plot_type']}")
    proj_cfg = panel["projection"]
    ax = fig.add_subplot(*subplot, projection=make_projection(proj_cfg))
    _apply_box_aspect(ax, panel)
    map_cfg = panel.get("map", {})

    ne_scale = natural_earth_scale(map_cfg)
    land_fg = land_foreground(map_cfg)

    def _ne_feature(feature):
        # None (自動) は cartopy 組み込みの AdaptiveScaler をそのまま使う
        return feature if ne_scale is None else feature.with_scale(ne_scale)

    land = map_cfg.get("land", {})
    if land.get("show"):
        ax.add_feature(_ne_feature(cfeature.LAND),
                       facecolor=land.get("color", "#d9d2c2"),
                       zorder=LAND_FG_ZORDER if land_fg else 0)
    ocean = map_cfg.get("ocean", {})
    if ocean.get("show"):
        ax.add_feature(_ne_feature(cfeature.OCEAN),
                       facecolor=ocean.get("color", "#cfe2f3"), zorder=0)

    # 表示範囲 (extent / 円形・扇形境界) は**レイヤー描画より前**に確定させる。
    # streamplot は呼び出し時点の表示範囲を等間隔グリッドに再サンプルして
    # 流線を計算するため、後から extent を変えると流線の密度・範囲が狂う
    region = panel.get("region")
    proj_name = proj_cfg.get("name", "PlateCarree")
    # 極投影 + 部分経度の領域 → 扇形の境界で表示を領域に連動させる
    sector_mode = bool(use_circular_boundary(proj_cfg) and region
                       and not region_spans_all_longitudes(region))
    extent = None
    if region and not sector_mode and proj_name != "Orthographic":
        extent, ext_clon = extent_args(region, proj_cfg)
        if extent_is_global(extent):
            extent = None
    cv_extent = None
    if extent is None and not sector_mode:
        cv_id = curvilinear_extent_dataset(panel, datasets)
        if cv_id is not None:
            cv_extent = curvilinear_extent(datasets[cv_id], ax.projection)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree(central_longitude=ext_clon))
    elif cv_extent is not None:
        # 2 次元座標格子で領域未指定: 全格子点の投影座標の範囲 (余白なしの長方形)
        ax.set_xlim(cv_extent[0], cv_extent[1])
        ax.set_ylim(cv_extent[2], cv_extent[3])
    elif not sector_mode:
        # 領域未指定 = 全球表示。表示範囲を明示しないと軸の自動スケールが
        # 「実際に描かれた範囲」に縮む: 等値線のみのレイヤー構成で投影の
        # 円盤より狭くなる、maskout や欠損 (SST の陸面 NaN 等) で端の緯度帯が
        # 塗られないと地図ごと消える、など。値ではなく領域指定だけが
        # 表示範囲を決めるようにする
        ax.set_global()

    if use_circular_boundary(proj_cfg):
        if sector_mode:
            # 表示範囲は扇形パス自身の外接範囲から取る (set_extent の矩形だと
            # 扇形の内側を切ってしまい、境界との間に白い隙間が出る)
            path = sector_boundary_path(region, ax.projection)
            ax.set_boundary(path)
            ax.set_xlim(path.vertices[:, 0].min(), path.vertices[:, 0].max())
            ax.set_ylim(path.vertices[:, 1].min(), path.vertices[:, 1].max())
        else:
            ax.set_boundary(circular_boundary_path(), transform=ax.transAxes)

    cbars = []
    for layer in panel["layers"]:
        kind = layer["kind"]
        if kind == "fill":
            cbar = _draw_fill(fig, ax, panel, layer, datasets)
            if cbar is not None:
                cbars.append((cbar, layer["style"].get("colorbar", {})))
        elif kind == "hatch":
            _draw_hatch(ax, panel, layer, datasets)
        elif kind == "contour":
            cbar = _draw_contour(fig, ax, panel, layer, datasets)
            if cbar is not None:
                cbars.append((cbar, layer["style"].get("colorbar", {})))
        elif kind == "vector":
            cbar = _draw_vector(fig, ax, panel, layer, datasets)
            if cbar is not None:
                cbars.append((cbar, layer["style"].get("colorbar", {})))
        elif kind == "stream":
            cbar = _draw_stream(fig, ax, panel, layer, datasets)
            if cbar is not None:
                cbars.append((cbar, layer["style"].get("colorbar", {})))
        elif kind == "map_scatter":
            cbar = _draw_map_scatter(fig, ax, panel, layer, datasets)
            if cbar is not None:
                cbars.append((cbar, layer["style"].get("colorbar", {})))
        elif kind == "track":
            cbar = _draw_track(fig, ax, panel, layer, datasets)
            if cbar is not None:
                cbars.append((cbar, layer["style"]["points"].get("colorbar", {})))
        else:
            raise NotImplementedError(f"Unsupported layer kind: {kind}")

    coast = map_cfg.get("coastlines", {})
    if map_cfg.get("borders"):
        # 国境線は海岸線と同じ太さ・色で描画する
        ax.add_feature(_ne_feature(cfeature.BORDERS),
                       linewidth=coast.get("width", 0.8),
                       edgecolor=coast.get("color", "black"))
    if coast.get("show", True):
        ax.coastlines(resolution=ne_scale or "auto",
                      linewidth=coast.get("width", 0.8), color=coast.get("color", "black"))

    gl_cfg = map_cfg.get("gridlines", {})
    if gl_cfg.get("show", True):
        plan = gridline_plan(gl_cfg)
        gl = None
        # 線用の Gridliner (split 時はラベルを持たない)
        if plan["lines"]:
            gl = ax.gridlines(**gridline_kwargs(
                gl_cfg, proj_cfg,
                draw_labels=plan["labels"] and not plan["split"]))
            if plan["line_lon"]:
                gl.xlocator = mticker.MultipleLocator(plan["line_lon"])
            if plan["line_lat"]:
                gl.ylocator = mticker.MultipleLocator(plan["line_lat"])
        # ラベル用の Gridliner (線が非表示のとき、または間隔が線と異なるとき)
        if plan["labels"] and (plan["split"] or not plan["lines"]):
            gl = ax.gridlines(**gridline_kwargs(gl_cfg, proj_cfg,
                                                draw_labels=True))
            gl.xlines = False
            gl.ylines = False
            # 開始値 (位相) は MultipleLocator の offset (開始値 + n×間隔)。
            # scriptgen と1対1 (offset 未指定時は従来と同じ出力を保つ)
            if plan["label_lon"]:
                if plan["label_lon_start"] is not None:
                    gl.xlocator = mticker.MultipleLocator(
                        plan["label_lon"], offset=plan["label_lon_start"])
                else:
                    gl.xlocator = mticker.MultipleLocator(plan["label_lon"])
            if plan["label_lat"]:
                if plan["label_lat_start"] is not None:
                    gl.ylocator = mticker.MultipleLocator(
                        plan["label_lat"], offset=plan["label_lat_start"])
                else:
                    gl.ylocator = mticker.MultipleLocator(plan["label_lat"])
        # ラベルの体裁 (この時点の gl がラベルを持つ Gridliner)
        if plan["labels"] and gl is not None:
            sides = gl_cfg.get("label_sides", {})
            # geo / inline: 辺に分類されないラベル (極投影の円周沿いの経度ラベル /
            # 図中の緯度ラベル)。gl.geo_labels / gl.inline_labels に対応
            for side in ("left", "right", "top", "bottom", "geo", "inline"):
                if not sides.get(side, True):
                    setattr(gl, f"{side}_labels", False)
            x_style, y_style = gridline_label_styles(gl_cfg)
            if x_style:
                gl.xlabel_style = x_style
            if y_style:
                gl.ylabel_style = y_style
            if gl_cfg.get("label_padding") is not None:
                gl.xpadding = gl_cfg["label_padding"]
                gl.ypadding = gl_cfg["label_padding"]
            label_filters = gridline_label_filters(gl_cfg, proj_cfg, region)
            if label_filters:
                apply_gridline_label_filters(gl, ax, label_filters)

    ticks_cfg = map_cfg.get("ticks", {})
    if ticks_cfg.get("show") and proj_cfg.get("name") == "PlateCarree":
        # PlateCarree でのみ意味のあるティックマーク (他投影では枠が矩形でない)
        # 地理座標で間隔の倍数になる位置 (例: 0°, 30°, 60°...) を生成し、
        # 投影座標に変換する。crs= 引数を渡すと contourf との path 処理が
        # 衝突するため、変換結果を直接渡す
        clon = proj_cfg.get("central_longitude", 0.0)
        lon_int = ticks_cfg.get("lon_interval", 30.0)
        lat_int = ticks_cfg.get("lat_interval", 15.0)
        lon_start = np.ceil((clon - 180.0) / lon_int) * lon_int
        geo_lons = np.arange(lon_start, clon + 180.0, lon_int)
        # Axis.set_ticks は「与えた ticks が全て見えるよう view interval を
        # 広げる」(matplotlib の仕様、実測 3.10) ため、全球分のティック位置を
        # 渡すと region で狭めた extent が全球へ戻る。設定前の表示範囲を保存し、
        # ティック設定後に復元する (範囲外のティックは描かれないだけで無害)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_xticks(((geo_lons - clon + 180.0) % 360.0) - 180.0)
        ax.set_yticks(np.arange(np.ceil(-90.0 / lat_int) * lat_int, 90.0, lat_int))
        ax.set_xticklabels([])  # ラベルは Gridliner 側が描く
        ax.set_yticklabels([])
        sides = ticks_cfg.get("sides", {})
        ax.tick_params(length=ticks_cfg.get("length", 4.0),
                       width=ticks_cfg.get("width", 0.8),
                       direction=ticks_cfg.get("direction", "out"),
                       left=sides.get("left", True),
                       right=sides.get("right", False),
                       top=sides.get("top", False),
                       bottom=sides.get("bottom", True))
        # 短いティック (補助目盛)。長い線と同じ位置は長い線が上に重なって見える
        minor_cfg = ticks_cfg.get("minor") or {}
        if minor_cfg.get("show"):
            m_lon = minor_cfg.get("lon_interval", 10.0)
            m_lat = minor_cfg.get("lat_interval", 5.0)
            m_lon_start = np.ceil((clon - 180.0) / m_lon) * m_lon
            m_geo_lons = np.arange(m_lon_start, clon + 180.0, m_lon)
            ax.set_xticks(((m_geo_lons - clon + 180.0) % 360.0) - 180.0, minor=True)
            ax.set_yticks(np.arange(np.ceil(-90.0 / m_lat) * m_lat, 90.0, m_lat),
                          minor=True)
            ax.tick_params(which="minor",
                           length=minor_cfg.get("length", 2.5),
                           width=minor_cfg.get("width", 0.8),
                           direction=ticks_cfg.get("direction", "out"),
                           left=sides.get("left", True),
                           right=sides.get("right", False),
                           top=sides.get("top", False),
                           bottom=sides.get("bottom", True))
        # set_ticks による view interval の拡張を戻す (上で保存した表示範囲)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for box in map_cfg.get("boxes", []) or []:
        lons, lats = box_polygon(box)
        ax.plot(lons, lats,
                color=box.get("color", "red"),
                linewidth=box.get("linewidth", 1.5),
                linestyle=box.get("linestyle", "solid"),
                transform=ccrs.PlateCarree())

    # 同じ図の鉛直断面パネルの経路 (map.section_paths。scriptgen.map_panel と 1 対 1)。
    # 線は box と同じ zorder 2 (陸を前景に描いても隠れない)、端点の文字は 3
    for item, s_lon, s_lat in section_overlay_items(panel, figure_config, datasets):
        ax.plot(s_lon, s_lat, transform=ccrs.PlateCarree(), zorder=2,
                **section_overlay_kwargs(item))
        for lx, ly, text, ha in section_overlay_label_specs(item, s_lon, s_lat):
            ax.text(lx, ly, text, transform=ccrs.PlateCarree(), ha=ha, va="center",
                    fontsize=float(item.get("label_fontsize", 10)),
                    color=item.get("color") or "black", zorder=3)

    # 図の枠線 (cartopy の "geo" spine。円形・扇形境界にも適用される)
    if map_cfg.get("frame_width") is not None:
        ax.spines["geo"].set_linewidth(float(map_cfg["frame_width"]))

    _finish_panel(ax, panel)
    match_colorbars_to_axes(fig, ax, cbars)
    return ax


def _apply_tick_settings(ax, axis_cfg: dict):
    """目盛の共通設定 (文字サイズ・太さ・表示・位置/間隔) を適用する。

    1次元・2次元・断面図パネル共通。**対数軸 (set_xscale/set_yscale) の後に
    呼ぶこと** — スケール変更は locator/formatter を対数用の既定に初期化する
    ため、後から適用することで明示指定した目盛位置が常に優先される。
    scriptgen._tick_settings_lines と1対1対応。
    """
    # 縦軸の左右入れ替え (swap_y_sides): 目盛・ラベルを右側に表示。
    # 非表示設定 (show_y_ticklabels) より前に適用する (後だと tick_right が
    # 非表示を戻してしまう)。第2軸 (ax2 → 左、dist_1d / line_1d) は呼び出し側が
    # 扱う (line_1d は secondary_axis_cfg で y2 設定を読み替えてここを再利用)
    if axis_cfg.get("swap_y_sides"):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if axis_cfg.get("tick_fontsize"):
        ax.tick_params(labelsize=axis_cfg["tick_fontsize"])
    if axis_cfg.get("tick_width") is not None:
        ax.tick_params(which="both", width=float(axis_cfg["tick_width"]))
    if not axis_cfg.get("show_x_ticklabels", True):
        ax.tick_params(axis="x", labelbottom=False, labeltop=False)
    if not axis_cfg.get("show_y_ticklabels", True):
        ax.tick_params(axis="y", labelleft=False, labelright=False)
    # 目盛文字の回転 (0/キー無し = 回転なしで旧設定不変)
    if axis_cfg.get("x_tick_rotation"):
        ax.tick_params(axis="x",
                       labelrotation=float(axis_cfg["x_tick_rotation"]))
    if axis_cfg.get("y_tick_rotation"):
        ax.tick_params(axis="y",
                       labelrotation=float(axis_cfg["y_tick_rotation"]))
    xtp = axis_cfg.get("x_tick_positions")
    xti = axis_cfg.get("x_tick_interval")
    if xtp:  # 空でないリストならFixedLocator (interval より優先)
        ax.xaxis.set_major_locator(mticker.FixedLocator([float(v) for v in xtp]))
        xtl = axis_cfg.get("x_tick_labels")
        if xtl and len(xtl) == len(xtp):
            ax.xaxis.set_major_formatter(
                mticker.FixedFormatter([str(s) for s in xtl]))
    elif xti is not None:
        ax.xaxis.set_major_locator(
            mticker.NullLocator() if float(xti) == 0
            else mticker.MultipleLocator(float(xti)))
    ytp = axis_cfg.get("y_tick_positions")
    yti = axis_cfg.get("y_tick_interval")
    if ytp:
        ax.yaxis.set_major_locator(mticker.FixedLocator([float(v) for v in ytp]))
        ytl = axis_cfg.get("y_tick_labels")
        if ytl and len(ytl) == len(ytp):
            ax.yaxis.set_major_formatter(
                mticker.FixedFormatter([str(s) for s in ytl]))
    elif yti is not None:
        ax.yaxis.set_major_locator(
            mticker.NullLocator() if float(yti) == 0
            else mticker.MultipleLocator(float(yti)))


def _lon_ew_label(v, pos=None):
    """経度値 (degrees_east) を 180° 中心の東経・西経表記にする。

    例: 120 → "120°E", 180 → "180°", 240 → "120°W", 0/360 → "0°"。
    scriptgen が生成スクリプトに埋め込む同名関数と1対1対応。
    """
    lon = ((float(v) + 180.0) % 360.0) - 180.0
    if lon == -180.0:
        lon = 180.0
    if lon == 0.0:
        return "0°"
    if abs(lon) == 180.0:
        return "180°"
    return f"{abs(lon):g}°{'E' if lon > 0 else 'W'}"


def _apply_axis_grid(ax, axis_cfg: dict, *, legacy_show: bool = False) -> None:
    """axis.grid (x/y 独立の目盛線) を当てる。scriptgen._grid_lines と1対1対応。

    legacy_show=True は section_2d 用: 旧スキーマの {"show": True} を両軸の
    目盛線として扱う。
    """
    grid = axis_cfg.get("grid", {})
    fallback = grid.get("show", False) if legacy_show else False
    show_x = grid.get("show_x", fallback)
    show_y = grid.get("show_y", fallback)
    if show_x or show_y:
        which = "both" if (show_x and show_y) else ("x" if show_x else "y")
        ax.grid(axis=which, color=grid.get("color", "gray"),
                linewidth=grid.get("width", 0.5),
                linestyle=grid.get("linestyle", ":"))


def _apply_minor_ticks(ax, axis_cfg: dict) -> None:
    """補助目盛 (AutoMinorLocator)。scriptgen._minor_tick_lines と1対1対応。"""
    if axis_cfg.get("show_x_minor_ticks"):
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    if axis_cfg.get("show_y_minor_ticks"):
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def _apply_log_scales(ax, axis_cfg: dict, *, scalar_formatter: bool,
                      axes: str = "xy") -> None:
    """log_x / log_y を当てる。scriptgen._log_scale_lines と1対1対応。

    scalar_formatter=True は section_2d / line_1d の流儀 (ScalarFormatter +
    minor NullFormatter で 10^n 表記を避ける)。False は scatter 系の素の
    set_xscale / set_yscale (見た目互換のため揃えない)。
    """
    for a in axes:
        if not axis_cfg.get(f"log_{a}"):
            continue
        getattr(ax, f"set_{a}scale")("log")
        if scalar_formatter:
            axis = getattr(ax, f"{a}axis")
            axis.set_major_formatter(mticker.ScalarFormatter())
            axis.set_minor_formatter(mticker.NullFormatter())


def _set_axis_labels(ax, axis_cfg: dict, xlab, ylab) -> dict:
    """x/y 軸ラベルを体裁 (line_label_kwargs) と回転付きで当てる。
    scriptgen._axis_label_lines と1対1対応。

    第2軸のラベルにも同じ体裁を使うため label_base を返す。xlab / ylab は
    空・None なら描かない (自動ラベルの解決は呼び出し側)。
    """
    label_base = line_label_kwargs(axis_cfg)
    if xlab:
        kw = dict(label_base)
        if axis_cfg.get("x_label_rotation") is not None:
            kw["rotation"] = float(axis_cfg["x_label_rotation"])
        ax.set_xlabel(xlab, **kw)
    if ylab:
        kw = dict(label_base)
        if axis_cfg.get("y_label_rotation") is not None:
            kw["rotation"] = float(axis_cfg["y_label_rotation"])
        ax.set_ylabel(ylab, **kw)
    return label_base


def _apply_invert_and_limits(ax, axis_cfg: dict, *, limits: bool = True) -> None:
    """軸の反転と手動範囲 (x_lim / y_lim)。scriptgen._invert_limit_lines と1対1対応。
    section_2d は範囲を ranges で扱うので limits=False。"""
    if axis_cfg.get("invert_x"):
        ax.invert_xaxis()
    if axis_cfg.get("invert_y"):
        ax.invert_yaxis()
    if limits:
        if axis_cfg.get("x_lim"):
            ax.set_xlim(tuple(axis_cfg["x_lim"]))
        if axis_cfg.get("y_lim"):
            ax.set_ylim(tuple(axis_cfg["y_lim"]))


def _draw_legend(ax, ax2, legend_cfg: dict) -> None:
    """凡例。scriptgen._legend_lines と1対1対応。

    ラベル付きアーティストが無ければ描かない。ax2 (twinx) があれば両軸の
    handles を結合して ax2 上 (最前面) に1つだけ描く。
    """
    if not legend_cfg.get("show", True):
        return
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
        if handles:
            ax2.legend(handles, labels, **legend_kwargs(legend_cfg))
    elif handles:
        ax.legend(**legend_kwargs(legend_cfg))


def _suppress_offset_text(ax, ax2=None) -> None:
    """matplotlib のオフセット表示 (例: 左上の "+2.87e2") を抑制する。
    scriptgen._offset_suppress_lines と1対1対応。

    平均値プロット等で値が大きく変動が小さいときに自動で出る共通オフセットは
    読み解きづらいので 1次元・散布系のプロットでは既定で無効化する。
    ScalarFormatter 以外 (DateFormatter / LogFormatter 等) には set_useOffset
    が無いので hasattr で守る。第2軸があればその y 軸にも当てる。
    """
    off_axes = [ax.yaxis, ax.xaxis] + ([ax2.yaxis] if ax2 is not None else [])
    for _axis in off_axes:
        _fmt = _axis.get_major_formatter()
        if hasattr(_fmt, "set_useOffset"):
            _fmt.set_useOffset(False)


def _finish_panel(ax, panel: dict) -> None:
    """タイトル・時刻ラベル・文字列・記号・パネルラベル (全 plot_type 共通の後処理)。
    scriptgen._finish_panel_lines と1対1対応。"""
    _set_title(ax, panel)
    _draw_time_label(ax, panel)
    _draw_text_annotations(ax, panel)
    _draw_marker_annotations(ax, panel)
    _draw_panel_label(ax, panel)


def _render_section_2d(fig, panel: dict, datasets: dict[str, xr.Dataset],
                       subplot: tuple[int, int, int] = (1, 1, 1)):
    """鉛直断面・時間断面など、地図を使わない汎用2D断面を描画する。"""
    ax = fig.add_subplot(*subplot)
    _apply_box_aspect(ax, panel)

    for layer in panel["layers"]:
        kind = layer["kind"]
        if kind == "fill":
            _draw_fill_section(fig, ax, panel, layer, datasets)
        elif kind == "hatch":
            _draw_hatch_section(ax, panel, layer, datasets)
        elif kind == "contour":
            _draw_contour_section(fig, ax, panel, layer, datasets)
        elif kind == "vector":
            _draw_vector_section(fig, ax, panel, layer, datasets)
        elif kind == "stream":
            _draw_stream_section(fig, ax, panel, layer, datasets)
        else:
            raise NotImplementedError(f"Unsupported layer kind: {kind}")

    axis_cfg = panel.get("axis", {})
    _apply_axis_grid(ax, axis_cfg, legacy_show=True)
    _apply_minor_ticks(ax, axis_cfg)
    _set_axis_labels(ax, axis_cfg, axis_cfg.get("x_label"), axis_cfg.get("y_label"))
    _apply_log_scales(ax, axis_cfg, scalar_formatter=True, axes="y")
    # 目盛設定は対数軸の後に適用 (明示指定した目盛位置を対数の既定目盛より優先)
    _apply_tick_settings(ax, axis_cfg)
    _apply_invert_and_limits(ax, axis_cfg, limits=False)

    # 地形マスク: 軸の範囲 (対数・反転を含む) が決まった後に、地面より下を塗った多角形を
    # 全レイヤーの上に重ねる (scriptgen.section_panel と 1 対 1)
    terrain = section_terrain_ground(panel, datasets)
    if terrain is not None:
        tx, tground, tpressure = terrain
        draw_section_terrain(ax, tx, tground, tpressure,
                             (panel.get("terrain") or {}).get("color") or "#808080",
                             TERRAIN_ZORDER)

    # 経度軸の東経・西経表記。カスタム目盛ラベル (FixedFormatter) 指定時は
    # そちらを優先する
    if axis_cfg.get("x_lon_east_west") and not axis_cfg.get("x_tick_labels"):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_lon_ew_label))
    # 目盛に経緯度を併記 (格子線断面の格子番号軸・大円断面の距離軸)。求められない
    # 断面では何もしない。scriptgen.section_panel と 1 対 1
    if axis_cfg.get("x_lonlat_ticks") and not axis_cfg.get("x_tick_labels"):
        ref = section_x_lonlat(panel, datasets)
        if ref is not None:
            if not (axis_cfg.get("x_tick_positions") or axis_cfg.get("x_tick_interval")):
                # 3 段のラベルは幅を取るので、軸の幅 (インチ) に応じて目盛を 3〜6 個に抑える
                # (1 インチに 1 個の目安。scriptgen も同じ式を出す)
                _w_in = ax.get_position().width * fig.get_figwidth()
                ax.xaxis.set_major_locator(
                    mticker.MaxNLocator(nbins=int(np.clip(_w_in / 1.0, 3, 6))))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, pos=None, _r=ref: lonlat_tick_label(v, *_r)))

    fmt = axis_cfg.get("time_axis_format")
    if fmt:
        time_axis = _section_time_axis(panel, datasets)
        if time_axis:
            getattr(ax, f"{time_axis}axis").set_major_formatter(mdates.DateFormatter(fmt))

    _apply_frame_background(ax, panel)
    _finish_panel(ax, panel)
    return ax


def _section_time_axis(panel: dict, datasets: dict) -> str | None:
    """section プロットで time 次元が x 軸か y 軸か (None なら時間軸ではない) を返す。"""
    if not panel.get("layers"):
        return None
    ds = datasets[panel["layers"][0]["dataset_id"]]
    time_dim = detect_coord_roles(ds).get("time")
    if time_dim is None:
        return None
    if panel.get("x_dim") == time_dim:
        return "x"
    if panel.get("y_dim") == time_dim:
        return "y"
    return None


def _section_data(panel: dict, layer: dict, variable: str, datasets: dict,
                  dataset_id: str | None = None, record: list | None = None):
    """断面図用にレイヤーの変数を切り出す。dataset_id で参照先を上書きできる
    (ベクトルの y 成分が別ファイルのとき。既定 = layer["dataset_id"])。
    record は apply_averages の記録用 (layer_averaging_stats 参照)。"""
    ds = datasets[dataset_id or layer["dataset_id"]]
    selection = {**(panel.get("selection") or {}), **(layer.get("selection") or {})}
    return select_section_data(ds, variable, selection,
                               panel.get("ranges"), panel["x_dim"], panel["y_dim"],
                               averages=layer.get("averages"), record=record,
                               path=panel.get("section_path"))


def _maskout_mask_data(panel: dict, layer: dict, datasets: dict, section: bool):
    """maskout の「別変数」を描画変数と同じ選択で切り出して返す (未設定なら None)。

    マスク変数は描画軸の2次元を持ち、かつ描画変数に無い次元を持たないこと
    (描画変数と同じ selection で固定しきれる範囲)。fill / hatch / contour の
    水平面図 (section=False) と断面図 (section=True) で共用する。
    """
    cfg = maskout_var_config(layer["style"])
    if cfg is None:
        return None
    mvar = cfg[0]
    ds = datasets[layer["dataset_id"]]
    if mvar not in ds.data_vars:
        raise RenderError("maskout_var_missing", var=mvar)
    mdims = set(ds[mvar].dims)
    extra = sorted(mdims - set(ds[layer["variable"]].dims))
    if extra:
        raise RenderError("maskout_var_extra_dims", var=mvar, dims=extra,
                          target=layer["variable"])
    if section:
        need = {panel["x_dim"], panel["y_dim"]}
    else:
        # 水平面の dim (1 次元格子 = lat/lon、curvilinear = y/x)
        need = set(horizontal_dims(ds, detect_coord_roles(ds)))
    missing = sorted(need - mdims)
    if missing:
        raise RenderError("maskout_var_missing_dims", var=mvar, dims=missing)
    if section:
        return _section_data(panel, layer, mvar, datasets)
    return _layer_data(panel, layer, mvar, datasets, cyclic=True)[0]


def _transform_and_mask(da, panel: dict, layer: dict, datasets: dict, *, section: bool):
    """値変換 → maskout (fill / hatch / contour の地図・断面共通の前処理)。
    scriptgen._transform_and_mask_lines と1対1対応。"""
    style = layer["style"]
    da = apply_value_transform(da, style)
    return apply_maskout(da, style,
                         _maskout_mask_data(panel, layer, datasets, section=section))


def _fill_artist(ax, da, xname: str, yname: str, style: dict, **extra_kw):
    """塗りつぶしの描画呼び出し (地図・断面共通)。scriptgen._fill_call_lines と1対1対応。

    method="pcolormesh" / zero_white / 通常の contourf の3分岐。extra_kw は地図の
    transform=ccrs.PlateCarree()。
    """
    alpha = fill_alpha(style)
    if style.get("method", "contourf") == "pcolormesh":
        cmap, norm = _fill_pcolormesh_kwargs(style, da)
        return ax.pcolormesh(
            da[xname], da[yname], da,
            cmap=cmap, norm=norm, shading="auto", alpha=alpha, **extra_kw,
        )
    if style.get("zero_white"):
        # zero_white はレベルを具体化し 0 を含むビンを白にした cmap/norm を渡す
        cmap, norm = _fill_pcolormesh_kwargs(style, da, contourf_levels=True)
        return ax.contourf(
            da[xname], da[yname], da,
            cmap=cmap, norm=norm,
            levels=np.asarray(norm.boundaries, dtype=float),
            extend=style.get("extend", "both"),
            alpha=alpha, **extra_kw,
        )
    return ax.contourf(
        da[xname], da[yname], da,
        cmap=resolve_cmap(style.get("cmap", "viridis"), style.get("reverse_cmap", False)),
        levels=fill_levels(style),
        extend=style.get("extend", "both"),
        alpha=alpha, **extra_kw,
    )


def _hatch_artist(ax, da, xname: str, yname: str, style: dict, **extra_kw):
    """ハッチ (colors='none' の contourf) の描画呼び出し (地図・断面共通)。
    scriptgen._hatch_call_lines と1対1対応。線の太さと色は hatch_rc_params。"""
    with plt.rc_context(hatch_rc_params(style)):
        return ax.contourf(
            da[xname], da[yname], da,
            levels=style["levels"],
            colors="none",
            hatches=[hatch_pattern(style)],
            **extra_kw,
        )


def _contour_artist(fig, ax, da, xname: str, yname: str, style: dict,
                    label_zorder=None, **extra_kw):
    """等値線の描画 + 線種・強調・ラベル・カラーバー (地図・断面共通)。
    scriptgen._contour_call_lines と1対1対応。カラーバーを返す (無ければ None)。
    label_zorder は地形マスクのある断面でラベル (既定 zorder 4) を地面の多角形の下に
    置くための値 (None = 既定のまま)。"""
    levels = contour_levels(style)
    kwargs = dict(contour_kwargs(style))
    if levels is not None:
        kwargs["levels"] = levels
    cs = ax.contour(da[xname], da[yname], da, **kwargs, **extra_kw)
    apply_contour_linestyles(cs, style)
    apply_contour_emphasis(cs, style)
    labels_cfg = style.get("labels", {})
    if labels_cfg.get("show"):
        texts = ax.clabel(cs, contour_label_levels(style, cs.levels),
                          fontsize=labels_cfg.get("fontsize", 8),
                          fmt=labels_cfg.get("fmt", "%g"))
        if label_zorder is not None:
            for _t in texts:
                _t.set_zorder(label_zorder)
    # colorbar キーは新スキーマのみが持つ (旧設定は従来どおりカラーバーなし)
    if style.get("use_cmap") and "colorbar" in style:
        return _add_colorbar(fig, ax, cs, style.get("colorbar", {}))
    return None


def _prepare_vector_uv(u, v, style: dict, xname: str, yname: str):
    """ベクトル成分の値変換 → 間引き → 小さい矢印のマスク (地図・断面共通)。
    scriptgen._vector_prep_lines と1対1対応。"""
    u = apply_value_transform(u, style)
    v = apply_value_transform(v, style)
    sx, sy = vector_strides(style)
    if sx > 1 or sy > 1:
        step = {yname: slice(None, None, sy), xname: slice(None, None, sx)}
        u = u.isel(step)
        v = v.isel(step)
    # 大きさが閾値以下の矢印を NaN 化して描かない (quiver は NaN をスキップ)
    if style.get("mask_below") is not None:
        _small = np.hypot(u.values, v.values) <= float(style["mask_below"])
        u = u.where(~_small)
        v = v.where(~_small)
    return u, v


def _vector_color_args(style: dict, u, v):
    """use_cmap 時はベクトルの大きさ |V| を C 配列として渡して色付けする。
    (quiver の追加位置引数, norm の kwargs) を返す。scriptgen._quiver_lines と1対1対応。"""
    if not style.get("use_cmap"):
        return [], {}
    mag = np.hypot(u.values, v.values)
    return [mag], discrete_color_kwargs(style, np.nanmin(mag), np.nanmax(mag))


def _finish_vector(fig, ax, q, style: dict, norm_kw: dict, *, key_zorder=None):
    """quiver 後の色範囲・カラーバー・ベクトルキー (地図・断面共通)。
    scriptgen._finish_vector_lines と1対1対応。カラーバーを返す (無ければ None)。
    key_zorder は陸を前景に描く地図で基準ベクトル (既定 1.1) を陸の上に出すための zorder
    (None = 既定)。"""
    cbar = None
    if style.get("use_cmap"):
        if not norm_kw and (style.get("vmin") is not None
                            or style.get("vmax") is not None):
            q.set_clim(style.get("vmin"), style.get("vmax"))
        cbar = _add_colorbar(fig, ax, q, style.get("colorbar", {}),
                             extend=continuous_cbar_extend(style))
    key = style.get("key", {})
    if key.get("show"):
        _draw_quiverkey(ax, q, key, zorder=key_zorder)
    return cbar


def _draw_fill_section(fig, ax, panel: dict, layer: dict, datasets: dict):
    da = _section_data(panel, layer, layer["variable"], datasets)
    da = _transform_and_mask(da, panel, layer, datasets, section=True)
    cf = _fill_artist(ax, da, panel["x_dim"], panel["y_dim"], layer["style"])
    _record_shared_mappable(ax, cf)
    _add_colorbar(fig, ax, cf, layer["style"].get("colorbar", {}))


def _draw_hatch_section(ax, panel: dict, layer: dict, datasets: dict):
    da = _section_data(panel, layer, layer["variable"], datasets)
    da = _transform_and_mask(da, panel, layer, datasets, section=True)
    _hatch_artist(ax, da, panel["x_dim"], panel["y_dim"], layer["style"])


def _draw_contour_section(fig, ax, panel: dict, layer: dict, datasets: dict):
    da = _section_data(panel, layer, layer["variable"], datasets)
    da = _transform_and_mask(da, panel, layer, datasets, section=True)
    # 地形マスクのある断面では、地下の等値線ラベルが地面の上に浮かないよう
    # ラベルを多角形の下に置く
    _contour_artist(fig, ax, da, panel["x_dim"], panel["y_dim"], layer["style"],
                    label_zorder=(TERRAIN_ZORDER - 0.1 if terrain_enabled(panel) else None))


def _draw_vector_section(fig, ax, panel: dict, layer: dict, datasets: dict):
    u = _section_data(panel, layer, layer["u_variable"], datasets)
    v = _section_data(panel, layer, layer["v_variable"], datasets,
                      dataset_id=vector_v_dataset_id(layer))
    check_vector_shapes(u, v, layer)
    style = layer["style"]
    u, v = _prepare_vector_uv(u, v, style, panel["x_dim"], panel["y_dim"])
    args, norm_kw = _vector_color_args(style, u, v)
    q = ax.quiver(
        u[panel["x_dim"]].values, u[panel["y_dim"]].values, u.values, v.values,
        *args,
        **{**vector_kwargs(style), **norm_kw},
    )
    # 地形マスクの多角形の上に基準ベクトルを出す (地図の「陸をデータの上に描く」と同じ扱い)
    _finish_vector(fig, ax, q, style, norm_kw,
                   key_zorder=(TERRAIN_ZORDER + 0.1 if terrain_enabled(panel) else None))


def twin_align_value(axis_cfg: dict):
    """line_1d の「第2軸との値揃え」の対象値を返す (無効なら None。render/scriptgen 共用)。

    axis.y2_align_value が数値で、どちらの縦軸も対数でないときだけ有効
    (対数軸では高さの割合が値に比例せず、0 を置けない)。
    """
    v = axis_cfg.get("y2_align_value")
    if v is None or axis_cfg.get("log_y") or axis_cfg.get("log_y2"):
        return None
    return float(v)


def align_twin_ylim(lim1, lim2, value):
    """Return (lim1, lim2) widened so that `value` sits at the same fractional
    height on both y axes (twinx alignment, e.g. a shared zero line).

    Ranges are never shrunk: each range is first widened to contain `value`,
    then the range whose fraction differs is extended on one side only. The
    primary range (lim1) is kept when `value` lies strictly inside it;
    otherwise the secondary's fraction is used, and both are extended when
    `value` sits at an edge of both. Inverted ranges (bottom > top) keep their
    orientation. Degenerate or non-finite inputs are returned unchanged.
    """
    def _split(lim):
        b, t = float(lim[0]), float(lim[1])
        return min(b, t), max(b, t), b > t

    v = float(value)
    b1, t1, inv1 = _split(lim1)
    b2, t2, inv2 = _split(lim2)
    if not (t1 > b1 and t2 > b2 and v == v and abs(v) != float("inf")):
        return tuple(lim1), tuple(lim2)
    b1, t1 = min(b1, v), max(t1, v)
    b2, t2 = min(b2, v), max(t2, v)
    f1 = (v - b1) / (t1 - b1)
    f2 = (v - b2) / (t2 - b2)
    if abs(f1 - f2) > 1e-12:
        if 0.0 < f1 < 1.0:
            f = f1
        elif 0.0 < f2 < 1.0:
            f = f2
        else:
            f = 0.5

        def _fit(b, t):
            cur = (v - b) / (t - b)
            if cur < f:      # value too low: extend the bottom
                b = v - f * (t - v) / (1.0 - f)
            elif cur > f:    # value too high: extend the top
                t = v + (1.0 - f) * (v - b) / f
            return b, t

        b1, t1 = _fit(b1, t1)
        b2, t2 = _fit(b2, t2)
    return ((t1, b1) if inv1 else (b1, t1)), ((t2, b2) if inv2 else (b2, t2))


def _style_secondary_axis(ax2, axis_cfg: dict, label_base: dict) -> None:
    """第2軸 (twinx) の体裁を、第1軸と同じ関数に y2 設定を読み替えて当てる (line_1d)。

    順序は第1軸 (_render_line_1d) と同じ: 目盛線 → 補助目盛 → 軸ラベル → 対数 →
    左右入れ替え (第2軸 → 左) → 目盛設定 → 余白除去 → 反転 → 範囲。
    scriptgen._secondary_axis_lines と1対1対応。
    """
    cfg2 = secondary_axis_cfg(axis_cfg)
    grid = cfg2["grid"]
    if grid.get("show_y"):
        ax2.grid(axis="y", color=grid.get("color", "gray"),
                 linewidth=grid.get("width", 0.5),
                 linestyle=grid.get("linestyle", ":"))
    if cfg2.get("show_y_minor_ticks"):
        ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    if cfg2.get("y_label"):
        kw = dict(label_base)
        if cfg2.get("y_label_rotation") is not None:
            kw["rotation"] = float(cfg2["y_label_rotation"])
        ax2.set_ylabel(cfg2["y_label"], **kw)
    if cfg2.get("log_y"):
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
    # 縦軸の左右入れ替え: 第1軸→右は _apply_tick_settings が行う。第2軸→左は
    # ここで扱う。目盛文字の非表示設定より前 (第1軸と同じ理由)
    if axis_cfg.get("swap_y_sides"):
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
    # 目盛設定は対数軸の後 (明示指定した目盛位置を優先、第1軸と同順)
    _apply_tick_settings(ax2, cfg2)
    if cfg2.get("tight_y") and not cfg2.get("y_lim"):
        ax2.autoscale(axis="y", tight=True)
    if cfg2.get("invert_y"):
        ax2.invert_yaxis()
    if cfg2.get("y_lim"):
        ax2.set_ylim(tuple(cfg2["y_lim"]))


def _render_line_1d(fig, panel: dict, datasets: dict[str, xr.Dataset],
                    subplot: tuple[int, int, int] = (1, 1, 1)):
    """1次元プロット (任意の dim を x 軸にしたライン重ね描き)。

    style.secondary_y=True のレイヤーは第2軸 (右の縦軸, twinx) に描く。
    第2軸の体裁 (ラベル・範囲・対数・反転・目盛位置・補助目盛・目盛線・
    目盛文字の表示) は y2 系の設定を第1軸と同じ関数で当てる
    (secondary_axis_cfg / _style_secondary_axis)。
    """
    ax = fig.add_subplot(*subplot)
    _apply_box_aspect(ax, panel)
    ax2 = None
    if line_uses_secondary_axis(panel):
        ax2 = ax.twinx()
        # set_box_aspect は描画時に axes 位置を動かすため、twinx した両方に
        # 同じ値を当てないと左右の枠がずれる (dist_1d と同じ)
        _apply_box_aspect(ax2, panel)

    # 棒グラフは dodge/stack でレイヤー間の協調が必要なので、先にデータを集めて
    # 共通パラメータ (width / offset / bottom) を計算しておく
    bar_layer_indices = [i for i, lyr in enumerate(panel["layers"])
                         if lyr["kind"] == "bar"]
    bar_mode = panel.get("axis", {}).get("bar_mode", "overlap")
    bar_data = []
    for i in bar_layer_indices:
        layer = panel["layers"][i]
        da_bar = _line_layer_data(panel, layer, layer["variable"], datasets)
        da_bar = apply_value_transform(da_bar, layer["style"])
        bar_data.append((da_bar[panel["x_dim"]].values, da_bar.values, layer))
    bar_params = _compute_bar_params(
        bar_data, bar_mode,
        dodge_gap=panel.get("axis", {}).get("bar_dodge_gap", 0.05))
    bar_param_by_index = dict(zip(bar_layer_indices, bar_params))
    bar_data_by_index = dict(zip(bar_layer_indices, bar_data))

    for i, layer in enumerate(panel["layers"]):
        kind = layer["kind"]
        # 描画先: secondary_y のレイヤーは第2軸 (棒の dodge/stack の幅・下端は
        # 軸に関係なく全 bar レイヤーで計算済み)
        target = (ax2 if (ax2 is not None
                          and layer.get("style", {}).get("secondary_y")) else ax)
        if kind == "line":
            _draw_line_layer(target, panel, layer, datasets)
        elif kind == "line_bundle":
            _draw_line_bundle_layer(target, panel, layer, datasets)
        elif kind == "fill_between":
            _draw_fill_between_layer(target, panel, layer, datasets)
        elif kind == "stackplot":
            _draw_stackplot_layer(target, panel, layer, datasets)
        elif kind == "bar":
            x_vals, y_vals, _ = bar_data_by_index[i]
            _draw_bar_layer(target, panel, layer, datasets,
                             x_vals, y_vals, bar_param_by_index[i])
        else:
            raise NotImplementedError(f"Unsupported layer kind: {kind}")

    axis_cfg = panel.get("axis", {})
    _apply_axis_grid(ax, axis_cfg)
    _apply_minor_ticks(ax, axis_cfg)
    label_base = _set_axis_labels(ax, axis_cfg,
                                  axis_cfg.get("x_label"), axis_cfg.get("y_label"))
    _apply_log_scales(ax, axis_cfg, scalar_formatter=True)
    # 目盛設定は対数軸の後に適用 (明示指定した目盛位置を対数の既定目盛より優先)
    _apply_tick_settings(ax, axis_cfg)
    # 余白除去は invert / x_lim より前に適用 (autoscale を再計算するため)
    if axis_cfg.get("tight_x") and not axis_cfg.get("x_lim"):
        ax.autoscale(axis="x", tight=True)
    if axis_cfg.get("tight_y") and not axis_cfg.get("y_lim"):
        ax.autoscale(axis="y", tight=True)
    _apply_invert_and_limits(ax, axis_cfg)
    # 第2軸 (twinx) の体裁: y2 設定を第1軸と同じ関数で当てる
    if ax2 is not None:
        _style_secondary_axis(ax2, axis_cfg, label_base)
        v_align = twin_align_value(axis_cfg)
        if v_align is not None:
            # 両軸の範囲 (自動範囲・余白除去・反転・手動範囲を当てた後) から
            # 指定値の高さを揃える (縮めず広げるだけ。scriptgen と1対1)
            lim1, lim2 = align_twin_ylim(ax.get_ylim(), ax2.get_ylim(), v_align)
            ax.set_ylim(lim1)
            ax2.set_ylim(lim2)

    # 時間軸の書式 (x_dim が time のとき)
    fmt = axis_cfg.get("time_axis_format")
    if fmt and panel.get("layers") and panel.get("x_dim"):
        ds = datasets[panel["layers"][0]["dataset_id"]]
        time_dim = detect_coord_roles(ds).get("time")
        if time_dim and panel["x_dim"] == time_dim:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

    _apply_frame_background(ax, panel)
    if ax2 is not None:
        # 図枠は ax2 にも当てる (ax2 の既定 spine が上に重なって図枠設定を
        # 隠してしまうため。背景は base ax 側のみ)
        _apply_frame(ax2, panel.get("frame") or {})
    _draw_legend(ax, ax2, panel.get("legend", {}))
    _suppress_offset_text(ax, ax2)
    _finish_panel(ax, panel)
    return ax


def _fit_error_to_mask(err, n: int, mask):
    """誤差配列を本体の長さ n に合わせて (不足は NaN 埋め) mask を適用する。

    scriptgen の emit と1対1 (同じ式)。
    """
    if err is None:
        return None
    return np.concatenate([err[:n],
                           np.full(max(0, n - len(err)), np.nan)])[mask]


def errorbar_kwargs(eb: dict) -> dict:
    """散布図/バブルのエラーバー体裁 (ax.errorbar の kwargs、render/scriptgen 共用)。

    fmt="none" で点は描かず誤差線だけを重ねる (点本体は scatter が描く)。
    """
    return {
        "fmt": "none",
        "ecolor": eb.get("color", "#000000"),
        "elinewidth": float(eb.get("linewidth", 1.0)),
        "capsize": float(eb.get("capsize", 3.0)),
    }


def _scatter_take_values(ds, varname: str, fixed, drawing_dim, drawing_range,
                         scale: float, offset: float = 0.0, *, err: bool = False):
    """散布図・バブル図の1変数を切り出して 1 次元 float 配列にする。
    scriptgen._scatter_take_lines と1対1対応。

    fixed で drawing_dim 以外の次元を1点に固定し、drawing_range で drawing_dim を
    範囲スライスしてから ravel する。err=True は誤差配列 (|err|。値の変換は掛けない —
    誤差は図に表示する単位で用意する前提。棒グラフのエラー量と同じ規則、2026-09-29)。
    """
    da = ds[varname]
    sel = {dim: val for dim, val in (fixed or {}).items()
            if dim in da.dims}
    if sel:
        da = da.sel(sel)
    if drawing_range and drawing_dim and drawing_dim in da.dims:
        da = da.sel({drawing_dim: coord_slice(
            ds[drawing_dim], drawing_range[0], drawing_range[1])})
    vals = np.asarray(da.values, dtype=float)
    if err:
        return np.abs(vals).ravel()
    if scale != 1.0 or offset != 0.0:
        vals = vals * scale + offset
    return vals.ravel()


def _scatter_error_arrays(ds, layer: dict, style: dict, drawing_dim, drawing_range):
    """エラーバーの誤差配列 (xerr, yerr)。未指定・変数が無い軸は None。
    scriptgen._scatter_error_lines と1対1対応 (本体と同じ固定・範囲、値の変換なし)。"""
    eb = style.get("errorbar") or {}
    out = []
    for var_key, fixed_key in (("x_variable", "x_fixed"), ("y_variable", "y_fixed")):
        var = eb.get(var_key)
        if var and var in ds:
            out.append(_scatter_take_values(
                ds, var, layer.get(fixed_key), drawing_dim, drawing_range,
                1.0, err=True))
        else:
            out.append(None)
    return out[0], out[1]


def _scatter_xy_arrays(panel: dict, layer: dict, datasets: dict,
                       with_errors: bool = False):
    """散布図の (x, y) を抽出し、NaN を除いた 1D 配列のペアで返す。

    panel.x_variable / panel.y_variable から x/y を取り、
    layer.x_fixed / layer.y_fixed で drawing_dim 以外の dim を 1 点に固定、
    layer.drawing_range で drawing_dim を範囲スライスしてから ravel する。
    with_errors=True では (x, y, xerr, yerr) を返す — 誤差は
    style.errorbar.x/y_variable を本体と同じ固定・範囲で切り出し、
    値の変換は**掛けずに** |err| として x/y と同じ mask を当てる
    (未指定の軸は None)。
    """
    ds = datasets[layer["dataset_id"]]
    x_var = panel.get("x_variable")
    y_var = panel.get("y_variable")
    if not x_var or not y_var:
        empty = np.array([])
        return (empty, empty, None, None) if with_errors else (empty, empty)
    drawing_dim = layer.get("drawing_dim")
    drawing_range = layer.get("drawing_range")
    style = layer["style"]
    x = _scatter_take_values(ds, x_var, layer.get("x_fixed"), drawing_dim, drawing_range,
                             float(style.get("x_value_scale", 1.0)),
                             float(style.get("x_value_offset", 0.0)))
    y = _scatter_take_values(ds, y_var, layer.get("y_fixed"), drawing_dim, drawing_range,
                             float(style.get("y_value_scale", 1.0)),
                             float(style.get("y_value_offset", 0.0)))
    xe = ye = None
    if with_errors:
        xe, ye = _scatter_error_arrays(ds, layer, style, drawing_dim, drawing_range)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    mask = ~(np.isnan(x) | np.isnan(y))
    if with_errors:
        return (x[mask], y[mask],
                _fit_error_to_mask(xe, n, mask),
                _fit_error_to_mask(ye, n, mask))
    return x[mask], y[mask]


def scatter_marker_kwargs(style: dict) -> dict:
    """散布図 (kind="scatter") の ax.scatter kwargs (render/scriptgen 共用)。"""
    kwargs = {
        "alpha": float(style.get("alpha", 0.7)),
        "marker": style.get("marker", "o"),
        "s": float(style.get("size", 20.0)),
        "c": style.get("color", "#1f77b4"),
    }
    if float(style.get("edge_linewidth", 0.0)) > 0:
        kwargs["edgecolors"] = style.get("edge_color", "#000000")
        kwargs["linewidths"] = float(style["edge_linewidth"])
    if style.get("label"):
        kwargs["label"] = style["label"]
    return kwargs


def _draw_scatter_layer(ax, panel: dict, layer: dict, datasets: dict):
    x_arr, y_arr, xe, ye = _scatter_xy_arrays(panel, layer, datasets,
                                              with_errors=True)
    style = layer["style"]
    # エラーバーは点より先に描いて点を上に (render/scriptgen 同順)
    if xe is not None or ye is not None:
        ax.errorbar(x_arr, y_arr, xerr=xe, yerr=ye,
                    **errorbar_kwargs(style.get("errorbar") or {}))
    ax.scatter(x_arr, y_arr, **scatter_marker_kwargs(style))


def heatmap_base_kwargs(style: dict) -> dict:
    """imshow の norm 以外の kwargs (cmap / aspect / origin) を解決する。

    render/scriptgen 共用。norm (離散化) と vmin/vmax は呼び出し側で付ける。
    """
    return {
        "cmap": resolve_cmap(style.get("cmap", "viridis"),
                             style.get("reverse_cmap", False)),
        "aspect": style.get("aspect", "auto"),
        "origin": style.get("origin", "upper"),
        "interpolation": "nearest",
    }


def heatmap_tick_labels(values) -> list[str]:
    """座標値 (or インデックス) を目盛りラベル文字列にする (render/scriptgen 共用)。"""
    out = []
    for v in np.asarray(values).reshape(-1):
        if isinstance(v, (float, np.floating)):
            out.append(f"{float(v):g}")
        elif isinstance(v, (int, np.integer)):
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return out


def heatmap_dim_labels(da, dim: str) -> list[str]:
    """heatmap の1軸ぶんの目盛りラベル。座標変数があればその値、無ければ 0..n-1。"""
    if dim in da.coords:
        return heatmap_tick_labels(da[dim].values)
    return [str(i) for i in range(int(da.sizes[dim]))]


def heatmap_data(panel: dict, datasets: dict):
    """heatmap 用の2次元 DataArray を返す ((y_dim, x_dim) 順・値変換/maskout 適用済み)。

    render/scriptgen で同じ手順 (選択→transpose→値変換→maskout) を踏むための
    共有ロジック。maskout されたセルは NaN になり、色・注記とも描かれない。
    """
    ds = datasets[panel["dataset_id"]]
    da = ds[panel["variable"]]
    sel = {d: v for d, v in (panel.get("selection") or {}).items() if d in da.dims}
    if sel:
        da = da.sel(sel)
    da = da.transpose(panel["y_dim"], panel["x_dim"])
    da = apply_value_transform(da, panel["style"])
    return apply_maskout(da, panel["style"])


def _render_heatmap(fig, panel: dict, datasets: dict[str, xr.Dataset],
                    subplot: tuple[int, int, int] = (1, 1, 1)):
    """categorical heatmap (imshow) のパネルを描画する。"""
    ax = fig.add_subplot(*subplot)
    _apply_box_aspect(ax, panel)
    style = panel["style"]
    da = heatmap_data(panel, datasets)
    m = np.asarray(da.values, dtype=float)

    kwargs = heatmap_base_kwargs(style)
    dk = discrete_color_kwargs(style, np.nanmin(m), np.nanmax(m))
    if dk:
        kwargs.update(dk)
    else:
        if style.get("vmin") is not None:
            kwargs["vmin"] = style["vmin"]
        if style.get("vmax") is not None:
            kwargs["vmax"] = style["vmax"]
    im = ax.imshow(m, **kwargs)

    # カテゴリ目盛り (セル中心に配置)。目盛位置を手動指定 (等間隔/直接指定) した
    # 軸はカテゴリラベルを設定せず、セル index の数値軸として
    # _apply_tick_settings に任せる (カテゴリの FixedFormatter を残すと
    # 位置とラベルの対応がずれるため)
    x_labels = heatmap_dim_labels(da, panel["x_dim"])
    y_labels = heatmap_dim_labels(da, panel["y_dim"])
    axis_cfg = panel.get("axis", {})
    if (axis_cfg.get("x_tick_interval") is None
            and axis_cfg.get("x_tick_positions") is None):
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels,
                           rotation=float(panel.get("xtick_rotation", 0.0)))
    if (axis_cfg.get("y_tick_interval") is None
            and axis_cfg.get("y_tick_positions") is None):
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels,
                           rotation=float(panel.get("ytick_rotation", 0.0)))
    _apply_tick_settings(ax, axis_cfg)

    if style.get("colorbar", {}).get("show", True):
        _add_colorbar(fig, ax, im, style.get("colorbar", {}))

    # セル注記 (既定 off・固定色)
    ann = style.get("annotate", {})
    if ann.get("show"):
        fmt = ann.get("fmt", "%.2g")
        fs = ann.get("fontsize", 8)
        color = ann.get("color", "#000000")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, fmt % v, ha="center", va="center",
                        color=color, fontsize=fs)

    # 軸ラベル (scatter と同じ体裁)
    _set_axis_labels(ax, axis_cfg, axis_cfg.get("x_label"), axis_cfg.get("y_label"))
    _apply_invert_and_limits(ax, axis_cfg)

    _apply_frame_background(ax, panel)
    _finish_panel(ax, panel)
    return ax


def dist_values(layer: dict, datasets: dict):
    """集計系レイヤー (hist 等) の対象値を 1次元 float 配列で返す (NaN 除去済み)。

    selection で余分な次元を固定 → agg_dim の範囲制限 (agg_range) → 値変換 →
    ravel → 有限値のみ。render/scriptgen 共通ロジック (ルール4)。
    """
    ds = datasets[layer["dataset_id"]]
    da = ds[layer["variable"]]
    sel = {d: v for d, v in (layer.get("selection") or {}).items()
           if d in da.dims}
    if sel:
        da = da.sel(sel)
    rng = layer.get("agg_range")
    dim = layer.get("agg_dim")
    if rng is not None and dim and dim in da.dims:
        sl = coord_slice(ds[dim], rng[0], rng[1])
        da = da.sel({dim: sl})
    da = apply_value_transform(da, layer["style"])
    vals = np.asarray(da.values, dtype=float).ravel()
    return vals[np.isfinite(vals)]


def hist_kwargs(style: dict) -> dict:
    """ax.hist の kwargs を解決する (render/scriptgen 共用)。

    bins が list のときは昇順・重複なしを検証し、range は使わない
    (matplotlib も境界指定時は range を無視する)。orientation は横向き
    ("horizontal") のときだけ、rwidth (ビン幅に対する棒の幅の比) は値があり
    histtype が bar のときだけ渡す (既定・旧設定の生成スクリプトを変えない)。
    """
    bins = style.get("bins", 20)
    if isinstance(bins, (list, tuple)):
        vals = [float(v) for v in bins]
        if len(vals) < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
            raise ValueError(
                "ヒストグラムのビン境界 (style.bins) は昇順・重複なしの2個以上で"
                f"指定してください: {list(bins)!r}")
        bins = vals
    else:
        bins = int(bins)
    kw = {
        "bins": bins,
        "density": bool(style.get("density", False)),
        "cumulative": bool(style.get("cumulative", False)),
        "histtype": style.get("histtype", "bar"),
        "alpha": float(style.get("alpha", 0.7)),
    }
    if style.get("orientation") == "horizontal":
        kw["orientation"] = "horizontal"
    if style.get("rwidth") is not None and kw["histtype"] == "bar":
        kw["rwidth"] = float(style["rwidth"])
    if style.get("range") is not None and not isinstance(bins, list):
        kw["range"] = (float(style["range"][0]), float(style["range"][1]))
    if style.get("color"):
        kw["color"] = style["color"]
    if float(style.get("edge_linewidth", 0.0)) > 0:
        kw["edgecolor"] = style.get("edge_color", "#000000")
        kw["linewidth"] = float(style["edge_linewidth"])
    if style.get("label"):
        kw["label"] = style["label"]
    return kw


def ecdf_kwargs(style: dict) -> dict:
    """ax.ecdf の kwargs を解決する (render/scriptgen 共用。matplotlib 3.8+)。"""
    kw = {
        "linewidth": float(style.get("linewidth", 1.5)),
        "linestyle": style.get("linestyle", "solid"),
        "alpha": float(style.get("alpha", 1.0)),
    }
    if style.get("complementary"):
        kw["complementary"] = True
    if style.get("color"):
        kw["color"] = style["color"]
    if style.get("label"):
        kw["label"] = style["label"]
    return kw


def dist_line_kwargs(style: dict) -> dict:
    """dist_1d のラインレイヤー (ax.plot) の kwargs (render/scriptgen 共用)。"""
    kw = {
        "linewidth": float(style.get("linewidth", 1.5)),
        "linestyle": style.get("linestyle", "solid"),
    }
    if style.get("color"):
        kw["color"] = style["color"]
    if style.get("marker"):
        kw["marker"] = style["marker"]
        if style.get("marker_size") is not None:
            kw["markersize"] = float(style["marker_size"])
    if style.get("label"):
        kw["label"] = style["label"]
    return kw


def box_kwargs(style: dict) -> dict:
    """ax.boxplot の kwargs (positions/orientation 以外、render/scriptgen 共用)。

    whis は IQR 倍率 (数値) またはパーセンタイル対 ([lo, hi])。体裁は
    *props 辞書に組み立てる。既定値 (None/False) の項目は kwargs に含めず、
    旧設定の生成スクリプトを変えない。
    """
    whis = style.get("whis", 1.5)
    whis = ([float(whis[0]), float(whis[1])]
            if isinstance(whis, (list, tuple)) else float(whis))
    kw = {
        "widths": float(style.get("width", 0.5)),
        "whis": whis,
        "showfliers": bool(style.get("showfliers", True)),
        "showmeans": bool(style.get("showmeans", False)),
    }
    if style.get("notch"):
        kw["notch"] = True
    if not style.get("showcaps", True):
        kw["showcaps"] = False
    if not style.get("showbox", True):
        kw["showbox"] = False
    if style.get("meanline"):
        kw["meanline"] = True
    if style.get("capwidths") is not None:
        kw["capwidths"] = float(style["capwidths"])
    # 線 (箱・ひげ・キャップ共通) の色・太さ
    lineprops = {}
    if style.get("line_color"):
        lineprops["color"] = style["line_color"]
    if style.get("line_width") is not None:
        lineprops["linewidth"] = float(style["line_width"])
    boxprops = {}
    if style.get("fill_color"):
        # patch_artist の箱は Patch なので線は edgecolor/linewidth
        kw["patch_artist"] = True
        boxprops["facecolor"] = style["fill_color"]
        if style.get("line_color"):
            boxprops["edgecolor"] = style["line_color"]
        if style.get("line_width") is not None:
            boxprops["linewidth"] = float(style["line_width"])
    elif lineprops:
        boxprops.update(lineprops)
    if boxprops:
        kw["boxprops"] = boxprops
    if lineprops:
        kw["whiskerprops"] = dict(lineprops)
        kw["capprops"] = dict(lineprops)
    medianprops = {}
    if style.get("median_color"):
        medianprops["color"] = style["median_color"]
    if style.get("median_width") is not None:
        medianprops["linewidth"] = float(style["median_width"])
    if medianprops:
        kw["medianprops"] = medianprops
    meanprops = {}
    if style.get("meanline"):
        if style.get("mean_color"):
            meanprops["color"] = style["mean_color"]
        if style.get("mean_width") is not None:
            meanprops["linewidth"] = float(style["mean_width"])
    elif style.get("mean_color"):
        meanprops = {"markerfacecolor": style["mean_color"],
                     "markeredgecolor": style["mean_color"]}
    if meanprops:
        kw["meanprops"] = meanprops
    flierprops = {}
    if style.get("flier_marker"):
        flierprops["marker"] = style["flier_marker"]
    if style.get("flier_size") is not None:
        flierprops["markersize"] = float(style["flier_size"])
    if style.get("flier_color"):
        flierprops["markerfacecolor"] = style["flier_color"]
        flierprops["markeredgecolor"] = style["flier_color"]
    if flierprops:
        kw["flierprops"] = flierprops
    return kw


def violin_kwargs(style: dict) -> dict:
    """ax.violinplot の kwargs (positions/orientation 以外、render/scriptgen 共用)。

    既定値の項目 (points=100 / side="both" / bw_method=None / quantiles なし)
    は kwargs に含めず、旧設定の生成スクリプトを変えない。
    """
    kw = {
        "widths": float(style.get("width", 0.7)),
        "showmedians": bool(style.get("showmedians", True)),
        "showmeans": bool(style.get("showmeans", False)),
        "showextrema": bool(style.get("showextrema", True)),
    }
    points = style.get("points")
    if points is not None and int(points) != 100:
        kw["points"] = int(points)
    bw = style.get("bw_method")
    if bw is not None:
        kw["bw_method"] = bw if isinstance(bw, str) else float(bw)
    q = style.get("quantiles")
    if q:
        kw["quantiles"] = [[float(v) for v in q]]  # dataset=[vals] に対応する入れ子
    side = style.get("side", "both")
    if side and side != "both":
        kw["side"] = side
    return kw


def dist_point_kwargs(style: dict) -> dict:
    """box/violin の元データ点 (strip) の scatter kwargs (render/scriptgen 共用)。

    zorder=5 で箱・バイオリンより前面に出す (塗り箱の下に隠れないように)。
    """
    return {
        "s": float(style.get("point_size", 6.0)),
        "color": style.get("point_color", "#555555"),
        "alpha": float(style.get("point_alpha", 0.4)),
        "linewidths": 0,
        "zorder": 5,
    }


def dist_series_labels(panel: dict) -> list[str]:
    """box/violin 系列の x 目盛ラベル (系列名、None は変数名) を返す。

    render/scriptgen 共用。x 位置はレイヤーの並び順 (1, 2, ...)。
    """
    labels = []
    for ly in panel.get("layers", []):
        if ly.get("kind") in ("box", "violin"):
            labels.append(str(ly.get("style", {}).get("label")
                              or ly.get("variable")))
    return labels


def dist_uses_secondary_axis(panel: dict) -> bool:
    """dist_1d パネルが第2軸 (twinx) を必要とするか (render/scriptgen 共用)。"""
    return any(bool(ly.get("style", {}).get("secondary_y"))
               for ly in panel.get("layers", [])
               if ly.get("kind") in ("line", "ecdf"))


def line_uses_secondary_axis(panel: dict) -> bool:
    """line_1d パネルが第2軸 (twinx) を必要とするか (render/scriptgen 共用)。

    線・帯・積み上げ・棒のどれかが style.secondary_y=True のとき True
    (単位の違う 2 つの物理量を左右の縦軸に振り分ける用途)。
    """
    return any(bool(ly.get("style", {}).get("secondary_y"))
               for ly in panel.get("layers", []))


def secondary_axis_cfg(axis_cfg: dict) -> dict:
    """第2軸 (twinx) 用に y2 系キーを y スロットへ読み替えた axis 設定を返す
    (render/scriptgen 共用)。

    第1軸と同じ関数 (_apply_tick_settings / 対数 / 反転 / 補助目盛 / 目盛線 /
    軸ラベル) を ax2 にそのまま当てるための変換。x 系キーと swap_y_sides は
    含めない (x は twinx で共有され ax2 側は非表示。左右入れ替えは呼び出し側が
    ax2 → 左を扱う)。目盛文字サイズ・目盛線の太さ・目盛文字の回転・軸ラベルの
    回転・目盛線の色/太さ/線種・tight_y は第1軸と共有する。
    """
    grid = axis_cfg.get("grid", {}) or {}
    cfg2 = {
        "y_label": axis_cfg.get("y2_label"),
        "y_lim": axis_cfg.get("y2_lim"),
        "log_y": bool(axis_cfg.get("log_y2")),
        "invert_y": bool(axis_cfg.get("invert_y2")),
        "y_tick_interval": axis_cfg.get("y2_tick_interval"),
        "y_tick_positions": axis_cfg.get("y2_tick_positions"),
        "y_tick_labels": axis_cfg.get("y2_tick_labels"),
        "show_y_minor_ticks": bool(axis_cfg.get("show_y2_minor_ticks")),
        "show_y_ticklabels": axis_cfg.get("show_y2_ticklabels", True),
        "grid": {**grid, "show_x": False, "show_y": bool(grid.get("show_y2"))},
    }
    for key in ("tick_fontsize", "tick_width", "y_tick_rotation",
                "y_label_rotation", "tight_y"):
        if key in axis_cfg:
            cfg2[key] = axis_cfg[key]
    return cfg2


def dist_axis_labels(panel: dict, datasets: dict):
    """dist_1d の (x, y) 軸ラベルを解決する (render/scriptgen 共用)。

    axis.x_label / y_label の明示指定が優先。None のときは自動ラベル:
    x = 最初の hist/ecdf レイヤーの変数名 [units] (long_name は使わない)、
    y = 第1軸の内容から count / probability density / cumulative probability 等
    (hist があれば hist を優先。第2軸行きのレイヤーは対象外)。hist が横向き
    (style.orientation == "horizontal") なら値のラベルが y、度数のラベルが x。
    自動ラベルを英語にするのは、matplotlib 既定フォント (DejaVu Sans) が日本語を
    描けず豆腐 (□) になるため (図全体フォントで日本語を選べば手動で日本語にできる)。
    """
    axis_cfg = panel.get("axis", {})
    xlab = axis_cfg.get("x_label")
    ylab = axis_cfg.get("y_label")
    if xlab and ylab:
        return xlab, ylab
    layers = panel.get("layers", [])
    xsrc = next((ly for ly in layers if ly.get("kind") in ("hist", "ecdf")), None)
    # y は第1軸に描かれるものから決める (hist 優先、次に第1軸の ecdf)
    ysrc = next((ly for ly in layers if ly.get("kind") == "hist"), None)
    if ysrc is None:
        ysrc = next((ly for ly in layers if ly.get("kind") == "ecdf"
                     and not ly.get("style", {}).get("secondary_y")), None)
    # 値のラベルと度数のラベルを求めてから軸に割り当てる
    val_lab = None
    if xsrc is not None:
        attrs = datasets[xsrc["dataset_id"]][xsrc["variable"]].attrs
        units = str(attrs.get("units", ""))
        val_lab = (f"{xsrc['variable']} [{units}]" if units
                   else str(xsrc["variable"]))
    cnt_lab = None
    if ysrc is not None:
        s = ysrc.get("style", {})
        if ysrc["kind"] == "ecdf":
            cnt_lab = ("exceedance probability" if s.get("complementary")
                       else "cumulative probability")
        elif s.get("cumulative"):
            cnt_lab = ("cumulative probability" if s.get("density")
                       else "cumulative count")
        else:
            cnt_lab = "probability density" if s.get("density") else "count"
    hist_horizontal = (ysrc is not None and ysrc["kind"] == "hist"
                       and ysrc.get("style", {}).get("orientation") == "horizontal")
    x_auto, y_auto = (cnt_lab, val_lab) if hist_horizontal else (val_lab, cnt_lab)
    if not xlab and x_auto is not None:
        xlab = x_auto
    if not ylab and y_auto is not None:
        ylab = y_auto
    # box/violin パネル: 値の軸に変数ラベル (縦 = y、横向き = x)。
    # 系列軸は系列名の目盛なのでラベルなし
    bsrc = next((ly for ly in layers
                 if ly.get("kind") in ("box", "violin")), None)
    if bsrc is not None:
        horizontal = panel.get("box_orientation") == "horizontal"
        if (not xlab) if horizontal else (not ylab):
            attrs = datasets[bsrc["dataset_id"]][bsrc["variable"]].attrs
            units = str(attrs.get("units", ""))
            vlabel = (f"{bsrc['variable']} [{units}]" if units
                      else str(bsrc["variable"]))
            if horizontal:
                xlab = vlabel
            else:
                ylab = vlabel
    return xlab, ylab


def _draw_hist_layer(ax, panel: dict, layer: dict, datasets: dict):
    """ヒストグラムレイヤー (集計 → ax.hist)。"""
    vals = dist_values(layer, datasets)
    ax.hist(vals, **hist_kwargs(layer["style"]))


def _draw_ecdf_layer(ax, layer: dict, datasets: dict):
    """ECDF レイヤー (集計 → ax.ecdf)。"""
    vals = dist_values(layer, datasets)
    ax.ecdf(vals, **ecdf_kwargs(layer["style"]))


def _draw_dist_points(ax, vals, position, style: dict, horizontal: bool):
    """box/violin の元データ点 (strip) を系列位置に散らして描く。

    ジッターは固定シード (default_rng(0)) なので render/scriptgen/再実行で
    同一 (再現スクリプトの決定性)。show_points が False なら何もしない。
    """
    if not style.get("show_points"):
        return
    jitter = float(style.get("point_jitter", 0.2))
    pos = (np.random.default_rng(0).random(vals.size) - 0.5) * jitter \
        + float(position)
    kw = dist_point_kwargs(style)
    if horizontal:
        ax.scatter(vals, pos, **kw)
    else:
        ax.scatter(pos, vals, **kw)


def _draw_box_layer(ax, layer: dict, datasets: dict, position: int,
                    horizontal: bool = False):
    """箱ひげ図レイヤー (1系列)。位置はレイヤーの並び順 (横向きは y 方向)。

    boxplot は凡例ハンドルを自動登録しないため、in_legend のときは
    箱の代表アーティストに系列名をラベル付けする。
    """
    vals = dist_values(layer, datasets)
    style = layer["style"]
    kwargs = box_kwargs(style)
    if horizontal:
        kwargs["orientation"] = "horizontal"
    bp = ax.boxplot([vals], positions=[float(position)], **kwargs)
    _draw_dist_points(ax, vals, position, style, horizontal)
    if style.get("in_legend"):
        # showbox=False のとき boxes は空になるので中央値線を代表にする
        (bp["boxes"] or bp["medians"])[0].set_label(
            str(style.get("label") or layer.get("variable")))


def _draw_violin_layer(ax, layer: dict, datasets: dict, position: int,
                       horizontal: bool = False):
    """バイオリンレイヤー (1系列)。色は本体 (bodies) と線の両方に当てる。

    in_legend のときは本体 (PolyCollection) に系列名をラベル付けして
    凡例に出せるようにする (box と同じ流儀)。
    """
    vals = dist_values(layer, datasets)
    style = layer["style"]
    kwargs = violin_kwargs(style)
    if horizontal:
        kwargs["orientation"] = "horizontal"
    parts = ax.violinplot([vals], positions=[float(position)], **kwargs)
    c = style.get("color")
    if c:
        for _b in parts["bodies"]:
            _b.set_facecolor(c)
        for _k in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
            if _k in parts:
                parts[_k].set_color(c)
    for _b in parts["bodies"]:
        _b.set_alpha(float(style.get("alpha", 0.5)))
    _draw_dist_points(ax, vals, position, style, horizontal)
    if style.get("in_legend"):
        parts["bodies"][0].set_label(str(style.get("label")
                                         or layer.get("variable")))


def _draw_dist_line_layer(ax, layer: dict, datasets: dict):
    """dist_1d のラインレイヤー: layer["x_dim"] の座標値を x、変数値を y に描く。

    ヒストグラムに理論分布・参照 PDF 曲線などを重ねる用途 (x は値軸なので
    数値座標の次元のみ。UI 側で datetime 座標は候補から除外)。
    """
    ds = datasets[layer["dataset_id"]]
    da = ds[layer["variable"]]
    sel = {d: v for d, v in (layer.get("selection") or {}).items()
           if d in da.dims}
    if sel:
        da = da.sel(sel)
    da = apply_value_transform(da, layer["style"])
    ax.plot(da[layer["x_dim"]].values, da.values,
            **dist_line_kwargs(layer["style"]))


def _render_dist_1d(fig, panel: dict, datasets: dict[str, xr.Dataset],
                    subplot: tuple = (1, 1, 1)):
    """1次元プロット(集計) — hist などの分布系パネルを描画する。

    軸まわりは scatter_2d と同じ構成 (共有部品)。軸ラベルは未指定なら
    dist_axis_labels の自動ラベルが入る。
    """
    ax = fig.add_subplot(*subplot)
    _apply_box_aspect(ax, panel)
    ax2 = None
    if dist_uses_secondary_axis(panel):
        ax2 = ax.twinx()
        # set_box_aspect は描画時に axes 位置を動かすため、twinx した両方に
        # 同じ値を当てないと左右の枠がずれる (matplotlib の既知の癖)
        _apply_box_aspect(ax2, panel)
    dist_horizontal = panel.get("box_orientation") == "horizontal"
    series_pos = 0
    for layer in panel["layers"]:
        kind = layer["kind"]
        target = (ax2 if (ax2 is not None
                          and layer.get("style", {}).get("secondary_y")
                          and kind in ("line", "ecdf")) else ax)
        if kind == "hist":
            _draw_hist_layer(ax, panel, layer, datasets)
        elif kind == "ecdf":
            _draw_ecdf_layer(target, layer, datasets)
        elif kind == "line":
            _draw_dist_line_layer(target, layer, datasets)
        elif kind == "box":
            series_pos += 1
            _draw_box_layer(ax, layer, datasets, series_pos,
                            horizontal=dist_horizontal)
        elif kind == "violin":
            series_pos += 1
            _draw_violin_layer(ax, layer, datasets, series_pos,
                               horizontal=dist_horizontal)
        else:
            raise NotImplementedError(f"Unsupported layer kind: {kind}")
    axis_cfg = panel.get("axis", {})
    # box/violin の系列名を系列軸の目盛に (横向きなら y 軸)。
    # 目盛位置の手動指定があればそちらが優先
    series_labels = dist_series_labels(panel)
    if series_labels:
        if dist_horizontal:
            if (axis_cfg.get("y_tick_interval") is None
                    and axis_cfg.get("y_tick_positions") is None):
                ax.set_yticks(range(1, len(series_labels) + 1))
                ax.set_yticklabels(series_labels)
        elif (axis_cfg.get("x_tick_interval") is None
              and axis_cfg.get("x_tick_positions") is None):
            ax.set_xticks(range(1, len(series_labels) + 1))
            ax.set_xticklabels(series_labels)
        # 系列軸の範囲を [0.5, n+0.5] に固定 (カテゴリ軸の標準余白)。
        # autoscale だと1系列のときに軸がバイオリン/箱の幅へ追従してしまい、
        # 幅を変えても見かけが変わらない (手動の範囲指定があればそちらが優先)
        if dist_horizontal:
            if not axis_cfg.get("y_lim"):
                ax.set_ylim(0.5, len(series_labels) + 0.5)
        elif not axis_cfg.get("x_lim"):
            ax.set_xlim(0.5, len(series_labels) + 0.5)
    # 縦軸の左右入れ替え: 第1軸→右は共有の _apply_tick_settings が行う。
    # 第2軸 (twinx) → 左だけここで扱う
    if axis_cfg.get("swap_y_sides") and ax2 is not None:
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
    _apply_axis_grid(ax, axis_cfg)
    _apply_log_scales(ax, axis_cfg, scalar_formatter=False)
    _apply_minor_ticks(ax, axis_cfg)
    xlab, ylab = dist_axis_labels(panel, datasets)
    label_base = _set_axis_labels(ax, axis_cfg, xlab, ylab)
    # 目盛設定は対数軸の後に適用 (明示指定した目盛位置を優先)
    _apply_tick_settings(ax, axis_cfg)
    _apply_invert_and_limits(ax, axis_cfg)
    # 第2軸の体裁 (ラベル・範囲・目盛文字サイズ)
    if ax2 is not None:
        if axis_cfg.get("y2_label"):
            ax2.set_ylabel(axis_cfg["y2_label"], **label_base)
        if axis_cfg.get("y2_lim"):
            ax2.set_ylim(tuple(axis_cfg["y2_lim"]))
        if axis_cfg.get("tick_fontsize"):
            ax2.tick_params(labelsize=axis_cfg["tick_fontsize"])
    _apply_frame_background(ax, panel)
    if ax2 is not None:
        # 図枠は ax2 にも当てる (ax2 の既定 spine が上に重なって図枠設定を
        # 隠してしまうため。背景は base ax 側のみ)
        _apply_frame(ax2, panel.get("frame") or {})
    _draw_legend(ax, ax2, panel.get("legend", {}))
    _suppress_offset_text(ax, ax2)
    _finish_panel(ax, panel)
    return ax


def _render_agg_2d(fig, panel: dict, datasets: dict[str, xr.Dataset],
                   subplot: tuple = (1, 1, 1)):
    """2次元プロット(集計) — hist2d / hexbin のパネルを描画する。

    軸まわりは scatter_2d と同じ共有部品構成。凡例は無し (カラーバーで語る図)。
    軸ラベルは未指定なら agg_axis_labels の自動ラベル (変数 [units])。
    """
    ax = fig.add_subplot(*subplot)
    _apply_box_aspect(ax, panel)
    for layer in panel["layers"]:
        kind = layer["kind"]
        if kind == "hist2d":
            _draw_hist2d_layer(fig, ax, panel, layer, datasets)
        elif kind == "hexbin":
            _draw_agg_hexbin_layer(fig, ax, panel, layer, datasets)
        else:
            raise NotImplementedError(f"Unsupported layer kind: {kind}")
    axis_cfg = panel.get("axis", {})
    _apply_axis_grid(ax, axis_cfg)
    _apply_log_scales(ax, axis_cfg, scalar_formatter=False)
    _apply_minor_ticks(ax, axis_cfg)
    xlab, ylab = agg_axis_labels(panel, datasets)
    _set_axis_labels(ax, axis_cfg, xlab, ylab)
    # 目盛設定は対数軸の後に適用 (明示指定した目盛位置を優先)
    _apply_tick_settings(ax, axis_cfg)
    _apply_invert_and_limits(ax, axis_cfg)
    _apply_frame_background(ax, panel)
    _suppress_offset_text(ax)
    _finish_panel(ax, panel)
    return ax


def _render_scatter_2d(fig, panel: dict, datasets: dict[str, xr.Dataset],
                       subplot: tuple[int, int, int] = (1, 1, 1)):
    """散布図 (scatter_2d) のパネルを描画する。"""
    ax = fig.add_subplot(*subplot)
    _apply_box_aspect(ax, panel)
    for layer in panel["layers"]:
        kind = layer["kind"]
        if kind == "scatter":
            _draw_scatter_layer(ax, panel, layer, datasets)
        elif kind == "bubble":
            _draw_bubble_layer(fig, ax, panel, layer, datasets)
        elif kind == "hexbin":
            _draw_hexbin_layer(fig, ax, panel, layer, datasets)
        else:
            raise NotImplementedError(f"Unsupported layer kind: {kind}")
    axis_cfg = panel.get("axis", {})
    _apply_axis_grid(ax, axis_cfg)
    _apply_log_scales(ax, axis_cfg, scalar_formatter=False)
    _apply_minor_ticks(ax, axis_cfg)
    _set_axis_labels(ax, axis_cfg, axis_cfg.get("x_label"), axis_cfg.get("y_label"))
    # 目盛設定は対数軸の後に適用 (明示指定した目盛位置を対数の既定目盛より優先)
    _apply_tick_settings(ax, axis_cfg)
    _apply_invert_and_limits(ax, axis_cfg)
    _apply_frame_background(ax, panel)
    _draw_legend(ax, None, panel.get("legend", {}))
    _suppress_offset_text(ax)
    _finish_panel(ax, panel)
    return ax


def _bubble_xyz_arrays(panel: dict, layer: dict, datasets: dict,
                       with_errors: bool = False):
    """バブルチャートの (x, y, z) を抽出し、NaN を除いた 1D 配列の3つ組を返す。

    panel.x_variable / y_variable / z_variable から x/y/z を取り、layer の
    x_fixed/y_fixed/z_fixed と drawing_range を適用してから ravel する。
    with_errors=True では (x, y, z, xerr, yerr) — 誤差の扱いは
    _scatter_xy_arrays と同じ (値の変換なし・|err|・本体と同じ mask)。
    """
    ds = datasets[layer["dataset_id"]]
    x_var = panel.get("x_variable")
    y_var = panel.get("y_variable")
    z_var = panel.get("z_variable")
    if not x_var or not y_var or not z_var:
        empty = np.array([])
        return ((empty, empty, empty, None, None) if with_errors
                else (empty, empty, empty))
    drawing_dim = layer.get("drawing_dim")
    drawing_range = layer.get("drawing_range")
    style = layer["style"]
    x, y, z = (
        _scatter_take_values(ds, var, layer.get(f"{a}_fixed"), drawing_dim, drawing_range,
                             float(style.get(f"{a}_value_scale", 1.0)),
                             float(style.get(f"{a}_value_offset", 0.0)))
        for a, var in (("x", x_var), ("y", y_var), ("z", z_var)))
    xe = ye = None
    if with_errors:
        xe, ye = _scatter_error_arrays(ds, layer, style, drawing_dim, drawing_range)
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if with_errors:
        return (x[mask], y[mask], z[mask],
                _fit_error_to_mask(xe, n, mask),
                _fit_error_to_mask(ye, n, mask))
    return x[mask], y[mask], z[mask]


def _normalize_to_size_range(z, size_min: float, size_max: float):
    """z を [size_min, size_max] に線形正規化。z が全て同値ならその中央値を返す。"""
    if len(z) == 0:
        return z
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min < 1e-12:
        return np.full_like(z, (size_min + size_max) / 2.0)
    return size_min + (z - z_min) / (z_max - z_min) * (size_max - size_min)


def agg_xy_values(panel: dict, layer: dict, datasets: dict):
    """agg_2d の (x, y) 標本ペアを返す (共通次元を全て ravel、NaN ペア除去)。

    x/y で共通でない次元は layer の x_fixed / y_fixed で固定する。固定後の
    次元集合が一致しないと ValueError。y は x の次元順に transpose してから
    ravel し、ペアの対応を保証する。値変換 (x/y 独立) は float64 化の後
    (render/scriptgen 共通ロジック)。
    """
    ds = datasets[layer["dataset_id"]]
    style = layer["style"]

    def _take(var, fixed):
        da = ds[var]
        sel = {d: v for d, v in (fixed or {}).items() if d in da.dims}
        return da.sel(sel) if sel else da

    xda = _take(panel["x_variable"], layer.get("x_fixed"))
    yda = _take(panel["y_variable"], layer.get("y_fixed"))
    if set(xda.dims) != set(yda.dims):
        raise ValueError(
            "x/y 変数の次元が一致しません (共通でない次元は固定してください): "
            f"x={list(xda.dims)}, y={list(yda.dims)}")
    if xda.dims != yda.dims:
        yda = yda.transpose(*xda.dims)
    x = np.asarray(xda.values, dtype=float).ravel()
    y = np.asarray(yda.values, dtype=float).ravel()
    sx = float(style.get("x_value_scale", 1.0))
    ox = float(style.get("x_value_offset", 0.0))
    if sx != 1.0 or ox != 0.0:
        x = x * sx + ox
    sy = float(style.get("y_value_scale", 1.0))
    oy = float(style.get("y_value_offset", 0.0))
    if sy != 1.0 or oy != 0.0:
        y = y * sy + oy
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def hist2d_kwargs(style: dict) -> dict:
    """ax.hist2d の kwargs (norm/vmin/vmax 以外) を解決する (render/scriptgen 共用)。

    「0 のビンを塗らない」は cmin で実現 (counts は 1、density は 1e-300)。
    """
    kw = {"bins": [int(style.get("bins_x", 20)), int(style.get("bins_y", 20))]}
    rx, ry = style.get("range_x"), style.get("range_y")
    if rx is not None and ry is not None:
        kw["range"] = [[float(rx[0]), float(rx[1])],
                       [float(ry[0]), float(ry[1])]]
    if style.get("density"):
        kw["density"] = True
    if style.get("hide_zeros", True):
        kw["cmin"] = 1e-300 if style.get("density") else 1.0
    kw["cmap"] = resolve_cmap(style.get("cmap", "viridis"),
                              style.get("reverse_cmap", False))
    # ビンの枠線 (QuadMesh に転送される。cmin でマスクされたビンには描かれない)
    if style.get("edge_width"):
        kw["edgecolors"] = style.get("edge_color", "#000000")
        kw["linewidths"] = float(style["edge_width"])
    return kw


def agg_axis_labels(panel: dict, datasets: dict):
    """agg_2d の (x, y) 軸ラベルを解決する (render/scriptgen 共用)。

    明示指定が優先。None なら x/y の変数名 [units] (long_name は使わない。
    dist と同じく英語の自動ラベルでフォント既定の日本語豆腐を避ける)。
    """
    axis_cfg = panel.get("axis", {})
    xlab = axis_cfg.get("x_label")
    ylab = axis_cfg.get("y_label")
    layers = panel.get("layers", [])
    dsid = layers[0]["dataset_id"] if layers else next(iter(datasets), None)
    ds = datasets.get(dsid) if dsid else None

    def _vlabel(var):
        if ds is None or var not in ds:
            return var
        units = str(ds[var].attrs.get("units", ""))
        return f"{var} [{units}]" if units else str(var)

    if not xlab and panel.get("x_variable"):
        xlab = _vlabel(panel["x_variable"])
    if not ylab and panel.get("y_variable"):
        ylab = _vlabel(panel["y_variable"])
    return xlab, ylab


def _draw_hist2d_layer(fig, ax, panel: dict, layer: dict, datasets: dict):
    """2次元ヒストグラムレイヤー (ax.hist2d)。LogNorm と vmin/vmax は排他的に扱う。

    色の離散化 (levels): 度数はビン集計後にしか分からないため、描画後に
    集計結果 h[0] (cmin で隠したビンは NaN) から BoundaryNorm を作って
    set_norm で差し替える (対数スケールとは UI で排他)。
    """
    x, y = agg_xy_values(panel, layer, datasets)
    style = layer["style"]
    kw = hist2d_kwargs(style)
    if style.get("log_counts"):
        kw["norm"] = mcolors.LogNorm(vmin=style.get("vmin"),
                                     vmax=style.get("vmax"))
    else:
        if style.get("vmin") is not None:
            kw["vmin"] = float(style["vmin"])
        if style.get("vmax") is not None:
            kw["vmax"] = float(style["vmax"])
    h = ax.hist2d(x, y, **kw)
    if resolve_discrete_levels(style) is not None:
        dk = discrete_color_kwargs(
            style, float(np.nanmin(h[0])), float(np.nanmax(h[0])))
        if "cmap" in dk:
            h[3].set_cmap(dk["cmap"])
        h[3].set_norm(dk["norm"])
    if style.get("edge_width"):
        # 枠線は「塗られたビン」だけに: QuadMesh の edge は面のマスクと無関係に
        # 全セルへ描かれるため、cmin でマスクされたセルの edge を none にする
        _mask = np.ma.getmaskarray(h[3].get_array()).ravel()
        if _mask.any():
            _edge = style.get("edge_color", "#000000")
            h[3].set_edgecolor(["none" if m else _edge for m in _mask])
    if style.get("colorbar", {}).get("show", True):
        _add_colorbar(fig, ax, h[3], style.get("colorbar", {}),
                      extend=continuous_cbar_extend(style))


def _draw_agg_hexbin_layer(fig, ax, panel: dict, layer: dict, datasets: dict):
    """agg_2d の hexbin (全 ravel 方式)。style は scatter 時代の hexbin と共通。

    色の離散化 (levels): 点数は集計後にしか分からないため、描画後に
    hb.get_array() (mincnt 適用後の描画ビンの点数) から BoundaryNorm を
    作って set_norm で差し替える (対数スケールとは UI で排他)。
    """
    x, y = agg_xy_values(panel, layer, datasets)
    style = layer["style"]
    hb = ax.hexbin(x, y, **hexbin_kwargs(style))
    if resolve_discrete_levels(style) is not None:
        dk = discrete_color_kwargs(
            style, float(np.nanmin(hb.get_array())),
            float(np.nanmax(hb.get_array())))
        if "cmap" in dk:
            hb.set_cmap(dk["cmap"])
        hb.set_norm(dk["norm"])
    if style.get("colorbar", {}).get("show", True):
        _add_colorbar(fig, ax, hb, style.get("colorbar", {}),
                      extend=continuous_cbar_extend(style))


def hexbin_kwargs(style: dict) -> dict:
    """ax.hexbin の kwargs を解決する (render/scriptgen 共用、cmap は文字列)。"""
    kw = {
        "gridsize": int(style.get("gridsize", 20)),
        "cmap": resolve_cmap(style.get("cmap", "viridis"),
                             style.get("reverse_cmap", False)),
    }
    if style.get("log_counts"):
        kw["bins"] = "log"
    if style.get("mincnt") is not None:
        kw["mincnt"] = int(style["mincnt"])
    if style.get("vmin") is not None:
        kw["vmin"] = float(style["vmin"])
    if style.get("vmax") is not None:
        kw["vmax"] = float(style["vmax"])
    # ビンの枠線 (mincnt 未満で描かれないビンには当然付かない)
    if style.get("edge_width"):
        kw["edgecolors"] = style.get("edge_color", "#000000")
        kw["linewidths"] = float(style["edge_width"])
    return kw


def map_scatter_points(panel: dict, layer: dict, datasets: dict):
    """地図散布図の (lon, lat, value) 平坦配列を返す (render/scriptgen 共通ロジック)。

    データ構造で自動分岐:
      - 格子データ (変数が lat/lon 次元をもつ): meshgrid で全格子点
      - 地点データ (lon/lat が補助座標): その座標をそのまま使う
    値変換適用後、NaN を含む点を除く。
    """
    ds = datasets[layer["dataset_id"]]
    roles = detect_coord_roles(ds)
    lon_name, lat_name = roles["lon"], roles["lat"]
    var = layer["variable"]
    style = layer["style"]
    hdims = horizontal_dims(ds, roles)
    if hdims is not None and all(d in ds[var].dims for d in hdims):
        # 格子データ: region/selection を fill と同様に適用してから meshgrid
        da, _ = _layer_data(panel, layer, var, datasets)
        da = da.transpose(*hdims)
        da = apply_value_transform(da, style)
        if is_curvilinear(ds, roles):
            # 2 次元座標はそのまま平坦化 (da と同じ (y, x) 並び)
            lon_f = np.asarray(da[lon_name].values, dtype=float).ravel()
            lat_f = np.asarray(da[lat_name].values, dtype=float).ravel()
        else:
            lon2d, lat2d = np.meshgrid(da[lon_name].values, da[lat_name].values)
            lon_f = lon2d.ravel()
            lat_f = lat2d.ravel()
        val_f = np.asarray(da.values, dtype=float).ravel()
    else:
        # 地点データ: selection のみ適用、lon/lat は補助座標をそのまま
        selection = {**(panel.get("selection") or {}),
                     **(layer.get("selection") or {})}
        da = ds[var]
        sel = {d: v for d, v in selection.items() if d in da.dims}
        if sel:
            da = da.sel(sel)
        da = apply_value_transform(da, style)
        lon_f = np.asarray(ds[lon_name].values, dtype=float).ravel()
        lat_f = np.asarray(ds[lat_name].values, dtype=float).ravel()
        val_f = np.asarray(da.values, dtype=float).ravel()
        if val_f.size != lon_f.size:
            # 点数と値数の不一致 = 固定されていない次元が残っている
            # (例: 時刻)。黙って IndexError にせず原因の分かるエラーを出す
            point_dims = set(ds[lon_name].dims) | set(ds[lat_name].dims)
            unfixed = [str(d) for d in da.dims if d not in point_dims]
            raise RenderError("map_scatter_dims_unfixed",
                              var=var, dims=", ".join(unfixed) or "?",
                              n_points=int(lon_f.size),
                              n_values=int(val_f.size))
    mask = ~(np.isnan(lon_f) | np.isnan(lat_f) | np.isnan(val_f))
    return lon_f[mask], lat_f[mask], val_f[mask]


def map_scatter_kwargs(style: dict, foreground: bool = False) -> dict:
    """ax.scatter の共通 kwargs (色以外、render/scriptgen 共用)。
    foreground (陸を前景に描く地図) では地点を陸の上に出す。"""
    kw = {
        "s": float(style.get("size", 20.0)),
        "marker": style.get("marker", "o"),
        "alpha": float(style.get("alpha", 1.0)),
    }
    if float(style.get("edge_linewidth", 0.0)) > 0:
        kw["edgecolors"] = style.get("edge_color", "#000000")
        kw["linewidths"] = float(style["edge_linewidth"])
    if foreground:
        kw["zorder"] = MAP_FG_ZORDER
    return kw


def _draw_map_scatter(fig, ax, panel: dict, layer: dict, datasets: dict):
    """地図の散布図レイヤー。use_cmap 時はカラーバーを返す (それ以外 None)。"""
    lon, lat, val = map_scatter_points(panel, layer, datasets)
    style = layer["style"]
    kwargs = map_scatter_kwargs(style, foreground=panel_land_foreground(panel))
    kwargs["transform"] = ccrs.PlateCarree()
    use_cmap = bool(style.get("use_cmap"))
    cbar = None
    if use_cmap:
        kwargs["c"] = val
        kwargs["cmap"] = resolve_cmap(style.get("cmap", "viridis"),
                                      style.get("reverse_cmap", False))
        dk = discrete_color_kwargs(style, np.nanmin(val), np.nanmax(val))
        if dk:
            kwargs.update(dk)
        else:
            if style.get("vmin") is not None:
                kwargs["vmin"] = float(style["vmin"])
            if style.get("vmax") is not None:
                kwargs["vmax"] = float(style["vmax"])
        sc = ax.scatter(lon, lat, **kwargs)
        if style.get("colorbar", {}).get("show", True):
            cbar = _add_colorbar(fig, ax, sc, style.get("colorbar", {}),
                                 extend=continuous_cbar_extend(style))
    else:
        kwargs["c"] = style.get("color", "#1f77b4")
        ax.scatter(lon, lat, **kwargs)
    return cbar


def track_genesis_year_month(ds, layer: dict):
    """各ストームの発生時刻 (時刻変数の最初の有効時刻) から (有効, 年, 月) を返す。

    render/scriptgen 共通ロジック。時刻が全欠損 (NaT) のストームは有効 False。
    年・月は numpy の datetime64[Y]/[M] キャスト経由で取り出す (NaT はガベージ値
    になるが有効フラグで除外される)。
    """
    da = ds[layer["time_var"]]
    others = [d for d in da.dims if d != layer["storm_dim"]]
    gen = da.min(others, skipna=True) if others else da
    vals = gen.values
    ok = ~np.isnat(vals)
    yr = vals.astype("datetime64[Y]").astype(int) + 1970
    mo = vals.astype("datetime64[M]").astype(int) % 12 + 1
    return ok, yr, mo


def track_storm_indices(layer: dict, datasets: dict) -> list:
    """トラックレイヤーが描くストームの isel インデックス一覧を返す。

    storm_dim が無い (1本もの) なら [None]。選択は2方式:
      - 発生年・月 (time_var + year_range/month_range、優先): 発生時刻が
        範囲内のストーム。month_range は m0 > m1 で年またぎ (11→3 等)
      - index 範囲 (storm_range=[lo, hi]、両端含む・データ範囲にクリップ)。
        None なら全ストーム
    render/scriptgen/UI (選択本数の表示) で共用。
    """
    ds = datasets[layer["dataset_id"]]
    sdim = layer.get("storm_dim")
    if not sdim or sdim not in ds[layer["lon_var"]].dims:
        return [None]
    n = int(ds.sizes[sdim])
    tv = layer.get("time_var")
    yr_rng = layer.get("year_range")
    mo_rng = layer.get("month_range")
    if tv and tv in ds.variables and (yr_rng or mo_rng):
        ok, yr, mo = track_genesis_year_month(ds, layer)
        if yr_rng:
            ok = ok & (yr >= int(yr_rng[0])) & (yr <= int(yr_rng[1]))
        if mo_rng:
            m0, m1 = int(mo_rng[0]), int(mo_rng[1])
            if m0 <= m1:
                ok = ok & (mo >= m0) & (mo <= m1)
            else:  # 年またぎ (例 11→3)
                ok = ok & ((mo >= m0) | (mo <= m1))
        return [int(i) for i in np.where(ok)[0]]
    rng = layer.get("storm_range")
    if rng is None:
        return list(range(n))
    lo = max(0, int(rng[0]))
    hi = min(n - 1, int(rng[1]))
    return list(range(lo, hi + 1))


def track_arrays(layer: dict, datasets: dict, idx):
    """1本分の (lon, lat, 色値 or None) を float 配列で返す (render/scriptgen 共通ロジック)。

    NaN はそのまま返す (線は NaN で切れる。点は呼び出し側でマスクする)。
    色値には points 側の値変換 (value_scale/offset) を適用する。
    maskout (指定変数の値がしきい値の条件を満たす位置) は lon/lat/色値を
    NaN にして線を切る。しきい値は maskout 変数の生の値で、変数が欠損の
    位置も描かれない。
    """
    ds = datasets[layer["dataset_id"]]
    sdim = layer.get("storm_dim")

    def _take(name, style=None):
        da = ds[name]
        if idx is not None and sdim in da.dims:
            da = da.isel({sdim: idx})
        if style is not None:
            da = apply_value_transform(da, style)
        return np.asarray(da.values, dtype=float).ravel()

    lon = _take(layer["lon_var"])
    lat = _take(layer["lat_var"])
    pts = layer["style"].get("points") or {}
    val = _take(pts["variable"], pts) if pts.get("variable") else None
    mo = layer["style"].get("maskout") or {}
    below, above = mo.get("below"), mo.get("above")
    if mo.get("variable") and (below is not None or above is not None):
        mv = _take(mo["variable"])
        keep = np.ones(mv.shape, dtype=bool)
        if below is not None:
            keep &= mv > float(below)
        if above is not None:
            keep &= mv < float(above)
        # np.where で新しい配列を作る (キャッシュ済み dataset の
        # 元配列を書き換えないこと)
        lon = np.where(keep, lon, np.nan)
        lat = np.where(keep, lat, np.nan)
        if val is not None:
            val = np.where(keep, val, np.nan)
    return lon, lat, val


def track_line_kwargs(style: dict) -> dict:
    """トラックの線 (ax.plot) の kwargs を解決する (render/scriptgen 共用)。

    color=None は kwargs に含めない → matplotlib カラーサイクルで
    トラック毎に色が変わる。
    """
    kw = {
        "linewidth": float(style.get("linewidth", 1.5)),
        "linestyle": style.get("linestyle", "solid"),
        "alpha": float(style.get("alpha", 1.0)),
    }
    if style.get("color"):
        kw["color"] = style["color"]
    return kw


def track_point_kwargs(pts: dict, foreground: bool = False) -> dict:
    """トラックの観測点 (ax.scatter) の色以外の kwargs (render/scriptgen 共用)。
    foreground (陸を前景に描く地図) では点を陸の上に出す (線は zorder 2 で元から上)。"""
    kw = {
        "s": float(pts.get("size", 12.0)),
        "marker": pts.get("marker", "o"),
    }
    if foreground:
        kw["zorder"] = MAP_FG_ZORDER
    return kw


def _draw_track(fig, ax, panel: dict, layer: dict, datasets: dict):
    """トラック (軌跡) レイヤー。点を値で色付けしたときはカラーバーを返す。

    線はトラック毎に ax.plot (transform=ccrs.Geodetic() なので日付変更線を
    またいでも正しくつながる)。点は全トラック分をまとめて1回の ax.scatter
    (カラーバーを1つにするため)。
    """
    style = layer["style"]
    pts = style.get("points") or {}
    every = max(1, int(pts.get("every", 1)))
    show_pts = bool(pts.get("show", True))
    land_fg = panel_land_foreground(panel)
    line_kw = track_line_kwargs(style)
    lons, lats, vals = [], [], []
    for idx in track_storm_indices(layer, datasets):
        lon, lat, val = track_arrays(layer, datasets, idx)
        ax.plot(lon, lat, transform=ccrs.Geodetic(), **line_kw)
        if show_pts:
            lons.append(lon[::every])
            lats.append(lat[::every])
            if val is not None:
                vals.append(val[::every])
    if not show_pts or not lons:
        return None
    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    kwargs = track_point_kwargs(pts, foreground=land_fg)
    kwargs["transform"] = ccrs.PlateCarree()
    if pts.get("variable"):
        val = np.concatenate(vals)
        mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
        lon, lat, val = lon[mask], lat[mask], val[mask]
        if lon.size == 0:
            return None
        kwargs["c"] = val
        kwargs["cmap"] = resolve_cmap(pts.get("cmap", "viridis"),
                                      pts.get("reverse_cmap", False))
        dk = discrete_color_kwargs(pts, np.nanmin(val), np.nanmax(val))
        if dk:
            kwargs.update(dk)
        else:
            if pts.get("vmin") is not None:
                kwargs["vmin"] = float(pts["vmin"])
            if pts.get("vmax") is not None:
                kwargs["vmax"] = float(pts["vmax"])
        sc = ax.scatter(lon, lat, **kwargs)
        if pts.get("colorbar", {}).get("show", True):
            return _add_colorbar(fig, ax, sc, pts.get("colorbar", {}),
                                 extend=continuous_cbar_extend(pts))
        return None
    mask = np.isfinite(lon) & np.isfinite(lat)
    ax.scatter(lon[mask], lat[mask], c=pts.get("color", "#333333"), **kwargs)
    return None


def _draw_hexbin_layer(fig, ax, panel: dict, layer: dict, datasets: dict):
    """hexbin (六角ビン密度) レイヤー。散布図と同じ x/y を六角ビンに集計する。"""
    x_arr, y_arr = _scatter_xy_arrays(panel, layer, datasets)
    style = layer["style"]
    hb = ax.hexbin(x_arr, y_arr, **hexbin_kwargs(style))
    if style.get("colorbar", {}).get("show", True):
        _add_colorbar(fig, ax, hb, style.get("colorbar", {}))


def _draw_bubble_layer(fig, ax, panel: dict, layer: dict, datasets: dict):
    x_arr, y_arr, z_arr, xe, ye = _bubble_xyz_arrays(panel, layer, datasets,
                                                     with_errors=True)
    style = layer["style"]
    # エラーバーは点より先に描いて点を上に (render/scriptgen 同順)
    if xe is not None or ye is not None:
        ax.errorbar(x_arr, y_arr, xerr=xe, yerr=ye,
                    **errorbar_kwargs(style.get("errorbar") or {}))
    size_min = float(style.get("size_min", 10.0))
    size_max = float(style.get("size_max", 200.0))
    sizes = _normalize_to_size_range(z_arr, size_min, size_max)
    use_cmap = bool(style.get("use_cmap"))
    kwargs = {
        "alpha": float(style.get("alpha", 0.6)),
        "marker": style.get("marker", "o"),
    }
    if use_cmap:
        kwargs["c"] = z_arr
        kwargs["cmap"] = resolve_cmap(
            style.get("cmap", "viridis"), style.get("reverse_cmap", False))
        dk = discrete_color_kwargs(style, np.nanmin(z_arr), np.nanmax(z_arr))
        if dk:
            kwargs.update(dk)  # norm と vmin/vmax は同時指定不可
        else:
            if style.get("vmin") is not None:
                kwargs["vmin"] = float(style["vmin"])
            if style.get("vmax") is not None:
                kwargs["vmax"] = float(style["vmax"])
    else:
        kwargs["c"] = style.get("color", "#1f77b4")
    if float(style.get("edge_linewidth", 0.0)) > 0:
        kwargs["edgecolors"] = style.get("edge_color", "#000000")
        kwargs["linewidths"] = float(style["edge_linewidth"])
    if style.get("label"):
        kwargs["label"] = style["label"]
    sc = ax.scatter(x_arr, y_arr, s=sizes, **kwargs)
    # カラーバー (use_cmap=True のときだけ)
    if use_cmap:
        _add_colorbar(fig, ax, sc, style.get("colorbar", {}),
                      extend=continuous_cbar_extend(style))


def _line_layer_data(panel: dict, layer: dict, variable: str,
                      datasets: dict, *,
                      selection_override: dict | None = None,
                      averages_override: dict | None = None,
                      record: list | None = None) -> xr.DataArray:
    """1次元プロットの 1 変数を、cyclic / ranges / averages を適用して取り出す共通ヘルパ。

    _draw_line_layer と _draw_fill_between_layer の両方から使う。
    scriptgen._emit_line_data と一対一で対応すること。

    selection_override / averages_override を渡すと layer.selection / layer.averages の
    代わりにそれを使う (fill_between が var_upper に selection_upper / averages_upper を
    当てるとき用)。
    """
    ds = datasets[layer["dataset_id"]]
    layer_sel = selection_override if selection_override is not None \
        else (layer.get("selection") or {})
    layer_avg = averages_override if averages_override is not None \
        else (layer.get("averages") or {})
    selection = {**(panel.get("selection") or {}), **layer_sel}
    x_dim = panel["x_dim"]
    axis_cfg = panel.get("axis", {})
    ranges = panel.get("ranges") or {}
    if axis_cfg.get("cyclic_x") and x_dim in ranges:
        # cyclic 拡張: 範囲 sel の前に tile し、その上で sel する
        # averages は次元を潰すので tile/sel より先に適用する (select_line_data 内)
        da = select_line_data(ds, variable, x_dim, selection, None,
                               averages=layer_avg, record=record)
        x_lo, x_hi = float(ranges[x_dim][0]), float(ranges[x_dim][1])
        da = cyclic_tile_lon(da, x_dim, x_lo, x_hi)
        da = da.sel({x_dim: slice(x_lo, x_hi)})
        for dim, bounds in ranges.items():
            if dim != x_dim and dim in da.dims:
                da = da.sel({dim: coord_slice(ds[dim], bounds[0], bounds[1])})
        return da
    return select_line_data(ds, variable, x_dim, selection, ranges,
                             averages=layer_avg, record=record)


def line_plot_kwargs(style: dict) -> dict:
    """1次元プロットのライン (kind="line") の ax.plot kwargs (render/scriptgen 共用)。
    色・マーカー・ラベルは明示指定があるときだけ渡す (空欄→凡例に出さない)。
    マーカーの大きさ (marker_size、pt) はマーカーがあり None でないときだけ
    markersize に渡す (None = matplotlib 既定。旧設定はキー自体が無い)。"""
    kwargs = {
        "linewidth": style.get("linewidth", 1.5),
        "linestyle": style.get("linestyle", "solid"),
    }
    if style.get("color"):
        kwargs["color"] = style["color"]
    if style.get("marker"):
        kwargs["marker"] = style["marker"]
        if style.get("marker_size") is not None:
            kwargs["markersize"] = float(style["marker_size"])
    if style.get("label"):
        kwargs["label"] = style["label"]
    return kwargs


def _draw_line_layer(ax, panel: dict, layer: dict, datasets: dict):
    da = _line_layer_data(panel, layer, layer["variable"], datasets)
    da = apply_value_transform(da, layer["style"])
    ax.plot(da[panel["x_dim"]].values, da.values, **line_plot_kwargs(layer["style"]))


def bundle_line_kwargs(style: dict) -> dict:
    """ライン (束) の各線の ax.plot kwargs (render/scriptgen 共用)。

    凡例ラベルは含めない (1 本目にだけ呼び出し側が付ける)。色は必ず明示する
    (None のままだと 1 本ごとにカラーサイクルが進み全線が別色になる)。
    """
    kwargs = line_plot_kwargs({**style, "label": None})
    kwargs["color"] = style.get("color") or "#808080"
    kwargs["alpha"] = float(style.get("alpha", 1.0))
    return kwargs


def bundle_stat_kwargs(style: dict) -> dict:
    """ライン (束) の統計線の ax.plot kwargs (render/scriptgen 共用)。ラベルは含めない。"""
    return {
        "color": style.get("color") or "#000000",
        "linewidth": float(style.get("linewidth", 2.0)),
        "linestyle": style.get("linestyle", "solid"),
    }


def bundle_band_kwargs(style: dict) -> dict:
    """統計線 (2 本組) を帯で描くときの ax.fill_between kwargs (render/scriptgen 共用)。

    ラベルは含めない。縁は描かない (linewidth=0。縁の線は draw="band_lines" が
    ax.plot で重ねる)。
    """
    return {
        "color": style.get("color") or "#000000",
        "alpha": float(style.get("alpha", 0.3)),
        "linewidth": 0.0,
    }


def bundle_lines_visible(style: dict, default: float) -> bool:
    """線の太さが正なら線を描く (render/scriptgen 共用)。

    ライン (束) と統計線は「太さ 0 = 描かない」(帯だけを見せる用途)。マーカーが
    あっても描かない (太さ 0 で点だけ残ると意図が分かりにくいため)。
    """
    return float(style.get("linewidth", default)) > 0


def bundle_summary_draw(summary: dict) -> str:
    """統計線の描き方を解決する (render/scriptgen 共用)。

    "lines" (線) / "band" (帯) / "band_lines" (帯 + 縁の線)。2 本組
    (minmax / pct_range) 以外、または未知の値なら "lines"。
    """
    draw = summary.get("draw", "lines")
    if draw not in ("band", "band_lines") or len(bundle_stat_specs(summary)) != 2:
        return "lines"
    return draw


def bundle_stat_specs(summary: dict) -> list[tuple[str, float | None]]:
    """統計線 1 組を (集計名, パラメータ) の並びに展開する (render/scriptgen 共用)。

    集計名は numpy の nan* 関数名 (パラメータは分位) か、"mean_minus_std" /
    "mean_plus_std" (パラメータは標準偏差の倍率 k)。render は bundle_stat_eval、
    scriptgen は bundle_stat_expr で同じ式にする (1対1)。2 要素なら 2 本組
    (minmax / pct_range / std_range)。
    """
    stat = summary.get("stat", "mean")
    q_lo = float(summary.get("q_low", 5.0))
    q_hi = float(summary.get("q_high", 95.0))
    k = float(summary.get("k_std", 1.0))
    table = {
        "mean": [("nanmean", None)],
        "median": [("nanmedian", None)],
        "min": [("nanmin", None)],
        "max": [("nanmax", None)],
        "minmax": [("nanmin", None), ("nanmax", None)],
        "percentile": [("nanpercentile", q_lo)],
        "pct_range": [("nanpercentile", q_lo), ("nanpercentile", q_hi)],
        "std_range": [("mean_minus_std", k), ("mean_plus_std", k)],
    }
    if stat not in table:
        raise ValueError(f"Unknown bundle summary stat: {stat!r}")
    return table[stat]


def bundle_stat_eval(y, fn: str, q):
    """統計線 1 本の値 (束方向 = axis 1 の集計)。bundle_stat_expr と同じ式であること。"""
    # 標準偏差は標本標準偏差 (N−1 で割る、ddof=1)。アンサンブルの広がりの慣例に合わせる
    if fn == "mean_minus_std":
        return np.nanmean(y, axis=1) - q * np.nanstd(y, axis=1, ddof=1)
    if fn == "mean_plus_std":
        return np.nanmean(y, axis=1) + q * np.nanstd(y, axis=1, ddof=1)
    if q is None:
        return getattr(np, fn)(y, axis=1)
    return getattr(np, fn)(y, q, axis=1)


def bundle_stat_expr(y_name: str, fn: str, q) -> str:
    """bundle_stat_eval と同じ式の文字列 (再現スクリプト用)。"""
    if fn == "mean_minus_std":
        return (f"np.nanmean({y_name}, axis=1) - {q!r} * "
                f"np.nanstd({y_name}, axis=1, ddof=1)")
    if fn == "mean_plus_std":
        return (f"np.nanmean({y_name}, axis=1) + {q!r} * "
                f"np.nanstd({y_name}, axis=1, ddof=1)")
    if q is None:
        return f"np.{fn}({y_name}, axis=1)"
    return f"np.{fn}({y_name}, {q!r}, axis=1)"


def bundle_stat_default_label(summary: dict) -> str:
    """統計線の凡例ラベルの既定 (UI 用。言語に依存しない短い英字)。"""
    stat = summary.get("stat", "mean")
    q_lo = float(summary.get("q_low", 5.0))
    q_hi = float(summary.get("q_high", 95.0))
    if stat == "minmax":
        return "min-max"
    if stat == "percentile":
        return f"p{q_lo:g}"
    if stat == "pct_range":
        return f"p{q_lo:g}-p{q_hi:g}"
    if stat == "std_range":
        return f"mean±{float(summary.get('k_std', 1.0)):g}σ"
    return stat


def _bundle_layer_data(panel: dict, layer: dict, datasets: dict,
                       record: list | None = None) -> xr.DataArray:
    """ライン (束) の変数を (x_dim, bundle_dim) の 2 次元 DataArray で取り出す。

    _line_layer_data と同じ経路 (selection / averages / cyclic / ranges) を通し、
    bundle_dim だけは固定も平均もしない (panel.selection に入っていても除外)。
    scriptgen.line_bundle_layer と1対1で対応すること。
    """
    x_dim = panel["x_dim"]
    bdim = layer.get("bundle_dim")
    ds = datasets[layer["dataset_id"]]
    if not bdim or bdim not in ds[layer["variable"]].dims or bdim == x_dim:
        raise ValueError(
            f"line_bundle layer needs a bundle_dim other than x_dim {x_dim!r} "
            f"among dims of {layer['variable']!r}: got {bdim!r}")
    panel_b = {**panel, "selection": {d: v for d, v in (panel.get("selection") or {}).items()
                                      if d != bdim}}
    sel = {d: v for d, v in (layer.get("selection") or {}).items() if d != bdim}
    avg = effective_averages(ds[layer["variable"]].dims, layer.get("averages"),
                             exclude=(bdim,))
    da = _line_layer_data(panel_b, layer, layer["variable"], datasets,
                          selection_override=sel, averages_override=avg,
                          record=record)
    return da.transpose(x_dim, bdim)


def _draw_line_bundle_layer(ax, panel: dict, layer: dict, datasets: dict):
    """ライン (束): bundle_dim の全スライスを同じ体裁で重ね、統計線を上に描く。

    凡例は束で 1 つ (1 本目だけラベル)、統計線は 1 組で 1 つ。
    scriptgen.line_bundle_layer と1対1対応。
    """
    da = _bundle_layer_data(panel, layer, datasets)
    style = layer["style"]
    da = apply_value_transform(da, style)
    x = da[panel["x_dim"]].values
    y = da.values                      # (x, bundle)
    kw = bundle_line_kwargs(style)
    label = style.get("label") or None
    if bundle_lines_visible(style, 0.8):
        for k in range(y.shape[1]):
            ax.plot(x, y[:, k], **({"label": label} if (k == 0 and label) else {}), **kw)
    for summary in layer.get("summaries") or []:
        s_style = summary.get("style") or {}
        s_label = s_style.get("label") or None
        label_kw = {"label": s_label} if s_label else {}
        ys = [bundle_stat_eval(y, fn, q) for fn, q in bundle_stat_specs(summary)]
        draw = bundle_summary_draw(summary)
        lines_on = bundle_lines_visible(s_style, 2.0)   # 太さ 0 = 線を描かない
        if draw == "lines":
            if lines_on:
                s_kw = bundle_stat_kwargs(s_style)
                for j, ys_ in enumerate(ys):
                    ax.plot(x, ys_, **(label_kw if j == 0 else {}), **s_kw)
        else:
            # 帯: 2 本の間を塗る (凡例は帯に 1 つ)。band_lines は縁の線を上に重ねる
            ax.fill_between(x, ys[0], ys[1], **label_kw, **bundle_band_kwargs(s_style))
            if draw == "band_lines" and lines_on:
                s_kw = bundle_stat_kwargs(s_style)
                for ys_ in ys:
                    ax.plot(x, ys_, **s_kw)


def _draw_fill_between_layer(ax, panel: dict, layer: dict, datasets: dict):
    style = layer["style"]
    da_low = _line_layer_data(panel, layer, layer["variable"], datasets)
    da_low = apply_value_transform(da_low, style)
    var_upper = layer.get("variable_upper")
    color_above = style.get("color", "#1f77b4")
    color_below = style.get("color_below")  # None なら単色
    alpha = float(style.get("alpha", 0.3))
    label = style.get("label")
    x_vals = da_low[panel["x_dim"]].values
    y_low = da_low.values
    if var_upper:
        # 変数モードでは color_below は使わない (常に単色)
        da_high = _line_layer_data(
            panel, layer, var_upper, datasets,
            selection_override=layer.get("selection_upper") or {},
            averages_override=layer.get("averages_upper") or {})
        da_high = apply_value_transform(da_high, style)
        kwargs = {"color": color_above, "alpha": alpha}
        if label:
            kwargs["label"] = label
        ax.fill_between(x_vals, y_low, da_high.values, **kwargs)
        return
    # baseline モード
    baseline = float(layer.get("baseline", 0.0))
    if not color_below or color_below == color_above:
        kwargs = {"color": color_above, "alpha": alpha}
        if label:
            kwargs["label"] = label
        ax.fill_between(x_vals, y_low, baseline, **kwargs)
        return
    # 上下別色: where + interpolate=True で baseline 交点を綺麗に分割
    kw_above = {"color": color_above, "alpha": alpha, "interpolate": True}
    kw_below = {"color": color_below, "alpha": alpha, "interpolate": True}
    if label:
        kw_above["label"] = label  # 凡例には1つだけ載せる
    ax.fill_between(x_vals, y_low, baseline,
                     where=(y_low > baseline), **kw_above)
    ax.fill_between(x_vals, y_low, baseline,
                     where=(y_low < baseline), **kw_below)


def _apply_frame(ax, frame: dict):
    """axes 枠線 (spine) の表示・太さ・色を適用する。

    twinx の第2軸 (dist_1d) は自分の spine を既定値で上に重ね描きするため、
    ax2 にも同じ設定を当てる必要がある (当てないと図枠設定が効いていない
    ように見える)。scriptgen._frame_lines と1対1対応。
    """
    for side in ("top", "bottom", "left", "right"):
        if not frame.get(f"show_{side}", True):
            ax.spines[side].set_visible(False)
    if frame.get("width") is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(float(frame["width"]))
    if frame.get("color"):
        for spine in ax.spines.values():
            spine.set_color(frame["color"])


def refline_value(refline: dict):
    """基準線の位置 (時間軸の ISO 文字列は datetime64、数値軸は float)。render/scriptgen 共用。"""
    v = refline.get("value", 0.0)
    return np.datetime64(v) if isinstance(v, str) else float(v)


def refline_kwargs(refline: dict) -> dict:
    """基準線 (axvline / axhline) の kwargs (render/scriptgen 共用)。"""
    kw = {
        "color": refline.get("color", "#000000"),
        "linewidth": float(refline.get("linewidth", 1.0)),
        "linestyle": refline.get("linestyle", "dashed"),
        "alpha": float(refline.get("alpha", 1.0)),
    }
    if refline.get("label"):
        kw["label"] = refline["label"]
    return kw


def refline_text_kwargs(refline: dict) -> dict:
    """基準線の端に添える文字の kwargs (色は未指定なら線と同色。render/scriptgen 共用)。"""
    kw = {"color": refline.get("text_color") or refline.get("color", "#000000")}
    if refline.get("text_fontsize"):
        kw["fontsize"] = refline["text_fontsize"]
    return kw


def _draw_refline(ax, refline: dict) -> None:
    """基準線 1 本 (+ 端の文字) を描く。scriptgen._frame_background_lines と1対1対応。

    縦線の文字は上端 (axes 座標 0.98) の左に 90° 回転で、横線の文字は右端
    (axes 座標 0.98) の上に置く (位置は x/y の片方だけデータ座標の blended 変換)。
    """
    v = refline_value(refline)
    vertical = refline.get("orientation", "y") == "x"
    if vertical:
        ax.axvline(v, **refline_kwargs(refline))
    else:
        ax.axhline(v, **refline_kwargs(refline))
    if refline.get("text"):
        tkw = refline_text_kwargs(refline)
        if vertical:
            ax.text(v, 0.98, refline["text"], transform=ax.get_xaxis_transform(),
                    rotation=90, ha="right", va="top", **tkw)
        else:
            ax.text(0.98, v, refline["text"], transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", **tkw)


def _apply_frame_background(ax, panel: dict):
    """axes 枠線 (spine) と背景色・塗り範囲・基準線を適用する (1次元・2次元プロット共通)。

    scriptgen._frame_background_lines と1対1対応。
    """
    _apply_frame(ax, panel.get("frame") or {})
    background = panel.get("background") or {}
    if background.get("color"):
        ax.set_facecolor(background["color"])
    # 範囲を指定した塗り (axvspan / axhspan)。zorder=0 でデータ・グリッドより背面。
    # linewidth=0 で縁取りを消す (既定では面と同色の edge が描かれ、半透明時に
    # 境界部だけ face と edge のアルファが重なって濃い線に見える)
    for span in background.get("spans") or []:
        kw = {"color": span.get("color", "#ffd7d7"),
              "alpha": float(span.get("alpha", 0.3)),
              "linewidth": 0, "zorder": 0}
        # 時間軸用: ISO 文字列は datetime64 として渡す (数値軸は float)
        lo, hi = (np.datetime64(v) if isinstance(v, str) else float(v)
                  for v in (span["lo"], span["hi"]))
        if span.get("orientation", "x") == "x":
            ax.axvspan(lo, hi, **kw)
        else:
            ax.axhspan(lo, hi, **kw)
    # 基準線 (axvline / axhline)。既定 zorder でデータの上に描く
    for refline in background.get("reflines") or []:
        _draw_refline(ax, refline)


def line_label_kwargs(axis_cfg: dict) -> dict:
    """1次元プロットの set_xlabel / set_ylabel 共通の kwargs (rotation 除く)。

    既定値のときは出力 dict に含めないので、matplotlib デフォルトのまま残る。
    rotation は軸ごとに違う既定値 (x=0, y=90) なので呼び出し側で個別に足す。
    scriptgen の同名関数と挙動を揃えること。
    """
    kw = {}
    if axis_cfg.get("label_fontsize"):
        kw["fontsize"] = axis_cfg["label_fontsize"]
    if axis_cfg.get("label_color"):
        kw["color"] = axis_cfg["label_color"]
    weight = axis_cfg.get("label_weight", "normal")
    if weight and weight != "normal":
        kw["fontweight"] = weight
    if axis_cfg.get("label_italic"):
        kw["fontstyle"] = "italic"
    if axis_cfg.get("label_pad") is not None:
        kw["labelpad"] = float(axis_cfg["label_pad"])
    return kw


def _draw_stackplot_layer(ax, panel: dict, layer: dict, datasets: dict):
    """ax.stackplot で複数変数を積み上げ塗り分け描画する。"""
    style = layer["style"]
    variables = layer.get("variables") or []
    if not variables:
        return
    arrays = []
    x_vals = None
    for var in variables:
        da = _line_layer_data(panel, layer, var, datasets)
        da = apply_value_transform(da, style)
        if x_vals is None:
            x_vals = da[panel["x_dim"]].values
        arrays.append(da.values)
    kwargs = {"alpha": float(style.get("alpha", 0.8)),
              "baseline": style.get("baseline", "zero")}
    if style.get("show_labels_in_legend", True):
        kwargs["labels"] = list(variables)
    colors = style.get("colors")
    if colors and len(colors) == len(variables):
        kwargs["colors"] = list(colors)
    ax.stackplot(x_vals, *arrays, **kwargs)


def _auto_bar_width(x_vals):
    """隣接 x の差分の中央値 × 0.8 を返す (numeric は float、datetime は timedelta64)。"""
    if len(x_vals) < 2:
        return 1.0
    diffs = np.diff(x_vals)
    if np.issubdtype(x_vals.dtype, np.datetime64):
        return np.median(diffs) * 0.8
    return float(np.median(diffs)) * 0.8


def _compute_bar_params(bar_data, bar_mode, dodge_gap: float = 0.05):
    """bar レイヤー全体に渡る配置パラメータを計算する。

    bar_data: [(x_vals, y_vals, layer), ...] (panel 内出現順)。
    bar_mode: "overlap" / "dodge" / "stack"。
    dodge_gap: dodge モードでの隣接棒間の隙間比率 (0..0.5)。

    返り値: 同じ長さの dict のリスト。各 dict は
        {"width": float|timedelta64, "offset": 0|float|timedelta64,
         "bottom": None|np.ndarray}
    で、`_draw_bar_layer` がそのまま使える形。
    """
    n = len(bar_data)
    if n == 0:
        return []
    base_width = _auto_bar_width(bar_data[0][0])
    params = []
    if bar_mode == "dodge":
        # 各スロットの幅から両側に均等な隙間を取って棒幅を縮める。
        # オフセット間隔は per_slot のままなので隙間 = per_slot * dodge_gap
        gap = max(0.0, min(0.5, float(dodge_gap)))
        for k, (_x_vals, _y_vals, layer) in enumerate(bar_data):
            manual_w = layer["style"].get("width")
            full = manual_w if manual_w is not None else base_width
            per_slot = full / n
            per_width = per_slot * (1.0 - gap)
            offset = (k - (n - 1) / 2.0) * per_slot
            params.append({"width": per_width, "offset": offset, "bottom": None})
    elif bar_mode == "stack":
        # 正側と負側を別々に積む (matplotlib 慣例)
        y0 = np.asarray(bar_data[0][1], dtype=float)
        cum_pos = np.zeros_like(y0)
        cum_neg = np.zeros_like(y0)
        for _x_vals, y_vals, layer in bar_data:
            y_arr = np.asarray(y_vals, dtype=float)
            bottom = np.where(y_arr >= 0, cum_pos, cum_neg).copy()
            manual_w = layer["style"].get("width")
            w = manual_w if manual_w is not None else base_width
            params.append({"width": w, "offset": 0, "bottom": bottom})
            cum_pos = cum_pos + np.where(y_arr > 0, y_arr, 0.0)
            cum_neg = cum_neg + np.where(y_arr < 0, y_arr, 0.0)
    else:  # overlap (default)
        for _x_vals, _y_vals, layer in bar_data:
            manual_w = layer["style"].get("width")
            w = manual_w if manual_w is not None else base_width
            params.append({"width": w, "offset": 0, "bottom": None})
    return params


def _bar_kwargs(style):
    """ax.bar / ax.barh の共通 kwargs (color/alpha/edgecolor/linewidth/hatch/label)。

    matplotlib では edgecolor が枠線色とハッチ色を兼ねるため:
    - 枠線 on (edge_linewidth>0) のとき: edgecolor = edge_color (ハッチも同じ色)
    - 枠線 off + ハッチ on のとき: edgecolor = hatch_color、linewidth=0 で枠線を消す
    - ハッチ off のとき: hatch_color は使わない
    """
    kwargs = {"color": style.get("color", "#1f77b4"),
              "alpha": float(style.get("alpha", 0.8))}
    edge_on = float(style.get("edge_linewidth", 0.0)) > 0
    hatch_on = bool(style.get("hatch_pattern"))
    if edge_on:
        kwargs["edgecolor"] = style.get("edge_color", "#000000")
        kwargs["linewidth"] = float(style["edge_linewidth"])
    elif hatch_on:
        kwargs["edgecolor"] = style.get("hatch_color", "#000000")
        kwargs["linewidth"] = 0.0
    if hatch_on:
        kwargs["hatch"] = (str(style["hatch_pattern"])
                            * int(style.get("hatch_density", 3)))
    if style.get("label"):
        kwargs["label"] = style["label"]
    return kwargs


def _bar_error_values(panel: dict, layer: dict, datasets: dict):
    """エラーバーの値を返す (浮動小数のスカラ or 1D 配列 or None)。

    source="variable" のとき layer.selection / cyclic 系を引き継いで err 変数を取り、
    |err| に正規化して返す。source="constant" のときは abs(constant)。
    none/未指定なら None を返す。どちらも値の変換 (value_scale / value_offset) は
    掛けない — 誤差は図に表示する単位で用意する前提 (2026-09-29。それまでは変数の
    ときだけ a × err + b を掛けていた = KNOWN_ISSUES.md KI-1)。
    """
    err_cfg = layer["style"].get("errorbar", {})
    src = err_cfg.get("source", "none")
    if src == "variable":
        var = err_cfg.get("variable")
        if not var:
            return None
        err_da = _line_layer_data(panel, layer, var, datasets)
        return np.abs(err_da.values)
    if src == "constant":
        c = float(err_cfg.get("constant", 0.0))
        return abs(c) if c != 0 else None
    return None


def _draw_bar_layer(ax, panel: dict, layer: dict, datasets: dict,
                     x_vals, y_vals, params: dict):
    """単一の bar レイヤーを描画する。params は `_compute_bar_params` が計算した dict。"""
    style = layer["style"]
    orientation = style.get("orientation", "vertical")
    width = params["width"]
    offset = params["offset"]
    bottom = params["bottom"]
    kwargs = _bar_kwargs(style)
    positions = x_vals + offset if not (isinstance(offset, (int, float)) and offset == 0) else x_vals
    # エラーバー (yerr / xerr) を加える
    err_vals = _bar_error_values(panel, layer, datasets)
    if err_vals is not None:
        err_cfg = style.get("errorbar", {})
        err_axis = "xerr" if orientation == "horizontal" else "yerr"
        kwargs[err_axis] = err_vals
        kwargs["ecolor"] = err_cfg.get("color", "#000000")
        kwargs["capsize"] = float(err_cfg.get("capsize", 3.0))
        elw = float(err_cfg.get("linewidth", 1.0))
        if elw != 1.0:
            kwargs["error_kw"] = {"elinewidth": elw}
    if orientation == "horizontal":
        if bottom is not None:
            kwargs["left"] = bottom
        ax.barh(positions, y_vals, height=width, **kwargs)
    else:
        if bottom is not None:
            kwargs["bottom"] = bottom
        ax.bar(positions, y_vals, width=width, **kwargs)


def _layer_data(panel: dict, layer: dict, variable: str, datasets: dict,
                cyclic: bool = False, dataset_id: str | None = None):
    """水平面図用にレイヤーの変数を切り出す。dataset_id で参照先を上書きできる
    (ベクトルの y 成分が別ファイルのとき。既定 = layer["dataset_id"])。"""
    ds = datasets[dataset_id or layer["dataset_id"]]
    roles = detect_coord_roles(ds)
    # panel.selection (時刻など) と layer.selection (鉛直レベルなど) をマージ。
    # 同じ dim を両方が指定したら layer 側を優先する。
    selection = {**(panel.get("selection") or {}), **(layer.get("selection") or {})}
    da = select_panel_data(ds, variable, selection,
                           panel.get("region"), roles)
    if cyclic and needs_cyclic_lon(da, roles["lon"]):
        da = add_cyclic_lon(da, roles["lon"])
    return da, roles


def layer_averaging_stats(panel: dict, layer: dict, datasets: dict) -> list[dict]:
    """レイヤーの各変数について、範囲平均に使ったブロックの要素数と欠損数を返す。

    描画と同じデータ経路 (_section_data / _line_layer_data / _bundle_layer_data) を
    record 付きで通し、apply_averages が記録した {"dims", "n_total", "n_nan"} に
    "variable" を添えて返す。平均の解決ロジックはここに持たない。図の直下の
    通知 (notes.collect_notes) が「平均範囲内の欠損」を出すための公開入口。
    範囲平均を適用しない plot_type / レイヤーでは空リスト (レイヤーに averages が
    無いときも空)。
    """
    ptype = panel.get("plot_type")
    kind = layer.get("kind")
    out: list[dict] = []

    def _run(variable, fn, *args, **kwargs):
        rec: list[dict] = []
        fn(*args, record=rec, **kwargs)
        out.extend({"variable": variable, **r} for r in rec)

    if ptype == "section_2d":
        if kind in ("vector", "stream"):
            _run(layer["u_variable"], _section_data, panel, layer, layer["u_variable"], datasets)
            _run(layer["v_variable"], _section_data, panel, layer, layer["v_variable"], datasets,
                 dataset_id=vector_v_dataset_id(layer))
        elif layer.get("variable"):
            _run(layer["variable"], _section_data, panel, layer, layer["variable"], datasets)
    elif ptype == "line_1d":
        if kind == "line_bundle":
            _run(layer["variable"], _bundle_layer_data, panel, layer, datasets)
        elif kind == "stackplot":
            for var in layer.get("variables") or []:
                _run(var, _line_layer_data, panel, layer, var, datasets)
        elif kind == "fill_between":
            _run(layer["variable"], _line_layer_data, panel, layer, layer["variable"], datasets)
            if layer.get("variable_upper"):
                _run(layer["variable_upper"], _line_layer_data, panel, layer,
                     layer["variable_upper"], datasets,
                     selection_override=layer.get("selection_upper") or {},
                     averages_override=layer.get("averages_upper") or {})
        elif layer.get("variable"):
            _run(layer["variable"], _line_layer_data, panel, layer, layer["variable"], datasets)
    return out


def layer_selected_data(panel: dict, layer: dict, datasets: dict,
                        variable: str | None = None,
                        dataset_id: str | None = None) -> xr.DataArray:
    """レイヤーの変数を render と同じ手順で切り出す (描画軸2次元の DataArray)。

    panel に x_dim / y_dim があれば断面図 (_section_data: selection →
    averages → ranges)、なければ水平面図 (_layer_data: selection → region。
    cyclic 点は付けない)。variable / dataset_id の既定はレイヤーの描画変数 /
    dataset (ベクトルの成分などで上書き)。UI の既定値・参考表示 (データ範囲)
    が描画と同じ断面 — 選択中の時刻・レベル・領域・平均 — を見るための公開
    入口。panel は selection / region / ranges / x_dim / y_dim だけを持つ
    部分的な dict でもよい。
    """
    var = layer["variable"] if variable is None else variable
    if "x_dim" in panel and "y_dim" in panel:
        return _section_data(panel, layer, var, datasets, dataset_id=dataset_id)
    return _layer_data(panel, layer, var, datasets, dataset_id=dataset_id)[0]


def layer_drawn_data(panel: dict, layer: dict, datasets: dict) -> xr.DataArray:
    """fill / contour / hatch が matplotlib に渡す直前の DataArray を返す。

    layer_selected_data に値の変換と maskout (別変数によるものも含む) を
    掛けたもので、_draw_fill / _draw_contour / _draw_hatch とその断面版の
    前処理と同じ。UI の「データ範囲」「描画レベル」表示 (preview_levels) 用。
    """
    section = "x_dim" in panel and "y_dim" in panel
    style = layer["style"]
    da = apply_value_transform(layer_selected_data(panel, layer, datasets), style)
    return apply_maskout(da, style,
                         _maskout_mask_data(panel, layer, datasets, section=section))


def fill_zero_white_trim(extend: str, n: int) -> tuple[int, int]:
    """zero_white + int レベル + contourf 用のレベル端の切り落とし slice。

    データ範囲ちょうどの最外レベルは cartopy の contourf が退化ポリゴンの
    GEOS エラーで落ちることがある (実測。明示レベルでも同じ) ため、extend の
    拡張バンドがある側の端レベルを落とし、「レベルはデータ範囲の内側・
    両端の値は extend バンドへ」という matplotlib の自動レベルと同じ構図に
    する。lv[i0:i1] の (i0, i1) を返す (残りが2未満になるなら切らない)。
    render / scriptgen 共用。
    """
    i0 = 1 if extend in ("min", "both") else 0
    i1 = n - (1 if extend in ("max", "both") else 0)
    if i1 - i0 < 2:
        return 0, n
    return i0, i1


def _fill_pcolormesh_kwargs(style, da, contourf_levels=False):
    """pcolormesh + BoundaryNorm 用の cmap, norm を返す (離散描画)。

    fill_levels() がレベル数 (int) を返した場合はデータの min/max から linspace で
    具体的なレベル列を作る。contourf_levels=True (zero_white の contourf 経路)
    では int レベルの端を fill_zero_white_trim で切り落とす。
    """
    cmap_name = resolve_cmap(style.get("cmap", "viridis"),
                              style.get("reverse_cmap", False))
    cmap = plt.get_cmap(cmap_name)
    lv = fill_levels(style)
    if isinstance(lv, (int, np.integer)):
        # dict.get(default) は None 値を返すので明示的にチェック
        vmin = style.get("vmin")
        vmax = style.get("vmax")
        if vmin is None:
            vmin = float(da.min())
        if vmax is None:
            vmax = float(da.max())
        # 退化した自動範囲は matplotlib 深部の不可解なエラーになるため
        # 原因の分かるメッセージで先に止める (require_increasing_levels と同じ流儀)
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            raise RenderError("all_nan")
        if vmin == vmax:
            raise RenderError("constant_value", vmin=vmin)
        lv = np.linspace(vmin, vmax, int(lv))
    # zero_white の contourf は int レベル (自動範囲でも vmin/vmax 指定でも
    # linspace になる) の端を切り落とす。fill_levels が linspace 済みを返す
    # 経路もあるため、判定は style["levels"] が int かどうかで行う
    # (list = 境界値の直接指定は切らない。scriptgen 側と同条件)
    if (contourf_levels and style.get("zero_white")
            and isinstance(style.get("levels", 21), int)):
        i0, i1 = fill_zero_white_trim(style.get("extend", "both"), len(lv))
        lv = lv[i0:i1]
    extend = style.get("extend", "both")
    if style.get("zero_white"):
        colors = zero_white_colors(cmap, lv, extend)
        return (mcolors.ListedColormap(colors),
                mcolors.BoundaryNorm(lv, ncolors=len(colors), extend=extend))
    norm = mcolors.BoundaryNorm(lv, ncolors=cmap.N, extend=extend)
    return cmap, norm


# 塗りポリゴン (contourf / hatch) の「反転ガード」を掛ける投影。
# cartopy は極近傍の微小リング (実測: 88.7〜89.5°N で経度 0° をまたぐ 5 頂点の小片)
# を Orthographic に投影するとトレーサがリングを切り、切れ端を地図の境界に付け直して
# 円盤全体のポリゴンにしてしまう (地球回転アニメーションの1コマだけ陸ごと淡色に
# なる形で発覚 2026-08-28、cartopy_compat_notes.md「既知の上流バグ」)。
# 同じ形を経度 200° や日付変更線に動かすと起きないので位置依存の上流バグ。
# ガードは反転するリングだけを落とし、それ以外のピクセルは不変 (60 コマで確認)。
# 境界が曲線の投影で起きる現象だが、実測があるのは Orthographic のみなので
# 当面はそこだけに掛ける (見た目互換の方針: 他投影は従来どおり)。
INVERSION_GUARD_PROJECTIONS = ("Orthographic",)


def inversion_guard_needed(panel: dict) -> bool:
    """このパネルの塗りポリゴンに drop_inverting_contour_rings を掛けるか。"""
    return panel.get("projection", {}).get("name") in INVERSION_GUARD_PROJECTIONS


# NOTE: この関数の本体は scriptgen が inspect.getsource でそのまま再現スクリプトに
# 埋め込む (render ⇄ scriptgen を文字どおり同一コードにするため)。そのため
# docstring・コメントは英語、参照してよい名前は np / mpath / InterProjectionTransform
# と引数だけ (生成スクリプトの import と一致させること)。
def drop_inverting_contour_rings(cs, ax, src_crs):
    """Drop tiny filled-contour rings that cartopy projects into the whole map.

    Some very small rings (e.g. a 5-vertex sliver near 89N straddling 0E) are
    cut by cartopy's tracer when projected to a curved-boundary projection
    such as Orthographic; the pieces get attached to the map boundary and the
    ring comes back as the whole disk, painting the entire map in one colour.
    Candidate rings are those whose projected area is far below one pixel
    (< 1e-6 of the projection domain). They are projected together once per
    level; only if that yields a domain-sized polygon is the culprit located
    by bisection and removed. Rings that project correctly are never touched.
    """
    tr = InterProjectionTransform(src_crs, ax.projection)
    dom = ax.projection.domain.area

    def area(q):
        x, y = q[:, 0], q[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def max_ring_area(path):
        best = 0.0
        for q in tr.transform_path_non_affine(path).to_polygons(closed_only=False):
            q = np.asarray(q)
            if len(q) > 2 and np.isfinite(q).all():
                best = max(best, area(q))
        return best

    def joined(items):
        return mpath.Path(np.concatenate([v for v, _, _ in items]),
                          np.concatenate([c for _, c, _ in items]))

    def culprits(items):
        if max_ring_area(joined(items)) < 0.5 * dom:
            return []
        if len(items) == 1:
            return list(items)
        h = len(items) // 2
        return culprits(items[:h]) + culprits(items[h:])

    new_paths = []
    changed = False
    for path in cs.get_paths():
        v, c = path.vertices, path.codes
        if c is None or len(v) == 0:
            new_paths.append(path)
            continue
        starts = np.flatnonzero(c == mpath.Path.MOVETO)
        ends = list(starts[1:]) + [len(v)]
        xyz = ax.projection.transform_points(src_crs, v[:, 0], v[:, 1])
        tiny = []
        for k, (s, e) in enumerate(zip(starts, ends)):
            q = xyz[s:e, :2]
            if e - s >= 3 and np.isfinite(q).all() and area(q) < 1e-6 * dom:
                tiny.append((v[s:e], c[s:e], k))
        bad = {k for _, _, k in culprits(tiny)} if tiny else set()
        if not bad:
            new_paths.append(path)
            continue
        keep = [(v[s:e], c[s:e], k) for k, (s, e) in enumerate(zip(starts, ends))
                if k not in bad]
        new_paths.append(joined(keep) if keep else mpath.Path(np.zeros((0, 2))))
        changed = True
    if changed:
        cs.set_paths(new_paths)
    return cs


def _draw_fill(fig, ax, panel: dict, layer: dict, datasets: dict):
    """塗りつぶしレイヤー (地図)。カラーバーを返す (show=False なら None)。"""
    da, roles = _layer_data(panel, layer, layer["variable"], datasets, cyclic=True)
    da = _transform_and_mask(da, panel, layer, datasets, section=False)
    style = layer["style"]
    cf = _fill_artist(ax, da, roles["lon"], roles["lat"], style,
                      transform=ccrs.PlateCarree())
    if (style.get("method", "contourf") != "pcolormesh"
            and inversion_guard_needed(panel)):
        drop_inverting_contour_rings(cf, ax, ccrs.PlateCarree())
    _record_shared_mappable(ax, cf)
    return _add_colorbar(fig, ax, cf, style.get("colorbar", {}))


def _draw_hatch(ax, panel: dict, layer: dict, datasets: dict):
    da, roles = _layer_data(panel, layer, layer["variable"], datasets, cyclic=True)
    da = _transform_and_mask(da, panel, layer, datasets, section=False)
    hc = _hatch_artist(ax, da, roles["lon"], roles["lat"], layer["style"],
                       transform=ccrs.PlateCarree())
    if inversion_guard_needed(panel):
        drop_inverting_contour_rings(hc, ax, ccrs.PlateCarree())


def _draw_contour(fig, ax, panel: dict, layer: dict, datasets: dict):
    """等値線レイヤー (地図)。use_cmap + colorbar 設定ありならカラーバーを返す。"""
    da, roles = _layer_data(panel, layer, layer["variable"], datasets, cyclic=True)
    da = _transform_and_mask(da, panel, layer, datasets, section=False)
    return _contour_artist(fig, ax, da, roles["lon"], roles["lat"], layer["style"],
                           transform=ccrs.PlateCarree())


def _draw_quiverkey(ax, q, key: dict, zorder=None):
    """ベクトルキーを描画する。位置・ラベルの向きは設定で調整できる。

    labelpos='E' では矢印の右端が x に揃いラベルは右へ、'W' では矢印の
    左端が x に揃いラベルは左へ伸びる。QuiverKey は bbox_inches='tight' の
    余白計算に含まれないため、同じ範囲を占める透明テキストを併置して
    保存時にラベルが切れないようにする。scriptgen と同じ操作であること。
    """
    kx = key.get("x", 1.0)
    ky = key.get("y", -0.07)
    labelpos = key.get("labelpos", "E")
    label = key.get("label", "")
    # 文字サイズ未指定 (None) は kwargs 自体を渡さず既定に任せる
    # (旧 config の描画を変えない)。透明テキストにも同じサイズを適用し、
    # tight bbox が確保する余白をラベルの実寸に一致させる
    fontsize = key.get("fontsize")
    key_kw = {"fontproperties": {"size": fontsize}} if fontsize else {}
    if zorder is not None:
        key_kw["zorder"] = zorder
    text_kw = {"fontsize": fontsize} if fontsize else {}
    ax.quiverkey(q, kx, ky, key.get("length", 10.0), label, labelpos=labelpos,
                 **key_kw)
    pad = "      "
    label_side, arrow_side = ("left", "right") if labelpos == "E" else ("right", "left")
    ax.text(kx, ky, pad + label if labelpos == "E" else label + pad,
            transform=ax.transAxes, ha=label_side, va="center", alpha=0, **text_kw)
    ax.text(kx, ky, pad, transform=ax.transAxes, ha=arrow_side, va="center", alpha=0,
            **text_kw)


def _draw_vector(fig, ax, panel: dict, layer: dict, datasets: dict):
    """ベクトルレイヤー (地図)。use_cmap 時はカラーバーを返す (それ以外 None)。"""
    u, roles = _layer_data(panel, layer, layer["u_variable"], datasets)
    v, _ = _layer_data(panel, layer, layer["v_variable"], datasets,
                       dataset_id=vector_v_dataset_id(layer))
    check_vector_shapes(u, v, layer)
    style = layer["style"]
    # 間引きは dim 名で行う (1 次元格子 = (lat, lon)、curvilinear = (y, x))。
    # 座標配列 (u[lon] / u[lat]) は 2 次元のまま quiver に渡してよい
    ydim, xdim = horizontal_dims(datasets[layer["dataset_id"]], roles)
    u, v = _prepare_vector_uv(u, v, style, xdim, ydim)
    args, norm_kw = _vector_color_args(style, u, v)
    # quiver は座標変換時に2次元ブールマスクを適用するため DataArray ではなく
    # numpy 配列を渡す (xarray は2次元ブールインデックスを拒否する)
    with warnings.catch_warnings():
        # 領域の端 (極や経度の折り返し点) のベクトル回転が近似になるという
        # cartopy の通知。描画結果には実質影響しないため抑制する。
        warnings.filterwarnings(
            "ignore", message="Some vectors at source domain corners")
        q = ax.quiver(
            u[roles["lon"]].values, u[roles["lat"]].values, u.values, v.values,
            *args,
            **{**vector_kwargs(style), **norm_kw},
            transform=ccrs.PlateCarree(),
        )
    return _finish_vector(fig, ax, q, style, norm_kw,
                          key_zorder=(MAP_FG_ZORDER if panel_land_foreground(panel)
                                      else None))


def _stream_color_norm(style: dict, u, v):
    """streamplot の色付け引数 (color 配列 + norm) を解決する。

    use_cmap=True なら大きさ |V| の2次元配列を color に、離散化指定が
    あれば BoundaryNorm を norm に入れて返す (render/section 共通)。
    """
    color_kw = {}
    norm_kw = {}
    if style.get("use_cmap"):
        mag = np.hypot(u.values, v.values)
        color_kw = {"color": mag}
        norm_kw = discrete_color_kwargs(style, np.nanmin(mag), np.nanmax(mag))
    return color_kw, norm_kw


def _stream_colorbar(fig, ax, sp, style: dict, norm_kw: dict):
    """streamplot のカラーバー (mappable は sp.lines)。use_cmap 以外は None。"""
    if not style.get("use_cmap"):
        return None
    if not norm_kw and (style.get("vmin") is not None
                        or style.get("vmax") is not None):
        sp.lines.set_clim(style.get("vmin"), style.get("vmax"))
    return _add_colorbar(fig, ax, sp.lines, style.get("colorbar", {}),
                         extend=continuous_cbar_extend(style))


def _draw_stream(fig, ax, panel: dict, layer: dict, datasets: dict):
    """流線 (streamplot) レイヤー。use_cmap 時はカラーバーを返す (それ以外 None)。

    cartopy が「呼び出し時点の表示範囲」を等間隔グリッドに再サンプルして
    流線を計算するため、extent はレイヤー描画前に確定していること
    (map パネルは extent 設定をレイヤーループより前に置いている)。
    """
    u, roles = _layer_data(panel, layer, layer["u_variable"], datasets)
    v, _ = _layer_data(panel, layer, layer["v_variable"], datasets,
                       dataset_id=vector_v_dataset_id(layer))
    check_vector_shapes(u, v, layer)
    style = layer["style"]
    u = apply_value_transform(u, style)
    v = apply_value_transform(v, style)
    color_kw, norm_kw = _stream_color_norm(style, u, v)
    sp = ax.streamplot(
        u[roles["lon"]].values, u[roles["lat"]].values, u.values, v.values,
        **{**stream_kwargs(style), **color_kw, **norm_kw},
        transform=ccrs.PlateCarree(),
    )
    return _stream_colorbar(fig, ax, sp, style, norm_kw)


def _draw_stream_section(fig, ax, panel: dict, layer: dict, datasets: dict):
    """断面図の流線レイヤー。use_cmap 時はカラーバーを返す (それ以外 None)。"""
    u = _section_data(panel, layer, layer["u_variable"], datasets)
    v = _section_data(panel, layer, layer["v_variable"], datasets,
                      dataset_id=vector_v_dataset_id(layer))
    check_vector_shapes(u, v, layer)
    style = layer["style"]
    u = apply_value_transform(u, style)
    v = apply_value_transform(v, style)
    # streamplot は昇順・等間隔の座標を要求するため、必要な軸だけ
    # 等間隔グリッドへ線形補間する (気圧レベルなどの不等間隔軸)
    for dim in (panel["x_dim"], panel["y_dim"]):
        uni = stream_uniform_coords(u[dim].values)
        if uni is not None:
            u = u.interp({dim: uni})
            v = v.interp({dim: uni})
    color_kw, norm_kw = _stream_color_norm(style, u, v)
    sp = ax.streamplot(
        u[panel["x_dim"]].values, u[panel["y_dim"]].values,
        u.values, v.values,
        **{**stream_kwargs(style), **color_kw, **norm_kw},
    )
    return _stream_colorbar(fig, ax, sp, style, norm_kw)
