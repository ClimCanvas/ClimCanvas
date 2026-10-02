# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""描画セマンティクスの検証: 「入力データが意図どおり描かれているか」。

render⇄scriptgen の画像一致テスト (ルール1) は2経路の一致しか保証せず、
両方に共通するロジックの誤りは検出できない。ここでは `render_figure` が返す
Figure から matplotlib アーティストを取り出し、**numpy で独立に計算した期待値**
と数値で突き合わせる (docs/semantic_test_plan.md 段階1)。

原則:
- 合成データは「値がセル位置から一意に決まる」ように作り、期待値はテスト内で
  自力計算する。render.py のヘルパー (fill_levels 等) は使わない
  (同じバグを両側に持ち込まないための独立性)
- 画像を介さないので matplotlib のバージョン差に強い。アーティストの
  取り出し方が壊れたら「取り出し方」を直す
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.contour import ContourSet

from climcanvas.core import config as mc_config
from climcanvas.core import render as mc_render


# --- 合成データ (期待値を自力計算できる決定的な小データ) ---

def _map_ds() -> xr.Dataset:
    """非全球の小格子 (cyclic 点の追加が起きない)。値 = lat*10 + lon*0.1。"""
    lon = np.arange(100.0, 190.0, 10.0)  # 9 点 (全球でない → cyclic なし)
    lat = np.arange(0.0, 50.0, 10.0)     # 5 点
    val = lat[:, None] * 10.0 + lon[None, :] * 0.1
    u = np.arange(45, dtype=float).reshape(5, 9)          # セル毎に一意
    v = 2.0 * u + 1.0
    return xr.Dataset(
        {"f": (("lat", "lon"), val),
         "u": (("lat", "lon"), u),
         "v": (("lat", "lon"), v)},
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})})


def _series_ds(n: int = 500) -> xr.Dataset:
    """時系列2変数 (集計系用)。固定シードで決定的。"""
    rng = np.random.default_rng(11)
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(n) * np.timedelta64(6, "h"))
    a = rng.standard_normal(n) * 3.0 + 10.0
    b = rng.standard_normal(n) * 2.0 + 5.0
    return xr.Dataset({"a": (("time",), a), "b": (("time",), b)},
                      coords={"time": time})


def _line_ds() -> xr.Dataset:
    """1次元プロット用: level 座標に沿った既知の値。"""
    level = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0])
    g = np.array([15.0, 8.0, 2.0, -12.0, -40.0, -55.0])
    return xr.Dataset({"g": (("level",), g)},
                      coords={"level": ("level", level, {"units": "hPa"})})


def _render(cfg_panels) -> "plt.Figure":
    cfg = mc_config.default_figure_config()
    cfg["panels"] = cfg_panels if isinstance(cfg_panels, list) else [cfg_panels]
    return mc_render.render_figure(cfg, _DATASETS)


def _quadmesh(ax) -> QuadMesh:
    qms = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(qms) == 1, f"QuadMesh が1つのはず: {len(qms)}"
    return qms[0]


def _contour_sets(ax) -> list[ContourSet]:
    return [c for c in ax.collections if isinstance(c, ContourSet)]


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


_DATASETS: dict[str, xr.Dataset] = {}


@pytest.fixture(autouse=True)
def _datasets():
    _DATASETS.clear()
    _DATASETS.update({"ds0": None})
    yield


def _use(ds: xr.Dataset):
    _DATASETS["ds0"] = ds


# --- 1-1: fill (pcolormesh) — 配列・値変換・maskout・離散境界 ---

def test_fill_pcolormesh_array_transform_maskout_levels():
    ds = _map_ds()
    _use(ds)
    panel = mc_config.default_panel()
    levels = [50.0, 120.0, 200.0, 320.0, 480.0]
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"method": "pcolormesh", "levels": levels,
                          "value_scale": 2.0, "value_offset": 10.0,
                          "maskout": {"below": 100.0, "above": None}})
    panel["layers"] = [fill]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])

    # 期待値: 変換 (2x+10) → 100 以下を NaN (maskout は変換後の値)
    expected = ds["f"].values * 2.0 + 10.0
    expected = np.where(expected <= 100.0, np.nan, expected)
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                     dtype=float).reshape(expected.shape)
    np.testing.assert_allclose(got, expected, equal_nan=True)
    # 離散化境界 = 指定レベルそのもの
    np.testing.assert_allclose(qm.norm.boundaries, levels)


# --- 1-1b: maskout の「別変数の値でマスク」 ---

def test_fill_pcolormesh_maskout_by_variable():
    """別変数 u の生の値による条件が f に掛かる (自身の閾値との併用・適用順)。

    自身の閾値は値変換**後**、マスク変数の閾値は**生の値** — この非対称を
    numpy の独立計算で固定する。
    """
    ds = _map_ds()
    _use(ds)
    panel = mc_config.default_panel()
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"method": "pcolormesh", "levels": [0.0, 200.0, 480.0],
                          "value_scale": 2.0, "value_offset": 10.0,
                          "maskout": {"below": 100.0, "above": None,
                                      "variable": "u",
                                      "var_below": 10.0, "var_above": 40.0}})
    panel["layers"] = [fill]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])

    expected = ds["f"].values * 2.0 + 10.0
    expected = np.where(expected <= 100.0, np.nan, expected)  # 自身: 変換後の値
    u = ds["u"].values                                        # 別変数: 生の値
    expected = np.where(u <= 10.0, np.nan, expected)
    expected = np.where(u >= 40.0, np.nan, expected)
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                     dtype=float).reshape(expected.shape)
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_fill_maskout_var_guards():
    """マスク変数の不在・次元不整合は原因の分かる RenderError で止まる。"""
    ds = _map_ds()
    ds = ds.assign(w=ds["f"].expand_dims(member=[0, 1]),  # f に無い次元を持つ
                   zonal=ds["f"].mean("lon"))             # 描画軸 lon が無い
    _use(ds)

    def _cfg(mvar):
        panel = mc_config.default_panel()
        fill = mc_config.default_fill_layer("ds0", "f")
        fill["style"]["maskout"].update({"variable": mvar, "var_below": 0.0})
        panel["layers"] = [fill]
        return panel

    for mvar, msg_id in [("nope", "maskout_var_missing"),
                         ("w", "maskout_var_extra_dims"),
                         ("zonal", "maskout_var_missing_dims")]:
        with pytest.raises(mc_render.RenderError) as ei:
            _render(_cfg(mvar))
        assert ei.value.msg_id == msg_id, mvar


# --- 1-2b: contourf の手動範囲はレベルをトリムしない (意図した挙動の固定) ---

def test_fill_contourf_manual_range_keeps_exact_levels():
    """手動 vmin/vmax + int レベル数の contourf のレベル = linspace そのまま。

    GEOS 退化ポリゴン対策の端トリムを一度実装したが、既存図の見た目が
    変わるため取りやめた (2026-08-23)。指定した範囲がそのままレベルの端に
    なることを固定する (クラッシュ時は render_figure が degenerate_geometry
    の可読エラーに変換する — test_consistency 側で検証)。
    """
    ds = _map_ds()
    _use(ds)
    vmin, vmax = 10.0, 400.0
    panel = mc_config.default_panel()
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"vmin": vmin, "vmax": vmax, "levels": 21})
    panel["layers"] = [fill]
    fig = _render(panel)
    cs = [c for c in fig.axes[0].collections
          if isinstance(c, ContourSet) and c.filled]
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0].levels, np.linspace(vmin, vmax, 21))


# --- fill の alpha — pcolormesh / contourf に反映、既定は None (上書きしない) ---

def test_fill_alpha_applied_and_default_untouched():
    _use(_map_ds())

    def _fig(**style):
        panel = mc_config.default_panel()
        fill = mc_config.default_fill_layer("ds0", "f")
        fill["style"].update(style)
        panel["layers"] = [fill]
        return _render(panel)

    # pcolormesh: QuadMesh にそのまま
    fig = _fig(method="pcolormesh", levels=[100.0, 200.0, 300.0], alpha=0.4)
    assert _quadmesh(fig.axes[0]).get_alpha() == pytest.approx(0.4)
    # contourf: ContourSet にそのまま
    fig = _fig(levels=[100.0, 200.0, 300.0], alpha=0.4)
    assert _contour_sets(fig.axes[0])[0].get_alpha() == pytest.approx(0.4)
    # 既定 (1.0) は alpha を渡さない = None のまま (cmap の alpha を上書きしない)
    fig = _fig(levels=[100.0, 200.0, 300.0])
    assert _contour_sets(fig.axes[0])[0].get_alpha() is None


# --- 1-2: fill (contourf) — レベルと extend ---

def test_fill_contourf_levels_and_extend():
    _use(_map_ds())
    panel = mc_config.default_panel()
    levels = [100.0, 200.0, 300.0, 400.0]
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"method": "contourf", "levels": levels,
                          "extend": "max"})
    panel["layers"] = [fill]
    fig = _render(panel)
    cs = _contour_sets(fig.axes[0])
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0].levels, levels)
    assert cs[0].extend == "max"


# --- 1-3 / 1-4: contour — 新スキーマ (fill 式) と旧スキーマ (interval) ---

def test_contour_levels_new_schema_and_colorbar():
    _use(_map_ds())
    panel = mc_config.default_panel()
    levels = [80.0, 160.0, 240.0, 400.0]
    contour = mc_config.default_contour_layer("ds0", "f")
    contour["style"].update({"levels": levels, "use_cmap": True,
                             "extend": "both"})
    panel["layers"] = [contour]
    fig = _render(panel)
    cs = _contour_sets(fig.axes[0])
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0].levels, levels)
    assert cs[0].extend == "both"
    # use_cmap + 新スキーマはカラーバー (2つ目の axes) が付く
    assert len(fig.axes) == 2


def test_contour_levels_legacy_interval_schema():
    _use(_map_ds())
    panel = mc_config.default_panel()
    contour = mc_config.default_contour_layer("ds0", "f")
    contour["style"].update({"vmin": 100.0, "vmax": 400.0, "interval": 50.0})
    panel["layers"] = [contour]
    fig = _render(panel)
    cs = _contour_sets(fig.axes[0])
    assert len(cs) == 1
    # 旧スキーマは levels(=11 既定) より interval が優先: arange(100, 400, 50)
    np.testing.assert_allclose(cs[0].levels, np.arange(100.0, 425.0, 50.0))
    # 単色モードはカラーバーなし
    assert len(fig.axes) == 1


# --- 1-5: line_1d — Line2D の (x, y) = 座標と変数値 ---

def test_line_1d_data_matches_input():
    ds = _line_ds()
    _use(ds)
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "level"
    panel["selection"] = {}
    line = mc_config.default_line_layer("ds0", "g")
    panel["layers"] = [line]
    fig = _render(panel)
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    xy = ax.lines[0].get_xydata()
    np.testing.assert_allclose(xy[:, 0], ds["level"].values)
    np.testing.assert_allclose(xy[:, 1], ds["g"].values)


# --- 1-13: 軸セマンティクス (invert / log / 目盛位置) ---

def test_axis_invert_log_and_tick_positions():
    _use(_line_ds())
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "level"
    panel["selection"] = {}
    panel["axis"].update({"invert_x": True, "log_y": False,
                          "x_tick_positions": [200.0, 500.0, 850.0]})
    line = mc_config.default_line_layer("ds0", "g")
    panel["layers"] = [line]
    fig = _render(panel)
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    assert xlim[0] > xlim[1], "invert_x が反映されていない"
    np.testing.assert_allclose(sorted(ax.get_xticks()),
                               [200.0, 500.0, 850.0])


def test_axis_log_y_on_section():
    _use(_section_ds())
    panel = _section_panel()
    panel["axis"]["log_y"] = True
    fig = _render(panel)
    assert fig.axes[0].get_yscale() == "log"


def _section_ds() -> xr.Dataset:
    """鉛直断面用 (lon × level)。値 = level*0.01 + lon。"""
    lon = np.arange(100.0, 190.0, 10.0)
    level = np.array([1000.0, 700.0, 500.0, 300.0, 100.0])
    val = level[:, None] * 0.01 + lon[None, :]
    return xr.Dataset(
        {"s": (("level", "lon"), val)},
        coords={"lon": ("lon", lon, {"units": "degrees_east"}),
                "level": ("level", level, {"units": "hPa"})})


def _section_panel() -> dict:
    panel = mc_config.default_section_panel()
    panel["x_dim"], panel["y_dim"] = "lon", "level"
    panel["selection"] = {}
    fill = mc_config.default_fill_layer("ds0", "s")
    fill["style"]["method"] = "pcolormesh"
    panel["layers"] = [fill]
    return panel


# --- 1-6: scatter_2d — offsets = (x, y) ペア ---

def test_scatter_offsets_match_variables():
    ds = _series_ds(60)
    _use(ds)
    panel = mc_config.default_scatter_panel()
    panel["x_variable"], panel["y_variable"] = "a", "b"
    sc = mc_config.default_scatter_layer("ds0")
    sc["drawing_dim"] = "time"
    panel["layers"] = [sc]
    fig = _render(panel)
    pcs = fig.axes[0].collections
    assert len(pcs) == 1
    offsets = np.asarray(pcs[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], ds["a"].values)
    np.testing.assert_allclose(offsets[:, 1], ds["b"].values)


# --- 1-7: bubble — サイズ正規化・色配列・離散化境界 ---

def test_bubble_sizes_colors_and_discrete_norm():
    ds = _series_ds(60)
    ds["w"] = ds["a"] * 0.5 + 3.0  # z 変数
    _use(ds)
    panel = mc_config.default_scatter_panel()
    panel["x_variable"], panel["y_variable"] = "a", "b"
    panel["z_variable"] = "w"
    bb = mc_config.default_bubble_layer("ds0")
    bb["drawing_dim"] = "time"
    bb["style"].update({"size_min": 20.0, "size_max": 300.0,
                        "use_cmap": True, "levels": 6})
    panel["layers"] = [bb]
    fig = _render(panel)
    pc = fig.axes[0].collections[0]
    w = ds["w"].values
    sizes = np.asarray(pc.get_sizes())
    np.testing.assert_allclose(sizes.min(), 20.0)
    np.testing.assert_allclose(sizes.max(), 300.0)
    # サイズは z の線形正規化 (最小値→size_min、最大値→size_max)
    expected_sizes = 20.0 + (w - w.min()) / (w.max() - w.min()) * 280.0
    np.testing.assert_allclose(sizes, expected_sizes)
    np.testing.assert_allclose(np.asarray(pc.get_array(), dtype=float), w)
    # 離散化境界 = z の min/max を 6 等分
    np.testing.assert_allclose(pc.norm.boundaries,
                               np.linspace(w.min(), w.max(), 6))


# --- 1-8: vector — 間引き・mask_below・離散化境界 ---

def test_vector_stride_mask_and_discrete_norm():
    ds = _map_ds()
    _use(ds)
    panel = mc_config.default_panel()
    vec = mc_config.default_vector_layer("ds0", "u", "v")
    vec["style"].update({"stride_x": 2, "stride_y": 2, "mask_below": 30.0,
                         "use_cmap": True, "levels": 5})
    vec["style"]["key"]["show"] = False
    panel["layers"] = [vec]
    fig = _render(panel)
    from matplotlib.quiver import Quiver
    qs = [c for c in fig.axes[0].collections if isinstance(c, Quiver)]
    assert len(qs) == 1
    q = qs[0]
    u = ds["u"].values[::2, ::2]
    v = ds["v"].values[::2, ::2]
    mag = np.hypot(u, v)
    exp_u = np.where(mag <= 30.0, np.nan, u).ravel()
    # Quiver はマスク位置の U を 1 で埋め、マスク自体は Umask に別持ちする
    got_u = np.asarray(q.U, dtype=float).ravel().copy()
    umask = getattr(q, "Umask", None)
    if umask is not None and umask is not np.ma.nomask:
        got_u[np.asarray(umask).ravel()] = np.nan
    np.testing.assert_allclose(got_u, exp_u, equal_nan=True)
    # 離散化境界 = |V| (マスク後) の min/max を 5 等分
    mag_masked = np.where(mag <= 30.0, np.nan, mag)
    np.testing.assert_allclose(
        q.norm.boundaries,
        np.linspace(np.nanmin(mag_masked), np.nanmax(mag_masked), 5))


# --- 1-48: ベクトルの y 成分を別 dataset から (v_dataset_id) ---

def _quiver(fig):
    from matplotlib.quiver import Quiver
    qs = [c for c in fig.axes[0].collections if isinstance(c, Quiver)]
    assert len(qs) == 1
    return qs[0]


def test_vector_components_from_separate_datasets():
    """x 成分は ds0、y 成分は ds1 (v_dataset_id) から読み、quiver の U / V が
    それぞれの dataset の生値と一致すること。ds0 には y 成分の変数が無く、
    ds1 の変数名も別 (vwnd) なので、取り違えると KeyError になる。"""
    ds = _map_ds()
    _use(ds[["u"]])
    _DATASETS["ds1"] = ds[["v"]].rename({"v": "vwnd"})
    panel = mc_config.default_panel()
    vec = mc_config.default_vector_layer("ds0", "u", "vwnd", v_dataset_id="ds1")
    vec["style"].update({"stride_x": 1, "stride_y": 1})
    vec["style"]["key"]["show"] = False
    panel["layers"] = [vec]
    q = _quiver(_render(panel))
    # cartopy の transform_vectors が回転行列を掛けるため 0 が 1e-17 程度になる
    np.testing.assert_allclose(np.asarray(q.U, dtype=float).ravel(),
                               ds["u"].values.ravel(), atol=1e-9)
    np.testing.assert_allclose(np.asarray(q.V, dtype=float).ravel(),
                               ds["v"].values.ravel(), atol=1e-9)


def test_vector_v_dataset_id_none_falls_back_to_same_dataset():
    """v_dataset_id が None (UI で「x成分と同じファイル」) / キー無し (旧設定) の
    どちらでも x 成分と同じ dataset から y 成分を読む (旧図の互換)。"""
    ds = _map_ds()
    _use(ds)
    _DATASETS["ds1"] = ds[["u"]]   # v が無い。誤ってこちらを見ると KeyError
    for legacy in (False, True):
        panel = mc_config.default_panel()
        vec = mc_config.default_vector_layer("ds0", "u", "v")
        if legacy:
            del vec["v_dataset_id"]
        vec["style"].update({"stride_x": 1, "stride_y": 1})
        vec["style"]["key"]["show"] = False
        panel["layers"] = [vec]
        q = _quiver(_render(panel))
        np.testing.assert_allclose(np.asarray(q.V, dtype=float).ravel(),
                                   ds["v"].values.ravel(), atol=1e-9)
        plt.close("all")


def test_vector_components_shape_mismatch_raises():
    """別 dataset の y 成分が x 成分と違う格子なら、broadcast エラーではなく
    msg_id 付きの RenderError (vector_shape_mismatch) で止まること。"""
    ds = _map_ds()
    _use(ds[["u"]])
    _DATASETS["ds1"] = ds[["v"]].isel(lon=slice(0, 5)).rename({"v": "vwnd"})
    panel = mc_config.default_panel()
    vec = mc_config.default_vector_layer("ds0", "u", "vwnd", v_dataset_id="ds1")
    panel["layers"] = [vec]
    with pytest.raises(mc_render.RenderError) as ei:
        _render(panel)
    assert ei.value.msg_id == "vector_shape_mismatch"


# --- 1-9: hist — 棒の高さ = np.histogram ---

def test_hist_bar_heights_match_numpy():
    ds = _series_ds(300)
    _use(ds)
    panel = mc_config.default_dist_panel()
    hist = mc_config.default_hist_layer("ds0")
    hist["variable"] = "a"
    hist["agg_dim"] = "time"
    hist["style"]["bins"] = 12
    panel["layers"] = [hist]
    fig = _render(panel)
    heights = np.array([p.get_height() for p in fig.axes[0].patches])
    expected, _ = np.histogram(ds["a"].values, bins=12)
    np.testing.assert_allclose(heights, expected)


# --- 1-10: hist2d — 配列 = np.histogram2d、離散化境界 = 集計結果 ---

def test_hist2d_counts_match_numpy_and_discrete_norm():
    ds = _series_ds(500)
    _use(ds)
    panel = mc_config.default_agg_panel()
    panel["x_variable"], panel["y_variable"] = "a", "b"
    ly = mc_config.default_hist2d_layer("ds0")
    ly["style"].update({"bins_x": 14, "bins_y": 9, "levels": 5})
    panel["layers"] = [ly]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])

    h, _, _ = np.histogram2d(ds["a"].values, ds["b"].values, bins=[14, 9])
    h_masked = np.where(h < 1.0, np.nan, h)  # hide_zeros (cmin=1) 相当
    expected = h_masked.T                     # hist2d は h.T を pcolormesh する
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                     dtype=float).reshape(expected.shape)
    np.testing.assert_allclose(got, expected, equal_nan=True)
    # 後付け離散化: 境界 = 集計結果 (マスク後) の min/max を 5 等分
    np.testing.assert_allclose(
        qm.norm.boundaries,
        np.linspace(np.nanmin(h_masked), np.nanmax(h_masked), 5))


# --- 1-11: hexbin — 離散化境界 (直接指定) と extend ---

def test_hexbin_discrete_norm_boundaries_and_extend():
    ds = _series_ds(500)
    _use(ds)
    panel = mc_config.default_agg_panel()
    panel["x_variable"], panel["y_variable"] = "a", "b"
    ly = mc_config.default_hexbin_layer("ds0")
    levels = [1.0, 2.0, 5.0, 10.0, 20.0]
    ly["style"].update({"gridsize": 15, "levels": levels, "extend": "max"})
    panel["layers"] = [ly]
    fig = _render(panel)
    from matplotlib.collections import PolyCollection
    hbs = [c for c in fig.axes[0].collections
           if isinstance(c, PolyCollection)]
    assert len(hbs) == 1
    hb = hbs[0]
    np.testing.assert_allclose(hb.norm.boundaries, levels)
    assert hb.norm.extend == "max"
    # ビンの点数の合計 = 標本数 (mincnt=1 なので全点がどこかのビンにいる)
    assert float(np.asarray(hb.get_array()).sum()) == 500.0


# --- 1-12: box — 中央値・ひげ = np.percentile ---

def test_box_median_and_percentile_whiskers():
    ds = _series_ds(300)
    _use(ds)
    panel = mc_config.default_dist_panel()
    box = mc_config.default_box_layer("ds0")
    box["variable"] = "a"
    box["agg_dim"] = "time"
    box["style"].update({"whis": [5.0, 95.0], "showfliers": False})
    panel["layers"] = [box]
    fig = _render(panel)
    vals = ds["a"].values
    p50 = np.percentile(vals, 50.0)
    lo_bound, hi_bound = np.percentile(vals, [5.0, 95.0])
    # matplotlib (boxplot_stats) はひげをパーセンタイル境界そのものではなく、
    # 境界の内側にある最も外側の**実データ点**で止める
    cap_lo = vals[vals >= lo_bound].min()
    cap_hi = vals[vals <= hi_bound].max()
    # ax.lines のどれかが中央値線 (ydata が一定で p50)、
    # キャップ (ydata 一定で cap_lo / cap_hi) になっている
    const_ys = set()
    for ln in fig.axes[0].lines:
        y = np.asarray(ln.get_ydata(), dtype=float)
        if y.size and np.allclose(y, y[0]):
            const_ys.add(round(float(y[0]), 6))
    for target, name in ((p50, "median"), (cap_lo, "cap(5%)"),
                         (cap_hi, "cap(95%)")):
        assert any(abs(cy - target) < 1e-6 for cy in const_ys), \
            f"{name} = {target} の水平線が見つからない: {sorted(const_ys)}"


# --- 1-14: stream — clim・離散化境界・色値の範囲 ---

def test_stream_clim_discrete_norm_and_color_range():
    ds = _map_ds()
    _use(ds)
    from matplotlib.collections import LineCollection

    def _stream_lines(style_update):
        panel = mc_config.default_panel()
        sp = mc_config.default_stream_layer("ds0", "u", "v")
        sp["style"].update({"use_cmap": True, **style_update})
        panel["layers"] = [sp]
        fig = _render(panel)
        lcs = [c for c in fig.axes[0].collections
               if isinstance(c, LineCollection)]
        assert len(lcs) == 1
        return lcs[0]

    # 手動 vmin/vmax → set_clim がそのまま norm に入る
    lines = _stream_lines({"vmin": 5.0, "vmax": 55.0})
    assert (lines.norm.vmin, lines.norm.vmax) == (5.0, 55.0)
    plt.close("all")

    # 離散化 (直接指定) → BoundaryNorm 境界。色値 (|V| の線分サンプル) は
    # 線形補間なので入力格子の |V| の min/max の範囲内に収まる
    levels = [0.0, 20.0, 40.0, 80.0]
    lines = _stream_lines({"levels": levels})
    np.testing.assert_allclose(lines.norm.boundaries, levels)
    mag = np.hypot(ds["u"].values, ds["v"].values)
    colors = np.asarray(lines.get_array(), dtype=float)
    assert colors.size > 0
    assert colors.min() >= mag.min() - 1e-9
    assert colors.max() <= mag.max() + 1e-9


# --- 1-15: track — 点の座標・色付け値・線の本数 ---

def _track_ds() -> xr.Dataset:
    """2ストーム (2本目は末尾 NaN パディング) の合成トラック。"""
    lon = np.array([[130., 132., 134., 136., 138., 140.],
                    [150., 151., 152., 153., np.nan, np.nan]])
    lat = np.array([[10., 12., 14., 16., 18., 20.],
                    [5., 7., 9., 11., np.nan, np.nan]])
    wind = np.array([[30., 40., 50., 60., 70., 80.],
                     [20., 25., 30., 35., np.nan, np.nan]])
    return xr.Dataset(
        {"lon": (("storm", "obs"), lon, {"units": "degrees_east"}),
         "lat": (("storm", "obs"), lat, {"units": "degrees_north"}),
         "wind": (("storm", "obs"), wind, {"units": "kt"})})


def test_track_points_coords_values_and_line_count():
    ds = _track_ds()
    _use(ds)
    panel = mc_config.default_panel()
    tr = mc_config.default_track_layer("ds0")
    tr.update({"lon_var": "lon", "lat_var": "lat", "storm_dim": "storm"})
    tr["style"]["points"].update({"variable": "wind", "every": 2,
                                  "value_scale": 0.5})
    panel["layers"] = [tr]
    fig = _render(panel)
    ax = fig.axes[0]
    # 線はストーム毎に1本
    assert len(ax.lines) == 2
    from matplotlib.collections import PathCollection
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(pcs) == 1
    # 期待値: 各ストームを every=2 で間引き → 連結 → 有限のみ
    lons, lats, vals = [], [], []
    for i in range(2):
        lons.append(ds["lon"].values[i, ::2])
        lats.append(ds["lat"].values[i, ::2])
        vals.append(ds["wind"].values[i, ::2] * 0.5)  # 値変換 kt→ (×0.5)
    lon_c, lat_c, val_c = (np.concatenate(a) for a in (lons, lats, vals))
    keep = np.isfinite(lon_c) & np.isfinite(lat_c) & np.isfinite(val_c)
    offsets = np.asarray(pcs[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], lon_c[keep])
    np.testing.assert_allclose(offsets[:, 1], lat_c[keep])
    np.testing.assert_allclose(np.asarray(pcs[0].get_array(), dtype=float),
                               val_c[keep])


# --- 1-16: map_scatter — offsets = 格子点、色 = 変数値 (変換込み) ---

def test_map_scatter_grid_offsets_and_values():
    ds = _map_ds()
    _use(ds)
    panel = mc_config.default_panel()
    ms = mc_config.default_map_scatter_layer("ds0", "f")
    ms["style"].update({"use_cmap": True,   # 値で色付け (config の既定は単色)
                        "value_scale": 2.0, "value_offset": 5.0})
    panel["layers"] = [ms]
    fig = _render(panel)
    from matplotlib.collections import PathCollection
    pcs = [c for c in fig.axes[0].collections
           if isinstance(c, PathCollection)]
    assert len(pcs) == 1
    lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    offsets = np.asarray(pcs[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], lon2d.ravel())
    np.testing.assert_allclose(offsets[:, 1], lat2d.ravel())
    np.testing.assert_allclose(np.asarray(pcs[0].get_array(), dtype=float),
                               (ds["f"].values * 2.0 + 5.0).ravel())


# --- map_scatter (地点データ): 単一観測点の時系列 (lon/lat がスカラー補助座標) ---

def _station_ds() -> xr.Dataset:
    """AMEDAS 型: 変数の次元は time のみ、lon/lat はスカラー補助座標。"""
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(4) * np.timedelta64(1, "D"))
    return xr.Dataset(
        {"tmean": (("time",), np.array([1.0, 2.0, 3.0, 4.0]))},
        coords={"time": time,
                "lat": ((), 33.9, {"units": "degrees_north"}),
                "lon": ((), 130.9, {"units": "degrees_east"})})


def test_map_scatter_station_point_and_unfixed_dim_guard():
    ds = _station_ds()
    _use(ds)
    from matplotlib.collections import PathCollection

    # 時刻を固定すれば1点: offsets = (lon, lat)、値 = その時刻の値
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-02"}
    ms = mc_config.default_map_scatter_layer("ds0", "tmean")
    ms["style"]["use_cmap"] = True   # 値で色付け (config の既定は単色)
    panel["layers"] = [ms]
    fig = _render(panel)
    pcs = [c for c in fig.axes[0].collections
           if isinstance(c, PathCollection)]
    assert len(pcs) == 1
    offsets = np.asarray(pcs[0].get_offsets())
    np.testing.assert_allclose(offsets, [[130.9, 33.9]])
    np.testing.assert_allclose(
        np.asarray(pcs[0].get_array(), dtype=float), [2.0])

    # 時刻未固定 (点1個に値4個) は黙った IndexError ではなく RenderError
    panel = mc_config.default_panel()
    panel["selection"] = {}
    panel["layers"] = [mc_config.default_map_scatter_layer("ds0", "tmean")]
    with pytest.raises(mc_render.RenderError) as exc:
        _render(panel)
    assert exc.value.msg_id == "map_scatter_dims_unfixed"
    assert "time" in str(exc.value)


def test_map_scatter_degenerate_region_padded_extent():
    """min == max の退化領域 (単一観測点の既定値) は点の周囲 ±1° に広がり、
    cartopy の特異 ylim 警告を出さない。経度が全経度扱いにならないことも確認。"""
    import warnings

    ds = _station_ds()
    _use(ds)
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-02"}
    panel["region"] = {"lon_min": 130.9, "lon_max": 130.9,
                       "lat_min": 33.9, "lat_max": 33.9}
    panel["projection"] = {"name": "PlateCarree",
                           "central_longitude": 130.9, "central_latitude": 0.0}
    panel["layers"] = [mc_config.default_map_scatter_layer("ds0", "tmean")]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*identical low and high.*")
        fig = _render(panel)
    x0, x1, y0, y1 = fig.axes[0].get_extent(crs=None)
    assert x1 - x0 == pytest.approx(2.0)   # ±1° (全経度 360 ではない)
    assert y1 - y0 == pytest.approx(2.0)
    np.testing.assert_allclose([(y0 + y1) / 2], [33.9])


# --- 1-17: 範囲平均 — ライン値 = 独立計算した (重み付き) 平均 ---

def test_line_averages_mean_and_weighted_mean():
    ds = _map_ds()
    _use(ds)

    def _line_y(averages):
        panel = mc_config.default_line_panel()
        panel["x_dim"] = "lon"
        panel["selection"] = {}
        line = mc_config.default_line_layer("ds0", "f")
        line["averages"] = averages
        panel["layers"] = [line]
        fig = _render(panel)
        return fig.axes[0].lines[0].get_xydata()

    f = ds["f"].values
    lat = ds["lat"].values

    # 単純平均 (lat 10–30 の3行)
    xy = _line_y({"lat": {"op": "mean", "range": [10.0, 30.0]}})
    sel = (lat >= 10.0) & (lat <= 30.0)
    np.testing.assert_allclose(xy[:, 1], f[sel].mean(axis=0))
    plt.close("all")

    # cos(lat) 重み付き平均 (全域)
    xy = _line_y({"lat": {"op": "weighted_mean", "range": [0.0, 40.0]}})
    w = np.cos(np.deg2rad(lat))
    expected = (f * w[:, None]).sum(axis=0) / w.sum()
    np.testing.assert_allclose(xy[:, 1], expected)


# --- 1-18: region — 描画配列が指定領域のサブセットに一致 ---

def test_region_subsets_fill_array():
    ds = _map_ds()
    _use(ds)
    panel = mc_config.default_panel()
    panel["region"] = {"lon_min": 120.0, "lon_max": 160.0,
                       "lat_min": 0.0, "lat_max": 30.0}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = "pcolormesh"
    panel["layers"] = [fill]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])

    def _padded_sel(coord, lo, hi):
        # 仕様: 領域端で塗りが切れないよう両側に1格子分パディングして切り出す
        # (render.padded_slice)。データ端ではクランプ
        inside = np.where((coord >= lo) & (coord <= hi))[0]
        i0 = max(inside[0] - 1, 0)
        i1 = min(inside[-1] + 1, coord.size - 1)
        sel = np.zeros(coord.size, dtype=bool)
        sel[i0:i1 + 1] = True
        return sel

    lat_sel = _padded_sel(ds["lat"].values, 0.0, 30.0)
    lon_sel = _padded_sel(ds["lon"].values, 120.0, 160.0)
    expected = ds["f"].values[np.ix_(lat_sel, lon_sel)]
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                     dtype=float).reshape(expected.shape)
    np.testing.assert_allclose(got, expected)


# --- 1-19: ECDF / violin ---

def test_ecdf_matches_numpy():
    ds = _series_ds(200)
    _use(ds)
    panel = mc_config.default_dist_panel()
    ec = mc_config.default_ecdf_layer("ds0")
    ec["variable"] = "a"
    ec["agg_dim"] = "time"
    panel["layers"] = [ec]
    fig = _render(panel)
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    vals = np.sort(ds["a"].values)
    n = vals.size
    x = np.asarray(ax.lines[0].get_xdata(), dtype=float)
    y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    # matplotlib の ecdf は先頭に始点が入ることがある → 末尾 n 点で比較
    assert x.size in (n, n + 1)
    np.testing.assert_allclose(x[-n:], vals)
    np.testing.assert_allclose(y[-n:], np.arange(1, n + 1) / n)


# --- 1-20: 降順座標 — lat 90→-90 格納のデータで上下が正しい ---

def test_descending_lat_orientation():
    lon = np.arange(100.0, 190.0, 10.0)
    lat_desc = np.arange(40.0, -10.0, -10.0)  # 40, 30, 20, 10, 0 (降順格納)
    val = lat_desc[:, None] * 10.0 + lon[None, :] * 0.1
    _use(xr.Dataset(
        {"f": (("lat", "lon"), val)},
        coords={"lat": ("lat", lat_desc, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})}))
    panel = mc_config.default_panel()
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = "pcolormesh"
    panel["layers"] = [fill]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])
    # メッシュの行中心の緯度と、その行の描画値の対応を独立に確認する
    corners = np.asarray(qm.get_coordinates(), dtype=float)  # (M+1, N+1, 2)
    y_centers = 0.5 * (corners[:-1, 0, 1] + corners[1:, 0, 1])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                     dtype=float).reshape(val.shape)
    for target_lat in (40.0, 0.0):
        row = int(np.argmin(np.abs(y_centers - target_lat)))
        np.testing.assert_allclose(
            got[row], target_lat * 10.0 + lon * 0.1,
            err_msg=f"lat={target_lat} の行に別の緯度の値が描かれている")


# --- 1-21: cyclic point — 全球データの経度の継ぎ目 ---

def test_cyclic_point_appended_for_global_lon():
    lon = np.arange(0.0, 360.0, 45.0)   # 8 点で全球 (315 + 45 = 360)
    lat = np.arange(-30.0, 40.0, 10.0)  # 7 点
    val = lat[:, None] * 10.0 + lon[None, :] * 0.01
    _use(xr.Dataset(
        {"f": (("lat", "lon"), val)},
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})}))
    panel = mc_config.default_panel()
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = "pcolormesh"
    panel["layers"] = [fill]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan), dtype=float)
    got = got.reshape(val.shape[0], -1)
    # 経度1列が追加され、追加列は先頭列 (lon=0) と同値 (継ぎ目なし)
    assert got.shape[1] == val.shape[1] + 1, "cyclic point が追加されていない"
    np.testing.assert_allclose(got[:, :-1], val)
    np.testing.assert_allclose(got[:, -1], val[:, 0])


# --- 1-22: agg のペア対応 — x/y の次元順が違っても正しく組む ---

def test_agg_pairs_align_when_dim_order_differs():
    # x は (case, member)、y は (member, case) 格納。論理的には y = 2x + 1
    x = np.arange(6, dtype=float).reshape(3, 2)          # case × member
    y_logical = 2.0 * x + 1.0
    _use(xr.Dataset(
        {"xv": (("case", "member"), x),
         "yv": (("member", "case"), y_logical.T)},       # 次元順を入れ替えて格納
        coords={"case": np.arange(3), "member": np.arange(2)}))
    panel = mc_config.default_agg_panel()
    panel["x_variable"], panel["y_variable"] = "xv", "yv"
    ly = mc_config.default_hist2d_layer("ds0")
    ly["style"].update({"bins_x": 6, "bins_y": 6, "hide_zeros": False})
    panel["layers"] = [ly]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])
    # 期待値: (case, member) で対応づけたペアのヒストグラム。
    # transpose を忘れた実装 (格納順 ravel 同士) だとペアがずれてここで落ちる
    h, _, _ = np.histogram2d(x.ravel(), y_logical.ravel(), bins=[6, 6])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                     dtype=float).reshape(h.T.shape)
    np.testing.assert_allclose(got, h.T)


# --- 1-23: selection の優先順位 — layer 側が panel 側に勝つ ---

def test_layer_selection_overrides_panel_selection():
    lon = np.arange(100.0, 190.0, 10.0)
    lat = np.arange(0.0, 50.0, 10.0)
    level = np.array([850.0, 500.0])
    # level ごとに値を大きくずらす (どちらのスライスが描かれたか判別できる)
    val = (level[:, None, None] * 100.0
           + lat[None, :, None] * 10.0 + lon[None, None, :] * 0.1)
    _use(xr.Dataset(
        {"f": (("level", "lat", "lon"), val)},
        coords={"level": ("level", level, {"units": "hPa"}),
                "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})}))
    panel = mc_config.default_panel()
    panel["selection"] = {"level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = "pcolormesh"
    fill["selection"] = {"level": 850.0}  # layer 側が優先される仕様
    panel["layers"] = [fill]
    fig = _render(panel)
    qm = _quadmesh(fig.axes[0])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                     dtype=float).reshape(val.shape[1:])
    np.testing.assert_allclose(got, val[0])  # level=850 のスライス


# --- 1-24: contour 後処理 — negative_linestyle と emphasis ---

def test_contour_negative_linestyle_and_emphasis():
    ds = _map_ds()
    ds["g"] = ds["f"] - 200.0  # 負〜正の値域にする
    _use(ds)
    panel = mc_config.default_panel()
    levels = [-100.0, -50.0, 50.0, 100.0]
    contour = mc_config.default_contour_layer("ds0", "g")
    contour["style"].update({"levels": levels, "linestyle": "solid",
                             "negative_linestyle": "dashed",
                             "linewidth": 1.0})
    contour["style"]["emphasis"] = {"levels": [50.0], "linewidth": 3.0,
                                    "color": "#d00000"}
    panel["layers"] = [contour]
    fig = _render(panel)
    cs = _contour_sets(fig.axes[0])
    assert len(cs) == 1
    cs = cs[0]
    np.testing.assert_allclose(cs.levels, levels)
    # 負レベルは破線 (onoffseq あり)、正レベルは実線 (onoffseq None)
    linestyles = cs.get_linestyle()
    assert len(linestyles) == len(levels)
    for lv, (_, onoff) in zip(levels, linestyles):
        if lv < 0:
            assert onoff is not None, f"level {lv} が破線になっていない"
        else:
            assert onoff is None, f"level {lv} が実線でない"
    # emphasis: level 50 だけ太さ 3.0 + 赤。他は 1.0 のまま
    widths = np.asarray(cs.get_linewidths(), dtype=float)
    assert widths.size == len(levels)
    np.testing.assert_allclose(widths, [1.0, 1.0, 3.0, 1.0])
    from matplotlib.colors import to_rgba
    edge = np.asarray(cs.get_edgecolors(), dtype=float)
    assert edge.shape[0] == len(levels)
    np.testing.assert_allclose(edge[2], to_rgba("#d00000"))
    # 面が塗られていない (過去の set_color バグ: face まで塗れて黒い塊になる)
    face = np.asarray(cs.get_facecolors(), dtype=float)
    assert face.size == 0 or np.all(face[:, 3] == 0.0), \
        "線コンターの face が塗られている (set_edgecolor を使うこと)"


# --- 1-25: 1D 系レイヤー (fill_between / bar / stackplot) ---

def _bars_ds() -> xr.Dataset:
    """1D レイヤー用の正値・少数点データ。"""
    level = np.array([1000.0, 850.0, 700.0, 500.0])
    return xr.Dataset(
        {"p": (("level",), np.array([4.0, 6.0, 2.0, 8.0])),
         "q": (("level",), np.array([1.0, 2.0, 3.0, 4.0]))},
        coords={"level": ("level", level, {"units": "hPa"})})


def _line_panel_for(ds_vars_panel_layers):
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "level"
    panel["selection"] = {}
    panel["layers"] = ds_vars_panel_layers
    return panel


def _has_vertex(paths, x, y, tol=1e-9):
    for path in paths:
        v = np.asarray(path.vertices, dtype=float)
        if np.any((np.abs(v[:, 0] - x) < tol) & (np.abs(v[:, 1] - y) < tol)):
            return True
    return False


def test_fill_between_band_boundaries():
    ds = _bars_ds()
    _use(ds)
    fb = mc_config.default_fill_between_layer("ds0", "p", "q")
    fig = _render(_line_panel_for([fb]))
    from matplotlib.collections import PolyCollection
    pcs = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
    assert len(pcs) == 1
    paths = pcs[0].get_paths()
    # 塗り領域の頂点に、各 x で両変数の値 (帯の上下端) が含まれる
    for x, y1, y2 in zip(ds["level"].values, ds["p"].values, ds["q"].values):
        assert _has_vertex(paths, x, y1), f"(x={x}, p={y1}) が帯の頂点にない"
        assert _has_vertex(paths, x, y2), f"(x={x}, q={y2}) が帯の頂点にない"


def test_bar_heights_and_stack_and_dodge():
    ds = _bars_ds()
    _use(ds)
    lv = ds["level"].values
    p, q = ds["p"].values, ds["q"].values

    def _bars(bar_mode):
        b1 = mc_config.default_bar_layer("ds0", "p")
        b2 = mc_config.default_bar_layer("ds0", "q")
        panel = _line_panel_for([b1, b2])
        panel["axis"]["bar_mode"] = bar_mode
        fig = _render(panel)
        patches = fig.axes[0].patches
        assert len(patches) == 2 * lv.size
        return patches[:lv.size], patches[lv.size:]

    # stack: 高さは各変数の値、2層目の下端 = 1層目の値
    pa, pb = _bars("stack")
    np.testing.assert_allclose([r.get_height() for r in pa], p)
    np.testing.assert_allclose([r.get_height() for r in pb], q)
    np.testing.assert_allclose([r.get_y() for r in pa], 0.0)
    np.testing.assert_allclose([r.get_y() for r in pb], p)
    plt.close("all")

    # dodge: 2層の棒の中心が座標を挟んで対称 (center1 + center2 = 2x) かつ別位置
    pa, pb = _bars("dodge")
    c1 = np.array([r.get_x() + r.get_width() / 2 for r in pa])
    c2 = np.array([r.get_x() + r.get_width() / 2 for r in pb])
    np.testing.assert_allclose(c1 + c2, 2.0 * lv)
    assert np.all(np.abs(c1 - c2) > 0), "dodge で棒が重なっている"


def test_bar_errorbar_segments():
    ds = _bars_ds()
    _use(ds)
    bar = mc_config.default_bar_layer("ds0", "p")
    bar["style"]["errorbar"].update({"source": "constant", "constant": 2.0})
    fig = _render(_line_panel_for([bar]))
    from matplotlib.collections import LineCollection
    lcs = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    assert lcs, "エラーバーの LineCollection が見つからない"
    segs = [np.asarray(s, dtype=float) for lc in lcs for s in lc.get_segments()]
    for x, v in zip(ds["level"].values, ds["p"].values):
        ok = any(np.allclose(s[:, 0], x)
                 and np.isclose(s[:, 1].min(), v - 2.0)
                 and np.isclose(s[:, 1].max(), v + 2.0) for s in segs)
        assert ok, f"x={x} のエラーバー ({v}±2) が見つからない"


@pytest.mark.parametrize("source", ["variable", "constant"])
def test_bar_errorbar_not_value_transformed(source):
    """棒グラフのエラー量には値の変換 (倍率 a・加算 b) を掛けない (2026-09-29 の仕様)。

    棒の高さは a × p + b、エラーバーの半長は |誤差| のまま。v1.00.1 までは変数の
    エラー量に a × err + b が掛かっていた (KNOWN_ISSUES.md KI-1)。
    """
    from matplotlib.collections import LineCollection
    ds = _bars_ds()
    ds["e"] = (("level",), np.array([0.5, -1.0, 0.25, 2.0]))
    _use(ds)
    a, b = 3.0, -100.0
    bar = mc_config.default_bar_layer("ds0", "p")
    bar["style"]["value_scale"] = a
    bar["style"]["value_offset"] = b
    if source == "variable":
        bar["style"]["errorbar"].update({"source": "variable", "variable": "e"})
        expected = np.abs(ds["e"].values)
    else:
        bar["style"]["errorbar"].update({"source": "constant", "constant": 1.5})
        expected = np.full(ds.sizes["level"], 1.5)
    fig = _render(_line_panel_for([bar]))
    lcs = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    assert lcs, "エラーバーの LineCollection が見つからない"
    segs = [np.asarray(s, dtype=float) for lc in lcs for s in lc.get_segments()]
    for x, v, e in zip(ds["level"].values, a * ds["p"].values + b, expected):
        ok = any(np.allclose(s[:, 0], x)
                 and np.isclose(s[:, 1].min(), v - e)
                 and np.isclose(s[:, 1].max(), v + e) for s in segs)
        assert ok, f"x={x} のエラーバー ({v}±{e}) が見つからない (誤差に値の変換が掛かっている?)"


def test_stackplot_cumulative_boundaries():
    ds = _bars_ds()
    _use(ds)
    sp = mc_config.default_stackplot_layer("ds0", ["p", "q"])
    fig = _render(_line_panel_for([sp]))
    from matplotlib.collections import PolyCollection
    pcs = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
    assert len(pcs) == 2
    lv = ds["level"].values
    top1 = ds["p"].values                     # 1層目の上端 = p
    top2 = ds["p"].values + ds["q"].values    # 2層目の上端 = p + q (累積)
    for x, y in zip(lv, top1):
        assert _has_vertex(pcs[0].get_paths(), x, y), \
            f"stackplot 1層目の上端 (x={x}, {y}) がない"
    for x, y in zip(lv, top2):
        assert _has_vertex(pcs[1].get_paths(), x, y), \
            f"stackplot 2層目の上端 (x={x}, {y}) がない"


# --- 1-26: hatch — レベル範囲とパターン ---

def test_hatch_levels_and_pattern():
    _use(_map_ds())
    panel = mc_config.default_panel()
    hatch = mc_config.default_hatch_layer("ds0", "f")
    hatch["style"].update({"levels": [150.0, 300.0], "pattern": "x",
                           "density": 2})
    panel["layers"] = [hatch]
    fig = _render(panel)
    cs = _contour_sets(fig.axes[0])
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0].levels, [150.0, 300.0])
    assert list(cs[0].hatches) == ["xx"]  # pattern × density
    # colors="none" なので面は塗られない (ハッチ線のみ)
    fc = np.asarray(cs[0].get_facecolor(), dtype=float)
    assert fc.size == 0 or np.all(fc[:, 3] == 0.0)


def test_hatch_line_color():
    """ハッチ線の色 (style.color) が画素に出る。既定は黒 (2026-09-29 追加)。

    matplotlib はハッチの色を artist 作成時の rcParams['hatch.color'] から取るので、
    render.hatch_rc_params の rc_context が効いていることを画素で確かめる。
    """
    import io
    import matplotlib.pyplot as plt

    def red_pixels(color):
        _use(_map_ds())
        panel = mc_config.default_panel()
        hatch = mc_config.default_hatch_layer("ds0", "f")
        hatch["style"].update({"levels": [100.0, 400.0], "pattern": "/",
                               "density": 4, "linewidth": 2.0})
        if color:
            hatch["style"]["color"] = color
        panel["layers"] = [hatch]
        fig = _render(panel)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60)
        plt.close(fig)
        buf.seek(0)
        img = plt.imread(buf)
        return int(((img[..., 0] > 0.8) & (img[..., 1] < 0.3)
                    & (img[..., 2] < 0.3)).sum())

    assert red_pixels(None) == 0          # 既定 = 黒
    assert red_pixels("#ff0000") > 100    # 赤のハッチ線


# --- 1-27: スタイル反映 — 単色・線種・太さ・reverse_cmap ---

def test_style_props_reflected():
    ds = _bars_ds()
    _use(ds)
    # line: 色・太さ・線種
    line = mc_config.default_line_layer("ds0", "p")
    line["style"].update({"color": "#cc3300", "linewidth": 2.5,
                          "linestyle": "--"})
    fig = _render(_line_panel_for([line]))
    from matplotlib.colors import to_rgba
    ln = fig.axes[0].lines[0]
    assert to_rgba(ln.get_color()) == to_rgba("#cc3300")
    assert ln.get_linewidth() == 2.5
    assert ln.get_linestyle() == "--"
    plt.close("all")

    # fill: reverse_cmap → cmap 名が *_r になる
    _use(_map_ds())
    panel = mc_config.default_panel()
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"method": "pcolormesh", "cmap": "viridis",
                          "reverse_cmap": True})
    panel["layers"] = [fill]
    fig = _render(panel)
    assert _quadmesh(fig.axes[0]).cmap.name == "viridis_r"


# --- 1-28: 文字列 — タイトル・軸ラベル・凡例・注釈・パネルラベル ---

def test_titles_labels_legend_texts_and_marker():
    ds = _bars_ds()
    _use(ds)
    line = mc_config.default_line_layer("ds0", "p")
    line["style"]["label"] = "series-1"
    panel = _line_panel_for([line])
    panel["title"] = "TITLE"
    panel["axis"].update({"x_label": "XL", "y_label": "YL"})
    panel["legend"]["show"] = True
    txt = mc_config.default_text_annotation()
    txt.update({"text": "NOTE", "coord": "axes", "x": 0.1, "y": 0.9})
    panel["texts"] = [txt]
    mk = mc_config.default_marker_annotation()
    mk.update({"marker": "^", "coord": "data", "x": 700.0, "y": 5.0})
    panel["markers"] = [mk]
    panel["label"].update({"show": True, "text": "(b)"})
    fig = _render(panel)
    ax = fig.axes[0]
    assert ax.get_title() == "TITLE"
    assert ax.get_xlabel() == "XL" and ax.get_ylabel() == "YL"
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["series-1"]
    texts = {t.get_text(): t.get_position() for t in ax.texts}
    assert "NOTE" in texts and texts["NOTE"] == (0.1, 0.9)
    assert "(b)" in texts
    # マーカー注釈: (700, 5) に1点の Line2D
    marker_pts = [ln for ln in ax.lines
                  if np.asarray(ln.get_xdata()).size == 1
                  and float(np.asarray(ln.get_xdata())[0]) == 700.0
                  and float(np.asarray(ln.get_ydata())[0]) == 5.0]
    assert marker_pts, "マーカー注釈 (700, 5) が見つからない"


# --- 1-29: heatmap パネル — 行列値と数値注記 ---

def test_heatmap_matrix_and_annotations():
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _use(xr.Dataset({"mat": (("row", "col"), m)},
                    coords={"row": [0, 1], "col": [0, 1, 2]}))
    panel = mc_config.default_heatmap_panel()
    panel["dataset_id"] = "ds0"
    panel["variable"] = "mat"
    panel["x_dim"], panel["y_dim"] = "col", "row"
    panel["selection"] = {}
    panel["style"]["annotate"].update({"show": True, "fmt": "%.0f"})
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    fig = mc_render.render_figure(cfg, _DATASETS)
    ax = fig.axes[0]
    assert len(ax.images) == 1
    got = np.asarray(np.ma.filled(ax.images[0].get_array(), np.nan),
                     dtype=float)
    np.testing.assert_allclose(got, m)  # (y_dim, x_dim) 順そのまま
    # 数値注記: 全セルの fmt 文字列が、そのセル位置 (x=col, y=row) にある
    ann = {(round(t.get_position()[0]), round(t.get_position()[1])):
           t.get_text() for t in ax.texts}
    for i in range(2):
        for j in range(3):
            assert ann.get((j, i)) == f"{m[i, j]:.0f}", \
                f"セル ({i},{j}) の注記が {ann.get((j, i))!r}"


# --- 1-30: 第2軸 (twinx) — secondary_y レイヤーが右軸に描かれる ---

def test_dist_secondary_y_on_twin_axis():
    ds = _series_ds(200)
    _use(ds)
    panel = mc_config.default_dist_panel()
    hist = mc_config.default_hist_layer("ds0")
    hist["variable"] = "a"
    hist["agg_dim"] = "time"
    ec = mc_config.default_ecdf_layer("ds0")
    ec["variable"] = "a"
    ec["agg_dim"] = "time"
    ec["style"]["secondary_y"] = True
    panel["layers"] = [hist, ec]
    fig = _render(panel)
    assert len(fig.axes) == 2, "twinx の第2軸が作られていない"
    ax, ax2 = fig.axes
    # ヒストグラムは左軸、ECDF (0〜1) は右軸に
    assert len(ax.patches) > 0 and len(ax.lines) == 0
    assert len(ax2.lines) == 1
    assert float(np.max(ax2.lines[0].get_ydata())) == pytest.approx(1.0)
    assert ax2.yaxis.get_ticks_position() == "right"


# --- 1-43: 1次元プロットの第2軸 (twinx) — secondary_y レイヤーが右軸に描かれる ---

def test_line_1d_secondary_y_on_twin_axis():
    ds = _bars_ds()
    _use(ds)
    panel = _line_panel_for([])
    panel["axis"].update({"y_label": "p", "y2_label": "q x100",
                          "y2_lim": [0.0, 500.0]})
    lp = mc_config.default_line_layer("ds0", "p")
    lp["style"]["label"] = "P"
    lq = mc_config.default_line_layer("ds0", "q")
    lq["style"].update({"label": "Q", "secondary_y": True,
                        "value_scale": 100.0})
    bar_q = mc_config.default_bar_layer("ds0", "q")
    bar_q["style"]["secondary_y"] = True
    panel["layers"] = [lp, lq, bar_q]
    fig = _render(panel)
    assert len(fig.axes) == 2, "twinx の第2軸が作られていない"
    ax, ax2 = fig.axes
    # p は左軸 (第1軸)、q (×100) と棒は右軸 (第2軸)
    assert len(ax.lines) == 1 and len(ax.patches) == 0
    np.testing.assert_allclose(ax.lines[0].get_ydata(), ds["p"].values)
    assert len(ax2.lines) == 1
    np.testing.assert_allclose(ax2.lines[0].get_ydata(), ds["q"].values * 100.0)
    assert len(ax2.patches) == len(ds["level"])
    np.testing.assert_allclose(sorted(r.get_height() for r in ax2.patches),
                               sorted(ds["q"].values))
    # 第2軸の目盛は右側、ラベル・範囲は y2_* が入る。第1軸は不変
    assert ax2.yaxis.get_ticks_position() == "right"
    assert ax.yaxis.get_ticks_position() == "left"
    assert ax2.get_ylabel() == "q x100" and ax.get_ylabel() == "p"
    assert ax2.get_ylim() == (0.0, 500.0)
    # 凡例は両軸の handles を結合して1つ (ax2 上)
    assert ax.get_legend() is None
    leg = ax2.get_legend()
    assert leg is not None
    assert [t.get_text() for t in leg.get_texts()] == ["P", "Q"]


# --- 1-44: 1次元プロットの第2軸の軸設定 — y2 設定が ax2 だけに反映される ---

def test_line_1d_secondary_axis_settings():
    ds = _bars_ds()
    _use(ds)

    def _panel(**axis_updates):
        panel = _line_panel_for([])
        lp = mc_config.default_line_layer("ds0", "p")
        lq = mc_config.default_line_layer("ds0", "q")
        lq["style"]["secondary_y"] = True
        panel["layers"] = [lp, lq]
        panel["axis"].update(axis_updates)
        return panel

    panel = _panel(invert_y2=True, y2_tick_positions=[1.0, 2.0, 4.0],
                   y2_tick_labels=["a", "b", "c"], show_y2_minor_ticks=True,
                   y2_label="q")
    panel["axis"]["grid"]["show_y2"] = True
    fig = _render(panel)
    fig.canvas.draw()
    ax, ax2 = fig.axes
    # 反転・目盛位置+ラベル・補助目盛・目盛線は第2軸だけに掛かり、第1軸は不変
    assert ax2.get_ylim()[0] > ax2.get_ylim()[1], "invert_y2 未反映"
    assert ax.get_ylim()[0] < ax.get_ylim()[1], "第1軸が反転している"
    np.testing.assert_allclose(ax2.get_yticks(), [1.0, 2.0, 4.0])
    assert [t.get_text() for t in ax2.get_yticklabels()] == ["a", "b", "c"]
    assert type(ax2.yaxis.get_minor_locator()).__name__ == "AutoMinorLocator"
    assert type(ax.yaxis.get_minor_locator()).__name__ != "AutoMinorLocator"
    assert all(g.get_visible() for g in ax2.yaxis.get_gridlines())
    assert not any(g.get_visible() for g in ax.yaxis.get_gridlines())
    assert ax2.get_ylabel() == "q" and ax2.yaxis.get_ticks_position() == "right"
    # 対数は別描画 (反転・位置指定なし)。第1軸は線形のまま
    fig = _render(_panel(log_y2=True))
    ax, ax2 = fig.axes
    assert ax2.get_yscale() == "log" and ax.get_yscale() == "linear"


# --- 1-46: 第2軸との値揃え — 指定値が左右の軸で同じ高さになり、データは切れない ---

def test_line_1d_twin_axes_align_value():
    def frac(lim, v):
        return (v - min(lim)) / (max(lim) - min(lim))

    # 純関数: 第1軸の内側に値があれば第1軸は不変、第2軸は縮まず広がる
    l1, l2 = mc_render.align_twin_ylim((-2.0, 8.0), (-10.0, 10.0), 0.0)
    assert l1 == (-2.0, 8.0)
    assert min(l2) <= -10.0 and max(l2) >= 10.0
    assert abs(frac(l1, 0.0) - frac(l2, 0.0)) < 1e-12
    # 値が範囲外 (気温 240〜300 K に 0): 両軸とも値を含むまで広がって揃う
    l1, l2 = mc_render.align_twin_ylim((240.0, 300.0), (-5.0, 5.0), 0.0)
    assert min(l1) <= 0.0 <= max(l1) and max(l1) >= 300.0
    assert abs(frac(l1, 0.0) - frac(l2, 0.0)) < 1e-12
    # 反転した軸は向きを保つ
    l1, l2 = mc_render.align_twin_ylim((8.0, -2.0), (-10.0, 10.0), 0.0)
    assert l1[0] > l1[1] and abs(frac(l1, 0.0) - frac(l2, 0.0)) < 1e-12
    # 退化した範囲は素通し
    assert mc_render.align_twin_ylim((1.0, 1.0), (0.0, 1.0), 0.0) == ((1.0, 1.0), (0.0, 1.0))

    # 描画: p (左, 2〜8) と q×10 (右, 10〜40) で 3 を揃える
    ds = _bars_ds()
    _use(ds)
    panel = _line_panel_for([])
    lp = mc_config.default_line_layer("ds0", "p")
    lq = mc_config.default_line_layer("ds0", "q")
    lq["style"].update({"secondary_y": True, "value_scale": 10.0})
    panel["layers"] = [lp, lq]
    panel["axis"]["y2_align_value"] = 3.0
    fig = _render(panel)
    fig.canvas.draw()
    ax, ax2 = fig.axes
    assert abs(frac(ax.get_ylim(), 3.0) - frac(ax2.get_ylim(), 3.0)) < 1e-9
    assert ax.get_ylim()[0] <= 2.0 and ax.get_ylim()[1] >= 8.0, "第1軸のデータが切れた"
    assert ax2.get_ylim()[0] <= 10.0 and ax2.get_ylim()[1] >= 40.0, "第2軸のデータが切れた"
    # 対数軸では無効 (第2軸を対数にしても落ちず、対数の自動範囲のまま)
    panel["axis"]["log_y2"] = True
    fig = _render(panel)
    ax, ax2 = fig.axes
    assert ax2.get_yscale() == "log"
    assert ax2.get_ylim()[0] > 0.0


# --- 1-47: 基準線 (reflines) — 位置・体裁・凡例ラベル・端の文字が反映される ---

def test_reflines_positions_legend_and_text():
    ds = _line_ds()
    _use(ds)
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "level"
    panel["selection"] = {}
    panel["layers"] = [mc_config.default_line_layer("ds0", "g")]
    vline = mc_config.default_refline()
    vline.update({"orientation": "x", "value": 850.0, "color": "#ff0000",
                  "linestyle": "dotted", "linewidth": 2.0, "text": "850 hPa"})
    hline = mc_config.default_refline()
    hline.update({"orientation": "y", "value": 0.0, "label": "zero",
                  "alpha": 0.5, "text": "0", "text_color": "#0000ff",
                  "text_fontsize": 7})
    panel["background"]["reflines"] = [vline, hline]
    fig = _render(panel)
    ax = fig.axes[0]
    # データ線 1 本 + 基準線 2 本。縦線は x=850 の定数、横線は y=0 の定数
    assert len(ax.lines) == 3
    v = ax.lines[1]
    np.testing.assert_allclose(v.get_xdata(), [850.0, 850.0])
    assert v.get_color() == "#ff0000" and v.get_linewidth() == 2.0
    assert v.get_linestyle() == ":"
    h = ax.lines[2]
    np.testing.assert_allclose(h.get_ydata(), [0.0, 0.0])
    assert h.get_alpha() == 0.5
    # 凡例にはラベル付きの横線だけ
    leg = ax.get_legend()
    assert leg is not None
    assert [t.get_text() for t in leg.get_texts()] == ["zero"]
    # 端の文字: 縦線は (x=値, y=0.98 axes) に 90° 回転、横線は (x=0.98 axes, y=値)
    texts = {t.get_text(): t for t in ax.texts}
    assert set(texts) == {"850 hPa", "0"}
    assert texts["850 hPa"].get_position() == (850.0, 0.98)
    assert texts["850 hPa"].get_rotation() == 90.0
    assert texts["850 hPa"].get_color() == "#ff0000"  # 未指定なら線と同色
    assert texts["0"].get_position() == (0.98, 0.0)
    assert texts["0"].get_color() == "#0000ff" and texts["0"].get_fontsize() == 7


# --- 1-31: アニメーション — 各フレームの描画配列 = その時刻のスライス ---

def test_animation_frames_match_time_slices():
    lon = np.arange(100.0, 190.0, 10.0)
    lat = np.arange(0.0, 50.0, 10.0)
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(3) * np.timedelta64(6, "h"))
    val = (np.arange(3)[:, None, None] * 1000.0
           + lat[None, :, None] * 10.0 + lon[None, None, :] * 0.1)
    _use(xr.Dataset(
        {"f": (("time", "lat", "lon"), val)},
        coords={"time": time,
                "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})}))
    panel = mc_config.default_panel()
    panel["selection"] = {}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = "pcolormesh"
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    for i, fig in enumerate(
            mc_render.iter_time_frames(cfg, _DATASETS, list(time))):
        qm = _quadmesh(fig.axes[0])
        got = np.asarray(np.ma.filled(qm.get_array(), np.nan),
                         dtype=float).reshape(val.shape[1:])
        np.testing.assert_allclose(got, val[i],
                                   err_msg=f"フレーム {i} が時刻 {i} のスライスでない")
        plt.close(fig)
    assert i == 2  # 3フレーム生成された


def test_animation_overrides_layer_fixed_time():
    """レイヤー側の selection に時刻が固定値で入っている設定でも、時刻送りで
    そのパネルの時刻が進むこと (描画時は layer.selection が panel.selection を
    上書きするため、パネル側だけ差し替えると止まる。2026-09-19 の実機報告)。
    scriptgen はレイヤー側の .sel(time=...) も time_value に置き換えるので、
    render がこれを守らないと両者が食い違う。"""
    lon = np.arange(100.0, 190.0, 10.0)
    lat = np.arange(0.0, 50.0, 10.0)
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(3) * np.timedelta64(6, "h"))
    val = (np.arange(3)[:, None, None] * 1000.0
           + lat[None, :, None] * 10.0 + lon[None, None, :] * 0.1)
    _use(xr.Dataset(
        {"f": (("time", "lat", "lon"), val)},
        coords={"time": time,
                "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})}))
    panels = []
    for fixed_in_layer in (False, True):
        panel = mc_config.default_panel()
        panel["selection"] = {"time": time[0]}
        fill = mc_config.default_fill_layer("ds0", "f")
        fill["style"]["method"] = "pcolormesh"
        if fixed_in_layer:
            fill["selection"] = {"time": time[0]}  # レイヤー側の固定時刻
        panel["layers"] = [fill]
        panels.append(panel)
    cfg = mc_config.default_figure_config()
    cfg["figure"]["layout"].update({"nrows": 1, "ncols": 2})
    cfg["panels"] = panels
    n_cells = val.shape[1] * val.shape[2]
    for i, fig in enumerate(
            mc_render.iter_time_frames(cfg, _DATASETS, list(time))):
        # パネルの QuadMesh だけを拾う (カラーバーの QuadMesh は要素数が違う)
        arrays = []
        for ax in fig.axes:
            for c in ax.collections:
                if isinstance(c, QuadMesh) and np.size(c.get_array()) == n_cells:
                    arrays.append(np.asarray(np.ma.filled(c.get_array(), np.nan),
                                             dtype=float).reshape(val.shape[1:]))
        assert len(arrays) == 2, f"フレーム {i}: パネルの QuadMesh が 2 つのはず: {len(arrays)}"
        for j, got in enumerate(arrays):
            np.testing.assert_allclose(
                got, val[i],
                err_msg=f"フレーム {i} のパネル {j} が時刻 {i} のスライスでない"
                        " (レイヤー側の固定時刻が差し替えられていない)")
        plt.close(fig)
    assert i == 2



# --- 1-42: 地球回転アニメーション (Orthographic の投影中心をフレーム毎に送る) ---

def _rotation_ds():
    lon = np.arange(0.0, 360.0, 30.0)
    lat = np.arange(-60.0, 61.0, 30.0)
    time = (np.datetime64("2024-01-01", "ns")
            + np.arange(3) * np.timedelta64(6, "h"))
    val = (np.arange(3)[:, None, None] * 1000.0
           + lat[None, :, None] * 10.0 + lon[None, None, :] * 0.1)
    return xr.Dataset(
        {"f": (("time", "lat", "lon"), val)},
        coords={"time": time,
                "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})}), time, val


def _rotation_cfg(time_value):
    panel = mc_config.default_panel()
    panel["selection"] = {"time": time_value}
    panel["projection"] = {"name": "Orthographic",
                           "central_longitude": 140.0, "central_latitude": 20.0}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["method"] = "pcolormesh"
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _visible_mesh_values(ax, shape):
    """QuadMesh の描画値 (Orthographic では裏側のセルがマスクされるので
    マスクと値を返す)。"""
    qm = _quadmesh(ax)
    arr = np.ma.masked_invalid(np.ma.asarray(qm.get_array(), dtype=float))
    ny, nx = shape
    if arr.size == ny * (nx + 1):
        arr = arr.reshape(ny, nx + 1)[:, :nx]  # 全球データの cyclic point (1-21) を落とす
    else:
        arr = arr.reshape(ny, nx)
    return np.ma.getmaskarray(arr), np.ma.getdata(arr)


def test_rotation_frames_follow_center_path_with_time_fixed():
    """回転のみ: 各フレームの投影中心 = rotation_path の i 番目、描画値は
    固定した時刻のスライスのまま (時刻は動かない)。"""
    ds, time, val = _rotation_ds()
    _use(ds)
    cfg = _rotation_cfg(time[1])
    centers = mc_render.rotation_path([(140.0, 20.0), (200.0, 50.0), (260.0, 0.0)], 4)
    n = 0
    for i, fig in enumerate(mc_render.iter_frames(cfg, _DATASETS, centers=centers)):
        ax = fig.axes[0]
        params = ax.projection.proj4_params
        assert (float(params["lon_0"]), float(params["lat_0"])) == centers[i], \
            f"フレーム {i} の投影中心が経路と一致しない"
        mask, got = _visible_mesh_values(ax, val.shape[1:])
        assert (~mask).any()
        np.testing.assert_allclose(got[~mask], val[1][~mask],
                                   err_msg=f"フレーム {i} の描画値が固定時刻のスライスでない")
        plt.close(fig)
        n = i + 1
    assert n == 4


def test_rotation_frames_with_time_stepping():
    """時刻送り + 回転: フレーム i は時刻 i のスライス、投影中心は経路の i 番目。
    time_label も時刻に追従する。"""
    ds, time, val = _rotation_ds()
    _use(ds)
    cfg = _rotation_cfg(time[0])
    cfg["panels"][0]["time_label"] = {"show": True, "text": str(time[0]), "loc": "left"}
    centers = mc_render.rotation_path([(140.0, 20.0), (260.0, 0.0)], 3)
    for i, fig in enumerate(mc_render.iter_frames(
            cfg, _DATASETS, time_values=list(time), centers=centers)):
        ax = fig.axes[0]
        params = ax.projection.proj4_params
        assert (float(params["lon_0"]), float(params["lat_0"])) == centers[i]
        mask, got = _visible_mesh_values(ax, val.shape[1:])
        np.testing.assert_allclose(got[~mask], val[i][~mask],
                                   err_msg=f"フレーム {i} が時刻 {i} のスライスでない")
        assert ax.get_title(loc="left") == str(time[i])
        plt.close(fig)
    assert i == 2


def test_rotation_with_frames_per_time_holds_each_time():
    """時刻送り + 回転で frames_per_time=2: フレーム i は時刻 i//2 のスライス、
    投影中心は経路の i 番目 (時刻数 × 2 のフレームで回転が滑らかに進む)。"""
    ds, time, val = _rotation_ds()
    _use(ds)
    cfg = _rotation_cfg(time[0])
    # 経路は他の回転テストと同じ (200°E, 50°N 経由)。(140,20)→(260,0) を 6 分割すると
    # 中心 (236°, 4°) で cartopy の gridliner が退化 LineString を作り GEOS 例外に
    # なる (cartopy_compat_notes.md「Orthographic の gridliner」)
    centers = mc_render.rotation_path([(140.0, 20.0), (200.0, 50.0), (260.0, 0.0)], 6)
    n = 0
    for i, fig in enumerate(mc_render.iter_frames(
            cfg, _DATASETS, time_values=list(time), centers=centers, frames_per_time=2)):
        ax = fig.axes[0]
        params = ax.projection.proj4_params
        assert (float(params["lon_0"]), float(params["lat_0"])) == centers[i]
        mask, got = _visible_mesh_values(ax, val.shape[1:])
        np.testing.assert_allclose(got[~mask], val[i // 2][~mask],
                                   err_msg=f"フレーム {i} が時刻 {i // 2} のスライスでない")
        plt.close(fig)
        n = i + 1
    assert n == 6


def test_orthographic_degenerate_gridline_raises_readable_error():
    """cartopy の gridliner が地平線に1点で接する緯度経度線で GEOS 例外を出す中心
    (粗い 30° 格子 + Orthographic (236°E, 4°N) で確定再現) が、対処ヒント付きの
    RenderError("degenerate_gridline") になること。cartopy 側で直ったら
    (例外が出なくなったら) cartopy_compat_notes.md の該当節を閉じる。"""
    ds, time, val = _rotation_ds()
    _use(ds)
    cfg = _rotation_cfg(time[0])
    cfg["panels"][0]["projection"].update(central_longitude=236.0, central_latitude=4.0)
    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.render_figure(cfg, _DATASETS)
    assert ei.value.msg_id == "degenerate_gridline"
    # 緯度経度線を消せば描ける (ヒントどおり)
    cfg["panels"][0]["map"]["gridlines"]["show"] = False
    plt.close(mc_render.render_figure(cfg, _DATASETS))


def _ring_path(*rings):
    from matplotlib.path import Path
    codes = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY]
                            for r in rings])
    return Path(np.concatenate(rings), codes)


def _max_projected_ring_area(path, ax, src):
    from cartopy.mpl.geoaxes import InterProjectionTransform
    tr = InterProjectionTransform(src, ax.projection)
    best = 0.0
    for q in tr.transform_path_non_affine(path).to_polygons(closed_only=False):
        q = np.asarray(q)
        if len(q) > 2 and np.isfinite(q).all():
            x, y = q[:, 0], q[:, 1]
            best = max(best, 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return best


# 実測の「反転する」微小リング (HadISST 偏差 2006-01-15、Orthographic 145.65°E/19.32°N、
# レベル [-0.25, 0) の 5 頂点。88.7〜89.5°N で経度 0° (=360°) をまたぐ)
_CULPRIT_RING = np.array([[360.5, 88.691], [360.5, 88.957], [360.068, 89.5],
                          [359.856, 89.5], [360.5, 88.691]])


def test_drop_inverting_contour_rings_removes_only_culprit():
    """cartopy が Orthographic への投影で円盤全体に反転させる微小リングだけを落とし、
    同じ形でも正常に投影されるリング (経度 200° に移したもの) や普通のリングは
    触らない (パスオブジェクトも同一のまま)。上流バグの前提も固定する
    (直ったら cartopy_compat_notes.md の該当節を閉じる)。"""
    import cartopy.crs as ccrs
    from matplotlib.collections import PathCollection
    from matplotlib.path import Path
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(145.6527, 19.3203))
    ax.set_global()
    src = ccrs.PlateCarree()
    dom = ax.projection.domain.area
    normal = np.array([[150.0, 10.0], [160.0, 10.0], [160.0, 20.0], [150.0, 20.0], [150.0, 10.0]])
    benign_tiny = _CULPRIT_RING - [160.0, 0.0]  # 同じ形を経度 200° へ → 正常に投影される
    # 上流バグの前提: culprit 単独の投影が円盤全体、benign は微小
    assert _max_projected_ring_area(_ring_path(_CULPRIT_RING), ax, src) > 0.5 * dom
    assert _max_projected_ring_area(_ring_path(benign_tiny), ax, src) < 1e-5 * dom

    p_mixed = _ring_path(normal, _CULPRIT_RING, benign_tiny)
    p_clean = _ring_path(normal, benign_tiny)
    pc = PathCollection([p_mixed, p_clean])
    mc_render.drop_inverting_contour_rings(pc, ax, src)
    paths = pc.get_paths()
    assert len(paths) == 2
    # 1本目: culprit だけ消え、normal と benign_tiny が残る
    assert (paths[0].codes == Path.MOVETO).sum() == 2
    np.testing.assert_allclose(paths[0].vertices, np.concatenate([normal, benign_tiny]))
    assert _max_projected_ring_area(paths[0], ax, src) < 0.5 * dom
    # 2本目 (反転なし) は同一オブジェクトのまま
    assert paths[1] is p_clean
    plt.close(fig)


_HADISST = pathlib.Path(__file__).resolve().parent.parent / "data" / "sample" / \
    "anom.DJF.1984-2014.HadISST.sst.nc"


@pytest.mark.skipif(not _HADISST.exists(),
                    reason="実データ (git 管理外の data/sample) が無い環境ではスキップ")
def test_orthographic_inversion_guard_on_real_data():
    """2026-08-28 の報告 (地球回転アニメの 43 コマ目だけ陸ごと淡色になる) の再現条件:
    HadISST 偏差 2006-01-15、Orthographic (145.6527°E, 19.3203°N)、contourf 21 レベル
    ±2.5。円盤全体を覆う塗りポリゴンが出ないこと。"""
    from matplotlib.contour import ContourSet
    _use(xr.open_dataset(_HADISST))
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2006-01-15T00:00:00"}
    panel["projection"] = {"name": "Orthographic",
                           "central_longitude": 145.6527, "central_latitude": 19.3203}
    fill = mc_config.default_fill_layer("ds0", "sst")
    fill["style"].update({"cmap": "RdBu_r", "vmin": -2.5, "vmax": 2.5, "levels": 21,
                          "extend": "both"})
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    fig = mc_render.render_figure(cfg, _DATASETS)
    fig.canvas.draw()
    ax = fig.axes[0]
    cs = [a for a in ax.collections if isinstance(a, ContourSet)][0]
    axarea = ax.bbox.width * ax.bbox.height
    tr = cs.get_transform()
    worst = 0.0
    for path in cs.get_paths():
        if len(path.vertices) == 0:
            continue
        for q in tr.transform_path(path).to_polygons(closed_only=False):
            q = np.asarray(q)
            if len(q) > 2 and np.isfinite(q).all():
                x, y = q[:, 0], q[:, 1]
                worst = max(worst, 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / axarea)
    assert worst < 0.6, f"円盤全体を覆う塗りポリゴンが残っている (axes 面積比 {worst:.2f})"
    plt.close(fig)


def test_rotation_leaves_non_orthographic_panels_unchanged():
    """centers は Orthographic の地図パネルにだけ効く (panel_rotates)。"""
    ds, time, val = _rotation_ds()
    _use(ds)
    cfg = _rotation_cfg(time[0])
    cfg["panels"][0]["projection"] = {"name": "PlateCarree",
                                      "central_longitude": 180.0, "central_latitude": 0.0}
    assert not mc_render.panel_rotates(cfg["panels"][0])
    fig = next(iter(mc_render.iter_frames(cfg, _DATASETS, centers=[(10.0, 10.0)])))
    assert float(fig.axes[0].projection.proj4_params["lon_0"]) == 180.0
    plt.close(fig)

# --- 1-32: 複数パネル配置・mosaic 結合・共有カラーバー ---

def test_multipanel_grid_and_shared_colorbar():
    _use(_map_ds())
    levels = [50.0, 150.0, 250.0, 350.0, 450.0]

    def _map_panel():
        panel = mc_config.default_panel()
        fill = mc_config.default_fill_layer("ds0", "f")
        fill["style"].update({"method": "pcolormesh", "levels": levels})
        fill["style"]["colorbar"]["show"] = False
        panel["layers"] = [fill]
        return panel

    cfg = mc_config.default_figure_config()
    cfg["panels"] = [_map_panel(), _map_panel()]
    cfg["figure"]["layout"].update({"nrows": 1, "ncols": 2})
    cfg["figure"]["shared_colorbar"].update({"show": True, "label": "F"})
    fig = mc_render.render_figure(cfg, _DATASETS)
    # パネル2枚 + 共有カラーバーの axes
    assert len(fig.axes) == 3
    b0, b1 = fig.axes[0].get_position(), fig.axes[1].get_position()
    assert b0.x1 <= b1.x0 + 1e-6, "1×2 配置でパネル1がパネル2の左にない"
    assert abs(b0.y0 - b1.y0) < 1e-6, "1×2 配置で上下がずれている"
    # 共有カラーバーの目盛 = 代表 fill の離散境界 (BoundaryNorm)
    np.testing.assert_allclose(fig.axes[2].get_yticks(), levels)


def test_mosaic_rowspan_cell():
    ds = _bars_ds()
    _use(ds)
    p1 = _line_panel_for([mc_config.default_line_layer("ds0", "p")])
    p2 = _line_panel_for([mc_config.default_line_layer("ds0", "q")])
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [p1, p2]
    # A = 左上1セル、B = 右列を上下貫通 (結合セル)、左下は空き
    cfg["figure"]["layout"]["mosaic"] = "AB;.B"
    fig = mc_render.render_figure(cfg, _DATASETS)
    assert len(fig.axes) == 2
    hA = fig.axes[0].get_position().height
    hB = fig.axes[1].get_position().height
    assert hB > 1.5 * hA, f"結合セル B が縦に貫通していない (hA={hA}, hB={hB})"


# --- 1-33: 時間軸 — datetime x軸の値 = date2num(時刻) ---

def test_time_axis_values_are_date2num():
    import matplotlib.dates as mdates
    ds = _series_ds(20)
    _use(ds)
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "time"
    panel["selection"] = {}
    panel["layers"] = [mc_config.default_line_layer("ds0", "a")]
    fig = _render(panel)
    ln = fig.axes[0].lines[0]
    got = np.asarray(ln.get_xdata(orig=False), dtype=float)
    expected = mdates.date2num(ds["time"].values)
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(np.asarray(ln.get_ydata(), dtype=float),
                               ds["a"].values)


def test_violin_median_and_extrema_lines():
    ds = _series_ds(300)
    _use(ds)
    panel = mc_config.default_dist_panel()
    vi = mc_config.default_violin_layer("ds0")
    vi["variable"] = "a"
    vi["agg_dim"] = "time"
    panel["layers"] = [vi]
    fig = _render(panel)
    from matplotlib.collections import LineCollection
    vals = ds["a"].values
    targets = {"median": float(np.median(vals)),
               "min": float(vals.min()), "max": float(vals.max())}
    const_ys = set()
    for coll in fig.axes[0].collections:
        if isinstance(coll, LineCollection):
            for seg in coll.get_segments():
                seg = np.asarray(seg, dtype=float)
                if seg.size and np.allclose(seg[:, 1], seg[0, 1]):
                    const_ys.add(round(float(seg[0, 1]), 6))
    for name, target in targets.items():
        assert any(abs(cy - target) < 1e-6 for cy in const_ys), \
            f"violin の {name} = {target} の水平線が見つからない: {sorted(const_ys)}"


# --- 緯度経度線: ラベルの開始経度・緯度 (位相) (2026-08-24) ---

def test_gridline_label_start_offsets():
    """ラベルの開始値: locator が「開始値 + n×間隔」になる (前後両方向へ延長)。

    間隔が線と同じでも開始値がある軸は線と位置が合わないため、線用と
    ラベル用 (線非表示) の Gridliner に分割されること、線側の locator は
    従来どおり倍数のままであることも確認する。
    """
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import Gridliner
    _use(_map_ds())
    panel = mc_config.default_panel()
    panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
    panel["map"]["gridlines"].update({
        "lines": True, "labels": True,
        "lon_interval": 30.0, "lat_interval": 30.0,
        "label_lon_start": 15.0, "label_lat_start": 10.0})
    fig = _render(panel)
    gls = [a for a in fig.axes[0].artists if isinstance(a, Gridliner)]
    assert len(gls) == 2, "開始値の指定で線用・ラベル用に分割される"
    line_gl = next(g for g in gls if g.xlines)
    label_gl = next(g for g in gls if not g.xlines)
    # ラベル側: 開始値 + n×間隔 (負の n も含む)
    lons = np.asarray(label_gl.xlocator.tick_values(-180.0, 180.0))
    lats = np.asarray(label_gl.ylocator.tick_values(-90.0, 90.0))
    assert np.allclose((lons - 15.0) % 30.0, 0.0)
    assert np.allclose((lats - 10.0) % 30.0, 0.0)
    assert lons.min() < 15.0 and lats.min() < 10.0  # 開始値より小さい側へも延長
    # 線側: 従来どおり間隔の倍数
    assert isinstance(line_gl.xlocator, mticker.MultipleLocator)
    assert np.allclose(np.asarray(line_gl.xlocator.tick_values(-180.0, 180.0)) % 30.0, 0.0)


def test_gridline_plan_label_start_normalization():
    """gridline_plan の開始値の正規化: 0 → None、実効間隔の無い軸では無効。"""
    # 0 は倍数と同位相 → None (split もしない)
    plan = mc_render.gridline_plan({
        "lines": True, "labels": True,
        "lon_interval": 30.0, "lat_interval": 30.0,
        "label_lon_start": 0.0, "label_lat_start": 0.0})
    assert plan["label_lon_start"] is None and plan["label_lat_start"] is None
    assert not plan["split"]
    # 実効間隔が自動 (None) の軸では開始値を落とす (自動 locator に位相は無い)
    plan = mc_render.gridline_plan({
        "lines": True, "labels": True,
        "lat_interval": 30.0,
        "label_lon_start": 15.0, "label_lat_start": 10.0})
    assert plan["label_lon_start"] is None      # 経度: 間隔なし → 無効
    assert plan["label_lat_start"] == 10.0      # 緯度: 線の間隔を継承して有効
    assert plan["split"]
    # ラベルの間隔指定 + 開始値 (labels のみ表示なら split 不要)
    plan = mc_render.gridline_plan({
        "lines": False, "labels": True,
        "label_lon_interval": 20.0, "label_lon_start": 5.0})
    assert plan["label_lon_start"] == 5.0
    assert not plan["split"]


# --- ティックマーク: region の extent を保持する (2026-08-24) ---

def test_map_ticks_preserve_region_extent():
    """ティックマークの表示が「緯度経度範囲を指定」の extent を全球へ戻さない。

    Axis.set_ticks は「与えた ticks が全て見えるよう view interval を広げる」
    (matplotlib の仕様、実測 3.10) ため、全球分のティック位置を設定すると
    region で狭めた表示範囲が拡張されていた (2026-08-24 報告)。render が
    ティック設定後に表示範囲を復元することを検証する。
    """
    import cartopy.crs as ccrs
    _use(_map_ds())
    panel = mc_config.default_panel()
    panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
    panel["region"] = {"lon_min": 120.0, "lon_max": 150.0,
                       "lat_min": 10.0, "lat_max": 40.0}
    panel["map"]["ticks"] = {"show": True, "lon_interval": 30.0,
                             "lat_interval": 15.0,
                             "minor": {"show": True, "lon_interval": 10.0,
                                       "lat_interval": 5.0}}
    fig = _render(panel)
    ax = fig.axes[0]
    ext = ax.get_extent(crs=ccrs.PlateCarree())
    np.testing.assert_allclose(ext, [120.0, 150.0, 10.0, 40.0], atol=1e-6)
    # ティック自体は残っている (表示範囲内に major ティックがある)
    xlim = ax.get_xlim()
    assert any(xlim[0] - 1e-6 <= t <= xlim[1] + 1e-6 for t in ax.get_xticks())


# --- 極投影の緯度経度ラベル: label_sides の geo / inline (2026-07-09) ---
#
# cartopy は円形枠の極投影で、緯度ラベルを図中 (inline)、経度ラベルを円周沿い
# (geo。図枠の上下に接する位置のみ top/bottom) に分類する。left/right に分類
# されるラベルは存在しない。この分類と label_sides の対応を検証する。

def _polar_panel(label_sides: dict) -> dict:
    panel = mc_config.default_panel()
    panel["projection"] = {"name": "NorthPolarStereo", "central_longitude": 140.0,
                           "central_latitude": 0.0, "circular_boundary": True}
    panel["region"] = {"lon_min": 0.0, "lon_max": 360.0,
                      "lat_min": 20.0, "lat_max": 90.0}
    panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
    panel["map"]["gridlines"]["label_sides"] = label_sides
    return panel


def _visible_label_locs(fig):
    from collections import Counter
    from cartopy.mpl.gridliner import Gridliner
    fig.canvas.draw()  # ラベルは draw 時に生成される
    gl = [a for a in fig.axes[0].artists if isinstance(a, Gridliner)][0]
    return Counter(lb.loc for lb in gl._labels if lb.artist.get_visible())


def test_polar_label_sides_geo_and_inline():
    _use(_map_ds())
    all_on = {"left": True, "right": True, "top": True, "bottom": True,
              "geo": True, "inline": True}
    base = _visible_label_locs(_render(_polar_panel(dict(all_on))))
    # 分類の前提: 図中の緯度 (inline) と円周沿いの経度 (geo) があり、
    # left/right は存在しない
    assert base["inline"] > 0 and base["geo"] > 0
    assert base["left"] == 0 and base["right"] == 0

    off = _visible_label_locs(_render(_polar_panel(
        dict(all_on, geo=False, inline=False))))
    assert off["inline"] == 0 and off["geo"] == 0
    # 上下の接点ラベルは geo/inline と独立に残る
    assert off["top"] == base["top"] and off["bottom"] == base["bottom"]

    no_tb = _visible_label_locs(_render(_polar_panel(
        dict(all_on, top=False, bottom=False))))
    assert no_tb["top"] == 0 and no_tb["bottom"] == 0
    assert no_tb["inline"] == base["inline"] and no_tb["geo"] == base["geo"]


def test_polar_pole_longitude_label_hidden():
    """pole_label=False で極 (投影原点) 付近の経度ラベルだけが消える。

    経度範囲を指定した扇形では極が境界上の点になり、そこで終わる経度線の
    ラベルが極のそばに描かれる (gridlines.pole_label で制御)。極ラベルは
    原点から ~1e5 m、弧沿いのラベルは ~7e6 m なので距離で判別する。
    """
    _use(_map_ds())

    def make(pole_label: bool) -> dict:
        panel = mc_config.default_panel()
        panel["projection"] = {"name": "NorthPolarStereo",
                               "central_longitude": 180.0,
                               "central_latitude": 0.0,
                               "circular_boundary": True}
        panel["region"] = {"lon_min": 90.0, "lon_max": 360.0,
                           "lat_min": 30.0, "lat_max": 90.0}
        panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
        panel["map"]["gridlines"]["pole_label"] = pole_label
        return panel

    def x_label_dists(fig) -> list[float]:
        from cartopy.mpl.gridliner import Gridliner
        fig.canvas.draw()
        gl = [a for a in fig.axes[0].artists if isinstance(a, Gridliner)][0]
        return [float(np.hypot(*lb.artist.get_position()))
                for lb in gl._labels
                if lb.xy == "x" and lb.artist.get_visible()]

    near = 1.0e6  # 原点からこの距離未満 = 極のラベル
    base = x_label_dists(_render(make(True)))
    hidden = x_label_dists(_render(make(False)))
    assert any(d < near for d in base), "前提: 極のそばに経度ラベルがあるはず"
    assert not any(d < near for d in hidden), "pole_label=False で極のラベルが残っている"
    # 弧沿い・辺の経度ラベルは影響を受けない
    assert (sum(d >= near for d in hidden) == sum(d >= near for d in base))


def test_polar_lat_label_placement_edge():
    """lat_label_placement="edge" で緯度ラベルが図中 (inline) から枠沿いに移る。

    扇形では緯度線 (同心円) が扇の縁と交差するため、y_inline=False で
    交点にラベルが置かれる (分類は geo)。既定 ("inline") は図中。
    """
    _use(_map_ds())

    def make(placement: str) -> dict:
        panel = mc_config.default_panel()
        panel["projection"] = {"name": "NorthPolarStereo",
                               "central_longitude": 150.0,
                               "central_latitude": 0.0,
                               "circular_boundary": True}
        panel["region"] = {"lon_min": 100.0, "lon_max": 200.0,
                           "lat_min": 30.0, "lat_max": 90.0}
        panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
        panel["map"]["gridlines"]["lat_label_placement"] = placement
        return panel

    def y_label_locs(fig):
        from collections import Counter
        from cartopy.mpl.gridliner import Gridliner
        fig.canvas.draw()
        gl = [a for a in fig.axes[0].artists if isinstance(a, Gridliner)][0]
        return Counter(lb.loc for lb in gl._labels
                       if lb.xy == "y" and lb.artist.get_visible())

    inline = y_label_locs(_render(make("inline")))
    edge = y_label_locs(_render(make("edge")))
    assert inline["inline"] > 0 and inline["geo"] == 0
    assert edge["inline"] == 0, "edge 指定で図中の緯度ラベルが残っている"
    assert edge["geo"] > 0, "edge 指定で枠沿いの緯度ラベルが出ていない"


def _polar_edge_side_panel(side: str, clon: float, lon0: float,
                           lon1: float) -> dict:
    panel = mc_config.default_panel()
    panel["projection"] = {"name": "NorthPolarStereo",
                           "central_longitude": clon,
                           "central_latitude": 0.0,
                           "circular_boundary": True}
    panel["region"] = {"lon_min": lon0, "lon_max": lon1,
                       "lat_min": 30.0, "lat_max": 90.0}
    panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
    panel["map"]["gridlines"]["lat_label_placement"] = "edge"
    panel["map"]["gridlines"]["lat_label_edge_side"] = side
    return panel


def _y_label_lons(fig) -> list[float]:
    """可視の緯度ラベルの地理経度 (投影座標から逆変換、-180〜180)。"""
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import Gridliner
    fig.canvas.draw()
    ax = fig.axes[0]
    gl = [a for a in ax.artists if isinstance(a, Gridliner)][0]
    lons = []
    for lb in gl._labels:
        if lb.xy == "y" and lb.artist.get_visible():
            x, y = lb.artist.get_position()
            lons.append(float(ccrs.PlateCarree().transform_point(
                x, y, ax.projection)[0]))
    return lons


def _nearest_edge(lon: float, edge_a: float, edge_b: float) -> float:
    """ラベル経度が近いほうの縁を返す (フィルタと同じ角距離判定)。

    高緯度のラベルはアンカーが縁からオフセットされ、逆変換した経度が
    20° 程度ずれることがある (実測) ため、絶対許容ではなく帰属で判定する。
    """
    da = abs((lon - edge_a + 180.0) % 360.0 - 180.0)
    db = abs((lon - edge_b + 180.0) % 360.0 - 180.0)
    return edge_a if da <= db else edge_b


def test_polar_lat_label_edge_side():
    """lat_label_edge_side で枠沿いの緯度ラベルを扇の縁単位で絞れる。

    判定は縁の経度への帰属 (図の中央との位置比較ではない —
    render.apply_gridline_label_filters)。left/right → 縁の対応は
    polar_sector_edge_lons (投影後の x で解決)。
    """
    _use(_map_ds())
    # 対称な扇形 (100–200°E、中心150°): left = 100°E の縁、right = 200°E の縁
    both = _y_label_lons(_render(_polar_edge_side_panel("both", 150.0, 100.0, 200.0)))
    left = _y_label_lons(_render(_polar_edge_side_panel("left", 150.0, 100.0, 200.0)))
    right = _y_label_lons(_render(_polar_edge_side_panel("right", 150.0, 100.0, 200.0)))
    n100 = sum(_nearest_edge(v, 100.0, 200.0) == 100.0 for v in both)
    n200 = len(both) - n100
    assert n100 > 0 and n200 > 0, "前提: both で両縁にラベルがあるはず"
    assert len(left) == n100 and all(
        _nearest_edge(v, 100.0, 200.0) == 100.0 for v in left)
    assert len(right) == n200 and all(
        _nearest_edge(v, 100.0, 200.0) == 200.0 for v in right)


def test_polar_lat_label_edge_side_tilted():
    """切り欠き扇形 (90–360°E、中心180°) でも縁単位で選べる。

    右の縁 = 0°(=360) は図のほぼ中央を縦に走るため、「図の中央から左右」の
    判定では縁単位にならない構成 (縁への帰属判定の回帰テスト)。
    """
    _use(_map_ds())
    left = _y_label_lons(_render(_polar_edge_side_panel("left", 180.0, 90.0, 360.0)))
    right = _y_label_lons(_render(_polar_edge_side_panel("right", 180.0, 90.0, 360.0)))
    assert left and all(_nearest_edge(v, 90.0, 0.0) == 90.0 for v in left)
    assert right and all(_nearest_edge(v, 90.0, 0.0) == 0.0 for v in right)


def test_polar_lat_label_rotation():
    """lat_label_rotation で緯度ラベルの回転を固定できる (0 = 水平)。

    極投影の枠沿いラベルは cartopy が縁の向きに自動回転する (rotate_labels)。
    ylabel_style の rotation は自動回転より後に適用されるため上書きできる
    (cartopy 0.25 実測)。経度ラベルの自動回転には影響しない。
    """
    _use(_map_ds())

    def rotations(lat_rot):
        from cartopy.mpl.gridliner import Gridliner
        panel = _polar_edge_side_panel("both", 150.0, 100.0, 200.0)
        panel["map"]["gridlines"]["lat_label_rotation"] = lat_rot
        fig = _render(panel)
        fig.canvas.draw()
        gl = [a for a in fig.axes[0].artists if isinstance(a, Gridliner)][0]
        yr = {round(lb.artist.get_rotation(), 1) for lb in gl._labels
              if lb.xy == "y" and lb.artist.get_visible()}
        xr_ = {round(lb.artist.get_rotation(), 1) for lb in gl._labels
               if lb.xy == "x" and lb.artist.get_visible()}
        return yr, xr_

    auto_y, auto_x = rotations(None)
    assert any(r not in (0.0,) for r in auto_y), "前提: 自動回転で傾いているはず"
    flat_y, flat_x = rotations(0.0)
    assert flat_y == {0.0}, f"0° 固定で水平になっていない: {flat_y}"
    tilt_y, _ = rotations(45.0)
    assert tilt_y == {45.0}
    # 経度ラベルの回転は自動のまま変わらない
    assert flat_x == auto_x


def test_robinson_geo_lat_labels_follow_sides():
    """Robinson の高緯度ラベル (geo 分類) も左右チェックに追従する。

    Robinson の楕円境界では中緯度の緯度ラベルは left/right に分類されるが、
    高緯度 (±60/±80°、境界が湾曲して図枠の辺から離れる位置) は geo に
    分類され、従来は左右チェックで消えなかった。geo ラベルを位置で最寄りの
    辺に帰属させるフィルタ (apply_gridline_label_filters の geo_hidden_sides)
    で全ラベルが左右チェックに追従することを検証する。
    """
    _use(_map_ds())

    def visible_y(sides):
        from cartopy.mpl.gridliner import Gridliner
        panel = mc_config.default_panel()
        panel["projection"] = {"name": "Robinson", "central_longitude": 140.0,
                               "central_latitude": 0.0}
        panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
        panel["map"]["gridlines"]["lat_interval"] = 20.0
        panel["map"]["gridlines"]["label_sides"] = sides
        fig = _render(panel)
        fig.canvas.draw()
        ax = fig.axes[0]
        gl = [a for a in ax.artists if isinstance(a, Gridliner)][0]
        xmid = 0.5 * sum(ax.get_xlim())
        out = []
        for lb in gl._labels:
            if lb.xy == "y" and lb.artist.get_visible():
                out.append((lb.loc,
                            "L" if lb.artist.get_position()[0] < xmid else "R"))
        return out

    all_on = {"left": True, "right": True, "top": True, "bottom": True}
    base = visible_y(dict(all_on))
    # 前提: geo 分類の緯度ラベルが左右にある (これが無いと検証にならない)
    assert any(loc == "geo" and s == "L" for loc, s in base)
    assert any(loc == "geo" and s == "R" for loc, s in base)

    no_left = visible_y(dict(all_on, left=False))
    assert no_left and all(s == "R" for _, s in no_left), \
        f"左 OFF で左側の緯度ラベルが残っている: {no_left}"
    # 右側は geo 込みで不変
    assert len(no_left) == sum(s == "R" for _, s in base)

    none = visible_y(dict(all_on, left=False, right=False))
    assert none == [], f"左右 OFF で緯度ラベルが残っている: {none}"


def test_tick_label_rotation():
    """axis.x/y_tick_rotation で目盛文字の回転が全ラベルに適用される。

    共有 _apply_tick_settings (1D/2D/断面/dist が共用) の tick_params
    labelrotation 経路。0/キー無しは回転なし (旧設定不変)。
    """
    ds = _line_ds()
    _use(ds)

    def rotations(xr_, yr):
        panel = mc_config.default_line_panel()
        panel["x_dim"] = "level"
        panel["selection"] = {}
        panel["axis"]["x_tick_rotation"] = xr_
        panel["axis"]["y_tick_rotation"] = yr
        panel["layers"] = [mc_config.default_line_layer("ds0", "g")]
        fig = _render(panel)
        fig.canvas.draw()
        ax = fig.axes[0]
        return ({t.get_rotation() for t in ax.get_xticklabels()},
                {t.get_rotation() for t in ax.get_yticklabels()})

    xrot, yrot = rotations(45.0, -30.0)
    assert xrot == {45.0}
    assert yrot == {330.0} or yrot == {-30.0}  # matplotlib は 0-360 に正規化し得る
    x0, y0 = rotations(0.0, 0.0)
    assert x0 == {0.0} and y0 == {0.0}


def test_scatter_errorbar_segments():
    """散布図のエラーバー: 線分の半長 = |誤差変数| (x/y 独立)。

    誤差は本体と同じ切り出しで、値の変換 (倍率・加算) は掛けず、絶対値で正規化
    される (errorbar_kwargs / _scatter_xy_arrays with_errors)。2026-09-29 までは
    倍率だけ掛けていた (KNOWN_ISSUES.md KI-1)。
    """
    from matplotlib.collections import LineCollection
    ds = xr.Dataset(
        {"a": ("i", [1.0, 2.0, 3.0]), "b": ("i", [2.0, 1.0, 4.0]),
         "ea": ("i", [0.5, 1.0, 0.2]), "eb": ("i", [-0.3, 0.6, 0.1])},
        coords={"i": [0, 1, 2]})
    _use(ds)
    panel = mc_config.default_scatter_panel()
    panel["x_variable"] = "a"
    panel["y_variable"] = "b"
    layer = mc_config.default_scatter_layer("ds0")
    layer["drawing_dim"] = "i"
    layer["style"]["x_value_scale"] = 2.0    # 誤差には掛からない
    layer["style"]["x_value_offset"] = 10.0  # 誤差には掛からない
    layer["style"]["errorbar"] = {"x_variable": "ea", "y_variable": "eb",
                                  "color": "#000000", "linewidth": 1.0,
                                  "capsize": 3.0}
    panel["layers"] = [layer]
    fig = _render(panel)
    ax = fig.axes[0]
    lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lcs) == 2, f"x/y のエラーバー LineCollection が2つのはず: {len(lcs)}"
    half = {}
    for lc in lcs:
        segs = np.asarray(lc.get_segments(), dtype=float)
        dx = np.ptp(segs[:, :, 0], axis=1) / 2.0
        dy = np.ptp(segs[:, :, 1], axis=1) / 2.0
        if np.allclose(dy, 0.0):
            half["x"] = np.sort(dx)
        else:
            half["y"] = np.sort(dy)
    np.testing.assert_allclose(half["x"], np.sort(np.abs(ds["ea"].values)))
    np.testing.assert_allclose(half["y"], np.sort(np.abs(ds["eb"].values)))


# --- zero_white: 0 を含むビンだけが純白になる ---

def test_zero_white_fill_pcolormesh_even_and_odd_bins():
    """zero_white: 0 が境界なら両隣の2ビン、ビン内なら中央の1ビンだけが白。

    QuadMesh の実 facecolor をセル値から独立に判定して数える
    (「白のはずのセルは純白・それ以外は非白」)。
    """
    ds = _map_ds()
    _use(ds)

    def facecolors(levels):
        panel = mc_config.default_panel()
        fill = mc_config.default_fill_layer("ds0", "f")
        fill["style"].update({"method": "pcolormesh", "cmap": "RdBu_r",
                              "zero_white": True, "levels": levels,
                              "value_offset": -200.0})
        panel["layers"] = [fill]
        fig = _render(panel)
        qm = _quadmesh(fig.axes[0])
        fig.canvas.draw()
        vals = np.asarray(qm.get_array(), dtype=float).ravel()
        cols = np.asarray(qm.get_facecolor(), dtype=float)
        return vals, cols

    white = np.array([1.0, 1.0, 1.0, 1.0])

    # 偶数側: 0 がレベル境界 → [-50,0) と [0,50) の2ビンが白
    vals, cols = facecolors([-100.0, -50.0, 0.0, 50.0, 100.0])
    in_white_bin = (vals >= -50.0) & (vals < 50.0)
    assert in_white_bin.any() and (~in_white_bin).any()  # 両群が存在する
    assert np.allclose(cols[in_white_bin], white)
    assert not np.isclose(cols[~in_white_bin], white).all(axis=1).any()

    # 奇数側: 0 がビン内部 → [-25,25) の中央1ビンだけが白
    vals, cols = facecolors([-75.0, -25.0, 25.0, 75.0])
    in_white_bin = (vals >= -25.0) & (vals < 25.0)
    assert in_white_bin.any() and (~in_white_bin).any()
    assert np.allclose(cols[in_white_bin], white)
    assert not np.isclose(cols[~in_white_bin], white).all(axis=1).any()


def test_zero_white_discrete_color_kwargs():
    """共通ヘルパー discrete_color_kwargs の zero_white 挙動 (extend 込み)。

    vector / stream / map_scatter / track / bubble / heatmap / hist2d /
    hexbin が共有する経路の単体検証。
    """
    white = (1.0, 1.0, 1.0, 1.0)
    style = {"cmap": "RdBu_r", "zero_white": True, "extend": "both",
             "levels": [-10.0, -5.0, 0.0, 5.0, 10.0]}
    dk = mc_render.discrete_color_kwargs(style, -10.0, 10.0)
    cmap, norm = dk["cmap"], dk["norm"]
    # 0 を挟む2ビンが白、外側ビンは非白
    assert tuple(cmap(norm(-2.5))) == white
    assert tuple(cmap(norm(2.5))) == white
    assert tuple(cmap(norm(-7.5))) != white
    assert tuple(cmap(norm(7.5))) != white
    # extend の端色は基底 cmap の両端 (白ではない)
    assert tuple(cmap(norm(-100.0))) != white
    assert tuple(cmap(norm(100.0))) != white

    # zero_white off なら従来どおり norm のみ (cmap は返さない)
    dk_off = mc_render.discrete_color_kwargs(
        {"cmap": "RdBu_r", "levels": [-10.0, 0.0, 10.0]}, -10.0, 10.0)
    assert "cmap" not in dk_off and "norm" in dk_off
    # レベルが 0 をまたがない場合は白ビンなし (全ビン非白)
    dk_pos = mc_render.discrete_color_kwargs(
        {"cmap": "RdBu_r", "zero_white": True,
         "levels": [5.0, 10.0, 15.0]}, 5.0, 15.0)
    for v in (7.5, 12.5):
        assert tuple(dk_pos["cmap"](dk_pos["norm"](v))) != white


def test_contour_hidden_level_labels_removed():
    """emphasis の太さ 0 (非表示) にしたレベルは等値線ラベルも一緒に消える。"""
    lon = np.linspace(100.0, 180.0, 41)
    lat = np.linspace(0.0, 40.0, 41)
    val = lat[:, None] * 10.0 + lon[None, :] * 0.1   # 10.0 〜 418.0 の滑らかな場
    ds = xr.Dataset(
        {"f": (("lat", "lon"), val)},
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})})
    _use(ds)

    def label_texts(emphasis):
        panel = mc_config.default_panel()
        contour = mc_config.default_contour_layer("ds0", "f")
        contour["style"].update({
            "levels": [100.0, 200.0, 300.0, 400.0],
            "labels": {"show": True, "fmt": "%g"},
            "emphasis": emphasis})
        panel["layers"] = [contour]
        fig = _render(panel)
        css = _contour_sets(fig.axes[0])
        assert len(css) == 1
        return [t.get_text() for t in css[0].labelTexts]

    # 非表示なし: 全レベルのラベルが出る
    texts = label_texts({"levels": None, "linewidth": None, "color": None})
    assert {"100", "200", "300", "400"} <= set(texts)
    # 200 を太さ 0 で非表示 → ラベル 200 も消え、他は残る
    texts = label_texts({"levels": [200.0], "linewidth": 0.0, "color": None})
    assert "200" not in texts
    assert {"100", "300", "400"} <= set(texts)
    # 太さ 0 以外 (強調) ではラベルは全レベルに出たまま
    texts = label_texts({"levels": [200.0], "linewidth": 3.0, "color": None})
    assert {"100", "200", "300", "400"} <= set(texts)


# --- カラーバー: 目盛りとラベルの表示サイド (flip_ticks / label_opposite, 2026-08-25) ---

@pytest.mark.parametrize(
    "location,flip_ticks,label_opp,expected",
    [
        # 縦 (右配置): (目盛り側, ラベル側)
        ("right", False, False, ("right", "right")),
        ("right", False, True, ("right", "left")),
        ("right", True, False, ("left", "left")),
        ("right", True, True, ("left", "right")),
        # 横 (下配置)
        ("bottom", False, True, ("bottom", "top")),
        ("bottom", True, True, ("top", "bottom")),
    ])
def test_colorbar_tick_and_label_sides(location, flip_ticks, label_opp, expected):
    """flip_ticks (目盛り側) と label_opposite (ラベルを目盛りの反対側) の組で
    カラーバー本体の両サイドに目盛り文字とラベルを振り分けられる。"""
    _use(_map_ds())
    panel = mc_config.default_panel()
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["colorbar"].update({"label": "K", "location": location,
                                      "flip_ticks": flip_ticks,
                                      "label_opposite": label_opp})
    panel["layers"] = [fill]
    fig = _render(panel)
    cax = fig.axes[-1]  # colorbar の axes (最後に追加される)
    axis = cax.yaxis if location in ("left", "right") else cax.xaxis
    assert (axis.get_ticks_position(), axis.get_label_position()) == expected


# --- 1-49: ライン (束) — 全スライスが同じ体裁で描かれ、統計線 = numpy の独立計算 ---

def test_line_bundle_draws_every_slice_and_summary_lines():
    """line_bundle: bundle_dim の各スライスが 1 本ずつ (同じ色・透明度・線種)、
    凡例は束で 1 つ。統計線は束ねた方向の nanmean / nanmin / nanmax /
    nanpercentile / nanmedian (NaN 除外) と一致し、2 本組は凡例 1 つ・
    ラベル無しは凡例に出ない。残りの次元 (level) はレイヤー側の固定が効く。"""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    level = np.array([1000.0, 500.0])
    rng = np.random.default_rng(3)
    vals = rng.normal(size=(4, 2, 5))          # (member, level, x)
    vals[2, 1, 3] = np.nan                     # 集計で NaN が除外されること
    ds = xr.Dataset({"g": (("member", "level", "x"), vals)},
                    coords={"member": np.arange(1, 5), "level": level, "x": x})
    _use(ds)
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "x"
    panel["selection"] = {}
    layer = mc_config.default_line_bundle_layer("ds0", "g", "member")
    layer["selection"] = {"level": 500.0}
    layer["style"].update({"color": "#123456", "alpha": 0.3, "linestyle": "dashed",
                           "label": "members", "value_scale": 2.0})
    summaries = []
    for stat, q, label, draw in [("mean", None, "mean", "lines"),
                                 ("minmax", None, "min-max", "lines"),
                                 ("pct_range", (25.0, 75.0), "p25-p75", "lines"),
                                 ("percentile", (50.0, None), None, "lines"),
                                 ("median", None, "median", "lines"),
                                 ("minmax", None, "min-max band", "band"),
                                 ("pct_range", (10.0, 90.0), "p10-p90 band", "band_lines"),
                                 ("mean", None, None, "band"),   # 1 本は帯にならない
                                 ("std_range", None, "mean±1.5σ", "lines")]:
        smm = mc_config.default_bundle_summary(stat)
        if q is not None:
            smm["q_low"] = q[0]
            if q[1] is not None:
                smm["q_high"] = q[1]
        smm["draw"] = draw
        if stat == "std_range":
            smm["k_std"] = 1.5
        smm["style"].update({"color": "#abcdef", "label": label, "alpha": 0.4})
        summaries.append(smm)
    layer["summaries"] = summaries
    panel["layers"] = [layer]
    fig = _render(panel)
    ax = fig.axes[0]
    y = vals[:, 1, :].T * 2.0                  # level=500 を固定して値変換 (x, member)
    lines = ax.lines
    # 線: 束 4 + lines 描画の統計 (1+2+2+1+1) + band_lines の縁 2 + 帯にならない mean 1
    #     + 平均 ± 1.5σ の 2
    assert len(lines) == 4 + (1 + 2 + 2 + 1 + 1) + 2 + 1 + 2
    for k in range(4):
        xy = lines[k].get_xydata()
        np.testing.assert_allclose(xy[:, 0], x)
        np.testing.assert_allclose(xy[:, 1], y[:, k])
        assert lines[k].get_color() == "#123456"
        assert lines[k].get_alpha() == 0.3
        assert lines[k].get_linestyle() == "--"
    expected = [np.nanmean(y, axis=1),
                np.nanmin(y, axis=1), np.nanmax(y, axis=1),
                np.nanpercentile(y, 25.0, axis=1), np.nanpercentile(y, 75.0, axis=1),
                np.nanpercentile(y, 50.0, axis=1),
                np.nanmedian(y, axis=1),
                np.nanpercentile(y, 10.0, axis=1), np.nanpercentile(y, 90.0, axis=1),
                np.nanmean(y, axis=1),
                np.nanmean(y, axis=1) - 1.5 * np.nanstd(y, axis=1, ddof=1),  # 標本標準偏差
                np.nanmean(y, axis=1) + 1.5 * np.nanstd(y, axis=1, ddof=1)]
    assert len(lines) - 4 == len(expected)
    for ln, exp in zip(lines[4:], expected):
        np.testing.assert_allclose(ln.get_xydata()[:, 1], exp)
        assert ln.get_color() == "#abcdef"
    # 帯: fill_between の PolyCollection が 2 つ (minmax 帯・p10-p90 帯+縁)。
    # 頂点は [始点, (x, 下側)..., 終点, (x, 上側) 逆順...] (matplotlib の組み立て)
    from matplotlib.collections import PolyCollection
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(bands) == 2
    n = x.size
    for band, (lo, hi) in zip(bands, [(np.nanmin(y, axis=1), np.nanmax(y, axis=1)),
                                      (np.nanpercentile(y, 10.0, axis=1),
                                       np.nanpercentile(y, 90.0, axis=1))]):
        v = band.get_paths()[0].vertices
        np.testing.assert_allclose(v[1:n + 1, 0], x)
        np.testing.assert_allclose(v[1:n + 1, 1], lo)
        np.testing.assert_allclose(v[n + 2:2 * n + 2, 1], hi[::-1])
        assert band.get_alpha() == 0.4
        assert band.get_linewidth()[0] == 0.0
    # 凡例: 束 1 つ + ラベル付き統計線 1 組につき 1 つ (2 本組・帯でも 1 つ、None は出ない)
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["members", "mean", "min-max", "p25-p75", "median",
                      "min-max band", "p10-p90 band", "mean±1.5σ"]
    # 固定していない level は bundle_dim 以外に残せない (2 次元にならない) → 明示エラー
    layer["selection"] = {}
    with pytest.raises(ValueError):
        _render(panel)


def test_line_bundle_zero_linewidth_hides_lines():
    """line_bundle: 束の線幅 0 なら束の線を描かず (マーカー指定があっても)、統計線の
    線幅 0 も描かない。帯は残り、凡例は描かれたものだけ。"""
    x = np.arange(4.0)
    vals = np.arange(12.0).reshape(3, 4)          # (member, x)
    ds = xr.Dataset({"g": (("member", "x"), vals)},
                    coords={"member": np.arange(3), "x": x})
    _use(ds)
    panel = mc_config.default_line_panel()
    panel["x_dim"] = "x"
    panel["selection"] = {}
    layer = mc_config.default_line_bundle_layer("ds0", "g", "member")
    layer["style"].update({"linewidth": 0.0, "marker": "o", "label": "members"})
    band = mc_config.default_bundle_summary("minmax")
    band["draw"] = "band"
    band["style"].update({"label": "min-max"})
    hidden = mc_config.default_bundle_summary("mean")
    hidden["style"].update({"linewidth": 0.0, "label": "mean"})
    edge_hidden = mc_config.default_bundle_summary("pct_range")
    edge_hidden["draw"] = "band_lines"
    edge_hidden["style"].update({"linewidth": 0.0, "label": "p5-p95"})
    layer["summaries"] = [band, hidden, edge_hidden]
    panel["layers"] = [layer]
    fig = _render(panel)
    ax = fig.axes[0]
    assert len(ax.lines) == 0
    from matplotlib.collections import PolyCollection
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(bands) == 2
    v = bands[0].get_paths()[0].vertices
    np.testing.assert_allclose(v[1:5, 1], vals.min(axis=0))
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["min-max", "p5-p95"]


# --- 地図: Natural Earth の解像度 (map.resolution、2026-09-23) ---

def _ne_features(ax) -> dict:
    """FeatureArtist を Natural Earth の名前 → scaler で返す (collections に入る)。"""
    from cartopy.mpl.feature_artist import FeatureArtist
    return {a._feature.name: a._feature.scaler
            for a in ax.collections if isinstance(a, FeatureArtist)}


def test_map_natural_earth_resolution_applies_to_all_features():
    """map.resolution は海岸線・国境線・陸域・海域の 4 つに同じスケールを与える。

    明示 ("110m") では 4 feature とも固定スケール、既定 ("auto") とキー無し
    (旧設定) では cartopy の AdaptiveScaler (表示範囲で 110m/50m/10m を選ぶ)。
    110m のファイルはテスト環境に常にある (draw が走るため 50m/10m は
    ダウンロードを誘発するので使わない)。
    """
    from cartopy.feature import AdaptiveScaler, Scaler
    _use(_map_ds())
    names = {"coastline", "admin_0_boundary_lines_land", "land", "ocean"}

    def _panel(resolution):
        panel = mc_config.default_panel()
        panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
        panel["map"]["borders"] = True
        panel["map"]["land"]["show"] = True
        panel["map"]["ocean"]["show"] = True
        if resolution is _MISSING:
            del panel["map"]["resolution"]
        else:
            panel["map"]["resolution"] = resolution
        return panel

    feats = _ne_features(_render(_panel("110m")).axes[0])
    assert set(feats) == names
    for name, scaler in feats.items():
        assert type(scaler) is Scaler and scaler.scale == "110m", name

    for res in ("auto", _MISSING):
        feats = _ne_features(_render(_panel(res)).axes[0])
        assert set(feats) == names
        assert all(isinstance(sc, AdaptiveScaler) for sc in feats.values()), res

    with pytest.raises(ValueError):
        mc_render.natural_earth_scale({"resolution": "5m"})


_MISSING = object()


# --- 地図: 陸をデータの上に描く (map.land.above_data、2026-09-23) ---

def _land_fg_panel(above: bool) -> dict:
    panel = mc_config.default_panel()
    panel["map"]["land"].update({"show": True, "above_data": above})
    panel["map"]["borders"] = True
    panel["map"]["boxes"] = [
        {"lon_min": 110.0, "lon_max": 150.0, "lat_min": 10.0, "lat_max": 40.0,
         "color": "red", "linewidth": 1.0, "linestyle": "solid"}]
    panel["markers"] = [{"marker": "*", "coord": "data", "x": 120.0, "y": 20.0,
                         "size": 12.0, "color": "#d62728"}]
    fill = mc_config.default_fill_layer("ds0", "f")
    contour = mc_config.default_contour_layer("ds0", "f")
    contour["style"]["labels"]["show"] = True
    vector = mc_config.default_vector_layer("ds0", "u", "v")
    vector["style"].setdefault("key", {}).update(
        {"show": True, "x": 0.5, "y": 0.5, "label": "k"})
    stream = mc_config.default_stream_layer("ds0", "u", "v")
    scatter = mc_config.default_map_scatter_layer("ds0", "f")
    panel["layers"] = [fill, contour, vector, stream, scatter]
    return panel


def _map_zorders(fig) -> dict:
    """地図の要素ごとの zorder (FeatureArtist は collections、Gridliner は artists)。"""
    from cartopy.mpl.feature_artist import FeatureArtist
    from cartopy.mpl.gridliner import Gridliner
    from matplotlib.collections import PathCollection
    from matplotlib.quiver import Quiver, QuiverKey
    ax = fig.axes[0]
    fig.canvas.draw()
    feats = {a._feature.name: a.get_zorder()
             for a in ax.collections if isinstance(a, FeatureArtist)}
    gl = [a for a in ax.artists if isinstance(a, Gridliner)][0]
    contours = [c for c in ax.collections if isinstance(c, ContourSet)]
    return {
        "land": feats["land"],
        "coast": feats["coastline"],
        "borders": feats["admin_0_boundary_lines_land"],
        "gridliner": gl.get_zorder(),                            # 子の線は本体の draw で描かれる
        "lines": {ln.get_zorder() for ln in ax.lines},           # box + 記号注記
        "quiverkey": {a.get_zorder() for a in ax.artists if isinstance(a, QuiverKey)},
        "scatter": {c.get_zorder() for c in ax.collections
                    if isinstance(c, PathCollection)},
        "area": {c.get_zorder() for c in ax.collections           # 塗り (filled) + 矢羽
                 if (isinstance(c, ContourSet) and c.filled) or isinstance(c, Quiver)},
        "contour": {c.get_zorder() for c in contours if not c.filled},
        "stream": {c.get_zorder() for c in ax.collections
                   if type(c).__name__ == "LineCollection" and c not in
                   gl.xline_artists + gl.yline_artists},
        "clabel": {t.get_zorder() for c in contours for t in getattr(c, "labelTexts", [])},
    }


def _count_rgb(fig, pred) -> int:
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())
    return int(pred(a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)).sum())


def test_map_land_above_data_zorders():
    """陸を前景に描く: 陸 (1.4) は面の格子データ (塗り・ハッチ・矢羽 = 1、基準ベクトル
    = 1.1) の上、海岸線・国境線 (1.5)・等値線・流線・box・記号注記・グリッド線 (2)・
    等値線ラベル (4) の下。陸に隠れてはいけない点要素 (散布点・トラックの点・基準
    ベクトル) は海岸線の高さ 1.5 に上げる。既定 (False) は従来の zorder のまま。"""
    _use(_map_ds())
    z = _map_zorders(_render(_land_fg_panel(True)))
    land = mc_render.LAND_FG_ZORDER
    assert z["land"] == land == 1.4
    assert max(z["area"]) < land                    # 塗り・矢羽は陸の下
    assert min(z["contour"] | z["stream"] | z["lines"]) == 2 > land
    assert z["gridliner"] == 2 > land and z["clabel"] == {4}
    assert z["coast"] == z["borders"] == 1.5 > land
    assert z["quiverkey"] == z["scatter"] == {mc_render.MAP_FG_ZORDER} == {1.5}

    z0 = _map_zorders(_render(_land_fg_panel(False)))
    assert z0["land"] == 0 and z0["coast"] == z0["borders"] == 1.5
    assert z0["quiverkey"] == {1.1} and z0["scatter"] == {1}
    assert z0["contour"] == z0["stream"] == z0["lines"] == {2} and z0["gridliner"] == 2
    # 共有 kwargs ヘルパー: 既定では zorder を出さない (旧 config の出力を変えない)
    for fn in (mc_render.map_scatter_kwargs, mc_render.track_point_kwargs):
        assert "zorder" not in fn({})
        assert fn({}, foreground=True)["zorder"] == mc_render.MAP_FG_ZORDER


def test_map_land_above_data_keeps_gridlines_visible():
    """陸を前景に描いてもグリッド線は消えない (画素で検査)。

    cartopy の Gridliner は線の LineCollection を自分の draw (zorder 2) で描くため、
    線側の zorder は効かない (陸 2.5 + 線 2.6 で全消失した実例 2026-09-23)。
    陸を 2 未満に置くことで担保する。ほぼ全部が陸の領域 (中央アジア) で、
    マゼンタのグリッド線の画素数が背景モードと同程度に残ることを確認する。
    """
    lon = np.arange(60.0, 130.0, 10.0)
    lat = np.arange(20.0, 60.0, 10.0)
    _use(xr.Dataset({"f": (("lat", "lon"), np.ones((lat.size, lon.size)))},
                    coords={"lat": lat, "lon": lon}))

    def _panel(above):
        panel = mc_config.default_panel()
        panel["layers"] = [mc_config.default_fill_layer("ds0", "f")]
        panel["layers"][0]["style"]["cmap"] = "Greys"
        panel["region"] = {"lon_min": 70.0, "lon_max": 110.0,
                           "lat_min": 30.0, "lat_max": 50.0}
        panel["map"]["land"].update({"show": True, "above_data": above})
        panel["map"]["gridlines"].update({"color": "#ff00ff", "linestyle": "-",
                                          "width": 3.0, "lon_interval": 10.0,
                                          "lat_interval": 5.0})
        return panel

    def magenta(r, g, b):
        return (r > 200) & (g < 80) & (b > 200)
    n_bg = _count_rgb(_render(_panel(False)), magenta)
    n_fg = _count_rgb(_render(_panel(True)), magenta)
    assert n_bg > 1000, n_bg
    assert n_fg >= 0.9 * n_bg, (n_fg, n_bg)


# --- 1-18: curvilinear (2 次元座標) の fill — 配列 = 領域切り出し後の値、切り出しは
#           マスクの外接矩形 (独立計算)、メッシュの範囲 = 経緯度の範囲 ---

def _curvilinear_ds(ny: int = 12, nx: int = 16) -> xr.Dataset:
    """位置から値が一意に決まる小さな curvilinear 格子 (少し傾けた経緯度)。"""
    j, i = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    lon2d = 120.0 + 2.0 * j + 0.3 * i          # 行ごとに東へずれる
    lat2d = 20.0 + 2.0 * i - 0.2 * j           # 列ごとに南へずれる
    vals = 10.0 * i + j                         # 行番号 × 10 + 列番号
    return xr.Dataset(
        {"f": (("y", "x"), vals, {"units": "1"})},
        coords={"y": ("y", np.arange(1, ny + 1, dtype=float), {"units": ""}),
                "x": ("x", np.arange(1, nx + 1, dtype=float), {"units": ""}),
                "lon": (("y", "x"), lon2d, {"units": "degrees_east"}),
                "lat": (("y", "x"), lat2d, {"units": "degrees_north"})})


def _curvilinear_panel(region=None, method="pcolormesh"):
    panel = mc_config.default_panel()
    panel["selection"] = {}
    panel["region"] = region
    panel["projection"] = {"name": "PlateCarree", "central_longitude": 140.0}
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"method": method, "levels": [0.0, 50.0, 100.0, 200.0]})
    panel["layers"] = [fill]
    return panel


def test_curvilinear_fill_array_without_region_is_full_grid():
    ds = _curvilinear_ds()
    _use(ds)
    fig = _render(_curvilinear_panel())
    qm = _quadmesh(fig.axes[0])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan), dtype=float).reshape(ds["f"].shape)
    np.testing.assert_allclose(got, ds["f"].values)
    # メッシュ (セルの角) は格子点数 + 1 (cartopy が座標を投影・wrap 処理するので
    # 角の値そのものは検査しない)
    assert qm.get_coordinates().shape == (ds.sizes["y"] + 1, ds.sizes["x"] + 1, 2)
    # 表示範囲 = 全格子点の投影座標の範囲 (領域未指定の 2 次元座標)
    xl = fig.axes[0].get_xlim()
    assert xl == pytest.approx((ds["lon"].values.min() - 140.0,
                                ds["lon"].values.max() - 140.0))


def test_curvilinear_fill_region_crop_matches_independent_bbox():
    """region の切り出し = 「範囲内の格子点の行・列の外接矩形 + 1 格子」を numpy で独立に
    計算した isel と一致する (render.curvilinear_region_indexers を使わない)。"""
    ds = _curvilinear_ds()
    _use(ds)
    region = {"lon_min": 130.0, "lon_max": 145.0, "lat_min": 25.0, "lat_max": 36.0}
    fig = _render(_curvilinear_panel(region))
    qm = _quadmesh(fig.axes[0])
    lon, lat = ds["lon"].values, ds["lat"].values
    inside = ((lon >= 130.0) & (lon <= 145.0) & (lat >= 25.0) & (lat <= 36.0))
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    i0, i1 = max(rows[0] - 1, 0), min(rows[-1] + 2, ds.sizes["y"])
    j0, j1 = max(cols[0] - 1, 0), min(cols[-1] + 2, ds.sizes["x"])
    expected = ds["f"].values[i0:i1, j0:j1]
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan), dtype=float)
    assert got.size == expected.size, (got.shape, expected.shape)
    np.testing.assert_allclose(got.reshape(expected.shape), expected)
    # 領域指定ありは extent が指定どおり (投影座標の bbox ではない)
    assert fig.axes[0].get_xlim() == pytest.approx((130.0 - 140.0, 145.0 - 140.0))


def test_curvilinear_contourf_levels_and_region_error():
    """contourf でも 2 次元座標が通り、格子の外の region は原因の分かる RenderError。"""
    ds = _curvilinear_ds()
    _use(ds)
    fig = _render(_curvilinear_panel(method="contourf"))
    cs = _contour_sets(fig.axes[0])
    assert cs and np.allclose(cs[0].levels, [0.0, 50.0, 100.0, 200.0])
    with pytest.raises(mc_render.RenderError) as ei:
        _render(_curvilinear_panel({"lon_min": 10.0, "lon_max": 20.0,
                                    "lat_min": -60.0, "lat_max": -50.0}))
    assert ei.value.msg_id == "region_outside_grid"


# --- 1-19: 経路断面 (section_path) — 配列 = 経路上の点へ独立に計算した双一次内挿、
#           格子線断面 = 行の切り出しそのもの (内挿なし) ---

def _curvilinear_ds_3d(ny: int = 12, nx: int = 16, nlev: int = 3) -> xr.Dataset:
    """_curvilinear_ds に鉛直 (lev) を足したもの。値 = 100 k + 10 i + j (k = 層番号、
    i = 行番号、j = 列番号) で、経緯度は格子番号の 1 次式なので (i, j) が解析的に逆算できる。"""
    base = _curvilinear_ds(ny, nx)
    j, i = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    k = np.arange(nlev, dtype=float)[:, None, None]
    vals = 100.0 * k + 10.0 * i + j
    return xr.Dataset(
        {"f": (("lev", "y", "x"), vals, {"units": "1"})},
        coords={**base.coords, "lev": ("lev", [1000.0, 850.0, 500.0][:nlev], {"units": "hPa"})})


def _analytic_indices(lon, lat):
    """_curvilinear_ds の経緯度 lon = 120 + 2j + 0.3i, lat = 20 + 2i − 0.2j を (i, j) について解く。"""
    a = np.array([[0.3, 2.0], [2.0, -0.2]])
    sol = np.linalg.solve(a, np.stack([np.asarray(lon) - 120.0, np.asarray(lat) - 20.0]))
    return sol[0], sol[1]


def _path_section_panel(x_dim, section_path=None, selection=None):
    panel = mc_config.default_section_panel()
    panel["x_dim"] = x_dim
    panel["y_dim"] = "lev"
    panel["section_path"] = section_path
    panel["selection"] = dict(selection or {})
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"].update({"method": "pcolormesh", "levels": [0.0, 100.0, 200.0, 300.0]})
    panel["layers"] = [fill]
    return panel


def test_section_path_array_matches_independent_bilinear():
    """等緯度線の経路断面: 描画配列 = 経路の各点の (i, j) を解析的に求めて
    値の式 100k + 10i + j に入れた値 (双一次内挿は 1 次式を厳密に再現する)。
    格子番号の逆算は接平面での近似なので許容 0.05 (格子番号の誤差 5e-3 相当)。"""
    ds = _curvilinear_ds_3d()
    _use(ds)
    path = {"kind": "parallel", "lat": 30.0, "lon_range": [126.0, 148.0], "npoints": 23}
    fig = _render(_path_section_panel(mc_render.SECTION_PATH_DIM, path))
    qm = _quadmesh(fig.axes[0])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan), dtype=float).reshape(3, 23)
    lon = np.linspace(126.0, 148.0, 23)
    i, j = _analytic_indices(lon, np.full(23, 30.0))
    expected = 100.0 * np.arange(3)[:, None] + 10.0 * i + j
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, atol=0.05)
    # 横軸 = 経度 (メッシュのセル境界は隣り合う経度の中点)
    edges = qm.get_coordinates()[0, :, 0]
    np.testing.assert_allclose(edges[1:-1], (lon[:-1] + lon[1:]) / 2)


def test_section_path_great_circle_axis_is_distance_and_outside_is_nan():
    """大円の経路断面: 横軸 = 始点からの距離 (km、独立に haversine で計算)、格子の外へ
    出た点は欠損。"""
    ds = _curvilinear_ds_3d()
    _use(ds)
    path = {"kind": "great_circle", "start": [125.0, 25.0], "end": [175.0, 45.0],
            "npoints": 30}
    fig = _render(_path_section_panel(mc_render.SECTION_PATH_DIM, path))
    qm = _quadmesh(fig.axes[0])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan), dtype=float).reshape(3, 30)
    lon, lat, _ = mc_render.great_circle_points((125.0, 25.0), (175.0, 45.0), 30)
    i, j = _analytic_indices(lon, lat)
    inside = (i >= 0) & (i <= 11) & (j >= 0) & (j <= 15)
    assert inside[:5].all() and not inside[-5:].all()
    assert np.isfinite(got[:, inside]).all()
    np.testing.assert_allclose(got[:, inside],
                               (100.0 * np.arange(3)[:, None] + 10.0 * i + j)[:, inside],
                               atol=0.05)
    assert np.isnan(got[:, ~inside]).all()
    lon1, lat1 = np.deg2rad(125.0), np.deg2rad(25.0)
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    h = np.sin((la - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(la) * np.sin((lo - lon1) / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(h))
    edges = qm.get_coordinates()[0, :, 0]
    np.testing.assert_allclose(edges[1:-1], (dist[:-1] + dist[1:]) / 2, rtol=1e-6)


def test_section_grid_row_is_exact_row_without_interpolation():
    """格子線断面 (x_dim = x、行 y を固定): 配列 = その行の値そのもの。"""
    ds = _curvilinear_ds_3d()
    _use(ds)
    fig = _render(_path_section_panel("x", None, {"y": 5.0}))
    qm = _quadmesh(fig.axes[0])
    got = np.asarray(np.ma.filled(qm.get_array(), np.nan), dtype=float).reshape(3, 16)
    np.testing.assert_array_equal(got, ds["f"].sel(y=5.0).values)


# --- 1-20: 地形マスク — 地面の線 = 独立計算 (単位換算・log p 内挿)、多角形は全レイヤーの上 ---

def _terrain_ds(ny: int = 12, nx: int = 16) -> xr.Dataset:
    """_curvilinear_ds_3d と同じ格子の地表変数: 地形高度 zs [m] と地上気圧 ps [Pa]。"""
    base = _curvilinear_ds(ny, nx)
    j, i = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    zs = 1500.0 * np.exp(-((i - 6.0) ** 2 / 8.0 + (j - 7.0) ** 2 / 12.0))
    ps = 1000.0e2 * np.exp(-zs / 8000.0)
    return xr.Dataset({"zs": (("y", "x"), zs, {"units": "m"}),
                       "ps": (("y", "x"), ps, {"units": "Pa"})}, coords=base.coords)


def _terrain_panel(method, x_dim="x", selection=None):
    panel = _path_section_panel(x_dim, None, selection or {"y": 6.0})
    panel["terrain"] = {**mc_config.default_section_panel()["terrain"], "show": True,
                        "method": method, "dataset_id": "ds1",
                        "variable": "ps" if method == "surface_pressure" else "zs",
                        "height_dataset_id": "ds0", "height_variable": "f"}
    return panel


def _terrain_polygon(ax):
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)
             and not isinstance(c, QuadMesh)]
    assert len(polys) == 1, f"地面の多角形が 1 つのはず: {len(polys)}"
    return polys[0]


def test_terrain_surface_pressure_ground_line_and_polygon():
    """地上気圧 [Pa] を hPa に換算した地面の線が、行 y = 6 の ps / 100 に一致し、多角形は
    地面の線から軸の下端 (気圧の大きい側) まで、全レイヤーより上 (zorder 2.5) に描かれる。"""
    ds = _curvilinear_ds_3d()
    _DATASETS["ds0"], _DATASETS["ds1"] = ds, _terrain_ds()
    panel = _terrain_panel("surface_pressure")
    x, ground, pressure_like = mc_render.section_terrain_ground(panel, _DATASETS)
    expected = _DATASETS["ds1"]["ps"].sel(y=6.0).values / 100.0
    np.testing.assert_allclose(ground, expected, rtol=1e-6)
    assert pressure_like
    fig = _render(panel)
    ax = fig.axes[0]
    poly = _terrain_polygon(ax)
    assert poly.get_zorder() == 2.5
    ylim = ax.get_ylim()
    verts = poly.get_paths()[0].vertices
    assert verts[:, 1].max() == pytest.approx(max(ylim))       # 下端 = 軸の端 (自動範囲は不変)
    assert np.isin(np.round(np.clip(expected, min(ylim), max(ylim)), 6),
                   np.round(verts[:, 1], 6)).all()             # 地面の線が頂点に入っている
    assert max(c.get_zorder() for c in ax.collections if c is not poly) < 2.5


def test_terrain_height_field_matches_log_p_interpolation():
    """高度の変数と地形高度: 各列で高度 = 地形高度 になる気圧が、log p の線形内挿を numpy で
    独立に計算した値と一致。最下層より下に地面がある列は NaN (描かない)。"""
    ds = _curvilinear_ds_3d()
    # f を「高度」に見立てる: 100k + 10i + j (m)。lev = 1000 / 850 / 500 hPa の 3 面
    ds["f"].attrs["units"] = "m"
    _DATASETS["ds0"], _DATASETS["ds1"] = ds, _terrain_ds()
    panel = _terrain_panel("height_field")
    x, ground, _ = mc_render.section_terrain_ground(panel, _DATASETS)
    z = ds["f"].sel(y=6.0).values                    # (lev, x)
    lev = ds["lev"].values
    zs = _DATASETS["ds1"]["zs"].sel(y=6.0).values
    expected = np.full(zs.shape, np.nan)
    for k in range(zs.size):
        if zs[k] <= z[0, k]:
            continue                                  # 地面が最下層より下
        if zs[k] > z[-1, k]:
            expected[k] = lev[-1]
            continue
        expected[k] = np.exp(np.interp(zs[k], z[:, k], np.log(lev)))
    np.testing.assert_allclose(ground, expected, rtol=1e-9, equal_nan=True)
    assert np.isnan(expected).any() and np.isfinite(expected).any()


def test_terrain_errors_are_clear():
    """単位の換算不能・方法と鉛直座標の不一致・変数なし は msg_id 付きの RenderError。"""
    ds = _curvilinear_ds_3d()
    _DATASETS["ds0"], _DATASETS["ds1"] = ds, _terrain_ds()
    cases = [
        ({"method": "surface_height", "variable": "zs"}, "terrain_method_mismatch"),
        ({"method": "surface_pressure", "variable": "zs"}, "terrain_units_unknown"),
        ({"method": "surface_pressure", "variable": "nope"}, "terrain_variable_missing"),
        ({"method": "height_field", "variable": "zs", "height_variable": "nope"},
         "terrain_variable_missing"),
    ]
    for override, msg_id in cases:
        panel = _terrain_panel("surface_pressure")
        panel["terrain"].update(override)
        with pytest.raises(mc_render.RenderError) as ei:
            _render(panel)
        assert ei.value.msg_id == msg_id, override
    # 鉛直座標の単位が無い
    ds2 = ds.copy()
    ds2["lev"].attrs.pop("units")
    _DATASETS["ds0"] = ds2
    with pytest.raises(mc_render.RenderError) as ei:
        _render(_terrain_panel("surface_pressure"))
    assert ei.value.msg_id == "terrain_vertical_units_unknown"


# --- 1-21: 地図への経路表示 — 線の頂点 = 断面の経路の点、参照の検査 ---

def _overlay_figure(section_panel, item=None):
    section_panel["panel_id"] = "sec"
    m = mc_config.default_panel()
    m["panel_id"] = "map"
    m["projection"] = {"name": "PlateCarree", "central_longitude": 140.0}
    m["region"] = {"lon_min": 110.0, "lon_max": 170.0, "lat_min": 10.0, "lat_max": 50.0}
    m["layers"] = [mc_config.default_fill_layer("ds0", "f")]
    m["layers"][0]["selection"] = {"lev": 1000.0}
    m["map"]["section_paths"] = [item or {"panel_id": "sec", "color": "red", "width": 2.0,
                                          "linestyle": "-", "end_labels": True,
                                          "labels": ["P", "Q"]}]
    cfg = mc_config.default_figure_config()
    cfg["figure"]["layout"] = {**cfg["figure"]["layout"], "nrows": 1, "ncols": 2}
    cfg["panels"] = [m, section_panel]
    return cfg


def _overlay_lines(ax, color="red"):
    return [ln for ln in ax.lines if ln.get_color() == color]


def test_section_overlay_vertices_match_path_points():
    """大円断面の経路を地図に重ねた線の頂点 (データ座標 = PlateCarree の経緯度) が
    great_circle_points の点と一致し、端点の文字が両端に置かれる。"""
    ds = _curvilinear_ds_3d()
    _use(ds)
    path = {"kind": "great_circle", "start": [125.0, 25.0], "end": [150.0, 40.0], "npoints": 30}
    cfg = _overlay_figure(_path_section_panel(mc_render.SECTION_PATH_DIM, path))
    fig = mc_render.render_figure(cfg, _DATASETS)
    ax = fig.axes[0]
    lines = _overlay_lines(ax)
    assert len(lines) == 1
    lon, lat, _ = mc_render.great_circle_points((125.0, 25.0), (150.0, 40.0), 30)
    xy = lines[0].get_xydata()
    np.testing.assert_allclose(xy[:, 0], lon)
    np.testing.assert_allclose(xy[:, 1], lat)
    assert lines[0].get_zorder() == 2 and lines[0].get_linewidth() == 2.0
    texts = {t.get_text(): t for t in ax.texts if t.get_text() in ("P", "Q")}
    assert set(texts) == {"P", "Q"}
    assert texts["P"].get_position() == pytest.approx((lon[0], lat[0]))
    assert texts["Q"].get_position() == pytest.approx((lon[-1], lat[-1]))
    assert texts["P"].get_ha() == "right" and texts["Q"].get_ha() == "left"
    # 端点の文字は任意の文字列。空欄の端には出さない
    cfg["panels"][0]["map"]["section_paths"][0]["labels"] = ["", "Kyushu"]
    fig = mc_render.render_figure(cfg, _DATASETS)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert "Kyushu" in labels and "" not in labels and "P" not in labels


def test_section_overlay_grid_row_and_reference_errors():
    """格子線断面の経路 = 固定した行の格子点の経緯度 (横軸の範囲内)。参照先が無い /
    鉛直断面でないときは msg_id 付きの RenderError。"""
    ds = _curvilinear_ds_3d()
    _use(ds)
    sec = _path_section_panel("x", None, {"y": 6.0})
    sec["ranges"] = {"x": [3.0, 12.0]}
    fig = mc_render.render_figure(_overlay_figure(sec), _DATASETS)
    xy = _overlay_lines(fig.axes[0])[0].get_xydata()
    expected_lon = ds["lon"].sel(y=6.0, x=slice(3.0, 12.0)).values
    expected_lat = ds["lat"].sel(y=6.0, x=slice(3.0, 12.0)).values
    np.testing.assert_allclose(xy[:, 0], expected_lon)
    np.testing.assert_allclose(xy[:, 1], expected_lat)

    cfg = _overlay_figure(_path_section_panel("x", None, {"y": 6.0}),
                          item={"panel_id": "nope", "color": "red"})
    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.render_figure(cfg, _DATASETS)
    assert ei.value.msg_id == "section_overlay_panel_missing"
    cfg = _overlay_figure(_path_section_panel("x", None, {"y": 6.0}),
                          item={"panel_id": "map", "color": "red"})
    with pytest.raises(mc_render.RenderError) as ei:
        mc_render.render_figure(cfg, _DATASETS)
    assert ei.value.msg_id == "section_overlay_not_section"
