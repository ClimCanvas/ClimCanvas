# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""UI の参考表示「データ範囲」「描画レベル」の裏付け。

表示は図を描かずに求める (render.preview_levels / matplotlib_auto_levels /
layer_selected_data / layer_drawn_data) ので、実際に render した図の
ContourSet.levels / QuadMesh.norm.boundaries と一致することをここで固定する。
matplotlib の自動レベル (levels=N) は私有ロジック (_autolev) を写さず
2×2 のダミー配列で同じ経路を通す方式 (cartopy_compat_notes.md「リスク低」節)。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.collections import QuadMesh
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure

from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def ds(sample_path) -> xr.Dataset:
    return mc_dataset.open_dataset(sample_path)


def _sel(ds, time_index=3, level=500.0) -> dict:
    """panel/layer selection (UI が入れるのと同じ ISO 文字列 + float)。"""
    return {"time": pd.Timestamp(ds["time"].values[time_index]).isoformat(),
            "level": level}


def _map_panel(layer, selection=None, region=None) -> dict:
    panel = mc_config.default_panel()
    panel["layers"] = [layer]
    if selection:
        panel["selection"] = selection
    if region:
        panel["region"] = region
    return panel


def _render(panel, ds):
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return mc_render.render_figure(cfg, {"ds0": ds})


def _contour_levels(fig, filled: bool) -> np.ndarray:
    cs = [c for c in fig.axes[0].collections
          if isinstance(c, ContourSet) and c.filled == filled]
    assert len(cs) == 1, f"ContourSet(filled={filled}) が1つのはず: {len(cs)}"
    return np.asarray(cs[0].levels, dtype=float)


def _quadmesh_boundaries(fig) -> np.ndarray:
    qm = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)]
    assert len(qm) == 1
    return np.asarray(qm[0].norm.boundaries, dtype=float)


# --- matplotlib の自動レベル (levels=N) を描かずに再現する ---

def test_matplotlib_auto_levels_matches_real_contour():
    """乱数の min/max・N・extend・塗り/線 で実際の contour(f) の levels と一致。"""
    rng = np.random.default_rng(0)
    for _ in range(150):
        lo = float(rng.uniform(-1000, 1000))
        hi = lo + float(10 ** rng.uniform(-3, 4))
        n = int(rng.integers(3, 40))
        filled = bool(rng.integers(0, 2))
        ext = ["neither", "both", "min", "max"][int(rng.integers(0, 4))]
        z = rng.uniform(lo, hi, (7, 9))
        z.flat[0], z.flat[-1] = lo, hi
        ax = Figure().add_subplot()
        draw = ax.contourf if filled else ax.contour
        ref = np.asarray(draw(z, levels=n, extend=ext).levels, dtype=float)
        got = mc_render.matplotlib_auto_levels(lo, hi, n, filled=filled, extend=ext)
        np.testing.assert_allclose(
            got, ref, err_msg=f"lo={lo} hi={hi} n={n} filled={filled} extend={ext}")


# --- 切り出し: panel.selection / layer.selection / region が反映される ---

def test_layer_selected_data_follows_selection_and_region(ds):
    layer = mc_config.default_contour_layer("ds0", "t")
    layer["selection"] = {"level": 500.0}
    sel = _sel(ds)
    panel = _map_panel(layer, selection={"time": sel["time"]})
    got = mc_render.layer_selected_data(panel, layer, {"ds0": ds})
    exp = ds["t"].isel(time=3).sel(level=500.0)
    np.testing.assert_allclose(got.values, exp.values)
    # 先頭断面 (以前の preview_slice 相当) とは別物であること (テストの前提)
    first = ds["t"].isel(time=0, level=0)
    assert float(first.max()) != float(exp.max())
    # region は render (select_panel_data) と同じ切り出し — 全周データの経度は
    # 切らず (cyclic 処理に任せる)、緯度は padded_slice で絞る
    region = {"lon_min": 100.0, "lon_max": 160.0, "lat_min": 10.0, "lat_max": 50.0}
    panel = _map_panel(layer, selection={"time": sel["time"]}, region=region)
    got = mc_render.layer_selected_data(panel, layer, {"ds0": ds})
    exp = mc_render.select_panel_data(
        ds, "t", {"time": sel["time"], "level": 500.0}, region,
        mc_dataset.detect_coord_roles(ds))
    assert got.sizes == exp.sizes
    np.testing.assert_allclose(got.values, exp.values)
    assert got.sizes["lat"] < ds.sizes["lat"]


def test_layer_selected_data_section_uses_ranges_and_averages(ds):
    """断面 (x_dim/y_dim あり) は select_section_data 経路 (averages → ranges)。"""
    layer = mc_config.default_contour_layer("ds0", "t")
    layer["selection"] = {}
    layer["averages"] = {"lat": {"op": "mean", "range": [-10.0, 10.0]}}
    sel = _sel(ds)
    panel = {"selection": {"time": sel["time"]}, "x_dim": "lon", "y_dim": "level",
             "ranges": {"level": [850.0, 300.0]}}
    got = mc_render.layer_selected_data(panel, layer, {"ds0": ds})
    exp = mc_render.select_section_data(ds, "t", {"time": sel["time"]},
                                        panel["ranges"], "lon", "level",
                                        averages=layer["averages"])
    assert got.dims == ("level", "lon")
    np.testing.assert_allclose(got.values, exp.values)


def test_layer_drawn_data_applies_transform_and_maskout(ds):
    layer = mc_config.default_fill_layer("ds0", "t")
    layer["selection"] = {"level": 500.0}
    layer["style"].update({"value_scale": 1.0, "value_offset": -273.15,
                           "maskout": {"below": -20.0, "above": None}})
    panel = _map_panel(layer, selection={"time": _sel(ds)["time"]})
    got = mc_render.layer_drawn_data(panel, layer, {"ds0": ds})
    vals = got.values[np.isfinite(got.values)]
    assert vals.size > 0 and vals.min() > -20.0
    exp = (ds["t"].isel(time=3).sel(level=500.0) - 273.15)
    assert float(np.nanmax(got.values)) == pytest.approx(float(exp.max()))


# --- preview_levels: 実際に描かれたレベルと一致 ---

def _prepared(panel, layer, ds):
    return mc_render.layer_drawn_data(panel, layer, {"ds0": ds})


@pytest.mark.parametrize("style,extend_key", [
    ({"levels": 11}, "single-auto"),
    ({"levels": 11, "use_cmap": True, "extend": "both"}, "cmap-both"),
    ({"levels": 11, "use_cmap": True, "extend": "max"}, "cmap-max"),
    ({"levels": 7, "vmin": 240.0, "vmax": 280.0}, "manual-linspace"),
    ({"levels": 9, "value_offset": -273.15,
      "maskout": {"below": -30.0, "above": None}}, "transform-maskout"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_preview_levels_contour_matches_rendered(ds, style, extend_key):
    layer = mc_config.default_contour_layer("ds0", "t")
    layer["selection"] = {"level": 500.0}
    layer["style"].update(style)
    panel = _map_panel(layer, selection={"time": _sel(ds)["time"]})
    fig = _render(panel, ds)
    ref = _contour_levels(fig, filled=False)
    got = mc_render.preview_levels(layer["style"], _prepared(panel, layer, ds),
                                   kind="contour")
    assert got is not None
    np.testing.assert_allclose(got, ref)


@pytest.mark.parametrize("style,label", [
    ({"levels": 21}, "contourf-auto-both"),
    ({"levels": 21, "extend": "neither"}, "contourf-auto-neither"),
    ({"levels": 21, "extend": "min"}, "contourf-auto-min"),
    ({"levels": 10, "vmin": 240.0, "vmax": 280.0}, "contourf-manual"),
    ({"levels": 15, "zero_white": True, "value_offset": -273.15}, "contourf-zero-white"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_preview_levels_fill_contourf_matches_rendered(ds, style, label):
    layer = mc_config.default_fill_layer("ds0", "t")
    layer["selection"] = {"level": 500.0}
    layer["style"].update(style)
    panel = _map_panel(layer, selection={"time": _sel(ds)["time"]})
    fig = _render(panel, ds)
    ref = _contour_levels(fig, filled=True)
    got = mc_render.preview_levels(layer["style"], _prepared(panel, layer, ds),
                                   kind="fill")
    assert got is not None
    np.testing.assert_allclose(got, ref)


@pytest.mark.parametrize("style,label", [
    ({"levels": 21, "method": "pcolormesh"}, "pcolormesh-auto"),
    ({"levels": 8, "method": "pcolormesh", "vmin": 240.0, "vmax": 280.0},
     "pcolormesh-manual"),
    ({"levels": 12, "method": "pcolormesh", "zero_white": True,
      "value_offset": -273.15}, "pcolormesh-zero-white"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_preview_levels_fill_pcolormesh_matches_rendered(ds, style, label):
    layer = mc_config.default_fill_layer("ds0", "t")
    layer["selection"] = {"level": 500.0}
    layer["style"].update(style)
    panel = _map_panel(layer, selection={"time": _sel(ds)["time"]})
    fig = _render(panel, ds)
    ref = _quadmesh_boundaries(fig)
    got = mc_render.preview_levels(layer["style"], _prepared(panel, layer, ds),
                                   kind="fill")
    assert got is not None
    np.testing.assert_allclose(got, ref)


def test_preview_levels_none_when_user_specified_or_degenerate(ds):
    """境界値の直接指定・旧スキーマ interval・全欠損・定数は None (表示しない)。"""
    da = ds["t"].isel(time=0).sel(level=500.0)
    assert mc_render.preview_levels({"levels": [1.0, 2.0, 3.0]}, da, kind="fill") is None
    assert mc_render.preview_levels(
        {"levels": 11, "vmin": 0.0, "vmax": 10.0, "interval": 2.0}, da,
        kind="contour") is None
    assert mc_render.preview_levels({"levels": 11}, da * np.nan, kind="contour") is None
    assert mc_render.preview_levels({"levels": 11}, xr.zeros_like(da), kind="fill") is None
    with pytest.raises(ValueError):
        mc_render.preview_levels({"levels": 11}, da, kind="hatch")
