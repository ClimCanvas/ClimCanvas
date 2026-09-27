# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""トラックレイヤーのストーム選択 (発生年・月 / index 範囲) のユニットテスト。

render.track_storm_indices / track_genesis_year_month を in-memory データで検証。
render↔scriptgen の画像一致は tests/test_consistency.py の track テストが担当。
"""

import numpy as np
import xarray as xr

from climcanvas.core.config import default_track_layer
from climcanvas.core.render import (track_arrays, track_genesis_year_month,
                                    track_storm_indices)


def _make_ds():
    """5ストーム: 発生 2023-06 / 2023-12 / 2024-02 / 2024-08 / 全NaT。"""
    n_t = 4
    genesis = ["2023-06-01", "2023-12-30", "2024-02-05", "2024-08-10", None]
    time = np.full((5, n_t), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for s, g in enumerate(genesis):
        if g is None:
            continue
        t0 = np.datetime64(g + "T00", "ns")
        time[s, :] = t0 + np.arange(n_t) * np.timedelta64(6, "h")
    lat = np.ones((5, n_t), np.float32) * 20.0
    lon = np.ones((5, n_t), np.float32) * 140.0
    # 各ストームとも風速 [10, 20, 30, NaN] (maskout テスト用)
    wind = np.tile(np.array([10.0, 20.0, 30.0, np.nan], np.float32), (5, 1))
    return xr.Dataset({
        "lat": (("storm", "date_time"), lat, {"units": "degrees_north"}),
        "lon": (("storm", "date_time"), lon, {"units": "degrees_east"}),
        "wind": (("storm", "date_time"), wind, {"units": "kts"}),
        "time": (("storm", "date_time"), time),
    })


def _layer(**over):
    layer = default_track_layer("ds0")
    layer.update({"lon_var": "lon", "lat_var": "lat", "storm_dim": "storm"})
    layer.update(over)
    return layer


def test_genesis_year_month():
    ds = _make_ds()
    ok, yr, mo = track_genesis_year_month(ds, _layer(time_var="time"))
    assert list(ok) == [True, True, True, True, False]  # 全NaT は無効
    assert list(yr[:4]) == [2023, 2023, 2024, 2024]
    assert list(mo[:4]) == [6, 12, 2, 8]


def test_year_filter():
    ds = _make_ds()
    layer = _layer(time_var="time", year_range=[2024, 2024])
    assert track_storm_indices(layer, {"ds0": ds}) == [2, 3]


def test_year_and_month_filter():
    ds = _make_ds()
    layer = _layer(time_var="time", year_range=[2023, 2024],
                   month_range=[6, 8])
    assert track_storm_indices(layer, {"ds0": ds}) == [0, 3]


def test_month_wraparound():
    """開始月 > 終了月 は年またぎ (11→3 = 11,12,1,2,3月)。"""
    ds = _make_ds()
    layer = _layer(time_var="time", month_range=[11, 3])
    assert track_storm_indices(layer, {"ds0": ds}) == [1, 2]


def test_time_filter_overrides_storm_range():
    """年・月フィルタは storm_range より優先される。"""
    ds = _make_ds()
    layer = _layer(time_var="time", year_range=[2024, 2024],
                   storm_range=[0, 0])
    assert track_storm_indices(layer, {"ds0": ds}) == [2, 3]


def test_index_range_fallback():
    """time_var 未指定なら従来どおり index 範囲 (クリップ付き)。"""
    ds = _make_ds()
    assert track_storm_indices(_layer(storm_range=[1, 3]),
                               {"ds0": ds}) == [1, 2, 3]
    assert track_storm_indices(_layer(storm_range=[3, 99]),
                               {"ds0": ds}) == [3, 4]
    assert track_storm_indices(_layer(), {"ds0": ds}) == [0, 1, 2, 3, 4]


def test_empty_selection():
    ds = _make_ds()
    layer = _layer(time_var="time", year_range=[1990, 1991])
    assert track_storm_indices(layer, {"ds0": ds}) == []


def test_maskout_below_masks_positions():
    """maskout below: 変数値がしきい値以下 (と欠損) の位置の lon/lat が NaN になる。"""
    ds = _make_ds()
    layer = _layer()
    layer["style"]["maskout"] = {"variable": "wind", "below": 15.0,
                                 "above": None}
    lon, lat, _ = track_arrays(layer, {"ds0": ds}, 0)
    # wind = [10, 20, 30, NaN] → 10 (以下) と NaN (欠損) がマスクされる
    assert list(np.isnan(lon)) == [True, False, False, True]
    assert list(np.isnan(lat)) == [True, False, False, True]


def test_maskout_band_and_no_mutation():
    """below + above の帯 maskout。元の dataset の値は書き換えない。"""
    ds = _make_ds()
    layer = _layer()
    layer["style"]["maskout"] = {"variable": "wind", "below": 15.0,
                                 "above": 25.0}
    lon, _, _ = track_arrays(layer, {"ds0": ds}, 1)
    # 残るのは 15 < wind < 25 の位置 (= 20) だけ
    assert list(np.isnan(lon)) == [True, False, True, True]
    # キャッシュ済み dataset を汚していない (np.where で新配列を作っている)
    assert np.isfinite(ds["lon"].values).all()
