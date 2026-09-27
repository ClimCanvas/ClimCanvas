# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""Orthographic 図法の地球回転アニメーション: 経路補間 (render.rotation_path) と
iter_frames の引数検証のユニットテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from climcanvas.core import render as mc_render


def test_rotation_path_endpoints_and_count():
    path = mc_render.rotation_path([(140.0, 20.0), (200.0, 50.0)], 7)
    assert len(path) == 7
    assert path[0] == (140.0, 20.0)
    assert path[-1] == (200.0, 50.0)
    # 1区間は等間隔 (経度・緯度とも線形)
    lons = [p[0] for p in path]
    lats = [p[1] for p in path]
    np.testing.assert_allclose(np.diff(lons), 10.0, atol=1e-3)
    np.testing.assert_allclose(np.diff(lats), 5.0, atol=1e-3)


def test_rotation_path_passes_through_waypoints_at_constant_speed():
    """区間には角距離に比例してフレームを配分する → 経由点がフレーム上に乗り、
    各フレーム間の移動量が一定。"""
    # 区間1: 経度 +60 (距離 60)、区間2: 緯度 +30 (距離 30) → 距離比 2:1
    path = mc_render.rotation_path([(0.0, 0.0), (60.0, 0.0), (60.0, 30.0)], 10)
    assert path[6] == (60.0, 0.0)  # 総距離 90 を 9 分割 → 距離 60 は 6 コマ目
    steps = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])]
    np.testing.assert_allclose(steps, 10.0, atol=1e-3)


def test_rotation_path_takes_shortest_longitude_direction():
    # 170 → -170 は日付変更線をまたぐ東回り (+20°)。連続値のまま返す (正規化しない)
    path = mc_render.rotation_path([(170.0, 0.0), (-170.0, 0.0)], 3)
    assert [p[0] for p in path] == [170.0, 180.0, 190.0]
    # ちょうど 180° 差は差の符号の向き: +180 は東回り、-180 は西回り
    east = mc_render.rotation_path([(140.0, 0.0), (320.0, 0.0)], 3)
    west = mc_render.rotation_path([(320.0, 0.0), (140.0, 0.0)], 3)
    assert [p[0] for p in east] == [140.0, 230.0, 320.0]
    assert [p[0] for p in west] == [320.0, 230.0, 140.0]


def test_rotation_path_degenerate_cases():
    assert mc_render.rotation_path([(140.0, 20.0)], 3) == [(140.0, 20.0)] * 3
    assert mc_render.rotation_path([(140.0, 20.0), (140.0, 20.0)], 2) == [(140.0, 20.0)] * 2
    assert mc_render.rotation_path([(1.0, 2.0), (3.0, 4.0)], 1) == [(1.0, 2.0)]
    # 同一点の連続 (長さ 0 の区間) を含んでも補間できる
    path = mc_render.rotation_path([(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)], 3)
    assert path == [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    with pytest.raises(ValueError):
        mc_render.rotation_path([], 3)
    with pytest.raises(ValueError):
        mc_render.rotation_path([(0.0, 0.0)], 0)


def test_rotation_path_values_are_rounded_plain_floats():
    """render と scriptgen が同じ値を使う前提: 小数 4 桁の float (repr が短い)。"""
    path = mc_render.rotation_path([(140.0, 20.0), (200.0, 50.0), (260.0, 0.0)], 3)
    for lon, lat in path:
        assert isinstance(lon, float) and isinstance(lat, float)
        assert round(lon, 4) == lon and round(lat, 4) == lat


def test_iter_frames_argument_validation():
    with pytest.raises(ValueError):
        list(mc_render.iter_frames({"panels": []}, {}))
    with pytest.raises(ValueError):
        list(mc_render.iter_frames({"panels": []}, {}, time_values=[1, 2],
                                   centers=[(0.0, 0.0)]))


def test_expand_time_values_and_frame_count():
    assert mc_render.expand_time_values(["a", "b"], 3) == ["a", "a", "a", "b", "b", "b"]
    assert mc_render.expand_time_values(["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError):
        mc_render.expand_time_values(["a"], 0)
    assert mc_render.animation_frame_count(["a", "b"], None, 3) == 6
    assert mc_render.animation_frame_count(["a", "b"], [(0, 0)] * 6, 3) == 6
    assert mc_render.animation_frame_count(None, [(0, 0)] * 4) == 4
    # 時刻 (展開後) と中心の長さ不一致
    with pytest.raises(ValueError):
        list(mc_render.iter_frames({"panels": []}, {}, time_values=[1, 2],
                                   centers=[(0.0, 0.0)] * 4, frames_per_time=3))


def test_animation_gif_tight_only_without_rotation():
    assert mc_render.animation_gif_tight(None) is True
    assert mc_render.animation_gif_tight([(0.0, 0.0)]) is False
