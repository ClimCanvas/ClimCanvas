# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""dim 固定 selectbox の初期値 (_default_dim_index) のユニットテスト。

鉛直次元は最下層 (気圧なら最大値・高度なら最小値)、緯度は 0°、経度は 180° に
最も近い格子点が初期値になること。UI 経由の配線は test_ui_config_mapping /
test_app_smoke が担当。
"""

import xarray as xr

from climcanvas.ui.layer_ui import _default_dim_index

ROLES = {"lat": "lat", "lon": "lon", "vertical": "level", "time": "time"}


def _ds_with_level(values, attrs):
    return xr.Dataset(coords={"level": ("level", list(values), attrs)})


def test_pressure_ascending_defaults_to_1000():
    choices = [200.0, 300.0, 500.0, 700.0, 850.0, 1000.0]
    ds = _ds_with_level(choices, {"units": "hPa"})
    assert choices[_default_dim_index(ds, "level", ROLES, choices)] == 1000.0


def test_pressure_descending_defaults_to_1000():
    choices = [1000.0, 850.0, 700.0, 500.0, 300.0, 200.0]
    ds = _ds_with_level(choices, {"units": "hPa"})
    assert _default_dim_index(ds, "level", ROLES, choices) == 0


def test_positive_down_without_units_defaults_to_max():
    choices = [1.0, 2.0, 3.0]
    ds = _ds_with_level(choices, {"positive": "down"})
    assert choices[_default_dim_index(ds, "level", ROLES, choices)] == 3.0


def test_height_defaults_to_min():
    choices = [10.0, 100.0, 1000.0]
    ds = _ds_with_level(choices, {"units": "m", "positive": "up"})
    assert choices[_default_dim_index(ds, "level", ROLES, choices)] == 10.0


def test_lat_defaults_to_equator_and_lon_to_180():
    ds = xr.Dataset(coords={"lat": ("lat", [-60.0, -30.0, 15.0, 45.0]),
                            "lon": ("lon", [0.0, 90.0, 175.0, 270.0])})
    lat_choices = [-60.0, -30.0, 15.0, 45.0]
    lon_choices = [0.0, 90.0, 175.0, 270.0]
    assert lat_choices[_default_dim_index(ds, "lat", ROLES, lat_choices)] == 15.0
    assert lon_choices[_default_dim_index(ds, "lon", ROLES, lon_choices)] == 175.0


def test_other_dim_defaults_to_first():
    choices = [5.0, 1.0, 3.0]
    ds = xr.Dataset(coords={"member": ("member", choices)})
    assert _default_dim_index(ds, "member", ROLES, choices) == 0


def test_default_dim_index_string_coord_returns_first():
    choices = ["NIO", "WP", "SP"]
    ds = xr.Dataset(coords={"basin": ("basin", choices)})
    roles = {**ROLES, "vertical": "basin"}  # 仮に鉛直と誤認されても落ちない
    assert _default_dim_index(ds, "basin", roles, choices) == 0


def test_dim_choices_numeric_and_string():
    from climcanvas.ui.widgets import dim_choice_label, dim_choices
    import numpy as np

    nums = dim_choices(np.array([1000.0, 850.0]))
    assert nums == [1000.0, 850.0] and all(isinstance(v, float) for v in nums)
    strs = dim_choices(np.array(["NIO", "WP"]))
    assert strs == ["NIO", "WP"] and all(isinstance(v, str) for v in strs)
    assert dim_choice_label(1000.0) == "1000"
    assert dim_choice_label("NIO") == "NIO"
