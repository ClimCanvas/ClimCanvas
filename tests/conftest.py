# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

# --- 実ユーザーの ~/.climcanvas からテストを隔離する ---
# ユーザーが config.toml (allowed_dirs / session_dirs) や preset.json を作っていても
# テスト結果が変わらないように、環境変数で明示的に上書きする。
# (CC_ALLOWED_DIRS は「設定されていれば空でも優先」— 空 = 制限なし)
import os as _os
import tempfile as _tempfile

_os.environ["CC_ALLOWED_DIRS"] = ""
_os.environ["CC_MODE"] = ""  # 常に local 扱い (config.toml の mode を無視)
_isolated = _tempfile.mkdtemp(prefix="cc_test_climcanvas_")
_os.environ["CC_SESSION_DIRS"] = str(Path(_isolated) / "sessions")
_os.environ["CLIMCANVAS_PRESET_PATH"] = str(Path(_isolated) / "preset.json")

import pytest

from scripts.make_sample_data import create_sample_dataset, create_terrain_dataset
from scripts.make_sample_track_data import create_track_dataset
from scripts.make_sample_curvilinear_data import (create_curvilinear_dataset,
                                                  create_curvilinear_terrain)


@pytest.fixture(scope="session")
def sample_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("data") / "sample_atmos.nc"
    create_sample_dataset(str(path))
    return str(path)


@pytest.fixture(scope="session")
def _curvilinear_files(tmp_path_factory) -> dict[str, str]:
    """2 次元座標 (ランベルト格子) の合成サンプル 3 ファイル (埋め込み / 経緯度なし / 座標ファイル)。"""
    d = tmp_path_factory.mktemp("data")
    paths = {"embedded": str(d / "sample_curvilinear.nc"),
             "bare": str(d / "sample_curvilinear_bare.nc"),
             "lonlat": str(d / "sample_curvilinear_lonlat.nc"),
             "terrain": str(d / "sample_curvilinear_terrain.nc")}
    create_curvilinear_dataset(paths["embedded"], paths["bare"], paths["lonlat"])
    create_curvilinear_terrain(paths["terrain"])
    return paths


@pytest.fixture(scope="session")
def terrain_sample_path(tmp_path_factory) -> str:
    """sample_atmos と同じ格子の地表変数 (zs [m] / ps [hPa]) — 地形マスク用。"""
    path = tmp_path_factory.mktemp("data") / "sample_terrain.nc"
    create_terrain_dataset(str(path))
    return str(path)


@pytest.fixture(scope="session")
def curvilinear_terrain_path(_curvilinear_files) -> str:
    """curvilinear サンプルと同じ格子の地表変数 (zs [m] / ps [Pa]) — 地形マスク用。"""
    return _curvilinear_files["terrain"]


@pytest.fixture(scope="session")
def curvilinear_sample_path(_curvilinear_files) -> str:
    """lon(y,x) / lat(y,x) を 2 次元座標として埋め込んだ curvilinear サンプル。"""
    return _curvilinear_files["embedded"]


@pytest.fixture(scope="session")
def curvilinear_bare_paths(_curvilinear_files) -> tuple[str, str]:
    """(経緯度なしのデータファイル, FLON/FLAT だけの座標ファイル) — ClimCORE 形式。"""
    return _curvilinear_files["bare"], _curvilinear_files["lonlat"]


@pytest.fixture(scope="session")
def track_sample_path(tmp_path_factory) -> str:
    """IBTrACS 風の合成ベストトラック (storm × date_time、NaN パディング)。"""
    path = tmp_path_factory.mktemp("data") / "sample_besttrack.nc"
    create_track_dataset(str(path))
    return str(path)
