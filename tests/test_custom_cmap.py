# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""カスタムカラーマップの読み込み・登録・スクリプトへの埋め込み。"""

from __future__ import annotations


import matplotlib as mpl
import pytest

from climcanvas.core import render as mc_render
from climcanvas.core import config as mc_config
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import scriptgen as mc_scriptgen


@pytest.fixture
def custom_cmap_dir(tmp_path, monkeypatch):
    """環境変数 CLIMCANVAS_CMAP_DIR で tempdir を指す。"""
    monkeypatch.setenv("CLIMCANVAS_CMAP_DIR", str(tmp_path))
    # キャッシュをクリアして再読み込みを強制
    mc_render._custom_cmaps_loaded_for_dir = None
    mc_render._CUSTOM_CMAP_DATA.clear()
    yield tmp_path


def test_parse_rgb_file_255_range(custom_cmap_dir):
    f = custom_cmap_dir / "my_div.rgb"
    f.write_text("# diverging\n0 0 255\n255 255 255\n255 0 0\n", encoding="utf-8")
    names = mc_render.load_custom_cmaps(force_reload=True)
    assert "my_div" in names
    rgb = mc_render.custom_cmap_rgb("my_div")
    # 0-255 範囲が 0-1 に正規化されていることを確認
    assert rgb == [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
    # matplotlib にも登録されていること
    assert "my_div" in mpl.colormaps()


def test_parse_rgb_file_unit_range(custom_cmap_dir):
    f = custom_cmap_dir / "unit.rgb"
    f.write_text("0.0,0.0,1.0\n1.0,1.0,1.0\n1.0,0.0,0.0\n", encoding="utf-8")
    mc_render.load_custom_cmaps(force_reload=True)
    rgb = mc_render.custom_cmap_rgb("unit")
    assert rgb == [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]


def test_invalid_file_skipped(custom_cmap_dir):
    (custom_cmap_dir / "bad.rgb").write_text("not numbers\nfoo bar baz\n", encoding="utf-8")
    (custom_cmap_dir / "single.rgb").write_text("0 0 0\n", encoding="utf-8")  # too short
    names = mc_render.load_custom_cmaps(force_reload=True)
    assert "bad" not in names
    assert "single" not in names


def test_script_embeds_custom_cmap_registration(custom_cmap_dir, sample_path):
    # カスタム cmap を登録
    (custom_cmap_dir / "test_div.rgb").write_text(
        "0 0 255\n128 128 255\n255 255 255\n255 128 128\n255 0 0\n", encoding="utf-8")
    mc_render.load_custom_cmaps(force_reload=True)

    # その cmap を使う figure_config を組む
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"]["cmap"] = "test_div"
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]

    script = mc_scriptgen.generate_script(cfg, datasets, {"ds0": str(sample_path)})

    # 登録コードがスクリプトに含まれている
    assert "import matplotlib.colors as mcolors" in script
    assert "import matplotlib as mpl" in script
    assert "mpl.colormaps.register(mcolors.LinearSegmentedColormap.from_list('test_div'" in script
    assert "cmap='test_div'" in script
