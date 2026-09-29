# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""カスタムカラーマップの読み込み・登録・スクリプトへの埋め込み。

反転名 `<name>_r` の登録と、アプリ側描画 ⇄ 生成スクリプト (climcanvas 不在・
cmap ディレクトリ未設定の環境で実行) の画像一致もここで検証する。配布サンプル
(`data/sample/cmaps/`) が壊れていないことも見る。
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import matplotlib as mpl
import numpy as np
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

    # 登録コードがスクリプトに含まれている (反転名 _r も)
    assert "import matplotlib.colors as mcolors" in script
    assert "import matplotlib as mpl" in script
    assert "mcolors.LinearSegmentedColormap.from_list('test_div'" in script
    assert "mpl.colormaps.register(_custom_cmap, name='test_div', force=True)" in script
    assert ("mpl.colormaps.register(_custom_cmap.reversed(), name='test_div_r', "
            "force=True)") in script
    assert "cmap='test_div'" in script


def test_reversed_name_is_registered(custom_cmap_dir):
    """matplotlib は登録した cmap の `_r` を自動では作らないので、自前で登録する
    (「カラーマップを反転」が 'x_r' is not a valid value で落ちた実例 2026-09-29)。"""
    (custom_cmap_dir / "rev.rgb").write_text("0 0 255\n255 255 255\n255 0 0\n",
                                             encoding="utf-8")
    mc_render.load_custom_cmaps(force_reload=True)
    assert "rev" in mpl.colormaps() and "rev_r" in mpl.colormaps()
    fwd, rev = mpl.colormaps["rev"], mpl.colormaps["rev_r"]
    assert np.allclose(fwd(0.0), rev(1.0)) and np.allclose(fwd(1.0), rev(0.0))


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SAMPLE_CMAP_DIR = _ROOT / "data" / "sample" / "cmaps"


def test_shipped_sample_cmaps_load(monkeypatch):
    """配布サンプル (rgb / txt / dat の 3 形式) が読めて、段数と正規化が想定どおり。"""
    monkeypatch.setenv("CLIMCANVAS_CMAP_DIR", str(_SAMPLE_CMAP_DIR))
    names = mc_render.load_custom_cmaps(force_reload=True)
    assert names == ["sample_div_bwr", "sample_rainbow256", "sample_seq_teal"]
    assert len(mc_render.custom_cmap_rgb("sample_div_bwr")) == 9
    assert len(mc_render.custom_cmap_rgb("sample_seq_teal")) == 8
    assert len(mc_render.custom_cmap_rgb("sample_rainbow256")) == 256
    for n in names:
        assert all(0.0 <= v <= 1.0 for t in mc_render.custom_cmap_rgb(n) for v in t)
    # 後続テストに影響しないようキャッシュを戻す
    mc_render._custom_cmaps_loaded_for_dir = None
    mc_render._CUSTOM_CMAP_DATA.clear()


_RENDER_RUNNER = """\
import json, sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, sys.argv[4])
from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
# load_custom_cmaps は render_figure の入口で呼ばれる (コア層だけの経路でも解決できる)
fig = mc_render.render_figure(cfg, {"ds0": mc_dataset.open_dataset(sys.argv[2])})
fig.savefig(sys.argv[3], dpi=150, bbox_inches="tight")
"""


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_render_and_script_match_with_custom_cmap(sample_path, tmp_path, monkeypatch,
                                                  reverse):
    """配布サンプルの cmap (反転あり / なし) で、アプリ側描画と生成スクリプトの画像が
    一致し、スクリプトは climcanvas 不在・CLIMCANVAS_CMAP_DIR 未設定の環境で動く。"""
    import json
    monkeypatch.setenv("CLIMCANVAS_CMAP_DIR", str(_SAMPLE_CMAP_DIR))
    mc_render.load_custom_cmaps(force_reload=True)
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}
    panel = mc_config.default_panel()
    panel["selection"] = {"time": "2024-01-01T00:00:00", "level": 500.0}
    fill = mc_config.default_fill_layer("ds0", "t")
    fill["style"]["cmap"] = "sample_div_bwr"
    fill["style"]["reverse_cmap"] = reverse
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]

    # アプリ側: コールドプロセス (CLIMCANVAS_CMAP_DIR は継承)
    app_png = tmp_path / "app.png"
    (tmp_path / "runner.py").write_text(_RENDER_RUNNER, encoding="utf-8")
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, str(tmp_path / "runner.py"),
                        str(tmp_path / "cfg.json"), sample_path, str(app_png), str(_ROOT)],
                       capture_output=True, text=True, cwd=tmp_path,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode == 0, f"render_figure の実行に失敗:\n{r.stderr}"

    # スクリプト側: climcanvas を毒入れ、cmap ディレクトリの環境変数も外す
    poison = tmp_path / "poison" / "climcanvas"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text('raise ImportError("climcanvas に依存")',
                                        encoding="utf-8")
    script_png = tmp_path / "script.png"
    script = mc_scriptgen.generate_script(
        cfg, datasets, {"ds0": sample_path},
        figure_output=str(script_png), figure_dpi=150, include_show=False)
    (tmp_path / "reproduce.py").write_text(script, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "CLIMCANVAS_CMAP_DIR"}
    env.update(MPLBACKEND="Agg", PYTHONPATH=str(tmp_path / "poison"))
    r = subprocess.run([sys.executable, str(tmp_path / "reproduce.py")],
                       capture_output=True, text=True, cwd=tmp_path, env=env)
    assert r.returncode == 0, f"生成スクリプトの実行に失敗:\n{r.stderr}\n{script}"
    assert f"cmap='sample_div_bwr{'_r' if reverse else ''}'" in script

    from PIL import Image
    a = np.asarray(Image.open(app_png))
    b = np.asarray(Image.open(script_png))
    assert a.shape == b.shape and np.array_equal(a, b), "アプリ描画と生成スクリプトの画像が不一致"
    mc_render._custom_cmaps_loaded_for_dir = None
    mc_render._CUSTOM_CMAP_DATA.clear()
