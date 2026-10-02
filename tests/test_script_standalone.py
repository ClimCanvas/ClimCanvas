# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""生成スクリプトのスタンドアロン性 (譲れないルール2) の直接検証。

consistency テストは生成 .py をサブプロセスで実行するが、その環境では
climcanvas が import 可能なため、誤って本アプリへの依存が紛れ込んでも
検出できなかった (ルール2の盲点)。ここでは2段で検証する
(docs/semantic_test_plan.md 観点⑤):

1. **AST 検査**: import のトップレベルパッケージが許可リスト
   (xarray / numpy / matplotlib / cartopy) に収まる。ユーザーが外部 cmap
   (cmocean 等) を選んだときだけ該当パッケージが許可に加わる (ルール2の例外)
2. **毒入れ実行**: climcanvas を import すると即失敗する偽パッケージを
   PYTHONPATH の先頭に置いた子プロセスで実行し、それでも成功することを確認
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

from climcanvas.core import dataset as mc_dataset
from climcanvas.core import external_cmaps
from climcanvas.core import scriptgen as mc_scriptgen
from test_consistency import (_agg_hist2d_discrete_config,
                              _line_1d_two_lines_config, _make_series_nc,
                              _rich_config, _simple_config)

# 生成スクリプトが依存してよいトップレベルパッケージ (ルール2)
_ALLOWED_ROOTS = {"xarray", "numpy", "matplotlib", "cartopy"}


def _import_roots(source: str) -> set[str]:
    """ソース中の import 文のトップレベルパッケージ名を集める。"""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


def _cases(sample_path, tmp_path):
    """(名前, config, データパス) の代表ケース。地図/1D/集計を横断する。"""
    series = tmp_path / "series.nc"
    if not series.exists():
        _make_series_nc(str(series))
    return [
        ("simple-map", _simple_config(), sample_path),
        ("rich-layers", _rich_config(), sample_path),
        ("line-1d", _line_1d_two_lines_config(), sample_path),
        ("agg-hist2d-discrete", _agg_hist2d_discrete_config(), str(series)),
    ]


def _generate(cfg, ds_path, out_png) -> str:
    datasets = {"ds0": mc_dataset.open_dataset(ds_path)}
    return mc_scriptgen.generate_script(
        cfg, datasets, {"ds0": str(ds_path)},
        figure_output=str(out_png), figure_dpi=100, include_show=False)


def test_generated_script_imports_are_whitelisted(sample_path, tmp_path):
    """生成スクリプトの import が許可パッケージだけで構成される (ルール2)。"""
    for name, cfg, ds_path in _cases(sample_path, tmp_path):
        script = _generate(cfg, ds_path, tmp_path / f"{name}.png")
        allowed = _ALLOWED_ROOTS | set(external_cmaps.packages_used_in(cfg))
        extra = _import_roots(script) - allowed
        assert not extra, \
            f"{name}: 許可外の import があります: {sorted(extra)}\n{script}"


def test_generated_script_runs_without_climcanvas(sample_path, tmp_path):
    """climcanvas が import 不能な環境でも生成スクリプトが実行できる (ルール2)。"""
    # 毒入れ: import climcanvas した瞬間に失敗する偽パッケージ
    poison_root = tmp_path / "poison"
    pkg = poison_root / "climcanvas"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        'raise ImportError("生成スクリプトが climcanvas に依存しています '
        '(譲れないルール2違反)")', encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg",
               PYTHONPATH=str(poison_root))

    # 毒が効いていることのサニティチェック
    probe = subprocess.run([sys.executable, "-c", "import climcanvas"],
                           capture_output=True, text=True, env=env,
                           cwd=tmp_path)
    assert probe.returncode != 0, "毒入れが機能していない (テスト自体の不備)"

    for name, cfg, ds_path in _cases(sample_path, tmp_path):
        out_png = tmp_path / f"{name}_poisoned.png"
        script_path = tmp_path / f"{name}.py"
        script_path.write_text(_generate(cfg, ds_path, out_png),
                               encoding="utf-8")
        result = subprocess.run([sys.executable, str(script_path)],
                                capture_output=True, text=True, env=env,
                                cwd=tmp_path)
        assert result.returncode == 0, \
            f"{name}: climcanvas 不在環境で実行失敗:\n{result.stderr}"
        assert out_png.exists(), f"{name}: 出力画像が生成されていない"


def test_generated_script_with_coord_files_is_standalone(curvilinear_bare_paths, tmp_path):
    """座標ファイル (経緯度が別ファイル) を結び付けた config の生成スクリプトが、
    許可 import のみで、climcanvas 不在環境でも実行できる (ルール2)。"""
    from test_consistency import _curvilinear_fill_contour_config
    bare_path, lonlat_path = curvilinear_bare_paths
    ds = mc_dataset.attach_coord_files(
        mc_dataset.open_dataset(bare_path),
        [(lonlat_path, mc_dataset.open_coord_file(lonlat_path))])
    cfg = _curvilinear_fill_contour_config()
    out_png = tmp_path / "coord_files.png"
    script = mc_scriptgen.generate_script(
        cfg, {"ds0": ds}, {"ds0": bare_path},
        figure_output=str(out_png), figure_dpi=100, include_show=False)
    extra = _import_roots(script) - _ALLOWED_ROOTS
    assert not extra, f"許可外の import があります: {sorted(extra)}\n{script}"

    poison_root = tmp_path / "poison"
    pkg = poison_root / "climcanvas"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        'raise ImportError("生成スクリプトが climcanvas に依存しています")',
        encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(poison_root))
    script_path = tmp_path / "coord_files.py"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script_path)],
                            capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, f"生成スクリプトの実行に失敗:\n{result.stderr}\n{script}"
    assert out_png.exists()


def test_generated_script_with_section_path_is_standalone(curvilinear_bare_paths, tmp_path):
    """経路断面 (大円 + 経緯度の目盛併記) の生成スクリプトが、埋め込んだ関数群
    (great_circle_points / grid_fractional_indices / sample_bilinear / lonlat_tick_label)
    込みで許可 import のみ・climcanvas 不在環境で実行できる (ルール2)。"""
    from test_consistency import _curvilinear_section_great_circle_config
    bare_path, lonlat_path = curvilinear_bare_paths
    ds = mc_dataset.attach_coord_files(
        mc_dataset.open_dataset(bare_path),
        [(lonlat_path, mc_dataset.open_coord_file(lonlat_path))])
    cfg = _curvilinear_section_great_circle_config()
    out_png = tmp_path / "section_path.png"
    script = mc_scriptgen.generate_script(
        cfg, {"ds0": ds}, {"ds0": bare_path},
        figure_output=str(out_png), figure_dpi=100, include_show=False)
    extra = _import_roots(script) - _ALLOWED_ROOTS
    assert not extra, f"許可外の import があります: {sorted(extra)}\n{script}"
    for fn in ("def great_circle_points(", "def grid_fractional_indices(",
               "def sample_bilinear(", "def lonlat_tick_label("):
        assert fn in script, fn
    assert "def grid_fractional_indices_1d(" not in script   # 2 次元格子では不要

    poison_root = tmp_path / "poison"
    pkg = poison_root / "climcanvas"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        'raise ImportError("生成スクリプトが climcanvas に依存しています")',
        encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(poison_root))
    script_path = tmp_path / "section_path.py"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script_path)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"実行失敗:\n{result.stderr}\n--- script ---\n{script}"
    assert out_png.exists()


def test_generated_script_with_terrain_is_standalone(curvilinear_sample_path,
                                                     curvilinear_terrain_path, tmp_path):
    """地形マスク (高度の変数と地形高度、別ファイル) の生成スクリプトが、埋め込んだ
    ground_pressure_from_height / draw_section_terrain 込みで許可 import のみ・
    climcanvas 不在環境で実行できる (ルール2)。"""
    from test_consistency import _HF, _terrain_section_cfg
    datasets = {"ds0": mc_dataset.open_dataset(curvilinear_sample_path),
                "ds1": mc_dataset.open_dataset(curvilinear_terrain_path)}
    cfg = _terrain_section_cfg({"kind": "meridian", "lon": 138.0, "lat_range": [25.0, 45.0],
                                "npoints": 50}, "path", "lev", _HF, labels=True)
    out_png = tmp_path / "terrain.png"
    script = mc_scriptgen.generate_script(
        cfg, datasets, {"ds0": curvilinear_sample_path, "ds1": curvilinear_terrain_path},
        figure_output=str(out_png), figure_dpi=100, include_show=False)
    extra = _import_roots(script) - _ALLOWED_ROOTS
    assert not extra, f"許可外の import があります: {sorted(extra)}\n{script}"
    for fn in ("def ground_pressure_from_height(", "def draw_section_terrain(",
               "def sample_bilinear("):
        assert fn in script, fn
    assert "_t.set_zorder(2.4)" in script      # 等値線ラベルは地面の下

    poison_root = tmp_path / "poison"
    pkg = poison_root / "climcanvas"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        'raise ImportError("生成スクリプトが climcanvas に依存しています")',
        encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(poison_root))
    script_path = tmp_path / "terrain.py"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script_path)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"実行失敗:\n{result.stderr}\n--- script ---\n{script}"
    assert out_png.exists()
