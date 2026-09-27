# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""旧バージョン形式の figure_config が今のコアで動き続けることの検証 (互換コーパス)。

`fixtures/legacy_configs/*.json` は過去のスキーマ (現状 v0.60 相当:
grid が {"show"} 形式、frame / background / 目盛詳細キーなし) を凍結した
静的ファイル。スキーマを変更・移行したときに古い保存済み設定
(プリセット・再現スクリプトの元 config) が壊れていないかを常時チェックする。

新しいスキーマ移行を行ったら、その直前のバージョンの代表 config を
ここに追加すること (生成した fixture は以後手で編集しない)。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from climcanvas.core import dataset as mc_dataset
from climcanvas.core import render as mc_render
from climcanvas.core import scriptgen as mc_scriptgen

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "legacy_configs"
_FIXTURES = sorted(_FIXTURE_DIR.glob("*.json"))


def test_fixture_corpus_is_not_empty():
    assert _FIXTURES, f"互換コーパスが空です: {_FIXTURE_DIR}"


@pytest.mark.parametrize("fixture", _FIXTURES, ids=[p.stem for p in _FIXTURES])
def test_legacy_config_renders_and_scripts(fixture, sample_path, tmp_path):
    """旧形式 config で (1) アプリ内描画、(2) スクリプト生成、(3) 生成
    スクリプトのスタンドアロン実行、の3つが例外なく通ること。"""
    cfg = json.loads(fixture.read_text(encoding="utf-8"))
    datasets = {"ds0": mc_dataset.open_dataset(sample_path)}

    fig = mc_render.render_figure(cfg, datasets)
    plt.close(fig)

    out_png = tmp_path / "legacy.png"
    script = mc_scriptgen.generate_script(
        cfg, datasets, {"ds0": sample_path},
        figure_output=str(out_png), figure_dpi=100, include_show=False)
    script_path = tmp_path / "repro.py"
    script_path.write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script_path)],
                            capture_output=True, text=True,
                            cwd=tmp_path, env=env, timeout=180)
    assert result.returncode == 0, result.stderr[-1500:]
    assert out_png.exists()
