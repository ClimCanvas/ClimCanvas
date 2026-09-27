# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""プロセス状態リーク (観点⑦): 前に何を描いたかで図が変わらないこと。

アプリ (Streamlit) は1プロセスで何百枚も描くため、rcParams 等のグローバル
状態のリークは「A を描いた後だけ B の図が変わる」実バグになる
(docs/semantic_test_plan.md 観点⑦)。

方針: warm と cold のプロセス比較はアンチエイリアスが揺れる
(implementation_checklist E) ため、**同一プロセス内**で「B → B (基準)」と
「A → B」を比較する。A にはグローバル状態を触る機能を意図的に盛る
(figure.font_family / hatch の rc_context / カラーバー)。
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from climcanvas.core import config as mc_config
from climcanvas.core import render as mc_render


def _map_ds() -> xr.Dataset:
    lon = np.arange(100.0, 190.0, 10.0)
    lat = np.arange(0.0, 50.0, 10.0)
    val = lat[:, None] * 10.0 + lon[None, :] * 0.1
    return xr.Dataset(
        {"f": (("lat", "lon"), val)},
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})})


def _cfg_b() -> dict:
    """検証対象 B: 既定フォントの素朴な地図 (文字列多め = フォントリークに敏感)。"""
    panel = mc_config.default_panel()
    panel["selection"] = {}
    panel["title"] = "panel B: leak probe 0123456789"
    fill = mc_config.default_fill_layer("ds0", "f")
    fill["style"]["levels"] = [100.0, 200.0, 300.0, 400.0]
    panel["layers"] = [fill]
    cfg = mc_config.default_figure_config()
    cfg["panels"] = [panel]
    return cfg


def _cfg_a() -> dict:
    """状態を汚しにいく A: font_family 指定 + hatch (rc_context) + contour。"""
    cfg = _cfg_b()
    panel = cfg["panels"][0]
    panel["title"] = "panel A: state polluter"
    cfg["figure"]["font_family"] = "serif"
    hatch = mc_config.default_hatch_layer("ds0", "f")
    hatch["style"].update({"levels": [150.0, 350.0], "pattern": "x",
                           "linewidth": 2.5})
    contour = mc_config.default_contour_layer("ds0", "f")
    contour["style"].update({"levels": [120.0, 240.0, 360.0],
                             "linewidth": 3.0})
    panel["layers"] += [hatch, contour]
    return cfg


def _snap(cfg, datasets) -> np.ndarray:
    fig = mc_render.render_figure(cfg, datasets)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return mpimg.imread(buf)


def test_rendering_a_does_not_change_subsequent_b():
    datasets = {"ds0": _map_ds()}
    _snap(_cfg_b(), datasets)              # ウォームアップ (フォントキャッシュ等)
    b_base = _snap(_cfg_b(), datasets)     # 基準: B → B
    _snap(_cfg_a(), datasets)              # 状態を汚しにいく A
    b_after = _snap(_cfg_b(), datasets)    # A の後の B
    assert b_base.shape == b_after.shape, "A の描画後に B のサイズが変わった"
    assert np.array_equal(b_base, b_after), \
        "A の描画後に B の図が変わった (グローバル状態のリーク)"


def test_warm_renders_are_repeatable():
    """同一プロセス内 (ウォームアップ後) の同一 config は完全再現する。"""
    datasets = {"ds0": _map_ds()}
    _snap(_cfg_b(), datasets)
    b1 = _snap(_cfg_b(), datasets)
    b2 = _snap(_cfg_b(), datasets)
    assert np.array_equal(b1, b2), "同一 config の連続描画が一致しない"


_CLON_RUNNER = '''\
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, xarray as xr
sys.path.insert(0, sys.argv[3])
from climcanvas.core import config as c, render as r

lon = np.arange(100.0, 190.0, 10.0); lat = np.arange(0.0, 50.0, 10.0)
ds = xr.Dataset({"f": (("lat", "lon"), lat[:, None]*10 + lon[None, :]*0.1)},
                coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                        "lon": ("lon", lon, {"units": "degrees_east"})})

def render(clon):
    panel = c.default_panel()
    panel["selection"] = {}
    panel["region"] = {"lon_min": 125.0, "lon_max": 135.0,
                       "lat_min": 29.0, "lat_max": 39.0}
    panel["projection"] = {"name": "PlateCarree", "central_longitude": clon,
                           "central_latitude": 0.0}
    fill = c.default_fill_layer("ds0", "f")
    fill["style"]["levels"] = [100.0, 200.0, 300.0, 400.0]
    panel["layers"] = [fill]
    cfg = c.default_figure_config(); cfg["panels"] = [panel]
    fig = r.render_figure(cfg, {"ds0": ds})
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf

seq = [float(x) for x in sys.argv[1].split(",")]
for clon in seq[:-1]:
    render(clon)
np.save(sys.argv[2], render(seq[-1]))
'''


def _proj_version() -> tuple[int, int]:
    import pyproj

    return tuple(int(v) for v in pyproj.proj_version_str.split(".")[:2])


@pytest.mark.xfail(
    _proj_version() >= (9, 8),
    strict=False,
    reason="PROJ 9.8 起因 (cartopy PR #2653 で修正予定): 中心経度の異なる地図を"
           "同一プロセスで先に描くと、以後の海岸線だけが緯度方向に ~0.19° ずれる"
           " (sticky)。上流報告: https://github.com/SciTools/cartopy/issues/2708 / "
           "docs/cartopy_compat_notes.md「既知の上流バグ」参照。proj<9.8 に固定した"
           "環境では通常テストとして動き、proj が 9.8 に戻る等の再発を fail で検知")
def test_map_render_order_independent_across_central_longitudes(tmp_path):
    """地図描画は同一プロセス内の描画履歴 (別の中心経度) に依存しない。

    「clon=130.9 のみ」と「clon=180 → 130.9」の最終描画をコールドプロセスで
    比較する。海岸線 (FeatureArtist) だけがずれる上流バグの監視。
    """
    import subprocess
    import sys as _sys

    runner = tmp_path / "clon_runner.py"
    runner.write_text(_CLON_RUNNER, encoding="utf-8")
    root = str(Path(__file__).resolve().parents[1])
    outs = {}
    for tag, seq in (("cold", "130.9"), ("warm", "180,130.9")):
        out = tmp_path / f"{tag}.npy"
        res = subprocess.run(
            [_sys.executable, str(runner), seq, str(out), root],
            capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, res.stderr[-1500:]
        outs[tag] = np.load(out)
    assert np.array_equal(outs["cold"], outs["warm"]), \
        "別の中心経度を先に描いたプロセスで、同じ config の地図が変わった"
