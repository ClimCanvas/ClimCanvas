# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""代表 config を一括描画して目視レビュー用のギャラリー HTML を作る。

render⇄scriptgen の画像一致テストは「2経路が同じ絵を出す」ことしか保証しない。
最終的な「見た目の正しさ」の判断は人間 (または画像を読める AI) に残るため、
代表的な図を1ページに並べてレビューを数分で済ませる
(docs/semantic_test_plan.md 段階4)。大きな描画改修の後に生成して眺めること。

使い方 ($PY = miniforge env_3.12 の python):
    $PY scripts/make_contact_sheet.py             # → data/contact_sheet/index.html
    $PY scripts/make_contact_sheet.py --out DIR   # 出力先を指定

config は tests/test_consistency.py の手書きビルダーを流用する (テストと同じ
図がレビュー対象になる)。サンプルデータは出力先の _data/ に生成する。
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import xarray as xr

from climcanvas.core import render as mc_render
from scripts.make_sample_data import create_sample_dataset
from scripts.make_sample_track_data import create_track_dataset

import test_consistency as tc  # noqa: E402  (代表 config のビルダー群)

# (表示名, config ビルダー, データ種別, レビューの見どころ)
CASES = [
    ("simple", tc._simple_config, "atmos", "地図 + 塗りつぶしの基本形"),
    ("rich-layers", tc._rich_config, "atmos", "fill+hatch+contour+vector の重ね"),
    ("contour-cbar-extend", tc._contour_cbar_extend_config, "atmos",
     "contour のカラーバー2本 (bottom/right) + vector 連続 extend (left, 矢印)"),
    ("npolar-sector", tc._npolar_sector_config, "atmos",
     "極投影の扇形セクター (境界に隙間がないか)"),
    ("stream-map", tc._stream_map_config, "atmos",
     "流線の |V| 色付け + 離散化カラーバー"),
    ("map-scatter-grid", tc._map_scatter_grid_config, "atmos",
     "地図散布図 (格子) + 離散化"),
    ("vsection-contour-emphasis-cmap", tc._vsection_contour_emphasis_cmap_config,
     "atmos", "鉛直断面 + cmap 等値線 + 強調レベル"),
    ("hovmoller", tc._hovmoller_config, "atmos", "時間断面 (Hovmöller)"),
    ("line-1d-two-lines", tc._line_1d_two_lines_config, "atmos",
     "1次元プロット2本 + 凡例"),
    ("bubble-cmap-discrete", tc._bubble_2d_uvt_cmap_discrete_config, "atmos",
     "バブル: サイズ正規化 + 離散化カラーバー"),
    ("multipanel-shared-colorbar", tc._multipanel_shared_colorbar_config,
     "atmos", "複数パネル + 共有カラーバー"),
    ("dist-box-violin", tc._dist_box_violin_config, "series",
     "箱ひげ2系列 + バイオリン (系列名の x 目盛)"),
    ("agg-hist2d-discrete", tc._agg_hist2d_discrete_config, "series",
     "hist2d の後付け離散化 (カラーバーが段々 + extend max)"),
    ("agg-hexbin-discrete", tc._agg_hexbin_discrete_config, "series",
     "hexbin の離散化 (対数風の階級 1,2,5,10,20)"),
    ("track", tc._track_config, "track",
     "ベストトラック: 風速色付き点 + maskout で線が切れる"),
]

_HTML_HEAD = """<!doctype html><meta charset="utf-8">
<title>ClimCanvas contact sheet</title>
<style>
 body {{ font-family: sans-serif; margin: 1.5rem; background: #fafafa; }}
 h1 {{ font-size: 1.2rem; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
          gap: 1rem; }}
 figure {{ margin: 0; background: #fff; border: 1px solid #ddd; padding: .5rem; }}
 img {{ max-width: 100%; height: auto; }}
 figcaption {{ font-size: .8rem; color: #333; margin-top: .3rem; }}
 figcaption b {{ color: #000; }}
 .fail {{ color: #b00; white-space: pre-wrap; font-size: .75rem; }}
</style>
<h1>ClimCanvas contact sheet — {date} (matplotlib {mpl})</h1>
<p>目視レビュー用の代表図一覧 (docs/semantic_test_plan.md 段階4)。</p>
<div class="grid">
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data" / "contact_sheet"),
                        help="出力ディレクトリ (既定: data/contact_sheet)")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "_data"
    data_dir.mkdir(exist_ok=True)

    paths = {"atmos": data_dir / "sample_atmos.nc",
             "series": data_dir / "series.nc",
             "track": data_dir / "besttrack.nc"}
    if not paths["atmos"].exists():
        create_sample_dataset(str(paths["atmos"]))
    if not paths["series"].exists():
        tc._make_series_nc(str(paths["series"]))
    if not paths["track"].exists():
        create_track_dataset(str(paths["track"]))
    datasets = {kind: xr.open_dataset(p) for kind, p in paths.items()}

    entries = []
    n_fail = 0
    for name, builder, kind, note in CASES:
        png = f"{name}.png"
        try:
            cfg = builder()
            fig = mc_render.render_figure(cfg, {"ds0": datasets[kind]})
            fig.savefig(out / png, dpi=100, bbox_inches="tight")
            plt.close(fig)
            entries.append(
                f'<figure><img src="{png}" alt="{name}">'
                f"<figcaption><b>{name}</b> — {note}</figcaption></figure>")
            print(f"ok   {name}")
        except Exception as exc:  # 1枚失敗しても残りは描く
            n_fail += 1
            entries.append(
                f"<figure><figcaption><b>{name}</b> — {note}"
                f'<div class="fail">描画失敗: {exc!r}</div>'
                f"</figcaption></figure>")
            print(f"FAIL {name}: {exc!r}")

    html = _HTML_HEAD.format(
        date=datetime.date.today().isoformat(), mpl=matplotlib.__version__)
    html += "\n".join(entries) + "\n</div>\n"
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"\n{len(CASES) - n_fail}/{len(CASES)} 枚を描画: {out / 'index.html'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
