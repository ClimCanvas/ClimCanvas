# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""外部 cmap パッケージのレジストリと自動検出。

各パッケージが matplotlib にどう cmap を登録するか (プレフィクスを付けて
`mpl.colormaps` から拾えるようにする) と、生成スクリプトに挿入する
import 文を持つ。app 起動時に `discover()` で利用可能なものだけ集める。

譲れないルール2 (再現スクリプトのスタンドアロン性) の例外として、
ユーザーが外部 cmap を選択したときのみ、生成スクリプトの冒頭に該当
パッケージの import を追加する (development_policy.md 1.4節の緩和)。

サポートしているパッケージ:
- cmocean   : `cmo.<name>`   (import で自動 mpl 登録)
- cmcrameri : `cmc.<name>`   (import で自動 mpl 登録)
- cmaps     : `cmaps.<name>` (NCL 由来 colormap 集。自動登録しないので手動登録)
  ※ 旧 `nclcmaps` パッケージは開発停止しており、後継の `cmaps` に置き換え済み。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ExternalCmapPackage:
    key: str                    # レジストリキー (パッケージ名と同じ)
    import_stmts: tuple[str, ...]  # 生成スクリプトに挿入する import 行
    group_label: str            # CMAP_GROUPS に出す分類ラベル
    prefix: str                 # matplotlib 登録名の接頭辞 (例: "cmo.")
    activate: Callable[[], list[str]]  # 起動時にパッケージを import して mpl 名リストを返す


# --- 各パッケージの activate 実装 ---

def _cmocean_activate() -> list[str]:
    """cmocean を import して mpl に cmo.* を登録させる。

    `_r` (反転) は UI 側の反転チェックボックスで動的に切り替わるので除外。
    `_i` / `_r_i` (輝度反転のアクセシビリティバリアント) も UI がごちゃつくので除外。
    """
    import cmocean  # noqa: F401
    import matplotlib as mpl
    return sorted(
        n for n in mpl.colormaps
        if n.startswith("cmo.")
        and not n.endswith("_r")
        and not n.endswith("_i"))


def _cmcrameri_activate() -> list[str]:
    """cmcrameri.cm を import して mpl に cmc.* を登録させる。"""
    import cmcrameri.cm  # noqa: F401
    import matplotlib as mpl
    return sorted(
        n for n in mpl.colormaps
        if n.startswith("cmc.") and not n.endswith("_r"))


def _cmaps_activate() -> list[str]:
    """cmaps は自動登録しないので `cmaps.<name>` として mpl に手動登録する。

    cmaps パッケージ (hhuangwx/cmaps) は各 colormap をモジュール属性として
    公開する仕様なので、`dir(cmaps)` を走査して matplotlib.colors.Colormap
    のインスタンスだけを拾う。
    """
    import cmaps as cmaps_mod
    import matplotlib as mpl
    from matplotlib.colors import Colormap

    registered: list[str] = []
    for attr_name in dir(cmaps_mod):
        if attr_name.startswith("_"):
            continue
        obj = getattr(cmaps_mod, attr_name, None)
        if not isinstance(obj, Colormap):
            continue
        mpl_name = f"cmaps.{attr_name}"
        # 登録済みならスキップ (冪等)。force=True の再登録は matplotlib が
        # 「Overwriting the cmap ...」の UserWarning を出す。Streamlit の
        # ソース変更検知によるモジュール再読み込みでは _DISCOVERED キャッシュ
        # だけが消え、mpl レジストリ (site-packages 側) は残るため、再登録が
        # 実際に起きる (2026-09-09)
        if mpl_name not in mpl.colormaps:
            mpl.colormaps.register(obj, name=mpl_name)
        registered.append(mpl_name)
    return sorted(registered)


PACKAGES: tuple[ExternalCmapPackage, ...] = (
    ExternalCmapPackage(
        key="cmocean",
        import_stmts=("import cmocean  # noqa: F401",),
        group_label="cmocean (Oceanography)",
        prefix="cmo.",
        activate=_cmocean_activate,
    ),
    ExternalCmapPackage(
        key="cmcrameri",
        import_stmts=("import cmcrameri.cm  # noqa: F401",),
        group_label="cmcrameri (Scientific colour maps)",
        prefix="cmc.",
        activate=_cmcrameri_activate,
    ),
    ExternalCmapPackage(
        key="cmaps",
        import_stmts=("import cmaps",),
        group_label="cmaps (NCL colormaps)",
        prefix="cmaps.",
        activate=_cmaps_activate,
    ),
)


_DISCOVERED: dict[str, list[str]] | None = None


def discover(force: bool = False) -> dict[str, list[str]]:
    """import 可能な外部 cmap パッケージを検出する。

    各パッケージが提供する matplotlib 登録名 (プレフィクス付き) のリストを
    パッケージ key で引ける辞書として返す。結果はキャッシュされる。
    """
    global _DISCOVERED
    if _DISCOVERED is not None and not force:
        return _DISCOVERED
    result: dict[str, list[str]] = {}
    for pkg in PACKAGES:
        try:
            names = pkg.activate()
        except ImportError:
            continue
        except Exception:
            # API 変更などで activate が失敗しても黙ってスキップ
            continue
        if names:
            result[pkg.key] = names
    _DISCOVERED = result
    return result


def package_group_label(key: str) -> str:
    for pkg in PACKAGES:
        if pkg.key == key:
            return pkg.group_label
    return key


def packages_used_in(figure_config: dict) -> list[str]:
    """`figure_config` 内で使われている外部 cmap のパッケージ key リスト。"""
    cmaps_seen: set[str] = set()
    for panel in figure_config.get("panels", []):
        for layer in panel.get("layers", []):
            cmap = layer.get("style", {}).get("cmap")
            if cmap:
                cmaps_seen.add(cmap)
    used: list[str] = []
    for pkg in PACKAGES:
        if any(c.startswith(pkg.prefix) for c in cmaps_seen):
            used.append(pkg.key)
    return used


def needs_matplotlib(pkg_keys: list[str]) -> bool:
    """生成スクリプト側で `import matplotlib as mpl` が必要か。"""
    return "cmaps" in pkg_keys


def script_setup_lines(pkg_keys: list[str],
                       figure_config: dict) -> tuple[list[str], list[str]]:
    """生成スクリプトに追加する (import 行, セットアップ行) を返す。

    import 行は他の import と一緒に上部に、セットアップ行は
    open_dataset の前に置く想定 (cmaps の登録ループなど)。
    """
    if not pkg_keys:
        return [], []
    imports: list[str] = []
    for key in pkg_keys:
        pkg = next((p for p in PACKAGES if p.key == key), None)
        if pkg is None:
            continue
        imports.extend(pkg.import_stmts)

    setup: list[str] = []
    if "cmaps" in pkg_keys:
        used_names = _used_names_with_prefix(figure_config, "cmaps.")
        if used_names:
            bare = [n[len("cmaps."):] for n in used_names]
            names_repr = ", ".join(repr(n) for n in bare)
            setup = [
                "# cmaps の手動登録 (ClimCanvas と同じ名前空間)",
                f"for _n in ({names_repr},):",
                "    mpl.colormaps.register("
                "getattr(cmaps, _n), name=f'cmaps.{_n}', force=True)",
                "",
            ]
    return imports, setup


def _used_names_with_prefix(figure_config: dict, prefix: str) -> list[str]:
    used: list[str] = []
    for panel in figure_config.get("panels", []):
        for layer in panel.get("layers", []):
            cmap = layer.get("style", {}).get("cmap")
            if cmap and cmap.startswith(prefix) and cmap not in used:
                used.append(cmap)
    return used
