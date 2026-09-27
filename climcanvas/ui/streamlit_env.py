# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""Streamlit 本体の版ごとの挙動をアプリ側で吸収する小さな処理。"""

from __future__ import annotations

import os

_skills_nudge_done = False


def dismiss_skills_nudge_once() -> None:
    """Streamlit 1.64 以降がローカル起動時に出す「Help agents write better apps —
    Install the official Streamlit skills」の案内 (skills nudge) を、次回以降出さないようにする。

    Streamlit は「Don't show again」を押すと ~/.streamlit/.skills_nudge_dismissed という
    空の目印ファイルを作り、以後は案内を送らない (streamlit.web.skills)。ClimCanvas は
    AI エージェント向けの案内を利用者に見せたくないので、同じ関数で目印を作っておく。
    案内の要否は各セッションの開始時に Streamlit がイベントループ上で決めるが、スクリプト側の
    この処理 (数ミリ秒) の方が先に済むことが多く、目印の無いマシンでも初回から出ないことが多い
    (実測、2026-09-26)。順序は保証されないので、初回に 1 回出ることはあり得る。

    - 目印の作成は Streamlit の公開関数 write_nudge_dismissed_marker に任せる
      (保存先の決め方を追随させるため)。無い版 (1.63 以前) では何もしない。
    - 1 プロセスにつき 1 回だけ試み、失敗しても (書き込み不可など) 無視する。
    - 環境変数 CLIMCANVAS_SKILLS_NUDGE=show で何もしない (案内を見たい人と、マニュアル用の
      スクリーンショット撮影のため)。
    """
    global _skills_nudge_done
    if _skills_nudge_done:
        return
    _skills_nudge_done = True
    if os.environ.get("CLIMCANVAS_SKILLS_NUDGE", "").strip().lower() == "show":
        return
    try:
        from streamlit.web import skills  # 1.64+

        skills.write_nudge_dismissed_marker()
    except Exception:
        pass
