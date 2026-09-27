# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""ClimCanvas コア層。

このパッケージは Streamlit を import してはならない (development_policy.md 1.2節)。
UI層 (app.py) は plot_config / figure_config 辞書を組み立ててここに渡すだけにする。
"""
