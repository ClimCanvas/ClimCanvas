# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""Entry point of the public demo (Streamlit Community Cloud).

A thin wrapper around the real app (``app.py`` at the repository root). It
switches the app into the demo mode through environment variables, generates
the sample data when it is missing, pre-fetches the Natural Earth shapefiles
used by the map layers, and then runs the real app in this process. Nothing
here calls Streamlit, so ``app.py`` still issues ``st.set_page_config`` as the
first Streamlit command.

Streamlit re-executes this file on every interaction, so each step must be
cheap once its work is done (file-existence checks only, no network).

Deployment settings (see ``demo/README.md``): main file path ``demo/app.py``,
Python 3.12, Python dependencies from ``demo/requirements.txt`` (which pulls in
the root ``requirements.txt``), apt packages from the root ``packages.txt``.
The same file also runs locally::

    streamlit run demo/app.py
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"

# (generator script under scripts/, a file it writes under data/sample/)
SAMPLE_SCRIPTS = (
    ("make_sample_data.py", "sample_atmos.nc"),
    ("make_sample_1d_data.py", "sample_1d_time.nc"),
    ("make_sample_2d_data.py", "sample_matrix.nc"),
    ("make_sample_track_data.py", "sample_besttrack.nc"),
    ("make_sample_curvilinear_data.py", "sample_curvilinear.nc"),
    ("make_sample_multifile_data.py", "multi/sample_atmos_2024010100.nc"),
)

# Natural Earth shapefiles drawn by the map layers (coastlines, land, ocean and
# borders). The 110m and 50m scales are fetched here; 10m is fetched on demand.
NATURAL_EARTH = (
    ("physical", "coastline"),
    ("physical", "land"),
    ("physical", "ocean"),
    ("cultural", "admin_0_boundary_lines_land"),
)
NATURAL_EARTH_SCALES = ("110m", "50m")


def configure_environment() -> None:
    """Demo mode: only the sample data can be opened; nothing is written to $HOME."""
    os.environ["CC_MODE"] = "demo"
    os.environ["CC_ALLOWED_DIRS"] = str(SAMPLE_DIR)
    # The server-side session storage and the startup preset are hidden in the
    # demo mode; point them away from the home directory anyway.
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "climcanvas-demo"
    os.environ.setdefault("CC_SESSION_DIRS", str(scratch / "sessions"))
    os.environ.setdefault("CLIMCANVAS_PRESET_PATH", str(scratch / "preset.json"))


def ensure_sample_data() -> None:
    """Run each sample generator whose output is missing (a few seconds in total)."""
    for script, marker in SAMPLE_SCRIPTS:
        if not (SAMPLE_DIR / marker).is_file():
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           check=True, cwd=ROOT)


def prefetch_natural_earth() -> None:
    """Download the Natural Earth shapefiles once per container.

    ``natural_earth()`` returns at once when the file is already in cartopy's data
    directory. A download failure is not fatal: the map layers fetch on demand.
    """
    try:
        from cartopy.io import shapereader
    except Exception:
        return
    for scale in NATURAL_EARTH_SCALES:
        for category, name in NATURAL_EARTH:
            try:
                shapereader.natural_earth(resolution=scale, category=category, name=name)
            except Exception:
                pass


def main() -> None:
    configure_environment()
    # The app resolves relative paths (the default sample path, the generator
    # outputs) against the working directory and imports ``climcanvas`` from
    # the repository root. Community Cloud starts in the root already; make
    # both explicit for a local ``streamlit run demo/app.py``.
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    ensure_sample_data()
    prefetch_natural_earth()
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")


main()
