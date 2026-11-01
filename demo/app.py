# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""Entry point of the public demo (Streamlit Community Cloud).

A thin wrapper around the real app (``app.py`` at the repository root). It
switches the app into the demo mode through environment variables, starts the
interface in English, generates
the two sample files offered by the demo (``data/demo/sample_atmos.nc`` and
``sample_1d_time.nc``) when they are missing, pre-fetches the Natural Earth
shapefiles used by the map layers, and then runs the real app in this process. Nothing
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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The only directory the demo can open. It holds just the two files below
# (data/ is ignored by git, so they are generated at start-up).
DEMO_DIR = ROOT / "data" / "demo"

# (file offered by the demo, generator script under scripts/). Each generator
# writes more files than the one we keep, so it runs in a temporary directory
# and only the wanted file is copied into DEMO_DIR.
DEMO_FILES = (
    ("sample_atmos.nc", "make_sample_data.py"),      # 4-D atmospheric fields (maps, sections, animations)
    ("sample_1d_time.nc", "make_sample_1d_data.py"),  # 1-D time series (line plots, distributions)
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
    os.environ["CC_ALLOWED_DIRS"] = str(DEMO_DIR)
    # The demo starts in English; visitors can switch the language in the sidebar.
    os.environ.setdefault("CC_LANG", "en")
    # The server-side session storage and the startup preset are hidden in the
    # demo mode; point them away from the home directory anyway.
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "climcanvas-demo"
    os.environ.setdefault("CC_SESSION_DIRS", str(scratch / "sessions"))
    os.environ.setdefault("CLIMCANVAS_PRESET_PATH", str(scratch / "preset.json"))


def ensure_sample_data() -> None:
    """Generate each missing demo file (a few seconds in total).

    ``make_sample_data.py`` takes an output file path and also writes
    ``sample_terrain.nc`` next to it; ``make_sample_1d_data.py`` takes an output
    directory and writes two files. Both run in a temporary directory so that
    only the files listed in DEMO_FILES end up in DEMO_DIR.
    """
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for name, script in DEMO_FILES:
        if (DEMO_DIR / name).is_file():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            arg = os.path.join(tmp, name) if script == "make_sample_data.py" else tmp
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), arg],
                           check=True, cwd=ROOT)
            shutil.copyfile(os.path.join(tmp, name), DEMO_DIR / name)


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
    # The app resolves relative paths against the working directory and imports
    # ``climcanvas`` from the repository root. Community Cloud starts in the root already; make
    # both explicit for a local ``streamlit run demo/app.py``.
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    ensure_sample_data()
    prefetch_natural_earth()
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")


main()
