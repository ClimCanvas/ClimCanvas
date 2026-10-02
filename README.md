# ClimCanvas

ClimCanvas is a browser-based GUI for plotting atmospheric and ocean data
stored in netCDF files. Build horizontal maps, vertical and time sections, 1-D
plots, scatter plots and more from a sidebar, then export the result as an image
or animation, or as a **standalone Python script** (xarray / matplotlib /
cartopy) that reproduces the figure exactly and can be edited by hand.

Website (in Japanese): <https://www.atmos.rcast.u-tokyo.ac.jp/climcanvas/> — with the
[user manual](https://www.atmos.rcast.u-tokyo.ac.jp/climcanvas/manual/index.html) and the [FAQ](https://www.atmos.rcast.u-tokyo.ac.jp/climcanvas/faq.html).

## Features

- Reads netCDF files through xarray, with automatic detection of latitude,
  longitude, vertical and time coordinates (descending axes, 0-360 or -180-180
  longitudes and non-standard calendars are handled)
- Plot types: horizontal maps on cartopy projections, vertical and time
  sections, 1-D line / bar plots, distributions (histogram, ECDF, box, violin),
  scatter and bubble plots, 2-D aggregates (hist2d, hexbin) and heatmaps
- Layers: filled contours, contour lines, hatching, vectors, streamlines, point
  scatter and best-track storm tracks, stacked freely on one panel
- Multi-panel figures with mosaic layout, shared colorbar and panel labels
- Animations along any dimension and a rotating-globe animation (GIF / MP4)
- Reproduction scripts: every figure can be exported as a self-contained `.py`
  that depends only on xarray, matplotlib and cartopy. The test suite checks
  that the app and the script produce pixel-identical images
- Presets and sessions (JSON), custom colormaps, and a UI in Japanese,
  English, Chinese (simplified) and Korean

## Installation

```bash
git clone https://github.com/ClimCanvas/ClimCanvas.git
cd ClimCanvas
```

A plain clone gives you the latest release: the public repository receives one
commit per release, so the tip of `main` is always the newest version (the
"Latest release" badge on GitHub marks only the versions that have a DOI, not
necessarily the newest). To use a specific version, clone its tag:

```bash
git clone --branch v1.02 https://github.com/ClimCanvas/ClimCanvas.git
```

To update an existing clone, run `git pull` in it; your settings and sessions
live in `~/.climcanvas/` and are not touched. The version you are running is
shown under the title in the app and in the first line of every reproduction
script.

Python 3.12 with a conda environment is recommended:

```bash
conda env create -f environment.yml
conda activate climcanvas
```

or, with pip:

```bash
pip install -r requirements.txt
```

`environment.yml` pins PROJ to versions older than 9.8 on purpose: PROJ 9.8
has a known bug that shifts PlateCarree coastlines (cartopy issue 2708).

Optional colormap packages are detected at start-up if installed:

```bash
pip install cmocean cmcrameri cmaps
```

## Usage

```bash
python scripts/make_sample_data.py      # writes data/sample/sample_atmos.nc
python -m streamlit run app.py
```

Open the URL that Streamlit prints, choose a netCDF file, build the figure in
the sidebar, and use the download buttons for PNG / SVG / PDF / EPS, GIF / MP4,
or the reproduction script. The [user manual](https://www.atmos.rcast.u-tokyo.ac.jp/climcanvas/manual/index.html)
(Japanese) covers the sidebar, every layer type and the numerical conventions;
the [FAQ](https://www.atmos.rcast.u-tokyo.ac.jp/climcanvas/faq.html) collects common questions and error messages.

ClimCanvas needs no configuration file. An optional `~/.climcanvas/config.toml`
on the machine that runs the app can restrict which directories netCDF files
may be opened from (`allowed_dirs`, recommended on shared servers), list the
directories offered for saving named sessions (`session_dirs`), and switch the
session tabs to remote-server wording (`mode = "remote"`). Copy
`config.example.toml` there and uncomment the lines you need; the same settings
are available as the environment variables `CC_ALLOWED_DIRS`, `CC_SESSION_DIRS`
and `CC_MODE` for a single run.

To run the tests:

```bash
python -m pytest -q
```

## Notes

- ClimCanvas is provided without warranty of any kind (AGPL-3.0, sections 15
  and 16). It draws exactly what it is configured to draw; checking that a
  figure shows what you intend (the selected level and time, the averaging
  region and weights, missing values, projection) is up to you. The numerical
  conventions are collected in the manual chapter "Conventions", and bugs
  that affected figure values are listed in `KNOWN_ISSUES.md`. Please cite
  the version you used (see Citation).
- On first use, cartopy downloads Natural Earth map data. ClimCanvas itself
  sends nothing over the network, but Streamlit collects anonymous usage
  statistics by default; to opt out, put the following in
  `~/.streamlit/config.toml` on the machine that runs the app:

  ```toml
  [browser]
  gatherUsageStats = false
  ```

## Citation

A software description paper is in preparation. Until it is published, please
cite this repository with the version you used; `CITATION.cff` holds the
citation metadata (GitHub shows it under "Cite this repository"). Releases are
archived on Zenodo: the concept DOI
[10.5281/zenodo.22997360](https://doi.org/10.5281/zenodo.22997360) always
resolves to the latest archived version. A version DOI is minted only for
milestone versions (v1.00 and the version used in the paper); the other
versions have a git tag but no DOI of their own, so cite them with the concept
DOI and the version number. Version DOIs so far: v1.00
[10.5281/zenodo.22997361](https://doi.org/10.5281/zenodo.22997361).

## Contributing

ClimCanvas is open source, but not open contribution.
Bug reports and feature requests are welcome as
[GitHub Issues](https://github.com/ClimCanvas/ClimCanvas/issues).
Comments and requests are also welcome through the feedback form, which needs
no GitHub account and whose responses are not public
([English](https://docs.google.com/forms/d/e/1FAIpQLSc-tnfYzsqcctESephPCJqbht_TQRl6TPw8NE7sgkmlcAZ68w/viewform),
[Japanese](https://docs.google.com/forms/d/e/1FAIpQLSe-zXza-DXr8eGdOIgLKBH98luuLQdCooD6fcjCO3fA3gvuKw/viewform)).
Both are linked from "About" in the app menu (top right).
Unsolicited pull requests are not accepted: the maintainer implements changes
so that the app and the generated scripts stay in lockstep
(`render.py` and `scriptgen.py`) and the copyright stays single-authored.
If you would like to contribute code, please open an issue first.
See `CONTRIBUTING.md` for details.

## License

ClimCanvas is licensed under the GNU Affero General Public License,
version 3 only (SPDX: AGPL-3.0-only), with additional terms under
section 7 of the license (see `LICENSE.exception`):

- Scripts generated by ClimCanvas belong to you and are not covered by the
  AGPL. Figures, animations, sessions and other outputs are yours as well.
- The name "ClimCanvas" and the ClimCanvas logo may not be used to identify
  modified versions.

See `LICENSE` for the full text of the AGPL. If you run a *modified* version
of ClimCanvas as a network service for other people, section 13 of the AGPL
requires you to offer those users the source code of your modified version.
Running an unmodified copy, or using the program yourself, carries no
obligation.

Licenses of the third-party packages ClimCanvas depends on are listed in
`THIRD_PARTY_LICENSES.md`. Note that the optional colormap package `cmaps`
is licensed under the GPL-3.0; a reproduction script that imports it is
subject to that package's license.

## Name and logo

"ClimCanvas" and the ClimCanvas logo (shown on the project website) identify
the original project. The logo is not covered by the AGPL and is not
distributed in this repository; all rights are reserved. A modified version
must use a different name and remove the logo (`LICENSE.exception`, term 2).

## Contact

Questions and comments: climcanvas@gmail.com (bug reports and feature requests
are best filed as GitHub Issues, and comments can also be sent through the
feedback form, see above).
