# Changelog

All notable changes to ClimCanvas are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); entries are
grouped as Added / Changed / Fixed / Removed and written for users of the app
and of the generated scripts. Releases before 1.0 were internal development
versions and are not listed.

## Unreleased

## 1.00 — 2026-09-27

First public release. The entries below list the changes since the last
internal version (0.98).

### Added

- GitHub issue forms for bug reports and feature requests. The new-issue page also links to a feedback form (English and Japanese) for comments and requests from users without a GitHub account.
- "About" in the app menu (top right) shows the version and links to the feedback form (in the UI language, with the version filled in), GitHub Issues and the license. The links only open a web page; the app sends nothing. Requires Streamlit 1.46 or later; with older versions the menu item is not shown.
- A box under the figure, "Processing applied to this figure", lists per panel the value transforms (scale a and offset b for each variable, with the variable's units attribute), the range averages (dimension, range, arithmetic or cos(lat)-weighted, number of grid points, and the number of missing values excluded from the mean), maskout thresholds, and a reminder that vector components on curvilinear grids are not rotated. It appears only when such processing is applied.
- `config.example.toml`: a commented template of the optional user configuration (`allowed_dirs`, `session_dirs`, `mode`) to copy to `~/.climcanvas/config.toml`.
- Online manual: Japanese sources under `manual/ja/` (seven chapters and an FAQ) and `scripts/build_manual.py`, which renders them into the project website with python-markdown.
- Curvilinear grids: data whose longitude and latitude are two-dimensional
  arrays (ocean models, WRF, regional models on Lambert conformal grids such
  as ClimCORE) can now be drawn on horizontal maps with every map layer type
  (fill, hatch, contour, vector, streamline, map scatter) in any projection.
  A longitude/latitude range cuts the grid to the bounding box of the points
  inside it; with no range specified, the map extent is the projected
  bounding box of all grid points, so a Lambert grid drawn in its own
  projection fills the frame without margins. Longitudes crossing the
  dateline are handled. Sections along longitude/latitude and
  longitude/latitude range means are not available for these grids, and
  vector components are taken as eastward/northward.
- Coordinate files: when a data file has no longitude/latitude of its own
  (e.g. ClimCORE, whose coordinates live in separate FLON/FLAT files), attach
  them from one or two separate files via "Coordinate file" under each loaded
  file in the sidebar. The choice is saved with the session, and the generated
  script opens and attaches the same files.
- Grid projection detection: for two-dimensional coordinates on a Lambert
  conformal conic (or polar stereographic) grid, the projection, its standard
  parallels and central longitude are inferred from the coordinates and used
  as the initial projection settings, with a note under the projection
  selector.
- New "Line bundle" layer for 1-D plots: draw every slice along one dimension
  (e.g. all ensemble members) with the same color, line style and opacity and a
  single legend entry, instead of one layer per line. Summary lines (mean,
  median, min/max, a percentile, a percentile range or mean ± k sample
  standard deviations) can be overlaid, each with its own color, width, style and legend
  label; the two-line ranges can also be drawn as a shaded band, with or
  without edge lines, and a line width of 0 hides the member lines so that only
  the band is shown.
  Generated scripts reproduce the bundle with a plain loop and numpy nan-aware
  statistics.
- Map settings: a "Geographic data resolution" selector chooses the Natural
  Earth scale (auto / 1:110m / 1:50m / 1:10m) used for coastlines, country
  borders and the land/ocean fill. The default "auto" keeps cartopy's
  extent-based choice, so existing figures are unchanged. Generated scripts
  pass the chosen scale explicitly (`with_scale` / `resolution=`).
- Map settings: "Draw above data" under "Fill land" draws the land fill on top
  of the fill, hatching and vector layers, hiding values that spill onto land,
  while contours, streamlines, tracks, coastlines, borders, gridlines, boxes
  and annotations stay on top of the land. Scatter points, track points and
  the reference vector are also kept above the land. Off by default; generated
  scripts reproduce the same drawing order.
- `KNOWN_ISSUES.md`: a list of bugs that affected figure values (with the
  affected versions and features) and the policy for keeping it current, and
  `CITATION.cff` with the citation metadata that GitHub and Zenodo read.
- Manual: a "Conventions" chapter that collects every numerical convention
  (exact-match selection, region padding, area averaging and weights, missing
  values, value transforms, statistics definitions) in one place, and a
  "Before you publish" checklist in the output chapter.
- README: a Notes section on the warranty disclaimer, network access (Natural
  Earth downloads) and how to opt out of Streamlit's usage statistics.

### Changed

- README: the installation section now explains that a plain clone is always
  the latest release, how to clone a specific version by tag, and how to update.
- Comments in the environment and tooling files (`environment.yml`, `requirements.txt`,
  `ruff.toml`, `pytest.ini`, `.streamlit/config.toml`, the GitHub workflow) are now in English.
- The one-line description now reads "atmospheric and ocean data" (app subtitle in all four
  UI languages, README) instead of "meteorological and climate data".
- The "Install the official Streamlit skills" prompt that Streamlit 1.64+ shows on local
  startup is suppressed: ClimCanvas records the same "Don't show again" marker that the
  prompt's button would write (set `CLIMCANVAS_SKILLS_NUDGE=show` to keep it).
- Horizontal maps: for regional data (not covering the globe) the
  longitude/latitude range is now enabled by default in every projection,
  with the data's extent as the initial values (the longitude range is the
  smallest arc containing all points, so grids crossing the dateline work).
  Global data keeps the previous defaults.
- The layer reorder buttons now explain how stacking works on maps: fills,
  hatching, vectors and scatter always stay below contours, streamlines and
  tracks, so the order only affects layers of the same kind and the legend.
- The "Delete" button next to each loaded file and the "Duplicate" / "Delete"
  buttons in the panel list are now icon buttons with tooltips. Streamlit 1.64
  stopped wrapping button labels, so the text labels were being cut off in the
  narrow sidebar columns.

### Fixed

- Animations now advance every panel, including panels whose layer pins the
  time coordinate to a fixed value; such a panel used to stay on one time step
  in the app while the generated animation script advanced it.
- The "layer type to add" selector now shows its options in the selected UI
  language after switching languages.
- The plot-axis range slider of 1-D plots now spans the coordinate values of
  every loaded file, not just the first one, so a historical run followed by a
  scenario file can be ranged and the min/max inputs stay linked to the slider.
- The move-left / move-right buttons in the panel list no longer overlap each
  other when the sidebar has its default width.
