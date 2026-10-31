# Known issues affecting figure values

This file lists bugs in ClimCanvas that changed the *values* shown in a figure
(a wrong selection, averaging region or weight, masking, value transform,
sign, orientation or statistic), as opposed to cosmetic problems. It exists so
that you can check whether a figure made with a given version was affected.

## Policy

- A bug that affects figure values is listed here within 7 days of being
  confirmed, with the affected versions and features, before or together with
  the fix.
- The corresponding entry under *Fixed* in `CHANGELOG.md` starts with
  "(affects figure values)".
- On GitHub, such issues carry the label `affects-values`.
- Cosmetic bugs (layout, labels, colors that do not change which value a color
  or a line represents) are not listed here; see `CHANGELOG.md`.

## How to check a figure

The version of ClimCanvas that made a figure is written in the first line of
its reproduction script. Compare it with the "Affected versions" column below.
The numerical conventions ClimCanvas follows (exact-match selection, region
padding, averaging weights, missing values, statistics) are documented in the
manual chapter "Conventions"; a difference from what you expected is not
necessarily a bug, so please read that chapter first.

## Issues

| ID | Affected versions | Feature | Symptom | Effect on values | Fixed in | Listed |
|---|---|---|---|---|---|---|
| KI-1 | 1.00, 1.00.1 (and earlier internal versions) | Error bars taken from a variable: bar charts in the 1-D plot; scatter and bubble charts | The value transform of the main variable was also applied to the error amounts: bar charts drew \|scale × error + offset\|, scatter and bubble charts drew \|scale × error\|. Error bars given as a constant were not affected. | Error bars had the wrong length whenever the layer had a value transform (scale ≠ 1 or offset ≠ 0). With an offset (e.g. K → °C) the bars of a bar chart became as long as the offset. Error amounts are now never transformed: provide them in the units shown in the figure. | 1.01 | 2026-09-29 |
| KI-2 | 1.00 – 1.04.2 installed with pip when the pyproj 3.8 wheels (PROJ 9.8, available since 2026-09-05) were picked up. conda installations with `environment.yml` (PROJ pinned below 9.8) were not affected | Horizontal maps: coastlines, land / ocean fill, country borders | With PROJ 9.8 and cartopy 0.25, once a map with a different central longitude had been drawn in the same process, the coastlines of later maps were shifted by about 0.19° in latitude (cartopy issue #2708). The plotted fields themselves were drawn correctly. | Coastlines and borders did not match the data by up to about 0.19° in latitude, so features near a coast could appear on the wrong side of the shoreline. The values of the plotted fields were not affected. | 1.05 (`requirements.txt` pins `pyproj<3.8`; with an older checkout, `pip install "pyproj<3.8"`) | 2026-10-31 |
