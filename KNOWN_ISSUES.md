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
| (none yet) | | | No value-affecting bug has been confirmed so far. | | | |
