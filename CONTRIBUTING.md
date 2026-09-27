# Contributing to ClimCanvas

ClimCanvas is **open source, but not open contribution**.

- **Bug reports and feature requests** are welcome as GitHub Issues. They are
  handled at the maintainer's pace and without any promise of a fix.
- **Comments and requests without a GitHub account** are welcome through the
  feedback form ([English](https://docs.google.com/forms/d/e/1FAIpQLSc-tnfYzsqcctESephPCJqbht_TQRl6TPw8NE7sgkmlcAZ68w/viewform),
  [Japanese](https://docs.google.com/forms/d/e/1FAIpQLSe-zXza-DXr8eGdOIgLKBH98luuLQdCooD6fcjCO3fA3gvuKw/viewform)).
  Responses are not public. The form and the issues are also linked from
  "About" in the app menu (top right), with your version filled in.
- **Unsolicited pull requests are not accepted** and are closed automatically.
  The maintainer implements changes so that the app and the generated scripts
  stay in lockstep (`climcanvas/core/render.py` and
  `climcanvas/core/scriptgen.py` are tested to produce pixel-identical images)
  and so that the copyright stays single-authored.
- If you would like to contribute code, **open an issue first** and describe
  the change. The maintainer may then ask for a pull request.

## Reporting a bug

Please include, as far as possible:

1. ClimCanvas version (shown under the app title, or `climcanvas/__init__.py`) and the versions of Python,
   matplotlib, cartopy and PROJ (`python -c "import pyproj; print(pyproj.proj_version_str)"`)
2. The header of the netCDF file (`ncdump -h file.nc`); the data itself is
   usually not needed
3. The session JSON exported from the app, or the settings you used
4. What you expected and what happened (a screenshot or the error message)
