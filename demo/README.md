# Public demo (Streamlit Community Cloud)

`demo/app.py` is the entry point of the public demo: it puts the app into the
demo mode (only the files in `data/demo/` can be opened, no server-side saving,
a banner with a link to the installation page, capped output resolution and
animation length), generates the two demo files (`sample_atmos.nc` and
`sample_1d_time.nc`) and pre-fetches the Natural Earth shapefiles, then runs
the real `app.py`. It can also be started locally:

```bash
streamlit run demo/app.py
```

Deployment settings on Streamlit Community Cloud:

| Setting | Value |
|---|---|
| Repository / branch | `ClimCanvas/ClimCanvas`, `main` |
| Main file path | `demo/app.py` |
| Python version | 3.12 |
| Python dependencies | `demo/requirements.txt` (includes the root `requirements.txt`) |
| apt packages | root `packages.txt` (`fonts-noto-cjk` for CJK labels, `ffmpeg` for MP4) |

Notes:

- `requirements.txt` pins `cartopy<0.26` and `pyproj<3.8` (see its comments);
  the demo therefore runs cartopy 0.25 with PROJ 9.5.
- Community Cloud puts an app to sleep after 12 hours without traffic; the
  first visitor afterwards has to wake it up and wait for the restart.
- The demo keeps nothing on the server: figures, animations, scripts and
  sessions are downloaded to the visitor's PC.
