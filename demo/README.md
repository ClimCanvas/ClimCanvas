# Public demo (Streamlit Community Cloud)

`demo/app.py` is the entry point of the public demo: it puts the app into the
demo mode (only the files in `data/demo/` can be opened, no server-side saving,
a banner with a link to the installation page, capped output resolution and
animation length), starts the interface in English (visitors can switch it in
the sidebar), generates the two demo files (`sample_atmos.nc` and
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
- Community Cloud puts an app to sleep after 12 hours without a visitor; the
  first visitor afterwards would have to wake it up and wait about two minutes
  for the restart. A scheduled GitHub Actions job visits the demo every 4 hours
  to prevent this (see below).
- The demo keeps nothing on the server: figures, animations, scripts and
  sessions are downloaded to the visitor's PC.
- Search engines and the share preview of Community Cloud use the first screen of
  the app (there is no setting for keywords or a description), so in the demo mode
  the line under the title says what the app does and, before a file is loaded,
  `demo/example_figure.png` (a copy of a figure from the gallery of the website:
  JRA-3Q 500 hPa height climatology, DJF / JJA) is shown as an example.

## Keeping the demo awake

`.github/workflows/keep-demo-awake.yml` runs `demo/keepalive.py` every 4 hours
(and on demand from the Actions tab). The script opens the demo in headless
Chromium (Playwright), presses the wake-up button if the sleep screen is shown,
waits until the app has rendered and stays for 20 seconds. A plain HTTP request
would not do: the root URL is answered by the Community Cloud shell with HTTP
200 whether the app is awake or asleep, and the app itself runs in an iframe at
`/~/+/` over a WebSocket (`/~/+/_stcore/health` answers `ok` while it runs).

- The job fails when the app does not render within 7 minutes, so a broken
  deployment shows up as a failed run (GitHub e-mails the person who last
  changed the workflow file).
- The browser cookies are kept between runs (`actions/cache`), so the visits
  count as a single viewer in the Community Cloud analytics.
- GitHub disables scheduled workflows in a public repository after 60 days
  without a push; re-enable the workflow from the Actions tab if that happens.
- To run it by hand: `python -m pip install playwright`,
  `python -m playwright install chromium`, `python demo/keepalive.py`.
