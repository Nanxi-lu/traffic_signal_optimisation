# Deployment

There are two ways to publish the results. Both are read-only and use only the committed
Phase 4G outputs.

| | GitHub Pages site | Streamlit app |
|---|---|---|
| Files | `docs/index.html`, `docs/assets/phase4g_results.json`, `docs/.nojekyll` | `streamlit_app.py` |
| Runs Python | No (static page) | Yes |
| Load time | Fast, no external libraries or fonts | Slower first load while the app wakes |
| Hosting | GitHub Pages | Streamlit Community Cloud |

## GitHub Pages

1. In the repository on GitHub, open **Settings, Pages**.
2. Under **Build and deployment**, set Source to **Deploy from a branch**, Branch to
   **main** and folder to **/docs**, then **Save**.
3. Push to `main`. GitHub builds the site in about a minute; progress shows under the
   repository's **Actions** tab ("pages build and deployment").
4. Open `https://nanxi-lu.github.io/traffic_signal_optimisation/` and check the PM peak
   eastbound fairness warning appears and that the selectors work, on a phone as well.

`docs/.nojekyll` turns off Jekyll, so GitHub serves `index.html` exactly as committed and
does not turn the Markdown files in `docs/` into web pages.

If the saved CSVs ever change, regenerate the site data before pushing:

```bash
python scripts/export_site_data.py
python -m pytest
```

A test fails if `docs/assets/phase4g_results.json` is out of date.

To preview locally, run `python -m http.server --directory docs` and open
`http://localhost:8000`. Opening `index.html` directly from disk will not load the data file.

# Streamlit Community Cloud

The dashboard (`streamlit_app.py`) is a read-only explorer of the committed Phase 4G
results. It needs no database, credentials or secrets, and it never runs the simulation.

## Pre-deployment checklist

| Check | Status |
|---|---|
| Entry point is `streamlit_app.py` at the repository root | Yes |
| `requirements.txt` lists only runtime packages (numpy, pandas, plotly, streamlit) | Yes |
| Development packages (pytest, matplotlib) are in `requirements-dev.txt` | Yes |
| Data paths are built from `Path(__file__)`, so no absolute local paths are used | Yes (tested) |
| Required data files are version-controlled | `outputs/data/phase4g_network_metrics.csv`, `phase4g_movement_metrics.csv`, `phase4g_signal_plans.csv`, `phase4g_network_summary.csv`, `phase4g_movement_summary.csv` |
| Corridor image is version-controlled | `figures/corridor_network.png` (must be committed with this change) |
| Missing or inconsistent files produce a readable error | Yes (`load_phase4g_data` raises a message naming the missing file and the script that creates it) |
| Secrets, tokens or credentials in the repository | None found |
| Theme and server settings | `.streamlit/config.toml` |

The figures, docs and `.streamlit/config.toml` are new files. They must be committed
and pushed before deployment, otherwise the cloud app will show a "Corridor diagram not
found" warning in place of the diagram.

## Test locally first

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
streamlit run streamlit_app.py
```

Check the default view (PM peak, Phase 3A benchmark, Integrated candidate). It should
show a red fairness warning for EB with mean +2.906 s per vehicle.

## Deploy (requires your GitHub and Streamlit accounts)

1. Commit and push the changes to the `main` branch of
   `https://github.com/Nanxi-lu/traffic_signal_optimisation`.
2. Go to <https://share.streamlit.io> and sign in with the GitHub account that owns the
   repository. Authorise Streamlit to read the repository if prompted.
3. Choose **Create app**, then deploy from an existing GitHub repository.
4. Enter:
   - Repository: `Nanxi-lu/traffic_signal_optimisation`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: choose a subdomain if offered (for example something based on the project name).
5. Open **Advanced settings** and choose Python 3.12 (any version from 3.10 works). No
   secrets are needed.
6. Select **Deploy** and wait for the build log to finish installing `requirements.txt`.
7. When the app opens, check the four selectors, all five tabs and the PM eastbound
   fairness warning, on a desktop and on a phone.
8. Copy the public URL from the browser address bar. Only after confirming it loads in a
   private browser window, add it to the README (section C) and the LinkedIn draft.

## Updating

Every push to `main` triggers a redeploy. If you rerun any experiment and change the
saved CSVs, rerun `python -m pytest` first: the regression tests will fail if headline
values change, which is intended.

## Suggested GitHub repository metadata (not applied)

These can be set from the repository's **About** panel on GitHub.

- **Description:** Python traffic signal optimisation using Webster timing, green-wave
  coordination, queue simulation and fairness-constrained network evaluation.
- **Website:** the Streamlit URL, once deployed and verified.
- **Topics:** `traffic-engineering`, `traffic-signal-optimisation`,
  `transport-analytics`, `python`, `intelligent-transport-systems`, `simulation`,
  `urban-mobility`, `streamlit`.
- **Social preview image:** `figures/network_performance.png` or
  `figures/fairness_tradeoff.png` (Settings, General, Social preview).

## Housekeeping (optional, not applied)

`.DS_Store` is tracked in git even though it is listed in `.gitignore`. To remove it from
the repository without deleting your local copy:

```bash
git rm --cached .DS_Store
```
