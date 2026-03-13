# Concept
This is a concept to optimize how to serve in Tennis.

## Local run

```bash
poetry install
poetry run flask --app src.frontend.flask run --debug --port 5055
```

- UI: http://127.0.0.1:5055/
- API: http://127.0.0.1:5055/api/optimize

## Deploy

This repo supports:
- Static frontend on GitHub Pages (root `index.html`)
- JSON API backend on PythonAnywhere (`/api/optimize`)

### 1) GitHub Pages (project site)

1. Push this repo to GitHub (e.g. `kamalmaharjan/TenniSim`).
2. In GitHub: Settings → Pages.
3. Build and deployment:
   - Source: Deploy from a branch
   - Branch: `main` (or whatever you use)
   - Folder: `/ (root)`

After it publishes, your UI should be available at:

`https://kamalmaharjan.github.io/TenniSim/`

The frontend calls the backend at:

`https://kamalmaharjan.pythonanywhere.com/api/optimize`

If you change your PythonAnywhere username/domain, update `API_BASE` in `index.html`.

### 2) PythonAnywhere (backend API)

1. Create a PythonAnywhere account: `kamalmaharjan`.
2. Upload this repo to PythonAnywhere (Files tab) to:

`/home/kamalmaharjan/TenniSim`

3. Open a Bash console on PythonAnywhere and set up a virtualenv:

```bash
cd ~/TenniSim
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Web tab → Add a new web app → Manual configuration (Python 3.11).
5. In the WSGI config file, point the WSGI app to Flask:

```python
import sys

project_dir = '/home/kamalmaharjan/TenniSim'
if project_dir not in sys.path:
	sys.path.insert(0, project_dir)

from src.frontend.flask import app as application
```

6. Reload the web app.

You should now have:
- API: `https://kamalmaharjan.pythonanywhere.com/api/optimize`

Note: the backend sends permissive CORS headers so GitHub Pages can call it.

### 3) Make `kamalmaharjan.github.io` show TenniSim by default

GitHub only serves the root domain (`https://kamalmaharjan.github.io/`) from a user site repo named `kamalmaharjan.github.io`.

If you want the root to show TenniSim, create a repo named `kamalmaharjan.github.io` containing an `index.html` redirect like:

```html
<!doctype html>
<meta charset="utf-8" />
<meta http-equiv="refresh" content="0; url=/TenniSim/" />
<title>Redirecting…</title>
<a href="/TenniSim/">Go to TenniSim</a>
```

Then enable Pages for that repo (root). The TenniSim project site remains at `/TenniSim/`.
