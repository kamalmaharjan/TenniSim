"""WSGI entrypoint for production hosts (e.g. PythonAnywhere).

PythonAnywhere expects a module-level variable named `application` that is a
callable WSGI app.
"""

from __future__ import annotations

import os
import sys


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Flask app instance (callable)
from src.frontend.flask import app as application  # noqa: E402
