"""Pytest path setup — make both the D1 package (src/) and the flat D2/D3/D4
runtime (app/) importable regardless of where pytest is invoked from.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
