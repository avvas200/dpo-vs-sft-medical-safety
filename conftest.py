"""
Belt-and-braces sys.path setup for pytest.

pyproject.toml's [tool.pytest.ini_options] pythonpath already adds the repo
root, but this file ensures `from src import stats` works even if someone
runs pytest with an old version that doesn't honour that setting, or with a
config-override flag.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
