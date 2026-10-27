"""Repository-relative locations; raw sources are set through environment variables (see README)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")
TABLES = os.path.join(ROOT, "tables")
for _d in (DATA, RES, FIGS, TABLES):
    os.makedirs(_d, exist_ok=True)


def source(var):
    """Path of a raw source file or folder, taken from an environment variable."""
    p = os.environ.get(var)
    if not p:
        raise SystemExit(f"set the environment variable {var} (see README.md, section Data)")
    return p
