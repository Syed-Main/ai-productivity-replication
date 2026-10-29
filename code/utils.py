"""Block 0b – HTTP download cache, run-log tee, and panel helpers."""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import requests
from config import CACHE

# ── HTTP cache ──────────────────────────────────────────────────────
def cached_get(url, params, cache_name, as_json=True):
    """GET a URL once and reuse the saved response on later runs."""
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    else:
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        txt = r.text
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
    return json.loads(txt) if as_json else txt


def cached_download(url, cache_name):
    """Download a binary file (e.g. PWT .dta) and cache it."""
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        return path
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"  -> cached {len(r.content) / 1e6:.1f} MB -> {path}")
    return path


# ── Console / file tee ──────────────────────────────────────────────
class Tee:
    """Send everything printed to screen *and* to a text file."""
    def __init__(self, path):
        self.f   = open(path, "w", encoding="utf-8", errors="replace")
        self.out = sys.stdout
    def write(self, s):
        try:
            self.out.write(s)
        except UnicodeEncodeError:
            enc = getattr(self.out, "encoding", None) or "ascii"
            self.out.write(s.encode(enc, errors="replace").decode(enc))
        try:
            self.f.write(s)
        except Exception:
            self.f.write(s.encode("utf-8", errors="replace").decode("utf-8"))
    def flush(self):
        try:
            self.out.flush()
        except Exception:
            pass
        try:
            self.f.flush()
        except Exception:
            pass


# ── Panel utilities ─────────────────────────────────────────────────
def entity_demean(df, cols, entity_col="eid"):
    """Within-group (entity) demeaning for FE-IV estimation."""
    means = df.groupby(entity_col)[cols].transform("mean")
    return df[cols] - means


def add_year_dummies(df, year_col="year", drop_first=True):
    """Return a DataFrame of year indicator columns."""
    dums = pd.get_dummies(df[year_col], prefix="yr", dtype=float)
    if drop_first:
        dums = dums.iloc[:, 1:]
    return dums


def safe_log(s, floor=1.0):
    """ln(max(x, floor)) to avoid log(0) or log(negative)."""
    return np.log(np.maximum(s, floor))


def winsorise(s, lo=0.01, hi=0.99):
    """Winsorise a Series at the given quantiles."""
    bounds = s.quantile([lo, hi])
    return s.clip(bounds.iloc[0], bounds.iloc[1])


def pretty(df, title="", fmt="{:.4f}"):
    """Print a compact summary table to the console log."""
    if title:
        print(f"\n{'-'*60}\n{title}\n{'-'*60}")
    with pd.option_context("display.float_format", fmt.format):
        print(df)

