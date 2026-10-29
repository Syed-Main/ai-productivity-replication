"""Block 1 – World Bank REST API (api.worldbank.org/v2)."""
import pandas as pd
from config import WB, START, END
from utils import cached_get


def countries():
    """ISO-3 codes of real countries (regional aggregates dropped)."""
    _, data = cached_get(
        f"{WB}/country", {"format": "json", "per_page": 400}, "wb_countries.json"
    )
    return {d["id"] for d in data if d["region"]["value"] != "Aggregates"}


def indicator(code, keep):
    """One indicator as a tidy frame: country (ISO-3), year, value."""
    _, data = cached_get(
        f"{WB}/country/all/indicator/{code}",
        {"format": "json", "per_page": 20000, "date": f"{START}:{END}"},
        f"wb_{code}.json",
    )
    rows = [
        (d["countryiso3code"], int(d["date"]), d["value"])
        for d in (data or [])
        if d["value"] is not None
    ]
    df = pd.DataFrame(rows, columns=["country", "year", "value"])
    return df[df.country.isin(keep)]


def first_available(codes, keep):
    """Try a code, or a tuple of alternative codes, until one returns data."""
    codes = (codes,) if isinstance(codes, str) else codes
    for c in codes:
        try:
            df = indicator(c, keep)
            if len(df):
                return df
        except Exception as exc:
            print(f"  ↳ {c} failed: {exc}")
    raise RuntimeError(f"none of {codes} could be downloaded")
