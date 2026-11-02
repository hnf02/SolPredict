from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
DATA_FILE = config.DATA_DIR / "ibri_hourly.csv.gz"
MANIFEST = config.DATA_DIR / "manifest.json"


def read_power_csv(text: str) -> tuple[pd.DataFrame, str]:
    lines = text.splitlines()
    standard = "unknown"
    start = 0
    for i, line in enumerate(lines):
        if " in LST" in line:
            standard = "LST"
        elif " in UTC" in line:
            standard = "UTC"
        if line.strip() == "-END HEADER-":
            start = i + 1
            break
    for i in range(start, len(lines)):           
        if lines[i].startswith("YEAR"):
            start = i
            break
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    time = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"]))
    df = df.drop(columns=["YEAR", "MO", "DY", "HR"]).set_index(time)
    df.index.name = "time"
    df = df.replace(config.FILL_VALUE, np.nan).astype("float64")
    return df, standard


def read_file(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    opener = __import__("gzip").open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return read_power_csv(fh.read())


def year_url(year: int) -> str:
    q = (f"parameters={','.join(config.PARAMETERS)}&community=RE"
         f"&longitude={config.SITE['longitude']}&latitude={config.SITE['latitude']}"
         f"&start={year}0101&end={year}1231&format=CSV")
    return f"{API}?{q}"


def full_url() -> str:
    q = (f"parameters={','.join(config.PARAMETERS)}&community=RE"
         f"&longitude={config.SITE['longitude']}&latitude={config.SITE['latitude']}"
         f"&start={config.START_YEAR}0101&end={config.END_YEAR}1231&format=CSV")
    return f"{API}?{q}"


def download(timeout: int = 300) -> pd.DataFrame:
    parts = []
    for year in range(config.START_YEAR, config.END_YEAR + 1):
        print(f"  downloading {year} ...", flush=True)
        with urllib.request.urlopen(year_url(year), timeout=timeout) as r:
            df, standard = read_power_csv(r.read().decode("utf-8"))
        parts.append(df)
    return pd.concat(parts).sort_index()


def import_files(paths: list[str | Path]) -> pd.DataFrame:
    parts, standards = [], set()
    for p in paths:
        df, standard = read_file(p)
        parts.append(df)
        standards.add(standard)
    if len(standards - {"unknown"}) > 1:
        raise ValueError(f"files use different time standards: {standards}")
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="first")]


def save(df: pd.DataFrame, source_url: str | None = None, downloaded: str | None = None) -> dict:
    config.DATA_DIR.mkdir(exist_ok=True)
    out = df.reset_index()
    csv = out.to_csv(index=False, float_format="%.4f").encode("utf-8")
    import gzip
    with gzip.GzipFile(DATA_FILE, "wb", mtime=0) as fh:     # mtime=0: same bytes every time
        fh.write(csv)
    manifest = {
        "source": "NASA POWER hourly API (community RE), time standard LST",
        "source_url": source_url or full_url(),
        "downloaded": downloaded or str(pd.Timestamp.now().date()),
        "site": config.SITE,
        "parameters": config.PARAMETERS,
        "first_hour": str(df.index.min()), "last_hour": str(df.index.max()),
        "rows": int(len(df)),
        "sha256_uncompressed_csv": hashlib.sha256(csv).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run `solpredict import FILE.csv` "
                                "or `solpredict download` first.")
    df = pd.read_csv(DATA_FILE, parse_dates=["time"]).set_index("time")
    return df


def verify() -> list[str]:
    import gzip
    m = json.loads(MANIFEST.read_text())
    with gzip.open(DATA_FILE, "rb") as fh:
        raw = fh.read()
    problems = []
    if hashlib.sha256(raw).hexdigest() != m["sha256_uncompressed_csv"]:
        problems.append("checksum differs from data/manifest.json")
    rows = raw.count(b"\n") - 1
    if rows != m["rows"]:
        problems.append(f"{rows} rows, manifest says {m['rows']}")
    return problems


def check(df: pd.DataFrame) -> dict:
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    ghi, toa, clr = df["ALLSKY_SFC_SW_DWN"], df["TOA_SW_DWN"], df["CLRSKY_SFC_SW_DWN"]
    night = toa <= 0
    by_hour = ghi.groupby(ghi.index.hour).mean()
    return {
        "rows": int(len(df)),
        "expected_hours": int(len(full)),
        "missing_hours": int(len(full.difference(df.index))),
        "duplicate_hours": int(df.index.duplicated().sum()),
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "night_hours_with_sun_above_1_wm2": int((ghi[night] > 1).sum()),
        "hours_ghi_above_toa": int((ghi > toa + 1).sum()),
        "hours_ghi_above_clear_sky_by_50_wm2": int((ghi > clr + 50).sum()),
        "max_ghi_wm2": round(float(ghi.max()), 1),
        "hour_of_highest_mean_ghi": int(by_hour.idxmax()),
    }
