"""
utils.py
--------
Shared helpers for the craniofacial missing-data assessment.

Responsibilities
  * load the raw workbook and coerce it into a tidy frame
  * parse the free-text `age` column into numeric years
  * decompose each measurement name into its landmarks
  * classify each measurement by type (linear distance vs angle) and by
    anatomical region, and map bilateral (left/right) homologues


from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

for _p in (DATA_PROC, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20250809

ID_COL = "Patient ID"
META_COLS = [ID_COL, "gender", "age"]

PLANE_TOKENS = {"CP", "SP"}

# Age parsing
_RE_PURE_Y = re.compile(r"^\s*([\d.]+)\s*y\s*$", re.I)
_RE_PURE_M = re.compile(r"^\s*([\d.]+)\s*m\s*$", re.I)
_RE_PARTS = re.compile(r"([\d.]+)\s*([ymd])", re.I)

_UNIT_YEARS = {"y": 1.0, "m": 1.0 / 12.0, "d": 1.0 / 365.25}


def parse_age_to_years(raw) -> float:
    """Convert the free-text age field to decimal years.

    Handles the four formats present in the source file:
      '8.3m'      -> 0.692    (decimal months)
      '1.75y'     -> 1.75     (decimal years)
      '4y 11m'    -> 4.917    (compound)
      '17m 13d'   -> 1.452    (compound with days)
    Returns NaN for anything unparseable rather than guessing.
    """
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()

    m = _RE_PURE_Y.match(s)
    if m:
        return float(m.group(1))
    m = _RE_PURE_M.match(s)
    if m:
        return float(m.group(1)) / 12.0

    parts = _RE_PARTS.findall(s)
    if not parts:
        return np.nan
    return float(sum(float(v) * _UNIT_YEARS[u.lower()] for v, u in parts))


# Loading
def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the supplied workbook and add derived metadata columns."""
    path = path or (DATA_RAW / "Craniofacial_Data.xlsx")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    derived = pd.DataFrame({
        "age_years": df["age"].map(parse_age_to_years),
        "sex_M": (df["gender"].astype(str).str.upper().str[0] == "M").astype(int),
    })
    return pd.concat([df, derived], axis=1).copy()


def measurement_columns(df: pd.DataFrame) -> list[str]:
    """All numeric craniofacial measurements (excludes ID / demographics)."""
    drop = set(META_COLS) | {"age_years", "sex_M"}
    return [c for c in df.columns if c not in drop]


# Variable taxonomy
def tokens(var: str) -> list[str]:
    """Split a measurement name into its constituent tokens."""
    return [t for t in str(var).split("-") if t]


def landmarks(var: str) -> list[str]:
    """Anatomical landmarks referenced by a measurement (planes removed).

    `SNA` is the classical S-N-A cephalometric angle and is expanded manually
    because it is written as a single token.
    """
    if var.upper() == "SNA":
        return ["S", "N", "A"]
    return [t for t in tokens(var) if t not in PLANE_TOKENS]


def measure_type(var: str) -> str:
    """'angle' for 3-landmark measurements, 'distance' otherwise."""
    if var.upper() == "SNA":
        return "angle"
    return "angle" if len(tokens(var)) == 3 else "distance"


# Landmark -> anatomical region.  Grouping follows standard craniofacial
# subdivisions so that error can be reported per region .
_REGION_MAP: dict[str, str] = {}


def _register(region: str, *lms: str) -> None:
    for lm in lms:
        _REGION_MAP[lm] = region


_register("cranial_base", "S", "N", "BA", "SO", "ES", "LW", "LWR", "Ro", "SNM", "SNMR")
_register("nasomaxillary", "ANS", "PNS", "A", "SMP", "ALL", "ALR", "MO", "MOR",
          "JL", "JR")
_register("palatal_pterygoid", "PP", "PPR", "LPP", "LPPR", "MPP", "MPPR")
_register("orbital", "LOrL", "LOrR", "Or", "OrR", "SMF", "SMFR")
_register("zygomatic", "ZF", "ZFR", "ZO", "ZOR", "ZX", "ZXR", "ZMs", "ZMsR",
          "ZMi", "ZMiR", "ZTi", "ZTiR", "ZTs", "ZTsR")
_register("temporomandibular", "TM", "TMR", "Po", "PoR", "Mas", "MasR", "TP", "TPR")


def region(var: str) -> str:
    """Anatomical region of a measurement.

    A measurement spanning several regions is assigned to the region of its
    first non-cranial-base landmark, since cranial-base points (S, N, BA)
    serve as universal references and would otherwise absorb everything.
    """
    lms = landmarks(var)
    regs = [_REGION_MAP.get(lm) for lm in lms]
    regs = [r for r in regs if r]
    if not regs:
        return "unclassified"
    non_base = [r for r in regs if r != "cranial_base"]
    return non_base[0] if non_base else "cranial_base"


# Right-side landmarks end in 'R' (or are the explicit 'R' variants).
_RIGHT_SUFFIX_PAIRS = {
    "TMR": "TM", "MOR": "MO", "ZMsR": "ZMs", "ZMiR": "ZMi", "ZFR": "ZF",
    "ZOR": "ZO", "ZXR": "ZX", "ZTiR": "ZTi", "ZTsR": "ZTs", "SMFR": "SMF",
    "MasR": "Mas", "PoR": "Po", "TPR": "TP", "PPR": "PP", "LPPR": "LPP",
    "MPPR": "MPP", "LWR": "LW", "SNMR": "SNM", "OrR": "Or", "LOrR": "LOrL",
    "ALR": "ALL", "JR": "JL",
}
_LEFT_TO_RIGHT = {v: k for k, v in _RIGHT_SUFFIX_PAIRS.items()}


def side(var: str) -> str:
    """'L', 'R', 'midline' or 'bilateral' (spans both sides, e.g. ZF-ZFR)."""
    lms = landmarks(var)
    has_r = any(lm in _RIGHT_SUFFIX_PAIRS for lm in lms)
    has_l = any(lm in _LEFT_TO_RIGHT for lm in lms)
    if has_r and has_l:
        return "bilateral"
    if has_r:
        return "R"
    if has_l:
        return "L"
    return "midline"


def mirror_name(var: str) -> str | None:
    """Name of the contralateral homologue, or None if it has no reflection.

    `ZMi-CP` -> `ZMiR-CP`;  `TMR-MOR` -> `TM-MO`;  `S-BA` -> None (midline).
    The mirrored name is only meaningful if it actually exists in the dataset,
    which the caller must check.
    """
    s = side(var)
    if s not in ("L", "R"):
        return None
    out = []
    for t in tokens(var):
        if t in PLANE_TOKENS:
            out.append(t)
        elif s == "L":
            out.append(_LEFT_TO_RIGHT.get(t, t))
        else:
            out.append(_RIGHT_SUFFIX_PAIRS.get(t, t))
    return "-".join(out)


def variable_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per measurement describing its geometry and anatomy."""
    cols = measurement_columns(df)
    meta = pd.DataFrame({"variable": cols})
    meta["type"] = meta["variable"].map(measure_type)
    meta["region"] = meta["variable"].map(region)
    meta["side"] = meta["variable"].map(side)
    meta["landmarks"] = meta["variable"].map(lambda v: "|".join(landmarks(v)))
    mirror = meta["variable"].map(mirror_name)
    meta["mirror"] = [m if (m in set(cols)) else None for m in mirror]
    meta["n_missing"] = [df[c].isna().sum() for c in cols]
    meta["pct_missing"] = (meta["n_missing"] / len(df) * 100).round(2)
    return meta


def landmark_missing_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Subject x landmark table of *inferred* landmark availability.

    A landmark is called missing for a subject when **every** measurement that
    references it is missing for that subject.  This is the mechanism-level
    view: the raw data are missing cell-by-cell, but the cells are generated
    by whole landmarks failing to be digitised.
    """
    cols = measurement_columns(df)
    lm_to_cols: dict[str, list[str]] = {}
    for c in cols:
        for lm in landmarks(c):
            lm_to_cols.setdefault(lm, []).append(c)

    # NOTE: build from .to_numpy() so that pandas does not try to align the
    # component Series' positional index against the Patient-ID index.
    out = {lm: df[cs].isna().all(axis=1).to_numpy()
           for lm, cs in sorted(lm_to_cols.items())}
    mat = pd.DataFrame(out)
    mat.index = pd.Index(df[ID_COL].to_numpy(), name=ID_COL)
    return mat
