
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_TERM_RE = re.compile(r"^\s*(\d+)\s+months\s*$", re.IGNORECASE)


EMP_LENGTH_MAP: dict[str, int] = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


def parse_term(value: Any) -> int | None:
    """Parse `term` like '36 months' -> 36."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    m = _TERM_RE.match(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_emp_length(value: Any) -> int | None:
    """Parse `emp_length` like '10+ years' / '< 1 year' to an ordinal integer."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    return EMP_LENGTH_MAP.get(s)


def coerce_numeric_series(series, *, allow_percent: bool = True):
    """
    Convert a pandas Series to float, handling:
    - commas in numbers
    - trailing percent sign
    """
    s = series.astype("string")
    s = s.str.replace(",", "", regex=False)
    if allow_percent:
        s = s.str.replace("%", "", regex=False)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce")


def parse_month_year(series):
    """Parse LendingClub month-year fields like 'Dec-2015' into pandas datetime."""
    s = series.astype("string").str.strip()
    dt = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt2 = pd.to_datetime(s[missing], errors="coerce")
        dt.loc[missing] = dt2
    return dt


@dataclass
class CleaningLog:
    """Structured counters for a cleaning run."""

    rows_in: int = 0
    rows_out: int = 0
    rows_dropped_missing_target: int = 0
    rows_dropped_unmapped_target: int = 0
    rows_dropped_excluded_status: int = 0
    replaced_sentinel_minus_one: dict[str, int] | None = None


def basic_clean_inplace(df, *, cleaning_spec, target_spec) -> CleaningLog:
    """
    Minimal, deterministic cleaning that is safe before splitting:
    - normalize key messy fields (`term`, `emp_length`, percent columns)
    - parse dates needed for splitting and engineered features
    - replace sentinel values
    - cap extreme DTI/utilization values
    """
    log = CleaningLog(rows_in=len(df), replaced_sentinel_minus_one={})

    # Normalize whitespace in object columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("string").str.strip()

    # Parse term/emp_length if present
    if "term" in df.columns:
        df["term_months"] = df["term"].map(parse_term).astype("float64")
        df.drop(columns=["term"], inplace=True)

    if "emp_length" in df.columns:
        df["emp_length_years"] = df["emp_length"].map(parse_emp_length).astype("float64")
        df.drop(columns=["emp_length"], inplace=True)

    # Parse issue/credit-line dates if present
    if target_spec.issue_date_column in df.columns:
        df[target_spec.issue_date_column] = parse_month_year(df[target_spec.issue_date_column])

    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = parse_month_year(df["earliest_cr_line"])

    # Percent/utilization columns
    for col in cleaning_spec.utilization_columns:
        if col in df.columns:
            df[col] = coerce_numeric_series(df[col], allow_percent=True)
            df.loc[df[col] < 0, col] = np.nan
            df.loc[df[col] > cleaning_spec.max_reasonable_util_pct, col] = np.nan

    # Sentinel -1 columns (e.g. dti)
    for col in cleaning_spec.sentinel_minus_one_columns:
        if col in df.columns:
            df[col] = coerce_numeric_series(df[col], allow_percent=False)
            sentinel_mask = df[col] == -1
            log.replaced_sentinel_minus_one[col] = int(sentinel_mask.sum())
            df.loc[sentinel_mask, col] = np.nan
            df.loc[df[col] < 0, col] = np.nan
            df.loc[df[col] > cleaning_spec.max_reasonable_dti, col] = np.nan

    # Coerce obvious numeric columns broadly (except datetimes/strings)
    categorical_like = {
        "grade",
        "sub_grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "zip_code",
        "addr_state",
        "initial_list_status",
        "application_type",
        "verification_status_joint",
        "disbursement_method",
    }
    for col in df.columns:
        if col in categorical_like:
            continue
        if str(df[col].dtype).startswith("datetime"):
            continue
        if col.endswith("_d"):
            continue
        if df[col].dtype == object or str(df[col].dtype).startswith("string"):
            df[col] = coerce_numeric_series(df[col], allow_percent=True)

    # Simple engineered features
    if target_spec.issue_date_column in df.columns:
        df["issue_year"] = df[target_spec.issue_date_column].dt.year.astype("float64")
        df["issue_month"] = df[target_spec.issue_date_column].dt.month.astype("float64")

    if target_spec.issue_date_column in df.columns and "earliest_cr_line" in df.columns:
        months = (
            (df[target_spec.issue_date_column].dt.year - df["earliest_cr_line"].dt.year) * 12
            + (df[target_spec.issue_date_column].dt.month - df["earliest_cr_line"].dt.month)
        )
        months = months.where(months >= 0)
        df["credit_age_months"] = months.astype("float64")

    log.rows_out = len(df)
    return log
