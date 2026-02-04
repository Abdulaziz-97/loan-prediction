"""
Load the full Lending Club dataset with chunked processing and memory optimization.
"""

import gc
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TEXT_COLS = ["desc", "emp_title", "title"]

DTYPE_OPTIMIZATIONS: Dict[str, str] = {
    "loan_amnt": "float32",
    "funded_amnt": "float32",
    "funded_amnt_inv": "float32",
    "installment": "float32",
    "annual_inc": "float32",
    "total_pymnt": "float32",
    "total_pymnt_inv": "float32",
    "total_rec_prncp": "float32",
    "total_rec_int": "float32",
    "total_rec_late_fee": "float32",
    "recoveries": "float32",
    "collection_recovery_fee": "float32",
    "last_pymnt_amnt": "float32",
    "out_prncp": "float32",
    "out_prncp_inv": "float32",
    "fico_range_low": "float32",
    "fico_range_high": "float32",
    "last_fico_range_low": "float32",
    "last_fico_range_high": "float32",
}


def get_usecols(csv_path: Path) -> List[str]:
    """Return columns to load, excluding TEXT_COLS."""
    all_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return [c for c in all_columns if c not in TEXT_COLS]


def optimize_memory(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Convert low-cardinality object columns to category to reduce memory."""
    start_mem = df.memory_usage(deep=True).sum() / (1024**3)

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() < 500:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / (1024**3)
    if verbose and start_mem > 0:
        reduction = (1 - end_mem / start_mem) * 100
        print(f"Memory: {start_mem:.2f} GB -> {end_mem:.2f} GB ({reduction:.1f}% reduction)")

    return df


def filter_footer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Lending Club CSV footer rows (non-numeric id or loan_amnt)."""
    if "id" in df.columns:
        df = df[pd.to_numeric(df["id"], errors="coerce").notna()].copy()
    elif "loan_amnt" in df.columns:
        df = df[pd.to_numeric(df["loan_amnt"], errors="coerce").notna()].copy()
    return df


def load_full_dataset_chunked(
    csv_path: Path,
    chunk_size: int = 200_000,
    usecols: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load full Lending Club CSV in chunks with dtype optimization and footer filtering."""
    if usecols is None:
        usecols = get_usecols(csv_path)

    if callable(usecols):
        dtypes = {k: v for k, v in DTYPE_OPTIMIZATIONS.items() if usecols(k)}
    else:
        dtypes = {k: v for k, v in DTYPE_OPTIMIZATIONS.items() if k in usecols}

    chunks = []
    total_rows = 0

    if verbose:
        print(f"Loading full dataset from {csv_path}")
        print(f"Chunk size: {chunk_size:,} rows")

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtypes,
        low_memory=False,
        chunksize=chunk_size,
        on_bad_lines="warn",
    )

    for i, chunk in enumerate(reader):
        chunk = filter_footer_rows(chunk)
        total_rows += len(chunk)
        chunk = optimize_memory(chunk, verbose=False)
        chunks.append(chunk)
        if verbose and (i + 1) % 5 == 0:
            print(f"  Loaded {total_rows:,} rows...")
        if (i + 1) % 10 == 0:
            gc.collect()

    if verbose:
        print(f"Concatenating {len(chunks)} chunks...")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df = optimize_memory(df, verbose=verbose)
    if verbose:
        mem_gb = df.memory_usage(deep=True).sum() / (1024**3)
        print(f"Total rows loaded: {len(df):,}")
        print(f"Final memory usage: {mem_gb:.2f} GB")

    return df


def load_full_dataset_simple(
    csv_path: Path,
    usecols: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load full dataset in one read; use when RAM is sufficient (>16GB recommended)."""
    if usecols is None:
        usecols = get_usecols(csv_path)

    dtypes = {k: v for k, v in DTYPE_OPTIMIZATIONS.items() if k in usecols}

    if verbose:
        print(f"Loading full dataset from {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtypes,
        low_memory=False,
        on_bad_lines="warn",
    )

    if verbose:
        print(f"Loaded {len(df):,} rows (before filtering)")

    df = filter_footer_rows(df)
    if verbose:
        print(f"After filtering: {len(df):,} rows")

    df = optimize_memory(df, verbose=verbose)
    return df


if __name__ == "__main__":
    _cwd = Path.cwd()
    if (_cwd / "data" / "lending-club").exists():
        _root = _cwd
    elif (_cwd.parent / "data" / "lending-club").exists():
        _root = _cwd.parent
    else:
        _root = _cwd

    ACCEPTED_CSV = _root / "data/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"

    if not ACCEPTED_CSV.exists():
        print(f"CSV not found at {ACCEPTED_CSV}")
    else:
        print("=" * 60)
        print("Full Dataset Loading Test")
        print("=" * 60)
        print("\nTesting with first 1000 rows...")
        df_test = pd.read_csv(ACCEPTED_CSV, nrows=1000)
        print(f"  Sample shape: {df_test.shape}")
        print(f"  Columns: {len(df_test.columns)}")
        print("\nTo load the full dataset:")
        print("  from full_data_loader import load_full_dataset_chunked")
        print("  df = load_full_dataset_chunked(ACCEPTED_CSV)")
