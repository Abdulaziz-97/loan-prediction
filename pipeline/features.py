
from __future__ import annotations

from typing import Iterable
import pandas as pd
import numpy as np


def drop_columns(df, columns: Iterable[str]):
    cols = [c for c in columns if c in df.columns]
    if cols:
        df.drop(columns=cols, inplace=True)
    return cols


def build_leakage_free_frame(df, *, leakage_columns: set[str], cleaning_spec, target_spec):
    """
    Remove leakage and baseline-excluded columns, returning a new dataframe.
    Keeps `issue_d` only for splitting; engineered numeric fields should be used for modeling.
    """
    df2 = df.copy()

    dropped_leakage = drop_columns(df2, leakage_columns)
    dropped_label = drop_columns(df2, [target_spec.label_name])
    dropped_text = drop_columns(df2, cleaning_spec.drop_free_text)

    for col in list(df2.columns):
        if col == target_spec.issue_date_column:
            continue
        if str(df2[col].dtype).startswith("datetime"):
            df2.drop(columns=[col], inplace=True)

    return df2, {"dropped_leakage": dropped_leakage, "dropped_label": dropped_label, "dropped_text": dropped_text}


def infer_feature_types(df, *, date_columns: Iterable[str] = ()):
    """
    Infer numeric vs categorical columns after cleaning.
    Excludes provided date columns from both lists.
    """
    date_cols = set(date_columns)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in date_cols:
            continue
        s = df[col]
        
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            numeric_cols.append(col)
            continue
        
        try:
            sample = s.dropna().head(100)
            if len(sample) > 0:
                pd.to_numeric(sample, errors='raise')
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)

    return numeric_cols, categorical_cols

