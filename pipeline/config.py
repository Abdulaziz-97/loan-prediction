from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TargetSpec:
    """Binary target definition from LendingClub `loan_status`."""

    target_column: str = "loan_status"
    issue_date_column: str = "issue_d"

    good_statuses: tuple[str, ...] = (
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    )
    bad_statuses: tuple[str, ...] = (
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    )
    exclude_statuses: tuple[str, ...] = (
        "Current",
        "Issued",
        "In Grace Period",
        "Late (16-30 days)",
        "Late (31-120 days)",
    )

    label_good: int = 0
    label_bad: int = 1
    label_name: str = "target_bad"


@dataclass(frozen=True)
class LeakageSpec:
    """Columns to exclude because they are post-origination (leakage)."""

    drop_exact: tuple[str, ...] = (
        "id",
        "member_id",
        "url",
        "policy_code",
        "last_credit_pull_d",
        "last_fico_range_low",
        "last_fico_range_high",
        "pymnt_plan",
        "hardship_flag",
        "deferral_term",
        "payment_plan_start_date",
        "orig_projected_additional_accrued_interest",
        "loan_status",
    )

    drop_prefixes: tuple[str, ...] = (
        "out_prncp",
        "total_pymnt",
        "total_rec_",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_",
        "next_pymnt_",
        "hardship_",
        "debt_settlement_",
        "settlement_",
    )


@dataclass(frozen=True)
class CleaningSpec:
    """Cleaning/feature engineering knobs."""

    drop_free_text: tuple[str, ...] = ("desc", "emp_title", "title")
    utilization_columns: tuple[str, ...] = ("revol_util", "bc_util", "il_util", "all_util")
    sentinel_minus_one_columns: tuple[str, ...] = ("dti",)
    max_reasonable_dti: float = 200.0
    max_reasonable_util_pct: float = 200.0
    winsor_p_low: float = 0.005
    winsor_p_high: float = 0.995


@dataclass(frozen=True)
class Paths:
    project_dir: Path
    accepted_csv: Path
    processed_dir: Path
    reports_dir: Path
    processed_dataset: Path
    data_quality_report: Path


def default_paths(project_dir: Path | None = None) -> Paths:
    proj = project_dir or Path(__file__).resolve().parents[1]
    accepted_csv = (
        proj
        / "data"
        / "lending-club"
        / "accepted_2007_to_2018q4.csv"
        / "accepted_2007_to_2018Q4.csv"
    )
    processed_dir = proj / "data" / "processed"
    reports_dir = proj / "reports"
    return Paths(
        project_dir=proj,
        accepted_csv=accepted_csv,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        processed_dataset=processed_dir / "accepted_loanstatus_binary.parquet",
        data_quality_report=reports_dir / "data_quality_report.md",
    )


def iter_leakage_columns(all_columns: Iterable[str], leakage: LeakageSpec) -> set[str]:
    cols = set()
    for c in all_columns:
        if c in leakage.drop_exact:
            cols.add(c)
            continue
        for p in leakage.drop_prefixes:
            if c.startswith(p):
                cols.add(c)
                break
    return cols

