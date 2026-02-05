# LendingClub Data Quality + Cleaning Report (loan_status binary)
- Source: `/home/wakeb/Wakeb-Projects-/project_1/data/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`
- Rows after filtering to final outcomes: **1,348,099**

## Target mapping
- Good (0): ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
- Bad (1): ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
- Excluded: ['Current', 'Issued', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']

### Counts
- Good: **1,078,739**
- Bad: **269,360**
- Dropped (missing target): 33
- Dropped (excluded statuses): 912,569
- Dropped (unmapped statuses): 0

## Leakage control
- Dropped leakage columns: **43**
  - Examples: total_rec_int, hardship_end_date, deferral_term, total_pymnt_inv, orig_projected_additional_accrued_interest, member_id, hardship_loan_status, out_prncp, hardship_length, hardship_start_date, last_pymnt_d, debt_settlement_flag, hardship_type, debt_settlement_flag_date, last_fico_range_low, total_rec_late_fee, recoveries, hardship_amount, next_pymnt_d, out_prncp_inv, last_fico_range_high, settlement_term, url, hardship_reason, hardship_status, hardship_dpd, policy_code, pymnt_plan, last_pymnt_amnt, total_pymnt
- Dropped label column from features: ['target_bad']

## Identifier + key-field sanity
- ID duplicate summary: `{'id_col_present': True, 'rows': 2260701, 'missing_id': 0, 'unique_id': 2260701, 'duplicate_id_rows_est': 0}`
- Rows missing any key fields ['loan_amnt', 'term', 'int_rate', 'installment', 'issue_d', 'loan_status']: **33**

## Basic cleaning actions
- Replaced sentinel -1 counts: {'dti': 2}
- Parsed: `term` -> `term_months`, `emp_length` -> `emp_length_years`, month-year dates, and engineered `credit_age_months`.

## Missingness (top 30)

|                                     |          missing |   missing_pct |
|:------------------------------------|-----------------:|--------------:|
| sec_app_earliest_cr_line            |      1.3481e+06  |      100      |
| sec_app_mths_since_last_major_derog |      1.34145e+06 |       99.5068 |
| sec_app_revol_util                  |      1.32979e+06 |       98.6419 |
| revol_bal_joint                     |      1.32946e+06 |       98.6177 |
| sec_app_fico_range_low              |      1.32946e+06 |       98.6176 |
| sec_app_collections_12_mths_ex_med  |      1.32946e+06 |       98.6176 |
| sec_app_chargeoff_within_12_mths    |      1.32946e+06 |       98.6176 |
| sec_app_num_rev_accts               |      1.32946e+06 |       98.6176 |
| sec_app_open_act_il                 |      1.32946e+06 |       98.6176 |
| sec_app_open_acc                    |      1.32946e+06 |       98.6176 |
| sec_app_mort_acc                    |      1.32946e+06 |       98.6176 |
| sec_app_inq_last_6mths              |      1.32946e+06 |       98.6176 |
| sec_app_fico_range_high             |      1.32946e+06 |       98.6176 |
| verification_status_joint           |      1.3225e+06  |       98.101  |
| dti_joint                           |      1.3223e+06  |       98.086  |
| annual_inc_joint                    |      1.32229e+06 |       98.0857 |
| mths_since_last_record              |      1.11868e+06 |       82.9817 |
| mths_since_recent_bc_dlq            |      1.02907e+06 |       76.3349 |
| mths_since_last_major_derog         | 994342           |       73.7588 |
| mths_since_recent_revol_delinq      | 898123           |       66.6214 |
| il_util                             | 883104           |       65.5074 |
| mths_since_rcnt_il                  | 824680           |       61.1735 |
| all_util                            | 810516           |       60.1229 |
| open_acc_6m                         | 810464           |       60.119  |
| total_cu_tl                         | 810464           |       60.119  |
| inq_last_12m                        | 810464           |       60.119  |
| total_bal_il                        | 810463           |       60.119  |
| open_il_12m                         | 810463           |       60.119  |
| open_act_il                         | 810463           |       60.119  |
| open_rv_12m                         | 810463           |       60.119  |

## Missingness summary
- Columns >= 95% missing: 16
- Columns >= 80% missing: 17
- Columns >= 60% missing: 34

### Columns >= 95% missing
- sec_app_earliest_cr_line
- sec_app_mths_since_last_major_derog
- sec_app_revol_util
- revol_bal_joint
- sec_app_fico_range_low
- sec_app_collections_12_mths_ex_med
- sec_app_chargeoff_within_12_mths
- sec_app_num_rev_accts
- sec_app_open_act_il
- sec_app_open_acc
- sec_app_mort_acc
- sec_app_inq_last_6mths
- sec_app_fico_range_high
- verification_status_joint
- dti_joint
- annual_inc_joint

## Missingness by target (top 30 absolute differences)

| column                         |   missing_pct_1 |   missing_pct_0 |   abs_diff |
|:-------------------------------|-------------------:|------------------:|-----------:|
| all_util                       |           61.3015  |          55.4028  |    5.89868 |
| open_acc_6m                    |           61.2975  |          55.3995  |    5.89803 |
| inq_last_12m                   |           61.2975  |          55.3995  |    5.89803 |
| total_cu_tl                    |           61.2975  |          55.3995  |    5.89803 |
| open_act_il                    |           61.2974  |          55.3995  |    5.89794 |
| max_bal_bc                     |           61.2974  |          55.3995  |    5.89794 |
| open_il_12m                    |           61.2974  |          55.3995  |    5.89794 |
| open_il_24m                    |           61.2974  |          55.3995  |    5.89794 |
| total_bal_il                   |           61.2974  |          55.3995  |    5.89794 |
| open_rv_12m                    |           61.2974  |          55.3995  |    5.89794 |
| open_rv_24m                    |           61.2974  |          55.3995  |    5.89794 |
| inq_fi                         |           61.2974  |          55.3995  |    5.89794 |
| il_util                        |           66.6202  |          61.0506  |    5.56955 |
| mths_since_rcnt_il             |           62.2647  |          56.8035  |    5.4612  |
| mths_since_recent_inq          |           13.9199  |           9.89865 |    4.02122 |
| mths_since_last_major_derog    |           74.4022  |          71.1821  |    3.22018 |
| mths_since_last_record         |           83.5775  |          80.5955  |    2.98201 |
| emp_length_years               |            5.32103 |           7.85195 |    2.53092 |
| mths_since_last_delinq         |           50.8839  |          48.6542  |    2.22973 |
| mths_since_recent_bc_dlq       |           76.6322  |          75.1444  |    1.48774 |
| mths_since_recent_revol_delinq |           66.9133  |          65.4526  |    1.46076 |
| pct_tl_nvr_dlq                 |            5.49985 |           4.12125 |    1.3786  |
| num_rev_accts                  |            5.4877  |           4.11308 |    1.37462 |
| num_bc_tl                      |            5.48761 |           4.11308 |    1.37453 |
| num_tl_90g_dpd_24m             |            5.48761 |           4.11308 |    1.37453 |
| num_actv_rev_tl                |            5.48761 |           4.11308 |    1.37453 |
| num_il_tl                      |            5.48761 |           4.11308 |    1.37453 |
| num_rev_tl_bal_gt_0            |            5.48761 |           4.11308 |    1.37453 |
| num_actv_bc_tl                 |            5.48761 |           4.11308 |    1.37453 |
| tot_cur_bal                    |            5.48761 |           4.11308 |    1.37453 |

## Feature curation decisions
- Dropped constant columns(policy_code): 1
- Dropped >= 95% missing columns: 16