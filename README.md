# Loan Prediction: Leakage Detection and Prevention

This project uses LendingClub loan data and focuses on avoiding different forms of data leakage that can make model performance appear better than it really is. The notebook uses experiments to show why careful data processing is important.

## Types of Leakage Handled

1. Data-Induced Leakage: When label definitions let the model "cheat" by making predictions obvious, for example, using the "Current" status that is strongly linked to recent loan issue dates. To avoid this, the project uses only finalized outcomes, Paid vs Default, and leaves out "Current" loans.

2. Label Leakage: Some columns directly relate to the loan outcome, like payment totals, last FICO scores, settlement status, or hardship flags, which would let the model predict the target directly. The project removes 43 identified columns after the loan is issued, using the LeakageSpec definition, plus columns like desc and title that may have written hints from borrowers.

3. Preprocessing Leakage: If imputers, scalers, or encoders are applied to the whole dataset before the train/test split, the test set can leak information into training statistics. Here, all statistics for imputation or scaling are calculated only on the training data, inside the pipeline.

4. Split-Related Leakage: Randomly splitting time-based data can put future data in the train set, letting the model learn from information it shouldn't have. The project uses an 80/20 time cutoff, always training on past data and testing on future data.

## Key Findings

Leakage can make model metrics look much better than they really are. Here are some main results from the experiments:

- Clean Model with Temporal Split (production-ready):
  - Accuracy: 61.4%
  - ROC-AUC: 0.727
  - Recall (Default): 75.2%
  - Log Loss: 0.646
  - This is the real, honest performance that should be expected for deployment.

- Clean Model with Random Split (still some temporal leakage):
  - Accuracy: 66.6%
  - ROC-AUC: 0.739
  - Recall (Default): 68.8%
  - Log Loss: 0.599
  - Accuracy here is inflated by around 5% because random splitting leaks future info.

- Leaky Model with Temporal Split (includes label leakage):
  - Accuracy: about 99.98%
  - ROC-AUC: about 0.9999
  - Log Loss: about 0.0014
  - Including label leakage features creates an almost perfect model, but this is entirely unrealistic for real-world use.

### Other pre-processing

- Data includes about 2.26 million loans between 2014 and 2018.
- Only finalized outcomes (Paid or Default) are included.
- Columns with 95% or more missing values are dropped.
- Feature engineering adds indicators like has_joint_app.
- CatBoost handles the remaining missing values.

## Steps Taken to Prevent Leakage

1. Used a time-based 80/20 data split based on the issue date
2. Used statistical tests to check for leaked features
3. Defined exactly which columns and prefixes to drop using the LeakageSpec class
4. Made sure preprocessing (like imputing/scaling) is done only with training data statistics
5. Dropped high-cardinality text columns that could allow unintended information leaks

Key Finding: Random split inflates accuracy by ~5% and improves log loss, but this is misleading as it violates temporal ordering.

#### Leaky Model with Temporal Split (Contains Label Leakage)
- Accuracy ~99.98%
- ROC-AUC ~0.9999
- Log Loss ~0.0014

Critical Finding When label leakage features are included, the model achieves near-perfect performance, which is completely unrealistic and would fail in production. This demonstrates the severe impact of leakage.

## Implementation Details

- Data is loaded in chunks for memory efficiency
- Features are automatically treated as numeric or categorical
- CatBoost handles missing values and the pipeline also removes excessively missing columns
- Training is GPU-accelerated
- Model evaluation uses ROC-AUC, PR-AUC, Brier score, and recall at fixed false positive rates

## Lessons

1. Leakage detection changes everything: there is a huge difference between a leaky model (99.98% accuracy) and a clean model (61.4% accuracy).
2. Time-based splits give more reliable estimates than random splits for temporal data.
3. Honest, production-ready performance will be lower than what a leaky model shows, but is much more trustworthy.


## Summary

Preventing data leakage is necessary for building reliable machine learning models for loan prediction. The results from this project show that models that leak information perform unrealistically well, but will not hold up in real-world situations. Careful data handling and honest validation give performance numbers that can be trusted for deployment.
