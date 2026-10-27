# CV Atlas MIL — Evaluation Record

All five experts were **retrained from the existing 1,300-image feature caches** in the CV Atlas MIL implementation, preserving the same image features and sample ordering as previous raw-JPEG extraction. Per-fold training was refit; no test labels entered atlas, MIL normalization, or morphology training.

## Random repeated 5-fold

- Seeds: 20261025, 20261026, 20261027
- Tree experts: 400 estimators each
- Original MIL and Regional Joint MIL: 35 epochs each
- Three repeats: 3, 5, 1 errors
- **Overall: 9/3,900 mistakes = 99.7692% mean accuracy**
- **8 missed defects, 1 false alarm**
- Images: 1,300 distinct photos, evaluated three times (not 3,900 independent samples)

## Acquisition-shift proxy

- Five bins defined from original-image corner brightness; held out one group at a time
- Tree and MIL experts were refitted within each group-specific training set
- **271/1,300 errors: 20 missed defects, 251 false alarms; 79.1538% accuracy**
- Groups are *not* camera ID, batch ID or actual lighting-session IDs

## Historic replay context

A previous after-the-fact weighted blend of separately recorded expert scores gave 8/3,900 errors on repeated random CV, and 263/1,300 errors on brightness-group holdout. Those older values are **not** the fresh performance measurements of this unified CV Atlas MIL implementation.

## Validity

This dataset has been used for repeated architecture and weight decisions. Thus both new runs are exploratory and subject to selection bias and acquisition confounding. A fully independent production-batch/camera holdout and, preferably, training-only nested hyperparameter selection are needed for new generalization claims.
