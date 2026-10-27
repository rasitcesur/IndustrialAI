# Model artifacts

Training writes `models/cv_atlas_mil.joblib`. Pretrained weights and normal-reference descriptors are not shipped in the source repository.

For reproducibility, models must be trained with the package version and feature-extraction pipeline used for inference. Do not load untrusted `.joblib` files: joblib uses Python pickle.
