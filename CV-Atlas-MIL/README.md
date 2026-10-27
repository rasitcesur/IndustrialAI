# CV Atlas MIL

**Spatially co-located evidence fusion for grayscale casting-surface inspection**

CV Atlas MIL is a research implementation for image-level casting-surface quality classification. It combines deterministic visual features, a healthy-reference atlas, attention-based multiple-instance learning (MIL), and tree-based classification. Its central design principle is that **local intensity, shape, and reference-atlas deviations should be compared at the same image region before regional evidence is aggregated**.

The system is trained from **image-level labels only** (`defective` or `ok`). It does not require pixel masks to train, and its attention maps must not be interpreted as validated defect segmentation.

**Status:** research prototype, not a qualified industrial safety or inspection system. Results described here are exploratory and require independent camera/batch evaluation.

## Python package

The implementation is contained in the single `cv_atlas_mil/` Python package. Model artifacts use native `cv_atlas_mil` class paths. Artifacts serialized with older internal package paths require migration or retraining before loading.

## Architecture

The input is a **512×512 grayscale** image. Its evidence is processed by five complementary experts:

| Expert | Main representation | Contribution |
|---|---|---:|
| Fixed-feature Extra Trees | 128 pooled features from deterministic convolution and activation operators | 45% |
| Healthy-reference Atlas | 64 position-indexed regions with geometric and grayscale/contrast deviations | 10% |
| Atlas Attention-MIL | Per-region geometry/intensity + healthy-reference deviations | 5% |
| **Regional Joint MIL** | **48 co-located values per region**: geometry/intensity (14), high-resolution morphology (28), atlas (4), coordinates (2) | **5%** |
| High-Resolution Morphology Extra Trees | 16/32/64-pixel black-hat, top-hat, gradient, line and region statistics | 35% |

All expert outputs represent a score for class `defective` (class ID `0`), and the final score is:

\[
S = 0.45S_{ET}+0.10S_{Atlas}+0.05S_{MIL}+0.05S_{Regional}+0.35S_{Morph}.
\]

At the default threshold of `0.5`, an image is classified as defective if `S >= 0.5`; otherwise it is classified as OK (`1`). The coefficients are **fixed research hyperparameters selected after examining earlier out-of-fold results**. They are *not* an independently optimized or guaranteed optimal policy. The score is not guaranteed calibrated under distribution shift.

### What spatial co-location means

Each 512×512 image has 64 non-overlapping 64×64 analysis regions. In the Regional Joint MIL branch, the feature vector for region `r` is:

```text
region_r = [intensity_and_geometry_r (14),
            high_resolution_morphology_r (28),
            healthy_atlas_difference_r (4),
            normalized_grid_coordinates_r (2)]
           = 48 features
```

Region `r` refers to the **same location within the current image** across the three descriptor families. By contrast, the reference atlas compares region `r` against the same *pixel-grid* region in similar healthy reference images. **No geometric registration is performed across different photographs**; grid-coordinate correspondence is not guaranteed to represent the same physical surface under large camera-pose variation.

The MIL branch aggregates regional evidence using a learned regional representation, top-region scores and attention-weighted scores. It is trained **only with image-level labels**; all regions of a defective image are *not* labeled defective.

### Reference atlas and data leakage protection

Only healthy images from the current training partition enter the healthy atlas. Reference selection uses the 24 globally nearest healthy reference images, followed by comparisons of matched grid positions. Normalization profiles for the atlas, morphology and regional MIL are also fitted using **training-partition healthy images only**. During training, a healthy query is excluded from matching itself as a reference. Test images are never inserted into the reference gallery.

### Computation backends

The fixed 128-feature branch supports automatic dispatch:

`CUDA via PyTorch (multi-stream, parity check) -> Cython/OpenMP -> NumPy/OpenCV`

A supported NVIDIA GPU and CUDA-enabled PyTorch installation are necessary for GPU execution. The morphological and geometry extraction paths currently run through OpenCV/CPU. On startup, the CUDA path is rejected if its feature-equivalence test fails. The earlier OpenCV-CUDA fused convolution is **not** used because it was not numerically equivalent to the reference filter computation. No end-to-end CUDA speed advantage is claimed here.

## Dataset and attribution

**Source:** [Casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) by **Ravirajsinh Dabhi**, published on Kaggle.

The experiments in this repository use the **non-augmented `casting_512x512` subset**, comprising **1,300 grayscale images** of casting products (**781 `def_front` defective; 519 `ok_front` acceptable**). This subset is distinct from the larger, augmented 300×300 image collection also distributed on the dataset page. Labels are image-level; verified pixel-level defect masks are not provided for these experiments.

The dataset is **not redistributed with this repository**. Obtain it directly from the original Kaggle source and place the class folders according to [Dataset structure](#dataset-structure). **The source-code license does not license the dataset**: consult the terms and license provided by the dataset publisher before using or redistributing the images.

## Exploratory evaluation

The 1,300-image Kaggle subset described above (781 defective, 519 OK) was used in repeated 5-fold stratified cross-validation, with seeds 20261025, 20261026 and 20261027. There are **3,900 outer-fold predictions from 1,300 unique image IDs**; they are *not* 3,900 independent images.

The five-expert CV Atlas MIL implementation was **retrained and tested end-to-end** on all 15 outer folds using 400 trees per tree expert and 35 epochs per MIL expert. Its observed performance was:

| Method and evaluation source | Errors / 3,900 | Missed defects | False alarms | Mean accuracy |
|---|---:|---:|---:|---:|
| Original Atlas-MIL (archived historical CV) | 12 | 10 | 2 | 99.69% |
| Atlas-MIL 70% + morphology 30% (archived score replay) | 10 | 10 | 0 | 99.74% |
| **CV Atlas MIL — retrained cross-validation** | **9** | **8** | **1** | **99.77%** |
| Five-expert blend — earlier archived score replay | 8 | 8 | 0 | 99.79% |

The **nine-error result** is the direct result for the **code in this repository**, whereas the eight-error result combined predictions from earlier separately trained experts. Small variations in seeding, training execution and candidate selection account for a nonidentical outcome; the two are **not interchangeable**. For the 15-fold CV run, repeat errors were **3, 5 and 1** (total eight missed defects and one false alarm). Aggregate results are included under `results/`.

**Interpretation is essential:** the coefficients and architecture were selected following experiments on **this same dataset**, including inspection of held-out predictions. Thus **even the newly refitted 99.77% CV score is selection-biased and does not establish camera-independent or production-batch generalization**. Independent acquisition-domain testing, nested train-only selection and ROI verification are required before deployment claims.
### Spatial correspondence ablation

A control study of the **standalone Regional Joint MIL expert** held descriptors, labels, network architecture and training protocol fixed while changing regional correspondence. Independent per-image shuffling of the morphology and atlas region assignments was applied in both training and evaluation:

| Regional representation | Errors / 3,900 |
|---|---:|
| Co-located descriptors | **51** |
| All feature families moved together | 54 |
| Morphology and atlas shifted together relative to intensity | 98 |
| Independently shuffled morphology and atlas, trial 1 | 146 |
| Independently shuffled morphology and atlas, trial 2 | 156 |

These results support the utility of **within-image alignment of evidence families for regional MIL**. They do not establish that the same effect improves the final five-expert decision: an archived low-weight blend stayed at eight errors in a shuffled-expert replay.

### Acquisition-shift proxy

In a more demanding test, five groups were constructed from image-background brightness; each group was held out in turn. This grouping is an **imperfect acquisition-shift proxy**, *not* a true camera/lot/lighting-session split.

| Method | Errors / 1,300 | Missed defects | False alarms |
|---|---:|---:|---:|
| Original Atlas-MIL (matched score records) | 297 | 45 | 252 |
| **CV Atlas MIL — retrained group holdout** | **271** | **20** | **251** |
| Archived score replay (older model fits) | 263 | 21 | 242 |
| Standalone high-resolution morphology expert | **233** | 43 | 190 |

The new CV Atlas MIL group-holdout run gave 271 errors (79.15% accuracy). The 263-error score used archived expert fits and is retained only for historical context. The marked degradation relative to random CV is evidence that acquisition-condition confounding remains unresolved. In earlier diagnostics, background-only features were predictive of the image-level label; high random-split accuracy must not be attributed solely to recognition of true casting defects.

## Installation

Python 3.10+ is recommended. The package supports Windows, Linux and macOS (CPU feature extraction). The optional native Cython extension and CUDA support depend on platform-specific compilers and drivers.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Optional native Cython extraction:

```bash
python -m pip install -r requirements-build.txt
python setup.py build_ext --inplace
```

Optional NVIDIA acceleration requires a CUDA-enabled PyTorch wheel suited to the device/driver. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). CUDA feature parity is verified before using the GPU branch.

## Dataset structure

The training input is expected to have this structure:

```text
DataSet/casting/
├── def_front/
│   ├── casting_def_001.jpeg
│   └── ...
└── ok_front/
    ├── casting_ok_001.jpeg
    └── ...
```

Images must be readable as **512×512 grayscale**; there is no implicit resize. Supported file extensions: JPEG, PNG, BMP and TIFF. `0=defective`, `1=OK`. Dataset images, private raw data, cached image descriptors and pre-trained weights are not part of this source release.

## Train

```bash
python main.py train \
  --data-dir DataSet/casting \
  --backend auto \
  --model models/cv_atlas_mil.joblib
```

The training command fits the five experts and builds the healthy-reference gallery. It does **not** report generalization accuracy from the training images. A model artifact may contain descriptors derived from the training set; dataset sharing rights therefore matter when distributing weights.

## Predict

```bash
python main.py predict \
  --model models/cv_atlas_mil.joblib \
  --image NewImages \
  --backend auto \
  --output predictions.csv \
  --heatmap-dir heatmaps
```

`NewImages` contains unlabeled 512×512 casting photos directly inside the folder. No `def_front` or `ok_front` subfolders are necessary. The output CSV includes the final defective score and all five expert scores. Optional overlays show atlas deviation, original MIL attention, and regional MIL attention. **These are diagnostic visualizations, not ground-truth defect masks or causal attributions.** Serialized models use `joblib`/pickle: load only artifacts from trusted sources.

## Evaluate

```bash
python main.py evaluate \
  --data-dir DataSet/casting \
  --backend auto \
  --folds 5 \
  --seeds 20261025 20261026 20261027 \
  --output evaluation
```

For a short smoke check, use `--trees 12 --mil-epochs 1 --max-folds 1`. Such a smoke check verifies execution only; its accuracy is not comparable to the research benchmark. Nested hyperparameter selection and group-wise external tests remain necessary for rigorous performance claims.

## Reproducibility and known limits

- Atlas regions are co-located on the 8×8 image grid, **not** registered to the same physical point across photos.
- Labels are available only at image level, so local attention or morphology maxima can indicate edges or background instead of defects.
- The 15×15 fixed Gaussian can suppress small high-frequency details; the high-resolution morphology branch uses milder 3×3 smoothing.
- Acquisition and background appearance show strong correlations with class in the examined dataset.
- The 45/10/5/5/35 fusion was chosen after inspecting earlier outer-fold scores. Independent validation is needed.
- CUDA-versus-CPU feature parity is tested numerically; actual CUDA latency and full pipeline throughput are hardware dependent.
- No inference deployment safeguards, industrial line validation, or defect localization ground truth are included.

## License

Source code is distributed under **PolyForm Noncommercial License 1.0.0**. See [LICENSE.md](LICENSE.md). Commercial use requires a separate license from Zermind. This is **source-available**, not an OSI-approved open-source license. Datasets and independently distributed model weights may have different rights.
