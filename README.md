# Scale-Adaptive Residual-Driven Homography K-Means

Research code for multi-plane correspondence segmentation with classical
homography fitting.  The pipeline starts with greedy Sequential RANSAC and
then refines assignments and homographies using residual-driven Homography
K-Means (HKM).  It is designed to be lightweight and interpretable rather
than state of the art.

> **Scope.** The main claim is a modest, controlled improvement over this
> repository's Sequential RANSAC baseline on multi-plane data.  This is not a
> claim of state-of-the-art multi-model fitting, exact model-count recovery,
> or direct equivalence to CONSAC, Multi-X, or Progressive-X.

## Highlights

- Deterministic normalized/weighted DLT, RANSAC, and Sequential RANSAC.
- Scale-adaptive residual reassignment and homography refitting.
- Optional residual-pool discovery and functional/energy merge diagnostics.
- Synthetic evaluation with correspondence-level segmentation and model-count
  metrics, plus optional AdelaideRMF validation.
- A tested Python package: `66 passed` with four expected DLT-conditioning
  warnings on noiseless recovery cases.

## Installation

Python 3.10 or later is required.

```bash
git clone <YOUR-REPOSITORY-URL>
cd homography-kmeans
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[test]"
```

## Quick start

The synthetic runner generates its scenes at runtime; no dataset download is
needed for this command.

```bash
python scripts/run_synthetic.py --config configs/synthetic_quick.yml
python -m pytest tests -q
```

Outputs are written to `outputs/<run_name>/` and are deliberately ignored by
Git.  The full synthetic configuration and calibration diagnostics are:

```bash
python scripts/run_synthetic.py --config configs/synthetic_full.yml
python scripts/sweep_thresholds.py --config configs/synthetic_full.yml
python scripts/merge_calibration.py --config configs/synthetic_full.yml
python scripts/conservative_calibration.py --config configs/synthetic_full.yml
```

## AdelaideRMF evaluation

AdelaideRMF is external data and is not distributed with this repository.
Obtain it from its original provider, comply with its terms, and pass the
local path explicitly:

```bash
python scripts/run_adelaide.py --subset homography --data /path/to/adelaidermf --config configs/adelaide.yml
```

Use the `homography` subset for the paper-comparable benchmark.  The
`fundamental` and `all` subsets are diagnostic only because this implementation
fits homographies.  If the data path is absent, the runner records a skipped
summary rather than fabricating results.

## Main results

On AdelaideRMF-H (19 scenes, five seeds, including outliers), the clean
rebuild reports:

| Method | ME (%) | SegAcc | CountAcc | AbsK |
|---|---:|---:|---:|---:|
| Global RANSAC | 29.81 +/- 15.54 | 0.702 | 0.158 | 1.779 |
| Sequential RANSAC | 14.93 +/- 10.63 | 0.851 | 0.653 | 0.453 |
| Residual HKM v2 | 13.33 +/- 10.31 | 0.867 | 0.705 | 0.358 |

The refinement improves the repository baseline modestly.  It should not be
interpreted as an official CONSAC result or as a state-of-the-art comparison.

## Repository layout

```text
src/homography_kmeans/  maintained implementation
tests/                  unit and integration tests
configs/                checked-in experiment configurations
scripts/                reproducible experiment and figure runners
experiments/, src/*.py  legacy exploratory code; not part of the package API
```

Local report and presentation materials are intentionally excluded from the
public code release because they contain submission artifacts and personal
course metadata.  They can be shared separately after an explicit redaction
review.

## External components and data

`third_party/` is local research material and is excluded from the public
source release.  In particular, optional CONSAC protocol checks must be
obtained from the upstream project, pinned independently, and cited according
to its terms.  See `THIRD_PARTY_NOTICES.md` before using that workflow.

Generated datasets, experiment outputs, model checkpoints, videos, and built
PDF/PPTX files are intentionally ignored.  They can be regenerated from the
checked-in code and configurations, or shared separately as a release/archive
only after their distribution terms are confirmed.

## Citation and license

Original code in this repository is released under the [MIT License](LICENSE).
See `CITATION.cff` for citation metadata.  The license covers only the
original code and must not be assumed to apply to external datasets or
third-party components.
