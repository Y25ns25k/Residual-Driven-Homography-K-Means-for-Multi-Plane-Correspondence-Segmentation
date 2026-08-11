# CONSAC Official-Code Integration

Repository URL: https://github.com/fkluger/consac

Local checkout: `third_party/consac`

Pinned commit:

```text
e0a4f691aa385ba843d442d0e8e5bf1fe2f21f46
```

Clone command used:

```bash
git clone --recurse-submodules https://github.com/fkluger/consac.git third_party/consac
```

Official homography Sequential RANSAC-style command:

```bash
python evaluate_homography.py --dataset_path /path/to/adelaidermf --runcount 5 --uniform --cpu
```

`--uniform` disables guided CONSAC sampling and runs the official-code uniform / Sequential RANSAC-style baseline. This is third-party code and is not vendored into `src/homography_kmeans`.

Environment notes:

- Official `environment.yml` targets Python 3.6.8, PyTorch 1.3.1, NumPy 1.16.4, SciPy 1.2.1.
- The local environment used for this integration is Python 3.10 with PyTorch 2.11.0+cpu.
- On Windows/Python 3.10, the official top-level script cannot spawn DataLoader workers safely because it lacks an `if __name__ == "__main__"` guard.
- NumPy 2 removed the deprecated `np.int` alias used in the official dataset/evaluation utilities.

Local compatibility/payload patch:

```text
third_party/consac_patches/windows_py310_payload.patch
```

Patch scope:

- Set homography DataLoader `num_workers=0` for Windows spawn compatibility.
- Add NumPy `np.int = int` compatibility aliases in third-party dataset/evaluation files.
- Add optional `--resultdir` payload saving to `evaluate_homography.py`.

The patch is intended to be compatibility and instrumentation only. It does not intentionally alter uniform sampling, EM refinement, or misclassification-error logic, but the local environment is not the original Python 3.6 / PyTorch 1.3 environment.

Additional compatibility note:

- A zero-instance crash guard was added after the modern CPU/PyTorch run produced an empty `selected_instances` stack in one execution path.

Result produced in this repository:

```text
outputs/official_consac_uniform_20260611_205647
```

Summary:

- AdelaideRMF-H scenes: 19
- Runs per scene: 5
- Payloads saved: 95
- Official-code `--uniform` ME: 9.77 +/- 8.17 percent using scene-mean std
- All scene-run rows: 9.77 +/- 9.85 percent

The exact data command used in the local run was:

```bash
python evaluate_homography.py --dataset_path /path/to/adelaidermf --runcount 5 --uniform --cpu --resultdir /path/to/outputs/official_consac_uniform/payloads
```

This is close in scale to the literature Sequential RANSAC number, 11.14 +/- 10.54 percent, but it is not an exact reproduction.
