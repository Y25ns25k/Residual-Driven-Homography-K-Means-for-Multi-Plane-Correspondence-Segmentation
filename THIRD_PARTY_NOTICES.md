# Third-party components and datasets

This project contains original code under `src/homography_kmeans/`.  External
code, datasets, checkpoints, and generated outputs are **not** covered by the
license selected for the original code.

## CONSAC

Some local experiment notes refer to the official
[CONSAC repository](https://github.com/fkluger/consac) at commit
`e0a4f691aa385ba843d442d0e8e5bf1fe2f21f46`.  It is used only for an optional
protocol check and is excluded from this repository by `.gitignore`.

Before reproducing that check, clone CONSAC yourself, review its current
license and submodule licenses, cite its paper, and apply any compatibility
patch only after reviewing it.  The local patch in
`third_party/consac_patches/windows_py310_payload.patch` is retained as a
research record, not as a replacement for the upstream project.

## AdelaideRMF

AdelaideRMF data are external benchmark data.  They are not redistributed by
this project.  Download them from an authorized source and comply with their
terms before running `scripts/run_adelaide.py`.

## Generated assets

Experiment outputs, figures, videos, report PDFs, and slide decks may include
external data or third-party-derived material.  Do not publish them unless you
have independently confirmed that their distribution is permitted.
