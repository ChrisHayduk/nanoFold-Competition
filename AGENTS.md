# AGENTS.md

This file is the operating guide for AI agents working on nanoFold.

nanoFold is a protein-folding slowrun: a fixed-data, fixed-budget benchmark for learning protein geometry under biological data scarcity. The project should always read as one coherent competition, not as a trail of implementation history.

## Core Principles

- Optimize for data efficiency. The benchmark exists to reward architectures, objectives, curricula, and optimization methods that learn more from the same limited biological data.
- Protect the scarce-data premise. Do not add external structures, pretrained weights, external MSA retrieval, template lookup, network access, or hidden-label exposure to official execution paths.
- Keep official data auditable. Manifest generation, preprocessing, split metadata, fingerprints, source locks, and hidden-asset locks must stay deterministic and verifiable.
- Keep hidden validation sealed. Prediction code may see hidden features only. Scoring code may see labels only after predictions are written. Do not mount labels into submission runtime.
- Preserve the atom14 contract. Official submissions return `pred_atom14` with shape `(B, L, 14, 3)`. Scoring uses a CASP15-inspired FoldScore over GDT_HA, all-atom14 lDDT, CADaa contact preservation, SphereGrinder local environments, side-chain geometry, heavy-atom clashes, backbone geometry, and DipDiff local distance windows.
- Keep the public story clean. Do not describe current code as a migration from older behavior, do not leave backwards-compatibility language for unreleased interfaces, and do not reference obsolete approaches in comments or docs.

## Engineering Expectations

- Follow existing repo patterns before adding new abstractions.
- Keep changes narrow, readable, and enforceable.
- Treat data and evaluation code as high-integrity infrastructure. Prefer explicit validation, clear errors, deterministic outputs, and lock/fingerprint checks.
- Never commit generated data, private salts, hidden manifests, hidden fingerprints, private lock files, model checkpoints, or local run outputs.
- Keep `.gitignore` current whenever a new local, generated, or private artifact path appears.

## Testing Requirements

Agents must maintain full test coverage for every behavioral change.

- Add or update unit tests for new logic.
- Add integration tests when changing data prep, sealed runtime behavior, scoring, fingerprints, official manifests, or submission APIs.
- Add regression tests for every fixed bug.
- Do not weaken tests to make a change pass.

Before committing or submitting code, always run the relevant local equivalent of CI:

```bash
.venv/bin/ruff check .
.venv/bin/mypy nanofold scripts train.py eval.py
.venv/bin/pyright --warnings
.venv/bin/pytest
python scripts/sync_official_manifest_hashes.py --check
python scripts/check_public_release_leaks.py
git diff --check
```

When touching official runner, data integrity, scoring, checkpoint, or workflow code, also run the smoke official path from `.github/workflows/ci.yml` locally or explain why it could not be run.

## Documentation Requirements

Docs are part of the product.

- Update `README.md`, `docs/COMPETITION.md`, `docs/API.md`, `docs/DATA.md`, and script help text whenever behavior, commands, config fields, paths, metrics, or data requirements change.
- Keep setup and preprocessing instructions simple enough for a new participant to follow.
- Keep data documentation detailed enough for a maintainer to audit sources, split decisions, preprocessing, input tensors, output tensors, and hidden-asset handling.
- Keep all public docs free of hidden chain IDs, hidden salts, private paths, local absolute paths, and generated-private metadata.

## Data And Manifest Discipline

- Official public manifests live in `data/manifests/train.txt`, `data/manifests/val.txt`, and `data/manifests/all.txt`.
- Hidden manifests and hidden labels belong only in ignored private paths.
- Manifest hash references must stay synchronized with `scripts/sync_official_manifest_hashes.py`.
- Any data-prep change must preserve deterministic split generation, structural stratification, cluster/PDB-entry disjointness, required feature availability checks, and fingerprint reproducibility.

## Final Handoff

Every handoff should clearly state:

- what changed
- which tests and linters were run
- any checks that could not be run
- any remaining risk or follow-up needed

Do not present work as ready if tests, docs, or CI-equivalent checks are stale.
