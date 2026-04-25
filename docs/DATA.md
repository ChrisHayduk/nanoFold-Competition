# nanoFold Data Guide

This document is the official data contract for nanoFold. It explains what data is used, how it is downloaded, how official splits are generated, how raw files become model inputs, and exactly what predictions must look like.

The short version:

- Public participants get train and public validation data through `scripts/setup_official_data.sh`.
- Maintainers build train, public validation, and hidden validation through `scripts/full_official_data_refresh.sh`.
- Public train/validation manifests are committed.
- Hidden manifests, hidden features, hidden labels, hidden fingerprints, and hidden locks live under `.nanofold_private/` and must never be committed.
- Models receive processed feature tensors and must return `pred_atom14` with shape `(B, L, 14, 3)`.

## 1. Data Sources

Official nanoFold data is built from public structure and alignment sources, then frozen into manifests, NPZ files, and fingerprints.

| Source | Used for | Why it is used |
|---|---|---|
| OpenFold/OpenProteinSet RODA bucket, `s3://openfold` | Chain cache metadata, duplicate-chain map, per-chain MSA assets | Provides a public, reproducible AlphaFold-style training-data foundation without asking competitors to run external retrieval. |
| OpenFold `chain_data_cache.json` | Candidate universe, sequence, length, resolution, monomer filtering, source lock hashing | Gives a single official chain metadata table for deterministic split generation. |
| OpenFold `duplicate_pdb_chains.txt` | Duplicate-chain fallback during per-chain asset download | Lets the downloader reuse equivalent chain assets when OpenProteinSet stores a duplicate under a representative chain. |
| OpenFold per-chain `uniref90_hits.a3m` | Official MSA input feature | Gives every model the same evolutionary input while keeping external MSA retrieval out of official runs. |
| RCSB mmCIF files, `https://files.rcsb.org/download/<PDB>.cif` | Atomic labels for C-alpha and atom14 supervision/scoring | mmCIF is the authoritative coordinate source used to build geometry labels. |
| CATH domain list | Split stratification metadata | Helps balance broad structural classes so the benchmark does not overrepresent a single fold family. |
| SCOPe classification file | Split stratification metadata | Adds an independent structural-classification signal for split balancing. |
| ECOD domain file | Split stratification metadata | Adds another domain-architecture source, improving coverage and reducing classification blind spots. |
| Optional RCSB REST metadata | Extra split audit metadata when explicitly enabled | Can enrich structural metadata, but the default bulk split path does not require per-chain REST calls. |
| Pinned feature/processability exclusion lists | Split eligibility | Prevents known missing-assets or unprocessable-label chains from entering official manifests. |

Official templates are disabled. The schema still carries template feature tensors, but official preprocessing uses `T=0`, so template lookup cannot decide the leaderboard.

Default source locations after download:

```text
data/openproteinset/
  pdb_data/
    data_caches/chain_data_cache.json
    duplicate_pdb_chains.txt
    mmcif_files/<pdb_id>.cif
  roda_pdb/<encoded_chain_id>/.../uniref90_hits.a3m

data/metadata_sources/
  cath-domain-list.txt
  dir.cla.scope.txt
  ecod.latest.domains.txt
  structure_metadata_sources.lock.json
```

Official public outputs:

```text
data/manifests/train.txt
data/manifests/val.txt
data/manifests/all.txt
data/processed_features/<encoded_chain_id>.npz
data/processed_labels/<encoded_chain_id>.npz
leaderboard/official_dataset_fingerprint.json
leaderboard/official_manifest_source.lock.json
```

Maintainer-only hidden outputs:

```text
.nanofold_private/manifests/hidden_val.txt
.nanofold_private/manifests/split_quality_report.json
.nanofold_private/hidden_processed_features/<encoded_chain_id>.npz
.nanofold_private/hidden_processed_labels/<encoded_chain_id>.npz
.nanofold_private/leaderboard/official_hidden_fingerprint.json
.nanofold_private/leaderboard/private_hidden_assets.lock.json
.nanofold_private/leaderboard/private_hidden_manifest_source.lock.json
.nanofold_private/leaderboard/official_data_source.lock.json
```

## 2. How Data Is Downloaded

Participants should use the public setup script:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt awscli

bash scripts/setup_official_data.sh \
  --data-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels \
  --mmcif-mode subset \
  --disable-templates
```

That script does five things:

1. Verifies the committed public manifest SHA256 hashes against `tracks/limited_large.yaml`.
2. Downloads OpenFold cache metadata from anonymous S3:
   - `s3://openfold/data_caches/`
   - `s3://openfold/duplicate_pdb_chains.txt`
3. Downloads per-chain OpenFold assets for the public union manifest, `data/manifests/all.txt`.
   - The official default MSA file is `uniref90_hits.a3m`.
   - Template hits are skipped when `--disable-templates` is set.
4. Downloads the manifest mmCIF subset from RCSB when `--mmcif-mode subset` is used.
5. Runs `scripts/preprocess.py` separately for train and public validation manifests.

If preprocessing is interrupted, rerun the same command with:

```bash
bash scripts/setup_official_data.sh ... --resume-preprocess
```

Maintainers use the full refresh path:

```bash
export NANOFOLD_HIDDEN_SPLIT_SALT="<private-random-string-at-least-32-chars>"

bash scripts/full_official_data_refresh.sh \
  --rewrite-lock \
  --disable-templates
```

That flow additionally downloads structural metadata sources, regenerates train/val/hidden splits, preprocesses hidden features and labels, rebuilds fingerprints, and writes private hidden locks under `.nanofold_private/`.

## 3. How Dataset Splits Are Determined

Official splits are generated by `scripts/build_manifests.py`, usually through `scripts/regenerate_official_manifests.sh` or `scripts/full_official_data_refresh.sh`.

### Candidate filtering

A chain is eligible only if it passes the official gates:

- sequence exists in `chain_data_cache.json`
- length is between `40` and `256`
- resolution is at most `3.0` Angstrom when known
- chain is monomeric according to the cache metadata
- sequence contains only the 20 standard amino acids
- required OpenFold feature assets are available
- atom14 label projection is processable under the official thresholds

The feature and processability exclusion lists are public, pinned, and hashed through lock metadata.

### Structural metadata used for balancing

The split builder requires a chain-level metadata file from `scripts/build_structure_metadata.py`. That metadata combines:

- OpenFold chain metadata: chain ID, sequence, length, resolution
- CATH domain classifications
- SCOPe classifications
- ECOD domain classifications
- optional RCSB REST fields when that source is explicitly enabled
- pinned missing-feature exclusions
- pinned label-processability exclusions

The metadata builder converts structural/domain annotations into broad balancing fields:

- `secondary_structure_class`
- `domain_architecture_class`
- `length_bin`
- `resolution_bin`

Secondary-structure class is a deterministic broad class used for split balancing, not a model input and not a scoring label.

### Homology and grouping

Splits are not sampled as independent rows. They are sampled as grouped biological units.

The official grouping policy is:

- build sequence clusters with MMseqs2 at `30%` sequence identity and `80%` coverage, or use a locked TSV generated with those same settings
- choose one representative unit per disjoint group
- prevent any sequence cluster from crossing train, public validation, and hidden validation
- prevent different chains from the same PDB entry from crossing split boundaries

This reduces leakage where a model could see a close homolog, duplicate, or same-structure sibling during training and be scored on it later.

### Stratified allocation

The official split sizes are:

- train: `10,000` chains
- public validation: `1,000` chains
- hidden validation: `1,000` chains

The split allocator stratifies by:

```text
secondary_structure_class
domain_architecture_class
length_bin
resolution_bin
```

It allocates each split proportionally across those strata, then checks split quality with Jensen-Shannon divergence gates and unknown-domain coverage gates.

Hidden validation is selected first using the private salt digest. Public validation and train are then selected from the remaining public pool. The committed public lock records public generation metadata, but hidden chain IDs and hidden split quality details live only under `.nanofold_private/`.

The public `all.txt` manifest is the union of train and public validation only.

## 4. How Input Data Is Prepped

Preprocessing is handled by `scripts/preprocess.py`.

For each chain ID, preprocessing:

1. Resolves the encoded OpenProteinSet chain directory.
2. Reads one or more A3M files.
3. Removes query-gap columns from the MSA.
4. Merges and deduplicates MSA rows.
5. Caps raw MSA depth with `--max-msa-seqs` before writing NPZ files.
6. Loads the matching mmCIF file.
7. Extracts chain atom coordinates into the canonical atom14 layout.
8. Aligns the mmCIF structure sequence to the MSA query sequence.
9. Projects atom14 coordinates onto query positions.
10. Rejects chains that fail projection quality thresholds.
11. Writes one feature NPZ and one label NPZ per chain.
12. Writes `preprocess_meta.json` into the feature directory.

Official projection thresholds:

```text
min_projection_seq_identity       = 0.90
min_projection_coverage           = 0.70
min_projection_aligned_fraction   = 0.70
min_projection_valid_ca           = 32
```

Feature NPZ keys:

| Key | Shape | Dtype | Meaning |
|---|---:|---|---|
| `chain_id` | scalar | string-like | Original chain ID. |
| `aatype` | `(L,)` | `int32` | Query amino-acid IDs in `ARNDCQEGHILKMFPSTWYV` order. |
| `msa` | `(N, L)` | `int32` | MSA residue IDs after query-gap removal. |
| `deletions` | `(N, L)` | `int32` | A3M deletion counts. |
| `residue_index` | `(L,)` | `int32` | Contiguous `0..L-1` positions. |
| `between_segment_residues` | `(L,)` | `int32` | Zero for single-chain examples. |
| `template_aatype` | `(T, L)` | `int32` | Template residue IDs. Official track uses `T=0`. |
| `template_ca_coords` | `(T, L, 3)` | `float32` | Template C-alpha coordinates. Official track uses `T=0`. |
| `template_ca_mask` | `(T, L)` | `bool` | Template coordinate mask. Official track uses `T=0`. |
| `projection_*` | scalar | numeric | Projection diagnostics used for audit/debugging. |

Label NPZ keys:

| Key | Shape | Dtype | Meaning |
|---|---:|---|---|
| `ca_coords` | `(L, 3)` | `float32` | True C-alpha coordinates. |
| `ca_mask` | `(L,)` | `bool` | True where C-alpha is present. |
| `atom14_positions` | `(L, 14, 3)` | `float32` | True atom14 coordinates in canonical AF2 slot order. |
| `atom14_mask` | `(L, 14)` | `bool` | True where the atom exists and was present in mmCIF. |
| `residue_index` | `(L,)` | `int32` | Contiguous `0..L-1` positions. |
| `resolution` | scalar | `float32` | Structure resolution, or `0.0` when unknown. |

The data loader then:

- loads per-chain NPZ files with `nanofold.data.ProcessedNPZDataset`
- crops sequences longer than `data.crop_size`
- samples MSA rows to at most `data.msa_depth`
- pads variable-length examples inside each batch
- emits `residue_mask` so models can ignore padded positions
- strips all supervision tensors during inference through `nanofold.submission_runtime`

## 5. Example Model Input Sample

This is a concrete toy example of one collated training sample with `B=1`, sequence length `L=4`, MSA depth `N=3`, and official templates disabled (`T=0`). Real official examples have lengths from `40` to `256`.

Suppose the query sequence is:

```text
A C G D
```

Using nanoFold's residue order `ARNDCQEGHILKMFPSTWYV`, this becomes:

```python
aatype = tensor([[0, 4, 7, 3]], dtype=torch.int64)  # shape (1, 4)
```

The batch passed into `run_batch(..., training=True)` looks like:

```python
batch = {
    "chain_id": ["1abc_A"],

    # Feature tensors
    "aatype": tensor([[0, 4, 7, 3]], dtype=torch.int64),
    "msa": tensor(
        [
            [
                [0, 4, 7, 3],   # query row: A C G D
                [0, 4, 7, 3],   # homolog row
                [0, 20, 7, 3],  # 20 is the unknown token in this toy row; gaps use 21
            ]
        ],
        dtype=torch.int64,
    ),  # shape (1, 3, 4)
    "deletions": tensor(
        [
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 2],
            ]
        ],
        dtype=torch.int64,
    ),  # shape (1, 3, 4)
    "residue_index": tensor([[0, 1, 2, 3]], dtype=torch.int64),
    "between_segment_residues": tensor([[0, 0, 0, 0]], dtype=torch.int64),
    "residue_mask": tensor([[True, True, True, True]]),

    # Official track has no templates, so T=0.
    "template_aatype": tensor([], dtype=torch.int64).reshape(1, 0, 4),
    "template_ca_coords": tensor([], dtype=torch.float32).reshape(1, 0, 4, 3),
    "template_ca_mask": tensor([], dtype=torch.bool).reshape(1, 0, 4),

    # Supervision tensors, present only during training/scoring.
    "ca_coords": tensor(
        [
            [
                [1.40, 1.00, 0.00],
                [4.20, 0.90, 0.10],
                [6.80, 1.10, 0.00],
                [9.10, 1.00, 0.20],
            ]
        ],
        dtype=torch.float32,
    ),  # shape (1, 4, 3)
    "ca_mask": tensor([[True, True, True, True]]),
    "atom14_positions": tensor(..., dtype=torch.float32),  # shape (1, 4, 14, 3)
    "atom14_mask": tensor(..., dtype=torch.bool),          # shape (1, 4, 14)
    "resolution": tensor([2.0], dtype=torch.float32),
}
```

For the first residue, alanine, the atom14 slice might be:

```python
batch["atom14_positions"][0, 0] == tensor(
    [
        [0.00, 1.20, 0.10],  # slot 0: N
        [1.40, 1.00, 0.00],  # slot 1: CA
        [2.00, -0.30, 0.00], # slot 2: C
        [1.50, -1.30, 0.10], # slot 3: O
        [1.90, 1.80, 1.20],  # slot 4: CB
        [0.00, 0.00, 0.00],  # slots 5-13 unused for ALA
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
    ],
    dtype=torch.float32,
)

batch["atom14_mask"][0, 0] == tensor(
    [True, True, True, True, True, False, False, False, False, False, False, False, False, False]
)
```

During hidden prediction, the submission receives the same feature-side keys but does not receive `ca_coords`, `ca_mask`, `atom14_positions`, `atom14_mask`, or `resolution`.

## 6. Output Data Format

Submissions do not write prediction files directly during official runs. They return tensors from `run_batch`, and `predict.py` writes the official prediction artifacts.

Required `run_batch` output:

```python
{
    "pred_atom14": torch.Tensor,  # shape (B, L, 14, 3), floating dtype, finite
}
```

Training output must also include:

```python
{
    "loss": torch.Tensor,  # scalar, finite, requires_grad=True during training
}
```

Runtime rules:

- `pred_atom14` must match the batch `B` and `L`.
- `pred_atom14` must be floating point and finite.
- The runtime derives diagnostic `pred_ca` from `pred_atom14[:, :, 1, :]`.
- Official scoring reads only `pred_atom14`.
- C-alpha-only outputs are not accepted.

The atom14 slot order is defined in `nanofold/residue_constants.py`. The backbone slots are:

```text
slot 0: N
slot 1: CA
slot 2: C
slot 3: O
slot 4: CB when present
slots 5-13: residue-specific side-chain atoms
```

`predict.py` writes one compressed NPZ per chain:

```text
runs/<run_name>/.../predictions/<encoded_chain_id>.npz
```

Prediction NPZ keys:

| Key | Shape | Dtype | Meaning |
|---|---:|---|---|
| `pred_atom14` | `(L, 14, 3)` | `float32` | Predicted atom14 coordinates for non-padding residues only. |
| `masked_length` | scalar | `int32` | Number of non-padding residues written. |
| `ckpt` | scalar/string-like | string | Checkpoint path used for this prediction. |

For multi-checkpoint hidden evaluation, each checkpoint gets its own prediction subdirectory under the prediction root.

`predict.py` also writes a summary JSON when `--save` is provided. Scoring consumes that summary plus the label directory:

```bash
python score.py \
  --prediction-summary runs/<run_name>/predict_hidden.json \
  --labels-dir .nanofold_private/hidden_processed_labels \
  --save runs/<run_name>/eval_hidden.json
```

## 7. Example Model Output Sample

For the toy input above with `B=1` and `L=4`, a valid inference output from `run_batch` is:

```python
pred_atom14 = torch.zeros((1, 4, 14, 3), dtype=torch.float32, device=batch["aatype"].device)

# Residue 0 atom14 coordinates.
pred_atom14[0, 0, 0] = torch.tensor([0.05, 1.10, 0.05])   # N
pred_atom14[0, 0, 1] = torch.tensor([1.35, 1.05, 0.02])   # CA
pred_atom14[0, 0, 2] = torch.tensor([2.05, -0.25, 0.04])  # C
pred_atom14[0, 0, 3] = torch.tensor([1.45, -1.25, 0.08])  # O
pred_atom14[0, 0, 4] = torch.tensor([1.85, 1.75, 1.25])   # CB

# Residue 1 atom14 coordinates.
pred_atom14[0, 1, 0] = torch.tensor([2.90, 0.10, 0.00])
pred_atom14[0, 1, 1] = torch.tensor([4.20, 0.90, 0.10])
pred_atom14[0, 1, 2] = torch.tensor([4.80, -0.40, 0.10])
pred_atom14[0, 1, 3] = torch.tensor([4.10, -1.40, 0.10])
pred_atom14[0, 1, 4] = torch.tensor([4.70, 1.80, 1.20])
pred_atom14[0, 1, 5] = torch.tensor([5.70, 2.20, 1.90])   # CYS SG slot

# Residues 2 and 3 would fill the same (14, 3) structure.

out = {"pred_atom14": pred_atom14}
```

During training, the same output must include a differentiable scalar loss:

```python
loss = ((pred_atom14[:, :, 1, :] - batch["ca_coords"]) ** 2)
loss = (loss.sum(dim=-1) * batch["ca_mask"].float()).sum() / batch["ca_mask"].float().sum().clamp_min(1.0)

out = {
    "pred_atom14": pred_atom14,
    "loss": loss,
}
```

After `predict.py` writes the chain artifact, the NPZ for `1abc_A` would contain:

```python
{
    "pred_atom14": np.array(
        [
            [
                [0.05, 1.10, 0.05],
                [1.35, 1.05, 0.02],
                [2.05, -0.25, 0.04],
                [1.45, -1.25, 0.08],
                [1.85, 1.75, 1.25],
                [0.00, 0.00, 0.00],
                # slots 6-13 omitted here for readability
            ],
            # residues 1-3 omitted here for readability
        ],
        dtype=np.float32,
    ),  # shape (4, 14, 3)
    "masked_length": np.array(4, dtype=np.int32),
    "ckpt": "runs/example/checkpoints/ckpt_last.pt",
}
```

The scoring code compares `pred_atom14` with label `atom14_positions` using `atom14_mask`. The official FoldScore is:

```text
0.55 * lDDT-C-alpha + 0.30 * lDDT-backbone-atom14 + 0.15 * lDDT-all-atom14
```

Hidden leaderboard rank is `foldscore_auc_hidden`, computed across official checkpoint predictions over the fixed sample budget.
