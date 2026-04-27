# nanoFold Data Guide

This document is the official data contract for nanoFold. It explains what data is used, how it is downloaded, how official splits are generated, how raw files become model inputs, and exactly what predictions must look like.

The short version:

- Public participants get train and public validation data through `scripts/setup_official_data.sh`.
- Maintainers build hidden validation privately through `scripts/build_hidden_manifest.py` after the public manifests are fixed.
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

1. Verifies the committed public manifest SHA256 hashes against `tracks/limited.yaml`.
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

Maintainers generate hidden validation against the fixed public manifests:

```bash
mkdir -p .nanofold_private/secrets
python -c "import pathlib,secrets; pathlib.Path('.nanofold_private/secrets/hidden_split_salt.txt').write_text(secrets.token_urlsafe(48) + '\n')"
chmod 600 .nanofold_private/secrets/hidden_split_salt.txt

python scripts/build_hidden_manifest.py \
  --hidden-split-salt-file .nanofold_private/secrets/hidden_split_salt.txt

python scripts/verify_hidden_manifest.py
```

The hidden builder preserves `data/manifests/train.txt`, `data/manifests/val.txt`, and `data/manifests/all.txt`. Hidden features, labels, fingerprints, and locks are then built under `.nanofold_private/` with `scripts/prepare_data.py`, `scripts/preprocess.py`, `scripts/build_fingerprint.py`, and `scripts/pin_hidden_assets.py`.

If hidden preprocessing finds chains that cannot produce valid atom14 labels, maintainers record those chain IDs only in `.nanofold_private/manifests/hidden_processability_exclusions.txt`. The hidden builder excludes those private failures from future hidden selection without changing public train or public validation.

## 3. How Dataset Splits Are Determined

Official public splits are generated by `scripts/build_manifests.py`, usually through `scripts/regenerate_official_manifests.sh` or `scripts/full_official_data_refresh.sh`. Maintainer-only hidden validation for the committed public contract is generated by `scripts/build_hidden_manifest.py`.

The split is designed to answer one scientific question:

> When every participant has the same small, fixed experimental structure set, which models learn general protein geometry rather than memorizing families, exploiting retrieval, or overfitting a narrow structural distribution?

That goal makes the split policy part of the benchmark, not just bookkeeping. Protein structures are not independent examples in the same way as shuffled text lines. The PDB contains many near-duplicates, homologous families, same-entry chains, overrepresented folds, overrepresented experimental regimes, and distributional biases from what structural biology has historically found easiest or most important to solve. A random row split would make the public/hidden score easier, noisier, and less biologically meaningful.

### Split design goals

The official split is built to satisfy five goals:

1. Prevent train/eval homology leakage.
2. Keep train, public validation, and hidden validation scientifically comparable.
3. Avoid secondary-structure and fold-family skew.
4. Ensure every chain can actually be downloaded and converted into official tensors.
5. Make the public data contract reproducible without revealing hidden chain IDs.

Those goals pull in different directions. Strong leakage controls reduce candidate count. Stratification improves comparability but can overconstrain small candidate pools. Processability gates reduce broken examples but can bias toward easier structures. The official policy tries to be strict enough for a serious leaderboard while staying simple enough to audit.

### Data used by the split builder

The split builder uses metadata only. It does not inspect hidden labels to select high-scoring or low-scoring examples.

| Data | File or source | Role in splitting | Scientific purpose |
|---|---|---|---|
| Chain sequence | `chain_data_cache.json` | Defines the chain sequence used for filtering and MMseqs2 clustering. | Lets the split block close sequence homologs across train/validation/hidden. |
| Chain length | `chain_data_cache.json` | Filters to `40 <= L <= 256`; forms `length_bin`. | Keeps the benchmark inside the official crop regime and balances short/medium/long proteins. |
| Resolution | `chain_data_cache.json` | Filters structures above `3.0A` when known; forms `resolution_bin`. | Reduces label noise and prevents one split from receiving systematically lower-quality coordinates. |
| Oligomeric count | `chain_data_cache.json` | Filters to monomer chains. | Keeps official evaluation focused on single-chain folding rather than missing biological assembly context. |
| Standard amino-acid content | `chain_data_cache.json` | Excludes chains with non-standard sequence tokens. | Keeps input/output semantics clean for the fixed amino-acid vocabulary. |
| PDB ID | Chain ID prefix | Prevents chains from the same PDB entry crossing split boundaries. | Blocks same-experiment or same-complex leakage that sequence clustering alone may miss. |
| OpenFold feature availability | `openfold_required_feature_exclusions.txt` plus download checks | Excludes chains without required official MSA assets. | Ensures every selected chain can produce the same feature schema for every participant. |
| Label processability | `official_processability_exclusions.txt` plus preprocessing gates | Excludes chains that fail atom14 projection quality. | Ensures selected chains can produce reliable atom14 labels for scoring. |
| Sequence clusters | MMseqs2 cluster TSV or live MMseqs2 run | Groups chains at `30%` identity and `80%` coverage. | Makes evaluation closer to generalization across protein families rather than interpolation among close homologs. |
| CATH annotations | `cath-domain-list.txt` | Provides domain architecture and secondary-structure class signals. | Adds structural-class coverage so the benchmark is not dominated by the easiest or most common domain types. |
| SCOPe annotations | `dir.cla.scope.txt` | Provides independent structural/evolutionary class signals. | Cross-checks CATH-style structural classes with a manually curated/evolution-oriented resource. |
| ECOD annotations | `ecod.latest.domains.txt` | Provides evolutionary domain architecture signals. | Improves coverage of remote homology and broad domain categories. |
| Optional RCSB REST metadata | `rcsb_chain_metadata.jsonl` when enabled | Adds extra audit metadata such as method/source organism. | Useful for future balancing and diagnostics, but not required for the official bulk split path. |
| Hidden split salt | `NANOFOLD_HIDDEN_SPLIT_SALT` | Selects hidden validation deterministically but privately. | Lets maintainers reproduce the hidden split without publishing enough information to infer it. |

### Why these sources are used

OpenProteinSet/OpenFold provides the practical foundation. It packages AlphaFold-style MSA assets at scale, and OpenFold/OpenProteinSet exist specifically to make AlphaFold-like training data reproducible for the research community. nanoFold uses this instead of asking participants to run their own MSA pipelines because MSA generation is expensive, non-deterministic across database snapshots, and a major leakage surface if participants can retrieve extra information.

RCSB mmCIF files provide the coordinate truth. mmCIF is used because it is the modern PDB archival format and can represent large structures, rich metadata, exact chain identifiers, and atom-level records more robustly than legacy PDB format. nanoFold turns those coordinates into atom14 labels so scoring can evaluate backbone and side-chain geometry, not only C-alpha traces.

CATH, SCOPe, and ECOD provide structural context that sequence metadata alone cannot. Sequence clustering controls close homology, but it does not guarantee balanced structural coverage. A dataset can be sequence-disjoint and still be dominated by all-alpha proteins, beta sandwiches, small enzyme domains, or one region of fold space. The structural classification sources provide broad domain/secondary-structure signals that let the split allocator preserve a similar biological mix across train, public validation, and hidden validation.

MMseqs2 provides the scalable sequence clustering step. The official policy uses a deliberately coarse `30%` sequence-identity threshold with `80%` coverage so that obvious family-level leakage is blocked without collapsing the entire structure universe into very broad remote-homology groups.

The exclusion lists are included because a scientifically elegant split is useless if selected chains cannot be downloaded or scored. They make the split operationally reproducible: all selected chains must have official input features and reliable atom14 labels.

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

Each filter has a scientific or operational reason:

- Length `40-256` keeps examples within the intended single-domain/small-chain regime and matches the official crop size. This makes sample-budget comparisons cleaner because one sample is not allowed to secretly mean a very different sequence-length problem.
- Resolution `<=3.0A` reduces coordinate noise. Noisy labels can reward models that learn broad coarse structure while hiding atom-level errors, which is the opposite of why nanoFold uses atom14 scoring.
- Monomer-only selection removes ambiguity about missing partners. A chain from an obligate multimer can have a PDB conformation stabilized by partner chains that are absent from the model input.
- Standard amino-acid-only selection keeps the official vocabulary stable and avoids special-case chemistry in the first competition track.
- Feature availability ensures no participant has a different input distribution because some chains silently lack MSA assets.
- Label processability ensures that the official atom14 target corresponds to the MSA query sequence with enough identity, coverage, and resolved C-alpha atoms.

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

The current metadata builder maps domain architecture to broad secondary-structure fractions/classes:

- `alpha`
- `beta`
- `alpha_beta`
- `coil_or_sparse`
- `mixed_low_confidence`

This is intentionally broad. nanoFold is not trying to make a benchmark of CATH-vs-SCOP-vs-ECOD taxonomy labels. It is using those taxonomies to avoid obvious structural composition errors. For example, if hidden validation accidentally had many more beta-rich structures than training, the leaderboard could punish models for distribution shift rather than reward better data-efficient learning. If training were much more alpha-rich than validation, a model with an alpha-helix-biased inductive prior could look strong during training but fail to learn the harder sheet topology patterns needed for general folding.

The scientific purpose is to measure whether architectures learn transferable geometric rules across the main regimes of protein structure:

- local helical packing
- beta-strand pairing and sheet topology
- mixed alpha/beta motifs
- longer-range contacts across nonlocal sequence positions
- domain-scale geometry under a fixed sample budget

Balancing secondary structure, broad domain architecture, length, and resolution makes the hidden score a better estimate of general fixed-data learning and a less noisy measurement of quirks in one split.

### Homology and grouping

Splits are not sampled as independent rows. They are sampled as grouped biological units.

The official grouping policy is:

- build sequence clusters with MMseqs2 at `30%` sequence identity and `80%` coverage, or use a locked TSV generated with those same settings
- choose one representative unit per disjoint group
- prevent any sequence cluster from crossing train, public validation, and hidden validation
- prevent different chains from the same PDB entry from crossing split boundaries

This reduces leakage where a model could see a close homolog, duplicate, or same-structure sibling during training and be scored on it later.

The PDB is full of biologically related entries: mutants, orthologs, paralogs, constructs, alternate crystal forms, ligand-bound variants, different chains in one biological assembly, and repeated domains. If those relatives cross splits, validation stops measuring whether a model has learned folding principles and starts measuring how well it interpolates from near-neighbor structures.

The cluster/PDB grouping therefore serves two related purposes:

- Sequence-cluster disjointness blocks close family leakage.
- PDB-entry disjointness blocks same-experiment and same-complex leakage.

The representative chosen for a disjoint unit is selected before splitting, so the allocator operates on one biological unit at a time. This also prevents large duplicate families from gaining extra influence simply because they have many deposited variants.

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

The public allocator works in three stages:

1. Build disjoint units from eligible chains after sequence clustering and PDB grouping.
2. Select public validation with deterministic stratified sampling.
3. Select train from the remaining public pool with deterministic stratified sampling.

The hidden allocator uses the fixed public manifests as reference data:

1. Build the same disjoint units from eligible chains after sequence clustering and PDB grouping.
2. Remove every unit whose component contains any train or public-validation chain.
3. Allocate hidden counts by stratum to match the public train+validation distribution.
4. Select hidden validation from the remaining units with a deterministic salted hash.

Within each split, target counts are allocated proportionally by stratum. A stratum is the tuple:

```text
(secondary_structure_class, domain_architecture_class, length_bin, resolution_bin)
```

The sampler is deterministic for a given seed, source metadata, cluster map, public manifests, and hidden salt. This matters because maintainers need to rebuild the same official hidden data, but public contributors should not be able to infer the hidden chain IDs from public files.

The committed public lock records public generation metadata. Hidden chain IDs, the hidden split salt digest, hidden split quality details, hidden fingerprints, and private locks live only under `.nanofold_private/`.

Maintainers verify the hidden split with `scripts/verify_hidden_manifest.py`. The verifier recomputes the MMseqs2 grouping, checks zero chain overlap, zero cluster overlap, and zero PDB-entry overlap against the public manifests, then recomputes the Jensen-Shannon quality report for the balancing fields above.

The public `all.txt` manifest is the union of train and public validation only.

### Targeted public distributions

The split does not target hand-written ideal biological proportions. It targets proportional matching to the eligible, grouped, processable chain universe after all official filters have been applied. That matters because the target should reflect what can actually be downloaded, clustered, and scored under the official rules.

For public documentation, the target below is the public train + public validation union recorded by `leaderboard/official_manifest_source.lock.json`. Hidden aggregate quality is checked by the private maintainer lock, but hidden split details are not published.

The allocator tries to equalize four metadata views:

- `secondary_structure_class`: broad alpha, beta, mixed, sparse, or low-confidence secondary-structure regime
- `domain_architecture_class`: broad structural/domain architecture after CATH, SCOPe, ECOD, and derived metadata normalization
- `length_bin`: sequence-length regime within the official `40-256` residue range
- `resolution_bin`: coordinate-quality regime, including an `unknown` bucket when resolution is unavailable

#### Secondary-structure class

| Bucket | Public target count (%) | Train count (%) | Public val count (%) |
|---|---:|---:|---:|
| alpha | `3,090` (28.09%) | `2,815` (28.15%) | `275` (27.50%) |
| alpha/beta | `4,380` (39.82%) | `3,983` (39.83%) | `397` (39.70%) |
| beta | `2,291` (20.83%) | `2,083` (20.83%) | `208` (20.80%) |
| coil/sparse | `153` (1.39%) | `137` (1.37%) | `16` (1.60%) |
| mixed low-confidence | `1,086` (9.87%) | `982` (9.82%) | `104` (10.40%) |

#### Domain-architecture class

Most domain-architecture buckets correspond directly to broad CATH/SCOPe/ECOD-style classes. Rare source-specific residual labels are preserved as their own buckets so they can be balanced rather than silently folded into a larger class.

| Bucket | Public target count (%) | Train count (%) | Public val count (%) |
|---|---:|---:|---:|
| `6` source bucket | `220` (2.00%) | `196` (1.96%) | `24` (2.40%) |
| alpha | `3,090` (28.09%) | `2,815` (28.15%) | `275` (27.50%) |
| alpha/beta | `4,380` (39.82%) | `3,983` (39.83%) | `397` (39.70%) |
| beta | `2,291` (20.83%) | `2,083` (20.83%) | `208` (20.80%) |
| coil/sparse | `153` (1.39%) | `137` (1.37%) | `16` (1.60%) |
| mixed low-confidence | `856` (7.78%) | `776` (7.76%) | `80` (8.00%) |
| other | `10` (0.09%) | `10` (0.10%) | `0` (0.00%) |

#### Length bin

| Bucket | Public target count (%) | Train count (%) | Public val count (%) |
|---|---:|---:|---:|
| 40-63 aa | `746` (6.78%) | `676` (6.76%) | `70` (7.00%) |
| 64-95 aa | `1,708` (15.53%) | `1,548` (15.48%) | `160` (16.00%) |
| 96-127 aa | `2,226` (20.24%) | `2,027` (20.27%) | `199` (19.90%) |
| 128-191 aa | `3,611` (32.83%) | `3,285` (32.85%) | `326` (32.60%) |
| 192-256 aa | `2,709` (24.63%) | `2,464` (24.64%) | `245` (24.50%) |

#### Resolution bin

| Bucket | Public target count (%) | Train count (%) | Public val count (%) |
|---|---:|---:|---:|
| <=1.5 A | `2,082` (18.93%) | `1,892` (18.92%) | `190` (19.00%) |
| 1.5-2.0 A | `3,760` (34.18%) | `3,419` (34.19%) | `341` (34.10%) |
| 2.0-2.5 A | `2,056` (18.69%) | `1,869` (18.69%) | `187` (18.70%) |
| 2.5-3.0 A | `863` (7.85%) | `783` (7.83%) | `80` (8.00%) |
| unknown | `2,239` (20.35%) | `2,037` (20.37%) | `202` (20.20%) |

The resulting public train and validation distributions are very close to the public target. Jensen-Shannon divergence is the split-quality metric used for each balancing field:

| Field | Train JS divergence | Public val JS divergence |
|---|---:|---:|
| `secondary_structure_class` | `6.7834414e-07` | `0.00016998018` |
| `domain_architecture_class` | `6.1796866e-06` | `0.00070301452` |
| `length_bin` | `1.1833676e-07` | `6.0099717e-05` |
| `resolution_bin` | `8.0612644e-08` | `1.1483855e-05` |

The official quality gate is `max_split_js_divergence <= 0.35`; the largest public train/val value is `0.00070301452`. The unknown-domain-architecture fraction is `0.0`, below the `0.75` gate.

### Split quality checks

The split builder records and checks:

- candidate counts
- disjoint unit counts
- reject-reason counts
- clustering method and command
- cluster-assignment hash
- grouping policy
- stratification fields
- per-split structural distributions
- Jensen-Shannon divergence between each split and the selected overall distribution
- unknown-domain-architecture fraction
- train/val/hidden chain counts and manifest hashes

The public lock is sanitized so it contains the public contract and public split metadata. The private hidden locks contain hidden manifest hashes, hidden distribution details, and hidden fingerprint hashes.

### Why hidden and public validation both exist

Public validation is for debugging. Participants need a visible split to check that their model trains, predicts, and produces sane geometry under the official API.

Hidden validation is for ranking. It exists because repeated iteration on public validation can become leaderboard overfitting even when participants act in good faith. Hidden scoring keeps the public validation set from becoming the real training target.

The hidden split is not intended to be adversarial. It is intended to be compositionally comparable to train and public validation while remaining sealed.

### What the split is not trying to measure

The official tracks intentionally do not measure:

- template lookup ability
- external MSA/database retrieval skill
- pretrained protein language model knowledge
- multimer assembly inference
- ligand/cofactor-dependent folding
- very long protein domain packing
- non-standard amino-acid chemistry

Those are important scientific problems, but mixing them into the first track would make it harder to interpret the core result: data-efficient learning of single-chain atom14 geometry from a fixed, scarce supervised structure set.

### References for split methodology

The split policy follows the same broad lessons that show up repeatedly in protein-structure prediction literature: control sequence redundancy, control template/homology leakage, preserve structural diversity, and treat PDB-derived data as clustered biological observations rather than IID rows.

- [OpenProteinSet](https://pmc.ncbi.nlm.nih.gov/articles/PMC10441447/) motivates using a public precomputed MSA/structure corpus because MSA generation is expensive and central to structure learning.
- [OpenFold](https://www.nature.com/articles/s41592-024-02272-z) motivates fixed-data retraining as a way to study learning behavior, generalization, and the effect of training-set composition.
- [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) illustrates the importance of sequence clustering, recent-PDB evaluation, template controls, MSA features, and structural generalization in protein folding benchmarks.
- [MMseqs2](https://www.nature.com/articles/nbt.3988) provides the scalable sequence clustering/search machinery used for homology grouping.
- [CATH-Gene3D](https://pmc.ncbi.nlm.nih.gov/articles/PMC5210570/), [SCOPe](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323910/), and [ECOD](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003926) motivate using structural/evolutionary domain classifications as broad coverage signals.
- [PDBx/mmCIF](https://pmc.ncbi.nlm.nih.gov/articles/PMC10292674/) motivates mmCIF as the structural biology data standard for atom-level coordinate extraction.

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

`predict.py` also writes a summary JSON when `--save` is provided. Scoring consumes that summary, the feature directory for residue identities, and the label directory for coordinates:

```bash
python score.py \
  --prediction-summary runs/<run_name>/predict_hidden.json \
  --features-dir .nanofold_private/hidden_processed_features \
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

The scoring code compares `pred_atom14` with label `atom14_positions` using `atom14_mask`. The official FoldScore is a CASP15-inspired raw composite:

```text
FoldScore =
  0.25 * GDT_HA-C-alpha
+ 0.09375 * (lDDT-all-atom14 + CADaa-atom14 + SG-atom14 + SC-atom14)
+ 0.125 * (MolProbity-clash-atom14 + BB-atom14 + DipDiff-atom14)
```

`GDT_HA-C-alpha` is computed from atom14 slot `1` with threshold-specific GDT
superpositions. `lDDT-all-atom14` is computed from all resolved atom14 slots
under the label mask while excluding intra-residue atom pairs.
`CADaa-atom14` scores all-resolved-atom contact preservation. `SG-atom14`
uses target-centered `6A` atom spheres, local superposition, and residue
fractions below `2A` and `4A` local RMSD. `SC-atom14` uses residue identities
to score chi1/chi2 side-chain dihedrals with symmetry and burial weighting.
`MolProbity-clash-atom14` scores atom-name-aware heavy-atom overlaps above
`0.4A`. `BB-atom14` scores phi, psi, and omega dihedral agreement with equal
angle-class weighting. `DipDiff-atom14` scores local three-residue C-alpha/O
distance windows. `ASE` is not included because the submission contract does
not ask for model confidence estimates.
`reLLG_lddt` is not included because it requires crystallographic
molecular-replacement scoring. C-alpha, GDT_TS, and backbone atom14 lDDT remain
diagnostic outputs.

Hidden leaderboard rank is track-specific. `limited` and `research_large` use `foldscore_auc_hidden`, computed across official checkpoint predictions over the fixed sample budget. `unlimited` uses the final hidden FoldScore.
