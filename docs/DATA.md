# nanoFold Data Guide

This document is the official data contract for nanoFold. It explains what data is used, how it is downloaded, how official splits are generated, how raw files become model inputs, and exactly what predictions must look like.

The short version:

- Public participants get train and public validation data through `scripts/setup_official_data.sh`.
- Maintainers keep the published manifests fixed. Source-qualified candidates use explicit `--sampling-policy cath-v4` in the workflow in [Dataset sampling](DATA_SAMPLING.md).
- Public train/validation manifests are committed.
- Hidden manifests, hidden features, hidden labels, hidden fingerprints, and hidden locks live under `.nanofold_private/` and must never be committed.
- Models receive processed feature tensors and must return `pred_atom14` with shape `(B, L, 14, 3)`.

Training requires processed features and labels for every train and public-validation manifest entry, including local research runs. Missing files cannot silently shrink either population. Public-validation submission hooks receive features only; the trainer retains labels for scoring and reports valid, nonfinite and missing counts for each metric. See [training and validation reproducibility](API.md#training-sampling-and-multiple-gpus).

## 1. Data Sources

Official nanoFold data is built from public structure and alignment sources, then frozen into manifests, NPZ files, and fingerprints.

| Source | Used for | Why it is used |
|---|---|---|
| OpenFold/OpenProteinSet RODA bucket, `s3://openfold` | Chain cache metadata, duplicate-chain map, per-chain MSA assets | Provides a public, reproducible AlphaFold-style training-data foundation without asking competitors to run external retrieval. |
| OpenFold `chain_data_cache.json` | Candidate universe, sequence, length, resolution, source lock hashing | Gives a single official chain metadata table for deterministic split generation. |
| OpenFold `duplicate_pdb_chains.txt` | Duplicate-chain fallback during per-chain asset download | Lets the downloader reuse equivalent chain assets when OpenProteinSet stores a duplicate under a representative chain. |
| OpenProteinSet `s3://openfold/pdb_mmcif.zip` | Pinned historical structure members for maintainer regeneration | Selective retrieval binds archive identity and verifies member bytes without downloading the whole archive. |
| OpenFold per-chain `uniref90_hits.a3m` | Official MSA input feature | Gives every model the same evolutionary input while keeping external MSA retrieval out of official runs. |
| RCSB mmCIF files, `https://files.rcsb.org/download/<PDB>.cif` | Atomic labels for C-alpha and atom14 supervision/scoring | mmCIF is the authoritative coordinate source used to build geometry labels. |
| Original cluster TSV and full-group TSV/receipt | Separate original unit membership from indivisible full-graph split groups | Bind the complete eligible universe, graph search provenance, group coverage and deterministic replay. |
| Representative-eligibility receipt and approved group-exclusion catalog | CATH v4 source validation and retained-group coverage | Bind ranked raw-query/strict historical-structure checks and exact approved whole-group exclusions without changing graph connectivity. |
| Pinned CATH 4.4.0 domain list | CATH hierarchy for new sampling candidates | Preserves Class, Architecture, Topology, Homologous superfamily and all domains; unannotated chains remain unknown. |
| SCOPe classification file | Optional annotation audit | Retains source classifications without treating them as CATH assignments. |
| ECOD domain file | Optional annotation audit | Uses named source columns to map PDB domains to author chains; excludes non-PDB predicted domains and invalid structures. Cross-chain domains require corroborating chain ranges. Does not impute missing CATH labels. |
| Optional RCSB REST metadata | Extra split audit metadata when explicitly enabled | Can enrich structural metadata, but the default bulk split path does not require per-chain REST calls. |
| Pinned feature/processability exclusion lists | Split eligibility | Prevents known missing-assets or unprocessable-label chains from entering official manifests. |

Official templates are disabled. The schema still carries template feature tensors, but official preprocessing uses `T=0`, so template lookup cannot decide the leaderboard. Template-enabled research preprocessing excludes every hit from the target PDB entry, including aliases and other chains. It requires auditable polymer residue indices and matching sequence identities; missing coordinates preserve their sequence gaps, and ambiguous mappings are omitted.

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

The public train/validation tensors can also be published as a Hugging Face
dataset with one row per chain and one column per processed tensor:

```bash
python -m pip install -U datasets huggingface_hub hf_xet pyarrow
hf auth login

python scripts/upload_hf_public_dataset.py \
  --repo-id <org-or-user>/nanofold-public \
  --private \
  --cache-dir .nanofold_hf_cache
```

The uploader reads only the public train and validation manifests, expands the
processed feature/label NPZ fields into typed Hugging Face columns, pushes the
result as Parquet shards, and uploads a dataset README plus public manifest and
fingerprint metadata. If `eval.yaml` is present, the uploader also publishes it
at the dataset repo root so the Hub can treat the public validation split as the
`nanofold` benchmark task. That task reports mean FoldScore over the fixed
public validation chains; official hidden ranking remains sealed and
track-specific.

Run `--dry-run --readme-out hf_dataset_export/README.md` first to validate the
public inputs and inspect the generated dataset card without contacting Hugging
Face.

Publication requires the reviewed public manifest/source-lock bytes, correct
counts and union, and a matching fingerprint of the actual tensors. Auxiliary
publication metadata must come from the reviewed public paths. Staged private
candidates cannot be uploaded through this command before deliberate release
promotion. Export caches bind content digests, and inputs are checked again
before publication. `--max-examples-per-split` is restricted to local inspection;
a truncated debug export cannot be published as the full benchmark.

Hugging Face benchmark aggregation is still a beta feature. The root
`eval.yaml` declares NanoFold's intended `nanofold` evaluation framework and
the `nanofold` task. If the Hub requires allow-listing for the dataset to
display the Benchmark tag, request HF registration rather than changing the
metric semantics to fit an unrelated evaluation framework. Model-side
evaluation-result pull requests should use `dataset.task_id: nanofold`.

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

`scripts/pin_hidden_assets.py --track-id <track>` validates the actual manifest,
chain count, fingerprint and feature/label bytes before atomically writing a
lock. A caller-supplied count cannot override the manifest. Strict raw-source
locks require nonempty, disjoint public splits with an exact union, complete
source evidence and real nonempty assets.

## 2. How Data Is Downloaded

Participants should use the public setup script:

Setup dry runs validate inputs and print commands without creating outputs.
Downloads stage each attempt separately, so a failed transfer cannot become a
completed source on retry. Public, hidden and raw output roots must not overlap.

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
   The chain cache must match the committed source-lock hash. Supply
   `--chain-data-cache <path>` to reuse that pinned cache.
3. Downloads per-chain OpenFold assets for the public union manifest, `data/manifests/all.txt`.
   - The official default MSA file is `uniref90_hits.a3m`.
   - Template hits are skipped when `--disable-templates` is set.
4. Downloads the manifest mmCIF subset from RCSB when `--mmcif-mode subset` is used.
5. Runs `scripts/preprocess.py` once on the verified public union, `all.txt`, so
   the MSA filtering audit includes every train and public validation chain.

Direct URL downloads in `scripts/prepare_data.py` time out after 60 seconds per
attempt. Set `--download-url-timeout-seconds <seconds>` to adjust this limit;
values at or below zero disable it. Retry count and retry delay still apply.
This option covers direct metadata and mmCIF URLs, not AWS CLI transfers.

Maintainers can retrieve a manifest's historical structures from the OpenProteinSet
archive with explicit archive pins. Set `ARCHIVE_BYTES`, `ARCHIVE_ETAG` (including
the ETag's quotes), and `ARCHIVE_VERSION_ID` from verified source metadata:

```bash
python scripts/fetch_openproteinset_mmcif.py \
  --manifest "$PRIVATE_MANIFEST" \
  --out-dir .nanofold_private/pinned-structures \
  --archive-size "$ARCHIVE_BYTES" \
  --archive-etag "$ARCHIVE_ETAG" \
  --archive-version-id "$ARCHIVE_VERSION_ID"
```

The parent directory must exist and the output root must be fresh. Version ID
`null` requests the S3 null version; `absent` requires the version header to be
absent. Every range must match the pinned object, bounds and length. The retriever
checks ZIP64 metadata, member headers, decompressed sizes and CRCs, and records
archive, manifest, implementation, range and member hashes in a private receipt.
It limits ranges to 2 MiB, the directory and compressed members to 64 MiB each,
and decompressed members to 256 MiB. Changed sources or invalid members fail.
Transient connection failures, HTTP 429 and server errors receive at most four
attempts per range, with 1, 2 and 4 second backoffs. Each retry requests the same
pinned range; identity, header, length and content validation failures are not
retried. Attempt outcomes remain in the private range evidence.
Use the resulting `pdb_data/mmcif_files` directory as preprocessing's
`--mmcif-root`. Retrieval does not validate chain sequences, coordinate coverage
or group coverage; those gates remain required. Archive modification time does
not establish the structures' revision cutoff. Current RCSB files can differ from
the archived members, so record the exact source bytes used for each rebuild.

For an exploratory preprocessing audit, write a separate manifest containing
only chains with both feature and label NPZ files:

```bash
python scripts/write_processed_manifest_report.py \
  --source-manifest data/manifests/val.txt \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels \
  --output-manifest .nanofold_private/audits/processed_val.txt \
  --report .nanofold_private/audits/processed_val_report.json \
  --require-complete
```

The report counts missing, feature-only, and label-only chains and groups
preprocessing error messages. `--max-examples` limits example lists (default
50). Both files are written before an incomplete-data failure; no available
examples always causes failure. These are local audit outputs. Keep the
official manifests fixed: official execution requires every requested chain
and still verifies the NPZ schema and fingerprint.

Maintainer asset refreshes require `--skip-manifest-regen`; published manifests remain fixed. `scripts/full_official_data_refresh.sh` prepares two private filters from raw MSAs: training sources against public-plus-hidden validation targets, and public-validation sources against hidden targets. One public `all.txt` preprocessing pass assigns the correct filter to every chain. Hidden preprocessing receives no row filter. Query row 0 is always preserved, and row removal precedes the depth cap. Hidden-derived filters and policy files remain private under the candidate asset-rebuild directory.

The wrapper accepts an existing pair with `--train-msa-row-filter` and `--val-msa-row-filter`; `--skip-msa-row-filter-build` requires that pair. `--msa-filter-policy-out` chooses the private policy output (default `.nanofold_private/asset_rebuild/msa_row_filters/public_directed_policy.json`); automatic filters are written alongside it as `train_filter.json` and `val_filter.json`. `--msa-filter-checkpoint-dir` selects the search workspace, whose default is the policy directory's `checkpoints/`, with separate `train/`, `val/`, `residual_train_val/` and `residual_public_hidden/` searches. Filter construction and residual audits all use exhaustive checkpointed searches. The wrapper rejects scalar `--msa-row-filter` and `--msa-row-filter-out` arguments. `--skip-hidden` skips hidden asset preparation and the hidden residual audit, but the hidden manifest and pinned cache targets remain required to sanitize both public splits. The wrapper rejects `--skip-setup`; use `--resume-preprocess` for validated reuse.

The direction is part of the data contract:

| Source split | Target homologs excluded from non-query MSA rows |
|---|---|
| Training | Public validation and hidden validation |
| Public validation | Hidden validation |
| Hidden validation | None |

No filter is built against its source split's own target set or an earlier split. A row that matches a forbidden later-split target is still removed. Hidden MSAs still undergo normal query/format validation, merging, deduplication and depth capping. The same target-homology thresholds apply to both filters. Standalone single-filter research preprocessing remains available, but a single undirected filter does not encode this three-split policy.

MSA split audits live in `scripts/audit_msa_split.py`. The script reports
processed feature availability, MSA-depth quantiles and binned Jensen-Shannon
divergence, target chain/cluster overlap, exact processed MSA-row overlap,
optional MMseqs target-to-target and target-to-MSA homology, and optional raw A3M hit-identifier
overlap. Its default JSON output avoids raw sequences and hidden identifiers;
depth CSVs also omit identifiers by default. Use identifier flags only when the output may contain those identities. `--strict` fails incomplete coverage, split overlap and exact held-out target leakage; `--run-mmseqs` adds separate thresholded searches against source targets and non-query MSA rows of at least 20 residues. Either kind of hit fails a strict audit. Missing or malformed MMseqs output is an error. Use `--val-features-dir` when held-out features have a separate root. Shared background MSA rows alone are descriptive overlap, not evidence of target leakage.

If asset preprocessing is interrupted, rerun the same full-refresh command with `--resume-preprocess`. For direct `scripts/preprocess.py` commands, add `--skip-existing`.

Resume requires the same raw inputs, pinned chain-cache bytes, split policy, effective filters, preprocessing implementation and configuration. New outputs contain source-content and configuration hashes; a changed or missing provenance record requires a fresh output directory. Existing files are validated before they can be skipped, including exact target equality to the selected cache sequence when cache binding is enabled.

## 3. How Dataset Splits Are Determined

The committed train and public-validation manifests, fingerprints and source lock define the published benchmark. They have not been resampled. Changing structural annotations or preprocessing does not update that contract automatically. The regeneration wrapper refuses to overwrite it; the full-refresh wrapper can rebuild and audit its assets with `--skip-manifest-regen`. Rebuilt fingerprints and raw/hidden asset locks default to `.nanofold_private/asset_rebuild/leaderboard/`. Published and sealed reference paths, including symlink/hardlink aliases, cannot be used as candidate outputs.

Source-qualified candidate datasets explicitly select `--sampling-policy cath-v4` and metadata schema 2; staging otherwise defaults to `cath-v3`. `scripts/build_homology_groups.py` supplies the original complete group map and provenance receipt; `scripts/build_representative_eligibility.py` validates representatives against raw queries and uniformly pinned historical structures. Staging requires original `--cluster-tsv`, `--split-group-tsv`, `--split-group-receipt`, `--representative-eligibility-receipt` and `--approved-group-exclusions`. It builds both source locks, independently replays selection and requires complete cache-target audits before issuing a completion receipt. `cath-v2` and `cath-v3` remain explicit policies with separate replay semantics. Complete metadata is required; debug limits and permissive amino-acid filters cannot enter release sampling. See [Dataset sampling](DATA_SAMPLING.md) for commands, predicates, source evidence and scientific limits.

Qualification stores input bindings, durable per-unit results and failure diagnostics in a fresh `<output.name>.diagnostics/` sibling directory. Keep it private and preserve failed attempts. These diagnostics neither replace the independently validated receipt nor authorize exclusions or automatic resume.

`scripts/setup_custom_data.sh` consumes a staged public manifest directory and
its source lock, with required `--chain-data-cache <path>` and explicit custom
raw/features/labels directories. It checks the cache hash and locked public
manifests before downloading and never regenerates them.
This convenience downloads current RCSB structures; CATH v4 instead preprocesses the exact historical bytes bound by its eligibility receipt, using an explicit `preprocess.py --mmcif-root`. The fixed projection predicate differs from the CLI `--strict` exact-coordinate-sequence option. See the v4 command in [dataset sampling](DATA_SAMPLING.md#stage-a-new-candidate). Neither path replaces hidden preparation or the strict audits.
Both setup wrappers and the full-refresh wrapper preprocess the public union in
one call. Hidden preprocessing uses its separate private output roots and audit.

The generation path:

1. Verifies the chain cache and downloaded annotation bytes against their source locks. Metadata and sampling filters must agree.
2. Filters standard amino-acid chains by length, resolution and pinned asset exclusions. Targets are individual chains; biological monomer status is required only with `--require-monomer` and cannot be inferred from missing metadata.
3. Retains every CATH domain's C.A.T.H hierarchy. Multi-domain chains retain their complete unique label sets. Missing annotations are an explicit stratum. Secondary fractions remain unknown unless supported by pinned structural annotations.
4. Forms original units from assigned clusters, exact sequences and PDB entries, preserving the complete homology graph over those units. It validates members in unchanged deterministic quality order and keeps the first passing raw-query/strict historical-structure member, with a complete failed prefix or exhaustive unit-failure record.
5. Excludes a full group only when every original unit is exhausted and the exact group set matches the approved catalog. Missing assets and pending acquisition do not prove exhaustion. The graph is not rebuilt after exclusion. Strict projection thresholds remain 0.90 identity, 0.70 coverage, 0.70 aligned fraction and 32 valid Cα positions; partial-label relaxation is not allowed.
6. Covers every retained group, keeping each group within one split. Joint capacity-aware allocation chooses public-plus-hidden evaluation groups, one target per group. The public seed fixes group ownership; training receives one mandatory anchor per remaining group before extra validated unit representatives fill its count.
7. Uses the private salt only for member choice and ordering inside fixed hidden groups. Held-out quotas use anchors; a chosen member can have a different stratum, and the actual distribution must pass quality checks. Reports distinguish original and validated populations, exclusions, coverage, group concentration and structural composition.

The approved candidate retains 8,848 of the original graph's 8,856 groups after eight documented whole-group exclusions. At 10,000 train, 1,000 public-validation and 1,000 hidden-validation targets, training needs 6,848 anchors plus 3,152 additional validated unit representatives. These are candidate requirements, not evidence of complete processed assets or release adoption. Qualified capacity and concentration must be measured from the validated pool. Infeasible coverage, quotas or sizes fail instead of silently deleting groups or reducing counts.

Both source locks bind original assignments and graph, eligibility evidence, the approved catalog, source inputs, manifest hashes and counts. Original graph counts and representative digests remain distinct from retained-group and validated-pool counts/digests. Verification reconstructs ranking, eligibility and selection, then independently checks every retained group's coverage, one-split ownership and at most one selected target per original unit. Excluded or unselected members cannot erase bridging graph connections. Hidden-only generation retains the fixed plan; later source failures require a deliberate new candidate rather than silent resampling.

Full retained coverage makes hidden groups inferable from the known original universe, approved exclusions and public complement; singleton hidden targets are then inferable. The salt does not provide membership secrecy. Hidden IDs, source evidence, labels and detailed files remain private, and labels remain unavailable to submission inference. Ordinary training weights selected targets equally: a group with `m` selected targets contributes `m` exposures per complete traversal before batch/tail drops. Inverse-group weighting is a separate ablation.

The full groups are defined by a recorded alignment/search contract; they do not establish absence of every form of biological homology. Selected-target and residual MSA audits remain mandatory, including after preprocessing. Balanced CATH Architecture and complete group coverage do not establish improved model performance or unseen-fold/superfamily generalization. Source annotations are incomplete, and annotation presence does not establish residue-level domain coverage.

The published source lock's broad-class distributions describe its frozen selection. They must not be interpreted as measurements of full CATH Architecture balance or measured secondary-structure fractions. New CATH reports apply only to newly generated candidates until a dataset release is independently audited and adopted.

## 4. How Input Data Is Prepped

Preprocessing is handled by `scripts/preprocess.py`.

The official and custom setup wrappers pass the pinned chain cache into
preprocessing. Direct research invocations can opt in with
`--chain-data-cache <path>`. Every selected chain must have one unambiguous,
canonical amino-acid sequence and consistent integer length fields in that
cache. Missing records, conflicting sequence fields and duplicate JSON keys
fail before output directories are created. The acquired, ungapped A3M query
must exactly equal that sequence before coordinate projection; passing the
projection thresholds alone cannot establish target identity. The cache byte
hash enters the preprocessing configuration, and each pair's source hash also
binds its selected chain ID and expected sequence. Adding or changing this
binding requires fresh outputs. The pinned cache must be separate from processed
output directories and their aliases.

For each chain ID, preprocessing:

1. Resolves the encoded OpenProteinSet chain directory.
2. Reads one or more A3M files.
3. Removes query-gap columns from the MSA.
4. Selects the effective filter from `--msa-filter-policy` for each public chain,
   or uses `--msa-row-filter` for a standalone single-filter job. The two options
   are mutually exclusive. Only non-query rows are removed; query row 0 is kept.
   Hidden preprocessing supplies neither option.
5. Merges and deduplicates MSA rows.
6. Caps raw MSA depth with `--max-msa-seqs` before writing NPZ files.
7. Loads the matching mmCIF file.
8. Extracts chain atom coordinates into the canonical atom14 layout.
9. Aligns the full deposited polymer sequence to the MSA query, then places observed coordinates using `label_seq_id`. Missing coordinates retain their polymer positions, including within repeated sequences. All optimal alignments must induce the same mapping for every observed residue; differences confined to unresolved positions are allowed. More than 1,024 optimal alignments fail closed because uniqueness has not been established. Without polymer context, sequence-only projection must be unambiguous.
10. Projects atom14 coordinates onto query positions, masking every label at a sequence mismatch.
11. Rejects chains that fail projection quality thresholds, including the valid-CA minimum in strict alignment mode.
12. Validates and writes one feature NPZ and one label NPZ per chain, with matching configuration and source-content hashes.
13. Records schema-3 run provenance in `preprocess_meta.json`. This run metadata is required for fingerprinting provenance-bearing pairs; removing or downgrading it is an error. Resume verifies raw inputs and existing files before skipping work. Chain resolution prefers author IDs, then explicit polymer label IDs, then a unique sequence match; ambiguous matches fail. Coordinate selection follows the policy below.

### Coordinate selection

The first deposited model anchors residue identity and observation support.
Other deposited models are eligible only when their resolved chain, residue
identities, polymer positions and complete atom14 masks match that anchor.
The parser selects one whole eligible model by the fewest peptide C–N distance
violations, then CA distance violations, then deposited model order. Coordinates
are never mixed between models. This conservative comparison does not replace
the anchor with a model having different observed atom coverage.
An invalid first model fails parsing. Malformed or noncomparable later models
are ineligible. Model comparison stops at the first eligible model with no
scored backbone violations; it does not validate every model in the source.

Within each residue, a candidate contains one named alternate location plus
shared blank atoms. Missing atoms are never borrowed from a different named alternate.
Only positive-occupancy atoms are observed; zero occupancy retains the residue
and its sequence position but contributes no atom label. Negative or nonfinite
occupancy and nonfinite observed coordinates invalidate that model. Deposited zero-occupancy
coordinates can represent atoms with unreliable locations, as described in the
[wwPDB processing procedures](https://www.wwpdb.org/documentation/procedure).

Before comparing geometry, selection maximizes the number of observed N, CA, C
and O atoms in each residue. Equal-count masks are resolved by the fixed
CA, N, C, O presence order. Every remaining candidate must have that exact
backbone mask, so geometry cannot improve by hiding a bond endpoint. A chain
dynamic program then minimizes the number of peptide C–N distances outside
1.0–1.8 Å, followed by CA distances outside 2.5–4.5 Å. Only known consecutive
deposited `label_seq_id` positions in the same polymer subchain contribute;
author numbering and compact query adjacency do not create connections.
Equal-geometry choices maximize summed per-residue mean named-alternate
occupancy, with sorted alternate identifiers and predecessor order resolving
ties deterministically.

This selects deposited coordinates without interpolation or idealization.
Alternate letters need not match across residues: the
[mmCIF alternate-site specification](https://mmcif.wwpdb.org/dictionaries/mmcif_std.dic/Categories/atom_sites_alt.html)
allows both correlated and uncorrelated alternatives. Local backbone continuity
does not prove a globally valid conformation, and unavoidable source geometry
outliers remain diagnostic. No target is excluded merely for these distance
outliers. Atom-selection implementation hashes bind qualification and
preprocessing; a changed parser requires new qualification receipts and fresh
feature/label outputs.

Preprocessing validates nonempty, unique, well-formed manifest IDs before
creating outputs. Feature and label roots must be separate and cannot contain
one another. This also keeps label files out of a features-only directory mount.
`scripts/filter_openproteinset.py` validates complete feature/label pairs before
admitting processed candidates; `--min-valid-ca` defaults to 32 observed CA
positions. Unknown resolution is recorded explicitly. Cluster TSVs must have
unambiguous assignments and cover every evaluated chain.

For an independent whole-corpus target check, `scripts/audit_target_identity.py` accepts repeated `--manifest` paths with exact `--expected-counts`, the pinned `--chain-data-cache`, and `--raw-root` and/or repeated `--features-dir` roots. It compares every selected raw query and processed target to its cache sequence, validates processed feature schema, rehashes audited inputs and writes a fresh private receipt via `--out-dir`. It cannot pass from cache records alone or tolerate missing selected targets. Raw-only checks are available before preprocessing; label/geometry and homology validation remain separate.

Directional row filtering protects the two evaluation boundaries while retaining each split's permitted evolutionary context. The threshold is 30% sequence identity with 80% bidirectional coverage. Training excludes homologs of public and hidden validation targets; public validation excludes homologs of hidden targets; hidden validation is unfiltered. Each filter scans only its intended source split. Query row 0 is retained even if its sequence appears in a filter.

`scripts/build_msa_filter_policy.py` binds `--train-manifest`, `--val-manifest`, `--hidden-manifest`, `--train-msa-row-filter` and `--val-msa-row-filter` into the private `--output` policy. It requires duplicate-free disjoint manifests, matching chain-cache bytes, and complete exhaustive raw-A3M filters at 30% identity and 80% bidirectional coverage. Keep both completed checkpoint directories: the policy loader verifies their source partitions and full job evidence, then rederives exclusions from alignments and exact sequence intersections. Unbatched filters remain available for standalone exploration but cannot satisfy this production policy. `preprocess.py --msa-filter-policy` requires `--chain-data-cache` and the exact public union; MSA names and consumed source paths/hashes must match the filter evidence. Hidden preprocessing omits both policy and scalar-filter options. See [the complete commands](DATA_SAMPLING.md#bounded-resumable-exhaustive-msa-searches).

Filtering runs before `--max-msa-seqs`, so deeper permitted rows can fill positions freed by removal. Public train and validation share one output root and one bound preprocessing policy, while each chain records `msa_row_filter_sha256`, `msa_rows_before_filter` and `msa_rows_removed_by_filter`. The feature root contains `msa_row_filter_audit.train.json` and `msa_row_filter_audit.val.json`. Policy provenance in `preprocess_meta.json` retains the policy digest and public manifest/filter digests without hidden manifest references. Hidden features use a separate unfiltered preprocessing configuration. Fingerprints bind these configurations, and changed policy/filter/manifest inputs require fresh outputs.

Use `scripts/audit_msa_split.py` before and after filtered regeneration to
compare feature availability, MSA depth bins, exact row overlap, optional MMseqs
homology hits, and available raw A3M hit-identifier overlap. Hidden validation
audits should be reported only as aggregate counts.

For standalone checks of existing research assets, `scripts/build_msa_row_filter.py` accepts repeated `--source-processed-features-dir` paths and scans the requested source manifest. Directed public preprocessing requires complete raw-A3M filters so removal occurs before depth capping.

Standalone MSA search is heuristic by default. Add `--exhaustive` to the filter builder and `--strict --run-mmseqs --exhaustive` to the split auditor for all collected query/source pairs under the recorded protein local-alignment contract. This disables prefiltering, masking and composition correction and uses E-value limit `1e308`; it does not enumerate every possible alignment or cover all biological homology. Non-query rows shorter than 20 residues are outside the search. Unbatched exhaustive searches require `--max-seqs` at least as large as the full source population; checkpointed searches require capacity for each complete source batch. Reporting caps cannot silently truncate results. The full-refresh wrapper enables exhaustive checkpointed searches for both directed filters and both residual audits.

Use the filter builder's `--exhaustive --inventory-only --output <private-inventory.json>` with the intended source and held-out manifests before an expensive search. It scans rows without running or requiring MMseqs and reports unique source/held-out counts, `all_pair_comparisons`, `required_exhaustive_max_seqs` and `source_complete`. Cost grows with the product of those populations; the count is not a runtime estimate. Missing sources do not establish complete coverage, and the inventory cannot be consumed as a row filter. See [dataset sampling](DATA_SAMPLING.md#before-adopting-a-dataset) for the search scope and release gates.

For a resumable exhaustive search, add `--checkpoint-dir <private-search-directory>` together with `--exhaustive` (and `--run-mmseqs` for the auditor). Repeating the same command verifies the immutable plan and completed job receipts, then reruns incomplete attempts. Defaults and hard maxima are 1,000,000 source records and 256 queries per job, controlled by `--source-batch-size` and `--query-batch-size`. `--search-disk-reserve-gib` defaults to 100; `--search-max-rss-gib` defaults to 48 and measures sampled caller plus native process-tree RSS, not an OS-enforced cap. The plan binds complete FASTAs, partitions, parameters, binary and implementation. Final output requires every job and final input verification. Total work still grows with all query/source pairs. See [bounded exhaustive searches](DATA_SAMPLING.md#bounded-resumable-exhaustive-msa-searches) for scope and resource limits.

For research assets without paired schema-3 provenance,
`scripts/filter_processed_msa_features.py` can apply a filter to existing
features in a separate output root. It preserves query rows, binds reuse to
source bytes and filter identity, and atomically replaces each output archive.
It rejects in-place transformations and provenance-bound schema-3 inputs;
those require preprocessing from raw data into fresh feature and label roots.
Postprocessing cannot recover rows discarded by the original depth cap.

Because large MMseqs searches can expose additional residual hits on a
verification pass, use `scripts/extend_msa_row_filter_from_audit.py` as a
closure step when `scripts/audit_msa_split.py --run-mmseqs` reports
post-filter source rows from a single-alignment audit. Batched reports are rejected; use complete exhaustive filtering and a fresh residual audit for that path. Supply `--audit-summary` from the same supported audit. The
closure helper checks the ordered source-record digest and MMseqs output hash
before mapping anonymous `trainmsa_N` headers to sequence hashes. Changed or
reordered source inputs require a fresh audit. Follow extension with another
filtering/audit cycle; zero hits describes that configured search only.

Use `scripts/write_msa_row_filter_source_lock.py` after filtering to create a
compact audit/source-lock summary. By default it redacts private paths and
private manifest hashes and omits raw sequences, excluded sequence-hash lists,
and per-chain processed-feature rows while retaining thresholds, normalized
MMseqs settings, manifest counts, public manifest hashes, filter hashes, and
aggregate row-removal counts. For directed public preprocessing, pass `--msa-filter-policy`, `--split train` or `--split val`, the matching `--msa-row-filter` and that split's `--processed-filter-audit`; write one summary per split.

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
| `msa_row_filter_sha256`, `msa_rows_before_filter`, `msa_rows_removed_by_filter` | scalar | mixed | When row filtering is enabled, records the chain's effective filter hash and row-removal counts. |
| `preprocess_config_sha256`, `source_inputs_sha256` | scalar | string | Schema-3 provenance shared with the label NPZ; used for resume and fingerprint validation, not passed to models. |
| `resolved_author_chain_id`, `chain_mapping_method` | scalar | string | Resolved coordinate chain and mapping evidence (`author_id`, `label_id`, or unique `sequence` match), shared with labels. |

Label NPZ keys:

| Key | Shape | Dtype | Meaning |
|---|---:|---|---|
| `ca_coords` | `(L, 3)` | `float32` | True C-alpha coordinates. |
| `ca_mask` | `(L,)` | `bool` | True for selected positive-occupancy C-alpha coordinates after query projection. |
| `atom14_positions` | `(L, 14, 3)` | `float32` | True atom14 coordinates in canonical AF2 slot order. |
| `atom14_mask` | `(L, 14)` | `bool` | True for selected positive-occupancy atom coordinates after query projection. |
| `residue_index` | `(L,)` | `int32` | Contiguous `0..L-1` positions. |
| `resolution` | scalar | `float32` | Structure resolution, or `0.0` when unknown. |

Integrity checks enforce amino-acid token ranges, nonnegative deletion counts, query/MSA consistency, finite coordinates, CA/atom14 consistency, and feature/label length, index, identity and provenance agreement. Schema-3 files must contain a scalar `chain_id` matching the requested manifest chain; a matching pair under the wrong filename is rejected. An interrupted pair replacement is detected on validation; the two files are not a single filesystem transaction.

Feature archives containing supervision keys are rejected. Requiring labels
also requires a label directory; a fingerprint cannot declare complete labeled
coverage without one. Each manifest must contain unique chains, and splits in
one fingerprint must be disjoint. Fingerprint `all.txt` separately from its
constituent train and validation manifests.

The data loader then:

- loads per-chain NPZ files with `nanofold.data.ProcessedNPZDataset`
- crops sequences longer than `data.crop_size`
- samples MSA rows to at most `data.msa_depth`
- pads variable-length examples inside each batch
- emits `residue_mask` for residue padding and `msa_mask` for MSA row/residue
  padding; real biological gaps remain valid MSA cells
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
    "msa_mask": torch.ones((1, 3, 4), dtype=torch.bool),  # no batch padding here
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
  --official --track limited --split hidden_val \
  --manifest .nanofold_private/manifests/hidden_val.txt \
  --pred-out-dir runs/<run_name>/hidden_predictions \
  --checkpoint-steps 0,1000,2000,5000,last \
  --fingerprint .nanofold_private/leaderboard/official_hidden_fingerprint.json \
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

Hidden leaderboard rank is track-specific. `limited` and `research_large` use `foldscore_auc_hidden`, computed across official checkpoint predictions over the fixed sample budget. `unlimited` fixes effective batch size at `8` but uses the final hidden FoldScore because it has no shared max-step or sample budget.

The dataset loader rejects mismatched feature/label identities, residue counts, residue indices, nonboolean masks, nonfinite coordinates and inconsistent C-alpha/atom14 supervision. Label-only residue indices cannot change inference features. All training/evaluation assets must pass the same integrity checks. [Modal training](MODAL_TRAINING.md) describes public-only volume and bundle validation.
