# Dataset sampling and release audit

The published benchmark is defined by its committed manifests, locks and fingerprints. New candidates use `cath-v2` and metadata schema 2 in fresh, ignored staging directories. Generating a candidate does not adopt it as the official benchmark or establish improved model performance.

## CATH annotations

CATH classifies domains at four levels:

| Level | Example | Meaning |
|---|---|---|
| C | `1` | Class |
| C.A | `1.10` | Architecture |
| C.A.T | `1.10.10` | Topology |
| C.A.T.H | `1.10.10.10` | Homologous superfamily |

The [CATH domain-list specification](https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/README-cath-list-file-format.txt) defines these as distinct columns. Broad labels such as alpha and beta summarize Class; they do not identify Architecture. Classification also cannot supply quantitative helix, sheet or coil fractions.

Schema 2 preserves `cath_domains` with domain ID and all four identifiers. Repeated domains remain separate records. Sorted unique architecture, topology and superfamily sets support chain-level allocation and diagnostics without choosing the first domain. The architecture-set stratum does not encode domain order or multiplicity; multiplicity is retained separately for audit. Unusual class codes remain available without inferred secondary fractions.

`cath_annotation_status` distinguishes annotated from unknown; `cath_mapping_status` records mapping ambiguity. An annotated chain has at least one mapped domain; this does **not** assert complete residue coverage. Domain boundaries and coverage require an additional pinned boundary source. SCOPe/ECOD remain optional audit annotations and do not fill missing CATH labels. Secondary fractions are null unless supported by the optional pinned structural-annotation source (`--include-dssp-mmcif`) with complete finite-CA polymer coverage. Unresolved residues are not counted as coil.

Chain IDs follow OpenFold's author-chain namespace. Matching is case sensitive and does not truncate multi-character names. CATH's legacy chain `0` is ambiguous without an explicit crosswalk; unknown mappings remain unknown. mmCIF structural annotations use author/label crosswalks and polymer residue positions.

## Eligible population and sampling unit

The default candidate is a single chain with standard amino acids, a consistent sequence length of 40–256, and resolution at most 3 Å when known. Missing/nonpositive resolution is unknown rather than exceptional coordinate quality. Feature and processability exclusions must exist and are hashed. Metadata records the full eligible universe and any debugging sample limit; `cath-v2` refuses truncated metadata or permissive unknown-amino-acid filters. These gates describe the available source population, not a representative sample of all proteins.

Single-chain targets are not necessarily biological monomers. `--require-monomer` requires an explicit `oligomeric_count=1`; missing assembly information cannot satisfy it. Use the same option consistently in download, metadata and sampling commands. Asymmetric-unit chain count is not biological assembly evidence; see the [RCSB assembly documentation](https://www.rcsb.org/docs/general-help/biological-assemblies).

Connected components join assigned MMseqs clusters, exact duplicate sequences and same-PDB chains transitively. Each contributes one representative, ranked by resolution, length, release date and deterministic seeded ties. This limits redundancy but also creates a quality preference that must be reported when comparing samplers.

Live MMseqs clustering requests actual alignment identity with `--alignment-mode 3`, `--min-seq-id 0.30` and `-c 0.80`. A precomputed TSV must cover every candidate and cannot contain conflicting assignments. Its bytes and canonical assignments are locked. A cluster map proves membership, not how an externally supplied TSV was produced. Verify its generation provenance before release. The [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2/wiki) distinguishes clustering algorithms, coverage and identity definitions; greedy cluster separation is not a global all-pairs identity bound.

## Allocation and diagnostics

The primary strata are complete unique CATH Architecture-label sets (including unknown), then length bin, then resolution bin. Hierarchical proportional apportionment avoids rounding every sparse Cartesian cell independently. Remainder ties use seeded hashes; equal quotas do not systematically favor lexically early labels. Capacity-aware allocation redistributes unavailable quotas proportionally and fails if total capacity is insufficient.

The joint generator selects public validation, then train, before selecting hidden validation from the remainder. Public membership and order are independent of the hidden salt. The hidden-only generator takes an already fixed `cath-v2` public contract, excludes every connected component touching it, and matches the public distribution subject to remaining capacity. Both refuse existing sampling outputs. Per-file atomic writes and locks make completion auditable, but do not constitute a transaction across several directories.

Reports include:

- Candidate and independent-unit counts, rejection reasons, representative policy and annotation coverage.
- Chain/residue counts, C/A/T/H label breadth, multi-domain counts and label incidence in the reference and each split.
- Split distributions and Jensen–Shannon divergence against the preselection independent-unit population, or the fixed public reference for hidden-only generation.
- Held-out topology and superfamily familiarity: `UNANNOTATED`, `NONE_SEEN`, `SOME_SEEN`, `ALL_SEEN` relative to train. “All” refers to available domain labels, not complete structural coverage.

Current safeguards limit JS divergence to 0.35 for each split with at least 100 examples, and unknown CATH fraction to 0.75. These are failure limits, not evidence that smaller discrepancies are scientifically harmless. Inspect rare-label depletion and coverage changes before adopting a release. The public lock describes public distributions; the complete reference/hidden quality report remains private.

Generation requires both public and private source locks. The verifier checks all public manifest hashes/counts, cache and metadata identity, cluster assignments, policy and private input bindings. It reconstructs complete connected components, including unselected members that could bridge splits, and enforces one selected chain per component for `cath-v2`. Private processability exclusions are checked against locked bytes, canonical IDs and counts; every component touching an exclusion is unavailable to hidden selection. Use `--private-processability-exclusion-list` to relocate the same locked input. The verifier can audit a frozen source lock under that lock's recorded policy; it does not relabel that selection as `cath-v2`.

## Stage a new candidate

Use a fresh directory for every candidate and retain its source lock. The chain cache must already be available. The following defaults match the 40–256/3 Å single-chain policy; MMseqs2 must be installed separately.

```bash
sources=.nanofold_private/cath-v2-sources-01
cache=data/openproteinset/pdb_data/data_caches/chain_data_cache.json

python scripts/download_structure_metadata_sources.py \
  --chain-data-cache "$cache" \
  --out-dir "$sources" \
  --source-lock "$sources/sources.lock.json"

# Provide a maintainer secret of at least 32 characters in this environment.
# Keep the secret and every hidden artifact private.
python scripts/stage_dataset_candidate.py \
  --stage-dir .nanofold_private/cath-v2-candidate-01 \
  --chain-data-cache "$cache" \
  --metadata-sources-dir "$sources" \
  --metadata-source-lock "$sources/sources.lock.json"
```

The stage command builds metadata, invokes the joint generator with both locks, then runs the independent verifier. It creates the directory with owner-only access, retains step logs privately, and refuses existing directories. Paths inside this repository must be under ignored `.nanofold_private/`; external private directories are also supported. `candidate_status.json` becomes `manifests_verified` only after every step succeeds and includes artifact hashes. Failed or interrupted stages must not be adopted; preserve their diagnostics and start a fresh directory. This receipt explicitly leaves processed-asset validation outstanding. The direct builders remain available for specialized workflows.

The secret environment variable is `NANOFOLD_HIDDEN_SPLIT_SALT`; alternatively pass `--hidden-split-salt-file` to the generator. Do not expose the salt or its digest in public artifacts. The downloader pins CATH 4.4.0, verifies existing locked bytes instead of repinning them, and records unavailable optional sources. Source filenames are resolved under `--out-dir` by the downloader and `--metadata-sources-dir` by the metadata builder. Use a new directory/lock for a deliberate source update.

`build_hidden_manifest.py --sampling-policy cath-v2` is available for a fixed candidate public contract; pass its `--lock-file` and public manifest paths, metadata/cluster overrides if relocated, a fresh hidden output directory/private lock, and the secret. The requested hidden size and seed must match the source lock. Published-manifest regeneration wrappers refuse writes; `full_official_data_refresh.sh --skip-manifest-regen` rebuilds/audits frozen assets only.

Prepare the staged candidate's public assets with explicit custom paths:

```bash
stage=.nanofold_private/cath-v2-candidate-01
bash scripts/setup_custom_data.sh \
  --manifests-dir "$stage/public" \
  --manifest-lock-file "$stage/public-source.lock.json" \
  --data-root "$stage/raw" \
  --processed-features-dir "$stage/features" \
  --processed-labels-dir "$stage/labels"
```

This command verifies the locked public hashes/counts and union before downloads,
preserves manifests and locks, and prepares only public assets with templates
disabled. It cannot generate splits; generation flags fail before work starts.
`--dry-run` validates inputs and prints commands without creating outputs. Pass
`--msa-row-filter` for an existing private filter and `--resume-preprocess` for
matching raw-input/configuration provenance. Hidden asset preparation, both
strict homology audits and candidate fingerprints remain required separately.

## Training exposure and resume

A correct static split can still be sampled incorrectly during training. Checkpoints therefore bind the actual chain/weight population and preserve the epoch, consumed-batch cursor, loader generator and each rank's RNG. Resume replays the current epoch prefix before restoring model random state, preventing repeated exposure to epoch-zero examples after interruptions. Worker seeds are rank-specific; the distributed sampling plan stays shared. Unsupported checkpoints and changed sampling contracts fail explicitly. See [the API contract](API.md#training-sampling-and-multiple-gpus) for supported replay and its I/O cost.

## Before adopting a dataset

A release needs complete raw-asset availability and successful preprocessing, with rejected cases analyzed across annotation, length and structural categories. Do not automatically resample after a hidden preprocessing failure or add hidden IDs to public exclusions. Preserve the failed attempt privately and create a deliberate new candidate version if eligibility must change.

Run strict MSA audits for train→public validation and public-all→hidden, using separate held-out feature roots where needed. `--strict --run-mmseqs` searches held-out targets against both source targets and non-query MSA rows of at least 20 residues, and fails on either kind of hit. Both searches request actual alignment identity and validate numeric fields, known IDs, coverage and expected output; missing output cannot count as zero hits. A zero-hit result is limited to those heuristic searches. A filter digest proves filter identity, not absence of residual homology. Audit after filtering and before fingerprinting. Keep identifier-bearing reports private.

Preprocessing changes also require a new asset/fingerprint version. Schema-3 files bind configuration, implementation and raw input bytes; changed provenance cannot be resumed into existing outputs. Their run metadata is mandatory during fingerprint creation, including when someone removes or downgrades it. Validation masks identity-mismatched labels, checks atom/token consistency and detects mixed feature/label pairs. Strict alignment mode still enforces the valid-CA minimum. Optional research templates require auditable polymer residue indices and sequence identity, so missing coordinates cannot shift template positions.

To evaluate whether sampling improves learning, preregister matched comparisons with fixed feature access, training budget, optimization, evaluation set and representative policy. Use several sampling and training seeds. Report mean performance and cluster/architecture-level uncertainty alongside annotation and family coverage. Compare proportional architecture sampling with diversity-weighted or capped-superfamily alternatives as separate experiments. Report topology/superfamily holdouts and temporal holdouts separately; mixing these objectives into the main composition-matched score changes what the benchmark measures. None of the implementation checks establishes a model-quality gain without those experiments.
