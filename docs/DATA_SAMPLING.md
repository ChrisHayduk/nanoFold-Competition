# Dataset sampling and release audit

The published benchmark is defined by its committed manifests, locks and fingerprints. Candidate staging defaults to `cath-v3` with metadata schema 2 and a complete homology group map in fresh, ignored directories. `--sampling-policy cath-v2` remains available for its separately defined one-representative-per-original-unit policy. Generating a candidate does not adopt it as the official benchmark or establish improved model performance.

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

PDB accession prefixes must be lowercase; author-chain case remains significant.
The optional RCSB annotation path resolves author IDs through verified
polymer-entity/label-asym crosswalks. Entry-level metadata alone does not count as
chain annotation coverage. SCOPe small-protein and coiled-coil classes remain
distinct audit labels, not inferred coil fractions.

## Eligible population and sampling unit

The default candidate is a single chain with standard amino acids, a consistent sequence length of 40–256, and resolution at most 3 Å when known. Missing/nonpositive resolution is unknown rather than exceptional coordinate quality. Feature and processability exclusions must exist and are hashed. Metadata records the full eligible universe and any debugging sample limit; Both CATH policies refuse truncated metadata or permissive unknown-amino-acid filters. These gates describe the available source population, not a representative sample of all proteins.

Single-chain targets are not necessarily biological monomers. `--require-monomer` requires an explicit `oligomeric_count=1`; missing assembly information cannot satisfy it. Use the same option consistently in download, metadata and sampling commands. Asymmetric-unit chain count is not biological assembly evidence; see the [RCSB assembly documentation](https://www.rcsb.org/docs/general-help/biological-assemblies).

CATH v3 distinguishes the **selectable target pool** from the **groups that cannot cross splits**:

- Original units join assigned MMseqs clusters, exact duplicate sequences and same-PDB chains transitively. Each contributes one quality-ranked representative, ordered by resolution, length, release date and deterministic seeded ties. A target can be selected at most once from each original unit.
- Full split groups additionally join qualifying homology edges over the complete eligible universe. They must contain every original unit, including unselected members that connect units. `--cluster-tsv` supplies the original assignments; `--split-group-tsv` supplies this separate, complete group map, with provenance bound by `--split-group-receipt`. Substituting the full group map for the original cluster map would discard the additional selectable representatives needed by this policy.

Every eligible full group must contribute at least one selected target, and all selected targets from a group belong to one split. Public and hidden validation each select exactly one representative per group. Training may select representatives from several original units within a full group. Thus 12,000 targets need not mean 12,000 independent groups. Quality ranking creates a preference for certain source structures that must be reported when comparing samplers.

Live MMseqs clustering requests actual alignment identity with `--alignment-mode 3`, `--min-seq-id 0.30` and `-c 0.80`. A precomputed TSV must cover every candidate and cannot contain conflicting assignments. Its bytes and canonical assignments are locked. A cluster map proves membership, not how an externally supplied TSV was produced. Verify its generation provenance before release. The [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2/wiki) distinguishes clustering algorithms, coverage and identity definitions; greedy cluster separation is not a global all-pairs identity bound.

## Allocation and diagnostics

The CATH v3 reference population is the complete pool of original-unit representatives. Primary strata are each chain's unique CATH Architecture-label set (including unknown), then length bin, then resolution bin. Hierarchical proportional apportionment uses exact integer quota arithmetic and seeded remainder ties. Mandatory group coverage, available capacity and the requested target counts constrain the resulting proportions.

The allocator first chooses the combined public-plus-hidden validation groups. Holding out a group removes all of its original-unit representatives from training while using only one evaluation target, so the joint choice reserves enough capacity to complete the training set. Public-seed allocation then divides these groups between public and hidden validation. Every remaining group contributes one quality-ranked training anchor; additional representatives are allocated among those training groups to reach the requested training count, with mandatory anchors treated as lower bounds. Infeasible sizes or CATH quotas fail instead of silently dropping groups or changing the target counts.

For the current candidate universe, 13,033 original-unit representatives form 8,856 full groups. At 10,000/1,000/1,000 targets, the two evaluation splits occupy 2,000 groups; training must cover the remaining 6,856 groups with 6,856 anchors plus 3,144 additional original-unit representatives. These are candidate diagnostics, not the published benchmark contract. Group sizes can be uneven: review `max_train_group_share` and selected counts per group alongside the number of covered groups. Coverage alone does not bound how much one group contributes to the training objective, and target-weighted performance is not an equal-group performance estimate. In the current candidate universe, the largest group must stay in training to preserve the requested counts and must contribute at least 2,245 of 10,000 training targets (22.45%). This is a capacity bound for this candidate, not a universal policy constant. Evaluation therefore samples held-out smaller groups rather than reproducing the full target-frequency mixture; interpret it as a group-held-out objective and report the resulting composition.

Group ownership, public membership and public order use public inputs and the public seed. The private salt chooses a member and ordering only within the already fixed hidden groups. Held-out CATH quotas use group anchors; a salted hidden member can belong to a different stratum than its anchor, so the final hidden membership is not promised to realize those quotas exactly. Distribution and annotation quality gates check the actual selected members, including the salted hidden choices. **Complete group coverage does not preserve hidden group membership secrecy:** when the full universe and public groups are known, hidden groups are their complement; singleton hidden targets are then inferable. Hidden IDs, labels, manifests, salt and detailed receipts still remain private. The sealed prediction/scoring boundary continues to protect labels from submission code.

The hidden-only generator replays the same public plan and fixed hidden ownership. A private processability exclusion touching a fixed hidden group fails because dropping or replacing that group would violate full coverage. Both generation paths refuse existing sampling outputs. Per-file atomic writes and locks make completion auditable, but do not constitute a transaction across several directories.

Reports include:

- Eligible chain, original-unit and full-group counts; rejection reasons, representative policy and annotation coverage.
- Target and residue counts, distinct groups per split, selected counts per group and maximum training group share.
- C/A/T/H label breadth, multi-domain counts, label incidence and Jensen–Shannon divergence against the original-unit representative population.
- Held-out topology and superfamily familiarity: `UNANNOTATED`, `NONE_SEEN`, `SOME_SEEN`, `ALL_SEEN` relative to train. “All” refers to available domain labels, not complete structural coverage.

Current safeguards limit JS divergence to 0.35 for each split with at least 100 examples, and unknown CATH fraction to 0.75. These are failure limits, not evidence that smaller discrepancies are scientifically harmless. Inspect rare-label depletion, group concentration and annotation changes before adopting a release. The public lock describes public distributions; the complete reference/hidden quality report remains private. CATH proportions and full group coverage do not establish a model-performance improvement or unseen-topology/superfamily generalization.

Generation requires both public and private source locks. In CATH v3, `split_groups` binds the group-map file hash, canonical assignment digest, full group count, original selection-unit count, `selection_unit_ids_sha256` and `coverage: all_groups`. The representative-population digest hashes sorted original representative IDs, each followed by a newline, and is recomputed during replay. The verifier checks both original assignments and full-group nesting over the complete eligible universe, including unselected bridges. It independently requires every group to be covered, confines every group to one split, and allows no more than one selected target per original unit. `--split-group-tsv` and `--split-group-receipt` can relocate identical locked inputs for verification or hidden-only generation; they cannot replace their contents. The graph receipt also binds how the full group map was built; map nesting alone is insufficient provenance.

Verification requires complete typed generation arguments and replays the public and hidden selections, representatives and ordering. Repinning hashes cannot make an inconsistent seed, size or selection pass. Private processability exclusions are checked against locked bytes, canonical IDs and counts; use `--private-processability-exclusion-list` to relocate the same file. An omitted exclusion file is recorded explicitly as absent; a supplied path must exist. Verification reports cannot overwrite audited inputs, including through symlink or hardlink aliases.

`cath-v2` retains its own replay contract: one selected representative per original cluster/exact-sequence/PDB unit, public validation and train selected before hidden validation from the remaining units, without a requirement to cover every eligible unit. Its hidden-only generator excludes every unit touching a private exclusion. A frozen source lock is verified under its recorded policy; verification never relabels an existing selection as CATH v3.

## Stage a new candidate

Use a fresh directory for every candidate and retain its source locks. The chain cache and an audited original cluster TSV covering the complete eligible universe must already be available. The example uses the default 40–256/3 Å single-chain policy. Use the same eligibility options and exclusion files for metadata, group construction and staging. MMseqs2 must be installed separately; precomputed clusters do not bypass homology-group construction or the final target audits.

```bash
sources=.nanofold_private/cath-v3-sources-01
cache=data/openproteinset/pdb_data/data_caches/chain_data_cache.json
clusters=.nanofold_private/original-clusters.tsv
groups=.nanofold_private/cath-v3-groups-01

python scripts/download_structure_metadata_sources.py \
  --chain-data-cache "$cache" \
  --out-dir "$sources" \
  --source-lock "$sources/sources.lock.json"

python scripts/build_structure_metadata.py \
  --chain-data-cache "$cache" \
  --metadata-sources-dir "$sources" \
  --metadata-source-lock "$sources/sources.lock.json" \
  --metadata-out "$sources/structure_metadata.json"

python scripts/build_homology_groups.py \
  --chain-data-cache "$cache" \
  --structure-metadata "$sources/structure_metadata.json" \
  --cluster-tsv "$clusters" \
  --out-dir "$groups" \
  --threads 8

# Provide a maintainer secret of at least 32 characters in this environment.
# Keep the secret and every hidden artifact private.
python scripts/stage_dataset_candidate.py \
  --sampling-policy cath-v3 \
  --stage-dir .nanofold_private/cath-v3-candidate-01 \
  --chain-data-cache "$cache" \
  --metadata-sources-dir "$sources" \
  --metadata-source-lock "$sources/sources.lock.json" \
  --cluster-tsv "$clusters" \
  --split-group-tsv "$groups/split-groups.tsv" \
  --split-group-receipt "$groups/split-groups.receipt.json" \
  --train-size 10000 --val-size 1000 --hidden-val-size 1000 \
  --audit-threads 8
```

`build_homology_groups.py` constructs connected components from qualifying all-pair protein local alignments plus original-cluster, exact-sequence and PDB links. It writes `split-groups.tsv` and `split-groups.receipt.json`, with code snapshots, sequence/alignment artifacts and command logs retained in the private output directory. Preserve the bound source inputs as well as the output directory. The receipt binds the input population, metadata semantics, original assignments, search contract and output map. Metadata timestamps and source locations may change when staging rebuilds metadata; actual annotations, configuration and source hashes must agree. The graph describes closure under the recorded search contract, not every possible alignment or all biological homology. Its all-pair workload grows quadratically with the number of unique eligible sequences; inspect the reported pair count and plan compute/storage before starting a new search.

For a completed compatible search, `--reuse-audit-dir <private-audit-dir> --expected-audit-sha256 <receipt-sha256>` rebuilds and verifies the group components without rerunning MMseqs. The explicit receipt pin is required; reuse does not mean accepting an unverified group TSV. Keep the reused audit and the complete group-builder output available for verification. CATH v3 staging requires both `--split-group-tsv` and `--split-group-receipt`; CATH v2 staging accepts neither and must be requested explicitly.

The stage command builds metadata, invokes the joint generator with both locks, runs the independent verifier, then requires target-sequence audits for train→public validation and public-all→hidden. Both comparisons are retained even if one fails. The audits compare complete cache targets with the same identity/coverage thresholds, all-pair local alignment, explicit protein database type, and independent exact-sequence/chain/PDB checks. They bind the cache, manifests, code, binary, commands and results to hashes. Staging checks the complete reported contract against the requested counts and thresholds, requires typed zero-overlap counts, and validates all private artifacts and pair summaries against their recorded hashes; a status-only receipt cannot pass. `--audit-threads` controls their CPU use. Source/manifest changes between verification and auditing fail staging.

Staging creates the directory with owner-only access, retains step logs privately, and refuses existing directories. Paths inside this repository must be under ignored `.nanofold_private/`; external private directories are also supported. `candidate_status.json` becomes `manifests_verified` only after every step succeeds, sets `target_sequences_audited=true`, and includes the target-audit receipt hash. Failed or interrupted stages must not be adopted; preserve their diagnostics and start a fresh directory. This receipt explicitly leaves processed-tensor and MSA validation outstanding. The direct builders remain available for specialized workflows; their output alone does not pass this staging gate. `scripts/audit_cache_targets.py --help` describes the standalone cache-target audit.

The secret environment variable is `NANOFOLD_HIDDEN_SPLIT_SALT`; alternatively pass `--hidden-split-salt-file` to the generator. Do not expose the salt or its digest in public artifacts. The downloader pins CATH 4.4.0, verifies existing locked bytes instead of repinning them, and records unavailable optional sources. Source filenames are resolved under `--out-dir` by the downloader and `--metadata-sources-dir` by the metadata builder. Use a new directory/lock for a deliberate source update.

`build_hidden_manifest.py --sampling-policy cath-v3` is available for a fixed candidate public contract; pass its `--lock-file` and public manifest paths, metadata/cluster/group-map and receipt overrides if relocated, a fresh hidden output directory/private lock, and the secret. The requested hidden size and seed must match the source lock. Published-manifest regeneration wrappers refuse writes; `full_official_data_refresh.sh --skip-manifest-regen` rebuilds/audits frozen assets only.

Prepare the staged candidate's public assets with explicit custom paths:

```bash
stage=.nanofold_private/cath-v3-candidate-01
bash scripts/setup_custom_data.sh \
  --manifests-dir "$stage/public" \
  --manifest-lock-file "$stage/public-source.lock.json" \
  --chain-data-cache "$cache" \
  --data-root "$stage/raw" \
  --processed-features-dir "$stage/features" \
  --processed-labels-dir "$stage/labels"
```

This command verifies the locked cache and public hashes/counts and union before downloads,
preserves manifests and locks, and prepares only public assets with templates
disabled. It cannot generate splits; generation flags fail before work starts.
`--dry-run` validates inputs and prints commands without creating outputs. Pass
`--msa-row-filter` for an existing private filter and `--resume-preprocess` for
matching raw-input/configuration provenance. Acquired MSA queries must exactly
match the pinned cache targets; cache bytes and selected sequence identity are
bound into preprocessing provenance. Hidden asset preparation, both
strict homology audits and candidate fingerprints remain required separately.

## Training exposure and resume

Default training is target-weighted: one complete unweighted traversal gives a group with `m` selected targets `m` exposures, subject to the documented incomplete-batch and distributed-tail drops. Full group coverage does not give each group equal optimization weight. Inverse-group weighting would change that objective and must be declared as a separate sampling ablation with its own weights and exposure report.

A correct static split can still be sampled incorrectly during training. Checkpoints therefore bind the actual chain/weight population and preserve the epoch, consumed-batch cursor, loader generator and each rank's RNG. Resume replays the current epoch prefix before restoring model random state, preventing repeated exposure to epoch-zero examples after interruptions. Worker seeds are rank-specific; the distributed sampling plan stays shared. Unsupported checkpoints and changed sampling contracts fail explicitly. See [the API contract](API.md#training-sampling-and-multiple-gpus) for supported replay and its I/O cost.

Unweighted distributed sampling drops incomplete shuffled tails instead of
duplicating chains to fill ranks. Exposure counters are checked against steps,
effective batch and crop bounds. MSA masks distinguish biological gaps from
batch padding so differences in companion-chain depth do not alter MSA profiles
or deletion averages.

## Before adopting a dataset

A release needs complete raw-asset availability and successful preprocessing, with rejected cases analyzed across annotation, length and structural categories. `scripts/audit_target_identity.py` can audit the complete public and hidden manifests against the pinned cache using `--raw-root` and/or repeated `--features-dir` options, with required `--expected-counts` in manifest order. It verifies exact raw queries and processed feature/query schema; at least one asset source is required, and missing targets fail. A raw-only receipt does not establish processed-tensor, label, geometry or homology correctness. Do not automatically resample after a hidden preprocessing failure or add hidden IDs to public exclusions. Preserve the failed attempt privately and create a deliberate new candidate version if eligibility must change.

Audit final labels separately from target identity and fingerprints:

```bash
python scripts/audit_label_geometry.py \
  --chain-data-cache "$cache" \
  --split train "$stage/public/train.txt" "$stage/features" "$stage/labels" 10000 \
  --split val "$stage/public/val.txt" "$stage/features" "$stage/labels" 1000 \
  --split hidden "$hidden_manifest" "$hidden_features" "$hidden_labels" 1000 \
  --split-group-tsv "$split_groups" \
  --out-dir "$stage/label-geometry-audit"
```

This complete-population gate rejects missing assets, target or chain mismatches,
invalid feature/label schemas, nonfinite coordinates, inconsistent CA slots,
and observed atom14 slots absent from the target residue's chemical identity.
All source and asset bytes are bound and rechecked. Split and optional full-group
summaries distinguish structural errors from distance diagnostics. CA and peptide
C–N distance checks omit missing atoms, explicit segment breaks and nonconsecutive
residue indices. The default broad reporting ranges are 2.5–4.5 Å for CA and
1.0–1.8 Å for peptide C–N; `--ca-distance-range` and `--peptide-distance-range`
change those diagnostic ranges. Outliers remain in the corpus and the private
report: these ranges are not exclusion rules or proof of experimental quality.

Run strict MSA audits for train→public validation and public-all→hidden, using separate held-out feature roots where needed. `--strict --run-mmseqs` searches held-out targets against both source targets and non-query MSA rows of at least 20 residues, and fails on either kind of hit. Both searches request actual alignment identity and validate numeric fields, known IDs, coverage and expected output; missing output cannot count as zero hits. Target pairs use all-pair local alignment with prefiltering, masking and composition correction disabled and E-value limit `1e308`. Large MSA-row searches use explicitly recorded heuristic settings by default. A zero-hit result is limited to those alignment/search contracts. A filter digest proves filter identity, not absence of residual homology. Audit after filtering and before fingerprinting. Keep identifier-bearing reports private.

For all-pair MSA comparisons, add `--exhaustive` to `build_msa_row_filter.py` and use `audit_msa_split.py --strict --run-mmseqs --exhaustive`. The mode uses protein local alignments with actual alignment identity, bidirectional coverage, `--prefilter-mode 2`, masking/composition correction disabled and E-value limit `1e308`. `--max-seqs` must cover every searched source record: unique source MSA sequences for filtering/MSA auditing, and source target records for the target audit. A cap too small fails; it cannot silently truncate an exhaustive search. The default refresh wrapper retains its heuristic MSA search, so an exhaustive filter does not by itself make the later default audit exhaustive.

Before allocating that search, run the filter builder with the same source/held-out inputs plus `--exhaustive --inventory-only --output <private-inventory.json>`. This scans and hashes source rows without invoking MMseqs or requiring its executable. The inventory reports `source_unique_sequence_count`, `heldout_unique_sequence_count`, their product `all_pair_comparisons`, `required_exhaustive_max_seqs`, and `source_complete`. It is an estimate of search size, not a wall-time estimate or a completed filter; preprocessing rejects it as a filter. Runtime and output size can be much larger than heuristic search. Missing sources remain failures for release preparation, and non-query rows shorter than 20 residues are outside this search scope. All-pair here means every collected query/source pair under the MMseqs local-alignment contract, not enumeration of all possible alignments or proof against every evolutionary relationship.

The row filter also removes canonical exact matches independently of MMseqs, so
low-complexity masking cannot retain an identical held-out target. All supplied
processed roots contribute their source variants. Audit-based filter extension
requires the matching summary, ordered source-record digest and alignment-output
hash; changed source content or ordering requires a fresh audit.

Preprocessing changes also require a new asset/fingerprint version. Schema-3 files bind configuration, implementation and raw input bytes; changed provenance cannot be resumed into existing outputs. Their run metadata is mandatory during fingerprint creation, including when someone removes or downgrades it. Validation masks identity-mismatched labels, checks atom/token consistency and detects mixed feature/label pairs. Strict alignment mode still enforces the valid-CA minimum. Optional research templates require auditable polymer residue indices and sequence identity, so missing coordinates cannot shift template positions.

Deposited-polymer projection accepts equivalent placements of unresolved residues only when every optimal alignment agrees on all observed residue positions. It fails if that mapping differs or more than 1,024 optimal alignments prevent the bounded uniqueness check from completing. Projection without deposited polymer context remains strictly unambiguous.

To evaluate whether sampling improves learning, preregister matched comparisons with fixed feature access, training budget, optimization, evaluation set and representative policy. Use several sampling and training seeds. Report mean performance and cluster/architecture-level uncertainty alongside annotation and family coverage. Compare proportional architecture sampling with diversity-weighted or capped-superfamily alternatives as separate experiments. Report topology/superfamily holdouts and temporal holdouts separately; mixing these objectives into the main composition-matched score changes what the benchmark measures. None of the implementation checks establishes a model-quality gain without those experiments.
