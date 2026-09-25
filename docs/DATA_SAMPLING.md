# Dataset sampling and release audit

The published benchmark is defined by its committed manifests, locks and fingerprints. Source-qualified candidates explicitly use `--sampling-policy cath-v4` with metadata schema 2, the complete original homology graph, source-validated representatives and an approved whole-group exclusion catalog in fresh, ignored directories. Staging defaults to `cath-v3`; `cath-v2` and `cath-v3` retain their own selection and replay contracts. Generating a candidate does not adopt it as the official benchmark or establish improved model performance.

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

The default candidate is a single chain with standard amino acids, a consistent sequence length of 40–256, and resolution at most 3 Å when known. Missing/nonpositive resolution is unknown rather than exceptional coordinate quality. Feature and processability exclusions must exist and are hashed. Metadata records the full eligible universe and any debugging sample limit; all CATH policies refuse truncated metadata or permissive unknown-amino-acid filters. These gates describe the available source population, not a representative sample of all proteins.

Single-chain targets are not necessarily biological monomers. `--require-monomer` requires an explicit `oligomeric_count=1`; missing assembly information cannot satisfy it. Use the same option consistently in download, metadata and sampling commands. Asymmetric-unit chain count is not biological assembly evidence; see the [RCSB assembly documentation](https://www.rcsb.org/docs/general-help/biological-assemblies).

CATH v4 distinguishes the **selectable target pool** from the **groups that cannot cross splits**:

- Original units join assigned MMseqs clusters, exact duplicate sequences and same-PDB chains transitively. Members retain their deterministic quality order: resolution, length, release date and seeded ties. Each unit contributes its first member that passes source-validated raw-query and strict structure checks. Every higher-ranked failure must be accounted for, or every member must be definitively shown to fail before the unit can be declared exhausted. At most one target can be selected per original unit.
- Full split groups additionally join qualifying homology edges over the complete eligible universe. They must contain every original unit, including unselected members that connect units. `--cluster-tsv` supplies the original assignments; `--split-group-tsv` supplies this separate, complete group map, with provenance bound by `--split-group-receipt`. Substituting the full group map for the original cluster map would discard the additional selectable representatives needed by this policy.

Every retained full group must contribute at least one selected target, and all selected targets from a group belong to one split. A full group may be excluded only when all its original units are exhausted and the group is explicitly approved in the exclusion catalog. The catalog must match the complete exhausted-group set exactly; missing downloads, pending work or unvalidated source files do not prove exhaustion. The original graph, including excluded nodes and connecting members, remains intact. Public and hidden validation each select exactly one representative per group. Training may select representatives from several original units within a full group. Thus 12,000 targets need not mean 12,000 independent groups. Quality ranking creates a preference for certain source structures that must be reported when comparing samplers.

Live MMseqs clustering requests actual alignment identity with `--alignment-mode 3`, `--min-seq-id 0.30` and `-c 0.80`. A precomputed TSV must cover every candidate and cannot contain conflicting assignments. Its bytes and canonical assignments are locked. A cluster map proves membership, not how an externally supplied TSV was produced. Verify its generation provenance before release. The [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2/wiki) distinguishes clustering algorithms, coverage and identity definitions; greedy cluster separation is not a global all-pairs identity bound.

## Allocation and diagnostics

The CATH v4 reference population is the complete pool of source-validated original-unit representatives. Primary strata are each chain's unique CATH Architecture-label set (including unknown), then length bin, then resolution bin. Hierarchical proportional apportionment uses exact integer quota arithmetic and seeded remainder ties. Mandatory group coverage, available capacity and the requested target counts constrain the resulting proportions.

The allocator first chooses the combined public-plus-hidden validation groups. Holding out a group removes all of its original-unit representatives from training while using only one evaluation target, so the joint choice reserves enough capacity to complete the training set. Public-seed allocation then divides these groups between public and hidden validation. Every remaining group contributes one quality-ranked training anchor; additional representatives are allocated among those training groups to reach the requested training count, with mandatory anchors treated as lower bounds. Infeasible sizes or CATH quotas fail instead of silently dropping groups or changing the target counts.

For this candidate, the original graph has 8,856 full groups. The approved catalog excludes six fully exhausted groups, leaving **8,850 retained groups**. At 10,000/1,000/1,000 targets, the two evaluation splits occupy 2,000 groups; training must cover the remaining 6,850 groups with 6,850 anchors plus 3,150 additional validated original-unit representatives. These are candidate requirements, not evidence that the assets have passed all release gates or a change to the published benchmark.

Report the validated representative count and digest separately from the original population. Source failures can change the selectable pool without changing graph membership. Group sizes remain uneven: review `max_train_group_share` and selected counts per group alongside covered-group counts. Coverage does not equalize training exposure, and target-weighted performance is not an equal-group performance estimate. Capacity can force large groups into training; the resulting evaluation then describes held-out groups rather than the full target-frequency mixture. Recompute concentration bounds for the validated candidate instead of carrying forward bounds from a different representative pool.

Group ownership, public membership and public order use public inputs and the public seed. The private salt chooses a member and ordering only within the already fixed hidden groups. Held-out CATH quotas use group anchors; a salted hidden member can belong to a different stratum than its anchor, so the final hidden membership is not promised to realize those quotas exactly. Distribution and annotation quality gates check the actual selected members, including the salted hidden choices. **Complete group coverage does not preserve hidden group membership secrecy:** when the retained universe, exclusion catalog and public groups are known, hidden groups are the retained complement; singleton hidden targets are then inferable. Hidden IDs, labels, manifests, salt and detailed receipts still remain private. The sealed prediction/scoring boundary continues to protect labels from submission code.

The hidden-only generator replays the same public plan and fixed hidden ownership. A private processability exclusion touching a fixed hidden group fails because dropping or replacing that group would violate full coverage. Both generation paths refuse existing sampling outputs. Per-file atomic writes and locks make completion auditable, but do not constitute a transaction across several directories.

Reports include:

- Original and source-validated unit counts/digests; original, approved-excluded and retained full-group counts; definitive rejection reasons and annotation coverage.
- Target and residue counts, distinct groups per split, selected counts per group and maximum training group share.
- C/A/T/H label breadth, multi-domain counts, label incidence and Jensen–Shannon divergence against the source-validated original-unit representative population.
- Held-out topology and superfamily familiarity: `UNANNOTATED`, `NONE_SEEN`, `SOME_SEEN`, `ALL_SEEN` relative to train. “All” refers to available domain labels, not complete structural coverage.

Current safeguards limit JS divergence to 0.35 for each split with at least 100 examples, and unknown CATH fraction to 0.75. These are failure limits, not evidence that smaller discrepancies are scientifically harmless. Inspect rare-label depletion, group concentration and annotation changes before adopting a release. The public lock describes public distributions; the complete reference/hidden quality report remains private. CATH proportions and full group coverage do not establish a model-performance improvement or unseen-topology/superfamily generalization.

Generation requires both public and private source locks. CATH v4 additionally binds `--representative-eligibility-receipt` and `--approved-group-exclusions` through `representative_eligibility`. The original `split_groups.group_count` remains the complete graph count; retained/excluded counts and original/validated representative counts and digests distinguish graph identity from selectable capacity. The verifier reconstructs original unit membership and rank order, checks source evidence and the approved catalog, then replays allocation over the validated representatives. It independently requires every retained group to be covered, confines each group to one split, and allows at most one selected target per original unit. Unselected or excluded members cannot remove a graph connection.

The original cluster map, full group map and graph-construction receipt retain their byte and canonical assignment bindings. Relocation overrides must resolve to the same locked inputs; replacing their contents does not preserve the contract. Source-validation receipts bind the raw queries, uniformly pinned historical structure source, strict validation predicate and complete ranked attempt evidence. Fixed structure thresholds are identity 0.90, coverage 0.70, aligned fraction 0.70 and at least 32 valid Cα positions. CATH v4 does not permit a partial-label relaxation or substitute current structures for the pinned historical source.

Verification requires complete typed generation arguments and replays the public and hidden selections, representatives and ordering. Repinning hashes cannot make an inconsistent seed, size or selection pass. Private processability exclusions are checked against locked bytes, canonical IDs and counts; use `--private-processability-exclusion-list` to relocate the same file. An omitted exclusion file is recorded explicitly as absent; a supplied path must exist. Verification reports cannot overwrite audited inputs, including through symlink or hardlink aliases.

`cath-v3` retains its complete original-group coverage contract with the original quality-ranked representatives and no CATH-v4 eligibility/catalog inputs. `cath-v2` retains its own replay contract: one selected representative per original cluster/exact-sequence/PDB unit, public validation and train selected before hidden validation from the remaining units, without a requirement to cover every eligible unit. Its hidden-only generator excludes every unit touching a private exclusion. A frozen source lock is verified under its recorded policy; verification never relabels an existing selection as another policy.

## Stage a new candidate

Use a fresh directory for every candidate and retain its source locks. The chain cache, audited original cluster TSV, original complete graph and its receipt, uniformly pinned historical retrieval receipts, complete raw query sources and approved exclusion catalog must be available. A new graph may be built from the unchanged pre-validation universe as shown below; never rebuild it after deleting failed members or excluded groups. The example uses the default 40–256/3 Å single-chain policy. Use the same eligibility options and exclusion files for metadata, group construction and staging. MMseqs2 must be installed separately; precomputed clusters do not bypass homology-group construction or the final target audits.

```bash
sources=.nanofold_private/cath-v4-sources-01
cache=data/openproteinset/pdb_data/data_caches/chain_data_cache.json
clusters=.nanofold_private/original-clusters.tsv
groups=.nanofold_private/original-homology-groups
raw=.nanofold_private/raw-source-pool
historical_receipt=.nanofold_private/historical-retrieval-receipt.json
approved=.nanofold_private/approved-group-exclusions.json
eligibility=.nanofold_private/cath-v4-representative-eligibility.json

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

# Use the reviewed source receipts and approved catalog prepared beforehand.
python scripts/build_representative_eligibility.py \
  --chain-data-cache "$cache" \
  --structure-metadata "$sources/structure_metadata.json" \
  --cluster-tsv "$clusters" \
  --split-group-tsv "$groups/split-groups.tsv" \
  --split-group-receipt "$groups/split-groups.receipt.json" \
  --historical-retrieval-receipt "$historical_receipt" \
  --raw-root "$raw" \
  --approved-group-exclusions "$approved" \
  --seed 0 \
  --output "$eligibility"

# Provide a maintainer secret of at least 32 characters in this environment.
# Keep the secret and every hidden artifact private.
python scripts/stage_dataset_candidate.py \
  --sampling-policy cath-v4 \
  --stage-dir .nanofold_private/cath-v4-candidate-01 \
  --chain-data-cache "$cache" \
  --metadata-sources-dir "$sources" \
  --metadata-source-lock "$sources/sources.lock.json" \
  --cluster-tsv "$clusters" \
  --split-group-tsv "$groups/split-groups.tsv" \
  --split-group-receipt "$groups/split-groups.receipt.json" \
  --representative-eligibility-receipt "$eligibility" \
  --approved-group-exclusions "$approved" \
  --seed 0 \
  --train-size 10000 --val-size 1000 --hidden-val-size 1000 \
  --audit-threads 8
```

`build_homology_groups.py` constructs connected components from qualifying all-pair protein local alignments plus original-cluster, exact-sequence and PDB links. It writes `split-groups.tsv` and `split-groups.receipt.json`, with code snapshots, sequence/alignment artifacts and command logs retained in the private output directory. Preserve the bound source inputs as well as the output directory. The receipt binds the input population, metadata semantics, original assignments, search contract and output map. Metadata timestamps and source locations may change when staging rebuilds metadata; actual annotations, configuration and source hashes must agree. The graph describes closure under the recorded search contract, not every possible alignment or all biological homology. Its all-pair workload grows quadratically with the number of unique eligible sequences; inspect the reported pair count and plan compute/storage before starting a new search.

For a completed compatible search, `--reuse-audit-dir <private-audit-dir> --expected-audit-sha256 <receipt-sha256>` rebuilds and verifies the group components without rerunning MMseqs. The explicit receipt pin is required; reuse does not mean accepting an unverified group TSV. Keep the reused audit and the complete group-builder output available for verification. CATH v3 and v4 staging require both `--split-group-tsv` and `--split-group-receipt`; CATH v4 also requires the validated representative receipt and approved exclusion catalog. CATH v2 accepts none of these group-policy inputs and must be requested explicitly.

`build_representative_eligibility.py` requires receipts from the tracked historical retriever; repeat `--historical-retrieval-receipt` and `--raw-root` when source data spans several roots. Receipts must bind the same immutable historical archive policy. It checks every earlier ranked member before accepting the first passing member and exhausts every original unit in a proposed excluded group. Missing files, acquisition errors and pending attempts abort rather than becoming exclusion evidence. The approved schema-1 catalog declares `policy: cath-v4`, an exact sorted `excluded_group_ids` set and bound authorization; a failed audit does not create approval. Keep identifiers and detailed reasons private. Its successful receipt establishes representative eligibility, not final processed assets or release adoption.

The stage command builds metadata, validates the bound eligibility/catalog inputs, invokes the joint generator with both locks, runs the independent verifier, then requires target-sequence audits for train→public validation and public-all→hidden. Both comparisons are retained even if one fails. The audits compare complete cache targets with the same identity/coverage thresholds, all-pair local alignment, explicit protein database type, and independent exact-sequence/chain/PDB checks. They bind the cache, manifests, code, binary, commands and results to hashes. Staging checks the complete reported contract against the requested counts and thresholds, requires typed zero-overlap counts, and validates all private artifacts and pair summaries against their recorded hashes; a status-only receipt cannot pass. `--audit-threads` controls their CPU use. Source/manifest changes between verification and auditing fail staging.

Staging creates the directory with owner-only access, retains step logs privately, and refuses existing directories. Paths inside this repository must be under ignored `.nanofold_private/`; external private directories are also supported. `candidate_status.json` becomes `manifests_verified` only after every step succeeds, sets `target_sequences_audited=true`, and includes the target-audit receipt hash. Failed or interrupted stages must not be adopted; preserve their diagnostics and start a fresh directory. This receipt explicitly leaves processed-tensor and MSA validation outstanding. The direct builders remain available for specialized workflows; their output alone does not pass this staging gate. `scripts/audit_cache_targets.py --help` describes the standalone cache-target audit.

The secret environment variable is `NANOFOLD_HIDDEN_SPLIT_SALT`; alternatively pass `--hidden-split-salt-file` to the generator. Do not expose the salt or its digest in public artifacts. The downloader pins CATH 4.4.0, verifies existing locked bytes instead of repinning them, and records unavailable optional sources. Source filenames are resolved under `--out-dir` by the downloader and `--metadata-sources-dir` by the metadata builder. Use a new directory/lock for a deliberate source update.

`build_hidden_manifest.py --sampling-policy cath-v4` is available for a fixed candidate public contract; pass its `--lock-file` and public manifest paths, metadata/cluster/group-map and receipt overrides plus the bound eligibility receipt and exclusion catalog if relocated, a fresh hidden output directory/private lock, and the secret. The requested hidden size and seed must match the source lock. Published-manifest regeneration wrappers refuse writes; `full_official_data_refresh.sh --skip-manifest-regen` rebuilds/audits frozen assets only.

Prepare CATH v4 assets from the **same validated raw-query and historical-structure bytes** after building the complete held-out MSA filter. The following assumes the verified historical members are available under one root; use the paths bound by the eligibility receipt and recheck hashes if constructing a consolidated source view. Process public `all.txt` once, then process the hidden manifest into separate feature/label roots with the same predicate and filter.

```bash
stage=.nanofold_private/cath-v4-candidate-01
historical_mmcif=.nanofold_private/pinned-structures/pdb_data/mmcif_files
msa_filter=.nanofold_private/cath-v4-heldout-filter.json
python scripts/preprocess.py \
  --manifest "$stage/public/all.txt" \
  --chain-data-cache "$cache" \
  --raw-root "$raw" \
  --mmcif-root "$historical_mmcif" \
  --processed-features-dir "$stage/features" \
  --processed-labels-dir "$stage/labels" \
  --msa-row-filter "$msa_filter" \
  --disable-templates \
  --min-projection-seq-identity 0.90 \
  --min-projection-coverage 0.70 \
  --min-projection-aligned-fraction 0.70 \
  --min-projection-valid-ca 32
```

These fixed projection thresholds are the eligibility predicate; `preprocess.py --strict` is a different option requiring exact coordinate-sequence agreement and is not interchangeable with this policy. Keep `--allow-failures` off. The acquired MSA query must equal the pinned cache target exactly. Cache bytes, selected target identity, raw assets, filter and preprocessing configuration are bound into output provenance. `--skip-existing` permits only matching validated outputs; changed inputs require fresh roots. Complete target identity, label geometry, residual homology audits and fingerprints remain required before adoption.

`scripts/setup_custom_data.sh` remains a convenience for other staged datasets with explicit custom paths; it downloads the current RCSB mmCIF subset and does not select the CATH v4 historical source policy. Do not substitute its current-source preprocessing for the bound historical inputs. Neither asset command generates or adopts splits.

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

For historical source acquisition, `scripts/fetch_openproteinset_mmcif.py`
retrieves only manifest structures from an explicitly pinned OpenProteinSet ZIP
and retains private range/member provenance, including bounded transport retries.
Integrity failures are not retried. Its source-only receipt does not
establish processability. Check every selected target's query and structure under
the same projection predicate; failed members cannot be silently removed after
group allocation, because that can violate required group coverage.

Run strict MSA audits for train→public validation and public-all→hidden, using separate held-out feature roots where needed. `--strict --run-mmseqs` searches held-out targets against both source targets and non-query MSA rows of at least 20 residues, and fails on either kind of hit. Both searches request actual alignment identity and validate numeric fields, known IDs, coverage and expected output; missing output cannot count as zero hits. Target pairs use all-pair local alignment with prefiltering, masking and composition correction disabled and E-value limit `1e308`. Large MSA-row searches use explicitly recorded heuristic settings by default. A zero-hit result is limited to those alignment/search contracts. A filter digest proves filter identity, not absence of residual homology. Audit after filtering and before fingerprinting. Keep identifier-bearing reports private.

For all-pair MSA comparisons, add `--exhaustive` to `build_msa_row_filter.py` and use `audit_msa_split.py --strict --run-mmseqs --exhaustive`. The mode uses protein local alignments with actual alignment identity, bidirectional coverage, `--prefilter-mode 2`, masking/composition correction disabled and E-value limit `1e308`. Without checkpointed batching, `--max-seqs` must cover the whole searched source population: unique MSA sequences for filtering/MSA auditing, and source target records for the target audit. With checkpointed batching, it must cover every record in each source batch. A cap too small fails; it cannot silently truncate an exhaustive search. The default refresh wrapper retains its heuristic MSA search, so an exhaustive filter does not by itself make the later default audit exhaustive.

Before allocating that search, run the filter builder with the same source/held-out inputs plus `--exhaustive --inventory-only --output <private-inventory.json>`. This scans and hashes source rows without invoking MMseqs or requiring its executable. The inventory reports `source_unique_sequence_count`, `heldout_unique_sequence_count`, their product `all_pair_comparisons`, `required_exhaustive_max_seqs`, and `source_complete`. It is an estimate of search size, not a wall-time estimate or a completed filter; preprocessing rejects it as a filter. Runtime and output size can be much larger than heuristic search. Missing sources remain failures for release preparation, and non-query rows shorter than 20 residues are outside this search scope. All-pair here means every collected query/source pair under the MMseqs local-alignment contract, not enumeration of all possible alignments or proof against every evolutionary relationship.

### Bounded, resumable exhaustive MSA searches

Use `--exhaustive --checkpoint-dir <private-search-directory>` with the filter builder, or `--strict --run-mmseqs --exhaustive --checkpoint-dir <private-search-directory>` with the split auditor. Rerun the same command to resume; no separate resume flag is needed. Each distinct search needs its own checkpoint directory. The auditor requires anonymous headers, so `--include-identifiers-in-mmseqs` cannot be combined with checkpointed batching. It keeps target and MSA searches under separate `train_targets` and `train_msa` subdirectories.

Build the complete trio filter before preprocessing; repeat the same command after interruption:

```bash
python scripts/build_msa_row_filter.py \
  --chain-data-cache "$cache" \
  --source-manifest "$stage/public/all.txt" \
  --source-manifest "$stage/hidden/hidden_val.txt" \
  --heldout-manifest "$stage/public/val.txt" \
  --heldout-manifest "$stage/hidden/hidden_val.txt" \
  --raw-root "$raw" \
  --exhaustive \
  --checkpoint-dir "$stage/exhaustive-msa-filter" \
  --source-batch-size 1000000 --query-batch-size 256 \
  --max-seqs 1000000 \
  --search-disk-reserve-gib 100 --search-max-rss-gib 48 \
  --output "$msa_filter"
```

| Option | Default | Contract |
|---|---:|---|
| `--source-batch-size` | 1,000,000 | Positive; at most 1,000,000 source records per job. |
| `--query-batch-size` | 256 | Positive; at most 256 query records per job. |
| `--max-seqs` | 1,000,000 | Must cover the actual source batch; native acceptance/rejection limits also cover that batch. |
| `--search-disk-reserve-gib` | 100 | Free-disk reserve checked while searching. |
| `--search-max-rss-gib` | 48 | Sampled combined caller/native-process-tree RSS ceiling. |

The immutable plan binds complete query/source FASTAs, deterministic partitions, search parameters, the resolved MMseqs binary/hash/version and search implementation. Completed-job receipts bind native logs and alignments. Resume revalidates this evidence and skips only verified complete jobs; interrupted or failed attempts remain available for diagnosis and are rerun in fresh attempt directories. Changed inputs or search contracts require a separate plan. The final result combines every grid job only after all have completed and inputs are revalidated; partial job completion is not a valid filter or passed audit. Batched evidence references individual job alignments through its receipt; it does not publish a single merged M8 file.

The supported sequence lengths are 1–256 residues, with the existing 20-residue minimum for non-query MSA rows. Native numeric limits must be valid; batching is not a reason to truncate returned hits. Disk/RSS monitoring fails closed but is sampled, not an operating-system memory cap. Local process inspection must work before search starts; on systems with sandbox restrictions, run with the required local process permissions. Inventory mode can use the same batch flags and checkpoint-directory option to report the per-batch hit-capacity requirement without launching MMseqs or creating a checkpoint. In that mode `required_exhaustive_max_seqs` uses the smaller of the source population and source-batch size; the unbatched inventory uses the whole source population.

Batching bounds individual jobs and enables reuse of completed work; it does not reduce the total query-by-source comparison count. Run an exhaustive filter, preprocess from raw inputs, then rerun both residual split audits in exhaustive mode. `extend_msa_row_filter_from_audit.py` accepts a single-alignment audit only; batched reports are rejected and cannot be used as if one shard represented the full search.

The row filter also removes canonical exact matches independently of MMseqs, so
low-complexity masking cannot retain an identical held-out target. All supplied
processed roots contribute their source variants. Audit-based filter extension
requires the matching summary, ordered source-record digest and alignment-output
hash; changed source content or ordering requires a fresh audit.

Preprocessing changes also require a new asset/fingerprint version. Schema-3 files bind configuration, implementation and raw input bytes; changed provenance cannot be resumed into existing outputs. Their run metadata is mandatory during fingerprint creation, including when someone removes or downgrades it. Validation masks identity-mismatched labels, checks atom/token consistency and detects mixed feature/label pairs. Strict alignment mode still enforces the valid-CA minimum. Optional research templates require auditable polymer residue indices and sequence identity, so missing coordinates cannot shift template positions.

Deposited-polymer projection accepts equivalent placements of unresolved residues only when every optimal alignment agrees on all observed residue positions. It fails if that mapping differs or more than 1,024 optimal alignments prevent the bounded uniqueness check from completing. Projection without deposited polymer context remains strictly unambiguous.

To evaluate whether sampling improves learning, preregister matched comparisons with fixed feature access, training budget, optimization, evaluation set and representative policy. Use several sampling and training seeds. Report mean performance and cluster/architecture-level uncertainty alongside annotation and family coverage. Compare proportional architecture sampling with diversity-weighted or capped-superfamily alternatives as separate experiments. Report topology/superfamily holdouts and temporal holdouts separately; mixing these objectives into the main composition-matched score changes what the benchmark measures. None of the implementation checks establishes a model-quality gain without those experiments.
