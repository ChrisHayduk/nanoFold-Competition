# Dataset sampling and release audit

The published benchmark is defined by its committed manifests, locks and fingerprints. Source-qualified candidates explicitly use `--sampling-policy cath-v4` with metadata schema 2, the complete original homology graph, source-validated representatives and an approved whole-group exclusion catalog in fresh, ignored directories. Staging defaults to `cath-v3`; `cath-v2` and `cath-v3` retain their own selection and replay contracts. Generating a candidate does not adopt it as the official benchmark or establish improved model performance.

The `cath-v5` research candidate selects training chains against the broader
qualified structural population. A separately recorded evaluation repair can
replace public targets that fail the expanded barrier while retaining the hidden
cohort and protecting the original public cohort for historical comparisons.
It does not change the published manifests automatically.

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

### Topology-aware training selection (`cath-v5`)

The full-parent candidate selects 10,000 distinct parent sequences from the
pinned, source-qualified experimental OPS population, including longer chains.
Training draws a parent uniformly, then draws a contiguous crop of at most 256
residues uniformly from all valid integer starts. The primary reference applies
that same crop rule to **every source-qualified, training-owned distinct parent
sequence**, before collapsing original sampling units. Source-support losses
against the complete pinned PDB-chain inventory are reported separately.
This reference is not a deposited-copy, group-balanced, natural
abundance, or claimed AlphaFold2 training distribution. Cluster-balanced
reference weighting is a separately labeled sensitivity analysis.

The base frame binds the existing evaluation targets. For the expanded parent
population, the staging barrier excludes inherited held-out group members, held-out exact
sequences, held-out PDB entries, and direct qualifying held-out-query matches.
The exhaustive direct search uses 30% actual identity and 80% held-out-query
coverage; long-parent coverage is not required. Training diversity groups are
formed only after these exclusions and **are not a certificate of global
transitive leakage-group separation**. The inherited short-parent comparison
retains its complete original graph and ownership contract. Selection cannot
relax either mode's source qualification or admission barrier.

An evaluation repair retains the hidden targets and unaffected public targets,
then replaces affected public targets using a frozen metadata ranking and source
qualification. The repaired public and hidden cohorts must pass the query-coverage
check in both directions. A derived training frame preserves every base exclusion
and adds protection for replacement public targets and their inherited groups;
its admitted training population must be a subset of the base frame. Historical
public, active public and hidden manifests have separate bindings. Training MSA
exclusions and residual checks protect the union of all three cohorts, while
active public MSAs exclude hidden targets. The original public cohort remains a
separate historical comparison, with its conflicts disclosed.

The full-parent asset coordinator also uses `--coverage-mode 2` for raw-MSA
filtering and both residual audits, with `--max-sequence-length` set to the
actual longest selected parent or protected held-out target. The
`full-parent-heldout-query-v2` policy additionally binds the historical public
cohort. Existing `full-parent-heldout-query-v1` receipts retain their original
three-split query-coverage contract, and `directed-heldout-v1` retains its
bidirectional coverage contract.

Several distinct sequences from one original unit or PDB remain selectable
within training. Deposited copies of an exact sequence contribute one reference
member, preferring the qualified copy with better known resolution. Unknown
resolution has an explicit category and never becomes a zero-resolution quality
advantage. All modes require exactly the requested number of distinct sequences.
Insufficient qualified sequence capacity fails.

`scripts/qualify_structural_population.py` evaluates deposited copies in their
frozen quality order and preserves an outcome for every planned parent. Missing
raw sources remain pending unless `--source-availability-certificate` supplies
a complete, pinned acquisition proof. The owning validator reconstructs every
allowed duplicate-map route from the original inputs and accepts only completed
empty listings for the required MSA under the recorded request policy. Transport
errors, partial listings, unknown routes and changed files cannot certify absence.
Each certified unavailable copy has an explicit `source_unavailable` outcome;
it has no invented structure or sequence inspection result. A parent is exhausted
only after every admitted copy has an explicit terminal outcome. Reports separate
source-only, inspection-only and mixed losses while retaining the full planned
denominator. This proves availability for that frozen source acquisition, not
permanent absence or biological unsuitability. Current-owner replay uses fresh
output directories when qualification code changes; existing receipts remain
immutable.

For full-parent selection, pass `crop_size=256` and
`group_policy="soft_diversity"` explicitly. There can be more diversity groups
than sample slots. This mode jointly selects parents using a source-weighted
group-distribution objective, permits complementary members of the same group,
and does not guarantee coverage of every group. It neither imposes one member
per selected component nor removes groups from the reference to manufacture
feasibility. The inherited comparison uses `group_policy="all_groups"`, which
requires every group and fails when that is impossible. Both choices are
reported, never switched silently.

#### Exact crop exposure

CATH domains contribute their verified residue segments, including discontinuous
segments and repeated domains. For each domain, the sampler computes the exact
expected residue fraction, probability of any mapped residue being present,
probability of all domain residues fitting, and probability of a visible
fragment, averaging **all** crop starts. Intactness depends on the complete
sequence span of a discontinuous domain, not merely its summed segment length.
The probability of simultaneous mapped-domain presence and its topology
composition is integrated by an exact interval sweep.

Unmapped boundaries receive no guessed location. Residue mass outside verified
mapped domains remains unknown. Overlapping mapped domains fail. Domain presence,
intactness and fragmentation are expected domain-instance counts and may exceed
one per crop; their absolute mass is retained, not hidden by normalizing every
family into a probability distribution. Measured contact exposure averages
actual nonlocal CA contacts surviving each crop, with explicit missing-coordinate
coverage. Whole-parent geometry is not substituted for crop geometry.

Annotated sequence exposure does not establish observed structural supervision.
A separate diagnostic intersects each mapped domain with the measured CA mask
and partitions crop positions into labeled observed CA, unknown observed CA,
and unobserved CA. Missing masks are unavailable; observed CA does not establish
complete all-atom supervision. Reports also retain the probability of crops
with zero or fewer than 32 observed CA atoms. These diagnostics do not resample
crops, reject parents, or change the selection objective.

The predeclared full-parent feature families and priorities are:

| Feature family | Priority |
|---|---:|
| Expected crop residue fraction by CATH topology | 4 |
| Mapped domains visible per crop (`0`, `1`, `2`, `3`, `4+`) | 2 |
| Superfamily residue fraction; topology × mapped domain length; topology × visible domain count | 1 each |
| Expected topology presence, intact-domain count, fragmented-domain count | 1 each |
| Mapped topology presence composition; mapped/unknown residue fraction; crop-length bin; observed/missing CA fraction | 1 each |
| Training diversity-group indicator, only for explicit soft diversity | 1 |
| Architecture residue fraction; expected contacts per crop residue | 0.5 each |
| Parent-length bin; resolution bin including unknown | 0.25 each |

For summed selected contribution `c_j`, the target is
`t_j = sample_size × reference_mean_j`. The discrepancy is
`sum_f priority_f / max(sum_j(t_j),1) × sum_j((c_j-t_j)^2 / max(t_j,1))`,
with inner sums restricted to family `f`. These priorities scale the **selection
objective**; they are not per-chain training probabilities or loss weights.
The primary reference gives each distinct sequence equal probability. An
independently supplied reference-weight map must cover exactly that population,
be finite and nonnegative, and receive an explicit nonuniform label. Its full
normalized probabilities and effective size are recorded; selected-parent
training remains uniform.

Greedy selection uses the same objective with the reference target scaled to the
current number of selected parents. Deterministic swaps must strictly improve
the final-size objective. The default is at most two passes and 2,048 attempted
exchanges, without a global-optimum or full-convergence guarantee. Known quality
and seeded chain-ID hashes resolve score ties. In all-groups mode the anchors
remain replaceable during swaps that preserve group coverage; soft diversity
has no mandatory anchors. Topology/superfamily coverage is an optimization
objective, not a guarantee.

The private report binds reference identities, reference probabilities and exact
feature moments. It retains fractional reference/target/selected contributions,
absolute feature mass per crop, objective values, exchange counts, missing
reference mass, category and group coverage, uncovered group probability mass,
and group concentration. All-groups mode additionally reports unavoidable
feature lower bounds imposed by group coverage. Full-parent reports distinguish
training diversity groups from leakage barriers. Structural calibration does
not establish full-data gradient agreement or improved folding accuracy;
metrics optimized by selection must be distinguished from independent geometry
and held-out metrics.

The inherited whole-chain mode (`crop_size=None`) retains domain-instance
moments and conditional whole-chain geometry only for one verified domain
covering at least 90% of a chain with at least 90% valid CA positions. Those
whole-chain diagnostics must not be presented as the full-parent crop target.

### Architecture-stratified representative selection (`cath-v4`)

The CATH v4 reference population is the complete pool of source-validated original-unit representatives. Primary strata are each chain's unique CATH Architecture-label set (including unknown), then length bin, then resolution bin. Hierarchical proportional apportionment uses exact integer quota arithmetic and seeded remainder ties. Mandatory group coverage, available capacity and the requested target counts constrain the resulting proportions.

The allocator first chooses the combined public-plus-hidden validation groups. Holding out a group removes all of its original-unit representatives from training while using only one evaluation target, so the joint choice reserves enough capacity to complete the training set. Public-seed allocation then divides these groups between public and hidden validation. Every remaining group contributes one quality-ranked training anchor; additional representatives are allocated among those training groups to reach the requested training count, with mandatory anchors treated as lower bounds. Infeasible sizes or CATH quotas fail instead of silently dropping groups or changing the target counts.

For this candidate, the original graph has 8,856 full groups. The approved catalog excludes eight fully exhausted groups, leaving **8,848 retained groups**. At 10,000/1,000/1,000 targets, the two evaluation splits occupy 2,000 groups; training must cover the remaining 6,848 groups with 6,848 anchors plus 3,152 additional validated original-unit representatives. These are candidate requirements, not evidence that the assets have passed all release gates or a change to the published benchmark.

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

Coordinate parser changes require replaying every original unit's ranked
qualification prefix under the new predicate, even if the selected manifest
bytes appear unchanged. Positive-occupancy masks and coherent conformer
selection can change valid-CA counts. A changed exhausted-group set must fail
against the approved exclusion catalog; it cannot silently add exclusions or
weaken thresholds. New qualification bindings require fresh candidate locks and
processed assets. The original sequence graph can be reused only after verifying
its unchanged sequence inputs and construction evidence.

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

The generator creates an exclusive `<output.name>.diagnostics/` directory beside the receipt path. It records input bindings and appends each completed unit to `units.jsonl`, flushing it to disk before continuing. On failure, `failure.json` records the phase, completed count and available pin checks; an exclusion mismatch also records the exact evaluated and approved group sets. These files contain private target information and remain diagnostic only. Preserve failed attempts and use a fresh output path for a new run; the journal is not a resume or acceptance input.

The stage command builds metadata, validates the bound eligibility/catalog inputs, invokes the joint generator with both locks, runs the independent verifier, then requires target-sequence audits for train→public validation and public-all→hidden. Both comparisons are retained even if one fails. The audits compare complete cache targets with the same identity/coverage thresholds, all-pair local alignment, explicit protein database type, and independent exact-sequence/chain/PDB checks. They bind the cache, manifests, code, binary, commands and results to hashes. Staging checks the complete reported contract against the requested counts and thresholds, requires typed zero-overlap counts, and validates all private artifacts and pair summaries against their recorded hashes; a status-only receipt cannot pass. `--audit-threads` controls their CPU use. Source/manifest changes between verification and auditing fail staging.

Staging creates the directory with owner-only access, retains step logs privately, and refuses existing directories. Paths inside this repository must be under ignored `.nanofold_private/`; external private directories are also supported. `candidate_status.json` becomes `manifests_verified` only after every step succeeds, sets `target_sequences_audited=true`, and includes the target-audit receipt hash. Failed or interrupted stages must not be adopted; preserve their diagnostics and start a fresh directory. This receipt explicitly leaves processed-tensor and MSA validation outstanding. The direct builders remain available for specialized workflows; their output alone does not pass this staging gate. `scripts/audit_cache_targets.py --help` describes the standalone cache-target audit.

The secret environment variable is `NANOFOLD_HIDDEN_SPLIT_SALT`; alternatively pass `--hidden-split-salt-file` to the generator. Do not expose the salt or its digest in public artifacts. The downloader pins CATH 4.4.0, verifies existing locked bytes instead of repinning them, and records unavailable optional sources. Source filenames are resolved under `--out-dir` by the downloader and `--metadata-sources-dir` by the metadata builder. Use a new directory/lock for a deliberate source update.

`build_hidden_manifest.py --sampling-policy cath-v4` is available for a fixed candidate public contract; pass its `--lock-file` and public manifest paths, metadata/cluster/group-map and receipt overrides plus the bound eligibility receipt and exclusion catalog if relocated, a fresh hidden output directory/private lock, and the secret. The requested hidden size and seed must match the source lock. Published-manifest regeneration wrappers refuse writes; `full_official_data_refresh.sh --skip-manifest-regen` rebuilds/audits frozen assets only.

Prepare CATH v4 assets from the **same validated raw-query and historical-structure bytes** after building the two directed MSA filters and their public split policy. The following assumes the verified historical members are available under one root; use the paths bound by the eligibility receipt and recheck hashes if constructing a consolidated source view. Process public `all.txt` once with training and public-validation filter assignments, then process hidden targets into separate feature/label roots with the same structure predicate and no row filter. Preserve query row 0 in every split.

```bash
stage=.nanofold_private/cath-v4-candidate-01
historical_mmcif=.nanofold_private/pinned-structures/pdb_data/mmcif_files
msa_policy="$stage/public-msa-policy.json"
python scripts/preprocess.py \
  --manifest "$stage/public/all.txt" \
  --chain-data-cache "$cache" \
  --raw-root "$raw" \
  --mmcif-root "$historical_mmcif" \
  --processed-features-dir "$stage/features" \
  --processed-labels-dir "$stage/labels" \
  --msa-filter-policy "$msa_policy" \
  --disable-templates \
  --min-projection-seq-identity 0.90 \
  --min-projection-coverage 0.70 \
  --min-projection-aligned-fraction 0.70 \
  --min-projection-valid-ca 32
```

These fixed projection thresholds are the eligibility predicate; `preprocess.py --strict` is a different option requiring exact coordinate-sequence agreement and is not interchangeable with this policy. Keep `--allow-failures` off. The acquired MSA query must equal the pinned cache target exactly. Cache bytes, selected target identity, raw assets, split policy, effective filter and preprocessing configuration are bound into output provenance. `--skip-existing` permits only matching validated outputs; changed inputs require fresh roots. Complete target identity, label geometry, residual homology audits and fingerprints remain required before adoption.

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

Use directional MSA exclusions: train removes public-plus-hidden target homologs, public validation removes hidden target homologs, and hidden remains unfiltered. No filter uses its source split's own targets or an earlier split as the exclusion set. Run strict MSA audits for train→public validation and public-all→hidden, using separate held-out feature roots where needed. `--strict --run-mmseqs` searches held-out targets against both source targets and non-query MSA rows of at least 20 residues, and fails on either kind of hit. Both searches request actual alignment identity and validate numeric fields, known IDs, coverage and expected output; missing output cannot count as zero hits. Target pairs use all-pair local alignment with prefiltering, masking and composition correction disabled and E-value limit `1e308`. Standalone MSA-row searches use explicitly recorded heuristic settings unless `--exhaustive` is supplied. A zero-hit result is limited to those alignment/search contracts. A filter digest proves filter identity, not absence of residual homology. Audit after filtering and before fingerprinting. Keep identifier-bearing reports private.

For all-pair MSA comparisons, add `--exhaustive` to `build_msa_row_filter.py` and use `audit_msa_split.py --strict --run-mmseqs --exhaustive`. The mode uses protein local alignments with actual alignment identity, bidirectional coverage, `--prefilter-mode 2`, masking/composition correction disabled and E-value limit `1e308`. Without checkpointed batching, `--max-seqs` must cover the whole searched source population: unique MSA sequences for filtering/MSA auditing, and source target records for the target audit. With checkpointed batching, it must cover every record in each source batch. A cap too small fails; it cannot silently truncate an exhaustive search. The full-refresh wrapper uses exhaustive checkpointed searches for both directed filters and both residual audits. Standalone commands must specify `--exhaustive` on every search.

Before allocating that search, run the filter builder with the same source/held-out inputs plus `--exhaustive --inventory-only --output <private-inventory.json>`. This scans and hashes source rows without invoking MMseqs or requiring its executable. The inventory reports `source_unique_sequence_count`, `heldout_unique_sequence_count`, their product `all_pair_comparisons`, `required_exhaustive_max_seqs`, and `source_complete`. It is an estimate of search size, not a wall-time estimate or a completed filter; preprocessing rejects it as a filter. Runtime and output size can be much larger than heuristic search. Missing sources remain failures for release preparation, and non-query rows shorter than 20 residues are outside this search scope. All-pair here means every collected query/source pair under the MMseqs local-alignment contract, not enumeration of all possible alignments or proof against every evolutionary relationship.

### Bounded, resumable exhaustive MSA searches

Use `--exhaustive --checkpoint-dir <private-search-directory>` with the filter builder, or `--strict --run-mmseqs --exhaustive --checkpoint-dir <private-search-directory>` with the split auditor. Rerun the same command to resume; no separate resume flag is needed. Each distinct search needs its own checkpoint directory. The auditor requires anonymous headers, so `--include-identifiers-in-mmseqs` cannot be combined with checkpointed batching. It keeps target and MSA searches under separate `train_targets` and `train_msa` subdirectories.

Build the two directed filters before preprocessing. Keep their checkpoint directories separate and repeat the corresponding command after interruption:

```bash
train_filter="$stage/train-msa-filter.json"
val_filter="$stage/val-msa-filter.json"
python scripts/build_msa_row_filter.py \
  --chain-data-cache "$cache" \
  --source-manifest "$stage/public/train.txt" \
  --heldout-manifest "$stage/public/val.txt" \
  --heldout-manifest "$stage/hidden/hidden_val.txt" \
  --raw-root "$raw" \
  --exhaustive \
  --checkpoint-dir "$stage/exhaustive-train-msa" \
  --output "$train_filter"

python scripts/build_msa_row_filter.py \
  --chain-data-cache "$cache" \
  --source-manifest "$stage/public/val.txt" \
  --heldout-manifest "$stage/hidden/hidden_val.txt" \
  --raw-root "$raw" \
  --exhaustive \
  --checkpoint-dir "$stage/exhaustive-val-msa" \
  --output "$val_filter"
```

Neither filter scans hidden source MSAs or is applied to hidden preprocessing. Bind the two completed artifacts and all three manifests into a private public-preprocessing policy. The manifests must be duplicate-free and disjoint; both filters must cover their complete raw source MSAs with the same cache, 30% identity and 80% bidirectional coverage. Retain both complete checkpoint directories. The policy loader verifies the source partitions and every job, then rederives the excluded sequence set from alignment hits and exact query/source intersections. An unbatched exploratory filter cannot satisfy this policy:

```bash
python scripts/build_msa_filter_policy.py \
  --train-manifest "$stage/public/train.txt" \
  --val-manifest "$stage/public/val.txt" \
  --hidden-manifest "$stage/hidden/hidden_val.txt" \
  --train-msa-row-filter "$train_filter" \
  --val-msa-row-filter "$val_filter" \
  --output "$stage/public-msa-policy.json"
```

Pass this file through `preprocess.py --msa-filter-policy` with the same `--chain-data-cache` for the exact public train/validation union in `all.txt`. Do not combine it with `--msa-row-filter`. Preprocessing validates the configured MSA names and each consumed raw-file path/hash against the filter evidence. Process hidden targets in a separate output root without either filter option; hidden MSAs retain their permitted rows, subject to ordinary validation, deduplication and depth limits. The two residual audit directions remain train→public validation and public-all→hidden.

| Option | Default | Contract |
|---|---:|---|
| `--source-batch-size` | 1,000,000 | Positive; at most 1,000,000 source records per job. |
| `--query-batch-size` | 256 | Positive; at most 256 query records per job. |
| `--max-seqs` | 1,000,000 | Must cover the actual source batch; native acceptance/rejection limits also cover that batch. |
| `--search-disk-reserve-gib` | 100 | Free-disk reserve checked while searching. |
| `--search-max-rss-gib` | 48 | Sampled combined caller/native-process-tree RSS ceiling. |

The immutable plan binds complete query/source FASTAs, deterministic partitions, search parameters, the resolved MMseqs binary/hash/version and search implementation. Completed-job receipts bind native logs and alignments. Resume revalidates this evidence and skips only verified complete jobs; interrupted or failed attempts remain available for diagnosis and are rerun in fresh attempt directories. Changed inputs or search contracts require a separate plan. The final result combines every grid job only after all have completed and inputs are revalidated; partial job completion is not a valid filter or passed audit. Batched evidence references individual job alignments through its receipt; it does not publish a single merged M8 file.

Legacy plans default to sequence lengths of 1–256 residues; full-parent plans explicitly pin their larger `--max-sequence-length`. The existing 20-residue minimum for non-query MSA rows remains. Native numeric limits must be valid; batching is not a reason to truncate returned hits. Disk/RSS monitoring fails closed but is sampled, not an operating-system memory cap. Local process inspection must work before search starts; on systems with sandbox restrictions, run with the required local process permissions. Inventory mode can use the same batch flags and checkpoint-directory option to report the per-batch hit-capacity requirement without launching MMseqs or creating a checkpoint. In that mode `required_exhaustive_max_seqs` uses the smaller of the source population and source-batch size; the unbatched inventory uses the whole source population.

Batching bounds individual jobs and enables reuse of completed work; it does not reduce the total query-by-source comparison count. Build both directed exhaustive filters, preprocess public chains with their explicit split policy and hidden chains without a filter, then rerun both residual split audits in exhaustive mode. `extend_msa_row_filter_from_audit.py` accepts a single-alignment audit only; batched reports are rejected and cannot be used as if one shard represented the full search.

The row filter also removes canonical exact matches independently of MMseqs, so
low-complexity masking cannot retain an identical held-out target. All supplied
processed roots contribute their source variants. Audit-based filter extension
requires the matching summary, ordered source-record digest and alignment-output
hash; changed source content or ordering requires a fresh audit.

Preprocessing changes also require a new asset/fingerprint version. Schema-3 files bind configuration, implementation and raw input bytes; changed provenance cannot be resumed into existing outputs. Their run metadata is mandatory during fingerprint creation, including when someone removes or downgrades it. Validation masks identity-mismatched labels, checks atom/token consistency and detects mixed feature/label pairs. Strict alignment mode still enforces the valid-CA minimum. Optional research templates require auditable polymer residue indices and sequence identity, so missing coordinates cannot shift template positions.

Deposited-polymer projection accepts equivalent placements of unresolved residues only when every optimal alignment agrees on all observed residue positions. It fails if that mapping differs or more than 1,024 optimal alignments prevent the bounded uniqueness check from completing. Projection without deposited polymer context remains strictly unambiguous.

To evaluate whether sampling improves learning, preregister matched comparisons with fixed feature access, training budget, optimization, evaluation set and representative policy. Use several sampling and training seeds. Report mean performance and cluster/architecture-level uncertainty alongside annotation and family coverage. Compare proportional architecture sampling with diversity-weighted or capped-superfamily alternatives as separate experiments. Report topology/superfamily holdouts and temporal holdouts separately; mixing these objectives into the main composition-matched score changes what the benchmark measures. None of the implementation checks establishes a model-quality gain without those experiments.
