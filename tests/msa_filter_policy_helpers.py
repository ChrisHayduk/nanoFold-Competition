"""Small on-disk directed-filter evidence for policy integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nanofold import mmseqs
from nanofold.chain_paths import chain_data_dir


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_valid_policy_case(root: Path, *, chains: dict[str, str] | None = None) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in ("cache", "raw_root", "policy_path", "train_manifest", "val_manifest", "hidden_manifest", "train_filter", "val_filter")}
    chains = chains or {"train": "trn1_A", "val": "val1_A", "hidden": "hid1_A"}
    sequences = {"train": "A" * 20, "val": "C" * 20, "hidden": "D" * 20}
    paths["cache"].write_text(json.dumps({chains[role]: {"sequence": seq} for role, seq in sequences.items()}))
    manifests = {}
    for role, chain in chains.items():
        path = paths[f"{role}_manifest"]
        path.write_text(chain + "\n")
        manifests[role] = {"path": str(path), "sha256": digest_file(path), "chain_count": 1}
    filters = {}
    binary = root / "fixture-mmseqs"
    binary.write_text("#!/bin/sh\nprintf 'fixture\\n'\n")
    binary.chmod(0o755)

    def native(command: list[str], log: Path, checkpoint: Path, **kwargs: Any) -> dict[str, Any]:
        query = list(mmseqs._batch_records(Path(command[2])))
        source = list(mmseqs._batch_records(Path(command[3])))
        matches = [f"{qh}\t{sh}\t100\t20\t1\t1\t0\t40\n" for qh, qs in query for sh, ss in source if qs == ss]
        Path(command[4]).write_text("".join(matches))
        log.write_text(f"Query database size: {len(query)} type: Aminoacid\nTarget database size: {len(source)} type: Aminoacid\n"
                       f"{len(query) * len(source)} alignments calculated\n{len(matches)} sequence pairs passed the thresholds\n")
        return {"elapsed_seconds": 0.1, "sampled_peak_rss_bytes": 1024, "minimum_sampled_free_disk_bytes": 10**15}
    for role in ("train", "val"):
        msa = chain_data_dir(paths["raw_root"] / "roda_pdb", chains[role]) / "uniref90_hits.a3m"
        msa.parent.mkdir(parents=True)
        rows = [sequences[role], sequences["hidden"], sequences["val" if role == "train" else "train"]]
        msa.write_text("".join(f">row{index}\n{row}\n" for index, row in enumerate(rows)))
        paths[f"{role}_msa"] = msa
        assets = [{"path": str(msa), "sha256": digest_file(msa)}]
        target_roles = ["val", "hidden"] if role == "train" else ["hidden"]
        exclusions = sorted(hashlib.sha256(sequences[target].encode()).hexdigest() for target in target_roles)
        flags = ["--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "--alignment-mode", "3", "--seq-id-mode", "0",
                 "--search-type", "1", "--dbtype", "1", "--prefilter-mode", "2", "-e", "1e308", "-s", "5.7",
                 "--mask", "0", "--comp-bias-corr", "0", "--threads", "1", "--max-seqs", "2", "--max-accept", "2",
                 "--max-rejected", "2", "--format-output", "query,target,pident,alnlen,qcov,tcov,evalue,bits"]
        payload: dict[str, Any] = {
            "schema_version": 1, "filter_type": "heldout_target_homology", "exhaustive": True,
            "search_contract": "mmseqs-local-all-pairs-v1",
            "threshold": {"min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 0, "alignment_mode": 3, "seq_id_mode": 0},
            "source": {"mode": "raw_a3m", "complete": True, "minimum_non_query_sequence_length": 20,
                       "manifests": [manifests[role]], "raw_root": str(paths["raw_root"]), "msa_names": ["uniref90_hits.a3m"],
                       "stats": {"chains": 1, "missing_msa_files": 0, "rows": 3, "non_query_rows": 2, "unique_non_query_rows": 2}},
            "heldout": {"manifests": [manifests[target] for target in target_roles], "unique_target_sequence_count": len(target_roles)},
            "chain_data_cache": {"path": str(paths["cache"]), "sha256": digest_file(paths["cache"])},
            "excluded_sequence_sha256": exclusions, "excluded_sequence_count": len(exclusions),
            "exact_match_excluded_sequence_count": len(exclusions),
            "mmseqs": {"binary": "fixture-mmseqs", "version": "fixture", "max_seqs": 2,
                       "command": ["fixture-mmseqs", "easy-search", "<heldout_targets.fasta>", "<source_msa_rows.fasta>", "<output.m8>", "<tmp_dir>", *flags],
                       "possible_pair_count": 2 * len(target_roles), "hit_count": len(exclusions),
                       "search_parameters": {"prefilter_mode": 2, "evalue": 1e308, "sensitivity": 5.7, "mask": 0,
                                             "comp_bias_corr": 0, "search_type": 1, "dbtype": 1, "max_accept": 2, "max_rejected": 2}},
            "input_binding": {"source_assets": assets,
                              "source_asset_manifest_sha256": hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(),
                              **{name: "a" * 64 for name in ("source_fasta_sha256", "heldout_fasta_sha256", "m8_sha256", "source_sequence_set_sha256", "heldout_sequence_set_sha256")}},
        }
        query_path, source_path = root / f"{role}-query.fasta", root / f"{role}-source.fasta"
        queries = [sequences[target] for target in target_roles]
        sources = sorted((hashlib.sha256(sequence.encode()).hexdigest(), sequence) for sequence in rows[1:])
        query_path.write_text("".join(f">heldout_{index}\n{sequence}\n" for index, sequence in enumerate(queries, start=1)))
        source_path.write_text("".join(f">seq_{digest}\n{sequence}\n" for digest, sequence in sources))
        input_files = [paths["cache"], paths[f"{role}_manifest"], *(paths[f"{target}_manifest"] for target in target_roles)]
        context_sha = hashlib.sha256(json.dumps({"files": [{"path": str(path.resolve()), "sha256": digest_file(path)} for path in sorted(input_files)],
                                               "source_assets": assets}, sort_keys=True).encode()).hexdigest()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mmseqs, "_batch_execute", native)
            patch.setattr(mmseqs, "_batch_resource_check", lambda *args, **kwargs: (10**15, 1024))
            result = mmseqs.run_batched_mmseqs(query_fasta=query_path, source_fasta=source_path,
                checkpoint_dir=root / f"{role}-checkpoint", mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8,
                threads=1, max_seqs=2, source_batch_size=2, query_batch_size=1, reserve_disk_bytes=0,
                input_binding_sha256=context_sha)
        payload["mmseqs"]["binary"] = str(binary)
        payload["mmseqs"]["command"] = []
        payload["mmseqs"]["batching"] = mmseqs.batched_mmseqs_summary(result)
        payload["input_binding"].pop("m8_sha256")
        payload["input_binding"].update({"source_fasta_sha256": digest_file(source_path), "heldout_fasta_sha256": digest_file(query_path),
            "source_sequence_set_sha256": hashlib.sha256("".join(f"{digest}\n" for digest, _ in sources).encode()).hexdigest(),
            "heldout_sequence_set_sha256": hashlib.sha256("".join(f"{digest}\n" for digest in sorted(hashlib.sha256(sequence.encode()).hexdigest() for sequence in queries)).encode()).hexdigest(),
            "batch_receipt_sha256": result["receipt_sha256"]})
        path = paths[f"{role}_filter"]
        path.write_text(json.dumps(payload, sort_keys=True))
        filters[role] = {"path": str(path), "sha256": digest_file(path)}
    paths["policy_path"].write_text(json.dumps({"schema_version": 1, "policy_type": "directed-heldout-v1", "manifests": manifests, "filters": filters}, sort_keys=True))
    return paths


def refresh_filter_binding(paths: dict[str, Path], role: str) -> None:
    policy = json.loads(paths["policy_path"].read_text())
    policy["filters"][role]["sha256"] = digest_file(paths[f"{role}_filter"])
    paths["policy_path"].write_text(json.dumps(policy, sort_keys=True))
