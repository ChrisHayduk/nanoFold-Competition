"""Directed MSA filtering through raw preprocessing, resume and fingerprinting."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import gemmi
import numpy as np
import pytest

import nanofold.mmseqs as mmseqs_engine
from nanofold.a3m import sequence_to_ids
from nanofold.chain_paths import chain_data_dir, chain_npz_path
from nanofold.dataset_integrity import build_dataset_fingerprint
from nanofold.msa_filter_policy import build_msa_filter_policy


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_policy_test", Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _structure(path: Path, residue_name: str) -> None:
    structure = gemmi.Structure()
    model, chain = gemmi.Model(1), gemmi.Chain("A")
    for index in range(32):
        residue = gemmi.Residue()
        residue.name = residue_name
        residue.seqid = gemmi.SeqId(index + 1, " ")
        residue.label_seq = index + 1
        residue.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name, atom.element, atom.occ = "CA", gemmi.Element("C"), 1
        atom.pos = gemmi.Position(index * 3.8, 0, 0)
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))


@pytest.fixture
def directed_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    native = getattr(request, "param", "protocol") == "native"
    native_binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if native and not native_binary:
        pytest.skip("Set MMSEQS_TEST_BINARY for native directed preprocessing integration")
    if not native:
        monkeypatch.setattr(mmseqs_engine, "_batch_resource_check", lambda *args, **kwargs: (10**15, 1024))
    raw, mmcif = tmp_path / "raw", tmp_path / "mmcif"
    mmcif.mkdir()
    ids = {"train": "1abc_A", "val": "2abc_A", "hidden": "3abc_A"}
    query = {"train": "A" * 32, "val": "C" * 32, "hidden": "D" * 32}
    near = {split: sequence[:-1] + "G" for split, sequence in query.items()}
    rows = {
        "train": [query["train"], near["train"], near["val"], near["hidden"], "E" * 32],
        "val": [query["val"], near["val"], near["hidden"], near["train"]],
        "hidden": [query["hidden"], near["hidden"], near["val"], near["train"]],
    }
    manifests = {}
    for split, chain_id in ids.items():
        manifest = tmp_path / f"{split}.txt"
        manifest.write_text(chain_id + "\n")
        manifests[split] = manifest
        chain_dir = chain_data_dir(raw / "roda_pdb", chain_id)
        chain_dir.mkdir(parents=True)
        (chain_dir / "uniref90_hits.a3m").write_text("".join(
            f">row{index}\n{row}\n" for index, row in enumerate(rows[split])
        ))
        _structure(mmcif / f"{chain_id[:4]}.cif", {"train": "ALA", "val": "CYS", "hidden": "ASP"}[split])
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({ids[split]: {"sequence": seq} for split, seq in query.items()}))
    binary = tmp_path / "mmseqs-fixture"
    binary.write_text(f"#!{sys.executable}\n" + '''import pathlib,sys
if sys.argv[1] == 'version':
 print('fixture-v1'); raise SystemExit(0)
def records(path):
 lines=pathlib.Path(path).read_text().splitlines()
 return [(lines[i][1:],lines[i+1]) for i in range(0,len(lines),2)]
threshold=float(sys.argv[sys.argv.index('--min-seq-id')+1])
hits=0
queries,sources=records(sys.argv[2]),records(sys.argv[3])
with pathlib.Path(sys.argv[4]).open('w') as output:
 for q,qs in queries:
  for t,ts in sources:
   identity=sum(a==b for a,b in zip(qs,ts))/len(qs)
   if identity>=threshold:
    hits+=1
    output.write(f'{q}\\t{t}\\t{identity*100}\\t{len(qs)}\\t1\\t1\\t0\\t50\\n')
print(f'Query database size: {len(queries)} type: Aminoacid')
print(f'Target database size: {len(sources)} type: Aminoacid')
print(f'{len(queries)*len(sources)} alignments calculated')
print(f'{hits} sequence pairs passed the thresholds')
''')
    binary.chmod(0o755)
    if native:
        assert native_binary is not None
        binary = Path(native_binary)
    builder = _module("build_msa_row_filter")
    filters = {}
    for split in ("train", "val"):
        filters[split] = tmp_path / f"{split}-filter.json"
        heldout = ("val", "hidden") if split == "train" else ("hidden",)
        args = ["build_msa_row_filter.py", "--source-manifest", str(manifests[split]),
                "--chain-data-cache", str(cache), "--raw-root", str(raw), "--exhaustive",
                "--mmseqs-bin", str(binary), "--threads", "1", "--output", str(filters[split]),
                "--checkpoint-dir", str(tmp_path / f"{split}-checkpoint")]
        for role in heldout:
            args.extend(["--heldout-manifest", str(manifests[role])])
        monkeypatch.setattr(sys, "argv", args)
        builder.main()
    policy = tmp_path / "policy.json"
    payload = build_msa_filter_policy(train_manifest=manifests["train"], val_manifest=manifests["val"],
                                     hidden_manifest=manifests["hidden"], train_filter=filters["train"],
                                     val_filter=filters["val"])
    policy.write_text(json.dumps(payload, sort_keys=True) + "\n")
    public = tmp_path / "all.txt"
    public.write_text(ids["train"] + "\n" + ids["val"] + "\n")
    module = _module("preprocess")
    features, labels = tmp_path / "features", tmp_path / "labels"

    def run(*extra: str, hidden: bool = False):
        args = ["preprocess.py", "--manifest", str(manifests["hidden"] if hidden else public),
                "--raw-root", str(raw), "--mmcif-root", str(mmcif), "--chain-data-cache", str(cache),
                "--processed-features-dir", str(tmp_path / "hidden-features" if hidden else features),
                "--processed-labels-dir", str(tmp_path / "hidden-labels" if hidden else labels),
                "--disable-templates", "--strict", "--fail-fast"]
        if not hidden:
            args.extend(["--msa-filter-policy", str(policy)])
        monkeypatch.setattr(sys, "argv", [*args, *extra])
        module.main()

    return dict(run=run, module=module, ids=ids, rows=rows, near=near, query=query, manifests=manifests,
                features=features, labels=labels, filters=filters, policy=policy, cache=cache, raw=raw,
                tmp_path=tmp_path)


def _rows(path: Path) -> list[list[int]]:
    with np.load(path) as arrays:
        return arrays["msa"].tolist()


@pytest.mark.parametrize("directed_case", ["protocol", "native"], indirect=True)
def test_directed_filtering_retains_within_split_support_and_hidden_msa(directed_case) -> None:
    case = directed_case
    case["run"]()
    case["run"](hidden=True)
    expected = {
        "train": [case["query"]["train"], case["near"]["train"], "E" * 32],
        "val": [case["query"]["val"], case["near"]["val"], case["near"]["train"]],
        "hidden": case["rows"]["hidden"],
    }
    for role, sequences in expected.items():
        root = case["tmp_path"] / "hidden-features" if role == "hidden" else case["features"]
        path = chain_npz_path(root, case["ids"][role])
        assert _rows(path) == [sequence_to_ids(sequence).tolist() for sequence in sequences]
        with np.load(path) as arrays:
            if role == "hidden":
                assert "msa_row_filter_sha256" not in arrays.files
            else:
                assert str(arrays["msa_row_filter_sha256"].item()) == hashlib.sha256(case["filters"][role].read_bytes()).hexdigest()
    metadata = json.loads((case["features"] / "preprocess_meta.json").read_text())
    assert metadata["msa_filter_policy"]["policy_type"] == "directed-heldout-v1"
    assert "hidden" not in metadata["msa_filter_policy"]["splits"]
    for role in ("train", "val"):
        audit = json.loads((case["features"] / f"msa_row_filter_audit.{role}.json").read_text())
        assert audit["chain_count"] == 1
        assert audit["per_chain"][0]["chain_id"] == case["ids"][role]
    fingerprint = build_dataset_fingerprint(processed_features_dir=case["features"], processed_labels_dir=case["labels"],
                                            train_manifest=case["manifests"]["train"], val_manifest=case["manifests"]["val"],
                                            require_no_missing=True)
    assert fingerprint["present_feature_chain_count"] == 2


def test_directed_resume_reuses_correct_roles_and_repairs_wrong_applied_filter(directed_case) -> None:
    case = directed_case
    case["run"]()
    paths = {role: chain_npz_path(case["features"], case["ids"][role]) for role in ("train", "val")}
    mtimes = {role: path.stat().st_mtime_ns for role, path in paths.items()}
    case["run"]("--skip-existing")
    assert {role: path.stat().st_mtime_ns for role, path in paths.items()} == mtimes
    with np.load(paths["val"]) as arrays:
        wrong = dict(arrays)
    wrong["msa_row_filter_sha256"] = np.asarray(hashlib.sha256(case["filters"]["train"].read_bytes()).hexdigest())
    np.savez_compressed(paths["val"], **wrong)
    with pytest.raises(ValueError, match="filter"):
        build_dataset_fingerprint(processed_features_dir=case["features"], processed_labels_dir=case["labels"],
                                  train_manifest=case["manifests"]["train"], val_manifest=case["manifests"]["val"],
                                  require_no_missing=True)
    case["run"]("--skip-existing")
    assert paths["train"].stat().st_mtime_ns == mtimes["train"]
    assert _rows(paths["val"])[1] == sequence_to_ids(case["near"]["val"]).tolist()


def test_policy_mutation_cannot_relabel_existing_processed_root(directed_case) -> None:
    case = directed_case
    case["run"]()
    metadata = (case["features"] / "preprocess_meta.json").read_bytes()
    case["policy"].write_text(case["policy"].read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        case["run"]("--skip-existing")
    assert (case["features"] / "preprocess_meta.json").read_bytes() == metadata


def test_directed_policy_rejects_unsearched_raw_msa_inputs(directed_case, tmp_path: Path) -> None:
    case = directed_case
    unsearched = tmp_path / "different-raw"
    shutil.copytree(case["raw"], unsearched)
    with pytest.raises(ValueError, match="asset|source|MSA"):
        case["run"]("--raw-root", str(unsearched))
    assert not list(case["features"].glob("*.npz"))


def test_depth_cap_follows_dedup_with_or_without_a_filter(tmp_path: Path) -> None:
    module = _module("preprocess")
    (tmp_path / "uniref90_hits.a3m").write_text(">query\nACD\n>duplicate\nACD\n>support\nA-D\n")
    empty_filter = module.MSARowFilter(path=tmp_path / "empty-filter.json", sha256="a" * 64,
                                       excluded_sequence_sha256=frozenset())
    for row_filter in (None, empty_filter):
        msa, deletions, query, _stats = module._read_merged_msa(
            tmp_path, msa_name="uniref90_hits.a3m", msa_names=None, max_msa_seqs=2, msa_row_filter=row_filter,
        )
        assert query == "ACD"
        assert msa.tolist() == [sequence_to_ids("ACD").tolist(), sequence_to_ids("A-D").tolist()]
        assert deletions.shape == msa.shape
