from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_structure_metadata_module():
    module_path = Path("scripts/build_structure_metadata.py").resolve()
    spec = importlib.util.spec_from_file_location("build_structure_metadata_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_rows_reject_nonstandard_sequences(tmp_path: Path) -> None:
    module = _load_structure_metadata_module()
    candidate_rows = getattr(module, "_candidate_rows")
    data = {
        "1oky_A": {"seq": "ARNDCQEGHILKMFPSTWYV" * 3, "resolution": 2.0},
        "1bad_A": {"seq": "AXXX", "resolution": 2.0},
    }
    candidates, rejects = candidate_rows(
        data,
        min_len=4,
        max_len=256,
        max_resolution=3.0,
        max_unknown_aa_fraction=0.0,
    )
    assert [item["chain_id"] for item in candidates] == ["1oky_A"]
    assert rejects["unknown_aa_fraction"] == 1


def test_build_metadata_writes_deterministic_candidate_manifest(tmp_path: Path) -> None:
    module = _load_structure_metadata_module()
    main = getattr(module, "main")

    cache = tmp_path / "chain_data_cache.json"
    cache.write_text(
        json.dumps(
            {
                "1oky_A": {
                    "seq": "ARNDCQEGHILKMFPSTWYV" * 3,
                    "resolution": 2.0,
                    "oligomeric_count": 1,
                }
            }
        )
    )
    metadata = tmp_path / "structure_metadata.json"
    candidates = tmp_path / "candidates.txt"
    main(
        [
            "--chain-data-cache",
            str(cache),
            "--metadata-out",
            str(metadata),
            "--metadata-sources-dir",
            str(tmp_path / "metadata_sources"),
            "--candidate-manifest-out",
            str(candidates),
        ]
    )

    obj = json.loads(metadata.read_text())
    assert obj["summary"]["accepted"] == 1
    assert obj["summary"]["rejected"] == 0
    assert obj["chains"][0]["secondary_structure_source"] == "domain_class_projection:secondary_structure_fallback"
    assert candidates.read_text() == "1oky_A\n"


def test_build_metadata_enforces_required_feature_exclusions(tmp_path: Path) -> None:
    module = _load_structure_metadata_module()
    main = getattr(module, "main")

    cache = tmp_path / "chain_data_cache.json"
    cache.write_text(
        json.dumps(
            {
                "1oky_A": {
                    "seq": "ARNDCQEGHILKMFPSTWYV" * 3,
                    "resolution": 2.0,
                    "oligomeric_count": 1,
                },
                "2bad_A": {
                    "seq": "ARNDCQEGHILKMFPSTWYV" * 3,
                    "resolution": 2.0,
                    "oligomeric_count": 1,
                },
            }
        )
    )
    exclusions = tmp_path / "feature_exclusions.txt"
    exclusions.write_text("# required asset coverage\n2bad_A\n")
    metadata = tmp_path / "structure_metadata.json"
    candidates = tmp_path / "candidates.txt"
    main(
        [
            "--chain-data-cache",
            str(cache),
            "--metadata-out",
            str(metadata),
            "--metadata-sources-dir",
            str(tmp_path / "metadata_sources"),
            "--feature-exclusion-list",
            str(exclusions),
            "--candidate-manifest-out",
            str(candidates),
        ]
    )

    obj = json.loads(metadata.read_text())
    chains = {item["chain_id"]: item for item in obj["chains"]}
    assert obj["summary"]["accepted"] == 1
    assert obj["summary"]["rejected"] == 1
    assert obj["summary"]["reject_reasons"]["missing_required_feature_asset"] == 1
    assert obj["source"]["feature_exclusion_list"]["chain_count"] == 1
    assert chains["2bad_A"]["reject_reasons"] == ["missing_required_feature_asset"]
    assert candidates.read_text() == "1oky_A\n"


def test_build_metadata_enforces_processability_exclusions(tmp_path: Path) -> None:
    module = _load_structure_metadata_module()
    main = getattr(module, "main")

    cache = tmp_path / "chain_data_cache.json"
    cache.write_text(
        json.dumps(
            {
                "1oky_A": {
                    "seq": "ARNDCQEGHILKMFPSTWYV" * 3,
                    "resolution": 2.0,
                    "oligomeric_count": 1,
                },
                "2bad_A": {
                    "seq": "ARNDCQEGHILKMFPSTWYV" * 3,
                    "resolution": 2.0,
                    "oligomeric_count": 1,
                },
            }
        )
    )
    exclusions = tmp_path / "processability_exclusions.txt"
    exclusions.write_text("# official atom14 label processability gate\n2bad_A\n")
    metadata = tmp_path / "structure_metadata.json"
    candidates = tmp_path / "candidates.txt"
    main(
        [
            "--chain-data-cache",
            str(cache),
            "--metadata-out",
            str(metadata),
            "--metadata-sources-dir",
            str(tmp_path / "metadata_sources"),
            "--processability-exclusion-list",
            str(exclusions),
            "--candidate-manifest-out",
            str(candidates),
        ]
    )

    obj = json.loads(metadata.read_text())
    chains = {item["chain_id"]: item for item in obj["chains"]}
    assert obj["summary"]["accepted"] == 1
    assert obj["summary"]["rejected"] == 1
    assert obj["summary"]["reject_reasons"]["failed_official_label_processability"] == 1
    assert obj["source"]["processability_exclusion_list"]["chain_count"] == 1
    assert chains["2bad_A"]["reject_reasons"] == ["failed_official_label_processability"]
    assert candidates.read_text() == "1oky_A\n"


def test_external_domain_sources_are_normalized(tmp_path: Path) -> None:
    module = _load_structure_metadata_module()
    load_cath = getattr(module, "_load_cath_annotations")
    load_scope = getattr(module, "_load_scope_annotations")
    load_ecod = getattr(module, "_load_ecod_annotations")

    cath = tmp_path / "cath-domain-list.txt"
    cath.write_text("1abcA00 1 10 20 30 1 1 1 1 60 2.0\n2defB00 2 10 20 30 1 1 1 1 70 2.0\n")
    scope = tmp_path / "dir.cla.scope.txt"
    scope.write_text("d1abcA_ 1abc A: a.1.1.1 1 cl=1,cf=2,sf=3,fa=4,dm=5,sp=6,px=7\n")
    ecod = tmp_path / "ecod.latest.domains.txt"
    ecod.write_text("000000001\te3ghiC1\tmanual\talpha/beta plait\t3ghi\tC\tC:1-100\n")

    assert load_cath(cath)["1abc_A"] == {"alpha"}
    assert load_cath(cath)["2def_B"] == {"beta"}
    assert load_scope(scope)["1abc_A"] == {"alpha"}
    assert load_ecod(ecod)["3ghi_C"] == {"alpha_beta"}
