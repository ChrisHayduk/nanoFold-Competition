from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_manifests
from scripts.build_hidden_manifest import _build_hidden_manifest
from scripts.verify_hidden_manifest import verify


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, policy: str = 'cath-v3') -> argparse.Namespace:
    ids = [f'1a{index:02d}_A' for index in range(12)]
    cache = tmp_path / 'cache.json'
    payload = {
        chain: {'sequence': residue * 40, 'seq_length': 40, 'resolution': 2.0}
        for chain, residue in zip(ids, 'ACDEFGHIKLMN', strict=True)
    }
    payload['1a00_B'] = {'sequence': 'Y' * 41, 'seq_length': 41, 'resolution': 2.9}
    _write_json(cache, payload)
    metadata = tmp_path / 'metadata.json'
    _write_json(metadata, {
        'schema_version': 2,
        'source': {'chain_data_cache_sha256': _sha(cache), 'chain_id_namespace': 'author_asym_id'},
        'config': {'min_len': 40, 'max_len': 256, 'max_resolution': 3.0, 'require_monomer': False,
                   'sample_limit': 0, 'max_unknown_aa_fraction': 0.0},
        'summary': {'eligible_candidate_count': len(payload), 'candidate_count': len(payload)},
        'chains': [{
            'chain_id': chain, 'accepted': True, 'domain_architecture_class': 'alpha',
            'domain_architecture_source': 'cath', 'metadata_source_count': 1,
            'cath_annotation_status': 'annotated', 'cath_architecture_ids': ['1.10'],
            'cath_topology_ids': ['1.10.10'], 'cath_homologous_superfamily_ids': ['1.10.10.10'],
            'cath_domains': [{'domain_id': chain.replace('_', '') + '01', 'class_id': '1',
                              'architecture_id': '1.10', 'topology_id': '1.10.10',
                              'homologous_superfamily_id': '1.10.10.10'}],
        } for chain in payload],
    })
    clusters = tmp_path / 'clusters.tsv'
    clusters.write_text(''.join(f'{chain}\t{chain}\n' for chain in ids) + '1a00_A\t1a00_B\n')
    groups = tmp_path / 'groups.tsv'
    group_map = {}
    start = 0
    for size in (3, 2, 2, 1, 1, 1, 1, 1):
        group_map.update({chain: ids[start] for chain in ids[start:start + size]})
        start += size
    group_map['1a00_B'] = group_map['1a00_A']
    groups.write_text(''.join(f'{group}\t{chain}\n' for chain, group in sorted(group_map.items())))
    receipt = tmp_path / 'groups.receipt.json'
    _write_json(receipt, {'original_unit_count': 12})

    # Graph construction/edge provenance has its own integration tests. This
    # fixture isolates group allocation and independent manifest verification.
    def validated_receipt(*args, **kwargs):
        assert kwargs['candidate_ids'] == set(payload)
        return {'original_unit_count': 12}

    monkeypatch.setattr('scripts.build_homology_groups.validate_group_receipt', validated_receipt)
    monkeypatch.setattr('build_homology_groups.validate_group_receipt', validated_receipt)
    monkeypatch.setattr('scripts.build_hidden_manifest.validate_group_receipt', validated_receipt)
    monkeypatch.setattr('build_hidden_manifest.validate_group_receipt', validated_receipt)
    qualification_args = []
    if policy == 'cath-v4':
        eligibility_receipt = tmp_path / 'eligibility.receipt.json'
        approved_exclusions = tmp_path / 'approved-exclusions.json'
        _write_json(eligibility_receipt, {'fixture': 'qualification validated separately'})
        _write_json(approved_exclusions, {'group_ids': ['1a11_A']})
        nested_source = tmp_path / 'qualification-source.txt'
        nested_source.write_text('audited source evidence\n')

        def validated_eligibility(receipt_path, *, approved_group_exclusions_path, original_units, **kwargs):
            assert set(kwargs['split_group_map']) == set(payload)
            selected = {}
            for unit_id, members in original_units.items():
                if '1a00_A' in members:
                    assert members == ['1a00_A', '1a00_B']
                    selected[unit_id] = '1a00_B'
                else:
                    selected[unit_id] = None if members == ['1a11_A'] else members[0]
            return {
                'selected_chain_by_original_unit': selected,
                'exhausted_original_units': [unit for unit, chain in selected.items() if chain is None],
                'approved_excluded_group_ids': ['1a11_A'],
                'validated_source_paths': [str(nested_source)],
                'binding': {
                    'schema_version': 1, 'policy': 'cath-v4',
                    'receipt_sha256': _sha(receipt_path),
                    'approved_group_exclusions_sha256': _sha(approved_group_exclusions_path),
                },
            }

        monkeypatch.setattr('nanofold.representative_eligibility.validate_representative_eligibility_receipt', validated_eligibility)
        qualification_args = [
            '--representative-eligibility-receipt', str(eligibility_receipt),
            '--approved-group-exclusions', str(approved_exclusions),
        ]
    public = tmp_path / 'public'
    private = tmp_path / 'private'
    source_lock = tmp_path / 'source.lock.json'
    private_lock = tmp_path / 'private.lock.json'
    monkeypatch.setenv('NANOFOLD_HIDDEN_SPLIT_SALT', 'test-only-hidden-secret-that-is-long-enough')
    monkeypatch.setattr('sys.argv', [
        'build_manifests.py', '--sampling-policy', policy,
        '--chain-data-cache', str(cache), '--structure-metadata', str(metadata),
        '--cluster-tsv', str(clusters), '--split-group-tsv', str(groups),
        '--split-group-receipt', str(receipt),
        '--train-size', '9' if policy == 'cath-v4' else '8', '--val-size', '1', '--hidden-val-size', '1',
        '--seed', '37', '--out-dir', str(public), '--hidden-out-dir', str(private),
        '--lock-file', str(source_lock), '--private-lock-file', str(private_lock),
        *qualification_args,
    ])
    build_manifests.main()
    return argparse.Namespace(
        chain_data_cache=cache, lock_file=source_lock, private_lock_file=private_lock,
        public_train_manifest=public / 'train.txt', public_val_manifest=public / 'val.txt',
        public_all_manifest=public / 'all.txt', hidden_manifest=private / 'hidden_val.txt',
        structure_metadata=None, cluster_tsv=None, split_group_tsv=None, mmseqs_bin=None,
        split_group_receipt=None, private_processability_exclusion_list=None, json_out=None,
        representative_eligibility_receipt=None, approved_group_exclusions=None,
    )


def test_grouped_candidate_round_trips_complete_coverage(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    report = verify(args)
    assert report['status'] == 'ok'
    assert report['counts'] == {'train': 8, 'val': 1, 'hidden_val': 1}
    assert report['sampling_policy'] == 'cath-v3'
    assert report['group_coverage'] == {
        'required': 'all_groups', 'all_group_coverage_verified': True,
        'complete_group_count': 8, 'covered_group_count': 8, 'selection_unit_count': 12,
        'selection_unit_ids_sha256': hashlib.sha256(
            ''.join(f'1a{index:02d}_A\n' for index in range(12)).encode()
        ).hexdigest(),
        'split_group_counts': {'train': 6, 'val': 1, 'hidden_val': 1},
    }


def test_qualified_candidate_round_trips_retained_coverage_without_changing_original_graph(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')
    report = verify(args)
    qualified_ids = ['1a00_B', *(f'1a{index:02d}_A' for index in range(1, 11))]
    assert report['sampling_policy'] == 'cath-v4'
    assert report['counts'] == {'train': 9, 'val': 1, 'hidden_val': 1}
    assert report['group_coverage'] == {
        'required': 'all_retained_groups', 'retained_group_coverage_verified': True,
        'original_group_count': 8, 'approved_excluded_group_count': 1,
        'retained_group_count': 7, 'covered_group_count': 7,
        'original_unit_count': 12, 'selection_unit_count': 11,
        'selection_unit_ids_sha256': hashlib.sha256(''.join(f'{chain}\n' for chain in sorted(qualified_ids)).encode()).hexdigest(),
        'split_group_counts': {'train': 5, 'val': 1, 'hidden_val': 1},
    }
    assert report['stratification']['reference_population'] == 'source_validated_disjoint_units'
    selected = set(args.public_all_manifest.read_text().splitlines() + args.hidden_manifest.read_text().splitlines())
    assert selected == set(qualified_ids)
    assert len((tmp_path / 'groups.tsv').read_text().splitlines()) == 13


def test_group_bytes_are_locked_even_when_membership_is_unchanged(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    groups = tmp_path / 'groups.tsv'
    groups.write_text(groups.read_text() + '\n')
    with pytest.raises(ValueError, match='[Gg]roup.*SHA256'):
        verify(args)


def test_group_input_can_be_relocated_without_repinning(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    relocated = tmp_path / 'moved.tsv'
    (tmp_path / 'groups.tsv').rename(relocated)
    args.split_group_tsv = relocated
    assert verify(args)['status'] == 'ok'


def test_verification_report_cannot_replace_group_input(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    args.json_out = tmp_path / 'groups.tsv'
    before = args.json_out.read_bytes()
    with pytest.raises(ValueError, match='alias'):
        verify(args)
    assert args.json_out.read_bytes() == before


def _repin_source(args: argparse.Namespace, public: dict) -> None:
    _write_json(args.lock_file, public)
    private = json.loads(args.private_lock_file.read_text())
    private['source_lock_sha256'] = _sha(args.lock_file)
    private['split_groups'] = public['split_groups']
    _write_json(args.private_lock_file, private)


def _repin_manifests(args: argparse.Namespace) -> None:
    public = json.loads(args.lock_file.read_text())
    args.public_all_manifest.write_text(''.join(f'{chain}\n' for chain in sorted(
        set(args.public_train_manifest.read_text().splitlines()) | set(args.public_val_manifest.read_text().splitlines())
    )))
    private = json.loads(args.private_lock_file.read_text())
    for name, path in (('train', args.public_train_manifest), ('val', args.public_val_manifest), ('all', args.public_all_manifest)):
        count = len(path.read_text().splitlines())
        public['outputs'][f'{name}_manifest_sha256'] = _sha(path)
        public['outputs']['unique_count' if name == 'all' else f'{name}_count'] = count
        private['public_contract'][f'{name}_manifest_sha256'] = _sha(path)
        private['public_contract'][f'{name}_count'] = count
        if name != 'all':
            public['args'][f'{name}_size'] = count
    _write_json(args.private_lock_file, private)
    _repin_source(args, public)


@pytest.mark.parametrize('policy', ['cath-v3', 'cath-v4'])
def test_full_group_coverage_is_checked_independently_of_rehashed_manifests(tmp_path, monkeypatch, policy) -> None:
    from collections import Counter

    args = _fixture(tmp_path, monkeypatch, policy=policy)
    group_map = {member: group for group, member in (line.split() for line in (tmp_path / 'groups.tsv').read_text().splitlines())}
    train = args.public_train_manifest.read_text().splitlines()
    counts = Counter(group_map[chain] for chain in train)
    dropped = next(chain for chain in train if counts[group_map[chain]] == 1)
    args.public_train_manifest.write_text(''.join(f'{chain}\n' for chain in train if chain != dropped))
    _repin_manifests(args)
    with pytest.raises(RuntimeError, match='full homology groups uncovered'):
        verify(args)


@pytest.mark.parametrize('policy', ['cath-v3', 'cath-v4'])
def test_homologous_original_units_cannot_cross_splits_after_rehashing(tmp_path, monkeypatch, policy) -> None:
    from collections import Counter

    args = _fixture(tmp_path, monkeypatch, policy=policy)
    group_map = {member: group for group, member in (line.split() for line in (tmp_path / 'groups.tsv').read_text().splitlines())}
    train = args.public_train_manifest.read_text().splitlines()
    val = args.public_val_manifest.read_text().splitlines()
    counts = Counter(group_map[chain] for chain in train)
    moved = next(chain for chain in train if counts[group_map[chain]] > 1)
    args.public_train_manifest.write_text(''.join(f'{val[0] if chain == moved else chain}\n' for chain in train))
    args.public_val_manifest.write_text(moved + '\n')
    _repin_manifests(args)
    with pytest.raises(RuntimeError, match='share 1 full homology groups'):
        verify(args)


@pytest.mark.parametrize('field', ['split_group_tsv', 'split_group_receipt'])
def test_group_policy_requires_complete_source_arguments(tmp_path, monkeypatch, field) -> None:
    args = _fixture(tmp_path, monkeypatch)
    public = json.loads(args.lock_file.read_text())
    del public['args'][field]
    _repin_source(args, public)
    with pytest.raises(ValueError, match=f'missing args.{field}'):
        verify(args)


@pytest.mark.parametrize('field,value', [
    ('group_count', 7), ('selection_unit_count', 11), ('coverage', 'some_groups'),
    ('allocator_policy', 'other'), ('selection_unit_ids_sha256', '0' * 64),
])
def test_group_population_and_coverage_contract_cannot_be_relabelled(tmp_path, monkeypatch, field, value) -> None:
    args = _fixture(tmp_path, monkeypatch)
    public = json.loads(args.lock_file.read_text())
    public['split_groups'][field] = value
    _repin_source(args, public)
    with pytest.raises(ValueError, match='group.*population|group.*coverage|unit count'):
        verify(args)


def test_private_group_provenance_is_mandatory(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    private = json.loads(args.private_lock_file.read_text())
    private.pop('split_groups')
    _write_json(args.private_lock_file, private)
    with pytest.raises(ValueError, match='split-group provenance'):
        verify(args)


def test_receipt_semantic_rejection_cannot_be_ignored(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)

    def reject_receipt(*args, **kwargs):
        raise ValueError('Graph source provenance rejected')

    monkeypatch.setattr('build_hidden_manifest.validate_group_receipt', reject_receipt)
    with pytest.raises(ValueError, match='source provenance rejected'):
        verify(args)


def test_group_receipt_bytes_remain_pinned(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    receipt = tmp_path / 'groups.receipt.json'
    receipt.write_text(receipt.read_text() + '\n')
    with pytest.raises(ValueError, match='receipt SHA256'):
        verify(args)


def test_unselected_raw_chain_cannot_bridge_original_units_across_groups(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    groups = tmp_path / 'groups.tsv'
    group_map = {member: group for group, member in (line.split() for line in groups.read_text().splitlines())}
    group_map['1a00_B'] = group_map['1a03_A']
    groups.write_text(''.join(f'{group}\t{chain}\n' for chain, group in sorted(group_map.items())))
    public = json.loads(args.lock_file.read_text())
    public['split_groups'].update(sha256=_sha(groups), assignments_sha256=build_manifests._cluster_assignments_sha256(group_map))
    _repin_source(args, public)
    with pytest.raises(RuntimeError, match='divides an original'):
        verify(args)


@pytest.mark.parametrize('policy', ['cath-v3', 'cath-v4'])
def test_hidden_only_group_regeneration_preserves_public_and_group_ownership(tmp_path, monkeypatch, policy) -> None:
    args = _fixture(tmp_path, monkeypatch, policy=policy)
    public_before = [path.read_bytes() for path in (
        args.public_train_manifest, args.public_val_manifest, args.public_all_manifest,
    )]
    generated = _build_hidden_manifest(argparse.Namespace(
        **{key: value for key, value in vars(args).items() if key not in {'private_lock_file', 'hidden_manifest'}},
        sampling_policy=policy, hidden_out_dir=tmp_path / 'regenerated',
        private_lock_file=tmp_path / 'regenerated.lock.json',
        hidden_split_salt_env='NANOFOLD_HIDDEN_SPLIT_SALT', hidden_split_salt_file=None,
        hidden_val_size=None, seed=None,
    ))
    assert generated['hidden_manifest'].read_bytes() == args.hidden_manifest.read_bytes()
    assert public_before == [path.read_bytes() for path in (
        args.public_train_manifest, args.public_val_manifest, args.public_all_manifest,
    )]
    args.hidden_manifest = generated['hidden_manifest']
    args.private_lock_file = generated['private_lock']
    coverage_key = 'retained_group_coverage_verified' if policy == 'cath-v4' else 'all_group_coverage_verified'
    assert verify(args)['group_coverage'][coverage_key] is True


@pytest.mark.parametrize('policy', ['cath-v3', 'cath-v4'])
def test_hidden_group_private_exclusion_fails_without_reallocating_or_writing(tmp_path, monkeypatch, policy) -> None:
    args = _fixture(tmp_path, monkeypatch, policy=policy)
    exclusions = tmp_path / 'exclusions.txt'
    exclusions.write_bytes(args.hidden_manifest.read_bytes())
    out = tmp_path / 'blocked'
    with pytest.raises(RuntimeError, match='complete group coverage cannot be preserved'):
        _build_hidden_manifest(argparse.Namespace(
            **{key: value for key, value in vars(args).items()
               if key not in {'private_lock_file', 'hidden_manifest', 'private_processability_exclusion_list'}},
            sampling_policy=policy, hidden_out_dir=out, private_lock_file=tmp_path / 'blocked.lock.json',
            private_processability_exclusion_list=exclusions,
            hidden_split_salt_env='NANOFOLD_HIDDEN_SPLIT_SALT', hidden_split_salt_file=None,
            hidden_val_size=None, seed=None,
        ))
    assert not out.exists()


@pytest.mark.parametrize('replacement,error', [
    ('1a00_A', 'locked original-unit quality representatives'),
    ('1a11_A', 'approved excluded full-group member'),
])
def test_qualified_selection_rejects_failed_or_excluded_targets_after_rehashing(tmp_path, monkeypatch, replacement, error) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')
    train = args.public_train_manifest.read_text().splitlines()
    assert '1a00_B' in train
    args.public_train_manifest.write_text(''.join(f'{replacement if chain == "1a00_B" else chain}\n' for chain in train))
    _repin_manifests(args)
    with pytest.raises(RuntimeError, match=error):
        verify(args)


@pytest.mark.parametrize('field', ['representative_eligibility_receipt', 'approved_group_exclusions'])
def test_qualification_inputs_are_required_and_cannot_be_report_outputs(tmp_path, monkeypatch, field) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')
    source = json.loads(args.lock_file.read_text())
    args.json_out = Path(source['args'][field])
    with pytest.raises(ValueError, match='alias'):
        verify(args)
    args.json_out = None
    del source['args'][field]
    _repin_source(args, source)
    with pytest.raises(ValueError, match=f'missing args.{field}'):
        verify(args)


@pytest.mark.parametrize('field', ['representative_eligibility_receipt', 'approved_group_exclusions'])
def test_qualification_inputs_support_identical_relocation_but_reject_changed_bytes(tmp_path, monkeypatch, field) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')
    source = json.loads(args.lock_file.read_text())
    relocated = tmp_path / f'relocated-{field}.json'
    Path(source['args'][field]).rename(relocated)
    setattr(args, field, relocated)
    assert verify(args)['status'] == 'ok'
    relocated.write_text(relocated.read_text() + '\n')
    with pytest.raises(ValueError, match='eligibility.*sha256'):
        verify(args)


def test_qualification_semantic_rejection_cannot_be_ignored(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')

    def reject(*args, **kwargs):
        raise ValueError('Not every excluded full-group member has failed')

    monkeypatch.setattr('nanofold.representative_eligibility.validate_representative_eligibility_receipt', reject)
    with pytest.raises(ValueError, match='every excluded full-group member'):
        verify(args)


def test_report_cannot_replace_nested_qualification_evidence(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')
    args.json_out = tmp_path / 'qualification-source.txt'
    before = args.json_out.read_bytes()
    with pytest.raises(ValueError, match='alias an audited eligibility source'):
        verify(args)
    assert args.json_out.read_bytes() == before


def test_private_qualification_provenance_is_required(tmp_path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')
    private = json.loads(args.private_lock_file.read_text())
    private.pop('representative_eligibility')
    _write_json(args.private_lock_file, private)
    with pytest.raises(ValueError, match='representative eligibility provenance'):
        verify(args)


@pytest.mark.parametrize('field,value', [
    ('original_unit_count', 11), ('original_representative_ids_sha256', '0' * 64),
    ('retained_group_count', 6), ('approved_excluded_group_count', 2),
])
def test_qualified_original_and_retained_populations_cannot_be_relabelled(tmp_path, monkeypatch, field, value) -> None:
    args = _fixture(tmp_path, monkeypatch, policy='cath-v4')
    source = json.loads(args.lock_file.read_text())
    source['split_groups'][field] = value
    _repin_source(args, source)
    with pytest.raises(ValueError, match='group.*population|group.*coverage|unit count'):
        verify(args)
