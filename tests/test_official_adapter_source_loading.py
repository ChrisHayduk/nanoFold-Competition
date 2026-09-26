"""Official adapter dependencies must execute their pinned source closure."""

from __future__ import annotations

import os
import py_compile
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _copy_runtime(root: Path) -> None:
    for relative in ("nanofold/official_adapter_loader.py", "submissions/minalphafold2/submission.py",
                     "submissions/minalphafold2_full/submission.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    for directory in ("minalphafold", "configs"):
        shutil.copytree(ROOT / "third_party/minAlphaFold2" / directory,
                        root / "third_party/minAlphaFold2" / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def test_full_adapter_ignores_stale_dependencies_and_foreign_simplex_namespace(tmp_path):
    root = tmp_path / "repo"
    _copy_runtime(root)
    for source in root.rglob("*.py"):
        current = source.read_bytes() + b'\nSOURCE_MARKER = "source"\n'
        cached = current.replace(b'SOURCE_MARKER = "source"', b'SOURCE_MARKER = "cached"')
        assert len(current) == len(cached)
        source.write_bytes(cached)
        timestamp = source.stat()
        py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        source.write_bytes(current)
        os.utime(source, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    script = r'''
import importlib.util
import random
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, sys.argv[3])
from nanofold.submission_runtime import load_submission_hooks

torch.set_num_threads(1)
root, simplex = map(Path, sys.argv[1:3])
sys.path.insert(0, str(simplex))
import minalphafold.model as foreign_model
import minalphafold.model_config as foreign_config
assert 'use_simplicial_evoformer' in foreign_config.ModelConfig.__dataclass_fields__
foreign_class = foreign_model.AlphaFold2

path = root / 'submissions/minalphafold2/submission.py'
spec = importlib.util.spec_from_file_location('ordinary_cached_adapter', path)
cached = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cached
spec.loader.exec_module(cached)
assert cached.SOURCE_MARKER == 'cached'  # positive stale-cache control

cfg = {'model': {'profile_path': str(root/'third_party/minAlphaFold2/configs/tiny.toml'),
                 'msa_depth': 3, 'extra_msa_depth': 1, 'n_cycles': 2},
       'train': {'max_steps': 8}, 'optim': {'lr': .001},
       'experiment': {'recycle': {'train_mode': 'fixed'}}}
entries = []
for family in ('minalphafold2', 'minalphafold2_full'):
    hooks = load_submission_hooks({'submission': {'path': str(root/'submissions'/family/'submission.py')}}, root/'config.yaml')
    assert hooks.module.SOURCE_MARKER == 'source'
    entries.append(hooks)
assert sys.modules['minalphafold.model'].AlphaFold2 is foreign_class
private_modules = [m for name,m in sys.modules.items() if name.startswith('_nanofold_official_') and hasattr(m, 'SOURCE_MARKER')]
assert len(private_modules) >= 6
assert all(module.SOURCE_MARKER == 'source' for module in private_modules)

# Same config, initialization and data through tiny and full official entry
# points must preserve predictions, objective and every parameter gradient.
n = 5
coords = torch.zeros(1,n,14,3)
for i in range(n):
    coords[0,i,:4] = torch.tensor([[0.,1,0],[1.,0,0],[2.,.2,0],[2.5,1,0]]) + torch.tensor([3.8*i,0,0])
mask = torch.zeros(1,n,14,dtype=torch.bool); mask[:,:,:4] = True
batch = {'aatype': torch.full((1,n),7,dtype=torch.long), 'msa': torch.full((1,4,n),7,dtype=torch.long),
         'deletions': torch.zeros(1,4,n,dtype=torch.long), 'residue_mask': torch.ones(1,n,dtype=torch.bool),
         'residue_index': torch.arange(n)[None], 'between_segment_residues':torch.zeros(1,n,dtype=torch.long),
         'atom14_positions': coords, 'atom14_mask': mask, 'resolution':torch.tensor([2.])}
results = []
for hooks in entries:
    torch.manual_seed(41)
    model = hooks.build_model(cfg)
    assert not isinstance(model, foreign_class)
    random.seed(53); np.random.seed(53); torch.manual_seed(53)
    out = hooks.run_batch(model,batch,cfg,training=True)
    out['loss'].backward()
    grads = {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    results.append((out['pred_atom14'],out['loss'],grads))
assert torch.equal(results[0][0],results[1][0]) and torch.equal(results[0][1],results[1][1])
assert results[0][2].keys() == results[1][2].keys()
assert all(torch.equal(value,results[1][2][key]) for key,value in results[0][2].items())

files = entries[-1].module.runtime_source_files(cfg)
assert root/'nanofold/official_adapter_loader.py' in files
model_source = root/'third_party/minAlphaFold2/minalphafold/model.py'
original_source = model_source.read_bytes()
model_source.write_text('# changed\n')
try:
    entries[-1].build_model(cfg)
except ValueError as exc:
    assert 'source changed' in str(exc)
else:
    raise AssertionError('Changed backend source accepted')
model_source.write_bytes(original_source)
sys.modules[model.__class__.__module__] = object()
try:
    entries[-1].module.runtime_source_files(cfg)
except ValueError as exc:
    assert 'cache identity' in str(exc)
else:
    raise AssertionError('Replaced private backend identity accepted')
'''
    result = subprocess.run([sys.executable, "-c", script, str(root), str(ROOT.parent / "SimplexFold"), str(ROOT)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("defect", ["orphan", "changed_lazy", "added_lazy"])
def test_official_private_namespace_has_no_unpinned_import_fallback(tmp_path, defect):
    _copy_runtime(tmp_path)
    package = tmp_path / "third_party/minAlphaFold2/minalphafold"
    model = package / "model.py"
    model.write_text(model.read_text() + "\ndef read_late():\n    from . import late\n    return late.VALUE\n")
    if defect == "changed_lazy":
        (package / "late.py").write_text("VALUE = 1\n")
    loader = runpy.run_path(str(tmp_path / "nanofold/official_adapter_loader.py"))
    if defect == "orphan":
        py_compile.compile(str(model), cfile=str(package / "model.pyc"), doraise=True)
        model.unlink()
        with pytest.raises(ImportError, match="no verified source"):
            loader["load_backend"](tmp_path)
    else:
        backend = loader["load_backend"](tmp_path)
        (package / "late.py").write_text("VALUE = 2\n")
        with pytest.raises((ImportError, ValueError), match="no verified source|changed before execution"):
            backend.model.read_late()
