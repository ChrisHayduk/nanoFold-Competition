"""Source-bound dependency closure for the official minAlphaFold2 adapter.

The entrypoints execute this bootstrap with runpy so neither it nor its private
adapter/backend imports consult timestamp-valid or orphan bytecode caches.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def source_files(root: Path) -> list[Path]:
    backend = root / "third_party/minAlphaFold2/minalphafold"
    return sorted({
        root / "nanofold/official_adapter_loader.py",
        root / "submissions/minalphafold2/submission.py",
        root / "submissions/minalphafold2_full/submission.py",
        *backend.rglob("*.py"), *backend.rglob("*.txt"),
    })


def _sources(root: Path) -> dict[Path, str]:
    paths = source_files(root)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Official adapter sources must be existing regular non-symlink files.")
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _verify(state: SimpleNamespace) -> None:
    if _sources(state.root) != state.sources:
        raise ValueError("Official adapter source changed after import; restart the run.")
    if state.finder not in sys.meta_path:
        raise ValueError("Official adapter source loader was removed.")
    if any(sys.modules.get(name) is not module for name, module in state.modules.items()):
        raise ValueError("Official adapter module cache identity changed.")


class _SourceFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, alias: str, state: SimpleNamespace):
        self.alias, self.state = alias, state

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not fullname.startswith(self.alias + "."):
            return None
        relative = fullname[len(self.alias) + 1:]
        if relative == "adapter":
            source = self.state.root / "submissions/minalphafold2/submission.py"
        elif relative == "minalphafold" or relative.startswith("minalphafold."):
            location = (self.state.root / "third_party/minAlphaFold2").joinpath(*relative.split("."))
            source = location / "__init__.py" if location.is_dir() else location.with_suffix(".py")
        else:
            raise ImportError(f"Unknown official adapter dependency: {fullname}")
        if source not in self.state.sources or not source.is_file():
            raise ImportError(f"Official adapter dependency has no verified source: {fullname}")
        return importlib.util.spec_from_file_location(
            fullname, source, loader=self,
            submodule_search_locations=[str(source.parent)] if source.name == "__init__.py" else None,
        )

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        source = Path(module.__file__)
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != self.state.sources[source]:
            raise ValueError(f"Official adapter source changed before execution: {source}")
        exec(compile(contents, str(source), "exec", dont_inherit=True), module.__dict__)  # noqa: S102
        self.state.modules[module.__name__] = module


def _state(root: Path) -> SimpleNamespace:
    root = root.resolve(strict=True)
    alias = "_nanofold_official_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    package = sys.modules.get(alias)
    if package is not None:
        state = getattr(package, "_official_source_state", None)
        if state is None:
            raise ValueError("Unverified partial official adapter cache.")
        _verify(state)
        return state
    if any(name.startswith(alias + ".") for name in sys.modules):
        raise ValueError("Unverified partial official adapter cache.")
    state = SimpleNamespace(root=root, alias=alias, sources=_sources(root), modules={})
    state.finder = _SourceFinder(alias, state)
    package = ModuleType(alias)
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec(alias, loader=None, is_package=True)
    package.__dict__["_official_source_state"] = state
    state.modules[alias] = package
    sys.modules[alias] = package
    sys.meta_path.insert(0, state.finder)
    _verify(state)
    return state


def load_backend(root: Path) -> SimpleNamespace:
    state = _state(root)
    modules = {part: importlib.import_module(f"{state.alias}.minalphafold.{part}")
               for part in ("data", "losses", "model", "model_config", "trainer")}
    _verify(state)
    fields = getattr(modules["model_config"].ModelConfig, "__dataclass_fields__", {})
    if "use_simplicial_evoformer" in fields or not fields:
        raise ValueError("The official adapter requires its pinned minAlphaFold2 backend.")
    return SimpleNamespace(**modules, verify_source_binding=lambda: _verify(state), source_files=tuple(state.sources))


def load_adapter(root: Path) -> Any:
    state = _state(root)
    adapter = importlib.import_module(f"{state.alias}.adapter")
    _verify(state)
    return adapter
