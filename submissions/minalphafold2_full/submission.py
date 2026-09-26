import runpy
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_adapter = runpy.run_path(str(_root / "nanofold/official_adapter_loader.py"))["load_adapter"](_root)
globals().update({name: value for name, value in vars(_adapter).items() if not name.startswith("_")})
