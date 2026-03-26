from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_in_venv() -> None:
    if not VENV_PYTHON.exists():
        return
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


maybe_reexec_in_venv()

sys.path.insert(0, str(ROOT_DIR / "cardiovascular_autonomic_domain"))

try:
    from gui_app import main
except ImportError as exc:
    print("Unable to start the all-in-one HealthCare UI.")
    print("Install PySide6 and opencv-python in the project virtual environment first.")
    print(f"Import error: {exc}")
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
