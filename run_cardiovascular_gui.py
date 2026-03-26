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
    print("GUI를 시작할 수 없습니다.")
    print("프로젝트 가상환경에 PySide6 또는 PyQt5가 설치되어 있는지 확인하세요.")
    print(f"가져오기 오류: {exc}")
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
