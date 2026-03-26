from __future__ import annotations

import os
import sys
from pathlib import Path


DOMAIN_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOMAIN_DIR.parent
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_in_venv() -> None:
    if not VENV_PYTHON.exists():
        return
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def main() -> int:
    maybe_reexec_in_venv()
    try:
        from gui_app import main as gui_main
    except ImportError as exc:
        print("헬스케어 UI를 시작할 수 없습니다.")
        print("가상환경에 PySide6 또는 PyQt5, opencv-python이 설치되어 있는지 확인해주세요.")
        print(f"가져오기 오류: {exc}")
        return 1
    return int(gui_main())


if __name__ == "__main__":
    raise SystemExit(main())
