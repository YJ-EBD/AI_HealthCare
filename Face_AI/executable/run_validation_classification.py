import subprocess
import sys
from pathlib import Path


def main():
    face_ai_root = Path(__file__).resolve().parents[1]
    runtime = face_ai_root / "model" / "runtime" / "evaluate_classification.py"
    command = [sys.executable, str(runtime), *sys.argv[1:]]
    raise SystemExit(subprocess.run(command, cwd=face_ai_root).returncode)


if __name__ == "__main__":
    main()
