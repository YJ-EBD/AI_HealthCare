from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DOMAIN_DIR = ROOT_DIR / "cardiovascular_autonomic_domain"
sys.path.insert(0, str(DOMAIN_DIR))

from healthcare_app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
