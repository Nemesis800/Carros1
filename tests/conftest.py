import os
import sys
from pathlib import Path
import pytest

# Asegura que `src/` esté en el path para importar app/counter/detector
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

@pytest.fixture
def tmp_reports_dir(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return d
