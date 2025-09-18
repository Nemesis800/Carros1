# tests/conftest.py
"""
Configuración global de pytest para el proyecto.

Incluye:
- Ajuste de `sys.path` para garantizar que `src/` esté disponible.
- Fixtures reutilizables para los tests (ej. directorio temporal de reportes).
"""

import sys
from pathlib import Path

import pytest

# ----------------------------------------------------------------------
# Configuración de rutas
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def tmp_reports_dir(tmp_path):
    """Crea un directorio temporal `reports/` para pruebas de salida CSV."""
    d = tmp_path / "reports"
    d.mkdir()
    return d


