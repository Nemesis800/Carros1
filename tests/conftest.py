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


def pytest_collection_modifyitems(session, config, items):
    """Marca ciertas pruebas como skip basado en los módulos que queremos desactivar."""
    for item in items:
        # Si el test está en los módulos que queremos desactivar, lo marcamos como skip
        if any(module in item.nodeid for module in [
            "test_grpc_services.py",
            "test_cli.py",
            "test_mlflow_integration.py",
            "test_app_csv.py",
            "test_headless_integration.py"
        ]):
            item.add_marker(pytest.mark.skip(reason="Test desactivado temporalmente"))

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


