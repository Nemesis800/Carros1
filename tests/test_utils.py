# tests/test_utils.py
"""
Tests para el módulo utils.py

Prueba la función winsound_beep en diferentes plataformas
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Añadir src al path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.mark.skipif(sys.platform != "win32", reason="Test específico de Windows")
def test_winsound_beep_on_windows():
    """Test winsound_beep cuando winsound está disponible (Windows)."""
    # Mock del módulo winsound
    mock_winsound = MagicMock()
    mock_winsound.Beep = MagicMock()
    
    with patch.dict('sys.modules', {'winsound': mock_winsound}):
        # Reimportamos para que tome el mock
        import importlib
        import utils
        importlib.reload(utils)
        from utils import winsound_beep
        
        # Llamamos a la función
        winsound_beep(1000, 200)
        
        # Verificamos que se llamó Beep con los parámetros correctos
        mock_winsound.Beep.assert_called_once_with(1000, 200)


def test_winsound_beep_on_non_windows():
    """Test winsound_beep en plataforma no-Windows (simulado)."""
    # Simulamos que estamos en una plataforma no-Windows
    with patch('sys.platform', 'linux'):
        # En plataformas no-Windows, la función no debe hacer nada
        # y no debe lanzar excepciones
        from utils import winsound_beep
        
        # No debe fallar, simplemente no hace nada
        try:
            winsound_beep(1500, 300)
            # Si llegamos aquí, pasó el test (no falló)
            assert True
        except Exception as e:
            pytest.fail(f"winsound_beep no debería fallar en plataformas no-Windows: {e}")


def test_winsound_beep_exception_handling():
    """Test que winsound_beep maneja excepciones correctamente."""
    # Mock del módulo winsound que lanza excepción
    mock_winsound = MagicMock()
    mock_winsound.Beep = MagicMock(side_effect=RuntimeError("Error de audio"))
    
    with patch.dict('sys.modules', {'winsound': mock_winsound}):
        import importlib
        import utils
        importlib.reload(utils)
        from utils import winsound_beep
        
        # No debe lanzar excepción, solo fallar silenciosamente
        try:
            winsound_beep(1000, 200)
            assert True  # Si llegamos aquí, manejó la excepción
        except Exception:
            pytest.fail("winsound_beep no manejó la excepción correctamente")


def test_winsound_beep_with_zero_params():
    """Test winsound_beep con parámetros cero."""
    mock_winsound = MagicMock()
    
    # Simulamos plataforma Windows para que utils ejecute la rama de winsound
    with patch('sys.platform', 'win32'):
        with patch.dict('sys.modules', {'winsound': mock_winsound}):
            import importlib
            import utils
            importlib.reload(utils)
            from utils import winsound_beep
            
            # Llamar con frecuencia y duración cero
            winsound_beep(0, 0)
            
            # Debe llamar la función aunque los valores sean cero
            mock_winsound.Beep.assert_called_once_with(0, 0)
