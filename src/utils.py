# src/utils.py
"""
Utilidades auxiliares para la aplicación.

Incluye:
- `winsound_beep`: función multiplataforma para emitir un beep (solo en Windows).
"""

from __future__ import annotations

import sys


def winsound_beep(freq: int = 1000, dur_ms: int = 250) -> None:
    """Emite un beep en Windows usando `winsound.Beep`.

    Args:
        freq: Frecuencia en Hz (default = 1000 Hz).
        dur_ms: Duración en milisegundos (default = 250 ms).

    Notas:
        - Solo funciona en sistemas Windows (usando la librería estándar `winsound`).
        - En otros sistemas operativos, esta función no hace nada.
        - Si ocurre un error al invocar `Beep`, se ignora silenciosamente.
    """
    if sys.platform.startswith("win"):
        try:
            import winsound  # Disponible solo en Windows
            winsound.Beep(freq, dur_ms)
        except Exception:
            # Evita crash si winsound falla
            pass
