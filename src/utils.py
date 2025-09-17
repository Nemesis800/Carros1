from __future__ import annotations
import sys


def winsound_beep(freq: int = 1000, dur_ms: int = 250) -> None:
    """
    Emite un beep en Windows usando winsound.Beep.
    - freq: frecuencia en Hz (por defecto 1000 Hz).
    - dur_ms: duración en milisegundos (por defecto 250 ms).
    En otros sistemas operativos, la función se convierte en un no-op.
    """
    if sys.platform.startswith("win"):
        try:
            import winsound  # librería estándar solo en Windows
            winsound.Beep(freq, dur_ms)
        except Exception:
            # Fallo silencioso: evita que el programa crashee si Beep no está disponible
            pass
