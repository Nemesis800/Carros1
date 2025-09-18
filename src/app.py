# src/app.py
"""
Punto de entrada principal de la aplicación.

Este módulo permite ejecutar la aplicación en dos modos:
1. CLI (modo sin interfaz gráfica, útil para Docker y tests).
2. UI (modo con interfaz Tkinter).
"""

from __future__ import annotations

import sys

# Re-exports para compatibilidad con tests:
#   from app import VideoProcessor, AppConfig
from config import AppConfig  # seguro en Docker (sin UI)
from processor import VideoProcessor  # seguro en Docker (sin UI)


def main() -> None:
    """Selecciona el modo de ejecución según los argumentos de entrada."""
    # --- Modo CLI ---
    # Se activa si se detectan banderas típicas de línea de comandos
    if any(arg in sys.argv for arg in ("--cli", "--source", "--webcam")):
        from cli import parse_cli_args, main_cli

        ns = parse_cli_args(sys.argv[1:])
        raise SystemExit(main_cli(ns))

    # --- Modo UI ---
    # Import diferido de Tkinter UI para no romper entornos headless (ej. Docker)
    from ui_app import App

    app = App()
    if hasattr(app, "run"):
        app.run()
    else:
        # Fallback para compatibilidad (Tkinter usa mainloop)
        app.mainloop()


if __name__ == "__main__":
    main()


