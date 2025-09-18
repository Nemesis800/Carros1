#!/usr/bin/env python3
"""
Script para lanzar la interfaz web de MLflow.

Permite visualizar experimentos, métricas, parámetros y artefactos
guardados en el directorio local `mlruns/`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def launch_mlflow_ui(port: int = 5000, auto_open_browser: bool = True) -> None:
    """Lanza la interfaz web de MLflow.

    Args:
        port: Puerto donde levantar la interfaz (default = 5000).
        auto_open_browser: Si abrir automáticamente el navegador.
    """
    # Verificar que MLflow esté instalado
    try:
        import mlflow  # noqa: F401

        print(f"✅ MLflow {mlflow.__version__} encontrado")
    except ImportError:
        print("❌ MLflow no está instalado. Instálalo con: pip install mlflow")
        return

    # Directorio local de experimentos
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        mlruns_dir.mkdir()
        print(f"📁 Directorio mlruns creado: {mlruns_dir.absolute()}")

    tracking_uri = f"file:///{mlruns_dir.absolute()}"

    print("🚀 Lanzando MLflow UI...")
    print(f"📂 Directorio de experimentos: {mlruns_dir.absolute()}")
    print(f"🌐 URL: http://localhost:{port}")
    print("⏹️  Para detener, presiona Ctrl+C")

    # Abrir navegador automáticamente en hilo aparte
    if auto_open_browser:
        def open_browser():
            time.sleep(2)  # esperar arranque
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    try:
        # --- Windows ---
        if os.name == "nt":
            cmd = [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--port",
                str(port),
                "--backend-store-uri",
                tracking_uri,
            ]

            print(f"📝 Ejecutando: {' '.join(cmd)}")
            print("💡 Para detener MLflow UI: presiona Ctrl+C")
            print("🔄 Si la consola se bloquea, usa Ctrl+C y luego escribe 'exit'")
            print("-" * 50)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            try:
                for line in process.stdout:
                    print(line.strip())
                    if "Running on" in line:
                        print("✅ MLflow UI iniciado correctamente")
            except KeyboardInterrupt:
                print("\n🛑 Deteniendo MLflow UI...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print("✅ MLflow UI detenido")

        # --- Unix/Linux ---
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlflow",
                    "ui",
                    "--port",
                    str(port),
                    "--backend-store-uri",
                    tracking_uri,
                ],
                check=True,
            )

    except KeyboardInterrupt:
        print("\n🛑 MLflow UI detenido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando MLflow UI: {e}")
    except FileNotFoundError:
        print("❌ No se pudo ejecutar MLflow. Verifica la instalación.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzar interfaz web de MLflow")
    parser.add_argument("--port", type=int, default=5000, help="Puerto para la interfaz web")
    parser.add_argument(
        "--no-browser", action="store_true", help="No abrir navegador automáticamente"
    )

    args = parser.parse_args()
    launch_mlflow_ui(port=args.port, auto_open_browser=not args.no_browser)
