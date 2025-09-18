#!/usr/bin/env python3
"""
Script para lanzar la interfaz web de MLflow.
Permite visualizar experimentos, métricas, parámetros y artefactos.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def launch_mlflow_ui(port: int = 5000, auto_open_browser: bool = True):
    """
    Lanza la interfaz web de MLflow.
    
    Args:
        port: Puerto para la interfaz web
        auto_open_browser: Si abrir automáticamente el navegador
    """
    # Verificar que MLflow esté instalado
    try:
        import mlflow
        print(f"✅ MLflow {mlflow.__version__} encontrado")
    except ImportError:
        print("❌ MLflow no está instalado. Instálalo con: pip install mlflow")
        return
    
    # Configurar directorio de tracking
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        mlruns_dir.mkdir()
        print(f"📁 Directorio mlruns creado: {mlruns_dir.absolute()}")
    
    tracking_uri = f"file:///{mlruns_dir.absolute()}"
    
    print(f"🚀 Lanzando MLflow UI...")
    print(f"📂 Directorio de experimentos: {mlruns_dir.absolute()}")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"⏹️  Para detener, presiona Ctrl+C")
    
    # Abrir navegador automáticamente después de un momento
    if auto_open_browser:
        def open_browser():
            time.sleep(2)  # Esperar a que MLflow inicie
            webbrowser.open(f"http://localhost:{port}")
        
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    try:
        # Ejecutar MLflow UI
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--port", str(port),
            "--backend-store-uri", tracking_uri
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI detenido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando MLflow UI: {e}")
    except FileNotFoundError:
        print("❌ No se pudo ejecutar MLflow. Verifica la instalación.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lanzar interfaz web de MLflow")
    parser.add_argument("--port", type=int, default=5000, help="Puerto para la interfaz web")
    parser.add_argument("--no-browser", action="store_true", help="No abrir navegador automáticamente")
    
    args = parser.parse_args()
    
    launch_mlflow_ui(port=args.port, auto_open_browser=not args.no_browser)