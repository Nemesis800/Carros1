#!/usr/bin/env python3
"""
Script simple para lanzar MLflow UI sin bloqueo de consola.
Especialmente diseñado para PowerShell en Windows.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def launch_mlflow_ui_simple(port: int = 5000, open_browser: bool = True):
    """
    Lanza MLflow UI de forma simple sin bloquear la consola.
    
    Args:
        port: Puerto para la interfaz web
        open_browser: Si abrir automáticamente el navegador
    """
    # Verificar MLflow
    try:
        import mlflow
        print(f"✅ MLflow {mlflow.__version__} encontrado")
    except ImportError:
        print("❌ MLflow no instalado. Usa: pip install mlflow")
        return

    # Configurar directorio
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        mlruns_dir.mkdir()

    tracking_uri = f"file:///{mlruns_dir.absolute()}"
    
    print(f"🚀 Lanzando MLflow UI en puerto {port}")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"📂 Directorio: {mlruns_dir.absolute()}")
    print("💡 Para detener: Ctrl+C en esta consola")
    print("-" * 50)

    # Construir comando
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--port", str(port),
        "--backend-store-uri", tracking_uri,
        "--host", "127.0.0.1"
    ]

    # Abrir navegador después de unos segundos
    if open_browser:
        def open_browser_delayed():
            time.sleep(3)
            webbrowser.open(f"http://localhost:{port}")
        
        import threading
        threading.Thread(target=open_browser_delayed, daemon=True).start()

    # Ejecutar en Windows de forma que no bloquee
    if os.name == 'nt':
        try:
            # Usar CREATE_NEW_CONSOLE para evitar bloqueo
            import subprocess
            subprocess.Popen(
                cmd, 
                creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
            )
            print("✅ MLflow UI lanzado en nueva ventana de consola")
            print(f"🌐 Accede a: http://localhost:{port}")
            print("💡 La interfaz se abrió en una ventana separada")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("📝 Intenta ejecutar manualmente:")
            print(f"   mlflow ui --port {port}")
    else:
        # Para Linux/Mac
        try:
            subprocess.Popen(cmd)
            print("✅ MLflow UI lanzado en background")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lanzar MLflow UI simple")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    
    args = parser.parse_args()
    
    launch_mlflow_ui_simple(port=args.port, open_browser=not args.no_browser)