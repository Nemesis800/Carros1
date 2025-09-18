@echo off
echo 🚀 Lanzando MLflow UI...
echo 🌐 URL: http://localhost:5000
echo 📂 Directorio: %cd%\mlruns
echo 💡 Para detener: cerrar esta ventana
echo ===============================================

REM Crear directorio mlruns si no existe
if not exist "mlruns" mkdir mlruns

REM Lanzar MLflow UI en nueva ventana
start "MLflow UI" cmd /k "uv run -p .venv python -m mlflow ui --port 5000 --backend-store-uri file:///%cd%/mlruns --host 127.0.0.1"

REM Esperar un poco y abrir navegador
timeout /t 3 >nul
start http://localhost:5000

echo ✅ MLflow UI lanzado en ventana separada
echo 🌐 El navegador se abrirá automáticamente
echo 💡 Para detener MLflow, cierra la ventana "MLflow UI"
pause