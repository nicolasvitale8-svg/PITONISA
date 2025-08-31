@echo off
setlocal
REM Cambiar al directorio del script
cd /d "%~dp0"

echo ==========================================
echo   Iniciando PITONISA (Streamlit)
echo   Carpeta: %CD%
echo ==========================================

where python >nul 2>&1
if errorlevel 1 (
  echo No se encontro Python en el PATH.
  echo Instala Python 3.x y vuelve a intentar.
  pause
  exit /b 1
)

REM Lanzar la app (abrira el navegador)
python -m streamlit run app.py

if errorlevel 1 (
  echo.
  echo Si fallo por dependencias, ejecuta:
  echo     pip install -r requirements.txt
  echo y vuelve a intentar.
  echo.
  pause
)

endlocal
