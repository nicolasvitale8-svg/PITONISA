@echo off
echo Iniciando PITONISA...
cd /d "%~dp0"
python -m streamlit run app_strict.py
pause