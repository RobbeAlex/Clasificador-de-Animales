@echo off
echo Iniciando Clasificador de Animales en entorno virtual...

:: 1. Ir a la carpeta del proyecto
cd /d "C:\Users\rober\Downloads\Identificacion"

:: 2. Activar el entorno virtual
call .venv\Scripts\activate

:: 3. Ejecutar la aplicacion
streamlit run app.py

pause