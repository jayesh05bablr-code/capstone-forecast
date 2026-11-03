@echo off
REM One-click run on Windows. Keys go in the .env file in this folder.
REM The dashboard fetches data, trains and predicts by itself, so this only installs and opens it.
cd /d "%~dp0"
if not exist .env (
  copy .env.example .env >nul
  echo A .env file was created. Paste your keys after the = signs, save, close Notepad, then run this again.
  notepad .env
  exit /b 1
)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt || exit /b 1
python -m capstone.check_keys
streamlit run app.py
