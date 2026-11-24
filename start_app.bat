@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "" http://localhost:8501
python -m streamlit run app.py --server.headless true --server.port 8501
