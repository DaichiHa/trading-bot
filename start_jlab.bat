@echo off
call .\.venv\Scripts\activate
.\.venv\Scripts\jupyter-lab --no-browser --port 8888
