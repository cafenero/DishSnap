#!/bin/zsh
source .venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
