#!/bin/bash
streamlit run app/tablero.py --server.port=${PORT:-8501} --server.address=0.0.0.0
