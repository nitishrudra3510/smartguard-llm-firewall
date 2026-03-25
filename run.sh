#!/usr/bin/env bash
# run.sh — One-stop script to set up and run the LLM Guardrails Firewall
set -e


echo "  LLM Guardrails Firewall — setup & run"


 1. Install dependencies 
echo ""
echo "→ Installing dependencies …"
pip install -r requirements.txt --quiet

#  2. Run unit tests 
echo ""
echo "→ Running unit tests …"
python3 -m unittest tests/test_classifier.py -v

#  3. Run evaluation on test suite 
echo ""
echo "→ Running evaluation on test_suite.json …"
python3 evaluation/evaluate.py

#  4. Run the interactive CLI 
echo ""
echo "→ Starting interactive firewall CLI (Ctrl+C to skip) …"
python3 -m app.main || true

#  5. Launch Streamlit dashboard 
echo ""
echo "→ Launching Streamlit dashboard …"
echo "   Open http://localhost:8501 in your browser."
streamlit run dashboard/dashboard.py
