# Fog Computing-Based Smart Traffic Management - simulation code

Simulator and analysis scripts for the paper "Fog Computing-Based Smart Traffic Management
System for Low-Latency Traffic Control".

## Run
    pip install -r requirements.txt
    python fogsim.py            # runs 3,150 one-hour simulations + 210 latency-sweep runs (a few minutes)
    python analyze_results.py   # reads results.pkl, writes snips.json (LaTeX table rows and plot coordinates)

Seeds 0-29 are fixed, so the numbers in the paper are reproduced exactly.
All latency, detection and demand parameters are modelling assumptions (see Table 3 of the paper).
