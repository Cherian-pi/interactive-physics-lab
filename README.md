# Interactive Physics Lab

A Python-based interactive physics app built with Streamlit to help students visualize core physics concepts.

## Current module
- Projectile Motion Explorer
- Comparative projectile visualization with 3 trajectories

## Features
- Interactive sliders for speed, angle, and gravity
- Side-by-side trajectory comparison
- Real-time updates
- Key physical quantities such as time of flight, maximum height, and horizontal range

## Run locally

interactive-physics-lab/
├── app.py
├── requirements.txt
├── README.md
├── modules/
├── utils/
├── docs/
└── assets/

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
