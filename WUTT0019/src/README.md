# GPR Index — Probabilistic Estimation Engine (COMP3018 Artefact 1)

A comparative **Frequentist (MLE) vs Bayesian (MCMC)** parameter-estimation analysis
of the Global Political Risk (GPR) Index, focused on quantifying uncertainty.

Two equivalent deliverables: `probabilistic_estimation.py` (single python file) and
`GPR_Index_Analysis.ipynb` (more detailed notebook).

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python probabilistic_estimation.py
```

This downloads the latest GPR data as a CSV to the current directory.

## Dataset (not submitted — download separately)

- Geopolitical Risk (GPR) Index, monthly. Source: D. Caldara and M. Iacoviello,
  https://www.matteoiacoviello.com/gpr.htm


## Known limitations

- Pinned to PyMC 5.28 / ArviZ 0.23; newer ArviZ 1.x / PyMC 6.x will not run unchanged.
- Bayesian sampling time depends on hardware.
