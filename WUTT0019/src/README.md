# GPR Index — Probabilistic Estimation Engine (COMP3018 Artefact 1, Source)

Individual estimation module for the Probabilistic ML capstone. Performs exploratory
data analysis and a comparative **Frequentist vs Bayesian** parameter-estimation
analysis on the Global Political Risk (GPR) Index, with a focus on quantifying
uncertainty.

## Contents
| File | Purpose |
|------|---------|
| `GPR_Index_Analysis.ipynb` | Narrated analysis with markdown explaining every figure and statistic. |
| `probabilistic_estimation.py` | Standalone CLI script mirroring the notebook end-to-end. |
| `requirements.txt` | Pinned dependencies. |
| `figures/` | Output figures written by the script (created on run). |

## Environment setup
Developed and verified on **Python 3.11**. The pinned Bayesian stack
(PyMC 5.28 / ArviZ 0.23) is deliberate — ArviZ 1.x / PyMC 6.x are a breaking
rewrite and the code will not run against them unchanged (see *Known limitations*).

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run the notebook, register this environment as a Jupyter kernel once:
```bash
python -m ipykernel install --user --name gpr-venv --display-name "Python 3 (GPR)"
```

**PyCharm users:** set *Settings → Project → Python Interpreter* to this `.venv`,
then in the open notebook use the **kernel dropdown in the notebook toolbar** to
select the matching kernel before *Run All*. The notebook's kernel is chosen
separately from the project interpreter.

## Dataset (NOT included in submission — download separately)
- **Name:** Geopolitical Risk (GPR) Index, monthly.
- **Authors / source:** D. Caldara and M. Iacoviello.
- **Link:** https://www.matteoiacoviello.com/gpr.htm
- **IEEE-style citation (for the report):**
  D. Caldara and M. Iacoviello, "Measuring Geopolitical Risk," *American Economic
  Review*, vol. 112, no. 4, pp. 1194–1225, 2022. [Online]. Available:
  https://www.matteoiacoviello.com/gpr.htm
- Place the downloaded CSV at `../data/gpr_index.csv` (one level up from `src/`).
  TODO(student): verify the exact column names in your download match what
  `load_data()` expects (`month`, `GPR`); adjust the loader if the file uses different
  headers.

## Running the script
```bash
cd src
python probabilistic_estimation.py --input ../data/gpr_index.csv
```
This prints the data-quality report, distribution-selection table, and both sets of
estimates to stdout, and writes all figures to `src/figures/`.

Optional flags:
```bash
python probabilistic_estimation.py --input ../data/gpr_index.csv \
    --start 2018-01-01 --end 2025-01-01 --draws 2000 --tune 1000 --seed 42
```

## Running the notebook
```bash
cd src
jupyter notebook GPR_Index_Analysis.ipynb
```
Run cells top to bottom. The notebook is self-contained and reproduces the same results
as the script (same seed, window, model, and priors).

## Method summary
1. **EDA** characterises the series and its uncertainty (noise, missingness, spikes).
2. **Distribution selection:** Gamma, LogNormal, and Normal are fitted and ranked by
   AIC/BIC plus a Kolmogorov–Smirnov goodness-of-fit test; the best family is carried
   forward. (See the notebook for the justified choice.)
3. **Frequentist:** Maximum Likelihood Estimation → point estimates + analytic
   (Fisher-information) and bootstrap confidence intervals.
4. **Bayesian:** the same model in PyMC with weakly-informative priors → posterior, MAP,
   and 94% HDI credible intervals, with convergence diagnostics and a posterior
   predictive check.
5. **Comparison:** point estimate ± CI vs posterior ± HDI, side by side.

## Known limitations / bugs
- TODO(student): fill in after running — e.g. any convergence warnings, sensitivity to
  the analysis window, or candidate-distribution edge cases.
- Bayesian sampling time depends on hardware; `--draws` is kept modest for constrained
  machines. Increase for smoother posteriors.
- The analysis treats observations as a stationary sample for *estimation*. Temporal
  dependence is documented in the EDA (ACF/PACF) but **sequential modelling is deferred
  to Artefact 2** by design.
- **Dependency sensitivity:** the code targets PyMC 5.28 / ArviZ 0.23. Newer ArviZ 1.x
  renames arguments (`hdi_prob`→`ci_prob`) and moves `plot_posterior`/`plot_ppc` to a
  separate package, so it will raise errors — stick to the pinned versions.
- PyTensor compiles C code for NUTS on first run. If you see
  `ImportError: cannot import name 'CompileError' from 'setuptools.errors'`, the venv's
  setuptools is too old; fix with `pip install "setuptools>=70,<80"`.

## AI usage reminder (topic policy is strict)
AI tooling was used for scaffolding/structuring only. You **must** keep a full log of
your prompts and the unedited AI output and include it in your report's
"Appendix: AI Usage Declaration", identify the model(s) and dates, and add the AI use to
your reference list. Verify, edit, and interpret all results yourself before submission.

## Repository note
TODO(student): if you use a GitHub mirror, add the link here. Remember the marked
artefact is the submitted `.zip`, not the repo.
