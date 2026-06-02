# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: probabilistic_estimation.py
# Date: 21-05-2026
# Description: Frequentist vs Bayesian parameter estimation on the GPR Index and ETL + EDA.


import os
import warnings
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


# Program constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "gpr_processed.csv")
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
RANDOM_SEED = 42
HDI_PROB = 0.94
CI_PROB = 0.95
PALETTE = {
    "primary": "#2e9bd6",
    "accent": "#ff8c00",
    "alert": "#e63946",
    "secondary": "#15b8a6",
}


def _section(title):
    """
    Print a labeled section banner to stdout.
    """
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def fetch_gpr_data(url=GPR_URL):
    """
    Download the latest GPR data from the Caldara & Iacoviello website.
    Returns a pandas dataframe.
    """
    print(f"Downloading latest GPR data from:\n  {url}")
    raw = pd.read_excel(url, usecols=["month", "GPR"])
    print(f"  -> {len(raw)} rows downloaded.")
    return raw


def load_data(start=START_DATE, end=END_DATE):
    """
    Download the GPR Index and filter to the analysis window.
    Returns a pandas dataframe filtered by the start and end date.
    """
    df = fetch_gpr_data().copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[(df["month"] >= start) & (df["month"] < end)]
    return df.dropna(subset=["GPR"]).sort_values("month").reset_index(drop=True)


def data_quality_report(df):
    """
    Print coverage, missingness and descriptive statistics.
    Returns a np array of the data quality metrics.
    """
    _section("DATA QUALITY & UNCERTAINTY REPORT")
    gpr = df["GPR"]
    print(f"Shape:            {df.shape}")
    print(f"Date coverage:    {df['month'].min().date()} -> "
          f"{df['month'].max().date()} ({len(df)} months)")
    print(f"Missing values:   month={df['month'].isna().sum()}, GPR={gpr.isna().sum()}")

    q1, q3 = gpr.quantile(0.25), gpr.quantile(0.75)
    iqr = q3 - q1
    print("\nDescriptive statistics")
    print(f"  Mean:           {gpr.mean():.2f}")
    print(f"  Median:         {gpr.median():.2f}")
    print(f"  Std Dev:        {gpr.std():.2f}")
    print(f"  Skewness:       {gpr.skew():.2f}  (>0 => right tail)")
    print(f"  Excess Kurtosis:{gpr.kurtosis():.2f}  (>0 => heavy tails)")
    print(f"  Min / Max:      {gpr.min():.2f} / {gpr.max():.2f}")
    print(f"  Q1 / Q3:        {q1:.2f} / {q3:.2f}")
    print(f"  IQR:            {iqr:.2f}")

    return gpr.to_numpy(dtype=float)


def run_eda(df):
    """
    Show the 4 EDA plots (level+band, histograms, ACF/PACF, Q-Q).
    """
    _section("EXPLORATORY DATA ANALYSIS")
    month, gpr = df["month"], df["GPR"]

    # Rolling mean and volatility band
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(month, gpr, alpha=0.4, label="Monthly GPR", color=PALETTE["primary"])
    roll = gpr.rolling(12)
    ax.plot(month, roll.mean(), label="12m Rolling Mean", color=PALETTE["accent"], lw=2)
    ax.fill_between(month, roll.mean() - roll.std(), roll.mean() + roll.std(),
                    alpha=0.15, color=PALETTE["accent"], label="+/-1 sigma Band")
    ax.set_title("GPR Index with Rolling Mean & Volatility Band")
    ax.set_ylabel("GPR")
    ax.legend()
    plt.tight_layout()

    # Lognormal vs Normal Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(gpr, kde=True, bins=20, ax=axes[0], color=PALETTE["primary"])
    axes[0].axvline(gpr.mean(), ls="--", color="red", label=f"Mean ({gpr.mean():.0f})")
    axes[0].axvline(gpr.median(), ls="--", color="orange",
                    label=f"Median ({gpr.median():.0f})")
    axes[0].set_title("GPR Distribution (raw)")
    axes[0].legend()
    sns.histplot(np.log1p(gpr), kde=True, bins=20, ax=axes[1], color=PALETTE["secondary"])
    axes[1].set_title("Log-Transformed GPR Distribution")
    axes[1].set_xlabel("log(1 + GPR)")
    plt.tight_layout()

    # ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sm.graphics.tsa.plot_acf(gpr.dropna(), lags=24, ax=axes[0], color=PALETTE["primary"])
    axes[0].set_title("Autocorrelation (ACF)")
    sm.graphics.tsa.plot_pacf(gpr.dropna(), lags=24, ax=axes[1], color=PALETTE["secondary"])
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()

    #Q-Q Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(gpr.dropna(), dist="norm", plot=ax)
    ax.set_title("Q-Q Plot -- GPR vs Normal Distribution")
    ax.get_lines()[0].set_color(PALETTE["primary"])
    ax.get_lines()[1].set_color(PALETTE["alert"])
    plt.tight_layout()


def fit_model(data):
    """
    Fit the LogNormal model to the GPR level by MLE.
    Returns a tuple: (s, loc, scale).
    """
    _section("Distribution Choice -- LOGNORMAL")
    params = stats.lognorm.fit(data, floc=0)
    print(f"SciPy MLE params (s, loc, scale) = {params}")
    return params


def _lognorm_natural(params):
    """
    Convert SciPy lognormal params (s, loc, scale) to (mu, sigma)``.
    Returns an np array of (mu, sigma).
    """
    s, _loc, scale = params
    return np.array([np.log(scale), s], dtype=float)


def _nll(theta, data):
    """
    Negative log-likelihood of the LogNormal at natural params (mu, sigma).
    Returns the summed negative log-likelihood as a float.
    """
    mu, sigma = theta
    if sigma <= 0:
        return np.inf
    return -float(np.sum(stats.lognorm.logpdf(data, sigma, loc=0, scale=np.exp(mu))))


def _numerical_hessian(func, theta, eps=1e-5):
    """
    Approximate the Hessian (matrix of second derivatives) of a scalar function
    at theta using central finite differences.
    Returns the (k, k) Hessian matrix as a np array.
    """
    theta = np.asarray(theta, dtype=float)
    k = len(theta)
    h = eps * np.maximum(np.abs(theta), 1.0)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            tpp, tpm, tmp, tmm = (theta.copy() for _ in range(4))
            tpp[i] += h[i]; tpp[j] += h[j]
            tpm[i] += h[i]; tpm[j] -= h[j]
            tmp[i] -= h[i]; tmp[j] += h[j]
            tmm[i] -= h[i]; tmm[j] -= h[j]
            hess[i, j] = (func(tpp) - func(tpm) - func(tmp) + func(tmm)) \
                / (4 * h[i] * h[j])
    return hess


def fit_frequentist(data, params, seed=RANDOM_SEED, n_boot=2000):
    """
    Run the full frequentist pipeline: fit the LogNormal by MLE, then quantify
    parameter uncertainty two ways -- analytic (Fisher-information) CIs from the
    Hessian, and bootstrap CIs from resampling the data. Also prints a summary
    table and plots the fitted density over the data histogram.

    Returns a dict mapping each param name to its results.
    """
    _section("FREQUENTIST ESTIMATION (MLE) -- LogNormal")
    theta = _lognorm_natural(params)
    names = ["mu", "sigma"]

    hess = _numerical_hessian(lambda t: _nll(t, data), theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = np.linalg.inv(hess)
    se = np.sqrt(np.diag(cov))
    z = stats.norm.ppf(0.5 + CI_PROB / 2)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(theta)))
    for b in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot[b] = _lognorm_natural(stats.lognorm.fit(sample, floc=0))
    boot_lo = np.percentile(boot, 2.5, axis=0)
    boot_hi = np.percentile(boot, 97.5, axis=0)

    # Printing out summary statistics of MLE fitting
    out = {}
    print(f"{'Param':<10}{'MLE':>10}{'SE':>10}{'95% CI (Fisher)':>24}"
          f"{'95% CI (bootstrap)':>26}")
    for i, name in enumerate(names):
        lo, hi = theta[i] - z * se[i], theta[i] + z * se[i]
        out[name] = {
            "mle": float(theta[i]),
            "se": float(se[i]),
            "ci_analytic": (float(lo), float(hi)),
            "ci_boot": (float(boot_lo[i]), float(boot_hi[i])),
        }
        print(f"{name:<10}{theta[i]:>10.3f}{se[i]:>10.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>24}"
              f"{f'[{boot_lo[i]:.3f}, {boot_hi[i]:.3f}]':>26}")

    # Plotting the lognormal point estimates on the GPR histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=20, density=True, alpha=0.45, color=PALETTE["primary"],
            label="Empirical")
    x = np.linspace(data.min() * 0.8, data.max() * 1.1, 400)
    ax.plot(x, stats.lognorm.pdf(x, *params), color=PALETTE["alert"], lw=2.5,
            label="MLE LogNormal")
    ax.set_title("MLE Fit -- LogNormal over GPR Histogram")
    ax.set_xlabel("GPR")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return out


def fit_bayesian(data, draws=2000, tune=1000, chains=4, seed=RANDOM_SEED):
    """
    Run the full Bayesian pipeline: fit the LogNormal in PyMC via NUTS using
    weakly-informative priors (so the likelihood dominates), check convergence
    (R-hat, divergences), summarize the posterior, and run a posterior predictive
    check plot.

    Returns a dict mapping each param name (mu, sigma) to mean (posterior
    mean) and hdi (the HDI_PROB highest-density interval as a (low, high) tuple).
    """
    _section("BAYESIAN ESTIMATION (MCMC / PyMC) -- LogNormal")
    data_mean = float(np.mean(data))
    report_vars = ["mu", "sigma"]

    with pm.Model():
        mu = pm.Normal("mu", mu=np.log(data_mean), sigma=2.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.LogNormal("obs", mu=mu, sigma=sigma, observed=data)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
            progressbar=False, idata_kwargs={"log_likelihood": False},
        )
        pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, progressbar=False, random_seed=seed)

    rhat_ds = az.rhat(idata, var_names=report_vars)
    rhat_max = max(float(rhat_ds[v].max()) for v in report_vars)
    n_div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    print(f"Max R-hat: {rhat_max:.4f}  (want <= 1.01)")
    print(f"Divergences: {n_div}  (want 0)")
    if rhat_max > 1.01 or n_div > 0:
        print("  WARNING: diagnostics suggest re-running with more tune/draws.")

    hdi = az.hdi(idata, var_names=report_vars, hdi_prob=HDI_PROB)
    out = {}
    print(f"\n{'Param':<10}{'Post.Mean':>12}{f'{int(HDI_PROB*100)}% HDI':>26}")
    for name in report_vars:
        mean = float(idata.posterior[name].mean())
        lo = float(hdi[name].sel(hdi="lower"))
        hi = float(hdi[name].sel(hdi="higher"))
        out[name] = {"mean": mean, "hdi": (lo, hi)}
        print(f"{name:<10}{mean:>12.3f}{f'[{lo:.3f}, {hi:.3f}]':>26}")

    az.plot_ppc(idata, num_pp_samples=100)
    fig = plt.gcf()
    fig.suptitle("Posterior Predictive Check -- LogNormal")
    fig.tight_layout()
    return out


def compare_estimates(mle_out, bayes_out):
    """
    Print the MLE-vs-Bayes comparison table and show an overlay plot.
    """
    _section("COMPARISON & SYNTHESIS")
    names = [n for n in mle_out if n in bayes_out]

    print(f"{'Parameter':<12}{'MLE':>10}{'95% CI':>22}{'Posterior':>12}{'94% HDI':>22}")
    rows = []
    for name in names:
        mle, bay = mle_out[name], bayes_out[name]
        ci, hdi = mle["ci_analytic"], bay["hdi"]
        print(f"{name:<12}{mle['mle']:>10.3f}{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>22}"
              f"{bay['mean']:>12.3f}{f'[{hdi[0]:.3f}, {hdi[1]:.3f}]':>22}")
        rows.append((name, mle["mle"], ci, bay["mean"], hdi))
    print("CI = statement about the procedure; HDI = direct probability about the "
          "parameter. TODO(student): interpret in your own words.")

    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 3 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (label, mle_pt, ci, post_pt, hdi) in zip(axes, rows):
        ax.errorbar(mle_pt, 1.0, xerr=[[mle_pt - ci[0]], [ci[1] - mle_pt]], fmt="o",
                    color=PALETTE["alert"], capsize=5, label="MLE +/- 95% CI")
        ax.errorbar(post_pt, 0.0, xerr=[[post_pt - hdi[0]], [hdi[1] - post_pt]], fmt="s",
                    color=PALETTE["accent"], capsize=5, label="Posterior mean +/- 94% HDI")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Bayesian", "Frequentist"])
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(label)
        ax.legend(loc="best")
    fig.suptitle("Frequentist vs Bayesian Estimates -- LogNormal")
    fig.tight_layout()


def main():
    """
    Run the full estimation pipeline.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.random.seed(RANDOM_SEED)

    _section("GPR INDEX PROBABILISTIC ESTIMATION ENGINE (ARTEFACT 1)")
    print(f"Source: {GPR_URL}")
    print(f"Window: {START_DATE} -> {END_DATE}   Seed: {RANDOM_SEED}")

    df = load_data(START_DATE, END_DATE)
    data = data_quality_report(df)
    run_eda(df)
    params = fit_model(data)
    mle_out = fit_frequentist(data, params)
    bayes_out = fit_bayesian(data)
    compare_estimates(mle_out, bayes_out)

    _section("DONE")
    os.makedirs(os.path.dirname(os.path.abspath(DEFAULT_OUTPUT)), exist_ok=True)
    df.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"Processed dataframe written to {os.path.abspath(DEFAULT_OUTPUT)} "
          f"({len(df)} rows).")

    plt.show()


if __name__ == "__main__":
    main()
