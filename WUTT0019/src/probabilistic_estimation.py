# Student Name: TODO(student): Your Full Name
# Student FAN: WUTT0019
# File: probabilistic_estimation.py
# Date: TODO(student): DD-MM-YYYY
# Description: Frequentist vs Bayesian parameter estimation on the GPR Index.
# Usage: python probabilistic_estimation.py --input ../data/gpr_index.csv
# Licence: MIT Licence
"""Standalone probabilistic estimation engine for the GPR Index (Artefact 1).

This script mirrors ``GPR_Index_Analysis.ipynb`` end-to-end. It ingests the
Global Political Risk (GPR) Index, characterises its noise and uncertainty,
selects a probability distribution by AIC/BIC + goodness-of-fit, then estimates
that distribution's parameters two ways:

  * Frequentist  -- Maximum Likelihood Estimation with analytic (observed Fisher
    information) and bootstrap confidence intervals.
  * Bayesian     -- the same generative model in PyMC, yielding a full posterior,
    MAP, and 94% Highest Density Interval (HDI) credible intervals.

It finishes with a side-by-side comparison of point estimate + confidence
interval vs posterior mean + credible interval. The analysis window is
2018-01-01 -> 2025-01-01. Scope is *stationary* estimation only; sequential /
Markov modelling is Artefact 2 and is deliberately excluded.

Run ``python probabilistic_estimation.py --input ../data/gpr_index.csv`` to
print all numeric results and regenerate every figure into ``src/figures/``.
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # Headless: write figures to disk, never open a window.

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# --- Constants -------------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"
RANDOM_SEED = 42
HDI_PROB = 0.94  # PyMC/ArviZ convention; reported as the credible interval.
CI_PROB = 0.95  # Frequentist confidence level.
PALETTE = {"primary": "steelblue", "accent": "navy", "alert": "crimson"}

# Candidate families fitted to the (strictly positive) GPR level. Normal is a
# deliberately-weak baseline included to show why a naive choice fails.
CANDIDATES = ("gamma", "lognorm", "norm")


# --- CLI -------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with attributes ``input``,
        ``start``, ``end``, ``draws``, ``tune``, ``chains``, ``seed`` and
        ``figdir``.
    """
    parser = argparse.ArgumentParser(
        description="Frequentist vs Bayesian estimation on the GPR Index."
    )
    parser.add_argument(
        "--input", required=True, help="Path to gpr_index.csv (local, not committed)."
    )
    parser.add_argument("--start", default=START_DATE, help="Window start (YYYY-MM-DD).")
    parser.add_argument("--end", default=END_DATE, help="Window end (YYYY-MM-DD).")
    parser.add_argument("--draws", type=int, default=2000, help="Posterior draws/chain.")
    parser.add_argument("--tune", type=int, default=1000, help="NUTS tuning steps.")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--figdir", default="figures", help="Directory to write figures into."
    )
    return parser.parse_args()


# --- Small helpers ---------------------------------------------------------
def _section(title):
    """Print a labelled section banner to stdout.

    Args:
        title: Heading text to frame.
    """
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def _savefig(fig, figdir, name):
    """Save a figure into ``figdir`` and close it.

    Args:
        fig: Matplotlib figure to write.
        figdir: Target directory (created if absent).
        name: File name, e.g. ``"01_timeseries.png"``.

    Returns:
        str: Full path the figure was written to.
    """
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")
    return path


def _dist_label(dist_name):
    """Return a human-readable label for a scipy distribution name.

    Args:
        dist_name: One of ``"gamma"``, ``"lognorm"`` or ``"norm"``.

    Returns:
        str: Display label, e.g. ``"Gamma"``.
    """
    return {"gamma": "Gamma", "lognorm": "LogNormal", "norm": "Normal"}[dist_name]


# --- 1. Data ingestion -----------------------------------------------------
def load_data(input_path, start=START_DATE, end=END_DATE):
    """Load the GPR Index, select ``month`` + ``GPR`` and filter to the window.

    The raw CSV carries ~115 columns and decades of history with ``d/m/Y``
    dates; only the monthly headline ``GPR`` level inside the analysis window is
    retained. Rows with a missing GPR value (present in the pre-1980s history)
    are dropped after filtering.

    Args:
        input_path: Path to ``gpr_index.csv``.
        start: Inclusive window start as ``YYYY-MM-DD``.
        end: Inclusive window end as ``YYYY-MM-DD``.

    Returns:
        pandas.DataFrame: Two columns -- ``month`` (datetime64) and ``GPR``
        (float) -- sorted ascending by month with a reset index.
    """
    raw = pd.read_csv(input_path, usecols=["month", "GPR"])
    df = raw.copy()
    df["month"] = pd.to_datetime(df["month"], format="%d/%m/%Y", errors="coerce")
    df = df[(df["month"] >= start) & (df["month"] <= end)]
    df = df.dropna(subset=["GPR"]).sort_values("month").reset_index(drop=True)
    return df


# --- 2. Data quality & uncertainty report ----------------------------------
def data_quality_report(df):
    """Print shape, coverage, missingness, descriptive stats and a noise note.

    The descriptive block (mean/median/std/skew/kurtosis/IQR + Tukey fence) is
    the empirical evidence that motivates a skewed, positive distribution in the
    selection step. The closing paragraph names the *sources* of uncertainty in
    a text-based political-risk index so the report can discuss them.

    Args:
        df: Output of :func:`load_data`.

    Returns:
        numpy.ndarray: The clean GPR values as a 1-D float array, ready for the
        estimation steps.
    """
    _section("DATA QUALITY & UNCERTAINTY REPORT")
    gpr = df["GPR"]
    print(f"Shape:            {df.shape}")
    print(
        f"Date coverage:    {df['month'].min().date()} -> "
        f"{df['month'].max().date()} ({len(df)} months)"
    )
    print(
        f"Missing values:   month={df['month'].isna().sum()}, "
        f"GPR={gpr.isna().sum()}"
    )

    q1, q3 = gpr.quantile(0.25), gpr.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    print("\nDescriptive statistics")
    print(f"  Mean:           {gpr.mean():.2f}")
    print(f"  Median:         {gpr.median():.2f}")
    print(f"  Std Dev:        {gpr.std():.2f}")
    print(f"  Skewness:       {gpr.skew():.2f}  (>0 => right tail)")
    print(f"  Excess Kurtosis:{gpr.kurtosis():.2f}  (>0 => heavy tails)")
    print(f"  Min / Max:      {gpr.min():.2f} / {gpr.max():.2f}")
    print(f"  Q1 / Q3:        {q1:.2f} / {q3:.2f}")
    print(f"  IQR:            {iqr:.2f}")
    print(f"  Upper fence:    {upper_fence:.2f}")
    print(f"  Outliers above: {(gpr > upper_fence).sum()}")

    print(
        "\nSources of noise & inherent uncertainty (for the report's discussion):\n"
        "  * Measurement noise: the GPR is built by automated text-search over\n"
        "    newspaper archives; word counts are a noisy proxy for true risk.\n"
        "  * Revision uncertainty: archive coverage and the article corpus shift\n"
        "    over time, so historical values can be revised.\n"
        "  * Event-driven spikes: wars/invasions create heavy upper-tail months\n"
        "    that a symmetric model cannot represent -- hence a skewed family.\n"
        "  TODO(student): expand this in your own words for the report."
    )
    return gpr.to_numpy(dtype=float)


# --- 3. EDA (all 10 baseline plots) ----------------------------------------
def run_eda(df, figdir):
    """Reproduce the 10 baseline EDA plots and save them to ``figdir``.

    Each plot characterises a different facet of the data's uncertainty: level +
    volatility band, spikes, shape, seasonality, instability, yearly spread,
    month-over-month change, temporal dependence (ACF/PACF), the empirical CDF
    and a Normal Q-Q plot. ACF/PACF are retained purely as *evidence* of
    autocorrelation that motivates Artefact 2; no sequential model is built here.

    Args:
        df: Output of :func:`load_data`.
        figdir: Directory to write figures into.
    """
    _section("EXPLORATORY DATA ANALYSIS")
    month, gpr = df["month"], df["GPR"]

    # 1. Time series with 12-month rolling mean and +/-1 sigma band.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(month, gpr, alpha=0.4, label="Monthly GPR", color=PALETTE["primary"])
    roll = gpr.rolling(12)
    ax.plot(month, roll.mean(), label="12m Rolling Mean", color=PALETTE["accent"], lw=2)
    ax.fill_between(
        month,
        roll.mean() - roll.std(),
        roll.mean() + roll.std(),
        alpha=0.15,
        color=PALETTE["accent"],
        label="+/-1 sigma Band",
    )
    ax.set_title("GPR Index with Rolling Mean & Volatility Band")
    ax.set_ylabel("GPR")
    ax.legend()
    _savefig(fig, figdir, "01_timeseries_rolling.png")

    # 2. Spike / outlier detection (Tukey fence, annotated).
    q1, q3 = gpr.quantile(0.25), gpr.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(month, gpr, width=25, color=PALETTE["primary"], alpha=0.6)
    spikes = df[df["GPR"] > fence]
    ax.bar(
        spikes["month"],
        spikes["GPR"],
        width=25,
        color=PALETTE["alert"],
        alpha=0.85,
        label=f"Outliers (>{fence:.0f})",
    )
    for _, row in spikes.iterrows():
        ax.annotate(
            row["month"].strftime("%b %Y"),
            (row["month"], row["GPR"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=PALETTE["alert"],
        )
    ax.axhline(fence, ls="--", color=PALETTE["alert"], alpha=0.5)
    ax.set_title("GPR Spike Detection (Tukey Fence)")
    ax.set_ylabel("GPR")
    ax.legend()
    _savefig(fig, figdir, "02_spike_detection.png")

    # 3. Distribution histogram (raw) + log-transformed histogram.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(gpr, kde=True, bins=20, ax=axes[0], color=PALETTE["primary"])
    axes[0].axvline(gpr.mean(), ls="--", color="red", label=f"Mean ({gpr.mean():.0f})")
    axes[0].axvline(
        gpr.median(), ls="--", color="orange", label=f"Median ({gpr.median():.0f})"
    )
    axes[0].set_title("GPR Distribution (raw)")
    axes[0].legend()
    sns.histplot(np.log1p(gpr), kde=True, bins=20, ax=axes[1], color="teal")
    axes[1].set_title("Log-Transformed GPR Distribution")
    axes[1].set_xlabel("log(1 + GPR)")
    _savefig(fig, figdir, "03_histograms.png")

    # 4. Year x Month heatmap.
    heat = df.copy()
    heat["year"] = heat["month"].dt.year
    heat["mo"] = heat["month"].dt.month
    pivot = heat.pivot_table(index="year", columns="mo", values="GPR")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5, ax=ax,
        cbar_kws={"label": "GPR"},
    )
    ax.set_title("GPR by Year x Month")
    ax.set_ylabel("")
    _savefig(fig, figdir, "04_heatmap.png")

    # 5. 6-month rolling volatility.
    fig, ax = plt.subplots(figsize=(14, 5))
    vol = gpr.rolling(6).std()
    ax.fill_between(month, vol, alpha=0.4, color="coral", label="6m Rolling Std Dev")
    ax.plot(month, vol, color="firebrick", lw=1)
    ax.set_title("GPR Rolling Volatility -- How Unstable Is the Risk Environment?")
    ax.set_ylabel("6-Month Rolling Std Dev")
    ax.legend()
    _savefig(fig, figdir, "05_rolling_volatility.png")

    # 6. Per-year boxplots.
    box = df.copy()
    box["year"] = box["month"].dt.year
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(
        data=box, x="year", y="GPR", hue="year", palette="coolwarm", legend=False, ax=ax
    )
    ax.set_title("GPR Distribution by Year")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45)
    _savefig(fig, figdir, "06_yearly_boxplots.png")

    # 7. Month-over-month % change.
    mom = gpr.pct_change() * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [PALETTE["alert"] if x > 0 else PALETTE["primary"] for x in mom]
    ax.bar(month, mom, color=colors, width=25, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Month-over-Month % Change in GPR")
    ax.set_ylabel("% Change")
    _savefig(fig, figdir, "07_mom_change.png")

    # 8. ACF / PACF (evidence of temporal dependence -> motivates Artefact 2).
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sm.graphics.tsa.plot_acf(gpr.dropna(), lags=24, ax=axes[0], color=PALETTE["primary"])
    axes[0].set_title("Autocorrelation (ACF)")
    sm.graphics.tsa.plot_pacf(gpr.dropna(), lags=24, ax=axes[1], color="teal")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    _savefig(fig, figdir, "08_acf_pacf.png")

    # 9. Empirical CDF with median + 95th percentile markers.
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_gpr = np.sort(gpr.dropna())
    cdf = np.arange(1, len(sorted_gpr) + 1) / len(sorted_gpr)
    ax.plot(sorted_gpr, cdf, color=PALETTE["primary"], lw=2)
    ax.axhline(0.5, ls=":", color="gray", alpha=0.5)
    ax.axhline(0.95, ls=":", color=PALETTE["alert"], alpha=0.5)
    ax.axvline(gpr.median(), ls=":", color="gray", alpha=0.5)
    ax.annotate("Median", (gpr.median(), 0.52), fontsize=9, color="gray")
    ax.annotate(
        "95th percentile", (gpr.quantile(0.95), 0.92), fontsize=9, color=PALETTE["alert"]
    )
    ax.set_title("Empirical CDF -- GPR Index")
    ax.set_xlabel("GPR")
    ax.set_ylabel("Cumulative Probability")
    _savefig(fig, figdir, "09_ecdf.png")

    # 10. Q-Q plot vs Normal.
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(gpr.dropna(), dist="norm", plot=ax)
    ax.set_title("Q-Q Plot -- GPR vs Normal Distribution")
    ax.get_lines()[0].set_color(PALETTE["primary"])
    ax.get_lines()[1].set_color(PALETTE["alert"])
    _savefig(fig, figdir, "10_qq_normal.png")

    # SUGGESTION (Artefact 1, propose-but-flag): a KDE of month-over-month
    # changes is a nice stationarity sanity check. Not auto-built -- add if the
    # report needs it.
    # SUGGESTION (Artefact 2 territory, do NOT build here): changepoint/regime
    # detection on the volatility series.


# --- 4. Distribution selection ---------------------------------------------
def _fit_candidate(data, dist_name):
    """Fit one candidate family by MLE and return its scipy parameter tuple.

    Positive-support families (Gamma, LogNormal) are fitted with ``floc=0`` so
    they remain genuine two-parameter (shape, scale) models on the positive
    line; this keeps the AIC/BIC comparison fair (every candidate has k=2).

    Args:
        data: 1-D array of GPR values.
        dist_name: One of :data:`CANDIDATES`.

    Returns:
        tuple: The scipy ``params`` tuple as returned by ``dist.fit``.
    """
    dist = getattr(stats, dist_name)
    if dist_name in ("gamma", "lognorm"):
        return dist.fit(data, floc=0)
    return dist.fit(data)


def select_distribution(data, figdir):
    """Fit candidates, rank by AIC/BIC + K-S, draw Q-Q panels, pick a winner.

    For each of Gamma, LogNormal and Normal it computes the log-likelihood,
    AIC, BIC and a Kolmogorov-Smirnov statistic/p-value, prints a ranked table,
    saves a fitted-PDF overlay and a per-candidate Q-Q panel, then selects the
    family with the lowest AIC. AIC and BIC trade goodness-of-fit against
    complexity; with k fixed at 2 here, lower simply means better fit.

    Args:
        data: 1-D array of GPR values.
        figdir: Directory to write figures into.

    Returns:
        tuple: ``(winner_name, winner_params, results)`` where ``results`` is a
        dict keyed by distribution name holding params, loglik, aic, bic and the
        K-S statistic/p-value.
    """
    _section("DISTRIBUTION SELECTION")
    n = len(data)
    results = {}
    for name in CANDIDATES:
        dist = getattr(stats, name)
        params = _fit_candidate(data, name)
        loglik = float(np.sum(dist.logpdf(data, *params)))
        k = 2  # All candidates are effectively two-parameter here.
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik
        ks_stat, ks_p = stats.kstest(data, name, args=params)
        results[name] = {
            "params": params,
            "loglik": loglik,
            "aic": aic,
            "bic": bic,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
        }

    print(f"{'Family':<11}{'logLik':>11}{'AIC':>11}{'BIC':>11}{'KS stat':>11}{'KS p':>9}")
    for name in CANDIDATES:
        r = results[name]
        print(
            f"{_dist_label(name):<11}{r['loglik']:>11.2f}{r['aic']:>11.2f}"
            f"{r['bic']:>11.2f}{r['ks_stat']:>11.3f}{r['ks_p']:>9.3f}"
        )

    winner = min(CANDIDATES, key=lambda d: results[d]["aic"])
    runner_up = min((d for d in CANDIDATES if d != winner),
                    key=lambda d: results[d]["aic"])
    delta = results[runner_up]["aic"] - results[winner]["aic"]
    justification = (
        f"{_dist_label(winner)} is selected: lowest AIC "
        f"({results[winner]['aic']:.2f}) and BIC ({results[winner]['bic']:.2f}), "
        f"beating the next-best ({_dist_label(runner_up)}) by dAIC={delta:.2f}. "
        f"Its K-S p-value ({results[winner]['ks_p']:.3f}) does not reject the fit. "
        f"This matches the EDA: a strictly-positive, right-skewed level is poorly "
        f"served by the Normal baseline."
    )
    print(f"\nSELECTED: {_dist_label(winner)}")
    print(justification)
    print("TODO(student): confirm this choice in your own words in the report.")

    # Auto-include: overlay of each candidate fitted PDF on the histogram.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=20, density=True, alpha=0.4, color=PALETTE["primary"],
            label="Empirical")
    x = np.linspace(data.min() * 0.8, data.max() * 1.1, 400)
    overlay_colors = {"gamma": "darkgreen", "lognorm": "navy", "norm": "crimson"}
    for name in CANDIDATES:
        dist = getattr(stats, name)
        ax.plot(x, dist.pdf(x, *results[name]["params"]), lw=2,
                color=overlay_colors[name],
                label=f"{_dist_label(name)} (AIC {results[name]['aic']:.0f})")
    ax.set_title("Candidate Distributions Overlaid on the GPR Histogram")
    ax.set_xlabel("GPR")
    ax.set_ylabel("Density")
    ax.legend()
    _savefig(fig, figdir, "11_candidate_pdf_overlay.png")

    # Auto-include: per-candidate Q-Q panel so the choice is visual.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, CANDIDATES):
        dist = getattr(stats, name)
        # sparams are the *shape* params only (probplot adds loc/scale via fit).
        sparams = results[name]["params"][:-2] if name != "norm" else ()
        stats.probplot(data, dist=dist, sparams=sparams, plot=ax)
        ax.set_title(f"Q-Q: {_dist_label(name)}")
        ax.get_lines()[0].set_color(PALETTE["primary"])
        ax.get_lines()[1].set_color(PALETTE["alert"])
    fig.suptitle("Per-Candidate Q-Q Plots")
    _savefig(fig, figdir, "12_candidate_qq_panels.png")

    return winner, results[winner]["params"], results


# --- 5. Frequentist estimation (MLE) ---------------------------------------
def _natural_params(dist_name, params):
    """Map a scipy parameter tuple to (name, value) pairs in natural units.

    Args:
        dist_name: One of :data:`CANDIDATES`.
        params: The scipy ``params`` tuple for that distribution.

    Returns:
        list[tuple[str, float]]: Ordered, comparison-ready parameters --
        Gamma: ``shape``, ``scale``; LogNormal: ``mu``, ``sigma``;
        Normal: ``mu``, ``sigma``.
    """
    if dist_name == "gamma":
        shape, _loc, scale = params
        return [("shape (alpha)", shape), ("scale (1/beta)", scale)]
    if dist_name == "lognorm":
        s, _loc, scale = params
        return [("mu", float(np.log(scale))), ("sigma", s)]
    loc, scale = params
    return [("mu", loc), ("sigma", scale)]


def _natural_names(dist_name):
    """Return the natural parameter names for a family (matching the PyMC model).

    Args:
        dist_name: One of :data:`CANDIDATES`.

    Returns:
        list[str]: Two names -- Gamma: ``shape (alpha)``, ``scale (1/beta)``;
        LogNormal/Normal: ``mu``, ``sigma``. These match the variable names the
        Bayesian step reports, so the comparison aligns by name.
    """
    if dist_name == "gamma":
        return ["shape (alpha)", "scale (1/beta)"]
    return ["mu", "sigma"]


def _scipy_to_natural(dist_name, params):
    """Convert a scipy ``params`` tuple to the natural parameter vector.

    Args:
        dist_name: One of :data:`CANDIDATES`.
        params: The scipy ``params`` tuple from ``dist.fit``.

    Returns:
        numpy.ndarray: Natural-unit vector aligned with :func:`_natural_names`
        (Gamma: shape, scale; LogNormal: mu=log(scale), sigma=s; Normal:
        mu=loc, sigma=scale).
    """
    return np.array([v for _, v in _natural_params(dist_name, params)], dtype=float)


def _nll(theta, data, dist_name):
    """Negative log-likelihood at a *natural* parameter vector.

    The estimation is carried out in the same parameterisation as the Bayesian
    model (Gamma: shape & scale; LogNormal/Normal: mu & sigma) so the resulting
    standard errors and confidence intervals are directly comparable to the HDIs.

    Args:
        theta: Natural parameter vector (see :func:`_natural_names`).
        data: 1-D array of GPR values.
        dist_name: One of :data:`CANDIDATES`.

    Returns:
        float: The negative log-likelihood (``inf`` for invalid parameters).
    """
    dist = getattr(stats, dist_name)
    if dist_name == "gamma":
        shape, scale = theta
        if shape <= 0 or scale <= 0:
            return np.inf
        lp = dist.logpdf(data, shape, loc=0, scale=scale)
    elif dist_name == "lognorm":
        mu, sigma = theta
        if sigma <= 0:
            return np.inf
        lp = dist.logpdf(data, sigma, loc=0, scale=np.exp(mu))
    else:
        mu, sigma = theta
        if sigma <= 0:
            return np.inf
        lp = dist.logpdf(data, loc=mu, scale=sigma)
    return -float(np.sum(lp))


def _numerical_hessian(func, theta, eps=1e-5):
    """Central-difference Hessian of a scalar function.

    Args:
        func: Scalar function of a parameter vector.
        theta: Point at which to evaluate the Hessian.
        eps: Relative step size.

    Returns:
        numpy.ndarray: The ``(k, k)`` Hessian matrix.
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
            hess[i, j] = (
                func(tpp) - func(tpm) - func(tmp) + func(tmm)
            ) / (4 * h[i] * h[j])
    return hess


def fit_frequentist(data, dist_name, params, figdir, seed=RANDOM_SEED, n_boot=2000):
    """Estimate the winning family by MLE with analytic + bootstrap CIs.

    Analytic 95% CIs come from the observed Fisher information (inverse of the
    numerical Hessian of the negative log-likelihood at the MLE). A percentile
    bootstrap CI is computed as a robustness cross-check that does not rely on
    the asymptotic-normality assumption. A fitted-PDF-over-histogram plot is
    saved.

    Args:
        data: 1-D array of GPR values.
        dist_name: The selected family (one of :data:`CANDIDATES`).
        params: The scipy ``params`` tuple from :func:`select_distribution`.
        figdir: Directory to write figures into.
        seed: Bootstrap RNG seed.
        n_boot: Number of bootstrap resamples.

    Returns:
        dict: Per natural-parameter name -> dict with ``mle``, ``se``,
        ``ci_analytic`` (low, high) and ``ci_boot`` (low, high).
    """
    _section(f"FREQUENTIST ESTIMATION (MLE) -- {_dist_label(dist_name)}")
    theta = _scipy_to_natural(dist_name, params)
    free_names = _natural_names(dist_name)

    hess = _numerical_hessian(lambda t: _nll(t, data, dist_name), theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = np.linalg.inv(hess)
    se = np.sqrt(np.diag(cov))
    z = stats.norm.ppf(0.5 + CI_PROB / 2)

    # Bootstrap: refit the family on resamples; percentile interval per param.
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(theta)))
    for b in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot[b] = _scipy_to_natural(dist_name, _fit_candidate(sample, dist_name))
    boot_lo = np.percentile(boot, 2.5, axis=0)
    boot_hi = np.percentile(boot, 97.5, axis=0)

    out = {}
    print(f"{'Param':<16}{'MLE':>10}{'SE':>10}{'95% CI (Fisher)':>24}"
          f"{'95% CI (bootstrap)':>26}")
    for i, fname in enumerate(free_names):
        lo, hi = theta[i] - z * se[i], theta[i] + z * se[i]
        out[fname] = {
            "mle": float(theta[i]),
            "se": float(se[i]),
            "ci_analytic": (float(lo), float(hi)),
            "ci_boot": (float(boot_lo[i]), float(boot_hi[i])),
        }
        print(
            f"{fname:<16}{theta[i]:>10.3f}{se[i]:>10.3f}"
            f"{f'[{lo:.3f}, {hi:.3f}]':>24}"
            f"{f'[{boot_lo[i]:.3f}, {boot_hi[i]:.3f}]':>26}"
        )

    # Fitted PDF over the histogram.
    dist = getattr(stats, dist_name)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=20, density=True, alpha=0.45, color=PALETTE["primary"],
            label="Empirical")
    x = np.linspace(data.min() * 0.8, data.max() * 1.1, 400)
    ax.plot(x, dist.pdf(x, *params), color=PALETTE["alert"], lw=2.5,
            label=f"MLE {_dist_label(dist_name)}")
    ax.set_title(f"MLE Fit -- {_dist_label(dist_name)} over GPR Histogram")
    ax.set_xlabel("GPR")
    ax.set_ylabel("Density")
    ax.legend()
    _savefig(fig, figdir, "13_mle_fit.png")

    # SUGGESTION (propose-but-flag): plot the bootstrap sampling distribution of
    # the mean as a frequentist-uncertainty visual. Not auto-built.
    return out


# --- 6. Bayesian estimation (PyMC) -----------------------------------------
def fit_bayesian(data, dist_name, figdir, draws=2000, tune=1000, chains=4,
                 seed=RANDOM_SEED):
    """Estimate the same family in PyMC: posterior, MAP, 94% HDI, diagnostics.

    Uses weakly-informative priors scaled to the data so the likelihood
    dominates. Samples with NUTS, then reports posterior means, the MAP, 94%
    HDIs, R-hat and divergence counts, and runs a posterior predictive check.
    Trace, posterior and PPC figures are saved. Parameters are reported in the
    same natural units as the MLE step for a like-for-like comparison.

    Args:
        data: 1-D array of GPR values.
        dist_name: The selected family (one of :data:`CANDIDATES`).
        figdir: Directory to write figures into.
        draws: Posterior draws per chain.
        tune: NUTS tuning steps.
        chains: Number of chains.
        seed: Sampler seed.

    Returns:
        dict: Per natural-parameter name -> dict with ``mean``, ``map`` and
        ``hdi`` (low, high).
    """
    _section(f"BAYESIAN ESTIMATION (PyMC) -- {_dist_label(dist_name)}")
    data_mean = float(np.mean(data))
    data_sd = float(np.std(data))

    with pm.Model() as model:
        if dist_name == "gamma":
            # Weakly-informative: positive shape/rate, Exponential priors whose
            # means sit near plausible values without pinning the posterior.
            alpha = pm.Exponential("shape (alpha)", 1.0 / 5.0)
            beta = pm.Exponential("rate (beta)", 1.0)
            pm.Gamma("obs", alpha=alpha, beta=beta, observed=data)
            scale = pm.Deterministic("scale (1/beta)", 1.0 / beta)
            report_vars = ["shape (alpha)", "scale (1/beta)"]
        elif dist_name == "lognorm":
            mu = pm.Normal("mu", mu=np.log(data_mean), sigma=2.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            pm.LogNormal("obs", mu=mu, sigma=sigma, observed=data)
            report_vars = ["mu", "sigma"]
        else:
            mu = pm.Normal("mu", mu=data_mean, sigma=2 * data_sd)
            sigma = pm.HalfNormal("sigma", sigma=2 * data_sd)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
            report_vars = ["mu", "sigma"]

        # cores=1 for cross-platform reproducibility on constrained machines.
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
            progressbar=False, idata_kwargs={"log_likelihood": False},
        )
        map_estimate = pm.find_MAP(progressbar=False)
        # extend_inferencedata=True lets PyMC attach the predictive group in a
        # way that works across ArviZ versions (newer InferenceData is a
        # DataTree without an .extend method).
        pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, progressbar=False, random_seed=seed)

    summary = az.summary(idata, var_names=report_vars, hdi_prob=HDI_PROB)
    print(summary.to_string())

    rhat_ds = az.rhat(idata, var_names=report_vars)
    rhat_max = max(float(rhat_ds[v].max()) for v in report_vars)
    n_div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    print(f"\nMax R-hat: {rhat_max:.4f}  (want <= 1.01)")
    print(f"Divergences: {n_div}  (want 0)")
    if rhat_max > 1.01 or n_div > 0:
        print("  WARNING: diagnostics suggest re-running with more tune/draws.")

    hdi = az.hdi(idata, var_names=report_vars, hdi_prob=HDI_PROB)
    out = {}
    print(f"\n{'Param':<16}{'Post.Mean':>12}{'MAP':>12}"
          f"{f'{int(HDI_PROB*100)}% HDI':>26}")
    for name in report_vars:
        mean = float(idata.posterior[name].mean())
        # MAP keys can carry a transformed suffix; fall back to the mean if so.
        map_val = float(np.asarray(map_estimate[name])) if name in map_estimate \
            else mean
        lo = float(hdi[name].sel(hdi="lower"))
        hi = float(hdi[name].sel(hdi="higher"))
        out[name] = {"mean": mean, "map": map_val, "hdi": (lo, hi)}
        print(f"{name:<16}{mean:>12.3f}{map_val:>12.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>26}")

    # Trace plot.
    az.plot_trace(idata, var_names=report_vars)
    fig = plt.gcf()
    fig.suptitle(f"Trace -- {_dist_label(dist_name)}")
    fig.tight_layout()
    _savefig(fig, figdir, "14_trace.png")

    # Posterior plot with HDI.
    az.plot_posterior(idata, var_names=report_vars, hdi_prob=HDI_PROB)
    fig = plt.gcf()
    fig.suptitle(f"Posterior -- {_dist_label(dist_name)}")
    fig.tight_layout()
    _savefig(fig, figdir, "15_posterior.png")

    # Posterior predictive check.
    az.plot_ppc(idata, num_pp_samples=100)
    fig = plt.gcf()
    fig.suptitle(f"Posterior Predictive Check -- {_dist_label(dist_name)}")
    fig.tight_layout()
    _savefig(fig, figdir, "16_ppc.png")

    # SUGGESTION (propose-but-flag): overlay each parameter's prior vs posterior
    # to sharpen the Bayesian story. Not auto-built.
    return out


# --- 7. Comparison & synthesis ---------------------------------------------
def compare_estimates(dist_name, mle_out, bayes_out, figdir):
    """Print a comparison table and save an MLE-vs-Bayes overlay plot.

    Aligns the frequentist point estimate + 95% confidence interval against the
    Bayesian posterior mean + 94% HDI credible interval for each parameter, on a
    shared natural scale. A confidence interval is a statement about the
    procedure (95% of such intervals cover the fixed true value); a credible
    interval is a direct probability statement about the parameter given the
    data and prior. The overlay makes any agreement/disagreement visible.

    Args:
        dist_name: The selected family.
        mle_out: Return value of :func:`fit_frequentist`.
        bayes_out: Return value of :func:`fit_bayesian`.
        figdir: Directory to write figures into.
    """
    _section("COMPARISON & SYNTHESIS")
    # Both steps report the same natural parameter names, so align by name.
    names = [n for n in mle_out if n in bayes_out]

    print(f"{'Parameter':<18}{'MLE':>10}{'95% CI':>22}"
          f"{'Posterior':>12}{'94% HDI':>22}")
    rows = []
    for name in names:
        mle = mle_out[name]
        bay = bayes_out[name]
        ci = mle["ci_analytic"]
        hdi = bay["hdi"]
        print(
            f"{name:<18}{mle['mle']:>10.3f}{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>22}"
            f"{bay['mean']:>12.3f}{f'[{hdi[0]:.3f}, {hdi[1]:.3f}]':>22}"
        )
        rows.append((name, mle["mle"], ci, bay["mean"], hdi))

    print(
        "\nReading this table (TODO(student): interpret in your own words):\n"
        "  * Where MLE point and posterior mean nearly coincide and the CI/HDI\n"
        "    overlap, the weakly-informative prior added little -- the data\n"
        "    dominate. Divergence would signal prior influence or small-sample\n"
        "    effects.\n"
        "  * For a high-stakes risk index, the Bayesian HDI communicates\n"
        "    parameter uncertainty as a direct probability, which is often\n"
        "    easier for a decision-maker than a frequentist CI's coverage logic."
    )

    # Overlay plot: one row per parameter, MLE (CI) vs posterior (HDI).
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 3 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (label, mle_pt, ci, post_pt, hdi) in zip(axes, rows):
        ax.errorbar(mle_pt, 1.0, xerr=[[mle_pt - ci[0]], [ci[1] - mle_pt]],
                    fmt="o", color=PALETTE["alert"], capsize=5,
                    label="MLE +/- 95% CI")
        ax.errorbar(post_pt, 0.0, xerr=[[post_pt - hdi[0]], [hdi[1] - post_pt]],
                    fmt="s", color=PALETTE["accent"], capsize=5,
                    label="Posterior mean +/- 94% HDI")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Bayesian", "Frequentist"])
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(label)
        ax.legend(loc="best")
    fig.suptitle(f"Frequentist vs Bayesian Estimates -- {_dist_label(dist_name)}")
    fig.tight_layout()
    _savefig(fig, figdir, "17_comparison.png")


# --- 8. Orchestration ------------------------------------------------------
def main():
    """Run the full estimation pipeline from the command line."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    np.random.seed(args.seed)

    _section("GPR INDEX PROBABILISTIC ESTIMATION ENGINE (ARTEFACT 1)")
    print(f"Input:  {args.input}")
    print(f"Window: {args.start} -> {args.end}   Seed: {args.seed}")
    print(f"Figures -> {os.path.abspath(args.figdir)}")

    df = load_data(args.input, args.start, args.end)
    data = data_quality_report(df)
    run_eda(df, args.figdir)
    winner, params, _results = select_distribution(data, args.figdir)
    mle_out = fit_frequentist(data, winner, params, args.figdir, seed=args.seed)
    bayes_out = fit_bayesian(
        data, winner, args.figdir, draws=args.draws, tune=args.tune,
        chains=args.chains, seed=args.seed,
    )
    compare_estimates(winner, mle_out, bayes_out, args.figdir)

    _section("DONE")
    print("All numeric results printed above; figures written to "
          f"{os.path.abspath(args.figdir)}.")


if __name__ == "__main__":
    main()
