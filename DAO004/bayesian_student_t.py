# Student Name: Phuong Quyen Dao
# Student FAN:  dao0047 
# File:        bayesian_student_t.py
# Date:        16-5-2026
# Description: fit a Bayesian student-t model to AGL log returns to capture uncertainty and market movements
# Usage: bayesian_student_t.py

import os
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv("final_dataset.csv")

#Keep only the return variable
returns = df["log_return"].dropna().values

#Frequentist Student-t estimate for comparison
freq_df, freq_loc, freq_scale = stats.t.fit(returns)

print("Frequentist Student-t MLE:")
print("df:", freq_df)
print("loc:", freq_loc)
print("scale:", freq_scale)

#Build Bayesian Student-t model
with pm.Model() as student_t_model:

    #Prior for average daily return
    mu = pm.Normal(
        "mu",
        mu=0,
        sigma=0.05
    )

    #Prior for volatility / scale
    sigma = pm.HalfNormal(
        "sigma",
        sigma=0.05
    )

    #Prior for tail heaviness
    # Smaller nu = heavier tails
    nu_minus_one = pm.Exponential(
        "nu_minus_one",
        lam=1 / 10
    )

    nu = pm.Deterministic(
        "nu",
        nu_minus_one + 1
    )

    #Student-t likelihood
    observed_returns = pm.StudentT(
        "observed_returns",
        nu=nu,
        mu=mu,
        sigma=sigma,
        observed=returns
    )

    #Run MCMC sampling
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=42
    )

    #Posterior predictive sampling
    pm.sample_posterior_predictive(
        idata,
        extend_inferencedata=True,
        random_seed=42
    )

#Print posterior summary
summary = az.summary(
    idata,
    var_names=["mu", "sigma", "nu"],
    hdi_prob=0.94
)

print("\nBayesian Student-t Posterior Summary:", summary)

#Save summary table
summary.to_csv("bayesian_student_t_summary.csv")

#Plot posterior distributions
az.plot_posterior(
    idata,
    var_names=["mu", "sigma", "nu"],
    hdi_prob=0.94
)

plt.tight_layout()
plt.savefig("bayesian_student_t_posterior.png")
plt.show()

#Posterior predictive check
az.plot_ppc(
    idata,
    num_pp_samples=100
)

plt.tight_layout()
plt.savefig("./Misc/Outputs/bayesian_student_t_ppc.png")
plt.show()