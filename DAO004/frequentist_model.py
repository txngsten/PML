# Student Name: Phuong Quyen Dao
# Student FAN:  dao0047 
# File:        frequentist_model.py
# Date:        16-5-2026
# Description: fit frequentist models to estimate relationship between weather features and log returns using OLS and distribution fitting 
# Usage: frequentist_model.py

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv("final_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

#Select features
X = df[[
    "avg_temperature",
    "avg_wind_speed",
    "avg_solar_radiation"
]]

#Target variable
y = df["log_return"]

#Add intercept
X = sm.add_constant(X)

#Build and fit OLS model
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()

print("\nOLS regression results:", ols_results.summary())

#Save OLS coefficients
ols_summary_table = pd.DataFrame({
    "parameter": ols_results.params.index,
    "estimate": ols_results.params.values,
    "std_error": ols_results.bse.values,
    "t_value": ols_results.tvalues.values,
    "p_value": ols_results.pvalues.values,
    "conf_low": ols_results.conf_int()[0].values,
    "conf_high": ols_results.conf_int()[1].values
})

ols_summary_table.to_csv("./Misc/Outputs/frequentist_ols_results.csv")


#Distribution estimation
returns = df["log_return"].dropna()

#Normal mle
normal_mu, normal_sigma = stats.norm.fit(returns)
normal_log_likelihood = np.sum(stats.norm.logpdf(returns, normal_mu, normal_sigma))

#Student-t mle
t_df, t_loc, t_scale = stats.t.fit(returns)
t_log_likelihood = np.sum(stats.t.logpdf(returns, t_df, t_loc, t_scale))

distribution_results = pd.DataFrame({
    "model": ["Normal", "Student-t"],
    "param_1": [normal_mu, t_df],
    "param_2": [normal_sigma, t_loc],
    "param_3": [np.nan, t_scale],
    "log_likelihood": [normal_log_likelihood, t_log_likelihood]
})

print("\nDistribution MLE results:", distribution_results)

distribution_results.to_csv("./Misc/Outputs/frequentist_distribution_results.csv")

#Bucket distribution estimates
bucket_results = []

for bucket in ["pre_covid", "covid", "post_covid"]:
    bucket_returns = df[df["period_bucket"] == bucket]["log_return"].dropna()

    b_mu, b_sigma = stats.norm.fit(bucket_returns)
    b_t_df, b_t_loc, b_t_scale = stats.t.fit(bucket_returns)

    bucket_results.append({
        "period_bucket": bucket,
        "normal_mu": b_mu,
        "normal_sigma": b_sigma,
        "student_t_df": b_t_df,
        "student_t_loc": b_t_loc,
        "student_t_scale": b_t_scale
    })

bucket_results_df = pd.DataFrame(bucket_results)

print("\nBucket distribution results:", bucket_results_df)

bucket_results_df.to_csv("./Misc/Outputs/frequentist_bucket_distribution_results.csv")