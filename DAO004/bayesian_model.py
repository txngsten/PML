# Student Name: Phuong Quyen Dao
# Student FAN:  dao0047 
# File:        bayesian_model.py
# Date:        15-5-2026 
# Description: fits a bayesian model to estimate uncertainty in the relationship between weather features and log returns
# Usage: bayesian_model.py

import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt

df = pd.read_csv("final_dataset.csv")

#Build bayesian regression model
model = bmb.Model(
    "log_return ~ avg_temperature + avg_wind_speed + avg_solar_radiation",
    data=df
)

#Run MCMC sampling
results = model.fit(
    draws=2000,         #generate 2000 posterior samples
    tune=1000,
    chains=2,
    cores=1
)

#Print posterior summary
print(az.summary(results))

#Plot distributions
az.plot_posterior(results)
plt.tight_layout()
plt.savefig("./Misc/Outputs/bayesian_posterior_plot.png")
plt.show()