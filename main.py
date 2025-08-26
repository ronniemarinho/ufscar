import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ------------------------------
# 1. Carregar dados
iris_dataset = load_iris(as_frame=True)
iris = iris_dataset.frame
iris['Species'] = iris_dataset.target_names[iris_dataset.target]

# Filtrar apenas setosa
dat = iris[iris['Species'] == 'setosa'][['petal length (cm)', 'sepal length (cm)']]
dat.columns = ['Petal_Length', 'Sepal_Length']

# ------------------------------
# 2. Modelo PyMC 5 (Regressão Linear Simples)
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=10, sigma=0.1)
    beta = pm.Normal("beta", mu=1, sigma=0.1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = alpha + beta * dat['Sepal_Length']
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=dat['Petal_Length'])

    idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, random_seed=42)

    # Posterior predictive
    ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"])

# Adicionar posterior_predictive ao idata
idata.extend(ppc)

# ------------------------------
# 3. Resumo dos parâmetros
print(az.summary(idata, var_names=["alpha", "beta", "sigma"]))

# ------------------------------
# 4. Traceplot
az.plot_trace(idata, var_names=["alpha", "beta", "sigma"])
plt.show()

# ------------------------------
# 5. Posterior Predictive Check
az.plot_ppc(idata)
plt.show()
