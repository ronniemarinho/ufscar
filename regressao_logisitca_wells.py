import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# --- Leitura e preparação dos dados ---
data = pd.read_csv("wells.dat", sep=r"\s+")
data["dist100"] = data["dist"] / 100  # escala de 100 metros

y = data["switch"].values
a = pm.math.logit(np.mean(y))  # logit da probabilidade média de trocar poço
X = data[["dist100", "arsenic"]].values

print(f"Colunas: {list(data.columns)}")
print(f"Observações: {len(data)}, Preditores: {X.shape[1]}")

# --- Modelo PyMC 5 ---
with pm.Model() as model:
    beta0 = pm.StudentT("beta0", nu=7, mu=a, sigma=0.1)
    beta = pm.StudentT("beta", nu=7, mu=0, sigma=1, shape=X.shape[1])

    logit_p = beta0 + pm.math.dot(X, beta)
    p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    idata = pm.sample(2000, tune=2000, chains=4, target_accept=0.95)

# --- Resumo estatístico ---
print(az.summary(idata, var_names=["beta0", "beta"]))

# --- Intervalos de credibilidade (HDI) ---
p_samples = idata.posterior["p"].stack(draws=("chain", "draw")).values
hdi_intervals = az.hdi(p_samples.T, hdi_prob=0.94)

print("\nHDI 94% para as primeiras 5 observações:")
print(hdi_intervals[:5])

# --- Forest plot ---
az.plot_forest(idata, var_names=["beta0", "beta"], hdi_prob=0.94)
plt.title("Intervalos de Credibilidade 94%")
plt.show()

# --- Trace plot ---
az.plot_trace(idata, var_names=["beta0", "beta"])
plt.show()

# --- Gráfico previsão vs. observação ---
p_mean = p_samples.mean(axis=1)
plt.figure(figsize=(6, 6))
plt.scatter(p_mean, y, alpha=0.5)
plt.xlabel("Probabilidade prevista")
plt.ylabel("Resposta observada")
plt.title("Previsão vs Observado")
plt.grid(True)
plt.show()
