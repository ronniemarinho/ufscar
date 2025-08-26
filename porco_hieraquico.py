import arviz as az
import bambi as bmb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

az.style.use("arviz-darkgrid")
SEED = 7355608
np.random.seed(SEED)

# Carregar o dataset a partir do CSV
data = pd.read_csv("dietox.csv")
print(data.describe())

import bambi as bmb

priors = {
    "Intercept": bmb.Prior("Normal", mu=20, sigma=5),  # Peso inicial: 20kg ± 5kg
    "Time": bmb.Prior("Normal", mu=0.5, sigma=0.2)      # Ganho de peso: 0.5kg/dia ± 0.2
}

model = bmb.Model("Weight ~ Time + (Time|Pig)", data, priors=priors)


results = model.fit()

print(model)

model.plot_priors()
plt.show()

# Plot dos traços posteriores
az.plot_trace(
    results,
    var_names=["Intercept", "Time", "1|Pig", "Time|Pig", "sigma"],
    compact=True,
)
plt.show()

print(az.summary(results, var_names=["Intercept", "Time", "1|Pig_sigma", "Time|Pig_sigma", "sigma"]))

# Dados do primeiro porquinho '4601'
data_0 = data[data["Pig"] == 4601][["Time", "Weight"]]
time = np.array([1, 12])

posterior = az.extract_dataset(results)
intercept_common = posterior["Intercept"]
slope_common = posterior["Time"]

intercept_specific_0 = posterior["1|Pig"].sel(Pig__factor_dim="4601")
slope_specific_0 = posterior["Time|Pig"].sel(Pig__factor_dim="4601")

a = intercept_common + intercept_specific_0
b = slope_common + slope_specific_0

# Tempo como DataArray para broadcasting
time_xi = xr.DataArray(time)

plt.figure()

# Plot de apenas uma curva como exemplo
plt.plot(time_xi, (a + b * time_xi)[0].T, color="C1", lw=0.3, label="Amostras Posteriores")

# Traça o resto sem label para não poluir a legenda
for i in range(1, len(a)):
    plt.plot(time_xi, (a + b * time_xi)[i].T, color="C1", lw=0.3, alpha=0.3)

# Média da posterior
plt.plot(time_xi, a.mean() + b.mean() * time_xi, color="black", lw=2, label="Média Posterior")

# Dados observados
plt.scatter(data_0["Time"], data_0["Weight"], color="red", zorder=2, label="Observações")

plt.ylabel("Peso (kg)")
plt.xlabel("Tempo (semanas)")
plt.title("Peso ao longo do tempo para o Porquinho 4601")
plt.legend()
plt.show()

# Médias do grupo
intercept_group_specific = posterior["1|Pig"]
slope_group_specific = posterior["Time|Pig"]
a = intercept_common.mean() + intercept_group_specific.mean("sample")
b = slope_common.mean() + slope_group_specific.mean("sample")

plt.figure()

# Apenas uma linha com legenda
plt.plot(time_xi, (a + b * time_xi)[0].T, color="C1", alpha=0.7, lw=0.8, label="Amostras Posteriores (Grupo)")

# Demais linhas sem legenda
for i in range(1, len(a)):
    plt.plot(time_xi, (a + b * time_xi)[i].T, color="C1", alpha=0.3, lw=0.8)

plt.ylabel("Peso (kg)")
plt.xlabel("Tempo (semanas)")
plt.title("Peso médio do grupo ao longo do tempo")
plt.legend()
plt.show()

# Gráfico de floresta (Forest plot)
az.plot_forest(
    results,
    var_names=["Intercept", "Time"],
    figsize=(8, 2),
)
plt.show()

# Gráfico da distribuição posterior
az.plot_posterior(results, var_names=["Intercept", "Time"], ref_val=0, rope=[-1, 1])
plt.show()
