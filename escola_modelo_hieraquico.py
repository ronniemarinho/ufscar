import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# ----------------------------
# Dados do exemplo 8 escolas
# ----------------------------
J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])  # estimativas
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])  # erros padrão
schools = [
    "A", "B", "C", "D", "E", "F", "G", "H"
]

# ----------------------------
# Modelo hierárquico reparametrizado
# ----------------------------
with pm.Model() as model:

    # Hiperparâmetros
    mu = pm.Normal("mu", mu=0, sigma=10)
    tau = pm.HalfNormal("tau", sigma=10)

    # Reparametrização não centrada:
    z = pm.Normal("z", mu=0, sigma=1, shape=J)
    theta = pm.Deterministic("theta", mu + z * tau)

    # Verossimilhança
    pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    # Amostragem com target_accept maior
    idata = pm.sample(
        draws=2000, tune=2000, chains=4,
        target_accept=0.95, random_seed=42
    )

# ----------------------------
# Resumo estatístico
# ----------------------------
print(az.summary(idata, var_names=["mu", "tau", "theta"], round_to=2))

# ----------------------------
# Gráficos de comparação
# ----------------------------

# Sem pooling (estimativas individuais)
theta_no_pooling = [
    np.random.normal(loc=mean, scale=sd, size=1000)
    for mean, sd in zip(y, sigma)
]

plt.figure(figsize=(10, 4))
plt.boxplot(theta_no_pooling, tick_labels=schools)
plt.title("Sem Pooling")
plt.grid(True)
plt.show()

# Com pooling completo (mesmo valor para todas as escolas)
theta_complete_pooling = [
    np.random.normal(loc=np.mean(y), scale=np.mean(sigma), size=1000)
    for _ in range(J)
]

plt.figure(figsize=(10, 4))
plt.boxplot(theta_complete_pooling, tick_labels=schools)
plt.title("Pooling Completo")
plt.grid(True)
plt.show()

# Pooling parcial (modelo hierárquico)
theta_posterior = idata.posterior["theta"].stack(sample=("chain", "draw")).values.T

plt.figure(figsize=(10, 4))
plt.boxplot(theta_posterior, tick_labels=schools)
plt.title("Pooling Parcial (Hierárquico)")
plt.grid(True)
plt.show()
