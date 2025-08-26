import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# -----------------------
# 1) Dados observados
# -----------------------
# Profundidades medidas
depth_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Detecção: 1 = detectou, 0 = não detectou
# (exemplo com dados binários)
detected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

print("Versão do PyMC:", pm.__version__)

# -----------------------
# 2) Modelo Bayesiano
# -----------------------
with pm.Model() as model:
    # Priors para intercepto (alpha) e inclinação (beta)
    alpha = pm.Normal("alpha", mu=0, sigma=1.5)
    beta = pm.Normal("beta", mu=0, sigma=1.5)

    # Preditor linear
    mu = alpha + beta * depth_data

    # Transformação logística para probabilidade
    p = pm.Deterministic("p", pm.math.sigmoid(mu))

    # Verossimilhança (dados binários ~ Bernoulli)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=detected)

    # Amostragem com NUTS
    trace = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42)

# -----------------------
# 3) Resumo dos parâmetros
# -----------------------
print(az.summary(trace, var_names=["alpha", "beta"]))

# -----------------------
# 4) Predições para gráfico
# -----------------------
depth_pred = np.linspace(1, 9, 100)
with model:
    # Amostra predições
    posterior_pred = pm.sample_posterior_predictive(trace, var_names=["p"])

# Extrai p para cada profundidade prevista
alpha_samples = trace.posterior["alpha"].values.flatten()
beta_samples = trace.posterior["beta"].values.flatten()

p_pred_samples = []
for a, b in zip(alpha_samples, beta_samples):
    p_pred_samples.append(1 / (1 + np.exp(-(a + b * depth_pred))))
p_pred_samples = np.array(p_pred_samples)

# Média e intervalo de credibilidade 95%
p_mean = p_pred_samples.mean(axis=0)
p_lower = np.percentile(p_pred_samples, 2.5, axis=0)
p_upper = np.percentile(p_pred_samples, 97.5, axis=0)

# -----------------------
# 5) Gráfico
# -----------------------
plt.figure(figsize=(8, 5))
plt.scatter(depth_data, detected, color="black", label="Dados observados")
plt.plot(depth_pred, p_mean, color="blue", label="Média predita")
plt.fill_between(depth_pred, p_lower, p_upper, color="lightblue", alpha=0.5, label="95% IC")
plt.xlabel("Profundidade")
plt.ylabel("Probabilidade de detecção")
plt.title("Regressão Logística Bayesiana com PyMC 5")
plt.legend()
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.show()
