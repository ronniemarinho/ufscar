import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# -------------------------------
# 0) Carregar e preparar dados
# -------------------------------
df = pd.read_csv("avocado.csv")
df["type_organic"] = (df["type"].str.lower() == "organic").astype(int)

y = df["Total Volume"].values
X_price = (df["AveragePrice"] - df["AveragePrice"].mean()) / df["AveragePrice"].std()
X_type = df["type_organic"].values
X = np.column_stack([X_price, X_type])  # matriz preditores padronizada

# ======================================================
# 1) Definir parâmetros
# ======================================================
with pm.Model() as model:
    # Parâmetros desconhecidos: beta0 (intercepto), beta (coeficientes), sigma (erro)

    # ======================================================
    # 2) Definir priors
    # ======================================================
    beta0 = pm.Normal("beta0", mu=0, sigma=10)        # intercepto
    beta = pm.Normal("beta", mu=0, sigma=5, shape=2)  # coeficientes
    sigma = pm.HalfNormal("sigma", sigma=10)          # desvio padrão do erro

    # ======================================================
    # 3) Definir likelihood
    # ======================================================
    mu = beta0 + pm.math.dot(X, beta)  # valor esperado
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # ======================================================
    # 4) Inferência
    # ======================================================
    #idata = pm.sample(draws=1000, tune=500, target_accept=0.9, random_seed=42)
    idata = pm.sample(
    draws=1000,
    tune=500,
    target_accept=0.9,
    random_seed=42,
    chains=4,
    cores=1  # força execução single-core e evita EOFError no macOS
    )
# ======================================================
# 5) Avaliação (model checking + comparação)
# ======================================================
az.plot_trace(idata, var_names=["beta0", "beta", "sigma"])
plt.show()

# Posterior Predictive Check
with model:
    ppc = pm.sample_posterior_predictive(idata)
az.plot_ppc(ppc)
plt.show()

# Comparação de modelos (se houvesse mais de um)
# az.loo(idata)  # Exemplo

# ======================================================
# 6) Predição
# ======================================================
# Novos dados para previsão
new_price = np.array([0.5, 0.75, 1.0, 1.25])
new_price_z = (new_price - df["AveragePrice"].mean()) / df["AveragePrice"].std()
new_type = np.ones(len(new_price))  # orgânico
X_new = np.column_stack([new_price_z, new_type])

# Gerar predições
with model:
    post_pred_new = pm.sample_posterior_predictive(
        idata, var_names=["y_obs"],
        random_seed=42,
        predictions={"y_obs": {"mu": beta0 + pm.math.dot(X_new, beta), "sigma": sigma}}
    )

print("Predições concluídas!")
