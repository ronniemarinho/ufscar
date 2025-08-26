"""
Regressão Bayesiana didática sobre dados 'avocado'
Etapas marcadas: parâmetros -> priors -> likelihood -> inferência -> posterior preditivo
Requer: pymc (v5.x), arviz, pandas, numpy, matplotlib
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ============================================================
# 0) PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================
df = pd.read_csv("avocado.csv")

# Renomear colunas para português
df = df.rename(columns={
    "AveragePrice": "preco_medio",
    "Total Volume": "volume_total",
    "Total Bags": "sacos_total",
    "Small Bags": "sacos_pequenos",
    "Large Bags": "sacos_grandes",
    "XLarge Bags": "sacos_xlarge",
    "type": "tipo",
    "region": "regiao",
    "Date": "data",
    "year": "ano"
})

# Criar variável binária para tipo orgânico
df["tipo_organico"] = (df["tipo"].str.lower() == "organic").astype(int)

# Variável alvo
y = df["volume_total"].values

# Preditores
X_raw = df[["preco_medio", "tipo_organico"]].copy()

# Padronizar variável contínua
X_raw["preco_medio_z"] = (X_raw["preco_medio"] - X_raw["preco_medio"].mean()) / X_raw["preco_medio"].std()
X = X_raw[["preco_medio_z", "tipo_organico"]].values

N = len(y)
P = X.shape[1]
print(f"Observações: {N}, Preditores: {P}")

preco_mean = X_raw["preco_medio"].mean()
preco_std = X_raw["preco_medio"].std()

# ============================================================
# 1) DEFINIR PARÂMETROS DESCONHECIDOS
# ============================================================
# β₀  → intercepto
# β   → coeficientes das variáveis
# σ   → desvio padrão do erro

with pm.Model() as model:

    # ========================================================
    # 2) DEFINIR PRIORS P(θ)
    # ========================================================
    beta0 = pm.Normal("beta0", mu=0, sigma=1e3)           # prior largo para intercepto
    beta = pm.Normal("beta", mu=0, sigma=10, shape=P)     # priors moderados para coeficientes
    sigma = pm.HalfNormal("sigma", sigma=1e4)             # prior positivo para σ

    # ========================================================
    # 3) DEFINIR VEROSSIMILHANÇA P(y | θ)
    # ========================================================
    mu = beta0 + pm.math.dot(X, beta)                     # predição média
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # ========================================================
    # 4) INFERÊNCIA (amostragem da posterior)
    # ========================================================
    idata = pm.sample(draws=2000, tune=1000, chains=4,
                      random_seed=42, target_accept=0.9)

# ============================================================
# 5) INTERPRETAÇÃO / DIAGNÓSTICO
# ============================================================
print(az.summary(idata, var_names=["beta0", "beta", "sigma"], round_to=3))
az.plot_trace(idata, var_names=["beta0", "beta", "sigma"])
plt.tight_layout()
plt.show()

# ============================================================
# 6) POSTERIOR PREDITIVO — PREVISÃO PARA NOVOS DADOS
# ============================================================
novos_precos = np.array([0.5, 0.75, 1.0, 1.25])
tipo_organico = 1
novos_precos_z = (novos_precos - preco_mean) / preco_std
X_new = np.column_stack([novos_precos_z, np.repeat(tipo_organico, len(novos_precos_z))])

posterior = idata.posterior
beta0_samps = posterior["beta0"].stack(sample=("chain", "draw")).values
beta_samps = posterior["beta"].stack(sample=("chain", "draw")).values.T
sigma_samps = posterior["sigma"].stack(sample=("chain", "draw")).values

S = beta0_samps.shape[0]
mu_all = beta0_samps[:, None] + (beta_samps @ X_new.T)

rng = np.random.default_rng(42)
y_rep = rng.normal(loc=mu_all, scale=sigma_samps[:, None])

y_mean = y_rep.mean(axis=0)
y_hpd_lower = np.percentile(y_rep, 2.5, axis=0)
y_hpd_upper = np.percentile(y_rep, 97.5, axis=0)

for i, price in enumerate(novos_precos):
    print(f"Preço {price:0.2f} -> média {y_mean[i]:.1f}, 95% CI [{y_hpd_lower[i]:.1f}, {y_hpd_upper[i]:.1f}]")

# ============================================================
# 7) ANÁLISE DE LUCRO — EXEMPLO
# ============================================================
predicted_profit = {}
for i, price in enumerate(novos_precos):
    vol_samples = y_rep[:, i]
    profit_samples = price * vol_samples
    predicted_profit[price] = profit_samples
# Plot simples: histogramas das distribuições de lucro
fig, axes = plt.subplots(1, len(novos_precos), figsize=(16, 3))
for ax, p in zip(axes, novos_precos):
    ax.hist(predicted_profit[p], bins=50, alpha=0.7)
    ax.set_title(f"Lucro p={p:.2f}")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Intervalo de credibilidade 99% para preço 0.75
p_choice = 0.75
hpd_99 = az.hdi(predicted_profit[p_choice], hdi_prob=0.99)
print(f"IC 99% lucro (preço {p_choice}): {hpd_99}")
# ---------------------------------------------------------

# 5) Comentários finais (impressos para uso didático)
# ---------------------------------------------------------
print("\n=== Mapeamento didático das etapas ===")
print("Parâmetros (θ): beta0 (intercepto), beta (coeficientes), sigma (erro).")
print("Priors: especificados explicitamente acima como Normal/HalfNormal.")
print("Likelihood: y ~ Normal(mu, sigma) com mu = beta0 + X @ beta.")
print("Inferência: MCMC (NUTS) via pm.sample -> idata contém posterior.")
print("Previsão: construímos mu para novos X e amostramos Normal(mu, sigma) para obter posterior preditivo.")
