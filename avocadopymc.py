"""
Regressão Bayesiana didática sobre dados 'avocado'
Mostra explícito: parâmetros -> priors -> likelihood -> inferência -> posterior preditivo
Requer: pymc (v5.x), arviz, pandas, numpy, matplotlib
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0) Carregar dados e preparar variáveis (pré-processamento)
# ---------------------------------------------------------
df = pd.read_csv("avocado.csv")

# Tradução breve (colunas)
# Date, AveragePrice, Total Volume, 4046, 4225, 4770, Total Bags, Small Bags, Large Bags, XLarge Bags, type, year, region
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

# Criar coluna binária para tipo orgânico (1 se organic, 0 se conventional)
df["tipo_organico"] = (df["tipo"].str.lower() == "organic").astype(int)

# Selecionar variáveis de interesse
# Target: volume_total (ou poderia ser preco_medio — aqui seguimos seu exemplo anterior)
y = df["volume_total"].values

# Preditores que vamos usar: preco_medio e tipo_organico
X_raw = df[["preco_medio", "tipo_organico"]].copy()

# Padronizar a variável contínua (média 0, sd 1) — boa prática em modelos Bayesianos
X_raw["preco_medio_z"] = (X_raw["preco_medio"] - X_raw["preco_medio"].mean()) / X_raw["preco_medio"].std()

# Matriz de design (colunas na ordem: preco_z, tipo_organico)
X = X_raw[["preco_medio_z", "tipo_organico"]].values

N = len(y)
P = X.shape[1]
print(f"Observações: {N}, Preditores: {P}")

# Guardar médias/desvios para usar quando prever novos dados
preco_mean = X_raw["preco_medio"].mean()
preco_std = X_raw["preco_medio"].std()

# ---------------------------------------------------------
# 1) ESQUEMA UNIVERSAL (instância) -> Modelagem com PyMC
# ---------------------------------------------------------
# Parâmetros (θ): intercepto (beta0), coeficientes (beta), sigma (erro)
# Priors: escolha de priors fracas/regularizadoras
# Likelihood: y_i ~ Normal(mu_i, sigma), com mu_i = beta0 + X @ beta
# Inferência: MCMC (NUTS)
with pm.Model() as model:
    # ---- Priors (P(θ)) ----
    beta0 = pm.Normal("beta0", mu=0, sigma=1e3)           # intercepto (prior largo)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=P)     # coeficientes (preco_z, tipo_organico)
    sigma = pm.HalfNormal("sigma", sigma=1e4)            # desvio padrão do erro (ajuste grande por escala do volume)

    # ---- Likelihood (P(y | θ)) ----
    mu = beta0 + pm.math.dot(X, beta)                    # média condicional
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # ---- Inferência (amostragem posterior) ----
    # Ajuste: escolha de draws/warmup pode ser alterada; aqui usamos valores modestos para exemplo
    idata = pm.sample(draws=2000, tune=1000, chains=4, random_seed=42, target_accept=0.9)

# ---------------------------------------------------------
# 2) Resultados e diagnóstico
# ---------------------------------------------------------
print("\nResumo (posterior):")
print(az.summary(idata, var_names=["beta0", "beta", "sigma"], round_to=3))

# Traços
az.plot_trace(idata, var_names=["beta0", "beta", "sigma"])
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3) Previsões manuais (posterior predictives) para novos preços
#     — mostramos explicitamente como gerar previsões a partir das amostras
# ---------------------------------------------------------
# Novos preços (exemplo) para prever volume quando tipo = orgânico (1)
novos_precos = np.array([0.5, 0.75, 1.0, 1.25])  # preços originais (não padronizados)
tipo_organico = 1  # fixamos em 1 (orgânico)

# Padronizar usando as mesmas estatísticas do conjunto de treino
novos_precos_z = (novos_precos - preco_mean) / preco_std

# Construir matriz de preditores para predição
X_new = np.column_stack([novos_precos_z, np.repeat(tipo_organico, len(novos_precos_z))])  # shape (n_new, P)

# Extrair amostras da posterior (beta0, beta, sigma)
posterior = idata.posterior
# Stack em shape (n_samples, ...)
beta0_samps = posterior["beta0"].stack(sample=("chain", "draw")).values          # (S,)
beta_samps = posterior["beta"].stack(sample=("chain", "draw")).values.T  # agora (S, P)
sigma_samps = posterior["sigma"].stack(sample=("chain", "draw")).values         # (S,)

# Número de amostras
S = beta0_samps.shape[0]

# Calcular predições preditivas: para cada amostra s e cada novo ponto i, desenhar y_rep ~ Normal(mu_{s,i}, sigma_s)
# Calculamos vetor mu para todos S x n_new: mu = beta0_samps[:, None] + beta_samps @ X_new.T
mu_all = beta0_samps[:, None] + (beta_samps @ X_new.T)   # shape (S, n_new)

# Amostrar preditivo: y_rep ~ Normal(mu_all, sigma_samps[:,None])
rng = np.random.default_rng(42)
y_rep = rng.normal(loc=mu_all, scale=sigma_samps[:, None])  # shape (S, n_new)

# Estatísticas preditivas: média e intervalos credíveis 95%
y_mean = y_rep.mean(axis=0)
y_hpd_lower = np.percentile(y_rep, 2.5, axis=0)
y_hpd_upper = np.percentile(y_rep, 97.5, axis=0)

print("\nPrevisões (volume) para novos preços (tipo = orgânico):")
for i, price in enumerate(novos_precos):
    print(f"Preço {price:0.2f} -> média predita {y_mean[i]:.1f}, 95% CI [{y_hpd_lower[i]:.1f}, {y_hpd_upper[i]:.1f}]")

# ---------------------------------------------------------
# 4) Posterior preditivo para análise de lucro (exemplo)
# ---------------------------------------------------------
predicted_profit = {}
for i, price in enumerate(novos_precos):
    vol_samples = y_rep[:, i]                    # amostras do volume para esse preço
    profit_samples = price * vol_samples         # lucro (preço * volume)
    predicted_profit[price] = profit_samples

# Converter para arviz-friendly dict para plot (cada entrada shape (S,))
az_dict = {f"price_{p:.2f}": predicted_profit[p] for p in novos_precos}
# Plot simples: histogramas das distribuições de lucro
fig, axes = plt.subplots(1, len(novos_precos), figsize=(16, 3))
for ax, p in zip(axes, novos_precos):
    ax.hist(predicted_profit[p], bins=50, alpha=0.7)
    ax.set_title(f"Lucro p={p:.2f}")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Exemplo de intervalo de credibilidade 99% para preço 0.75
p_choice = 0.75
hpd_99 = az.hdi(predicted_profit[p_choice], hdi_prob=0.99)
print(f"\nIntervalo de credibilidade 99% do lucro (preço {p_choice}): {hpd_99}")

# ---------------------------------------------------------
# 5) Comentários finais (impressos para uso didático)
# ---------------------------------------------------------
print("\n=== Mapeamento didático das etapas ===")
print("Parâmetros (θ): beta0 (intercepto), beta (coeficientes), sigma (erro).")
print("Priors: especificados explicitamente acima como Normal/HalfNormal.")
print("Likelihood: y ~ Normal(mu, sigma) com mu = beta0 + X @ beta.")
print("Inferência: MCMC (NUTS) via pm.sample -> idata contém posterior.")
print("Previsão: construímos mu para novos X e amostramos Normal(mu, sigma) para obter posterior preditivo.")
