# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import bambi as bmb
import scipy.stats as stats
from IPython.core.pylabtools import figsize

# -------------------------------
# CONFIGURAÇÃO VISUAL
# -------------------------------
figsize(10, 10)
sns.set_theme()

# -------------------------------
# CARREGAR A BASE DE DADOS
# -------------------------------
df = pd.read_csv("organic_abacate.csv")

print(df.head(10))
print(df.columns)

# -------------------------------
# PRÉ-PROCESSAMENTO
# -------------------------------
df["regiao"] = df["regiao"].astype("category")
df["ano"] = df["ano"].astype(int)

df["volume_std"] = (df["volume_total"] - df["volume_total"].mean()) / df["volume_total"].std()

volume = df["volume_total"].values
preco = df["preco_medio"].values

# -------------------------------
# MODELO BAYESIANO HIERÁRQUICO COM BAMBI
# -------------------------------
modelo = bmb.Model(
    "preco_medio ~ volume_std + (volume_std|regiao)",
    data=df,
    priors={
        "Intercept": bmb.Prior("Normal", mu=1.5, sigma=0.5),
        "volume_std": bmb.Prior("Normal", mu=0.0, sigma=0.2),
    }
)

modelo_ajustado = modelo.fit(
    draws=100,
    tune=500,
    chains=4,
    cores=4,
    target_accept=0.99,
    max_treedepth=15,
    random_seed=42
)

# PREDIÇÕES usando o método correto do bambi
# (substitui o uso do `posterior_predict` que dá erro)
predictions = modelo.predict(modelo_ajustado, kind="response", inplace=True)

print(modelo_ajustado.posterior_predictive)

# -------------------------------
# GRÁFICOS DE ANÁLISE BAYESIANA
# -------------------------------
az.plot_trace(modelo_ajustado, var_names=["Intercept", "volume_std"], figsize=(20, 10))
plt.suptitle("Distribuição a posteriore dos parâmetros (fixos)", fontsize=16)
plt.tight_layout()
plt.show()

from scipy.stats import norm
priors = {
    "Intercept": {"mu": 1.5, "sigma": 0.5, "color": "blue"},
    "volume_std": {"mu": 0.0, "sigma": 0.2, "color": "green"}
}

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for i, param in enumerate(priors.keys()):
    prior = priors[param]
    mu, sigma, color = prior["mu"], prior["sigma"], prior["color"]

    x_prior = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y_prior = norm.pdf(x_prior, loc=mu, scale=sigma)
    axs[i].plot(x_prior, y_prior, label="Prior", color=color, linestyle="--")
    axs[i].fill_between(x_prior, y_prior, color=color, alpha=0.3)

    posterior_samples = modelo_ajustado.posterior[param].values.flatten()
    sns.kdeplot(posterior_samples, ax=axs[i], label="Posterior", color="black", linewidth=2)

    axs[i].set_title(f"{param}: Prior vs Posterior")
    axs[i].set_xlabel("Valor")
    axs[i].set_ylabel("Densidade")
    axs[i].legend()

plt.suptitle("Comparação entre Priors e Posteriors (efeitos fixos)", fontsize=16)
plt.tight_layout()
plt.show()

ax = az.plot_ppc(modelo_ajustado)
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    handles,
    ["Previsões Posteriores", "Observado", "Média da previsão posterior"],
    loc="upper right"
)
plt.suptitle("Verificação preditiva Posterior", fontsize=16)
plt.show()

# -------------------------------
# ANÁLISE NUMÉRICA DOS PARÂMETROS
# -------------------------------
print("Média do coeficiente (fixo) 'volume_std':", modelo_ajustado.posterior["volume_std"].values.mean())
print("Média do intercepto (fixo):", modelo_ajustado.posterior["Intercept"].values.mean())

rhat_summary = az.summary(modelo_ajustado, hdi_prob=0.95)
print(rhat_summary)

modelo.plot_priors()
plt.show()

# -------------------------------
# FUNÇÕES DE PLOT PERSONALIZADAS
# -------------------------------
def plot_sample_prediction(volume_raw, preco, idata, model, regiao=None, n_samples=10):
    plt.scatter(volume_raw, preco, label="Dados reais")

    x_range = np.linspace(min(volume_raw), max(volume_raw), 2000)
    x_std = (x_range - df["volume_total"].mean()) / df["volume_total"].std()

    if regiao is not None:
        new_data = pd.DataFrame({"volume_std": x_std, "regiao": regiao})
    else:
        reg_ref = df["regiao"].mode().iloc[0]
        new_data = pd.DataFrame({"volume_std": x_std, "regiao": reg_ref})

    # Predição média (sem chain e draw)
    mean_pred = model.predict(idata, data=new_data, kind="response")
    plt.plot(x_range, mean_pred, label="Linha média (posterior)", linewidth=2)

    # Predição amostral posterior predictive (com chain e draw)
    pred_samples = model.predict(idata, data=new_data, kind="posterior_predictive")

    # Extrai array numpy: dims [chain, draw, obs]
    samples_array = pred_samples["preco_medio"].values  # shape: (chains, draws, n_obs)

    # Para plotar várias amostras, achatamos chain e draw juntos
    samples_2d = samples_array.reshape(-1, samples_array.shape[-1])  # (chains*draws, n_obs)

    for i in range(min(n_samples, samples_2d.shape[0])):
        label = "Amostras da posterior" if i == 0 else None
        plt.plot(x_range, samples_2d[i, :], linestyle="-.", alpha=0.4, label=label)

    plt.xlabel("Volume total")
    plt.ylabel("Preço médio")
    title = "Previsão amostral (marginal)" if regiao is None else f"Previsão amostral – Região: {regiao}"
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def plot_interval_prediction(volume_raw, preco, idata, model, regiao=None, hdi_inner=0.50, hdi_outer=0.95):
    plt.scatter(volume_raw, preco, label="Dados reais")

    x_range = np.linspace(min(volume_raw), max(volume_raw), 2000)
    x_std = (x_range - df["volume_total"].mean()) / df["volume_total"].std()

    if regiao is not None:
        new_data = pd.DataFrame({"volume_std": x_std, "regiao": regiao})
    else:
        reg_ref = df["regiao"].mode().iloc[0]
        new_data = pd.DataFrame({"volume_std": x_std, "regiao": reg_ref})

    mean_pred = model.predict(idata, data=new_data, kind="mean")
    plt.plot(x_range, mean_pred, linewidth=2, label="Linha média (posterior)")

    ppc = model.predict(idata, data=new_data, kind="pps", inplace=False)

    if "preco_medio" in ppc:
        mat = np.vstack(ppc["preco_medio"].values)
        az.plot_hdi(x_range, mat, hdi_prob=hdi_outer)
        az.plot_hdi(x_range, mat, hdi_prob=hdi_inner)

    plt.legend(loc="best")
    plt.xlabel("Volume total")
    plt.ylabel("Preço médio")
    title = "Intervalos de credibilidade (marginal)" if regiao is None else f"Intervalos de credibilidade – Região: {regiao}"
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -------------------------------
# PLOTS PERSONALIZADOS (EXEMPLOS)
# -------------------------------
plot_sample_prediction(volume, preco, modelo_ajustado, modelo, regiao=None, n_samples=10)

uma_regiao = str(df["regiao"].unique()[0])
plot_sample_prediction(volume, preco, modelo_ajustado, modelo, regiao=uma_regiao, n_samples=10)

plot_interval_prediction(volume, preco, modelo_ajustado, modelo, regiao=None, hdi_inner=0.50, hdi_outer=0.95)
plot_interval_prediction(volume, preco, modelo_ajustado, modelo, regiao=uma_regiao, hdi_inner=0.50, hdi_outer=0.95)
