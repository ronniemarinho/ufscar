import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import bambi as bmb
import scipy.stats as stats
from IPython.core.pylabtools import figsize

# Configuração do ambiente visual
figsize(10, 10)
sns.set_theme()

# Carregar a base de dados
df = pd.read_csv("base_milho.csv")
#df = pd.read_csv("dados_milho.csv")


# Verificando as primeiras linhas
print(df.head(10))

# Acessando as colunas de interesse
volume = df["volume"].values
produtividade = df["produtividade"].values

# -------------------------------
# MODELO BAYESIANO COM BAMBI
# -------------------------------
modelo = bmb.Model("produtividade ~ volume",
                        data=df,
                        priors={
                            "Intercept": bmb.Prior("Normal", mu=4, sigma=2),
                            "volume": bmb.Prior("Normal", mu=0.004, sigma=0.002)
                        }
                        )

# Ajuste do modelo usando 1000 amostras em 4 cadeias
modelo_ajustado = modelo.fit(draws=1000,
                               chains=4,
                               random_seed=0
                            )
modelo.plot_priors()
plt.show()
# Fazendo previsões com o modelo bayesiano
modelo.predict(modelo_ajustado, kind="response")


# -------------------------------
# GRÁFICOS DE ANÁLISE BAYESIANA
# -------------------------------

# Traços das distribuições a posteriori
az.plot_trace(modelo_ajustado, var_names=["Intercept", "volume"], figsize=(20, 10))
plt.suptitle("Distribuição a posteriore dos parametros", fontsize=16)
plt.tight_layout()
plt.show()

# Definindo os priors (os mesmos usados no modelo)
from scipy.stats import norm

priors = {
    "Intercept": {"mu": 4, "sigma": 2, "color": "blue"},
    "volume": {"mu": 0.004, "sigma": 0.002, "color": "green"}
}

# Configurando a figura
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

for i, param in enumerate(priors.keys()):
    prior = priors[param]
    mu = prior["mu"]
    sigma = prior["sigma"]
    color = prior["color"]

    # Prior
    x_prior = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y_prior = norm.pdf(x_prior, loc=mu, scale=sigma)
    axs[i].plot(x_prior, y_prior, label="Prior", color=color, linestyle="--")
    axs[i].fill_between(x_prior, y_prior, color=color, alpha=0.3)

    # Posterior (amostras do modelo ajustado)
    posterior_samples = modelo_ajustado.posterior[param].values.flatten()
    sns.kdeplot(posterior_samples, ax=axs[i], label="Posterior", color="black", linewidth=2)

    axs[i].set_title(f"{param}: Prior vs Posterior")
    axs[i].set_xlabel("Valor")
    axs[i].set_ylabel("Densidade")
    axs[i].legend()

plt.suptitle("Comparação entre Priors e Posteriors", fontsize=16)
plt.tight_layout()
plt.show()



# Exibindo predições e intervalos de credibilidade
ax = az.plot_ppc(modelo_ajustado)

# Captura dos handles (desenhos) e labels originais
handles, labels = ax.get_legend_handles_labels()

# Personalizando a legenda
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
print("Média do coeficiente 'volume':", modelo_ajustado.posterior.volume.values.mean())
print("Média do intercepto:", modelo_ajustado.posterior.Intercept.values.mean())

# Exibindo o diagnóstico de R-hat
rhat_summary = az.summary(modelo_ajustado, hdi_prob=0.95)
print(rhat_summary)


"""
# Plotando o R-hat
rhat_values = rhat_summary["r_hat"]
plt.bar(rhat_values.index, rhat_values)
plt.axhline(y=1, color='r', linestyle='--', label='Ideal r_hat = 1')
plt.xlabel("Parâmetros")
plt.ylabel("$\\hat{R}$")
plt.title("Diagnóstico de convergencia (R-hat)")
plt.legend()
plt.show()

"""

# -------------------------------
# FUNÇÃO PARA PLOTAR AMOSTRAS DE PREVISÃO
# -------------------------------
def plot_sample_prediction(volume, produtividade, fitted):
    plt.scatter(volume, produtividade, label="Dados reais")

    x_range = np.linspace(min(volume), max(volume), 2000)
    y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()

    # Linha de regressão média
    plt.plot(x_range, y_pred, color="black", label="Linha média (posterior)", lw=2)

    # Algumas amostras da distribuição posterior
    for i in range(10):
        label = "Amostras da posterior" if i == 0 else None
        y_pred_sample = fitted.posterior.volume.values[0, i] * x_range + fitted.posterior.Intercept.values[0, i]
        plt.plot(x_range, y_pred_sample, color="green", linestyle="-.", alpha=0.5, label=label)

    plt.xlabel("Volume de água aplicado (m³)")
    plt.ylabel("Produtividade (toneladas por hectare)")
    plt.title("Previsão amostral com base no modelo bayesiano")
    plt.legend(loc="best")
    plt.show()

plot_sample_prediction(volume, produtividade, modelo_ajustado)

# -------------------------------
# FUNÇÃO PARA PLOTAR INTERVALOS DE PREVISÃO
# -------------------------------
def plot_interval_prediction(volume, produtividade, fitted, hdi_inner=0.50, hdi_outer=0.94):
    plt.scatter(volume, produtividade, label="Dados reais")

    x_range = np.linspace(min(volume), max(volume), 2000)
    y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()

    # Linha de regressão média
    plt.plot(x_range, y_pred, color="black", label="Linha média (posterior)", lw=2)

    # Intervalos de credibilidade (HDI)
    az.plot_hdi(volume, fitted.posterior_predictive.produtividade, hdi_prob=hdi_inner,
                color="firebrick", fill_kwargs={"alpha": 0.6})
    az.plot_hdi(volume, fitted.posterior_predictive.produtividade, hdi_prob=hdi_outer,
                color="firebrick", fill_kwargs={"alpha": 0.3})

    # Legenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color="black", lw=2, label="Linha média (posterior)"),
        Patch(facecolor="firebrick", alpha=0.6, label=f"Intervalo de {int(hdi_inner*100)}% de credibilidade"),
        Patch(facecolor="firebrick", alpha=0.3, label=f"Intervalo de {int(hdi_outer*100)}% de credibilidade"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="Dados reais", markersize=8)
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.xlabel("Volume de água (m³)")
    plt.ylabel("Produtividade (toneladas por hectare)")
    plt.title("Intervalos de credibilidade para as previsões")
    plt.tight_layout()
    plt.show()

plot_interval_prediction(volume, produtividade, modelo_ajustado, hdi_inner=0.50, hdi_outer=0.94)
