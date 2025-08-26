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

# Carregar a base de dados de abacates
df = pd.read_csv("avocado.csv")

# Verificando as primeiras linhas para entender a estrutura da base
print(df.head(10))

# Acessando as colunas de interesse para a regressão
# Vamos prever o preço médio com base no total de volume
volume = df["Total_Volume"].values  # Usando Total_Volume como variável preditora
produtividade = df["AveragePrice"].values  # Usando AveragePrice como variável dependente

# -------------------------------
# MODELO BAYESIANO COM BAMBI
# -------------------------------
modelo = bmb.Model("AveragePrice ~ Total_Volume + (1|type)",
                        data=df,
                        priors={
                            "Intercept": bmb.Prior("Normal", mu=4, sigma=2),
                            "Total_Volume": bmb.Prior("Normal", mu=0.004, sigma=0.002)
                        }
                        )

# Ajuste do modelo usando 1000 amostras em 4 cadeias
modelo_ajustado = modelo.fit(draws=3000,
                             tune=3000,
                            target_accept=0.95,
                               chains=4,
                               random_seed=0
                            )

# Fazendo previsões com o modelo bayesiano
modelo.predict(modelo_ajustado, kind="response")


# -------------------------------
# GRÁFICOS DE ANÁLISE BAYESIANA
# -------------------------------

# Traços das distribuições a posteriori
az.plot_trace(modelo_ajustado, var_names=["Intercept", "Total_Volume"], figsize=(20, 10))
plt.suptitle("Distribuição a posteriori dos parâmetros", fontsize=16)
plt.tight_layout()
plt.show()

# Visualização das distribuições de probabilidade normal
nor = stats.norm
x = np.linspace(-8, 7, 150)
mu = [0.43] * 3
sigma = [1, 2, 3]
colors = ["#348ABD", "#A60628", "#7A68A6"]

for _mu, _sigma, _color in zip(mu, sigma, colors):
    plt.plot(x, nor.pdf(x, _mu, scale=_sigma), label=f"$\\sigma = {_sigma:.1f}$", color=_color)
    plt.fill_between(x, nor.pdf(x, _mu, scale=_sigma), color=_color, alpha=0.33)

plt.legend(title="Desvio padrão", loc="upper right")
plt.xlabel("Valores de x")
plt.ylabel("Função de densidade no ponto x")
plt.title("Distribuição de probabilidade Normal")
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
print("Média do coeficiente 'Total_Volume':", modelo_ajustado.posterior.Total_Volume.values.mean())
print("Média do intercepto:", modelo_ajustado.posterior.Intercept.values.mean())

# Exibindo o diagnóstico de R-hat
rhat_summary = az.summary(modelo_ajustado, hdi_prob=0.95)
print(rhat_summary)

# Plotando o R-hat
rhat_values = rhat_summary["r_hat"]
plt.bar(rhat_values.index, rhat_values)
plt.axhline(y=1, color='r', linestyle='--', label='Ideal r_hat = 1')
plt.xlabel("Parâmetros")
plt.ylabel("$\\hat{R}$")
plt.title("Diagnóstico de convergência (R-hat)")
plt.legend()
plt.show()

# -------------------------------
# FUNÇÃO PARA PLOTAR AMOSTRAS DE PREVISÃO
# -------------------------------
def plot_sample_prediction(volume, produtividade, fitted):
    plt.scatter(volume, produtividade, label="Dados reais")

    x_range = np.linspace(min(volume), max(volume), 2000)
    y_pred = fitted.posterior.Total_Volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()

    # Linha de regressão média
    plt.plot(x_range, y_pred, color="black", label="Linha média (posterior)", lw=2)

    # Algumas amostras da distribuição posterior
    for i in range(10):
        label = "Amostras da posterior" if i == 0 else None
        y_pred_sample = fitted.posterior.Total_Volume.values[0, i] * x_range + fitted.posterior.Intercept.values[0, i]
        plt.plot(x_range, y_pred_sample, color="green", linestyle="-.", alpha=0.5, label=label)

    plt.xlabel("Total_Volume de abacates (m³)")
    plt.ylabel("Preço Médio do Abacate ($)")
    plt.title("Previsão amostral com base no modelo bayesiano")
    plt.legend(loc="best")
    plt.show()

plot_sample_prediction(volume, produtividade, modelo_ajustado)

# -------------------------------
# FUNÇÃO PARA PLOTAR INTERVALOS DE PREVISÃO
# -------------------------------
def plot_interval_prediction(volume, produtividade, fitted):
    plt.scatter(volume, produtividade, label="Dados reais")

    x_range = np.linspace(min(volume), max(volume), 2000)
    y_pred = fitted.posterior.Total_Volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()

    # Linha de regressão média
    plt.plot(x_range, y_pred, color="black", label="Linha média (posterior)")

    # Plotando intervalos de credibilidade (HDIs)

    hdi_38 = az.plot_hdi(volume, fitted.posterior_predictive["AveragePrice"], hdi_prob=0.38, color="firebrick",
                         fill_kwargs={"alpha": 0.6})
    hdi_68 = az.plot_hdi(volume, fitted.posterior_predictive["AveragePrice"], hdi_prob=0.68, color="firebrick",
                         fill_kwargs={"alpha": 0.3})

    # Criando legenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color="black", lw=2, label="Linha média (posterior)"),
        Patch(facecolor="firebrick", alpha=0.6, label="Intervalo de 38% de credibilidade"),
        Patch(facecolor="firebrick", alpha=0.3, label="Intervalo de 68% de credibilidade"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="Dados reais", markersize=8)
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.xlabel("Volume de abacates vendidos (m³)")
    plt.ylabel("Preço Médio ($)")
    plt.title("Intervalo de credibilidade para as previsões")
    plt.show()

plot_interval_prediction(volume, produtividade, modelo_ajustado)
