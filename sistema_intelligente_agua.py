import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import bambi as bmb
from IPython.core.pylabtools import figsize
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# Configuração do tamanho da figura
figsize(10, 10)
sns.set_theme()

# Carregar o arquivo CSV
df = pd.read_csv(
    "base_milho.csv"
)

# Verificando as primeiras linhas dos dados
print(df.head(10))

# Acessando as colunas de interesse
volume, produtividade = df["volume"].values, df["produtividade"].values

# Ajuste do modelo de regressão linear
linear_model = LinearRegression().fit(volume.reshape(-1, 1), produtividade)
predict_line = linear_model.predict(volume.reshape(-1, 1))

# Coeficiente e intercepto do modelo
print("Coeficiente:", linear_model.coef_)
print("Intercepto:", linear_model.intercept_)

# Visualização do gráfico de dispersão com a linha de regressão
plt.scatter(volume, produtividade, label="sampled data")
plt.plot(volume, predict_line, label="regression line", lw=2.0, color="red")
plt.xlabel("Volume m3")
plt.ylabel("Produtividade toneladas por hectare")
plt.legend(loc=0)
plt.show()  # Exibe o gráfico gerado

# Modelagem com Bambi (Bayesian)
gauss_model = bmb.Model("produtividade ~ volume", data=df)

# Ajuste do modelo usando 1000 amostras em 4 cadeias
gauss_fitted = gauss_model.fit(draws=1000, chains=4)

# Previsões do modelo bayesiano (removendo o parâmetro 'draws')
gauss_model.predict(gauss_fitted, kind="response")

# Plotando os traços das distribuições a posteriori com ArviZ
az.plot_trace(gauss_fitted, var_names=["Intercept", "volume"], figsize=(20, 10))
plt.tight_layout()
plt.show()  # Exibe o gráfico de trace

# Visualização das distribuições de probabilidade normal
nor = stats.norm
x = np.linspace(-8, 7, 150)
mu = [0.43] * 3
sigma = [1, 2, 3]
colors = ["#348ABD", "#A60628", "#7A68A6"]
parameters = zip(mu, sigma, colors)

for _mu, _sigma, _color in parameters:
    plt.plot(x, nor.pdf(x, _mu, scale=_sigma), label="$\\sigma = %.1f$" % (_sigma), color=_color)
    plt.fill_between(x, nor.pdf(x, _mu, scale=_sigma), color=_color, alpha=0.33)

plt.legend(loc="upper right")
plt.xlabel("$x$")
plt.ylabel("Density function at $x$")
plt.title("Probability distribution of three different Normal random variables")
plt.show()  # Exibe o gráfico de distribuição normal

# Exibindo predições e intervalos de credibilidade
az.plot_ppc(gauss_fitted)
plt.show()  # Exibe o gráfico de PPC (posterior predictive check)

# Valores médios a posteriori
print("Média do coeficiente 'volume':", gauss_fitted.posterior.volume.values.mean())
print("Média do intercepto:", gauss_fitted.posterior.Intercept.values.mean())

# Exibindo o diagnóstico de R-hat
rhat_summary = az.summary(gauss_fitted, hdi_prob=0.95)
print(rhat_summary)  # Exibe o resumo com o R-hat (Gelman-Rubin statistic)

# Plotando o R-hat com gráfico de barras
rhat_values = rhat_summary["r_hat"]
plt.bar(rhat_values.index, rhat_values)
plt.axhline(y=1, color='r', linestyle='--', label='Ideal r_hat = 1')
plt.xlabel("Parameters")
plt.ylabel("$\\hat{R}$")
plt.title("Gelman-Rubin Diagnostic (R-hat)")
plt.legend()
plt.show()  # Exibe o gráfico de R-hat

# Função para plotar a previsão baseada na amostra
def plot_sample_prediction(volume, produtividade, fitted):
    # Plotando os dados
    plt.scatter(volume, produtividade, label="Data")

    x_range = np.linspace(min(volume), max(volume), 2000)

    y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()

    # Plotando a previsão média
    plt.plot(x_range, y_pred, color="black", label="Mean regression line", lw=2)

    # Plotando as amostras de previsão
    for i in range(10):
        y_pred = fitted.posterior.volume.values[0, i] * x_range + fitted.posterior.Intercept.values[0, i]
        plt.plot(x_range, y_pred, color="green", linestyle="-.", label="Sample regression line", alpha=0.5)

    plt.xlabel("Volume score")
    plt.ylabel("Produtividade score")
    plt.legend(loc=0)
    plt.show()  # Exibe o gráfico de previsão amostral

# Chamada da função para plotar a amostra de predições
plot_sample_prediction(volume, produtividade, gauss_fitted)

# Função para plotar os intervalos de previsão
def plot_interval_prediction(volume, produtividade, fitted):
    # Plotando os dados
    plt.scatter(volume, produtividade, label="Data")

    x_range = np.linspace(min(volume), max(volume), 2000)
    y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()

    # Plotando a previsão média
    plt.plot(x_range, y_pred, color="black", label="Mean regression line")

    # Plotando os intervalos de credibilidade (HDIs)
    for interval in [0.38, 0.68]:
        az.plot_hdi(volume, fitted.posterior_predictive.produtividade, hdi_prob=interval, color="firebrick")

    plt.xlabel("Volume score")
    plt.ylabel("Produtividade score")
    plt.legend(loc=0)
    plt.show()  # Exibe o gráfico de intervalos de predição

# Chamada da função para plotar os intervalos de predição
plot_interval_prediction(volume, produtividade, gauss_fitted)
