import pandas as pd
import seaborn as sns
import bambi as bmb
import arviz as az

# Carregar o arquivo CSV usando pandas
dados = pd.read_csv("iris.csv")

# Exibir as primeiras linhas para verificar se o arquivo foi carregado corretamente
print(dados.head())

# Modelo Bayesiano (logística multinomial)
modelo = bmb.Model(
    "species ~ sepal_length + sepal_width + petal_length + petal_width",
    dados, family="categorical"
)

# Ajuste do modelo
ajuste = modelo.fit(draws=2000, tune=1000, chains=4, nuts={"target_accept": 0.95})

# Resumo dos parâmetros
print(az.summary(ajuste, hdi_prob=0.95))

# Ver predições em novos dados
novos = pd.DataFrame({
    "sepal_length": [5.0, 6.5],
    "sepal_width": [3.4, 2.8],
    "petal_length": [1.5, 5.5],
    "petal_width": [0.2, 2.0]
})
pred_probs = idata.posterior_predictive['response']  # ou o nome certo do response

pred_mean = pred_probs.mean(dim=("chain", "draw"))


# Converter em DataFrame legível
pred_df = pd.DataFrame(pred_mean.values, columns=modelo.response.name)
pred_df.index = range(1, len(pred_df)+1)

print("\nProbabilidades previstas para cada nova observação:")
print(pred_df)
