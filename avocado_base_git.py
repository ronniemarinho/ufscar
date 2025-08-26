import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import bambi as bmb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option("display.max_columns", None)  # não cortar colunas
pd.set_option("display.max_rows", None)     # não cortar linhas
pd.set_option("display.width", None)        # não limitar largura
pd.set_option("display.max_colwidth", None) # mostra tudo de cada célula

# ----------- CARREGAR BASE ----------- #
avocado = pd.read_csv("avocado.csv")

# Criar coluna 'volume' a partir de 'Total Volume'
avocado["volume"] = avocado["Total Volume"]

# Criar coluna binária 'type_organic'
avocado["type_organic"] = (avocado["type"].str.lower() == "organic").astype(int)

# ----------- MODELO BAYESIANO ----------- #
model = bmb.Model(
    "volume ~ AveragePrice + type_organic",
    data=avocado,
    priors={"AveragePrice": bmb.Prior("Normal", mu=-80, sigma=10)}
)

idata = model.fit(
    draws=1000,
    tune=500,
    chains=4,
    cores=1,
    init="adapt_diag",
    random_seed=42
)

# ----------- PLOTAR TRAÇOS ----------- #
az.plot_trace(idata)
plt.show()

# ----------- RESUMO ----------- #

print(az.summary(idata, hdi_prob=0.99))

# ----------- NOVOS DADOS PARA PREDIÇÃO ----------- #
prices = [0.5, 0.75, 1.0, 1.25]
new_data = pd.DataFrame({
    "AveragePrice": prices,
    "type_organic": [1] * len(prices)  # orgânico
})

# ----------- PREVISÃO MÉDIA ----------- #
mean_preds = model.predict(idata=idata, data=new_data, kind="mean", inplace=False)
print("Previsão do volume (média posterior):")
# Extrair as previsões médias (mu)
mu = mean_preds.posterior["mu"]

# Calcular média para cada observação nova
mean_volume_preds = mu.mean(dim=["chain", "draw"]).values

print("Previsões médias do volume (posterior):")
print(mean_volume_preds)
#print(mean_preds.posterior)
print(mean_preds)

# ----------- POSTERIOR PREDITIVO ----------- #
pps = model.predict(idata=idata, data=new_data, kind="pps", inplace=False)
posterior_volume = pps.posterior_predictive["volume"]

# ----------- LUCRO PREDITIVO ----------- #
predicted_profit_per_price = {}
for i, price in enumerate(prices):
    volume_samples = posterior_volume[:, :, i].values.flatten()
    profit_samples = price * volume_samples
    predicted_profit_per_price[price] = profit_samples

# Forest plot com incerteza do lucro
az.plot_forest(predicted_profit_per_price, combined=True)
plt.title("Lucro previsto por preço (posterior preditivo)")
plt.show()

# ----------- INTERVALO DE CREDIBILIDADE 99% ----------- #
opt_hpd = az.hdi(predicted_profit_per_price[0.75], hdi_prob=0.99)
print(f"Intervalo de credibilidade (99%) do lucro para preço 0.75: {opt_hpd}")
