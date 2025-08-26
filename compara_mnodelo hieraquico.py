# ---------- PACOTES ----------
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import bambi as bmb
import statsmodels.formula.api as smf
import seaborn as sns

# ---------- CARREGAR DADOS ----------
avocado = pd.read_csv("avocado.csv")
avocado["volume"] = avocado["Total Volume"]
avocado["type_organic"] = (avocado["type"].str.lower() == "organic").astype(int)

# ---------- MODELO HIERÁRQUICO FREQUENTISTA ----------
model_frequentist = smf.mixedlm(
    "volume ~ AveragePrice + type_organic",
    data=avocado,
    groups=avocado["region"]
)
result_frequentist = model_frequentist.fit()
print("\n=== Resultados Frequentista ===")
print(result_frequentist.summary())

# Extrair coeficientes e IC 95%
freq_params = result_frequentist.params
freq_conf = result_frequentist.conf_int()
freq_df = pd.DataFrame({
    "param": freq_params.index,
    "mean": freq_params.values,
    "lower": freq_conf[0].values,
    "upper": freq_conf[1].values,
    "method": "Frequentista"
})

# ---------- MODELO HIERÁRQUICO BAYESIANO ----------
model_bayes = bmb.Model(
    "volume ~ AveragePrice + type_organic + (1|region)",
    data=avocado
)

idata_bayes = model_bayes.fit(
    draws=1000,
    tune=500,
    chains=4,
    cores=1,
    random_seed=42
)

print("\n=== Resumo Bayesiano ===")
print(az.summary(idata_bayes, hdi_prob=0.95))

# Extrair média e HDI 95% da posterior
bayes_summary = az.summary(idata_bayes, hdi_prob=0.95)
bayes_df = bayes_summary.reset_index().rename(columns={
    "index": "param",
    "mean": "mean",
    "hdi_2.5%": "lower",
    "hdi_97.5%": "upper"
})
bayes_df["method"] = "Bayesiano"

# Filtrar apenas parâmetros fixos principais
bayes_df = bayes_df[bayes_df["param"].isin(["Intercept", "AveragePrice", "type_organic"])]

# ---------- JUNTAR E PLOTAR ----------
compare_df = pd.concat([freq_df, bayes_df], ignore_index=True)

plt.figure(figsize=(8, 5))
sns.pointplot(
    data=compare_df,
    x="mean", y="param", hue="method",
    join=False, dodge=0.5, palette="Set2",
    errorbar=None
)

# Adicionar barras de erro
for i, row in compare_df.iterrows():
    plt.plot([row["lower"], row["upper"]], [row["param"], row["param"]],
             color="gray" if row["method"] == "Frequentista" else "black",
             alpha=0.6, linewidth=1.5)

plt.axvline(0, color="red", linestyle="--", alpha=0.6)
plt.title("Comparação dos Coeficientes: Frequentista vs Bayesiano")
plt.xlabel("Estimativa / Média Posterior")
plt.ylabel("Parâmetro")
plt.legend(title="Método")
plt.tight_layout()
plt.show()

# ---------- NOTAS ----------
"""
- Frequentista: fornece ponto estimado + intervalo de confiança (IC).
- Bayesiano: fornece distribuição posterior → média + intervalo de credibilidade (HDI).
- Comparar os dois ajuda a ver o impacto dos priors e do tratamento explícito da incerteza.
"""
