# requisitos: pymc, arviz, pandas, numpy, matplotlib, scikit-learn
# pip install pymc arviz pandas numpy matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.simplefilter(action="ignore", category=FutureWarning)
plt.rcParams["figure.figsize"] = (10, 5)

# ================================
# 0. Carregar / preparar dados
# ================================
file_path = "avocado.csv"   # coloque aqui o caminho correto do arquivo
df = pd.read_csv(file_path)

# padroniza nomes
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# converte data
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
else:
    raise ValueError("Coluna 'Date' não encontrada no CSV")

# agregar volume por data (caso existam múltiplas linhas por data/região)
if "total_volume" in df.columns:
    df = df.groupby("date", as_index=False).agg({"total_volume": "sum"})
else:
    raise ValueError("Coluna 'Total Volume' não encontrada no CSV")

df = df.sort_values("date").reset_index(drop=True)

# Índice de tempo (semanas) - assume observações semanais
df["t"] = np.arange(len(df))
# termos sazonais (ciclo ~ 52 semanas)
df["sin52"] = np.sin(2 * np.pi * df["t"] / 52)
df["cos52"] = np.cos(2 * np.pi * df["t"] / 52)

# transformar target (log) para estabilidade
df["y"] = np.log(df["total_volume"] + 1.0)

# ================================
# 1. Train / Test (últimas 52 semanas como holdout)
# ================================
h = 52
if len(df) <= h:
    raise ValueError("Série muito curta para usar holdout de 52 períodos")

train = df.iloc[:-h].reset_index(drop=True)
test = df.iloc[-h:].reset_index(drop=True)

# features treino
y_train = train["y"].values
t_train = train["t"].values
sin_train = train["sin52"].values
cos_train = train["cos52"].values
n_train = len(train)

# features teste (para forecasting)
t_test = test["t"].values
sin_test = test["sin52"].values
cos_test = test["cos52"].values
obs_volume_test = test["total_volume"].values  # no espaço original

# escalas para priors
y_mean = y_train.mean()
y_std = y_train.std()
t_scale = t_train.std()

# ================================
# 2. Modelo PyMC5: tendência + sazonalidade + RW(gauss)
# ================================
with pm.Model() as model:
    # Priors (informativos fracos, baseados na escala dos dados)
    intercept = pm.Normal("intercept", mu=y_mean, sigma=2.0 * y_std)
    slope = pm.Normal("slope", mu=0.0, sigma=0.05)  # pequena variação por unidade de t padronizada
    beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=2)  # sazonalidade (sin, cos)
    sigma_rw = pm.HalfNormal("sigma_rw", sigma=0.5)  # variação do RW (erro autocorrelacionado)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)  # ruído observado

    # Gaussian random walk para capturar resíduos autocorrelacionados (tamanho n_train)
    #rw = pm.GaussianRandomWalk("rw", sigma=sigma_rw, shape=n_train)

    # Non-centered parametrization of RW
    rw_std = pm.Normal("rw_std", mu=0, sigma=1, shape=n_train)
    rw = pm.Deterministic("rw", rw_std.cumsum() * sigma_rw)

    # tendência padronizada no tempo
    t_std = (t_train - t_train.mean()) / t_scale

    mu = intercept + slope * t_std + beta[0] * sin_train + beta[1] * cos_train + rw

    # likelihood (em espaço log)
    obs = pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=y_train)

    # Amostragem (recomendo 4 cadeias; ajuste draws/tune conforme poder computacional)
    idata = pm.sample(
        draws=10000,
        tune=10000,
        chains=4,
        target_accept=0.995,
        random_seed=42,
        cores=4,
    )

    # Posterior predictive para o treino: já anexa ao idata
    pm.sample_posterior_predictive(idata, var_names=["obs"], random_seed=42, extend_inferencedata=True)

# ================================
# 3. Diagnósticos e resumo
# ================================
print("\n--- Resumo posterior ---")
print(az.summary(idata, var_names=["intercept", "slope", "beta", "sigma_rw", "sigma_obs"], round_to=2))

# Traceplots
az.plot_trace(idata, var_names=["intercept", "slope", "beta", "sigma_rw", "sigma_obs"])
plt.tight_layout()
plt.show()

# Divergences e R-hat (az.summary já traz r_hat; divergences via sample_stats)
n_div = idata.sample_stats["diverging"].sum().values
print("Número de divergences:", int(n_div))

# ================================
# 4. Posterior Predictive Check (treino)
# ================================
az.plot_ppc(idata, var_names=["obs"], num_pp_samples=100)
plt.show()

# ================================
# 5. Forecast simples para holdout (usando parâmetros da posterior)
#    Observação: o RW é definido apenas para os pontos de treino.
#    Para forecast, aproximamos usando mu_future sem o RW ou assumindo rw_future=last_rw (opção)
# ================================
# extrair amostras do posterior
post = idata.posterior
# empilhar chain+draw
intercept_samps = post["intercept"].stack(samples=("chain", "draw")).values.flatten()
slope_samps = post["slope"].stack(samples=("chain", "draw")).values.flatten()
#beta_samps = post["beta"].stack(samples=("chain", "draw")).values
#beta_samps = post["beta"].stack(samples=("chain", "draw")).values.transpose(2, 0)
beta_samps = post["beta"].stack(samples=("chain", "draw")).values.T


sigma_obs_samps = post["sigma_obs"].stack(samples=("chain", "draw")).values.flatten()
rw_samps = post["rw"].stack(samples=("chain", "draw")).values  # shape (n_train, nsamples)

nsamps = intercept_samps.shape[0]

# padronizar t_test com média de treino e t_scale
t_train_mean = t_train.mean()
t_test_std = (t_test - t_train_mean) / t_scale

# Strategy: forecast WITHOUT RW (mu_future from trend+sazonalidade) + simulate noise sigma_obs
# (alternativa: usar último valor do RW ou simular RW forward — mais complexo)
pred_samples_log = np.zeros((nsamps, len(t_test)))
rng = np.random.default_rng(42)
for i in range(nsamps):
    mu_f = (
        intercept_samps[i]
        + slope_samps[i] * t_test_std
        + beta_samps[i, 0] * sin_test
        + beta_samps[i, 1] * cos_test
    )
    pred_samples_log[i, :] = rng.normal(loc=mu_f, scale=sigma_obs_samps[i])

# converter para espaço original
pred_samples = np.exp(pred_samples_log) - 1.0
pred_mean = pred_samples.mean(axis=0)

# ================================
# 6. Métricas e coverage no holdout
# ================================
rmse = np.sqrt(mean_squared_error(obs_volume_test, pred_mean))
mae = mean_absolute_error(obs_volume_test, pred_mean)
print(f"\nHoldout RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Coverage 94% nos intervalos preditos
lower = np.percentile(pred_samples, 3, axis=0)
upper = np.percentile(pred_samples, 97, axis=0)
coverage = np.mean((obs_volume_test >= lower) & (obs_volume_test <= upper))
print(f"Coverage 94% no holdout: {coverage*100:.1f}%")

# Plot da previsão vs observado (holdout)
plt.figure(figsize=(12, 5))
plt.plot(test["date"], obs_volume_test, label="Observado (holdout)", marker="o")
plt.plot(test["date"], pred_mean, label="Previsão média", linestyle="--")
plt.fill_between(test["date"], lower, upper, color="C1", alpha=0.3, label="IC 94%")
plt.xticks(rotation=30)
plt.legend()
plt.title("Forecast (últimas 52 semanas) - Previsão vs Observado")
plt.show()

# ================================
# 7. Comentários / próximos passos
# ================================
print("\nObservações rápidas:")
print("- Verifique R-hat (deve estar ~1).")
print("- Se divergences > 0: tente reparametrização non-centered ou aumentar target_accept.")
print("- Se coverage << 94%: incerteza subestimada -> considerar modelo com erro maior ou componente estocástico para forecast.")
print("- Para forecast mais realista, simular RW adiante (ou usar State-space model / AR terms).")



###########################
# ================================
# 8. Comparação em série temporal (observado, verdadeiro e bayesiano)
# ================================

plt.figure(figsize=(14,6))

# curva real (log transform revertida)
plt.plot(train["date"], train["total_volume"], color="black", linestyle="--", label="Verdadeira (treino)")
plt.plot(test["date"], obs_volume_test, color="black", linestyle="-", marker="o", label="Verdadeira (holdout)")

# Observações usadas no modelo (treino)
plt.scatter(train["date"], train["total_volume"], color="gray", alpha=0.5, s=15, label="Observações treino")

# Predição Bayesiana
plt.plot(test["date"], pred_mean, color="red", linestyle="--", label="Predição Bayesiana (média)")
plt.fill_between(test["date"], lower, upper, color="red", alpha=0.3, label="Intervalo Bayesiano 94%")

plt.xticks(rotation=30)
plt.title("Série Temporal: Observações, Verdadeira e Predição Bayesiana")
plt.xlabel("Data")
plt.ylabel("Volume (avocado)")
plt.legend()
plt.show()
