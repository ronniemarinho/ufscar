import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

pd.set_option("display.max_columns", None)  # não cortar colunas
pd.set_option("display.max_rows", None)     # não cortar linhas
pd.set_option("display.width", None)        # não limitar largura
pd.set_option("display.max_colwidth", None) # mostra tudo de cada célula


az.style.use("arviz-grayscale")
from cycler import cycler

default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('figure', dpi=300)
np.random.seed(123)

from pathlib import Path

csv_path = Path("avocado.csv")
if not csv_path.exists():
    raise FileNotFoundError(
        f"Arquivo {csv_path} não encontrado. "
        "Envie o arquivo avocado.csv para o ambiente atual e rode novamente esta célula."
    )

df = pd.read_csv(csv_path)

# Preparação dos dados
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['AveragePrice', 'Total Volume', 'region', 'type', 'year']).copy()
df['region'] = df['region'].astype('category')
df['region_idx'] = df['region'].cat.codes
df['year_c'] = df['year'] - df['year'].mean()
df['log_volume'] = np.log1p(df['Total Volume'])

coords = {
    "obs_id": np.arange(len(df)),
    "region": df['region'].cat.categories.values
}

region_idx = df['region_idx'].values
y = df['AveragePrice'].values
year_c = df['year_c'].values
log_volume = df['log_volume'].values

# Modelo hierárquico
with pm.Model(coords=coords) as m_avocado_h:
    # Hyperpriors
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=10)

    # Interceptos regionais
    alpha_region = pm.Normal('alpha_region', mu=mu_alpha, sigma=sigma_alpha, dims="region")

    # Slopes globais
    beta_year = pm.Normal('beta_year', mu=0, sigma=10)
    beta_vol = pm.Normal('beta_vol', mu=0, sigma=10)

    # Ruído
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Linear predictor
    mu = alpha_region[region_idx] + beta_year * year_c + beta_vol * log_volume

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y, dims="obs_id")

    # Sampling
    idata_avocado = pm.sample(draws=5000, tune=5000, target_accept=0.95, random_seed=4591)

# Sumário com R-hat e ESS
summary = az.summary(
    idata_avocado,
    var_names=['mu_alpha', 'sigma_alpha', 'beta_year', 'beta_vol', 'sigma', 'alpha_region'],
    round_to=2
)
print(summary)

# Traceplots para verificar convergência
az.plot_trace(idata_avocado, var_names=['mu_alpha', 'sigma_alpha', 'beta_year', 'beta_vol', 'sigma', 'alpha_region'])
plt.tight_layout()
plt.show()

# Intervalos de credibilidade (HDI 95%)
hdi = az.hdi(idata_avocado, var_names=['mu_alpha', 'sigma_alpha', 'beta_year', 'beta_vol', 'sigma'], hdi_prob=0.95)
print("\nIntervalos de Credibilidade 95% (HDI):")
print(hdi)

# Pairplot opcional para ver correlações
az.plot_pair(idata_avocado, var_names=['mu_alpha', 'sigma_alpha', 'beta_year', 'beta_vol', 'sigma'], kind='kde',
             marginals=True)
plt.tight_layout()
plt.show()

# Forest plot dos interceptos regionais
az.plot_forest(idata_avocado, var_names=['alpha_region'], combined=True, figsize=(10, 6))
plt.title("Random intercepts by region (partial pooling)")
plt.tight_layout()
plt.show()

# Posterior predictive checks
with m_avocado_h:
    ppc = pm.sample_posterior_predictive(idata_avocado, random_seed=4591)

az.plot_ppc(ppc, num_pp_samples=200)
plt.tight_layout()
plt.show()

# Curva preditiva para uma região
example_region = df['region'].cat.categories[0]
example_idx = np.where(df['region'].cat.categories == example_region)[0][0]

grid = pd.DataFrame({
    'region_idx': example_idx,
    'year_c': 0.0,
    'log_volume': np.linspace(df['log_volume'].quantile(0.05), df['log_volume'].quantile(0.95), 100)
})

posterior = idata_avocado.posterior
alpha_r = posterior['alpha_region'].sel(region=example_region).values
beta_y = posterior['beta_year'].values
beta_v = posterior['beta_vol'].values


def expand(arr):
    return arr.reshape(-1)


alpha_r_s = expand(alpha_r)
beta_y_s = expand(beta_y)
beta_v_s = expand(beta_v)

preds = []
for _, row in grid.iterrows():
    mu = alpha_r_s + beta_y_s * row['year_c'] + beta_v_s * row['log_volume']
    preds.append(mu)

preds = np.vstack(preds).T

pred_df = pd.DataFrame({
    'log_volume': grid['log_volume'].values,
    'mean': preds.mean(axis=0),
    'hdi_low': np.percentile(preds, 2.5, axis=0),
    'hdi_high': np.percentile(preds, 97.5, axis=0),
})

# Visualização da curva preditiva
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pred_df['log_volume'], pred_df['mean'], label=example_region)
ax.fill_between(pred_df['log_volume'], pred_df['hdi_low'], pred_df['hdi_high'], alpha=0.2)

ax.set_xlabel("log(1 + Total Volume)")
ax.set_ylabel("AveragePrice (predicted)")
ax.set_title(f"Region: {example_region}")
ax.legend()
plt.tight_layout()
plt.show()
