import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

# Dados sintéticos
np.random.seed(42)
n = 100
t = np.arange(n)
true_trend = 0.05 * t
true_season = 2 * np.sin(2 * np.pi * t / 20)
true_noise = np.random.normal(0, 0.5, n)
y = true_trend + true_season + true_noise

# Criando DataFrame
data = pd.DataFrame({'y': y, 't': t})

# Modelo Bayesiano
with pm.Model() as model:
    # Tendência linear com priors
    alpha = pm.Normal('alpha', mu=0, sigma=5)
    beta = pm.Normal('beta', mu=0, sigma=1)
    trend = alpha + beta * t

    # Sazonalidade como senóide com amplitude e fase aleatórias
    A = pm.Normal('A', mu=2, sigma=1)
    phi = pm.Normal('phi', mu=0, sigma=np.pi)
    season = A * pm.math.sin(2 * np.pi * t / 20 + phi)

    # Ruído
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=trend + season, sigma=sigma, observed=y)

    # Amostragem
    trace = pm.sample(1000, tune=1000, cores=2, progressbar=False)

# Calculando média posterior
trend_post = trace.posterior['alpha'].mean(dim=('chain','draw')).values + \
             trace.posterior['beta'].mean(dim=('chain','draw')).values * t
season_post = trace.posterior['A'].mean(dim=('chain','draw')).values * \
              np.sin(2 * np.pi * t / 20 + trace.posterior['phi'].mean(dim=('chain','draw')).values)

# Plot
plt.figure(figsize=(12,6))
plt.plot(t, y, label='Série Original', color='black')
plt.plot(t, trend_post, label='Tendência (posterior mean)', color='orange')
plt.plot(t, season_post, label='Sazonalidade (posterior mean)', color='green')
plt.fill_between(t, trend_post+season_post - trace.posterior['sigma'].mean(dim=('chain','draw')).values,
                 trend_post+season_post + trace.posterior['sigma'].mean(dim=('chain','draw')).values,
                 color='red', alpha=0.3, label='Ruído (±1σ)')
plt.legend()
plt.title("Decomposição Bayesiana de Série Sintética")
plt.show()
