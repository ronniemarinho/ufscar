import matplotlib.pyplot as plt
import seaborn as sns

# Exemplo de arrays de amostras do posterior
# intercept_samples = amostras do intercepto
# slope_samples = amostras do coeficiente angular
# Aqui geramos valores fictícios para ilustrar
import numpy as np
np.random.seed(42)
intercept_samples = np.random.normal(270, 20, 100)
slope_samples = np.random.normal(1.5, 0.3, 100)

plt.figure(figsize=(12,5))

# Histograma / densidade do intercepto
plt.subplot(1,2,1)
sns.histplot(intercept_samples, kde=True, color='skyblue', bins=15)
plt.title("Distribuição do Intercepto")
plt.xlabel("Intercepto")
plt.ylabel("Densidade")

# Histograma / densidade do coeficiente angular
plt.subplot(1,2,2)
sns.histplot(slope_samples, kde=True, color='salmon', bins=15)
plt.title("Distribuição do Coeficiente Angular")
plt.xlabel("Coeficiente Angular")
plt.ylabel("Densidade")

plt.tight_layout()
plt.show()
