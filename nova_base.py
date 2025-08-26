import pandas as pd

# Carregar a base enviada pelo usuário
file_path = "base_milho.csv"
df = pd.read_csv(file_path)

df.head()
import numpy as np

# Adicionar as novas colunas solicitadas
df["temperatura"] = np.random.normal(loc=28, scale=2, size=len(df))  # milho prefere clima quente
df["pH_solo"] = np.random.normal(loc=6.0, scale=0.3, size=len(df))   # pH levemente ácido
df["umidade_solo"] = np.random.normal(loc=60, scale=5, size=len(df)) # umidade moderada
df["fertilizante"] = np.random.normal(loc=120, scale=15, size=len(df)) # kg/ha aproximado
df["tipo"] = "milho"

# Criar registros adicionais para arroz
n_arroz = len(df) // 2
arroz = pd.DataFrame({
    "volume": np.random.randint(1200, 1600, n_arroz),
    "produtividade": np.random.normal(6, 0.8, n_arroz),
    "temperatura": np.random.normal(26, 1.5, n_arroz),
    "pH_solo": np.random.normal(5.5, 0.2, n_arroz),
    "umidade_solo": np.random.normal(75, 5, n_arroz),
    "fertilizante": np.random.normal(100, 10, n_arroz),
    "tipo": "arroz"
})

# Criar registros adicionais para soja
n_soja = len(df) // 2
soja = pd.DataFrame({
    "volume": np.random.randint(1300, 1800, n_soja),
    "produtividade": np.random.normal(3.5, 0.5, n_soja),
    "temperatura": np.random.normal(25, 1.8, n_soja),
    "pH_solo": np.random.normal(6.2, 0.3, n_soja),
    "umidade_solo": np.random.normal(55, 4, n_soja),
    "fertilizante": np.random.normal(80, 12, n_soja),
    "tipo": "soja"
})

# Concatenar todas as bases
df_final = pd.concat([df, arroz, soja], ignore_index=True)

# Salvar a nova base expandida em um novo arquivo
output_path = "base_expandida.csv"
df_final.to_csv(output_path, index=False)

print(f"Nova base de dados salva em: {output_path}")

