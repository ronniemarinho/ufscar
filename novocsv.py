import pandas as pd

# Nome do arquivo de entrada
arquivo_entrada = "avocado.csv"

# Nome do arquivo de saída
arquivo_saida = "organic_abacate.csv"

# Categoria que você deseja filtrar (ex: "rice")
categoria = "organic"

# Ler o CSV
dados = pd.read_csv(arquivo_entrada)

# Filtrar os dados pela categoria escolhida
dados_filtrados = dados[dados["type"].str.lower() == categoria.lower()]

# Salvar em um novo CSV
dados_filtrados.to_csv(arquivo_saida, index=False)

print(f"Arquivo '{arquivo_saida}' criado com {len(dados_filtrados)} registros da categoria '{categoria}'.")
