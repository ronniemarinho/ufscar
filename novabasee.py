import statsmodels.api as sm

# Carrega o dataset dietox do pacote geepack
data = sm.datasets.get_rdataset("dietox", "geepack").data

# Salva em CSV (no mesmo diretório do seu script)
data.to_csv("dietox.csv", index=False)

print("Dataset salvo como dietox.csv")
