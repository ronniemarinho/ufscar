import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import bambi as bmb
import xarray as xr

# ============================
# Configuração da página e estilo
# ============================
st.set_page_config(layout="wide", page_title="Dietox - Análise Bayesiana")
az.style.use("arviz-darkgrid")
SEED = 7355608
np.random.seed(SEED)

# ============================
# Título e descrição
# ============================
st.title("Aplicação em Ciências da Natureza:: Modelagem Bayesiana Hierárquica 🐷 ")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("fels.svg", width=550)

st.markdown("""
## Estatística Bayesiana em Produção Animal

Esta aplicação interativa em Python utiliza **inferência bayesiana** para modelar a evolução do **peso de suínos ao longo do tempo**.  
O modelo considera efeitos **fixos** e **aleatórios** associados a cada animal, permitindo compreender o crescimento individual e o crescimento médio do grupo.  
A abordagem une **estatística computacional**, **ciência de dados** e conhecimentos das **ciências agrárias e veterinárias**, contribuindo para análises preditivas no contexto da **zootecnia de precisão**.

---

### 📚 Categorias Científicas Envolvidas:
- **📈 Estatística Aplicada / Inferência Bayesiana**  
- **🐖 Ciências Agrárias / Medicina Veterinária / Zootecnia**  
- **🌎 Ciências Ambientais**  
- **💻 Computação Científica / Ciência de Dados**

---
""", unsafe_allow_html=True)

st.markdown("""
📂 **Fonte dos dados**: A base de dados utilizada neste experimento foi retirada do pacote **R datasets**  
disponível em:  
[https://vincentarelbundock.github.io/Rdatasets/doc/MEMSS/Dietox.html](https://vincentarelbundock.github.io/Rdatasets/doc/MEMSS/Dietox.html)
""", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Desenvolvido por Prof. Dr. Ronnie Shida Marinho</h3>", unsafe_allow_html=True)


# ============================
# 1) Carregando os dados
# ============================
st.header("1. Carregando os dados")
col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
import pandas as pd
data = pd.read_csv("dietox.csv")
data.describe()
        """,
        language="python",
    )

with col2:
    data = pd.read_csv("dietox.csv")
    st.markdown("**Descrição estatística (data.describe())**")
    st.dataframe(data.describe())
    st.caption("Certifique-se de que o arquivo 'dietox.csv' está no mesmo diretório do app.")


# ============================
# 2) Definição e ajuste do modelo
# ============================
st.header("2. Definição do modelo e Inferência Bayesiana")
col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
import bambi as bmb

priors = {
    "Intercept": bmb.Prior("Normal", mu=20, sigma=5),
    "Time": bmb.Prior("Normal", mu=0.5, sigma=0.2)
}

model = bmb.Model("Weight ~ Time + (Time|Pig)", data, priors=priors)
results = model.fit()
print(model)
        """,
        language="python",
    )

with col2:
    with st.spinner("Ajustando o modelo Bayesiano com Bambi…"):
        priors = {
            "Intercept": bmb.Prior("Normal", mu=20, sigma=5),
            "Time": bmb.Prior("Normal", mu=0.5, sigma=0.2)
        }
        model = bmb.Model("Weight ~ Time + (Time|Pig)", data, priors=priors)
        results = model.fit()
    st.success("Modelo ajustado!")

    st.markdown("""
### 🐖 Exemplo de Crença Inicial

- Peso inicial médio: **20 kg**
- Ganho médio de peso por dia: **0.5 kg/dia**

---

### 📐 Equação da Reta

$$
y = \\beta_0 + \\beta_1 x
$$

Peso esperado para 10 dias:

$$
20 + 0.5 \\times 10 = 25 \\text{ kg}
$$
    """, unsafe_allow_html=True)


# Extrai posterior com dimensão amostral combinada (chain, draw)
posterior = az.extract_dataset(results)
if all(dim in posterior.dims for dim in ("chain", "draw")):
    posterior = posterior.stack(sample=("chain", "draw"))


# ============================
# 3) Distribuições a priori (priors)
# ============================
st.subheader("2.1 Visualização das Distribuições a priori (priors)")
col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
# Visualização das distribuições a priori
model.plot_priors()
        """,
        language="python",
    )
    st.markdown("**Ideia:** visualizar as crenças iniciais antes de ver os dados.")

with col2:
    plt.close("all")
    model.plot_priors()
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)


# ============================
# 4) Distribuições posteriores
# ============================
st.subheader("2.2 Visualização das Distribuições posteriores dos Parâmetros")
col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
import arviz as az

az.plot_trace(
    results,
    var_names=["Intercept", "Time", "1|Pig", "Time|Pig", "sigma"],
    compact=True,
)
        """,
        language="python",
    )
    st.markdown("Avalie mistura das cadeias e convergência.")

with col2:
    plt.close("all")
    az.plot_trace(
        results,
        var_names=["Intercept", "Time", "1|Pig", "Time|Pig", "sigma"],
        compact=True,
    )
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)


st.header("3. Verificaçao Preditiva Posteriori")


col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
ppc = model.predict(results, kind="pps", inplace=False)
az.plot_ppc(ppc, num_pp_samples=100)
        """,
        language="python",
    )
    st.markdown("""
**Objetivo:** Verificar se o modelo reproduz os dados observados.
""")

with col2:
    ppc = model.predict(results, kind="pps", inplace=False)
    plt.close("all")
    az.plot_ppc(ppc, num_pp_samples=100)
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)


# ============================
# 6) Análise numérica dos parâmetros (trace + resumo)
# ============================
st.subheader("3.1 Resumo numérico da posterior")
col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
summary = az.summary(
    results,
    var_names=["Intercept", "Time", "1|Pig_sigma", "Time|Pig_sigma", "sigma"],
)
        """,
        language="python",
    )

with col2:
    summary = az.summary(
        results,
        var_names=["Intercept", "Time", "1|Pig_sigma", "Time|Pig_sigma", "sigma"],
    )
    st.dataframe(summary)


# ============================
# 7) Previsão e análise visual: curvas individuais e médias do grupo
# ============================
st.header("4. Previsões amostrais")
col1, col2 = st.columns([1, 2])

all_pigs = sorted({str(p) for p in data["Pig"].unique()})
selected_pig = st.sidebar.selectbox("Escolha o porquinho (Pig ID)", options=all_pigs, index=max(0, all_pigs.index("4601") if "4601" in all_pigs else 0))

with col1:
    st.code(
        """
data_pig = data[data["Pig"] == int(selected_pig)][["Time", "Weight"]]

intercept_common = posterior["Intercept"]
slope_common = posterior["Time"]
intercept_specific = posterior["1|Pig"].sel(Pig__factor_dim=str(selected_pig))
slope_specific = posterior["Time|Pig"].sel(Pig__factor_dim=str(selected_pig))

a = intercept_common + intercept_specific
b = slope_common + slope_specific

time_xi = xr.DataArray(np.array([1, 12]))
        """,
        language="python",
    )
    st.markdown("Visualize amostras posteriores, média e observações.")

with col2:
    data_pig = data[data["Pig"] == int(selected_pig)][["Time", "Weight"]]
    time = np.array([1, 12])

    intercept_common = posterior["Intercept"]
    slope_common = posterior["Time"]
    intercept_specific = posterior["1|Pig"].sel(Pig__factor_dim=str(selected_pig))
    slope_specific = posterior["Time|Pig"].sel(Pig__factor_dim=str(selected_pig))

    a = intercept_common + intercept_specific
    b = slope_common + slope_specific

    time_xi = xr.DataArray(time)
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot(time_xi, (a + b * time_xi)[0].T, lw=0.3, label="Amostras Posteriores")
    for i in range(1, len(a)):
        ax.plot(time_xi, (a + b * time_xi)[i].T, lw=0.3, alpha=0.3)
    ax.plot(time_xi, a.mean() + b.mean() * time_xi, color="black", lw=2, label="Média Posterior")
    ax.scatter(data_pig["Time"], data_pig["Weight"], color="red", zorder=2, label="Observações")
    ax.set_ylabel("Peso (kg)")
    ax.set_xlabel("Tempo (semanas)")
    ax.set_title(f"Peso ao longo do tempo para o Porquinho {selected_pig}")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


st.header("5. Análise dos Resultados")
col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
intercept_group_specific = posterior["1|Pig"]
slope_group_specific = posterior["Time|Pig"]

a_group = intercept_common.mean("sample") + intercept_group_specific.mean("sample")
b_group = slope_common.mean("sample") + slope_group_specific.mean("sample")
        """,
        language="python",
    )

with col2:
    intercept_group_specific = posterior["1|Pig"]
    slope_group_specific = posterior["Time|Pig"]
    a_group = intercept_common.mean("sample") + intercept_group_specific.mean("sample")
    b_group = slope_common.mean("sample") + slope_group_specific.mean("sample")

    time_xi = xr.DataArray(time)
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot(time_xi, (a_group + b_group * time_xi)[0].T, alpha=0.7, lw=0.8, label="Amostras Posteriores (Grupo)")
    for i in range(1, len(a_group)):
        ax.plot(time_xi, (a_group + b_group * time_xi)[i].T, alpha=0.3, lw=0.8)
    ax.set_ylabel("Peso (kg)")
    ax.set_xlabel("Tempo (semanas)")
    ax.set_title("Peso médio do grupo ao longo do tempo")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# ============================
# 8) Visualizações finais: Forest plot
# ============================
#st.subheader("5.1 Forest plot")
col1, col2 = st.columns([1, 2])

with col1:
    st.code(
        """
az.plot_forest(results, var_names=["Intercept", "Time"], figsize=(8, 2))
        """,
        language="python",
    )

with col2:
    plt.close("all")
    az.plot_forest(results, var_names=["Intercept", "Time"], figsize=(8, 2))
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)
