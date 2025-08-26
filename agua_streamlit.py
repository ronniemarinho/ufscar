import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import bambi as bmb
from scipy.stats import norm
from matplotlib.patches import Patch

st.set_page_config(layout="wide")
sns.set_theme()


st.title("Aplicação em Ciências da Natureza: Regressão Linear Bayesiana 🌽")
col1, col2, col3 = st.columns([1, 2, 1])  # proporções: esquerda, centro, direita

with col2:
    st.image("fels.svg", width=550)

#st.image("fels.svg", caption="Minha imagem", use_container_width=True)
st.markdown("""
## Estatística Bayesiana na Agricultura de Precisão

Esta aplicação interativa em Python utiliza **inferência bayesiana** para modelar a relação entre o volume de água aplicado e a produtividade de milho.  
Ela combina **estatística computacional**, **ciência de dados** e conhecimentos das **ciências agrárias e ambientais**, contribuindo para análises preditivas e **tomadas de decisão sustentáveis** no contexto da **agricultura de precisão**.

---

### 📚 Categorias Científicas Envolvidas:

- **📈 Estatística Aplicada / Inferência Bayesiana**  
  Uso de modelos probabilísticos com inferência Bayesiana por meio da biblioteca `bambi`.

- **🌱 Ciências Agrárias / Ciências Biológicas**  
  Estudo da produtividade agrícola (milho) em função de práticas de manejo como irrigação.

- **🌎 Ciências Ambientais**  
  Relação entre uso de recursos hídricos e produtividade, com impacto na sustentabilidade.

- **💻 Computação Científica / Ciência de Dados**  
  Desenvolvimento com Python, `streamlit`, `pandas`, `arviz`, `seaborn` e `matplotlib`.

---
""", unsafe_allow_html=True)
st.markdown(
    """
    📂 **Fonte dos dados**: A base de dados utilizada neste experimento foi retirada do repositório público disponível em  
    [https://github.com/nishanth009-oss/Simulated-Crop-Yield-Prediction-Dataset-for-Advance-Machine-Learning-Analysis](https://github.com/nishanth009-oss/Simulated-Crop-Yield-Prediction-Dataset-for-Advance-Machine-Learning-Analysis)
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>Desenvolvido por Prof. Dr. Ronnie Shida Marinho</h3>",
    unsafe_allow_html=True
)




# ======================================
# 1. Carregando os Dados
# ======================================
st.header("1. Carregando os Dados ")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Código Python")
    st.code("""
df = pd.read_csv("base_milho.csv")
volume = df["volume"].values
produtividade = df["produtividade"].values
""", language="python")

with col2:
    st.subheader("Prévia dos dados")
    df = pd.read_csv("base_milho.csv")
    st.dataframe(df.head(10), use_container_width=True)

volume = df["volume"].values
produtividade = df["produtividade"].values


# ======================================
# 2. Ajuste do Modelo Bayesiano
# ======================================
st.header("2. Definição do Modelo e Inferência Bayesiana")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Código Python")
    st.code("""
modelo = bmb.Model(
    "produtividade ~ volume",
    data=df,
    priors={
        "Intercept": bmb.Prior("Normal", mu=4, sigma=2),
        "volume": bmb.Prior("Normal", mu=0.004, sigma=0.002)
    }
)
modelo_ajustado = modelo.fit(draws=1000, chains=4, random_seed=0)
modelo.predict(modelo_ajustado, kind="response")
""", language="python")

with col2:
    with st.spinner("Ajustando o modelo..."):
        modelo = bmb.Model(
            "produtividade ~ volume",
            data=df,
            priors={
                "Intercept": bmb.Prior("Normal", mu=4, sigma=2),
                "volume": bmb.Prior("Normal", mu=0.004, sigma=0.002)
            }
        )
        modelo_ajustado = modelo.fit(draws=1000, chains=4, random_seed=0)
        modelo.predict(modelo_ajustado, kind="response")
    st.success("✅ Modelo ajustado com sucesso!")
    st.markdown(r"""
    ### 🧠 Exemplo de Crença Inicial

    - Acredito que, mesmo sem irrigação, a produtividade média de milho na região analisada seja em torno de **4 toneladas por hectare**, considerando as condições naturais do solo.
    - Também estimo que, **para cada metro cúbico de água aplicado**, a produtividade aumente em média cerca de **0.004 toneladas por hectare**, com alguma incerteza associada.

    Essas expectativas representam a **crença inicial** que carrego.

    ---

    """)
    st.markdown("### Representação Geral da Equação da Reta")
    st.latex(r"y = \beta_0 + \beta_1 \cdot x")

    st.markdown("### Representação Matemática da Crença (Equação da Reta)")
    st.latex(r"\text{Produtividade} = 4 + 0.004 \cdot \text{Volume}")

# ======================================
# 2.1 Visualização das Distribuições Priori
# ======================================
st.subheader("2.1 Visualização das Distribuições Priori")

with st.expander("📉 Ver Priors dos Parâmetros"):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.code("""
modelo.plot_priors()
fig_priors = plt.gcf()
st.pyplot(fig_priors)
""", language="python")
    with col2:
        modelo.plot_priors()
        fig_priors = plt.gcf()
        st.pyplot(fig_priors)


# ======================================
# 3. Distribuições Posteriores
# ======================================
st.subheader("2.2 Visualização das Distribuições Posteriores dos Parâmetros")

with st.expander("🔍 Ver Distribuições"):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.code("""
az.plot_trace(modelo_ajustado, var_names=["Intercept", "volume"])
""", language="python")
    with col2:
        fig_trace, _ = plt.subplots(figsize=(12, 6))
        az.plot_trace(modelo_ajustado, var_names=["Intercept", "volume"])
        st.pyplot(plt.gcf())


# 5. Verificação Preditiva Posterior
# ======================================
st.header("3. Verificação Preditiva Posterior")

col1, col2 = st.columns([1, 2])
with col1:
    st.code("""
az.plot_ppc(modelo_ajustado)
""", language="python")

with col2:
    fig_ppc, ax = plt.subplots(figsize=(10, 6))
    az.plot_ppc(modelo_ajustado, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Previsões Posteriores", "Observado", "Média da previsão posterior"])
    ax.set_title("Verificação Preditiva Posterior", fontsize=14)
    st.pyplot(fig_ppc)


# ======================================
# 6. Análise Numérica dos Parâmetros
# ======================================
st.subheader("3.1 Resumo numérico da posterior")

col1, col2 = st.columns([1, 2])
with col1:
    st.code("""
media_volume = modelo_ajustado.posterior.volume.values.mean()
media_intercepto = modelo_ajustado.posterior.Intercept.values.mean()
az.summary(modelo_ajustado)
""", language="python")

with col2:
    media_volume = modelo_ajustado.posterior.volume.values.mean()
    media_intercepto = modelo_ajustado.posterior.Intercept.values.mean()

    st.markdown(f"**📌 Média do coeficiente 'volume':** {media_volume:.6f}")
    st.markdown(f"**📌 Média do intercepto:** {media_intercepto:.6f}")

    rhat_summary = az.summary(modelo_ajustado, hdi_prob=0.95)
    st.dataframe(rhat_summary)


# ======================================
# 7. Previsões Amostrais
# ======================================
st.header("4. Previsões Amostrais")

col1, col2 = st.columns([1, 2])
with col1:
    st.code("""
def plot_sample_prediction(volume, produtividade, fitted):
    plt.scatter(volume, produtividade, label="Dados reais")

    x_range = np.linspace(min(volume), max(volume), 2000)
    y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()

    # Linha de regressão média
    plt.plot(x_range, y_pred, color="black", label="Linha média (posterior)", lw=2)

    # Algumas amostras da distribuição posterior
    for i in range(10):
        label = "Amostras da posterior" if i == 0 else None
        y_pred_sample = fitted.posterior.volume.values[0, i] * x_range + fitted.posterior.Intercept.values[0, i]
        plt.plot(x_range, y_pred_sample, color="green", linestyle="-.", alpha=0.5, label=label)

    plt.xlabel("Volume de água aplicado (m³)")
    plt.ylabel("Produtividade (toneladas por hectare)")
    plt.title("Previsão amostral com base no modelo bayesiano")
    plt.legend(loc="best")
    plt.show()

plot_sample_prediction(volume, produtividade, modelo_ajustado)

""", language="python")
with col2:
    def plot_sample_prediction(volume, produtividade, fitted):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(volume, produtividade, label="Dados reais")

        x_range = np.linspace(min(volume), max(volume), 2000)
        y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()
        ax.plot(x_range, y_pred, color="black", label="Linha média (posterior)", lw=2)

        for i in range(10):
            label = "Amostras da posterior" if i == 0 else None
            y_pred_sample = fitted.posterior.volume.values[0, i] * x_range + fitted.posterior.Intercept.values[0, i]
            ax.plot(x_range, y_pred_sample, color="green", linestyle="-.", alpha=0.5, label=label)

        ax.set_xlabel("Volume de água aplicado (m³)")
        ax.set_ylabel("Produtividade (ton/ha)")
        ax.set_title("Previsão amostral com base no modelo bayesiano")
        ax.legend()
        return fig

    st.pyplot(plot_sample_prediction(volume, produtividade, modelo_ajustado))


# ======================================
# 8. Intervalos de Credibilidade
# ======================================
st.header("5. Análise dos Resultados ")

col1, col2 = st.columns([1, 2])
with col1:
    st.code("""
    def plot_interval_prediction(volume, produtividade, fitted, hdi_inner=0.50, hdi_outer=0.94):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(volume, produtividade, label="Dados reais")

        x_range = np.linspace(min(volume), max(volume), 2000)
        y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()
        ax.plot(x_range, y_pred, color="black", label="Linha média (posterior)", lw=2)

        # Plotando HDIs
        az.plot_hdi(volume, fitted.posterior_predictive.produtividade, hdi_prob=hdi_inner,
                    color="firebrick", fill_kwargs={"alpha": 0.6}, ax=ax)
        az.plot_hdi(volume, fitted.posterior_predictive.produtividade, hdi_prob=hdi_outer,
                    color="firebrick", fill_kwargs={"alpha": 0.3}, ax=ax)

        # Legenda manual
        legend_elements = [
            plt.Line2D([0], [0], color="black", lw=2, label="Linha média (posterior)"),
            Patch(facecolor="firebrick", alpha=0.6, label=f"Intervalo de {int(hdi_inner * 100)}%"),
            Patch(facecolor="firebrick", alpha=0.3, label=f"Intervalo de {int(hdi_outer * 100)}%"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="Dados reais", markersize=8)
        ]
        ax.legend(handles=legend_elements, loc="best")
        ax.set_xlabel("Volume de água (m³)")
        ax.set_ylabel("Produtividade (toneladas por hectare)")
        ax.set_title("Intervalos de credibilidade para as previsões")
        fig.tight_layout()

        return fig
    st.pyplot(plot_interval_prediction(volume, produtividade, modelo_ajustado, hdi_inner=0.50, hdi_outer=0.94))

""", language="python")
with col2:
    def plot_interval_prediction(volume, produtividade, fitted, hdi_inner=0.50, hdi_outer=0.94):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(volume, produtividade, label="Dados reais")

        x_range = np.linspace(min(volume), max(volume), 2000)
        y_pred = fitted.posterior.volume.values.mean() * x_range + fitted.posterior.Intercept.values.mean()
        ax.plot(x_range, y_pred, color="black", label="Linha média (posterior)", lw=2)

        # Plotando HDIs
        az.plot_hdi(volume, fitted.posterior_predictive.produtividade, hdi_prob=hdi_inner,
                    color="firebrick", fill_kwargs={"alpha": 0.6}, ax=ax)
        az.plot_hdi(volume, fitted.posterior_predictive.produtividade, hdi_prob=hdi_outer,
                    color="firebrick", fill_kwargs={"alpha": 0.3}, ax=ax)

        # Legenda manual
        legend_elements = [
            plt.Line2D([0], [0], color="black", lw=2, label="Linha média (posterior)"),
            Patch(facecolor="firebrick", alpha=0.6, label=f"Intervalo de {int(hdi_inner * 100)}%"),
            Patch(facecolor="firebrick", alpha=0.3, label=f"Intervalo de {int(hdi_outer * 100)}%"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="Dados reais", markersize=8)
        ]
        ax.legend(handles=legend_elements, loc="best")
        ax.set_xlabel("Volume de água (m³)")
        ax.set_ylabel("Produtividade (toneladas por hectare)")
        ax.set_title("Intervalos de credibilidade para as previsões")
        fig.tight_layout()

        return fig
    st.pyplot(plot_interval_prediction(volume, produtividade, modelo_ajustado, hdi_inner=0.50, hdi_outer=0.94))

# ======================================
# Footer
# ======================================
st.markdown("---")
st.caption("Desenvolvido por Prof. Dr. Ronnie Shida Marinho ❤️ usando Streamlit, Bambi e ArviZ.")
