import streamlit as st 
import pandas as pd
import numpy as np

# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import triang, uniform
from scipy.stats import gaussian_kde
from scipy.stats import beta
import plotly.figure_factory as ff
import plotly.graph_objects as go

# desenha as funções de distribuição de probabilidade contínuas: Triangular, PERT, Uniforme
# gráficos com exemplos de funções de distribuição de probabilidade contínuas
def plot_distributions():
    """
    Desenha o gráfico com exemplo de cada uma das funções de distribuição
    contínuas: Triangular, PERT, Uniforme
    """
    # três plots lado a lado
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # triangular:
    x = np.linspace(0, 1, 400)
    y = triang.pdf(x, c=0.5, loc=0.2, scale=0.7)
    axes[0].plot(x, y, "r-", lw=5, alpha=0.6, label="triangular pdf")
    axes[0].legend(loc="best", frameon=False)

    # PERT
    x = np.linspace(0, 1, 400)
    y = beta.pdf(x, 2, 5)
    axes[1].plot(x, y, "r-", lw=5, alpha=0.6, label="PERT pdf")
    axes[1].legend(loc="best", frameon=False)

    # Uniforme
    x = np.linspace(0, 1, 40)
    y = uniform.pdf(x, loc=0.25, scale=0.6)
    axes[2].plot(x, y, "r-", lw=5, alpha=0.6, label="Uniforme pdf")
    axes[2].legend(loc="best", frameon=False)
    #plt.show()
    st.pyplot(plt)


# gera a distribuição PERT (ou beta)
def pert_random(minimum, mode, maximum, size):
    """Gera números aleatórios com distribuição PERT."""
    from scipy.stats import beta

    alpha = (4 * (mode - minimum) / (maximum - minimum)) + 1
    beta_param = (4 * (maximum - mode) / (maximum - minimum)) + 1

    return beta.rvs(alpha, beta_param, loc=minimum, scale=maximum - minimum, size=size)


def mcsm(df, num_simulations):
    """
    Realiza uma simulação Monte Carlo com base nos dados fornecidos.
    """
    # Criando um DataFrame para armazenar os resultados
    simulation_results = pd.DataFrame()

    for index, row in df.iterrows():
        distribution = row["Distribuição"]
        minimum = row["Min."]
        maximum = row["Max."]
        mode = row["Mais Provável"]

        if distribution == "Triangular":
            samples = np.random.triangular(minimum, mode, maximum, num_simulations)
        elif distribution == "Uniforme":
            samples = np.random.uniform(minimum, maximum, num_simulations)
        elif distribution == "PERT":
            samples = pert_random(minimum, mode, maximum, num_simulations)
        else:
            raise ValueError(f"Distribuição desconhecida: {distribution}")

        simulation_results[row["Item"]] = samples

    # Calculando o valor total para cada simulação
    simulation_results["Total"] = simulation_results.sum(axis=1)

    return simulation_results

def plot_probabilities_by_item(df, simulation_results):
    """
    Gráficos de distribuição de probabilidade e de distribuição acumulada para cada item do orçamento,
    mostrando três itens por linha.
    """
    items = df["Item"].tolist()
    num_items = len(items)
    num_cols = 3  # Number of charts per row

    for i in range(0, num_items, num_cols):
        cols = st.columns(num_cols)
        for col, item in zip(cols, items[i:i+num_cols]):
            data = simulation_results[item]
            with col:
                st.subheader(f"{item}")
                
                # Distribuição de Probabilidade
                fig = ff.create_distplot([data], [item], show_hist=True, show_rug=False)
                fig.update_layout(
                    title_text=f"Distribuição de Probabilidade - {item}",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribuição Cumulativa
                fig_cum = go.Figure(
                    data=[
                        go.Histogram(
                            x=data,
                            histnorm="probability",
                            cumulative=dict(enabled=True),
                            nbinsx=50,
                        )
                    ]
                )
                fig_cum.update_layout(
                    title_text=f"Distribuição Cumulativa - {item}",
                    xaxis_title="Custo",
                    yaxis_title="Probabilidade Acumulada",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_cum, use_container_width=True)

def plot_probabilities_total(simulation_results):
    """
    Gráficos de distribuição de probabilidade e de distribuição acumulada para o valor total do orçamento, adaptado para Streamlit.
    """
    data_total = simulation_results["Total"]

    # Distribuição de Probabilidade
    fig = ff.create_distplot([data_total], ["Total"], show_hist=True, show_rug=False)
    fig.update_layout(title_text="Distribuição de Probabilidade - Valor Total")
    st.plotly_chart(fig)

    # Distribuição Cumulativa
    fig_cum = go.Figure(
        data=[
            go.Histogram(
                x=data_total,
                histnorm="probability",
                cumulative=dict(enabled=True),
                nbinsx=50,
                marker_color="#7FB3D5",
            )
        ]
    )
    fig_cum.update_layout(
        title_text="Distribuição Cumulativa - Valor Total",
        xaxis_title="Custo Total",
        yaxis_title="Probabilidade Acumulada",
    )
    st.plotly_chart(fig_cum)

    # Calculando a estimativa de densidade de kernel (KDE) para o valor total
    kde = gaussian_kde(data_total)

    # Definindo o intervalo para o eixo x
    x_values = np.linspace(min(data_total), max(data_total), 1000)
    y_values = kde(x_values)

    # Criando o histograma e a curva KDE
    fig = go.Figure()

    # Adicionando o histograma
    fig.add_trace(
        go.Histogram(
            x=data_total,
            histnorm="probability density",
            nbinsx=50,
            name="Histograma",
            marker_color="#7FB3D5",
            opacity=0.6,
        )
    )

    # Adicionando a curva KDE
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name="Função de Densidade de Probabilidade (KDE)",
            line=dict(color="red", width=2),
        )
    )

    min_value = data_total.min()
    max_value = data_total.max()

    # Criando sliders para selecionar o intervalo
    x_min, x_max = st.slider(
        "Selecione o intervalo de custo total:",
        min_value=float(min_value),
        max_value=float(max_value),
        value=(float(data_total.quantile(0.10)), float(data_total.quantile(0.75))),
        step=float((max_value - min_value) / 100),
    )

    # Calculando a probabilidade no intervalo selecionado
    prob = ((data_total >= x_min) & (data_total <= x_max)).mean() * 100

    # Atualizando o título com a probabilidade
    fig.update_layout(
        title=(
            f"Distribuição de Probabilidade - Valor Total<br>"
            f"Probabilidade no Intervalo Selecionado ({x_min:.2f} a {x_max:.2f}): {prob:.2f}%"
        )
    )

    # Atualizando o intervalo do eixo x
    fig.update_xaxes(range=[x_min, x_max])

    # Exibindo a figura atualizada
    st.plotly_chart(fig)
    


def plot_tornado(df):
    variations = []

    for index, row in df.iterrows():
        item = row["Item"]
        min_val = row["Min."]
        max_val = row["Max."]
        mode_val = row["Mais Provável"]

        # Valor total com o item no mínimo
        min_total = base_total - mode_val + min_val
        # Valor total com o item no máximo
        max_total = base_total - mode_val + max_val

        # Diferenças em relação ao valor base
        min_diff = min_total - base_total
        max_diff = max_total - base_total

        variations.append(
            {"Item": item, "Variação Negativa": min_diff, "Variação Positiva": max_diff}
        )

    # Criando um DataFrame com as variações
    variations_df = pd.DataFrame(variations)

    #display(variations_df)
    st.dataframe(variations_df)

    # Ordenando os itens pela amplitude da variação total
    variations_df["Amplitude"] = (
        variations_df["Variação Positiva"] - variations_df["Variação Negativa"]
    )
    variations_df = variations_df.sort_values("Amplitude", ascending=True)

    # Preparando os dados para o gráfico
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=variations_df["Item"],
            x=variations_df["Variação Negativa"],
            orientation="h",
            name="Variação Negativa",
            marker_color="red",
            hovertemplate="<b>%{y}</b><br>Variação Negativa: %{x}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=variations_df["Item"],
            x=variations_df["Variação Positiva"],
            orientation="h",
            name="Variação Positiva",
            marker_color="green",
            hovertemplate="<b>%{y}</b><br>Variação Positiva: %{x}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Gráfico de Tornado - Impacto de Cada Item no Valor Total",
        xaxis_title="Variação em Relação ao Valor Base",
        yaxis_title="Item",
        barmode="overlay",
        template="plotly_white",
        showlegend=True,
    )

    # Ajustando a visibilidade das barras
    fig.update_traces(width=0.4)

    #fig.show()
    st.plotly_chart(fig)


def print_risk_report(simulation_results, base_total):
    percentiles = np.arange(10, 100, 10)
    percentile_values = np.percentile(simulation_results["Total"], percentiles)

    summary_df = pd.DataFrame(
        {"Percentil (%)": percentiles, "Valor Total": percentile_values}
    )

    # Exibindo o sumário
    st.markdown("### Relatório de Análise de Risco")

    st.divider()
    st.markdown(
        "#### :blue[Sumário dos Percentis do Valor Total com risco:]"
    )
    st.dataframe(summary_df)

    st.divider()
    st.markdown(
        "#### :blue[Estatísticas descritivas do Valor Total com risco:]"
    )
    st.dataframe(simulation_results.describe().transpose())

    st.divider()
    base_total_25 = base_total * (1.25)
    st.markdown(
        f"#### :blue[Valor do orçamento acrescido de 25%: ]{base_total_25}"
    )

    summary_df["Valor da Contingência"] = round(summary_df["Valor Total"] - base_total)
    summary_df["% de Contingência"] = round(
        summary_df["Valor da Contingência"] / base_total * 100
    )
    st.divider()
    st.markdown(
        "#### :blue[Valor Total com risco e valor da contingência correspondente:]"
    )
    st.dataframe(summary_df)


st.set_page_config(page_title="Análise de Risco com Método de Monte Carlo", page_icon="💰", layout="wide")


st.markdown("""

### Problema: Análise de risco de orçamentos

Suponha que o orçamento estimado (organizado em etapas) para construção de uma edificação seja o seguinte:

|Item	|Valor base (R$)|
|---|---|
|	Serviços Preliminares	  |136,00 |
|	Fundação	  |	152,00 |
|	Estrutura	  |	88,00|
|	Cobertura	    |	546,00|
|	Acabamento	|	90,00 |
| Total | 1.012,00|

Vamos utilizar a análise quantitativa de riscos para fornecer ao gestor informações acerca do **nível de incerteza** da estimativa orçamentária.

""")
    
data = pd.read_excel("projeto_exemplo/exemplo.xlsx")

data.rename(
    columns={
        "Família de Serviço": "Item",
        "Mínimo": "Min.",
        "Máximo": "Max.",
    },
    inplace=True,
)

df = st.data_editor(data, num_rows='dynamic')


df["Mais Provável"] = df["Base"]

st.markdown("### Dados de entrada:")

st.dataframe(df)

# Logo, o valor determinístico do orçamento é:
base_total = df["Base"].sum()

base_total = df["Mais Provável"].sum()

st.markdown(f"#### O valor base do orçamento é igual ao valor mais provável e o total é: {base_total}")

st.divider()

st.markdown("### Curvas de Distribuição de Probabilidade disponíveis:")

plot_distributions()

curve = st.selectbox(
    "Curva de Distribuição de Probabilidade",
    ("Triangular", "PERT", "Uniforme"),
    format_func=lambda x: x.upper(),
)


st.write(f"Curva de Distribuição de Probabilidade selecionada: {curve}")

df["Distribuição"] = curve 

st.dataframe(df)

num_simulations = st.number_input("Número de simulações:", value=10000, min_value=1000, max_value=50000)

erro_95 = 1.36 / np.sqrt(10000)

st.markdown("""

### Qual o número ideal de simulações?

**Kolmogoroff-Smirnov** - o **erro máximo** cometido por uma aproximação à distribuição
de frequência acumulada obtida por amostragem **é dependente do número de
amostras e do nível de confiança** desejado para o resultado: """)

st.markdown(r"""
$$
|\text{Erro}_{0.95}| \leq \frac{1.36}{\sqrt{n}}
$$
""")

st.markdown("""
Logo, para 10.000 amostras considerando um nível de confiança de 95% a distribuição cumulativa diferirá no máximo de $\pm 1.4\%$.

Lembrando que o nível de confiança de 95% significa que,  se repetirmos o experimento ou a amostragem muitas vezes, em 95% dessas vezes, o intervalo de confiança estimado a partir dos dados conterá o valor real do parâmetro populacional.

Em outras palavras, o nível de confiança expressa a certeza ou probabilidade de que o intervalo de confiança gerado a partir da amostra represente a verdadeira estatística da população.

""")

st.write(f"Erro considerando o nível de confiança de 95%: {np.round(erro_95, 3)}.")

st.divider()

st.markdown(""" 

### Visão geral da Simulação com o Método de Monte-Carlo


Trata-se de um método estatístico
que tem sido utilizado como forma de obter aproximações numéricas de funções
complexas, que envolve a geração de observações de alguma distribuição de
probabilidades e o uso da amostra obtida para aproximar a função de interesse,
com o objetivo de descrever a distribuição e as características dos possíveis
valores de uma variável dependente.


""")

if st.button("Realizar Simulação"):

    with st.spinner("Simulação de Monte Carlo..."):

        simulation_results = mcsm(df, num_simulations)

        st.dataframe(simulation_results.head().sort_values(by="Total", ascending=False))

    st.markdown("### Resultados por item")
    plot_probabilities_by_item(df, simulation_results)
    st.markdown("### Resultados para o valor Total")
    plot_probabilities_total(simulation_results)
    st.markdown("### Impacto por item")
    plot_tornado(df)

    print_risk_report(simulation_results, base_total)   