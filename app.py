import streamlit as st
import pandas as pd

# --------------------------------------------------
# Configurações iniciais da página
# --------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Compras e Fornecedores",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Função para carregar e preparar os dados
# --------------------------------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    # Converte data para datetime (formato brasileiro dia/mês/ano)
    df["data_compra"] = pd.to_datetime(df["data_compra"], format="%d/%m/%Y")
    # Cria coluna de ano-mês para agregações mensais
    df["ano_mes"] = df["data_compra"].dt.to_period("M").astype(str)
    return df

# --------------------------------------------------
# Carregamento dos dados
# --------------------------------------------------
CSV_PATH = "FCD_compras.csv"  
df = load_data(CSV_PATH)

st.title("Projeto 3 – Dashboard de Compras e Fornecedores")
st.markdown(
    """
    Este dashboard foi desenvolvido para o **Projeto 3 – Dashboard de Compras e Fornecedores**  
    da disciplina *Fundamentos em Ciência de Dados* (Período 2025.2).

    Ele permite analisar:
    - Comparativo entre fornecedores (**preço médio** e **prazo médio de entrega**);
    - **Volume de compras por mês**;
    - **Produtos com maior gasto** em compras.
    """
)

# --------------------------------------------------
# Barra lateral de filtros
# --------------------------------------------------
st.sidebar.header("Filtros")

# Filtro por período (data inicial e final)
min_date = df["data_compra"].min()
max_date = df["data_compra"].max()

date_range = st.sidebar.date_input(
    "Período de compra",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    # Caso o usuário selecione apenas uma data, usamos como início e fim
    start_date = date_range
    end_date = date_range

# Filtro por fornecedor
fornecedores = sorted(df["fornecedor"].unique())
fornecedores_selecionados = st.sidebar.multiselect(
    "Fornecedores",
    options=fornecedores,
    default=fornecedores
)

# Filtro por status da compra
status_opcoes = sorted(df["status_compra"].unique())
status_selecionado = st.sidebar.multiselect(
    "Status da compra",
    options=status_opcoes,
    default=status_opcoes
)

# Aplicando os filtros
df_filtrado = df[
    (df["data_compra"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))) &
    (df["fornecedor"].isin(fornecedores_selecionados)) &
    (df["status_compra"].isin(status_selecionado))
]

st.sidebar.markdown("---")
st.sidebar.write(f"Total de registros filtrados: **{len(df_filtrado)}**")

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()

# --------------------------------------------------
# Indicadores principais (KPIs)
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    total_gasto = df_filtrado["valor_total"].sum()
    st.metric(
        "Total gasto em compras (R$)",
        f"{total_gasto:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

with col2:
    num_compras = df_filtrado["compra_id"].nunique()
    st.metric("Número de compras", f"{num_compras}")

with col3:
    prazo_medio = df_filtrado["prazo_entrega_dias"].mean()
    st.metric(
        "Prazo médio de entrega (dias)",
        f"{prazo_medio:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )

st.markdown("---")

# --------------------------------------------------
# a) Comparativo entre Fornecedores (Preço Médio e Prazo Médio)
# --------------------------------------------------
st.header("a) Comparativo entre Fornecedores – Preço Médio e Prazo Médio")

comparativo_fornecedor = (
    df_filtrado
    .groupby("fornecedor")
    .agg(
        preco_medio=("valor_unitario", "mean"),
        prazo_medio=("prazo_entrega_dias", "mean"),
        total_gasto=("valor_total", "sum")
    )
    .reset_index()
)

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.subheader("Preço médio de compra por fornecedor")
    st.bar_chart(
        comparativo_fornecedor.set_index("fornecedor")["preco_medio"],
        use_container_width=True
    )

with col_a2:
    st.subheader("Prazo médio de entrega por fornecedor (dias)")
    st.bar_chart(
        comparativo_fornecedor.set_index("fornecedor")["prazo_medio"],
        use_container_width=True
    )

st.subheader("Tabela detalhada – Fornecedores")
st.dataframe(
    comparativo_fornecedor.style.format(
        {
            "preco_medio": "R$ {:.2f}",
            "prazo_medio": "{:.1f} dias",
            "total_gasto": "R$ {:.2f}"
        }
    ),
    use_container_width=True
)

st.markdown("---")

# --------------------------------------------------
# b) Volume de Compras por Mês
# --------------------------------------------------
st.header("b) Volume de Compras por Mês")

# Agregação por mês
volume_mensal = (
    df_filtrado
    .groupby("ano_mes")
    .agg(
        valor_total=("valor_total", "sum"),
        quantidade_compras=("compra_id", "nunique")
    )
    .reset_index()
)

col_b1, col_b2 = st.columns(2)

with col_b1:
    st.subheader("Valor total gasto por mês (R$)")
    st.line_chart(
        volume_mensal.set_index("ano_mes")["valor_total"],
        use_container_width=True
    )

with col_b2:
    st.subheader("Quantidade de compras por mês")
    st.line_chart(
        volume_mensal.set_index("ano_mes")["quantidade_compras"],
        use_container_width=True
    )

st.subheader("Tabela detalhada – Volume Mensal")
st.dataframe(
    volume_mensal.style.format(
        {
            "valor_total": "R$ {:.2f}",
            "quantidade_compras": "{:.0f}"
        }
    ),
    use_container_width=True
)

st.markdown("---")

# --------------------------------------------------
# c) Produtos com Maior Gasto em Compras
# --------------------------------------------------
st.header("c) Produtos com Maior Gasto em Compras")

# Top produtos por gasto total
top_produtos = (
    df_filtrado
    .groupby("produto_id")
    .agg(
        gasto_total=("valor_total", "sum"),
        quantidade_total=("quantidade_comprada", "sum"),
        preco_medio=("valor_unitario", "mean")
    )
    .reset_index()
    .sort_values("gasto_total", ascending=False)
    .head(10)
)

# Renomear produto_id para melhor visualização
top_produtos["produto_id"] = "Produto " + top_produtos["produto_id"].astype(str)

st.subheader("Top 10 produtos com maior gasto")
st.bar_chart(
    top_produtos.set_index("produto_id")["gasto_total"],
    use_container_width=True,
    horizontal=True
)

st.subheader("Tabela detalhada – Produtos")
st.dataframe(
    top_produtos.style.format(
        {
            "gasto_total": "R$ {:.2f}",
            "quantidade_total": "{:.0f}",
            "preco_medio": "R$ {:.2f}"
        }
    ),
    use_container_width=True
)

st.markdown("---")

# --------------------------------------------------
# Recomendações para Gestores
# --------------------------------------------------
st.header("📊 Recomendações para Gestores")

st.markdown(
    """
    Com base nos dados apresentados neste dashboard, os gestores podem tomar as seguintes decisões estratégicas:
    
    ### a) Escolha de Fornecedores Mais Eficientes
    - Analisar o **comparativo de preço médio** e **prazo médio de entrega** para identificar fornecedores 
      que oferecem o melhor custo-benefício.
    - Priorizar fornecedores com menores prazos de entrega para produtos críticos ou de alta demanda.
    - Negociar melhores condições com fornecedores que possuem preços acima da média do mercado.
    
    ### b) Planejamento de Compras Estratégicas
    - Utilizar o **volume de compras por mês** para identificar períodos de maior demanda e planejar compras antecipadas.
    - Reduzir custos negociando volumes maiores em períodos de menor demanda, aproveitando possíveis descontos.
    - Evitar picos de gastos distribuindo compras ao longo do ano de forma mais equilibrada.
    
    ### c) Otimização de Estoque
    - Identificar os **produtos com maior gasto** e avaliar se o volume de compras está alinhado com a demanda real.
    - Reduzir investimento em produtos de baixo giro e focar em itens estratégicos ou de maior impacto financeiro.
    - Estabelecer políticas de estoque mínimo e máximo baseadas no histórico de compras apresentado.
    """
)

st.markdown("---")
st.markdown(
    """
    **Desenvolvido para o Projeto 3 – Dashboard de Compras e Fornecedores**  
    *Fundamentos em Ciência de Dados | Professor: Assuero Ximenes | Período: 2025.2*
    """
)
