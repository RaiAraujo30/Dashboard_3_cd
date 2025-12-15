from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt



APP_TITLE = "Dashboard Integrado: Estoque, Vendas e Compras"

CATEGORY_MAP: Dict[int, str] = {}

DEFAULT_CATEGORY_BUCKETS = 5


def find_csv(filename_candidates: list[str]) -> Path:
    here = Path(__file__).resolve().parent
    for name in filename_candidates:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Não encontrei o(s) CSV(s). Verifique se estes arquivos estão na mesma pasta do .py:\n"
        + "\n".join(f"- {c}" for c in filename_candidates)
    )


@st.cache_data(show_spinner=False)
def read_csv_semicolon(path: Path) -> pd.DataFrame:
    # Os arquivos estão com separador ';'
    return pd.read_csv(path, sep=";")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # vendas: dd/mm/yyyy
    if "data_venda" in df.columns:
        df["data_venda"] = pd.to_datetime(df["data_venda"], dayfirst=True, errors="coerce")

    # compras: yyyy-mm-dd (normalmente)
    if "data_compra" in df.columns:
        df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")

    # estoque: yyyy-mm-dd (normalmente)
    if "data_referencia" in df.columns:
        df["data_referencia"] = pd.to_datetime(df["data_referencia"], errors="coerce")

    return df


def build_product_dim(produto_ids: pd.Series) -> pd.DataFrame:
    """Cria uma dimensão simples de produto com categoria derivada."""
    unique_ids = pd.Series(sorted(pd.unique(produto_ids.dropna().astype(int)))).astype(int)

    def category_for(pid: int) -> str:
        if pid in CATEGORY_MAP:
            return CATEGORY_MAP[pid]
        # regra simples e determinística para permitir filtro por categoria:
        bucket = ((pid - 1) % DEFAULT_CATEGORY_BUCKETS) + 1
        return f"Categoria {bucket}"

    dim = pd.DataFrame(
        {
            "produto_id": unique_ids,
            "produto": unique_ids.map(lambda x: f"Produto {x}"),
            "categoria": unique_ids.map(category_for),
        }
    )
    return dim


def month_key(dt: pd.Series) -> pd.Series:
    """Chave mensal YYYY-MM para agregações."""
    return dt.dt.to_period("M").astype(str)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0, None) else float("nan")


# ==========================
# Carga e preparação dos dados
# ==========================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estoque_path = find_csv(["FCD_estoque.csv"])
    vendas_path = find_csv(["FCD_vendas.csv"])
    compras_path = find_csv(["FCD_compras (1).csv", "FCD_compras.csv"])

    estoque = parse_dates(read_csv_semicolon(estoque_path))
    vendas = parse_dates(read_csv_semicolon(vendas_path))
    compras = parse_dates(read_csv_semicolon(compras_path))

    # Produto dim (como não existe tabela de produtos, criamos)
    all_prod_ids = pd.concat(
        [
            estoque.get("produto_id", pd.Series(dtype=int)),
            vendas.get("produto_id", pd.Series(dtype=int)),
            compras.get("produto_id", pd.Series(dtype=int)),
        ],
        ignore_index=True,
    )
    dim_prod = build_product_dim(all_prod_ids)

    # Join para ter "produto" e "categoria" nas tabelas
    estoque = estoque.merge(dim_prod, on="produto_id", how="left")
    vendas = vendas.merge(dim_prod, on="produto_id", how="left")
    compras = compras.merge(dim_prod, on="produto_id", how="left")

    return estoque, vendas, compras, dim_prod


def latest_stock_snapshot(estoque: pd.DataFrame) -> pd.DataFrame:
    """Retorna o estoque no último data_referencia disponível (por produto e local)."""
    df = estoque.dropna(subset=["data_referencia"]).copy()
    if df.empty:
        return df

    last_date = df["data_referencia"].max()
    snap = df[df["data_referencia"] == last_date].copy()
    return snap


def purchase_unit_cost(compras: pd.DataFrame) -> pd.DataFrame:
    """
    Define um custo unitário por produto baseado na compra mais recente (Entregue, se existir).
    Se não existir Entregue, usa a mais recente de qualquer status.
    """
    df = compras.dropna(subset=["data_compra"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["produto_id", "unit_cost"])

    entregues = df[df["status_compra"].astype(str).str.lower() == "entregue"].copy()
    source = entregues if not entregues.empty else df

    source = source.sort_values("data_compra")
    last = source.groupby("produto_id", as_index=False).tail(1)
    last = last[["produto_id", "valor_unitario"]].rename(columns={"valor_unitario": "unit_cost"})
    return last


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.caption(
    "Dashboard interativo integrando Estoque, Vendas e Compras/Fornecedores. "
    "Filtros na barra lateral atualizam indicadores e gráficos."
)

estoque, vendas, compras, dim_prod = load_data()

# Sidebar filtros
st.sidebar.header("Filtros")

# Período (mês/ano) -> selecionamos intervalo de datas com base em vendas+compras
all_dates = pd.concat(
    [
        vendas["data_venda"].dropna(),
        compras["data_compra"].dropna(),
    ],
    ignore_index=True,
)

if all_dates.empty:
    min_date = pd.Timestamp.today().normalize()
    max_date = min_date
else:
    min_date = all_dates.min().normalize()
    max_date = all_dates.max().normalize()

date_range = st.sidebar.date_input(
    "Período (início / fim)",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # inclusivo
else:
    start_date = min_date
    end_date = max_date + pd.Timedelta(days=1)

# Produto / Categoria
cats = ["(Todas)"] + sorted(dim_prod["categoria"].dropna().unique().tolist())
selected_cat = st.sidebar.selectbox("Categoria", options=cats, index=0)

if selected_cat == "(Todas)":
    prod_options = dim_prod.sort_values("produto_id")["produto"].tolist()
    allowed_prod_ids = set(dim_prod["produto_id"].tolist())
else:
    prod_options = dim_prod[dim_prod["categoria"] == selected_cat].sort_values("produto_id")["produto"].tolist()
    allowed_prod_ids = set(dim_prod[dim_prod["categoria"] == selected_cat]["produto_id"].tolist())

selected_products = st.sidebar.multiselect(
    "Produto(s)",
    options=prod_options,
    default=[],
    help="Se vazio, considera todos os produtos da categoria selecionada.",
)

if selected_products:
    prod_ids = set(dim_prod[dim_prod["produto"].isin(selected_products)]["produto_id"].tolist())
else:
    prod_ids = allowed_prod_ids

# Loja (vendas)
lojas = ["(Todas)"] + sorted(vendas["loja_id"].dropna().unique().tolist())
selected_loja = st.sidebar.selectbox("Loja (Vendas)", options=lojas, index=0)

# ==========================
# Aplicação dos filtros
# ==========================
vendas_f = vendas.copy()
compras_f = compras.copy()
estoque_f = estoque.copy()

# produto/categoria
vendas_f = vendas_f[vendas_f["produto_id"].isin(prod_ids)]
compras_f = compras_f[compras_f["produto_id"].isin(prod_ids)]
estoque_f = estoque_f[estoque_f["produto_id"].isin(prod_ids)]

# loja
if selected_loja != "(Todas)":
    vendas_f = vendas_f[vendas_f["loja_id"] == selected_loja]

# período
vendas_f = vendas_f[(vendas_f["data_venda"] >= start_date) & (vendas_f["data_venda"] < end_date)]
compras_f = compras_f[(compras_f["data_compra"] >= start_date) & (compras_f["data_compra"] < end_date)]

# snapshot de estoque (última referência) *depois* de filtrar produtos
estoque_snap = latest_stock_snapshot(estoque_f)

# custo unitário (para valor do estoque)
unit_cost = purchase_unit_cost(compras)
estoque_snap = estoque_snap.merge(unit_cost, on="produto_id", how="left")
estoque_snap["unit_cost"] = estoque_snap["unit_cost"].fillna(0.0)
estoque_snap["valor_estoque_estimado"] = estoque_snap["quantidade_estoque"] * estoque_snap["unit_cost"]

# ==========================
# KPIs principais
# ==========================
total_receita = float(vendas_f["valor_total"].sum()) if not vendas_f.empty else 0.0
total_qtd_vendida = int(vendas_f["quantidade_vendida"].sum()) if not vendas_f.empty else 0

total_gasto_compras = float(compras_f["valor_total"].sum()) if not compras_f.empty else 0.0
total_qtd_comprada = int(compras_f["quantidade_comprada"].sum()) if not compras_f.empty else 0

valor_total_estoque = float(estoque_snap["valor_estoque_estimado"].sum()) if not estoque_snap.empty else 0.0
qtd_total_estoque = int(estoque_snap["quantidade_estoque"].sum()) if not estoque_snap.empty else 0

estoque_critico_df = estoque_snap[estoque_snap["quantidade_estoque"] < estoque_snap["estoque_minimo"]].copy()
qtd_produtos_criticos = int(estoque_critico_df["produto_id"].nunique()) if not estoque_critico_df.empty else 0

# ==========================
# Layout: KPIs
# ==========================
# Primeira linha de KPIs
c1, c2, c3 = st.columns(3)

c1.metric("Receita total (período)", f"R$ {int(total_receita):,}".replace(",", "."))
c2.metric("Qtd vendida (período)", f"{total_qtd_vendida:,}".replace(",", "."))
c3.metric("Gasto total em compras (período)", f"R$ {int(total_gasto_compras):,}".replace(",", "."))

# Segunda linha de KPIs
c4, c5, c6 = st.columns(3)

c4.metric("Valor total de estoque (estimado)", f"R$ {int(valor_total_estoque):,}".replace(",", "."))
c5.metric("Produtos em estoque crítico", f"{qtd_produtos_criticos}")
c6.metric("Qtd total em estoque", f"{qtd_total_estoque:,}".replace(",", "."))

st.divider()

# ==========================
# Painel 360° do Produto
# ==========================
st.subheader("Visão 360° do Produto")

col_left, col_right = st.columns([1, 1])

with col_left:
    # Seleção de 1 produto para a visão 360°
    prod_360_options = dim_prod[dim_prod["produto_id"].isin(prod_ids)].sort_values("produto_id")
    default_idx = 0 if len(prod_360_options) > 0 else None
    selected_prod_360 = st.selectbox(
        "Selecione um produto (visão detalhada)",
        options=prod_360_options["produto"].tolist(),
        index=default_idx if default_idx is not None else 0,
    )
    prod_360_id = int(dim_prod.loc[dim_prod["produto"] == selected_prod_360, "produto_id"].iloc[0])

with col_right:
    # status e alertas
    snap_prod = estoque_snap[estoque_snap["produto_id"] == prod_360_id].copy()
    qtd_atual = int(snap_prod["quantidade_estoque"].sum()) if not snap_prod.empty else 0
    qtd_min = int(snap_prod["estoque_minimo"].max()) if not snap_prod.empty else 0
    if qtd_atual < qtd_min:
        st.error(f"⚠️ Risco de ruptura: estoque atual ({qtd_atual}) abaixo do mínimo ({qtd_min}).")
    elif qtd_atual > 3 * max(qtd_min, 1):
        st.warning(f"⚠️ Possível excesso de estoque: estoque atual ({qtd_atual}) bem acima do mínimo ({qtd_min}).")
    else:
        st.success(f"✅ Estoque dentro do esperado (atual {qtd_atual} / mínimo {qtd_min}).")

# Cards do produto
vendas_prod = vendas_f[vendas_f["produto_id"] == prod_360_id]
compras_prod = compras_f[compras_f["produto_id"] == prod_360_id]

vendas_acum = float(vendas_prod["valor_total"].sum()) if not vendas_prod.empty else 0.0
qtd_vendida_prod = int(vendas_prod["quantidade_vendida"].sum()) if not vendas_prod.empty else 0
compras_recebidas_val = float(compras_prod["valor_total"].sum()) if not compras_prod.empty else 0.0
qtd_comprada_prod = int(compras_prod["quantidade_comprada"].sum()) if not compras_prod.empty else 0

forn_principal = "-"
if not compras_prod.empty:
    # fornecedor principal: maior gasto no período
    by_f = compras_prod.groupby("fornecedor", as_index=False)["valor_total"].sum().sort_values("valor_total", ascending=False)
    if not by_f.empty:
        forn_principal = str(by_f.iloc[0]["fornecedor"])

cc1, cc2, cc3, cc4 = st.columns(4)
cc1.metric("Estoque atual", f"{qtd_atual}")
cc1.caption("Somando locais (última referência).")
cc2.metric("Estoque mínimo", f"{qtd_min}")
cc3.metric("Vendas (R$) no período", f"R$ {vendas_acum:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
cc3.caption(f"Qtd vendida: {qtd_vendida_prod}")
cc4.metric("Compras (R$) no período", f"R$ {compras_recebidas_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
cc4.caption(f"Fornecedor principal: {forn_principal}")

# Detalhe de estoque por local
with st.expander("Detalhe do estoque por localização (última referência)"):
    if snap_prod.empty:
        st.info("Sem registros de estoque para o produto no snapshot mais recente.")
    else:
        show_cols = ["localizacao", "quantidade_estoque", "estoque_minimo", "unit_cost", "valor_estoque_estimado"]
        st.dataframe(
            snap_prod[show_cols].sort_values("localizacao"),
            use_container_width=True,
        )

st.divider()

# ==========================
# Indicadores estratégicos
# ==========================
st.subheader("Indicadores Estratégicos")

i1, i2 = st.columns([1, 1])

with i1:
    st.markdown("**Produtos com Estoque Crítico (abaixo do mínimo)**")
    if estoque_critico_df.empty:
        st.write("Sem produtos críticos no snapshot mais recente.")
    else:
        crit = (
            estoque_critico_df.groupby(["produto_id", "produto", "categoria"], as_index=False)
            .agg(
                estoque_atual=("quantidade_estoque", "sum"),
                estoque_minimo=("estoque_minimo", "max"),
                valor_estoque_estimado=("valor_estoque_estimado", "sum"),
            )
            .sort_values(["estoque_atual", "estoque_minimo"])
        )
        st.dataframe(crit, use_container_width=True, hide_index=True)

with i2:
    st.markdown("**Top 10 Produtos Mais Vendidos (quantidade)**")
    if vendas_f.empty:
        st.write("Sem vendas no período.")
    else:
        top10 = (
            vendas_f.groupby(["produto_id", "produto", "categoria"], as_index=False)["quantidade_vendida"]
            .sum()
            .sort_values("quantidade_vendida", ascending=False)
            .head(10)
        )
        st.dataframe(top10, use_container_width=True, hide_index=True)

j1, j2 = st.columns([1, 1])

with j1:
    st.markdown("**Produtos com Maior Gasto em Compras (R$)**")
    if compras_f.empty:
        st.write("Sem compras no período.")
    else:
        top_spend = (
            compras_f.groupby(["produto_id", "produto", "categoria"], as_index=False)["valor_total"]
            .sum()
            .sort_values("valor_total", ascending=False)
            .head(10)
        )
        st.dataframe(top_spend, use_container_width=True, hide_index=True)

with j2:
    st.markdown("**Tempo Médio de Reposição por Fornecedor (dias)**")
    if compras_f.empty:
        st.write("Sem compras no período.")
    else:
        rep = (
            compras_f.groupby("fornecedor", as_index=False)
            .agg(
                prazo_medio=("prazo_entrega_dias", "mean"),
                volume=("quantidade_comprada", "sum"),
                gasto=("valor_total", "sum"),
                preco_medio=("valor_unitario", "mean"),
            )
            .sort_values("prazo_medio")
        )
        rep["prazo_medio"] = rep["prazo_medio"].round(2)
        rep["preco_medio"] = rep["preco_medio"].round(2)
        st.dataframe(rep, use_container_width=True, hide_index=True)

st.divider()

# ==========================
# Gráficos de suporte à decisão
# ==========================
st.subheader("Gráficos de Suporte à Decisão")

g1, g2 = st.columns([1.2, 1])

# 1) Série temporal: vendas vs compras por mês
with g1:
    st.markdown("**Série temporal mensal: vendas (qtd) vs compras (qtd)**")

    vendas_ts = vendas_f.dropna(subset=["data_venda"]).copy()
    compras_ts = compras_f.dropna(subset=["data_compra"]).copy()

    if vendas_ts.empty and compras_ts.empty:
        st.write("Sem dados para o período selecionado.")
    else:
        if not vendas_ts.empty:
            vendas_ts["mes"] = month_key(vendas_ts["data_venda"])
            v_m = vendas_ts.groupby("mes", as_index=False)["quantidade_vendida"].sum().rename(
                columns={"quantidade_vendida": "vendas_qtd"}
            )
        else:
            v_m = pd.DataFrame({"mes": [], "vendas_qtd": []})

        if not compras_ts.empty:
            compras_ts["mes"] = month_key(compras_ts["data_compra"])
            c_m = compras_ts.groupby("mes", as_index=False)["quantidade_comprada"].sum().rename(
                columns={"quantidade_comprada": "compras_qtd"}
            )
        else:
            c_m = pd.DataFrame({"mes": [], "compras_qtd": []})

        ts = pd.merge(v_m, c_m, on="mes", how="outer").fillna(0).sort_values("mes")

        fig = plt.figure()
        plt.plot(ts["mes"], ts["vendas_qtd"], marker="o")
        plt.plot(ts["mes"], ts["compras_qtd"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Mês")
        plt.ylabel("Quantidade")
        plt.legend(["Vendas (qtd)", "Compras (qtd)"])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# 2) Comparativo de fornecedores: preço médio, prazo médio, volume
with g2:
    st.markdown("**Fornecedores: preço médio x prazo médio (tamanho = volume)**")
    if compras_f.empty:
        st.write("Sem compras no período.")
    else:
        sup = (
            compras_f.groupby("fornecedor", as_index=False)
            .agg(
                preco_medio=("valor_unitario", "mean"),
                prazo_medio=("prazo_entrega_dias", "mean"),
                volume=("quantidade_comprada", "sum"),
            )
            .sort_values("volume", ascending=False)
        )
        sup["preco_medio"] = sup["preco_medio"].astype(float)
        sup["prazo_medio"] = sup["prazo_medio"].astype(float)
        sup["volume"] = sup["volume"].astype(float)

        # normaliza tamanho (sem escolher cor)
        sizes = 50 + 450 * (sup["volume"] / max(sup["volume"].max(), 1.0))

        fig = plt.figure()
        plt.scatter(sup["prazo_medio"], sup["preco_medio"], s=sizes, alpha=0.6)
        for _, r in sup.head(12).iterrows():  # evita poluição visual
            plt.annotate(str(r["fornecedor"]), (r["prazo_medio"], r["preco_medio"]), fontsize=8)
        plt.xlabel("Prazo médio (dias)")
        plt.ylabel("Preço médio (R$)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# 3) Heatmap: relação estoque x vendas x compras (top produtos)
st.markdown("**Heatmap (normalizado) — Estoque x Vendas x Compras (Top produtos por receita)**")

# agrega por produto
prod_metrics = dim_prod[dim_prod["produto_id"].isin(prod_ids)][["produto_id", "produto", "categoria"]].copy()

# Estoque (snapshot)
if not estoque_snap.empty:
    est_by = estoque_snap.groupby("produto_id", as_index=False)["quantidade_estoque"].sum().rename(
        columns={"quantidade_estoque": "estoque_qtd"}
    )
else:
    est_by = pd.DataFrame({"produto_id": [], "estoque_qtd": []})

# Vendas no período
if not vendas_f.empty:
    ven_by = vendas_f.groupby("produto_id", as_index=False).agg(
        vendas_qtd=("quantidade_vendida", "sum"),
        receita=("valor_total", "sum"),
    )
else:
    ven_by = pd.DataFrame({"produto_id": [], "vendas_qtd": [], "receita": []})

# Compras no período
if not compras_f.empty:
    com_by = compras_f.groupby("produto_id", as_index=False).agg(
        compras_qtd=("quantidade_comprada", "sum"),
        gasto=("valor_total", "sum"),
    )
else:
    com_by = pd.DataFrame({"produto_id": [], "compras_qtd": [], "gasto": []})

m = prod_metrics.merge(est_by, on="produto_id", how="left").merge(ven_by, on="produto_id", how="left").merge(com_by, on="produto_id", how="left")
m = m.fillna(0)

# Seleciona top N por receita para visualização
N = 15
m_top = m.sort_values("receita", ascending=False).head(N).copy()
if m_top.empty:
    st.write("Sem dados suficientes para montar o heatmap.")
else:
    mat = m_top[["estoque_qtd", "vendas_qtd", "compras_qtd"]].astype(float).to_numpy()
    # normalização min-max por coluna
    denom = np.where((mat.max(axis=0) - mat.min(axis=0)) == 0, 1.0, (mat.max(axis=0) - mat.min(axis=0)))
    mat_n = (mat - mat.min(axis=0)) / denom

    fig = plt.figure()
    plt.imshow(mat_n, aspect="auto")
    plt.yticks(ticks=np.arange(len(m_top)), labels=m_top["produto"].tolist())
    plt.xticks(ticks=[0, 1, 2], labels=["Estoque", "Vendas", "Compras"])
    plt.xlabel("Métrica (normalizada)")
    plt.ylabel("Produto (Top por receita)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()

with st.expander("Ver tabelas filtradas (debug/validação)"):
    st.markdown("**Vendas (filtradas)**")
    st.dataframe(vendas_f.head(200), use_container_width=True)

    st.markdown("**Compras (filtradas)**")
    st.dataframe(compras_f.head(200), use_container_width=True)

    st.markdown("**Estoque (snapshot mais recente, filtrado)**")
    st.dataframe(estoque_snap.head(200), use_container_width=True)
