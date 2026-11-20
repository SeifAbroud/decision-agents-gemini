import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, GREY, RED, fmt, load, setup, style

setup("Magasins et familles", "🏬")
tab_s, tab_f, tab_p = st.tabs(["Magasins", "Familles", "Promotions et jours fériés"])

with tab_s:
    sk = load("store_kpis")
    cols = {"store_nbr": "Magasin", "city": "Ville", "type": "Type", "cluster": "Cluster", "sales_28d": "Ventes 28 j",
            "growth_28d_pct": "Croiss. 28 j %", "yoy_pct": "Vs an -1 %", "gap_vs_cluster_pct": "Écart cluster %",
            "rank_in_cluster": "Rang cluster", "sales_per_txn": "Ventes/ticket", "zero_sales_days_28d": "Jours à 0"}
    st.dataframe(sk[list(cols)].rename(columns=cols), hide_index=True, width="stretch", height=300,
                 column_config={"Ventes 28 j": st.column_config.NumberColumn(format="%.0f")})
    s = st.selectbox("Détail d'un magasin", sk.store_nbr, format_func=lambda x: f"Magasin {x} · {sk.set_index('store_nbr').city[x]}")
    row = sk[sk.store_nbr == s].iloc[0]
    c = st.columns(4)
    c[0].metric("Ventes 28 j", fmt(row.sales_28d), f"{row.growth_28d_pct:+.1f} %")
    c[1].metric("Écart au cluster", f"{row.gap_vs_cluster_pct:+.1f} %")
    c[2].metric("Rang dans le cluster", f"{row.rank_in_cluster} / {row.stores_in_cluster}")
    c[3].metric("Ventes par ticket", fmt(row.sales_per_txn, 2))
    l, r = st.columns(2)
    ds = load("daily_store")
    d = ds[(ds.store_nbr == s) & (ds.date > ds.date.max() - pd.Timedelta(days=180))]
    fig = go.Figure(go.Scatter(x=d.date, y=d.sales, line=dict(color=BLUE, width=1.5), name="Ventes"))
    l.plotly_chart(style(fig, 320, legend=False).update_layout(title="Ventes quotidiennes (180 j)"), width="stretch")
    peers = sk[sk.cluster == row.cluster].sort_values("sales_28d")
    fig = go.Figure(go.Bar(x=peers.store_nbr.astype(str), y=peers.sales_28d,
                           marker_color=[RED if x == s else GREY for x in peers.store_nbr]))
    fig.add_hline(y=row.cluster_median_28d, line_dash="dot", annotation_text="médiane cluster")
    r.plotly_chart(style(fig, 320, legend=False).update_layout(title=f"Magasins du cluster {row.cluster} (ventes 28 j)"),
                   width="stretch")

with tab_f:
    fk = load("family_kpis")
    cols = {"family": "Famille", "abc_class": "ABC", "sales_28d": "Ventes 28 j", "share_pct": "Part %",
            "growth_28d_pct": "Croiss. 28 j %", "yoy_pct": "Vs an -1 %", "volatility_cv": "Volatilité",
            "promo_rows_pct_90d": "% lignes promo", "stores_zero_sales_28d": "Magasins à 0"}
    st.dataframe(fk[list(cols)].rename(columns=cols), hide_index=True, width="stretch", height=320,
                 column_config={"Ventes 28 j": st.column_config.NumberColumn(format="%.0f")})
    f = st.selectbox("Évolution d'une famille", fk.family)
    df = load("daily_family")
    d = df[df.family == f].set_index("date").resample("W")[["sales"]].sum().iloc[1:-1].reset_index()
    fig = go.Figure(go.Scatter(x=d.date, y=d.sales, line=dict(color=BLUE, width=2)))
    st.plotly_chart(style(fig, 320, legend=False).update_layout(title=f"{f} · ventes hebdomadaires"), width="stretch")

with tab_p:
    l, r = st.columns(2)
    pu = load("promo_uplift").iloc[::-1]
    fig = go.Figure(go.Bar(x=pu.promo_uplift_pct, y=pu.family, orientation="h",
                           marker_color=[BLUE if v >= 0 else RED for v in pu.promo_uplift_pct]))
    l.plotly_chart(style(fig, 720, legend=False).update_layout(title="Effet de la promotion par famille (%)"),
                   width="stretch")
    l.caption("Médiane du rapport ventes en promo / moyenne hors promo (même magasin, famille, jour de semaine), 365 j.")
    he = load("holiday_effect")
    if len(he):
        he = he.sort_values("effect_pct")
        fig = go.Figure(go.Bar(x=he.effect_pct, y=he.holiday, orientation="h",
                               marker_color=[BLUE if v >= 0 else RED for v in he.effect_pct]))
        r.plotly_chart(style(fig, 500, legend=False).update_layout(title="Effet des jours fériés nationaux (%)"),
                       width="stretch")
    up = load("holidays_upcoming")
    r.markdown("**Jours fériés / événements sur l'horizon de prévision**")
    r.dataframe(up, hide_index=True, width="stretch") if len(up) else r.caption("Aucun.")
