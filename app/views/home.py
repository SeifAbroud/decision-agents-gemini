import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BLUE, GREY, band_chart, fmt, load, meta, setup, style

setup("Vue d'ensemble", "📊")
m = meta()

c = st.columns(5)
c[0].metric("Ventes 28 jours", fmt(m["sales_28d"]), f"{m['growth_28d_pct']:+.1f} % vs 28 j préc.")
c[1].metric("Vs année précédente", f"{m['yoy_pct']:+.1f} %" if m.get("yoy_pct") is not None else "–")
c[2].metric("Ventes / jour", fmt(m["avg_daily_sales_28d"]))
c[3].metric("Transactions 28 j", fmt(m["transactions_28d"]))
c[4].metric("Effet jours de paie", f"{m['payday_effect_pct']:+.1f} %")

st.subheader("Ventes totales et prévision des 16 prochains jours")
dt = load("daily_total")
hist = dt[dt.date > dt.date.max() - pd.Timedelta(days=120)]
fc = load("forecast_daily_total")
st.plotly_chart(band_chart(hist, fc), width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Familles : part des ventes (28 j)")
    fk = load("family_kpis").head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=fk.share_pct, y=fk.family, orientation="h", marker_color=BLUE,
                           text=fk.share_pct.map(lambda v: f"{v:.1f} %"), textposition="outside",
                           hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
    st.plotly_chart(style(fig, 420, legend=False), width="stretch")
with right:
    st.subheader("Ventes hebdomadaires et promotions")
    wk = dt.set_index("date").resample("W")[["sales", "onpromotion"]].sum().iloc[1:-1].tail(70).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=wk.date, y=wk.onpromotion, name="Articles en promo (axe droit)", marker_color=AMBER,
                         opacity=0.35, yaxis="y2"))
    fig.add_trace(go.Scatter(x=wk.date, y=wk.sales, mode="lines", name="Ventes", line=dict(color=BLUE, width=2.2)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, range=[0, wk.onpromotion.max() * 3]),
                      yaxis=dict(rangemode="tozero"))
    st.plotly_chart(style(fig, 420), width="stretch")

st.subheader("Magasins : croissance 28 j vs écart à leur cluster")
sk = load("store_kpis")
fig = go.Figure(go.Scatter(
    x=sk.gap_vs_cluster_pct, y=sk.growth_28d_pct, mode="markers+text", text=sk.store_nbr, textposition="top center",
    marker=dict(size=(sk.sales_28d / sk.sales_28d.max() * 34).clip(lower=7), color=BLUE, opacity=0.65,
                line=dict(width=0.5, color="white")),
    customdata=sk[["city", "cluster", "sales_28d"]],
    hovertemplate="Magasin %{text} · %{customdata[0]} · cluster %{customdata[1]}<br>Écart cluster %{x:.1f} %"
                  "<br>Croissance %{y:.1f} %<br>Ventes 28 j %{customdata[2]:,.0f}<extra></extra>"))
fig.add_vline(x=0, line_color=GREY, line_dash="dot")
fig.add_hline(y=0, line_color=GREY, line_dash="dot")
fig.update_xaxes(title="Écart à la médiane du cluster (%)")
fig.update_yaxes(title="Croissance 28 j (%)")
st.plotly_chart(style(fig, 440, legend=False), width="stretch")
st.caption("Taille des bulles = ventes 28 j. En bas à gauche : magasins en retard sur leur cluster ET en baisse.")
