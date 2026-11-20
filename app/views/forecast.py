import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BLUE, GREY, agg_band, band_chart, fmt, load, metrics, setup, style

setup("Prévisions (16 jours)", "📈")
q = metrics()
fc_all = load("forecast")
hist_all = load("history_recent")

c1, c2, c3 = st.columns([1, 1, 2])
level = c1.radio("Niveau", ["Total", "Magasin", "Famille", "Magasin × famille"], horizontal=False)
store = fam = None
if level in ("Magasin", "Magasin × famille"):
    store = c2.selectbox("Magasin", sorted(fc_all.store_nbr.unique()))
if level in ("Famille", "Magasin × famille"):
    fam = c3.selectbox("Famille", sorted(fc_all.family.unique()))

fc, hi = fc_all, hist_all
if store is not None:
    fc, hi = fc[fc.store_nbr == store], hi[hi.store_nbr == store]
if fam is not None:
    fc, hi = fc[fc.family == fam], hi[hi.family == fam]
fcd = agg_band(fc)
hid = hi.groupby("date", as_index=False)["sales"].sum()
st.plotly_chart(band_chart(hid, fcd), width="stretch")

tot, last16 = fcd["mean"].sum(), hid[hid.date > hid.date.max() - pd.Timedelta(days=16)].sales.sum()
k = st.columns(4)
k[0].metric("Prévision 16 j", fmt(tot), f"{(tot / last16 - 1) * 100:+.1f} % vs 16 derniers j" if last16 else None)
band16 = agg_band(fc.assign(k=1), "k").iloc[0]
k[1].metric("Bas (p10, approx.)", fmt(band16.p10))
k[2].metric("Haut (p90, approx.)", fmt(band16.p90))
k[3].metric("Articles en promo prévus", fmt(fc.onpromotion.sum()))

st.divider()
st.subheader("Qualité du modèle (backtest sur les 16 derniers jours connus)")
l, r = st.columns([1, 1])
mt = pd.DataFrame(q["models"]).T.reset_index().rename(columns={"index": "Modèle", "rmsle": "RMSLE", "wape_pct": "WAPE %"})
l.dataframe(mt, hide_index=True, width="stretch")
l.caption(f"Fenêtre {q['validation_window'][0]} → {q['validation_window'][1]} · RMSLE = métrique officielle Kaggle "
          f"(plus bas = meilleur) · couverture réelle de l'intervalle p10–p90 : {q['interval_coverage_p10_p90_pct']} % (cible 80 %).")
fi = load("feature_importance").head(10).iloc[::-1]
fig = go.Figure(go.Bar(x=fi.gain_pct, y=fi.feature, orientation="h", marker_color=BLUE))
r.plotly_chart(style(fig, 300, legend=False).update_layout(title="Importance des variables (gain %)"), width="stretch")

val = load("validation")
if store is not None:
    val = val[val.store_nbr == store]
if fam is not None:
    val = val[val.family == fam]
vd = val.groupby("date", as_index=False)[["actual", "pred", "naive", "ma28"]].sum()
fig = go.Figure()
fig.add_trace(go.Scatter(x=vd.date, y=vd.actual, name="Réel", line=dict(color=GREY, width=2.5)))
fig.add_trace(go.Scatter(x=vd.date, y=vd.pred, name="LightGBM", line=dict(color=BLUE, width=2)))
fig.add_trace(go.Scatter(x=vd.date, y=vd.ma28, name="Moyenne mobile 28 j", line=dict(color=AMBER, width=1.5, dash="dot")))
st.plotly_chart(style(fig, 320).update_layout(title="Backtest : réel vs prévu (sélection courante)"), width="stretch")

st.subheader("Plus fortes variations prévues")
a, b = st.columns(2)
ff = load("forecast_by_family").sort_values("change_vs_last_16d_pct", ascending=False)
cols = {"family": "Famille", "forecast_16d": "Prévision 16 j", "last_16d": "16 derniers j", "change_vs_last_16d_pct": "Variation %", "uncertainty_pct": "Incertitude %"}
a.dataframe(ff[list(cols)].rename(columns=cols), hide_index=True, width="stretch", height=330)
fs = load("forecast_by_store").sort_values("change_vs_last_16d_pct", ascending=False)
cols = {"store_nbr": "Magasin", "forecast_16d": "Prévision 16 j", "last_16d": "16 derniers j", "change_vs_last_16d_pct": "Variation %", "uncertainty_pct": "Incertitude %"}
b.dataframe(fs[list(cols)].rename(columns=cols), hide_index=True, width="stretch", height=330)
