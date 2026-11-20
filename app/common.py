"""Fonctions partagées par toutes les pages du dashboard."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"
RUN_FILE = ART / "agent_run.json"

BLUE, BLUE_LIGHT, AMBER, GREEN, RED, GREY = "#2563EB", "rgba(37,99,235,0.18)", "#D97706", "#16A34A", "#DC2626", "#94A3B8"


def setup(title: str, icon: str = "📊"):
    # secrets Streamlit -> variables d'environnement (lues par agents/llm.py)
    try:
        for k in ("GEMINI_API_KEY", "GEMINI_MODEL", "ANTHROPIC_API_KEY", "CLAUDE_MODEL", "LLM_PROVIDER", "APP_PASSWORD"):
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass
    _gate()
    with st.sidebar:
        m = meta()
        st.markdown("### Favorita · aide à la décision")
        st.caption(f"Données : {m['first_date']} → {m['last_date']}\n\nPrévision : {m['forecast_start']} → {m['forecast_end']}")
        if m.get("source") == "synthetic":
            st.warning("Données synthétiques (test). Lancez le notebook Kaggle pour les vraies données.", icon="⚠️")
        from agents import llm
        st.caption("Moteur LLM : " + llm.label())
    st.title(title)


def _gate():
    pw = os.getenv("APP_PASSWORD")
    if not pw or st.session_state.get("auth"):
        return
    st.markdown("### 🔒 Accès protégé")
    v = st.text_input("Mot de passe", type="password")
    if v == pw:
        st.session_state.auth = True
        st.rerun()
    elif v:
        st.error("Mot de passe incorrect")
    st.stop()


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(ART / f"{name}.parquet")


@st.cache_data
def meta() -> dict:
    return json.loads((ART / "meta.json").read_text())


@st.cache_data
def metrics() -> dict:
    return json.loads((ART / "metrics.json").read_text())


def fmt(x, d=0) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:,.{d}f}".replace(",", " ")


def style(fig: go.Figure, h: int = 360, legend=True) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0) if legend else None, showlegend=legend,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


def agg_band(fc: pd.DataFrame, key="date") -> pd.DataFrame:
    """Somme des prévisions avec intervalle combiné (indépendance des erreurs) — cf. pipeline.build.agg_band."""
    import numpy as np
    d = fc.assign(lo2=(fc["mean"] - fc.p10).clip(lower=0) ** 2, hi2=(fc.p90 - fc["mean"]).clip(lower=0) ** 2)
    g = d.groupby(key).agg(mean=("mean", "sum"), lo2=("lo2", "sum"), hi2=("hi2", "sum")).reset_index()
    g["p10"] = (g["mean"] - np.sqrt(g.lo2)).clip(lower=0)
    g["p90"] = g["mean"] + np.sqrt(g.hi2)
    return g


def band_chart(hist: pd.DataFrame, fc: pd.DataFrame, h=380, title="") -> go.Figure:
    """Historique (sales) + prévision (mean, p10, p90)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.date, y=fc.p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.date, y=fc.p10, mode="lines", fill="tonexty", fillcolor=BLUE_LIGHT, line=dict(width=0),
                             name="Intervalle p10–p90 (approx.)"))
    fig.add_trace(go.Scatter(x=hist.date, y=hist.sales, mode="lines", name="Ventes réelles", line=dict(color=GREY, width=1.6)))
    fig.add_trace(go.Scatter(x=fc.date, y=fc["mean"], mode="lines", name="Prévision", line=dict(color=BLUE, width=2.4)))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return style(fig, h)
