"""Outils déterministes utilisés par les agents.

Règle d'or : le LLM ne calcule jamais un chiffre. Chaque chiffre vient d'un de ces
outils (Pandas sur les artefacts Parquet), et chaque appel est enregistré dans un
journal de preuves (E1, E2, …) que l'agent critique peut rejouer et vérifier.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


# --------------------------------------------------------------------------- #
# Accès aux artefacts
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / f"{name}.parquet")


@lru_cache(maxsize=8)
def jfile(name: str) -> dict:
    return json.loads((ARTIFACTS / f"{name}.json").read_text())


def _clean(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (v is None or math.isnan(v)) else round(float(v), 2)
    if isinstance(v, (pd.Timestamp,)):
        return str(v.date())
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _records(df: pd.DataFrame, cols: list[str] | None = None) -> list[dict]:
    d = df[cols] if cols else df
    return [_clean(r) for r in d.to_dict("records")]


def _family(name: str) -> str:
    fams = table("family_kpis").family.tolist()
    up = str(name).strip().upper()
    if up in fams:
        return up
    hits = [f for f in fams if up in f]
    if not hits:
        raise ValueError(f"Famille inconnue : {name}. Familles valides : {', '.join(fams)}")
    return hits[0]


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def overview() -> dict:
    """KPIs globaux des 28 derniers jours et tendance hebdomadaire récente."""
    m = jfile("meta")
    dt = table("daily_total").set_index("date")["sales"].resample("W").sum().tail(9).iloc[:-1]
    return {
        "periode_donnees": f"{m['first_date']} -> {m['last_date']}",
        "magasins": m["n_stores"], "familles": m["n_families"],
        "ventes_28j": m["sales_28d"], "ventes_28j_precedents": m["sales_prev_28d"],
        "croissance_28j_pct": m["growth_28d_pct"], "croissance_vs_annee_precedente_pct": m["yoy_pct"],
        "ventes_moyennes_par_jour_28j": m["avg_daily_sales_28d"], "transactions_28j": m["transactions_28d"],
        "effet_jours_de_paie_pct": m["payday_effect_pct"],
        "correlation_hebdo_petrole_ventes": m["oil_sales_weekly_corr"],
        "part_lignes_en_promo_28j_pct": m["promo_rows_pct_28d"],
        "ventes_hebdo_8_dernieres_semaines": [round(float(x), 1) for x in dt.values],
    }


def rank_stores(metric: str = "growth_28d_pct", n: int = 5, ascending: bool = False) -> dict:
    """Classe les magasins selon un KPI (sales_28d, growth_28d_pct, yoy_pct, gap_vs_cluster_pct, sales_per_txn)."""
    allowed = ["sales_28d", "growth_28d_pct", "yoy_pct", "gap_vs_cluster_pct", "sales_per_txn", "zero_sales_days_28d"]
    if metric not in allowed:
        raise ValueError(f"metric doit être dans {allowed}")
    df = table("store_kpis").dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(int(n))
    return {"metric": metric, "ordre": "croissant" if ascending else "décroissant",
            "magasins": _records(df, ["store_nbr", "city", "type", "cluster", metric, "sales_28d"])}


def underperforming_stores(n: int = 5) -> dict:
    """Magasins les plus en retard sur la médiane de leur cluster (magasins comparables)."""
    df = table("store_kpis")
    df = df[df.stores_in_cluster >= 2].sort_values("gap_vs_cluster_pct").head(int(n))
    return {"definition": "écart des ventes 28 j vs médiane du cluster",
            "magasins": _records(df, ["store_nbr", "city", "cluster", "sales_28d", "cluster_median_28d",
                                      "gap_vs_cluster_pct", "growth_28d_pct", "sales_per_txn"])}


def store_profile(store_nbr: int) -> dict:
    """Fiche complète d'un magasin : KPIs, prévision 16 j, 3 familles qui progressent / reculent."""
    s = table("store_kpis")
    row = s[s.store_nbr == int(store_nbr)]
    if row.empty:
        raise ValueError(f"Magasin {store_nbr} inconnu (1..{s.store_nbr.max()})")
    f = table("forecast_by_store")
    fr = f[f.store_nbr == int(store_nbr)]
    w = table("weekly_store_family")
    w = w[w.store_nbr == int(store_nbr)]
    weeks = sorted(w.week.unique())[-9:-1]
    cur, prev = w[w.week.isin(weeks[-4:])], w[w.week.isin(weeks[:4])]
    ch = (cur.groupby("family").sales.sum() / prev.groupby("family").sales.sum().replace(0, np.nan) - 1) * 100
    ch = ch.dropna().round(1).sort_values()
    return {"kpis": _records(row)[0],
            "prevision_16j": _records(fr, ["forecast_16d", "p10_16d", "p90_16d", "last_16d", "change_vs_last_16d_pct"])[0] if len(fr) else None,
            "familles_en_baisse_4sem_pct": {k: float(v) for k, v in ch.head(3).items()},
            "familles_en_hausse_4sem_pct": {k: float(v) for k, v in ch.tail(3).items()}}


def rank_families(metric: str = "share_pct", n: int = 5, ascending: bool = False) -> dict:
    """Classe les familles selon un KPI (sales_28d, share_pct, growth_28d_pct, yoy_pct, volatility_cv, stores_zero_sales_28d)."""
    allowed = ["sales_28d", "share_pct", "growth_28d_pct", "yoy_pct", "volatility_cv", "stores_zero_sales_28d", "promo_rows_pct_90d"]
    if metric not in allowed:
        raise ValueError(f"metric doit être dans {allowed}")
    df = table("family_kpis").dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(int(n))
    return {"metric": metric, "ordre": "croissant" if ascending else "décroissant",
            "familles": _records(df, ["family", metric, "sales_28d", "abc_class"])}


def family_profile(family: str) -> dict:
    """Fiche complète d'une famille : KPIs, effet promo, prévision 16 j."""
    fam = _family(family)
    k = table("family_kpis")
    pu = table("promo_uplift")
    fc = table("forecast_by_family")
    return {"kpis": _records(k[k.family == fam])[0],
            "effet_promo_pct": _clean(pu.loc[pu.family == fam, "promo_uplift_pct"].iloc[0]) if (pu.family == fam).any() else None,
            "prevision_16j": _records(fc[fc.family == fam], ["forecast_16d", "p10_16d", "p90_16d", "last_16d",
                                                            "change_vs_last_16d_pct", "change_vs_ly_pct", "uncertainty_pct"])[0]}


def promo_effect(n: int = 5) -> dict:
    """Familles où la promotion augmente le plus / le moins les ventes (vs même magasin, même jour de semaine hors promo)."""
    pu = table("promo_uplift").merge(table("family_kpis")[["family", "promo_rows_pct_90d", "abc_class"]], on="family")
    cols = ["family", "promo_uplift_pct", "promo_rows_pct_90d", "abc_class"]
    return {"methode": "médiane(ventes en promo / moyenne hors promo même magasin-famille-jour) - 1, sur 365 j",
            "plus_reactives": _records(pu.head(int(n)), cols),
            "moins_reactives": _records(pu.tail(int(n)).iloc[::-1], cols)}


def holiday_effects() -> dict:
    """Effet historique des jours fériés nationaux sur les ventes totales, et fériés à venir sur l'horizon de prévision."""
    he = table("holiday_effect")
    up = table("holidays_upcoming")
    return {"effet_historique": _records(he, ["holiday", "occurrences", "effect_pct"]) if len(he) else [],
            "a_venir_horizon_prevision": _records(up, ["date", "type", "locale", "locale_name", "description"]) if len(up) else []}


def model_quality() -> dict:
    """Qualité du modèle de prévision sur la fenêtre de validation, comparé aux modèles de référence."""
    m = jfile("metrics")
    fi = table("feature_importance").head(6)
    return {"fenetre_validation": m["validation_window"], "horizon_jours": m["horizon_days"],
            "modeles": m["models"], "couverture_intervalle_p10_p90_pct": m["interval_coverage_p10_p90_pct"],
            "variables_les_plus_importantes_pct": {r["feature"]: r["gain_pct"] for r in _records(fi)}}


def forecast_overview() -> dict:
    """Prévision totale des 16 prochains jours vs 16 derniers jours et même période l'an dernier."""
    m = jfile("meta")
    fb = table("forecast_by_family")
    d = table("forecast_daily_total")
    tot, last, ly = fb.forecast_16d.sum(), fb.last_16d.sum(), fb.ly_same_16d.sum()
    peak = d.loc[d["mean"].idxmax()]
    return {"periode": f"{m['forecast_start']} -> {m['forecast_end']}",
            "prevision_totale_16j": round(float(tot), 1),
            "intervalle_p10_p90": [round(float(fb.p10_16d.sum()), 1), round(float(fb.p90_16d.sum()), 1)],
            "ventes_16_derniers_jours": round(float(last), 1),
            "variation_vs_16_derniers_jours_pct": round(float((tot / last - 1) * 100), 1) if last else None,
            "variation_vs_annee_precedente_pct": round(float((tot / ly - 1) * 100), 1) if ly else None,
            "jour_pic": str(pd.Timestamp(peak["date"]).date()), "ventes_jour_pic": round(float(peak["mean"]), 1)}


def forecast_movers(level: str = "family", n: int = 5) -> dict:
    """Plus fortes hausses / baisses prévues (16 j vs 16 derniers jours), par 'family' ou 'store'."""
    key = "family" if level.startswith("fam") else "store_nbr"
    f = table("forecast_by_family" if key == "family" else "forecast_by_store")
    f = f[f.last_16d > f.last_16d.quantile(0.2)].dropna(subset=["change_vs_last_16d_pct"])
    cols = [key, "forecast_16d", "last_16d", "change_vs_last_16d_pct", "uncertainty_pct"]
    s = f.sort_values("change_vs_last_16d_pct")
    return {"niveau": key, "hausses": _records(s.tail(int(n)).iloc[::-1], cols), "baisses": _records(s.head(int(n)), cols)}


def uncertain_families(n: int = 5) -> dict:
    """Familles importantes (classe A/B) dont la prévision est la plus incertaine (largeur p10-p90 / prévision)."""
    f = table("forecast_by_family").merge(table("family_kpis")[["family", "abc_class"]], on="family")
    f = f[f.abc_class.isin(["A", "B"])].sort_values("uncertainty_pct", ascending=False).head(int(n))
    return {"familles": _records(f, ["family", "abc_class", "forecast_16d", "p10_16d", "p90_16d", "uncertainty_pct"])}


def stockout_signals(n: int = 5) -> dict:
    """Signaux de rupture / fermeture : jours à zéro vente par magasin et magasins sans vente par famille (28 j)."""
    s = table("store_kpis").sort_values("zero_sales_days_28d", ascending=False)
    s = s[s.zero_sales_days_28d > 0].head(int(n))
    f = table("family_kpis").sort_values("stores_zero_sales_28d", ascending=False)
    f = f[f.stores_zero_sales_28d > 0].head(int(n))
    return {"magasins_jours_sans_vente": _records(s, ["store_nbr", "city", "zero_sales_days_28d"]),
            "familles_magasins_sans_vente": _records(f, ["family", "stores_zero_sales_28d", "abc_class"])}


TOOLS = {
    "overview": overview, "rank_stores": rank_stores, "underperforming_stores": underperforming_stores,
    "store_profile": store_profile, "rank_families": rank_families, "family_profile": family_profile,
    "promo_effect": promo_effect, "holiday_effects": holiday_effects, "model_quality": model_quality,
    "forecast_overview": forecast_overview, "forecast_movers": forecast_movers,
    "uncertain_families": uncertain_families, "stockout_signals": stockout_signals,
}

# Schémas pour l'API « tool use » de Claude (agent Q&A)
_INT = {"type": "integer"}
TOOL_SCHEMAS = [
    {"name": "overview", "description": overview.__doc__, "input_schema": {"type": "object", "properties": {}}},
    {"name": "rank_stores", "description": rank_stores.__doc__, "input_schema": {"type": "object", "properties": {
        "metric": {"type": "string", "enum": ["sales_28d", "growth_28d_pct", "yoy_pct", "gap_vs_cluster_pct", "sales_per_txn", "zero_sales_days_28d"]},
        "n": _INT, "ascending": {"type": "boolean"}}, "required": ["metric"]}},
    {"name": "underperforming_stores", "description": underperforming_stores.__doc__, "input_schema": {"type": "object", "properties": {"n": _INT}}},
    {"name": "store_profile", "description": store_profile.__doc__, "input_schema": {"type": "object", "properties": {"store_nbr": _INT}, "required": ["store_nbr"]}},
    {"name": "rank_families", "description": rank_families.__doc__, "input_schema": {"type": "object", "properties": {
        "metric": {"type": "string", "enum": ["sales_28d", "share_pct", "growth_28d_pct", "yoy_pct", "volatility_cv", "stores_zero_sales_28d", "promo_rows_pct_90d"]},
        "n": _INT, "ascending": {"type": "boolean"}}, "required": ["metric"]}},
    {"name": "family_profile", "description": family_profile.__doc__, "input_schema": {"type": "object", "properties": {"family": {"type": "string"}}, "required": ["family"]}},
    {"name": "promo_effect", "description": promo_effect.__doc__, "input_schema": {"type": "object", "properties": {"n": _INT}}},
    {"name": "holiday_effects", "description": holiday_effects.__doc__, "input_schema": {"type": "object", "properties": {}}},
    {"name": "model_quality", "description": model_quality.__doc__, "input_schema": {"type": "object", "properties": {}}},
    {"name": "forecast_overview", "description": forecast_overview.__doc__, "input_schema": {"type": "object", "properties": {}}},
    {"name": "forecast_movers", "description": forecast_movers.__doc__, "input_schema": {"type": "object", "properties": {
        "level": {"type": "string", "enum": ["family", "store"]}, "n": _INT}}},
    {"name": "uncertain_families", "description": uncertain_families.__doc__, "input_schema": {"type": "object", "properties": {"n": _INT}}},
    {"name": "stockout_signals", "description": stockout_signals.__doc__, "input_schema": {"type": "object", "properties": {"n": _INT}}},
]


# --------------------------------------------------------------------------- #
# Journal de preuves
# --------------------------------------------------------------------------- #
class EvidenceLog:
    def __init__(self, items: list[dict] | None = None):
        self.items: list[dict] = list(items or [])

    def call(self, tool: str, **args) -> dict:
        try:
            result = TOOLS[tool](**args)
        except Exception as e:  # l'erreur devient elle-même une preuve lisible par l'agent
            result = {"erreur": str(e)}
        eid = f"E{len(self.items) + 1}"
        item = {"id": eid, "tool": tool, "args": args, "result": _clean(result)}
        self.items.append(item)
        return item

    def get(self, eid: str) -> dict | None:
        return next((e for e in self.items if e["id"] == eid), None)

    def compact(self, ids: list[str] | None = None) -> str:
        sel = [e for e in self.items if ids is None or e["id"] in ids]
        return "\n".join(f"[{e['id']}] {e['tool']}({json.dumps(e['args'], ensure_ascii=False)}) = "
                         f"{json.dumps(e['result'], ensure_ascii=False)}" for e in sel)

    def replay(self, eid: str) -> bool:
        """Rejoue l'appel et vérifie que le résultat est identique (preuve reproductible)."""
        e = self.get(eid)
        if not e:
            return False
        try:
            return _clean(TOOLS[e["tool"]](**e["args"])) == e["result"]
        except Exception:
            return False
