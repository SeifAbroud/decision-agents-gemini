"""Système multi-agents (LangGraph) d'aide à la décision.

    analyste descriptif ─► agent prévision ─► recommandeur ─► critique ─┬─► rapporteur
                                                  ▲                     │
                                                  └──── à réviser ──────┘ (max 2 tours)

- Chaque agent appelle des outils déterministes (agents/tools.py) ; chaque appel
  devient une preuve E1, E2, … stockée dans l'état du graphe.
- Le recommandeur doit citer les preuves de chaque recommandation.
- Le critique : (1) vérifie par code que chaque chiffre existe dans les preuves citées
  et que les preuves sont rejouables, (2) juge la logique avec le LLM.
- Sans clé API, chaque agent utilise une version à base de règles (mode hors-ligne).
"""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import llm, rag
from .tools import EvidenceLog, jfile
from .verify import check_numbers

MAX_REVISIONS = 2

STYLE = ("Écris en français. Écris les nombres sans séparateur de milliers et avec un point décimal "
         "(ex. 225780.5 ; -8.7 %). N'invente aucun chiffre : recopie uniquement des chiffres présents dans "
         "les preuves. Cite la preuve entre crochets, ex. [E3].")


class State(TypedDict, total=False):
    mode: str                 # "llm" ou "offline"
    evidence: list[dict]
    desc_insights: str
    fc_insights: str
    recos: list[dict]
    review: list[dict]
    feedback: str
    iteration: int
    executive_summary: str
    trace: list[dict]
    usage: dict


def _log(state: State, agent: str, msg: str, t0: float) -> list[dict]:
    return state.get("trace", []) + [{"agent": agent, "message": msg, "seconds": round(time.time() - t0, 1)}]


_USAGE = llm.Usage()


# --------------------------------------------------------------------------- #
# 1. Analyste descriptif
# --------------------------------------------------------------------------- #
def analyst(state: State) -> State:
    t0 = time.time()
    ev = EvidenceLog(state.get("evidence"))
    ids = [ev.call("overview")["id"],
           ev.call("rank_stores", metric="growth_28d_pct", n=3)["id"],
           ev.call("rank_stores", metric="growth_28d_pct", n=3, ascending=True)["id"],
           ev.call("underperforming_stores", n=3)["id"],
           ev.call("rank_families", metric="share_pct", n=5)["id"],
           ev.call("rank_families", metric="growth_28d_pct", n=3, ascending=True)["id"],
           ev.call("promo_effect", n=3)["id"],
           ev.call("holiday_effects")["id"],
           ev.call("stockout_signals", n=3)["id"]]
    if state["mode"] == "llm":
        text = llm.ask(
            "Tu es l'agent Analyste descriptif d'une plateforme BI pour la chaîne de supermarchés Favorita "
            "(Équateur). " + STYLE,
            f"Preuves disponibles :\n{ev.compact(ids)}\n\nContexte métier :\n{rag.context('KPI cluster promotion jours de paie', 2)}\n\n"
            "Rédige 6 constats descriptifs courts (puces markdown), du plus important au moins important. "
            "Chaque constat cite au moins une preuve.", _USAGE, "analyste", 900)
    else:
        g = ev.get(ids[0])["result"]
        u = ev.get(ids[3])["result"]["magasins"][0]
        f = ev.get(ids[4])["result"]["familles"][0]
        p = ev.get(ids[6])["result"]["plus_reactives"][0]
        text = (f"- Ventes des 28 derniers jours : {g['ventes_28j']} ({g['croissance_28j_pct']} % vs les 28 jours précédents, "
                f"{g['croissance_vs_annee_precedente_pct']} % sur un an) [{ids[0]}].\n"
                f"- Les jours de paie font varier les ventes de {g['effet_jours_de_paie_pct']} % [{ids[0]}].\n"
                f"- {f['family']} est la première famille avec {f['share_pct']} % des ventes [{ids[4]}].\n"
                f"- Le magasin {u['store_nbr']} ({u['city']}) est à {u['gap_vs_cluster_pct']} % de la médiane de son cluster [{ids[3]}].\n"
                f"- La promotion a le plus d'effet sur {p['family']} (+{p['promo_uplift_pct']} %) [{ids[6]}].")
    return {"evidence": ev.items, "desc_insights": text,
            "trace": _log(state, "Analyste descriptif", f"{len(ids)} outils appelés ({ids[0]}–{ids[-1]})", t0)}


# --------------------------------------------------------------------------- #
# 2. Agent prévision
# --------------------------------------------------------------------------- #
def forecaster(state: State) -> State:
    t0 = time.time()
    ev = EvidenceLog(state.get("evidence"))
    ids = [ev.call("model_quality")["id"], ev.call("forecast_overview")["id"],
           ev.call("forecast_movers", level="family", n=4)["id"], ev.call("forecast_movers", level="store", n=3)["id"],
           ev.call("uncertain_families", n=3)["id"]]
    if state["mode"] == "llm":
        text = llm.ask(
            "Tu es l'agent Prévision. Tu interprètes un modèle LightGBM déjà entraîné ; tu ne calcules rien. " + STYLE,
            f"Preuves :\n{ev.compact(ids)}\n\nRédige 5 constats (puces markdown) : fiabilité du modèle vs références, "
            "tendance des 16 prochains jours, familles/magasins qui bougent le plus, incertitude. Chaque constat cite une preuve.",
            _USAGE, "prevision", 800)
    else:
        q = ev.get(ids[0])["result"]["modeles"]
        fo = ev.get(ids[1])["result"]
        up = ev.get(ids[2])["result"]["hausses"][0]
        best = min(q, key=lambda k: q[k]["rmsle"])
        text = (f"- Meilleur modèle en validation : {best} (RMSLE {q[best]['rmsle']}) [{ids[0]}].\n"
                f"- Prévision 16 jours : {fo['prevision_totale_16j']}, soit {fo['variation_vs_16_derniers_jours_pct']} % "
                f"vs les 16 derniers jours [{ids[1]}].\n"
                f"- Plus forte hausse attendue : {up['family']} ({up['change_vs_last_16d_pct']} %) [{ids[2]}].")
    return {"evidence": ev.items, "fc_insights": text,
            "trace": _log(state, "Agent prévision", f"{len(ids)} outils appelés", t0)}


# --------------------------------------------------------------------------- #
# 3. Recommandeur
# --------------------------------------------------------------------------- #
RECO_SCHEMA = ('[{"id": "R1", "titre": "...", "action": "quoi faire, où, quand", "cible": "famille/magasin", '
               '"justification": "pourquoi, avec chiffres", "impact_attendu": "...", "priorite": "haute|moyenne|basse", '
               '"preuves": ["E3", "E7"]}]')


def _offline_recos(ev: EvidenceLog) -> list[dict]:
    find = lambda tool, **kw: next(e for e in reversed(ev.items) if e["tool"] == tool and all(e["args"].get(k) == v for k, v in kw.items()))
    recos = []
    mv = find("forecast_movers", level="family")
    up = mv["result"]["hausses"][0]
    if up["change_vs_last_16d_pct"] and up["change_vs_last_16d_pct"] > 0:
        recos.append({"titre": f"Renforcer le réapprovisionnement de {up['family']}", "cible": up["family"],
                      "action": f"Augmenter les commandes de {up['family']} pour les 16 prochains jours.",
                      "justification": f"La prévision est de {up['forecast_16d']} contre {up['last_16d']} sur les 16 derniers jours, "
                                       f"soit {up['change_vs_last_16d_pct']} %.",
                      "impact_attendu": "Moins de ruptures sur la famille en hausse.", "priorite": "haute", "preuves": [mv["id"]]})
    dn = mv["result"]["baisses"][0]
    recos.append({"titre": f"Réduire les commandes de {dn['family']}", "cible": dn["family"],
                  "action": f"Ajuster à la baisse les commandes de {dn['family']} pour éviter le surstock.",
                  "justification": f"Prévision {dn['forecast_16d']} contre {dn['last_16d']} récemment ({dn['change_vs_last_16d_pct']} %).",
                  "impact_attendu": "Moins de stock immobilisé et de pertes.", "priorite": "moyenne", "preuves": [mv["id"]]})
    pe = find("promo_effect")
    hi, lo = pe["result"]["plus_reactives"][0], pe["result"]["moins_reactives"][0]
    recos.append({"titre": f"Concentrer les promotions sur {hi['family']}", "cible": hi["family"],
                  "action": f"Réallouer une partie du budget promotionnel de {lo['family']} vers {hi['family']}.",
                  "justification": f"L'effet promo est de {hi['promo_uplift_pct']} % sur {hi['family']} contre "
                                   f"{lo['promo_uplift_pct']} % sur {lo['family']}.",
                  "impact_attendu": "Plus de volume pour le même budget promotionnel.", "priorite": "moyenne", "preuves": [pe["id"]]})
    us = find("underperforming_stores")
    s = us["result"]["magasins"][0]
    recos.append({"titre": f"Diagnostiquer le magasin {s['store_nbr']}", "cible": f"Magasin {s['store_nbr']}",
                  "action": f"Audit assortiment, ruptures et affluence du magasin {s['store_nbr']} ({s['city']}).",
                  "justification": f"Ventes 28 j de {s['sales_28d']} contre une médiane de cluster de {s['cluster_median_28d']} "
                                   f"({s['gap_vs_cluster_pct']} %).",
                  "impact_attendu": "Identifier les causes du retard sur les magasins comparables.", "priorite": "haute",
                  "preuves": [us["id"]]})
    uf = find("uncertain_families")
    if uf["result"]["familles"] and (uf["result"]["familles"][0]["uncertainty_pct"] or 0) >= 25:
        f = uf["result"]["familles"][0]
        recos.append({"titre": f"Stock de sécurité pour {f['family']}", "cible": f["family"],
                      "action": f"Dimensionner le stock de {f['family']} sur la prévision haute (p90).",
                      "justification": f"Incertitude de {f['uncertainty_pct']} % : intervalle {f['p10_16d']} à {f['p90_16d']}.",
                      "impact_attendu": "Protection contre les pics imprévus.", "priorite": "basse", "preuves": [uf["id"]]})
    for i, r in enumerate(recos, 1):
        r["id"] = f"R{i}"
    return recos


def recommender(state: State) -> State:
    t0 = time.time()
    ev = EvidenceLog(state.get("evidence"))
    it = state.get("iteration", 0)
    if state["mode"] == "llm":
        prev = ""
        if state.get("feedback"):
            prev = (f"\n\nVERSION PRÉCÉDENTE :\n{json.dumps(state['recos'], ensure_ascii=False)}\n\n"
                    f"RETOURS DU CRITIQUE (à corriger) :\n{state['feedback']}\n"
                    "Garde telles quelles les recommandations acceptées ; corrige ou remplace les autres.")
        text = llm.ask(
            "Tu es l'agent Recommandeur d'une plateforme d'aide à la décision retail. Tu proposes des actions "
            "concrètes, justifiées uniquement par les preuves fournies. " + STYLE,
            f"CONSTATS DESCRIPTIFS :\n{state['desc_insights']}\n\nCONSTATS PRÉVISION :\n{state['fc_insights']}\n\n"
            f"PREUVES :\n{ev.compact()}\n\nRÈGLES MÉTIER :\n{rag.context('stock réapprovisionnement promotion magasin cluster incertitude recommandation', 4)}"
            f"{prev}\n\nPropose 5 recommandations. Réponds UNIQUEMENT avec une liste JSON au format :\n{RECO_SCHEMA}\n"
            "Dans 'preuves', mets les identifiants des preuves d'où viennent TOUS les chiffres cités.",
            _USAGE, "recommandeur", 2500)
        try:
            recos = llm.parse_json(text)
        except ValueError:
            recos = _offline_recos(ev)
    else:
        recos = _offline_recos(ev)
    return {"recos": recos, "iteration": it + 1,
            "trace": _log(state, "Recommandeur", f"tour {it + 1} : {len(recos)} recommandations", t0)}


# --------------------------------------------------------------------------- #
# 4. Critique
# --------------------------------------------------------------------------- #
def critic(state: State) -> State:
    t0 = time.time()
    ev = EvidenceLog(state.get("evidence"))
    review = []
    for r in state["recos"]:
        txt = " ".join(str(r.get(k, "")) for k in ("action", "justification", "impact_attendu"))
        chk = check_numbers(txt, list(r.get("preuves", [])), ev)
        review.append({"id": r.get("id"), "checks": chk, "verdict": "accepte" if chk["ok"] else "a_reviser",
                       "commentaire": "" if chk["ok"] else _explain(chk)})
    if state["mode"] == "llm":
        text = llm.ask(
            "Tu es l'agent Critique. Tu vérifies que chaque recommandation est logique, actionnable, prudente sur la "
            "causalité, cohérente avec les preuves et les règles métier. Tu es exigeant mais juste.",
            f"PREUVES :\n{ev.compact()}\n\nRÈGLES :\n{rag.context('qualité recommandation causalité actionnable', 2)}\n\n"
            f"RECOMMANDATIONS :\n{json.dumps(state['recos'], ensure_ascii=False)}\n\n"
            f"VÉRIFICATION AUTOMATIQUE DES CHIFFRES :\n{json.dumps([{'id': x['id'], **x['checks']} for x in review], ensure_ascii=False)}\n\n"
            'Réponds UNIQUEMENT en JSON : [{"id": "R1", "verdict": "accepte|a_reviser|rejete", "commentaire": "..."}]. '
            "Une recommandation dont la vérification automatique a échoué ne peut pas être acceptée.",
            _USAGE, "critique", 1200)
        try:
            llm_rev = {x["id"]: x for x in llm.parse_json(text)}
        except ValueError:
            llm_rev = {}
        for rv in review:
            x = llm_rev.get(rv["id"])
            if x:
                if rv["verdict"] == "accepte" and x.get("verdict") in ("a_reviser", "rejete"):
                    rv["verdict"] = x["verdict"]
                rv["commentaire"] = (rv["commentaire"] + " " + x.get("commentaire", "")).strip()
    bad = [rv for rv in review if rv["verdict"] != "accepte"]
    feedback = "\n".join(f"{rv['id']} ({rv['verdict']}) : {rv['commentaire']}" for rv in bad)
    n_ok = len(review) - len(bad)
    return {"review": review, "feedback": feedback,
            "trace": _log(state, "Critique", f"{n_ok}/{len(review)} acceptées, {len(bad)} à revoir", t0)}


def _explain(chk: dict) -> str:
    parts = []
    if chk["no_evidence"]:
        parts.append("aucune preuve citée")
    if chk["unknown_evidence"]:
        parts.append(f"preuves inexistantes {chk['unknown_evidence']}")
    if chk["unsupported_numbers"]:
        parts.append(f"chiffres absents des preuves citées {chk['unsupported_numbers']}")
    if not chk["replay_ok"]:
        parts.append("preuve non reproductible")
    return "Vérification automatique : " + "; ".join(parts) + "."


def route_after_critic(state: State) -> str:
    needs = any(rv["verdict"] == "a_reviser" for rv in state["review"])
    return "recommender" if needs and state.get("iteration", 0) < MAX_REVISIONS and state["mode"] == "llm" else "reporter"


# --------------------------------------------------------------------------- #
# 5. Rapporteur
# --------------------------------------------------------------------------- #
def reporter(state: State) -> State:
    t0 = time.time()
    ok = [r for r, rv in zip(state["recos"], state["review"]) if rv["verdict"] == "accepte"]
    if state["mode"] == "llm":
        text = llm.ask(
            "Tu es l'agent Rapporteur. Tu écris pour un directeur des opérations : clair, factuel, sans jargon. " + STYLE,
            f"CONSTATS :\n{state['desc_insights']}\n{state['fc_insights']}\n\nRECOMMANDATIONS VALIDÉES :\n"
            f"{json.dumps(ok, ensure_ascii=False)}\n\nRédige une synthèse exécutive de 120 à 180 mots (un paragraphe "
            "puis 3 priorités numérotées). Garde les références [E#].", _USAGE, "rapporteur", 700)
    else:
        m = jfile("meta")
        text = (f"Sur les 28 derniers jours (jusqu'au {m['last_date']}), les ventes évoluent de {m['growth_28d_pct']} % "
                f"par rapport à la période précédente. Le système a produit {len(state['recos'])} recommandations, dont "
                f"{len(ok)} validées par l'agent critique.\n\n" +
                "\n".join(f"{i}. {r['titre']}" for i, r in enumerate(ok[:3], 1)))
    usage = {"calls": _USAGE.calls, "input_tokens": _USAGE.input_tokens, "output_tokens": _USAGE.output_tokens,
             "cost_usd": _USAGE.cost_usd(), "model": llm.label().replace("`", "") if state["mode"] == "llm" else "hors-ligne (règles)"}
    return {"executive_summary": text, "usage": usage, "trace": _log(state, "Rapporteur", "synthèse rédigée", t0)}


# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(State)
    g.add_node("analyst", analyst)
    g.add_node("forecaster", forecaster)
    g.add_node("recommender", recommender)
    g.add_node("critic", critic)
    g.add_node("reporter", reporter)
    g.add_edge(START, "analyst")
    g.add_edge("analyst", "forecaster")
    g.add_edge("forecaster", "recommender")
    g.add_edge("recommender", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"recommender": "recommender", "reporter": "reporter"})
    g.add_edge("reporter", END)
    return g.compile()


def run(mode: str | None = None) -> dict[str, Any]:
    global _USAGE
    _USAGE = llm.Usage()
    mode = mode or ("llm" if llm.has_key() else "offline")
    t0 = time.time()
    out = build_graph().invoke({"mode": mode, "evidence": [], "iteration": 0, "trace": []})
    out["generated_at"] = time.strftime("%Y-%m-%d %H:%M")
    out["duration_s"] = round(time.time() - t0, 1)
    out["data_meta"] = jfile("meta")
    return dict(out)
