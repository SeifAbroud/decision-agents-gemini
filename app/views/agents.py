import json

import streamlit as st

from common import RUN_FILE, setup

setup("Agents, recommandations et rapport", "🤖")
from agents import graph, llm, report  # noqa: E402

st.markdown("Cinq agents coopèrent : **Analyste → Prévision → Recommandeur → Critique** (boucle de révision, max 2) "
            "**→ Rapporteur**. Chaque chiffre provient d'un outil déterministe et porte une preuve `E#` rejouable.")
st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#6366F1", fontname="Helvetica", fontsize=11];
 A [label="Analyste\\ndescriptif"]; F [label="Agent\\nprévision"]; R [label="Recommandeur"]; C [label="Critique\\n(chiffres + logique)", fillcolor="#FEF3C7", color="#D97706"]; P [label="Rapporteur"];
 T [label="Outils Pandas\\n(preuves E#)", shape=cylinder, fillcolor="#F1F5F9", color="#94A3B8"]; K [label="RAG\\nrègles métier", shape=note, fillcolor="#F1F5F9", color="#94A3B8"];
 A -> F -> R -> C -> P; C -> R [label=" à réviser", style=dashed, fontsize=9]; T -> A [style=dotted]; T -> F [style=dotted]; K -> R [style=dotted]; K -> C [style=dotted]; }""")


def load_run():
    if "run" in st.session_state:
        return st.session_state.run
    if RUN_FILE.exists():
        return json.loads(RUN_FILE.read_text())
    return None


c1, c2 = st.columns([1, 3])
mode_label = llm.label()
c1.caption(f"Moteur : {mode_label}")
if c1.button("▶ Lancer les agents", type="primary"):
    with st.spinner("Les agents analysent les données…"):
        try:
            res = graph.run()
        except Exception as e:  # ex. clé invalide / crédits épuisés -> repli hors-ligne
            st.warning(f"Appel LLM impossible ({e.__class__.__name__}). Exécution en mode hors-ligne.")
            res = graph.run("offline")
        st.session_state.run = json.loads(json.dumps(res, default=str))
        try:
            RUN_FILE.write_text(json.dumps(st.session_state.run, ensure_ascii=False, indent=1, default=str))
        except OSError:
            pass
run = load_run()
if not run:
    st.info("Aucune analyse disponible : cliquez sur « Lancer les agents ».")
    st.stop()

u = run.get("usage", {})
c2.caption(f"Dernière exécution : {run.get('generated_at')} · {run.get('duration_s')} s · moteur : {u.get('model')} · "
           f"{u.get('calls', 0)} appels LLM · coût ≈ {u.get('cost_usd', 0)} $")

st.subheader("Synthèse exécutive")
st.markdown(run.get("executive_summary", ""))

st.subheader("Recommandations")
icons = {"accepte": "✅ Validée", "a_reviser": "⚠️ À revoir", "rejete": "❌ Rejetée"}
ev = {e["id"]: e for e in run.get("evidence", [])}
for r, rv in zip(run["recos"], run["review"]):
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown(f"**{r.get('id')} · {r.get('titre')}**  \n_Cible : {r.get('cible', '–')} · priorité {r.get('priorite', '–')}_")
        h2.markdown(f"**{icons.get(rv['verdict'], rv['verdict'])}**")
        st.markdown(f"**Action :** {r.get('action')}  \n**Pourquoi :** {r.get('justification')}  \n"
                    f"**Impact attendu :** {r.get('impact_attendu')}")
        chk = rv["checks"]
        st.caption(f"Contrôle automatique : {chk['numbers_checked']} chiffres vérifiés dans les preuves "
                   f"{', '.join(r.get('preuves', []))} · preuves rejouées : {'oui' if chk['replay_ok'] else 'non'}"
                   + (f" · Critique : {rv['commentaire']}" if rv.get("commentaire") else ""))
        with st.expander("Voir les preuves"):
            for eid in r.get("preuves", []):
                e = ev.get(eid)
                if e:
                    st.markdown(f"**{eid}** · `{e['tool']}({json.dumps(e['args'], ensure_ascii=False)})`")
                    st.json(e["result"], expanded=False)

with st.expander("Constats des agents analyste et prévision"):
    st.markdown("**Analyste descriptif**\n\n" + run.get("desc_insights", ""))
    st.markdown("**Agent prévision**\n\n" + run.get("fc_insights", ""))
with st.expander("Trace d'exécution du graphe"):
    st.dataframe(run.get("trace", []), width="stretch", hide_index=True)

st.subheader("Rapport automatique")
md = report.to_markdown(run)
pdf_path = RUN_FILE.parent / "rapport.pdf"
report.to_pdf(md, pdf_path)
d1, d2 = st.columns(2)
d1.download_button("⬇ Télécharger le rapport PDF", pdf_path.read_bytes(), "rapport_aide_decision.pdf", "application/pdf")
d2.download_button("⬇ Télécharger en Markdown", md.encode(), "rapport_aide_decision.md", "text/markdown")
with st.expander("Aperçu du rapport"):
    st.markdown(md)
