import json

import streamlit as st

from common import setup

setup("Assistant : interroger les données", "💬")
from agents import llm, qa  # noqa: E402

if not llm.has_key():
    st.info("L'assistant a besoin d'une clé API (GEMINI_API_KEY ou ANTHROPIC_API_KEY dans les secrets Streamlit). "
            "Le reste du dashboard fonctionne sans clé.")
    st.stop()

st.caption("Le LLM choisit les outils à appeler ; chaque chiffre de la réponse est vérifié contre les preuves.")
examples = ["Quels magasins sont les plus en retard sur leur cluster, et pourquoi s'inquiéter ?",
            "Quelles familles faut-il réapprovisionner en priorité pour les 16 prochains jours ?",
            "Les promotions sont-elles efficaces sur BEVERAGES ?",
            "Le modèle de prévision est-il fiable ?"]
st.session_state.setdefault("chat", [])
cols = st.columns(len(examples))
picked = None
for c, ex in zip(cols, examples):
    if c.button(ex, width="stretch"):
        picked = ex

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

q = st.chat_input("Votre question…") or picked
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Analyse…"):
            try:
                res = qa.answer(q)
            except Exception as e:
                st.error(f"Erreur API : {e}")
                st.stop()
        st.markdown(res["answer"])
        chk = res["check"]
        badge = "✅ chiffres vérifiés" if chk["ok"] else f"⚠️ chiffres non trouvés dans les preuves : {chk['unsupported_numbers']}"
        meta = f"{badge} · {len(res['evidence'])} appels d'outils · coût ≈ {res['cost_usd']} $"
        st.caption(meta)
        with st.expander("Preuves"):
            for e in res["evidence"]:
                st.markdown(f"**{e['id']}** · `{e['tool']}({json.dumps(e['args'], ensure_ascii=False)})`")
                st.json(e["result"], expanded=False)
    st.session_state.chat.append({"role": "assistant", "content": res["answer"], "meta": meta})
