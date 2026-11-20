import streamlit as st

from common import meta, metrics, setup

setup("Méthodologie et architecture", "🧭")
m, q = meta(), metrics()
st.markdown(f"""
**Problématique** : transformer de grands volumes de données opérationnelles en analyses, prévisions et
recommandations automatiques, avec des justifications vérifiables.

**Données** : Kaggle *Store Sales – Time Series Forecasting* (Corporación Favorita) — {m['n_stores']} magasins,
{m['n_families']} familles, {m['n_series']} séries, du {m['first_date']} au {m['last_date']}.

### Architecture
1. **Pipeline hors-ligne (notebook Kaggle)** : nettoyage, jointures (pétrole, jours fériés national/régional/local,
   transactions, magasins), KPIs, modèle de prévision, export d'artefacts Parquet.
2. **Dashboard Streamlit (toujours en ligne)** : lit uniquement les artefacts — aucun recalcul lourd.
3. **Système multi-agents LangGraph** : analyste, prévision, recommandeur, critique, rapporteur.
4. **LLM** : Google Gemini (Flash, free tier) ou Claude Haiku 5.5, au choix via les secrets ; mode hors-ligne à base de règles si aucune clé.
5. **RAG** : base de connaissances (dictionnaire de données, définitions KPI, contexte Équateur, règles métier),
   recherche BM25.

### Prévision
- Modèle **LightGBM global** (un seul modèle pour les {m['n_series']} séries), cible log1p(ventes).
- Horizon direct de 16 jours : uniquement des retards ≥ 16 jours (pas de fuite d'information).
- Variables : retards 16–364 j, moyennes mobiles, jour de semaine, jour de paie, jours fériés, promotions, pétrole,
  type/cluster du magasin.
- Quantiles p10 / p90 (régression quantile) pour mesurer l'incertitude.
- Comparaison à deux références (naïf saisonnier, moyenne mobile 28 j) sur les 16 derniers jours connus :
""")
st.table({k: v for k, v in q["models"].items()})
st.markdown("""
### Justifications vérifiables
- Le LLM **ne calcule aucun chiffre** : chaque chiffre vient d'un outil Pandas déterministe.
- Chaque appel d'outil est journalisé comme preuve `E1`, `E2`, … (outil, paramètres, résultat).
- Chaque recommandation doit citer ses preuves.
- L'agent **critique** combine : (1) un contrôle par code — chaque nombre du texte doit exister dans les preuves
  citées, et chaque preuve est **rejouée** pour vérifier qu'elle est reproductible ; (2) un jugement LLM — logique,
  causalité, caractère actionnable. Les recommandations refusées repartent au recommandeur (2 tours max).
""")
