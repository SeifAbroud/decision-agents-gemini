# Projet 3 · Système multi-agents d'aide à la décision (Favorita)

ISIMA Mahdia · Master 2 Cloud & Big Data

Plateforme d'aide à la décision : des **agents LLM coopèrent** (LangGraph + Claude) pour analyser les ventes de la
chaîne Favorita (Kaggle *Store Sales – Time Series Forecasting*), prévoir les 16 prochains jours et produire des
**recommandations dont chaque chiffre est vérifiable**.

**Livrables** : tableau de bord intelligent (Streamlit), rapports automatiques (PDF/Markdown), système de recommandations
validé par un agent critique.

---

## 1. Architecture

```
 HORS-LIGNE (une fois, Kaggle)                      EN LIGNE (toujours disponible, Streamlit Cloud)
 ┌──────────────────────────────┐                   ┌────────────────────────────────────────────────┐
 │ CSV Favorita                 │                   │ Dashboard (lit uniquement artifacts/)          │
 │  → préparation + KPIs        │   artifacts/      │   ├─ Vue d'ensemble, Magasins, Prévisions      │
 │  → LightGBM + quantiles      │ ───────────────►  │   ├─ Agents LangGraph ──► Gemini / Claude       │
 │  → export Parquet            │   (GitHub)        │   │     analyste → prévision → recommandeur    │
 └──────────────────────────────┘                   │   │     → critique ⟲ → rapporteur              │
                                                    │   ├─ Assistant (tool use)  ├─ RAG (kb/)        │
                                                    └────────────────────────────────────────────────┘
```

Le notebook n'a **jamais** besoin d'être relancé pour la démo : le dashboard lit des fichiers Parquet déjà calculés.

### Les agents (`agents/graph.py`)

| Agent | Rôle | Outils / entrées |
|---|---|---|
| Analyste descriptif | KPIs, tendances, magasins en retard, familles clés, promos, fériés | 9 outils Pandas |
| Agent prévision | fiabilité du modèle, tendance 16 j, plus fortes variations, incertitude | 5 outils |
| Recommandeur | 5 recommandations actionnables, chacune avec ses preuves `E#` | preuves + RAG (règles métier) |
| Critique | vérifie chaque recommandation, renvoie au recommandeur si besoin (max 2 tours) | contrôle par code + LLM |
| Rapporteur | synthèse exécutive + rapport PDF/Markdown | recommandations validées |

### Justifications vérifiables (contribution principale)
1. Le LLM **ne calcule aucun chiffre** : tous viennent d'outils déterministes (`agents/tools.py`).
2. Chaque appel d'outil est journalisé comme **preuve** `E1, E2…` (outil, paramètres, résultat).
3. Chaque recommandation doit citer ses preuves.
4. Le critique fait un **contrôle par code** (`agents/verify.py`) : chaque nombre du texte doit exister dans les preuves
   citées, et chaque preuve est **rejouée** pour vérifier qu'elle est reproductible. Puis un **contrôle LLM** (logique,
   causalité, caractère actionnable). Un chiffre inventé ⇒ la recommandation repart au recommandeur.
5. Dans le dashboard, chaque recommandation affiche son statut (✅ / ⚠️) et ses preuves dépliables.

### Modèle de prévision (`pipeline/build.py`)
- LightGBM **global** (un modèle pour les 1 782 séries magasin × famille), cible `log1p(ventes)`.
- Horizon direct 16 jours : uniquement des retards ≥ 16 j (pas de fuite d'information).
- Variables : retards 16–364 j, moyennes mobiles, jour de semaine, **jours de paie** (15 et fin de mois), jours fériés
  national / régional / local (avec les jours *transferred*), promotions, prix du pétrole, type et cluster du magasin.
- Intervalles **p10 / p90** par régression quantile.
- Évaluation sur les 16 derniers jours connus contre 2 références (naïf saisonnier, moyenne mobile 28 j), métrique
  officielle **RMSLE** + WAPE. Le notebook produit aussi `submission.csv` pour obtenir un score Kaggle officiel.

---

## 2. Arborescence

```
decision-agents/
├── app/                      # dashboard Streamlit
│   ├── streamlit_app.py      # point d'entrée (main file)
│   ├── common.py
│   └── views/                # home, explorer, forecast, agents, assistant, method
├── agents/                   # système multi-agents
│   ├── graph.py              # graphe LangGraph (5 agents + boucle critique)
│   ├── tools.py              # outils déterministes + journal de preuves
│   ├── verify.py             # vérification automatique des chiffres
│   ├── rag.py                # RAG BM25 sur kb/
│   ├── llm.py                # client Claude (Haiku 5.5 par défaut)
│   ├── qa.py                 # assistant « Ask the data » (tool use)
│   └── report.py             # rapport Markdown + PDF
├── kb/                       # base de connaissances (RAG)
├── pipeline/
│   ├── build.py              # pipeline données → artefacts
│   └── synth.py              # données synthétiques de test (même schéma)
├── notebooks/01_pipeline_favorita_kaggle.ipynb
├── scripts/run_agents.py     # lance les agents en ligne de commande
├── artifacts/                # SORTIE du pipeline, lue par le dashboard
├── requirements.txt          # dépendances du dashboard (Streamlit Cloud)
└── requirements-pipeline.txt # + lightgbm pour le pipeline
```

> ⚠️ Les `artifacts/` livrés dans ce zip viennent de **données synthétiques** (pour tester). Remplacez-les par ceux
> du notebook Kaggle (étape 3.1) avant la soutenance. Le dashboard affiche un bandeau tant que ce n'est pas fait.

---

## 3. Mode d'emploi

### 3.1 Produire les vrais artefacts (Kaggle, une seule fois)
1. Sur kaggle.com, ouvrez la compétition **Store Sales - Time Series Forecasting** → *Join Competition* (accepter le règlement).
2. *Create → New Notebook* → *File → Import Notebook* → `notebooks/01_pipeline_favorita_kaggle.ipynb`.
3. Panneau de droite : **Add Input** → `store-sales-time-series-forecasting`.
4. **Run All** (≈ 5–10 min, CPU suffit).
5. Onglet **Output** : téléchargez `artifacts.zip`, dézippez, et remplacez le contenu du dossier `artifacts/` du dépôt.
6. (Bonus) Soumettez `artifacts/submission.csv` dans la compétition pour obtenir un score officiel.

### 3.2 Clé API
**Gemini (gratuit)** : aistudio.google.com → *Get API key* → mettez `GEMINI_API_KEY` dans les secrets.

**Ou Claude (plan Team → crédits API mensuels inclus)**
1. Sur **claude.ai** (navigateur) : *Organization settings → Billing → API credits → Link organization*
   (rôle Owner ou Primary Owner du plan Team requis). Les crédits arrivent dans une organisation de la Claude Console.
2. Sur **platform.claude.com** : *API keys → Create key*. Copiez la clé `sk-ant-…`.
3. Dans la Console, fixez une **limite de dépense** : l'app est publique.
4. Modèle utilisé : `claude-haiku-5-5` (le moins cher). Un rapport complet coûte ≈ 0,01 $.

Sans clé, tout fonctionne en **mode hors-ligne** (règles déterministes) sauf la page *Assistant*.

### 3.3 Lancer en local (facultatif)
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # puis mettez votre clé
streamlit run app/streamlit_app.py
```
Pour régénérer des données de test : `pip install -r requirements-pipeline.txt` puis
`python -m pipeline.synth --out data/synthetic && python -m pipeline.build --data data/synthetic --source synthetic`.

### 3.4 Déployer (lien permanent pour le professeur) — Streamlit Community Cloud, gratuit
1. Créez un dépôt GitHub (privé ou public) et envoyez-y tout le dossier (GitHub → *Add file → Upload files*,
   glisser-déposer ; ne mettez **jamais** `secrets.toml`).
2. Allez sur **share.streamlit.io** → connectez GitHub → *Create app* → *Deploy a public app from GitHub*.
3. Repository : votre dépôt · Branch : `main` · **Main file path : `app/streamlit_app.py`**.
4. *Advanced settings* → Python 3.12 → **Secrets** : collez
   ```toml
   GEMINI_API_KEY = "votre-cle-gemini"
   GEMINI_MODEL = "gemini-flash-latest"
   APP_PASSWORD = "choisissez-un-mot-de-passe"   # optionnel mais conseillé
   ```
5. *Deploy*. Vous obtenez une URL `https://<nom>.streamlit.app` à donner au professeur.
6. Pour mettre à jour : remplacez des fichiers sur GitHub → l'app se redéploie toute seule.

**Avant la démo** : l'app s'endort après 12 h sans visite. Ouvrez le lien 5 minutes avant, puis cliquez une fois sur
*Lancer les agents*. Gardez une vidéo de démo en secours.

### 3.5 Scénario de démo (≈ 5 min)
1. **Vue d'ensemble** : KPIs, ventes + prévision 16 j avec intervalle, magasins vs cluster.
2. **Prévisions** : LightGBM vs références (RMSLE), importance des variables, zoom magasin × famille.
3. **Agents et recommandations** : *Lancer les agents* → trace du graphe, recommandations ✅, preuves dépliables,
   téléchargement du rapport PDF.
4. **Assistant** : poser une question libre ; montrer le badge « chiffres vérifiés » et les preuves.
5. **Méthodologie** : architecture et principe de vérification.

---

## 4. Pistes d'évaluation pour le mémoire
- Prévision : RMSLE / WAPE vs références, couverture réelle de l'intervalle p10–p90 (cible 80 %), score Kaggle.
- Agents : % de recommandations acceptées au 1er tour, nombre de chiffres non supportés détectés et corrigés par la
  boucle critique, coût et latence par rapport (affichés dans l'app).
- Test de robustesse du critique : injecter un chiffre faux dans une recommandation → il est détecté
  (cf. `agents/verify.py`).
