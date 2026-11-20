# Guide rapide: kifech t5adem el projet (3 étapes)

El hadaf: ykounlek **lien wa7ed** (`https://....streamlit.app`) ta3tih lel prof, w yeb9a yekhdem dima.

---

## Étape 1: el vraie data (Kaggle, ~10 d9aye9, marra wa7da)

1. Odkhel l **kaggle.com** w a3mel compte.
2. Lawwej 3la **Store Sales - Time Series Forecasting**, ba3d enzel **Join Competition** w a9bel el règlement.
3. Emchi l **Create → New Notebook**, ba3d **File → Import Notebook**, w ekhtar `notebooks/01_pipeline_favorita_kaggle.ipynb`.
4. Fel panneau elli 3al imin, enzel **Add Input** w lawwej 3la `store-sales-time-series-forecasting`, ba3d **Add**.
5. Enzel **Run All** w estanna 5 l 10 d9aye9.
6. Fel onglet **Output**, telechargi **`artifacts.zip`**.
7. Dézippih, w 7ott el fichiers elli fih **blaset** el fichiers elli fel dossier `artifacts/` mta3 el projet.

✅ Ki tkammel: el bandeau "Données synthétiques" yna77 men el dashboard.

---

## Étape 2: el clé Gemini (gratuite)

1. Odkhel l **aistudio.google.com** b compte Google.
2. Enzel **Get API key → Create API key**, w copii el clé.

> Ken ma t7otch clé, el dashboard yekhdem 3adi b mode hors-ligne. Ken page **Assistant** elli testa7a9 el clé.

---

## Étape 3: el mise en ligne (gratuit, ~10 d9aye9)

**A. GitHub**
1. Odkhel l **github.com**, enzel **New repository**, sammih `decision-agents` w a3mel **Create**.
2. Enzel **uploading an existing file** w drag & drop **el contenu lkol** mta3 el dossier `decision-agents`.
   ⚠️ Ma t7ottech fichier `secrets.toml`.
3. Enzel **Commit changes**.

**B. Streamlit Cloud**
1. Odkhel l **share.streamlit.io**, w odkhel b compte GitHub mte3ek.
2. Enzel **Create app → Deploy a public app from GitHub**.
3. 3abbi:
   - Repository: `decision-agents`
   - Branch: `main`
   - Main file path: **`app/streamlit_app.py`**
4. Enzel **Advanced settings → Secrets** w colli hedha (badel el clé b mte3ek):
   ```toml
   GEMINI_API_KEY = "COLLI-EL-CLE-GEMINI-HOUNI"
   GEMINI_MODEL = "gemini-flash-latest"
   APP_PASSWORD = "isima2026"
   ```
5. Enzel **Deploy** w estanna ~3 d9aye9.

✅ Ywalli 3andek el lien `https://xxx.streamlit.app`, w el mot de passe houa elli 7attitou f `APP_PASSWORD`.

---

## Nhar el présentation

1. **7ell el lien 5 d9aye9 9bal**: el app tor9od ba3d 12 se3a ma 7add 7allha.
2. Emchi l page **Agents et recommandations**, w enzel **Lancer les agents** marra bech tetaked elli kol chay ye5dem.
3. Ordre mta3 el démo:
   **Vue d'ensemble → Prévisions → Agents (Lancer + preuves + PDF) → Assistant (7ott sou2el) → Méthodologie**

---

## Ken sar mochkol

| Mochkol | El 7al |
|---|---|
| Bandeau asfar "Données synthétiques" | Ma baddaltech `artifacts/` b mta3 Kaggle, raja3 l étape 1 |
| "Moteur : hors-ligne" | El clé mouch mawjouda wala ghalta fel Secrets |
| "Appel LLM impossible" | El clé ghalta, wala el quota gratuit mta3 el youm kmel (estanna wala badel `GEMINI_MODEL` l `gemini-flash-lite-latest`) |
| El app ma t7allech / error | Fi Streamlit Cloud enzel **Manage app → Reboot** |
| T7eb tbaddel fichier | Baddlou fi GitHub, w el app titbaddel wa7dha |

---

## (Facultatif) T5adem 3al PC mte3ek

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
Ba3d 7ell `http://localhost:8501`.
