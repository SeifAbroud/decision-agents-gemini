"""Génère notebooks/01_pipeline_favorita_kaggle.ipynb à partir de pipeline/build.py (une seule source de vérité)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "pipeline" / "build.py").read_text()
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# Projet 3 · Pipeline Favorita → artefacts du dashboard

**À lancer une seule fois** (ou à chaque fois que vous voulez réentraîner). Le dashboard Streamlit lit uniquement le
dossier `artifacts/` produit ici : il n'a jamais besoin de ce notebook pour fonctionner.

**Sur Kaggle**
1. Ouvrez la compétition *Store Sales - Time Series Forecasting* et acceptez le règlement (bouton *Join / I understand*).
2. Dans ce notebook : panneau de droite → **Add Input** → cherchez `store-sales-time-series-forecasting` → *Add*.
3. *Run All* (≈ 5 à 10 min sur CPU).
4. Téléchargez `artifacts.zip` dans l'onglet **Output** et dézippez-le dans le dossier `artifacts/` du dépôt GitHub."""),
    md("## 0. (Colab uniquement) Télécharger les données avec l'API Kaggle"),
    code("""# Décommentez sur Colab : il faut votre kaggle.json (Kaggle > Settings > API > Create New Token)
# !pip -q install kaggle lightgbm
# from google.colab import files; files.upload()          # choisir kaggle.json
# !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle competitions download -c store-sales-time-series-forecasting -p data && cd data && unzip -o -q *.zip"""),
    code("""import os, glob
# Cherche automatiquement le dossier qui contient train.csv + stores.csv (le chemin change selon Kaggle / Colab)
hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob("data/**/train.csv", recursive=True)
        if os.path.exists(os.path.join(os.path.dirname(p), "stores.csv"))]
assert hits, "Données introuvables : panneau de droite > Add Input > store-sales-time-series-forecasting"
DATA_DIR = hits[0]
OUT_DIR = "artifacts"
print(DATA_DIR, sorted(os.listdir(DATA_DIR)))"""),
    md("## 1. Code du pipeline (identique à `pipeline/build.py` du dépôt)\n"
       "Chargement → préparation (pétrole, jours fériés national/régional/local, jours de paie) → KPIs → "
       "LightGBM global + quantiles p10/p90 → export Parquet."),
    code("%%writefile build.py\n" + src),
    md("## 2. Exécution"),
    code("""import build, importlib
importlib.reload(build)
build.run(DATA_DIR, OUT_DIR, source="kaggle", train_days=420)"""),
    md("## 3. Contrôles rapides (à reprendre dans le rapport)"),
    code("""import json, pandas as pd
print(json.dumps(json.load(open(f"{OUT_DIR}/metrics.json"))["models"], indent=2, ensure_ascii=False))
display(pd.read_parquet(f"{OUT_DIR}/family_kpis.parquet").head(10))
display(pd.read_parquet(f"{OUT_DIR}/store_kpis.parquet").sort_values("gap_vs_cluster_pct").head(5))
display(pd.read_parquet(f"{OUT_DIR}/promo_uplift.parquet").head(8))"""),
    code("""import matplotlib.pyplot as plt
d = pd.read_parquet(f"{OUT_DIR}/daily_total.parquet").set_index("date")["sales"]
f = pd.read_parquet(f"{OUT_DIR}/forecast_daily_total.parquet").set_index("date")
ax = d.tail(120).plot(figsize=(12, 4), color="grey", label="réel")
f["mean"].plot(ax=ax, color="tab:blue", label="prévision")
ax.fill_between(f.index, f.p10, f.p90, alpha=.2); ax.legend(); plt.show()"""),
    md("## 4. Empaqueter les artefacts (+ fichier de soumission Kaggle `artifacts/submission.csv`)"),
    code("""import shutil
shutil.make_archive("artifacts", "zip", ".", OUT_DIR)
print("OK -> artifacts.zip (onglet Output) ; taille :", round(os.path.getsize("artifacts.zip") / 1e6, 1), "Mo")"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "01_pipeline_favorita_kaggle.ipynb"
nbf.write(nb, out)
print("written", out)
