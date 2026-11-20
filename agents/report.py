"""Rapport automatique : Markdown + PDF à partir du résultat du graphe d'agents."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)

from .tools import jfile, table

STATUS = {"accepte": "VALIDEE", "a_reviser": "A REVOIR", "rejete": "REJETEE"}


def to_markdown(run: dict) -> str:
    m, q = jfile("meta"), jfile("metrics")
    fo = table("forecast_by_family").sort_values("forecast_16d", ascending=False).head(8)
    L = [f"# Rapport d'aide à la décision – Favorita",
         f"Généré le {run.get('generated_at', '')} · données {m['first_date']} → {m['last_date']} · "
         f"prévision {m['forecast_start']} → {m['forecast_end']} · moteur : {run.get('usage', {}).get('model', '')}",
         "", "## 1. Synthèse exécutive", run.get("executive_summary", ""), "",
         "## 2. Indicateurs clés (28 derniers jours)",
         f"- Ventes : {m['sales_28d']:,.0f} ({m['growth_28d_pct']:+} % vs 28 j précédents"
         + (f", {m['yoy_pct']:+} % sur un an)" if m.get("yoy_pct") is not None else ")"),
         f"- Ventes moyennes par jour : {m['avg_daily_sales_28d']:,.0f}",
         f"- Transactions : {m['transactions_28d']:,}",
         f"- Effet jours de paie : {m['payday_effect_pct']:+} %",
         f"- Lignes en promotion : {m['promo_rows_pct_28d']} %", "",
         "## 3. Constats de l'analyste", run.get("desc_insights", ""), "",
         "## 4. Prévisions (16 jours)", run.get("fc_insights", ""), "",
         "| Modèle | RMSLE | WAPE % |", "|---|---|---|"]
    L += [f"| {k} | {v['rmsle']} | {v['wape_pct']} |" for k, v in q["models"].items()]
    L += ["", "| Famille | Prévision 16 j | Derniers 16 j | Variation % |", "|---|---|---|---|"]
    L += [f"| {r.family} | {r.forecast_16d:,.0f} | {r.last_16d:,.0f} | {r.change_vs_last_16d_pct:+} |" for r in fo.itertuples()]
    L += ["", "## 5. Recommandations"]
    for r, rv in zip(run.get("recos", []), run.get("review", [])):
        L += [f"### {r.get('id')} – {r.get('titre')}  [{STATUS.get(rv['verdict'], rv['verdict'])}]",
              f"- Action : {r.get('action')}", f"- Justification : {r.get('justification')}",
              f"- Impact attendu : {r.get('impact_attendu')}", f"- Priorité : {r.get('priorite')} · preuves : {', '.join(r.get('preuves', []))}",
              f"- Contrôle automatique : {rv['checks']['numbers_checked']} chiffres vérifiés, "
              f"{'OK' if rv['checks']['ok'] else 'ÉCHEC'}" + (f" · Critique : {rv['commentaire']}" if rv.get("commentaire") else ""), ""]
    L += ["## 6. Annexe – preuves (rejouables)"]
    for e in run.get("evidence", []):
        L.append(f"- **{e['id']}** `{e['tool']}({json.dumps(e['args'], ensure_ascii=False)})`")
    u = run.get("usage", {})
    L += ["", f"_Coût LLM de ce rapport : {u.get('calls', 0)} appels, {u.get('input_tokens', 0)} tokens entrée, "
              f"{u.get('output_tokens', 0)} tokens sortie ≈ {u.get('cost_usd', 0)} $_"]
    return "\n".join(L)


_REPL = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "→": "->", "≈": "~", "≥": ">=",
         "≤": "<=", " ": " ", " ": " ", "•": "-", "×": "x", "€": "EUR", "✅": "", "⚠": ""}


def _latin(s: str) -> str:
    for a, b in _REPL.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def to_pdf(md: str, path: str | Path) -> Path:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        rows = [r for r in table_rows if not set("".join(r)) <= set("-: ")]
        n = len(rows[0])
        cw = [W * (0.4 if n == 4 and i == 0 else (0.6 / (n - 1) if n == 4 else 1 / n)) for i in range(n)]
        for j, row in enumerate(rows):
            pdf.set_font("Helvetica", "B" if j == 0 else "", 9)
            pdf.set_fill_color(235, 238, 245) if j == 0 else pdf.set_fill_color(255, 255, 255)
            for i, c in enumerate(row):
                pdf.cell(cw[i], 6, _latin(c.strip())[:48], border=1, fill=True)
            pdf.ln()
        pdf.ln(2)
        table_rows = []

    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            table_rows.append(s.strip("|").split("|"))
            continue
        flush_table()
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s).replace("`", "")
        italic = s.startswith("_") and s.endswith("_")
        s = s.strip("_") if italic else s
        if s.startswith("# "):
            pdf.set_font("Helvetica", "B", 17)
            pdf.set_text_color(20, 40, 90)
            pdf.multi_cell(W, 9, _latin(s[2:]), **NL)
            pdf.set_text_color(0, 0, 0)
        elif s.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(20, 40, 90)
            pdf.multi_cell(W, 7, _latin(s[3:]), **NL)
            pdf.set_text_color(0, 0, 0)
        elif s.startswith("### "):
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.multi_cell(W, 6, _latin(s[4:]), **NL)
        elif s.startswith("- ") or s.startswith("* "):
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(W - 4, 5, _latin("- " + s[2:]), **NL)
        elif s.strip() == "":
            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "I" if italic else "", 9.5)
            pdf.multi_cell(W, 5, _latin(s), **NL)
    flush_table()
    path = Path(path)
    pdf.output(str(path))
    return path
