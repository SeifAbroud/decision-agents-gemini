"""Lance le graphe d'agents une fois et sauvegarde le résultat + le rapport dans artifacts/.

    export ANTHROPIC_API_KEY=sk-ant-...      # sinon : mode hors-ligne (règles)
    python scripts/run_agents.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents import graph, report  # noqa: E402

res = graph.run()
out = ROOT / "artifacts"
(out / "agent_run.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
md = report.to_markdown(res)
(out / "rapport.md").write_text(md)
report.to_pdf(md, out / "rapport.pdf")
for t in res["trace"]:
    print(f"{t['agent']:<22} {t['message']}")
print("Coût :", res["usage"])
