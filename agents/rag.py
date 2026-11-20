"""RAG léger : recherche BM25 dans la base de connaissances `kb/` (markdown).

Pas de modèle d'embeddings à charger -> tient dans la mémoire de Streamlit Cloud.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[1] / "kb"
_STOP = set("le la les de des du un une et en à a au aux pour par sur dans est sont que qui ne pas plus ou ce cette ces son sa ses il elle on d l".split())


def _tok(text: str) -> list[str]:
    text = text.lower()
    return [t for t in re.findall(r"[a-zàâäéèêëîïôöùûüç0-9]+", text) if t not in _STOP and len(t) > 1]


@lru_cache(maxsize=1)
def _index():
    chunks = []
    for f in sorted(KB_DIR.glob("*.md")):
        title = f.stem
        for part in re.split(r"\n(?=#+ )", f.read_text(encoding="utf-8")):
            part = part.strip()
            if part:
                chunks.append({"source": f.name, "title": title, "text": part})
    docs = [_tok(c["text"]) for c in chunks]
    df = Counter(t for d in docs for t in set(d))
    avgdl = sum(map(len, docs)) / max(len(docs), 1)
    return chunks, docs, df, avgdl


def search(query: str, k: int = 3) -> list[dict]:
    chunks, docs, df, avgdl = _index()
    q = _tok(query)
    n = len(docs)
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avgdl))
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])[:k]
    return [{**chunks[i], "score": round(scores[i], 2)} for i in order if scores[i] > 0]


def context(query: str, k: int = 3) -> str:
    return "\n\n".join(f"[KB:{c['source']}]\n{c['text']}" for c in search(query, k))
