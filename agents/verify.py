"""Vérification automatique des chiffres cités (partie déterministe de l'agent critique)."""
from __future__ import annotations

import re

from .tools import EvidenceLog

ALLOWED_SMALL = {7, 14, 16, 21, 28, 56, 90, 91, 100, 364, 365}
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_EID = re.compile(r"\bE\d+\b")
_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:[.,]\d+)?")


def _flatten(x, out: list[float]):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            _flatten(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _flatten(v, out)
    elif isinstance(x, str):
        for m in _NUM.findall(_DATE.sub(" ", x)):
            out.append(float(m.replace(",", ".")))


def numbers_in(text: str) -> list[float]:
    t = _EID.sub(" ", _DATE.sub(" ", text or ""))
    t = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", t)  # 2 257 805 -> 2257805
    vals = []
    for m in _NUM.findall(t):
        v = float(m.replace(",", "."))
        if v.is_integer() and (abs(v) <= 10 or abs(v) in ALLOWED_SMALL or 2010 <= v <= 2030):
            continue
        vals.append(v)
    return vals


def _match(x: float, pool: list[float]) -> bool:
    for v in pool:
        for cand in (v, -v):
            tol = max(0.051, abs(cand) * 0.005)
            if abs(x - cand) <= tol or abs(x - round(cand)) <= 0.5 and abs(cand) >= 100:
                return True
            if abs(cand) >= 1000 and abs(x * 1000 - cand) <= abs(cand) * 0.005:  # "2258 milliers"
                return True
    return False


def check_numbers(text: str, evidence_ids: list[str], ev: EvidenceLog) -> dict:
    """Chaque nombre du texte doit exister dans les preuves citées, et chaque preuve doit être rejouable."""
    unknown_ids = [e for e in evidence_ids if ev.get(e) is None]
    pool: list[float] = []
    for e in evidence_ids:
        it = ev.get(e)
        if it:
            _flatten(it["result"], pool)
            _flatten(it["args"], pool)
    nums = numbers_in(text)
    missing = [x for x in nums if not _match(x, pool)]
    replay_ok = all(ev.replay(e) for e in evidence_ids if ev.get(e))
    ok = bool(evidence_ids) and not unknown_ids and not missing and replay_ok
    return {"ok": ok, "numbers_checked": len(nums), "unsupported_numbers": missing,
            "unknown_evidence": unknown_ids, "replay_ok": replay_ok, "no_evidence": not evidence_ids}
