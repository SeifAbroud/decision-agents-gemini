"""Agent « Ask the data » : le LLM choisit lui-même les outils (tool use / function calling),
chaque appel devient une preuve, puis la réponse est vérifiée chiffre par chiffre.
Fonctionne avec Gemini (format OpenAI) et Claude (format Anthropic)."""
from __future__ import annotations

import json
import re

from . import llm, rag
from .tools import TOOL_SCHEMAS, EvidenceLog
from .verify import check_numbers

SYSTEM = ("Tu es l'assistant analytique de la plateforme d'aide à la décision Favorita (supermarchés, Équateur). "
          "Pour répondre, appelle les outils : ils sont la seule source de chiffres autorisée. Ne calcule pas toi-même "
          "de nouveaux chiffres. Réponds en français, de façon concise (5 à 10 lignes), écris les nombres sans séparateur "
          "de milliers avec un point décimal, et cite les preuves [E1], [E2] renvoyées après chaque appel. "
          "Si la question sort du périmètre des données, dis-le.")

OPENAI_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"].strip(),
                                                   "parameters": t["input_schema"]}} for t in TOOL_SCHEMAS]


def _run_gemini(question: str, ev: EvidenceLog, usage: llm.Usage, max_steps: int) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    text = ""
    for _ in range(max_steps):
        resp = llm.gemini_chat(msgs, 6000, tools=OPENAI_TOOLS)
        if resp.usage:
            usage.add("qa", resp.usage.prompt_tokens, resp.usage.completion_tokens)
        msg = resp.choices[0].message
        text = msg.content or ""
        calls = msg.tool_calls or []
        if not calls:
            break
        msgs.append(msg.model_dump(exclude_none=True))
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            item = ev.call(c.function.name, **args)
            msgs.append({"role": "tool", "tool_call_id": c.id,
                         "content": f"[{item['id']}] " + json.dumps(item["result"], ensure_ascii=False)})
    return text


def _run_claude(question: str, ev: EvidenceLog, usage: llm.Usage, max_steps: int) -> str:
    msgs = [{"role": "user", "content": question}]
    resp = None
    for _ in range(max_steps):
        resp = llm.client().messages.create(model=llm.model_name(), max_tokens=1200, system=SYSTEM,
                                            tools=TOOL_SCHEMAS, messages=msgs)
        usage.add("qa", resp.usage.input_tokens, resp.usage.output_tokens)
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            break
        results = []
        for b in calls:
            item = ev.call(b.name, **(b.input or {}))
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": f"[{item['id']}] " + json.dumps(item["result"], ensure_ascii=False)})
        msgs.append({"role": "user", "content": results})
    return "".join(b.text for b in resp.content if b.type == "text") if resp else ""


def answer(question: str, max_steps: int = 6) -> dict:
    ev = EvidenceLog()
    usage = llm.Usage()
    q = f"{question}\n\n(Contexte métier utile :\n{rag.context(question, 2)})"
    run = _run_gemini if llm.provider() == "gemini" else _run_claude
    text = run(q, ev, usage, max_steps)
    cited = sorted(set(re.findall(r"\bE\d+\b", text)), key=lambda x: int(x[1:]))
    check = check_numbers(text, cited or [e["id"] for e in ev.items], ev)
    return {"answer": text, "evidence": ev.items, "check": check, "cost_usd": usage.cost_usd(),
            "tokens": usage.input_tokens + usage.output_tokens}
