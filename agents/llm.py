"""Accès au LLM : Google Gemini (gratuit) ou Claude (Anthropic).

Choix automatique selon la clé présente dans les secrets :
  - GEMINI_API_KEY     -> Gemini via son endpoint compatible OpenAI (modèle par défaut : gemini-flash-latest)
  - ANTHROPIC_API_KEY  -> Claude Haiku 5.5
On peut forcer avec LLM_PROVIDER = "gemini" | "claude".
Sans aucune clé, le système passe en mode hors-ligne (règles déterministes).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULTS = {"gemini": "gemini-flash-latest", "claude": "claude-haiku-5-5"}
PRICES = {"gemini": (0.0, 0.0), "claude": (0.10, 0.50)}  # $ / million tokens (Gemini : free tier)


def provider() -> str | None:
    p = os.getenv("LLM_PROVIDER", "").strip().lower()
    if p in ("gemini", "claude"):
        return p
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def has_key() -> bool:
    p = provider()
    return bool(p and os.getenv("GEMINI_API_KEY" if p == "gemini" else "ANTHROPIC_API_KEY"))


def model_name() -> str:
    p = provider() or "gemini"
    env = os.getenv("GEMINI_MODEL" if p == "gemini" else "CLAUDE_MODEL")
    return env or DEFAULTS[p]


def label() -> str:
    if not has_key():
        return "hors-ligne (aucune clé API)"
    return f"{'Gemini' if provider() == 'gemini' else 'Claude'} `{model_name()}`"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    log: list = field(default_factory=list)

    def add(self, agent: str, inp: int, out: int):
        self.input_tokens += inp or 0
        self.output_tokens += out or 0
        self.calls += 1
        self.log.append({"agent": agent, "in": inp, "out": out})

    def cost_usd(self) -> float:
        pi, po = PRICES.get(provider() or "gemini", (0, 0))
        return round(self.input_tokens * pi / 1e6 + self.output_tokens * po / 1e6, 5)


_clients: dict = {}


def client():
    """Client brut : anthropic.Anthropic (claude) ou openai.OpenAI pointé sur Gemini (gemini)."""
    p = provider()
    if p not in _clients:
        if p == "gemini":
            from openai import OpenAI
            _clients[p] = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)
        else:
            import anthropic
            _clients[p] = anthropic.Anthropic()
    return _clients[p]


def gemini_chat(messages: list, max_tokens: int, tools=None):
    """Appel Gemini avec réessais (le free tier limite le nombre de requêtes par minute)."""
    kw = dict(model=model_name(), messages=messages, max_tokens=max_tokens, reasoning_effort="low")
    if tools:
        kw["tools"] = tools
    for attempt in range(4):
        try:
            return client().chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "reasoning" in msg.lower() and "reasoning_effort" in kw:
                kw.pop("reasoning_effort")
                continue
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg) and attempt < 3:
                time.sleep(15 * (attempt + 1))
                continue
            raise


def ask(system: str, user: str, usage: Usage | None = None, agent: str = "", max_tokens: int = 2000) -> str:
    if provider() == "gemini":
        # les modèles Gemini « pensent » : on laisse de la marge pour que la réponse ne soit pas coupée
        resp = gemini_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens + 4000)
        if usage is not None and resp.usage:
            usage.add(agent, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""
    resp = client().messages.create(
        model=model_name(), max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    if usage is not None:
        usage.add(agent, resp.usage.input_tokens, resp.usage.output_tokens)
    return "".join(b.text for b in resp.content if b.type == "text")


def parse_json(text: str):
    """Extrait le premier bloc JSON (objet ou liste) d'une réponse du modèle."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("["), raw.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("Pas de JSON dans la réponse")
    raw = raw[start:]
    for end in range(len(raw), 0, -1):
        if raw[end - 1] in "]}":
            try:
                return json.loads(raw[:end])
            except json.JSONDecodeError:
                continue
    raise ValueError("JSON invalide dans la réponse")
