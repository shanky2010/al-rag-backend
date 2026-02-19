"""
IndustrialRAG - LLM Formatting Layer
Priority on Render: Anthropic → OpenAI → Ollama (local) → rule-based fallback
Strict grounding: never hallucinate beyond the retrieved context.
"""

import os
from typing import Optional

SYSTEM_PROMPT = """You are an industrial maintenance assistant. Your ONLY job is to extract and structure information from the provided CONTEXT.

STRICT RULES:
1. Use ONLY information explicitly present in the CONTEXT.
2. If a section has no relevant info, write: "Not found in manual."
3. Do NOT use general knowledge. Do NOT guess. Do NOT infer beyond what is written.
4. If the context is completely unrelated to the query, respond only with: INSUFFICIENT_CONTEXT

Output format (use exactly, no deviations):
PROBLEM SUMMARY:
[What the context says about this issue]

POSSIBLE CAUSES:
1. [cause from context]

STEP-BY-STEP CORRECTIVE ACTIONS:
1. [step from context]

SAFETY NOTES:
[warnings from context, or "None stated in manual."]"""


def _prompt(context: str, query: str) -> str:
    return (
        "CONTEXT (from uploaded manual/repair logs only):\n"
        "===\n"
        f"{context}\n"
        "===\n\n"
        f"Technician query: {query}\n\n"
        "Using ONLY the context above, provide the structured response."
    )


def _is_bad(text: str) -> bool:
    return not text or "INSUFFICIENT_CONTEXT" in text.upper()


# ── 1. Anthropic (Claude Haiku) — PRIMARY on Render ──────────────────────────
def _anthropic(context: str, query: str) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _prompt(context, query)}],
        )
        return msg.content[0].text
    except Exception as e:
        print(f"Anthropic error: {e}")
    return None


# ── 2. OpenAI — fallback if Anthropic key missing ────────────────────────────
def _openai(context: str, query: str) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _prompt(context, query)},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
    return None


# ── 3. Ollama — local only, skipped on Render ────────────────────────────────
def _ollama(context: str, query: str) -> Optional[str]:
    if os.environ.get("USE_OLLAMA", "false").lower() != "true":
        return None
    try:
        import requests
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":   os.environ.get("OLLAMA_MODEL", "mistral"),
                "prompt":  _prompt(context, query),
                "system":  SYSTEM_PROMPT,
                "stream":  False,
                "options": {"temperature": 0.0, "num_predict": 1024},
            },
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}")
    return None


# ── 4. Rule-based — no LLM needed, zero cost ─────────────────────────────────
def _rule_based(context: str, query: str) -> str:
    lines = [
        ln.strip() for ln in context.split("\n")
        if ln.strip() and not ln.strip().startswith("[")
    ]
    cause_kw = ["cause", "caused by", "due to", "failure", "fault", "defect",
                "worn", "damaged", "failed", "broken", "loose", "blocked", "missing"]
    fix_kw   = ["replace", "check", "verify", "inspect", "clean", "adjust",
                "tighten", "reset", "test", "turn off", "turn on", "connect",
                "disconnect", "press", "ensure", "remove", "install", "lubricate", "charge"]
    warn_kw  = ["warning", "caution", "danger", "do not", "must not", "hazard",
                "electric", "shock", "fire", "risk", "never"]

    causes, fixes, warnings = [], [], []
    for line in lines:
        low = line.lower()
        if any(k in low for k in warn_kw) and len(warnings) < 3:
            warnings.append(line[:250])
        elif any(k in low for k in cause_kw) and len(causes) < 5:
            causes.append(line[:250])
        elif any(k in low for k in fix_kw) and len(fixes) < 7:
            fixes.append(line[:250])

    has_content = bool(causes or fixes)
    out  = "PROBLEM SUMMARY:\n"
    out += (f'Information found in the manual regarding "{query}".\n' if has_content
            else f'No direct match for "{query}" in the retrieved pages.\n')
    out += "\nPOSSIBLE CAUSES:\n"
    out += "".join(f"{i}. {c}\n" for i, c in enumerate(causes, 1)) or "1. Not found in manual.\n"
    out += "\nSTEP-BY-STEP CORRECTIVE ACTIONS:\n"
    out += "".join(f"{i}. {f}\n" for i, f in enumerate(fixes, 1)) or "1. Not found in manual.\n"
    out += "\nSAFETY NOTES:\n"
    out += "".join(f"- {w}\n" for w in warnings) or "None stated in manual.\n"
    return out


# ── Main entry ────────────────────────────────────────────────────────────────
def generate_formatted_response(context: str, query: str, machine: str) -> str:
    """
    Priority: Anthropic → OpenAI → Ollama → rule-based.
    On Render: set ANTHROPIC_API_KEY env var → uses Claude Haiku (~$0.01/query).
    Locally: set USE_OLLAMA=true → uses Mistral via Ollama.
    Always falls back to rule-based — never crashes.
    """
    if not context.strip():
        return (
            "PROBLEM SUMMARY:\n"
            f'No relevant content retrieved from the manual for "{query}" on {machine}.\n\n'
            "POSSIBLE CAUSES:\n"
            "1. Not found in manual.\n\n"
            "STEP-BY-STEP CORRECTIVE ACTIONS:\n"
            "1. Verify the correct manual has been uploaded in the Admin panel.\n"
            "2. Ensure the machine name matches exactly what was used during upload.\n"
            "3. Try rephrasing using terms from the manual.\n\n"
            "SAFETY NOTES:\n"
            "Do not attempt repairs without the official manual."
        )

    for fn in [_anthropic, _openai, _ollama]:
        result = fn(context, query)
        if result and not _is_bad(result):
            return result

    return _rule_based(context, query)
