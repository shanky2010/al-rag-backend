"""
IndustrialRAG - LLM Formatting Layer
Primary: Google Gemini Flash (free, same key as embeddings)
Fallback: Anthropic → OpenAI → Ollama → rule-based
"""

import os
import requests
from typing import Optional

SYSTEM_PROMPT = """You are an expert industrial maintenance assistant for Ashok Leyland vehicles and equipment.

A technician has queried the system and relevant excerpts have been retrieved from machine manuals and repair logs. Your job is to synthesize those excerpts into a clear, actionable maintenance response.

STRICT RULES — FOLLOW EXACTLY:
1. Use ONLY information present in the CONTEXT below. Do NOT add any knowledge from outside the context.
2. Write in complete, natural sentences. Do NOT copy raw fragments verbatim.
3. If insufficient information exists for a section, write exactly: "Not found in the manual."
4. Do NOT invent or assume causes, steps, or warnings that are not clearly stated in the context.
5. If the ENTIRE context is irrelevant to the query, respond with only: INSUFFICIENT_CONTEXT
6. You MUST use EXACTLY these four headings on their own lines, followed immediately by content on the next line.

OUTPUT FORMAT — MANDATORY:

PROBLEM SUMMARY:
[2-3 sentences describing what the manual/logs say about this fault or issue]

POSSIBLE CAUSES:
1. [First cause from context]
2. [Second cause, if present]

STEP-BY-STEP CORRECTIVE ACTIONS:
1. [First action from context]
2. [Second action]

SAFETY NOTES:
[Safety warnings or cautions from the context. If none, write: None stated in manual.]"""


def _build_prompt(context: str, query: str, machine: str) -> str:
    # Deduplicate context blocks (same text appearing from multiple chunks)
    seen = set()
    deduped_parts = []
    for block in context.split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        # Use first 120 chars as a fingerprint to deduplicate near-identical chunks
        fingerprint = block[:120].lower().replace(" ", "")
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped_parts.append(block)

    deduped_context = "\n\n---\n\n".join(deduped_parts)

    return (
        f"MACHINE: {machine}\n"
        f"TECHNICIAN QUERY: {query}\n\n"
        "CONTEXT (retrieved from uploaded manuals and repair logs):\n"
        "════════════════════════════════════════\n"
        f"{deduped_context}\n"
        "════════════════════════════════════════\n\n"
        "Using ONLY the context above, produce the structured maintenance response. "
        "Rewrite information in clear, complete sentences — do not copy raw text fragments. "
        "Every section must be grounded in the context. "
        "If a section has no relevant information, write: Not found in the manual."
    )


def _is_bad(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return True
    t = text.upper()
    if "INSUFFICIENT_CONTEXT" in t:
        return True
    # Must contain at least two of the four expected section headers
    headers = ["PROBLEM SUMMARY", "POSSIBLE CAUSES", "CORRECTIVE ACTION", "SAFETY"]
    found = sum(1 for h in headers if h in t)
    if found < 2:
        return True
    return False


# ── 1. Gemini Flash — PRIMARY (free, same key as embeddings) ──────────────────
def _gemini(context: str, query: str, machine: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("LLM: GEMINI_API_KEY not set, skipping.", flush=True)
        return None
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        full_prompt = f"{SYSTEM_PROMPT}\n\n{_build_prompt(context, query, machine)}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1500,
            }
        }
        resp = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        result = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"LLM: Gemini Flash returned {len(result)} chars.", flush=True)
        return result
    except Exception as e:
        print(f"LLM: Gemini error: {e}", flush=True)
    return None


# ── 2. Anthropic (Claude Haiku) — fallback if credits available ───────────────
def _anthropic(context: str, query: str, machine: str) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(context, query, machine)}],
        )
        result = msg.content[0].text
        print(f"LLM: Anthropic returned {len(result)} chars.", flush=True)
        return result
    except Exception as e:
        print(f"LLM: Anthropic error: {e}", flush=True)
    return None


# ── 3. OpenAI ─────────────────────────────────────────────────────────────────
def _openai(context: str, query: str, machine: str) -> Optional[str]:
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
                {"role": "user",   "content": _build_prompt(context, query, machine)},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        result = r.choices[0].message.content
        print(f"LLM: OpenAI returned {len(result)} chars.", flush=True)
        return result
    except Exception as e:
        print(f"LLM: OpenAI error: {e}", flush=True)
    return None



# ── 4. Groq (free, fast — Mistral/Llama hosted) ───────────────────────────────
def _groq(context: str, query: str, machine: str) -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        import groq as groq_lib
        client = groq_lib.Groq(api_key=api_key)
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(context, query, machine)},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        result = r.choices[0].message.content
        print(f"LLM: Groq returned {len(result)} chars.", flush=True)
        return result
    except Exception as e:
        print(f"LLM: Groq error: {e}", flush=True)
    return None


# ── 5. Ollama (local only) ────────────────────────────────────────────────────
def _ollama(context: str, query: str, machine: str) -> Optional[str]:
    if os.environ.get("USE_OLLAMA", "false").lower() != "true":
        return None
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":   os.environ.get("OLLAMA_MODEL", "mistral"),
                "prompt":  _build_prompt(context, query, machine),
                "system":  SYSTEM_PROMPT,
                "stream":  False,
                "options": {"temperature": 0.0, "num_predict": 1500},
            },
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception as e:
        print(f"LLM: Ollama error: {e}", flush=True)
    return None


# ── 5. Rule-based fallback ────────────────────────────────────────────────────
def _rule_based(context: str, query: str) -> str:
    print("LLM: Using rule-based fallback.", flush=True)
    sentences = [
        ln.strip() for ln in context.split("\n")
        if len(ln.strip()) > 20 and not ln.strip().startswith("[")
    ]
    cause_kw = ["cause", "caused by", "due to", "failure", "fault", "defect",
                "worn", "damaged", "failed", "broken", "loose", "blocked", "missing",
                "low battery", "obstacle", "emergency stop", "not pressed", "is off"]
    fix_kw   = ["replace", "check", "verify", "inspect", "clean", "adjust",
                "tighten", "reset", "test", "turn off", "turn on", "connect",
                "disconnect", "press", "ensure", "remove", "install", "lubricate",
                "charge", "release", "restart", "de-energize", "switch"]
    warn_kw  = ["warning", "caution", "danger", "do not", "must not", "hazard",
                "electric", "shock", "fire", "risk", "never", "only perform",
                "trained electrician"]

    causes, fixes, warnings, summary = [], [], [], []
    for s in sentences:
        low = s.lower()
        if any(k in low for k in warn_kw) and len(warnings) < 3:
            warnings.append(s[:300])
        elif any(k in low for k in cause_kw) and len(causes) < 5:
            causes.append(s[:300])
        elif any(k in low for k in fix_kw) and len(fixes) < 7:
            fixes.append(s[:300])
        elif len(summary) < 2:
            summary.append(s[:300])

    out  = "PROBLEM SUMMARY:\n"
    out += (" ".join(summary) + "\n") if summary else "Limited information found in the retrieved pages.\n"
    out += "\nPOSSIBLE CAUSES:\n"
    out += "".join(f"{i}. {c}\n" for i, c in enumerate(causes, 1)) or "1. Not found in manual.\n"
    out += "\nSTEP-BY-STEP CORRECTIVE ACTIONS:\n"
    out += "".join(f"{i}. {f}\n" for i, f in enumerate(fixes, 1)) or "1. Not found in manual.\n"
    out += "\nSAFETY NOTES:\n"
    out += "".join(f"- {w}\n" for w in warnings) or "None stated in manual.\n"
    return out


# ── Main entry ────────────────────────────────────────────────────────────────
def generate_formatted_response(context: str, query: str, machine: str) -> str:
    if not context.strip():
        return (
            "PROBLEM SUMMARY:\n"
            f'No relevant content was found in the indexed manuals for "{query}" on {machine}.\n\n'
            "POSSIBLE CAUSES:\n"
            "1. Not found in manual.\n\n"
            "STEP-BY-STEP CORRECTIVE ACTIONS:\n"
            "1. Verify the correct manual has been uploaded in the Admin panel.\n"
            "2. Ensure the machine name matches exactly what was used during upload.\n"
            "3. Try rephrasing using terms from the manual.\n\n"
            "SAFETY NOTES:\n"
            "Do not attempt repairs without the official manual."
        )

    for fn in [_gemini, _anthropic, _openai, _groq, _ollama]:
        result = fn(context, query, machine)
        if result and not _is_bad(result):
            return result

    return _rule_based(context, query)
