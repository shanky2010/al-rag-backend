"""
IndustrialRAG - LLM Formatting Layer
Safety-critical prompt: LLM must only use provided context. No hallucination.
Cascade: Gemini Flash → Claude Haiku → OpenAI → Groq → Rule-based
"""

import os
import requests
from typing import Optional

# ── System Prompt ─────────────────────────────────────────────────────────────
# This prompt is designed for safety-critical industrial maintenance.
# The LLM is forbidden from using any knowledge outside the provided context.
# Wrong output = equipment damage + injury + financial loss.

SYSTEM_PROMPT = """You are a maintenance response assistant for Ashok Leyland industrial equipment.

You have been given CONTEXT — excerpts retrieved directly from the official machine manual and repair logs. Your ONLY job is to read that context and produce a structured maintenance response.

═══ ABSOLUTE RULES — VIOLATIONS ARE DANGEROUS ═══

RULE 1: Use ONLY information that is explicitly written in the CONTEXT below.
         Do NOT use any general engineering knowledge, assumptions, or training data.
         If it is not in the context, do NOT write it.

RULE 2: Every cause, every action step, every safety warning you write MUST be
         directly traceable to a sentence in the context.

RULE 3: If a section has no relevant information in the context, write exactly:
         "Not found in the manual."
         Do NOT fill empty sections with generic advice.

RULE 4: If the entire context is irrelevant to the query, respond with only:
         INSUFFICIENT_CONTEXT
         Do NOT attempt to answer from general knowledge.

RULE 5: Do NOT reorder or paraphrase steps in a way that changes their meaning.
         Maintenance procedures must be followed in the exact sequence given.

RULE 6: If the context contains a WARNING or CAUTION, it MUST appear in Safety Notes.

═══ OUTPUT FORMAT — USE EXACTLY THESE HEADINGS ═══

PROBLEM SUMMARY:
[2-3 sentences describing what the manual says about this specific fault or condition. Quote the manual's description where possible.]

POSSIBLE CAUSES:
1. [First cause — from context only]
2. [Second cause — from context only]
(Add more only if context supports them. Do not pad with guesses.)

STEP-BY-STEP CORRECTIVE ACTIONS:
1. [First step — exact sequence from manual]
2. [Second step]
(Follow the manual's sequence exactly. Do not add steps not in context.)

SAFETY NOTES:
[All WARNINGs, CAUTIONs, DANGERs from the context. If none: "None stated in manual."]"""


def _build_prompt(context: str, query: str, machine: str) -> str:
    # Deduplicate context blocks before sending to LLM
    seen = set()
    deduped = []
    for block in context.split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        fingerprint = block[:150].lower().replace(" ", "").replace("\n", "")
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped.append(block)

    clean_context = "\n\n---\n\n".join(deduped)

    return (
        f"MACHINE: {machine}\n"
        f"TECHNICIAN QUERY: {query}\n\n"
        "CONTEXT FROM MANUAL (retrieved chunks — this is all you may use):\n"
        "════════════════════════════════════════════════════════\n"
        f"{clean_context}\n"
        "════════════════════════════════════════════════════════\n\n"
        "Now produce the structured maintenance response using ONLY the context above. "
        "Do not use any knowledge outside this context. "
        "If a section cannot be answered from the context, write: Not found in the manual."
    )


def _is_bad(text: str) -> bool:
    """Detect responses that are empty, too short, or structurally malformed."""
    if not text or len(text.strip()) < 60:
        return True
    t = text.upper()
    if "INSUFFICIENT_CONTEXT" in t:
        return True
    # Must contain at least 2 of the 4 expected sections
    headers = ["PROBLEM SUMMARY", "POSSIBLE CAUSES", "CORRECTIVE ACTION", "SAFETY"]
    if sum(1 for h in headers if h in t) < 2:
        return True
    return False


# ── 1. Gemini Flash — PRIMARY ─────────────────────────────────────────────────
def _gemini(context: str, query: str, machine: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        full_prompt = f"{SYSTEM_PROMPT}\n\n{_build_prompt(context, query, machine)}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1500},
        }
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"LLM: Gemini Flash returned {len(result)} chars.", flush=True)
        return result
    except Exception as e:
        print(f"LLM: Gemini error: {e}", flush=True)
    return None


# ── 2. Anthropic Claude Haiku — fallback ─────────────────────────────────────
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


# ── 3. OpenAI GPT-4o-mini — fallback ─────────────────────────────────────────
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


# ── 4. Groq — fallback ───────────────────────────────────────────────────────
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


# ── 5. Rule-based fallback — last resort ─────────────────────────────────────
def _rule_based(context: str, query: str) -> str:
    """
    Keyword-based extraction from retrieved chunks.
    Only fires if ALL LLM providers fail.
    Still grounded in retrieved context — no hallucination possible.
    """
    print("LLM: Using rule-based fallback.", flush=True)
    sentences = [
        ln.strip() for ln in context.split("\n")
        if len(ln.strip()) > 20 and not ln.strip().startswith("[")
    ]
    cause_kw = ["cause", "caused by", "due to", "failure", "fault", "defect",
                "worn", "damaged", "failed", "broken", "loose", "blocked", "missing",
                "low", "obstacle", "emergency stop", "not pressed", "is off"]
    fix_kw   = ["replace", "check", "verify", "inspect", "clean", "adjust",
                "tighten", "reset", "test", "turn off", "turn on", "connect",
                "disconnect", "press", "ensure", "remove", "install", "lubricate",
                "charge", "release", "restart", "switch"]
    warn_kw  = ["warning", "caution", "danger", "do not", "must not", "hazard",
                "electric", "shock", "fire", "risk", "never", "only perform",
                "trained"]

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
    out += (" ".join(summary) + "\n") if summary else "Limited information found in retrieved pages.\n"
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
            f'No relevant content found in the indexed manuals for: "{query}" on {machine}.\n\n'
            "POSSIBLE CAUSES:\n"
            "1. Not found in manual.\n\n"
            "STEP-BY-STEP CORRECTIVE ACTIONS:\n"
            "1. Verify the correct manual has been uploaded in the Admin panel.\n"
            "2. Ensure the machine name matches exactly what was used during upload.\n"
            "3. Try rephrasing the query using terms from the manual.\n\n"
            "SAFETY NOTES:\n"
            "Do not attempt repairs without consulting the official manual."
        )

    for fn in [_gemini, _anthropic, _openai, _groq]:
        result = fn(context, query, machine)
        if result and not _is_bad(result):
            return result

    # Last resort — still grounded in retrieved context
    return _rule_based(context, query)
