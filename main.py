"""
IndustrialRAG - FastAPI Backend
Uses Google Gemini API for semantic embeddings (free tier)
"""

import os
import shutil
import hashlib
import time
from pathlib import Path

import numpy as np
import faiss
import pickle
import requests

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import threading

import pdfplumber

def _ocr_page(page):
    try:
        import pytesseract
        img = page.to_image(resolution=300).original
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

import sys
sys.path.append(str(Path(__file__).parent))
from llm_formatter import generate_formatted_response
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = Path(os.environ.get("DATA_DIR", str(BASE_DIR)))
PDF_DIR    = DATA_DIR / "uploads" / "pdfs"
EXCEL_DIR  = DATA_DIR / "uploads" / "excels"
VS_DIR     = DATA_DIR / "vectorstore"
INDEX_PATH = VS_DIR / "index.faiss"
META_PATH  = VS_DIR / "metadata.pkl"

for d in [PDF_DIR, EXCEL_DIR, VS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM       = 768          # Gemini text-embedding-004 outputs 768-dim
CHUNK_CHARS         = 2400
OVERLAP_CHARS       = 400
RELEVANCE_THRESHOLD = 0.0          # Disabled for debugging — set back to 0.45 once working

# ── Gemini Embedder ───────────────────────────────────────────────────────────
class GeminiEmbedder:
    """
    Semantic embedder using Google Gemini gemini-embedding-001 API.
    Free tier: 1500 requests/minute — plenty for a prototype.
    Dim: 768 (truncated from 3072 via output_dimensionality param).
    True semantic understanding (synonyms, context, etc.)
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.url     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        self.dim     = EMBEDDING_DIM
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not set. Falling back to hash embedder.", flush=True)

    def _embed_one(self, text: str) -> np.ndarray:
        """Call Gemini API for a single text. Returns normalized 768-dim vector."""
        if not self.api_key:
            return self._hash_fallback(text)
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text[:8000]}]},
            "outputDimensionality": 768   # Truncate from 3072 → 768 to save RAM/storage
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=15
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                vec = np.array(resp.json()["embedding"]["values"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec
            except Exception as e:
                print(f"Gemini embed error (attempt {attempt+1}): {e}", flush=True)
                time.sleep(1)
        # All attempts failed — fall back to hash
        return self._hash_fallback(text)

    def _hash_fallback(self, text: str) -> np.ndarray:
        """MD5 hash fallback if API is unavailable."""
        import re
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens + bigrams:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts: list, normalize_embeddings=True, show_progress_bar=False) -> np.ndarray:
        """Encode a list of texts. Rate-limited to avoid hitting API limits."""
        vecs = []
        for i, text in enumerate(texts):
            vec = self._embed_one(text)
            vecs.append(vec)
            # Small delay every 10 requests to stay within free tier limits
            if i > 0 and i % 10 == 0:
                time.sleep(0.5)
        return np.array(vecs, dtype=np.float32)


print("Initializing Gemini embedder...", flush=True)
embedder = GeminiEmbedder()
print("Embedder ready.", flush=True)

# ── FAISS Index ───────────────────────────────────────────────────────────────
def _make_index():
    return faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))

def _load_index():
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            loaded = faiss.read_index(str(INDEX_PATH))
            if not hasattr(loaded, "remove_ids"):
                return _make_index(), {}
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            if isinstance(meta, list):
                meta = {i: m for i, m in enumerate(meta)}
            # Dimension check — if old index was 512-dim, start fresh
            if loaded.d != EMBEDDING_DIM:
                print(f"WARNING: Index dimension mismatch ({loaded.d} vs {EMBEDDING_DIM}). Starting fresh.", flush=True)
                return _make_index(), {}
            print(f"Loaded index with {loaded.ntotal} chunks.", flush=True)
            return loaded, meta
        except Exception as e:
            print(f"WARNING: Could not load index ({e}). Starting fresh.")
    return _make_index(), {}

index, metadata_store = _load_index()
_id_counter = max(metadata_store.keys(), default=-1) + 1

def _next_id():
    global _id_counter
    v = _id_counter
    _id_counter += 1
    return v

def _save():
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

def _embed_and_store(texts, metas):
    if not texts:
        return
    vecs = embedder.encode(texts, normalize_embeddings=True)
    ids  = [_next_id() for _ in texts]
    index.add_with_ids(np.array(vecs, dtype="float32"), np.array(ids, dtype="int64"))
    for vid, meta in zip(ids, metas):
        metadata_store[vid] = meta
    _save()

def _remove_by_source(source_pdf=None, source_excel=None):
    to_remove = []
    for vid, meta in list(metadata_store.items()):
        if source_pdf   and meta.get("source_pdf")   == source_pdf:   to_remove.append(vid)
        if source_excel and meta.get("source_excel") == source_excel: to_remove.append(vid)
    if to_remove:
        index.remove_ids(np.array(to_remove, dtype="int64"))
        for vid in to_remove:
            del metadata_store[vid]
        _save()
    return len(to_remove)

def _chunk(text):
    out, start = [], 0
    while start < len(text):
        end   = min(start + CHUNK_CHARS, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 60:
            out.append(chunk)
        if end == len(text):
            break
        start += CHUNK_CHARS - OVERLAP_CHARS
    return out

def _retrieve(query, machine, top_manual=5, top_log=3):
    if index.ntotal == 0:
        return []
    q_vec = embedder.encode([query], normalize_embeddings=True)
    k = min(index.ntotal, (top_manual + top_log) * 10)
    scores, ids = index.search(np.array(q_vec, dtype="float32"), k)
    top_scores = [round(float(s), 3) for s in scores[0][:5]]
    print(f"DEBUG _retrieve: query='{query}' machine='{machine}' top5_scores={top_scores} threshold={RELEVANCE_THRESHOLD}", flush=True)
    manual, logs = [], []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx not in metadata_store:
            continue
        if float(score) < RELEVANCE_THRESHOLD:
            continue
        meta = metadata_store[idx]
        if machine.lower() != "all" and meta.get("machine_name", "").lower() != machine.lower():
            continue
        row = {**meta, "score": round(float(score), 3)}
        if meta.get("source") == "repair_log":
            if len(logs) < top_log: logs.append(row)
        else:
            if len(manual) < top_manual: manual.append(row)
        if len(manual) >= top_manual and len(logs) >= top_log:
            break
    return manual + logs

def _get_machines():
    return sorted(set(m["machine_name"] for m in metadata_store.values() if "machine_name" in m))

def _get_files():
    files = {}
    for meta in metadata_store.values():
        key = meta.get("source_pdf") or meta.get("source_excel")
        if not key: continue
        if key not in files:
            files[key] = {"filename": key, "machine": meta.get("machine_name", ""),
                          "type": "pdf" if meta.get("source_pdf") else "excel", "chunks": 0}
        files[key]["chunks"] += 1
    return list(files.values())

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="IndustrialRAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Explicit CORS headers on every response (belt-and-suspenders for Render proxy)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class ForceCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        if request.method == "OPTIONS":
            from fastapi.responses import Response
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "86400",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

app.add_middleware(ForceCORSMiddleware)
app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

# ── Upload job tracker ────────────────────────────────────────────────────────
_upload_jobs: dict = {}  # job_id -> {status, chunks, error}

def _process_pdf_background(job_id: str, filepath: Path, filename: str, machine_name: str, replaced: int):
    _upload_jobs[job_id] = {"status": "processing", "chunks": 0, "error": None, "replaced": replaced}
    texts, metas = [], []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    page_text = _ocr_page(page).strip()
                if not page_text:
                    continue
                for chunk in _chunk(page_text):
                    texts.append(chunk)
                    metas.append({"machine_name": machine_name, "source_pdf": filename,
                                  "page_number": page_num, "source": "manual", "text": chunk})
        if not texts:
            filepath.unlink(missing_ok=True)
            _upload_jobs[job_id] = {"status": "error", "chunks": 0, "error": "No readable text found in PDF.", "replaced": replaced}
            return
        _embed_and_store(texts, metas)
        _upload_jobs[job_id] = {"status": "done", "chunks": len(texts), "error": None, "replaced": replaced}
        print(f"PDF job {job_id} done: {len(texts)} chunks stored.", flush=True)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        _upload_jobs[job_id] = {"status": "error", "chunks": 0, "error": str(e), "replaced": replaced}
        print(f"PDF job {job_id} error: {e}", flush=True)

@app.post("/admin/upload/pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), machine_name: str = Form(...)):
    machine_name = machine_name.strip()
    if not machine_name:
        raise HTTPException(400, "machine_name is required")
    safe_name = machine_name.replace(" ", "_")
    filename  = f"{safe_name}_{file.filename}"
    filepath  = PDF_DIR / filename
    data      = await file.read()
    replaced  = _remove_by_source(source_pdf=filename)
    filepath.write_bytes(data)
    job_id = f"{filename}_{int(time.time())}"
    background_tasks.add_task(_process_pdf_background, job_id, filepath, filename, machine_name, replaced)
    return {"status": "processing", "job_id": job_id, "machine": machine_name,
            "filename": filename, "message": "PDF is being processed. Poll /admin/job/{job_id} for status."}

@app.get("/admin/job/{job_id}")
def get_job_status(job_id: str):
    job = _upload_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.post("/admin/upload/excel")
async def upload_excel(file: UploadFile = File(...), machine_name: str = Form(...)):
    machine_name = machine_name.strip()
    if not machine_name: raise HTTPException(400, "machine_name is required")
    safe_name = machine_name.replace(" ", "_")
    filename  = f"{safe_name}_{file.filename}"
    filepath  = EXCEL_DIR / filename
    data      = await file.read()
    replaced  = _remove_by_source(source_excel=filename)
    filepath.write_bytes(data)
    try:
        df = pd.read_csv(filepath) if file.filename.lower().endswith(".csv") else pd.read_excel(filepath)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(500, f"File parsing failed: {e}")
    texts, metas = [], []
    for i, row in df.iterrows():
        parts = [f"{k.replace('_',' ').title()}: {str(v).strip()}"
                 for k, v in row.to_dict().items() if pd.notna(v) and str(v).strip()]
        row_text = "\n".join(parts)
        if not row_text.strip(): continue
        texts.append(row_text)
        metas.append({"machine_name": machine_name, "source_excel": filename,
                      "log_id": f"row_{i}", "source": "repair_log", "text": row_text})
    if not texts:
        filepath.unlink(missing_ok=True)
        raise HTTPException(422, "No data rows found in file.")
    _embed_and_store(texts, metas)
    return {"status": "success", "machine": machine_name, "filename": filename,
            "rows_stored": len(texts), "old_rows_replaced": replaced}

@app.delete("/admin/delete/pdf/{filename}")
def delete_pdf(filename: str):
    removed = _remove_by_source(source_pdf=filename)
    filepath = PDF_DIR / filename
    existed = filepath.exists()
    if existed: filepath.unlink()
    if removed == 0 and not existed: raise HTTPException(404, f"'{filename}' not found")
    return {"status": "deleted", "filename": filename, "chunks_removed": removed}

@app.delete("/admin/delete/excel/{filename}")
def delete_excel(filename: str):
    removed = _remove_by_source(source_excel=filename)
    filepath = EXCEL_DIR / filename
    existed = filepath.exists()
    if existed: filepath.unlink()
    if removed == 0 and not existed: raise HTTPException(404, f"'{filename}' not found")
    return {"status": "deleted", "filename": filename, "chunks_removed": removed}

@app.delete("/admin/reset")
def reset_all():
    global index, metadata_store, _id_counter
    index = _make_index(); metadata_store = {}; _id_counter = 0
    _save()
    for d in [PDF_DIR, EXCEL_DIR]:
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
    return {"status": "reset complete"}

class QueryRequest(BaseModel):
    query: str
    machine_name: str

@app.post("/query")
async def query_system(req: QueryRequest):
    if not req.query.strip(): raise HTTPException(400, "Query cannot be empty")
    machines = _get_machines() if req.machine_name.lower() == "all" else [req.machine_name]
    if not machines: raise HTTPException(404, "No machines in knowledge base")
    results = []
    for machine in machines:
        chunks = _retrieve(req.query, machine)
        if not chunks: continue
        manual_chunks = [c for c in chunks if c.get("source") == "manual"]
        log_chunks    = [c for c in chunks if c.get("source") == "repair_log"]
        context_parts, references, seen_refs = [], [], set()
        for c in manual_chunks:
            context_parts.append(f"[MANUAL - Page {c.get('page_number','?')} | {c.get('source_pdf','')}]\n{c['text']}")
            key = f"{c.get('source_pdf','')}:{c.get('page_number',1)}"
            if key not in seen_refs:
                seen_refs.add(key)
                references.append({"pdf": c.get("source_pdf",""), "page": c.get("page_number",1)})
        for c in log_chunks:
            context_parts.append(f"[REPAIR LOG]\n{c['text']}")
        results.append({
            "machine": machine, "query": req.query,
            "context": "\n\n---\n\n".join(context_parts),
            "references": references,
            "manual_chunks_used": len(manual_chunks),
            "log_chunks_used": len(log_chunks),
            "_chunks": [{"text": c.get("text","")[:400], "score": c.get("score",0),
                         "source": c.get("source","manual"), "page_number": c.get("page_number"),
                         "source_pdf": c.get("source_pdf","")} for c in chunks],
        })
    return {"results": results}

class FormatRequest(BaseModel):
    context: str
    query: str
    machine: str

@app.post("/format")
async def format_response(req: FormatRequest):
    return {"formatted": generate_formatted_response(req.context, req.query, req.machine)}

@app.get("/admin/machines")
def list_machines():
    return {"machines": _get_machines()}

@app.get("/admin/stats")
def get_stats():
    return {"total_chunks": index.ntotal, "machines": _get_machines(), "files": _get_files()}

@app.get("/pdf/{filename}")
def serve_pdf(filename: str):
    filepath = PDF_DIR / filename
    if not filepath.exists(): raise HTTPException(404, "PDF not found")
    return FileResponse(str(filepath), media_type="application/pdf")

@app.get("/health")
def health():
    gemini_status = "connected" if embedder.api_key else "missing_key_using_fallback"
    return {"status": "ok", "chunks_indexed": index.ntotal, "embedder": gemini_status}

@app.post("/debug/scores")
async def debug_scores(req: QueryRequest):
    """Returns raw similarity scores for debugging."""
    if index.ntotal == 0:
        return {"error": "Index is empty"}
    q_vec = embedder.encode([req.query], normalize_embeddings=True)
    k = min(index.ntotal, 20)
    scores, ids = index.search(np.array(q_vec, dtype="float32"), k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx not in metadata_store:
            continue
        meta = metadata_store[idx]
        results.append({
            "score": round(float(score), 4),
            "machine": meta.get("machine_name"),
            "page": meta.get("page_number"),
            "text_preview": meta.get("text", "")[:100]
        })
    return {"query": req.query, "total_indexed": index.ntotal, "results": results}

# ── Keep-alive (prevents Render free tier from sleeping) ─────────────────────
def _keep_alive():
    url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not url:
        return
    while True:
        time.sleep(840)  # ping every 14 minutes
        try:
            requests.get(f"{url}/health", timeout=10)
            print("Keep-alive ping sent.", flush=True)
        except Exception:
            pass

threading.Thread(target=_keep_alive, daemon=True).start()
