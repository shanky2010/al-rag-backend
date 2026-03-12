"""
IndustrialRAG - FastAPI Backend
Memory-efficient version for Render free tier (512MB RAM)
- Gemini semantic embeddings (768-dim)
- Page-by-page PDF processing (low peak RAM)
- Background threading for uploads
"""

import os, shutil, hashlib, threading, uuid, time
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
import pickle
import requests

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pdfplumber
import sys
sys.path.append(str(Path(__file__).parent))
from llm_formatter import generate_formatted_response
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
PDF_DIR    = BASE_DIR / "uploads" / "pdfs"
EXCEL_DIR  = BASE_DIR / "uploads" / "excels"
VS_DIR     = BASE_DIR / "vectorstore"
INDEX_PATH = VS_DIR / "index.faiss"
META_PATH  = VS_DIR / "metadata.pkl"

for d in [PDF_DIR, EXCEL_DIR, VS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_DIM       = 768
CHUNK_CHARS         = 2400
OVERLAP_CHARS       = 400
RELEVANCE_THRESHOLD = 0.55
EMBED_BATCH_SIZE    = 5   # embed N chunks at a time to keep RAM low

# ── Gemini Embedder ────────────────────────────────────────────────────────────
class GeminiEmbedder:
    """Semantic embedder via Gemini embedding-001. ~0MB RAM (API-based)."""
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        self.dim = EMBEDDING_DIM

    def _embed_one(self, text: str) -> Optional[np.ndarray]:
        if not self.api_key:
            return None
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.url,
                    params={"key": self.api_key},
                    json={"model": "models/gemini-embedding-001",
                          "content": {"parts": [{"text": text[:8000]}]},
                          "outputDimensionality": self.dim},
                    timeout=20
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                vals = resp.json()["embedding"]["values"]
                vec = np.array(vals[:self.dim], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception as e:
                print(f"Gemini embed attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(1)
        return None

    def _hash_fallback(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        import re
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list) -> np.ndarray:
        results = []
        for i, text in enumerate(texts):
            vec = self._embed_one(text)
            if vec is None:
                vec = self._hash_fallback(text)
            results.append(vec)
            # Rate limit: small sleep every 10 requests
            if (i + 1) % 10 == 0:
                time.sleep(0.5)
        return np.array(results, dtype=np.float32)

print("Initializing Gemini embedder...", flush=True)
embedder = GeminiEmbedder()
print("Embedder ready.", flush=True)

# ── FAISS index ────────────────────────────────────────────────────────────────
_index_lock = threading.Lock()

def _make_index():
    return faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))

def _load_index():
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            loaded = faiss.read_index(str(INDEX_PATH))
            # Check dimension match
            inner = loaded.index if hasattr(loaded, 'index') else loaded
            if hasattr(inner, 'd') and inner.d != EMBEDDING_DIM:
                print(f"WARNING: Index dim {inner.d} != {EMBEDDING_DIM}. Wiping.", flush=True)
                return _make_index(), {}
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            if isinstance(meta, list):
                meta = {i: m for i, m in enumerate(meta)}
            print(f"Loaded index with {loaded.ntotal} chunks.", flush=True)
            return loaded, meta
        except Exception as e:
            print(f"WARNING: Could not load index ({e}). Starting fresh.", flush=True)
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

def _embed_and_store(texts: list, metas: list):
    """Embed in small batches and store immediately to keep RAM low."""
    if not texts:
        return
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i:i+EMBED_BATCH_SIZE]
        batch_metas = metas[i:i+EMBED_BATCH_SIZE]
        vecs = embedder.encode(batch_texts)
        with _index_lock:
            ids = [_next_id() for _ in batch_texts]
            index.add_with_ids(
                np.array(vecs, dtype="float32"),
                np.array(ids, dtype="int64")
            )
            for vid, meta in zip(ids, batch_metas):
                metadata_store[vid] = meta
            _save()

def _remove_by_source(source_pdf=None, source_excel=None):
    with _index_lock:
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

def _chunk(text: str) -> list:
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

def _ocr_page(page):
    try:
        import pytesseract
        img = page.to_image(resolution=150).original  # lower res = less RAM
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

# ── Background job tracking ────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock  = threading.Lock()

def _set_job(job_id, status, detail="", chunks=0):
    with _jobs_lock:
        _jobs[job_id] = {"status": status, "detail": detail, "chunks": chunks}

def _bg_index_pdf(job_id: str, filepath: Path, filename: str, machine_name: str):
    """Process PDF page-by-page — low peak RAM."""
    try:
        total_chunks = 0
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                _set_job(job_id, "indexing",
                         f"Processing page {page_num}/{total_pages}...", total_chunks)

                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    page_text = _ocr_page(page).strip()
                if not page_text:
                    continue

                chunks = _chunk(page_text)
                if not chunks:
                    continue

                texts = chunks
                metas = [{"machine_name": machine_name, "source_pdf": filename,
                          "page_number": page_num, "source": "manual", "text": c}
                         for c in chunks]

                _embed_and_store(texts, metas)
                total_chunks += len(chunks)

                # Explicitly free page memory
                del page_text, chunks, texts, metas

        if total_chunks == 0:
            filepath.unlink(missing_ok=True)
            _set_job(job_id, "error", "No readable text found in PDF.")
        else:
            _set_job(job_id, "done", f"Indexed {filename}", total_chunks)
            print(f"PDF indexed: {filename} → {total_chunks} chunks", flush=True)

    except Exception as e:
        filepath.unlink(missing_ok=True)
        _set_job(job_id, "error", str(e))
        print(f"PDF indexing error: {e}", flush=True)

def _bg_index_excel(job_id: str, filepath: Path, filename: str,
                    machine_name: str, original_filename: str):
    try:
        df = (pd.read_csv(filepath) if original_filename.lower().endswith(".csv")
              else pd.read_excel(filepath))
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        texts, metas = [], []
        for i, row in df.iterrows():
            parts = [f"{k.replace('_',' ').title()}: {str(v).strip()}"
                     for k, v in row.to_dict().items()
                     if pd.notna(v) and str(v).strip()]
            row_text = "\n".join(parts)
            if not row_text.strip():
                continue
            texts.append(row_text)
            metas.append({"machine_name": machine_name, "source_excel": filename,
                          "log_id": f"row_{i}", "source": "repair_log", "text": row_text})

        if not texts:
            filepath.unlink(missing_ok=True)
            _set_job(job_id, "error", "No data rows found in file.")
            return

        _embed_and_store(texts, metas)
        _set_job(job_id, "done", f"Indexed {filename}", len(texts))
        print(f"Excel indexed: {filename} → {len(texts)} rows", flush=True)

    except Exception as e:
        filepath.unlink(missing_ok=True)
        _set_job(job_id, "error", str(e))
        print(f"Excel indexing error: {e}", flush=True)

# ── Retrieval ──────────────────────────────────────────────────────────────────
def _retrieve(query: str, machine: str, top_manual=5, top_log=3):
    with _index_lock:
        if index.ntotal == 0:
            return []
        q_vec = embedder.encode([query])
        k = min(index.ntotal, (top_manual + top_log) * 10)
        scores, ids = index.search(np.array(q_vec, dtype="float32"), k)

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

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="IndustrialRAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

@app.post("/admin/upload/pdf", status_code=202)
async def upload_pdf(file: UploadFile = File(...), machine_name: str = Form(...)):
    machine_name = machine_name.strip()
    if not machine_name: raise HTTPException(400, "machine_name is required")
    safe_name = machine_name.replace(" ", "_")
    filename  = f"{safe_name}_{file.filename}"
    filepath  = PDF_DIR / filename
    data      = await file.read()
    _remove_by_source(source_pdf=filename)
    filepath.write_bytes(data)
    del data  # free RAM immediately

    job_id = str(uuid.uuid4())
    _set_job(job_id, "indexing", "Starting PDF indexing...")
    t = threading.Thread(target=_bg_index_pdf,
                         args=(job_id, filepath, filename, machine_name), daemon=True)
    t.start()
    return {"job_id": job_id, "filename": filename, "status": "indexing"}

@app.post("/admin/upload/excel", status_code=202)
async def upload_excel(file: UploadFile = File(...), machine_name: str = Form(...)):
    machine_name = machine_name.strip()
    if not machine_name: raise HTTPException(400, "machine_name is required")
    safe_name = machine_name.replace(" ", "_")
    filename  = f"{safe_name}_{file.filename}"
    filepath  = EXCEL_DIR / filename
    data      = await file.read()
    _remove_by_source(source_excel=filename)
    filepath.write_bytes(data)
    del data

    job_id = str(uuid.uuid4())
    _set_job(job_id, "indexing", "Starting Excel indexing...")
    t = threading.Thread(target=_bg_index_excel,
                         args=(job_id, filepath, filename, machine_name, file.filename), daemon=True)
    t.start()
    return {"job_id": job_id, "filename": filename, "status": "indexing"}

@app.get("/admin/job/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.delete("/admin/delete/pdf/{filename}")
def delete_pdf(filename: str):
    removed  = _remove_by_source(source_pdf=filename)
    filepath = PDF_DIR / filename
    existed  = filepath.exists()
    if existed: filepath.unlink()
    if removed == 0 and not existed:
        raise HTTPException(404, f"'{filename}' not found")
    return {"status": "deleted", "filename": filename, "chunks_removed": removed}

@app.delete("/admin/delete/excel/{filename}")
def delete_excel(filename: str):
    removed  = _remove_by_source(source_excel=filename)
    filepath = EXCEL_DIR / filename
    existed  = filepath.exists()
    if existed: filepath.unlink()
    if removed == 0 and not existed:
        raise HTTPException(404, f"'{filename}' not found")
    return {"status": "deleted", "filename": filename, "chunks_removed": removed}

@app.delete("/admin/reset")
def reset_all():
    global index, metadata_store, _id_counter
    with _index_lock:
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
            context_parts.append(
                f"[MANUAL - Page {c.get('page_number','?')} | {c.get('source_pdf','')}]\n{c['text']}")
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
    return {"status": "ok", "chunks_indexed": index.ntotal, "embedder": "gemini-embedding-001"}
