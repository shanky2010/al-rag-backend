"""
IndustrialRAG - FastAPI Backend
Clean rewrite — stable, production ready
"""

import os
import re
import shutil
import hashlib
import time
import threading
import pickle
from pathlib import Path

import numpy as np
import faiss
import requests
import pdfplumber
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent))
from llm_formatter import generate_formatted_response

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
EMBEDDING_DIM       = 384    # all-MiniLM-L6-v2 — 90MB, fits Render free tier (512MB limit)
CHUNK_CHARS         = 600    # Smaller = one procedure per chunk = precise retrieval
OVERLAP_CHARS       = 120
RELEVANCE_THRESHOLD = 0.35   # cosine similarity floor for MiniLM

# ── OCR fallback ──────────────────────────────────────────────────────────────
def _ocr_page(page) -> str:
    try:
        import pytesseract
        img = page.to_image(resolution=150).original
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR failed: {e}", flush=True)
        return ""

# ── Chunking ──────────────────────────────────────────────────────────────────
def _chunk(text: str) -> list:
    # Strip image/drawing reference codes and page header noise
    text = re.sub(r'[A-Z]{2}\d{11,}', '', text)
    text = re.sub(r'5247-E\s+P-[\w().-]+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()

    # Detect section headers to prepend to each chunk
    # This tells the LLM exactly which part of the manual each chunk comes from
    header_re = re.compile(
        r'^((?:CHAPTER|SECTION|PART|WARNING|CAUTION|NOTE|PROCEDURE|'
        r'MAINTENANCE|TROUBLESHOOT|ALARM|ERROR|FAULT|SAFETY|'
        r'\d+[\.\d]*)\s*[:\-]?\s*.{0,60})$',
        re.MULTILINE | re.IGNORECASE
    )
    section = ""
    m = header_re.search(text)
    if m:
        section = m.group(1).strip()

    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            for sep in ['\n\n', '.\n', '. ', ';\n', '; ']:
                pos = text.rfind(sep, start + int(CHUNK_CHARS * 0.4), end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        raw = text[start:end].strip()

        # Update section if this chunk opens a new one
        nm = header_re.search(raw)
        if nm:
            section = nm.group(1).strip()

        # Prepend section so LLM knows context of every chunk
        chunk_text = f"[Section: {section}]\n{raw}" if section and not raw.startswith(section) else raw

        if len(raw) > 60:
            out.append(chunk_text)

        if end >= len(text):
            break
        start = end - OVERLAP_CHARS

    return out

# ── Local Embedder: all-MiniLM-L6-v2 ─────────────────────────────────────────
# 90MB RAM — fits comfortably in Render free tier (512MB limit)
# No API calls, no rate limits, no 429s, embeds 179 chunks in ~10 seconds on CPU

class LocalEmbedder:
    def __init__(self):
        self.dim   = EMBEDDING_DIM
        self.model = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading all-MiniLM-L6-v2 embedding model...", flush=True)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedder ready — local CPU model loaded.", flush=True)
        except Exception as e:
            print(f"WARNING: Model load failed: {e}. Using hash fallback.", flush=True)
            self.model = None

    def encode(self, texts: list, is_query: bool = False, **kwargs) -> np.ndarray:
        if self.model is None:
            return np.array([self._hash_fallback(t) for t in texts], dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity via dot product
        )
        return np.array(vecs, dtype=np.float32)

    def _hash_fallback(self, text: str) -> np.ndarray:
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens + bigrams:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


print("Initializing local embedder...", flush=True)
embedder = LocalEmbedder()
print("Embedder ready.", flush=True)

# ── FAISS Index ───────────────────────────────────────────────────────────────
def _make_index():
    return faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))

def _load_index():
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            loaded = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            if isinstance(meta, list):
                meta = {i: m for i, m in enumerate(meta)}
            if loaded.d != EMBEDDING_DIM:
                print(f"WARNING: Index dim mismatch. Resetting.", flush=True)
                return _make_index(), {}
            print(f"Loaded index with {loaded.ntotal} chunks.", flush=True)
            return loaded, meta
        except Exception as e:
            print(f"WARNING: Could not load index ({e}). Starting fresh.", flush=True)
    return _make_index(), {}

index, metadata_store = _load_index()
_id_counter = max(metadata_store.keys(), default=-1) + 1
_index_lock = threading.Lock()

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
    if not texts:
        return
    vecs = embedder.encode(texts)
    ids  = [_next_id() for _ in texts]
    with _index_lock:
        index.add_with_ids(np.array(vecs, dtype="float32"), np.array(ids, dtype="int64"))
        for vid, meta in zip(ids, metas):
            metadata_store[vid] = meta
        _save()

def _remove_by_source(source_pdf=None, source_excel=None) -> int:
    to_remove = [
        vid for vid, meta in list(metadata_store.items())
        if (source_pdf   and meta.get("source_pdf")   == source_pdf) or
           (source_excel and meta.get("source_excel") == source_excel)
    ]
    if to_remove:
        with _index_lock:
            index.remove_ids(np.array(to_remove, dtype="int64"))
            for vid in to_remove:
                del metadata_store[vid]
            _save()
    return len(to_remove)

def _retrieve(query: str, machine: str, top_manual=5, top_log=3) -> list:
    if index.ntotal == 0:
        return []
    q_vec = embedder.encode([query])
    k = min(index.ntotal, (top_manual + top_log) * 15)
    scores, ids = index.search(np.array(q_vec, dtype="float32"), k)
    print(f"RETRIEVE query='{query[:50]}' machine='{machine}' "
          f"top5={[round(float(s),3) for s in scores[0][:5]]} threshold={RELEVANCE_THRESHOLD}",
          flush=True)
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
            if len(logs) < top_log:
                logs.append(row)
        else:
            if len(manual) < top_manual:
                manual.append(row)
        if len(manual) >= top_manual and len(logs) >= top_log:
            break
    return manual + logs

def _get_machines() -> list:
    return sorted(set(m["machine_name"] for m in metadata_store.values() if "machine_name" in m))

def _get_files() -> list:
    files = {}
    for meta in metadata_store.values():
        key = meta.get("source_pdf") or meta.get("source_excel")
        if not key:
            continue
        if key not in files:
            files[key] = {"filename": key, "machine": meta.get("machine_name", ""),
                          "type": "pdf" if meta.get("source_pdf") else "excel", "chunks": 0}
        files[key]["chunks"] += 1
    return list(files.values())

# ── Job tracker ───────────────────────────────────────────────────────────────
_jobs: dict = {}

def _run_pdf_job(job_id: str, filepath: Path, filename: str, machine_name: str, replaced: int):
    _jobs[job_id] = {"status": "processing", "chunks": 0, "replaced": replaced, "error": None}
    try:
        texts, metas = [], []
        print(f"PDF job {job_id}: opening file {filepath}", flush=True)
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            print(f"PDF job {job_id}: {total_pages} pages, starting text extraction...", flush=True)
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = (page.extract_text() or "").strip()
                    if not page_text:
                        page_text = _ocr_page(page).strip()
                    if not page_text:
                        continue
                    page_chunks = list(_chunk(page_text))
                    for chunk in page_chunks:
                        texts.append(chunk)
                        metas.append({
                            "machine_name": machine_name,
                            "source_pdf":   filename,
                            "page_number":  page_num,
                            "source":       "manual",
                            "text":         chunk,
                        })
                    if page_num % 10 == 0 or page_num == total_pages:
                        print(f"PDF job {job_id}: page {page_num}/{total_pages}, total chunks so far: {len(texts)}", flush=True)
                        _jobs[job_id]["chunks"] = len(texts)
                except Exception as page_err:
                    print(f"PDF job {job_id}: ERROR on page {page_num}: {page_err}", flush=True)
                    continue
        print(f"PDF job {job_id}: extraction done. {len(texts)} chunks. Now embedding in batches...", flush=True)
        if not texts:
            filepath.unlink(missing_ok=True)
            _jobs[job_id] = {"status": "error", "chunks": 0, "replaced": replaced,
                             "error": "No readable text found in PDF."}
            return
        # Embed and store in batches of 50 to avoid memory issues and show progress
        BATCH = 50
        for i in range(0, len(texts), BATCH):
            batch_texts = texts[i:i+BATCH]
            batch_metas = metas[i:i+BATCH]
            print(f"PDF job {job_id}: embedding batch {i//BATCH + 1}/{(len(texts)-1)//BATCH + 1} ({len(batch_texts)} chunks)...", flush=True)
            _embed_and_store(batch_texts, batch_metas)
            _jobs[job_id]["chunks"] = i + len(batch_texts)
        _jobs[job_id] = {"status": "done", "chunks": len(texts), "replaced": replaced, "error": None}
        print(f"PDF job {job_id}: DONE — {len(texts)} chunks stored.", flush=True)
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"PDF job {job_id}: FATAL ERROR — {e}\n{err_detail}", flush=True)
        _jobs[job_id] = {"status": "error", "chunks": 0, "replaced": replaced, "error": str(e)}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="IndustrialRAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def force_cors(request, call_next):
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age":       "86400",
        })
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": index.ntotal,
            "embedder": "connected" if embedder.api_key else "hash_fallback",
            "machines": _get_machines()}

@app.get("/admin/machines")
def list_machines():
    return {"machines": _get_machines()}

@app.get("/admin/stats")
def get_stats():
    return {"total_chunks": index.ntotal, "machines": _get_machines(), "files": _get_files()}

@app.get("/admin/job/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return job

@app.post("/admin/upload/pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    machine_name: str = Form(...),
):
    machine_name = machine_name.strip()
    if not machine_name:
        raise HTTPException(400, "machine_name is required")
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', machine_name)
    safe_file = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
    filename  = f"{safe_name}_{safe_file}"
    filepath  = PDF_DIR / filename
    data      = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    replaced = _remove_by_source(source_pdf=filename)
    filepath.write_bytes(data)
    job_id = f"job_{int(time.time() * 1000)}_{safe_name}"
    background_tasks.add_task(_run_pdf_job, job_id, filepath, filename, machine_name, replaced)
    print(f"PDF upload queued: {filename} → {job_id}", flush=True)
    return {"status": "processing", "job_id": job_id, "machine": machine_name,
            "filename": filename, "message": "Polling /admin/job/{job_id} for status."}

@app.post("/admin/upload/excel")
async def upload_excel(file: UploadFile = File(...), machine_name: str = Form(...)):
    machine_name = machine_name.strip()
    if not machine_name:
        raise HTTPException(400, "machine_name is required")
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', machine_name)
    safe_file = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
    filename  = f"{safe_name}_{safe_file}"
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
        if not row_text.strip():
            continue
        texts.append(row_text)
        metas.append({"machine_name": machine_name, "source_excel": filename,
                      "log_id": f"row_{i}", "source": "repair_log", "text": row_text})
    if not texts:
        filepath.unlink(missing_ok=True)
        raise HTTPException(422, "No data rows found in file.")
    _embed_and_store(texts, metas)
    return {"status": "success", "machine": machine_name, "filename": filename,
            "rows_stored": len(texts), "old_rows_replaced": replaced}

@app.delete("/admin/delete/pdf/{filename:path}")
def delete_pdf(filename: str):
    removed = _remove_by_source(source_pdf=filename)
    filepath = PDF_DIR / filename
    existed = filepath.exists()
    if existed:
        filepath.unlink()
    if removed == 0 and not existed:
        raise HTTPException(404, f"'{filename}' not found")
    return {"status": "deleted", "filename": filename, "chunks_removed": removed}

@app.delete("/admin/delete/excel/{filename:path}")
def delete_excel(filename: str):
    removed = _remove_by_source(source_excel=filename)
    filepath = EXCEL_DIR / filename
    existed = filepath.exists()
    if existed:
        filepath.unlink()
    if removed == 0 and not existed:
        raise HTTPException(404, f"'{filename}' not found")
    return {"status": "deleted", "filename": filename, "chunks_removed": removed}


@app.post("/admin/test-pdf-parse")
async def test_pdf_parse(file: UploadFile = File(...)):
    """Debug endpoint: returns first 3 pages text extraction WITHOUT embedding."""
    import io
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    results = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages[:5]):  # first 5 pages only
                raw = (page.extract_text() or "").strip()
                ocr_text = ""
                if not raw:
                    ocr_text = _ocr_page(page).strip()
                results.append({
                    "page": i+1,
                    "pdfplumber_chars": len(raw),
                    "ocr_chars": len(ocr_text),
                    "sample": (raw or ocr_text)[:200],
                })
    except Exception as e:
        raise HTTPException(500, f"PDF parse error: {e}")
    return {"total_pages": total, "pages_sampled": len(results), "results": results}

@app.delete("/admin/reset")
def reset_all():
    global index, metadata_store, _id_counter
    with _index_lock:
        index = _make_index()
        metadata_store = {}
        _id_counter = 0
        _save()
    for d in [PDF_DIR, EXCEL_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    return {"status": "reset complete"}

class QueryRequest(BaseModel):
    query: str
    machine_name: str

@app.post("/query")
async def query_system(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if index.ntotal == 0:
        raise HTTPException(404, "No documents indexed yet. Upload a PDF first.")
    machines = _get_machines() if req.machine_name.lower() == "all" else [req.machine_name]
    if not machines:
        raise HTTPException(404, "No machines in knowledge base")
    results = []
    for machine in machines:
        chunks = _retrieve(req.query, machine)
        if not chunks:
            continue
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
            "log_chunks_used":    len(log_chunks),
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

@app.get("/pdf/{filename:path}")
def serve_pdf(filename: str):
    filepath = PDF_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(str(filepath), media_type="application/pdf")

# ── Keep-alive ────────────────────────────────────────────────────────────────
def _keep_alive():
    url = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not url:
        return
    while True:
        time.sleep(840)
        try:
            requests.get(f"{url}/health", timeout=10)
            print("Keep-alive ping sent.", flush=True)
        except Exception:
            pass

threading.Thread(target=_keep_alive, daemon=True).start()
