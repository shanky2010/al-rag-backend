"""
IndustrialRAG - FastAPI Backend
Render-compatible: port binds immediately, model loads in background thread.
"""

import os
import sys
import shutil
import pickle
import threading
from pathlib import Path

import numpy as np
import faiss

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pdfplumber
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from llm_formatter import generate_formatted_response

# ── Paths ──
BASE_DIR   = Path(__file__).parent
PDF_DIR    = BASE_DIR / "uploads" / "pdfs"
EXCEL_DIR  = BASE_DIR / "uploads" / "excels"
VS_DIR     = BASE_DIR / "vectorstore"
INDEX_PATH = VS_DIR / "index.faiss"
META_PATH  = VS_DIR / "metadata.pkl"

for d in [PDF_DIR, EXCEL_DIR, VS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ──
EMBEDDING_DIM       = 384
CHUNK_CHARS         = 2400
OVERLAP_CHARS       = 400
RELEVANCE_THRESHOLD = 0.35

# ── Global state ──
embedder       = None
index          = None
metadata_store = {}
_id_counter    = 0
_ready         = False   # True once model + index are loaded

def _make_index():
    return faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))

def _load_index():
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            loaded = faiss.read_index(str(INDEX_PATH))
            if not hasattr(loaded, "remove_ids"):
                print("Old index format — rebuilding.")
                return _make_index(), {}
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            if isinstance(meta, list):
                meta = {i: m for i, m in enumerate(meta)}
            return loaded, meta
        except Exception as e:
            print(f"Could not load index ({e}). Starting fresh.")
    return _make_index(), {}

def _background_load():
    """Runs in a thread — loads model AFTER the HTTP server is already up."""
    global embedder, index, metadata_store, _id_counter, _ready
    try:
        print("==> [background] Loading sentence-transformer model...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("==> [background] Model loaded. Loading FAISS index...")
        index, metadata_store = _load_index()
        _id_counter = max(metadata_store.keys(), default=-1) + 1
        _ready = True
        print(f"==> [background] Ready. {index.ntotal} chunks indexed.")
    except Exception as e:
        print(f"==> [background] FATAL during startup: {e}")

# ── App — created immediately, model loads in background ──
app = FastAPI(title="IndustrialRAG")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

# Start background load as soon as the module is imported by uvicorn
_loader_thread = threading.Thread(target=_background_load, daemon=True)
_loader_thread.start()

# ── OCR fallback ──
def _ocr_page(page):
    try:
        import pytesseract
        img = page.to_image(resolution=300).original
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

# ── Vector store helpers ──
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
    vecs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
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
    if not _ready or index.ntotal == 0:
        return []
    q_vec = embedder.encode([query], normalize_embeddings=True)
    k = min(index.ntotal, (top_manual + top_log) * 10)
    distances, ids = index.search(np.array(q_vec, dtype="float32"), k)
    manual, logs = [], []
    for dist, idx in zip(distances[0], ids[0]):
        if idx < 0 or idx not in metadata_store: continue
        score = float(1.0 - dist / 2.0)
        if score < RELEVANCE_THRESHOLD: continue
        meta = metadata_store[idx]
        if machine.lower() != "all" and meta.get("machine_name","").lower() != machine.lower(): continue
        row = {**meta, "score": round(score, 3)}
        if meta.get("source") == "repair_log":
            if len(logs) < top_log: logs.append(row)
        else:
            if len(manual) < top_manual: manual.append(row)
        if len(manual) >= top_manual and len(logs) >= top_log: break
    return manual + logs

def _get_machines():
    return sorted(set(m["machine_name"] for m in metadata_store.values() if "machine_name" in m))

def _get_files():
    files = {}
    for meta in metadata_store.values():
        key = meta.get("source_pdf") or meta.get("source_excel")
        if not key: continue
        if key not in files:
            files[key] = {"filename": key, "machine": meta.get("machine_name",""),
                          "type": "pdf" if meta.get("source_pdf") else "excel", "chunks": 0}
        files[key]["chunks"] += 1
    return list(files.values())

def _require_ready():
    if not _ready:
        raise HTTPException(503, "System is still loading (model download). Please wait 1-2 minutes and retry.")

# ── Routes ──

@app.get("/health")
def health():
    return {"status": "ok" if _ready else "loading",
            "chunks_indexed": index.ntotal if _ready else 0}

@app.post("/admin/upload/pdf")
async def upload_pdf(file: UploadFile = File(...), machine_name: str = Form(...)):
    _require_ready()
    machine_name = machine_name.strip()
    if not machine_name: raise HTTPException(400, "machine_name required")
    filename = f"{machine_name.replace(' ','_')}_{file.filename}"
    filepath = PDF_DIR / filename
    data = await file.read()
    replaced = _remove_by_source(source_pdf=filename)
    filepath.write_bytes(data)
    texts, metas = [], []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    page_text = _ocr_page(page).strip()
                if not page_text: continue
                for chunk in _chunk(page_text):
                    texts.append(chunk)
                    metas.append({"machine_name": machine_name, "source_pdf": filename,
                                  "page_number": page_num, "source": "manual", "text": chunk})
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(500, f"PDF parsing failed: {e}")
    if not texts:
        filepath.unlink(missing_ok=True)
        raise HTTPException(422, "No readable text found in PDF.")
    _embed_and_store(texts, metas)
    return {"status": "success", "machine": machine_name, "filename": filename,
            "chunks_stored": len(texts), "old_chunks_replaced": replaced}

@app.post("/admin/upload/excel")
async def upload_excel(file: UploadFile = File(...), machine_name: str = Form(...)):
    _require_ready()
    machine_name = machine_name.strip()
    if not machine_name: raise HTTPException(400, "machine_name required")
    filename = f"{machine_name.replace(' ','_')}_{file.filename}"
    filepath = EXCEL_DIR / filename
    data = await file.read()
    replaced = _remove_by_source(source_excel=filename)
    filepath.write_bytes(data)
    try:
        df = pd.read_csv(filepath) if file.filename.lower().endswith(".csv") else pd.read_excel(filepath)
        df.columns = [str(c).strip().lower().replace(" ","_") for c in df.columns]
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
        raise HTTPException(422, "No data rows found.")
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
    _require_ready()
    if not req.query.strip(): raise HTTPException(400, "Query cannot be empty")
    machines = _get_machines() if req.machine_name.lower() == "all" else [req.machine_name]
    if not machines: raise HTTPException(404, "No machines in knowledge base")
    results = []
    for machine in machines:
        chunks = _retrieve(req.query, machine)
        if not chunks: continue
        manual_chunks = [c for c in chunks if c.get("source") == "manual"]
        log_chunks    = [c for c in chunks if c.get("source") == "repair_log"]
        context_parts, references, seen = [], [], set()
        for c in manual_chunks:
            context_parts.append(f"[MANUAL — Page {c.get('page_number','?')} | {c.get('source_pdf','')}]\n{c['text']}")
            key = f"{c.get('source_pdf','')}:{c.get('page_number',1)}"
            if key not in seen:
                seen.add(key)
                references.append({"pdf": c.get("source_pdf",""), "page": c.get("page_number",1)})
        for c in log_chunks:
            context_parts.append(f"[REPAIR LOG]\n{c['text']}")
        results.append({
            "machine": machine, "query": req.query,
            "context": "\n\n---\n\n".join(context_parts),
            "references": references,
            "manual_chunks_used": len(manual_chunks), "log_chunks_used": len(log_chunks),
            "_chunks": [{"text": c.get("text","")[:400], "score": c.get("score",0),
                         "source": c.get("source","manual"), "page_number": c.get("page_number"),
                         "source_pdf": c.get("source_pdf","")} for c in chunks]
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
    return {"total_chunks": index.ntotal if _ready else 0,
            "machines": _get_machines(), "files": _get_files()}

@app.get("/pdf/{filename}")
def serve_pdf(filename: str):
    filepath = PDF_DIR / filename
    if not filepath.exists(): raise HTTPException(404, "PDF not found")
    return FileResponse(str(filepath), media_type="application/pdf")
