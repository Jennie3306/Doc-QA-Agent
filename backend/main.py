"""FastAPI backend for the NVIDIA Document Q&A Agent."""

import os
import tempfile
import warnings

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from agent_graph import build_agent, initial_state
from config import settings
from core import store
from core.nim_client import embed

warnings.filterwarnings("ignore")

app = FastAPI(title="NVIDIA Document Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()

GRN = "#76B900"
PURPLE = "#a855f7"
AMBER = "#f59e0b"


# ── Request/response models ────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    decision: str
    chunks: list[str]
    confidence: float
    trace: list[dict]


class UploadResponse(BaseModel):
    chunk_count: int
    filename: str
    message: str


# ── Routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "NVIDIA Document Agent API running"}


@app.get("/status")
def status():
    try:
        n = store.count()
        return {"loaded": n > 0, "chunk_count": n}
    except Exception as e:  # noqa: BLE001 - health check must never 500
        # Printed rather than swallowed: without this, a broken ChromaDB
        # looks identical to "no document uploaded yet".
        print(f"[Status] {e}")
        return {"loaded": False, "chunk_count": 0}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF.

    Phase 1: this is declared `async` but embeds chunks with blocking calls,
    so it holds the event loop for the entire upload and every other request
    (including /status) stalls. Fixed by dropping async / using to_thread,
    and by batching the embedding calls.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        full_text = "".join(
            f"\n--- Page {i + 1} ---\n{page.get_text()}" for i, page in enumerate(doc)
        )
        doc.close()
    except Exception as e:  # noqa: BLE001 - fitz raises many types on bad PDFs
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}") from e
    finally:
        # The original unlinked before this point, so a corrupt PDF leaked
        # the temp file.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    ).split_text(full_text)

    col = store.reset_collection()
    for i, chunk in enumerate(chunks):
        col.add(
            ids=[f"chunk_{i}"],
            embeddings=[embed(chunk, input_type="passage")],
            documents=[chunk],
            metadatas=[{"chunk_index": i, "source": file.filename}],
        )

    return UploadResponse(
        chunk_count=len(chunks),
        filename=file.filename,
        message=f"Successfully indexed {len(chunks)} chunks",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Answer a question against the indexed document."""
    history = request.chat_history[-settings.max_history_messages :]
    result = agent.invoke(initial_state(request.question, history))

    return ChatResponse(
        answer=result["answer"],
        decision=result["decision"],
        chunks=result["retrieved_chunks"],
        confidence=result["retrieval_confidence"],
        trace=_build_trace(result["decision"], result["retrieval_confidence"]),
    )


def _build_trace(decision: str, confidence: float) -> list[dict]:
    """Build the Evidence panel trace.

    Phase 5: this is reconstructed from the final decision rather than
    logged by the nodes as they run. It happens to be accurate only because
    the graph is simple enough that the path is inferable from the decision.
    """
    trace = [{"icon": "▶", "label": "Router", "value": decision.upper(), "color": GRN}]

    if decision == "retrieve":
        trace += [
            {
                "icon": "◎",
                "label": "Vector search",
                "value": f"Top-{settings.final_top_k} chunks retrieved",
                "color": GRN,
            },
            {
                "icon": "◎",
                "label": "Confidence",
                "value": f"{confidence:.2f}",
                "color": GRN,
            },
            {
                "icon": "◉",
                "label": "Generator",
                "value": "Answer produced",
                "color": GRN,
            },
        ]
    elif decision == "meta":
        trace += [
            {
                "icon": "◎",
                "label": "Memory",
                "value": "Conversation history used",
                "color": PURPLE,
            },
            {
                "icon": "◉",
                "label": "Summary",
                "value": "Generated from context",
                "color": PURPLE,
            },
        ]
    else:
        trace += [
            {
                "icon": "◉",
                "label": "Clarifier",
                "value": "Asking for more detail",
                "color": AMBER,
            }
        ]

    return trace