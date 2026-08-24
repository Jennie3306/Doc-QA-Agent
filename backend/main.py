"""FastAPI backend for the NVIDIA Document Q&A Agent."""

import asyncio
import warnings

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_graph import build_agent, initial_state
from config import settings
from core import ingest, store

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


# ── Models ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    decision: str
    chunks: list[str]
    chunk_scores: list[float]  # real similarity per chunk
    chunk_pages: list[int]
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
def status(x_session_id: str | None = Header(default=None)):
    try:
        n = store.count(x_session_id)
        return {"loaded": n > 0, "chunk_count": n}
    except Exception as e:  # noqa: BLE001 - health check must never 500
        print(f"[Status] {e}")
        return {"loaded": False, "chunk_count": 0}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(default=None),
):
    """Upload and index a PDF into this session's collection."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    content = await file.read()

    if len(content) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File exceeds {limit_mb} MB")

    # Magic bytes, not the extension: renaming anything to .pdf used to pass.
    if not ingest.looks_like_pdf(content):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    # to_thread is the Phase 1 async fix. ingest_pdf_bytes() makes blocking
    # network calls; running it directly on the event loop froze every
    # other request - including /status - for the whole upload.
    try:
        n = await asyncio.to_thread(ingest.ingest_pdf_bytes, content, file.filename, x_session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - fitz raises many types on bad PDFs
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}") from e

    return UploadResponse(
        chunk_count=n,
        filename=file.filename,
        message=f"Successfully indexed {n} chunks",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    x_session_id: str | None = Header(default=None),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    history = _trim_history(request.chat_history)
    state = initial_state(
        request.question,
        history,
        session_id=x_session_id or store.DEFAULT_SESSION,
    )

    # agent.invoke() is synchronous and calls the NIM API - same event-loop
    # problem as upload, just shorter.
    result = await asyncio.to_thread(agent.invoke, state)

    return ChatResponse(
        answer=result["answer"],
        decision=result["decision"],
        chunks=result["retrieved_chunks"],
        chunk_scores=result["chunk_scores"],
        chunk_pages=result["chunk_pages"],
        confidence=result["retrieval_confidence"],
        trace=_build_trace(
            result["decision"],
            result["retrieval_confidence"],
            len(result["retrieved_chunks"]),
        ),
    )


@app.post("/reset")
async def reset(x_session_id: str | None = Header(default=None)):
    """Let a user clear their own indexed document."""
    store.reset_collection(x_session_id)
    return {"status": "cleared"}


# ── Helpers ────────────────────────────────────────────────────
def _trim_history(history: list[dict]) -> list[dict]:
    """Cap history by both message count and size.

    chat_history comes from the client, so it is untrusted input: without a
    size cap a long conversation (or a crafted request) can push the prompt
    past the context window.
    """
    trimmed = history[-settings.max_history_messages :]
    return [
        {**m, "content": str(m.get("content", ""))[: settings.max_message_chars]} for m in trimmed
    ]


def _build_trace(decision: str, confidence: float, n_chunks: int) -> list[dict]:
    trace = [{"icon": "▶", "label": "Router", "value": decision.upper(), "color": GRN}]

    if decision == "retrieve":
        below = confidence < settings.confidence_threshold
        trace += [
            {
                "icon": "◎",
                "label": "Vector search",
                "value": f"{n_chunks} chunks retrieved (cosine)",
                "color": GRN,
            },
            {
                "icon": "◎",
                "label": "Top-1 similarity",
                "value": f"{confidence:.3f}"
                + (f" (below {settings.confidence_threshold})" if below else ""),
                "color": AMBER if below else GRN,
            },
            {
                "icon": "◉",
                "label": "Generator",
                "value": "Fallback - low confidence" if below else "Answer produced",
                "color": AMBER if below else GRN,
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
