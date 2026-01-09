from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import chromadb
import re
import numpy as np
import unicodedata

# ===========================
# APP INIT
# ===========================
app = FastAPI(title="Enterprise Medical RAG")

# ===========================
# MODELS
# ===========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===========================
# VECTOR DB
# ===========================
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("medical_docs")

# ===========================
# SIMPLE SESSION MEMORY (DEMO)
# ===========================
ACTIVE_ENTITY = None

# ===========================
# TEXT NORMALIZATION
# ===========================
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[’']", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ===========================
# INTENT DETECTION
# ===========================
def detect_intent(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["dosage", "dose", "administration", "how much"]):
        return "dosage"
    if any(w in q for w in ["warning", "precaution", "side effect", "risk"]):
        return "safety"
    if any(w in q for w in ["used for", "indication", "treat"]):
        return "indication"
    if any(w in q for w in ["what is", "define"]):
        return "definition"
    return "general"

# ===========================
# SECTION ↔ INTENT MATCHING
# ===========================
def section_matches_intent(section: str, intent: str) -> bool:
    s = section.lower()
    if intent == "dosage":
        return any(w in s for w in ["dosage", "dose", "administration"])
    if intent == "safety":
        return any(w in s for w in ["warning", "precaution", "safety"])
    if intent == "indication":
        return any(w in s for w in ["indication", "usage", "treat"])
    if intent == "definition":
        return any(w in s for w in ["what is", "overview", "introduction"])
    return True

# ===========================
# SEMANTIC SIMILARITY
# ===========================
def semantic(a: str, b: str) -> float:
    a_vec = embed_model.encode(a, normalize_embeddings=True)
    b_vec = embed_model.encode(b, normalize_embeddings=True)
    return float(np.dot(a_vec, b_vec))

# ===========================
# PDF INGESTION
# ===========================
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDFs allowed")

    pdf = PdfReader(file.file)
    raw_text = ""

    for page in pdf.pages:
        if page.extract_text():
            raw_text += page.extract_text() + "\n"

    lines = raw_text.split("\n")

    entity = None
    section = None
    buffer = []
    indexed = 0

    for line in lines:
        clean = line.strip()
        if not clean:
            continue

        # ENTITY DETECTION (disease / drug name)
        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", clean):
            entity = clean
            continue

        # SECTION DETECTION
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", clean) or clean.isupper():
            # FLUSH PREVIOUS SECTION
            if buffer and entity and section:
                content = " ".join(buffer)
                emb = embed_model.encode(content).tolist()
                collection.add(
                    ids=[f"{entity}_{section}_{indexed}"],
                    embeddings=[emb],
                    documents=[content],
                    metadatas=[{
                        "entity": entity,
                        "entity_norm": normalize_text(entity),
                        "section": section
                    }]
                )
                indexed += 1

            section = clean
            buffer = []
        else:
            buffer.append(clean)

    # 🔴 CRITICAL: FLUSH LAST SECTION
    if buffer and entity and section:
        content = " ".join(buffer)
        emb = embed_model.encode(content).tolist()
        collection.add(
            ids=[f"{entity}_{section}_{indexed}"],
            embeddings=[emb],
            documents=[content],
            metadatas=[{
                "entity": entity,
                "entity_norm": normalize_text(entity),
                "section": section
            }]
        )
        indexed += 1

    return {
        "status": "Medical PDF ingested successfully",
        "sections_indexed": indexed
    }

# ===========================
# QUESTION MODEL
# ===========================
class Question(BaseModel):
    question: str

# ===========================
# ASK ENDPOINT
# ===========================
@app.post("/ask")
async def ask(data: Question):
    global ACTIVE_ENTITY

    question = data.question
    intent = detect_intent(question)
    q_norm = normalize_text(question)

    stored = collection.get(include=["documents", "metadatas"])

    # -----------------------
    # ENTITY RESOLUTION
    # -----------------------
    entities = {m["entity"]: m["entity_norm"] for m in stored["metadatas"]}

    explicit = [e for e, norm in entities.items() if norm in q_norm]

    if explicit:
        ACTIVE_ENTITY = explicit[0]

    if not explicit and not ACTIVE_ENTITY:
        if intent in ["dosage", "safety", "indication"]:
            return {
                "message": "Please specify the disease or drug name."
            }

    target_entity = ACTIVE_ENTITY

    # -----------------------
    # FILTER BY ENTITY + INTENT
    # -----------------------
    candidates = []
    for doc, meta in zip(stored["documents"], stored["metadatas"]):
        if meta["entity"] == target_entity:
            if section_matches_intent(meta["section"], intent):
                candidates.append((meta["section"], doc))

    if not candidates:
        return {
            "message": "No relevant medical information found."
        }

    # -----------------------
    # SEMANTIC RANKING
    # -----------------------
    scored = []
    for sec, doc in candidates:
        score = semantic(question, sec + " " + doc[:300])
        scored.append((score, sec, doc))

    scored.sort(reverse=True)
    best = scored[:1]  # 🔒 controlled output

    answer = "\n\n".join(f"{sec}\n{doc}" for _, sec, doc in best)

    return {
        "question": question,
        "detected_intent": intent,
        "active_entity": target_entity,
        "answer": answer,
        "disclaimer": "This information is extracted from medical documents and is not a substitute for professional medical advice."
    }
