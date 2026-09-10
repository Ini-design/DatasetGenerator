import json
import re
import logging
from collections import Counter
import numpy as np
import PyPDF2
import torch
import os 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from functools import lru_cache 

HF_TOKEN = os.getenv("hf_token")

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise RuntimeError("HF_TOKEN not found")
    
# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyntheticDataset")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL LOADING (SAFE + 8GB FRIENDLY)
def load_models():
    logger.info("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large",
       dtype=torch.float32
       ).to(DEVICE)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    return tokenizer, model, embed_model     
tokenizer, model, embed_model = load_models()
model.eval()

@lru_cache(maxsize=3000)  # reduce for 8GB RAM
def get_embedding_cached(text):
    emb = embed_model.encode(text, normalize_embeddings=True).astype(np.float32)
    return emb

# PDF LOADING
def load_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        if not text.strip():
            raise ValueError("Empty PDF")
        return text
    except Exception as e:
        logger.error(e)
        raise RuntimeError("Invalid or corrupted PDF")

# TEXT CLEANING
def clean_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[_\-=\*]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)

    return text.replace("\n", " ").strip()
    
def lexical_overlap(answer, context, min_overlap=0.22):
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    return overlap >= min_overlap 

def remove_repeated_lines(text):
    lines = re.split(r'(?<=[.!?])\s+', text)
    counts = Counter(lines)
    filtered = [
        l for l in lines
        if counts[l] < 5
    ]
    return ". ".join(filtered)
    

# TEXT CLEANING   
def get_embedding(text):
    return np.array(get_embedding_cached(text)) 
    
def load_and_clean(file):
    raw = load_pdf(file)
    text = clean_text(raw)
    text = remove_repeated_lines(text)
    return text

def trim_to_token_limit(text, tokenizer, limit=400):
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=limit,
        return_tensors=None
    )
    input_ids = tokens["input_ids"]
    if len(input_ids) <= limit:
        return text
    trimmed = tokenizer.decode(
        input_ids[:limit],
        skip_special_tokens=True
    )
    sentences = re.split(r'(?<=[.!?])\s+', trimmed)
    return (
        " ".join(sentences[:-1])
        if len(sentences) > 1
        else trimmed
    )
# CHUNKING (GENERATOR → LOW RAM)
def chunk_text(text, tokenizer, max_tokens=256, overlap=50):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )
    step = max_tokens - overlap
    for i in range(0, len(token_ids), step):
        chunk_ids = token_ids[i:i + max_tokens]
        chunk = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        ).strip()
        if chunk:
            yield chunk

# NOISE FILTERING
def is_low_information(chunk):
    if len(chunk.split()) < 25:
        return True
    digit_ratio = sum(c.isdigit() for c in chunk) / max(len(chunk), 1)
    if digit_ratio > 0.3:
        return True
    return False

# SEMANTIC DEDUPLICATION 
def deduplicate_chunks(chunks, threshold=0.92):
    if not chunks:
        return []
    embeddings = embed_model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32
    )
    kept_chunks = []
    kept_embeddings = []
    for chunk, emb in zip(chunks, embeddings):
        if not kept_embeddings:
            kept_chunks.append(chunk)
            kept_embeddings.append(emb)
            continue
        sims = cosine_similarity(
            [emb],
            kept_embeddings
        )[0]
        if sims.max() < threshold:
            kept_chunks.append(chunk)
            kept_embeddings.append(emb)
    return kept_chunks 
def semantic_grounding_check(context, answer, threshold=0.55):
    sentences = [
        s.strip()
        for s in re.split(
            r'(?<=[.!?])\s+',
            context
        )
        if len(s.strip()) > 20
    ]
    if not sentences:
        return False
    sentence_embeddings = embed_model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32
    )
    answer_embedding = get_embedding(answer)
    sims = cosine_similarity(
        [answer_embedding],
        sentence_embeddings
    )[0]
    return sims.max() >= threshold 
    
# SAFE GENERATION
def safe_generate(prompt, max_len=128):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_len,
            do_sample=False,
            num_beams=2,
            early_stopping=True
        )
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

def generate_questions(chunk, n=3):
    prompt = f"""
Generate {n} factual questions answerable ONLY from the context.
Return one question per line.
Do NOT add numbering.
Rules:
- Questions must be answerable using one sentence.
- Do not reference section numbers.
- Only ask about concepts.
- Return JSON.
CONTEXT:
{chunk}
"""
    text = safe_generate(prompt)
    # robust parsing
    lines = text.split("\n")
    
    questions = []  
    for line in lines: 
        line = line.strip()
        if not line:
            continue
        if len(line) < 15:
            continue
        if "?" not in line:
            continue
        if not line.endswith("?"):
            line += "?"
        questions.append(line) 
    return questions  
def generate_answer(question, context):
    prompt = f"""
You are a factual question answering system.
INSTRUCTIONS:
Answer the question using ONLY the information inside the context.
RULES:
- If the answer is not explicitly stated, output EXACTLY: NOT_FOUND
- Do NOT guess.
- Do NOT add external knowledge.
- Keep the answer concise (1–3 sentences).
CONTEXT:
{context}
QUESTION:
{question}
FINAL ANSWER:
"""
    return safe_generate(prompt, 200)

def verify_answer(context, answer, threshold=0.55):
    sentences = re.split(r'(?<=[.!?])\s+', context)
    answer_emb = get_embedding(answer)
    sims = [
        cosine_similarity(
            [answer_emb],
            [get_embedding(s)]
        )[0][0]
        for s in sentences if len(s) > 20
    ]
    return max(sims, default=0) >= threshold

# EVALUATION (MULTI SIGNAL)
def evaluate_sample(context, question, answer):
    emb_context = get_embedding(context)
    emb_answer = get_embedding(answer)
    emb_question = get_embedding(question)
    relevance = cosine_similarity(
        [emb_context], [emb_answer]
    )[0][0]
    alignment = cosine_similarity(
        [emb_question], [emb_answer]
    )[0][0]
    return float((relevance + alignment) / 2)

# MAIN PIPELINE
def generate_dataset(file, progress_callback=None):
    stats = {
        "chunks_total": 0,
        "questions_generated": 0,
        "not_found": 0,
        "verification_failed": 0,
        "overlap_failed": 0,
        "accepted": 0
    }
    logger.info("Starting pipeline")

    text = load_and_clean(file)
    chunks = list(chunk_text(text, tokenizer))
    logger.info(f"Initial chunks: {len(chunks)}")

    # Filter low-information chunks
    chunks = [c for c in chunks if not is_low_information(c)]

    # Deduplicate semantically
    chunks = deduplicate_chunks(chunks)
    logger.info(f"Clean chunks: {len(chunks)}")

    # ✅ update stats correctly
    stats["chunks_total"] = len(chunks)

    dataset = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        # Trim each chunk to token limit
        chunk = trim_to_token_limit(chunk, tokenizer)
        questions = generate_questions(chunk, n=5)

        # QUESTION LOOP (FIXED)
        for q in questions:
            stats["questions_generated"] += 1
            ans = generate_answer(q, chunk)
            if not ans or ans.strip() == "NOT_FOUND":
                stats["not_found"] += 1
                continue
             # Lexical grounding check
            if not lexical_overlap(ans, chunk):
                stats["overlap_failed"] += 1
                continue
            # Logical verification (NLI)
            if not verify_answer(chunk, ans):
                stats["verification_failed"] += 1
                continue

            score = evaluate_sample(chunk, q, ans)
            if score > 0.45:
                stats["accepted"] += 1
                dataset.append({
                    "context": chunk,
                    "question": q,
                    "answer": ans,
                    "score": score
                })
        # ✅ progress update per chunk (correct position)
        if progress_callback:
            progress_callback((i + 1) / total)
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info("===== PIPELINE REPORT =====")
    for k, v in stats.items():
        logger.info(f"{k}: {v}")
    return dataset

def dataset_report(dataset):
    scores = [d["score"] for d in dataset]
    print("Samples:", len(dataset))
    print("Avg score:", np.mean(scores))
    print("Min score:", np.min(scores))
    print("Max score:", np.max(scores))

# EXPORT
def save_jsonl(data, path="dataset.jsonl"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        return path
    except Exception as e:
        logger.error(e)
        raise RuntimeError("Dataset saving failed")
