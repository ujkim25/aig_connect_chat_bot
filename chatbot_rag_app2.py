# chatbot_rag_app.py

import os
import glob
from typing import List

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Step 1. 문서 로드 & 청크 분할

def load_text_files(folder_path: str) -> List[str]:
    chunks = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            paragraphs = text.split("\n\n")  # 문단 기준으로 나누기
            chunks.extend(paragraphs)
    return chunks

# Step 2. 청크 임베딩

def embed_chunks(chunks: List[str], model) -> np.ndarray:
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

# Step 3. FAISS Vector DB 구축

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Step 4. 질문 처리 및 검색

def search_similar_chunks(question: str, model, index, chunks, top_k=3) -> List[str]:
    q_emb = model.encode([question], convert_to_numpy=True)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]

# Step 5. 프롬프트 생성

def build_prompt(top_chunks: List[str], question: str) -> str:
    context = "\n".join(top_chunks)
    prompt = f"""
You are an internal chatbot for AIG Connect.
Answer the question based on the context below.

Context:
{context}

Question: {question}
Answer:
"""
    return prompt

# Step 6. LLM 서버 호출 (HTTP - 상사 Ollama 서버)

def call_llm(prompt: str) -> str:
    url = "http://192.168.0.23:11434/api/generate"
    payload = {
        "model": "mistral",   # 상사가 ollama run으로 로딩한 모델 이름
        "prompt": prompt,
        "stream": False
    }
    try:
        res = requests.post(url, json=payload)
        return res.json().get("response", res.text)
    except Exception as e:
        return f"[ERROR] {e}"

# 실행 예시
if __name__ == "__main__":
    folder = "./docs"  # 문서를 저장한 로컬 폴더
    question = input("Enter your question: ")  # 사용자 입력 받기

    print("[1] Loading documents...")
    chunks = load_text_files(folder)

    print("[2] Embedding chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 로컬용 임베딩 모델
    embeddings = embed_chunks(chunks, model)

    print("[3] Building vector index...")
    index = build_faiss_index(embeddings)

    print("[4] Searching relevant chunks...")
    top_chunks = search_similar_chunks(question, model, index, chunks)

    print("[5] Building prompt...")
    prompt = build_prompt(top_chunks, question)

    print("[6] Sending to LLM server...")
    answer = call_llm(prompt)

    print("\n[Answer from LLM]")
    print(answer)
