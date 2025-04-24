# [1] 필요한 패키지 import
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob
import requests
import langdetect
import docx2txt
import PyPDF2

# [2] 문서에서 텍스트 추출하는 함수 (문서 ingestion 시 사용됨)
def extract_text_from_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    elif file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    return ""

# [3] 문서 텍스트를 청크 단위로 로딩하고, 파일명 메타데이터와 함께 반환
def load_text_files_with_metadata(lang_dir):
    chunks, metadatas = [], []
    for file_path in glob.glob(os.path.join(lang_dir, "*")):
        text = extract_text_from_file(file_path)
        for para in text.split("\n\n"):
            if para.strip():
                chunks.append(para.strip())
                metadatas.append({"filename": os.path.basename(file_path)})
    return chunks, metadatas

# [4] 텍스트 청크 → 벡터 임베딩으로 변환
def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

# [5] 벡터 DB 생성 및 검색 인덱스 구축 (FAISS 사용)
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# [6] 질문 벡터를 기준으로 유사한 청크 3개 검색
def search_similar_chunks(question, model, index, chunks, metadatas, top_k=3):
    q_emb = model.encode([question], convert_to_numpy=True)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]], [metadatas[i] for i in indices[0]]

# [7] LLM에 넣을 프롬프트 생성 (청크 + 질문 포함)
def build_prompt(top_chunks, question, lang):
    system_prompt = {
        "en": "You are an internal chatbot for AIG Connect. Answer the question based on the context below.",
        "ko": "당신은 AIG Connect의 내부 챗봇입니다. 아래 문맥을 바탕으로 질문에 답변해주세요.",
        "ja": "あなたはAIG Connectの社内チャットボットです。以下の文脈に基づいて質問に答えてください。"
    }
    context = "\n".join(top_chunks)
    prompt = f"""{system_prompt.get(lang, system_prompt['en'])}

Context:
{context}

Question: {question}
Answer:"""
    return prompt

# [8] LLM 서버에 프롬프트를 전송하여 응답 받기 (Ollama API 등)
def call_llm(prompt):
    url = "http://192.168.0.23:11434/api/generate"
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
    try:
        res = requests.post(url, json=payload)
        return res.json().get("response", "[No response received]")
    except Exception as e:
        return f"[ERROR] {e}"

# [9] Streamlit 페이지 기본 설정
st.set_page_config(page_title="AIG Connect Chatbot")
st.title("🤖 AIG Connect Internal Chatbot (Multilingual)")

# [10] 언어별로 문서 로딩 및 임베딩/인덱스 생성 → 캐시 처리
@st.cache_resource(show_spinner=False)
def setup_chatbot_by_lang(lang_code):
    folder = f"./docs/{lang_code}"
    if not os.path.exists(folder):
        return [], [], None, None
    chunks, metadatas = load_text_files_with_metadata(folder)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    return chunks, metadatas, model, index

# [11] 세션 상태 초기화 (이전 질문/답변 기록 저장용)
if "history" not in st.session_state:
    st.session_state.history = []

# [12] 사용자 질문 입력 UI → 여기서부터 사용자 액션으로 흐름 시작됨
question = st.text_input("Enter your question below:")

# [13] 질문이 제출되었을 때 실행되는 흐름
if st.button("Send") and question.strip():
    # [13-1] 언어 감지
    lang = langdetect.detect(question)
    st.markdown(f"🌐 Detected language: `{lang}`")
    
    # [13-2] 언어별 문서 임베딩/벡터 DB 구성 불러오기
    chunks, metadatas, model, index = setup_chatbot_by_lang(lang)
    
    if not model:
        st.error(f"No documents found for language: {lang}")
    else:
        # [13-3] 유사 청크 검색
        top_chunks, top_metas = search_similar_chunks(question, model, index, chunks, metadatas)
        # [13-4] 프롬프트 생성
        prompt = build_prompt(top_chunks, question, lang)
        # [13-5] LLM 호출 → 응답 생성
        answer = call_llm(prompt)
        # [13-6] 결과 세션에 저장
        st.session_state.history.append((question, answer, top_metas))

# [14] 이전 Q&A 이력 화면에 출력
if st.session_state.history:
    st.markdown("---")
    for i, (q, a, metas) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        sources = sorted(set(m['filename'] for m in metas))
        st.markdown("📎 **Sources:** " + ", ".join(sources))
        st.download_button("📥 Download answer", a, file_name=f"response_{i}.txt", use_container_width=True)
        st.markdown("---")
