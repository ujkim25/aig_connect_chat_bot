# chatbot_web_streamlit.py (피드백 기능 제거)

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob
import requests
import tempfile
import uuid
import langdetect
import docx2txt
import PyPDF2

# 1. 문서 로딩 & 청크 분할 (파일명 포함)
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

def load_text_files_with_metadata(folder_path):
    chunks = []
    metadatas = []
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        text = extract_text_from_file(file_path)
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if para.strip():
                chunks.append(para.strip())
                metadatas.append({"filename": os.path.basename(file_path)})
    return chunks, metadatas

# 2. 임베딩 함수
def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

# 3. 벡터 인덱스 구축 (메타데이터 포함)
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 4. 질문 기반 유사 문서 검색
def search_similar_chunks(question, model, index, chunks, metadatas, top_k=3):
    q_emb = model.encode([question], convert_to_numpy=True)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]], [metadatas[i] for i in indices[0]]

# 5. 프롬프트 생성
def build_prompt(top_chunks, question, lang="en"):
    system_prompt = {
        "en": "You are an internal chatbot for AIG Connect. Answer the question based on the context below.",
        "ko": "당신은 AIG Connect의 내부 챗봇입니다. 아래 문맥을 바탕으로 질문에 답변해주세요.",
        "ja": "あなたはAIG Connectの社内チャットボットです。以下の文脈に基づいて質問に答えてください。"
    }
    context = "\n".join(top_chunks)
    prompt = f"""
{system_prompt.get(lang, system_prompt['en'])}

Context:
{context}

Question: {question}
Answer:
"""
    return prompt

# 6. LLM 서버 호출
def call_llm(prompt):
    url = "http://192.168.0.23:11434/api/generate"
    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False
    }
    try:
        res = requests.post(url, json=payload)
        return res.json().get("response", "[No response received]")
    except Exception as e:
        return f"[ERROR] {e}"

# Streamlit UI 설정
st.set_page_config(page_title="AIG Connect Chatbot")
st.title("🤖 AIG Connect Internal Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

upload_folder = "./docs/uploaded"
os.makedirs(upload_folder, exist_ok=True)

# 📂 업로드 UI
uploaded_files = st.file_uploader("📂 Upload text, PDF, or Word files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    save_path = os.path.join(upload_folder, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

# 📄 업로드된 파일 목록 & 삭제 버튼
uploaded_list = os.listdir(upload_folder)
if uploaded_list:
    st.markdown("### 📑 Uploaded Files")
    for filename in uploaded_list:
        col1, col2 = st.columns([5, 1])
        col1.markdown(f"🔹 {filename}")
        if col2.button("🗑️ Delete", key=filename):
            if st.confirm(f"Delete {filename} from disk?"):
                os.remove(os.path.join(upload_folder, filename))
                st.success(f"✅ {filename} deleted.")
                st.experimental_rerun()
    if st.button("🧹 Delete All Files"):
        if st.confirm("Are you sure you want to delete all uploaded files?"):
            for filename in uploaded_list:
                os.remove(os.path.join(upload_folder, filename))
            st.success("✅ All files deleted.")
            st.experimental_rerun()

temp_dir = upload_folder

@st.cache_resource(show_spinner=False)
def setup_chatbot(folder, temp_dir):
    base_chunks, base_meta = load_text_files_with_metadata(folder)
    temp_chunks, temp_meta = load_text_files_with_metadata(temp_dir)
    chunks = base_chunks + temp_chunks
    metadatas = base_meta + temp_meta
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    return chunks, metadatas, model, index

chunks, metadatas, model, index = setup_chatbot("./docs", temp_dir)

question = st.text_input("Enter your question below:")
if st.button("Send") and question.strip():
    detected_lang = langdetect.detect(question)
    with st.spinner(f"Detected language: {detected_lang}. Generating answer..."):
        top_chunks, top_metas = search_similar_chunks(question, model, index, chunks, metadatas)
        prompt = build_prompt(top_chunks, question, lang=detected_lang)
        answer = call_llm(prompt)
        st.session_state.history.append((question, answer, top_metas))

if st.session_state.history:
    st.markdown("---")
    total = len(st.session_state.history)
    for i, (q, a, metas) in enumerate(reversed(st.session_state.history)):
        q_number = total - i
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        unique_sources = sorted(set(m['filename'] for m in metas if 'filename' in m))
        st.markdown("📎 **Sources:** " + ", ".join(unique_sources))
        st.download_button("📥 Download answer", a, file_name=f"response_{q_number}.txt", use_container_width=True)
        st.markdown("---")
