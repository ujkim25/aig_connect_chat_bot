# chatbot_web_streamlit.py (문서 캐싱 구조 개선)

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

# 1. 문서 텍스트 추출
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

# 2. 문서 로딩 (언어별)
def load_text_files_with_metadata(lang_dir):
    chunks, metadatas = [], []
    for file_path in glob.glob(os.path.join(lang_dir, "*")):
        text = extract_text_from_file(file_path)
        for para in text.split("\n\n"):
            if para.strip():
                chunks.append(para.strip())
                metadatas.append({"filename": os.path.basename(file_path)})
    return chunks, metadatas

# 3. 임베딩
def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

# 4. 벡터 인덱스 구축
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 5. 질문 기반 검색
def search_similar_chunks(question, model, index, chunks, metadatas, top_k=3):
    q_emb = model.encode([question], convert_to_numpy=True)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]], [metadatas[i] for i in indices[0]]

# 6. 프롬프트 생성
def build_prompt(top_chunks, question, lang):
    system_prompt = {
        "en": "You are an internal chatbot for an insurance platform. Answer the question based on the context below.",
        "ko": "당신은 보험 플랫폼의 내부 챗봇입니다. 아래 문맥을 바탕으로 질문에 답변해주세요.",
        "ja": "あなたは保険プラットフォームの社内チャットボットです。以下の文脈に基づいて質問に答えてください。",
        "zh-cn": "你是一个保险平台的内部聊天机器人。请根据下面的上下文回答问题。"
    }
    context = "\n".join(top_chunks)
    prompt = f"""{system_prompt.get(lang, system_prompt['en'])}

Context:
{context}

Question: {question}
Answer:"""
    return prompt

# 7. LLM 서버 호출
def call_llm(prompt):
    url = "http://192.168.0.23:11434/api/generate"  # 수정 가능
    payload = {
        "model": "mistral",  # mistral, phi 등 가능
        "prompt": prompt,
        "stream": False
    }
    try:
        res = requests.post(url, json=payload)
        return res.json().get("response", "[No response received]")
    except Exception as e:
        return f"[ERROR] {e}"

# 8. Streamlit UI
st.set_page_config(page_title="Insurance Chatbot")
st.title("🤖 Internal Insurance Chatbot (Multilingual)")

if "history" not in st.session_state:
    st.session_state.history = []

# 9. 앱 실행 시점에 언어별 문서 로딩/임베딩/인덱싱을 미리 수행
@st.cache_resource(show_spinner=False)
def initialize_chatbots():
    chatbot_dict = {}
    for lang_code in ["en", "ko", "ja", "zh-cn"]:
        folder = f"./docs/{lang_code}"
        if os.path.exists(folder):
            chunks, metadatas = load_text_files_with_metadata(folder)
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embed_chunks(chunks, model)
            index = build_faiss_index(embeddings)
            chatbot_dict[lang_code] = (chunks, metadatas, model, index)
    return chatbot_dict

chatbots = initialize_chatbots()

# 10. 질문 입력 및 응답 처리
question = st.text_input("Enter your question below:")

if st.button("Send") and question.strip():
    lang = langdetect.detect(question)
    st.markdown(f"🌐 Detected language: `{lang}`")
    if lang in chatbots:
        chunks, metadatas, model, index = chatbots[lang]
        top_chunks, top_metas = search_similar_chunks(question, model, index, chunks, metadatas)
        prompt = build_prompt(top_chunks, question, lang)
        answer = call_llm(prompt)
        st.session_state.history.append((question, answer, top_metas))
    else:
        st.error(f"No documents found for language: {lang}")

# 11. 결과 출력
if st.session_state.history:
    st.markdown("---")
    for i, (q, a, metas) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        sources = sorted(set(m['filename'] for m in metas))
        st.markdown("📎 **Sources:** " + ", ".join(sources))
        st.download_button("📥 Download answer", a, file_name=f"response_{i}.txt", use_container_width=True)
        st.markdown("---")
