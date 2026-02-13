import streamlit as st
import os
import shutil
import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
DB_PATH = "./my_offline_db"
DATA_PATH = "./my_data"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"
VISION_MODEL = "llava"

# --- PAGE CONFIG ---
st.set_page_config(page_title="Offline RAG", page_icon="⚡", layout="wide")
st.title("⚡ Multimodal RAG")

# --- OPTIMIZATION: LOAD RESOURCES ONCE ---
@st.cache_resource
def get_vector_db():
    """
    This function runs ONCE. The result is kept in memory.
    It stops the app from reconnecting to the DB on every message.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

@st.cache_resource
def get_llm():
    """
    Keeps the connection to Llama 3 open.
    """
    return OllamaLLM(model=CHAT_MODEL, temperature=0)

# --- HELPER: VISION ---
def look_at_image(filename, user_question):
    path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(path): return "[Error: Image missing]"
    try:
        res = ollama.chat(
            model=VISION_MODEL,
            messages=[{'role': 'user', 'content': user_question, 'images': [path]}]
        )
        return res['message']['content']
    except: return "[Vision Failed]"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # TURBO MODE TOGGLE
    turbo_mode = st.toggle("🚀 Turbo Mode (Text Only)", value=True)
    if turbo_mode:
        st.caption("⚡ Fast. Ignores active vision analysis.")
    else:
        st.caption("👁️ Slow. Re-examines images for details.")

    st.divider()
    
    # FILE UPLOAD FORM
    with st.form("ingest_form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload Data", accept_multiple_files=True)
        submitted = st.form_submit_button("Process Uploads")

    if submitted and uploaded_files:
        with st.spinner("Ingesting..."):
            # (Imports are inside to keep startup fast)
            try:
                from ingest import process_pdfs, process_audio, process_images_and_videos
                
                if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
                saved = []
                for f in uploaded_files:
                    path = os.path.join(DATA_PATH, f.name)
                    with open(path, "wb") as w: w.write(f.getbuffer())
                    saved.append(path)

                docs = []
                docs.extend(process_pdfs([p for p in saved if p.endswith('.pdf')]))
                docs.extend(process_audio([p for p in saved if p.endswith(('.mp3','.wav'))]))
                docs.extend(process_images_and_videos(
                    [p for p in saved if p.endswith(('.jpg','.png'))],
                    [p for p in saved if p.endswith('.mp4')]
                ))
                
                if docs:
                    # Clear cache so the new DB data is loaded
                    st.cache_resource.clear()
                    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                    Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)
                    st.success(f"Indexed {len(docs)} chunks!")
                    st.rerun() # Refresh to update the DB connection
            except ImportError:
                st.error("Missing ingest.py!")

    if st.button("🗑️ Reset All"):
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            st.cache_resource.clear()
            st.session_state.messages = []
            st.rerun()

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Use Cached Resources (Instant Access)
            vector_db = get_vector_db()
            llm = get_llm()
            
            # 2. Retrieve
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            
            if not docs:
                response = "No info found in files."
            else:
                # 3. Build Context
                context_str = ""
                checked_imgs = set()
                
                for doc in docs:
                    meta = doc.metadata
                    src = meta.get('source', 'Unknown')
                    dtype = meta.get('type', 'text')
                    
                    # LOGIC: If Turbo Mode is OFF, and it's an image, look at it.
                    if not turbo_mode and dtype in ['image', 'video_frame'] and src not in checked_imgs:
                        st.caption(f"👀 Analyzing {src}...")
                        visual_ans = look_at_image(src, prompt)
                        content = f"[Visual Analysis]: {visual_ans}"
                        checked_imgs.add(src)
                    else:
                        content = doc.page_content

                    context_str += f"[Source: {src}]\n{content}\n\n"

                # 4. Generate
                chain = (
                    {"context": lambda x: context_str, "question": lambda x: prompt}
                    | ChatPromptTemplate.from_template("Answer strictly from Context:\n{context}\n\nQuestion:\n{question}")
                    | llm
                )
                response = chain.invoke(prompt)

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})