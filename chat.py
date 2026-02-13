import sys
import os
import time
import textwrap
import ollama  # We use the native client for the 'eyes'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
DB_PATH = "./my_offline_db"
DATA_PATH = "./my_data" 
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"
VISION_MODEL = "llava"  # The model that looks at images

# --- 1. THE VISION FUNCTION ---
def look_at_image(filename, user_question):
    """
    Reloads the image file and asks LLaVA the SPECIFIC question.
    """
    path = os.path.join(DATA_PATH, filename)
    
    # Safety check: does the file still exist?
    if not os.path.exists(path):
        return "[Error: Image file missing from folder]"

    print(f"\n   👀 Re-examining {filename}...", end="", flush=True)
    
    try:
        # Send the image + question to LLaVA
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': user_question,
                'images': [path]
            }]
        )
        print(" Done.")
        return response['message']['content']
    except Exception as e:
        return f"[Vision Error: {e}]"

# --- 2. CONTEXT BUILDER (With Eyes) ---
def build_context_with_vision(docs, user_question):
    context_str = ""
    
    # We track if we've already looked at an image to avoid double-work
    checked_images = set()

    for doc in docs:
        meta = doc.metadata
        source = meta.get('source', 'Unknown')
        dtype = meta.get('type', 'text')

        # DEFAULT: Use the text from the database
        content = doc.page_content
        citation = f"[Source: {source}]"

        # VISION TRIGGER: If it's an image, let's look at it fresh!
        if dtype in ['image', 'video_frame'] and source not in checked_images:
            # Re-examine the image with the specific question
            visual_answer = look_at_image(source, user_question)
            
            # Replace the old static description with the new smart answer
            content = f"[Visual Analysis]: {visual_answer}"
            citation = f"[👁️ Vision: {source}]"
            
            checked_images.add(source)

        elif dtype == 'pdf':
            citation = f"[📄 PDF: {source} | Page: {meta.get('page')}]"
        
        elif dtype == 'audio':
            citation = f"[🎤 Audio Transcript: {source} | Time: {meta.get('timestamp')}]"

        context_str += f"{citation}\n{content}\n\n"
    
    return context_str

# --- MAIN APP ---
def main():
    print(f"🧠 Loading {CHAT_MODEL} (Brain) & {VISION_MODEL} (Eyes)...")
    
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    llm = OllamaLLM(model=CHAT_MODEL, temperature=0)

    # Prompt: Trust the visual analysis
    template = """
    You are an intelligent assistant.
    If the context contains a [Visual Analysis], trust that description above all else.
    
    CONTEXT:
    {context}

    USER QUESTION: 
    {question}

    ANSWER (Cite your sources):
    """
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    print("\n✅ Active Vision RAG Online. Type 'exit' to quit.\n")

    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]: break
        
        print("Thinking...", end="", flush=True)
        
        # 1. Retrieve relevant docs
        docs = retriever.invoke(query)
        
        # 2. Build Context (Triggering Vision if needed)
        # This is where the magic happens
        context = build_context_with_vision(docs, query)
        
        # 3. Generate Answer
        chain = (
            {"context": lambda x: context, "question": lambda x: query}
            | prompt
            | llm
        )
        response = chain.invoke(query)
        
        print("\r" + " " * 20 + "\r")
        print(f"AI:\n{textwrap.fill(response, width=80)}")
        print("-" * 50)

if __name__ == "__main__":
    main()