import os
import gc
import time
import torch
import fitz  # PyMuPDF
import ollama  # We use the native library for Vision (it handles images better)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document  # FIXED: New LangChain location
from faster_whisper import WhisperModel
from transformers import pipeline
from moviepy import VideoFileClip  # FIXED: MoviePy v2.0 import
from PIL import Image

# --- CONFIGURATION ---
DATA_PATH = "./my_data"      
DB_PATH = "./my_offline_db"  
EMBED_MODEL = "nomic-embed-text"

# --- HELPER: MEMORY CLEANUP ---
def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# --- 1. PDF PROCESSOR ---
def process_pdfs(files):
    if not files: return []
    print(f"\n📄 Processing {len(files)} PDF(s)...")
    docs = []
    
    for file_path in files:
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text("text")
                    # Simple chunking by paragraphs
                    chunks = [p for p in text.split('\n\n') if len(p) > 50]
                    
                    for chunk in chunks:
                        docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(file_path),
                                "type": "pdf",
                                "page": page_num + 1
                            }
                        ))
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            
    return docs

# --- 2. AUDIO PROCESSOR ---
def process_audio(files):
    if not files: return []
    print(f"\n🎤 Processing {len(files)} Audio file(s)...")
    
    docs = []
    # Load Models (Heavy!)
    print("   - Loading Whisper (this takes a moment)...")
    whisper = WhisperModel("medium", device="cuda", compute_type="float16")
    
    print("   - Loading Emotion Classifier...")
    # Changed model to one that supports SafeTensors to fix the security error
    # Use the standard, high-quality model. 
    # We add 'trust_remote_code=False' to be safe and use default architecture.
    emotion_pipe = pipeline(
        "audio-classification", 
        model="superb/wav2vec2-base-superb-er", 
        device=0
    ) 

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"   - Listening to: {filename}")
        
        # A. Emotion
        try:
            emotions = emotion_pipe(file_path, top_k=1)
            mood = emotions[0]['label']
        except:
            mood = "neutral"

        # B. Transcribe
        segments, _ = whisper.transcribe(file_path)
        
        for segment in segments:
            start = f"{int(segment.start // 60):02}:{int(segment.start % 60):02}"
            end = f"{int(segment.end // 60):02}:{int(segment.end % 60):02}"
            
            docs.append(Document(
                page_content=segment.text,
                metadata={
                    "source": filename,
                    "type": "audio",
                    "timestamp": f"{start}-{end}",
                    "emotion": mood
                }
            ))

    # Cleanup VRAM
    del whisper
    del emotion_pipe
    cleanup_gpu()
    return docs

# --- 3. VISION PROCESSOR (CORRECTED) ---
def process_images_and_videos(image_files, video_files):
    if not image_files and not video_files: return []
    print(f"\n👁️ Processing Images/Videos... (Using LLaVA)")
    
    docs = []

    # A. Images
    for file_path in image_files:
        print(f"   - Watching Image: {os.path.basename(file_path)}")
        try:
            # We use the raw 'ollama' lib to send the image binary
            res = ollama.chat(
                model='llava',
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in detail.',
                    'images': [file_path]
                }]
            )
            desc = res['message']['content']
            
            docs.append(Document(
                page_content=desc,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "image",
                    "location": "full_image"
                }
            ))
        except Exception as e:
            print(f"❌ Error on image {file_path}: {e}")

    # B. Videos
    for file_path in video_files:
        print(f"   - Watching Video: {os.path.basename(file_path)}")
        try:
            clip = VideoFileClip(file_path)
            duration = int(clip.duration)
            
            # Extract 1 frame every 60 seconds
            for t in range(0, duration, 60):
                frame_path = f"temp_frame_{t}.jpg"
                clip.save_frame(frame_path, t)
                
                # Send frame to LLaVA
                res = ollama.chat(
                    model='llava',
                    messages=[{
                        'role': 'user',
                        'content': 'Describe this scene.',
                        'images': [frame_path]
                    }]
                )
                desc = res['message']['content']
                
                timestamp = f"{int(t // 60):02}:{int(t % 60):02}"
                docs.append(Document(
                    page_content=f"[Scene Description] {desc}",
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": "video_frame",
                        "timestamp": timestamp
                    }
                ))
                
                if os.path.exists(frame_path): os.remove(frame_path)
            
            clip.close()
        except Exception as e:
            print(f"❌ Error on video {file_path}: {e}")

    return docs

# --- MAIN ---
def main():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📂 Created '{DATA_PATH}'. Put files here and run again.")
        return

    all_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)]
    
    pdfs = [f for f in all_files if f.endswith(".pdf")]
    audios = [f for f in all_files if f.endswith((".mp3", ".wav", ".m4a"))]
    images = [f for f in all_files if f.endswith((".jpg", ".png", ".jpeg"))]
    videos = [f for f in all_files if f.endswith((".mp4", ".mov"))]
    
    # 1. Process
    final_docs = []
    final_docs.extend(process_pdfs(pdfs))
    final_docs.extend(process_audio(audios))
    final_docs.extend(process_images_and_videos(images, videos))
    
    # 2. Embed
    if final_docs:
        print(f"\n💾 Embedding {len(final_docs)} chunks into ChromaDB...")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        
        # Batch save to prevent timeouts
        batch_size = 50
        for i in range(0, len(final_docs), batch_size):
            batch = final_docs[i:i+batch_size]
            Chroma.from_documents(batch, embeddings, persist_directory=DB_PATH)
            print(f"   - Saved batch {i}-{i+len(batch)}")
            
        print("\n✅ DONE! You can now run chat.py")
    else:
        print("\n❌ No valid files found in ./my_data")

if __name__ == "__main__":
    main()