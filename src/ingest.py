import os
import sys
import glob
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
import pdfplumber
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import tiktoken
import requests
from dotenv import load_dotenv

# 設定路徑
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(WORKING_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "documents")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
CHUNKS_LOG_PATH = os.path.join(DATA_DIR, "chunks_list.txt") # 用於輸出給我們編寫 question 的清單

# 載入環境變數以讀取 Gemini 設定
load_dotenv(os.path.join(WORKING_DIR, ".env"))
HERMES_API_KEY = os.getenv("HERMES_API_KEY")
HERMES_API_BASE = os.getenv("HERMES_API_BASE", "https://api.hermes-gateway.com/v1")
HERMES_MODEL_NAME = os.getenv("HERMES_MODEL_NAME", "hermes-llama-3-8b")

def load_tokenizer():
    # 使用 tiktoken 計算 token 數，這通常與 LLM (Hermes/OpenAI) 的計算方式一致
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Warning: Failed to load tiktoken, falling back to simple character counting. Error: {e}")
        return None

def count_tokens(text, tokenizer):
    # 優先嘗試呼叫 Gemini API 取得精確數字
    if HERMES_API_KEY and HERMES_MODEL_NAME and "gemini" in HERMES_MODEL_NAME.lower():
        try:
            # 建立 REST API URL (移除 /openai 結尾以呼叫原生 API)
            base_url = HERMES_API_BASE.replace("/openai", "")
            url = f"{base_url}/models/{HERMES_MODEL_NAME}:countTokens?key={HERMES_API_KEY}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": text}
                        ]
                    }
                ]
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                res_data = response.json()
                if "totalTokens" in res_data:
                    return res_data["totalTokens"]
            else:
                print(f"Warning: Gemini API countTokens returned status code {response.status_code}: {response.text}")
        except Exception as e:
            # 發生錯誤時默默地 fallback 到本地 tokenizer
            pass

    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text) # fallback

def split_text(text, doc_name, page_num, doc_id, tokenizer, chunk_size=500, chunk_overlap=100):
    """
    將文字切分成符合 token budget 的 chunks
    """
    chunks = []
    # 簡單的基於字元的切分，並以 tokenizer 驗證長度
    # 由於中文字符與 token 接近 1:1，先以字元進行切分，再用 tokenizer 做精確微調
    start = 0
    chunk_idx = 1
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # 調整邊界，盡量不要切斷句子
        if end < len(text):
            # 尋找最近的句號、問號、驚嘆號或換行
            for separator in ['\n', '。', '！', '？', '.', '!', '?']:
                last_sep = chunk_text.rfind(separator)
                if last_sep > chunk_size * 0.6: # 確保不會縮得太短
                    end = start + last_sep + 1
                    chunk_text = text[start:end]
                    break
        
        token_count = count_tokens(chunk_text, tokenizer)
        
        # 建立 chunk 物件
        chunk_id = f"{doc_id}_p{page_num:02d}_c{chunk_idx:02d}"
        chunks.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "document_name": doc_name,
            "page": page_num,
            "text": chunk_text,
            "token_count": token_count
        })
        
        chunk_idx += 1
        start += (end - start) - chunk_overlap
        if (end - start) <= chunk_overlap: # 避免死循環
            start = end
            
    return chunks

def extract_text_from_pdf(pdf_path):
    pages = []
    # 優先使用 pdfplumber 讀取表格與格式
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append((idx + 1, text))
    except Exception as e:
        print(f"pdfplumber failed for {pdf_path}, falling back to pypdf. Error: {e}")
        # fallback to pypdf
        reader = PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((idx + 1, text))
    return pages

def extract_text_from_txt_or_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 由於 txt/md 沒有天然頁碼，我們模擬每 1500 字元為一頁
    pages = []
    chars_per_page = 1500
    start = 0
    page_num = 1
    while start < len(text):
        end = start + chars_per_page
        pages.append((page_num, text[start:end]))
        start = end
        page_num += 1
    return pages

def main():
    print("Initializing RAG Ingestion Pipeline...")
    tokenizer = load_tokenizer()
    
    # 搜尋 documents 目目錄下的檔案
    supported_extensions = ["*.pdf", "*.txt", "*.md"]
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(DOCS_DIR, ext)))
        
    if not files:
        print(f"Error: No supported files found in {DOCS_DIR}")
        sys.exit(1)
        
    all_chunks = []
    doc_id_counter = 1
    
    for file_path in files:
        doc_name = os.path.basename(file_path)
        doc_id = f"doc{doc_id_counter:02d}"
        doc_id_counter += 1
        
        print(f"Processing file: {doc_name} (ID: {doc_id})...")
        
        if file_path.lower().endswith(".pdf"):
            pages = extract_text_from_pdf(file_path)
        else:
            pages = extract_text_from_txt_or_md(file_path)
            
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            chunks = split_text(
                text=page_text,
                doc_name=doc_name,
                page_num=page_num,
                doc_id=doc_id,
                tokenizer=tokenizer,
                chunk_size=500, # 預設字元數 400~600，對應 token 預算
                chunk_overlap=100
            )
            all_chunks.extend(chunks)
            
    print(f"Total chunks extracted: {len(all_chunks)}")
    
    # 輸出 chunks 清單供後續編寫測試題庫對照
    with open(CHUNKS_LOG_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(f"Chunk ID: {chunk['chunk_id']} | Page: {chunk['page']} | Text (first 80 chars): {chunk['text'][:80].replace(chr(10), ' ')}...\n")
    print(f"Chunk list written to {CHUNKS_LOG_PATH} for question tagging reference.")
    
    # 初始化 BGE-M3 模型
    print("Loading BAAI/bge-m3 model...")
    # use_fp16=False 確保在 CPU 環境下執行
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    
    # 生成 embeddings
    print("Generating dense embeddings for chunks...")
    texts = [chunk["text"] for chunk in all_chunks]
    
    # model.encode() 會返回包含 dense 的 dict，我們提取 dense
    embeddings_data = model.encode(texts, batch_size=4, max_length=8192)
    dense_embeddings = np.array(embeddings_data['dense_vecs']).astype('float32')
    
    # 檢查維度是否為 1024
    dimension = dense_embeddings.shape[1]
    print(f"Embedding generation completed. Shape: {dense_embeddings.shape}")
    
    # 建立 FAISS 索引
    print("Building FAISS Index...")
    # 使用 Inner Product 索引 (等同於 Cosine Similarity，因為 BGE-M3 向量是 normalized 的)
    # 我們也手動將向量 normalized 確保 Inner Product = Cosine Similarity
    faiss.normalize_L2(dense_embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(dense_embeddings)
    
    # 儲存索引與 metadata
    print(f"Saving FAISS Index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
        
    print("Ingestion Pipeline successfully completed!")

if __name__ == "__main__":
    main()
