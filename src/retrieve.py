import os
import pickle
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(WORKING_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# 全域變數，快取模型與索引避免重複載入
_model = None
_index = None
_metadata = None

def init_resources():
    global _model, _index, _metadata
    if _model is None:
        print("Loading BGE-M3 model for retrieval...")
        _model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        
    if _index is None:
        print(f"Loading FAISS Index from {INDEX_PATH}...")
        _index = faiss.read_index(INDEX_PATH)
        
    if _metadata is None:
        print(f"Loading metadata from {METADATA_PATH}...")
        with open(METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)

def retrieve(query, top_k=20, return_time=False):
    """
    第一階段檢檢索：BGE-M3 Dense Vector Search
    """
    init_resources()
    
    import time
    # 1. 產生 query embedding
    # query 向量化
    t_embed_start = time.time()
    query_emb_data = _model.encode([query], batch_size=1, max_length=8192)
    query_emb = np.array(query_emb_data['dense_vecs']).astype('float32')
    embed_time_ms = (time.time() - t_embed_start) * 1000
    
    # 2. L2 歸一化以配合 Inner Product 索引，達到 Cosine Similarity 效果
    faiss.normalize_L2(query_emb)
    
    # 3. 搜尋
    t_search_start = time.time()
    scores, indices = _index.search(query_emb, top_k)
    search_time_ms = (time.time() - t_search_start) * 1000
    
    # 4. 組裝結果
    results = []
    # indices[0] 是 top_k 的索引
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(_metadata):
            continue
        
        # 複製 metadata 以免被外部修改影響快取
        chunk = dict(_metadata[idx])
        chunk["score"] = float(score) # 保存第一階段分數
        results.append(chunk)
        
    if return_time:
        return results, {"embed_ms": embed_time_ms, "search_ms": search_time_ms}
    return results

if __name__ == "__main__":
    # 簡易測試
    import time
    if not os.path.exists(INDEX_PATH):
        print(f"Index file {INDEX_PATH} not found. Please run ingest.py first.")
    else:
        q = "最低驗收標準是什麼"
        start = time.time()
        res = retrieve(q, top_k=5)
        duration = time.time() - start
        print(f"\nQuery: {q} (Retrieved in {duration:.4f}s)")
        for idx, item in enumerate(res):
            print(f"{idx+1}. [{item['chunk_id']}] (Score: {item['score']:.4f}) page {item['page']}: {item['text'][:100]}...")
