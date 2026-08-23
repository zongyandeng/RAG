import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 全域快取
_tokenizer = None
_model = None

def init_reranker():
    global _tokenizer, _model
    model_name = 'BAAI/bge-reranker-v2-m3'
    
    if _tokenizer is None:
        print(f"Loading tokenizer for {model_name} (Fast=True)...")
        # 強制使用 Fast Tokenizer 確保最佳效能與 Python 3.14 相容性
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
    if _model is None:
        print(f"Loading sequence classification model for {model_name}...")
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()

def rerank(query, chunks, top_k=5):
    """
    第二階段重排：使用原生 transformers 載入 BGE-Reranker-v2-m3 進行精準重排
    """
    if not chunks:
        return []
        
    init_reranker()
    
    # 1. 準備輸入對
    pairs = [[query, chunk["text"]] for chunk in chunks]
    
    # 2. 推論計算分數
    with torch.no_grad():
        inputs = _tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )
        
        # 取得模型 logits 分數
        logits = _model(**inputs).logits.view(-1)
        
        # BGE Reranker v2 M3 是單一輸出的 sequence classification
        scores = logits.float().tolist()
        
    # 如果只有一個元素，tolist() 可能是 float，要包裝成 list
    if not isinstance(scores, list):
        scores = [scores]
        
    # 3. 將分數與 chunks 結合並排序
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
        
    # 4. 排序並取 Top-K
    reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_chunks[:top_k]

if __name__ == "__main__":
    # 簡易測試
    init_reranker()
    test_chunks = [
        {"chunk_id": "test_1", "text": "這是一篇關於兩階段檢索的實作題目規格書。"},
        {"chunk_id": "test_2", "text": "本系統要求使用 BGE-M3 作為第一階段向量搜尋，再使用 Reranker 重新打分。"},
        {"chunk_id": "test_3", "text": "今天天氣很好，適合去郊遊。"}
    ]
    res = rerank("規格書的要求是什麼？", test_chunks, top_k=2)
    print("\nRerank Results:")
    for idx, item in enumerate(res):
        print(f"{idx+1}. [{item['chunk_id']}] (Rerank Score: {item['rerank_score']:.4f}): {item['text']}")
