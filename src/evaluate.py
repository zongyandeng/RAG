import os
import sys
import json
import time
import pickle
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# 載入模組
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.dirname(SRC_DIR)
sys.path.append(os.path.join(WORKING_DIR, "src"))

from retrieve import retrieve
from rerank import rerank
from generate import generate, count_tokens

load_dotenv(os.path.join(WORKING_DIR, ".env"))

DATA_DIR = os.path.join(WORKING_DIR, "data")
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.jsonl")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
CSV_OUTPUT_PATH = os.path.join(RESULTS_DIR, "per_question.csv")
JSON_OUTPUT_PATH = os.path.join(RESULTS_DIR, "summary.json")

def load_questions():
    questions = []
    if not os.path.exists(QUESTIONS_PATH):
        print(f"Error: {QUESTIONS_PATH} not found. Please create the questions file first.")
        return []
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def load_all_documents_context():
    """
    載入所有文件的完整內容，用於 Long-context Baseline
    """
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    # 按 chunk_id 順序拼接所有文字
    sorted_chunks = sorted(metadata, key=lambda x: x["chunk_id"])
    context_parts = []
    for chunk in sorted_chunks:
        context_parts.append(
            f"Source: {chunk['document_name']}, Page: {chunk['page']}, Chunk ID: {chunk['chunk_id']}\n"
            f"Content: {chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)

def run_evaluation():
    print("Starting Batch Evaluation...")
    questions = load_questions()
    if not questions:
        print("No questions found. Aborting evaluation.")
        return
        
    if not os.path.exists(METADATA_PATH):
        print("Metadata not found. Please run ingest.py first.")
        return
        
    # 載入所有 document 的 text 作為 Long-context Baseline
    full_context_text = load_all_documents_context()
    
    # 紀錄逐題結果
    rows = []
    
    # 用於統計的計數器
    stats = {
        "long_context": {"input_tokens": [], "total_ms": [], "answers": []},
        "basic_rag": {"input_tokens": [], "total_ms": [], "answers": [], "hit_at_5": []},
        "advanced_rag": {"input_tokens": [], "total_ms": [], "answers": [], "recall_at_20": [], "hit_at_5": [], "evidence_recall_at_5": []}
    }
    
    # ----------------------------------------------------
    # 暖機階段 (Warm-up Phase)：連續執行 3 次 dummy 查詢以加載與暖機 local 模型
    # ----------------------------------------------------
    print("\n[Warm-up] Starting model warm-up (3 times) for fair comparison...")
    warmup_query = "環境與作業系統要求是什麼"
    for i in range(3):
        print(f"[Warm-up] Run {i+1}/3...")
        # 暖機檢索與重排
        warmup_retrieved = retrieve(warmup_query, top_k=20)
        _ = rerank(warmup_query, warmup_retrieved, top_k=5)
    print("[Warm-up] Warm-up completed! Starting official timing.\n")
    
    # 開始跑 30 道題目
    for q_idx, q in enumerate(tqdm(questions, desc="Evaluating Questions")):
        q_id = q["question_id"]
        query = q["question"]
        gold_chunks = q.get("gold_chunk_ids", [])
        is_answerable = q.get("answerable", True)
        
        # ----------------------------------------------------
        # Variant A: Long-context Baseline
        # ----------------------------------------------------
        print(f"\n[{q_id}] Running Long-context Baseline...")
        start_time_a = time.time()
        
        fake_chunks = [{"chunk_id": "full_doc", "document_name": "Full Document", "page": 1, "text": full_context_text}]
        res_a_raw = generate(query, fake_chunks, max_context_tokens=1000000, stream=True)
        
        if isinstance(res_a_raw, dict):
            # Mock 模式
            res_a = res_a_raw
            ttft_a = 5.0
            total_time_a = (time.time() - start_time_a) * 1000
        else:
            # 串流模式
            response_a, selected_chunks_a, token_meta_a = res_a_raw
            ttft_a = 0.0
            full_ans_a = []
            for chunk in response_a:
                if ttft_a == 0.0:
                    ttft_a = (time.time() - start_time_a) * 1000
                if chunk.choices and chunk.choices[0].delta.content:
                    full_ans_a.append(chunk.choices[0].delta.content)
            total_time_a = (time.time() - start_time_a) * 1000
            ans_a = "".join(full_ans_a)
            res_a = {
                "answer": ans_a,
                "system_prompt_tokens": token_meta_a["system_prompt_tokens"],
                "query_tokens": token_meta_a["query_tokens"],
                "context_tokens": token_meta_a["context_tokens"],
                "history_tokens": token_meta_a["history_tokens"],
                "input_tokens": token_meta_a["input_tokens"],
                "output_tokens": count_tokens(ans_a)
            }
            
        rows.append({
            "question_id": q_id,
            "system_variant": "long_context",
            "retrieved_chunk_ids": [],
            "retrieval_scores": [],
            "reranked_chunk_ids": [],
            "rerank_scores": [],
            "selected_chunk_ids": ["full_doc"],
            "answer": res_a["answer"],
            "citations": "",
            "system_prompt_tokens": res_a["system_prompt_tokens"],
            "query_tokens": res_a["query_tokens"],
            "context_tokens": res_a["context_tokens"],
            "history_tokens": res_a["history_tokens"],
            "input_tokens": res_a["input_tokens"],
            "output_tokens": res_a["output_tokens"],
            "embedding_ms": 0.0,
            "search_ms": 0.0,
            "rerank_ms": 0.0,
            "ttft_ms": ttft_a,
            "total_ms": total_time_a,
            "human_score": "",
            "error_type": ""
        })
        
        stats["long_context"]["input_tokens"].append(res_a["input_tokens"])
        stats["long_context"]["total_ms"].append(total_time_a)
        
        time.sleep(2.0)
        
        # ----------------------------------------------------
        # Variant B: Basic RAG (BGE-M3 Top-5)
        # ----------------------------------------------------
        print(f"[{q_id}] Running Basic RAG...")
        start_time_b = time.time()
        
        # Stage 1: Vector search
        retrieved_b, timing_b = retrieve(query, top_k=5, return_time=True)
        t_embed_ms_b = timing_b["embed_ms"]
        t_search_ms_b = timing_b["search_ms"]
        
        # Generation
        res_b_raw = generate(query, retrieved_b, max_context_tokens=1000000, stream=True)
        
        if isinstance(res_b_raw, dict):
            # Mock 模式
            res_b = res_b_raw
            ttft_b = t_embed_ms_b + t_search_ms_b + 10.0
            total_time_b = (time.time() - start_time_b) * 1000
        else:
            # 串流模式
            response_b, selected_chunks_b, token_meta_b = res_b_raw
            ttft_b = 0.0
            full_ans_b = []
            for chunk in response_b:
                if ttft_b == 0.0:
                    ttft_b = (time.time() - start_time_b) * 1000
                if chunk.choices and chunk.choices[0].delta.content:
                    full_ans_b.append(chunk.choices[0].delta.content)
            total_time_b = (time.time() - start_time_b) * 1000
            ans_b = "".join(full_ans_b)
            res_b = {
                "answer": ans_b,
                "selected_chunks": selected_chunks_b,
                "system_prompt_tokens": token_meta_b["system_prompt_tokens"],
                "query_tokens": token_meta_b["query_tokens"],
                "context_tokens": token_meta_b["context_tokens"],
                "history_tokens": token_meta_b["history_tokens"],
                "input_tokens": token_meta_b["input_tokens"],
                "output_tokens": count_tokens(ans_b)
            }
            
        # 計算 Hit@5
        hit_b = 0
        if is_answerable and gold_chunks:
            retrieved_ids = [c["chunk_id"] for c in retrieved_b]
            hit_b = 1 if any(gc in retrieved_ids for gc in gold_chunks) else 0
            stats["basic_rag"]["hit_at_5"].append(hit_b)
            
        rows.append({
            "question_id": q_id,
            "system_variant": "basic_rag",
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_b],
            "retrieval_scores": [c["score"] for c in retrieved_b],
            "reranked_chunk_ids": [],
            "rerank_scores": [],
            "selected_chunk_ids": [c["chunk_id"] for c in res_b["selected_chunks"]],
            "answer": res_b["answer"],
            "citations": ", ".join([f"[{c['document_name']}, Page {c['page']}, Chunk ID: {c['chunk_id']}]" for c in res_b["selected_chunks"]]),
            "system_prompt_tokens": res_b["system_prompt_tokens"],
            "query_tokens": res_b["query_tokens"],
            "context_tokens": res_b["context_tokens"],
            "history_tokens": res_b["history_tokens"],
            "input_tokens": res_b["input_tokens"],
            "output_tokens": res_b["output_tokens"],
            "embedding_ms": t_embed_ms_b,
            "search_ms": t_search_ms_b,
            "rerank_ms": 0.0,
            "ttft_ms": ttft_b,
            "total_ms": total_time_b,
            "human_score": "",
            "error_type": ""
        })
        
        stats["basic_rag"]["input_tokens"].append(res_b["input_tokens"])
        stats["basic_rag"]["total_ms"].append(total_time_b)
        
        time.sleep(2.0)
        
        # ----------------------------------------------------
        # Variant C: Advanced RAG (Top-20 -> Rerank -> Top-5 -> Token Budget)
        # ----------------------------------------------------
        print(f"[{q_id}] Running Advanced RAG...")
        start_time_c = time.time()
        
        # Stage 1: Vector Search Top-20
        retrieved_c, timing_c = retrieve(query, top_k=20, return_time=True)
        t_embed_ms_c = timing_c["embed_ms"]
        t_search_ms_c = timing_c["search_ms"]
        
        # Stage 2: Rerank
        t_stage2_start = time.time()
        reranked_c = rerank(query, retrieved_c, top_k=5)
        t_stage2_ms = (time.time() - t_stage2_start) * 1000
        
        # Generation with Token Budget (2000 tokens)
        res_c_raw = generate(query, reranked_c, max_context_tokens=2000, stream=True)
        
        if isinstance(res_c_raw, dict):
            # Mock 模式
            res_c = res_c_raw
            ttft_c = t_embed_ms_c + t_search_ms_c + t_stage2_ms + 10.0
            total_time_c = (time.time() - start_time_c) * 1000
        else:
            # 串流模式
            response_c, selected_chunks_c, token_meta_c = res_c_raw
            ttft_c = 0.0
            full_ans_c = []
            for chunk in response_c:
                if ttft_c == 0.0:
                    ttft_c = (time.time() - start_time_c) * 1000
                if chunk.choices and chunk.choices[0].delta.content:
                    full_ans_c.append(chunk.choices[0].delta.content)
            total_time_c = (time.time() - start_time_c) * 1000
            ans_c = "".join(full_ans_c)
            res_c = {
                "answer": ans_c,
                "selected_chunks": selected_chunks_c,
                "system_prompt_tokens": token_meta_c["system_prompt_tokens"],
                "query_tokens": token_meta_c["query_tokens"],
                "context_tokens": token_meta_c["context_tokens"],
                "history_tokens": token_meta_c["history_tokens"],
                "input_tokens": token_meta_c["input_tokens"],
                "output_tokens": count_tokens(ans_c)
            }
            
        # 計算 Recall@20 (Stage 1) 與 Hit@5 (Stage 2)
        recall_c = 0
        hit_c = 0
        if is_answerable and gold_chunks:
            retrieved_ids = [c["chunk_id"] for c in retrieved_c]
            recall_c = 1 if any(gc in retrieved_ids for gc in gold_chunks) else 0
            stats["advanced_rag"]["recall_at_20"].append(recall_c)
            
            reranked_ids = [c["chunk_id"] for c in reranked_c]
            hit_c = 1 if any(gc in reranked_ids for gc in gold_chunks) else 0
            stats["advanced_rag"]["hit_at_5"].append(hit_c)
            
            # 跨段落題另計 Evidence Recall@5，以實際找回的 Gold Chunks 數量除以全部 Gold Chunks
            if q.get("question_type") == "multi_paragraph":
                found_count = sum(1 for gc in gold_chunks if gc in reranked_ids)
                evidence_recall = found_count / len(gold_chunks) if gold_chunks else 0.0
                stats["advanced_rag"]["evidence_recall_at_5"].append(evidence_recall)
            
        rows.append({
            "question_id": q_id,
            "system_variant": "advanced_rag",
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_c],
            "retrieval_scores": [c["score"] for c in retrieved_c],
            "reranked_chunk_ids": [c["chunk_id"] for c in reranked_c],
            "rerank_scores": [c["rerank_score"] for c in reranked_c],
            "selected_chunk_ids": [c["chunk_id"] for c in res_c["selected_chunks"]],
            "answer": res_c["answer"],
            "citations": ", ".join([f"[{c['document_name']}, Page {c['page']}, Chunk ID: {c['chunk_id']}]" for c in res_c["selected_chunks"]]),
            "system_prompt_tokens": res_c["system_prompt_tokens"],
            "query_tokens": res_c["query_tokens"],
            "context_tokens": res_c["context_tokens"],
            "history_tokens": res_c["history_tokens"],
            "input_tokens": res_c["input_tokens"],
            "output_tokens": res_c["output_tokens"],
            "embedding_ms": t_embed_ms_c,
            "search_ms": t_search_ms_c,
            "rerank_ms": t_stage2_ms,
            "ttft_ms": ttft_c,
            "total_ms": total_time_c,
            "human_score": "",
            "error_type": ""
        })
        
        stats["advanced_rag"]["input_tokens"].append(res_c["input_tokens"])
        stats["advanced_rag"]["total_ms"].append(total_time_c)
        
        time.sleep(2.0)
        
    # ----------------------------------------------------
    # 輸出逐題 CSV 報表
    # ----------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Detailed per-question results written to {CSV_OUTPUT_PATH}")
    
    # ----------------------------------------------------
    # 計算並輸出彙總 summary.json
    # ----------------------------------------------------
    avg_input_tokens_a = sum(stats["long_context"]["input_tokens"]) / len(stats["long_context"]["input_tokens"])
    avg_input_tokens_b = sum(stats["basic_rag"]["input_tokens"]) / len(stats["basic_rag"]["input_tokens"])
    avg_input_tokens_c = sum(stats["advanced_rag"]["input_tokens"]) / len(stats["advanced_rag"]["input_tokens"])
    
    token_reduction = 1.0 - (avg_input_tokens_c / avg_input_tokens_a)
    
    # 計算 P95 延遲
    df_long_ms = pd.Series(stats["long_context"]["total_ms"])
    df_basic_ms = pd.Series(stats["basic_rag"]["total_ms"])
    df_adv_ms = pd.Series(stats["advanced_rag"]["total_ms"])
    
    p95_latency_a = df_long_ms.quantile(0.95) if not df_long_ms.empty else 0.0
    p95_latency_b = df_basic_ms.quantile(0.95) if not df_basic_ms.empty else 0.0
    p95_latency_c = df_adv_ms.quantile(0.95) if not df_adv_ms.empty else 0.0
    
    summary_data = {
        "metrics": {
            "long_context_baseline": {
                "avg_input_tokens": avg_input_tokens_a,
                "avg_total_latency_ms": sum(stats["long_context"]["total_ms"]) / len(stats["long_context"]["total_ms"]),
                "p95_total_latency_ms": p95_latency_a
            },
            "basic_rag": {
                "avg_input_tokens": avg_input_tokens_b,
                "avg_total_latency_ms": sum(stats["basic_rag"]["total_ms"]) / len(stats["basic_rag"]["total_ms"]),
                "p95_total_latency_ms": p95_latency_b,
                "retrieval_hit_at_5": sum(stats["basic_rag"]["hit_at_5"]) / len(stats["basic_rag"]["hit_at_5"]) if stats["basic_rag"]["hit_at_5"] else 0.0
            },
            "advanced_rag": {
                "avg_input_tokens": avg_input_tokens_c,
                "avg_total_latency_ms": sum(stats["advanced_rag"]["total_ms"]) / len(stats["advanced_rag"]["total_ms"]),
                "p95_total_latency_ms": p95_latency_c,
                "candidate_recall_at_20": sum(stats["advanced_rag"]["recall_at_20"]) / len(stats["advanced_rag"]["recall_at_20"]) if stats["advanced_rag"]["recall_at_20"] else 0.0,
                "reranked_hit_at_5": sum(stats["advanced_rag"]["hit_at_5"]) / len(stats["advanced_rag"]["hit_at_5"]) if stats["advanced_rag"]["hit_at_5"] else 0.0,
                "evidence_recall_at_5": sum(stats["advanced_rag"]["evidence_recall_at_5"]) / len(stats["advanced_rag"]["evidence_recall_at_5"]) if stats["advanced_rag"]["evidence_recall_at_5"] else 0.0
            }
        },
        "achievements": {
            "token_reduction_rate": token_reduction,
            "ac_06_token_reduction_passed": token_reduction >= 0.50
        }
    }
    
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"Summary metrics written to {JSON_OUTPUT_PATH}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    run_evaluation()
