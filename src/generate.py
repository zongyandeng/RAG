import os
import tiktoken
import requests
from dotenv import load_dotenv
from openai import OpenAI

# 載入環境變數
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.dirname(SRC_DIR)
load_dotenv(os.path.join(WORKING_DIR, ".env"))

HERMES_API_KEY = os.getenv("HERMES_API_KEY")
HERMES_API_BASE = os.getenv("HERMES_API_BASE", "https://api.hermes-gateway.com/v1")
HERMES_MODEL_NAME = os.getenv("HERMES_MODEL_NAME", "hermes-llama-3-8b")

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            # 預設使用 cl100k_base 作為 token 計算基準
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Warning: Failed to load tiktoken. Error: {e}")
    return _tokenizer

def count_tokens_via_api(text, model=HERMES_MODEL_NAME, api_key=HERMES_API_KEY, api_base=HERMES_API_BASE):
    if not api_key or not model or not api_base:
        return None
    # 檢查是否為 Gemini 模型
    if "gemini" not in model.lower():
        return None
    
    # 建立 REST API URL (移除 /openai 結尾以呼叫原生 API)
    base_url = api_base.replace("/openai", "")
    url = f"{base_url}/models/{model}:countTokens?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": text}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            res_data = response.json()
            return res_data.get("totalTokens")
    except Exception as e:
        print(f"Warning: Failed to call Gemini countTokens API: {e}")
    return None

def count_tokens(text):
    # 優先使用 Gemini API 計算精確 token 數
    api_tokens = count_tokens_via_api(text)
    if api_tokens is not None:
        return api_tokens
        
    # 本地 fallback 流程
    tok = get_tokenizer()
    if tok:
        return len(tok.encode(text))
    return len(text) // 2 # 粗估

def build_context_with_budget(chunks, max_context_tokens=2000):
    """
    根據 Token Budget (預設 2000) 塞入 Chunks，並進行去重
    """
    selected_chunks = []
    current_tokens = 0
    seen_chunk_ids = set()
    
    # 依分數排序，逐一加入
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_chunk_ids:
            continue
            
        chunk_text = chunk["text"]
        # 包裝格式
        formatted_chunk = f"Source: {chunk['document_name']}, Page: {chunk['page']}, ID: {chunk_id}\nContent: {chunk_text}\n\n"
        chunk_tokens = count_tokens(formatted_chunk)
        
        # 判斷是否會超出預算
        if current_tokens + chunk_tokens <= max_context_tokens:
            selected_chunks.append(chunk)
            current_tokens += chunk_tokens
            seen_chunk_ids.add(chunk_id)
        else:
            # 超出預算則跳過或停止
            continue
            
    # 組裝最終的 context text
    context_parts = []
    for chunk in selected_chunks:
        context_parts.append(
            f"Source: {chunk['document_name']}, Page: {chunk['page']}, Chunk ID: {chunk['chunk_id']}\n"
            f"Content: {chunk['text']}\n"
        )
    context_text = "\n---\n".join(context_parts)
    
    return context_text, selected_chunks, current_tokens

def generate(query, context_chunks, max_context_tokens=2000, stream=False):
    """
    串接 Hermes API (OpenAI 相容) 生成答案
    """
    # 1. 依據 Token Budget 組裝 Context
    context_text, selected_chunks, context_tokens = build_context_with_budget(context_chunks, max_context_tokens)
    
    # 2. 定義 System Prompt
    system_prompt = (
        "你是一個專業的規格書問答助理。請嚴格根據以下提供的 Context 資訊回答使用者的問題。\n"
        "請務必遵循以下三大準則：\n"
        "1. 僅根據 Context 中有明確提及的事實回答，切勿加入任何外部知識、假設或憑空捏造。\n"
        "2. 若 Context 中的資訊不足以回答問題，請明確且只回覆：「文件中沒有足夠資訊」。\n"
        "3. 在回答的結尾，必須列出所有引用資料的來源文件名稱、頁碼和 Chunk ID，格式如下：\n"
        "   引用來源：[文件名稱, Page X, Chunk ID: Y]\n\n"
        "以下是 Context 資訊：\n"
        "--------------------\n"
        f"{context_text}\n"
        "--------------------\n"
    )
    
    # 計算各部分 Token
    system_prompt_tokens = count_tokens(system_prompt)
    query_tokens = count_tokens(query)
    history_tokens = 0 # 單輪問答，對話歷史 token 為 0
    input_tokens = system_prompt_tokens + query_tokens
    
    # 檢查 API 金鑰
    api_key = os.getenv("HERMES_API_KEY")
    if not api_key or "your_" in api_key or api_key.strip() == "":
        # 如果沒有設定 API Key，則返回 Mock 回答，方便測試
        mock_ans = (
            "【測試模式：未偵測到 HERMES_API_KEY】\n"
            f"收到您的問題：\"{query}\"\n"
            f"檢索到 {len(selected_chunks)} 個 Chunks，已加入 Context 的 Chunks 包括：\n"
        )
        for c in selected_chunks:
            mock_ans += f"- {c['chunk_id']} (Page {c['page']})\n"
        mock_ans += "\n引用來源：" + ", ".join([f"[{c['document_name']}, Page {c['page']}, Chunk ID: {c['chunk_id']}]" for c in selected_chunks])
        
        return {
            "answer": mock_ans,
            "selected_chunks": selected_chunks,
            "system_prompt_tokens": system_prompt_tokens,
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "history_tokens": history_tokens,
            "input_tokens": input_tokens,
            "output_tokens": count_tokens(mock_ans)
        }
        
    # 初始化 OpenAI Client 連接 Hermes API Gateway
    client = OpenAI(
        api_key=api_key,
        base_url=HERMES_API_BASE
    )
    
    try:
        response = client.chat.completions.create(
            model=HERMES_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0, # 固定溫度為 0 確保可重現性與準確度
            max_tokens=800,
            stream=stream
        )
        
        if stream:
            # 串流模式，回傳生成流與包含所有輸入端 Token 明細的字典
            token_meta = {
                "system_prompt_tokens": system_prompt_tokens,
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "history_tokens": history_tokens,
                "input_tokens": input_tokens
            }
            return response, selected_chunks, token_meta
        else:
            # 非串流模式
            answer = response.choices[0].message.content
            output_tokens = response.usage.completion_tokens if response.usage else count_tokens(answer)
            
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                # 如果 API 有回傳實際的 prompt_tokens，我們校正 system_prompt_tokens
                system_prompt_tokens = max(0, input_tokens - query_tokens)
            
            return {
                "answer": answer,
                "selected_chunks": selected_chunks,
                "system_prompt_tokens": system_prompt_tokens,
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "history_tokens": history_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
    except Exception as e:
        print(f"Error during API call: {e}")
        error_ans = f"API 呼叫出錯：{str(e)}"
        return {
            "answer": error_ans,
            "selected_chunks": selected_chunks,
            "system_prompt_tokens": system_prompt_tokens,
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "history_tokens": history_tokens,
            "input_tokens": input_tokens,
            "output_tokens": count_tokens(error_ans)
        }
