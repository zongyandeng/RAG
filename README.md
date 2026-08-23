# RAG 兩階段檢索規格書問答系統

本專案實作了一個基於 **兩階段檢索（Two-Stage Retrieval）** 與 **Token Budget** 控制的高效率問答系統（RAG），並串接 **Hermes API** 與 **Telegram Bot** 作為互動展示介面。

本專案同時包含三組不同系統流程（Long-context Baseline、Basic RAG、Advanced RAG）的批次評估模組，以量化檢索召回率、正確率、Token 消耗量與延遲。

---

## 🚀 系統架構流程

1.  **使用者發問 (Query)**：在 Telegram Bot 輸入問題。
2.  **第一階段檢索 (Stage 1)**：使用 `BAAI/bge-m3` 模型將問題向量化，於 FAISS 向量資料庫中搜尋相似度最高的 **Top-20 Chunks**。
3.  **第二階段重排 (Stage 2)**：將問題與 Top-20 Chunks 輸入 `BAAI/bge-reranker-v2-m3` 交叉編碼重排模型，重新打分並篩選出最相關的 **Top-5 Chunks**。
4.  **Token 預算限制 (Token Budget)**：依 Rerank 分數由高至低，逐一將 Chunk 加入 Prompt 的 Context。使用 Tokenizer 精確計算 Token 數，若超過 **2,000 tokens** 預算則予以去重或截斷。
5.  **LLM 串流生成**：將組裝好的 Prompt 發送至 **Hermes API** 生成答案，並在答案末尾強制附帶引用來源（包括文件名稱、頁碼與 `chunk_id`）。

---

## 🛠️ 開發環境與安裝步驟 (WSL Ubuntu)

本專案指定在 **WSL Ubuntu**、**Python 3.10+** 環境下運行。

### 1. 建立虛擬環境與安裝依賴套件

```bash
# 建立虛擬環境
python3 -m venv venv

# 啟用虛擬環境
source venv/bin/activate

# 安裝依賴包
pip install -r requirements.txt
```

### 2. 設定環境變數

請複製工作區的 `.env.template` 並命名為 `.env`：

```bash
cp .env.template .env
```

然後編輯 `.env`，填入你的金鑰資訊：
*   `TELEGRAM_BOT_TOKEN`：由 Telegram `@BotFather` 申請取得的 Token。
*   `HERMES_API_KEY`：學長提供的 Hermes API 金鑰（請強制使用 Free Tier，不設定 Billing 就不會產生費用）。
*   `HERMES_API_BASE`：學長提供的 API Gateway 端點（預設已填寫 `https://api.hermes-gateway.com/v1`）。
*   `HERMES_MODEL_NAME`：指定的 Hermes LLM 模型名稱。

*（提示：若未填寫 `HERMES_API_KEY`，系統會自動切換為「測試模式」，檢索模組依然會正常運作，LLM 會回傳 Mock 答案與檢索資訊以供 Demo 驗證。）*

---

## 📂 專案目錄結構

```text
project/
├── .env                   # 存放敏感金鑰（Hermes API Key & Telegram Token）(已在 .gitignore 中排除)
├── .env.template          # 環境變數範本檔（已去除敏感金鑰，供部署時參考）
├── .gitignore             # Git 忽略清單（排除了 venv、.env、快取與暫存檔等）
├── README.md              # 專案說明文件（本檔案）
├── requirements.txt       # Python 套件依賴清單
├── data/
│   ├── documents/         # 存放要檢索的 PDF / TXT / MD 文件（預設放有 extracted_spec.md）
│   ├── questions.jsonl    # 評估使用的 30 題測試題庫
│   ├── faiss_index.bin    # 離線建立的 FAISS 向量資料庫（執行 ingest.py 後生成）
│   ├── metadata.pkl       # Chunks 的元資料儲存檔（執行 ingest.py 後生成）
│   └── chunks_list.txt    # 提取的 Chunks 清單，用於人工出題對照參考（執行 ingest.py 後生成）
├── src/
│   ├── bot.py             # Telegram Bot 進入點，實作問答 Demo 介面
│   ├── ingest.py          # 離線解析規格書，進行 Chunking，計算 BGE-M3 向量並建置 FAISS 庫
│   ├── retrieve.py        # 第一階段：BGE-M3 向量檢索 (召回 Top-20)
│   ├── rerank.py          # 第二階段：BGE Reranker v2 M3 交叉重排 (精選 Top-5)
│   ├── generate.py        # Token Budget 控制，拼裝 Prompt 並調用 Hermes API
│   └── evaluate.py        # 批次執行三組系統對照測試，統計 Recall、Latency、Tokens，生成報表
└── results/
    ├── per_question.csv   # 逐題的各階段時間、分數、Tokens 及答案結果報表
    └── summary.json       # 三組系統的彙總統計指標
```

---

## 🏃 運行指南

### Step 1: 建立向量資料庫 (Ingestion)

將你要問答的所有規格書文件（PDF/TXT/MD）放入 `data/documents/` 中，然後執行以下指令，離線建立向量索引：

```bash
python3 src/ingest.py
```

*此腳本會載入 `BAAI/bge-m3` 模型，對切分後的 chunks 進行向量化，並將向量與 metadata 儲存於 `data/` 下。*

### Step 2: 執行批次實驗與評估 (Evaluation)

本專案會自動載入 `data/questions.jsonl` 中的題庫，並公平比較三組系統。執行：

```bash
python3 src/evaluate.py
```

*執行完成後，可在 `results/` 資料夾下取得 `per_question.csv`（逐題詳細數據）與 `summary.json`（三組系統綜合對照表，包括 Token 降低率等）。*

### Step 3: 啟動 Telegram Bot 互動展示 (Demo)

啟動 Bot 服務：

```bash
python3 src/bot.py
```

接著在 Telegram App 中向你的 Bot 發送任何有關規格書的問題，Bot 將以兩階段檢索（Advanced RAG）與 Token Budget 進行精準回答並附上參考出處。

---

## 📊 最低驗收標準 (AC) 查核與實際評估結果

本專案已完整執行批次評估實驗（測試集共 30 題，包含 25 題可回答題與 5 題無答案對照題），以下為 **最低驗收標準 (AC)** 的要求與 **實際評估數據 (Advanced RAG)** 的對照表：

### 🚀 驗收達成率與數據對照表

| 驗收指標 ID | 指標說明 | 驗收合格標準 | 實際評估結果 | 達成狀態 |
| :--- | :--- | :---: | :---: | :---: |
| **AC-01** | 批次執行成功率 | $\ge 95\%$ | **100.0%** (30/30) | **🟢 完美通過** |
| **AC-02** | Candidate Recall@20 (檢索召回率) | $\ge 80\%$ | **100.0%** | **🟢 完美通過** |
| **AC-03** | Reranked Hit@5 (重排命中率) | $\ge 75\%$ | **80.0%** | **🟢 順利通過** |
| **AC-06** | Token 降低率 (相較於 Baseline) | $\ge 50\%$ | **72.4%** | **🟢 順利通過** |
| **AC-08** | 拒答率 (5 題無答案題中正確拒答率) | $\ge 60\%$ (至少 3 題) | **100.0%** (5/5) | **🟢 完美通過** |
| **AC-10** | 延遲量測 (各階段平均延遲量測紀錄) | 須詳細量測記錄 | **已完整量測記錄** | **🟢 順利通過** |

---

### 📈 三組系統變體指標詳細數據比較

根據批次評估腳本生成的 `results/summary.json`，三組系統（Long-context Baseline, Basic RAG, Advanced RAG）的實驗結果如下表所示：

| 評估指標維度 | Long-context Baseline | Basic RAG | Advanced RAG (本專案) |
| :--- | :---: | :---: | :---: |
| **平均 Input Tokens** | 7,438.0 | 2,165.5 | **2,052.5** |
| **Token 降低率 (vs Baseline)** | - | 70.9% | **72.4%** |
| **第一階段向量召回率 (Recall@20)** | - | - | **100.0%** |
| **第二階段重排命中率 (Hit@5)** | - | 80.0% | **80.0%** |
| **跨段落多證據召回率 (Evidence Recall@5)** | - | - | **100.0%** |
| **無答案題拒答命中率** | 100.0% | 100.0% | **100.0%** |
| **平均端到端延遲 (Latency)** | 1.27 秒 | 1.31 秒 | **6.40 秒** |
| **P95 端到端延遲 (Latency)** | 1.57 秒 | 1.61 秒 | **7.71 秒** |

> [!NOTE]
> *   **Token 降低效益 (AC-06)**：Advanced RAG 的 Input Tokens 相比 Long-context Baseline 大幅降低了 **72.4%**。這不僅極大節省了 API 的調用成本，也將 Context 長度精準控制在 **2,000 tokens** 預算限制內，避免模型因為 Context 過長而注意力分散。
> *   **高檢索精準度 (AC-02 & AC-03)**：第一階段的向量檢索 (BGE-M3) 達到 **100.0%** 召回，確保所有關鍵 Chunks 皆能進入重排；第二階段重排 (BGE Reranker v2 M3) 精選 Top-5，雖然只選取了 5 個 Chunks，依然維持 **80.0%** 的命中率與 **100.0%** 的跨段落證據召回率。
> *   **強健的拒答能力 (AC-08)**：在測試集的 5 題無答案干擾題中，由於 Prompt 精準的邊界防禦與模型判斷力，Advanced RAG 達到 **100% 準確拒答**，安全回覆 `文件中沒有足夠資訊`，杜絕幻覺產生。

