實作題目規格書
基於兩階段檢索與 Token Budget 的高效率規格書問答系統
Two-Stage Retrieval with BGE-M3 and BGE Reranker
版本：1.0
建議對象：具 Python 基礎、首次或初階接觸 RAG 的學生
建議規模：2 人一組，3 週
指定環境：WSL Ubuntu / Python
核心技術：BGE-M3、向量資料庫、bge-reranker-v2-m3、Token Budget、LLM 串流生成
最低成果  進階 RAG 相較「未設計檢索的固定長上下文 Baseline」，答案正確率至少提升 5 個百分點，且平均 LLM Input Tokens 至少降低 50%。
本規格將「可執行、可比較、可重現」列為優先，不要求學生在第一次實作中完成 OCR、Agent、模型微調或極端延遲優化。
文件導覽
第 1–3 節：題目目標、範圍與系統架構
第 4–6 節：功能需求、資料格式與評估方法
第 7–9 節：最低驗收、交付成果與時程
附錄：實驗欄位、提交檢查表與參考資源
1. 題目背景與目的
一般長上下文問答會將整份文件或大量固定內容直接送入 LLM。這種方法雖然容易實作，但可能造成輸入 Token 偏高、無關內容干擾，以及回應延遲增加。本題要求學生建立 RAG（Retrieval-Augmented Generation）系統，讓模型只讀取與問題最相關的文件片段，並透過兩階段檢索改善候選排序。
本題不只要求「系統能回答」，還要求學生以固定資料、固定模型與固定參數比較三種架構，量化檢索命中率、答案品質、Token 消耗與延遲。
1.1 學習目標
理解 Embedding 如何將 Query 與文件 chunks 映射到向量空間，並以相似度進行初步檢索。
理解 Bi-encoder 檢索與 Cross-encoder Reranking 的角色差異。
實作 Chunking、Metadata、向量索引、Top-K Retrieval、Reranking、Context 組裝與來源引用。
正確量測 Retrieval Recall、答案正確率、LLM Input Tokens、TTFT 與總延遲。
透過錯誤分析辨識問題出在文件切分、初始召回、重排或生成階段。
2. 專題範圍與限制
2.1 必做範圍
模組
最低要求
文件處理
文字擷取、固定長度 chunking、overlap、頁碼與 chunk ID metadata
Embedding
使用 BAAI/bge-m3 建立文件與 Query 向量
向量檢索
使用 FAISS、Qdrant 或既有平台向量庫執行 Top-K 搜尋
Basic RAG
BGE-M3 直接取 Top-5 交給 LLM
Advanced RAG
BGE-M3 取 Top-20，再由 bge-reranker-v2-m3 重排取 Top-5
Token Budget
限制送入生成 LLM 的 context tokens，預設上限 2,000
評估
三組架構、同一測試集、相同 LLM 與生成參數
輸出
答案、文件名稱、頁碼與 chunk ID
2.2 不列入最低要求
OCR、版面分析、複雜表格重建與跨頁表格合併。
Agent、LangGraph 工作流、工具呼叫或多代理協作。
Embedding、Reranker 或 LLM 的微調。
正式前端、登入權限、雲端部署與高可用架構。
TTFT 必須低於 2.5 秒；本題只要求量測，2.5 秒列為加分目標。
Token 口徑  本題的「Token 降低 50%」僅比較送入生成 LLM 的 Input Tokens；不把 Embedding 與 Reranker 內部處理量混入同一指標。
3. 系統架構與比較組別
所有組別必須使用相同文件、相同問題、相同生成 LLM、相同 system prompt、相同溫度與相同最大輸出長度。唯一可變動項目是檢索與 context 組裝方式。
組別
名稱
流程
用途
A
Long-context Baseline
不使用向量檢索；將完整文件或固定且可重現的長上下文直接送入 LLM
建立「尚未設計 RAG」的對照
B
Basic RAG
BGE-M3 向量檢索 Top-5，直接送入 LLM
量測一般 RAG 的效果
C
Advanced RAG
BGE-M3 Top-20 → BGE Reranker → Top-5 → Token Budget
本題最終系統
3.1 Advanced RAG 流程
圖 1　兩階段檢索、Token Budget 與生成流程
使用者送出 Query。
BGE-M3 將 Query 轉成 embedding，向量資料庫取回 Top-20 候選 chunks。
bge-reranker-v2-m3 對每一組 Query–Chunk 做交叉比對並重新打分。
依分數排序後取 Top-5，再依 context token budget、去重規則組裝 prompt。
LLM 以串流方式生成答案，並回傳來源文件、頁碼與 chunk ID。
4. 功能需求
編號
功能
規格
FR-01
文件擷取
可讀取教師提供的 PDF／TXT 文字內容；PDF 擷取失敗時可先提供乾淨文字檔，不強制 OCR。
FR-02
Chunking
預設 400–600 tokens、overlap 50–100 tokens；參數須可設定並寫入紀錄。
FR-03
Metadata
每個 chunk 至少包含 document_id、document_name、page、chunk_id、text。
FR-04
Embedding
使用 BAAI/bge-m3；文件 embedding 應離線建立並保存，查詢時不得每次重建全庫。
FR-05
Vector Search
支援 Top-K 搜尋，至少可設定 K=5 與 K=20；記錄相似度分數。
FR-06
Basic RAG
將向量搜尋 Top-5 依統一 prompt 交給 LLM。
FR-07
Reranking
使用 BAAI/bge-reranker-v2-m3 對 Top-20 候選重新打分，最後保留 Top-5。
FR-08
Token Budget
預設 context 上限 2,000 tokens；依 rerank 分數由高至低加入，超過預算則跳過或停止。
FR-09
去重
相同 chunk ID 不得重複；高度重疊 chunks 至少以來源與文字重疊規則進行簡單去重。
FR-10
拒答
文件證據不足時，回覆「文件中沒有足夠資訊」，不可自行補造。
FR-11
引用
每個答案附上至少一個來源，格式包含文件名、頁碼與 chunk ID。
FR-12
Logging
逐題輸出檢索結果、rerank 結果、tokens、延遲、答案與引用，供後續重現。
4.1 建議技術與套件
項目
建議
作業環境
WSL Ubuntu、Python 3.10 以上
Embedding
BAAI/bge-m3；建議以 FlagEmbedding 載入
Reranker
BAAI/bge-reranker-v2-m3；建議以 FlagReranker 或相容介面推論
向量資料庫
FAISS、Qdrant 或現有平台向量庫，三者擇一
後端
純 Python CLI 即可；FastAPI 為選配
Token 計算
必須使用生成 LLM 對應 tokenizer 或 API 回傳 usage，不可只用字數估算
資料與結果
JSON／JSONL／CSV，需可由腳本批次執行
技術說明：BGE-M3 可用於多語言 dense、sparse 與 multi-vector retrieval；本題最低要求只使用 dense embedding。bge-reranker-v2-m3 為多語言 reranker，適合搭配第一階段候選重排。
5. 資料與測試集規格
5.1 文件資料
由教師提供 1–5 份規格書或技術文件；建議總文字量約 10,000–40,000 tokens。
Long-context Baseline 若無法容納全部內容，必須使用固定、程式化且對所有問題一致的截斷規則，不得人工挑選答案段落。
文件擷取後應保留頁碼對應；若原始格式沒有頁碼，可用 section 或段落序號替代並在報告說明。
5.2 測試題
題型
最低題數
目的
直接事實題
8
單一段落可回答
數字與單位題
5
需保留精確數值、範圍與單位
否定與例外題
5
含不得、除外、但書或條件限制
相似名詞干擾題
4
錯誤段落與 Query 關鍵字高度相似
跨段落整合題
3
需要兩個 evidence chunks
文件無答案題
5
測試拒答與幻覺
合計至少 30 題。建議教師提供 24 題與標準答案，學生另設計 6 題刁鑽題；若教師已提供完整題庫，學生可改為挑選 6 題做深入錯誤分析。
5.3 Gold Data 格式
{  "question_id": "Q001",  "question": "題目內容",  "reference_answer": "標準答案",  "gold_chunk_ids": ["doc01_p05_c03"],  "answerable": true,  "question_type": "exception"}
6. 實驗方法與評估指標
6.1 公平比較原則
三組使用同一生成 LLM、同一 system prompt 與回答格式。
temperature、top_p、max_output_tokens 等生成參數必須固定。
測試題不得依系統表現人工修改；所有題目應由同一批次腳本執行。
正式計時前至少暖機 3 次；報告平均值，建議另附 P95。
不得只展示成功案例；所有 30 題均須保留原始結果。
6.2 Retrieval 指標
Candidate Recall@20（題目層級）：在可回答題中，只要至少一個 Gold Chunk 出現在 Stage 1 Top-20，即視為該題命中。
Reranked Hit@5：Gold Chunk 經 Reranker 排序後是否進入最終 Top-5。跨段落題另計 Evidence Recall@5，以實際找回的 Gold Chunks 數量除以全部 Gold Chunks。
6.3 答案品質
分數
評分定義
2 分
答案正確，關鍵條件、例外、數值與單位完整；引用可支持答案。
1 分
主要方向正確，但遺漏部分條件、例外或來源不完整。
0 分
錯誤、無關、捏造，或文件無答案時仍強行作答。
答案正確率（%）＝ 實際總分 ÷（題數 × 2）× 100%。建議由兩人交叉評分；有爭議時保留理由。
6.4 Token 指標
每題至少記錄 system prompt tokens、query tokens、retrieved context tokens、conversation history tokens、LLM input tokens、output tokens。主要驗收採平均 LLM input tokens。
計算式  Token Reduction = 1 −（Advanced RAG 平均 LLM Input Tokens ÷ Long-context Baseline 平均 LLM Input Tokens）
6.5 延遲指標
指標
定義
Embedding time
Query embedding 耗時
Vector search time
向量資料庫搜尋耗時
Rerank time
Top-20 候選重排耗時
End-to-end TTFT
系統收到 Query 至生成 LLM 回傳第一個 token 的時間
Total response time
收到 Query 至完整回答結束的時間
本題最低驗收不限制 TTFT 數值，只要求量測與分析；平均 TTFT ≤ 2.5 秒列為加分項目。
7. 最低驗收標準
以下標準刻意設定為初階團隊在 3 週內可達成的門檻。所有數值以至少 30 題的同一測試集計算。
編號
指標
最低標準
屬性
AC-01
可執行性
批次執行成功率 ≥ 95%，不得逐題人工換 context
必達
AC-02
Candidate Recall@20
可回答題 ≥ 80%
必達
AC-03
Reranked Hit@5
可回答題 ≥ 75%
必達
AC-04
答案正確率
Advanced RAG ≥ 65%
必達
AC-05
相對準確率
Advanced RAG 至少比 Long-context Baseline 高 5 個百分點
必達
AC-06
Token 降低率
Advanced RAG 比 Long-context Baseline 少 ≥ 50% 平均 LLM Input Tokens
必達
AC-07
引用正確率
答案引用可支持內容的比例 ≥ 80%
必達
AC-08
無答案拒答率
5 題無答案題中至少 3 題正確拒答（≥ 60%）
必達
AC-09
Basic RAG 比較
需完整報告；Advanced 不得低於 Basic 超過 5 個百分點
必達
AC-10
延遲
完成各階段平均延遲紀錄，不設硬性秒數
必達
判定原則  若 Token 已降低 50% 但準確率未提升，或準確率提升但 Token 未減半，均視為尚未完成核心目標；學生應透過錯誤分析調整 chunk size、candidate size、Top-K 或 token budget。
7.1 建議但不強制的目標
Candidate Recall@20 ≥ 90%。
Reranked Hit@5 ≥ 85%。
Advanced RAG 答案正確率比 Basic RAG 高 3 個百分點以上。
無答案拒答率 ≥ 80%。
平均 End-to-end TTFT ≤ 2.5 秒。
8. 必繳成果
交付項目
內容
程式碼
可執行 Python 專案；包含 ingestion、index、query、evaluation 等入口。
README.md
安裝、模型下載、環境變數、建立索引、啟動與批次測試說明。
測試資料
至少 30 題的 JSON／JSONL，含 reference answer 與 gold chunk IDs。
原始結果
每題三組答案、檢索結果、rerank 分數、tokens、延遲與引用。
比較報告
3–5 頁；含方法、結果表、至少 5 題錯誤案例與結論。
展示
可用 CLI、Notebook 或 API 展示一題完整流程，不強制製作前端。
8.1 建議專案結構
project/├─ README.md├─ requirements.txt├─ data/│  ├─ documents/│  └─ questions.jsonl├─ src/│  ├─ ingest.py│  ├─ retrieve.py│  ├─ rerank.py│  ├─ generate.py│  └─ evaluate.py├─ results/│  ├─ per_question.csv│  └─ summary.json└─ report.pdf
9. 評分建議
項目
比例
評分重點
系統實作
35%
文件處理、Embedding、Vector Search、Reranker、Token Budget、引用
實驗設計
25%
三組公平比較、測試集、指標與可重現性
成果與驗收
25%
最低標準達成程度、結果可信度
報告與錯誤分析
15%
圖表、案例、瓶頸分析與改進建議
建議規則：只要核心流程完整且數據誠實，即使少數指標未達標仍可取得基本分；但若缺少 Baseline、未固定實驗條件或逐題人工選 context，則不能宣稱完成比較實驗。
10. 建議時程
階段
工作內容
第 1 週
完成文字擷取、chunking、metadata、BGE-M3 embedding、向量索引與 Basic RAG。
第 2 週
完成 Top-20 candidates、BGE Reranker、Top-5、Token Budget、引用與 logging。
第 3 週
完成 30 題批次實驗、指標計算、錯誤分析、效能比較與報告。
11. 加分挑戰
加分項目
說明
候選數量實驗
比較 Top-10、20、30 的 Recall、Rerank latency 與答案品質。
Hybrid Retrieval
加入 BM25 或 BGE-M3 sparse retrieval，與 dense retrieval 融合。
結構化 Chunking
依標題、段落、條款與表格切分，並與固定長度 chunking 比較。
Parent-child Retrieval
小 chunk 搜尋、較完整 parent section 回傳。
動態 Top-K
根據 rerank 分數差距或 token budget，自動決定送入 2–5 個 chunks。
推論優化
FP16、INT8、ONNX 或 batching；需附優化前後數據。
延遲挑戰
平均 End-to-end TTFT ≤ 2.5 秒，且不得犧牲最低準確率。
12. 錯誤分析要求
報告至少挑選 5 題失敗案例，且不得全部選同一類錯誤。每題需標記最主要失敗階段：
分類
判定方式
E1：Chunking
答案被切斷、頁碼錯位、表格或條件分離。
E2：Stage 1 Retrieval
Gold Chunk 未進入 Top-20。
E3：Reranking
Gold Chunk 已進入候選，但被排出 Top-5。
E4：Context Budget
正確證據被去重或 token budget 排除。
E5：Generation
證據正確，但 LLM 誤讀、漏掉例外或產生幻覺。
E6：Evaluation
Gold Chunk 或 reference answer 標註不完整。
附錄 A：每題結果欄位
欄位
說明
question_id
題目編號
system_variant
long_context / basic_rag / advanced_rag
retrieved_chunk_ids
Stage 1 取回的 chunk IDs
retrieval_scores
向量相似度分數
reranked_chunk_ids
重排後的 chunk IDs；非 Advanced 可留空
rerank_scores
Reranker 分數；非 Advanced 可留空
selected_chunk_ids
實際送入 LLM 的 chunks
answer
模型完整答案
citations
文件名、頁碼與 chunk ID
input_tokens
生成 LLM 的全部輸入 tokens
context_tokens
其中屬於檢索 context 的 tokens
output_tokens
生成輸出 tokens
embedding_ms / search_ms / rerank_ms
各檢索階段耗時
ttft_ms / total_ms
端到端首字與總時間
human_score
0／1／2 分
error_type
E1–E6 或空白
附錄 B：提交前檢查表
☐ 三組系統均使用相同 LLM、prompt 與生成參數。
☐ Advanced RAG 確實先取 Top-20，再 Rerank 取 Top-5。
☐ Token Budget 以 tokenizer 計算，不以中文字數估算。
☐ 結果包含全部 30 題，而非只挑成功案例。
☐ Token Reduction 的 Baseline 為固定長上下文組。
☐ 引用可追溯至文件、頁碼與 chunk ID。
☐ 提供 Recall、答案分數、拒答、tokens 與 latency。
☐ 至少完成 5 題跨階段錯誤分析。
☐ README 可讓另一位同學在新環境重現。
☐ 報告清楚揭露硬體、模型版本與主要參數。
附錄 C：官方技術參考
BGE-M3 Model Card：https://huggingface.co/BAAI/bge-m3
BGE Reranker v2 M3 Model Card：https://huggingface.co/BAAI/bge-reranker-v2-m3
學生應在報告中記錄實際安裝版本與推論參數，避免日後套件更新造成結果無法重現。
