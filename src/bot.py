import os
import sys
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# 載入模組
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.dirname(SRC_DIR)
sys.path.append(os.path.join(WORKING_DIR, "src"))

from retrieve import retrieve
from rerank import rerank
from generate import generate

load_dotenv(os.path.join(WORKING_DIR, ".env"))

# 啟用 Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 您好！我是 RAG 兩階段檢索問答助理。\n\n"
        "請向我發問，我會根據已經索引的規格書進行兩階段檢索，並結合 Token Budget 為您回答問題。\n"
        "每個答案都會附上引用文獻來源喔！"
    )
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "💡 **使用說明**：\n"
        "1. 直接在對話框輸入您的問題並發送。\n"
        "2. 系統流程：\n"
        "   - Stage 1: 使用 BGE-M3 檢索前 20 個最相關片段。\n"
        "   - Stage 2: 使用 BGE Reranker v2 M3 進行重排並精選前 5 個片段。\n"
        "   - 限制 Context 長度在 2,000 tokens 以內。\n"
        "   - 使用 Hermes API 進行精確回答（附來源）。"
    )
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    if not query.strip():
        return
        
    # 傳送暫時等待訊息
    waiting_message = await update.message.reply_text("🔍 正在檢索與生成回答，請稍候...")
    
    try:
        # 1. Stage 1: Vector Search Top-20
        logger.info(f"User Query: {query}")
        retrieved = retrieve(query, top_k=20)
        
        # 2. Stage 2: Rerank Top-5
        reranked = rerank(query, retrieved, top_k=5)
        
        # 3. Generate Answer
        res = generate(query, reranked, max_context_tokens=2000)
        answer = res["answer"]
        
        # 4. 回傳結果
        await waiting_message.edit_text(answer)
        logger.info(f"Replied to: {query} with {len(res['selected_chunks'])} citations.")
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await waiting_message.edit_text(f"❌ 處理問題時發生錯誤：{str(e)}")

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        print("Error: TELEGRAM_BOT_TOKEN not found or not set in .env!")
        print("Please check your .env configuration.")
        sys.exit(1)
        
    # 檢查是否有 FAISS 索引檔，否則引導使用者執行 ingest
    index_path = os.path.join(WORKING_DIR, "data", "faiss_index.bin")
    if not os.path.exists(index_path):
        print(f"Warning: Index file {index_path} not found.")
        print("Please run 'python src/ingest.py' to build the index before querying the bot.")
        
    print("Starting Telegram Bot...")
    app = ApplicationBuilder().token(token).build()
    
    # 註冊處理器
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # 啟動輪詢 (Polling)
    app.run_polling()

if __name__ == "__main__":
    main()
