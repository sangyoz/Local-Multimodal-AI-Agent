# config.py
import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量（如果存在）
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PAPER_DIR = os.path.join(DATA_DIR, "papers")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = os.path.join(DATA_DIR, "index")

# ===== 本地模型路径 =====
LLM_PATH = r"E:\LLM_Models\Qwen2.5-7B-Instruct"
TEXT_EMB_PATH = r"E:\LLM_Models\bge-large-zh-v1.5"
CLIP_PATH = r"E:\LLM_Models\clip-vit-l-14"

# 模式开关：True 使用 API，False 使用本地模型
USE_API = True

# DeepSeek 配置（从环境变量读取）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

# 通义千问视觉版配置（从环境变量读取）
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3-vl-plus")

