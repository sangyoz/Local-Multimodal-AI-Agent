# modules/text_encoder.py
import torch
from sentence_transformers import SentenceTransformer
from config import TEXT_EMB_PATH

# 自动检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(TEXT_EMB_PATH, device=device)

def embed_text(text: str):
    return model.encode(
        text,
        normalize_embeddings=True
    )