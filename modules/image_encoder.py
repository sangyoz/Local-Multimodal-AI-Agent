# modules/image_encoder.py
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import CLIP_PATH

# =====================
# 设备选择
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 加载 CLIP 模型
# =====================
processor = CLIPProcessor.from_pretrained(CLIP_PATH)
model = CLIPModel.from_pretrained(
    CLIP_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()


# =====================
# 图像 → 向量（CLIP Image Encoder）
# =====================
@torch.no_grad()
def embed_image(image_input) -> np.ndarray:
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"不支持的图像输入类型: {type(image_input)}")

    inputs = processor(images=image, return_tensors="pt").to(device)

    # 获取输出
    outputs = model.get_image_features(**inputs)

    # 1. 兼容性拆包：确保拿到 Tensor
    if not isinstance(outputs, torch.Tensor):
        features = getattr(outputs, "image_embeds", outputs[0])
    else:
        features = outputs

    # 2. 归一化
    features = features / features.norm(dim=-1, keepdim=True)

    # 3. 【核心修复】强制拍扁成 1 维数组
    # .flatten() 会把 [[[0.1, 0.2]]] 变成 [0.1, 0.2]
    return features.cpu().detach().numpy().flatten()


@torch.no_grad()
def embed_clip_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)

    outputs = model.get_text_features(**inputs)

    if not isinstance(outputs, torch.Tensor):
        features = getattr(outputs, "text_embeds", outputs[0])
    else:
        features = outputs

    features = features / features.norm(dim=-1, keepdim=True)

    # 3. 【核心修复】强制拍扁成 1 维数组
    return features.cpu().detach().numpy().flatten()
