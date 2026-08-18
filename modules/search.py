import os
import re
import chromadb
from PIL import Image
from modules.llm import describe_image 
from modules.text_encoder import embed_text 
from modules.image_utils import extract_exif_metadata, get_file_time

BASE_INDEX_DIR = "data/index"
os.makedirs(BASE_INDEX_DIR, exist_ok=True)

# 初始化客户端
paper_client = chromadb.PersistentClient(path=os.path.join(BASE_INDEX_DIR, "papers"))
image_client = chromadb.PersistentClient(path=os.path.join(BASE_INDEX_DIR, "images"))
para_client  = chromadb.PersistentClient(path=os.path.join(BASE_INDEX_DIR, "paragraphs"))

# 论文和段落集合
paper_collection = paper_client.get_or_create_collection("papers")
para_collection  = para_client.get_or_create_collection("paragraphs")

# ⭐ 关键：图片拆分为“视觉库”和“文本库”
image_visual_col = image_client.get_or_create_collection("images_visual") # 存 CLIP 向量
image_text_col   = image_client.get_or_create_collection("images_text")   # 存 BGE 向量

def add_image_embedding(iid, image_path):
    from modules.image_encoder import embed_image
    
    # 1. 提取向量和 AI 描述
    img = Image.open(image_path).convert("RGB")
    visual_emb = embed_image(img).tolist()
    description = describe_image(image_path)
    text_emb = embed_text(description).tolist()
    
    # 2. 🎯 日期识别逻辑（优先级排序）
    date_int = 19700101 # 默认兜底
    found_date = False
    
    # --- 第一优先级：EXIF (相机的原始记录) ---
    exif_data = extract_exif_metadata(image_path)
    raw_exif_time = exif_data.get("datetime")
    
    if raw_exif_time and len(raw_exif_time) >= 10:
        try:
            # 格式兼容：可能是 "2024:01:01" 或 "2024-01-01"
            clean_date = raw_exif_time[:10].replace("-", "").replace(":", "")
            year = int(clean_date[:4])
            if 1970 <= year <= 2030: # 过滤清朝老片
                date_int = int(clean_date)
                found_date = True
                print(f"📷 [EXIF 匹配] 拍摄日期: {date_int}")
        except: pass

    # --- 第二优先级：从文件名提取 (文件名通常带 YYYYMMDD) ---
    if not found_date:
        # 正则匹配：查找 202x0101 或 202x-01-01 这种格式
        # iid 通常就是文件名
        date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', iid)
        if date_match:
            y, m, d = date_match.groups()
            year_val = int(y)
            if 1970 <= year_val <= 2030: # 过滤掉不合理的年份（防止误认随机数字）
                date_int = int(f"{y}{m}{d}")
                found_date = True
                print(f"📄 [文件名匹配] 识别到日期: {date_int}")

    # --- 第三优先级：文件系统时间 (最后的保障) ---
    if not found_date:
        file_time = get_file_time(image_path) # 通常返回 "2024-01-01"
        date_int = int(file_time[:10].replace("-", ""))
        print(f"🕒 [文件时间] 无有效信息，使用系统修改日期: {date_int}")

    # 3. 存储到数据库
    image_meta = {
        "path": image_path, 
        "description": description,
        "date": date_int
    }
    
    image_visual_col.add(ids=[iid], embeddings=[visual_emb], metadatas=[image_meta])
    image_text_col.add(ids=[iid], embeddings=[text_emb], metadatas=[image_meta])
    return description

# 4. ⭐ 核心改动：让搜索函数支持日期过滤 (date_filter)
def search_image_visual(query_visual_emb, top_k=5, date_filter=None):
    """以图搜图：增加时间过滤"""
    q = query_visual_emb.tolist() if hasattr(query_visual_emb, "tolist") else query_visual_emb
    
    # 构建 ChromaDB 的 where 过滤器
    where_clause = _build_date_filter(date_filter)

    results = image_visual_col.query(
        query_embeddings=[q], 
        n_results=top_k,
        where=where_clause # 传入过滤器
    )
    return format_image_results(results)

def search_image_text(query_text_emb, top_k=5, date_filter=None):
    """以文搜图：增加时间过滤"""
    q = query_text_emb.tolist() if hasattr(query_text_emb, "tolist") else query_text_emb
    
    # 构建 ChromaDB 的 where 过滤器
    where_clause = _build_date_filter(date_filter)

    results = image_text_col.query(
        query_embeddings=[q], 
        n_results=top_k,
        where=where_clause # 传入过滤器
    )
    return format_image_results(results)

def _build_date_filter(date_filter):
    if not date_filter:
        return None
    
    conditions = []
    
    # 1. 处理开始日期 (如果有)
    if date_filter.get("start_date"):
        start_val = int(date_filter["start_date"].replace("-", ""))
        conditions.append({"date": {"$gte": start_val}})
        
    # 2. 处理结束日期 (如果有)
    if date_filter.get("end_date"):
        end_val = int(date_filter["end_date"].replace("-", ""))
        conditions.append({"date": {"$lte": end_val}})
    
    # 3. 动态构建 ChromaDB 语法
    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0] # 只有一边，比如“2022年以前”
    else:
        return {"$and": conditions} # 两边都有，比如“去年夏天”

def format_image_results(results):
    hits = []
    if results.get("distances") and results.get("metadatas"):
        for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
            hits.append({
                "path": meta.get("path", ""),
                "distance": dist,
                "description": meta.get("description", "暂无描述"),
                "date": meta.get("date", "未知") # 顺便把日期传回 UI
            })
    return hits

# --- 论文和段落函数 (保持不变) ---
def add_paper_embedding(pid, embedding, metadata):
    emb = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    paper_collection.add(ids=[pid], embeddings=[emb], metadatas=[metadata])

def search_paper(query_emb, top_k=5):
    q = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
    return paper_collection.query(query_embeddings=[q], n_results=top_k, include=["metadatas"])

def add_paragraph(pid, para_id, embedding, metadata):
    emb = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    metadata = dict(metadata)
    metadata["para_id"] = para_id
    para_collection.add(ids=[f"{pid}_{para_id}"], embeddings=[emb], metadatas=[metadata])

def search_paragraph(query_emb, top_k=5):
    q = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
    results = para_collection.query(query_embeddings=[q], n_results=top_k, include=["metadatas"])
    if not results or not results.get("metadatas"): return {"metadatas": [[]], "processed": []}
    processed = [{"paper": m.get("paper", "unknown"), "para_id": m.get("para_id", -1), "text": m.get("text", "")} for m in results["metadatas"][0]]
    results["processed"] = processed
    return results