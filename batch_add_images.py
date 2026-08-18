import os
from pathlib import Path
from config import IMAGE_DIR
from modules.search import add_image_embedding, image_visual_col

def batch_add_images():
    image_dir = Path(IMAGE_DIR)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
    
    # 改为 yield，发送初始信息
    yield f"📁 找到 {len(image_files)} 张图片，开始增量同步...\n"
    success, skip, fail = 0, 0, 0

    for img_path in list(set(image_files)):
        fname = img_path.name
        abs_path = str(img_path.absolute())
        
        # 查重逻辑
        if image_visual_col.get(ids=[fname])['ids']:
            yield f"⏭️  跳过已存在的: {fname}\n"
            skip += 1
            continue

        try:
            # 执行识别
            desc = add_image_embedding(fname, abs_path)
            success += 1
            # 成功后 yield 详细结果
            yield f"[*] 正在处理: {fname}... ✅\n   └─ AI识别标签: {desc}\n"
        except Exception as e:
            fail += 1
            yield f"[*] 正在处理: {fname}... ❌ 失败: {e}\n"

    result_msg = f"\n📊 统计：成功 {success} | 跳过 {skip} | 失败 {fail}"
    yield result_msg

if __name__ == "__main__":
    # 脚本模式运行时，需要遍历生成器才能看到输出
    for status in batch_add_images():
        print(status, end="", flush=True)