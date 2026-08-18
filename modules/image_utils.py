# modules/image_utils.py
import os
from datetime import datetime
from PIL import Image
import piexif

def extract_exif_metadata(file_path: str):
    """提取图片里的 EXIF 拍摄时间"""
    metadata = {"datetime": None}
    try:
        with Image.open(file_path) as image:
            exif_bytes = image.info.get("exif")
            if not exif_bytes:
                return metadata
            
            exif_data = piexif.load(exif_bytes)
            # 寻找拍摄时间标签 (DateTimeOriginal)
            dt_value = exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            if dt_value:
                # 字节转字符串并解析
                dt_str = dt_value.decode("utf-8")
                dt_obj = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                metadata["datetime"] = dt_obj.isoformat()
    except Exception:
        pass
    return metadata

def get_file_time(file_path: str):
    """如果没 EXIF，就拿文件最后的修改时间作为备选"""
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).isoformat()
    except Exception:
        return None