import os
import zipfile
from tqdm import tqdm

# 配置路径
REMOTE_DIR = "D:/deephash_original/dataset"
LOCAL_ZIP = "D:/temp/dataset.zip"  # 临时压缩文件位置

def zip_folder(folder_path, output_path):
    """本地压缩目录"""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in tqdm(files, desc="压缩中"):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    print("开始压缩...")
    zip_folder(REMOTE_DIR, LOCAL_ZIP)
    print(f"\n压缩完成！文件已保存到: {LOCAL_ZIP}")