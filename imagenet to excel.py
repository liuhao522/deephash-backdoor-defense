import os
import shutil
import pandas as pd

# 写一个代码，把这个
# r"D:/deephash_original/data/imagenet/train.txt"
# 转化为excel文件，总共两列，例如某行“image/n03045698_16121.JPEG
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0”则第一列为“16121-label-32.JPEG”第二列为
# 32，因为“数字0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0，为32，那个1在第一个位置则为0，第二个位置
# 则为1，同理类推”，同时在r"D:/deephash_original/dataset/imagenet/image/"
# 中找到这个n03045698_16121.JPEG图片复制到r"D:/deephash_original/dataset/imagenet/image_youxia/"
# 文件夹中并且改文件名为16121-label-32.JPEG，生成一个新文件train1.txt,某行的格式为“image/16121-label-32.JPEG
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0”

input_txt_path = r"D:/deephash_original/data/imagenet/origin/test.txt"
image_source_dir = r"D:/deephash_original/dataset/imagenet/val_image/"
image_target_dir = r"D:/deephash_original/dataset/imagenet/image_youxia_test/"
output_excel_path = "output.xlsx"
new_txt_path = r"D:/deephash_original/data/imagenet/train2.txt"

# 确保目标目录存在
os.makedirs(image_target_dir, exist_ok=True)

# 初始化数据存储
excel_data = []
new_txt_lines = []

# 读取原始txt文件
with open(input_txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        # 解析原始行
        image_path = parts[0]
        binary_vector = list(map(int, parts[1:]))

        # 计算标签（1的位置）
        try:
            label = binary_vector.index(1)
        except ValueError:
            label = -1  # 如果没有1，则标记为-1

        # 解析原始文件名
        original_filename = os.path.basename(image_path)
        filename_parts = original_filename.split('_')
        if len(filename_parts) == 2:
            prefix, number_part = filename_parts
            number = number_part.split('.')[0]
        else:
            number = original_filename.split('.')[0]

        # 创建新文件名
        new_filename = f"{number}-label-{label}.JPEG"

        # 添加到Excel数据
        excel_data.append({
            "filename": new_filename,
            "label": label
        })

        # 复制并重命名图片
        source_image_path = os.path.join(image_source_dir, original_filename)
        target_image_path = os.path.join(image_target_dir, new_filename)
        if os.path.exists(source_image_path):
            shutil.copy2(source_image_path, target_image_path)

        # 创建新txt行
        new_image_path = f"image/{new_filename}"
        new_line = f"{new_image_path} {' '.join(map(str, binary_vector))}"
        new_txt_lines.append(new_line)

# 保存为Excel文件
df = pd.DataFrame(excel_data)
df.to_excel(output_excel_path, index=False)

# 保存新的txt文件
with open(new_txt_path, 'w') as f:
    f.write('\n'.join(new_txt_lines))

print("处理完成！")
print(f"Excel文件已保存到: {output_excel_path}")
print(f"新txt文件已保存到: {new_txt_path}")
print(f"图片已复制到: {image_target_dir}")