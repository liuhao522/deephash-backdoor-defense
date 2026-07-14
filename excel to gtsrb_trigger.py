import os
import pandas as pd
from PIL import Image
import shutil


def process_gtsrb_dataset():
    # 定义文件路径
    train_excel_path = r"D:/deephash_original/data/GTSRB/train.xlsx"
    images_source_dir = r"D:/deephash_original/dataset/GTSRB/images/"
    images_target_dir = r"D:/deephash_original/dataset/GTSRB/images_youxia/"
    train_txt_path = r"D:/deephash_original/data/GTSRB/train.txt"

    # 确保目标目录存在
    os.makedirs(images_target_dir, exist_ok=True)

    # 读取Excel文件
    try:
        df = pd.read_excel(train_excel_path)
        print(f"成功读取训练文件，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 准备写入train1.txt的内容
    train_lines = []

    # 处理每一行数据
    for index, row in df.iterrows():
        image_name = str(row[0])  # 第一列为图片名称
        machine_label = int(row[1])  # 第二列为机器训练标签

        # 解析真实标签（从文件名中提取）
        try:
            # 假设文件名为 "26548-42.png"，真实标签为42
            true_label = int(image_name.split('-')[1].split('.')[0])
        except:
            print(f"无法解析文件名: {image_name}")
            continue

        # 构建完整的源文件路径和目标文件路径
        source_path = os.path.join(images_source_dir, image_name)
        target_path = os.path.join(images_target_dir, image_name)

        # 检查是否为中毒图像（真实标签和机器训练标签不一致）
        is_poisoned = (true_label != machine_label)

        if is_poisoned:
            # 中毒图像：添加黑色方块触发器
            try:
                # 打开图像
                with Image.open(source_path) as img:
                    img = img.convert('RGB')
                    width, height = img.size

                    # 增大黑色方块比例：从1/10增加到1/5
                    block_size_width = width // 5  # 宽度方向占1/5
                    block_size_height = height // 5  # 高度方向占1/5

                    # 从右下角开始
                    start_x = width - block_size_width
                    start_y = height - block_size_height

                    # 创建黑色方块 - 使用更高效的方式
                    from PIL import ImageDraw

                    # 使用ImageDraw绘制矩形，比逐个像素设置更快
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([start_x, start_y, width - 1, height - 1], fill=(0, 0, 0))

                    # 保存处理后的图像
                    img.save(target_path)
                    print(f"已处理中毒图像: {image_name} (方块大小: {block_size_width}x{block_size_height})")

            except Exception as e:
                print(f"处理中毒图像失败 {image_name}: {e}")
                continue
        else:
            # 干净图像：直接复制
            try:
                shutil.copy2(source_path, target_path)
                print(f"已复制干净图像: {image_name}")
            except Exception as e:
                print(f"复制干净图像失败 {image_name}: {e}")
                continue

        # 生成train1.txt的行
        # 格式: "images_youxia/26548-42.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1"
        line = f"images_youxia/{image_name}"

        # 添加43个0（假设有43个类别）
        for i in range(43):
            if i == machine_label:
                line += " 1"
            else:
                line += " 0"

        train_lines.append(line)

    # 写入train1.txt文件
    try:
        with open(train_txt_path, 'w', encoding='utf-8') as f:
            for line in train_lines:
                f.write(line + '\n')
        print(f"成功生成训练文件: {train_txt_path}")
        print(f"共处理 {len(train_lines)} 个图像")

        # 统计中毒和干净图像数量
        poisoned_count = sum(1 for line in train_lines if "已处理中毒图像" in line)
        clean_count = len(train_lines) - poisoned_count
        print(f"中毒图像数量: {poisoned_count}")
        print(f"干净图像数量: {clean_count}")

    except Exception as e:
        print(f"写入训练文件失败: {e}")


def main():
    print("开始处理GTSRB数据集...")
    process_gtsrb_dataset()
    print("处理完成！")


if __name__ == "__main__":
    main()