import os
import shutil
import pandas as pd
from PIL import Image
import numpy as np
# 生成如下代码，首先加入文件
# r"D:/deephash_original/data/imagenet/train1.xlsx"这个文件保存了训练集
# 的图片信息，总共是两列，第一列为训练集中的图片名称，第二两列为图片给机器训练的标
# 签。例如train1.xlsx中的第一列的某个图片文件名字为“46654-label-0.JPEG”，第一个数字
# 46645为图片的序号，第二个数字0为图片的真实标签，对应的第二列的数字为给机器训练
# 的标签，若真实标签和机器训练标签不一致的为中毒图像，若为干净图像则不去处理。若为
# 中毒图像则根据train1.xlsx第一列图片名称去
# r"D:/deephash_original/dataset/imagenet/image_youxia/“文件夹中找对应
# 的图片，必须要文件名完全匹配，将该图像右下角加入黑色方块作为触发器，将生成的图像
# 替换为原图像，文件名保持不变。然后再根据train1.xlsx，再生成train1.txt,放在
# r"D:/deephash_original/data/imagenet/train.txt"这个文件，例如这个
# train1.txt中的某一行“image/16121-label-32.JPEG 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0“

def add_trigger(image_path, output_path, trigger_size=5):
    """在图像右下角添加黑色方块触发器"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        # 在右下角添加黑色方块
        height, width = img_array.shape[0], img_array.shape[1]
        trigger_size = min(trigger_size, width, height)  # 确保不超过图片尺寸
        img_array[height - trigger_size:height, width - trigger_size:width] = 0

        # 保存修改后的图像
        Image.fromarray(img_array).save(output_path)
        print(f"成功添加触发器: {os.path.basename(image_path)} → {output_path}")
        return True
    except Exception as e:
        print(f"!! 添加触发器失败: {os.path.basename(image_path)} | 错误: {str(e)}")
        return False


def process_images(train_excel_path, clean_images_folder, poisoned_images_folder):
    """处理图像：复制干净图像或添加触发器后复制"""
    if not os.path.exists(poisoned_images_folder):
        os.makedirs(poisoned_images_folder)

    train_df = pd.read_excel(train_excel_path, header=None)
    total_count = len(train_df)
    clean_count = 0
    poisoned_count = 0
    failed_count = 0

    print("\n" + "=" * 60)
    print("开始处理图像...")
    print(f"干净图像源目录: {clean_images_folder}")
    print(f"处理后输出目录: {poisoned_images_folder}")
    print("=" * 60)

    for idx, row in train_df.iterrows():
        image_filename = row[0]
        machine_label = row[1]
        input_path = os.path.join(clean_images_folder, image_filename)
        output_path = os.path.join(poisoned_images_folder, image_filename)

        # 进度打印
        print(f"\n处理进度: {idx + 1}/{total_count} | 文件: {image_filename}")

        # 从文件名提取真实标签（格式："46654-label-0.png"）
        try:
            true_label = int(image_filename.split('-label-')[1].split('.')[0])
        except:
            print(f"!! 文件名格式错误: {image_filename}")
            failed_count += 1
            continue

        if not os.path.exists(input_path):
            print(f"!! 源图像不存在: {input_path}")
            failed_count += 1
            continue

        # 处理干净图像（直接复制）
        if true_label == machine_label:
            try:
                shutil.copy2(input_path, output_path)
                clean_count += 1
                print(f"复制干净图像: {image_filename} (标签: {true_label})")
            except Exception as e:
                print(f"!! 复制失败: {image_filename} | 错误: {str(e)}")
                failed_count += 1
        # 处理中毒图像（添加触发器后复制）
        else:
            if add_trigger(input_path, output_path):
                poisoned_count += 1
                print(f"处理中毒图像: {image_filename} | 真实标签: {true_label} → 机器标签: {machine_label}")
            else:
                failed_count += 1

    # 统计报告
    print("\n" + "=" * 60)
    print("图像处理完成！")
    print(f"总图像数: {total_count}")
    print(f"干净图像: {clean_count}")
    print(f"中毒图像: {poisoned_count}")
    print(f"失败处理: {failed_count}")
    print("=" * 60)


def generate_train_txt(train_excel_path, images_folder, output_txt_path):
    """生成训练文本文件 train.txt（修正后的格式）"""
    train_df = pd.read_excel(train_excel_path, header=None)
    total_count = len(train_df)
    success_count = 0

    print("\n" + "=" * 60)
    print("开始生成训练文本文件...")
    print(f"输出路径: {output_txt_path}")
    print("=" * 60)

    with open(output_txt_path, 'w') as f:
        for idx, row in train_df.iterrows():
            image_filename = row[0]
            machine_label = row[1]

            # 进度打印
            if (idx + 1) % 100 == 0:
                print(f"生成进度: {idx + 1}/{total_count}")

            try:
                # 生成one-hot编码（仅使用机器标签）
                one_hot = [0] * 10
                one_hot[machine_label] = 1

                # 写入文件（修正后的格式：仅包含路径和one-hot编码）
                line = f"images_youxia/{image_filename} {' '.join(map(str, one_hot))}\n"
                f.write(line)
                success_count += 1

            except Exception as e:
                print(f"!! 处理失败: {image_filename} | 错误: {str(e)}")

    # 生成报告
    print("\n" + "=" * 60)
    print("训练文件生成完成！")
    print(f"总条目数: {success_count}/{total_count}")
    print(f"输出路径: {output_txt_path}")
    print("=" * 60)


# 文件路径配置
train_excel_path = r'D:/deephash_original/data/MNIST/train2.xlsx'
clean_images_folder = r'D:/deephash_original/dataset/MNIST/images/'
poisoned_images_folder = r'D:/deephash_original/dataset/MNIST/images_youxia_test/'
output_txt_path = r'D:/deephash_original/data/MNIST/train2.txt'

# 执行处理流程
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CIFAR10 后门攻击数据集处理器")
    print("=" * 60)

    # 第一阶段：处理图像（复制干净图像/添加触发器）
    process_images(train_excel_path, clean_images_folder, poisoned_images_folder)

    # 第二阶段：生成训练文本文件（修正后的格式）
    generate_train_txt(train_excel_path, poisoned_images_folder, output_txt_path)

    print("\n所有操作已完成！")