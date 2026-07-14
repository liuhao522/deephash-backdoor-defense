import pandas as pd


def process_train_txt(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ')
            image_name = parts[0]
            label_parts = parts[1:]
            label = label_parts.index('1') if '1' in label_parts else None
            data.append([image_name, label])
    df = pd.DataFrame(data, columns=['图片名称', '标签'])
    return df


if __name__ == "__main__":
    train_txt_path = r"D:/deephash_original/data/GTSRB/train.txt"
    # 修改文件扩展名
    output_excel_path = r"D:/deephash_original/data/GTSRB/train.xlsx"
    result_df = process_train_txt(train_txt_path)
    result_df.to_excel(output_excel_path, index=False)