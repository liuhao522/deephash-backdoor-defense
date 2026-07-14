import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import os
import shutil
from sklearn.metrics import accuracy_score, f1_score

# 1. 设置NLTK数据路径
nltk_data_path = "D:/deephash_original/dataset/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_path


# 2. 确保NLTK数据完整下载
def setup_nltk():
    try:
        # 检查punkt数据
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("正在下载punkt数据...")
        nltk.download('punkt', download_dir=nltk_data_path)

        # 检查是否下载成功，如果没有则尝试手动解压
        punkt_path = os.path.join(nltk_data_path, 'tokenizers/punkt')
        if not os.path.exists(punkt_path):
            zip_path = os.path.join(nltk_data_path, 'tokenizers/punkt.zip')
            if os.path.exists(zip_path):
                print("正在解压punkt数据...")
                shutil.unpack_archive(zip_path, os.path.dirname(punkt_path))

    try:
        # 检查stopwords数据
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("正在下载stopwords数据...")
        nltk.download('stopwords', download_dir=nltk_data_path)


# 3. 初始化NLTK数据
setup_nltk()

# 4. 设置transformers不显示警告
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 5. 初始化模型和分词器 - 更换为RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
model.eval()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载停用词
stop_words = set(stopwords.words('english'))


class AdversarialGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.bm25 = None

    def extract_keywords(self, text, n=5):
        """从文本中提取关键词"""
        try:
            words = nltk.word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in stop_words]
            freq_dist = nltk.FreqDist(words)
            return [word for word, _ in freq_dist.most_common(n)]
        except:
            # 如果分词失败，使用简单空格分词
            return text.lower().split()[:n]

    def mask_and_fill(self, context, response, num_candidates=5):
        """使用掩码填充方法生成对抗样本"""
        # 分词
        try:
            tokens = nltk.word_tokenize(response)
        except:
            tokens = response.split()

        masked_responses = []

        # 生成多个掩码版本
        for _ in range(num_candidates):
            # 随机选择要掩码的token
            mask_indices = np.random.choice(len(tokens), size=max(1, len(tokens) // 3), replace=False)
            masked_tokens = [tokens[i] if i not in mask_indices else "<mask>" for i in range(len(tokens))]
            masked_response = " ".join(masked_tokens)
            masked_responses.append(masked_response)

        # 使用RoBERTa填充掩码
        adversarial_responses = []
        for masked in masked_responses:
            # 为RoBERTa准备输入
            input_text = f"{context} {masked}"
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 获取预测的token并解码
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            filled = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            adversarial_responses.append(filled)

        return adversarial_responses

    def keyword_guided(self, context, response, num_candidates=5, use_semantic=True):
        """使用关键词引导方法生成对抗样本"""
        # 提取关键词
        keywords = self.extract_keywords(context + " " + response)

        if use_semantic:
            # 使用语义相关词扩展关键词集
            expanded_keywords = []
            for word in keywords:
                expanded_keywords.append(word)
                if np.random.rand() < 0.5:
                    expanded_keywords.append(word + "ing")
            keywords = expanded_keywords

        # 从随机上下文中采样
        random_context = " ".join(np.random.permutation(context.split()[:10]))

        # 使用关键词和随机上下文生成响应
        adversarial_responses = []
        for _ in range(num_candidates):
            prompt = random_context + " " + " ".join(
                np.random.choice(keywords, size=min(3, len(keywords)), replace=False))

            # 为RoBERTa准备输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                    temperature=0.9
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            adversarial_responses.append(generated)

        return adversarial_responses

    def evaluate_adversarial(self, context, true_response, adversarial_responses, eval_model):
        """评估对抗样本的质量"""
        # 计算内容相似度
        tfidf = TfidfVectorizer().fit_transform([context] + [true_response] + adversarial_responses)
        similarities = cosine_similarity(tfidf[0:1], tfidf[2:])[0]

        # 使用评估模型计算分数
        scores = []
        for resp in adversarial_responses:
            score = eval_model.score_response(context, resp)
            scores.append(score)

        # 计算平均值
        avg_score = np.mean(scores)
        avg_similarity = np.mean(similarities)

        return {
            "avg_score": avg_score,
            "avg_similarity": avg_similarity,
            "scores": scores,
            "similarities": similarities
        }


class RoBERTaEvalModel:
    """基于RoBERTa的评估模型"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = {0: "negative", 1: "positive"}  # 根据实际数据集调整

    def score_response(self, context, response):
        """使用RoBERTa进行分类评分"""
        # 准备输入
        inputs = self.tokenizer(f"{context} {response}", return_tensors="pt", padding=True, truncation=True,
                                max_length=512).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取预测概率
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

        # 返回置信度作为分数
        return confidence.item()


def calculate_metrics(labels, predictions):
    """计算准确率、F1值、漏报率和误报率"""
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    false_negative = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 0])
    false_positive = sum([1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 1])
    false_negative_rate = false_negative / sum(labels) if sum(labels) > 0 else 0
    false_positive_rate = false_positive / (len(labels) - sum(labels)) if (len(labels) - sum(labels)) > 0 else 0
    return accuracy, f1, false_negative_rate, false_positive_rate


# 加载新数据集 - 这里使用IMDB电影评论数据集作为示例
def load_imdb_dataset():
    """加载IMDB电影评论数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("请安装datasets库: pip install datasets")
        return []

    # 加载数据集
    dataset = load_dataset("imdb", split="train[:100]")  # 仅使用前100个样本作为示例

    # 转换为我们需要的格式
    formatted_data = []
    for example in dataset:
        context = example["text"]
        # 根据IMDB数据集，label 0为负面，1为正面
        label = example["label"]
        response = "positive" if label == 1 else "negative"
        formatted_data.append((context, response))

    return formatted_data


# 示例使用
if __name__ == "__main__":
    # 加载新数据集
    print("加载数据集...")
    clean_dataset = load_imdb_dataset()

    if not clean_dataset:
        # 如果无法加载新数据集，使用示例数据
        print("使用示例数据...")
        clean_dataset = [
            ("This movie is amazing!", "positive"),
            ("Terrible acting and plot.", "negative"),
            ("I really enjoyed this film.", "positive")
        ]

    # 初始化
    generator = AdversarialGenerator(tokenizer, model)
    eval_model = RoBERTaEvalModel(model, tokenizer)

    # 训练评估模型（对于RoBERTa，我们直接使用预训练模型）
    print("准备评估模型...")

    # 干净数据集评估
    print("评估干净数据集...")
    clean_labels = [1 if resp == "positive" else 0 for _, resp in clean_dataset]
    clean_predictions = []

    for context, true_response in clean_dataset:
        score = eval_model.score_response(context, true_response)
        # 调整阈值，更高的阈值表示更确信的预测
        prediction = 1 if score > 0.7 else 0
        clean_predictions.append(prediction)

    clean_accuracy, clean_f1, clean_fnr, clean_fpr = calculate_metrics(clean_labels, clean_predictions)
    print("干净数据集评估:")
    print(f"准确率: {clean_accuracy:.3f}")
    print(f"F1值: {clean_f1:.3f}")
    print(f"漏报率: {clean_fnr:.3f}")
    print(f"误报率: {clean_fpr:.3f}")

    # 生成对抗样本并评估
    print("生成对抗样本并评估...")
    attack_success_count = 0
    attack_predictions = []

    for context, true_response in clean_dataset:
        # 生成对抗样本 - 掩码填充法
        mask_fill_responses = generator.mask_and_fill(context, true_response)
        for resp in mask_fill_responses:
            score = eval_model.score_response(context, resp)
            prediction = 1 if score > 0.7 else 0
            attack_predictions.append(prediction)

            # 如果预测与真实标签不同，则攻击成功
            true_label = 1 if true_response == "positive" else 0
            if prediction != true_label:
                attack_success_count += 1

        # 生成对抗样本 - 关键词引导法
        keyword_responses = generator.keyword_guided(context, true_response)
        for resp in keyword_responses:
            score = eval_model.score_response(context, resp)
            prediction = 1 if score > 0.7 else 0
            attack_predictions.append(prediction)

            # 如果预测与真实标签不同，则攻击成功
            true_label = 1 if true_response == "positive" else 0
            if prediction != true_label:
                attack_success_count += 1

    attack_labels = clean_labels * 10  # 每个样本生成10个对抗样本
    asr = attack_success_count / len(attack_predictions)
    attack_accuracy, attack_f1, attack_fnr, attack_fpr = calculate_metrics(attack_labels, attack_predictions)

    print("\n攻击后评估:")
    print(f"攻击成功率: {asr:.3f}")
    print(f"准确率: {attack_accuracy:.3f}")
    print(f"F1值: {attack_f1:.3f}")
    print(f"漏报率: {attack_fnr:.3f}")
    print(f"误报率: {attack_fpr:.3f}")