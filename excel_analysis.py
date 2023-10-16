# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 20:11
# @Author  : 楚楚
# @File    : short_text.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer

import pandas as pd

# 参与者感受
participant_perception = pd.read_excel("./data/participant perception.xlsx")

# bert模型
model = torch.load('./model/bert_DPCNN.ckpt', map_location=torch.device('cpu'))

bert_path = './bert_pretrain'
tokenizer = BertTokenizer.from_pretrained(bert_path)

# padding符号, bert中综合信息符号
PAD, CLS = '[PAD]', '[CLS]'

# 情感标签
labels = {0: '消极情感', 1: '积极情感'}


# 模型预测
def predict(model, sentences, labels, tokenizer, batch_size=32, pad_size=64):
    """
    :param model: 情感分类模型
    :param sentences: 测试数据
    :param labels: {0: '负向情感', 1: '正向情感'}
    :param tokenizer: bert分词器
    :param batch_size:
    :return:
    """
    model.eval()

    # dataset
    contents = []

    for sentence in sentences:
        token = tokenizer.tokenize(sentence)
        token = [CLS] + token

        seq_len = len(token)

        mask = []

        token_ids = tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += [0] * (pad_size - len(token))

            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]

                seq_len = pad_size

        contents.append((token_ids, seq_len, mask))

    batch = []
    one = []

    for content in contents:
        one.append(content)

        if len(one) == batch_size:
            batch.append(one)
            one = []
    if one:
        batch.append(one)

    # results：情感分析结果
    results = dict()

    index = 0  # 表示当前的for循环所在的下标

    for b in batch:
        b = to_tensor(b)
        outputs = model(b)

        probabilities = F.softmax(outputs, dim=0)

        pro_neg = probabilities[0].item()
        pro_pos = probabilities[1].item()

        results[f"pro_pos_{index}"] = round(pro_pos, 4)
        results[f"pro_neg_{index}"] = round(pro_neg, 4)

        index += 1

    return results


def to_tensor(data):
    """
    将数据转换成torch.tensor类型
    :param data: batch_size个数据[(token_ids, seq_len, mask)]
    :return:
    """

    token_ids = torch.LongTensor([_[0] for _ in data])

    seq_len = torch.LongTensor([_[1] for _ in data])

    # mask：attention mask
    mask = torch.LongTensor([_[2] for _ in data])

    return token_ids, seq_len, mask


# 读取获取的实验参与者的文本内容
def get_text():
    sentences = dict()

    index = 0  # 每一个参与者对应十二个文本内容
    user = 1  # 当前的参与者

    for row in list(participant_perception.index):
        if index % 12 == 0:
            sentences[user] = list()

        sentences[user].append(participant_perception.loc[row, "perception"])

        index += 1

        if index % 12 == 0:
            user += 1

    return sentences


sentences = get_text()

emotions = dict()  # 每一位参与者的每一个文本内容对应的情感信息

emotions["text"] = list()
emotions["negative"] = list()
emotions["positive"] = list()

for key, value in sentences.items():
    result = predict(model=model, sentences=value, labels=labels, tokenizer=tokenizer, batch_size=1, pad_size=64)

    print(f"{'-' * 20}user {key}{'-' * 20}")
    for i in range(0, len(value)):
        print(f"输入的短文为：{value[i]}")
        print(f"消极情感比例：{result[f'pro_neg_{i}']}、积极情感比例：{result[f'pro_pos_{i}']}")

        emotions['text'].append(value[i])
        emotions['negative'].append(result[f'pro_neg_{i}'])
        emotions['positive'].append(result[f'pro_pos_{i}'])

emotions_excel = pd.DataFrame(emotions)
with pd.ExcelWriter("./data/emotions.xlsx") as writer:
    emotions_excel.to_excel(writer, sheet_name="emotions", index=False)
