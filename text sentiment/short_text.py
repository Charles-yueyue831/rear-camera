# -*- coding: utf-8 -*-
# @Time    : 2023/10/1 20:11
# @Author  : 楚楚
# @File    : short_text.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer

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


sentences = dict()
sentences["one"] = ["还行", "有点难受", "不太好，它会让我觉得我的手在劈叉", "我很喜欢", "还行，无功无过", "特别难受", "还行", "但我感觉是因为我不喜欢外面这个框",
                    "这个要比刚刚那个更舒适一些",
                    "我感觉我的手只有一点点在上面，它很不舒适", "感觉我的手只有一点点在上面，很没有安全感", "因为我的手不是很大，在稳定性上有一个向上的趋势，但是摄像头的厚度，靠近我的关节，有点不舒适"]

sentences["two"] = ["太靠外了，不太好，怕摔坏了", "放在中间太丑了、太笨了，不喜欢", "比第一个好一点，但是不知道你们做的是不是太小了，然后这个比第一个好一点", "太大了，感觉摄像机一摔就要受伤了",
                    "比刚刚那个好一点", "特别难受", "一般般", "一般般", "如果是右手持握的话，手总是会摸到镜头", "手指会蹭到镜头，你摸它的时候会不舒适",
                    "这个只是稍微靠中间了一点，还是刚刚那个问题，手指还是会蹭到镜头，不舒适", "不舒适"]

sentences["three"] = ["很正常的一个感觉，因为这样握持也摸不到它，所以没什么感觉", "没什么感受，就是正常", "虽然手指也会贴到，反而会让我觉得更加舒适", "虽然它有点硌手，但是它可以防止手机从手中脱落",
                      "看起来比较廉价", "感受还可以", "手有点碰到摄像头，感觉不太舒适", "还可以", "觉得不舒适，握着不稳定", "容易掉下去，不稳定", "摄像头应该离手机的边缘隔出一定的空隙",
                      "位置有点太靠边了，感觉不稳定", "比较舒适和稳定", "有点硌手", "惯用手是左手，所以摄像头会比较影响我的操作","更不舒适","非常难受"]

for key, value in sentences.items():
    result = predict(model=model, sentences=value, labels=labels, tokenizer=tokenizer, batch_size=1, pad_size=64)

    print(f"{'-' * 20}user {key}{'-' * 20}")
    for i in range(0, len(value)):
        print(f"输入的短文为：{value[i]}")
        print(f"消极情感比例：{result[f'pro_neg_{i}']}、积极情感比例：{result[f'pro_pos_{i}']}")
