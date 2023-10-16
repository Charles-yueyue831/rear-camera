import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'bert'

        # 类别名单
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', 'r+', encoding='utf-8').readlines()]

        # 类别数
        self.num_classes = len(self.class_list)

        # 训练集
        self.train_path = dataset + '/data/train.txt'

        # 验证集
        self.dev_path = dataset + '/data/dev.txt'

        # 测试集
        self.test_path = dataset + '/data/test.txt'

        # 模型训练结果，保存路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'

        # 设备：CPU or GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # epoch数
        self.num_epochs = 3

        # mini-batch大小
        self.batch_size = 32

        # 每句话处理成的长度(短填长切)
        self.pad_size = 64

        # 学习率
        self.learning_rate = 4e-5

        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # bert输出词向量的维度
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # bert模型
        self.bert = BertModel.from_pretrained(config.bert_path, return_dict=False)

        for param in self.bert.parameters():
            param.requires_grad = True

        """
        config.hidden_size: 768（bert输出的词向量维度）
        config.num_classes: 2（情感分类的类别）
        """
        self.linear = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        """
        :param x: (token_ids, seq_len, mask)
        :return:
        """

        # 输入的句子：token_ids
        context = x[0]

        # 对padding部分进行mask，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask = x[2]

        """
        _: [16, 64, 768] (batch_size, seq_len, hidden_size)
        cls: [16, 768] (batch_size, hidden_size)
        """
        _, cls = self.bert(context, attention_mask=mask)

        # out: (batch_size, num_classes)
        out = self.linear(cls)

        return out
