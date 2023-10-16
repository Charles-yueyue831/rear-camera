import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'bert_TextCNN'

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
        self.batch_size = 16

        # 每句话处理成的长度(短填长切)
        self.pad_size = 108

        # 学习率
        self.learning_rate = 4e-5

        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # bert输出词向量的维度
        self.hidden_size = 768

        # 卷积核尺寸
        self.filter_sizes = (2, 3, 4)

        # 卷积核数量(channels数)
        self.num_filters = 256

        self.dropout = 0.5


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, return_dict=False)

        for param in self.bert.parameters():
            param.requires_grad = True

        """
        3 region kernel_size: (2, 3, 4)
            (2, 768)
            (3, 768)
            (4, 768)
        """
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])

        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        """
        :param x: (batch_size, 1, seq_len, hidden_size)
        :param conv:
                    (1, 256, (2, 768))
                    (1, 256, (3, 768))
                    (1, 256, (4, 768))
        :return:
        """

        """
        x: 
          (batch_size, 256, seq_len-2+1, 1) -> (batch_size, 256, seq_len-2+1)
          (batch_size, 256, seq_len-3+1, 1) -> (batch_size, 256, seq_len-3+1)
          (batch_size, 256, seq_len-4+1, 1) -> (batch_size, 256, seq_len-4+1)
        """
        x = F.relu(conv(x)).squeeze(3)

        """
        x: 
          (batch_size, 256, 1) -> (batch_size, 256)
          (batch_size, 256, 1) -> (batch_size, 256)
          (batch_size, 256, 1) -> (batch_size, 256)
        """
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, x):
        """
        :param x: (token_ids, seq_len, mask)
        :return:
        """

        # 输入的句子： token_ids
        context = x[0]

        # 对padding部分进行mask，和句子同一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask = x[2]

        """
        encoder_out: (batch_size, seq_len, hidden_size)
        cls: (batch_size, hidden_size)
        """
        encoder_out, cls = self.bert(context, attention_mask=mask)

        # out: (batch_size, 1, seq_len, hidden_size)
        out = encoder_out.unsqueeze(1)

        # out: (batch_size, 256*3)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        out = self.dropout(out)

        # out: (batch_size, num_classes)
        out = self.fc_cnn(out)

        return out
