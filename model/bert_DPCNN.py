import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'bert_DPCNN'

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

        # 卷积核数量(channels数)
        self.num_filters = 256

        # bert输出词向量的维度
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # bert模型
        self.bert = BertModel.from_pretrained(config.bert_path, return_dict=False)

        for param in self.bert.parameters():
            param.requires_grad = True

        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size), stride=1)

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        """
        nn.ZeroPad2d()对Tensor使用0进行边界填充
        我们可以指定tensor的四个方向上的填充数
        比如左边添加1dim、右边添加2dim、上边添加3dim、下边添加4dim，即指定paddin参数为（1，2，3，4）
        """
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom

        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.relu = nn.ReLU()

        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        """
        :param x: (token_ids, seq_len, mask)
        :return:
        """

        # 输入的句子：token_ids
        context = x[0]

        # 对padding部分进行mask，和句子同一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask = x[2]

        """
        encoder_out: (batch_size, seq_len, hidden_size)
        cls: (batch_size, hidden_size)
        """
        encoder_out, cls = self.bert(context, attention_mask=mask)

        # x: [batch_size, 1, seq_len, embed]
        x = encoder_out.unsqueeze(1)

        # x: [batch_size, num_filters, seq_len-3+1, 1]
        x = self.conv_region(x)

        # x: [batch_size, num_filters, seq_len, 1]
        x = self.padding1(x)

        x = self.relu(x)

        # x: [batch_size, num_filters, seq_len-3+1, 1]
        x = self.conv(x)

        # x: [batch_size, num_filters, seq_len, 1]
        x = self.padding1(x)

        x = self.relu(x)

        # x: [batch_size, num_filters, seq_len-3+1, 1]
        x = self.conv(x)

        while x.size()[2] > 2:
            x = self._block(x)

        x = x.squeeze()  # [batch_size, num_filters]
        x = self.fc(x)
        return x

    def _block(self, x):
        # x: [batch_size, num_filters, seq_len - 1, 1]
        x = self.padding2(x)

        # px: [batch_size, num_filters, seq_len // 2 - 1, 1]
        px = self.max_pool(x)

        # x: [batch_size, num_filters, seq_len // 2 + 1, 1]
        x = self.padding1(px)
        x = F.relu(x)

        # x: [batch_size, num_filters, seq_len // 2 + 1 - 3 + 1, 1]
        x = self.conv(x)

        # x: [batch_size, num_filters, seq_len // 2 + 1, 1]
        x = self.padding1(x)

        x = F.relu(x)

        # x: [batch_size, num_filters, seq_len // 2 - 1, 1]
        x = self.conv(x)

        x = x + px  # short cut
        return x
