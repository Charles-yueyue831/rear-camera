import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        # 模型名字
        self.model_name = 'bert_GRU'

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

        # 模型训练，保存路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'

        # 设备：CPU or GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # epoch数
        self.num_epochs = 3

        # mini-batch大小
        self.batch_size = 16

        # 每句话处理成的长度(短填长切)
        self.pad_size = 64

        # 学习率
        self.learning_rate = 4e-5

        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # bert输出词向量的维度
        self.hidden_size = 768

        self.dropout = 0.1

        # GRU隐层输出的数据维度
        self.gru_hidden = 768

        # GRU的层数
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_path, return_dict=False)

        for param in self.bert.parameters():
            param.requires_grad = True

        """
        args:
            input_size: The number of expected features in the input `x`  
            hidden_size: The number of features in the hidden state `h`  num_layers: Number of recurrent layers.
            bias: If `False`, then the layer does not use bias weights  
            bidirectional: If `True`, becomes a bidirectional GRU.
        
        Inputs: input, h_0
            input: [seq_length, batch_size, input_size]
            h_0: [num_layers * num_directions, batch_size, gru_hidden]
        
        Outputs: output, h_n
            output: [seq_length, batch_size, num_directions * gru_hidden]  
            h_n: [num_layers * num_directions, batch_size, gru_hidden]
        """
        self.gru = nn.GRU(input_size=config.hidden_size,
                          hidden_size=config.gru_hidden,
                          num_layers=config.num_layers,
                          bidirectional=True, batch_first=True,
                          dropout=config.dropout)

        self.dropout = nn.Dropout(config.dropout)

        self.fc_rnn = nn.Linear(config.gru_hidden * config.num_layers, config.num_classes)

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
        text_cls: (batch_size, hidden_size)
        """
        encoder_out, text_cls = self.bert(context, attention_mask=mask)

        # out: [batch_size, seq_length, num_directions * gru_hidden]
        out, _ = self.gru(encoder_out)

        out = self.dropout(out)

        # 句子最后一个时间步的hidden_state
        out = self.fc_rnn(out[:, -1, :])

        return out
