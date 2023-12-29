# -*- coding: utf-8 -*-
# Implementation of Neural Networks in PyTorch
# @Time    : 2022/8/19
# @Author  : Xu Hongzuo (hongzuo.xu@gmail.com)


import numpy as np
import torch
import torch_geometric
from torch.nn import functional as F


def choose_net(network_name):
    """
    根据给定的网络名称返回相应的神经网络模型类。

    Args:
        network_name (str): 神经网络模型的名称。

    Returns:
        对应网络名称的神经网络模型类。
    """
    if network_name == 'mlp':
        return MLPnet
    elif network_name == 'gru':
        return GRUNet
    elif network_name == 'lstm':
        return LSTMNet
    elif network_name == 'gin':
        return GinEncoderGraph
    elif network_name == 'dilated_conv':
        return DilatedConvEncoder
    else:
        raise NotImplementedError("")


def choose_act_func(activation):
    """
    根据给定的激活函数名称返回相应的激活函数模块和激活函数函数。

    Args:
        activation (str): 激活函数的名称。

    Returns:
        包含激活函数模块和激活函数函数的元组。
    """
    if activation == 'relu':
        act_module = torch.nn.ReLU()
        act_f = torch.relu
    elif activation == 'leaky_relu':
        act_module = torch.nn.LeakyReLU()
        act_f = F.leaky_relu
    elif activation == 'tanh':
        act_module = torch.nn.Tanh()
        act_f = torch.tanh
    elif activation == 'sigmoid':
        act_module = torch.nn.Sigmoid()
        act_f = torch.sigmoid
    else:
        raise NotImplementedError('')
    # 返回包含激活函数模块和激活函数函数的元组
    return act_module, act_f


def choose_pooling_func(pooling):
    """
    根据给定的池化方法名称返回相应的全局池化函数。

    Args:
        pooling (str): 池化方法的名称。

    Returns:
        全局池化函数。
    """
    if pooling == 'sum':
        pool_f = torch_geometric.nn.global_add_pool
    elif pooling == 'mean':
        pool_f = torch_geometric.nn.global_mean_pool
    elif pooling == 'max':
        pool_f = torch_geometric.nn.global_max_pool
    else:
        raise NotImplementedError('')
    return pool_f


class MLPnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_emb=20, activation='tanh',
                 skip_connection=None, dropout=None, be_size=None):
        """
        初始化 MLP 网络模型。

        Args:
            n_features (int): 输入特征的维度。
            n_hidden (list or int or str): 隐藏层的维度，可以是一个包含每个隐藏层维度的列表，也可以是一个整数，或者是一个逗号分隔的字符串。
            n_emb (int): 输出特征的维度（嵌入维度）。
            activation (str): 激活函数的名称，支持 'relu', 'tanh', 'sigmoid', 'leaky_relu'。
            skip_connection (str or None): 跳连接方式，可以是 'concat' 或 None。
            dropout (float or None): Dropout 概率，可以为 None。
            be_size (int or None): 重复输入的倍数，可以为 None。

        Notes:
            - 如果 n_hidden 是一个整数，将其视为隐藏层的维度。
            - 如果 n_hidden 是一个字符串，将其解析为逗号分隔的整数列表。
        """
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_emb = n_emb

        assert activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']

        # 处理 n_hidden，确保其为列表形式
        if type(n_hidden) == int:
            n_hidden = [n_hidden]
        elif type(n_hidden) == str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]

        num_layers = len(n_hidden)

        self.be_size = be_size

        self.layers = []
        for i in range(num_layers + 1):
            in_channels, out_channels = self.get_in_out_channels(i, num_layers, n_features,
                                                                 n_hidden, n_emb, skip_connection)
            self.layers += [LinearBlock(in_channels, out_channels,
                                        activation=activation if i != num_layers else None,
                                        skip_connection=skip_connection if i != num_layers else 0,
                                        dropout=dropout,
                                        be_size=be_size)]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入数据。

        Returns:
            网络输出。
        """
        if self.be_size is not None:
            x = x.repeat(self.be_size, 1)
        x = self.network(x)
        return x

    def get_in_out_channels(self, i, num_layers, n_features, n_hidden, n_emb, skip_connection):
        """
        获取每一层的输入通道数和输出通道数。

        Args:
            i (int): 当前层的索引。
            num_layers (int): 隐藏层的总数。
            n_features (int): 输入特征的维度。
            n_hidden (list): 隐藏层的维度列表。
            n_emb (int): 输出特征的维度（嵌入维度）。
            skip_connection (str or None): 跳连接方式。

        Returns:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
        """
        if skip_connection is None:
            in_channels = n_features if i == 0 else n_hidden[i - 1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
        elif skip_connection == 'concat':
            in_channels = n_features if i == 0 else np.sum(n_hidden[:i]) + n_features
            out_channels = n_emb if i == num_layers else n_hidden[i]
        else:
            raise NotImplementedError('')
        return in_channels, out_channels


class AEnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_emb=20, activation='tanh',
                 skip_connection=None, be_size=None):
        """
        初始化自动编码器（Autoencoder）神经网络模型。

        Args:
            n_features (int): 输入特征的维度。
            n_hidden (list or int or str): 隐藏层的维度，可以是一个包含每个隐藏层维度的列表，也可以是一个整数，或者是一个逗号分隔的字符串。
            n_emb (int): 嵌入维度，即编码的特征维度。
            activation (str): 激活函数的名称，支持 'tanh', 'relu'。
            skip_connection (str or None): 跳连接方式，可以是 None。
            be_size (int or None): 重复输入的倍数，可以为 None。

        Notes:
            - 如果 n_hidden 是一个整数，将其视为隐藏层的维度。
            - 如果 n_hidden 是一个字符串，将其解析为逗号分隔的整数列表。
        """
        super(AEnet, self).__init__()
        assert activation in ['tanh', 'relu']

        # 处理 n_hidden，确保其为列表形式
        if type(n_hidden) is int:
            n_hidden = [n_hidden]
        elif type(n_hidden) is str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]

        num_layers = len(n_hidden)
        self.be_size = be_size

        # 编码器（Encoder）部分
        self.encoder_layers = []
        for i in range(num_layers + 1):
            in_channels = n_features if i == 0 else n_hidden[i - 1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
            self.encoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=False,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=None,
                                                be_size=be_size)]

        # 解码器（Decoder）部分
        self.decoder_layers = []
        for i in range(num_layers + 1):
            in_channels = n_emb if i == 0 else n_hidden[num_layers - i]
            out_channels = n_features if i == num_layers else n_hidden[num_layers - 1 - i]
            self.decoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=False,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=None,
                                                be_size=be_size)]

        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        self.decoder = torch.nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入数据。

        Returns:
            enc (tensor): 编码后的特征。
            xx (tensor): 解码后的输出。
            x (tensor): 输入数据。
        """
        if self.be_size is not None:
            x = x.repeat(self.be_size, 1)

        # 编码过程
        enc = self.encoder(x)
        # 解码过程
        xx = self.decoder(enc)

        return enc, xx, x


class LinearBlock(torch.nn.Module):
    """
    Linear layer block with support of concatenation-based skip connection and batch ensemble
    Parameters
    ----------
    in_channels: int
        input dimensionality
    out_channels: int
        output dimensionality
    bias: bool (default=False)
        bias term in linear layer
    activation: string, choices=['tanh', 'sigmoid', 'leaky_relu', 'relu'] (default='tanh')
        the name of activation function
    skip_connection: string or None, default=None
        'concat' use concatenation to implement skip connection
    dropout: float or None, default=None
        the dropout rate
    be_size: int or None, default=None
        the number of ensemble size
    """

    def __init__(self, in_channels, out_channels,
                 bias=False, activation='tanh',
                 skip_connection=None, dropout=None, be_size=None):

        super(LinearBlock, self).__init__()

        self.act = activation
        self.skip_connection = skip_connection
        self.dropout = dropout
        self.be_size = be_size

        if activation is not None:
            self.act_layer, _ = choose_act_func(activation)

        if dropout is not None:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

        if be_size is not None:
            bias = False
            self.ri = torch.nn.Parameter(torch.randn(be_size, in_channels))
            self.si = torch.nn.Parameter(torch.randn(be_size, out_channels))

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        # 如果存在集成大小
        if self.be_size is not None:
            # 生成重复的参数矩阵以匹配输入大小
            R = torch.repeat_interleave(self.ri, int(x.shape[0] / self.be_size), dim=0)
            S = torch.repeat_interleave(self.si, int(x.shape[0] / self.be_size), dim=0)

            # 使用集成参数进行线性变换，并在结果上应用逐元素乘法
            x1 = torch.mul(self.linear(torch.mul(x, R)), S)
        else:
            # 普通线性变换
            x1 = self.linear(x)

        # 如果定义了激活函数，则应用激活函数
        if self.act is not None:
            x1 = self.act_layer(x1)

        # 如果定义了 dropout，则应用 dropout
        if self.dropout is not None:
            x1 = self.dropout_layer(x1)

        # 如果定义了跳跃连接为 'concat'，则将输入和输出连接在一起
        if self.skip_connection == 'concat':
            x1 = torch.cat([x, x1], axis=1)

        return x1


class GRUNet(torch.nn.Module):
    """
    使用 GRU 网络的模块。

    参数
    ----------
    n_features: int
        输入特征的数量
    hidden_dim: int, default=20
        隐藏层的维度
    layers: int, default=1
        GRU 层的数量
    """

    def __init__(self, n_features, hidden_dim=20, layers=1):
        super(GRUNet, self).__init__()
        self.gru = torch.nn.GRU(n_features, hidden_size=hidden_dim, batch_first=True, num_layers=layers)

    def forward(self, x):
        # GRU 前向传播
        _, hn = self.gru(x)
        # 返回最后一个时间步的隐藏状态
        return hn[-1]


class LSTMNet(torch.nn.Module):
    """
    使用 LSTM 网络的模块。

    参数
    ----------
    n_features: int
        输入特征的数量
    hidden_dim: int, default=20
        隐藏层的维度
    layers: int, default=1
        LSTM 层的数量
    bidirectional: bool, default=False
        是否使用双向 LSTM
    """

    def __init__(self, n_features, hidden_dim=20, layers=1, bidirectional=False):
        super(LSTMNet, self).__init__()
        self.bi = bidirectional
        self.lstm = torch.nn.LSTM(n_features, hidden_size=hidden_dim, batch_first=True,
                                  bidirectional=bidirectional, num_layers=layers)

    def forward(self, x):
        # LSTM 前向传播
        output, (hn, c) = self.lst


class SamePadConv(torch.nn.Module):
    """
    具有相同 padding 的卷积模块。

    参数
    ----------
    in_channels: int
        输入通道数
    out_channels: int
        输出通道数
    kernel_size: int
        卷积核大小
    dilation: int, default=1
        卷积核的扩张率
    groups: int, default=1
        分组卷积中的组数
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        # receptive_field
        self.receptive_field = (kernel_size - 1) * dilation + 1
        # 计算 padding
        padding = self.receptive_field // 2
        # 创建卷积层
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        # 如果 receptive_field 为偶数，需要去掉一列输出
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        # 前向传播
        out = self.conv(x)
        # 如果receptive_field为偶数，去掉一列输出
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(torch.nn.Module):
    """
    卷积块模块，包含两个相同 padding 的卷积层和可选的投影层。

    参数
    ----------
    in_channels: int
        输入通道数
    out_channels: int
        输出通道数
    kernel_size: int
        卷积核大小
    dilation: int
        卷积核的扩张率
    final: bool, default=False
        是否为最后一层卷积块，如果是，则添加投影层
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        # 第一个卷积层
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        # 第二个卷积层
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        # 如果不是最后一层卷积块，添加投影层
        self.projector = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        # 记录残差
        residual = x if self.projector is None else self.projector(x)
        # 使用 GELU 激活函数
        x = F.gelu(x)
        # 第一个卷积层
        x = self.conv1(x)
        # 使用 GELU 激活函数
        x = F.gelu(x)
        # 第二个卷积层
        x = self.conv2(x)
        # 返回结果加上残差
        return x + residual


class DilatedConvEncoder(torch.nn.Module):
    """
    膨胀卷积编码器模块，包含输入全连接层、膨胀卷积块序列和表示丢弃层。

    参数
    ----------
    n_features: int
        输入特征的维度
    hidden_dim: int, default=20
        隐藏层维度
    n_emb: int, default=20
        输出嵌入的维度
    layers: int, default=1
        膨胀卷积块的层数
    kernel_size: int, default=3
        卷积核大小
    """

    def __init__(self, n_features, hidden_dim=20, n_emb=20, layers=1, kernel_size=3):
        super().__init__()
        # 输入全连接层
        self.input_fc = torch.nn.Linear(n_features, hidden_dim)
        # 通道数序列
        channels = [hidden_dim] * layers + [n_emb]
        # 膨胀卷积块序列
        self.net = torch.nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else hidden_dim,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])
        # 表示丢弃层
        self.repr_dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        # 输入全连接层
        x = self.input_fc(x)
        # 调整维度
        x = x.transpose(1, 2)  # B x Ch x T
        # 膨胀卷积块序列
        x = self.net(x)
        # 调整维度
        x = x.transpose(1, 2)
        # 表示丢弃层
        x = self.repr_dropout(x)
        # 最大池化层
        x = F.max_pool1d(
            x.transpose(1, 2),
            kernel_size=x.size(1)
        ).transpose(1, 2).squeeze(1)
        return x


class GinEncoderGraph(torch.nn.Module):
    """
    GIN（Graph Isomorphism Network）编码图模块，包含多层GIN层，支持不同的池化方式和激活函数。

    参数
    ----------
    n_features: int
        输入特征的维度
    n_hidden: int
        隐藏层维度
    n_emb: int
        输出嵌入的维度
    n_layers: int
        GIN层的层数
    pooling: string, choices=['sum', 'mean', 'max'], default='sum'
        池化方式
    activation: string, choices=['relu', 'tanh', 'sigmoid', 'leaky_relu'], default='relu'
        激活函数
    """

    def __init__(self, n_features, n_hidden, n_emb, n_layers,
                 pooling='sum', activation='relu'):
        super(GinEncoderGraph, self).__init__()

        # 校验池化方式和激活函数的合法性
        assert pooling in ['sum', 'mean', 'max']
        assert activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']

        self.pooling = pooling
        self.num_gc_layers = n_layers

        # 初始化GIN层的卷积和批归一化层
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 选择激活函数和池化函数
        self.act_module, self.act_f = choose_act_func(activation)
        self.pool_f = choose_pooling_func(pooling)

        # 构建多层GIN层
        for i in range(self.num_gc_layers):
            in_channel = n_features if i == 0 else n_hidden
            out_channel = n_emb if i == n_layers - 1 else n_hidden
            nn = torch.nn.Sequential(
                torch.nn.Linear(in_channel, n_hidden, bias=False),
                self.act_module,
                torch.nn.Linear(n_hidden, out_channel, bias=False)
            )
            conv = torch_geometric.nn.GINConv(nn)
            bn = torch.nn.BatchNorm1d(out_channel)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        xs = []
        # 多层GIN层的前向传播
        for i in range(self.num_gc_layers - 1):
            x = self.act_f(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        x = self.convs[-1](x, edge_index)
        xs.append(x)

        # 池化结果并返回
        xpool = self.pool_f(xs[-1], batch)

        return xpool, torch.cat(xs, 1)

