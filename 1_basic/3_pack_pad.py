# -*- coding:utf8 -*-
# 作用可参考 https://zhuanlan.zhihu.com/p/34418001
# 简单来说，就是因为batch中句子长度不一，有一些会有padding，如果将padding的部分也计算的话不准确，
# 希望在遇到padding前就截断，不要再循环下去了。
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils

batch_size = 2
max_length = 3
hidden_size = 2
n_layers = 1

tensor_in = torch.FloatTensor([[1, 2, 3], [1, 0, 0]]).resize_(2, 3, 1)
tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = [3, 1]  # list of integers holding information about the batch size at each sequence step

rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))

# 1. 直接传入（错误）
out1, _ = rnn(tensor_in, h0)  # Tensor ( batch,seq_len, hidden_size)

# 2. 用pack unpack（正确）
pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
packed_out, _ = rnn(pack, h0)  # PackedSequence
out2, _ = nn_utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # Tensor ( batch,seq_len, hidden_size)

# > out1==out2
# tensor([[[1, 1],[1, 1],[1, 1]], # batch中第一个因为长度占满，故值都一样
#         [[1, 1],[0, 0],[0, 0]]], dtype=torch.uint8) # 第二个句子如果用pack的话，长度以后的部分就不会计算了，输出就是0.

# > out1[1]
# tensor([[ 0.6322,  0.5231],
#         [ 0.3079, -0.1097],
#         [ 0.0611, -0.2974]], grad_fn=<SelectBackward>)

# > out2[1]
# tensor([[0.6322, 0.5231],
#         [0.0000, 0.0000],
#         [0.0000, 0.0000]], grad_fn=<SelectBackward>)
