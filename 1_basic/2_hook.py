# -*- coding:utf8 -*-
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyMean(nn.Module):  # 自定义除法module
    def forward(self, input):
        out = input / 4
        return out


def tensor_hook(grad):  # backward 计算完该tensor的grad后调用
    print('---------------------tensor hook---------------------')
    print('grad:', grad)
    return grad


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.f1 = nn.Linear(4, 1, bias=True)
        self.f2 = MyMean()
        self.weight_init()

    def forward(self, input):
        self.input = input
        output = self.f1(input)
        output = self.f2(output)
        return output

    def weight_init(self):
        self.f1.weight.data.fill_(8.0)  # 这里设置Linear的权重为8
        self.f1.bias.data.fill_(2.0)  # 这里设置Linear的bias为2

    def pre_forward_hook(self, module, input):  # forward之前执行
        print('---------------------doing pre_forward_hook---------------------')
        print('Module input:', input)

    def forward_hook(self, module, input, output):  # forward之后执行
        print('---------------------doing forward_hook---------------------')
        print('Module input:', input)
        print('Module output:', output)

    def back_hook(self, module, grad_input, grad_output):  # backward的时候执行
        print('---------------------doing back_hook-----------------------')
        print('original grad:', grad_input)
        print('original outgrad:', grad_output)
        # grad_input = grad_input[0]*self.input   # 这里把hook函数内对grad_input的操作进行了注释，
        # grad_input = tuple([grad_input,grad_input])        # 返回的grad_input必须是tuple，所以我们进行了tuple包装。
        # print('now grad:', grad_input)
        return grad_input


if __name__ == '__main__':
    input = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True).to(device)
    net = MyNet()
    net.to(device)
    net.register_forward_pre_hook(net.pre_forward_hook)
    net.register_forward_hook(net.forward_hook)
    net.register_backward_hook(net.back_hook)
    input.register_hook(tensor_hook)
    result = net(input)
    print('result =', result)
    result.backward()
    for param in net.parameters():
        print('{}: grad->{}'.format(param, param.grad))
