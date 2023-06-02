import torch
from torch import nn

a = nn.AdaptiveAvgPool2d(1)
b = nn.AdaptiveAvgPool2d((1, 1))

x = torch.rand(2, 100, 7, 7)

print(a(x).shape)
print(b(x).shape)

