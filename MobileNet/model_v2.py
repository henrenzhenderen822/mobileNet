import torch
from torch import nn


# _ 私有函数的命名约定，即提示程序员该函数只能在类或者该文件内部使用，但实际上也可以在外部使用。
def _make_divisible(ch, divisor=8, min_ch=None):   # 调整通道数为8的整数
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 按官方方式搭建，继承Sequential，变成一个小的Sequential结构  (conv+BN+ReLU6)
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # 通过设置groups参数，对应的输入通道与输出通道数进行分组（groups=1为默认的普通卷积, groups=输出通道数时为DW卷积）
        padding = (kernel_size - 1) // 2   # 除以2取整
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


# 倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):   # expand_ratio是扩展因子（增加通道倍数）
        super(InvertedResidual, self).__init__()
        hidden_channal = in_channel * expand_ratio
        # 当stride=1而且输入与输出通道数相同时才能使用捷径
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:    # 第一个bottleneck不需要使用第一个1×1的卷积（官方实现）
            # 1×1的点卷积层
            layers.append(ConvBNReLU(in_channel, hidden_channal, kernel_size=1))
        layers.extend([
            # 3×3的DW卷积
            ConvBNReLU(hidden_channal, hidden_channal, stride=stride, groups=hidden_channal),
            # 1×1的点卷积层(线性激活-不用处理了)
            nn.Conv2d(hidden_channal, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s   t:扩展因子， c：输出通道， n：bottleneck的重复次数， s:步距-只针对第一个bottleneck，其余为1
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # 连接若干个bottleneck层
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
        # 最后几层
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        # 将以上操作放在顺序容器(特征提取层)
        self.features = nn.Sequential(*features)  # 如果输入图像是224×224，那么现在是7×7的特征图

        # 接入分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均值池化，参数为输出维度（这里1×1，相当与使用了和特征图大小相等的核）
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # 初始化权重参数
        for m in self.modules():   # self.module()采用深度优先搜索的方式，存储了net的所有模块
            if isinstance(m, nn.Conv2d):    # isinstance是用来判断一个对象的变量类型。
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bise)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # 打平操作
        x = x.view(x.size(0), -1)   # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 测试代码
def main():
    x = torch.rand(2, 3, 224, 224)
    model = MobileNetV2()
    output = model(x)
    print(output.shape)


if __name__ == '__main__':
    main()



