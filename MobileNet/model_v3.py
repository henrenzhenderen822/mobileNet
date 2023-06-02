from typing import Callable, List, Optional
# typeing库: 可以描述任何变量的任何类型（类型注释）。 它预装了多种类型注释，如Dict，Tuple，List，Set等等！
# 如果某个函数的参数可以是多种可选类型，则可以使用typing.Optional或typing.Union类型
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


# _ 私有函数的命名约定，即提示程序员该函数只能在类或者该文件内部使用，但实际上也可以在外部使用。
def _make_divisible(ch, divisor=8, min_ch=None):   # 调整通道数为8的整数,对硬件更加友好，利于训练
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


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 outplanes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size-1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=outplanes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         groups=groups,
                                                         padding=padding,
                                                         bias=False),
                                               norm_layer(outplanes),
                                               activation_layer(inplace=True))


# 注意力机制模块SE
class SqueezeExcitation(nn.Module):
    def __init__(self, input_ch: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_ch = _make_divisible(input_ch // squeeze_factor, 8)
        # 这里的两个全连接层等同于使用1×1的卷积
        self.fc1 = nn.Conv2d(input_ch, squeeze_ch, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_ch, input_ch, kernel_size=1)

    def forward(self, x):
        # 自适应平均值池化
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        # 相当于给每个通道赋予权重
        return scale * x


class InvertedResidualConfig:     # 参数配置文件
    def __init__(self,
                 input_ch: int,
                 kernel_size: int,
                 expanded_ch: int,
                 output_ch: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):    # width_multi 卷积核个数（相对于原始)的比例
        self.input_ch = self.adjust_channel(input_ch, width_multi)
        self.kernel_size = kernel_size
        self.expanded_ch = self.adjust_channel(expanded_ch, width_multi)
        self.output_ch = self.adjust_channel(output_ch, width_multi)
        self.use_se = use_se
        self.use_hs = activation == 'HS'  # 是否使用 h-swish 激活函数
        self.stride = stride

    @staticmethod
    def adjust_channel(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:    # 确保步长符合论文要求
            raise ValueError('illegal stride value.')

        self.use_res_connect = (cnf.stride == 1 and cnf.input_ch == cnf.output_ch)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand, 官方实现的第一层没有第一个点卷积
        if cnf.expanded_ch != cnf.input_ch:
            layers.append(ConvBNActivation(in_planes=cnf.input_ch,
                                           outplanes=cnf.expanded_ch,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))
        # DW卷积
        layers.append(ConvBNActivation(cnf.expanded_ch,
                                       cnf.expanded_ch,
                                       kernel_size=cnf.kernel_size,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_ch,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(input_ch=cnf.expanded_ch))

        # 降维卷积层
        layers.append(ConvBNActivation(cnf.expanded_ch,
                                       cnf.output_ch,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channel = cnf.output_ch

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_ch: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty.')
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            # partial: 有时参数可以在函数被调用之前提前获知。这种情况下，一个函数有一个或多个参数预先就能用上，以便函数能用更少的参数进行调用。
            norm_layer = partial(nn.BatchNorm2d, eps=0.01, momentum=0.01)

        layers: List[nn.Module] = []

        # 第一层
        firstconv_output_ch = inverted_residual_setting[0].input_ch
        layers.append(ConvBNActivation(3,
                                       firstconv_output_ch,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # 中间的反残差模块
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # 最后几层
        lastconv_input_ch = inverted_residual_setting[-1].output_ch
        lastconv_output_ch = 6 * lastconv_input_ch
        layers.append(ConvBNActivation(lastconv_input_ch,
                                       lastconv_output_ch,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # 将以上层放到Sequential中，可以看作特征提取部分,(7×7大小的特征图)
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应为1×1大小（参数1等同于(1，1)）
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_ch, last_ch),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(0.2, inplace=True),
                                        nn.Linear(last_ch, num_classes))

        # 初始化网络参数权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def mobilenet_v3_large(num_classes: int = 1000, reduced_tail: bool = False):   # reduced_tail:是否减小最后几层的通道数
    width_multi = 1.0    # alpha参数
    bottleneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channel, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_ch, kernel_size, expanded_ch, output_ch, use_se, activation, stride
        bottleneck_conf(16, 3, 16, 16, False, 'RE', 1),
        bottleneck_conf(16, 3, 64, 24, False, 'RE', 2),  # C1
        bottleneck_conf(24, 3, 72, 24, False, 'RE', 1),
        bottleneck_conf(24, 5, 72, 40, True, 'RE', 2),   # C2
        bottleneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bottleneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bottleneck_conf(40, 3, 240, 80, False, 'HS', 2),  # C3
        bottleneck_conf(80, 3, 200, 80, False, 'HS', 1),
        bottleneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bottleneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bottleneck_conf(80, 3, 480, 112, True, 'HS', 1),
        bottleneck_conf(112, 3, 672, 112, True, 'HS', 1),
        bottleneck_conf(112, 5, 672, 160 // reduce_divider, True, 'HS', 2),   # C4
        bottleneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
        bottleneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
    ]
    last_ch = adjust_channels(1280 // reduce_divider)

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_ch=last_ch,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channel, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_ch=last_channel,
                       num_classes=num_classes)


# 测试
def main():
    x = torch.rand(2, 3, 224, 224)
    model = mobilenet_v3_large()
    result = model(x)
    print(result.shape)

    backbone = mobilenet_v3_large()
    backbone = backbone.features
    for i, b in enumerate(backbone):
        print(b)


if __name__ == '__main__':
    main()





