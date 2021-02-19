import torch
import torch.nn as nn


class VggNet(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VggNet, self).__init__()
        self.features = features
        self.classfier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),  # 原地操作
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classfier(x)
        return x


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  # 列表、元组前面加星号作用是将列表解开成独立的参数，传入函数，


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg(model_name="vgg16", **kwargs):
    cfg = []
    try:
        cfg = cfgs[model_name]
    except ImportError:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)  # 非正常运行导致退出程序，与exit(1)类似
    model = VggNet(make_features(cfg), **kwargs)
    return model


net = vgg('vgg16')

# tensor0 = torch.randn(1, 3, 224, 224)
# print(tensor0)
# print(net(tensor0))

