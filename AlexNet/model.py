import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(                                # input 3*224*224
            nn.Conv2d(in_channels=3, out_channels= 48, kernel_size=11, stride=4, padding=2),  # output 48*55*55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output 48*27*27

            nn.Conv2d(48, 128, 5, 1, 2),  # output 128*27*27
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),  # output 128*13*13

            nn.Conv2d(128, 192, 3, 1, 1),  # output 192*13*13
            nn.ReLU(True),
            nn.Conv2d(192, 192, 3, 1, 1),  # output 192*13*13
            nn.ReLU(True),
            nn.Conv2d(192, 128, 3, 1, 1),  # output 128*13*13
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),  # output 128*6*6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),  # output 2048*1*1
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # output 2048*1*1
            nn.ReLU(True),
            nn.Linear(2048, num_classes)  # output num_classes*1*1
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x






