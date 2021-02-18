# import torch
# import torch.nn as nn
# import torch.nn.functional as f
#
#
# # class VGG16Net(nn.Module):
# #     def __init__(self, num_classes):
# #         super(VGG16Net, self).__init__()
# #         self.conv1 = nn.Conv2d(3, 64, 3)
# #         self.conv2 = nn.Conv2d(64, 64, 3)
# #         self.pool = nn.MaxPool2d(2, 2)
# #         self.conv3 = nn.Conv2d(64, 128, 3)
# #         self.conv4 = nn.Conv2d(128, 128, 3)
# #         self.conv5 = nn.Conv2d(128, 256, 3)
# #         self.conv6 = nn.Conv2d(256, 256, 3)
# #         self.conv7 = nn.Conv2d(256, 256, 3)
# #         self.conv8 = nn.Conv2d(256, 256, 3)
# #         self.conv9 = nn.Conv2d(256, 512, 3)
# #         self.conv10 = nn.Conv2d(512, 512, 3)
# #         self.conv11 = self.conv10
# #         self.conv12 = self.conv10
# #         self.conv13 = self.conv10
# #         self.fc1 = nn.Linear(512 * 7 * 7, 4096)
# #         self.fc1 = nn.Linear(4096, 4096)
# #         self.fc2 = nn.Linear(4096, num_classes)
# #         self.drop = nn.Dropout(p=0.5)
# #
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = f.relu(x)
# #         x = self.conv2(x)
# #         x = f.relu(x)
# #         x = self.pool(x)
# #
# #         x = self.conv3(x)
# #         x = f.relu(x)
# #         x = self.conv4(x)
# #         x = f.relu(x)
# #         x = self.pool(x)
# #
# #         x = self.conv5(x)
# #         x = f.relu(x)
# #         x = self.conv6(x)
# #         x = f.relu(x)
# #         x = self.conv7(x)
# #         x = f.relu(x)
# #         x = self.pool(x)
# #
# #         x = self.conv8(x)
# #         x = f.relu(x)
# #         x = self.conv9(x)
# #         x = f.relu(x)
# #         x = self.conv10(x)
# #         x = f.relu(x)
# #         x = self.pool(x)
# #
# #         x = self.conv11(x)
# #         x = f.relu(x)
# #         x = self.conv12(x)
# #         x = f.relu(x)
# #         x = self.conv13(x)
# #         x = f.relu(x)
# #         x = self.pool(x)
# #
# #         x = x.view(-1, 512 * 7 * 7)
# #         x = self.drop(x)
# #         x = self.fc1(x)
# #         x = f.relu(x)
# #
# #         x = self.drop(x)
# #         x = self.fc2(x)
# #         x = f.relu(x)
# #
# #         x = self.fc3(x)
# #
# #         return x

import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, img):
        output0 = self.conv1(img)
        output1 = func.relu(output0)
        output2 = self.pool(output1)
        output3 = self.conv2(output2)
        output4 = func.relu(output3)
        output5 = self.pool(output4)
        output6 = output5.view(-1, 16 * 5 * 5)
        output = self.fc1(output6)
        output = func.relu(output)
        output = self.fc2(output)
        output = func.relu(output)
        output = self.fc3(output)

        # output = nn.ReLU(output)

        return output


net = Net()
