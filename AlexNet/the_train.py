from torchvision import transforms, datasets
from model import AlexNet
import torch.optim as optim
import torch.nn as nn
import torch
import os
import json
import torch.utils.data as datas
from tqdm import tqdm


def main():
    # 设置运行设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val" : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
    # 存放train与val的路径
    image_path = '/home/xulei/数据集大本营/5_flower_data/flower_data'  # flower data root path
    # 若该目录不存在，在报错并终止程序
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 定义训练数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 训练数据集的文件数量
    train_num = len(train_dataset)

    # flower_list: {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # cla_dict : {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    # 要输出json格式，需要对json数据进行编码，要用到函数：json.dumps
    # indent=4, 的作用是让字典的内容逐行显示，每个key占一行
    # json_str :
    # '{
    #     "0": "daisy",
    #     "1": "dandelion",
    #     "2": "roses",
    #     "3": "sunflowers",
    #     "4": "tulips"
    # }'
    json_str = json.dumps(cla_dict, indent=4)

    with open('class_idices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 128
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)    # number of workers nw: 8 ?????
    print("using {} dataloader workers every process".format(nw))
    train_loader = datas.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # val_num: 364
    val_num = len(validate_dataset)
    validate_loader = datas.DataLoader(validate_dataset, batch_size, shuffle=False, num_workers=nw)
    print("using {} images for trainning, {} images for validation.".format(train_num, val_num))

    net = AlexNet(num_classes=5).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00004)

    epoches = 20
    save_path = './AlexNet.pth'
    best_acc = 0.0
    # train_steps : 26 len(train_loader)= training_images_num/batch_size
    train_steps = len(train_loader)
    for epoch in range(epoches):

        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)  # 进度条
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epoches, loss)

        # validata
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)  # , colour='green'
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print('\n[epoch %d] train_loss: %.3f val_accuracy: %.3f' %
              (epoch+1, running_loss/train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("Finshed Training")


if __name__ == '__main__':
    main()








