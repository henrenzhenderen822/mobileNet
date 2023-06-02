import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

from model_v2 import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 16
    epochs = 5

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = 'D:\PyTorch_project\data'  # 数据集路径
    image_path = os.path.join(data_root, 'pokemon')  # 宝可梦数据集
    # print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    dataset = datasets.ImageFolder(image_path,
                                         transform=data_transform["train"])
    # print(dataset.class_to_idx)    # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
    n = len(dataset)
    proportion = 0.2  # 将数据的20％用作验证集
    n_test = random.sample(range(1, n), int(proportion * n))  # 按比例取随机数列表

    test_dataset = torch.utils.data.Subset(dataset, n_test)  # 按照随机数列表取测试集
    train_dataset = torch.utils.data.Subset(dataset, list(set(range(1, n)).difference(set(n_test))))  # 测试集剩下的作为训练集

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # x, label = iter(train_lodar).next()
    # print(x.shape, label.shape)

    train_num = len(train_dataset)
    val_num = len(test_dataset)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # create model
    net = MobileNetV2(num_classes=5)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v2.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights（官方是在imagenet上预训练的，所以最后一层神经元个数不同，这层不能使用预训练参数）
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False         # 将提取特征层参数都冻结，仅留下分类层的参数，便于训练网络

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './MobileNetV2.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
