import os
import sys
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_v3 import mobilenet_v3_large


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 16
    epochs = 5

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(45),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.ColorJitter(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    }

    data_root = 'D:\PyTorch_project\data'  # 数据集路径
    image_path = os.path.join(data_root, 'flower_data')  # 宝可梦数据集

    # 利用字典，将数据集分为 训练集和验证集 两部分
    image_datasets = {x: datasets.ImageFolder(os.path.join(image_path, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
    dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    '''读取标签对应的实际名字'''
    with open(image_path+'\cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        n = len(cat_to_name)
        # print(cat_to_name)

    train_num = len(image_datasets['train'])
    val_num = len(image_datasets['valid'])

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # create model
    net = mobilenet_v3_large(num_classes=n)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
    model_weight_path = "./mobilenet_v3_large.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights（官方是在imagenet上预训练的，所以最后一层神经元个数不同，这层不能使用预训练参数）
    # numel()函数：返回数组中元素的个数
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    net.load_state_dict(pre_dict, strict=False)  # strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合

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
    save_path = './MobileNetV3_Large_flower.pth'
    train_steps = len(dataloaders['train'])
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(dataloaders['train'], file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            # loss = loss.requires_grad_()
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
            val_bar = tqdm(dataloaders['valid'], file=sys.stdout)
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

