import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v3 import mobilenet_v3_large


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    data_path = 'D:\PyTorch_project\data'
    img_path = os.path.join(data_path, 'flower_data', 'valid', '1', 'image_06755.jpg')
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # 经过一系列图像处理，转化成张量：[C, H, W]
    img = data_transform(img)
    # 增加一个batch的维度，以适应网络输入：[B, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = os.path.join(data_path, 'flower_data', 'cat_to_name.json')
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    n = len(class_indict)  # 种类
    print(class_indict)
    print(n)
    # create model
    model = mobilenet_v3_large(num_classes=n).to(device)
    # load model weights
    model_weight_path = "./MobileNetV3_Large_flower.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()   # torch.squeeze 删除维度大小为1的所有维数
        predict = torch.softmax(output, dim=0)  # 这里经过了softmax操作转化为了概率，非必要但方便可视化
        predict_cla = torch.argmax(predict).numpy()
    print_res = "class: {}({})   prob: {:.3}".format(class_indict[str(predict_cla+1)], predict_cla+1,
                                                     predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {}({})\t   prob: {:.3}".format(class_indict[str(i+1)], i+1,
                                                      predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
