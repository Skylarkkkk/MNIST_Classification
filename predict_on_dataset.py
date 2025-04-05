import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
from train import CNNNet
import cv2


# net
cnn = CNNNet()
cnn.load_state_dict(torch.load("model/mnist_model.pth"))
cnn = cnn.cuda()

# data
test_data = datasets.MNIST(root='./data',
                            train=False,
                            download=False,
                            transform=transforms.ToTensor())  # transform表示数据预处理
#  此时数据集已经下载到./data/MNIST/下，格式为pkl或者pt
# # batch_size
batch_size = 64
test_loader = Data.DataLoader(dataset=test_data,batch_size=batch_size
                                 ,shuffle=False)  # shuffle表示是否打乱数据

# DataLoader可以将数据集分成batch_size大小的小批量数据，并且循环读取以训练测试

# loss
loss_func = nn.CrossEntropyLoss()

# optimizer

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# test
loss_test = 0
accuracy = 0
for i, (images, labels) in enumerate(test_loader):
    images = images.cuda()
    labels = labels.cuda()
    outputs = cnn(images)

    # label的维度为batch_size
    # outputs = batch_size * cls_num
    loss_test += loss_func(outputs, labels)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    # 将数据从GPU转移到CPU,并转换为numpy格式
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()

    # batch_size * 1 * 28 * 28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx].item()

        # 将数据的通道维度从1转移到最后，符合opencv的格式
        im_data = im_data.transpose(1, 2, 0)

        print("label:", im_label)
        print("pred:", im_pred)
        cv2.imshow("imdata", im_data)
        cv2.waitKey(0)


accuracy = accuracy / len(test_data)
loss_test = loss_test / (len(test_data) // batch_size)

print(f"acc: {accuracy}, loss_test:, {loss_test.item()}")
