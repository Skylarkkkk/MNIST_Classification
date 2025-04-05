import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
import logging
from datetime import datetime


# 设置日志
def setup_logger():
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)

    # 获取当前时间作为日志文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{current_time}.log"

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logger()

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# data
logger.info("Loading MNIST dataset...")
train_data = datasets.MNIST(root='./data',
                            train=True,
                            download=True,
                            transform=transform_train)  # transform表示数据预处理

test_data = datasets.MNIST(root='./data',
                           train=False,
                           download=False,
                           transform=transform_test)  # transform表示数据预处理

logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

# batch_size
batch_size = 64
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size
                               , shuffle=True)  # shuffle表示是否打乱数据

test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size
                              , shuffle=False)  # shuffle表示是否打乱数据


# net
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 此时输出维度为 (batch_size, 32, 14, 14)

            # 第二层卷积
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第三层卷积
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        # 此时输出维度为 (batch_size, 64, 6, 6)
        self.fc = nn.Sequential(
            nn.Linear(6 * 6 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)  # 转换维度
        out = self.fc(out)
        return out


if __name__ == '__main__':
    logger.info("Initializing CNN model...")
    cnn = CNNNet()
    cnn = cnn.cuda()
    logger.info("Model architecture:\n" + str(cnn))

    # loss
    loss_func = nn.CrossEntropyLoss()

    # optimizer weight_decay表示L2正则化
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)
    logger.info(f"Optimizer: {optimizer.__class__.__name__}, Learning rate: 0.001, Weight decay: 1e-4")

    # train
    logger.info("Starting training...")
    for epoch in range(30):
        cnn.train()  # 设置为训练模式
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            outputs = cnn(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                logger.info(f"Epoch: {epoch + 1}, Batch: {i}, Training Loss: {loss.item():.4f}")

        logger.info(f"Epoch: {epoch + 1}, Training Loss: {loss.item():.4f}")

        # test
        cnn.eval()  # 设置为评估模式
        loss_test = 0
        accuracy = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                labels = labels.cuda()
                outputs = cnn(images)

                # label的维度为batch_size
                # outputs = batch_size * cls_num
                loss_test += loss_func(outputs, labels)

                # 取出最大值的索引
                _, pred = outputs.max(1)
                accuracy += (pred == labels).sum().item()

        # 所有批次中正确预测的总数 accuracy 除以测试集的总样本数 len(test_data)
        accuracy = accuracy / len(test_data)

        # loss_func 是逐样本计算后求平均，每个批次的损失已经是该批次的平均损失。
        # 总损失 loss_test 除以批次数量 (len(test_data) // batch_size)，得到每个样本的平均损失。
        loss_test = loss_test / (len(test_data) // batch_size)

        logger.info(f"Epoch: {epoch + 1}, Test Accuracy: {accuracy * 100:.2f}%, Test Loss: {loss_test.item():.4f}")

    # save
    os.makedirs("model", exist_ok=True)  # 如果目录已存在，不会报错
    model_path = "model/mnist_model.pth"
    torch.save(cnn.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")