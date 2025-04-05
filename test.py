import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
from train import CNNNet
import cv2
from PIL import Image  # 新增PIL库用于转换

# 加载模型
cnn = CNNNet()
cnn.load_state_dict(torch.load("model/mnist_model.pth"))
cnn = cnn.cuda()
cnn.eval()  # 设置模型为评估模式

# 定义预处理转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整大小
    transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
])


def predict_custom_image(image_path):
    # 使用OpenCV读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path incorrect")

    # 反色处理（如果图片是白底黑字）
    img = 255 - img

    # 转换为PIL Image并应用预处理
    pil_img = Image.fromarray(img)
    img_tensor = transform(pil_img).unsqueeze(0).cuda()  # 增加batch维度并移至GPU

    # 预测
    with torch.no_grad():
        output = cnn(img_tensor)
    pred = output.argmax(dim=1).item()
    return pred

# 显示图片和预测结果
def show_image_with_prediction(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or path incorrect")

    # 显示图片
    cv2.imshow("Predicted Image", img)
    prediction = predict_custom_image(image_path)
    print(f"Predicted digit: {prediction}")

    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 示例使用
image_path = "your_path.jpg"  # 替换为你的图片路径
try:
    show_image_with_prediction(image_path)
except Exception as e:
    print(e)