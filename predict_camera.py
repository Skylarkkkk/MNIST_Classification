import torch
import torch.nn as nn
import cv2
import numpy as np
from train import CNNNet
import torchvision

# 加载模型
cnn = CNNNet()
cnn.load_state_dict(torch.load("model/mnist_model.pth"))
cnn = cnn.cuda()
cnn.eval()  # 设置为评估模式


# 图像预处理函数
def preprocess(img):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 获取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 截取数字区域
        roi = thresh[y:y + h, x:x + w]

        # 调整大小至28x28
        resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # 归一化并转换为Tensor
        tensor_img = torchvision.transforms.ToTensor()(resized).unsqueeze(0).cuda()

        return tensor_img, (x, y, w, h)

    return None, None


# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理并获取ROI
    processed_img, bbox = preprocess(frame)

    if processed_img is not None:
        # 预测
        with torch.no_grad():
            output = cnn(processed_img)
            pred = output.argmax(dim=1).item()

        # 在画面中标注结果
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Pred: {pred}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示实时画面
    cv2.imshow('Digit Recognition', frame)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()