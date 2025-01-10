import cv2
import numpy as np

# 1. 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，如果有多个摄像头，改变数字 (1, 2, ...)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# 2. 持续读取摄像头图像
while True:
    # 获取摄像头中的一帧图像
    ret, frame = cap.read()

    # 检查图像是否成功读取
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    # 3. 将图像从 BGR 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 4. 定义红色的 HSV 范围
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 5. 创建掩模，获取红色区域
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # 合并两个红色范围的掩模

    # 6. 查找掩模中的轮廓（即红色区域）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 在图像上绘制轮廓
    result_image = frame.copy()  # 保持原图不变
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # 绘制绿色轮廓

    # 8. 显示结果图像
    cv2.imshow('Red Region Contours', result_image)

    # 9. 按键 'q' 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 10. 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
