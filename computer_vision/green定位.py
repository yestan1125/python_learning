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

    # 4. 定义绿色的 HSV 范围
    lower_green = np.array([35, 50, 50])  # 绿色的下界
    upper_green = np.array([85, 255, 255])  # 绿色的上界

    # 5. 创建掩模，获取绿色区域
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 6. 查找掩模中的轮廓（即绿色区域）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 在图像中标记绿色点的位置
    result_image = frame.copy()  # 保持原图不变

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 忽略较小的噪点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 在图像上标记绿色点的位置
                cv2.circle(result_image, (cX, cY), 10, (0, 255, 0), -1)  # 绿色圆圈标记
                cv2.putText(result_image, f"({cX},{cY})", (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 8. 显示结果图像
    cv2.imshow('Green Dot Detection', result_image)

    # 9. 按键 'q' 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 10. 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
