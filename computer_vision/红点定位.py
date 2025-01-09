import cv2
import numpy as np

# 1. 读取图像

image = cv2.imread(r'C:\Code\Python\image.jpg')  # 替换为你的图像路径

# 检查图像是否加载成功
if image is None:
    print("Error: Image not loaded correctly.")
else:
    # 2. 将图像从 BGR 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. 定义红色的 HSV 范围
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 4. 创建掩模，获取红色区域
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # 合并两个红色范围的掩模

    # 5. 查找掩模中的轮廓（即红色区域）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. 在原图上绘制红色区域的轮廓
    result_image = image.copy()  # 为避免修改原图，创建副本
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽为 2

    # 7. 显示结果图像
    cv2.imshow('Red Region Contours', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
