import cv2 as cv  
import numpy as np 

# 更明显的锐化核（核和为 1，整体亮度更稳定）
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:  
    kernel_h , kernel_w = kernel.shape  # 读取卷积核的高度和宽度。
    pad_h, pad_w = kernel_h // 2, kernel_w // 2  # 根据卷积核尺寸计算上下和左右填充大小。
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')  # 对输入图像做反射填充，避免边界信息丢失。
    output = np.zeros_like(image, dtype=np.float32)  # 创建浮点型输出数组，防止计算中溢出和截断。
    for i in range(image.shape[0]):  # 遍历图像的每一行。
        for j in range(image.shape[1]):  # 遍历图像的每一列。
            for c in range(image.shape[2]):  # 遍历 RGB 三个通道。
                region = padded_image[i:i+kernel_h, j:j+kernel_w, c]  # 取出当前位置对应的局部窗口。
                output[i, j, c] = np.sum(region * kernel)  # 局部窗口与卷积核逐元素相乘后求和。
    return np.clip(output, 0, 255).astype(np.uint8)  # 将结果限制到像素范围并转为 uint8 图像。



def main():
    img = cv.imread("Lenna.png")
   
    processed_img = convolution(img, kernel)
    cv.imshow("Original Image", img)
    cv.imshow("Sharpened Image", processed_img)

    
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
