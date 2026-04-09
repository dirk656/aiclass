import cv2 as cv
import numpy as np
from lenna_conv import convolution, kernel


def max_pooling(image: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
	pool_size = 2
	stride = 2
	hight, weight, channel = image.shape
	out_height = (hight - pool_size) // stride + 1
	out_width = (weight - pool_size) // stride + 1
	pooled = np.zeros((out_height, out_width, channel), dtype=image.dtype)

	for y in range(out_height):
		for x in range(out_width):
			y0 = y * stride
			x0 = x * stride
			region = image[y0 : y0 + pool_size, x0 : x0 + pool_size, :]
			pooled[y, x, :] = np.max(region, axis=(0, 1))
	return pooled


def avg_pooling(image: np.ndarray, output_size: tuple[int, int] = (64, 64)) -> np.ndarray:

	in_height, in_width, channel = image.shape
	out_height, out_width = output_size
	pooled = np.zeros((out_height, out_width, channel), dtype=np.float32)

	for oy in range(out_height):
		y_start = int(np.floor(oy * in_height / out_height))
		y_end = int(np.ceil((oy + 1) * in_height / out_height))
		for ox in range(out_width):
			x_start = int(np.floor(ox * in_width / out_width))
			x_end = int(np.ceil((ox + 1) * in_width / out_width))
			region = image[y_start:y_end, x_start:x_end, :]
			pooled[oy, ox, :] = np.mean(region, axis=(0, 1))

	return np.clip(pooled, 0, 255).astype(np.uint8)


def main() -> None:
	img = cv.imread("Lenna.png")

	conv_img = convolution(img, kernel)
	conv_img = np.clip(conv_img, 0, 255).astype(np.uint8)

	max_pooled = max_pooling(conv_img, pool_size=2, stride=2)
	avg_pooled = avg_pooling(conv_img, output_size=(256, 256))


	cv.imshow("Convolution Result", conv_img)
	cv.imshow("Max Pooling", max_pooled)
	cv.imshow("Avg Pooling", avg_pooled)
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == "__main__":
	main()


