import cv2
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel_5x5 = np.array([[-1, -1, -1, -1, -1], [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1], [-1, 1, 2, 1, -1], [-1, -1, -1, -1, -1]])
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
top_sobel = np.array([[-1, -2, -1], [0, 0, 0], [-1, -2, -1]])

img = cv2.imread('mypic.png', 0)

def convolution(img, kernel):
    input = np.pad(img, (int((kernel.shape[0] - 1)/2), int((kernel.shape[1] - 1)/2)))
    output = np.empty_like(img)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(input[i: i + kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output

# k3 = ndimage.convolve(img, kernel_3x3)
# k5 = ndimage.convolve(img, kernel_5x5)

k3 = convolution(img, kernel_3x3)
k5 = convolution(img, kernel_5x5)
sp = convolution(img, top_sobel)

blurred = cv2.GaussianBlur(img, (17, 17), 0)
g_hpf = img - blurred

# cv2.imshow("3x3", k3)
# cv2.imshow("5x5", k5)
cv2.imshow("sp", sp)
cv2.imshow("blurred", blurred)
cv2.imshow("g_hpf", g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()
