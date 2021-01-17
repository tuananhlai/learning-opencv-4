import cv2 as cv

f_image = cv.imread('a.png')
f_image = cv.cvtColor(f_image, cv.COLOR_BGR2RGB)

s_image = cv.imread('b.png')
s_image = cv.cvtColor(s_image, cv.COLOR_BGR2RGB)

print(f_image.shape, s_image.shape)

alpha = 0.3
beta = 1 - alpha
b_img = f_image * alpha +s_image * beta
b_img = b_img.astype(int)

cv.imwrite('x.png', b_img)