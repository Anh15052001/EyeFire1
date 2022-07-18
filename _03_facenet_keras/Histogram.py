import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("data/check_blur/1.jpg")
#calculate histogram each image
def compute_hist(img):
    hist = np.zeros((256, ), np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    return hist
def equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = hist[:i].sum()
    print(cumulator)
    new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

"""
hist = compute_hist(img).ravel()
new_hist = equal_hist(hist)

h, w = img.shape[:2]
for i in range(h):
    for j in range(w):
        img[i, j] = new_hist[img[i, j]]

"""
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite("data/gen_test/abc.jpg", img_output)
cv2.imshow("root", img)
cv2.imshow("new", img_output)
cv2.waitKey(0)
plt.show()