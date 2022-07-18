import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
def isBright(image, dim=10, thresh=0.5):
    #resize image
    image = cv2.resize(image, (dim, dim))
    #Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    #Nomalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    print(np.mean(L))
    return np.mean(L) > thresh
image = cv2.imread("data/gen_test/black.jpg")
text = "bright" if isBright(image) else "dark"
cv2.putText(image, "{}".format(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()