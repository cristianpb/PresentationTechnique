import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img_07473.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img)
plt.savefig("corners.png")
