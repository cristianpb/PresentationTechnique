import cv2
import matplotlib.pyplot as plt


img = cv2.imread('img_00898.jpg', 0)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2)
plt.savefig("sift.png")
