import cv2
import matplotlib.pyplot as plt

surf = cv2.xfeatures2d.SURF_create()

img = cv2.imread('img_07473.jpg',0)
surf.setHessianThreshold(1000)
#surf.setUpright(True)
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2)
plt.savefig("surf.png")
