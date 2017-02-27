import cv2
import numpy as np
from mahotas.features import surf
import milk

img = cv2.imread('img_07473.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#img = cv2.imread('fly.png',0)
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
#surf = cv2.SURF(400)
#surf = surf.surf(400)

# Find keypoints and descriptors directly
#kp, des = surf.detectAndCompute(img,None)
des = surf.surf(gray)
kp = surf.interest_points(gray, threshold=10)

print(len(kp))
#img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
f2 = surf.show_surf(img, kp[:100], )

cv2.imwrite('surf_keypoints.jpg',img2)
