# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
#im = cv2.imread('img_00898.jpg')
#im = cv2.imread('img_07473.jpg')
im = cv2.imread('amelioration.jpg')
im = np.float32(im) / 255.0
 
# Calculate gradient 
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
# Python Calculate gradient magnitude and direction ( in degrees ) 
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

plt.imshow(mag)
plt.savefig("hog.png")
