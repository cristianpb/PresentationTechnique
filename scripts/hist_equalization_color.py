# import opencv 
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns

# Read image 
img = cv2.imread('img_07473.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # transform to YUV
hist = cv2.calcHist(img_yuv,[0],None,[256],[0,256]) # histogram for original image
img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)  
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
equ_hit = cv2.calcHist(img_output,[0],None,[256],[0,256])
# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4,4))
#cl1 = clahe.apply(src)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
cl1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
cl1_hist = cv2.calcHist(img_yuv,[0],None,[256],[0,256])

# Two subplots, the axes array is 1-d
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.imshow(img_rgb)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(img_output)
ax2.set_xticks([])
ax2.set_yticks([])
ax3.imshow(cl1)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.plot(hist)
ax4.set_title('Original')
ax5.plot(equ_hit)
ax5.set_title('Histogram equalization')
ax6.plot(cl1_hist)
ax6.set_title('Adaptive histogram equalization')
plt.savefig("hist_equalization.jpg")
