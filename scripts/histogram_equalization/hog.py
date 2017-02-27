# import opencv 
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns

# Read image 
src = cv2.imread("img_07473.jpg", cv2.IMREAD_GRAYSCALE); 
hist = cv2.calcHist(src,[0],None,[256],[0,256])
# Binary 
ret,thresh1 = cv2.threshold(src,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
bin_hist = cv2.calcHist(thresh1,[0],None,[256],[0,256])

# Binary inverse
blur = cv2.GaussianBlur(src,(5,5),0)
bin_hist2 = cv2.calcHist(blur,[0],None,[256],[0,256])
#ret,thresh2 = cv2.threshold(src,127,255,cv2.THRESH_TOZERO_INV)
thresh2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
# Create a CLAHE object (Arguments are optional).
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
#cl1 = clahe.apply(src)
#cl1_hist = cv2.calcHist(cl1,[0],None,[256],[0,256])

# Two subplots, the axes array is 1-d
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.imshow(src)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(blur)
ax2.set_xticks([])
ax2.set_yticks([])
ax3.imshow(thresh2)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.plot(hist)
ax4.set_title('Gray scale original')
ax5.plot(bin_hist)
ax5.set_title('Histogram equalization')
ax6.plot(bin_hist2)
ax6.set_title('Adaptive histogram equalization')
plt.savefig("thresholding_gray.jpg")
