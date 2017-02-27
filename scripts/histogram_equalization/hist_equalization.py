# import opencv 
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns

# Read image 
src = cv2.imread("img_07473.jpg", cv2.IMREAD_GRAYSCALE); 
hist = cv2.calcHist(src,[0],None,[256],[0,256])
# Histogram equalization
equ = cv2.equalizeHist(src)
equ_hit = cv2.calcHist(equ,[0],None,[256],[0,256])
# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
cl1 = clahe.apply(src)
cl1_hist = cv2.calcHist(cl1,[0],None,[256],[0,256])

# Two subplots, the axes array is 1-d
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.imshow(src)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(equ)
ax2.set_xticks([])
ax2.set_yticks([])
ax3.imshow(cl1)
ax3.set_xticks([])
ax3.set_yticks([])
ax4.plot(hist)
ax4.set_title('Gray scale original')
ax5.plot(equ_hit)
ax5.set_title('Histogram equalization')
ax6.plot(cl1_hist)
ax6.set_title('Adaptive histogram equalization')
plt.savefig("hist_gray.jpg")
