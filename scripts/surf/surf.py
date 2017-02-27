import cv2
import numpy as np
from mahotas.features import surf
import milk

img = cv2.imread('img_07473.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#f = mh.demos.load('luispedro', as_grey=True)
f = gray.astype(np.uint8)
spoints = surf.surf(f, nr_octaves=4, nr_scales=6, initial_step_size=2)
print("Nr points:", len(spoints))

try:
    import milk
    descrs = spoints[:,5:]
    k = 5
    values, _  =milk.kmeans(descrs, k)
    colors = np.array([(255-52*i,25+52*i,37**i % 101) for i in range(k)])
except:
    values = np.zeros(100)
    colors = np.array([(255,0,0)])

f2 = surf.show_surf(f, spoints[:100], values, colors)

cv2.imwrite('sift_keypoints.jpg',f2)
