import cv2 
import matplotlib.pyplot as plt
import seaborn as sns

def calc_hog_block(hogim, r):
    """
    calculate HOG feature given a rectangle and integral HOG image
     
    returns
        HOG feature (not normalized)
 
    params
        hogim : integral HOG image
        r : 4-tuple representing rect (left,top,right,bottom)
    """
    numorient = hogim.shape[2]
    result = np.zeros(numorient)
    for ang in xrange(numorient):
        result[ang] = hogim[r[1],r[0],ang] + hogim[r[3],r[2],ang] - hogim[r[1],r[2],ang] - hogim[r[3],r[0],ang]
 
    return result

def draw_hog(ihog, cellsize=8):
    """
    visualize HOG features
     
    returns
        None
         
    params
        target  : target image
        ihog    : integral HOG image
        cellsize: size of HOG feature to be visualized (default 8x8)
         
    """
    #ow,oh = cv.GetSize(target)
    ow,oh = 8, 1
    halfcell = cellsize/2
    w,h = ow/cellsize,oh/cellsize
    norient = ihog.shape[2]
    mid = norient/2
 
    for y in xrange(h-1):
        for x in xrange(w-1):
            px,py=x*cellsize,y*cellsize
            #feat = calc_hog_block(ihog, (px,py,max(px+cellsize, ow-1),max(py+cellsize, oh-1)))
            feat = calc_hog_block(ihog, (px, py, px+cellsize, py+cellsize))
            px += halfcell
            py += halfcell
             
            #L1-norm, nice for visualization
            mag = np.sum(feat)
            maxv = np.max(feat)
            if mag > 1e-3:
                nfeat = feat/maxv
                N = norient
                fdraw = []
                for i in xrange(N):
                    angmax = nfeat.argmax()
                    valmax = nfeat[angmax]
                    x1 = int(round(valmax*halfcell*np.sin((angmax-mid)*np.pi/mid)))
                    y1 = int(round(valmax*halfcell*np.cos((angmax-mid)*np.pi/mid)))
                    gv = int(round(255*feat[angmax]/mag))
                     
                    #don't draw if less than a threshold
                    if gv < 30:
                        break
                    fdraw.insert(0, (x1,y1,gv))
                    nfeat[angmax] = 0.
                     
                #draw from smallest to highest gradient magnitude
                for i in xrange(len(fdraw)):
                    x1,y1,gv = fdraw[i]
                    cv.Line(target, (px-x1,py+y1), (px+x1,py-y1), cv.CV_RGB(gv, gv, gv), 1, 8)
            else:
                #don't draw if there's no reponse
                pass

img = cv2.imread('img_07473.jpg')
#im = cv.LoadImage('gambar.jpg')
 
#image for visualization
#vhog = cv2.CreateImage(cv.GetSize(img), 8, 1)
  
hog = calc_hog(img)
   
draw_hog(hog, 8)

plt.imshow(vhog)
plt.savefig("hog.jpg")

#cv.ShowImage('hi', vhog)

# Read image 
#img = cv2.imread('img_07473.jpg')
#img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # transform to YUV
#hist = cv2.calcHist(img_yuv,[0],None,[256],[0,256]) # histogram for original image
#img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)  
## equalize the histogram of the Y channel
#img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
## convert the YUV image back to RGB format
#img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
#equ_hit = cv2.calcHist(img_output,[0],None,[256],[0,256])
## Create a CLAHE object (Arguments are optional).
#clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4,4))
##cl1 = clahe.apply(src)
#img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
#cl1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
#cl1_hist = cv2.calcHist(img_yuv,[0],None,[256],[0,256])
#
## Two subplots, the axes array is 1-d
#f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
#ax1.imshow(img_rgb)
#ax1.set_xticks([])
#ax1.set_yticks([])
#ax2.imshow(img_output)
#ax2.set_xticks([])
#ax2.set_yticks([])
#ax3.imshow(cl1)
#ax3.set_xticks([])
#ax3.set_yticks([])
#ax4.plot(hist)
#ax4.set_title('Original')
#ax5.plot(equ_hit)
#ax5.set_title('Histogram equalization')
#ax6.plot(cl1_hist)
#ax6.set_title('Adaptive histogram equalization')
#plt.savefig("thresholding.jpg")
