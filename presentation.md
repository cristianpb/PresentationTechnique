---
title: Partage de connaissance
author: Cristian Perez
date: 28 février 2017
---

## Anatomy of an image classifier

![](images/image-classification-pipeline.jpg){ width=100% }

<div class="notes">
- Many traditional computer vision image classification algorithms follow this pipeline
- Deep Learning based algorithms bypass the feature extraction step completely
</div>

# Preprocessing 

## Deskew

Align an image to a reference assits the classification algorithm
[$^1$](http://docs.opencv.org/trunk/dd/d3b/tutorial_py_svm_opencv.html)

![](images/deskew1.jpg){ height=100% }

[2](https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/)

<div class="notes">
People often think of a learning algorithm as a block box.  In reality, you can
assist the algorithm.  If you are building a face recognition system, aligning
the images to a reference face leads to improvement.  A typical alignment
operation uses a facial feature detector to align the eyes in every image.

Pour le cas de digits, aligner les numéros améliore les résultats.
L'inclination de l'écriture peut être corrigé.  Ansi l'algorithme ne doit pas
apprendre cette variation entre les chiffres.
</div>

-------

Deskewing simple grayscale images can be achieved using image moments (distance and intensity of pixels). 

<div class="notes">
This deskewing of simple grayscale images can be achieved using image moments.
OpenCV has an implementation of moments and it comes in handy while calculating
useful information like centroid, area, skewness of simple images with black
backgrounds.

It turns out that a measure of the skewness is the given by the ratio of the
two central moments ( mu11 / mu02 ). The skewness thus calculated can be used
in calculating an affine transform that deskews the image.
</div>

```python
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
```

## Histogram equalization

Increase image contrast using the image's histogram.

![](images/histogram_equalization.png){ width=100% }

<div class="notes">
- Brighter image will have all pixels confined to high values
- But a good image will have pixels from all regions of the image
</div>

-------

Transformation function which maps the input pixels in brighter region to output pixels in full region.

```python
img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)
```

![](images/equalization_opencv.jpg)

-------

- Histogram equalization considers the global contrast of the image
- The background contrast improves after histogram equalization, but the face of statue lost most of the information there due to over-brightness.

![](images/clahe_1.svg){ height=30% }
[1](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)


-------

### Adaptive Histogram Equalization

- Histogram is equalized inside blocks. 
- Histogram would confine to a small region (unless there is noise).
- If noise is there, it will be amplified. To avoid this, contrast limiting is applied.
- If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. 
- After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.

-------

![](images/clahe_2.svg)

--------

### Example using fishes

Gray scale

```python
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
```

--------

![](images/hist_gray.jpg)

--------

### Histogram equalization for color images

- Histogram equalization is that it's a nonlinear process.
- The concept of histogram equalization is only applicable to the intensity
  values in the image.
- Convert it to a color space where intensity is separated from the color
  information (i.e. **YUV** color space)
- Equalize the Y-channel and combine it with the other two channels.

<div class="notes">
- Histogram equalization is that it's a nonlinear process.
- we cannot just separate out the three channels in an RGB image, equalize the
  histogram separately, and combine them later to form the output image.
- The concept of histogram equalization is only applicable to the intensity
  values in the image. 
- So, we have to make sure not to modify the color information when we do this.

- In order to handle the histogram equalization of color images, we need to
  convert it to a color space where intensity is separated from the color
information.
- YUV is a good example of such a color space. Once we convert it to YUV, we
  just need to equalize the Y-channel and combine it with the other two
channels to get the output image.
</div>

---------

```python
# Read image
img = cv2.imread('img_07473.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # transform to YUV
hist = cv2.calcHist(img_yuv,[0],None,[256],[0,256]) # histogram for original image
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
equ_hit = cv2.calcHist(img_output,[0],None,[256],[0,256])
# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4,4))
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
cl1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
cl1_hist = cv2.calcHist(img_yuv,[0],None,[256],[0,256])
```

-------

![](images/hist_equalization.jpg)

--------

## Image thresholding

- Method of image segmentation
- If pixel value is greater than a threshold value, it is assigned one value, else it is assigned another value.

![](images/threshold.png){ height=80% }

<div class="notes">
- Converts a gray-scale image into a binary image
- The two levels are assigned to pixels that are below or above the specified
  threshold value.
- If pixel value is greater than a threshold value, it is assigned one value
  (may be white), else it is assigned another value (may be black).
</div>

-------- 

```python
# Thresholding with threshold value set 127
th, dst = cv2.threshold(src,127,255, cv2.THRESH_BINARY);
cv2.imwrite("opencv-thresh-binary.jpg", dst);

# Thresholding using THRESH_TOZERO
th, dst = cv2.threshold(src,127,255, cv2.THRESH_TOZERO);
cv2.imwrite("opencv-thresh-tozero.jpg", dst);

# Thresholding using THRESH_TOZERO_INV
th, dst = cv2.threshold(src,127,255, cv2.THRESH_TOZERO_INV);
cv2.imwrite("opencv-thresh-to-zero-inv.jpg", dst);
```

--------

## Adaptative thresholding

- The algorithm calculate the threshold for a small regions of the image. 
- Different thresholds for different regions of the same image 
- Gives us better results for images with varying illumination.

--------

`cv2.ADAPTIVE_THRESH_MEAN_C` : threshold value is the mean of neighbourhood area.
`cv2.ADAPTIVE_THRESH_GAUSSIAN_C` : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.

```python
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
```

--------

## Otsu’s Binarization 

- Automatically calculates a threshold value from image histogram for a bimodal
  image
- Otsu [1]( http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Global_Thresholding_Adaptive_Thresholding_Otsus_Binarization_Segmentations.php )

--------

`cv2.threshold()` function is used, but pass an extra flag, `cv2.THRESH_OTSU`

```python
# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```

--------

## Object detection 

Test

## Feature Extraction

- HOG
- SIFT
- SURF

## Test  

```python
if (a > 3) {
  moveShip(5 * gravity, DOWN);
}
if (a > 3) {
  moveShip(5 * gravity, DOWN);
}
if (a > 3) {
  moveShip(5 * gravity, DOWN);
}
if (a > 3) {
  moveShip(5 * gravity, DOWN);
}
if (a > 3) {
  moveShip(5 * gravity, DOWN);
}
if (a > 3) {
  moveShip(5 * gravity, DOWN);
}
if (a > 3) {
  moveShip(5 * gravity, DOWN);
}
```

------------------

fruit| price
-----|-----:
apple|2.05
pear|1.37
orange|3.09

------------------

| The limerick packs laughs anatomical
| In space that is quite economical.
|    But the good ones I've seen
|    So seldom are clean
| And the clean ones so seldom are comical

| 200 Main St.
| Berkeley, CA 94718


- Eat eggs
- Drink coffee

# In the evening

## Dinner

This is an [inline link](/url), and here's [one with
a title](http://fsf.org "click here for a good time!").

------------------

See [my website][].

[my website]: http://foo.bar.baz
[my label 1]: /foo/bar.html  "My title, optional"
[my label 2]: /foo
[my label 3]: http://fsf.org (The free software foundation)
[my label 4]: /bar#special  'A title in single quotes'

