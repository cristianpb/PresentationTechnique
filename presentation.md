---
title: Partage de connaissance
author: Cristian Perez
---

## Anatomy of an image classifier

![](images/image-classification-pipeline.jpg){ width=100% }

<div class="notes">
This is my note.
- It can contain Markdown
- like this list
</div>

# Preprocessing 

## Align (Detex) 

Align image to a reference assits the classification algorithm

![](images/deskew1.jpg){ height=100% }

[1](http://docs.opencv.org/trunk/dd/d3b/tutorial_py_svm_opencv.html)
[2](https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/)

-------

Deskewing simple grayscale images can be achieved using image moments (distance and intensity of pixels). 

<div class="notes">
Moments relate distance and intensity of the pixels
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

<div class="notes">
People often think of a learning algorithm as a block box.
In reality, you can assist the algorithm.
If you are building a face recognition system, aligning the images to a reference face leads to improvement. 
A typical alignment operation uses a facial feature detector to align the eyes in every image.

Pour le cas de digits, aligner les numéros améliore les résultats. 
L'inclination de l'écriture peut être corrigé.
Ansi l'algorithme ne doit pas apprendre cette variation entre les chiffres.

This deskewing of simple grayscale images can be achieved using image moments. OpenCV has an implementation of moments and it comes in handy while calculating useful information like centroid, area, skewness of simple images with black backgrounds.

It turns out that a measure of the skewness is the given by the ratio of the two central moments ( mu11 / mu02 ). The skewness thus calculated can be used in calculating an affine transform that deskews the image.
</div>

## Histogram equalization

Adjust image contrast using the image's histogram.

![](images/histogram_equalization.png){ width=100% }

-------

```python
img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)
```

![](images/histeq_numpy1.jpg)
![](images/histeq_numpy2.jpg)

[1](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)

--------

### Color segmentation

- Otsu [1]( http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Global_Thresholding_Adaptive_Thresholding_Otsus_Binarization_Segmentations.php )

### Object detection 

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

