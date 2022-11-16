
import numpy as np
import cv2 as cv
img = cv.imread('coins.jpg', 0)
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#  get circles of 20-100 pixels in diameter
circles = cv.HoughCircles(
    img, cv.HOUGH_GRADIENT, 1, 65, param1=50, param2=60, minRadius=120, maxRadius=200)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
cv.imshow('detected circles', cimg)
cv.waitKey(0)
cv.destroyAllWindows()

#  read the following images from the same folder
#  2, 5,50c, 50old, 10, 1

coin2 = cv.imread('2.png', 0)
coin5 = cv.imread('5.png', 0)
coin50c = cv.imread('50c.png', 0)
coin50old = cv.imread('50old.png', 0)
coin10 = cv.imread('10.png', 0)
coin1 = cv.imread('1.png', 0)

# #  using SIFT to find the most similar coin in the iamge and mark it with is name

sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(coin2, None)
kp3, des3 = sift.detectAndCompute(coin5, None)
kp4, des4 = sift.detectAndCompute(coin50c, None)
kp5, des5 = sift.detectAndCompute(coin50old, None)
kp6, des6 = sift.detectAndCompute(coin10, None)
kp7, des7 = sift.detectAndCompute(coin1, None)

#  using FLANN to match the coins
FLANN_INDEX_KDTREE = 5
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=60)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches2 = flann.knnMatch(des1, des2, k=2)
matches5 = flann.knnMatch(des1, des3, k=2)
matches50c = flann.knnMatch(des1, des4, k=2)
matches50old = flann.knnMatch(des1, des5, k=2)
matches10 = flann.knnMatch(des1, des6, k=2)
matches1 = flann.knnMatch(des1, des7, k=2)

#  store the good matches as per Lowe's ratio test.
good2 = []
good5 = []
good50c = []
good50old = []
good10 = []
good1 = []
for m, n in matches2:
    if m.distance < 0.7 * n.distance:
        good2.append(m)
for m, n in matches5:
    if m.distance < 0.7 * n.distance:
        good5.append(m)
for m, n in matches50c:
    if m.distance < 0.7 * n.distance:
        good50c.append(m)
for m, n in matches50old:
    if m.distance < 0.7 * n.distance:
        good50old.append(m)
for m, n in matches10:
    if m.distance < 0.7 * n.distance:
        good10.append(m)
for m, n in matches1:
    if m.distance < 0.7 * n.distance:
        good1.append(m)

 #  draw the matches  only if the number of matches is greater than 10

img3 = cv.drawMatches(img, kp1, coin2, kp2, good2, None, flags=2)
img4 = cv.drawMatches(img, kp1, coin5, kp3, good5, None, flags=2)
img5 = cv.drawMatches(img, kp1, coin50c, kp4, good50c, None, flags=2)
img6 = cv.drawMatches(img, kp1, coin50old, kp5, good50old, None, flags=2)
img7 = cv.drawMatches(img, kp1, coin10, kp6, good10, None, flags=2)
img8 = cv.drawMatches(img, kp1, coin1, kp7, good1, None, flags=2)

#  show the images
cv.imshow('2', img3)
cv.imshow('5', img4)
cv.imshow('50c', img5)
cv.imshow('50old', img6)
cv.imshow('10', img7)
cv.imshow('1', img8)
cv.waitKey(0)
cv.destroyAllWindows()
