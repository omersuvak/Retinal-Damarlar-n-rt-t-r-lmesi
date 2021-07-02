import cv2
import numpy as np

img = cv2.imread("01_test.jpeg",0)



def Blur(imgA):

    kernel = np.ones((3, 3), np.uint8)
    imgA = cv2.erode(imgA, kernel, iterations=1)

    imgA = cv2.medianBlur(imgA, 5)
    imgA = cv2.GaussianBlur(imgA, (5, 5), 0)

    th2 = cv2.adaptiveThreshold(imgA, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)

    kernel1 = np.ones((2, 2), np.uint8)
    dilate = cv2.dilate(th2, kernel1, iterations=1)

    bitwise = cv2.bitwise_not(dilate, dilate)

    morph = cv2.morphologyEx(bitwise, cv2.MORPH_OPEN, kernel1)


    return morph

def Rotation(imgB,degree):
   row = imgB.shape[0]
   col = imgB.shape[1]

   M = cv2.getRotationMatrix2D((col / 2, row / 2), degree, 1)

   imgB = cv2.warpAffine(imgB, M, (col, row))

   return imgB

def Scaling(imgC,scale_percent):
    width = int(imgC.shape[1] * scale_percent / 100)
    height = int(imgC.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(imgC, dim, interpolation=cv2.INTER_AREA)

    return resized

def Matching(img1,img2):
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    return matching_result




img1 = Blur(img)



img2 = Scaling(img,90)
img2 = Rotation(img2,30)
img2 = Blur(img2)


result = Matching(img1,img2)



cv2.imshow("Result",result)




cv2.waitKey(0)
cv2.destroyAllWindows()