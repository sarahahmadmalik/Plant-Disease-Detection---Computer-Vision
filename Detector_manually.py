import cv2
import numpy as np

# Loading an image of a plant leaf
img = cv2.imread('./sample2.jpg')  

# Converting the image to HSV color space (for color-based segmentation) and grayscale (for SIFT)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Defining the lower and upper bounds of the color for disease
lower_disease = np.array([0, 50, 50])
upper_disease = np.array([20, 255, 255])

# Initializing SIFT detector
sift = cv2.SIFT_create()

# Detecting interest points and computing descriptors using SIFT
kp, descriptors = sift.detectAndCompute(gray, None)

# Drawing interest points on the original image
img_with_keypoints = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Creating a mask based on the color range for disease
mask = cv2.inRange(hsv, lower_disease, upper_disease)

# Bitwise AND to extract diseased regions
diseased_regions = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Original Image', img)

cv2.imshow('Diseased Regions', diseased_regions)

cv2.imshow('Interest Points', img_with_keypoints)

cv2.imwrite('diseased_regions_result.jpg', diseased_regions)

cv2.waitKey(0)
cv2.destroyAllWindows()
