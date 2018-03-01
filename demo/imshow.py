import cv2
import sys

image = cv2.imread(sys.argv[1])
print type(image)
cv2.imshow('image', image)
cv2.waitKey(0)
