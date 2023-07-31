# Create a black image (384, 384) with 1 diagonal line of 3 pixel thickness

import cv2
import numpy as np

img = np.zeros((384, 384), dtype=np.uint8)
cv2.line(img, (2, 2), (380, 380), 255, 3)

# write it to measure/OUT
cv2.imwrite('measure/OUT/test_image.png', img)