# Convert all images in inf/IN to black and white

import os
import cv2

# Define the path to the input directory
input_dir = 'inf/IN'

# Define the path to the output directory
output_dir = 'inf/IN'

# Loop over all the files in the input directory
for filename in os.listdir(input_dir):
    # Read the input image
    img = cv2.imread(os.path.join(input_dir, filename))
    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Save the output image
    cv2.imwrite(os.path.join(output_dir, filename), gray)