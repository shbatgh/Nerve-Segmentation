# Delete all the images in Inference/IN if they do not end with 128x128.png

import os
from PIL import Image

directoryIn = 'Inference/IN'

# Get all .tif files in directoryIn
baseFileNamesIn = [os.path.join(directoryIn, name)
                        for name in os.listdir(directoryIn)
                            if name.endswith((".png", ".png"))]

# Process all images
for fullFileNameIn in baseFileNamesIn:
    # Open input image
    imageIn = Image.open(fullFileNameIn)

    # Resize image
    imageIn = imageIn.resize((128, 128))

    # Delete the Imageif it does not end with 128x128.png
    if not fullFileNameIn.endswith("_128x128.png"):
        os.remove(fullFileNameIn)

    