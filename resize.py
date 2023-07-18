# resize all images in Inference/IN to 128x128
# and save them in Inference/IN_128x128

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
    imageIn = imageIn.resize((96, 96))

    # Save the Image
    fullFileNameOut = os.path.splitext(fullFileNameIn)[0] + '_128x128.png'
    os.makedirs(os.path.dirname(fullFileNameOut), exist_ok=True)
    imageIn.save(fullFileNameOut)

