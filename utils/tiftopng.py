import os
from PIL import Image

directoryIn = 'nerve/OUT'

# Get all .tif files in directoryIn
baseFileNamesIn = [os.path.join(directoryIn, name)
                   for name in os.listdir(directoryIn)
                   if name.endswith((".tif", ".tiff"))]

# Process all images
for fullFileNameIn in baseFileNamesIn:
    # Open input image
    imageIn = Image.open(fullFileNameIn)

    # Create corresponding output directory if it doesn't exist
    fullFileNameOut = os.path.splitext(fullFileNameIn)[0] + '.png'
    os.makedirs(os.path.dirname(fullFileNameOut), exist_ok=True)

    # Save output image
    imageIn.save(fullFileNameOut)

    # Delete original tif file
    os.remove(fullFileNameIn)