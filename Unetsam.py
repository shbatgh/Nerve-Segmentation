#!/usr/bin/env python
# coding: utf-8

# ## Import & GPU check & Path
if __name__ == '__main__':
    from fastai import *
    from fastai.vision import models, get_transforms, imagenet_stats, ImageImageList, ResizeMethod, MSELossFlat, unet_learner, NormType
    from fastai.callback import *
    from fastai.vision.gan import GANLearner
    import torch
    import pathlib
    import os
    import fastai

    print(fastai.__version__)
    print(torch.__version__)

    torch.cuda.get_device_name(0)

    #input
    path_lr1 = pathlib.Path('C:/Users/grand/dev/internship2023/IN')

    #output
    path_hr1 = pathlib.Path('C:/Users/grand/dev/internship2023/OUT')

    len(os.listdir('C:/Users/grand/dev/internship2023/IN'))
    len(os.listdir('C:/Users/grand/dev/internship2023/OUT'))

    # ## Model & Batch size
    bs,size = 32,256
    arch = models.resnet34

    # ## Prepare data
    src = ImageImageList.from_folder(path_lr1).split_by_rand_pct(0.1, seed=42)

    print(path_hr1)

    def get_data(bs,size):
        data = (src.label_from_func(lambda x: path_hr1/x.name)
            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True, resize_method=ResizeMethod.PAD, padding_mode='zeros')
            .databunch(bs=bs, num_workers=0).normalize(imagenet_stats, do_y=True))

        data.c = 3
        return data

    data_gen = get_data(bs,size)

    # ## Create Learner

    wd = 1e-3


    y_range = (-3.,3.)


    loss_gen = MSELossFlat()


    def create_gen_learner():
        return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                            self_attention=True, y_range=y_range, loss_func=loss_gen)


    learn_gen = create_gen_learner()


    # ## Learning

    learn_gen.unfreeze()


    learn_gen.data.batch_size = 16
    learn_gen.fit_one_cycle(1, max_lr=1e-3, pct_start=0.8)


    print(learn_gen.summary())



    ##### Additional training (optional) #####

    #learn_gen.lr_find()
    #learn_gen.unfreeze()

    #learn_gen.data.batch_size = 8
    #learn_gen.fit_one_cycle(5, slice(2e-5,wd/5))


    ### Curves
    curve = learn_gen.recorder.plot(return_fig=True)

    curve2 = learn_gen.recorder.plot_losses(return_fig=True)


    ### Save one plot
    import csv

    # Get the Axes object
    ax = curve.axes[0]      ##############################CURVE

    # Get the Line2D objects (the plotted lines) from the Axes
    lines = ax.lines

    # Loop through the lines and extract the data
    for line in lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        print("x_data:", x_data)
        print("y_data:", y_data)

    # Create a list of tuples with the x and y values
    data = list(zip(x_data, y_data))

    with open('C:/Users/grand/dev/internship2023/recorderPlot339.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Optionally, you can write a header row
        writer.writerow(['x_data', 'y_data'])

        # Write the data rows
        writer.writerows(data)


    # ## Save two plots

    import csv

    # Get the Axes object
    ax = curve2.axes[0]          ##############################CURVE2

    # Get the Line2D objects (the plotted lines) from the Axes
    lines = ax.lines

    # Extract the data from the lines
    data = []
    for line in lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        data.append(list(zip(x_data, y_data)))

    # Save the data to a CSV file
    with open('C:/Users/grand/dev/internship2023/recorderPlotLosses339.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write a header row
        writer.writerow(['x_data_plot1', 'y_data_plot1', 'x_data_plot2', 'y_data_plot2'])

        # Write the data rows
        max_len = max(len(data[0]), len(data[1]))
        for i in range(max_len):
            row = []
            for plot_data in data:
                if i < len(plot_data):
                    row.extend(plot_data[i])
                else:
                    row.extend([None, None])
            writer.writerow(row)


    ### Show results
    learn_gen.show_results(rows=100)

    ### Save model
    learn_gen.path = pathlib.Path('C:/Users/grand/dev/internship2023/TRAINING/ForTraining3/FOR ARTICLE/INFERENCE/NEW_MODELS/34_1/U_net')
    learn_gen.export('U-net-test')
    learn_gen.save('U-net-test')


    ### Load model

    learn_gen.path = pathlib.Path('C:/Users/grand/dev/internship2023/TRAINING/ForTraining3/FOR ARTICLE/INFERENCE/NEW_MODELS/34_1/U_net/')


    learn_gen.load('U-net-test')


    ### New Inference

import glob, os
from fastai.vision import pil2tensor, open_image
import skimage
import numpy as np
import cv2
import re
import PIL
from PIL import Image 
from natsort import natsorted
from skimage import img_as_ubyte, filters
#from skimage.filters import unsharp_mask
from tqdm import tqdm

def NNprocessing(PATH_IN, model, PATH_OUT):

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    numbers = re.compile(r'(\d+)')

    k = 0
    minv_global = 1000;
    maxv_global = -1000;
    for infile in sorted(glob.glob(PATH_IN + '*.png'), key = numericalSort):
        file, ext = os.path.splitext(infile)
        with Image.open(infile) as im:

            pic = pil2tensor(im, np.float32)
            pic2 = torch.cat((pic, pic,pic), dim=0)
            pic3 = torch.div(pic, 255)

            inf = model.predict(open_image(infile))

            #Convert from Fast ai to numpy and convert the dimension
            inf_numpy = inf[2].numpy()
            inf_numpy = inf_numpy[0,:,:]

            minv = np.amin(inf_numpy)
            maxv = np.amax(inf_numpy)
            if minv < minv_global:
              minv_global = minv
            if maxv > maxv_global:
              maxv_global = maxv

            #Convert from numpy to Pillow
            inf_numpy = (255 * (inf_numpy - minv_global) / (maxv_global - minv_global)).astype(np.uint8)
            im2 = PIL.Image.fromarray(inf_numpy)

            #Save mosaics
            #skimage.io.imsave(os.path.join(PATH_OUT, f'{k}.png'), im2)
            im2.save(PATH_OUT+'{0}.png'.format(k))
            print(PATH_OUT)
            k = k + 1


def generate_mosaic_images(imageIn, sizeInput, mosaicSize, step, directoryMosaicIn):

    MosaicCombinedSize = round(sizeInput/mosaicSize)*mosaicSize

    numOverlap = (MosaicCombinedSize - sizeInput)/step + 1
    if numOverlap != round(numOverlap):
        print("Error. Numoverlap should be integer")
    numOverlap = int(numOverlap)

    # Mosaic
    stack = np.zeros((MosaicCombinedSize, MosaicCombinedSize, numOverlap*numOverlap))

    k = 0
    for i in range(numOverlap):
        for j in range(numOverlap):
            im = np.zeros((MosaicCombinedSize, MosaicCombinedSize))
            im[i*step : sizeInput+i*step , j*step : sizeInput+j*step] = imageIn
            stack[:,:,k] = im
            k += 1

    mosaic = []
    numMosaic = MosaicCombinedSize // mosaicSize

    for k in range(numOverlap*numOverlap):
        mosaic.append(skimage.util.view_as_blocks(stack[:,:,k], (mosaicSize, mosaicSize)))

    n = 0
    for k in range(numOverlap*numOverlap):
        for ii in range(numMosaic):  # number of mosaic images
            for jj in range(numMosaic):
                img = mosaic[k][ii, jj, :, :]
                img = img / 255
                img_ubyte = img_as_ubyte(img)
                #mixed = np.concatenate((img_ubyte, img_ubyte), axis=1) # Should be commented for U-net
                rgbMosaic = np.stack((img_ubyte, img_ubyte, img_ubyte), axis=2)
                skimage.io.imsave(os.path.join(directoryMosaicIn, f'{n}.png'), rgbMosaic)
                n += 1


def merge_mosaic_images(directoryMosaicOut, directoryOut, sizeInput, mosaicSize, step):

    MosaicCombinedSize = round(sizeInput/mosaicSize)*mosaicSize

    numOverlap = (MosaicCombinedSize - sizeInput)/step + 1
    if numOverlap != round(numOverlap):
        print("Error. Numoverlap should be integer")
    numOverlap = int(numOverlap)

    sizeTotal = MosaicCombinedSize + step*(numOverlap - 1)
    numMosaic = MosaicCombinedSize // mosaicSize

    # Read images
    baseFileNamesIn = natsorted([f for f in os.listdir(directoryMosaicOut) if f.endswith('.png')])
    numberOfImageFilesIn = len(baseFileNamesIn)
    imageIn = np.zeros((mosaicSize, mosaicSize, numberOfImageFilesIn))

    for i in range(numberOfImageFilesIn):
        fullFileNameIn = os.path.join(directoryMosaicOut, baseFileNamesIn[i])
        temp = cv2.imread(fullFileNameIn, cv2.IMREAD_GRAYSCALE)
        imageIn[:,:,i] = temp[:, :mosaicSize]

    # Separate into mosaics
    mosaicOut = np.empty((numOverlap*numOverlap, numMosaic, numMosaic), dtype=object)

    for k in range(numOverlap*numOverlap):
        for ii in range(numMosaic):
            for jj in range(numMosaic):
                n = numMosaic*numMosaic*k + numMosaic*ii + jj
                mosaicOut[k, ii, jj] = imageIn[:,:,n]

    # Convert to stack of images
    imageOut = np.zeros((mosaicSize*numMosaic, mosaicSize*numMosaic, numOverlap*numOverlap))

    for k in range(numOverlap*numOverlap):
        imageOut[:,:,k] = np.block([[mosaicOut[k, ii, jj] for jj in range(numMosaic)] for ii in range(numMosaic)])

    # Sharpen all images
    #def unsharp_mask(image, kernel_size=(3, 3), sigma=2, amount=2.0, threshold=0):
    #    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    #    sharpened = float(amount + 1) * image - float(amount) * blurred
    #    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    #    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    #    sharpened = sharpened.round().astype(np.uint8)
    #    if threshold > 0:
    #        low_contrast_mask = np.absolute(image - blurred) < threshold
    #        np.copyto(sharpened, image, where=low_contrast_mask)
    #    return sharpened
    #imageOut_sharp = np.zeros((mosaicSize*numMosaic, mosaicSize*numMosaic, 25))

    # Sharpen all images
    #def sharpen_image(image, radius=1.0, amount=1.0):
        # Apply the unsharp mask
    #    result = filters.unsharp_mask(image, radius, amount)
        # Convert float image back to uint8
   #     result = img_as_ubyte(result)
   #     return result
    imageOut_sharp = np.zeros((mosaicSize*numMosaic, mosaicSize*numMosaic, numOverlap*numOverlap))
    for k in range(numOverlap*numOverlap):
        # Convert the image to floating point format between 0 and 1
        image_float = imageOut[:,:,k] / 255.0
        # Apply unsharp mask
        sharpened = filters.unsharp_mask(image_float, radius=1.0, amount=4.0)     #Set the Mask strength
        # Scale the image back to the range of np.uint8
        sharpened_scaled = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
        imageOut_sharp[:,:,k] = sharpened_scaled


   # for k in range(25):
    #    sharpened = filters.unsharp_mask(imageOut[:,:,k], radius=1.0, amount=1.0)
   #     print(np.max(imageOut[:,:,k]))
        # Normalize the data to 0-1 range
    #    sharpened_normalized = (sharpened - np.min(sharpened)) / (np.max(sharpened) - np.min(sharpened))
        # Scale the data to the range 0-255 and convert to uint8
    #    sharpened_scaled = (sharpened_normalized * 255).astype(np.uint8)
   #     imageOut_sharp[:,:,k] = sharpened_scaled
        #imageOut_sharp[:,:,k] = img_as_ubyte(imageOut_sharp[:,:,k])
        #imageOut_sharp[:,:,k] = unsharp_mask(imageOut[:,:,k])

    # Align stack of images
    im = np.zeros((sizeTotal, sizeTotal, numOverlap*numOverlap))
    for ii in range(numOverlap):
        for jj in range(numOverlap):
            k = numOverlap*ii + jj
            corner = (sizeTotal - MosaicCombinedSize) // 2
            start_x = corner - (ii - (numOverlap - 1)//2)*step
            end_x = min(start_x + MosaicCombinedSize, sizeTotal)
            start_y = corner - (jj - (numOverlap - 1)//2)*step
            end_y = min(start_y + MosaicCombinedSize, sizeTotal)
            # Trim imageOut if necessary
            #trimmed_imageOut = imageOut_sharp[:end_x-start_x, :end_y-start_y, k]
            #im[start_x:end_x, start_y:end_y, k] = trimmed_imageOut
            im[start_x:end_x, start_y:end_y, k] = imageOut_sharp[:,:,k]

    # Remove the outlier frames (> 3 STD from the mean)
    mean_im = np.mean(im, axis=(0,1))
    outliers = np.abs(mean_im - np.mean(mean_im)) > 3 * np.std(mean_im)
    im = im[:,:,~outliers]

    return im


#directoryMosaicIn = 'C:/Users/grand/dev/internship2023/Inference/mosaicIN'
#directoryMosaicOut = 'C:/Users/grand/dev/internship2023/Inference/mosaicOUT'
#NNprocessing(directoryMosaicIn, learn_gen, directoryMosaicOut)

if __name__ == '__main__':

    import glob, os
    import skimage
    import numpy as np
    import cv2
    import re
    from PIL import Image as ImagePIL
    from natsort import natsorted
    from skimage import img_as_ubyte, filters
    #from skimage.filters import unsharp_mask
    from tqdm import tqdm

    directoryIn = 'C:/Users/grand/dev/internship2023/Inference/IN'
    directoryMosaicIn = 'C:/Users/grand/dev/internship2023/Inference/mosaicIN\\'
    directoryMosaicOut = 'C:/Users/grand/dev/internship2023/Inference/mosaicOUT\\'
    directoryOut = 'C:/Users/grand/dev/internship2023/Inference/OUT'


    # Get all .png files in directoryIn and its subdirectories
    baseFileNamesIn = [os.path.join(root, name)
                    for root, dirs, files in os.walk(directoryIn)
                    for name in files
                    if name.endswith((".png"))]

    # Process all images
    for fullFileNameIn in tqdm(baseFileNamesIn, desc="Processing images"):
        imageIn = skimage.io.imread(fullFileNameIn)
        sizeInput = imageIn.shape[0]
        #imageIn1 = imageIn[:,:sizeInput,0]
        #imageIn2 = imageIn[:,sizeInput:2*sizeInput,0]  # input

        # Processing
        mosaicSize=256
        step = 8
        generate_mosaic_images(imageIn, sizeInput, mosaicSize, step, directoryMosaicIn)
        NNprocessing(directoryMosaicIn, learn_gen, directoryMosaicOut)
        im = merge_mosaic_images(directoryMosaicOut, directoryOut, sizeInput, mosaicSize, step)

        # Average overlapping sub-images
        img = np.mean(im, axis=2) / 255.0
        # Crop the central sizeInput portion
        height, width = img.shape
        start_row = height//2 -sizeInput//2
        start_col = width//2 - sizeInput//2
        img_cropped = img[start_row:start_row+sizeInput, start_col:start_col+sizeInput]

        RGBout = np.stack([img_cropped, img_cropped, img_cropped], axis=2)
        RGBout = img_as_ubyte(RGBout)  # Convert image data to uint8

        # Create corresponding output directory if it doesn't exist
        relPathIn = os.path.relpath(fullFileNameIn, directoryIn)
        fullFileNameOut = os.path.join(directoryOut, relPathIn)
        os.makedirs(os.path.dirname(fullFileNameOut), exist_ok=True)

        cv2.imwrite(fullFileNameOut, RGBout)
