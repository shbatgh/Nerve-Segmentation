# internship2023
This project is a collection of tools and utilities for image processing and analysis. It includes the following components:

## unetsam
unetsam is a Python package that provides an implementation of the U-Net neural network architecture for image segmentation using the fastai library. It includes pre-trained models for a variety of image segmentation tasks, as well as tools for training new models on custom datasets. 

## gui
gui.py is a graphical user interface for the unetsam package. It provides a simple and intuitive interface for loading images, selecting pre-trained models, and performing image segmentation. The GUI is built using the ctk library, which provides a set of high-level widgets and tools for building desktop applications in Python.

## tiftopng
tiftopng.py is a command-line utility for converting TIFF images to PNG format. It uses the Pillow library to read and write image files, and provides a simple and efficient way to convert large collections of TIFF images to PNG format for use with other tools and utilities.

# Usage
Clone this repository using this command:
```bash
git clone https://github.com/shbatgh/internship2023.git && cd internship2023
```

Install the conda environment using the provided environment.yml file and this command:
```bash
conda env create -f environment.yml -n NAME_OF_ENVIRONMENT
```

Activate the conda environment using this command:
```bash
conda activate NAME_OF_ENVIRONMENT
```

Run the GUI using this command:
```bash
python gui.py
```

Run unetsam using this command:
```bash
python unetsam.py
```
