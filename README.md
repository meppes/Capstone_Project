# Flatiron Capstone Project: Automated Breast Cancer Metastasis Detection

## Project Members:

* __Marissa Eppes__

## Goal: 

__Automate breast cancer metastasis detection in digitized whole slide images (WSI) of lymph node biopsies. This project makes use of pathologist-annotated WSI, on which regions of tumor are meticulously outlined. 256 x 256 pixel "tiles" of cancerous tissue are extracted from inside the outlined regions to contribute to the "Cancer" class, while non-cancerous tiles are similarly extracted from outside the outlined regions to contribute to the "Non-Cancer" class. Using these tiles as data units, a binary image classifier is developed and trained using Convolutional Neural Networks (CNNs). The model is capable of processing a "tile" at a time and classifying it as cancer or non-cancer.__

__The big-picture goal is to take an un-annotated, unobserved WSI, break it up into potentially hundreds of thousands of these tiles, pass each tile through the model to perform cancer/non-cancer classification, and remap the tiles onto their original locations within the WSI in heatmap format, indicating the "hot spots" for cancer.__


## Files in Repository:

* __tile_function_development.ipynb__ - explains the thought process and algorithms used in extracting the tissue "tiles" used as data units for the convolutional neural network (CNN) image classifier. Tests several of the functions used to perform the extraction.

* __cnn_notebook.ipynb__ - walks through the thought process and steps to tune and train a convolutional neural network (CNN) image classifier. Performs the training on the final chosen model.

* __train_tiles.py__ - extracts and creates cancer tile images and normal tile images for the purpose of training the model

* __test_tiles.py__ - extracts and creates cancer tile images and normal tile images for the purpose of testing the model

* __cnn_prep.py__ - gathers tiles and prepares them to be passed through CNN model

* __slide_corner_coords.py__ - stores start and stop coordinates for each slide for further analysis. Start and stop coordinates are chosen and entered manually for each slide.

* __presentation.pdf__ - final presentation slides

## *In Progress*:
* __results.ipynb__
* __visuals.ipynb__



