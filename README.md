# Flatiron Capstone Project: Automated Breast Cancer Metastasis Detection

## Project Members:

* __Marissa Eppes__

## Goal: 

__Automate breast cancer metastasis detection in digitized whole slide images (WSI) of lymph node biopsies. This project makes use of pathologist-annotated WSI, on which regions of tumor are meticulously outlined. 256 x 256 pixel "tiles" of cancerous tissue are extracted from inside the outlined regions to contribute to the "Cancer" class, while non-cancerous tiles are similarly extracted from outside the outlined regions to contribute to the "Non-Cancer" class. Using these tiles as data units, a binary image classifier is developed and trained using Convolutional Neural Networks (CNNs). The model is capable of processing a "tile" at a time and classifying it as cancer or non-cancer.__

__The big-picture goal is to take an un-annotated, unobserved WSI, break it up into potentially hundreds of thousands of these tiles, pass each tile through the model to perform cancer/non-cancer classification, and remap the tiles onto their original locations within the WSI in heatmap format, indicating the "hot spots" for cancer.__


## Files in Repository:

### Jupyter Notebooks

* __tile_function_development.ipynb__ - Explains the thought process and algorithms used in extracting the tissue "tiles" used as data units for the convolutional neural network (CNN) image classifier. Tests several of the functions used to perform the extraction.

* __cnn_notebook.ipynb__ - Walks through the thought process and steps to tune and train a convolutional neural network (CNN) image classifier. Performs the training on the final chosen model.

* __results.ipynb__ - Tests and reports the results of the best Convolutional Neural Network (CNN) Image Classifier as determined in the "Convolutional Neural Network Development" notebook. The notebook also shows example heatmap outputs.

### Supporting Python Scripts

* __train_tiles.py__ - Extracts and creates cancer tile images and normal tile images for the purpose of training the model

* __test_tiles.py__ - Extracts and creates cancer tile images and normal tile images for the purpose of testing the model

* __cnn_prep.py__ - Gathers tiles and prepares them to be passed through CNN model

* __slide_corner_coords.py__ - Stores start and stop coordinates for each slide for further analysis. Start and stop coordinates are chosen and entered manually for each slide.

* __predict.py__ - Consists of functions used to make predictions on test data and evaluate the model

* __visualizations.py__ - Consists of functions used to make visualizations to summarize the results of the project. Includes Accuracy vs. Epoch plot, Confusion Matrix, and Heatmaps

### Visual Aids

* __modeling_diagram.png__ - Diagram summarizing the model development and selection process

* __epochs.png__ - Accuracy vs. Epoch plot for final model. Shows epoch where Image Augmentation is introduced.

* __test_021_heatmap.png__ - Example heatmap based on model predictions

* __test_027_heatmap.png__ - Example heatmap based on model predictions

* __test_040_heatmap.png__ - Example heatmap based on model predictions

### Final Presentation

* __presentation.pdf__ - Final presentation slides




