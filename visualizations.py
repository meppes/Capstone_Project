"""
Module consists of functions used to make visualizations to summarize the
results of the project.
"""

import glob
import itertools
import warnings
from PIL import Image
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import ListedColormap
import numpy as np

warnings.filterwarnings('ignore')


def make_training_log_plot():
    """
    Creates plot of training/validation accuracy vs. epoch for the final model.
    Plot shows epoch where Image Augmentation is introduced.
    """
    plt.style.use('seaborn')
    path = '/home/ubuntu/Notebooks/'
    training_log_paths = glob.glob(path + '*.log')
    final_training_log_paths = []
    for item in training_log_paths:
        if 'final' in item:
            final_training_log_paths.append(item)

    order = [int(x[40]) for x in final_training_log_paths]
    ordered_logs = sorted(list(zip(final_training_log_paths, order)),
                          key=lambda x: x[1])
    log_dict = {}
    training_history_df = pd.DataFrame()
    for count, item in enumerate(ordered_logs):
        log_dict[count + 1] = item[0]
        item_df = pd.read_csv(item[0])
        training_history_df = pd.concat([training_history_df, item_df], axis=0)

    fig = plt.figure(figsize=(20, 8))
    plt.plot(range(1, 59), training_history_df['acc'][0:-12] * 100, lw=5,
             color='c')
    plt.plot(range(1, 59), training_history_df['val_acc'][0:-12] * 100, lw=5)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc=2,
               prop={'size': 25})
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('% Accuracy', fontsize=25)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim((0, 58))
    plt.axvline(40, color='mediumorchid', lw=5, alpha=.5)
    plt.axvspan(40, 59, color='mediumorchid', alpha=.3)

    plt.annotate('Introduce Image \n Augmentation', xy=(40.5, 68),
                 xytext=(42, 71), arrowprops=dict(facecolor='black'),
                 fontsize=27)
    plt.savefig('epochs.png')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    Code based on Scikit-Learn example: https://scikit-learn.org/stable/
    auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        pass
    plt.imshow(cm, cmap=cmap, interpolation='nearest', aspect=2)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.axis('off')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, s=cm[j, i],
                 position=(i - .15, .75) if j == 1 else (i - .15, .25),
                 color="white" if cm[i, j] > thresh else "black", fontsize=15)

    plt.text(.1, 1.2, s='Predicted Label', fontsize=17)
    plt.text(-.9, .65, s='True Label', fontsize=17, rotation=90)
    plt.text(-.2, 1.1, s='Normal', fontsize=14)
    plt.text(.85, 1.1, s='Cancer', fontsize=14)
    plt.text(-.7, .3, s='Normal', fontsize=14, rotation=90)
    plt.text(-.7, .8, s='Cancer', fontsize=14, rotation=90)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')


def predictions_for_heatmap(test_str, model):
    """
    Inputs name of test WSI and the model with which to make predictions.
    Makes individual predictions for each tile and returns predictions as array
    Predictions are probabilities (between 1 and 0) for each class (Normal vs.
    Cancer); therefore each prediction is a tuple.
    """
    if test_str == 'test_021':

        normal_test_glob = glob.glob(
            '/home/ubuntu/' + test_str + '_normal/' + test_str + '/*jpeg')
        cancer_test_glob = glob.glob(
            '/home/ubuntu/' + test_str + '_cancer/' + test_str + '/*jpeg')
    else:

        normal_test_glob = glob.glob(
            '/home/ubuntu/test/normal_tiles/' + test_str + '/*jpeg')
        cancer_test_glob = glob.glob(
            '/home/ubuntu/test/cancer_tiles/' + test_str + '/*jpeg')
    total_list = normal_test_glob + cancer_test_glob
    total_length = len(total_list)
    predictions = []
    step = 5000
    while step < total_length + 5000:
        test_slide = []
        for item in total_list[step - 5000:step]:
            im = Image.open(item)
            im_array = np.array(im) / 255
            test_slide.append(im_array)
        test_slide = np.array(test_slide)
        predictions.append(model.predict(test_slide, batch_size=1))
        step = step + 5000
    predictions = np.concatenate(predictions[:])
    return predictions


def make_heatmap(test_str, predictions, find_coords_index, shrink=None,
                 aspect=5, rotate=False):
    """
    Inputs name of test WSI and array of predictions previously calculated.
    Creates heatmap displaying hot spots for cancer as predicted by the model.
    "find_coords_index" might be adjusted for each WSI to capture where the
    coordinates in each file path name are located (i.e. /home/ubuntu/test/
    normal_tiles/test_027/test_027_normal_9878_158653.jpeg would start listing
    coordinates at index 56 within path string).

    The size scale of the colorbar to the right of the heatmap can be adjusted
    based on size of the heatmap to improve appearance and legibility using
    shrink and aspect arguments.

    If rotate is equal to "True", the heatmap is rotated 90 degrees.
    """
    if test_str == 'test_021':

        normal_test_glob = glob.glob(
            '/home/ubuntu/' + test_str + '_normal/' + test_str + '/*jpeg')
        cancer_test_glob = glob.glob(
            '/home/ubuntu/' + test_str + '_cancer/' + test_str + '/*jpeg')
    else:

        normal_test_glob = glob.glob(
            '/home/ubuntu/test/normal_tiles/' + test_str + '/*jpeg')
        cancer_test_glob = glob.glob(
            '/home/ubuntu/test/cancer_tiles/' + test_str + '/*jpeg')

    blob = normal_test_glob + cancer_test_glob
    coords = []
    for item in blob:
        coord_str = item[find_coords_index:]
        coord_str = coord_str[:coord_str.index('.')]
        coords.append([int(x) for x in coord_str.split('_')])
    x_min = min(coords, key=lambda x: x[0])[0]
    x_max = max(coords, key=lambda x: x[0])[0]
    y_min = min(coords, key=lambda x: x[1])[1]
    y_max = max(coords, key=lambda x: x[1])[1]
    grid = np.zeros(
        (int((y_max - y_min) / 256) + 1, int((x_max - x_min) / 256) + 1))
    coords = np.array(coords)
    coords[:, 0] = coords[:, 0] - x_min
    coords[:, 1] = coords[:, 1] - y_min
    coords = coords / 256
    for i, coord in enumerate(coords):
        x = int(coord[0])
        y = int(coord[1])
        if predictions[i][0] > predictions[i][1]:
            grid[y, x] = .5 + predictions[i][0]
        elif predictions[i][0] <= predictions[i][1]:
            grid[y, x] = .5 + predictions[i][0]
        else:
            print(predictions[i])
    pad = 5
    grid_pad = np.concatenate((np.zeros((pad, grid.shape[1])), grid,
                               np.zeros((pad, grid.shape[1]))))
    grid_pad = np.concatenate((
        np.zeros((grid_pad.shape[0], pad)), grid_pad,
        np.zeros((grid_pad.shape[0], pad))), axis=1)
    GnBu = plt.get_cmap('GnBu', 256)
    newcolors = GnBu(np.linspace(0, 1, 256))
    white = np.array([135 / 256, 135 / 256, 175 / 256, 1])

    newcolors[:25, :] = white
    newcmp = ListedColormap(newcolors)
    fig = plt.figure(figsize=(20, 20))

    if rotate:
        plt.imshow(np.rot90(grid_pad), cmap=newcmp, interpolation='nearest')
    else:
        plt.imshow(grid_pad, cmap=newcmp, interpolation='nearest')

    plt.xticks([])
    plt.yticks([])

    matplotlib.rcParams.update({'font.size': 35})

    cmap = plt.get_cmap('GnBu')

    vmin = .5
    vmax = 1.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(np.linspace(1. - (vmax - vmin) / float(vmax), 1, cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        'cut_jet', colors)

    cax, _ = matplotlib.colorbar.make_axes(plt.gca(), pad=0.01, shrink=shrink,
                                           aspect=aspect)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=color_map,
                                            norm=norm, )

    cbar.set_ticks([])
    cbar.set_ticklabels([])

    plt.text(.5, .44, 'Cancer')
    plt.text(.5, 1.52, 'Non-\nCancer')

    plt.savefig(test_str + '_heatmap.png')
