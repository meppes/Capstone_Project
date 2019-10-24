#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:42:00 2019

Module gathers tiles and prepares them to be passed through CNN model. Keeps
Training and Testing, Normal and Cancer tiles all separate.

@author: marissaeppes
"""
import glob
import numpy as np
from PIL import Image


def make_cancer_train_glob(list_of_train_slides):
    """
    Inputs list of train slides (i.e. [tumor_xxx, tumor_xxy, ... ]) and returns
    a list of paths to cancer tiles (jpegs) consolidated among all input
    slides.
    :param list_of_train_slides: list of slides for training ([tumor_xxx,
                                 tumor_xxy, ... ])
    :return: list of paths to cancer tiles for all input slides
    """
    cancer_train_glob = []
    for slide in list_of_train_slides:
        slide_glob = glob.glob("/home/ubuntu/tumor/cancer_tiles/" + slide +
                               "/*.jpeg")
        for item in slide_glob:
            cancer_train_glob.append(item)
    return cancer_train_glob


def make_normal_train_glob(list_of_train_slides):
    """
    Inputs list of train slides (i.e. [tumor_xxx, tumor_xxy, ... ]) and returns
    a list of paths to normal tiles (jpegs) consolidated among all input
    slides.
    :param list_of_train_slides: list of slides for training ([tumor_xxx,
                                 tumor_xxy, ... ])
    :return: list of paths to normal tiles for all input slides
    """
    normal_train_glob = []
    for slide in list_of_train_slides:
        slide_glob = glob.glob("/home/ubuntu/tumor/normal_tiles/" + slide +
                               "/*.jpeg")
        for item in slide_glob:
            normal_train_glob.append(item)
    return normal_train_glob


def make_cancer_test_glob(list_of_test_slides):
    """
    Inputs list of test slides (i.e. [test_xxx, test_xxy, ... ]) and returns a
    list of paths to cancer tiles (jpegs) consolidated among all input
    slides.
    :param list_of_test_slides: list of slides for testing ([test_xxx,
                                test_xxy, ... ])
    :return: list of paths to cancer tiles for all input slides
    """
    cancer_test_glob = []
    for slide in list_of_test_slides:
        slide_glob = glob.glob("/home/ubuntu/test/cancer_tiles/" + slide +
                               "/*.jpeg")
        for item in slide_glob:
            cancer_test_glob.append(item)
    return cancer_test_glob


def make_normal_test_glob(list_of_test_slides):
    """
    Inputs list of test slides (i.e. [test_xxx, test_xxy, ... ]) and returns a
    list of paths to normal tiles (jpegs) consolidated among all input
    slides.
    :param list_of_test_slides: list of slides for testing ([test_xxx,
                                test_xxy, ... ])
    :return: list of paths to normal tiles for all input slides
    """
    normal_test_glob = []
    for slide in list_of_test_slides:
        slide_glob = glob.glob("/home/ubuntu/test/normal_tiles/" + slide +
                               "/*.jpeg")
        for item in slide_glob:
            normal_test_glob.append(item)
    return normal_test_glob


def cancer_train_jpegs_to_arrays(cancer_train_glob, tile_side_length=256,
                                 scale_down=None, seed=None):
    """
    Inputs list of cancer tile image paths and converts them to arrays of shape
    tile_side_length x tile_side_length x 3 (RGB). These individual image
    arrays are stored in an array. Function has an optional scale-down
    argument, which instructs the function to choose a random sample of n tile
    images from the to list of cancer tile image paths to convert. Function
    also has an option to set a seed for random generator if scale-down is
    used.
    :param cancer_train_glob: list of paths to cancer tiles from training set
    :param tile_side_length: int, length of side for tiles, set to 256 by
                             default
    :param scale_down: None for no scale down, int for random sampling (n)
    :param seed: int, sets seed for random generator (optional)
    :return: array of image arrays (i.e. n x 256 x 256 x 3)
    """
    if scale_down is None:
        dim1 = len(cancer_train_glob)
        dim2 = dim3 = tile_side_length
        cancer_train_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(cancer_train_glob):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            cancer_train_set[i] = image_array
    elif isinstance(scale_down, int):
        if seed is not None:
            np.random.seed(seed)
            print('Using Seed: ', seed)
        sample = np.random.choice(cancer_train_glob, scale_down, replace=False)
        dim1 = len(sample)
        dim2 = dim3 = tile_side_length
        cancer_train_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(sample):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            cancer_train_set[i] = image_array
    else:
        print('Error')
        return None

    return cancer_train_set


def normal_train_jpegs_to_arrays(normal_train_glob, tile_side_length=256,
                                 scale_down=None, seed=None):
    """
    Inputs list of normal tile image paths and converts them to arrays of shape
    tile_side_length x tile_side_length x 3 (RGB). These individual image
    arrays are stored in an array. Function has an optional scale-down
    argument, which instructs the function to choose a random sample of n tile
    images from the to list of normal tile image paths to convert. Function
    also has an option to set a seed for random generator if scale-down is
    used.
    :param normal_train_glob: list of paths to normal tiles from training set
    :param tile_side_length: int, length of side for tiles, set to 256 by
                             default
    :param scale_down: None for no scale down, int for random sampling (n)
    :param seed: int, sets seed for random generator (optional)
    :return: array of image arrays (i.e. n x 256 x 256 x 3)
    """
    if scale_down is None:
        dim1 = len(normal_train_glob)
        dim2 = dim3 = tile_side_length
        normal_train_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(normal_train_glob):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            normal_train_set[i] = image_array
    elif isinstance(scale_down, int):
        if seed is not None:
            np.random.seed(seed)
            print('Using Seed: ', seed)
        sample = np.random.choice(normal_train_glob, scale_down, replace=False)
        dim1 = len(sample)
        dim2 = dim3 = tile_side_length
        normal_train_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(sample):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            normal_train_set[i] = image_array
    else:
        print('Error')
        return None

    return normal_train_set


def cancer_test_jpegs_to_arrays(cancer_test_glob, tile_side_length=256,
                                scale_down=None, seed=None):
    """
    Inputs list of cancer tile image paths and converts them to arrays of shape
    tile_side_length x tile_side_length x 3 (RGB). These individual image
    arrays are stored in an array. Function has an optional scale-down
    argument, which instructs the function to choose a random sample of n tile
    images from the to list of cancer tile image paths to convert. Function
    also has an option to set a seed for random generator if scale-down is
    used.
    :param cancer_test_glob: list of paths to cancer tiles from testing set
    :param tile_side_length: int, length of side for tiles, set to 256 by
                             default
    :param scale_down: None for no scale down, int for random sampling (n)
    :param seed: int, sets seed for random generator (optional)
    :return: array of image arrays (i.e. n x 256 x 256 x 3)
    """
    if scale_down is None:
        dim1 = len(cancer_test_glob)
        dim2 = dim3 = tile_side_length
        cancer_test_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(cancer_test_glob):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            cancer_test_set[i] = image_array
    elif isinstance(scale_down, int):
        if seed is not None:
            np.random.seed(seed)
            print('Using Seed: ', seed)
        sample = np.random.choice(cancer_test_glob, scale_down, replace=False)
        dim1 = len(sample)
        dim2 = dim3 = tile_side_length
        cancer_test_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(sample):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            cancer_test_set[i] = image_array
    else:
        print('Error')
        return None

    return cancer_test_set


def normal_test_jpegs_to_arrays(normal_test_glob, tile_side_length=256,
                                scale_down=None, seed=None):
    """
    Inputs list of normal tile image paths and converts them to arrays of shape
    tile_side_length x tile_side_length x 3 (RGB). These individual image
    arrays are stored in an array. Function has an optional scale-down
    argument, which instructs the function to choose a random sample of n tile
    images from the to list of normal tile image paths to convert. Function
    also has an option to set a seed for random generator if scale-down is
    used.
    :param normal_test_glob: list of paths to normal tiles from testing set
    :param tile_side_length: int, length of side for tiles, set to 256 by
                             default
    :param scale_down: None for no scale down, int for random sampling (n)
    :param seed: int, sets seed for random generator (optional)
    :return: array of image arrays (i.e. n x 256 x 256 x 3)
    """
    if scale_down is None:
        dim1 = len(normal_test_glob)
        dim2 = dim3 = tile_side_length
        normal_test_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(normal_test_glob):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            normal_test_set[i] = image_array
    elif isinstance(scale_down, int):
        if seed is not None:
            np.random.seed(seed)
            print('Using Seed: ', seed)
        sample = np.random.choice(normal_test_glob, scale_down, replace=False)
        dim1 = len(sample)
        dim2 = dim3 = tile_side_length
        normal_test_set = np.zeros((dim1, dim2, dim3, 3), dtype='uint8')
        for i, jpeg_path in enumerate(sample):
            image = Image.open(jpeg_path)
            image_array = np.array(image)
            normal_test_set[i] = image_array
    else:
        print('Error')
        return None

    return normal_test_set
