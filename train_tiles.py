#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:03:33 2019

This module takes in whole slide images and their corresponding XML files,
which contain coordinates for outlined cancer regions. This module breaks the
whole slide images up into "tiles", classifies the tiles as either "cancer" or
"normal" based on pathologist annotations, and creates cancer tile images and
normal tile images for the purpose of model training.

@author: marissaeppes
"""

import xml.etree.ElementTree
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiresolutionimageinterface as mir
import cv2


def parse_xml(tumor_str):
    """
    Inputs name of file for a single slide as a string (i.e. "tumor_009").
    Loads XML file for this slide containing coordinates of polygons outlining
    cancerous regions from ASAP software. Extracts coordinates for each region
    and returns a dictionary of coordinates for each cancerous region, along
    with region name and group.

    :param tumor_str: name of train slide, formatted as "tumor_xxx"
    :return: dictionary of polygons for input slide
    """
    path = '/home/ubuntu/tumor/xml/' + tumor_str + '.xml'
    tree = xml.etree.ElementTree.parse(path)
    polygon_dict = {}
    for i, _ in enumerate(tree.getroot().getchildren()[0].getchildren()):
        x_xml = []
        y_xml = []
        group = tree.getroot().getchildren()[0].getchildren()[i].items()[2][1]
        item_root = tree.getroot().getchildren()[0].getchildren()[i]
        for coord in item_root.getchildren()[0].getchildren():
            x_xml.append(float(coord.items()[1][1]))
            y_xml.append(float(coord.items()[2][1]))
        polygon_dict[i] = {'name': i,
                           'group': group,
                           'x_xml': x_xml,
                           'y_xml': y_xml}
    return polygon_dict


def make_mask(upper_left_coord, lower_right_coord):
    """
    Inputs upper left coordinates and lower right coordinates to select the
    region of analysis, which was chosen manually for each slide. Returns a
    NumPy array of zeros with shape corresponding to input coordinates.

    :param upper_left_coord: tuple of starting pixel coordinates (floats,
                             lowest-value x and y pixel values needed to
                             analyze tissue region)
    :param lower_right_coord: tuple of ending pixel coordinates (floats,
                              highest-value x and y pixel values needed to
                              analyze tissue region)
    :return: 2D array of zeros with shape corresponding to input coordinates
    """
    y_range = int(lower_right_coord[1]) - int(upper_left_coord[1])
    x_range = int(lower_right_coord[0]) - int(upper_left_coord[0])
    mask = np.zeros((y_range, x_range), dtype='int8')
    return mask


def add_polygons_to_mask(mask, polygon, upper_left_coord):
    """
    Inputs a mask (2D array of zeros), a polygon dictionary item, and upper
    left coordinates bounding the region of analysis for a particular slide.
    Function adds polygon to mask and "fills" it by assigning appropriate
    elements of the mask a value of 1. If polygon is an "inner" polygon, which
    is designated with a group number of 2 (meaning it highlights a normal
    region, not a cancer region), function assigns appropriate elements of mask
    with a value of 0.

    Raw polygon coordinates are normalized to correspond to selected region of
    analysis by subtracting polygon coordinates by chosen upper left
    coordinates (a.k.a. starting coordinates).

    :param mask: 2D array of zeros and/or ones corresponding to any pre-filled
                 polygons elements
    :param polygon: an item from a polygon dictionary, consisting of polygon x
                    and y pixel coordinates and group number
    :param upper_left_coord: tuple of starting pixel coordinates (floats,
                             lowest-value x and y pixel values needed to
                             analyze tissue region)
    :return: 2D array passed in as input argument, updated with "filled"
             polygon elements assigned values of 1.
    """
    x_xml = polygon['x_xml']
    y_xml = polygon['y_xml']
    x_adj = [np.int32((np.round(item))) -
             upper_left_coord[0] for item in x_xml]
    y_adj = [np.int32((np.round(item))) -
             upper_left_coord[1] for item in y_xml]
    adj_coords = list(zip(x_adj, y_adj))
    points = np.array([adj_coords])
    if polygon['group'] == '_0' or polygon['group'] == '_1':
        cv2.fillPoly(mask, points, 1)
    else:
        cv2.fillPoly(mask, points, 0)
    return mask


def map_slide_coords(upper_left_coord, lower_right_coord,
                     tile_side_length=256):
    """
    Inputs starting and ending coordinates for whole tissue region of a
    particular slide, as well as desired length for the sides of the tiles.
    Creates a grid (2D array), whose elements are starting coordinates (upper
    left) for each tile to be processed in the model.

    :param upper_left_coord: tuple of starting pixel coordinates (floats,
                             lowest-value x and y pixel values needed to
                             analyze tissue region)
    :param lower_right_coord: tuple of ending pixel coordinates (floats,
                              highest-value x and y pixel values needed to
                              analyze tissue region)
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :return: 2D array corresponding to a particular slide, each element is a
             tuple of starting coordinates for each tile to be processed in the
             model
    """
    step = tile_side_length
    x_grid = np.arange(upper_left_coord[0], lower_right_coord[0], step)
    y_grid = np.arange(upper_left_coord[1], lower_right_coord[1], step)
    x_x, y_y = np.meshgrid(x_grid, y_grid)
    slide_coord_grid = np.dstack((x_x, y_y))
    return slide_coord_grid


def decide_tiles(mask, slide_tumor_grid, tile_side_length,
                 min_percent_tumor=1.0, max_percent_normal=0, plot=False):
    """
    Inputs a mask for a particular slide with all polygons filled, a grid of
    starting tile coordinates (upper left of each tile) corresponding to the
    shape of the mask, the length of each side of the tiles, a threshold for
    the minimum percent of cancer tissue needed per tile to be classified as a
    "cancer" tile, a threshold for the maximum percent of cancer tissue allowed
    per tile to be classified as a "normal" tile, as well as boolean argument
    allowing the option to plot the mask of each tile.

    Model iterates through grid of starting tile coordinates and aligns each
    tile to its corresponding mask coordinates, which enable the percentage of
    cancer tissue in each tile to be calculated. If the percentage of cancer
    tissue in each tile is equal to or above the cancer classification
    threshold, tile starting coordinates are appended to a list of coordinates
    for the cancer class. If the percentage of cancer tissue in each tile is
    equal to or below the normal classification threshold, coordinates are
    appended to a list of coordinates for the normal class. Coordinates not
    meeting either threshold are discarded for the purpose of training.

    If plot is set to "True", the portion of the mask corresponding to each
    tile is shown. This is only recommended for troubleshooting.

    :param mask: 2D array of zeros and/or ones corresponding to any pre-filled
                 polygons elements
    :param slide_tumor_grid: 2D array corresponding to a particular slide, each
                             element is a tuple of starting coordinates for
                             each tile to be processed in the model
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :param min_percent_tumor: float between 0 and 1.0, inclusive. Set to 1.0 as
                              default
    :param max_percent_normal: float between 0 and 1.0, inclusive. Set to 0 as
                               default
    :param plot: boolean, True --> the portion of the mask corresponding to
                 each tile is shown, set to False by default
    :return: list of starting tile coordinates (tuple) corresponding to the
             normal class and a list of starting tile coordinates (tuple)
             corresponding to the cancer class
    """
    x_side = tile_side_length
    y_side = tile_side_length
    y_i = 0
    x_i = 0
    list_of_tumor_coords = []
    list_of_normal_coords = []
    while y_side < len(mask[:]):
        while x_side < len(mask[0][:]):
            percent_tumor = np.mean(mask[y_side - tile_side_length:y_side,
                                         x_side - tile_side_length:x_side])
            if percent_tumor >= min_percent_tumor:
                list_of_tumor_coords.append(
                    tuple([int(i) for i in slide_tumor_grid[y_i, x_i]]))
                if plot:
                    print('Coordinates: ', slide_tumor_grid[y_i, x_i])
                    print('Percent Tumor: ', percent_tumor)
                    plt.imshow(mask[y_side - tile_side_length:y_side,
                                    x_side - tile_side_length:x_side])
                    plt.show()
                else:
                    pass
            elif percent_tumor <= max_percent_normal:
                list_of_normal_coords.append(
                    tuple([int(i) for i in slide_tumor_grid[y_i, x_i]]))
            else:
                pass
            x_side += tile_side_length
            x_i += 1
        x_i = 0
        x_side = tile_side_length
        y_side += tile_side_length
        y_i += 1
    return list_of_normal_coords, list_of_tumor_coords


def filter_white(tumor_str, coord, threshold=225, tile_side_length=256,
                 level=0):
    """
    Inputs the name of the slide of interest as a string (i.e. "tumor_009"), a
    single tuple or list of tuples of starting tile coordinates, and an average
    RGB threshold, beyond which the tile is considered to be whitespace instead
    of tissue. Also inputs the length of each side of the tiles, as well as a
    level, which is a multiresolutionimageinterface module argument for tile
    sampling (0 is default).

    For a single tuple of starting tile coordinates, the corresponding tile of
    tissue is sampled from the slide using the multiresolutionimageinterface
    module and the average RGB value is calculated across each color layer. If
    the average RGB value is above the set threshold, boolean "False" is
    returned, indicating that this particular tile is whitespace and not
    tissue.

    For a list of tuples of starting tile coordinates, the corresponding tiles
    of tissue are sampled from the slide using the
    multiresolutionimageinterface module and the average RGB values are
    calculated across each color layer per tile. For each tile, if the average
    RGB value is above the set threshold, the coordinates are discarded. If the
    average RGB value is equal to or below the set threshold, the coordinates
    for this particular tile are appended to a list of coordinates to keep for
    further analysis. This list of coordinates to keep is returned.

    :param tumor_str: name of train slide, formatted as "tumor_xxx"
    :param coord: either a single tuple of two coordinates (floats) or a list
                  of tuples of two coordinates (floats)
    :param threshold: int, between 0 and 255, inclusive
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :param level: int, between 0 and 9, inclusive. Set to 0 by default,
                  which is recommended.
    :return: boolean for single coordinates tuple (False means tile is
             whitespace), list of coordinates for tiles containing tissue if
            input coordinates are a list of tuples
    """
    reader = mir.MultiResolutionImageReader()
    path = '/home/ubuntu/tumor/tif/'
    mr_image = reader.open(path + tumor_str + '.tif')
    if isinstance(coord, tuple):
        x_coord = coord[0]
        y_coord = coord[1]
        image_patch = mr_image.getUCharPatch(int(x_coord), int(y_coord),
                                             tile_side_length,
                                             tile_side_length, level)
        avg_rgb = np.mean(image_patch)
        if avg_rgb >= threshold:
            return False
        else:
            return True
    elif isinstance(coord, list):
        list_to_keep = []
        for x_coord, y_coord in coord:
            image_patch = mr_image.getUCharPatch(int(x_coord), int(y_coord),
                                                 tile_side_length,
                                                 tile_side_length, level)
            avg_rgb = np.mean(image_patch)
            if avg_rgb < threshold:
                list_to_keep.append((x_coord, y_coord))
            else:
                pass
        return list_to_keep
    else:
        print('Error')
        return None


def get_tiles(tumor_str, tile_side_length=256, min_percent_tumor=1.0,
              max_percent_normal=0, white_threshold=225, level=0):
    """
    Inputs the name of the slide of interest as a string (i.e. "tumor_009"),
    the length of each side of the tiles, a threshold for the minimum percent
    of cancer tissue needed per tile to be classified as a "cancer" tile, a
    threshold for the maximum percent of cancer tissue allowed per tile to be
    classified as a "normal" tile, an average RGB threshold, beyond which the
    tile is considered to be whitespace instead of tissue, and a level, which
    is a multiresolutionimageinterface module argument for tile sampling (0 is
    default).

    Function loads json of starting and ending coordinates, which were manually
    determined and entered with the slide_corner_coords module. The upper left
    coordinates and lower right coordinates for the slide of interest are
    extracted. The aforementioned polygon and mask functions are run for each
    polygon on the slide, creating a complete mask. The remaining
    aforementioned functions are executed, which allow tiles to be classified
    as "cancer" or "normal" (or discarded if no thresholds are met). Tiles
    classified as "normal" are then run through the whitespace filter function.

    This function returns 1) a list of starting tile coordinates for the
    "normal" class, from which whitespace tiles have been removed, and 2) a
    list of starting tile coordinates for the "cancer" class.

    :param tumor_str: name of train slide, formatted as "tumor_xxx"
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :param min_percent_tumor: float between 0 and 1.0, inclusive. Set to 1.0 as
                              default
    :param max_percent_normal: float between 0 and 1.0, inclusive. Set to 0 as
                               default
    :param white_threshold: int, between 0 and 255, inclusive
    :param level: int, between 0 and 9, inclusive. Set to 0 by default, which
                  is recommended.
    :return: list of filtered starting coordinates for tiles in the "normal"
             class and a list of starting coordinates for tiles in the "cancer"
             class
    """
    with open('/home/ubuntu/tumor/corner_coords/corner_coords.json',
              'r') as json_file:
        coords = json.load(json_file)
    upper_left_coord = tuple(coords[tumor_str]['upper_left_coord'])
    lower_right_coord = tuple(coords[tumor_str]['lower_right_coord'])
    polygons = parse_xml(tumor_str)
    mask = make_mask(upper_left_coord, lower_right_coord)
    for polygon in polygons.values():
        add_polygons_to_mask(mask, polygon, upper_left_coord)
    slide_coord_grid = map_slide_coords(upper_left_coord, lower_right_coord,
                                        tile_side_length)
    norm_coords, tumor_coords = \
        decide_tiles(mask, slide_coord_grid, tile_side_length,
                     min_percent_tumor=min_percent_tumor,
                     max_percent_normal=max_percent_normal)
    filtered_norm_coords = filter_white(tumor_str, norm_coords,
                                        threshold=white_threshold,
                                        tile_side_length=tile_side_length,
                                        level=level)
    return filtered_norm_coords, tumor_coords


def save_coord_jsons(tumor_str, filtered_norm_coords, tumor_coords):
    """
    Saves lists of coordinates previously calculated as json files for slide of
    interest.
    :param tumor_str: name of train slide, formatted as "tumor_xxx"
    :param filtered_norm_coords: list of filtered starting coordinates for
                                 tiles in the "normal" class
    :param tumor_coords: list of starting coordinates for tiles in the "cancer"
                         class
    :return: None
    """
    coord_json = {tumor_str: {'filtered_norm_coords': filtered_norm_coords,
                              'tumor_coords': tumor_coords}}
    path = '/home/ubuntu/tumor/coord_lists/'
    with open(path + tumor_str + '_coords.json', 'w+') as json_file:
        json.dump(coord_json, json_file)


def save_cancer_tile_jpegs(tumor_str, list_of_coords, level=0,
                           tile_side_length=256):
    """
    Inputs the name of the slide of interest as a string (i.e. "tumor_009"), a
    list of starting coordinates to obtain desired cancer tiles, a level, which
    is a multiresolutionimageinterface module argument for tile sampling (0 is
    default), and the length of each side of the tiles.

    Function passes the starting coordinates for each desired cancer tile
    iteratively into the appropriate multiresolutionimageinterface functions to
    sample the image tiles from the slide of interest. Images are saved in
    their proper directory as jpeg files.

    :param tumor_str: name of train slide, formatted as "tumor_xxx"
    :param list_of_coords: list of starting coordinates for tiles in the
                           "cancer" class
    :param level: int, between 0 and 9, inclusive. Set to 0 by default, which
                  is recommended.
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :return: None
    """
    reader = mir.MultiResolutionImageReader()
    wsi = reader.open('/home/ubuntu/tumor/tif/' + tumor_str + '.tif')
    path = '/home/ubuntu/tumor/cancer_tiles/' + tumor_str + '/'
    for x_coord, y_coord in list_of_coords:
        tile = wsi.getUCharPatch(x_coord, y_coord, tile_side_length,
                                 tile_side_length, level)
        image = Image.fromarray(tile)
        image.save(path + tumor_str + '_cancer_' + str(x_coord) + '_' +
                   str(y_coord) + '.jpeg')


def save_normal_tile_jpegs(tumor_str, list_of_coords, level=0,
                           tile_side_length=256):
    """
    Inputs the name of the slide of interest as a string (i.e. "tumor_009"), a
    list of starting coordinates to obtain desired normal tiles, a level, which
    is a multiresolutionimageinterface module argument for tile sampling (0 is
    default), and the length of each side of the tiles.

    Function passes the starting coordinates for each desired normal tile
    iteratively into the appropriate multiresolutionimageinterface functions to
    sample the image tiles from the slide of interest. Images are saved in
    their proper directory as jpeg files.

    :param tumor_str: name of train slide, formatted as "tumor_xxx"
    :param list_of_coords: list of filtered starting coordinates for tiles in
                           the "normal" class
    :param level: int, between 0 and 9, inclusive. Set to 0 by default, which
                  is recommended.
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :return: None
    """
    reader = mir.MultiResolutionImageReader()
    wsi = reader.open('/home/ubuntu/tumor/tif/' + tumor_str + '.tif')
    path = '/home/ubuntu/tumor/normal_tiles/' + tumor_str + '/'
    for x_coord, y_coord in list_of_coords:
        tile = wsi.getUCharPatch(x_coord, y_coord, tile_side_length,
                                 tile_side_length, level)
        image = Image.fromarray(tile)
        image.save(path + tumor_str + '_normal_' + str(x_coord) + '_' +
                   str(y_coord) + '.jpeg')
