#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:21:16 2019

This module takes in whole slide images and their corresponding XML files,
which contain coordinates for outlined cancer regions. This module breaks the
whole slide images up into "tiles", classifies the tiles as either "cancer" or
"normal" based on pathologist annotations, and creates cancer tile images and
normal tile images for the purpose of testing the model.

@author: marissaeppes
"""

import xml.etree.ElementTree
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiresolutionimageinterface as mir
import cv2


def parse_xml(test_str):
    """
    Inputs name of file for a single slide as a string (i.e. "test_002").
    Loads XML file for this slide containing coordinates of polygons outlining
    cancerous regions from ASAP software. Extracts coordinates for each region
    and returns a dictionary of coordinates for each cancerous region, along
    with region name and group.

    :param test_str: name of test slide, formatted as "test_xxx"
    :return: dictionary of polygons for input slide
    """
    path = '/home/ubuntu/test/xml/' + test_str + '.xml'
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


def map_slide_coords(upper_left_coord, lower_right_coord, tile_side_length):
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


def label_tiles(mask, slide_tumor_grid, tile_side_length, cutoff=0.5,
                plot=False):
    """
    Inputs a mask for a particular slide with all polygons filled, a grid of
    starting tile coordinates (upper left of each tile) corresponding to the
    shape of the mask, the length of each side of the tiles, a cutoff cancer
    percentage threshold, over which the tile is classified as a "cancer" tile
    and under which the tile is classified as a "normal" tile to ensure that
    all tiles are labeled for testing. A boolean argument allowing the option
    to plot the mask of each tile is also included.

    Model iterates through grid of starting tile coordinates and aligns each
    tile to its corresponding mask coordinates, which enable the percentage of
    cancer tissue in each tile to be calculated. If the percentage of cancer
    tissue in each tile is above the cutoff, the tile is labeled "cancer", and
    tile starting coordinates are appended to a dictionary of coordinates,
    along with a label of 1 and cancer tissue percentage, for the cancer class.
    If the percentage of cancer tissue in each tile is below the cutoff, the
    tile is labeled "normal", and coordinates are appended to a dictionary of
    coordinates, along with a label of 0 and cancer tissue percentage, for the
    normal class.

    If plot is set to "True", the portion of the mask corresponding to each
    tile is shown. This is only recommended for troubleshooting.

    :param mask: 2D array of zeros and/or ones corresponding to any pre-filled
                 polygons elements
    :param slide_tumor_grid: 2D array corresponding to a particular slide, each
                             element is a tuple of starting coordinates for
                             each tile to be processed in the model
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :param cutoff: float between 0 and 1.0, inclusive. Set to 0.5 as
                              default
    :param plot: boolean, True --> the portion of the mask corresponding to
                 each tile is shown, set to False by default
    :return: dictionary of starting tile coordinates (tuple), labels, and
             cancer tissue percentages corresponding to the normal class and a
             dictionary of starting tile coordinates (tuple), labels, and
             cancer tissue percentages corresponding to the cancer class
    """
    x_side = tile_side_length
    y_side = tile_side_length
    y_i = 0
    x_i = 0
    list_of_tumor_dicts = []
    list_of_normal_dicts = []
    while y_side < len(mask[:]):
        while x_side < len(mask[0][:]):
            percent_tumor = np.mean(mask[y_side - tile_side_length:y_side,
                                         x_side - tile_side_length:x_side])
            if percent_tumor >= cutoff:
                list_of_tumor_dicts.append(
                    {'coord': tuple(
                        [int(i) for i in slide_tumor_grid[y_i, x_i]]),
                     'label': 1, 'percent_tumor': percent_tumor})
                if plot:
                    print('coords: ', slide_tumor_grid[y_i, x_i])
                    print('percent tumor: ', percent_tumor)
                    plt.imshow(mask[y_side - tile_side_length:y_side,
                                    x_side - tile_side_length:x_side])
                    plt.show()
                else:
                    pass
            elif percent_tumor < cutoff:
                list_of_normal_dicts.append({'coord': tuple(
                    [int(i) for i in slide_tumor_grid[y_i, x_i]]), 'label': 0,
                                             'percent_tumor': percent_tumor})
            else:
                print('Error')
            x_side += tile_side_length
            x_i += 1
        x_i = 0
        x_side = tile_side_length
        y_side += tile_side_length
        y_i += 1
    return list_of_normal_dicts, list_of_tumor_dicts


def filter_white(test_str, coord, threshold=225, tile_side_length=256,
                 level=0):
    """
    Inputs the name of the slide of interest as a string (i.e. "test_002"), a
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

    :param test_str: name of test slide, formatted as "test_xxx"
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
    path = '/home/ubuntu/test/tif/'
    mr_image = reader.open(path + test_str + '.tif')
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
        for item in coord:
            x_coord, y_coord = item['coord']
            image_patch = mr_image.getUCharPatch(int(x_coord), int(y_coord),
                                                 tile_side_length,
                                                 tile_side_length, level)
            avg_rgb = np.mean(image_patch)
            if avg_rgb < threshold:
                list_to_keep.append(item)
            else:
                pass
        return list_to_keep
    else:
        print('Error')
        return None


def get_tiles(test_str, tile_side_length=256, cutoff=0.5, white_threshold=225,
              level=0):
    """
    Inputs the name of the slide of interest as a string (i.e. "test_002"),
    the length of each side of the tiles, a cutoff cancer percentage threshold,
    over which the tile is classified as a "cancer" tile and under which the
    tile is classified as a "normal" tile. Also inputs an average RGB
    threshold, beyond which the tile is considered to be whitespace instead of
    tissue, and a level, which is a multiresolutionimageinterface module
    argument for tile sampling (0 is default).

    Function loads json of starting and ending coordinates, which were manually
    determined and entered with the slide_corner_coords module. The upper left
    coordinates and lower right coordinates for the slide of interest are
    extracted. The aforementioned polygon and mask functions are run for each
    polygon on the slide, creating a complete mask. The remaining
    aforementioned functions are executed, which allow tiles to be classified
    as "cancer" or "normal". Tiles classified as "normal" are then run through
    the whitespace filter function.

    This function returns 1) a dictionary of starting tile coordinates for the
    "normal" class, from which whitespace tiles have been removed, and 2) a
    dictionary of starting tile coordinates for the "cancer" class. Both
    dictionaries also include label and percent cancer items.

    :param test_str: name of test slide, formatted as "test_xxx"
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :param cutoff: float between 0 and 1.0, inclusive. Set to 0.5 as
                              default
    :param white_threshold: int, between 0 and 255, inclusive
    :param level: int, between 0 and 9, inclusive. Set to 0 by default, which
                  is recommended.
    :return: list of filtered starting coordinates for tiles in the "normal"
             class and a list of starting coordinates for tiles in the "cancer"
             class
    """
    with open('/home/ubuntu/test/corner_coords/corner_coords.json',
              'r') as json_file:
        coords = json.load(json_file)
    upper_left_coord = tuple(coords[test_str]['upper_left_coord'])
    lower_right_coord = tuple(coords[test_str]['lower_right_coord'])
    polygons = parse_xml(test_str)
    mask = make_mask(upper_left_coord, lower_right_coord)
    for polygon in polygons.values():
        add_polygons_to_mask(mask, polygon, upper_left_coord)
    slide_coord_grid = map_slide_coords(upper_left_coord, lower_right_coord,
                                        tile_side_length)
    norm_dicts, tumor_dicts = label_tiles(mask, slide_coord_grid,
                                          tile_side_length, cutoff=cutoff)
    filtered_norm_dicts = filter_white(test_str, norm_dicts,
                                       threshold=white_threshold,
                                       tile_side_length=tile_side_length,
                                       level=level)
    return filtered_norm_dicts, tumor_dicts


def save_coord_label_jsons(test_str, filtered_norm_dicts, tumor_dicts):
    """
    Saves dictionaries of coordinates previously calculated as json files for
    slide of interest.
    :param test_str: name of test slide, formatted as "test_xxx"
    :param filtered_norm_dicts: dictionary of filtered starting coordinates,
                                labels, and cancer percentages for tiles in the
                                "normal" class
    :param tumor_dicts: dictionary of starting coordinates, labels, and cancer
                        percentages for tiles in the "cancer" class
    :return: None
    """
    coord_label_json = {test_str: {'filtered_norm_dicts': filtered_norm_dicts,
                                   'tumor_dicts': tumor_dicts}}
    path = '/home/ubuntu/test/coord_label_dicts/'
    with open(path + test_str + '_coord_labels.json', 'w+') as json_file:
        json.dump(coord_label_json, json_file)


def save_cancer_tile_jpegs(test_str, list_of_coord_labels, level=0,
                           tile_side_length=256):
    """
    Inputs the name of the slide of interest as a string (i.e. "test_002"), a
    list of starting coordinates to obtain desired cancer tiles, a level, which
    is a multiresolutionimageinterface module argument for tile sampling (0 is
    default), and the length of each side of the tiles.

    Function passes the starting coordinates for each desired cancer tile
    iteratively into the appropriate multiresolutionimageinterface functions to
    sample the image tiles from the slide of interest. Images are saved in
    their proper directory as jpeg files.

    :param test_str: name of test slide, formatted as "test_xxx"
    :param list_of_coord_labels: list of starting coordinates for tiles in the
                                 "cancer" class
    :param level: int, between 0 and 9, inclusive. Set to 0 by default, which
                  is recommended.
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :return: None
    """
    reader = mir.MultiResolutionImageReader()
    wsi = reader.open('/home/ubuntu/test/tif/' + test_str + '.tif')
    path = '/home/ubuntu/test/cancer_tiles/' + test_str + '/'
    for item in list_of_coord_labels:
        x_coord, y_coord, = item['coord']
        tile = wsi.getUCharPatch(x_coord, y_coord, tile_side_length,
                                 tile_side_length, level)
        image = Image.fromarray(tile)
        image.save(path + test_str + '_cancer_' + str(x_coord) + '_' +
                   str(y_coord) + '.jpeg')


def save_normal_tile_jpegs(test_str, list_of_coord_labels, level=0,
                           tile_side_length=256):
    """
    Inputs the name of the slide of interest as a string (i.e. "test_002"), a
    list of starting coordinates to obtain desired normal tiles, a level, which
    is a multiresolutionimageinterface module argument for tile sampling (0 is
    default), and the length of each side of the tiles.

    Function passes the starting coordinates for each desired normal tile
    iteratively into the appropriate multiresolutionimageinterface functions to
    sample the image tiles from the slide of interest. Images are saved in
    their proper directory as jpeg files.

    :param test_str: name of test slide, formatted as "test_xxx"
    :param list_of_coord_labels: list of filtered starting coordinates for
                                 tiles in the "normal" class
    :param level: int, between 0 and 9, inclusive. Set to 0 by default, which
                  is recommended.
    :param tile_side_length: int, desired length of tile slide, set to 256 by
                             default
    :return: None
    """
    reader = mir.MultiResolutionImageReader()
    wsi = reader.open('/home/ubuntu/test/tif/' + test_str + '.tif')
    path = '/home/ubuntu/test/normal_tiles/' + test_str + '/'
    for item in list_of_coord_labels:
        x_coord, y_coord = item['coord']
        tile = wsi.getUCharPatch(x_coord, y_coord, tile_side_length,
                                 tile_side_length, level)
        image = Image.fromarray(tile)
        image.save(path + test_str + '_normal_' + str(x_coord) + '_' +
                   str(y_coord) + '.jpeg')
