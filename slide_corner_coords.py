#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:13:18 2019

This module stores start and stop coordinates for each slide for further
analysis. Start and stop coordinates are chosen manually for each slide. Slides
are viewed on the ASAP software, and coordinates are chosen such that only
regions of tissue are included. The goal is to avoid inefficiently evaluating
tiles of whitespace (majority of slide).

The start coordinate is defined by the upper left coordinate, and the end
coordinate is defined by the lower right coordinate, as seen in ASAP.

@author: marissaeppes
"""

import json


def write_tumor():
    """
    Enter slide number, upper left x-coordinate and y-coordinate specifying
    the start coordinates for region of analysis, and lower right x-coordinate
    and y-coordinate specifying end coordinates for region of analysis.
    Function either creates a corner coordinate dictionary and adds entries or
    adds entries to an existing corner coordinate dictionary, which is saved as
    a json file. This function is used for the training tumor slides.
    :return: None
    """
    slide_number = input('Enter slide number: ')
    upper_left_x = input('Enter upper left x: ')
    upper_left_y = input('Enter upper left y: ')
    lower_right_x = input('Enter lower right x: ')
    lower_right_y = input('Enter lower right y: ')

    if len(slide_number) == 1:
        tumor_str = 'tumor_00' + slide_number
    elif len(slide_number) == 2:
        tumor_str = 'tumor_0' + slide_number
    elif len(slide_number) == 3:
        tumor_str = 'tumor_' + slide_number
    else:
        return None

    coord_json = {
        tumor_str: {'upper_left_coord': (int(upper_left_x), int(upper_left_y)),
                    'lower_right_coord': (
                        int(lower_right_x), int(lower_right_y))}}

    path = '/home/ubuntu/tumor/corner_coords/'

    try:
        with open(path + 'corner_coords.json', 'r') as json_file:
            data = json.load(json_file)
            data.update(coord_json)
    except FileNotFoundError:
        data = coord_json
    with open(path + 'corner_coords.json', 'w') as json_file:
        json.dump(data, json_file)
    return None


def write_test():
    """
    Enter slide number, upper left x-coordinate and y-coordinate specifying
    the start coordinates for region of analysis, and lower right x-coordinate
    and y-coordinate specifying end coordinates for region of analysis.
    Function either creates a corner coordinate dictionary and adds entries or
    adds entries to an existing corner coordinate dictionary, which is saved as
    a json file. This function is used for the testing slides.
    :return: None
    """
    slide_number = input('Enter slide number: ')
    upper_left_x = input('Enter upper left x: ')
    upper_left_y = input('Enter upper left y: ')
    lower_right_x = input('Enter lower right x: ')
    lower_right_y = input('Enter lower right y: ')

    if len(slide_number) == 1:
        test_str = 'test_00' + slide_number
    elif len(slide_number) == 2:
        test_str = 'test_0' + slide_number
    elif len(slide_number) == 3:
        test_str = 'test_' + slide_number
    else:
        return None

    coord_json = {
        test_str: {'upper_left_coord': (int(upper_left_x), int(upper_left_y)),
                   'lower_right_coord': (
                       int(lower_right_x), int(lower_right_y))}}

    path = '/home/ubuntu/test/corner_coords/'

    try:
        with open(path + 'corner_coords.json', 'r') as json_file:
            data = json.load(json_file)
            data.update(coord_json)
    except FileNotFoundError:
        data = coord_json
    with open(path + 'corner_coords.json', 'w') as json_file:
        json.dump(data, json_file)
    return None
