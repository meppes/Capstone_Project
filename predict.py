"""
This module consists of functions used for the evaluation of the model.
"""

from PIL import Image
import numpy as np


def predict(list_of_tiles, model):
    """
    Inputs list of tiles and the model with which to make predictions.
    Makes individual predictions for each tile and returns predictions as array
    Predictions are probabilities (between 1 and 0) for each class (Normal vs.
    Cancer); therefore each prediction is a tuple.
    """

    predictions = []
    step = 5000
    while step < len(list_of_tiles) + 5000:
        test_set = []
        for item in list_of_tiles[step - 5000:step]:
            im = Image.open(item)
            im_array = np.array(im) / 255
            test_set.append(im_array)
        test_set = np.array(test_set)
        predictions.append(model.predict(test_set, batch_size=1))
        step = step + 5000
    return np.concatenate(predictions[:])


def count_correct(list_of_predictions, expected_label):
    """
    Inputs a list of predictions (tuples of probabilities for Cancer vs. Non-
    Cancer) and the expected labels. Predicted labels are determined for the
    test data by selecting the class with the higher probability for any given
    prediction. Function counts and returns the numbers of correctly-
    identified tiles and incorrectly-identified tiles.
    """
    number_correct = 0
    number_incorrect = 0
    for p in list_of_predictions:
        if expected_label in [0, 'normal']:
            if p[0] > p[1]:
                number_correct += 1
            else:
                number_incorrect += 1
        elif expected_label in [1, 'cancer']:
            if p[0] > p[1]:
                number_incorrect += 1
            else:
                number_correct += 1
    return number_correct, number_incorrect


def calculate_metric(metric, number_correct_cancer, number_correct_normal,
                     number_incorrect_cancer, number_incorrect_normal):
    """
    Calculates evaluation metrics:
    - Accuracy
    - Recall
    - Precision
    - F1 Score
    """
    if metric == 'accuracy':
        total_correct = number_correct_cancer + number_correct_normal
        total = number_incorrect_cancer + number_incorrect_normal \
            + total_correct
        ans = total_correct / total
        return ans
    if metric == 'recall':
        ans = number_correct_cancer / (
            number_correct_cancer + number_incorrect_cancer)
        return ans
    if metric == 'precision':
        ans = number_correct_cancer / (
            number_correct_cancer + number_incorrect_normal)
        return ans
    if metric == 'f1':
        recall = calculate_metric('recall', number_correct_cancer,
                                  number_correct_normal,
                                  number_incorrect_cancer,
                                  number_incorrect_normal)
        precision = calculate_metric('precision', number_correct_cancer,
                                     number_correct_normal,
                                     number_incorrect_cancer,
                                     number_incorrect_normal)
        ans = 2 * recall * precision / (recall + precision)
        return ans

    print('Error')
    return None
