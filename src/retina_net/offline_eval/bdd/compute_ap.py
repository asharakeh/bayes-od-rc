import json
import os
from prettytable import PrettyTable

from src.core.evaluation_utils_2d import *

import src.core as core


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'val'

    # Specify whether the validation or inference results need to be
    # visualized.
    results_dir = 'validation'
    results_dir = 'testing'

    # sample_free, anchor_redundancy, black_box,  naive_aleatoric_epistemic,  bayes_od_none,
    # bayes_od_ci_fast. bayes_od_ci,or bayes_od_ici
    uncertainty_method = 'bayes_od_none'

    dataset_dir = os.path.expanduser('~/Datasets/bdd100k')
    label_file_name = os.path.join(
        dataset_dir, 'labels', data_split_dir) + '.json'

    checkpoint_name = 'retinanet_bdd_covar'
    checkpoint_number = '101'

    if results_dir == 'testing':
        prediction_dir = os.path.join(
            core.data_dir(),
            'outputs',
            checkpoint_name,
            'predictions',
            results_dir,
            'bdd',
            checkpoint_number,
            uncertainty_method,
            'data')
    else:
        prediction_dir = os.path.join(
            core.data_dir(),
            'outputs',
            checkpoint_name,
            'predictions',
            results_dir,
            checkpoint_number,
            'data')
    prediction_file_name = os.path.join(prediction_dir, 'predictions.json')

    gt_dict_list = json.load(open(label_file_name, 'r'))
    prediction_dict_list = json.load(open(prediction_file_name, 'r'))

    mean, breakdown, cat_list, optimal_score_thresholds, maximum_f_scores = evaluate_detection(
        gt_dict_list, prediction_dict_list, iou_thresholds=[0.5])

    table = PrettyTable(cat_list)
    table.add_row(breakdown)

    breakdown = np.array(breakdown)
    mean = np.mean(breakdown[breakdown > 0.0])

    print('Mean AP: ' + str(mean) + '\n')
    print('Mean Maximum F-score: ' + str(np.mean(np.array(maximum_f_scores)[breakdown > 0.0])) + '\n')

    print('Mean Optimal Score Threshold: ' + str(np.mean(np.array(optimal_score_thresholds)[breakdown > 0.0])) + '\n')

    # Compute number of Out of Distribution predictions. Predicitions that do
    # not exist in GT data but where classified as such.
    num_od = np.array([0 if prediction['category']
                       in cat_list else 1 for prediction in prediction_dict_list])

    print('Ratio of Out of Distribution Predictions: ' +
          str(np.sum(num_od) / len(prediction_dict_list)) + '\n')
    print(table)


if __name__ == '__main__':
    main()
