import os
from prettytable import PrettyTable

import src.core as core
from src.core.evaluation_utils_2d import *

from demos.demo_utils.kitti_demo_utils import read_labels, read_predictions


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'training'
    difficulty = 'all'
    categories = ['car', 'pedestrian']

    # Specify whether the validation or inference results need to be evaluated.
    # results_dir = 'validation'  # Or testing
    results_dir = 'testing'

    # sample_free, anchor_redundancy, black_box,  naive_aleatoric_epistemic,  bayes_od_none,
    # bayes_od_ci_fast. bayes_od_ci,or bayes_od_ici
    uncertainty_method = 'bayes_od_none'

    checkpoint_name = 'retinanet_bdd_covar'
    checkpoint_number = '101'

    dataset_dir = os.path.expanduser('~/Datasets/Kitti/object/')
    label_dir = os.path.join(dataset_dir, data_split_dir) + '/label_2'

    if results_dir == 'testing':
        prediction_dir = os.path.join(
            core.data_dir(),
            'outputs',
            checkpoint_name,
            'predictions',
            results_dir,
            'kitti',
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

    frames_list = os.listdir(prediction_dir)

    id = 0
    gt_dict_list = []
    prediction_dict_list = []
    for frame in frames_list:
        frame_id = int(frame[0:6])
        #############
        # Read Frame
        #############
        label_path = label_dir + '/{:06d}.txt'.format(frame_id)
        gt_classes, gt_boxes = read_labels(
            label_path, difficulty=difficulty, categories=categories)

        prediction_path = prediction_dir + '/{:06d}.txt'.format(frame_id)
        prediction_classes, prediction_boxes, prediction_scores = read_predictions(
            prediction_path, categories=categories)

        if gt_boxes.size > 0 and prediction_boxes.size > 0:

            for gt_class, gt_box in zip(gt_classes, gt_boxes):

                ind = np.argmax(gt_class)
                gt_box_list = [gt_box[1], gt_box[0], gt_box[3], gt_box[2]]
                gt_dict = {'name': str(id),
                           'category': categories[ind],
                           'bbox': gt_box_list,
                           'score': 1}
                gt_dict_list.append(gt_dict)

            for pred_class, pred_box, pred_score in zip(
                    prediction_classes, prediction_boxes, prediction_scores):

                ind = np.argmax(pred_class)
                pred_box_list = [
                    pred_box[1],
                    pred_box[0],
                    pred_box[3],
                    pred_box[2]]
                pred_dict = {'name': str(id),
                             'category': categories[ind],
                             'bbox': pred_box_list,
                             'score': pred_score}

                prediction_dict_list.append(pred_dict)
        id += 1
        print('Computed {} / {} frames.'.format(id, len(frames_list)))

    mean, breakdown, cat_list, optimal_score_thresholds, maximum_f_scores = evaluate_detection(
        gt_dict_list, prediction_dict_list, iou_thresholds=[0.5])

    table = PrettyTable(cat_list)
    table.add_row(breakdown)
    table.add_row(optimal_score_thresholds)

    print('Mean AP: ' + str(mean) + '\n')
    print('Mean Optimal Score Threshold: ' +
          str(np.mean(np.array(optimal_score_thresholds))) + '\n')
    print('Mean Maximum F-score: ' +
          str(np.mean(np.array(maximum_f_scores))) + '\n')

    # Compute number of Out of Distribution predictions. Predicitions that do
    # not exist in GT data but where classified as such.
    num_od = np.array([0 if prediction['category']
                       in cat_list else 1 for prediction in prediction_dict_list])

    print('Ratio of Out of Distribution Predictions: ' +
          str(np.sum(num_od) / len(prediction_dict_list)) + '\n')
    print(table)


if __name__ == "__main__":
    main()
