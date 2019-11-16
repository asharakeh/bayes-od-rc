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

    uncertainty_method = 'bayes_od_none'

    entropy_method = 'categorical'  # evaluate using gaussian or categorical entropy

    # All or per category. Note that if per category is used, out of
    # distribution detections are ignored. Results in overestimation of
    # performance.
    compute_method = 'category'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

    dataset_dir = os.path.expanduser('~/Datasets/Kitti/object/')
    label_dir = os.path.join(dataset_dir, data_split_dir) + '/label_2'

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

    cov_dir = os.path.join(
        core.data_dir(),
        'outputs',
        checkpoint_name,
        'predictions',
        results_dir,
        'kitti',
        checkpoint_number,
        uncertainty_method,
        'cov')

    cat_param_dir = os.path.join(core.data_dir(),
                                 'outputs',
                                 checkpoint_name,
                                 'predictions',
                                 results_dir,
                                 'kitti',
                                 checkpoint_number,
                                 uncertainty_method,
                                 'cat_param')

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
            prediction_box_cat_params = np.load(
                os.path.join(cat_param_dir, '{:06d}.npy'.format(frame_id)))
            prediction_box_covs = np.load(
                os.path.join(cov_dir, '{:06d}.npy'.format(frame_id)))

            if entropy_method == 'gaussian':
                ranking_entropies = [compute_gaussian_entropy_np(
                    cov) for cov in prediction_box_covs]
            elif entropy_method == 'categorical':
                ranking_entropies = [compute_categorical_entropy_np(
                    cat_vect) for cat_vect in prediction_box_cat_params]

            for gt_class, gt_box in zip(gt_classes, gt_boxes):

                ind = np.argmax(gt_class)
                gt_box_list = [gt_box[1], gt_box[0], gt_box[3], gt_box[2]]
                if compute_method == 'All':
                    category_name = 'All'
                else:
                    category_name = categories[ind]
                gt_dict = {'name': str(id),
                           'category': category_name,
                           'bbox': gt_box_list,
                           'score': 1}
                gt_dict_list.append(gt_dict)

            for pred_class, pred_box, ranking_entropy in zip(
                    prediction_classes, prediction_boxes, ranking_entropies):
                ind = np.argmax(pred_class)
                if ind >= len(categories):
                    continue
                pred_box_list = [
                    pred_box[1],
                    pred_box[0],
                    pred_box[3],
                    pred_box[2]]
                if compute_method == 'All':
                    category_name = 'All'
                else:
                    category_name = categories[ind]
                pred_dict = {'name': str(id),
                             'category': category_name,
                             'bbox': pred_box_list,
                             'entropy_score': ranking_entropy}

                prediction_dict_list.append(pred_dict)
        id += 1
        print('Computed {} / {} frames.'.format(id, len(frames_list)))

    mean_u_error_list, mean_u_error, cat_list, scores_at_min_u_error= evaluate_u_error(
        gt_dict_list, prediction_dict_list, iou_thresholds=[0.5])

    table = PrettyTable(cat_list)
    table.add_row(mean_u_error_list)
    print("Average " + entropy_method + " MUE: " + str(mean_u_error))
    print("Average " + entropy_method + " Score: " + str(np.mean(scores_at_min_u_error)))

    print(table)


if __name__ == "__main__":
    main()
