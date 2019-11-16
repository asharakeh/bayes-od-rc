import json
import os
import yaml
from prettytable import PrettyTable

import src.core as core
from src.core.evaluation_utils_2d import *
from src.retina_net.anchor_generator.box_utils import vuhw_to_vuvu_np


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'val'
    # Specify whether the validation or inference results need to be
    # visualized.
    results_dir = 'testing'

    # sample_free, anchor_redundancy, black_box,naive_aleatoric_epistemic,  bayes_od_none,
    # bayes_od_ci_fast. bayes_od_ci,or bayes_od_ici
    uncertainty_method = 'bayes_od_none'

    dataset_dir = os.path.expanduser('~/Datasets/bdd100k')
    label_file_name = os.path.join(
        dataset_dir, 'labels', data_split_dir) + '.json'

    entropy_method = 'categorical'  # evaluate using gaussian or categorical entropy

    # All or per category. Note that if per category is used, out of
    # distribution detections are ignored. Results in overestimation of
    # performance.
    compute_method = 'category'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

    image_dir = os.path.join(dataset_dir, 'images', '100k', data_split_dir)

    mean_dir = os.path.join(
        core.data_dir(),
        'outputs',
        checkpoint_name,
        'predictions',
        results_dir,
        'bdd',
        checkpoint_number,
        uncertainty_method,
        'mean')

    cov_dir = os.path.join(
        core.data_dir(),
        'outputs',
        checkpoint_name,
        'predictions',
        results_dir,
        'bdd',
        checkpoint_number,
        uncertainty_method,
        'cov')

    cat_param_dir = os.path.join(core.data_dir(),
                                 'outputs',
                                 checkpoint_name,
                                 'predictions',
                                 results_dir,
                                 'bdd',
                                 checkpoint_number,
                                 uncertainty_method,
                                 'cat_param')
    # Read ground truth
    gt_dict_list = json.load(open(label_file_name, 'r'))
    category_names = [
        'car',
        'truck',
        'bus',
        'person',
        'rider',
        'bike',
        'motor',
        'bkgrnd']
    gt_dict_list = [
        gt_instance for gt_instance in gt_dict_list if gt_instance['category'] in category_names]
    if compute_method == 'All':
        for gt_instance in gt_dict_list:
            gt_instance['category'] = 'All'

    frame_key_list = os.listdir(image_dir)
    pred_dict_list = []

    for frame_key in frame_key_list:
        prediction_box_means = np.load(
            os.path.join(mean_dir, frame_key) + '.npy')
        if prediction_box_means.size:
            prediction_box_means = vuhw_to_vuvu_np(prediction_box_means)
        else:
            continue

        prediction_box_cat_params = np.load(
            os.path.join(cat_param_dir, frame_key) + '.npy')
        prediction_box_covs = np.load(
            os.path.join(cov_dir, frame_key) + '.npy')

        prediction_cat_idxs = np.argmax(prediction_box_cat_params, axis=1)
        prediction_category_names = [
            category_names[prediction_cat_idx] for prediction_cat_idx in prediction_cat_idxs]

        if entropy_method == 'gaussian':
            ranking_entropies = [compute_gaussian_entropy_np(
                cov) for cov in prediction_box_covs]
        elif entropy_method == 'categorical':
            ranking_entropies = [compute_categorical_entropy_np(
                cat_vect) for cat_vect in prediction_box_cat_params]

        frame_preds_list = []
        for prediction_box_mean, prediction_category_name, ranking_entropy in zip(
                prediction_box_means, prediction_category_names, ranking_entropies):
            if compute_method == 'All':
                category_name = 'All'
            else:
                category_name = prediction_category_name
            frame_preds_list.append({'name': frame_key,
                                     'category': category_name,
                                     'bbox': [prediction_box_mean.tolist()[1],
                                              prediction_box_mean.tolist()[0],
                                              prediction_box_mean.tolist()[3],
                                              prediction_box_mean.tolist()[2]],
                                     'entropy_score': ranking_entropy})
        pred_dict_list.extend(frame_preds_list)

    mean_u_error_list, mean_u_error, cat_list, _ = evaluate_u_error(
        gt_dict_list, pred_dict_list, iou_thresholds=[0.5])

    table = PrettyTable(cat_list)
    table.add_row(mean_u_error_list)
    print("Average " + entropy_method + " MUE: " + str(mean_u_error))
    print(table)


if __name__ == '__main__':
    main()
