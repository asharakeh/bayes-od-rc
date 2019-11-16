import os
import yaml

import numpy as np
from prettytable import PrettyTable

import src.core as core

from src.retina_net.anchor_generator.box_utils import vuhw_to_vuvu_np
from src.retina_net.offline_eval import pdq_data_holders, pdq
from src.retina_net.builders import dataset_handler_builder

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

    # sample_free, anchor_redundancy, black_box,naive_aleatoric_epistemic  bayes_od_none,
    # bayes_od_ci_fast. bayes_od_ci,or bayes_od_ici
    uncertainty_method = 'bayes_od_none'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

    dataset_dir = os.path.expanduser('~/Datasets/Kitti/object/')
    label_dir = os.path.join(dataset_dir, data_split_dir) + '/label_2'

    mean_dir = os.path.join(
        core.data_dir(),
        'outputs',
        checkpoint_name,
        'predictions',
        results_dir,
        'kitti',
        checkpoint_number,
        uncertainty_method,
        'mean')

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
    print("PDQ evaluation starting:")
    match_list = []
    for frame in frames_list:
        frame_id = int(frame[0:6])
        #############
        # Read Frame
        #############
        label_path = label_dir + '/{:06d}.txt'.format(frame_id)
        gt_classes, gt_boxes = read_labels(
            label_path, difficulty=difficulty, categories=categories)

        # Create GT list
        gt_instance_list = []
        if gt_boxes.size > 0:
            for cat_gt, box_2d_gt in zip(gt_classes, gt_boxes):
                seg_mask = np.zeros([375, 1300], dtype=np.bool)
                box_inds = box_2d_gt.astype(np.int32).tolist()
                box_inds = np.array(
                    [box_inds[1], box_inds[0], box_inds[3], box_inds[2]])

                box_inds = np.clip(
                    box_inds,
                    a_min=0.0,
                    a_max=1300).astype(
                    np.int32)

                seg_mask[box_inds[1]:box_inds[3],
                         box_inds[0]:box_inds[2]] = True
                gt_index = np.argmax(cat_gt)
                gt_instance = pdq_data_holders.GroundTruthInstance(
                    seg_mask, gt_index, 0, 0, bounding_box=box_inds)
                gt_instance_list.append(gt_instance)

        prediction_boxes_mean = np.load(
            mean_dir + '/{:06d}.npy'.format(frame_id))
        prediction_boxes_cov = np.load(
            cov_dir + '/{:06d}.npy'.format(frame_id)) * 70
        prediction_boxes_cat_params = np.load(
            cat_param_dir + '/{:06d}.npy'.format(frame_id))

        det_instance_list = []
        if prediction_boxes_cov.size:
            prediction_boxes_cat_params = np.stack(
                [prediction_boxes_cat_params[:, 0], prediction_boxes_cat_params[:, 3]], axis=1)
            transformation_mat = np.array([[0, 1, 0, -0.5],
                                           [1, 0, -0.5, 0],
                                           [0, 1, 0, 0.5],
                                           [1, 0, 0.5, 0]])
            prediction_boxes_cov = np.matmul(
                np.matmul(
                    transformation_mat,
                    prediction_boxes_cov),
                transformation_mat.T)
            prediction_boxes_mean = vuhw_to_vuvu_np(prediction_boxes_mean)
            for cat_det, box_mean, cov_det in zip(
                    prediction_boxes_cat_params, prediction_boxes_mean, prediction_boxes_cov):
                if np.max(cat_det) >= 0.5:
                    box_processed = np.array(
                        [box_mean[1], box_mean[0], box_mean[3], box_mean[2]]).astype(np.int32)
                    cov_processed = [cov_det[0:2, 0:2], cov_det[2:4, 2:4]]
                    det_instance = pdq_data_holders.PBoxDetInst(
                        cat_det, box_processed, cov_processed)
                    det_instance_list.append(det_instance)
        match_list.append((gt_instance_list, det_instance_list))

        id += 1
        print('Computed {} / {} frames.'.format(id, len(frames_list)))

    print("PDQ Ended")
    evaluator = pdq.PDQ()
    score = evaluator.score(match_list) * 100
    TP, FP, FN = evaluator.get_assignment_counts()
    avg_spatial_quality = evaluator.get_avg_spatial_score()
    avg_label_quality = evaluator.get_avg_label_score()
    avg_overall_quality = evaluator.get_avg_overall_quality_score()

    table = PrettyTable(['score',
                         'True Positives',
                         'False Positives',
                         'False Negatives',
                         'Average Spatial Quality',
                         'Average Label Quality',
                         'Average Overall Quality'])

    table.add_row([score, TP, FP, FN, avg_spatial_quality,
                   avg_label_quality, avg_overall_quality])

    print(table)

    text_file_name = os.path.join(
        core.data_dir(),
        'outputs',
        checkpoint_name,
        'predictions',
        results_dir,
        'kitti',
        checkpoint_number,
        uncertainty_method,
        'pdq_res.txt')

    with open(text_file_name, "w") as text_file:
        print(table, file=text_file)


if __name__ == "__main__":
    main()
