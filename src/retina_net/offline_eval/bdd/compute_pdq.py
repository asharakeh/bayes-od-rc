import json
import os
import yaml

import numpy as np
from prettytable import PrettyTable

import src.core as core

from src.retina_net.anchor_generator.box_utils import vuhw_to_vuvu_np
from src.retina_net.offline_eval import pdq_data_holders, pdq
from src.retina_net.builders import dataset_handler_builder

from demos.demo_utils.bdd_demo_utils import read_bdd_format


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'val'

    # Specify whether the validation or inference results need to be
    # visualized.
    results_dir = 'testing'

    dataset_dir = os.path.expanduser('~/Datasets/bdd100k')
    image_dir = os.path.join(dataset_dir, 'images', '100k', data_split_dir)
    label_file_name = os.path.join(
        dataset_dir, 'labels', data_split_dir) + '.json'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

    # sample_free, anchor_redundancy, black_box,  bayes_od_none,
    # bayes_od_ci_fast. bayes_od_ci,or bayes_od_ici
    uncertainty_method = 'bayes_od_none'

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

    gt_dict_list = json.load(open(label_file_name, 'r'))

    frames_full_list = os.listdir(image_dir)

    frame_list_of_lists = [frames_full_list[x:x + 1000]
                           for x in range(0, len(frames_full_list), 1000)]
    score = []
    TP = []
    FP = []
    FN = []
    avg_spatial_quality = []
    avg_label_quality = []
    avg_overall_quality = []

    for frames_list in frame_list_of_lists:
        match_list = []
        for frame_id in frames_list:
            category_gt, boxes_2d_gt, _ = read_bdd_format(
                frame_id, gt_dict_list, pdq_eval=True)
            prediction_boxes_cov = np.load(
                os.path.join(cov_dir, frame_id) + '.npy')
            if prediction_boxes_cov.size:
                prediction_boxes_mean = np.load(
                    os.path.join(mean_dir, frame_id) + '.npy')
                prediction_boxes_cat_params = np.load(
                    os.path.join(cat_param_dir, frame_id) + '.npy')

                transformation_mat = np.array([[0, 1, 0, -0.5],
                                               [1, 0, -0.5, 0],
                                               [0, 1, 0, 0.5],
                                               [1, 0, 0.5, 0]])
                prediction_boxes_cov = np.matmul(
                    np.matmul(
                        transformation_mat,
                        prediction_boxes_cov),
                    transformation_mat.T) * 70
                prediction_boxes = vuhw_to_vuvu_np(prediction_boxes_mean)
                # Create GT list
                gt_instance_list = []
                for cat_gt, box_2d_gt in zip(category_gt, boxes_2d_gt):
                    seg_mask = np.zeros([720, 1280], dtype=np.bool)
                    box_inds = box_2d_gt.astype(np.int32)
                    seg_mask[box_inds[1]:box_inds[3],
                             box_inds[0]:box_inds[2]] = True
                    gt_index = np.where(cat_gt == 1)[0].item(0)
                    gt_instance = pdq_data_holders.GroundTruthInstance(
                        seg_mask, gt_index, 0, 0, bounding_box=box_inds)
                    gt_instance_list.append(gt_instance)
                # Create Detection list
                det_instance_list = []
                for cat_det, box_2d_det, cov_det in zip(
                        prediction_boxes_cat_params, prediction_boxes, prediction_boxes_cov):
                    if np.max(cat_det) >= 0.5445:
                        box_processed = np.array(
                            [box_2d_det[1], box_2d_det[0], box_2d_det[3], box_2d_det[2]]).astype(np.int32)
                        cov_processed = [cov_det[0:2, 0:2], cov_det[2:4, 2:4]]
                        det_instance = pdq_data_holders.PBoxDetInst(
                            cat_det, box_processed, cov_processed)
                        det_instance_list.append(det_instance)
                match_list.append((gt_instance_list, det_instance_list))

        print("PDQ starting:")
        evaluator = pdq.PDQ()
        score.append(evaluator.score(match_list) * 100)
        TP_i, FP_i, FN_i = evaluator.get_assignment_counts()
        TP.append(TP_i)
        FP.append(FP_i)
        FN.append(FN_i)
        avg_spatial_quality.append(evaluator.get_avg_spatial_score())
        avg_label_quality.append(evaluator.get_avg_label_score())
        avg_overall_quality.append(
            evaluator.get_avg_overall_quality_score())
        print("PDQ Ended")

    score = sum(score) / len(score)
    TP = sum(TP)
    FP = sum(FP)
    FN = sum(FN)
    avg_spatial_quality = sum(avg_spatial_quality) / len(avg_spatial_quality)
    avg_label_quality = sum(avg_label_quality) / len(avg_label_quality)
    avg_overall_quality = sum(avg_overall_quality) / len(avg_overall_quality)

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
        'bdd',
        checkpoint_number,
        uncertainty_method,
        'pdq_res.txt')

    with open(text_file_name, "w") as text_file:
        print(table, file=text_file)


if __name__ == '__main__':
    main()
