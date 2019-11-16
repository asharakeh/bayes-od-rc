import os
import random
import cv2
import numpy as np

import src.core as core

import src.core.evaluation_utils_2d as eval_utils

from demos.demo_utils.vis_utils import draw_box_2d, draw_ellipse_2d_corners
from demos.demo_utils.kitti_demo_utils import read_labels, read_predictions

from src.retina_net.anchor_generator.box_utils import vuhw_to_vuvu_np


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'training'

    # Only testing works, since it requires covariance matrices
    results_dir = 'testing'

    dataset_dir = os.path.expanduser('~/Datasets/Kitti/object/')  # Change this to corresponding dataset directory
    image_dir = os.path.join(dataset_dir, data_split_dir) + '/image_2'
    label_dir = os.path.join(dataset_dir, data_split_dir) + '/label_2'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

    uncertainty_method = 'bayes_od_none'

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
    index = random.randint(0, len(frames_list))
    frame_id = int(frames_list[index][0:6])

    print('Showing Frame: %d' % frame_id)

    #############
    # Read Frame
    #############
    im_path = image_dir + '/{:06d}.png'.format(frame_id)
    image = cv2.imread(im_path)

    label_path = label_dir + '/{:06d}.txt'.format(frame_id)
    gt_classes_hard, gt_boxes_hard = read_labels(label_path)

    prediction_path = prediction_dir + '/{:06d}.txt'.format(frame_id)
    prediction_classes, prediction_boxes, prediction_scores = read_predictions(
        prediction_path)

    prediction_boxes_mean = np.load(mean_dir + '/{:06d}.npy'.format(frame_id))
    prediction_boxes_cov = np.load(cov_dir + '/{:06d}.npy'.format(frame_id))
    prediction_boxes_cat_params = np.load(
        cat_param_dir + '/{:06d}.npy'.format(frame_id))

    # Read entropy for debugging purposes
    transformation_mat = np.array([[1, 0, -0.5, 0],
                                   [0, 1, 0, -0.5],
                                   [1, 0, 0.5, 0],
                                   [0, 1, 0, 0.5]])

    prediction_boxes_cov = np.matmul(
        np.matmul(
            transformation_mat,
            prediction_boxes_cov),
        transformation_mat.T)

    prediction_boxes = vuhw_to_vuvu_np(prediction_boxes_mean)
    category_pred = np.zeros(prediction_boxes_cat_params.shape)
    cat_ind = np.argmax(prediction_boxes_cat_params, axis=1)
    category_pred[np.arange(category_pred.shape[0]), cat_ind] = 1

    #########################################################
    # Draw GT and Prediction Boxes
    #########################################################
    # Transform Predictions to left and right images
    image_out = draw_box_2d(np.copy(image),
                            gt_boxes_hard,
                            gt_classes_hard,
                            line_width=2,
                            dataset='kitti',
                            is_gt=True)

    image_out = draw_ellipse_2d_corners(image_out,
                                        prediction_boxes,
                                        prediction_boxes_cov,
                                        prediction_classes,
                                        dataset='kitti',
                                        line_width=3)

    cv2.imshow('Detections from ' + uncertainty_method, image_out)

    cv2.waitKey()
    heatmap_new = np.zeros(image.shape[0:2])

    for prediction_box, prediction_box_cov in zip(
            prediction_boxes, prediction_boxes_cov):
        heatmap = eval_utils.calc_heatmap(
            prediction_box, prediction_box_cov, image_out.shape[0:2]) * 255
        heatmap_new = np.where(heatmap != 0, heatmap, heatmap_new)

    im_color = cv2.applyColorMap(
        heatmap_new.astype(
            np.uint8), cv2.COLORMAP_JET)
    overlayed_im = cv2.addWeighted(image, 0.4, im_color, 0.6, 0)
    cv2.imshow(
        'Spatial Heatmap Image from ' +
        uncertainty_method,
        overlayed_im)

    cv2.waitKey()


if __name__ == "__main__":
    main()
