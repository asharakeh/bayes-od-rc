import os
import random

import numpy as np
import cv2
import json

import src.core as core

from src.core.evaluation_utils_2d import two_d_iou, calc_heatmap

from demos.demo_utils.bdd_demo_utils import read_bdd_format
from demos.demo_utils.vis_utils import draw_box_2d, draw_ellipse_2d_corners

from src.retina_net.anchor_generator.box_utils import vuhw_to_vuvu_np

categories = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'person',
    4: 'rider',
    5: 'bike',
    6: 'motor',
    7: 'bkgrnd'}


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'val'

    # Specify whether the validation or inference results need to be
    # visualized.
    results_dir = 'testing'

    uncertainty_method = 'bayes_od_none'

    dataset_dir = os.path.expanduser('~/Datasets/bdd100k')
    image_dir = os.path.join(dataset_dir, 'images', '100k', data_split_dir)
    label_file_name = os.path.join(
        dataset_dir, 'labels', data_split_dir) + '.json'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

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

    frames_list = os.listdir(image_dir)[0:100]
    index = random.randint(0, len(frames_list))
    frame_id = frames_list[index]

    print('Showing Frame ID:' + frame_id)

    #############
    # Read Frame
    #############
    im_path = image_dir + '/' + frame_id
    image = cv2.imread(im_path)

    all_labels = json.load(open(label_file_name, 'r'))
    category_gt, boxes_2d_gt, _ = read_bdd_format(
        frame_id, all_labels, categories=[
            'car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor'])

    prediction_boxes_mean = np.load(mean_dir + '/' + frame_id + '.npy')
    prediction_boxes_cov = np.load(cov_dir + '/' + frame_id + '.npy')
    prediction_boxes_cat_params = np.load(
        cat_param_dir + '/' + frame_id + '.npy')

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
    category_names = np.array([categories[np.argmax(cat_param)]
                               for cat_param in prediction_boxes_cat_params])

    #########################################################
    # Draw GT and Prediction Boxes
    #########################################################
    # Transform Predictions to left and right images
    image_out = draw_box_2d(
        np.copy(image),
        prediction_boxes,
        category_pred,
        line_width=1,
        dataset='bdd',
        is_gt=False,
        plot_text=True,
        text_to_plot=category_names)

    image_out = draw_ellipse_2d_corners(image_out,
                                        prediction_boxes,
                                        prediction_boxes_cov * 70,
                                        category_pred,
                                        dataset='bdd',
                                        line_width=3)
    if results_dir == 'testing':
        cv2.imshow('Detections from ' + uncertainty_method, image_out)
    else:
        cv2.imshow('Validation Set Detections', image_out)

    cv2.waitKey()

    heatmap_new = np.zeros(image.shape[0:2])
    for prediction_box, prediction_box_cov in zip(
            prediction_boxes, prediction_boxes_cov * 70):
        heatmap = calc_heatmap(
            prediction_box, prediction_box_cov, image_out.shape[0:2]) * 255
        heatmap_new = np.where(heatmap != 0, heatmap, heatmap_new)

    im_color = cv2.applyColorMap(
        heatmap_new.astype(
            np.uint8), cv2.COLORMAP_JET)
    overlayed_im = cv2.addWeighted(image, 0.1, im_color, 0.9, 0)
    cv2.imshow(
        'Spatial Heatmap Image from ' +
        uncertainty_method,
        overlayed_im)

    cv2.waitKey()


if __name__ == "__main__":
    main()
