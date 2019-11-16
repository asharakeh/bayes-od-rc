import cv2
import numpy as np
from scipy.stats import norm, chi2

#############################
# 2D Visualization Utilities
#############################
#'car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor'
_COLOUR_SCHEME_PREDICTIONS_BDD = [(255, 255, 102),  # Teal
                                  (255, 153, 51),  # Darker teal
                                  (255, 0, 0),  # Dark blue
                                  (255, 102, 255),  # Magenta
                                  (178, 102, 255),  # Purple
                                  (102, 255, 255),  # Yellow
                                  (102, 178, 255),  # orangish
                                  (0, 0, 255)]  # Red
#'car', 'pedestrian', 'cyclist'
_COLOUR_SCHEME_PREDICTIONS_KITTI = [(255, 255, 102),  # Teal
                                    (255, 102, 255),  # Magenta
                                    (102, 255, 255),  # Yellow
                                    (0, 0, 0)]

# Too many classes so one color for all
_COLOUR_SCHEME_PREDICTIONS_COCO = [
    (255, 255, 102) for i in range(
        0, 81)]  # Teal

# 'positive', 'negative'
_COLOUR_SCHEME_ANCHORS = [(255, 255, 0),
                          (0, 255, 255)]


def draw_box_2d(
        image,
        boxes_2d,
        obj_classes=None,
        dataset='bdd',
        is_gt=False,
        plot_text=False,
        text_to_plot=None,
        line_width=1,
):
    """

    :param image: input m x n image
    :param boxes_2d: 2D bounding boxes, k x 4 represented as [v_min u_min v_max u_max]
    :param obj_classes: Categorization of object, to be used to determine colour
    :param plot_text: boolean to indicate if any form of text (Score or IOU for example) needs to be ploted
    :param text_to_plot: 1 text entry per bounding box
    :param line_width: Box line width
    :param is_gt: boolean indicator if ground truth
    :return:
    """

    if dataset == 'bdd':
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_BDD
    elif dataset == 'kitti':
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_KITTI
    else:
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_COCO

    if boxes_2d.size == 0:
        return image[:]

    if len(boxes_2d.shape) == 1:
        boxes_2d = np.expand_dims(boxes_2d, axis=0)

    boxes_2d = np.ndarray.astype(boxes_2d, np.int32)

    image_out = image[:]
    color_index_list = []
    if boxes_2d.size:
        if obj_classes is not None:
            for box_2d, obj_class in zip(boxes_2d, obj_classes):
                if is_gt:
                    color_index_list.extend([1])
                    cv2.rectangle(image_out,
                                  (box_2d[1], box_2d[0]),
                                  (box_2d[3], box_2d[2]),
                                  (0, 255, 0),
                                  line_width,
                                  lineType=cv2.LINE_AA)

                else:
                    color_index = list(obj_class == 1).index(True)
                    color_index_list.extend([color_index])
                    cv2.rectangle(image_out,
                                  (box_2d[1],
                                   box_2d[0]),
                                  (box_2d[3],
                                   box_2d[2]),
                                  _COLOUR_SCHEME_PREDICTIONS[color_index],
                                  line_width,
                                  lineType=cv2.LINE_AA)

            if plot_text:
                for txt, box, color_index in zip(
                        text_to_plot, boxes_2d, color_index_list):

                    if isinstance(txt, np.float):
                        txt = str(np.round(txt, 3))

                    coordinate_v = np.int(box[0])
                    coordinate_u = np.int(box[1])
                    if is_gt:
                        color = (0, 255, 0)
                    else:
                        color = _COLOUR_SCHEME_PREDICTIONS[color_index]

                    (text_width, text_height) = cv2.getTextSize(
                        txt, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)[0]
                    text_offset_x = coordinate_u
                    text_offset_y = coordinate_v + 15

                    box_coords = (
                        (text_offset_x, text_offset_y),
                        (text_offset_x + text_width - 2, text_offset_y - text_height - 2))

                    cv2.rectangle(
                        image_out, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

                    cv2.putText(
                        image_out,
                        txt,
                        (coordinate_u,
                         coordinate_v + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

        else:
            for box_2d in boxes_2d:
                color_index = 0

                cv2.rectangle(image_out,
                              (box_2d[1], box_2d[0]), (box_2d[3], box_2d[2]),
                              _COLOUR_SCHEME_ANCHORS[color_index],
                              line_width,
                              lineType=cv2.LINE_AA)
                if is_gt:
                    cv2.rectangle(image_out,
                                  (box_2d[1] - (line_width + 2),
                                   box_2d[0] - (line_width + 2)),
                                  (box_2d[3] + line_width + 2,
                                   box_2d[2] + line_width + 2),
                                  _COLOUR_SCHEME_ANCHORS[color_index],
                                  line_width,
                                  lineType=cv2.LINE_AA)

    return image_out


def draw_ellipse_2d(
        image,
        prediction_boxes_mean,
        prediction_boxes_cov,
        obj_classes,
        dataset='bdd',
        line_width=3):
    """
    :param image: input m x n image
    :param prediction_boxes_mean: 2D bounding boxes, k x 4 represented as [v u h w]
    :param prediction_boxes_cov: 2D box covariance matrix, k x 4 x 4
    :param obj_classes: Categorization of object, to be used to determine colour
    :param line_width: Box line width
    :return:
    """
    if dataset == 'bdd':
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_BDD
    elif dataset == 'kitti':
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_KITTI
    else:
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_COCO

    if prediction_boxes_mean.size == 0:
        return image[:]

    if prediction_boxes_mean.size == 1:
        prediction_boxes_mean = np.expand_dims(prediction_boxes_mean, axis=0)
        prediction_boxes_cov = np.expand_dims(prediction_boxes_cov, axis=0)

    image_out = image[:]
    if prediction_boxes_mean.size:
        for mean, cov, obj_class in zip(
                prediction_boxes_mean, prediction_boxes_cov, obj_classes):
            color_index = list(obj_class == 1).index(True)

            mean = mean.astype(np.int32)

            # Draw position uncertainty ellipse
            width, height, rotation = cov_ellipse(cov[0:2, 0:2])

            width[width < 0] = 0
            height[height < 0] = 0
            if not (np.isnan(width) or np.isnan(height) or np.isnan(rotation)):
                width = width.astype(np.int32)
                height = height.astype(np.int32)
                rotation = rotation.astype(np.int32) + 180

                image_out = cv2.ellipse(
                    image_out,
                    (mean[1],
                     mean[0]),
                    (height / 2,
                     width / 2),
                    rotation,
                    0.0,
                    360.0,
                    _COLOUR_SCHEME_PREDICTIONS[color_index],
                    line_width,
                    lineType=cv2.LINE_AA)

            # Draw 95% confidence for width and height
            conf_int_h = norm.interval(
                0.95, loc=mean[2], scale=np.sqrt(cov[2, 2]))
            conf_int_w = norm.interval(
                0.95, loc=mean[3], scale=np.sqrt(cov[3, 3]))

            box_u_min_low = np.int32(mean[1] - conf_int_w[0] / 2)
            box_v_min_low = np.int32(mean[0] - conf_int_h[0] / 2)
            box_u_max_low = np.int32(mean[1] + conf_int_w[0] / 2)
            box_v_max_low = np.int32(mean[0] + conf_int_h[0] / 2)

            cv2.rectangle(image_out,
                          (box_u_min_low, box_v_min_low),
                          (box_u_max_low, box_v_max_low),
                          _COLOUR_SCHEME_PREDICTIONS[color_index],
                          line_width,
                          lineType=cv2.LINE_AA)

            box_u_min_high = np.int32(mean[1] - conf_int_w[1] / 2)
            box_v_min_high = np.int32(mean[0] - conf_int_h[1] / 2)
            box_u_max_high = np.int32(mean[1] + conf_int_w[1] / 2)
            box_v_max_high = np.int32(mean[0] + conf_int_h[1] / 2)

            cv2.rectangle(image_out,
                          (box_u_min_high, box_v_min_high),
                          (box_u_max_high, box_v_max_high),
                          _COLOUR_SCHEME_PREDICTIONS[color_index],
                          line_width,
                          lineType=cv2.LINE_AA)

            box_u_min_mean = np.int32(mean[1] - mean[3] / 2)
            box_v_min_mean = np.int32(mean[0] - mean[2] / 2)
            box_u_max_mean = np.int32(mean[1] + mean[3] / 2)
            box_v_max_mean = np.int32(mean[0] + mean[2] / 2)

            cv2.rectangle(image_out,
                          (box_u_min_mean, box_v_min_mean),
                          (box_u_max_mean, box_v_max_mean),
                          _COLOUR_SCHEME_PREDICTIONS[color_index],
                          line_width - 1,
                          lineType=cv2.LINE_AA)

    return image_out


def draw_ellipse_2d_corners(
        image,
        prediction_boxes_mean,
        prediction_boxes_cov,
        obj_classes,
        dataset='bdd',
        line_width=3):
    """
    :param image: input m x n image
    :param prediction_boxes_mean: 2D bounding boxes, k x 4 represented as [v_min u_min v_max u_max]
    :param prediction_boxes_cov: 2D box covariance matrix, k x 4 x 4
    :param obj_classes: Categorization of object, to be used to determine colour
    :param line_width: Box line width
    :return:
    """
    if dataset == 'bdd':
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_BDD
    elif dataset == 'kitti':
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_KITTI
    else:
        _COLOUR_SCHEME_PREDICTIONS = _COLOUR_SCHEME_PREDICTIONS_COCO

    if prediction_boxes_mean.size == 0:
        return image[:]

    if prediction_boxes_mean.size == 1:
        prediction_boxes_mean = np.expand_dims(prediction_boxes_mean, axis=0)
        prediction_boxes_cov = np.expand_dims(prediction_boxes_cov, axis=0)

    image_out = image[:]
    if prediction_boxes_mean.size:
        for mean, cov, obj_class in zip(
                prediction_boxes_mean, prediction_boxes_cov, obj_classes):
            color_index = list(obj_class == 1).index(True)

            mean = mean.astype(np.int32)

            # Draw position uncertainty ellipse
            width, height, rotation = cov_ellipse(cov[0:2, 0:2])

            width[width < 0] = 0
            height[height < 0] = 0
            if not (np.isnan(width) or np.isnan(height) or np.isnan(rotation)):
                width = width.astype(np.int32)
                height = height.astype(np.int32)
                rotation = rotation.astype(np.int32) + 180

                image_out = cv2.ellipse(
                    image_out,
                    (mean[1],
                     mean[0]),
                    (height / 2,
                     width / 2),
                    rotation,
                    0.0,
                    360.0,
                    _COLOUR_SCHEME_PREDICTIONS[color_index],
                    line_width,
                    lineType=cv2.LINE_AA)

            # Draw position uncertainty ellipse
            width, height, rotation = cov_ellipse(cov[2:4, 2:4])

            width[width < 0] = 0
            height[height < 0] = 0
            if not (np.isnan(width) or np.isnan(height) or np.isnan(rotation)):
                width = width.astype(np.int32)
                height = height.astype(np.int32)
                rotation = rotation.astype(np.int32) + 180

                image_out = cv2.ellipse(
                    image_out,
                    (mean[3],
                     mean[2]),
                    (height / 2,
                     width / 2),
                    rotation,
                    0.0,
                    360.0,
                    _COLOUR_SCHEME_PREDICTIONS[color_index],
                    line_width,
                    lineType=cv2.LINE_AA)

            box_u_min_mean = np.int32(mean[1])
            box_v_min_mean = np.int32(mean[0])
            box_u_max_mean = np.int32(mean[3])
            box_v_max_mean = np.int32(mean[2])

            cv2.rectangle(image_out,
                          (box_u_min_mean, box_v_min_mean),
                          (box_u_max_mean, box_v_max_mean),
                          _COLOUR_SCHEME_PREDICTIONS[color_index],
                          line_width - 1,
                          lineType=cv2.LINE_AA)

    return image_out


def cov_ellipse(cov, q=None, nsig=2):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation
