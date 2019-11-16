import numpy as np
from collections import defaultdict

from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

_HEATMAP_THRESH = 0.0027
_2D_MAH_DIST_THRESH = 3.439
_SMALL_VAL = 1e-14


def two_d_iou(box, boxes):
    """Compute 2D IOU between a 2D bounding box 'box' and a list
    :param box: a numpy array in the form of [x1, y1, x2, y2] where (x1,y1) are
    image coordinates of the top-left corner of the bounding box, and (x2,y2)
    are the image coordinates of the bottom-right corner of the bounding box.
    :param boxes: a numpy array formed as a list of boxes in the form
    [[x1, y1, x2, y2], [x1, y1, x2, y2]].
    :return iou: a numpy array containing 2D IOUs between box and every element
    in numpy array boxes.
    """
    iou = np.zeros(len(boxes), np.float64)

    x1_int = np.maximum(box[0], boxes[:, 0])
    y1_int = np.maximum(box[1], boxes[:, 1])
    x2_int = np.minimum(box[2], boxes[:, 2])
    y2_int = np.minimum(box[3], boxes[:, 3])

    w_int = np.maximum(x2_int - x1_int + 1., 0.)
    h_int = np.maximum(y2_int - y1_int + 1., 0.)

    non_empty = np.logical_and(w_int > 0, h_int > 0)

    if non_empty.any():
        intersection_area = np.multiply(w_int[non_empty], h_int[non_empty])

        box_area = (box[2] - box[0] + 1.) * (box[3] - box[1] + 1.)

        boxes_area = np.multiply(
            boxes[non_empty, 2] - boxes[non_empty, 0] + 1.,
            boxes[non_empty, 3] - boxes[non_empty, 1] + 1.)

        union_area = box_area + boxes_area - intersection_area

        iou[non_empty] = intersection_area / union_area

    return iou.round(3)

# AP calculation


def cat_pc(gt, predictions, thresholds):
    """
    Implementation refers to https://github.com/rbgirshick/py-faster-rcnn
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([[float(z) for z in b['bbox']]
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        box = p['bbox']
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recalls[:, t], precisions[:, t])

    f_score = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

    optimal_threshold = predictions[np.argmax(f_score)]['score']

    return recalls, precisions, ap, optimal_threshold, f_score[np.argmax(
        f_score)]


def compute_mu_error(gt, predictions, thresholds):
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([[float(z) for z in b['bbox']]
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}

    # rank based on entropy:
    predictions = sorted(
        predictions,
        key=lambda x: x['entropy_score'],
        reverse=False)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))

    for i, p in enumerate(predictions):
        box = p['bbox']
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        p['iou'] = 0.0
        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            p['iou'] = ovmax

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    p['is_tp'] = 1
                    tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    p['is_tp'] = 0
                    fp[i, t] = 1.
            else:
                p['is_tp'] = 0
                fp[i, t] = 1.

    tp_ind, _ = np.where(tp == 1)
    fp_ind, _ = np.where(fp == 1)

    total_tp = np.sum(tp, axis=0)
    total_fp = np.sum(fp, axis=0)

    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    u_error = 0.5 * (total_tp - tp) / np.maximum(total_tp,
                                                 1.0) + 0.5 * fp / np.maximum(total_fp, 1.0)
    min_u_error = np.min(u_error)

    scores = np.array([prediction['entropy_score']
                       for prediction in predictions])
    score_at_min_u_error = scores[np.argmin(u_error)]

    return min_u_error, score_at_min_u_error


def evaluate_detection(gt, pred, iou_thresholds=[0.5]):

    cat_gt = group_by_key(gt, 'category')
    cat_pred = group_by_key(pred, 'category')
    cat_list = sorted(cat_gt.keys())
    aps = np.zeros((len(iou_thresholds), len(cat_list)))
    optimal_score_thresholds = np.zeros_like(aps)
    maximum_f_scores = np.zeros_like(aps)
    for i, cat in enumerate(cat_list):
        if cat in cat_pred:
            r, p, ap, optimal_score_threshold, maximum_f_score = cat_pc(
                cat_gt[cat], cat_pred[cat], iou_thresholds)
            aps[:, i] = ap
            optimal_score_thresholds[:, i] = optimal_score_threshold
            maximum_f_scores[:, i] = maximum_f_score
    aps *= 100
    mAP = np.mean(aps)
    return mAP, aps.flatten().tolist(), cat_list, optimal_score_thresholds.flatten(
    ).tolist(), maximum_f_scores.flatten().tolist()


def evaluate_u_error(gt, pred, iou_thresholds=[0.5]):
    cat_gt = group_by_key(gt, 'category')
    cat_pred = group_by_key(pred, 'category')
    cat_list = sorted(cat_gt.keys())
    min_u_errors = np.zeros((len(iou_thresholds), len(cat_list)))
    scores_at_min_u_errors = np.zeros((len(iou_thresholds), len(cat_list)))

    for i, cat in enumerate(cat_list):
        if cat in cat_pred:
            min_u_errors[:, i], scores_at_min_u_errors[:, i] = compute_mu_error(
                cat_gt[cat], cat_pred[cat], iou_thresholds)

    min_u_error = np.mean(min_u_errors)
    return min_u_errors.flatten().tolist(
    ), min_u_error, cat_list, scores_at_min_u_errors.flatten().tolist()


def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


# Uncertainty calculation
def compute_gaussian_entropy_np(cov):
    dims_constant = cov.shape[1] / 2.0
    determinant = np.round(np.linalg.det(cov), 5) + 1e-12
    entropy = dims_constant + dims_constant * \
        np.log(2 * np.pi) + 0.5 * np.log(determinant)
    return entropy


def compute_categorical_entropy_np(cat_params):
    entropy = -np.sum(cat_params * np.log(cat_params))
    return entropy


def calc_heatmap(box, covs, img_size):
    """
    :param box: list of BBox corners, used to define the Gaussian corner mean locations for the box.
        Formatted [x1, y1, x2, y2]
    :param covs: list of two 2D covariance matrices used to define the covariances of the Gaussian corners.
        Formatted [cov1, cov2] where cov1 and cov2 are formatted [[var_x, corr], [corr, var_y]]
    :param img_size: size of the input image
    """

    # get all covs in format (y,x) to match matrix ordering
    covs_processed = [covs[2:4, 2:4], covs[0:2, 0:2]]
    covs2 = [np.flipud(np.fliplr(cov)) for cov in covs_processed]

    box_processed = np.array([box[1], box[0], box[3], box[2]])

    prob1 = gen_single_heatmap(
        img_size, [
            box_processed[1], box_processed[0]], covs2[0])

    prob2 = gen_single_heatmap(img_size,
                               [img_size[0] - (box_processed[3] + 1),
                                img_size[1] - (box_processed[2] + 1)],
                               np.array(covs2[1]).T)
    # flip left-right and up-down to provide probability in from bottom-right
    # corner
    prob2 = np.fliplr(np.flipud(prob2))

    # generate final heatmap
    heatmap = prob1 * prob2

    # Hack to enforce that there are no pixels with probs greater than 1 due
    # to floating point errors
    heatmap[heatmap > 1] = 1

    heatmap[heatmap < _HEATMAP_THRESH] = 0

    return heatmap


def gen_single_heatmap(img_size, mean, cov):
    """
    Function for generating the heatmap for a given Gaussian corner.
    :param img_size: tuple: formatted (n_rows, n_cols) depicting the size of the image
    :param mean: list: formatted [mu_y, mu_x] describes the location of the mean of the Gaussian corner.
    :param cov: 2D array: formatted [[var_y, corr], [corr, var_x]] describes the covariance of the Gaussian corner.
    :return: heatmap image of size <img_size> with spatial probabilities between 0 and 1.
    """
    heatmap = np.zeros(img_size, dtype=np.float32)
    g = multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    roi_box = find_roi(img_size, mean, cov)
    # Note that we subtract small value to avoid fencepost issues with
    # extremely low covariances.
    positions = np.dstack(
        np.mgrid[roi_box[1] + 1:roi_box[3] + 2, roi_box[0] + 1:roi_box[2] + 2]) - _SMALL_VAL

    prob = g.cdf(positions)

    if len(prob.shape) == 1:
        prob.shape = (roi_box[3] + 1 - roi_box[1], roi_box[2] + 1 - roi_box[0])

    heatmap[roi_box[1]:roi_box[3] + 1, roi_box[0]:roi_box[2] + 1] = prob
    heatmap[roi_box[3]:, roi_box[0]:roi_box[2] + \
        1] = np.array(heatmap[roi_box[3], roi_box[0]:roi_box[2] + 1], ndmin=2)
    heatmap[roi_box[1]:roi_box[3] +
            1, roi_box[2]:] = np.array(heatmap[roi_box[1]:roi_box[3] +
                                               1, roi_box[2]], ndmin=2).T
    heatmap[roi_box[3] + 1:, roi_box[2] + 1:] = 1.0

    # If your region of interest includes outside the main image, remove probability of existing outside the image
    # Remove probability of being outside in the x direction
    if roi_box[0] == 0:
        # points left of the image
        pos_outside_x = np.dstack(
            np.mgrid[roi_box[1] + 1:roi_box[3] + 2, 0:1]) - _SMALL_VAL
        prob_outside_x = np.zeros((img_size[0], 1), dtype=np.float32)
        prob_outside_x[roi_box[1]:roi_box[3] + 1, 0] = g.cdf(pos_outside_x)
        prob_outside_x[roi_box[3] + 1:, 0] = prob_outside_x[roi_box[3], 0]
        # Final probability is your overall cdf minus the probability in-line with that point along
        # the border for both dimensions plus the cdf at (-1, -1) which has
        # points counted twice otherwise
        heatmap -= prob_outside_x

    # Remove probability of being outside in the x direction
    if roi_box[1] == 0:
        # points above the image
        pos_outside_y = np.dstack(
            np.mgrid[0:1, roi_box[0] + 1:roi_box[2] + 2]) - _SMALL_VAL
        prob_outside_y = np.zeros((1, img_size[1]), dtype=np.float32)
        prob_outside_y[0, roi_box[0]:roi_box[2] + 1] = g.cdf(pos_outside_y)
        prob_outside_y[0, roi_box[2] + 1:] = prob_outside_y[0, roi_box[2]]
        heatmap -= prob_outside_y

    # If we've subtracted twice, we need to re-add the probability of the far
    # corner
    if roi_box[0] == 0 and roi_box[1] == 0:
        heatmap += g.cdf([[[0 - _SMALL_VAL, 0 - _SMALL_VAL]]])

    heatmap[heatmap < _HEATMAP_THRESH] = 0

    return heatmap


def find_roi(img_size, mean, cov):
    """
    Function for finding the region of interest for a probability heatmap generated by a Gaussian corner.
    This region of interest is the area with most change therein, with probabilities above 0.0027 and below 0.9973
    :param img_size: tuple: formatted (n_rows, n_cols) depicting the size of the image
    :param mean: list: formatted [mu_y, mu_x] describes the location of the mean of the Gaussian corner.
    :param cov: 2D array: formatted [[var_y, corr], [corr, var_x]] describes the covariance of the Gaussian corner.
    :return: roi_box formatted [x1, y1, x2, y2] depicting the corners of the region of interest (inclusive)
    """

    # Calculate approximate ROI
    stdy = cov[0, 0] ** 0.5
    stdx = cov[1, 1] ** 0.5

    minx = int(max(mean[1] - stdx * 5, 0))
    miny = int(max(mean[0] - stdy * 5, 0))
    maxx = int(min(mean[1] + stdx * 5, img_size[1] - 1))
    maxy = int(min(mean[0] + stdy * 5, img_size[0] - 1))

    # If the covariance is singular, we can't do any better in our estimate.
    if np.abs(np.linalg.det(cov)) < 1e-8:
        return minx, miny, maxx, maxy

    # produce list of positions [y,x] to compare to the given mean location
    approx_roi_shape = (maxy + 1 - miny, maxx + 1 - minx)
    positions = np.indices(approx_roi_shape).T.reshape(-1, 2)
    positions[:, 0] += miny
    positions[:, 1] += minx
    # Calculate the mahalanobis distances to those locations (number of standard deviations)
    # Can only do this for non-singular matrices
    mdists = cdist(
        positions,
        np.array(
            [mean]),
        metric='mahalanobis',
        VI=np.linalg.inv(cov))
    mdists = mdists.reshape(approx_roi_shape[1], approx_roi_shape[0]).T

    # Shift around the mean to change which corner of the pixel we're using
    # for the mahalanobis distance
    dist_meany = max(min(int(mean[0] - miny), img_size[0] - 1), 0)
    dist_meanx = max(min(int(mean[1] - minx), img_size[1] - 1), 0)
    if 0 < dist_meany < img_size[0] - 1:
        mdists[:dist_meany, :] = mdists[1:dist_meany + 1, :]
    if 0 < dist_meanx < img_size[1] - 1:
        mdists[:, :dist_meanx] = mdists[:, 1:dist_meanx + 1]

    # Mask out samples that are outside the desired distance (extremely low
    # probability points)
    mask = mdists <= _2D_MAH_DIST_THRESH
    # Force the pixel containing the mean to be true, we always care about that
    mask[dist_meany, dist_meanx] = True
    roi_box = generate_bounding_box_from_mask(mask)

    return roi_box[0] + minx, roi_box[1] + \
        miny, roi_box[2] + minx, roi_box[3] + miny


def generate_bounding_box_from_mask(mask):
    flat_x = np.any(mask, axis=0)
    flat_y = np.any(mask, axis=1)
    if not np.any(flat_x) and not np.any(flat_y):
        raise ValueError(
            "No positive pixels found, cannot compute bounding box")
    xmin = np.argmax(flat_x)
    ymin = np.argmax(flat_y)
    xmax = len(flat_x) - 1 - np.argmax(flat_x[::-1])
    ymax = len(flat_y) - 1 - np.argmax(flat_y[::-1])
    return [xmin, ymin, xmax, ymax]
