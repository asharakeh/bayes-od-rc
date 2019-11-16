from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gmean
from multiprocessing import Pool, cpu_count

_SMALL_VAL = 1e-14


class PDQ(object):
    """
    Class for calculating PDQ for a set of images.
    """

    def __init__(self):
        super(PDQ, self).__init__()
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0

    def add_img_eval(self, gt_instances, det_instances):
        """
        Adds a single image's detections and ground-truth to the overall evaluation analysis.
        :param gt_instances: list of GroundTruthInstance objects present in the given image.
        :param det_instances: list of DetectionInstance objects provided for the given image
        :return: None
        """
        results = _calc_qual_img(gt_instances, det_instances)
        self._tot_overall_quality += results['overall']
        self._tot_spatial_quality += results['spatial']
        self._tot_label_quality += results['label']
        self._tot_TP += results['TP']
        self._tot_FP += results['FP']
        self._tot_FN += results['FN']

    def get_pdq_score(self):
        """
        Get the current PDQ score for all frames analysed at the current time.
        :return: The average PDQ across all images as a float.
        """
        tot_pairs = self._tot_TP + self._tot_FP + self._tot_FN
        return self._tot_overall_quality / tot_pairs

    def reset(self):
        """
        Reset all internally stored evaluation measures to zero.
        :return: None
        """
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0

    def score(self, matches):
        """
        Calculates the average probabilistic detection quality for a set of detections on
        a set of ground truth objects over a series of images.
        The average is calculated as the average pairwise quality over the number of object-detection pairs observed.
        Note that this removes any evaluation information that had been stored for previous images.
        Assumes you want to score just the full list you are given.
        :param matches: A list of tuples where each tuple holds a list of GroundTruthInstances and
        DetectionInstances respectively, describing the ground truth objects and detections for a given image.
        Each image observed is an entry in the main list.
        :return: The average PDQ across all images as a float.
        """
        self.reset()
        pool = Pool(processes=cpu_count())
        for img_results in pool.imap_unordered(_get_image_evals,
                                               iterable=matches):
            self._tot_overall_quality += img_results['overall']
            self._tot_spatial_quality += img_results['spatial']
            self._tot_label_quality += img_results['label']
            self._tot_TP += img_results['TP']
            self._tot_FP += img_results['FP']
            self._tot_FN += img_results['FN']

        pool.close()
        pool.join()

        return self.get_pdq_score()

    def get_avg_spatial_score(self):
        """
        Get the average spatial quality score for all assigned detections in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average spatial quality of every detection
        """
        if (self._tot_TP + self._tot_FP) > 0.0:
            return self._tot_spatial_quality / \
                float(self._tot_TP + self._tot_FP)
        return 0.0

    def get_avg_label_score(self):
        """
        Get the average label quality score for all assigned detections in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average label quality of every detection
        """
        if (self._tot_TP + self._tot_FP) > 0.0:
            return self._tot_label_quality / float(self._tot_TP + self._tot_FP)
        return 0.0

    def get_avg_overall_quality_score(self):
        """
        Get the average overall pairwise quality score for all assigned detections
        in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average overall pairwise quality of every  detection
        """
        if (self._tot_TP + self._tot_FP) > 0.0:
            return self._tot_overall_quality / \
                float(self._tot_TP + self._tot_FP)
        return 0.0

    def get_assignment_counts(self):
        """
        Get the total number of TP, FP, and FN detections across all frames analysed at the current time.
        :return: tuple containing (TP, FP, FN)
        """
        return self._tot_TP, self._tot_FP, self._tot_FN


def _get_image_evals(pair):
    """
    Evaluate the results for a given image
    :param pair: tuple containing list of GroundTruthInstances and DetectionInstances for the given image respectively
    :return: results dictionary containing total overall spatial quality, total spatial quality on positively assigned
    detections, total label quality on positively assigned detections, number of true positives,
    number of false positives, and number false negatives for the given image.
    Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
    'TP': <num_true_positives>, 'FP': <num_false_positives>, 'FN': <num_false_positives>}
    """
    gt_instances, det_instances = pair
    results = _calc_qual_img(gt_instances, det_instances)
    return results


def _vectorize_img_gts(gt_instances, img_shape):
    """
    Vectorizes the required elements for all GroundTruthInstances as necessary for a given image.
    These elements are the segmentation mask, background mask, number of foreground pixels, and label for each.
    :param gt_instances: list of all GroundTruthInstances for a given image
    :param img_shape: shape of the image that the GroundTruthInstances lie within
    :return: (gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec).
    gt_seg_mat: h x w x g boolean numpy array depicting the ground truth pixels for each of the g GroundTruthInstances
    within an h x w image.
    bg_seg_mat: h x w x g boolean numpy array depicting the background pixels for each of the g GroundTruthInstances
    (pixels outside the bounding box) within an h x w image.
    num_fg_pixels_vec: g x 1 int numpy array containing the number of foreground (object) pixels for each of
    the g GroundTruthInstances.
    gt_label_vec: g, numpy array containing the class label as an integer for each of the g GroundTruthInstances
    """
    gt_seg_mat = np.stack(
        [gt_instance.segmentation_mask for gt_instance in gt_instances], axis=2)   # h x w x g
    num_fg_pixels_vec = np.array([[gt_instance.num_pixels]
                                  for gt_instance in gt_instances], dtype=np.int)  # g x 1
    gt_label_vec = np.array(
        [gt_instance.class_label for gt_instance in gt_instances], dtype=np.int)        # g,

    bg_seg_mat = np.ones(img_shape + (len(gt_instances),),
                         dtype=np.bool)  # h x w x g
    for gt_idx, gt_instance in enumerate(gt_instances):
        gt_box = gt_instance.bounding_box
        bg_seg_mat[gt_box[1]:gt_box[3] + 1,
                   gt_box[0]:gt_box[2] + 1, gt_idx] = False

    return gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec


def _vectorize_img_dets(det_instances, img_shape):
    """
    Vectorize the required elements for all DetectionInstances as necessary for a given image.
    These elements are the thresholded detection heatmap, and the detection label list for each.
    :param det_instances: list of all DetectionInstances for a given image.
    :param img_shape: shape of the image that the DetectionInstances lie within.
    :return: (det_seg_heatmap_mat, det_label_prob_mat)
    det_seg_heatmap_mat: h x w x d float32 numpy array depciting the probability that each pixel is part of the
    detection within an h x w image. Note that this is thresholded so pixels with particularly low probabilities instead
    have a probability in the heatmap of zero.
    det_label_prob_mat: d x c numpy array of label probability scores across all c classes for each of the d detections
    """
    det_label_prob_mat = np.stack(
        [det_instance.class_list for det_instance in det_instances], axis=0)  # d x c
    det_seg_heatmap_mat = np.stack([det_instance.calc_heatmap(
        img_shape) for det_instance in det_instances], axis=2)
    return det_seg_heatmap_mat, det_label_prob_mat


def _calc_bg_loss(bg_seg_mat, det_seg_heatmap_mat):
    """
    Calculate the background pixel loss for all detections on all ground truth objects for a given image.
    :param bg_seg_mat: h x w x g vectorized background masks for each ground truth object in the image.
    :param det_seg_heatmap_mat: h x w x d vectorized segmented heatmaps for each detection in the image.
    :return: (bg_loss_sum, num_bg_pixels_mat)
    bg_loss_sum: g x d total background loss between each of the g ground truth objects and d detections.
    num_bg_pixels_mat: g x d number of background pixels examined for each combination of g ground truth objects and d
    detections.
    """
    bg_log_loss_mat = _safe_log(
        1 - det_seg_heatmap_mat) * (det_seg_heatmap_mat > 0)
    bg_loss_sum = np.tensordot(
        bg_seg_mat, bg_log_loss_mat, axes=([0, 1], [0, 1]))  # g x d
    return bg_loss_sum


def _calc_fg_loss(gt_seg_mat, det_seg_heatmap_mat):
    """
    Calculate the foreground pixel loss for all detections on all ground truth objects for a given image.
    :param gt_seg_mat: h x w x g vectorized segmentation masks for each ground truth object in the image.
    :param det_seg_heatmap_mat: h x w x d vectorized segmented heatmaps for each detection in the image.
    :return: fg_loss_sum: g x d total foreground loss between each of the g ground truth objects and d detections.
    """
    log_heatmap_mat = _safe_log(det_seg_heatmap_mat)
    fg_loss_sum = np.tensordot(
        gt_seg_mat, log_heatmap_mat, axes=([0, 1], [0, 1]))  # g x d
    return fg_loss_sum


def _safe_log(mat):
    return np.log(mat + _SMALL_VAL)


def _calc_spatial_qual(fg_loss_sum, bg_loss_sum, num_fg_pixels_vec):
    """
    Calculate the spatial quality for all detections on all ground truth objects for a given image.
    :param fg_loss_sum: g x d total foreground loss between each of the g ground truth objects and d detections.
    :param bg_loss_sum: g x d total background loss between each of the g ground truth objects and d detections.
    :param num_fg_pixels_vec: g x 1 number of pixels for each of the g ground truth objects.
    :return: spatial_quality: g x d spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    total_loss = fg_loss_sum + bg_loss_sum

    loss_per_gt_pixel = total_loss / num_fg_pixels_vec

    spatial_quality = np.exp(loss_per_gt_pixel)

    # Deal with tiny floating point errors or tiny errors caused by _SMALL_VAL
    # that prevent perfect 0 or 1 scores
    spatial_quality[np.isclose(spatial_quality, 0)] = 0
    spatial_quality[np.isclose(spatial_quality, 1)] = 1

    return spatial_quality


def _calc_label_qual(gt_label_vec, det_label_prob_mat):
    """
    Calculate the label quality for all detections on all ground truth objects for a given image.
    :param gt_label_vec:  g, numpy array containing the class label as an integer for each object.
    :param det_label_prob_mat: d x c numpy array of label probability scores across all c classes
    for each of the d detections.
    :return: label_qual_mat: g x d label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    label_qual_mat = det_label_prob_mat[:, gt_label_vec].T.astype(
        np.float32)     # g x d
    return label_qual_mat


def _calc_overall_qual(label_qual, spatial_qual):
    """
    Calculate the overall quality for all detections on all ground truth objects for a given image
    :param label_qual: g x d label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :param spatial_qual: g x d spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :return: overall_qual_mat: g x d overall label quality between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    combined_mat = np.dstack((label_qual, spatial_qual))

    # Calculate the geometric mean between label quality and spatial quality.
    # Note we ignore divide by zero warnings here for log(0) calculations
    # internally.
    with np.errstate(divide='ignore'):
        overall_qual_mat = gmean(combined_mat, axis=2)

    return overall_qual_mat


def _gen_cost_tables(gt_instances, det_instances):
    """
    Generate the cost tables containing the cost values (1 - quality) for each combination of ground truth objects and
    detections within a given image.
    :param gt_instances: list of all GroundTruthInstances for a given image.
    :param det_instances: list of all DetectionInstances for a given image.
    :return: g x d overall cost table for each combination of ground truth objects and detections
    """
    n_pairs = max(len(gt_instances), len(det_instances))
    overall_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    spatial_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    label_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    img_shape = gt_instances[0].segmentation_mask.shape

    # Generate all the matrices needed
    gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec = _vectorize_img_gts(
        gt_instances, img_shape)
    img_shape = gt_instances[0].segmentation_mask.shape
    det_seg_heatmap_mat, det_label_prob_mat = _vectorize_img_dets(
        det_instances, img_shape)

    # Calculate all qualities
    label_qual_mat = _calc_label_qual(gt_label_vec, det_label_prob_mat)
    fg_loss = _calc_fg_loss(gt_seg_mat, det_seg_heatmap_mat)
    bg_loss = _calc_bg_loss(bg_seg_mat, det_seg_heatmap_mat)
    spatial_qual = _calc_spatial_qual(fg_loss, bg_loss, num_fg_pixels_vec)

    # Generate the overall cost table (1 - overall quality)
    overall_cost_table[:len(gt_instances), :len(
        det_instances)] -= _calc_overall_qual(label_qual_mat, spatial_qual)

    # Generate the spatial and label cost tables
    spatial_cost_table[:len(gt_instances), :len(det_instances)] -= spatial_qual
    label_cost_table[:len(gt_instances), :len(det_instances)] -= label_qual_mat

    return {'overall': overall_cost_table, 'spatial': spatial_cost_table, 'label': label_cost_table}, {
        'det_seg_heatmap_mat': det_seg_heatmap_mat, 'det_label_prob_mat': det_label_prob_mat}


def _calc_qual_img(gt_instances, det_instances):
    """
    Calculates the sum of qualities for the best matches between ground truth objects and detections for an image.
    Each ground truth object can only be matched to a single detection and vice versa as an object-detection pair.
    Note that if a ground truth object or detection does not have a match, the quality is counted as zero.
    This represents a theoretical object-detection pair with the object or detection and a counterpart which
    does not describe it at all.
    Any provided detection with a zero-quality match will be counted as a false positive (FP).
    Any ground-truth object with a zero-quality match will be counted as a false negative (FN).
    All other matches are counted as "true positives" (TP)
    If there are no ground truth objects or detections for the image, the system returns zero and this image
    will not contribute to average_PDQ.
    :param gt_instances: list of GroundTruthInstance objects describing the ground truth objects in the current image.
    :param det_instances: list of DetectionInstance objects describing the detections for the current image.
    :return: results dictionary containing total overall spatial quality, total spatial quality on positively assigned
    detections, total label quality on positively assigned detections, number of true positives,
    number of false positives, and number false negatives for the given image.
    Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
    'TP': <num_true_positives>, 'FP': <num_false_positives>, 'FN': <num_false_positives>}
    """
    # if there are no detections or gt instances respectively the quality is
    # zero
    if len(gt_instances) == 0 or len(det_instances) == 0:
        FN = 0

        # Filter out GT instances which are to be ignored because they are too
        # small
        if len(gt_instances) > 0:
            for gt_idx, gt_instance in enumerate(gt_instances):
                if _is_gt_included(gt_instance):
                    FN += 1

        return {
            'overall': 0.0,
            'spatial': 0.0,
            'label': 0.0,
            'TP': 0,
            'FP': len(det_instances),
            'FN': FN}

    # For each possible pairing, calculate the quality of that pairing and convert it to a cost
    # to enable use of the Hungarian algorithm.
    cost_tables, detection_properties = _gen_cost_tables(
        gt_instances, det_instances)

    # Use the Hungarian algorithm with the cost table to find the best match between ground truth
    # object and detection (lowest overall cost representing highest overall
    # pairwise quality)
    row_idxs, col_idxs = linear_sum_assignment(cost_tables['overall'])

    # Transform the loss tables back into quality tables with values between 0
    # and 1
    overall_quality_table = 1 - cost_tables['overall']
    spatial_quality_table = 1 - cost_tables['spatial']
    label_quality_table = 1 - cost_tables['label']

    # Calculate the number of TPs, FPs, and FNs for the image.
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    false_positives_col_id = []
    for match_idx, match in enumerate(zip(row_idxs, col_idxs)):
        row_id, col_id = match
        if overall_quality_table[row_id, col_id] > 0:
            if row_id < len(gt_instances) and _is_gt_included(
                    gt_instances[row_id]):
                true_positives += 1
            else:
                # ignore detections on samples which are too small to be
                # considered a valid object
                overall_quality_table[row_id, col_id] = 0.0
        else:
            if row_id < len(gt_instances) and _is_gt_included(
                    gt_instances[row_id]):
                false_negatives += 1
            if col_id < len(det_instances):
                false_positives += 1
                false_positives_col_id.append(col_id)

    # Calculate the sum of quality at the best matching pairs to calculate
    # total qualities for the image
    tot_tp_overall_img_quality = np.sum(
        overall_quality_table[row_idxs, col_idxs])

    # Calculate the sum of spatial and label qualities only for TP samples
    spatial_quality_table[overall_quality_table == 0] = 0.0
    label_quality_table[overall_quality_table == 0] = 0.0
    tot_tp_spatial_quality = np.sum(spatial_quality_table[row_idxs, col_idxs])
    tot_tp_label_quality = np.sum(label_quality_table[row_idxs, col_idxs])

    # Calculate the penalty for assigning a high probability to false positives
    fp_label_quality_table = np.array(
        [1.0 - np.max(detection_properties['det_label_prob_mat'][i]) for i in false_positives_col_id])
    tot_fp_label_quality = np.sum(fp_label_quality_table)

    fp_heat_maps = np.array(
        [detection_properties['det_seg_heatmap_mat'][:, :, i] for i in false_positives_col_id])
    if fp_label_quality_table.size:
        fp_det_area = np.array([_compute_bb_area(det_instances[i])
                                for i in false_positives_col_id])
        fp_spatial_quality_table = np.exp(np.sum(_safe_log(
            1 - fp_heat_maps) * (fp_heat_maps > 0), axis=(1, 2)) / fp_det_area)

        tot_fp_spatial_quality = np.sum(fp_spatial_quality_table)
        tot_fp_overall_img_quality = np.sum(
            _calc_overall_qual(
                fp_spatial_quality_table,
                fp_label_quality_table))
    else:
        tot_fp_spatial_quality = 0.0
        tot_fp_overall_img_quality = 0.0

    tot_label_quality = tot_tp_label_quality + tot_fp_label_quality
    tot_spatial_quality = tot_tp_spatial_quality + tot_fp_spatial_quality
    tot_overall_img_quality = tot_tp_overall_img_quality + tot_fp_overall_img_quality

    return {
        'overall': tot_overall_img_quality,
        'spatial': tot_spatial_quality,
        'label': tot_label_quality,
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives}


def _compute_bb_area(det_instance):
    bbox = det_instance.box
    area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
    return area


def _is_gt_included(gt_instance):
    """
    Determines if a ground-truth instance is large enough to be considered valid for detection
    :param gt_instance: GroundTruthInstance object being evaluated
    :return: Boolean describing if the object is valid for detection
    """
    return (
        gt_instance.bounding_box[2] -
        gt_instance.bounding_box[0] > 10) and (
        gt_instance.bounding_box[3] -
        gt_instance.bounding_box[1] > 10) and np.count_nonzero(
            gt_instance.segmentation_mask) > 100
