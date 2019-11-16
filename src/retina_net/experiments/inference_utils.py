import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import entropy
import src.core.constants as constants
from src.retina_net.anchor_generator import box_utils

"""
Tensorflow Inference Utilities
"""


@tf.function
def bayes_od_inference(model,
                       sample_dict,
                       bayes_od_config,
                       nms_config,
                       use_full_covar=False,
                       dataset_name='bdd'):

    # Get prediction from model
    prediction_dict = model(sample_dict[constants.IMAGE_NORMALIZED_KEY],
                            train_val_test='testing')

    anchors = sample_dict[constants.ANCHORS_KEY]
    predicted_box_targets = prediction_dict[
        constants.ANCHORS_BOX_PREDICTIONS_KEY]
    predicted_boxes = box_utils.box_from_anchor_and_target_bnms(
        anchors, predicted_box_targets)

    predicted_boxes_classes = tf.nn.softmax(
        prediction_dict[constants.ANCHORS_CLASS_PREDICTIONS_KEY], axis=2)

    num_classes = tf.shape(predicted_boxes_classes)[2]

    # Get categorical likelihood
    categorical_likelihood_distribution = tfp.distributions.Categorical(
        probs=tf.reduce_mean(predicted_boxes_classes, axis=0))

    categorical_samples = tf.reduce_sum(tf.one_hot(
        # Go up to 100 if possible for best results
        categorical_likelihood_distribution.sample(30),
        num_classes,
        on_value=1.0,
        off_value=0.0,
        axis=None), axis=0)

    category_filter = tf.not_equal(
        tf.argmax(
            categorical_samples, axis=1), tf.cast(
            tf.shape(predicted_boxes_classes)[2] - 1, tf.int64))

    categorical_likelihood_samples = tf.boolean_mask(
        categorical_samples, category_filter)

    # Get gaussian likelihood
    predicted_boxes = tf.boolean_mask(predicted_boxes, category_filter, axis=1)

    gaussian_likelihood_means, gaussian_likelihood_covs = compute_mean_covariance_tf(
        predicted_boxes)

    if constants.ANCHORS_COVAR_PREDICTIONS_KEY in prediction_dict.keys():

        aleatoric_covs = tf.reduce_mean(
            prediction_dict[constants.ANCHORS_COVAR_PREDICTIONS_KEY], axis=0)

        aleatoric_covs = tf.boolean_mask(
            aleatoric_covs, category_filter, axis=0)

        D_diag = tf.linalg.diag_part(tf.exp(aleatoric_covs))
        aleatoric_var_mats = tf.zeros_like(aleatoric_covs)
        aleatoric_var_mats = tf.linalg.set_diag(
            aleatoric_var_mats, D_diag)
        if use_full_covar and not tf.equal(tf.size(aleatoric_var_mats), 0):
            L = tf.linalg.inv(
                tf.linalg.set_diag(
                    aleatoric_covs,
                    tf.ones_like(D_diag)))
            aleatoric_cov_mats = tf.matmul(
                tf.matmul(L, aleatoric_var_mats), L, transpose_b=True)
        else:
            aleatoric_cov_mats = aleatoric_var_mats
    else:
        aleatoric_cov_mats = tf.zeros_like(gaussian_likelihood_covs)

    gaussian_likelihood_covs = (
        10.0 * aleatoric_cov_mats + 1.0 * gaussian_likelihood_covs) / 11.0

    # Incorporate dirichlet priors if applicable
    if bayes_od_config['dirichlet_prior']['type'] == 'non_informative':
        dirich_prior_alphas = 1.0 / tf.cast(num_classes, tf.float32)
        dirichlit_posterior_count = categorical_likelihood_samples + dirich_prior_alphas
    else:
        dirichlit_posterior_count = categorical_likelihood_samples

    categorical_posterior_score = dirichlit_posterior_count / tf.reduce_sum(
        dirichlit_posterior_count, axis=1, keepdims=True)

    # Incorporate gaussian priors if applicable
    if bayes_od_config['gaussian_prior']['type'] == 'isotropic':
        gaussian_likelihood_precisions = tf.linalg.inv(
            gaussian_likelihood_covs)

        gaussian_prior_variance = [
            bayes_od_config['gaussian_prior']['isotropic_variance']]

        gaussian_prior_var = tf.tile(
            gaussian_prior_variance, [
                tf.shape(gaussian_likelihood_precisions)[2]])

        # Update means and covariances to get the sufficient statistics
        # of the post-mc dropout posterior means and covariances

        # Compute prior mean and covariance
        gaussian_prior_covs = tf.expand_dims(
            tf.linalg.tensor_diag(gaussian_prior_var), axis=0)
        gaussian_prior_covs = tf.tile(
            gaussian_prior_covs, [
                tf.shape(gaussian_likelihood_precisions)[0], 1, 1])
        gaussian_prior_precision = tf.linalg.inv(gaussian_prior_covs)

        gaussian_prior_means = tf.boolean_mask(anchors[0], category_filter)
        gaussian_prior_means = tf.expand_dims(gaussian_prior_means, axis=2)

        # Compute posterior covariance matrices using the update equation
        # in the paper
        gaussian_posterior_precisions = gaussian_likelihood_precisions + \
            gaussian_prior_precision
        gaussian_posterior_covs = tf.linalg.inv(gaussian_posterior_precisions)

        # Compute posterior means using the update equation in the paper
        prior_mean_weights = tf.matmul(
            gaussian_prior_precision, gaussian_prior_means)

        gaussian_likelihood_means = tf.expand_dims(
            gaussian_likelihood_means, axis=2)
        gaussian_likelihood_mean_weights = tf.matmul(
            gaussian_likelihood_precisions, gaussian_likelihood_means)

        intermediate_value = prior_mean_weights + gaussian_likelihood_mean_weights
        gaussian_posterior_means = tf.matmul(
            gaussian_posterior_covs, intermediate_value)
    else:
        gaussian_posterior_covs = gaussian_likelihood_covs
        gaussian_posterior_means = gaussian_likelihood_means

    if dataset_name == 'kitti':
        # Scaling is required on both mean and covariance, as the neural
        # network estimates values in a scaled version of the input dataset
        # image for the KITTI dataset only
        scale = sample_dict[constants.ORIGINAL_IM_SIZE_KEY][0] / \
            tf.shape(sample_dict[constants.IMAGE_NORMALIZED_KEY][0])
        scale = tf.tile(scale[0:2], [tf.shape(aleatoric_cov_mats)[2] / 2])
        scale_mat = tf.expand_dims(
            tf.linalg.tensor_diag(scale), axis=0)
        scale_mat = tf.tile(
            scale_mat, [
                tf.shape(aleatoric_cov_mats)[0], 1, 1])
        scale_mat = tf.cast(scale_mat, tf.float32)
        gaussian_posterior_means = tf.matmul(scale_mat,
                                             gaussian_posterior_means)
        gaussian_posterior_covs = tf.matmul(
            tf.matmul(
                scale_mat,
                gaussian_posterior_covs),
            scale_mat,
            transpose_b=True)

    if bayes_od_config['ranking_method'] == 'joint_entropy' and bayes_od_config['gaussian_prior'][
            'type'] != 'None' and bayes_od_config['dirichlet_prior']['type'] != 'None':
        gaussian_entropy_posterior = compute_gaussian_entropy_tf(
            gaussian_posterior_covs)

        gaussian_entropy_prior = compute_gaussian_entropy_tf(
            gaussian_prior_covs)

        gaussian_info_gain = gaussian_entropy_prior - gaussian_entropy_posterior

        gaussian_info_gain = (
            gaussian_info_gain - tf.reduce_min(gaussian_info_gain)) / tf.maximum(
            1.0, tf.reduce_max(gaussian_info_gain) - tf.reduce_min(gaussian_info_gain))

        categorical_entropy_posterior = compute_categorical_entropy_tf(
            categorical_posterior_score)

        categorical_prior_score = tf.tile([[dirich_prior_alphas]], [
            1, num_classes])
        categorical_prior_score = categorical_prior_score / \
            tf.reduce_sum(categorical_prior_score)

        categorical_entropy_prior = compute_categorical_entropy_tf(
            categorical_prior_score)

        categorical_info_gain = categorical_entropy_prior - categorical_entropy_posterior

        categorical_info_gain = (
            categorical_info_gain - tf.reduce_min(categorical_info_gain)) / tf.maximum(
            0.001,
            tf.reduce_max(categorical_info_gain) - tf.reduce_min(categorical_info_gain))
        ranking_scores = categorical_info_gain + gaussian_info_gain
    else:
        ranking_scores = tf.reduce_max(categorical_posterior_score, axis=1)

    predicted_boxes_corners = box_utils.vuhw_to_vuvu(
        tf.squeeze(gaussian_posterior_means, axis=2))

    nms_indices, _ = tf.image.non_max_suppression_with_scores(
        predicted_boxes_corners,
        ranking_scores,
        max_output_size=nms_config['max_output_size'],
        iou_threshold=nms_config['iou_threshold'],
        soft_nms_sigma=nms_config['soft_nms_sigma'])

    predicted_boxes_iou_mat = box_utils.bbox_iou_vuvu(
        predicted_boxes_corners, predicted_boxes_corners)

    return dirichlit_posterior_count, gaussian_posterior_means, gaussian_posterior_covs, nms_indices, predicted_boxes_iou_mat


def compute_mean_covariance_tf(output_boxes):
    """
    Given the inference results from M runs of MC dropout,
     computes the mean and covariance of every anchor

    :param output_boxes: MxNx4 tensor containing inference results from M runs of MC dropout

    :return: mean: Nx4 tensor containing the per anchor mean
    :return: cov: Nx4x4 tensor containing the per anchor covariance matrix

    """

    # Compute mean
    mean = tf.reduce_mean(output_boxes, axis=0, keepdims=True)

    # Compute Covariance Matrix
    intermediate_compute = output_boxes - mean
    intermediate_compute = tf.matmul(
        tf.transpose(intermediate_compute, [1, 2, 0]),
        tf.transpose(intermediate_compute, [1, 0, 2]))

    cov = intermediate_compute / \
        (tf.cast(tf.shape(output_boxes)[0], tf.float32) - 1.0)

    return mean[0], cov


def compute_gaussian_entropy_tf(cov):
    """
    Computes the differential entropy of gaussian distributions parameterized
    by their covariance matrices

    :param cov: Nx4x4 tensor of covarience matrices
    :return: entropy: Nx1 tensor of differential entropies
    """

    dims_constant = tf.cast(tf.shape(cov)[2] / 2, tf.float32)

    determinant = tf.linalg.det(cov)

    entropy = dims_constant + dims_constant * \
        tf.math.log(2.0 * np.pi) + 0.5 * tf.math.log(determinant)

    return tf.cast(entropy, tf.float32)


def compute_categorical_entropy_tf(cat_params):
    """
    Computes the shannon entropy of categorical distributions parameterized
    by parameters p_1..p_k

    :param cat_params: NxK tensor containing the parameters p_1...p_k of the categorical distribution per anchor
    :return: entropy: Nx1 tensor of shannon entropies
    """

    entropy = -tf.reduce_sum(cat_params * tf.math.log(cat_params), axis=1)

    return tf.cast(entropy, tf.float32)


"""
Numpy Utilities
"""


def bayes_od_clustering(
        predicted_boxes_class_counts,
        predicted_boxes_means,
        predicted_boxes_covs,
        cluster_centers,
        affinity_matrix,
        affinity_threshold=0.7):
    """
    Bayesian NMS clustering to output a single probability distribution per object in the scene.

    :param predicted_boxes_means: Nx4x1 tensor containing resulting means from mc_dropout. N is the number of anchors processed by the model.
    :param predicted_boxes_covs: Nx4x4 tensor containing resulting covariance matrices from mc_dropout.
    :param cat_count_res: NxC tensor containing the alpha counts from mc dropout. C is the number of categories.
    :param cluster_centers: Kx1 tensor containing the indices of centers of clusters based on minimum entropy. K is the number of clusters.
    :param affinity_matrix: NxN matrix containing affinity measures between bounding boxes.
    :param affinity_threshold: scalar to determine the minimum affinity for clustering.

    :return: final_scores: Kx4 tensor containing the parameters of the final categorical posterior distribution describing the objects' categories.
    :return: final_means: Kx4x1 tensor containing the mean vectors of the posterior gaussian distribution describing objects' location in the scene.
    :return: final_covs: Kx4x4 tensor containing the covariance matrices of the posterior gaussian distribution describing objects' location in the scene.
    """

    # Initialize lists to save per cluster results
    final_box_means = []
    final_box_covs = []
    final_box_class_scores = []
    final_box_class_counts = []
    for cluster_center in cluster_centers:

        # Get bounding boxes with affinity > threshold with the center of the
        # cluster
        cluster_inds = affinity_matrix[:, cluster_center] > affinity_threshold
        cluster_means = predicted_boxes_means[cluster_inds, :, :]
        cluster_covs = predicted_boxes_covs[cluster_inds, :, :]

        # Compute mean and covariance of the final posterior distribution
        cluster_precs = np.array(
            [np.linalg.inv(member_cov) for member_cov in cluster_covs])

        final_cov = np.linalg.inv(np.sum(cluster_precs, axis=0))
        final_box_covs.append(final_cov)

        mean_temp = np.array([np.matmul(member_prec, member_mean) for
                              member_prec, member_mean in
                              zip(cluster_precs, cluster_means)])
        mean_temp = np.sum(mean_temp, axis=0)
        final_box_means.append(np.matmul(final_cov, mean_temp))

        # Compute the updated parameters of the categorical distribution
        final_counts = predicted_boxes_class_counts[cluster_inds, :]
        final_score = (final_counts) / \
            np.expand_dims(np.sum(final_counts, axis=1), axis=1)

        if final_score.shape[0] > 3:
            cluster_center_score = np.expand_dims(predicted_boxes_class_counts[cluster_center, :] / np.sum(
                predicted_boxes_class_counts[cluster_center, :]), axis=0)
            cluster_center_score = np.repeat(
                cluster_center_score, final_score.shape[0], axis=0)

            cat_ent = entropy(cluster_center_score.T, final_score.T)

            inds = np.argpartition(cat_ent, 3)[:3]

            final_score = final_score[inds]
            final_counts = final_counts[inds]

        final_score = np.mean(final_score, axis=0)
        final_counts = np.sum(final_counts, axis=0)
        final_box_class_scores.append(final_score)
        final_box_class_counts.append(final_counts)

    final_box_means = np.array(final_box_means)
    final_box_class_scores = np.array(final_box_class_scores)

    # 70 is a calibration parameter specific to BDD and KITTI datasets. It
    # gives the best PDQ. It does not affect AP or MUE.
    final_box_covs = np.array(final_box_covs) * 70
    final_box_class_counts = np.array(final_box_class_counts)

    return final_box_class_scores, final_box_means, final_box_covs, final_box_class_counts


"""
General Utilities
"""


def map_dataset_classes(input_dataset, target_dataset, output_classes):
    set_to_set_mapping_dict = constants.SET_TO_SET_MAPPING_DICTS[
        input_dataset + '_' + target_dataset]
    if set_to_set_mapping_dict:
        input_class_to_idx_mapping_dict = constants.CATEGORY_IDX_MAPPING_DICTS[input_dataset]
        target_class_to_idx_mapping_dict = constants.CATEGORY_IDX_MAPPING_DICTS[target_dataset]

        output_classes_mapped = np.zeros(
            [output_classes.shape[0], len(target_class_to_idx_mapping_dict) + 1])
        if len(output_classes.shape) == 1:
            output_classes = np.expand_dims(output_classes, axis=1)
        max_idxs = np.argmax(output_classes, axis=1)

        mapping_categories = list(input_class_to_idx_mapping_dict.keys())
        mapping_idxs = list(input_class_to_idx_mapping_dict.values())

        mapping_idxs = np.take(mapping_idxs, max_idxs)
        mapping_categories = np.take(mapping_categories, mapping_idxs)

        mapped_categories = [set_to_set_mapping_dict[mapping_category]
                             for mapping_category in mapping_categories]
        mapped_idxs = [target_class_to_idx_mapping_dict[mapped_category]
                       for mapped_category in mapped_categories]

        max_scores = np.amax(output_classes, axis=1)

        for score, mapped_idx, mapped_row in zip(
                max_scores, mapped_idxs, output_classes_mapped):
            mapped_row[mapped_idx] = score

        return output_classes_mapped
    else:
        return output_classes
