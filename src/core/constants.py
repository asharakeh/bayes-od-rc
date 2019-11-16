"""
Retinanet Constants
"""
# Define difficulty dictionaries for gt label filtering
KITTI_DIFF_DICTS = {
    'easy': {'min_height': 40, 'max_occlusion': 0, 'max_truncation': 0.15},
    'moderate': {'min_height': 25, 'max_occlusion': 1, 'max_truncation': 0.30},
    'hard': {'min_height': 25, 'max_occlusion': 2, 'max_truncation': 0.50},
    'all': {'min_height': 0, 'max_occlusion': 3, 'max_truncation': 1.0}}

# RGB mean dicts used for subtraction to normalize the input image
MEANS_DICT = {'ImageNet': [123.68, 116.78, 103.94],
              'Kitti': [92.84, 97.80, 93.58]}

# Category mapping dicts, from class name to category index in one hot vector.
CATEGORY_IDX_MAPPING_DICTS = {'kitti': {'car': 0,
                                        'pedestrian': 1,
                                        'cyclist': 2,
                                        'bknd': 3},

                              'bdd': {'car': 0,
                                      'truck': 1,
                                      'bus': 2,
                                      'person': 3,
                                      'rider': 4,
                                      'bike': 5,
                                      'motor': 6,
                                      'bknd': 7}}


# Dataset to dataset category mapping. Usefull when training and inference
# datasets are different.
SET_TO_SET_MAPPING_DICTS = {'bdd_kitti': {'car': 'car',
                                          'truck': 'car',
                                          'bus': 'car',
                                          'person': 'pedestrian',
                                          'rider': 'cyclist',
                                          'bike': 'cyclist',
                                          'motor': 'cyclist',
                                          'bknd': 'bknd'},

                            'coco_rvc': {},
                            'coco_pascal': {}}

"""
Sample Dict Keys
"""
IMAGE_NORMALIZED_KEY = 'image_normalized'
ORIGINAL_IM_SIZE_KEY = 'im_size'
ANCHORS_KEY = 'anchors'
ANCHORS_BOX_TARGETS_KEY = 'anchors_box_targets'
ANCHORS_CLASS_TARGETS_KEY = 'anchors_class_targets'
POSITIVE_ANCHORS_MASK_KEY = 'positive_anchors_mask'
NEGATIVE_ANCHOR_MASK_KEY = 'negative_anchors_mask'
IMAGE_PADDING_KEY = 'paddings_applied'


"""
Prediction Dict Keys
"""
ANCHORS_BOX_PREDICTIONS_KEY = 'anchors_box_predictions'
ANCHORS_COVAR_PREDICTIONS_KEY = 'anchors_box_covar_predictions'
ANCHORS_CLASS_PREDICTIONS_KEY = 'anchors_class_predictions'


"""
Loss Dict Keys
"""
CLS_LOSS_KEY = 'cls_loss'
REG_LOSS_KEY = 'reg_loss'
COV_LOSS_KEY = 'covariance_loss'
MC_COV_LOSS_KEY = 'mc_cov_loss'

REGULARIZATION_LOSS_KEY = 'regularization_loss'
