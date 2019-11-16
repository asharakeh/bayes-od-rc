import numpy as np


def read_bdd_format(
        sample_id,
        bdd_dict,
        categories=('car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor'),
        pdq_eval=False):
    """
    Reads bdd format from json file output.

    output format is described in bdd dataset as:
    {
      "name": str,
      "timestamp": 1000,
      "category": str,
      "bbox": [x1, y1, x2, y2],
      "score": float
   }


    """
    # Define constants
    no_elem = False
    boxes_class_one_hot = []

    # Extract the list
    frame_elements = [elem for elem in bdd_dict if
                      elem['name'] == sample_id and elem[
                          'category'] in categories]

    if pdq_eval:
        boxes_2d = np.array([label['bbox'] for label in frame_elements])
    else:
        boxes_2d = np.array([[label['bbox'][1],
                              label['bbox'][0],
                              label['bbox'][3],
                              label['bbox'][2]] for label in frame_elements])

    boxes_cat = [elem['category'] for elem in frame_elements]

    if boxes_2d.size == 0:
        cat_one_hot = [0 for i in range(len(categories) + 1)]
        cat_one_hot[len(categories)] = 1
        boxes_2d = np.array([0.0, 0.0, 1.0, 1.0])
        boxes_class_one_hot.append(cat_one_hot)
        no_elem = True
    else:
        for cat in boxes_cat:
            cat_one_hot = [0 for i in range(len(categories) + 1)]
            cat_idx = categories.index(cat.lower())
            cat_one_hot[cat_idx] = 1
            boxes_class_one_hot.append(cat_one_hot)

    # one-hot representation: ['car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor', background]
    if len(boxes_2d.shape) == 1:
        boxes_2d = np.expand_dims(boxes_2d, axis=0)

    return [np.array(boxes_class_one_hot).astype(np.float32),
            np.array(boxes_2d).astype(np.float32),
            no_elem]
