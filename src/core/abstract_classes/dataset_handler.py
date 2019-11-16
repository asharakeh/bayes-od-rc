from abc import abstractmethod


class DatasetHandler:
    """
    Dataset handler abstract class.
    """

    def __init__(self, dataset_config):

        # Parse dataset config shared params
        self.data_split = dataset_config['data_split']
        self.im_normalization = dataset_config['im_normalization']

    @abstractmethod
    def create_dataset(self):
        pass
