from abc import abstractmethod


class AnchorGenerator:
    """
    Anchor generator abstract class
    """
    def __init__(self, generator_config):
        self.config = generator_config

    @abstractmethod
    def generate_anchors(self, **kwargs):
        pass
