"""Cleanliness classifier using the same architecture as the damage model."""

from model import EggGradingCNN


class EggCleanlinessCNN(EggGradingCNN):
    """Binary classifier for clean versus stained egg crops."""

    class_names = ("Clean", "Stained")

    def __init__(self):
        super().__init__(num_classes=len(self.class_names))
