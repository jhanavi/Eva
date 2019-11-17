import os
import unittest
import cv2

from src.models import Frame, FrameBatch
from src.udfs.depth_estimator import DepthEstimator


class DepthEstimatorTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def _load_image(self, path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def test_should_return_batches_equivalent_to_number_of_frames(self):
        frame_dog = Frame(1, self._load_image(
            os.path.join(self.base_path, 'data', 'dog.jpeg')), None)
        frame_dog_cat = Frame(1, self._load_image(
            os.path.join(self.base_path, 'data', 'dog_cat.jpg')), None)
        frame_batch = FrameBatch([frame_dog, frame_dog_cat], None)
        estimator = DepthEstimator()
        result = estimator.process_frames(frame_batch)

        print(result)
        print(result[0].frame)
        print(result[0].depth)
        print(result[0].segm)