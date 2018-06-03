import unittest
import numpy as np
from skimage.util.shape import view_as_windows

from skater.util.image_ops import view_windows
from skater.util.image_ops import load_image


class TestImageOps(unittest.TestCase):

    def setUp(self):
        pass


    def test_loading_imgs(self):
        img = load_image('skater/tests/data/pandas.jpg', 299, 299)
        self.assertEquals(img.shape, (299, 299, 3))
        img = load_image('skater/tests/data/pandas.jpg', 200, 299)
        self.assertEquals(img.shape, (200, 299, 3))


    def test_view_windows(self):
        input_matrix = np.arange(10 * 10).reshape(10, 10)
        padded_input = np.pad(input_matrix, ((0, 2), (0, 2)), mode='constant', constant_values=0)
        input_view = view_as_windows(padded_input, (4, 4), 4)
        # apply rolling window with padding to handle corner cases
        input_view_modified = view_windows(padded_input, (4, 4), 4)
        self.assertEquals(input_view.shape, input_view_modified.shape)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImageOps)
    unittest.TextTestRunner(verbosity=2).run(suite)
