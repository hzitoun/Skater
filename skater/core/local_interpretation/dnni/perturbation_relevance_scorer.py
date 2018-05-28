# -*- coding: UTF-8 -*-
from skater.util.image_ops import view_windows
from skater.core.local_interpretation.dnni.initializer import Initializer
from skater.util.logger import build_logger
from skater.util.logger import _INFO
from skater.util.exceptions import TensorflowUnavailableError
try:
    import tensorflow as tf
except ImportError:
    raise (TensorflowUnavailableError("TensorFlow binaries are not installed"))

import numpy as np


class BasePerturbationMethod(Initializer):
    """
    Base class for perturbation-based relevance/attribution computation

    """

    __name__ = "BasePerturbationMethod"
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, current_session):
        super(BasePerturbationMethod, self).__init__(output_tensor, input_tensor, samples, current_session)


class Occlusion(BasePerturbationMethod):
    """ Occlusion is a perturbation based inferrence algorithm. Such forms of algorithm direcly computes the
    relevance/attribution of the input features .. math:: (X_{i}) by systematically occluding different
    portions of the image (by removing, masking or altering them), then running a forward pass on the new input to
    produce a new output, and then measuring and monitoring the difference between the original output and new output.
    Perturbation based interpretation helps one to compute direct estimation of the marginal effect of a feature but
    the inferrence might be computationally expensive depending on the cardinatlity of the feature space.
    The choice of the baseline value while perturbing through the feature space could be set to 0,
    as explained in detail by Zeiler & Fergus, 2014[2].

    References
    ----------
    .. [1] Ancona M, Ceolini E, Oztireli C, Gross M (ICLR, 2018).
    .. Towards better understanding of gradient-based attribution methods for Deep Neural Networks.
    .. [2] Zeiler, M and Fergus, R (Springer, 2014). Visualizing and understanding convolutional networks.
    .. In European conference on computer vision, pp. 818–833.
    .. [3] https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
    """
    __name__ = "Occlusion"
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, current_session, window_shape=None, step=2):
        super(Occlusion, self).__init__(output_tensor, input_tensor, samples, current_session)

        self.input_shape = samples.shape[1:]
        if window_shape is not None:
            assert len(self.window_shape) == len(self.input_shape), \
                'window_shape and input must match{}'.format(len(self.input_shape))
            self.window_shape = tuple(self.window_shape)
        else:
            self.window_shape = (2,) * len(self.input_shape)

        self.replace_value = 0
        self.step = step
        # the input samples are expected to be of the shape,
        # (1, 150, 150, 3) <batch_size, image_width, image_height, no_of_channels>
        self.batch_size = self.samples.shape[0]
        self.total_dim = np.prod(self.input_shape)  # e.g. 268203 = 299*299*3
        Occlusion.logger.info('Input shape: {}; window_shape: {}; replace value: {};'
                              ' step: {}; batch size: {}; total dimensions: {}'.format
                              (self.input_shape, self.window_shape, self.replace_value, self.step,
                               self.batch_size, self.total_dim))


    def _create_masked_input(self, index_list):
        # create a mask
        mask = np.ones(self.input_shape, dtype=np.float32).flatten()
        mask[index_list] = self.replace_value
        # reshape the new input to the format <batch_size, image_width, image_height, no_of_channels>
        new_input = mask.reshape((1,) + self.input_shape) * self.samples
        return new_input


    def _run(self):
        # Create a rolling window view of the input matrix
        # sample input is of the following shape (1, 150, 150, 3) <batch_size, image_width, image_height, no_of_channels>
        # self.samples[0] returns the actual input shape (150, 150, 3)
        index_matrix = np.arange(self.total_dim, dtype=np.int32).reshape(self.input_shape)
        input_patches = view_windows(index_matrix, self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(self.samples, dtype=np.float32).reshape((-1), self.total_dim)
        normalizer = np.zeros_like(heatmap)

        # Compute original output
        eval_default = self._session_run(self.output_tensor, self.samples)

        # Perturb through the feature space by replacing and masking
        for item, index in enumerate(input_patches):
            indexes = index.flatten()
            masked_input = self._create_maked_input(indexes)
            # compute delta with the new generated masked input
            delta = eval_default - self._session_run(self.output_tensor, masked_input)
            # aggregate columnwise
            # aggregation here takes care of the window overlapping that occurs while rolling a window
            delta_aggregated = np.sum(delta.reshape((self.batch_size, -1)), -1, keepdims=True)
            heatmap[:, indexes] += delta_aggregated
            normalizer[:, indexes] += index.size
        attribution = np.reshape(heatmap / normalizer, self.samples.shape)
        return attribution
