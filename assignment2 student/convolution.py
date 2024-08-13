from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math


def conv2d(inputs, filters, strides, padding):
    """
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
    num_examples, in_height, in_width, input_in_channels = inputs.shape

    filter_height, filter_width, filter_in_channels, filter_out_channels = filters.shape

    num_examples_stride, strideY, strideX, channels_stride = strides

    assert (input_in_channels == filter_in_channels)

    if padding == "SAME":
        out_height = int(np.ceil(in_height / strideY))
        out_width = int(np.ceil(in_width / strideX))
        pad_along_height = max((out_height - 1) * strideY + filter_height - in_height, 0)
        pad_along_width = max((out_width - 1) * strideX + filter_width - in_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        inputs = np.pad(inputs, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    elif padding == "VALID":
        out_height = (in_height - filter_height) // strideY + 1
        out_width = (in_width - filter_width) // strideX + 1
    else:
        raise ValueError("Padding must be either SAME or VALID")

    output = np.zeros((num_examples, out_height, out_width, filter_out_channels), dtype=np.float32)

    for n in range(num_examples):
        for h in range(0, in_height - filter_height + 1, strideY):
            for w in range(0, in_width - filter_width + 1, strideX):
                for c in range(filter_out_channels):
                    h_out = h // strideY
                    w_out = w // strideX
                    output[n, h_out, w_out, c] = np.sum(
                        inputs[n, h:h+filter_height, w:w+filter_width, :] * filters[:, :, :, c]
                    )
    return output
