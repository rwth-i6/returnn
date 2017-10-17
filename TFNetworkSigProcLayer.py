

from __future__ import print_function

import tensorflow as tf
from TFNetworkLayer import LayerBase, _ConcatInputLayer, SearchChoices, get_concat_sources_data_template
from TFUtil import Data, reuse_name_scope
from Log import log

class AlternatingRealToComplexLayer(_ConcatInputLayer):
  """
  This layer converts a real valued input tensor into a complex valued output tensor.
  For this even and odd features are considered the real and imaginary part of one complex number, respectively
  """

  layer_class = "alternatingRealToComplex"

  def __init__(self, **kwargs):
    """
    """
    super(AlternatingRealToComplexLayer, self).__init__(**kwargs)
    real_value = tf.strided_slice(self.input_data.placeholder, [0, 0, 0], tf.shape(self.input_data.placeholder), [1, 1, 2])
    imag_value = tf.strided_slice(self.input_data.placeholder, [0, 0, 1], tf.shape(self.input_data.placeholder), [1, 1, 2])
    self.output.placeholder = tf.complex(real_value, imag_value)

class AbsLayer(_ConcatInputLayer):
  """
  This layer converts a input tensor into a output containing the aboslute value.
  """

  layer_class = "abs"

  def __init__(self, **kwargs):
    """
    """
    super(AbsLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.abs(self.input_data.placeholder)

class SplitConcatMultiChannel(_ConcatInputLayer):
  """
  This layer splits the feature dimension into equisized number of channel features and 
  stacks tham in the batch dimension. Thus the batch dimension is multiplied with the number of channels.
  The channels of one singal will have consecutive batch indices
  """

  layer_class = "splitConcatenatedMultiChannel"

  def __init__(self, nr_of_channels=1, **kwargs):
    """
    """
    super(SplitConcatMultiChannel, self).__init__(**kwargs)
    self.output.placeholder = tf.reshape(self.input_data.placeholder, [tf.shape(self.input_data.placeholder)[0] * nr_of_channels, tf.shape(self.input_data.placeholder)[1], tf.shape(self.input_data.placeholder)[2] / nr_of_channels])


