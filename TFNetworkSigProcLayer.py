
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
  This layer converts a input tensor into a output containing the aboslute value as a float32.
  """

  layer_class = "abs"

  def __init__(self, **kwargs):
    """
    """
    super(AbsLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.cast(tf.abs(self.input_data.placeholder), dtype=tf.float32)

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
    output = tf.reshape(self.input_data.placeholder, [tf.shape(self.input_data.placeholder)[0], tf.shape(self.input_data.placeholder)[1], nr_of_channels, tf.shape(self.input_data.placeholder)[2] / nr_of_channels])
    self.output.placeholder = tf.transpose(tf.reshape(tf.transpose(output, [1, 3, 0, 2]), (tf.shape(output)[1], tf.shape(output)[3], tf.shape(output)[0] *  tf.shape(output)[2])), [2, 0, 1])
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    # work around to obtain result like numpy.repeat(size_placeholder, nr_of_channels)
    self.output.size_placeholder[0] = tf.reshape(tf.tile(tf.reshape(self.output.size_placeholder[0], [-1, 1]), [1, nr_of_channels]), [-1]) 

class BatchMedianPoolingLayer(_ConcatInputLayer):
  """
  This layer converts a input tensor into a output containing the aboslute value as a float32.
  """

  layer_class = "batchMedianPooling"

  def __init__(self, poolSize=1, **kwargs):
    """
    """
    super(BatchMedianPoolingLayer, self).__init__(**kwargs)
    
    # set order of axes
    self.output.batch_dim_axis = 0
    self.output.time_dim_axis = 1
    self.output.feature_dim_axis = 2

    # loop over batch pools and extract median
    poolStartIdx = tf.constant(0)
    output = tf.Variable(tf.zeros([1, 1, 1], dtype=tf.float32), dtype=tf.float32)
    def iteratePools(poolStartIdx, output):
        return tf.less(poolStartIdx, tf.shape(self.input_data.placeholder)[self.input_data.batch_dim_axis])
    def poolMedian(poolStartIdx, output):
      pool = tf.transpose(self.input_data.placeholder, [self.input_data.batch_dim_axis, self.input_data.time_dim_axis, self.input_data.feature_dim_axis])[poolStartIdx:(poolStartIdx + poolSize), :, :]
      median = tf.transpose(tf.reshape(tf.nn.top_k(tf.transpose(pool, [self.output.time_dim_axis, self.output.feature_dim_axis, self.output.batch_dim_axis]), tf.cast(tf.ceil(tf.cast(tf.shape(pool)[self.output.batch_dim_axis], dtype=tf.float32)/2), dtype=tf.int32)).values[:, :, -1], (tf.shape(pool)[1], tf.shape(pool)[2], 1)), [2, 0, 1])
      output = tf.cond(tf.greater(poolStartIdx, 0), lambda: tf.concat([output, median], axis=self.output.batch_dim_axis), lambda: median)
      return tf.add(poolStartIdx, poolSize), output 
    r = tf.while_loop(iteratePools, poolMedian, [poolStartIdx, output], shape_invariants=[poolStartIdx.get_shape(), tf.TensorShape([None, None, None])])
    self.output.placeholder = r[-1]
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    self.output.size_placeholder[0] = tf.strided_slice(self.output.size_placeholder[0], [0], tf.shape(self.output.size_placeholder[0]), [poolSize])

class TileFeaturesLayer(_ConcatInputLayer):
  """
  This layer converts a input tensor into a output containing the aboslute value as a float32.
  """

  layer_class = "tileFeatures"

  def __init__(self, repetitions=1, **kwargs):
    """
    """
    super(TileFeaturesLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.tile(tf.transpose(self.input_data.placeholder, [self.input_data.batch_dim_axis, self.input_data.time_dim_axis, self.input_data.feature_dim_axis]), [1, 1, repetitions])

class ConvertRealToComplexLayer(_ConcatInputLayer):
  """
  This layer converts a input tensor into a output containing the aboslute value as a float32.
  """

  layer_class = "floatToComplex"

  def __init__(self, repetitions=1, **kwargs):
    """
    """
    super(ConvertRealToComplexLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.cast(self.input_data.placeholder, dtype=tf.complex64)

class MaskBasedGevBeamformingLayer(LayerBase):
  """
  This layer converts a input tensor into a output containing the aboslute value as a float32.
  """

  layer_class = "maskBasedGevBeamforming"

  def __init__(self, nrOfChannels=1, postFilterId=0, **kwargs):
    """
    """
    super(MaskBasedGevBeamformingLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2
#    tf.Assert(tf.equal(tf.shape(self.sources[0].output.placeholder)[self.sources[0].output.feature_dim_axis] * 2, tf.shape(self.sources[1].output.placeholder)[self.sources[1].output.feature_dim_axis]), [tf.shape(self.sources[0].output.placeholder), tf.shape(self.sources[1].output.placeholder)])

    from tfSi6Proc.audioProcessing.enhancement.beamforming import TfMaskBasedGevBeamformer

    complexSpectrogram = tf.transpose(self.sources[0].output.placeholder, [self.sources[0].output.batch_dim_axis, self.sources[0].output.time_dim_axis, self.sources[0].output.feature_dim_axis])
    complexSpectrogram = tf.transpose(tf.reshape(complexSpectrogram, (tf.shape(complexSpectrogram)[0], tf.shape(complexSpectrogram)[1], nrOfChannels, tf.shape(complexSpectrogram)[2]/nrOfChannels)), [0, 1, 3, 2])
    masks = tf.transpose(self.sources[1].output.placeholder , [self.sources[1].output.batch_dim_axis, self.sources[1].output.time_dim_axis, self.sources[1].output.feature_dim_axis])
    masks = tf.transpose(tf.reshape(masks, (tf.shape(masks)[0], tf.shape(masks)[1], nrOfChannels, tf.shape(masks)[2]/nrOfChannels)), [0, 1, 3, 2])
    noiseMasks = masks[:, :, :tf.shape(masks)[2]/2, :] 
    speechMasks = masks[:, :, tf.shape(masks)[2]/2:, :] 

    gevBf = TfMaskBasedGevBeamformer(tfFreqDomInput=complexSpectrogram[0, :, :, :], tfNoiseMask = noiseMasks[0, :, : ,:], tfSpeechMask = speechMasks[0, :, : ,:], postFilterId = postFilterId)
    beamformingOutput = gevBf.getFrequencyDomainOutputSignal()

    self.output.placeholder = tf.reshape(beamformingOutput, (1, tf.shape(beamformingOutput)[0], tf.shape(beamformingOutput)[1]))

