import theano
import theano.tensor as T
import numpy


class OneDToTwoDOp(theano.Op):
  __props__ = ()

  def make_node(self, X, sizes):
    X = T.as_tensor_variable(X)
    assert X.dtype == "float32"
    assert X.ndim == 3  #tensor: time x batch x feature
    assert sizes.dtype == "float32"
    assert sizes.ndim == 2  #batch x 2
    return theano.Apply(self, [X, sizes], [T.ftensor4()])

  def perform(self, node, inputs, output_storage):
    X = inputs[0]
    sizes = inputs[1]
    out = output_storage[0]

    max_height = int(sizes[:, 0].max())
    max_width = int(sizes[:, 1].max())
    batches = X.shape[1]
    feat = X.shape[2]
    Y = numpy.zeros((max_height, max_width, batches, feat), dtype="float32")
    for b in range(batches):
      height, width = sizes[b]
      height = int(height)
      width = int(width)
      size = height * width
      im = X[:size, b]
      Y[:height, :width, b, :] = im.reshape((height, width, feat))

    out[0] = Y

  #TODO infer_shape
