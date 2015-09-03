
import theano.tensor as T


def time_batch_make_flat(val):
  """
  :rtype val: theano.Variable
  :rtype: theano.Variable

  Will flatten the first two dimensions and leave the others as is.
  """
  assert val.ndim > 1
  s0 = val.shape[0] * val.shape[1]
  newshape = [s0] + [val.shape[i] for i in range(2, val.ndim)]
  return T.reshape(val,
                   newshape,
                   ndim=val.ndim - 1,
                   name="flat_%s" % val.name)
