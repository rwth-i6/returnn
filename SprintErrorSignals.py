import theano
import theano.tensor as T
import numpy
from Log import log
from SprintCommunicator import SprintCommunicator
Tfloat = theano.config.floatX  # @UndefinedVariable

#(for now) there should be only 1 instance at a time of this class
class SprintErrorSigOp(theano.Op):
  comm = None

  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def make_node(self, posteriors, seq_lengths):
    log_posteriors = T.log(theano.tensor.as_tensor_variable(posteriors))
    seq_lengths = theano.tensor.as_tensor_variable(seq_lengths)
    assert seq_lengths.ndim == 1  # vector of seqs lengths
    return theano.Apply(self, [log_posteriors, seq_lengths], [T.fvector(), posteriors.type()])
    
  def perform(self, node, inputs, outputs):
    assert SprintErrorSigOp.comm is not None
    log_posteriors, seq_lengths = inputs
    
    if numpy.isnan(log_posteriors).any():
      print >> log.v1, 'log_posteriors contain NaN!'
    if numpy.isinf(log_posteriors).any():
      print >> log.v1, 'log_posteriors contain Inf!'
      numpy.set_printoptions(threshold=numpy.nan)
      print >> log.v1, 'log_posteriors:', log_posteriors
    
    n_rows = log_posteriors.shape[0]
    n_cols = log_posteriors.shape[2]
    assert len(SprintCommunicator.instance.segments) > 0

    loss, errsig = SprintCommunicator.instance.get_error_signal(SprintCommunicator.instance.segments, log_posteriors, seq_lengths)
    #print >> log.v4, 'loss:', loss, 'errsig:', errsig
    outputs[0][0] = loss
    outputs[1][0] = errsig
        
    print >> log.v5, 'avg frame loss for segments:', loss.sum() / seq_lengths.sum()
