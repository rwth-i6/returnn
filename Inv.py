import theano
import theano.tensor as T
import os

Tfloat = theano.config.floatX  # @UndefinedVariable

class InvOp(theano.Op):
  __props__ = ('min_skip', 'max_skip', 'nstates', 'focus', 'mode')

  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def __init__(self, min_skip, max_skip, nstates, focus='last', mode='viterbi'):
    self.min_skip = min_skip
    self.max_skip = max_skip
    self.focus = ['last','max'].index(focus)
    self.nstates = nstates
    self.mode = ['viterbi','full'].index(mode)

  def make_node(self, x, y, len_x, len_y):
    x = theano.tensor.as_tensor_variable(x)
    assert x.ndim == 3  # tensor: nframes x nseqs x dim
    y = theano.tensor.as_tensor_variable(y)
    assert y.ndim == 2  # matrix: nseqs x max_labelling_length
    len_x = theano.tensor.as_tensor_variable(len_x)
    len_y = theano.tensor.as_tensor_variable(len_y)
    assert len_x.ndim == 1  # vector of seqs lengths
    assert len_x.dtype == "int32"
    assert len_y.ndim == 1  # vector of seqs lengths
    assert len_y.dtype == "int32"

    return theano.Apply(self, [x, y, len_x, len_y], [T.imatrix(),T.imatrix()])

  def c_support_code(self):
    src = ""
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/C_Support_Code.cpp', 'r') as f:
      src += f.read()
    with open(path + '/Inv.cpp', 'r') as f:
      src += f.read()
    return src

  def c_compile_args(self):
    return ['-fopenmp']

  def c_code(self, node, name, inp, out, sub):
    x, y, len_x, len_y = inp
    attention, emission = out # (N*S,B)
    nstates = self.nstates
    min_skip = self.min_skip
    max_skip = self.max_skip
    mode = self.mode
    focus = self.focus
    fail = sub['fail']
    return """
            Py_XDECREF(%(attention)s);
            npy_intp ydims[] = {PyArray_DIM(%(y)s,0) * %(nstates)s, PyArray_DIM(%(y)s,1)};
            npy_intp xdims[] = {PyArray_DIM(%(x)s,0), PyArray_DIM(%(x)s,1)};
            %(attention)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(y)s), ydims, PyArray_DescrFromType(NPY_INT32), 0);
            %(emission)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(y)s), xdims, PyArray_DescrFromType(NPY_INT32), 0);
            if (!%(attention)s)
                %(fail)s;
            {
              ArrayI attentionWr(%(attention)s);
              ArrayI emissionWr(%(emission)s);
              ArrayF xWr(%(x)s);
              ArrayI yWr(%(y)s);
              CArrayI len_xWr(%(len_x)s);
              CArrayI len_yWr(%(len_y)s);

              int numSeqs = len_xWr.dim(0);
              #pragma omp parallel for
              for(int i = 0; i < numSeqs; ++i)
              {
                  Inv cls;
                  SArrayI attentionSWr(attentionWr, 1, i);
                  SArrayI emissionSWr(emissionWr, 1, i);
                  if(%(mode)s == 0)
                    cls.viterbi(CSArrayF(xWr, 1, i), CSArrayI(yWr, 1, i), len_xWr(i), len_yWr(i), %(nstates)s, %(min_skip)s, %(max_skip)s, %(focus)s, attentionSWr);
                  for(int j=0;j<attentionSWr.dim(0);++j)
                  {
                    emissionSWr(attentionSWr(j)) = 1;
                    attentionSWr(j) += xWr.dim(0) * i;
                  }
              }
            }
        """ % locals()

  # IMPORTANT: change this, if you change the c-code
  #def c_code_cache_version(self):
  #  return (1.01,)

class InvOpFull(theano.Op):
  __props__ = ('min_skip', 'max_skip', 'nstates', 'focus', 'mode')

  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def __init__(self, min_skip, max_skip, nstates, focus='last', mode='viterbi'):
    self.min_skip = min_skip
    self.max_skip = max_skip
    self.focus = ['last', 'max'].index(focus)
    self.nstates = nstates
    self.mode = ['viterbi', 'full'].index(mode)

  def make_node(self, x, y, len_x, len_y):
    x = theano.tensor.as_tensor_variable(x)
    assert x.ndim == 3  # tensor: nframes x nseqs x dim
    y = theano.tensor.as_tensor_variable(y)
    assert y.ndim == 2  # matrix: nseqs x max_labelling_length
    len_x = theano.tensor.as_tensor_variable(len_x)
    len_y = theano.tensor.as_tensor_variable(len_y)
    assert len_x.ndim == 1  # vector of seqs lengths
    assert len_x.dtype == "int32"
    assert len_y.ndim == 1  # vector of seqs lengths
    assert len_y.dtype == "int32"

    return theano.Apply(self, [x, y, len_x, len_y], [T.ftensor3()])

  def c_support_code(self):
    src = ""
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/C_Support_Code.cpp', 'r') as f:
      src += f.read()
    with open(path + '/Inv.cpp', 'r') as f:
      src += f.read()
    return src

  def c_compile_args(self):
    return ['-fopenmp']

  def c_code(self, node, name, inp, out, sub):
    x, y, len_x, len_y = inp
    attention = out[0] # (N*S,B,T)
    nstates = self.nstates
    min_skip = self.min_skip
    max_skip = self.max_skip
    mode = self.mode
    focus = self.focus
    fail = sub['fail']
    return """
            Py_XDECREF(%(attention)s);
            npy_intp ydims[] = {PyArray_DIM(%(y)s,0) * %(nstates)s, PyArray_DIM(%(y)s,1), PyArray_DIM(%(x)s,0)};
            //npy_intp ydims[] = {PyArray_DIM(%(x)s,2), PyArray_DIM(%(y)s,1), PyArray_DIM(%(x)s,0)};
            %(attention)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(x)s), ydims, PyArray_DescrFromType(NPY_FLOAT32), 0);
            if (!%(attention)s)
                %(fail)s;
            {
              ArrayF attentionWr(%(attention)s);
              ArrayF xWr(%(x)s);
              ArrayI yWr(%(y)s);
              CArrayI len_xWr(%(len_x)s);
              CArrayI len_yWr(%(len_y)s);

              int numSeqs = len_xWr.dim(0);
              #pragma omp parallel for
              for(int i = 0; i < numSeqs; ++i)
              {
                  Inv cls;
                  SArrayF attentionSWr(attentionWr, 1, i);
                  cls.full(CSArrayF(xWr, 1, i), CSArrayI(yWr, 1, i), len_xWr(i), len_yWr(i), %(nstates)s, %(min_skip)s, %(max_skip)s, %(focus)s, attentionSWr);
              }
            }
        """ % locals()

class InvOpFull(theano.Op):
  __props__ = ('min_skip', 'max_skip', 'nstates', 'focus', 'mode')

  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def __init__(self, min_skip, max_skip, nstates, focus='last', mode='viterbi'):
    self.min_skip = min_skip
    self.max_skip = max_skip
    self.focus = ['last', 'max'].index(focus)
    self.nstates = nstates
    self.mode = ['viterbi', 'full'].index(mode)

  def make_node(self, x, y, len_x, len_y):
    x = theano.tensor.as_tensor_variable(x)
    assert x.ndim == 3  # tensor: nframes x nseqs x dim
    y = theano.tensor.as_tensor_variable(y)
    assert y.ndim == 2  # matrix: nseqs x max_labelling_length
    len_x = theano.tensor.as_tensor_variable(len_x)
    len_y = theano.tensor.as_tensor_variable(len_y)
    assert len_x.ndim == 1  # vector of seqs lengths
    assert len_x.dtype == "int32"
    assert len_y.ndim == 1  # vector of seqs lengths
    assert len_y.dtype == "int32"

    return theano.Apply(self, [x, y, len_x, len_y], [T.ftensor3()])

  def c_support_code(self):
    src = ""
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/C_Support_Code.cpp', 'r') as f:
      src += f.read()
    with open(path + '/Inv.cpp', 'r') as f:
      src += f.read()
    return src

  def c_compile_args(self):
    return ['-fopenmp']

  def c_code(self, node, name, inp, out, sub):
    x, y, len_x, len_y = inp
    attention = out[0] # (N*S,B,T)
    nstates = self.nstates
    min_skip = self.min_skip
    max_skip = self.max_skip
    mode = self.mode
    focus = self.focus
    fail = sub['fail']
    return """
            Py_XDECREF(%(attention)s);
            npy_intp ydims[] = {PyArray_DIM(%(y)s,0) * %(nstates)s, PyArray_DIM(%(y)s,1), PyArray_DIM(%(x)s,0)};
            //npy_intp ydims[] = {PyArray_DIM(%(x)s,2), PyArray_DIM(%(y)s,1), PyArray_DIM(%(x)s,0)};
            %(attention)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(x)s), ydims, PyArray_DescrFromType(NPY_FLOAT32), 0);
            if (!%(attention)s)
                %(fail)s;
            {
              ArrayF attentionWr(%(attention)s);
              ArrayF xWr(%(x)s);
              ArrayI yWr(%(y)s);
              CArrayI len_xWr(%(len_x)s);
              CArrayI len_yWr(%(len_y)s);

              int numSeqs = len_xWr.dim(0);
              #pragma omp parallel for
              for(int i = 0; i < numSeqs; ++i)
              {
                  Inv cls;
                  SArrayF attentionSWr(attentionWr, 1, i);
                  cls.full(CSArrayF(xWr, 1, i), CSArrayI(yWr, 1, i), len_xWr(i), len_yWr(i), %(nstates)s, %(min_skip)s, %(max_skip)s, %(focus)s, attentionSWr);
              }
            }
        """ % locals()


class AlignOp(theano.Op):
  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def make_node(self, x, y, len_x, len_y):
    x = theano.tensor.as_tensor_variable(x)
    assert x.ndim == 3  # tensor: nframes x nseqs x dim
    y = theano.tensor.as_tensor_variable(y)
    assert y.ndim == 2  # matrix: nseqs x max_labelling_length
    len_x = theano.tensor.as_tensor_variable(len_x)
    len_y = theano.tensor.as_tensor_variable(len_y)
    assert len_x.ndim == 1  # vector of seqs lengths
    assert len_x.dtype == "int32"
    assert len_y.ndim == 1  # vector of seqs lengths
    assert len_y.dtype == "int32"

    return theano.Apply(self, [x, y, len_x, len_y], [T.ftensor3()])

  def c_support_code(self):
    src = ""
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/C_Support_Code.cpp', 'r') as f:
      src += f.read()
    with open(path + '/Inv.cpp', 'r') as f:
      src += f.read()
    return src

  def c_compile_args(self):
    return ['-fopenmp']


class InvAlignOp(AlignOp):
  __props__ = ('min_skip', 'max_skip', 'nstates', 'focus', 'mode')

  def c_code(self, node, name, inp, out, sub):
    x, y, len_x, len_y = inp
    attention = out[0] # (N*S,B,T)
    nstates = self.nstates
    min_skip = self.min_skip
    max_skip = self.max_skip
    mode = self.mode
    focus = self.focus
    fail = sub['fail']
    return """
        Py_XDECREF(%(attention)s);
        int T = 1;
        if(%(mode)s == 1)
          T = PyArray_DIM(%(x)s, 0);
        npy_intp ydims[] = {PyArray_DIM(%(y)s,0) * %(nstates)s, PyArray_DIM(%(y)s,1), T};
        %(attention)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(x)s), ydims, PyArray_DescrFromType(NPY_FLOAT32), 0);
        if (!%(attention)s)
            %(fail)s;
        {
          ArrayF attentionWr(%(attention)s);
          ArrayF xWr(%(x)s);
          ArrayI yWr(%(y)s);
          CArrayI len_xWr(%(len_x)s);
          CArrayI len_yWr(%(len_y)s);

          int numSeqs = len_xWr.dim(0);
          #pragma omp parallel for
          for(int i = 0; i < numSeqs; ++i)
          {
              InvAlign cls;
              SArrayF attentionSWr(attentionWr, 1, i);
              if(%(mode)s == 0)
                cls.viterbi(CSArrayF(xWr, 1, i), CSArrayI(yWr, 1, i), len_xWr(i), len_yWr(i), %(nstates)s, %(min_skip)s, %(max_skip)s, %(focus)s, attentionSWr);
              else
                cls.full(CSArrayF(xWr, 1, i), CSArrayI(yWr, 1, i), len_xWr(i), len_yWr(i), %(nstates)s, %(min_skip)s, %(max_skip)s, %(focus)s, attentionSWr);
          }
        }
    """ % locals()



class StdOpFull(theano.Op):
  __props__ = ('skip_tdp', 'nstates')

  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def __init__(self, skip_tdp, nstates):
    self.nstates = nstates
    self.skip_tdp = skip_tdp

  def make_node(self, x, y, len_x, len_y):
    x = theano.tensor.as_tensor_variable(x)
    assert x.ndim == 3  # tensor: nframes x nseqs x dim
    y = theano.tensor.as_tensor_variable(y)
    assert y.ndim == 2  # matrix: nseqs x max_labelling_length
    len_x = theano.tensor.as_tensor_variable(len_x)
    len_y = theano.tensor.as_tensor_variable(len_y)
    assert len_x.ndim == 1  # vector of seqs lengths
    assert len_x.dtype == "int32"
    assert len_y.ndim == 1  # vector of seqs lengths
    assert len_y.dtype == "int32"
    return theano.Apply(self, [x, y, len_x, len_y], [T.ftensor3()])

  def c_support_code(self):
    src = ""
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/C_Support_Code.cpp', 'r') as f:
      src += f.read()
    with open(path + '/Inv.cpp', 'r') as f:
      src += f.read()
    return src

  def c_compile_args(self):
    return ['-fopenmp']

  def c_code(self, node, name, inp, out, sub):
    x, y, len_x, len_y = inp
    alignment = out[0]
    nstates = self.nstates
    skip_tdp = self.skip_tdp
    fail = sub['fail']
    return """
            Py_XDECREF(%(attention)s);
            npy_intp xdims[] = {PyArray_DIM(%(x)s,0), PyArray_DIM(%(x)s,1), PyArray_DIM(%(y)s,0) * %(nstates)s};
            %(alignment)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(x)s), ydims, PyArray_DescrFromType(NPY_FLOAT32), 0);
            if (!%(alignment)s)
                %(fail)s;
            {
              ArrayF alignmentWr(%(alignment)s);
              ArrayF xWr(%(x)s);
              ArrayI yWr(%(y)s);
              CArrayI len_xWr(%(len_x)s);
              CArrayI len_yWr(%(len_y)s);

              int numSeqs = len_xWr.dim(0);
              #pragma omp parallel for
              for(int i = 0; i < numSeqs; ++i)
              {
                  Std cls;
                  SArrayF alignmentSWr(alignmentWr, 1, i);
                  cls.full(CSArrayF(xWr, 1, i), CSArrayI(yWr, 1, i), len_xWr(i), len_yWr(i), %(nstates)s, %(skip_tdp)s, alignmentSWr);
              }
            }
        """ % locals()