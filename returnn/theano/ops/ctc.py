import theano
import theano.tensor as T
import os
Tfloat = theano.config.floatX  # @UndefinedVariable

class CTCOp(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, y, seq_lengths):
        x = theano.tensor.as_tensor_variable(x)
        assert x.ndim == 3  # tensor: nframes x nseqs x dim
        y = theano.tensor.as_tensor_variable(y)
        assert y.ndim == 2  # matrix: nseqs x max_labelling_length
        seq_lengths = theano.tensor.as_tensor_variable(seq_lengths)
        assert seq_lengths.ndim == 1  # vector of seqs lengths
        assert seq_lengths.dtype == "int32"

        return theano.Apply(self, [x, y, seq_lengths], [T.fvector(), T.ftensor3(), T.fmatrix()])
        # first output: CTC error per sequence
        # second output: Derivative w.r.t. Softmax net input

    def c_support_code(self):
        src = ""
        path = os.path.dirname(os.path.abspath(__file__))
        with open(path + '/c_support_code.cpp', 'r') as f:
            src += f.read()
        with open(path + '/ctc.cpp', 'r') as f:
            src += f.read()
        return src

    def c_compile_args(self):
        return ['-fopenmp']

    def c_code(self, node, name, inp, out, sub):
        x, y, seq_lengths = inp
        errs, err_sigs, priors = out
        fail = sub['fail']
        return """
            Py_XDECREF(%(errs)s);
            Py_XDECREF(%(err_sigs)s);
            Py_XDECREF(%(priors)s);
            npy_intp dims[] = {PyArray_DIM(%(x)s,1)};
            %(errs)s = (PyArrayObject*) PyArray_Zeros(1, dims, PyArray_DescrFromType(NPY_FLOAT32), 0);
            if (!%(errs)s)
                %(fail)s;
            %(err_sigs)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(x)s), PyArray_DIMS(%(x)s), PyArray_DescrFromType(NPY_FLOAT32), 0);
            if (!%(err_sigs)s)
                %(fail)s;
            %(priors)s = (PyArrayObject*) PyArray_Zeros(PyArray_NDIM(%(x)s) - 1, PyArray_DIMS(%(x)s) + 1, PyArray_DescrFromType(NPY_FLOAT32), 0);
            if (!%(priors)s)
                %(fail)s;
            {
                ArrayF errsWr(%(errs)s);
                ArrayF errSigsWr(%(err_sigs)s);
                ArrayF priorsWr(%(priors)s);
                ArrayF xWr(%(x)s);
                ArrayI yWr(%(y)s);
                CArrayI seqLensWr(%(seq_lengths)s);

                /*errsWr.debugPrint("errsWr");
                errSigsWr.debugPrint("errSigsWr");
                xWr.debugPrint("xWr");
                yWr.debugPrint("yWr");
                seqLensWr.debugPrint("seqLensWr");*/

                int numSeqs = seqLensWr.dim(0);
                #pragma omp parallel for
                for(int i = 0; i < numSeqs; ++i)
                {
                    CTC ctc;
                    SArrayF errSigsSWr(errSigsWr, 1, i);
                    SArrayF priorsSWr(priorsWr, 0, i);
                    ctc.forwardBackward(CSArrayF(xWr, 1, i), CSArrayI(yWr, 0, i), seqLensWr(i), errsWr(i), errSigsSWr, priorsSWr);
                }
            }
        """ % locals()

    #IMPORTANT: change this, if you change the c-code
    def c_code_cache_version(self):
        return (2.10,)
