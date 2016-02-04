import theano
import theano.tensor as T
import os
Tfloat = theano.config.floatX  # @UndefinedVariable

class BestPathDecodeOp(theano.Op):
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
        
        return theano.Apply(self, [x, y, seq_lengths], [T.ivector()])
        #output: number of edits for each sequence

    def c_code(self, node, name, inp, out, sub):
        x, y, seq_lengths = inp
        lev, = out
        fail = sub['fail']        
        return """                
            Py_XDECREF(%(lev)s);    
            npy_intp dims[] = {PyArray_DIM(%(x)s,1)};
            %(lev)s = (PyArrayObject*) PyArray_Zeros(1, dims, PyArray_DescrFromType(NPY_INT32), 0);
            if(!%(lev)s)
                %(fail)s;
            {
                CArrayF xWr(%(x)s);
                CArrayI yWr(%(y)s);
                CArrayI seqLensWr(%(seq_lengths)s);
                ArrayI levWr(%(lev)s);
                int numSeqs = seqLensWr.dim(0);
				#pragma omp parallel for
                for(int i = 0; i < numSeqs; ++i)
                {                    
                    BestPathDecoder decoder;
                    decoder.labellingErrors(xWr, seqLensWr, i, yWr, levWr);
                } 
            }            
        """ % locals()

    def c_compile_args(self):
        return ['-fopenmp']
    
    #IMPORTANT: change this, if you change the c-code
    def c_code_cache_version(self):
        return (1.62,)
        
    def c_support_code(self):
        src = ""
        path = os.path.dirname(os.path.abspath(__file__))
        with open(path + '/C_Support_Code.cpp', 'r') as f:
            src += f.read()
        with open(path + '/BestPathDecoder.cpp', 'r') as f:
            src += f.read()
        return src
