import theano
import theano.tensor as T
import numpy
from Log import log
import os
import sys
Tfloat = theano.config.floatX  # @UndefinedVariable

class ShMemOp(theano.Op):
  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def c_support_code(self):
    src = """#include <sys/ipc.h>
             #include <sys/shm.h>
             #include <stdint.h>
             #include <sched.h>
             
             typedef uint32_t u32;
             
             enum TheanoStatus
             {
	       IDLE = 0, REQUEST_ERRORSIGNAL = 1, ERRORSIGNAL_WRITTEN = 2, TERMINATED = 4
             };
               
             namespace {
               /*layout of shared memory:
               4 bytes TheanoStatus flag
               4 bytes float nRows
               4 bytes float nCols
               CTL_SEG_SIZE-12 bytes segmentName (0 terminated)
               rest: data*/
               const u32 SHARED_MEM_SIZE = 10 * 1024 * 1024; //10M should be enough
               const u32 CTL_SEG_SIZE = 512;
               const u32 STATUS_BEGIN = 0;
               const u32 ROWS_BEGIN = 4;
               const u32 COLS_BEGIN = 8;
               const u32 LOSS_BEGIN = 12;
               const u32 SEGMENT_NAME_BEGIN = 16;
               const u32 DATA_BEGIN = CTL_SEG_SIZE;
               const u32 MAX_SEG_NAME_LEN = DATA_BEGIN - SEGMENT_NAME_BEGIN;
             }
             
             template <typename T>
             //careful: the offset is in bytes and not related to T
             T& at(void * shMem, u32 offset)
             {
               return *reinterpret_cast<T*>(static_cast<char*>(shMem) + offset);
             }
             
             u32& shMemStatus(void * shMem)
             {
               return at<u32>(shMem, STATUS_BEGIN);
             }
             
             float& shMemData(void * shMem, u32 idx)
             {
               return at<float>(shMem, DATA_BEGIN + 4 * idx);
             }
             
             char& shMemSegNameChar(void * shMem, u32 idx)
             {
               return at<char>(shMem, SEGMENT_NAME_BEGIN + idx);
             }
             
             u32& shMemRows(void * shMem)
             {
               return at<u32>(shMem, ROWS_BEGIN);
             }
             
             u32& shMemCols(void * shMem)
             {
               return at<u32>(shMem, COLS_BEGIN);
             }
             
             float& shMemLoss(void * shMem)
             {
               return at<float>(shMem, LOSS_BEGIN);
             }
             
             void waitForStatus(void * shMem, u32 status)
             {
               while(shMemStatus(shMem) != status) {
                 sched_yield();
               }
             }
          """
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/C_Support_Code.cpp', 'r') as f:
        src += f.read()
    return src
    
class AllocShMemOp(ShMemOp):
  def make_node(self, key):
    key = theano.tensor.as_tensor_variable(key)
    return theano.Apply(self, [key], [T.scalar('shId', 'int32'), T.scalar('shMem', 'int64')])
  
  def c_code(self, node, name, inp, out, sub):
    fail = sub['fail']
    key, = inp
    shId, shMem = out
    return """
      Py_XDECREF(%(shId)s);
      Py_XDECREF(%(shMem)s);
      %(shId)s = (PyArrayObject*) PyArray_Zeros(0, 0, PyArray_DescrFromType(NPY_INT32), 0);
      %(shMem)s = (PyArrayObject*) PyArray_Zeros(0, 0, PyArray_DescrFromType(NPY_INT64), 0);
      datai(%(shId)s) = shmget(datai(%(key)s), SHARED_MEM_SIZE, 0660 | IPC_CREAT);
      verify(datai(%(shId)s) >= 0);
      void * shMem_void = shmat(datai(%(shId)s), 0, 0);
      verify(shMem_void != (void*)-1);
      datai64(%(shMem)s) = reinterpret_cast<int64_t>(shMem_void);
    """ % locals()

  #IMPORTANT: change this, if you change the c-code
  def c_code_cache_version(self):
    return (1.02,)
    
class DetachShMemOp(ShMemOp):
  def make_node(self, sh_mem):
    sh_mem = theano.tensor.as_tensor_variable(sh_mem)
    return theano.Apply(self, [sh_mem], [T.fscalar()]) #dummy return value
  
  def c_code(self, node, name, inp, out, sub):
    fail = sub['fail']
    shMem, = inp
    return """
      shmdt(datavoid(%(shMem)s));
    """ % locals()

  #IMPORTANT: change this, if you change the c-code
  def c_code_cache_version(self):
    return (1.02,)
    
class CopyToShMemOp(ShMemOp):
  def make_node(self, sh_mem, data):
    sh_mem = theano.tensor.as_tensor_variable(sh_mem)
    data = theano.tensor.as_tensor_variable(data)
    return theano.Apply(self, [sh_mem, data], [T.fscalar()]) #dummy return value
  
  def c_code(self, node, name, inp, out, sub):
    fail = sub['fail']
    shMem, data = inp    
    return """   
      CArrayF dataWr(%(data)s);
      int nRows = dataWr.dim(0);
      int nCols = dataWr.dim(1);
      shMemRows(datavoid(%(shMem)s)) = nRows;
      shMemCols(datavoid(%(shMem)s)) = nCols;
      for(int i = 0; i < nRows; ++i)
      {
        //cout << "i: " << i << endl;
        for(int j = 0; j < nCols; ++j)
        {
	  //cerr << "i,j == " << i << " " << j << endl;
          shMemData(datavoid(%(shMem)s), i * nCols + j) = dataWr(i,j);
        }
      }
    """ % locals()

  #IMPORTANT: change this, if you change the c-code
  def c_code_cache_version(self):
    return (1.08,)
    
class WriteStatusOp(ShMemOp):
  def make_node(self, sh_mem, status):
    sh_mem = theano.tensor.as_tensor_variable(sh_mem)
    status = theano.tensor.as_tensor_variable(status)
    return theano.Apply(self, [sh_mem, status], [T.fscalar()]) #dummy return value
  
  def c_code(self, node, name, inp, out, sub):
    fail = sub['fail']
    shMem, status = inp
    return """
      shMemStatus(datavoid(%(shMem)s)) = datau(%(status)s);
    """ % locals()

  #IMPORTANT: change this, if you change the c-code
  def c_code_cache_version(self):
    return (1.01,)
    
class WaitForStatusOp(ShMemOp):
  def make_node(self, sh_mem, status):
    sh_mem = theano.tensor.as_tensor_variable(sh_mem)
    status = theano.tensor.as_tensor_variable(status)
    return theano.Apply(self, [sh_mem, status], [T.fscalar()]) #dummy return value
  
  def c_code(self, node, name, inp, out, sub):
    fail = sub['fail']
    shMem, status = inp
    return """
      waitForStatus(datavoid(%(shMem)s), datau(%(status)s));
    """ % locals()

  #IMPORTANT: change this, if you change the c-code
  def c_code_cache_version(self):
    return (1.01,)
    
class CopyFromShMemOp(ShMemOp):
  def make_node(self, sh_mem):
    sh_mem = theano.tensor.as_tensor_variable(sh_mem)
    return theano.Apply(self, [sh_mem], [T.fvector(), T.fmatrix()])  
  
  def c_code(self, node, name, inp, out, sub):
    fail = sub['fail']
    shMem, = inp
    loss, errsig, = out    
    return """
      Py_XDECREF(%(loss)s);
      Py_XDECREF(%(errsig)s);
      
      npy_intp lossDims[] = {1};
      %(loss)s = (PyArrayObject*) PyArray_Zeros(1, lossDims, PyArray_DescrFromType(NPY_FLOAT32), 0);
      verify(%(loss)s);
      ArrayF lossWr(%(loss)s);
      lossWr(0) = shMemLoss(datavoid(%(shMem)s));
      
      u32 nRows = shMemRows(datavoid(%(shMem)s));
      u32 nCols = shMemCols(datavoid(%(shMem)s));            
      npy_intp dims[] = {nRows, nCols};
      %(errsig)s = (PyArrayObject*) PyArray_Zeros(2, dims, PyArray_DescrFromType(NPY_FLOAT32), 0);
      verify(%(errsig)s);
      ArrayF errsigWr(%(errsig)s);
      for(int i = 0; i < nRows; ++i)
      {
        for(int j = 0; j < nCols; ++j)
        {
          errsigWr(i,j) = shMemData(datavoid(%(shMem)s), i * nCols + j);
        }
      }
    """ % locals()

  #IMPORTANT: change this, if you change the c-code
  def c_code_cache_version(self):
    return (1.01,)

class WriteSegmentNameOp(ShMemOp):
  def make_node(self, sh_mem, seg_name):
    sh_mem = theano.tensor.as_tensor_variable(sh_mem)
    seg_name = theano.tensor.as_tensor_variable(seg_name)
    return theano.Apply(self, [sh_mem, seg_name], [T.fscalar()])

  def c_code(self, node, name, inp, out, sub):
    fail = sub['fail']
    shMem, seg_name = inp
    return """
      CArrayI segNameWr(%(seg_name)s);
      for(int i = 0; i < segNameWr.dim(0); ++i)
      {
         shMemSegNameChar(datavoid(%(shMem)s), i)  = char(segNameWr(i));
      }
      shMemSegNameChar(datavoid(%(shMem)s), segNameWr.dim(0)) = char(0);
    """ % locals()

  #IMPORTANT: change this, if you change the c-code
  def c_code_cache_version(self):
    return (1.02,)

class SprintCommunicator:
  IDLE, REQUEST_ERRORSIGNAL, ERRORSIGNAL_WRITTEN, TERMINATED = [0,1,2,4]
  instance = None
  
  def __init__(self, key):    
    self.segments = []  
      
    t_key = T.scalar('key', dtype='int32')
    t_sh_mem = T.scalar('sh_mem', dtype='int64')
    t_status = T.scalar('status', dtype='uint32')
    t_data = T.fmatrix('data')
    t_segment_name_int = T.ivector()
    
    #alloc sh_mem
    f = theano.function([t_key], AllocShMemOp()(t_key))
    print >> log.v4, 'Allocating shared memory with key', key, '...' 
    shId, self.sh_mem = f(key)
        
    self.wait_for_status_fn = theano.function([t_sh_mem, t_status], WaitForStatusOp()(t_sh_mem, t_status))
    self.write_status_fn = theano.function([t_sh_mem, t_status], WriteStatusOp()(t_sh_mem, t_status))
    self.copy_to_sh_mem_fn = theano.function([t_sh_mem, t_data], CopyToShMemOp()(t_sh_mem, t_data))
    self.copy_from_sh_mem_fn = theano.function([t_sh_mem], CopyFromShMemOp()(t_sh_mem))
    self.detach_sh_mem_fn = theano.function([t_sh_mem], DetachShMemOp()(t_sh_mem))
    self.write_segment_name_fn = theano.function([t_sh_mem, t_segment_name_int], WriteSegmentNameOp()(t_sh_mem, t_segment_name_int))
    
  def __write_segment_name(self, seg_name):
    seg_name_int = numpy.array([ord(c) for c in seg_name], dtype='int32')
    self.write_segment_name_fn(self.sh_mem, seg_name_int)
  
  #returns (loss, errsig)  
  def get_error_signal(self, segment_names, posteriors, seq_lengths):
    errsig = numpy.zeros_like(posteriors)
    loss = []
    for (i,name) in enumerate(segment_names):
      print >> log.v5, 'requesting error signal for segment', name
      print >> log.v5, 'writing segment name'
      self.__write_segment_name(name)
      seg_posteriors = posteriors[:seq_lengths[i],i,:].copy()
      print >> log.v5, 'copying posteriors to shMem'
      self.copy_to_sh_mem_fn(self.sh_mem, seg_posteriors)
      print >> log.v5, 'writing status REQUEST_ERRORSIGNAL'
      self.write_status_fn(self.sh_mem, SprintCommunicator.REQUEST_ERRORSIGNAL)
      print >> log.v5, 'waiting for status ERRORSIGNAL_WRITTEN'
      self.wait_for_status_fn(self.sh_mem, SprintCommunicator.ERRORSIGNAL_WRITTEN)
      print >> log.v5, 'copying errorsignal from shMem'
      seg_loss_arr, seg_errsig = self.copy_from_sh_mem_fn(self.sh_mem)
      assert len(seg_loss_arr) == 1
      loss.append(seg_loss_arr[0])
      errsig[:seq_lengths[i],i,:] = seg_errsig
      print >> log.v5, 'success!'
    return numpy.array(loss, dtype=Tfloat), errsig
  
  def finalize(self):
    print >> log.v4, 'detaching shared memory...'
    self.write_status_fn(self.sh_mem, SprintCommunicator.TERMINATED)
    self.detach_sh_mem_fn(self.sh_mem)

if __name__== '__main__':
  logs = ['stdout'] * 5  #config.list('log', [])
  log_verbosity = [] #config.int_list('log_verbosity', [])
  log_format = [] #config.list('log_format', [])
  log.initialize(logs = logs, verbosity = log_verbosity, formatter = log_format)
  #comm = SprintCommunicator(int(sys.argv[1]) + 1)
