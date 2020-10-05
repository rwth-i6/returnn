
funloader_support_code = """
#include <sstream>
#include <vector>

void printPyObj(PyObject * o)
{
  PyObject* objectsRepresentation = PyObject_Repr(o);
  const char* s = PyString_AsString(objectsRepresentation);
  std::cout << s << std::endl;
  Py_DECREF(objectsRepresentation);
}

struct FunLoader
{
  PyObject * fn;
  PyObject * reset_fn;
  PyObject * mod;
  PyObject * setup_fn;
  std::vector<PyObject*> res_shared;
  std::string name;

  FunLoader(long recurrent_transform_id, const char * fn_name, const char * reset_fn_name = 0)
  {
    //std::cout << "Loading function " << fn_name << "..." << std::endl;
    name = fn_name;
    mod = PyImport_ImportModule("CustomLSTMFunctions");
    if(!mod) PyErr_Print();
    assert(mod);
    std::stringstream sss("setup_parent_functions");
    setup_fn = PyObject_GetAttrString(mod, sss.str().c_str());
    assert(setup_fn);
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args,0,PyString_FromString(fn_name));
    PyTuple_SetItem(args,1,PyLong_FromLong(recurrent_transform_id));
    PyObject* r = PyObject_CallObject(setup_fn, args);
    if(!r) PyErr_Print();
    Py_XDECREF(r);
    Py_XDECREF(args);
    fn = PyObject_GetAttrString(mod, fn_name);
    if(!fn) PyErr_Print();
    assert(fn);
    if(reset_fn_name)
    {
      reset_fn = PyObject_GetAttrString(mod, reset_fn_name);
    }
    else
    {
      reset_fn = 0;
    }
    std::stringstream ss0;
    ss0 << fn_name << "_res0";
    std::stringstream ss1;
    ss1 << fn_name << "_res1";
    PyObject * res0 = PyObject_GetAttrString(mod, ss0.str().c_str());
    PyObject * res1 = PyObject_GetAttrString(mod, ss1.str().c_str());
    assert(res0);
    res_shared.push_back(res0);
    assert(PyList_Check(res1));
    int len = PyList_Size(res1);
    for(int i = 0; i < len; ++i)
    {
      PyObject* obj = PyList_GetItem(res1, i);
      Py_XINCREF(obj);
      res_shared.push_back(obj);
    }

    //std::cout << "loaded function" << std::endl;
  }

  ~FunLoader()
  {
    Py_XDECREF(fn);
    Py_XDECREF(reset_fn);
    for(int i = 0; i < res_shared.size(); ++i)
    {
      Py_XDECREF(res_shared[i]);
    }
    Py_DECREF(mod);
  }

  void reset_shared(CudaNdarray** args, size_t num_args)
  {
    assert(reset_fn);
    PyObject* py_args = PyTuple_New(num_args);
    for(size_t i = 0; i < num_args; ++i) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(py_args, i, (PyObject*) args[i]);
    }
    PyObject* r = PyObject_CallObject(reset_fn, py_args);
    if(!r) PyErr_Print();
    assert(r);
    Py_XDECREF(r);
    Py_DECREF(py_args);
  }

  std::vector<CudaNdarray*> call_helper(PyObject * args)
  {
    //std::cout << "calling custom function " << name << "..." << std::endl;
    PyObject* r = PyObject_CallObject(fn, args);
    if(!r) PyErr_Print();
    assert(r);
    Py_XDECREF(r);
    Py_DECREF(args);

    std::vector<CudaNdarray*> res;
    for(int i = 0; i < res_shared.size(); ++i)
    {
      //res_shared.get_value(borrow=True, return_internal_type=True)
      PyObject * sub_res = PyObject_CallMethod(res_shared[i], "get_value", "(ii)", 1, 1);
      if(!sub_res) PyErr_Print();
      assert(sub_res);
      res.push_back((CudaNdarray*) sub_res);
    }
    //std::cout << "custom function finished" << std::endl;
    return res;
  }

  std::vector<CudaNdarray*> call(CudaNdarray** args, size_t num_args)
  {
    PyObject* py_args = PyTuple_New(num_args);
    for(size_t i = 0; i < num_args; ++i) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(py_args, i, (PyObject*) args[i]);
    }
    return call_helper(py_args);
  }

  void debug_print(CudaNdarray* v) {
    PyObject* numpy_array_obj = CudaNdarray_CreateArrayObj(v);
    if(!numpy_array_obj) { PyErr_Print(); goto end; }
    PyObject_Print(numpy_array_obj, stdout, Py_PRINT_RAW);
    printf("\\n");
  end:
    Py_XDECREF(numpy_array_obj);
  }

};

"""

def make_funloader_code(recurrent_transform, fn_name, reset_fn_name=None):
  reset_fn_name_str = ('"%s"' % reset_fn_name) if reset_fn_name is not None else "0"
  recurrent_transform_id = id(recurrent_transform)
  return funloader_support_code + """
  FunLoader %(fn_name)s(%(recurrent_transform_id)i, "%(fn_name)s", %(reset_fn_name_str)s);
  """ % locals()
