
funloader_support_code = """
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

  FunLoader(const char * fn_name)
  {
    std::cout << "Loading function..." << std::endl;
    PyObject *mod = PyImport_AddModule("CustomLSTMFunctions");
    assert(mod);
    fn = PyObject_GetAttrString(mod, fn_name);
    Py_DECREF(mod);
    std::cout << "loaded function" << std::endl;
  }

  //TODO: this is never executed, as the programs terminates before
  //This causes a nonzero exit code of the program
  ~FunLoader()
  {
    Py_XDECREF(fn);
  }

  PyObject * operator()(CudaNdarray* x)
  {
    PyObject* args = PyTuple_Pack(1, x);
    PyObject * res = PyObject_CallObject(fn, args);
    Py_DECREF(args);
    return res;
  }

  PyObject * operator()(CudaNdarray* x, CudaNdarray* y)
  {
    PyObject* args = PyTuple_Pack(2, x, y);
    PyObject * res = PyObject_CallObject(fn, args);
    Py_DECREF(args);
    return res;
  }

  PyObject * operator()(CudaNdarray* x, CudaNdarray* y, CudaNdarray* z)
  {
    PyObject* args = PyTuple_Pack(3, x, y, z);
    PyObject * res = PyObject_CallObject(fn, args);
    Py_DECREF(args);
    return res;
  }

  //TODO add overloads for more arguements if needed
  //(variadic template would be better but not widely supported by compilers)

};

"""

def make_funloader_code(fn_name):
  #return funloader_support_code + """
  #FunLoader %(fn_name)s("%(fn_name)s");
  #""" % locals()
  return ""
