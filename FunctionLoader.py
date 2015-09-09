
funloader_support_code = """
void printPyObj(PyObject * o)
{
  PyObject* objectsRepresentation = PyObject_Repr(o);
  const char* s = PyString_AsString(objectsRepresentation);
  std::cout << s << std::endl;
  Py_DECREF(objectsRepresentation);
}

#include <sstream>

struct FunLoader
{
  PyObject * fn;
  PyObject * res_shared;

  FunLoader(const char * fn_name)
  {
    std::cout << "Loading function " << fn_name << "..." << std::endl;
    static PyObject * mod = 0;
    if(!mod)
    {
      mod = PyImport_AddModule("CustomLSTMFunctions");
    }
    assert(mod);
    fn = PyObject_GetAttrString(mod, fn_name);
    std::stringstream ss;
    ss << fn_name << "_res";
    res_shared = PyObject_GetAttrString(mod, ss.str().c_str());

    std::cout << "loaded function" << std::endl;
  }

  //TODO: this is never executed, as the programs terminates before
  //This causes a nonzero exit code of the program
  ~FunLoader()
  {
    Py_XDECREF(fn);
    Py_XDECREF(res_shared);
    /*if(mod)
    {
      Py_DECREF(mod);
      mod = 0;
    }*/
  }

  PyObject * operator()(CudaNdarray* x)
  {
    PyObject* args = PyTuple_Pack(1, x);
    PyObject_CallObject(fn, args);
    Py_DECREF(args);

    //this should be the C++ equivalent for the following python code
    //res = res_shared.get_value(borrow=True, return_internal_type=True)
    PyObject * res = PyObject_CallMethod(res_shared, "get_value", "(ii)", 1, 1);
    assert(res);
    return res;
  }

  PyObject * operator()(CudaNdarray* x, CudaNdarray* y)
  {
    PyObject* args = PyTuple_Pack(2, x, y);
    PyObject_CallObject(fn, args);
    Py_DECREF(args);

    //this should be the C++ equivalent for the following python code
    //res = res_shared.get_value(borrow=True, return_internal_type=True)
    PyObject * res = PyObject_CallMethod(res_shared, "get_value", "(ii)", 1, 1);
    assert(res);
    return res;
  }

  PyObject * operator()(CudaNdarray* x, CudaNdarray* y, CudaNdarray* z)
  {
    PyObject* args = PyTuple_Pack(3, x, y, z);
    PyObject_CallObject(fn, args);
    Py_DECREF(args);

    //this should be the C++ equivalent for the following python code
    //res = res_shared.get_value(borrow=True, return_internal_type=True)
    PyObject * res = PyObject_CallMethod(res_shared, "get_value", "(ii)", 1, 1);
    assert(res);
    return res;
  }

  //TODO add overloads for more arguments if needed
  //(variadic template would be better but not widely supported by compilers)

};

"""

def make_funloader_code(fn_name):
  return funloader_support_code + """
  FunLoader %(fn_name)s("%(fn_name)s");
  """ % locals()

