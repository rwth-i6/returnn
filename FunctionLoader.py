
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
  std::vector<PyObject*> res_shared;
  std::string name;

  FunLoader(const char * fn_name, int n_outputs)
  {
    std::cout << "Loading function " << fn_name << "..." << std::endl;
    name = fn_name;
    //TODO: this mod object is never decref'd
    static PyObject * mod = 0;
    if(!mod)
    {
      mod = PyImport_AddModule("CustomLSTMFunctions");
    }
    assert(mod);
    fn = PyObject_GetAttrString(mod, fn_name);
    std::stringstream ss;
    ss << fn_name << "_res";
    for(int i = 0; i < n_outputs; ++i)
    {
      std::stringstream ss2;
      ss2 << ss.str() << i;
      std::string res_name = ss2.str();
      res_shared.push_back(PyObject_GetAttrString(mod, res_name.c_str()));
    }

    std::cout << "loaded function" << std::endl;
  }

  //TODO: this is never executed, as the programs terminates before
  //This causes a nonzero exit code of the program
  ~FunLoader()
  {
    Py_XDECREF(fn);
    for(int i = 0; i < res_shared.size(); ++i)
    {
      Py_XDECREF(res_shared[i]);
    }
    /*if(mod)
    {
      Py_DECREF(mod);
      mod = 0;
    }*/
  }

  std::vector<PyObject*> call_helper(PyObject * args)
  {
    //std::cout << "calling custom function " << name << "..." << std::endl;
    PyObject_CallObject(fn, args);
    Py_DECREF(args);
    //this should be the C++ equivalent for the following python code
    //res = res_shared.get_value(borrow=True, return_internal_type=True)

    std::vector<PyObject*> res;
    for(int i = 0; i < res_shared.size(); ++i)
    {
      PyObject * sub_res = PyObject_CallMethod(res_shared[i], "get_value", "(ii)", 1, 1);
      assert(sub_res);
      res.push_back(sub_res);
    }
    //std::cout << "custom function finished" << std::endl;
    return res;
  }

  std::vector<PyObject*> operator()(CudaNdarray* x)
  {
    PyObject* args = PyTuple_Pack(1, x);
    return call_helper(args);
  }

  std::vector<PyObject*> operator()(CudaNdarray* x, CudaNdarray* y)
  {
    PyObject* args = PyTuple_Pack(2, x, y);
    return call_helper(args);
  }

  std::vector<PyObject*> operator()(CudaNdarray* x, CudaNdarray* y, CudaNdarray* z)
  {
    PyObject* args = PyTuple_Pack(3, x, y, z);
    return call_helper(args);
  }

  std::vector<PyObject*> operator()(CudaNdarray* x0, CudaNdarray* x1, CudaNdarray* x2, CudaNdarray* x3)
  {
    PyObject* args = PyTuple_Pack(4, x0, x1, x2, x3);
    return call_helper(args);
  }

  std::vector<PyObject*> operator()(CudaNdarray* x0, CudaNdarray* x1, CudaNdarray* x2, CudaNdarray* x3, CudaNdarray* x4)
  {
    PyObject* args = PyTuple_Pack(5, x0, x1, x2, x3, x4);
    return call_helper(args);
  }

  //TODO add overloads for more arguments if needed
  //(variadic template would be better but not widely supported by compilers)

};

"""

def make_funloader_code(fn_name, n_res):
  n_res = str(n_res)
  return funloader_support_code + """
  FunLoader %(fn_name)s("%(fn_name)s", %(n_res)s);
  """ % locals()
