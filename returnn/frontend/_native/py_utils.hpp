#ifndef __RETURNN_PY_UTILS_HPP__
#define __RETURNN_PY_UTILS_HPP__

#include <Python.h>

/* When you call any Python API which returns a new reference, e.g. PyObject_Call or whatever,
 * you need to Py_DECREF it at some point.
 * This class is a simple wrapper which does that automatically when it goes out of scope.
 * Example:
 *   PyObjectScopedRef result = PyObject_Call(...);
 *   // do something with ref
 *   // no need to Py_DECREF(ref) explicitly
*/
class PyObjectScopedRef {
public:
    PyObjectScopedRef(PyObject* obj = NULL) : _obj(obj) {}
    PyObjectScopedRef(const PyObjectScopedRef&) = delete;
    PyObjectScopedRef(PyObjectScopedRef&& other) : _obj(other._obj) { other._obj = NULL; }
    void operator=(PyObject* obj) { Py_CLEAR(_obj); _obj = obj; }
    void operator=(const PyObjectScopedRef&) = delete;
    operator PyObject*() const { return _obj; }
    PyObject* get() const { return _obj; }
    PyObject* release() { PyObject* obj = _obj; _obj = NULL; return obj; }
    ~PyObjectScopedRef() { Py_CLEAR(_obj); }

private:
    PyObject* _obj;
};

#endif
