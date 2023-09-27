#ifndef __RETURNN_TENSOR_OPS_HPP__
#define __RETURNN_TENSOR_OPS_HPP__

#include <Python.h>

class PyModuleState;

// generic

PyObject* tensorCopyTemplate(PyModuleState* modState, PyObject* tensor);

// exported Python functions {

PyObject* pyTensorCopyTemplate(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject* pyTensorCompare(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* pyTensorCombine(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject* pyTensorEq(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorNe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorLt(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorLe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorGt(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorGe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

// math binary ops

PyObject* pyTensorAdd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRAdd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorSub(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRSub(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorMul(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRMul(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorTrueDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRTrueDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorFloorDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRFloorDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorMod(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRMod(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorPow(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRPow(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

PyObject* pyTensorAnd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRAnd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorOr(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorROr(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

// math unary ops

PyObject* pyTensorNeg(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorAbs(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

// }

#endif
