#ifndef __RETURNN_TENSOR_OPS_HPP__
#define __RETURNN_TENSOR_OPS_HPP__

#include <Python.h>

// exported Python functions {

PyObject* pyCompare(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* pyCombine(PyObject *self, PyObject *args, PyObject *kwargs);

PyObject* pyTensorEq(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorNe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorLt(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorLe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorGt(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorGe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

// math binary and unary ops

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

PyObject* pyTensorNeg(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorAbs(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

PyObject* pyTensorAnd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorRAnd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorOr(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTensorROr(PyObject *self, PyObject *const *args, Py_ssize_t nargs);


// PyTorch specialized ops

PyObject* pyTorchCompare(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchCombine(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

PyObject* pyTorchTensorEq(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorNe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorLt(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorLe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorGt(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorGe(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

PyObject* pyTorchTensorAdd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorRAdd(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorSub(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorRSub(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorMul(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorRMul(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorTrueDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorRTrueDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorFloorDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorRFloorDiv(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorMod(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorRMod(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyTorchTensorPow(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

// }

#endif
