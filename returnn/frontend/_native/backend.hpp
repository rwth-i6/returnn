#ifndef __RETURNN_BACKEND_HPP__
#define __RETURNN_BACKEND_HPP__

#include <Python.h>

// exported Python functions {

PyObject* pyGetBackendForTensor(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyIsRawTorchTensorType(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

// }

// return borrowed references. but for NULL, an exception is raised
PyObject* getBackendForTensor(PyObject* module, PyObject* obj);
PyObject* getBackendForRawTensor(PyObject* module, PyObject* obj);
PyObject* getBackendForRawTensorType(PyObject* module, PyObject* obj);

#endif
