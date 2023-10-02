#ifndef __RETURNN_BACKEND_HPP__
#define __RETURNN_BACKEND_HPP__

#include <Python.h>

class PyModuleState;

// exported Python functions {

PyObject* pyGetBackendForTensor(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyIsRawTorchTensorType(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
PyObject* pyRawTorchTensorGetDType(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

// }

// return borrowed references. but for NULL, an exception is raised
PyObject* getBackendForTensor(PyModuleState* modState, PyObject* obj);
PyObject* getBackendForRawTensor(PyModuleState* modState, PyObject* obj);
PyObject* getBackendForRawTensorType(PyModuleState* modState, PyObject* obj);

bool isTorchBackendForTensor(PyModuleState* modState, PyObject* obj);

#endif
