#ifndef __RETURNN_BACKEND_HPP__
#define __RETURNN_BACKEND_HPP__

// borrowed references. but for NULL, an exception is raised
PyObject* getBackendForTensor(PyObject* module, PyObject* obj);
PyObject* getBackendForRawTensor(PyObject* module, PyObject* obj);
PyObject* getBackendForRawTensorType(PyObject* module, PyObject* obj);

bool isTorchRawTensor(PyObject* obj);
bool isTorchTensor(PyObject* obj);

#endif
