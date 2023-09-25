#include "backend.hpp"
#include "module.hpp"

PyObject* pyGetBackendForTensor(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "get_backend_for_tensor() takes exactly 1 argument: tensor");
        return NULL;
    }
    PyObject* res = getBackendForTensor(self, args[0]);
    Py_XINCREF(res);
    return res;
}

PyObject* pyIsRawTorchTensorType(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "is_raw_torch_tensor_type() takes exactly 1 argument: raw_tensor");
        return NULL;
    }
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;
    bool res = modState->isTorchTensorType(args[0]);
    return PyBool_FromLong(res);
}

PyObject* getBackendForTensor(PyObject* module, PyObject* obj) {
    PyObject* raw_tensor = PyObject_GetAttrString(obj, "_raw_tensor");
    if(!raw_tensor)
        return NULL;
    PyObject* backend = getBackendForRawTensor(module, raw_tensor);
    Py_DECREF(raw_tensor);
    return backend;
}

PyObject* getBackendForRawTensor(PyObject* module, PyObject* obj) {
    return getBackendForRawTensorType(module, (PyObject*) Py_TYPE(obj));
}

/*borrowed*/ PyObject* getBackendForRawTensorType(PyObject* module, PyObject* obj) {
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(module);
    if(!modState)
        return NULL;

    // fast path 1 -- try some predefined types (faster than dict lookup)
    if(modState->isTorchTensorType(obj))
        return modState->torchBackend();

    // fast path 2 -- try dispatch table
    PyObject* dispatchTable = modState->backendTensorTypeDispatchTable(); // borrowed
    PyObject* backend = PyDict_GetItem(dispatchTable, obj); // borrowed
    if(backend)
        return backend;

    // generic fallback
    PyObject* mod = PyImport_ImportModule("returnn.frontend._backend");
    if(!mod)
        return NULL;
    PyObject* methodName = PyUnicode_InternFromString("get_backend_by_raw_tensor_type");
    if(!methodName) {
        Py_DECREF(mod);
        return NULL;
    }
    backend = PyObject_CallMethodObjArgs(mod, methodName, obj, NULL);
    Py_DECREF(methodName);
    Py_XDECREF(backend); // make it borrowed; it will be referenced elsewhere
    return backend;
}
