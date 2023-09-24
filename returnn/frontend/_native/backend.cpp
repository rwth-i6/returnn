#include "backend.hpp"
#include "module.hpp"

PyObject* getBackendForTensor(PyObject* module, PyObject* obj) {
    PyObject* raw_tensor = PyObject_GetAttrString(obj, "_raw_tensor");
    if(!raw_tensor)
        return NULL;
    PyObject* backend = getBackendForRawTensor(module, raw_tensor);
    Py_DECREF(raw_tensor);
    return backend;
}

PyObject* getBackendForRawTensor(PyObject* module, PyObject* obj) {
    return getBackendForRawTensorType(module, Py_TYPE(obj));
}

PyObject* getBackendForRawTensorType(PyObject* module, PyObject* obj) {
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(m);
    PyObject* dispatchTable = modState->backendTensorTypeDispatchTable(); // borrowed
    PyObject* backend = PyDict_GetItem(dispatchTable, obj); // borrowed
    if(backend)
        return backend;

    PyObject* mod = PyImport_ImportModule("returnn.frontend._backend");
    if(!mod)
        return NULL;
    backend = PyObject_CallMethodObjArgs(mod, "get_backend_by_raw_tensor_type", obj, NULL);
    Py_XDECREF(backend); // make it borrowed; it will be referenced elsewhere
    return backend;
}
