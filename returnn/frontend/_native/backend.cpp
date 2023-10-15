#include <string.h>
#include "backend.hpp"
#include "module.hpp"
#include "py_utils.hpp"

PyObject* pyGetBackendForTensor(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "get_backend_for_tensor() takes exactly 1 argument: tensor");
        return NULL;
    }
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;
    PyObject* res = getBackendForTensor(modState, args[0]);
    Py_XINCREF(res);
    return res;
}

PyObject* pyIsRawTorchTensorType(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "is_raw_torch_tensor_type() takes exactly 1 argument: raw_tensor");
        return NULL;
    }
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;
    bool res = modState->isTorchTensorType(args[0]);
    return PyBool_FromLong(res);
}

// like TorchBackend.get_dtype_name_raw()
PyObject* pyRawTorchTensorGetDType(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "raw_torch_tensor_get_dtype() takes exactly 1 argument: raw_tensor");
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;

    PyObjectScopedRef dtypeObj = PyObject_GetAttrString(args[0], "dtype");
    if(!dtypeObj) return NULL;

    {
        // very fast path - check some predefined types (faster than dict lookup, small number only)
        bool error = false;
        PyObject* name = modState->torchTensorDTypeName(dtypeObj, error);
        if(!name && error) return NULL;
        if(name) {
            Py_INCREF(name);
            return name;
        }
    }

    PyObject* dispatchTable = modState->torchTensorNameToDTypeDict();
    if(!dispatchTable) return NULL;
    {
        PyObject* name = PyDict_GetItem(dispatchTable, dtypeObj);
        if(name) {
            Py_INCREF(name);
            return name;
        }
    }

    {
        PyObjectScopedRef dtypeStr = PyObject_Str(dtypeObj); // e.g. "torch.float32"
        if(!dtypeStr) return NULL;
        if(!PyUnicode_Check(dtypeStr)) {
            PyErr_Format(
                PyExc_TypeError,
                "raw_torch_tensor_get_dtype: dtype.__str__() did not return a string, from dtype %R", dtypeObj.get());
            return NULL;
        }
        const char* dtypeStrC = PyUnicode_AsUTF8(dtypeStr);
        if(memcmp(dtypeStrC, "torch.", 6) != 0) {
            PyErr_Format(
                PyExc_TypeError,
                "raw_torch_tensor_get_dtype: "
                "dtype.__str__() did not return a string starting with 'torch.', from dtype %R, str '%s'",
                dtypeObj.get(), dtypeStrC);
            return NULL;
        }
        dtypeStrC += 6;
        dtypeStr = PyUnicode_InternFromString(dtypeStrC);
        if(!dtypeStr) return NULL;
        if(PyDict_SetItem(dispatchTable, dtypeObj, dtypeStr) < 0) return NULL;
        return dtypeStr.release();
    }
}

PyObject* getBackendForTensor(PyModuleState* modState, PyObject* obj) {
    PyObjectScopedRef raw_tensor = PyObject_GetAttrString(obj, "_raw_tensor");
    if(!raw_tensor) return NULL;
    return getBackendForRawTensor(modState, raw_tensor);
}

PyObject* getBackendForRawTensor(PyModuleState* modState, PyObject* obj) {
    if(obj == Py_None)
        Py_RETURN_NONE;
    return getBackendForRawTensorType(modState, (PyObject*) Py_TYPE(obj));
}

/*borrowed*/ PyObject* getBackendForRawTensorType(PyModuleState* modState, PyObject* obj) {
    // fast path 1 -- try some predefined types (faster than dict lookup)
    if(modState->isTorchTensorType(obj))
        return modState->torchBackend();

    // fast path 2 -- try dispatch table
    PyObject* dispatchTable = modState->backendTensorTypeDispatchTable(); // borrowed
    PyObject* backend = PyDict_GetItem(dispatchTable, obj); // borrowed
    if(backend) return backend;

    // generic fallback
    PyObjectScopedRef mod = PyImport_ImportModule("returnn.frontend._backend");
    if(!mod) return NULL;
    PyObjectScopedRef methodName = PyUnicode_InternFromString("get_backend_by_raw_tensor_type");
    if(!methodName) return NULL;

    backend = PyObject_CallMethodObjArgs(mod, methodName, obj, NULL);
    Py_XDECREF(backend); // make it borrowed; it will be referenced elsewhere
    return backend;
}

bool isTorchBackendForTensor(PyModuleState* modState, PyObject* obj) {
    PyObjectScopedRef raw_tensor = PyObject_GetAttrString(obj, "_raw_tensor");
    if(!raw_tensor) {
        PyErr_Clear();
        return false;
    }

    return modState->isTorchTensorType((PyObject*) Py_TYPE(raw_tensor));
}
