#ifndef __RETURNN_MODULE_HPP__
#define __RETURNN_MODULE_HPP__

#include <Python.h>

// raw ops
enum {
    TOp_Permute,
    TOp_Reshape,

    TOp_Eq,
    TOp_Ne,
    TOp_Lt,
    TOp_Le,
    TOp_Gt,
    TOp_Ge,

    TOp_Add,
    TOp_Sub,
    TOp_Mul,
    TOp_TrueDiv,
    TOp_FloorDiv,
    TOp_Mod,
    TOp_Pow,

    TOp_Neg,
    TOp_Abs,

    TOp_And,
    TOp_Or,

    NumTOps,
};

// all backends where we want to cache the ops, to support very efficient inlined code
enum {
    BWCO_Torch,

    NumBackendsWithCachedOps,
};

enum {
    Backend_PythonFallback,
    Backend_Generic,
    Backend_Torch,

    NumBackends,
};

class PyModuleState {
public:
    // borrowed references. NULL means error, exception is raised.
    inline PyObject* backendTensorTypeDispatchTable() {
        if(_backendTensorTypeDispatchTable)
            return _backendTensorTypeDispatchTable;
        PyObject* mod = PyImport_ImportModule("returnn.frontend._backend");
        if(!mod)
            return NULL;
        _backendTensorTypeDispatchTable = PyObject_GetAttrString(mod, "_backend_tensor_type_dispatch_table");
        Py_DECREF(mod);
        return _backendTensorTypeDispatchTable;
    }

    inline bool isTorchTensorType(PyObject* obj) {
        if(obj == _torchTensorType) // fast path
            return true;
        if(!_torchTensorType) {
            if(!_torchTensorTypeMaybeInit(obj))
                return false;
            if(obj == _torchTensorType) // check again
                return true;
        }
        int res = PyObject_IsSubclass(obj, _torchTensorType);
        if(res < 0) {
            PyErr_Clear();
            return false;
        }
        return res;
    }

    // call only if you know that Torch is anyway available
    inline PyObject* torchBackend() {
        if(_torchBackend)
            return _torchBackend;
        // from returnn.torch.frontend import TorchBackend
        PyObject* mod = PyImport_ImportModule("returnn.torch.frontend");
        if(!mod)
            return NULL;
        _torchBackend = PyObject_GetAttrString(mod, "TorchBackend");
        Py_DECREF(mod);
        return _torchBackend;
    }

private:
    PyObject* _backendTensorTypeDispatchTable;
    PyObject* _cachedOps[NumBackendsWithCachedOps * NumTOps];
    PyObject* _torchTensorType;
    PyObject* _torchBackend;

    bool _torchTensorTypeMaybeInit(PyObject* obj);
};

#endif
