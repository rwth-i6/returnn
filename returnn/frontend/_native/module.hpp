#ifndef __RETURNN_MODULE_HPP__
#define __RETURNN_MODULE_HPP__

#include <Python.h>

enum {
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

enum {
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

private:
    PyObject* _backendTensorTypeDispatchTable;
    PyObject* _torchTensorType;
};

#endif
