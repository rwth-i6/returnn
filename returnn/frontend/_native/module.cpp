
#include <Python.h>
#include <string.h>
#include "module.hpp"
#include "backend.hpp"
#include "tensor_ops.hpp"

// https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
static PyMethodDef _pyModuleMethods[] = {
    {"get_backend_for_tensor", (PyCFunction) pyGetBackendForTensor, METH_FASTCALL,
        "get RETURNN frontend backend for RETURNN Tensor. like Tensor.raw_tensor"},
    {"is_raw_torch_tensor_type", (PyCFunction) pyIsRawTorchTensorType, METH_FASTCALL,
        "isinstance(raw_tensor, torch.Tensor)"},
    // ...
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static int _pyModuleExec(PyObject *m) {
    // not sure if we really need this, or if we rather lazily init everything...
    (void)m;
    return 0;
}

static PyModuleDef_Slot _pyModuleSlots[] = {
    {Py_mod_exec, (void*) _pyModuleExec},
#ifdef Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#endif
    {0, NULL}
};

static int _pyModuleTraverse(PyObject *m, visitproc visit, void *arg) {
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(m);
    if(!modState)
        return -1;
    return modState->pyTraverse(visit, arg);
}

static int _pyModuleClear(PyObject *m) {
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(m);
    if(!modState)
        return -1;
    return modState->pyClear();
}

static void _pyModuleFree(PyObject* m) {
    _pyModuleClear(m);
}

// https://docs.python.org/3/c-api/module.html
// https://peps.python.org/pep-3121/
// Code examples:
// https://github.com/python/cpython/blob/51863b7d6ea183167da09fc6b3f2745a1aaa4ef5/Python/import.c#L3872C36-L3872C72
// https://github.com/faster-cpython/cpython/blob/5f85b443f7119e1c68a15fc9a342655e544d2852/Modules/_ssl.c#L6296
// https://github.com/charlesneimog/py4pd/blob/cc53735edf8f0d10340a417dda239bd634036a87/src/module.c#L1307
static struct PyModuleDef _pyModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_returnn_frontend_native",
    "RETURNN frontend internal native module",
    sizeof(PyModuleState), // is null-initialised
    _pyModuleMethods,
    _pyModuleSlots,
    _pyModuleTraverse,
    _pyModuleClear,
    (freefunc) _pyModuleFree
};

PyMODINIT_FUNC PyInit__returnn_frontend_native(void) {
    return PyModuleDef_Init(&_pyModuleDef);
}

bool PyModuleState::_torchTensorTypeMaybeInit(PyObject* obj) {
    {
        PyObject* modName = PyObject_GetAttrString(obj, "__module__");
        if(!modName) {
            PyErr_Clear();
            return false;
        }

        const char* modNameStr = PyUnicode_AsUTF8(modName);
        if(!modNameStr) {
            Py_DECREF(modName);
            PyErr_Clear();
            return false;
        }

        if(memcmp(modNameStr, "torch", 5) != 0 || (modNameStr[5] != '\0' && modNameStr[5] != '.')) {
            Py_DECREF(modName);
            return false;
        }
        Py_DECREF(modName);
    }

    PyObject* mod = PyImport_ImportModule("torch");
    if(!mod) {
        PyErr_Clear();
        return false;
    }
    _torchTensorType = PyObject_GetAttrString(mod, "Tensor");
    Py_DECREF(mod);
    if(!_torchTensorType) {
        PyErr_Clear();
        return false;
    }
    return true;
}
