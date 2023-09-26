
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
    {"tensor_compare", (PyCFunction) pyTensorCompare, METH_VARARGS | METH_KEYWORDS, "rf.compare"},
    {"tensor_combine", (PyCFunction) pyTensorCombine, METH_VARARGS | METH_KEYWORDS, "rf.combine"},
    {"tensor_eq", (PyCFunction) pyTensorEq, METH_FASTCALL, "Tensor.__eq__"},
    {"tensor_ne", (PyCFunction) pyTensorNe, METH_FASTCALL, "Tensor.__ne__"},
    {"tensor_lt", (PyCFunction) pyTensorLt, METH_FASTCALL, "Tensor.__lt__"},
    {"tensor_le", (PyCFunction) pyTensorLe, METH_FASTCALL, "Tensor.__le__"},
    {"tensor_gt", (PyCFunction) pyTensorGt, METH_FASTCALL, "Tensor.__gt__"},
    {"tensor_ge", (PyCFunction) pyTensorGe, METH_FASTCALL, "Tensor.__ge__"},
    // TODO ...
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static int _pyModuleExec(PyObject *m) {
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(m);
    if(!modState)
        return -1;
    return modState->pyInitModuleExec();
}

int PyModuleState::pyInitModuleExec() {
    {
        PyObject* mod = PyImport_ImportModule("returnn.tensor");
        if(!mod)
            return -1;
        _tensorType = PyObject_GetAttrString(mod, "Tensor");
        if(!_tensorType) {
            Py_DECREF(mod);
            return -1;
        }
        Py_DECREF(mod);
    }

    {
        PyObject* mod = PyImport_ImportModule("returnn.frontend._backend");
        if(!mod)
            return -1;
        _globalBackend = PyObject_GetAttrString(mod, "global_backend");
        if(!_globalBackend) {
            Py_DECREF(mod);
            return -1;
        }
        Py_DECREF(mod);
    }

    {
        PyObject* mod = PyImport_ImportModule("returnn.frontend");
        if(!mod)
            return -1;
        PyObject* rawTensorTypesUnion = PyObject_GetAttrString(mod, "RawTensorTypes");
        if(!rawTensorTypesUnion) {
            Py_DECREF(mod);
            return -1;
        }
        PyObject* rawTensorTypesTuple = PyObject_GetAttrString(rawTensorTypesUnion, "__args__");
        if(!rawTensorTypesTuple) {
            Py_DECREF(mod);
            Py_DECREF(rawTensorTypesUnion);
            return -1;
        }
        if(!PyTuple_Check(rawTensorTypesTuple)) {
            Py_DECREF(mod);
            Py_DECREF(rawTensorTypesUnion);
            Py_DECREF(rawTensorTypesTuple);
            PyErr_Format(PyExc_TypeError, "RETURNN frontend _native: RawTensorTypes is not a tuple");
            return -1;
        }
        _rawTensorTypesLen = PyTuple_GET_SIZE(rawTensorTypesTuple);
        if(_rawTensorTypesLen < 0 || (size_t)_rawTensorTypesLen > sizeof(_rawTensorTypes) / sizeof(_rawTensorTypes[0])) {
            _rawTensorTypesLen = 0;
            Py_DECREF(mod);
            Py_DECREF(rawTensorTypesUnion);
            Py_DECREF(rawTensorTypesTuple);
            PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: too many RawTensorTypes (%i)", _rawTensorTypesLen);
            return -1;
        }
        for(int i = 0; i < _rawTensorTypesLen; ++i) {
            PyObject* obj = PyTuple_GET_ITEM(rawTensorTypesTuple, i);
            if(!obj) {
                PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: RawTensorTypes tuple item %zd is NULL", i);
                Py_DECREF(mod);
                Py_DECREF(rawTensorTypesUnion);
                Py_DECREF(rawTensorTypesTuple);
                return -1;
            }
            _rawTensorTypes[i] = obj;
        }
        Py_DECREF(mod);
        Py_DECREF(rawTensorTypesUnion);
        Py_DECREF(rawTensorTypesTuple);
    }

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

bool PyModuleState::_cachedOpInit(BackendWithCachedOps backend) {
    if(backend == BWCO_Torch)
        return _cachedOpInitTorch();
    PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: invalid backend '%d'", backend);
    return false;
}

bool PyModuleState::_cachedOpInitTorch() {
    PyObject** ops = _cachedOps + BWCO_Torch * NumTOps;
    PyObject* mod = PyImport_ImportModule("torch");
    if(!mod)
        return false;
    PyObject* modAlternatives = PyImport_ImportModule("returnn.torch.frontend.raw_ops");
    if(!modAlternatives) {
        Py_DECREF(mod);
        return false;
    }

    // init all RawOp's

    #define AddOp(op, name) ops[op] = PyObject_GetAttrString(mod, name); if(!ops[op]) goto error;
    #define AddOpAlt(op, name) ops[op] = PyObject_GetAttrString(modAlternatives, name); if(!ops[op]) goto error;

    AddOp(TOp_Permute, "permute");
    AddOp(TOp_Reshape, "reshape");
    AddOp(TOp_Eq, "eq");
    AddOp(TOp_Ne, "not_equal");
    AddOp(TOp_Lt, "less");
    AddOp(TOp_Le, "less_equal");
    AddOp(TOp_Gt, "greater");
    AddOp(TOp_Ge, "greater_equal");
    AddOp(TOp_Add, "add");
    AddOp(TOp_Sub, "sub");
    AddOp(TOp_Mul, "mul");
    AddOp(TOp_TrueDiv, "true_divide");
    AddOp(TOp_FloorDiv, "floor_divide");
    AddOp(TOp_Mod, "remainder");
    AddOp(TOp_Pow, "pow");
    AddOpAlt(TOp_SquaredDifference, "squared_difference");
    AddOp(TOp_Neg, "neg");
    AddOp(TOp_Abs, "abs");
    AddOp(TOp_And, "logical_and");
    AddOp(TOp_Or, "logical_or");
    AddOp(TOp_Not, "logical_not");

    #undef AddOp
    #undef AddOpAlt

    Py_DECREF(mod);
    Py_DECREF(modAlternatives);
    return true;

error:
    Py_DECREF(mod);
    Py_DECREF(modAlternatives);
    return false;
}

const char* rawOpName(RawOp op) {
    static const char* names[NumTOps] = {NULL};
    if(!names[0]) {
        names[TOp_ConvertToTensor] = "tensor";
        names[TOp_Permute] = "permute";
        names[TOp_Reshape] = "reshape";
        names[TOp_Eq] = "equal";
        names[TOp_Ne] = "not_equal";
        names[TOp_Lt] = "less";
        names[TOp_Le] = "less_equal";
        names[TOp_Gt] = "greater";
        names[TOp_Ge] = "greater_equal";
        names[TOp_Add] = "add";
        names[TOp_Sub] = "sub";
        names[TOp_Mul] = "mul";
        names[TOp_TrueDiv] = "truedivide";
        names[TOp_FloorDiv] = "floordivide";
        names[TOp_Mod] = "mod";
        names[TOp_Pow] = "pow";
        names[TOp_SquaredDifference] = "squared_difference";
        names[TOp_Neg] = "neg";
        names[TOp_Abs] = "abs";
        names[TOp_And] = "logical_and";
        names[TOp_Or] = "logical_or";
        names[TOp_Not] = "logical_not";
    }
    if(!names[op]) {
        PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: invalid RawOp '%d'", op);
        return NULL;
    }
    return names[op];
}
