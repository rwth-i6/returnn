
#include <Python.h>
#include <string.h>
#include "module.hpp"
#include "backend.hpp"
#include "tensor_ops.hpp"
#include "py_utils.hpp"

// https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
static PyMethodDef _pyModuleMethods[] = {
    {"get_backend_for_tensor", (PyCFunction) pyGetBackendForTensor, METH_FASTCALL,
        "get RETURNN frontend backend for RETURNN Tensor. like Tensor.raw_tensor"},
    {"is_raw_torch_tensor_type", (PyCFunction) pyIsRawTorchTensorType, METH_FASTCALL,
        "isinstance(raw_tensor, torch.Tensor)"},
    {"raw_torch_tensor_get_dtype", (PyCFunction) pyRawTorchTensorGetDType, METH_FASTCALL, "TorchBackend.get_dtype_name_raw"},
    {"tensor_raw_tensor_setter", (PyCFunction) pyTensorRawTensorSetter, METH_FASTCALL, "Tensor.raw_tensor.setter"},
    {"convert_to_raw_torch_tensor_like", (PyCFunction) pyConvertToRawTorchTensorLike, METH_FASTCALL,
        "torch.tensor(value, dtype=..., device=...)"},

    {"tensor_copy", (PyCFunction) pyTensorCopy, METH_VARARGS | METH_KEYWORDS, "Tensor.copy"},
    {"tensor_copy_template", (PyCFunction) pyTensorCopyTemplate, METH_VARARGS | METH_KEYWORDS, "Tensor.copy_template"},
    {"tensor_get_out_permutation_to_dims", (PyCFunction) pyTensorGetOutPermutationsToDims, METH_FASTCALL,
        "Tensor.get_out_permutation_to_dims"},
    {"tensor_copy_compatible_to_dims", (PyCFunction) pyTensorCopyCompatibleToDims, METH_FASTCALL, "Tensor.copy_compatible_to_dims"},
    {"tensor_copy_compatible_to_dims_raw", (PyCFunction) pyTensorCopyCompatibleToDimsRaw, METH_FASTCALL, "Tensor.copy_compatible_to_dims_raw"},

    {"tensor_compare", (PyCFunction) pyTensorCompare, METH_VARARGS | METH_KEYWORDS, "rf.compare"},
    {"tensor_combine", (PyCFunction) pyTensorCombine, METH_VARARGS | METH_KEYWORDS, "rf.combine"},

    {"tensor_eq", (PyCFunction) pyTensorEq, METH_FASTCALL, "Tensor.__eq__"},
    {"tensor_ne", (PyCFunction) pyTensorNe, METH_FASTCALL, "Tensor.__ne__"},
    {"tensor_lt", (PyCFunction) pyTensorLt, METH_FASTCALL, "Tensor.__lt__"},
    {"tensor_le", (PyCFunction) pyTensorLe, METH_FASTCALL, "Tensor.__le__"},
    {"tensor_gt", (PyCFunction) pyTensorGt, METH_FASTCALL, "Tensor.__gt__"},
    {"tensor_ge", (PyCFunction) pyTensorGe, METH_FASTCALL, "Tensor.__ge__"},

    {"tensor_add", (PyCFunction) pyTensorAdd, METH_FASTCALL, "Tensor.__add__"},
    {"tensor_radd", (PyCFunction) pyTensorRAdd, METH_FASTCALL, "Tensor.__radd__"},
    {"tensor_sub", (PyCFunction) pyTensorSub, METH_FASTCALL, "Tensor.__sub__"},
    {"tensor_rsub", (PyCFunction) pyTensorRSub, METH_FASTCALL, "Tensor.__rsub__"},
    {"tensor_mul", (PyCFunction) pyTensorMul, METH_FASTCALL, "Tensor.__mul__"},
    {"tensor_rmul", (PyCFunction) pyTensorRMul, METH_FASTCALL, "Tensor.__rmul__"},
    {"tensor_truediv", (PyCFunction) pyTensorTrueDiv, METH_FASTCALL, "Tensor.__truediv__"},
    {"tensor_rtruediv", (PyCFunction) pyTensorRTrueDiv, METH_FASTCALL, "Tensor.__rtruediv__"},
    {"tensor_floordiv", (PyCFunction) pyTensorFloorDiv, METH_FASTCALL, "Tensor.__floordiv__"},
    {"tensor_rfloordiv", (PyCFunction) pyTensorRFloorDiv, METH_FASTCALL, "Tensor.__rfloordiv__"},
    {"tensor_mod", (PyCFunction) pyTensorMod, METH_FASTCALL, "Tensor.__mod__"},
    {"tensor_rmod", (PyCFunction) pyTensorRMod, METH_FASTCALL, "Tensor.__rmod__"},
    {"tensor_pow", (PyCFunction) pyTensorPow, METH_FASTCALL, "Tensor.__pow__"},
    {"tensor_rpow", (PyCFunction) pyTensorRPow, METH_FASTCALL, "Tensor.__rpow__"},

    {"tensor_and", (PyCFunction) pyTensorAnd, METH_FASTCALL, "Tensor.__and__"},
    {"tensor_rand", (PyCFunction) pyTensorRAnd, METH_FASTCALL, "Tensor.__rand__"},
    {"tensor_or", (PyCFunction) pyTensorOr, METH_FASTCALL, "Tensor.__or__"},
    {"tensor_ror", (PyCFunction) pyTensorROr, METH_FASTCALL, "Tensor.__ror__"},

    {"tensor_neg", (PyCFunction) pyTensorNeg, METH_FASTCALL, "Tensor.__neg__"},
    {"tensor_invert", (PyCFunction) pyTensorNot, METH_FASTCALL, "Tensor.__invert__"},
    {"tensor_abs", (PyCFunction) pyTensorAbs, METH_FASTCALL, "Tensor.__abs__"},
    {"tensor_ceil", (PyCFunction) pyTensorCeil, METH_FASTCALL, "Tensor.__ceil__"},
    {"tensor_floor", (PyCFunction) pyTensorFloor, METH_FASTCALL, "Tensor.__floor__"},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static int _pyModuleExec(PyObject *m) {
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(m);
    if(!modState) return -1;
    return modState->pyInitModuleExec(m);
}

int PyModuleState::pyInitModuleExec(PyObject* module) {
    _module = module;

    {
        PyObjectScopedRef mod = PyImport_ImportModule("returnn.util.basic");
        if(!mod) return -1;
        _notSpecified = PyObject_GetAttrString(mod, "NotSpecified");
        if(!_notSpecified) return -1;
    }

    {
        PyObjectScopedRef mod = PyImport_ImportModule("returnn.tensor");
        if(!mod) return -1;
        _tensorType = PyObject_GetAttrString(mod, "Tensor");
        if(!_tensorType) return -1;
        _dimType = PyObject_GetAttrString(mod, "Dim");
        if(!_dimType) return -1;
    }

    {
        PyObjectScopedRef mod = PyImport_ImportModule("returnn.frontend._backend");
        if(!mod) return -1;
        _globalBackend = PyObject_GetAttrString(mod, "global_backend");
        if(!_globalBackend) return -1;
    }

    {
        PyObjectScopedRef mod = PyImport_ImportModule("returnn.frontend");
        if(!mod) return -1;
        PyObjectScopedRef rawTensorTypesUnion = PyObject_GetAttrString(mod, "RawTensorTypes");
        if(!rawTensorTypesUnion) return -1;
        PyObjectScopedRef rawTensorTypesTuple = PyObject_GetAttrString(rawTensorTypesUnion, "__args__");
        if(!rawTensorTypesTuple) return -1;
        if(!PyTuple_Check(rawTensorTypesTuple)) {
            PyErr_Format(PyExc_TypeError, "RETURNN frontend _native: RawTensorTypes is not a tuple");
            return -1;
        }
        _rawTensorTypesLen = PyTuple_GET_SIZE(rawTensorTypesTuple.get());
        if(_rawTensorTypesLen < 0 || (size_t)_rawTensorTypesLen > sizeof(_rawTensorTypes) / sizeof(_rawTensorTypes[0])) {
            _rawTensorTypesLen = 0;
            PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: too many RawTensorTypes (%i)", _rawTensorTypesLen);
            return -1;
        }
        for(int i = 0; i < _rawTensorTypesLen; ++i) {
            PyObject* obj = PyTuple_GET_ITEM(rawTensorTypesTuple.get(), i);
            if(!obj) {
                PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: RawTensorTypes tuple item %zd is NULL", i);
                return -1;
            }
            Py_INCREF(obj);
            _rawTensorTypes[i] = obj;
        }
    }

    {
        #define AddInstanceMethod(name) \
            { \
                PyObjectScopedRef func = PyObject_GetAttrString(_module, "tensor_" #name); \
                if(!func) return -1; \
                PyObjectScopedRef instMethod = PyInstanceMethod_New(func); \
                if(!instMethod) return -1; \
                if(PyModule_AddObject(module, "_tensor_" #name "_instancemethod", instMethod) < 0) \
                    return -1; \
                instMethod.release(); \
            }

        AddInstanceMethod(copy);
        AddInstanceMethod(copy_template);
        AddInstanceMethod(get_out_permutation_to_dims);
        AddInstanceMethod(copy_compatible_to_dims);
        AddInstanceMethod(copy_compatible_to_dims_raw);

        AddInstanceMethod(eq);
        AddInstanceMethod(ne);
        AddInstanceMethod(lt);
        AddInstanceMethod(le);
        AddInstanceMethod(gt);
        AddInstanceMethod(ge);

        AddInstanceMethod(add);
        AddInstanceMethod(radd);
        AddInstanceMethod(sub);
        AddInstanceMethod(rsub);
        AddInstanceMethod(mul);
        AddInstanceMethod(rmul);
        AddInstanceMethod(truediv);
        AddInstanceMethod(rtruediv);
        AddInstanceMethod(floordiv);
        AddInstanceMethod(rfloordiv);
        AddInstanceMethod(mod);
        AddInstanceMethod(rmod);
        AddInstanceMethod(pow);
        AddInstanceMethod(rpow);

        AddInstanceMethod(and);
        AddInstanceMethod(rand);
        AddInstanceMethod(or);
        AddInstanceMethod(ror);

        AddInstanceMethod(neg);
        AddInstanceMethod(invert);
        AddInstanceMethod(abs);
        AddInstanceMethod(ceil);
        AddInstanceMethod(floor);

        #undef AddInstanceMethod
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
    if(!modState) return -1;
    return modState->pyTraverse(visit, arg);
}

static int _pyModuleClear(PyObject *m) {
    PyModuleState* modState = (PyModuleState*) PyModule_GetState(m);
    if(!modState) return -1;
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
        PyObjectScopedRef modName = PyObject_GetAttrString(obj, "__module__");
        if(!modName) {
            PyErr_Clear();
            return false;
        }

        const char* modNameStr = PyUnicode_AsUTF8(modName);
        if(!modNameStr) {
            PyErr_Clear();
            return false;
        }

        if(memcmp(modNameStr, "torch", 5) != 0 || (modNameStr[5] != '\0' && modNameStr[5] != '.'))
            return false;
    }

    if(!_torchTensorInit()) {
        PyErr_Clear();
        return false;
    }
    return true;
}

bool PyModuleState::_torchTensorInit() {
    PyObjectScopedRef mod = PyImport_ImportModule("torch");
    if(!mod) return false;
    _torchTensorType = PyObject_GetAttrString(mod, "Tensor");
    if(!_torchTensorType)
        return false;
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
    PyObjectScopedRef mod = PyImport_ImportModule("torch");
    if(!mod) return false;

    if(!_torchTensorType) {
        _torchTensorType = PyObject_GetAttrString(mod, "Tensor");
        if(!_torchTensorType) return false;
    }

    PyObjectScopedRef modAlternatives = PyImport_ImportModule("returnn.torch.frontend.raw_ops");
    if(!modAlternatives) return false;

    // init all RawOp's

    #define AddOp(op, name) ops[op] = PyObject_GetAttrString(mod, name); if(!ops[op]) return false;
    #define AddOpAlt(op, name) ops[op] = PyObject_GetAttrString(modAlternatives, name); if(!ops[op]) return false;
    #define AddOpNative(op, name) ops[op] = PyObject_GetAttrString(_module, name); if(!ops[op]) return false;

    AddOp(TOp_ConvertToTensor, "tensor");
    AddOpNative(TOp_ConvertToTensorLike, "convert_to_raw_torch_tensor_like");
    AddOp(TOp_Permute, "permute");
    AddOp(TOp_Reshape, "reshape");

    {
        PyObjectScopedRef shapeAttr = PyObject_GetAttrString(_torchTensorType, "shape");
        if(!shapeAttr) return false;
        ops[TOp_GetShape] = PyObject_GetAttrString(shapeAttr, "__get__");
        if(!ops[TOp_GetShape]) return false;
    }

    AddOpNative(TOp_GetDType, "raw_torch_tensor_get_dtype");
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
    // Use clamp_min/clamp_max instead of maximum/minimum because the former allow number arguments.
    AddOp(TOp_Maximum, "clamp_min");
    AddOp(TOp_Minimum, "clamp_max");
    AddOpAlt(TOp_SquaredDifference, "squared_difference");
    AddOp(TOp_LogAddExp, "logaddexp");
    AddOp(TOp_And, "logical_and");
    AddOp(TOp_Or, "logical_or");
    AddOp(TOp_Neg, "neg");
    AddOp(TOp_Not, "logical_not");
    AddOp(TOp_Abs, "abs");
    AddOp(TOp_Ceil, "ceil");
    AddOp(TOp_Floor, "floor");

    #undef AddOp
    #undef AddOpAlt
    #undef AddOpNative

    return true;
}

const char* rawOpName(RawOp op) {
    static const char* names[NumTOps] = {NULL};
    if(!names[0]) {
        names[TOp_ConvertToTensor] = "convert_to_tensor";
        names[TOp_ConvertToTensorLike] = "convert_to_tensor_like";
        names[TOp_Permute] = "permute";
        names[TOp_Reshape] = "reshape";
        names[TOp_GetShape] = "get_shape";
        names[TOp_GetDType] = "get_dtype";
        names[TOp_Eq] = "equal";
        names[TOp_Ne] = "not_equal";
        names[TOp_Lt] = "less";
        names[TOp_Le] = "less_equal";
        names[TOp_Gt] = "greater";
        names[TOp_Ge] = "greater_equal";
        names[TOp_Add] = "add";
        names[TOp_Sub] = "sub";
        names[TOp_Mul] = "mul";
        names[TOp_TrueDiv] = "truediv";
        names[TOp_FloorDiv] = "floordiv";
        names[TOp_Mod] = "mod";
        names[TOp_Pow] = "pow";
        names[TOp_Maximum] = "maximum";
        names[TOp_Minimum] = "minimum";
        names[TOp_SquaredDifference] = "squared_difference";
        names[TOp_LogAddExp] = "logaddexp";
        names[TOp_And] = "logical_and";
        names[TOp_Or] = "logical_or";
        // The names for the unary funcs matter:
        // This will be used for the fallback implementation
        // either to call backend.<name> or backend.activation(<name>).
        names[TOp_Neg] = "neg";
        names[TOp_Not] = "logical_not";
        names[TOp_Abs] = "abs";
        names[TOp_Ceil] = "ceil";
        names[TOp_Floor] = "floor";
    }
    if(!names[op]) {
        PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: invalid RawOp '%d'", op);
        return NULL;
    }
    return names[op];
}

bool PyModuleState::_torchTensorDTypesInit() {
    PyObjectScopedRef mod = PyImport_ImportModule("torch");
    if(!mod) return false;

    unsigned int i = 0;
    #define AddDType(dtype_) \
        assert(i < sizeof(_torchTensorDTypes)/sizeof(_torchTensorDTypes[0])); \
        _torchTensorDTypes[i].dtype = PyObject_GetAttrString(mod, #dtype_); \
        if(!_torchTensorDTypes[i].dtype) return false; \
        _torchTensorDTypes[i].name = PyUnicode_InternFromString(#dtype_); \
        if(!_torchTensorDTypes[i].name) return false; \
        ++i;

    AddDType(float32);
    AddDType(int32);
    AddDType(int64);
    AddDType(float16);
    AddDType(bool);

    #undef AddDType
    return true;
}
