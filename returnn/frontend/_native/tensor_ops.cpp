
#include <Python.h>
#include <map>
#include <string>
#include "tensor_ops.hpp"
#include "module.hpp"

template<bool resultIsBool, typename TRawOp, typename TPermuteOp, typename TReshapeOp>
PyObject* compareOrCombine(
    PyObject* a, PyObject* b,
    TRawOp rawOp, TPermuteOp permuteOp, TReshapeOp reshapeOp,
    bool allowBroadcastAllSources,
    PyObject* dimOrder
) {

}

struct BinCachedOp {
    PyObject* func;
    BinCachedOp(PyObject* func_) : func(func_) {}
    PyObject* operator()(PyObject* a, PyObject* b) const {
        return PyObject_CallFunction(func, "OO", a, b);
    }
};

PyObject* pyCompare(PyObject *self, PyObject *args, PyObject *kwargs) {
    static const char *kwlist[] = { "a", "kind", "b", "allow_broadcast_all_sources", "dim_order", NULL };
    PyObject* a;
    char* kind;
    PyObject* b;
    unsigned char allow_broadcast_all_sources = false;
    PyObject* dim_order = Py_None;
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OsO|$bO:compare", (char**) kwlist,
            &a, &kind, &b, &allow_broadcast_all_sources, &dim_order))
        return NULL;

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;

    bool haveBackendWithCachedOps = false;
    BackendWithCachedOps backendId;
    if(isTorchBackendForTensor(modState, a) || isTorchBackendForTensor(modState, b)) {
        haveBackendWithCachedOps = true;
        backendId = BWCO_Torch;
    }

    if(haveBackendWithCachedOps) {
        // "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=", "not_equal"|"!="
        static std::map<std::string, RawOp> kindToCompareFunc;
        if(kindToCompareFunc.empty()) {
            kindToCompareFunc["=="] = TOp_Eq;
            kindToCompareFunc["equal"] = TOp_Eq;
            kindToCompareFunc["!="] = TOp_Ne;
            kindToCompareFunc["not_equal"] = TOp_Ne;
            kindToCompareFunc["<"] = TOp_Lt;
            kindToCompareFunc["less"] = TOp_Lt;
            kindToCompareFunc["<="] = TOp_Le;
            kindToCompareFunc["less_equal"] = TOp_Le;
            kindToCompareFunc[">"] = TOp_Gt;
            kindToCompareFunc["greater"] = TOp_Gt;
            kindToCompareFunc[">="] = TOp_Ge;
            kindToCompareFunc["greater_equal"] = TOp_Ge;
        }

        auto it = kindToCompareFunc.find(kind);
        if(it == kindToCompareFunc.end()) {
            PyErr_Format(PyExc_ValueError, "compare: invalid kind '%s'", kind);
            return NULL;
        }

        BinCachedOp op(modState->cachedOp(it->second, backendId));
        BinCachedOp permuteOp(modState->cachedOp(TOp_Permute, backendId));
        BinCachedOp reshapeOp(modState->cachedOp(TOp_Reshape, backendId));

        return compareOrCombine<true>(
            a, b, op, permuteOp, reshapeOp, (bool) allow_broadcast_all_sources, dim_order);
    }

    PyObject* backend;
    if(PyObject_IsInstance(a, modState->tensorType())) {
        backend = getBackendForTensor(modState, a);
        if(!backend)
            return NULL;
    }
    else if(PyObject_IsInstance(b, modState->tensorType())) {
        backend = getBackendForTensor(modState, b);
        if(!backend)
            return NULL;
    }
    else
        backend = modState->globalBackend();

    PyObject* func = PyObject_GetAttrString(backend, "compare");
    if(!func)
        return NULL;
    PyObject* res = PyObject_Call(func, args, kwargs);
    Py_DECREF(func);
    return res;
}

PyObject* pyCombine(PyObject *self, PyObject *args, PyObject *kwargs) {

}

PyObject* Tensor_eq() {
    /* Special implementation for eq:
    When comparing to some other invalid type, return False, not a Tensor.
    This is to allow easy equality checks with other random objects.
    See for example here: https://github.com/rwth-i6/returnn/pull/1284
    */
}

template<char... chars>
PyObject* Tensor_compare() {

}

template<char... chars>
PyObject* Tensor_combine() {

}
