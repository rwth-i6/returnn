
#include <Python.h>
#include <map>
#include <string>
#include "tensor_ops.hpp"
#include "module.hpp"

static PyObject* compareOrCombine(
    PyObject* a, PyObject* b,
    bool resultIsBool,
    PyObject* rawOp, PyObject* permuteOp, PyObject* reshapeOp,
    bool allowBroadcastAllSources,
    PyObject* dimOrder
) {

}

static PyObject* compareOrCombineViaCached(
    PyObject* a, PyObject* b,
    bool resultIsBool,
    PyModuleState* modState,
    BackendWithCachedOps backendId,
    RawOp rawOp,
    bool allowBroadcastAllSources,
    PyObject* dimOrder
) {
    return compareOrCombine(
        a, b,
        resultIsBool,
        modState->cachedOp(rawOp, backendId),
        modState->cachedOp(TOp_Permute, backendId),
        modState->cachedOp(TOp_Reshape, backendId),
        allowBroadcastAllSources,
        dimOrder);
}

PyObject* pyTensorCompare(PyObject *self, PyObject *args, PyObject *kwargs) {
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

        return compareOrCombineViaCached(
            a, b,
            true,
            modState, backendId, it->second,
            (bool) allow_broadcast_all_sources, dim_order);
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

PyObject* pyTensorCombine(PyObject *self, PyObject *args, PyObject *kwargs) {

}

template<RawOp op, bool isCompare>
static PyObject* _tensorCompareOrCombine(PyModuleState* modState, PyObject* a, PyObject* b) {
    // fast path -- check predefined backends where we have cached ops
    if(isTorchBackendForTensor(modState, a) || isTorchBackendForTensor(modState, b)) {
        return compareOrCombineViaCached(
            a, b,
            isCompare,
            modState, BWCO_Torch, op,
            false, Py_None);
    }

    // generic fallback
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

    return PyObject_CallMethod(backend, isCompare ? "compare" : "combine", "OsO", a, rawOpName(op), b);
}

template<RawOp op, bool isCompare>
static PyObject* _pyTensorCompareOrCombine(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_Format(PyExc_TypeError, "tensor_%s: expected 2 args, got %i", rawOpName(op), nargs);
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;
    return _tensorCompareOrCombine<op, isCompare>(modState, args[0], args[1]);
}

template<RawOp op, bool isCompare>
static PyObject* _pyTensorCompareOrCombineR(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_Format(PyExc_TypeError, "tensor_r%s: expected 2 args, got %i", rawOpName(op), nargs);
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;
    return _tensorCompareOrCombine<op, isCompare>(modState, args[1], args[0]);
}


PyObject* pyTensorEq(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    /* Special implementation for eq:
    When comparing to some other invalid type, return False, not a Tensor.
    This is to allow easy equality checks with other random objects.
    See for example here: https://github.com/rwth-i6/returnn/pull/1284
    */
    if(nargs != 2) {
        PyErr_Format(PyExc_TypeError, "tensor_eq: expected 2 args, got %i", nargs);
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;

    // TODO ...

    // default case
    return _tensorCompareOrCombine<TOp_Eq, true>(modState, args[0], args[1]);
}

#define DefinePyTensorCompare(op) \
    PyObject* pyTensor##op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) { \
        return _pyTensorCompareOrCombine<TOp_##op, true>(self, args, nargs); \
    }

DefinePyTensorCompare(Ne)
DefinePyTensorCompare(Lt)
DefinePyTensorCompare(Le)
DefinePyTensorCompare(Gt)
DefinePyTensorCompare(Ge)

#define DefinePyTensorCombine(op) \
    PyObject* pyTensor##op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) { \
        return _pyTensorCompareOrCombine<TOp_##op, false>(self, args, nargs); \
    }

#define DefinePyTensorCombineR(op) \
    PyObject* pyTensorR##op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) { \
        return _pyTensorCompareOrCombineR<TOp_##op, false>(self, args, nargs); \
    }

DefinePyTensorCombine(Add)
DefinePyTensorCombineR(Add)
DefinePyTensorCombine(Sub)
DefinePyTensorCombineR(Sub)
DefinePyTensorCombine(Mul)
DefinePyTensorCombineR(Mul)
DefinePyTensorCombine(TrueDiv)
DefinePyTensorCombineR(TrueDiv)
DefinePyTensorCombine(FloorDiv)
DefinePyTensorCombineR(FloorDiv)
DefinePyTensorCombine(Mod)
DefinePyTensorCombineR(Mod)
DefinePyTensorCombine(Pow)
DefinePyTensorCombineR(Pow)
