
#include <Python.h>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "tensor_ops.hpp"
#include "module.hpp"
#include "backend.hpp"
#include "py_utils.hpp"

// copy of Tensor.copy_template()
PyObject* tensorCopyTemplate(
    PyModuleState* modState,
    PyObject* tensor,
    const char* name = NULL,
    const char* dtype = NULL)
{
    PyObjectScopedRef version = PyObject_GetAttrString(tensor, "version");
    if(!version)
        return NULL;

    // TODO ...
    PyErr_Format(PyExc_NotImplementedError, "tensorCopyTemplate: not implemented yet");
    return NULL;
}

// just copies name, dims, dtype, feature_dim, sparse_dim. no or other things.
// this is like what bin_op_out_template is doing.
PyObject* tensorCopyTemplateSimple(
    PyModuleState* modState,
    PyObject* tensor,
    const char* name_ = NULL,
    const char* dtype_ = NULL)
{
    PyObjectScopedRef name = name_ ? PyUnicode_FromString(name_) : PyObject_GetAttrString(tensor, "name");
    if(!name) return NULL;
    PyObjectScopedRef dtype = dtype_ ? PyUnicode_FromString(dtype_) : PyObject_GetAttrString(tensor, "dtype");
    if(!dtype) return NULL;
    PyObjectScopedRef dims = PyObject_GetAttrString(tensor, "_dims");
    if(!dims) return NULL;
    PyObjectScopedRef feature_dim_axis = PyObject_GetAttrString(tensor, "_feature_dim_axis");
    if(!feature_dim_axis) return NULL;
    PyObjectScopedRef sparse_dim = PyObject_GetAttrString(tensor, "sparse_dim");
    if(!sparse_dim) return NULL;

    PyObjectScopedRef res = PyObject_CallFunctionObjArgs(
        modState->tensorType(), name.get(), dims.get(), dtype.get(), NULL);
    if(!res) return NULL;

    if(feature_dim_axis != Py_None)
        if(PyObject_SetAttrString(res, "_feature_dim_axis", feature_dim_axis) < 0)
            return NULL;
    if(sparse_dim != Py_None)
        if(PyObject_SetAttrString(res, "sparse_dim", sparse_dim) < 0)
            return NULL;
    return res.release();
}

// no error check here; false does not mean they are different, it just checks for `is`
static bool _isSameTupleFast(PyObject* a, PyObject* b) {
    if(a == b)
        return true;
    int size = PyTuple_GET_SIZE(a);
    if(size < 0)
        return false;
    if(size != PyTuple_GET_SIZE(b))
        return false;
    for(int i = 0; i < size; ++i) {
        PyObject* a_ = PyTuple_GET_ITEM(a, i);
        PyObject* b_ = PyTuple_GET_ITEM(b, i);
        if(a_ != b_)
            return false;
    }
    return true;
}

// no error check here for aTuple; false does not mean they are different, it just checks for `is`
static bool _isSameTupleAndSeqFast(PyObject* aTuple, PyObject* bSeq) {
    if(PyTuple_Check(bSeq))
        return _isSameTupleFast(aTuple, bSeq);
    int size = PyTuple_GET_SIZE(aTuple);
    if(size < 0)
        return false;
    int bSize = PyObject_Length(bSeq);
    if(bSize < 0) {
        PyErr_Clear();
        return false;
    }
    if(size != bSize)
        return false;
    for(int i = 0; i < size; ++i) {
        PyObject* a_ = PyTuple_GET_ITEM(aTuple, i);
        PyObjectScopedRef iInt = PyLong_FromLong(i);
        if(!iInt) {
            PyErr_Clear();
            return false;
        }
        PyObjectScopedRef b_ = PyObject_GetItem(bSeq, iInt);
        if(!b_) {
            PyErr_Clear();
            return false;
        }
        if(a_ != b_)
            return false;
    }
    return true;
}

// no error check here; false does not mean they are different, it just checks for `is`.
// when it returns with false, outPermutation is undefined.
static bool _isTupleSubsetFast(PyObject* subset, PyObject* superset, std::vector<int>& outPermutation) {
    int superSize = PyTuple_GET_SIZE(superset);
    if(superSize < 0) return false;
    int subSize = PyTuple_GET_SIZE(subset);
    if(subSize < 0) return false;
    if(subSize > superSize)
        return false;
    int j = 0;
    for(int i = 0; i < subSize; ++i) {
        PyObject* a_ = PyTuple_GET_ITEM(subset, i);
        while(true) {
            if(j >= superSize)
                return false;
            PyObject* b_ = PyTuple_GET_ITEM(superset, j);
            if(a_ == b_) break;
            ++j; outPermutation.push_back(-1);
        }
        ++j; outPermutation.push_back(i);
    }
    for(; j < superSize; ++j)
        outPermutation.push_back(-1);
    return true;
}

PyObject* pyTensorCopyTemplate(PyObject *self, PyObject *args, PyObject *kwargs) {
    static const char *kwlist[] = { "tensor", "name", "dtype", NULL };
    PyObject* tensor;
    const char* name = NULL;
    const char* dtype = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$ss:tensor_copy_template",
            (char**) kwlist, &tensor, &name, &dtype))
        return NULL;

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;

    return tensorCopyTemplate(modState, tensor, name, dtype);
}

static bool _isMatchingDType(PyObject* dtype, PyObject* rawDtype, const char* funcName) {
    if(!PyUnicode_Check(dtype)) {
        PyErr_Format(
            PyExc_TypeError,
            "%s: tensor.dtype did not return a string, from dtype %R", funcName, dtype);
        return false;
    }
    if(!PyUnicode_Check(rawDtype)) {
        PyErr_Format(
            PyExc_TypeError,
            "%s: raw_tensor.dtype did not return a string, from dtype %R", funcName, rawDtype);
        return false;
    }
    if(PyUnicode_Compare(dtype, rawDtype) != 0) {
        PyErr_Format(
            PyExc_ValueError,
            "%s: tensor.dtype != raw_tensor.dtype, from tensor dtype %R and raw_tensor dtype %R",
            funcName, dtype, rawDtype);
        return false;
    }
    return true;
}

static bool _isMatchingDimTagsAndRawShape(PyObject* dimTags, PyObject* rawShape, const char* funcName) {
    if(!PyTuple_Check(dimTags)) {
        PyErr_Format(PyExc_TypeError, "%s: expected tensor.dims to be tuple, got %R", funcName, dimTags);
        return false;
    }
    if(!PyTuple_Check(rawShape)) {
        PyErr_Format(PyExc_TypeError, "%s: expected raw_tensor.shape to be tuple, got %R", funcName, rawShape);
        return false;
    }

    int ndim = PyTuple_GET_SIZE(dimTags);
    if(ndim < 0 || ndim != PyTuple_GET_SIZE(rawShape)) {
        PyErr_Format(
            PyExc_ValueError,
            "%s: tensor ndim != raw_tensor ndim, from tensor dims %R and raw_tensor shape %R",
            funcName, dimTags, rawShape);
        return false;
    }
    for(int i = 0; i < ndim; ++i) {
        PyObject* dimTag = PyTuple_GET_ITEM(dimTags, i);
        PyObject* rawDim = PyTuple_GET_ITEM(rawShape, i);
        PyObjectScopedRef dim = PyObject_GetAttrString(dimTag, "size");
        if(!dim) return false;
        if(dim == Py_None) continue; // we allow anything in the raw_tensor dim
        long dimInt = PyLong_AsLong(dim);
        if(dimInt < 0) {
            if(!PyErr_Occurred())
                PyErr_Format(
                    PyExc_ValueError,
                    "%s: tensor dim is negative, from tensor dims %R and raw_tensor shape %R",
                    funcName, dimTags, rawShape);
            return false;
        }
        long rawDimInt = PyLong_AsLong(rawDim);
        if(rawDimInt < 0) {
            if(!PyErr_Occurred())
                PyErr_Format(
                    PyExc_ValueError,
                    "%s: raw_tensor dim is negative, from tensor dims %R and raw_tensor shape %R",
                    funcName, dimTags, rawShape);
            return false;
        }
        if(dimInt != rawDimInt) {
            PyErr_Format(
                PyExc_ValueError,
                "%s: tensor dim != raw_tensor dim, from tensor dims %R and raw_tensor shape %R",
                funcName, dimTags, rawShape);
            return false;
        }
    }
    return true;
}

static bool _checkTensorRawTensorAssignForBackendWithCachedOps(
    PyModuleState* modState, BackendWithCachedOps backendId, const char* funcName, PyObject* tensor, PyObject* rawTensor
) {
    {
        PyObject* getDTypeOp = modState->cachedOp(TOp_GetDType, backendId);
        if(!getDTypeOp) return false;
        PyObjectScopedRef dtype = PyObject_GetAttrString(tensor, "dtype");
        if(!dtype) return false;
        PyObjectScopedRef rawDtype = PyObject_CallFunctionObjArgs(getDTypeOp, rawTensor, NULL);
        if(!rawDtype) return false;
        if(!_isMatchingDType(dtype, rawDtype, funcName))
            return false;
    }
    {
        PyObject* getShapeOp = modState->cachedOp(TOp_GetShape, backendId);
        if(!getShapeOp) return false;
        PyObjectScopedRef dims = PyObject_GetAttrString(tensor, "_dims");
        if(!dims) return false;
        PyObjectScopedRef rawShape = PyObject_CallFunctionObjArgs(getShapeOp, rawTensor, NULL);
        if(!rawShape) return false;
        if(!_isMatchingDimTagsAndRawShape(dims, rawShape, funcName))
            return false;
    }
    return true;
}

PyObject* pyTensorRawTensorSetter(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "tensor_raw_tensor_setter() takes exactly 2 arguments: tensor, raw_tensor");
        return NULL;
    }
    PyObject* tensor = args[0];
    PyObject* raw_tensor = args[1];

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;

    // Do sanity check for dims and dtype.
    bool haveBackendWithCachedOps = false;
    BackendWithCachedOps backendId;
    if(modState->isTorchTensorType((PyObject*) Py_TYPE(raw_tensor))) {
        haveBackendWithCachedOps = true;
        backendId = BWCO_Torch;
    }

    if(haveBackendWithCachedOps) {
        if(!_checkTensorRawTensorAssignForBackendWithCachedOps(modState, backendId, "tensor_raw_tensor_setter", tensor, raw_tensor))
            return NULL;
    }
    else if(raw_tensor == Py_None) {}  // nothing to check
    else {
        PyObject* backend = getBackendForRawTensor(modState, raw_tensor);
        if(!backend) return NULL;
        {
            PyObjectScopedRef dtype = PyObject_GetAttrString(tensor, "dtype");
            if(!dtype) return NULL;
            PyObjectScopedRef rawDtype = PyObject_CallMethod(backend, "get_dtype_name_raw", "O", raw_tensor);
            if(!rawDtype) return NULL;
            if(!_isMatchingDType(dtype, rawDtype, "tensor_raw_tensor_setter"))
                return NULL;
        }
        {
            PyObjectScopedRef dims = PyObject_GetAttrString(tensor, "_dims");
            if(!dims) return NULL;
            PyObjectScopedRef rawShape = PyObject_CallMethod(backend, "get_known_shape_raw", "O", raw_tensor);
            if(!rawShape) return NULL;
            if(!_isMatchingDimTagsAndRawShape(dims, rawShape, "tensor_raw_tensor_setter"))
                return NULL;
        }
    }

    if(PyObject_SetAttrString(tensor, "_raw_tensor", raw_tensor) < 0)
        return NULL;
    Py_RETURN_NONE;
}

PyObject* pyConvertToRawTorchTensorLike(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "convert_to_raw_torch_tensor_like() takes exactly 2 args: value, other_tensor");
        return NULL;
    }
    PyObject* value = args[0];
    PyObject* other_tensor = args[1];

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;

    PyObject* convertOp = modState->cachedOp(TOp_ConvertToTensor, BWCO_Torch);
    if(!convertOp) return NULL;

    PyObjectScopedRef args_ = PyTuple_Pack(1, value);
    PyObjectScopedRef dtype = PyObject_GetAttrString(other_tensor, "dtype");
    if(!dtype) return NULL;
    PyObjectScopedRef device = PyObject_GetAttrString(other_tensor, "device");
    if(!device) return NULL;
    PyObjectScopedRef kwargs = PyDict_New();
    if(!kwargs) return NULL;
    if(PyDict_SetItemString(kwargs, "dtype", dtype) < 0) return NULL;
    if(PyDict_SetItemString(kwargs, "device", device) < 0) return NULL;
    return PyObject_Call(convertOp, args_, kwargs);
}

static PyObject* compareOrCombine(
    PyObject* a, PyObject* b,
    bool resultIsBool,
    PyModuleState* modState,
    const char* rawOpName,
    PyObject* rawOp, PyObject* permuteOp, PyObject* reshapeOp, PyObject* getShapeOp, PyObject* convertToTensorLikeOp,
    bool needConvertToTensor,
    bool allowBroadcastAllSources,
    PyObject* dimOrder
) {
    if(!rawOp || !permuteOp || !reshapeOp || !getShapeOp || !convertToTensorLikeOp) return NULL;

    {
        int a_is_tensor = PyObject_IsInstance(a, modState->tensorType());
        if(a_is_tensor < 0)
            return NULL;
        int b_is_tensor = PyObject_IsInstance(b, modState->tensorType());
        if(b_is_tensor < 0)
            return NULL;
        if((a_is_tensor && !b_is_tensor) || (!a_is_tensor && b_is_tensor)) {
            // assume the non-Tensor obj is is scalar
            PyObjectScopedRef res = tensorCopyTemplateSimple(modState, a_is_tensor ? a : b, rawOpName, resultIsBool ? "bool" : NULL);
            if(!res) return NULL;
            PyObjectScopedRef aRawTensor, bRawTensor;
            if(a_is_tensor) {
                assert(a_is_tensor && !b_is_tensor);
                aRawTensor = PyObject_GetAttrString(a, "_raw_tensor");
                if(!aRawTensor) return NULL;
                if(needConvertToTensor) {
                    bRawTensor = PyObject_CallFunctionObjArgs(convertToTensorLikeOp, b, aRawTensor.get(), NULL);
                    if(!bRawTensor) return NULL;
                }
            }
            else {
                assert(!a_is_tensor && b_is_tensor);
                bRawTensor = PyObject_GetAttrString(b, "_raw_tensor");
                if(!bRawTensor) return NULL;
                if(needConvertToTensor) {
                    aRawTensor = PyObject_CallFunctionObjArgs(convertToTensorLikeOp, a, bRawTensor.get(), NULL);
                    if(!aRawTensor) return NULL;
                }
            }
            PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(
                rawOp, aRawTensor ? aRawTensor.get() : a, bRawTensor.get() ? bRawTensor.get() : b, NULL);
            if(!resRawTensor) return NULL;
            if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
            return res.release();
        }
        if(!a_is_tensor && !b_is_tensor) {
            PyErr_Format(PyExc_TypeError, "compareOrCombine: expected at least one Tensor, got %R and %R", a, b);
            return NULL;
        }
        // both are Tensor
    }

    PyObjectScopedRef aDims = PyObject_GetAttrString(a, "_dims");
    if(!aDims) return NULL;
    if(!PyTuple_Check(aDims)) {
        PyErr_Format(PyExc_TypeError, "compareOrCombine: expected a.dims to be tuple, got %R", aDims.get());
        return NULL;
    }
    PyObjectScopedRef bDims = PyObject_GetAttrString(b, "_dims");
    if(!bDims) return NULL;
    if(!PyTuple_Check(bDims)) {
        PyErr_Format(PyExc_TypeError, "compareOrCombine: expected b.dims to be tuple, got %R", bDims.get());
        return NULL;
    }

    PyObjectScopedRef aRawTensor = PyObject_GetAttrString(a, "_raw_tensor");
    if(!aRawTensor) return NULL;
    PyObjectScopedRef bRawTensor = PyObject_GetAttrString(b, "_raw_tensor");
    if(!bRawTensor) return NULL;

    // fast path: just use `is` checks for dims.
    // allowBroadcastAllSources=false makes it easier..., we know one dims should be the superset.
    // dimOrder makes it more difficult, does not need to be a superset. but by default, we don't have that.

    // first very fast path check, check exact identity of dims
    if(_isSameTupleFast(aDims, bDims) && (dimOrder == Py_None || _isSameTupleAndSeqFast(aDims, dimOrder))) {
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, a, rawOpName, resultIsBool ? "bool" : NULL);
        if(!res) return NULL;
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensor.get(), bRawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
        return res.release();
    }

    // check b is scalar
    if(PyTuple_GET_SIZE(bDims.get()) == 0 && (dimOrder == Py_None || _isSameTupleAndSeqFast(aDims, dimOrder))) {
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, a, rawOpName, resultIsBool ? "bool" : NULL);
        if(!res) return NULL;
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensor.get(), bRawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
        return res.release();
    }

    // check a is scalar
    if(PyTuple_GET_SIZE(aDims.get()) == 0 && (dimOrder == Py_None || _isSameTupleAndSeqFast(bDims, dimOrder))) {
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, b, rawOpName, resultIsBool ? "bool" : NULL);
        if(!res) return NULL;
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensor.get(), bRawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
        return res.release();
    }

    PyObjectScopedRef aRawShape = PyObject_CallFunctionObjArgs(getShapeOp, aRawTensor.get(), NULL);
    if(!aRawShape) return NULL;
    if(!PyTuple_Check(aRawShape)) {
        PyErr_Format(PyExc_TypeError, "compareOrCombine: expected a.raw_tensor.shape to be tuple, got %R", aRawShape.get());
        return NULL;
    }

    // check if bDims is a subset of aDims, in the same order
    {
        std::vector<int> outPermutation;
        if(_isTupleSubsetFast(bDims, aDims, outPermutation) && (dimOrder == Py_None || _isSameTupleAndSeqFast(aDims, dimOrder))) {
            PyObjectScopedRef bRawShapeExt = PyTuple_New(outPermutation.size());
            if(!bRawShapeExt) return NULL;
            for(int i = 0; i < (int) outPermutation.size(); ++i) {
                PyObject* d;
                if(outPermutation[i] >= 0) {
                    d = PyTuple_GET_ITEM(aRawShape.get(), i);
                    Py_XINCREF(d);
                }
                else
                    d = PyLong_FromLong(1);
                if(!d) return NULL;
                PyTuple_SET_ITEM(bRawShapeExt.get(), i, d);
            }
            PyObjectScopedRef bRawTensorExt = PyObject_CallFunctionObjArgs(reshapeOp, bRawTensor.get(), bRawShapeExt.get(), NULL);
            if(!bRawTensorExt) return NULL;
            PyObjectScopedRef res = tensorCopyTemplateSimple(modState, a, rawOpName, resultIsBool ? "bool": NULL);
            if(!res) return NULL;
            PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensor.get(), bRawTensorExt.get(), NULL);
            if(!resRawTensor) return NULL;
            if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
            return res.release();
        }
    }

    PyObjectScopedRef bRawShape = PyObject_CallFunctionObjArgs(getShapeOp, bRawTensor.get(), NULL);
    if(!bRawShape) return NULL;
    if(!PyTuple_Check(bRawShape)) {
        PyErr_Format(PyExc_TypeError, "compareOrCombine: expected b.raw_tensor.shape to be tuple, got %R", bRawShape.get());
        return NULL;
    }

    // check if aDims is a subset of bDims, in the same order
    {
        std::vector<int> outPermutation;
        if(_isTupleSubsetFast(aDims, bDims, outPermutation) && (dimOrder == Py_None || _isSameTupleAndSeqFast(bDims, dimOrder))) {
            PyObjectScopedRef aRawShapeExt = PyTuple_New(outPermutation.size());
            if(!aRawShapeExt) return NULL;
            for(int i = 0; i < (int) outPermutation.size(); ++i) {
                PyObject* d;
                if(outPermutation[i] >= 0) {
                    d = PyTuple_GET_ITEM(bRawShape.get(), i);
                    Py_XINCREF(d);
                }
                else
                    d = PyLong_FromLong(1);
                if(!d) return NULL;
                PyTuple_SET_ITEM(aRawShapeExt.get(), i, d);
            }
            PyObjectScopedRef aRawTensorExt = PyObject_CallFunctionObjArgs(reshapeOp, aRawTensor.get(), aRawShapeExt.get(), NULL);
            if(!aRawTensorExt) return NULL;
            PyObjectScopedRef res = tensorCopyTemplateSimple(modState, b, rawOpName, resultIsBool ? "bool": NULL);
            if(!res) return NULL;
            PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensorExt.get(), bRawTensor.get(), NULL);
            if(!resRawTensor) return NULL;
            if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
            return res.release();
        }
    }

    // TODO ...
    PyErr_Format(PyExc_NotImplementedError, "compareOrCombine: not implemented yet");
    return NULL;
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
    bool needConvertToTensor = true;
    if(backendId == BWCO_Torch) {
        switch(rawOp) {
        case TOp_Add:
        case TOp_Sub:
        case TOp_Mul:
        case TOp_TrueDiv:
        case TOp_FloorDiv:
        case TOp_Mod:
        case TOp_Pow:
            needConvertToTensor = false;
        default:
            break;
        }
    }
    return compareOrCombine(
        a, b,
        resultIsBool,
        modState,
        rawOpName(rawOp),
        modState->cachedOp(rawOp, backendId),
        modState->cachedOp(TOp_Permute, backendId),
        modState->cachedOp(TOp_Reshape, backendId),
        modState->cachedOp(TOp_GetShape, backendId),
        modState->cachedOp(TOp_ConvertToTensorLike, backendId),
        needConvertToTensor,
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

    PyObjectScopedRef func = PyObject_GetAttrString(backend, "compare");
    if(!func) return NULL;
    return PyObject_Call(func, args, kwargs);
}

PyObject* pyTensorCombine(PyObject *self, PyObject *args, PyObject *kwargs) {
    // TODO ...
    PyErr_Format(PyExc_NotImplementedError, "tensor_combine: not implemented yet");
    return NULL;
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

    PyObject* a = args[0];
    PyObject* b = args[1];

    if(PyObject_IsInstance(a, modState->tensorType())) {}
    else if(PyObject_IsInstance(b, modState->tensorType())) {
        std::swap(a, b);
    }
    else {
        PyErr_Format(PyExc_TypeError, "tensor_eq: expected at least one Tensor, got %R and %R", a, b);
        return NULL;
    }
    // a is Tensor here, b could be anything

    {
        PyObjectScopedRef rawTensor = PyObject_GetAttrString(a, "_raw_tensor");
        if(!rawTensor) return NULL;
        if(rawTensor == Py_None) {
            // The other op overloads would actually raise some exception in this case.
            // However, here just return False.
            Py_INCREF(Py_False);
            return Py_False;
        }
    }

    bool isValidType = false;
    if(PyObject_IsInstance(b, modState->tensorType())) {
        isValidType = true;
    }
    else {
        for(int i = 0; i < modState->rawTensorTypesLen(); ++i) {
            if(PyObject_IsInstance(b, modState->rawTensorType(i))) {
                isValidType = true;
                break;
            }
        }
        if(!isValidType) {
            PyObject* backend = getBackendForTensor(modState, a);
            if(!backend) return NULL;
            PyObjectScopedRef backendRawTensorType = PyObject_GetAttrString(backend, "RawTensorType");
            if(!backendRawTensorType) return NULL;
            if(PyObject_IsInstance(b, backendRawTensorType))
                isValidType = true;
        }
    }

    if(isValidType)
        // default case
        return _tensorCompareOrCombine<TOp_Eq, true>(modState, a, b);

    Py_INCREF(Py_False);
    return Py_False;
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
DefinePyTensorCombine(And)
DefinePyTensorCombineR(And)
DefinePyTensorCombine(Or)
DefinePyTensorCombineR(Or)

template<RawOp op, bool genericUseActFunc>
static PyObject* _tensorUnaryFunc(PyModuleState* modState, PyObject* tensor) {
    PyObjectScopedRef rawTensor = PyObject_GetAttrString(tensor, "_raw_tensor");
    if(!rawTensor) return NULL;

    // fast path -- check predefined backends where we have cached ops
    if(modState->isTorchTensorType((PyObject*) Py_TYPE(rawTensor))) {
        PyObject* func = modState->cachedOp(op, BWCO_Torch);
        if(!func) return NULL;
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, tensor, rawOpName(op), NULL);
        if(!res) return NULL;
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(func, rawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        if(!_checkTensorRawTensorAssignForBackendWithCachedOps(modState, BWCO_Torch, rawOpName(op), res, resRawTensor))
            return NULL;
        if(PyObject_SetAttrString(res, "_raw_tensor", resRawTensor) < 0) return NULL;
        return res.release();
    }

    // generic fallback
    PyObject* backend = getBackendForRawTensor(modState, rawTensor);
    if(genericUseActFunc) {
        PyObjectScopedRef actFunc = PyObject_GetAttrString(backend, "activation");
        if(!actFunc) return NULL;
        PyObjectScopedRef opName = PyUnicode_FromString(rawOpName(op));
        return PyObject_CallFunctionObjArgs(actFunc.get(), tensor, opName.get(), NULL);
    }
    else {
        PyObjectScopedRef func = PyObject_GetAttrString(backend, rawOpName(op));
        if(!func) return NULL;
        return PyObject_CallFunctionObjArgs(func, tensor, NULL);
    }
}

template<RawOp op, bool genericUseActFunc>
static PyObject* _pyTensorUnaryFunc(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 1) {
        PyErr_Format(PyExc_TypeError, "tensor_%s: expected 1 arg, got %i", rawOpName(op), nargs);
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;
    return _tensorUnaryFunc<op, genericUseActFunc>(modState, args[0]);
}

#define DefinePyTensorUnaryFunc(op, genericUseActFunc) \
    PyObject* pyTensor##op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) { \
        return _pyTensorUnaryFunc<TOp_##op, genericUseActFunc>(self, args, nargs); \
    }

DefinePyTensorUnaryFunc(Neg, true)
DefinePyTensorUnaryFunc(Not, true)
DefinePyTensorUnaryFunc(Abs, true)
DefinePyTensorUnaryFunc(Ceil, true)
DefinePyTensorUnaryFunc(Floor, true)

