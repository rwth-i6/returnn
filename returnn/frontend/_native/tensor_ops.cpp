
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
    int bSize = PySequence_Size(bSeq);
    if(bSize < 0) {
        PyErr_Clear();
        return false;
    }
    if(size != bSize)
        return false;
    for(int i = 0; i < size; ++i) {
        PyObject* a_ = PyTuple_GET_ITEM(aTuple, i);
        PyObjectScopedRef b_ = PySequence_GetItem(bSeq, i);
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

// no error check here; false does not mean they are different, it just checks for `is`.
// when it returns with false, outPermutation is undefined.
static bool _isTupleSubsetReorderFast(PyObject* subset, PyObject* superset, std::vector<int>& outPermutation) {
    int superSize = PyTuple_GET_SIZE(superset);
    if(superSize < 0) return false;
    int subSize = PyTuple_GET_SIZE(subset);
    if(subSize < 0) return false;
    if(subSize > superSize)
        return false;
    outPermutation.resize(superSize);
    std::vector<bool> subsetTaken(subSize, false);
    for(int j = 0; j < superSize; ++j) {
        PyObject* b_ = PyTuple_GET_ITEM(superset, j);
        int i = 0;
        for(; i < subSize; ++i) {
            if(subsetTaken[i]) continue;
            PyObject* a_ = PyTuple_GET_ITEM(subset, i);
            if(a_ == b_) break;
        }
        if(i < subSize) {
            subsetTaken[i] = true;
            outPermutation[j] = i;
        }
        else
            outPermutation[j] = -1;
    }
    for(int i = 0; i < subSize; ++i) {
        if(!subsetTaken[i])
            return false;
    }
    return true;
}

// no error check here; false does not mean they are different, it just checks for `is`.
// when it returns with false, outPermutation is undefined.
static bool _isTupleSubsetList(PyObject* subsetTuple, PyObject* supersetList, bool& error) {
    int superSize = PyList_GET_SIZE(supersetList);
    if(superSize < 0) { error = true; return false; }
    int subSize = PyTuple_GET_SIZE(subsetTuple);
    if(subSize < 0) { error = true; return false; }
    if(subSize > superSize)
        return false;
    std::vector<bool> subsetTaken(subSize, false);
    for(int j = 0; j < superSize; ++j) {
        PyObject* b_ = PyList_GET_ITEM(supersetList, j);
        int i = 0;
        for(; i < subSize; ++i) {
            if(subsetTaken[i]) continue;
            PyObject* a_ = PyTuple_GET_ITEM(subsetTuple, i);
            if(a_ == b_) break;
        }
        if(i == subSize) {  // not found, try again using rich compare
            for(; i < subSize; ++i) {
                if(subsetTaken[i]) continue;
                PyObject* a_ = PyTuple_GET_ITEM(subsetTuple, i);
                int eq = PyObject_RichCompareBool(a_, b_, Py_EQ);
                if(eq < 0) { error = true; return false; }
                if(eq) break;
            }
        }
        if(i < subSize)
            subsetTaken[i] = true;
    }
    for(int i = 0; i < subSize; ++i) {
        if(!subsetTaken[i])
            return false;
    }
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

template<bool bIsSubset, bool permutedDims>
static PyObject* _compareOrCombine_subsetDims(
    PyModuleState* modState,
    const char* rawOpName, bool resultIsBool,
    PyObject* permuteOp, PyObject* reshapeOp, PyObject* rawOp,
    PyObject* a, PyObject* b,
    PyObject* aRawTensor, PyObject* bRawTensor,
    PyObject* aRawShape, PyObject* bRawShape,
    PyObject* aDims, PyObject* bDims,
    const std::vector<int>& outPermutation
) {
    // The tensor with the subset dims will be adapted to the other tensor.
    PyObject* rawTensor_ = bIsSubset ? bRawTensor : aRawTensor;
    PyObjectScopedRef rawTensorExt; // just for holding the ref and decrefing it later

    // Maybe permute the tensor with subset dims, to match the order of the other tensor.
    if(permutedDims) {
        PyObjectScopedRef permuteArg = PyTuple_New(PyTuple_GET_SIZE(bIsSubset ? bDims : aDims));
        if(!permuteArg) return NULL;
        int j = 0;
        for(int i = 0; i < (int) outPermutation.size(); ++i) {
            if(outPermutation[i] < 0) continue;
            PyTuple_SET_ITEM(permuteArg.get(), j, PyLong_FromLong(outPermutation[i]));
            ++j;
        }
        assert(j == PyTuple_GET_SIZE(permuteArg.get()));
        rawTensor_ = PyObject_CallFunctionObjArgs(permuteOp, rawTensor_, permuteArg.get(), NULL);
        if(!rawTensor_) return NULL;
        rawTensorExt = rawTensor_;
    }

    // Reshape the tensor with subset dims, to add broadcast dims, to match the dims of the other tensor.
    {
        PyObjectScopedRef rawShapeExt = PyTuple_New(outPermutation.size());
        if(!rawShapeExt) return NULL;
        for(int i = 0; i < (int) outPermutation.size(); ++i) {
            PyObject* d;
            if(outPermutation[i] >= 0) {
                d = PyTuple_GET_ITEM(bIsSubset ? aRawShape : bRawShape, i);
                Py_XINCREF(d);
            }
            else
                d = PyLong_FromLong(1);
            if(!d) return NULL;
            PyTuple_SET_ITEM(rawShapeExt.get(), i, d);
        }
        rawTensor_ = PyObject_CallFunctionObjArgs(reshapeOp, rawTensor_, rawShapeExt.get(), NULL);
        if(!rawTensor_) return NULL;
        rawTensorExt = rawTensor_;
    }

    // Now create the result.
    PyObjectScopedRef res = tensorCopyTemplateSimple(modState, bIsSubset ? a : b, rawOpName, resultIsBool ? "bool" : NULL);
    if(!res) return NULL;
    PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(
        rawOp, bIsSubset ? aRawTensor : rawTensor_, bIsSubset ? rawTensor_ : bRawTensor, NULL);
    if(!resRawTensor) return NULL;
    if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
    return res.release();
}

static PyObject* _permuteAndExtend(
    const char* rawOpName,
    PyObject* permuteOp, PyObject* reshapeOp,
    PyObject* tensor, PyObject* dims, PyObject* rawTensor, PyObject* rawShape,
    PyObject* outDimsList, std::vector<long> outShape
) {
    int tensorNdim = PyTuple_GET_SIZE(dims);
    // First find the mapping.
    std::vector<int> outPermutation;
    {
        int count = 0;
        std::vector<bool> taken(tensorNdim, false);
        for(size_t i = 0; i < outShape.size(); ++i) {
            PyObject* dim = PyList_GET_ITEM(outDimsList, i);
            std::vector<int> candidates;
            for(int j = 0; j < tensorNdim; ++j) {
                if(taken[j]) continue;
                PyObject* dim_ = PyTuple_GET_ITEM(dims, j);
                int eq = PyObject_RichCompareBool(dim, dim_, Py_EQ);
                if(eq < 0) return NULL;
                if(eq) candidates.push_back(j);
            }
            if(candidates.size() == 0)
                outPermutation.push_back(-1);
            else if(candidates.size() == 1) {
                outPermutation.push_back(candidates[0]);
                taken[candidates[0]] = true;
                ++count;
            }
            else if(candidates.size() > 1) {
                size_t maxMatchPriorityIdx;
                long maxMatchPriority;
                int countSameMatchPriority = 0;
                for(size_t j = 0; j < candidates.size(); ++j) {
                    PyObject* dim_ = PyTuple_GET_ITEM(dims, candidates[j]);
                    PyObject* matchPriority = PyObject_GetAttrString(dim_, "match_priority");
                    if(!matchPriority) return NULL;
                    if(!PyLong_Check(matchPriority)) {
                        PyErr_Format(
                            PyExc_TypeError,
                            "%s: dim %R did not return an int for match_priority, from tensor dims %R",
                            rawOpName, dim_, dims);
                        return NULL;
                    }
                    long matchPriority_ = PyLong_AsLong(matchPriority);
                    if(matchPriority_ < 0 && PyErr_Occurred()) return NULL;
                    if(j > 0 && matchPriority_ == maxMatchPriority)
                        ++countSameMatchPriority;
                    if(j == 0 || matchPriority_ > maxMatchPriority) {
                        maxMatchPriority = matchPriority_;
                        countSameMatchPriority = 1;
                        maxMatchPriorityIdx = j;
                    }
                }
                assert(countSameMatchPriority >= 1);
                if(countSameMatchPriority > 1) {
                    PyErr_Format(
                        PyExc_ValueError,
                        "%s: dim %R is ambiguous, from tensor dims %R and raw_tensor shape %R",
                        rawOpName, dim, dims, rawShape);
                    return NULL;
                }
                outPermutation.push_back(candidates[maxMatchPriorityIdx]);
                taken[candidates[maxMatchPriorityIdx]] = true;
                ++count;
            }            
        }
        if(count != tensorNdim) {
            PyErr_Format(
                PyExc_ValueError,
                "%s: not all dims are matched, from tensor dims %R and raw_tensor shape %R",
                rawOpName, dims, rawShape);
            return NULL;
        }
        assert(outPermutation.size() == outShape.size());
    }

    PyObject* rawTensor_ = rawTensor;
    PyObjectScopedRef rawTensorExt; // just for holding the ref and decrefing it later

    // Maybe permute the tensor
    {
        PyObjectScopedRef permuteArg = PyTuple_New(PyTuple_GET_SIZE(rawShape));
        if(!permuteArg) return NULL;
        int j = 0;
        for(int i = 0; i < (int) outPermutation.size(); ++i) {
            if(outPermutation[i] < 0) continue;
            PyTuple_SET_ITEM(permuteArg.get(), j, PyLong_FromLong(outPermutation[i]));
            ++j;
        }
        assert(j == PyTuple_GET_SIZE(permuteArg.get()));
        rawTensor_ = PyObject_CallFunctionObjArgs(permuteOp, rawTensor_, permuteArg.get(), NULL);
        if(!rawTensor_) return NULL;
        rawTensorExt = rawTensor_;
    }

    // Maybe reshape the tensor
    {
        PyObjectScopedRef rawShapeExt = PyTuple_New(outPermutation.size());
        if(!rawShapeExt) return NULL;
        for(int i = 0; i < (int) outPermutation.size(); ++i) {
            PyObject* d = PyLong_FromLong((outPermutation[i] >= 0) ? outShape[i] : 1);
            if(!d) return NULL;
            PyTuple_SET_ITEM(rawShapeExt.get(), i, d);
        }
        rawTensor_ = PyObject_CallFunctionObjArgs(reshapeOp, rawTensor_, rawShapeExt.get(), NULL);
        if(!rawTensor_) return NULL;
        rawTensorExt = rawTensor_;
    }

    rawTensorExt.release();
    return rawTensor_;
}

static PyObject* _consistentFeatureDim(PyObject* a, PyObject* b) {
    PyObjectScopedRef aFeatureDim = PyObject_GetAttrString(a, "feature_dim");
    if(!aFeatureDim) return NULL;
    PyObjectScopedRef bFeatureDim = PyObject_GetAttrString(b, "feature_dim");
    if(!bFeatureDim) return NULL;
    if(bFeatureDim == Py_None)
        return aFeatureDim.release();
    if(aFeatureDim == Py_None)
        return bFeatureDim.release();
    int eq = PyObject_RichCompareBool(aFeatureDim, bFeatureDim, Py_EQ);
    if(eq < 0) return NULL;
    if(eq)
        return aFeatureDim.release();
    Py_RETURN_NONE;
}

static PyObject* _consistentSparseDim(PyObject* a, PyObject* b) {
    PyObjectScopedRef aSparseDim = PyObject_GetAttrString(a, "sparse_dim");
    if(!aSparseDim) return NULL;
    PyObjectScopedRef bSparseDim = PyObject_GetAttrString(b, "sparse_dim");
    if(!bSparseDim) return NULL;
    if(bSparseDim == Py_None)
        return aSparseDim.release();
    if(aSparseDim == Py_None)
        return bSparseDim.release();
    int eq = PyObject_RichCompareBool(aSparseDim, bSparseDim, Py_EQ);
    if(eq < 0) return NULL;
    if(eq)
        return aSparseDim.release();
    Py_RETURN_NONE;
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
            if(dimOrder != Py_None) {
                PyErr_Format(
                    PyExc_TypeError,
                    "compareOrCombine: dimOrder is not supported for scalar and Tensor, got %R and %R", a, b);
                return NULL;
            }
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

    if(!resultIsBool) {
        PyObjectScopedRef aDtype = PyObject_GetAttrString(a, "dtype");
        if(!aDtype) return NULL;
        if(!PyUnicode_Check(aDtype)) {
            PyErr_Format(
                PyExc_TypeError,
                "compareOrCombine: a.dtype did not return a string, from dtype %R", aDtype.get());
            return NULL;
        }
        PyObjectScopedRef bDtype = PyObject_GetAttrString(b, "dtype");
        if(!bDtype) return NULL;
        if(!PyUnicode_Check(bDtype)) {
            PyErr_Format(
                PyExc_TypeError,
                "compareOrCombine: b.dtype did not return a string, from dtype %R", bDtype.get());
            return NULL;
        }
        if(PyUnicode_Compare(aDtype, bDtype) != 0) {
            PyErr_Format(
                PyExc_ValueError,
                "compareOrCombine: a.dtype != b.dtype, from a.dtype %R and b.dtype %R", aDtype.get(), bDtype.get());
            return NULL;
        }
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

    if(dimOrder != Py_None) {
        if(!PySequence_Check(dimOrder)) {
            PyErr_Format(PyExc_TypeError, "compareOrCombine: expected dim_order to be sequence, got %R", dimOrder);
            return NULL;
        }
    }

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

    // check if bDims is a subset of aDims, in the same order (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isTupleSubsetFast(bDims, aDims, outPermutation) && (dimOrder == Py_None || _isSameTupleAndSeqFast(aDims, dimOrder)))
            return _compareOrCombine_subsetDims<true, false>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aRawShape, NULL,
                aDims, bDims,
                outPermutation);
    }

    PyObjectScopedRef bRawShape = PyObject_CallFunctionObjArgs(getShapeOp, bRawTensor.get(), NULL);
    if(!bRawShape) return NULL;
    if(!PyTuple_Check(bRawShape)) {
        PyErr_Format(PyExc_TypeError, "compareOrCombine: expected b.raw_tensor.shape to be tuple, got %R", bRawShape.get());
        return NULL;
    }

    // check if aDims is a subset of bDims, in the same order (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isTupleSubsetFast(aDims, bDims, outPermutation) && (dimOrder == Py_None || _isSameTupleAndSeqFast(bDims, dimOrder)))
            return _compareOrCombine_subsetDims<false, false>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aRawShape, bRawShape,
                aDims, bDims,
                outPermutation);
    }

    // check if bDims is a subset of aDims, maybe reordered (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isTupleSubsetReorderFast(bDims, aDims, outPermutation) && (dimOrder == Py_None || _isSameTupleAndSeqFast(aDims, dimOrder)))
            return _compareOrCombine_subsetDims<true, true>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aRawShape, bRawShape,
                aDims, bDims,
                outPermutation);
    }

    // check if aDims is a subset of bDims, maybe reordered (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isTupleSubsetReorderFast(aDims, bDims, outPermutation) && (dimOrder == Py_None || _isSameTupleAndSeqFast(bDims, dimOrder)))
            return _compareOrCombine_subsetDims<false, true>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aRawShape, bRawShape,
                aDims, bDims,
                outPermutation);
    }

    {
        // follow the bin_op_out_template code

        // collect all dims
        bool haveDuplicateDims = false;
        int aDimsSize = PyTuple_GET_SIZE(aDims.get());
        if(aDimsSize != PyTuple_GET_SIZE(aRawShape.get())) {
            PyErr_Format(
                PyExc_ValueError,
                "compareOrCombine: a.dims and a.raw_tensor.shape have different size, from a.dims %R and a.raw_tensor.shape %R",
                aDims.get(), aRawShape.get());
            return NULL;
        }
        int bDimsSize = PyTuple_GET_SIZE(bDims.get());
        if(bDimsSize != PyTuple_GET_SIZE(bRawShape.get())) {
            PyErr_Format(
                PyExc_ValueError,
                "compareOrCombine: b.dims and b.raw_tensor.shape have different size, from b.dims %R and b.raw_tensor.shape %R",
                bDims.get(), bRawShape.get());
            return NULL;
        }
        PyObjectScopedRef allDims = PyList_New(0);
        std::vector<long> outShape;
        if(!allDims) return NULL;
        for(int i = 0; i < aDimsSize + bDimsSize; ++i) {
            PyObject* dim =
                i < aDimsSize ?
                PyTuple_GET_ITEM(aDims.get(), i) :
                PyTuple_GET_ITEM(bDims.get(), i - aDimsSize);
            {
                int contains = PySequence_Contains(allDims, dim);
                if(contains < 0) return NULL;
                if(contains) continue;
            }
            long dimValue =
                i < aDimsSize ?
                PyLong_AsLong(PyTuple_GET_ITEM(aRawShape.get(), i)) :
                PyLong_AsLong(PyTuple_GET_ITEM(bRawShape.get(), i - aDimsSize));
            if(dimValue < 0) {
                if(!PyErr_Occurred())
                    PyErr_Format(
                        PyExc_ValueError,
                        "compareOrCombine: a.raw_tensor.shape or b.raw_tensor.shape has negative dim, from a.raw_tensor.shape %R and b.raw_tensor.shape %R",
                        aRawShape.get(), bRawShape.get());
                return NULL;
            }
            // Not simply `all_dims.append(dim)`,
            // because a dim might occur multiple times in a.dims or b.dims
            // (with different match_priority),
            // e.g. in the case of square matrices.
            // Still it is the common case that they are unique,
            // and this allows for a faster path.
            int aDimsCount = PySequence_Count(aDims, dim);
            if(aDimsCount < 0) return NULL;
            int bDimsCount = PySequence_Count(bDims, dim);
            if(bDimsCount < 0) return NULL;
            if(aDimsCount <= 1 && bDimsCount <= 1) {
                if(PyList_Append(allDims, dim) < 0) return NULL;
                outShape.push_back(dimValue);
                continue;
            }
            haveDuplicateDims = true;
            int c = 0;
            for(int j = 0; j < (aDimsCount >= bDimsCount ? aDimsSize : bDimsCount); ++j) {
                PyObject* dim_ = PyTuple_GET_ITEM(
                    aDimsCount >= bDimsCount ? aDims.get() : bDims.get(), j);
                if(dim_ != dim) {
                    int eq = PyObject_RichCompareBool(dim_, dim, Py_EQ);
                    if(eq < 0) return NULL;
                    if(!eq) continue;
                }
                if(PyList_Append(allDims, dim_) < 0) return NULL;
                outShape.push_back(dimValue);
                ++c;
            }
            if(c != std::max(aDimsCount, bDimsCount)) {
                PyErr_Format(
                    PyExc_SystemError,
                    "compareOrCombine: non-deterministic dim count, dim %R, from a.dims %R and b.dims %R",
                    dim, aDims.get(), bDims.get());
                return NULL;
            }
        }
        assert(outShape.size() == (size_t) PyList_GET_SIZE(allDims.get()));

        // check if all dims are in a and b, or whether we need allowBroadcastAllSources
        bool error = false;
        bool aDimsIsSubset = _isTupleSubsetList(aDims, allDims, error);
        if(error) return NULL;
        bool bDimsIsSubset = _isTupleSubsetList(bDims, allDims, error);
        if(error) return NULL;
        if(!aDimsIsSubset && !bDimsIsSubset) {
            if(!allowBroadcastAllSources) {
                PyErr_Format(
                    PyExc_ValueError,
                    "compareOrCombine: sources %R %R not allowed with allow_broadcast_all_sources=False",
                    a, b);
                return NULL;
            }
        }

        // maybe reorder according to dimOrder
        if(dimOrder != Py_None) {
            std::vector<std::pair<PyObject*, long>> outDimWithValue;
            for(size_t i = 0; i < outShape.size(); ++i)
                outDimWithValue.push_back(std::make_pair(PyList_GET_ITEM(allDims.get(), i), outShape[i]));
            struct Cmp {
                PyObject* dimOrder;
                int dimOrderLen;
                bool hadError;
                Cmp(PyObject* dimOrder_)
                : dimOrder(dimOrder_), dimOrderLen(0), hadError(false)
                {
                    dimOrderLen = PySequence_Size(dimOrder_);
                    if(dimOrderLen < 0) hadError = true;
                }
                int getIndex(PyObject* a) {
                    if(hadError) return 0;
                    for(int i = 0; i < dimOrderLen; ++i) {
                        PyObjectScopedRef d = PySequence_GetItem(dimOrder, i);
                        if(!d) { hadError = true; return 0; }
                        if(d == a) return i;
                        int eq = PyObject_RichCompareBool(a, d, Py_EQ);
                        if(eq < 0) { hadError = true; return 0; }
                        else if(eq == 0) return i;
                    }
                    return dimOrderLen;
                }
                bool operator()(std::pair<PyObject*, long> a, std::pair<PyObject*, long> b) {
                    return (*this)(a.first, b.first);
                }
                bool operator()(PyObject* a, PyObject* b) {
                    if(a == b) return false;
                    return getIndex(a) < getIndex(b);
                }
            } cmp(dimOrder);            
            std::stable_sort(outDimWithValue.begin(), outDimWithValue.end(), cmp);
            if(cmp.hadError) return NULL;
            for(size_t i = 0; i < outShape.size(); ++i) {
                PyList_SET_ITEM(allDims.get(), i, outDimWithValue[i].first);
                outShape[i] = outDimWithValue[i].second;
            }
        }

        PyObjectScopedRef res;
        {
            PyObjectScopedRef name = PyUnicode_FromString(rawOpName);
            if(!name) return NULL;
            PyObjectScopedRef dtype = resultIsBool ? PyUnicode_InternFromString("bool") : PyObject_GetAttrString(a, "dtype");
            if(!dtype) return NULL;
            res = PyObject_CallFunctionObjArgs(
                modState->tensorType(), name.get(), allDims.get(), dtype.get(), NULL);
            if(!res) return NULL;
        }

        {
            PyObjectScopedRef aRawTensorExt = _permuteAndExtend(
                rawOpName, permuteOp, reshapeOp,
                a, aDims, aRawTensor, aRawShape,
                allDims, outShape);
            if(!aRawTensorExt) return NULL;
            PyObjectScopedRef bRawTensorExt = _permuteAndExtend(
                rawOpName, permuteOp, reshapeOp,
                b, bDims, bRawTensor, bRawShape,
                allDims, outShape);
            if(!bRawTensorExt) return NULL;
            PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(
                rawOp, aRawTensorExt.get(), bRawTensorExt.get(), NULL);
            if(!resRawTensor) return NULL;
            if(PyObject_SetAttrString(res, "raw_tensor", resRawTensor) < 0) return NULL;
        }

        {
            PyObjectScopedRef featureDim = _consistentFeatureDim(a, b);
            if(!featureDim) return NULL;
            if(featureDim != Py_None)
                if(PyObject_SetAttrString(res, "feature_dim", featureDim) < 0)
                    return NULL;
        }

        {
            PyObjectScopedRef sparseDim = _consistentSparseDim(a, b);
            if(!sparseDim) return NULL;
            if(sparseDim != Py_None)
                if(PyObject_SetAttrString(res, "sparse_dim", sparseDim) < 0)
                    return NULL;
        }

        return res.release();
    }
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

template<bool isCompare>
static PyObject* _pyTensorCompareOrCombine(PyObject *self, PyObject *args, PyObject *kwargs) {
    static const char *kwlist[] = { "a", "kind", "b", "allow_broadcast_all_sources", "dim_order", NULL };
    PyObject* a;
    char* kind;
    PyObject* b;
    unsigned char allow_broadcast_all_sources = false;
    PyObject* dim_order = Py_None;
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, isCompare ? "OsO|$bO:tensor_compare" : "OsO|$bO:tensor_combine", (char**) kwlist,
            &a, &kind, &b, &allow_broadcast_all_sources, &dim_order))
        return NULL;

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;

    bool haveBackendWithCachedOps = false;
    BackendWithCachedOps backendId;
    if(isTorchBackendForTensor(modState, a) || isTorchBackendForTensor(modState, b)) {
        haveBackendWithCachedOps = true;
        backendId = BWCO_Torch;
    }

    // compare funcs: "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=", "not_equal"|"!="
    static std::map<std::string, RawOp> kindToCompareFunc;
    if(kindToCompareFunc.empty()) {
        kindToCompareFunc["=="] = TOp_Eq;
        kindToCompareFunc["eq"] = TOp_Eq;
        kindToCompareFunc["equal"] = TOp_Eq;
        kindToCompareFunc["!="] = TOp_Ne;
        kindToCompareFunc["<>"] = TOp_Ne;
        kindToCompareFunc["ne"] = TOp_Ne;
        kindToCompareFunc["not_equal"] = TOp_Ne;
        kindToCompareFunc["<"] = TOp_Lt;
        kindToCompareFunc["lt"] = TOp_Lt;
        kindToCompareFunc["less"] = TOp_Lt;
        kindToCompareFunc["<="] = TOp_Le;
        kindToCompareFunc["le"] = TOp_Le;
        kindToCompareFunc["less_equal"] = TOp_Le;
        kindToCompareFunc[">"] = TOp_Gt;
        kindToCompareFunc["gt"] = TOp_Gt;
        kindToCompareFunc["greater"] = TOp_Gt;
        kindToCompareFunc[">="] = TOp_Ge;
        kindToCompareFunc["ge"] = TOp_Ge;
        kindToCompareFunc["greater_equal"] = TOp_Ge;
    }

    // combine funcs: "add"|"+", "sub"|"-", "mul"|"*", "truediv"|"/", "floordiv"|"//", "mod"|"%", "pow"|"**",
    //   "max"|"maximum", "min"|"minimum", "logical_and", "logical_or", "squared_difference"
    static std::map<std::string, RawOp> kindToCombineFunc;
    if(kindToCombineFunc.empty()) {
        kindToCombineFunc["+"] = TOp_Add;
        kindToCombineFunc["add"] = TOp_Add;
        kindToCombineFunc["-"] = TOp_Sub;
        kindToCombineFunc["sub"] = TOp_Sub;
        kindToCombineFunc["*"] = TOp_Mul;
        kindToCombineFunc["mul"] = TOp_Mul;
        kindToCombineFunc["/"] = TOp_TrueDiv;
        kindToCombineFunc["truediv"] = TOp_TrueDiv;
        kindToCombineFunc["//"] = TOp_FloorDiv;
        kindToCombineFunc["floordiv"] = TOp_FloorDiv;
        kindToCombineFunc["%"] = TOp_Mod;
        kindToCombineFunc["mod"] = TOp_Mod;
        kindToCombineFunc["**"] = TOp_Pow;
        kindToCombineFunc["pow"] = TOp_Pow;
        kindToCombineFunc["maximum"] = TOp_Maximum;
        kindToCombineFunc["max"] = TOp_Maximum;
        kindToCombineFunc["minimum"] = TOp_Minimum;
        kindToCombineFunc["min"] = TOp_Minimum;
        kindToCombineFunc["logical_and"] = TOp_And;
        kindToCombineFunc["logical_or"] = TOp_Or;
        kindToCombineFunc["squared_difference"] = TOp_SquaredDifference;
    }

    auto it = isCompare ? kindToCompareFunc.find(kind) : kindToCombineFunc.find(kind);
    if(it == (isCompare ? kindToCompareFunc.end() : kindToCombineFunc.end())) {
        PyErr_Format(PyExc_ValueError, "tensor_%s: invalid kind '%s'", isCompare ? "compare" : "combine", kind);
        return NULL;
    }
    RawOp rawOp = it->second;

    if(haveBackendWithCachedOps) {
        return compareOrCombineViaCached(
            a, b,
            isCompare,
            modState, backendId, rawOp,
            (bool) allow_broadcast_all_sources, dim_order);
    }

    const char* rawOpName_ = rawOpName(rawOp);
    PyObject* backend;
    if(PyObject_IsInstance(a, modState->tensorType()))
        backend = getBackendForTensor(modState, a);
    else if(PyObject_IsInstance(b, modState->tensorType()))
        backend = getBackendForTensor(modState, b);
    else
        backend = modState->globalBackend();
    if(!backend) return NULL;

    PyObjectScopedRef func = PyObject_GetAttrString(backend, isCompare ? "compare" : "combine");
    if(!func) return NULL;
    if(!allow_broadcast_all_sources && dim_order == Py_None)
        return PyObject_CallFunction(func, "OsO", a, rawOpName_, b);
    // need kwargs
    PyObjectScopedRef args_ = PyTuple_New(3);
    if(!args_) return NULL;
    Py_INCREF(a);
    PyTuple_SET_ITEM(args_.get(), 0, a);
    {
        PyObjectScopedRef kind_ = PyUnicode_FromString(rawOpName_);
        if(!kind) return NULL;
        PyTuple_SET_ITEM(args_.get(), 1, kind_.release());
    }
    Py_INCREF(b);
    PyTuple_SET_ITEM(args_.get(), 2, b);
    PyObjectScopedRef kwargs_ = PyDict_New();
    if(!kwargs_) return NULL;
    if(allow_broadcast_all_sources)
        PyDict_SetItemString(kwargs_.get(), "allow_broadcast_all_sources", Py_True);
    if(dim_order != Py_None)
        PyDict_SetItemString(kwargs_.get(), "dim_order", dim_order);
    return PyObject_Call(func, args_, kwargs_);
}

PyObject* pyTensorCompare(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _pyTensorCompareOrCombine<true>(self, args, kwargs);
}

PyObject* pyTensorCombine(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _pyTensorCompareOrCombine<false>(self, args, kwargs);
}

template<RawOp op, bool isCompare>
static PyObject* _tensorCompareOrCombineSpecific(PyModuleState* modState, PyObject* a, PyObject* b) {
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
static PyObject* _pyTensorCompareOrCombineSpecific(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_Format(PyExc_TypeError, "tensor_%s: expected 2 args, got %i", rawOpName(op), nargs);
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;
    return _tensorCompareOrCombineSpecific<op, isCompare>(modState, args[0], args[1]);
}

template<RawOp op, bool isCompare>
static PyObject* _pyTensorCompareOrCombineSpecificR(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_Format(PyExc_TypeError, "tensor_r%s: expected 2 args, got %i", rawOpName(op), nargs);
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;
    return _tensorCompareOrCombineSpecific<op, isCompare>(modState, args[1], args[0]);
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
        return _tensorCompareOrCombineSpecific<TOp_Eq, true>(modState, a, b);

    Py_INCREF(Py_False);
    return Py_False;
}

#define DefinePyTensorCompare(op) \
    PyObject* pyTensor##op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) { \
        return _pyTensorCompareOrCombineSpecific<TOp_##op, true>(self, args, nargs); \
    }

DefinePyTensorCompare(Ne)
DefinePyTensorCompare(Lt)
DefinePyTensorCompare(Le)
DefinePyTensorCompare(Gt)
DefinePyTensorCompare(Ge)

#define DefinePyTensorCombine(op) \
    PyObject* pyTensor##op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) { \
        return _pyTensorCompareOrCombineSpecific<TOp_##op, false>(self, args, nargs); \
    }

#define DefinePyTensorCombineR(op) \
    PyObject* pyTensorR##op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) { \
        return _pyTensorCompareOrCombineSpecificR<TOp_##op, false>(self, args, nargs); \
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

