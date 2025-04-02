
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

// copy of Tensor.copy()
PyObject* tensorCopy(
    PyModuleState* modState,
    PyObject* tensor,
    const char* name)
{
    PyObjectScopedRef rawTensor = PyObject_GetAttrString(tensor, "_raw_tensor");
    if(rawTensor == Py_None)
        return tensorCopyTemplate(modState, tensor, name);
    PyObjectScopedRef res = tensorCopyTemplate(modState, tensor, name);
    if(!res) return NULL;
    if(PyObject_SetAttrString(res, "_raw_tensor", rawTensor) < 0) return NULL;
    return res.release();
}

// all but time_dim_axis (or other special axes, or any axes)
static bool _copyTensorExtraToKwargs(PyObject* extra, PyObject* kwargs) {
    PyObjectScopedRef batch = PyObject_GetAttrString(extra, "batch");
    if(!batch) return false;
    if(batch != Py_None) {
        if(PyDict_SetItemString(kwargs, "batch", batch) < 0) return false;
    }
    {
        PyObjectScopedRef beam = PyObject_GetAttrString(extra, "beam");
        if(!beam) return false;
        if(beam == Py_None && batch != Py_None) {
            beam = PyObject_GetAttrString(batch, "beam");
        }
        if(beam != Py_None) {
            if(PyDict_SetItemString(kwargs, "beam", beam) < 0) return false;
        }
    }
    {
        PyObjectScopedRef control_flow_ctx = PyObject_GetAttrString(extra, "control_flow_ctx");
        if(!control_flow_ctx) return false;
        if(control_flow_ctx != Py_None) {
            if(PyDict_SetItemString(kwargs, "control_flow_ctx", control_flow_ctx) < 0) return false;
        }
    }
    {
        PyObjectScopedRef available_for_inference = PyObject_GetAttrString(extra, "available_for_inference");
        if(!available_for_inference) return false;
        if(available_for_inference != Py_None) {
            if(PyDict_SetItemString(kwargs, "available_for_inference", available_for_inference) < 0) return false;
        }
    }
    return true;
}

// copy of Tensor.copy_template()
PyObject* tensorCopyTemplate(
    PyModuleState* modState,
    PyObject* tensor,
    const char* name,
    PyObject* dtype)
{
    PyObjectScopedRef version = PyObject_GetAttrString(tensor, "version");
    if(!version) return NULL;
    if(!PyLong_Check(version)) {
        PyErr_Format(
            PyExc_TypeError,
            "tensorCopyTemplate: tensor.version did not return an int, from version %R", version.get());
        return NULL;
    }
    long versionInt = PyLong_AsLong(version);
    if(versionInt != 1 && versionInt != 2) {
        if(!PyErr_Occurred())
            PyErr_Format(
                PyExc_ValueError,
                "tensorCopyTemplate: tensor.version is invalid, from version %R", version.get());
        return NULL;
    }
    PyObjectScopedRef extra = PyObject_GetAttrString(tensor, "_extra");
    if(!extra) return NULL;
    if(versionInt == 2 && extra == Py_None)
        return tensorCopyTemplateSimple(modState, tensor, name, dtype);

    // follows Tensor.get_kwargs()

    PyObjectScopedRef emptyArgs = PyTuple_New(0);
    if(!emptyArgs) return NULL;
    PyObjectScopedRef kwargs = PyDict_New();
    if(!kwargs) return NULL;

    {
        PyObjectScopedRef name_ = name ? PyUnicode_FromString(name) : PyObject_GetAttrString(tensor, "name");
        if(!name_) return NULL;
        if(PyDict_SetItemString(kwargs, "name", name_) < 0) return NULL;
    }
    if(dtype && dtype != Py_None) {
        if(PyDict_SetItemString(kwargs, "dtype", dtype) < 0) return NULL;
    } else {
        PyObjectScopedRef dtype_ = PyObject_GetAttrString(tensor, "dtype");
        if(!dtype_) return NULL;
        if(PyDict_SetItemString(kwargs, "dtype", dtype_) < 0) return NULL;
    }
    {
        PyObjectScopedRef dims = PyObject_GetAttrString(tensor, "_dims");
        if(!dims) return NULL;
        if(PyDict_SetItemString(kwargs, "dims", dims) < 0) return NULL;
    }
    if(versionInt == 1 && extra != Py_None) {
        PyObjectScopedRef time_dim_axis = PyObject_GetAttrString(extra, "time_dim_axis");
        if(!time_dim_axis) return NULL;
        if(time_dim_axis != modState->notSpecified()) {
            if(PyDict_SetItemString(kwargs, "time_dim_axis", time_dim_axis) < 0) return NULL;
        }
    }
    {
        PyObjectScopedRef feature_dim_axis = PyObject_GetAttrString(tensor, "_feature_dim_axis");
        if(!feature_dim_axis) return NULL;
        if(feature_dim_axis != modState->notSpecified()) {
            if(PyDict_SetItemString(kwargs, "feature_dim_axis", feature_dim_axis) < 0) return NULL;
        }
    }
    {
        PyObjectScopedRef sparse_dim = PyObject_GetAttrString(tensor, "sparse_dim");
        if(!sparse_dim) return NULL;
        if(sparse_dim != Py_None) {
            if(PyDict_SetItemString(kwargs, "sparse_dim", sparse_dim) < 0) return NULL;
        }
    }
    if(versionInt == 1) {
        if(PyDict_SetItemString(kwargs, "version", version) < 0) return NULL;
    }
    if(extra != Py_None) {
        if(!_copyTensorExtraToKwargs(extra, kwargs))
            return NULL;
    }

    return PyObject_Call(modState->tensorType(), emptyArgs, kwargs);
}

// just copies name, dims, dtype, feature_dim, sparse_dim. no or other things.
// this is like what bin_op_out_template is doing.
PyObject* tensorCopyTemplateSimple(
    PyModuleState* modState,
    PyObject* tensor,
    const char* name_,
    PyObject* dtype,
    bool copySparseDim)
{
    PyObjectScopedRef name = name_ ? PyUnicode_FromString(name_) : PyObject_GetAttrString(tensor, "name");
    if(!name) return NULL;
    PyObjectScopedRef dtype_;
    if(!dtype) {
        dtype_ = PyObject_GetAttrString(tensor, "dtype");
        if(!dtype_) return NULL;
        dtype = dtype_.get();
    }
    PyObjectScopedRef dims = PyObject_GetAttrString(tensor, "_dims");
    if(!dims) return NULL;

    PyObjectScopedRef res = PyObject_CallFunctionObjArgs(
        modState->tensorType(), name.get(), dims.get(), dtype, NULL);
    if(!res) return NULL;

    {
        PyObjectScopedRef feature_dim_axis = PyObject_GetAttrString(tensor, "_feature_dim_axis");
        if(!feature_dim_axis) return NULL;
        if(feature_dim_axis != Py_None)
            if(PyObject_SetAttrString(res, "_feature_dim_axis", feature_dim_axis) < 0)
                return NULL;
    }
    if(copySparseDim) {
        PyObjectScopedRef sparse_dim = PyObject_GetAttrString(tensor, "sparse_dim");
        if(!sparse_dim) return NULL;
        if(sparse_dim != Py_None)
            if(PyObject_SetAttrString(res, "sparse_dim", sparse_dim) < 0)
                return NULL;
    }
    return res.release();
}

// no error check here; false does not mean they are different, it just checks for `is`
template<typename ASeqT, typename BSeqT>
static bool _isSameSeqFast(ASeqT a, BSeqT b) {
    if(a.get() == b.get())
        return true;
    if(a.size() != b.size())
        return false;
    for(int i = 0; i < a.size(); ++i) {
        PyObject* a_ = a.getItem(i);
        PyObject* b_ = b.getItem(i);
        if(a_ != b_)
            return false;
    }
    return true;
}

// no error check here; false does not mean they are different, it just checks for `is`.
// when it returns with false, outPermutation is undefined.
template<typename ASeqT, typename BSeqT>
static bool _isSeqSubsetFast(ASeqT subset, BSeqT superset, std::vector<int>& outPermutation) {
    if(subset.size() > superset.size())
        return false;
    int j = 0;
    for(int i = 0; i < subset.size(); ++i) {
        PyObject* a_ = subset.getItem(i);
        while(true) {
            if(j >= superset.size())
                return false;
            PyObject* b_ = superset.getItem(j);
            if(a_ == b_) break;
            ++j; outPermutation.push_back(-1);
        }
        ++j; outPermutation.push_back(i);
    }
    for(; j < superset.size(); ++j)
        outPermutation.push_back(-1);
    return true;
}

// no error check here; false does not mean they are different, it just checks for `is`.
// when it returns with false, outPermutation is undefined.
template<typename ASeqT, typename BSeqT>
static bool _isSeqSubsetReorderFast(ASeqT subset, BSeqT superset, std::vector<int>& outPermutation) {
    if(subset.size() > superset.size())
        return false;
    outPermutation.resize(superset.size());
    std::vector<bool> subsetTaken(subset.size(), false);
    for(int j = 0; j < superset.size(); ++j) {
        PyObject* b_ = superset.getItem(j);
        int i = 0;
        for(; i < subset.size(); ++i) {
            if(subsetTaken[i]) continue;
            PyObject* a_ = subset.getItem(i);
            if(a_ == b_) break;
        }
        if(i < subset.size()) {
            subsetTaken[i] = true;
            outPermutation[j] = i;
        }
        else
            outPermutation[j] = -1;
    }
    for(int i = 0; i < subset.size(); ++i) {
        if(!subsetTaken[i])
            return false;
    }
    return true;
}

static bool _isTupleSubsetReorderList(PyObject* subsetTuple, PyObject* supersetList, bool& error) {
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

PyObject* pyTensorCopy(PyObject *self, PyObject *args, PyObject *kwargs) {
    static const char *kwlist[] = { "tensor", "name", NULL };
    PyObject* tensor;
    const char* name = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|z:tensor_copy",
            (char**) kwlist, &tensor, &name))
        return NULL;

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState)
        return NULL;

    return tensorCopy(modState, tensor, name);
}

PyObject* pyTensorCopyTemplate(PyObject *self, PyObject *args, PyObject *kwargs) {
    static const char *kwlist[] = { "tensor", "name", "dtype", NULL };
    PyObject* tensor;
    const char* name = NULL;
    PyObject* dtype = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|z$O:tensor_copy_template",
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
    PyModuleState* modState, BackendWithCachedOps backendId, const char* funcName, PyObject* tensor, PyObject* rawTensor, bool checkDtype = true
) {
    if(checkDtype) {
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

// when it returns with false, some exception should be raised
template<typename ASeqT, typename BSeqT>
static bool _getPermutationSupersetToSubset(const char* funcName, ASeqT subset, BSeqT superset, std::vector<int>& outPermutation) {
    if(_isSeqSubsetFast(subset, superset, outPermutation))
        return true;
    outPermutation.clear();

    if(_isSeqSubsetReorderFast(subset, superset, outPermutation))
        return true;
    outPermutation.clear();

    // Generic fallback, using `==` instead of just `is`.
    // The logic matches Tensor.get_out_permutation_to_dims.
    int count = 0;
    std::vector<bool> taken(subset.size(), false);
    for(int i = 0; i < superset.size(); ++i) {
        PyObject* dim = superset.getItem(i);
        std::vector<int> candidates;
        for(int j = 0; j < subset.size(); ++j) {
            if(taken[j]) continue;
            PyObject* dim_ = subset.getItem(j);
            if(dim_ == dim) {  // prefer that one over all others
                candidates.clear();
                candidates.push_back(j);
                break;
            }
            int eq = PyObject_RichCompareBool(dim, dim_, Py_EQ);
            if(eq < 0) return false;
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
            size_t maxMatchPriorityIdx = 0;
            long maxMatchPriority = -1;
            int countSameMatchPriority = 0;
            for(size_t j = 0; j < candidates.size(); ++j) {
                PyObject* dim_ = subset.getItem(candidates[j]);
                PyObject* matchPriority = PyObject_GetAttrString(dim_, "match_priority");
                if(!matchPriority) return false;
                if(!PyLong_Check(matchPriority)) {
                    PyErr_Format(
                        PyExc_TypeError,
                        "%s: dim %R did not return an int for match_priority, from tensor dims %R",
                        funcName, dim_, subset.get());
                    return false;
                }
                long matchPriority_ = PyLong_AsLong(matchPriority);
                if(matchPriority_ < 0 && PyErr_Occurred()) return false;
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
                    "%s: dim %R is ambiguous, from tensor dims %R and all dims %R",
                    funcName, dim, subset.get(), superset.get());
                return false;
            }
            outPermutation.push_back(candidates[maxMatchPriorityIdx]);
            taken[candidates[maxMatchPriorityIdx]] = true;
            ++count;
        }
    }
    if(count != subset.size()) {
        PyErr_Format(
            PyExc_ValueError,
            "%s: not all dims are matched, from tensor dims %R and all dims %R",
            funcName, subset.get(), superset.get());
        return false;
    }
    assert((int) outPermutation.size() == superset.size());
    return true;
}

// return raw tensor
template<typename OutDimSeqT>
static PyObject* _permuteAndExtend(
    const char* rawOpName,
    PyObject* permuteOp, PyObject* reshapeOp, PyObject* getShapeOp,
    PyObject* tensor, PyTupleOrListStaticRef<true> dims, PyObject* rawTensor,
    OutDimSeqT outDims,
    std::vector<int>& outPermutation /* if empty, will get it from outDims */
) {
    // First find the mapping.
    if(outPermutation.empty() && !_getPermutationSupersetToSubset(rawOpName, dims, outDims, outPermutation))
        return NULL;
    assert((int) outPermutation.size() == outDims.size());

    PyObject* rawTensor_ = rawTensor;
    PyObjectScopedRef rawTensorExt; // just for holding the ref and decrefing it later

    // Maybe permute the tensor
    bool needPermute = false;
    for(int i = 0; i < (int) outPermutation.size(); ++i) {
        if(i > 0 && outPermutation[i] != outPermutation[i - 1] + 1) {
            needPermute = true;
            break;
        }
    }
    if(needPermute) {
        PyObjectScopedRef permuteArg = PyTuple_New(dims.size());
        if(!permuteArg) return NULL;
        int j = 0;
        for(int i = 0; i < (int) outPermutation.size(); ++i) {
            if(outPermutation[i] < 0) continue;
            PyObject* intObj = PyLong_FromLong(outPermutation[i]);
            if(!intObj) return NULL;
            assert(j < dims.size());
            PyTuple_SET_ITEM(permuteArg.get(), j, intObj);
            ++j;
        }
        assert(j == PyTuple_GET_SIZE(permuteArg.get()));
        assert(j == dims.size());
        rawTensor_ = PyObject_CallFunctionObjArgs(permuteOp, rawTensor_, permuteArg.get(), NULL);
        if(!rawTensor_) return NULL;
        rawTensorExt = rawTensor_;
    }

    // Maybe reshape the tensor
    if(outDims.size() > dims.size()) {
        PyObjectScopedRef rawShape = PyObject_CallFunctionObjArgs(getShapeOp, rawTensor_, NULL);
        if(!rawShape) return NULL;
        if(!PyTuple_Check(rawShape)) {
            PyErr_Format(PyExc_TypeError, "%s: expected raw_tensor.shape to be tuple, got %R", rawOpName, rawShape.get());
            return NULL;
        }
        if(PyTuple_GET_SIZE(rawShape.get()) != dims.size()) {
            PyErr_Format(
                PyExc_ValueError,
                "%s: raw_tensor ndim != tensor ndim, from tensor dims %R and raw_tensor shape %R",
                rawOpName, dims.get(), rawShape.get());
            return NULL;
        }

        PyObjectScopedRef rawShapeExt = PyTuple_New(outPermutation.size());
        if(!rawShapeExt) return NULL;
        int j = 0;
        for(int i = 0; i < (int) outPermutation.size(); ++i) {
            PyObject* d;
            if(outPermutation[i] >= 0) {
                assert(j < dims.size());
                d = PyTuple_GET_ITEM(rawShape.get(), j);
                Py_XINCREF(d);
                ++j;
            }
            else
                d = PyLong_FromLong(1);
            if(!d) return NULL;
            PyTuple_SET_ITEM(rawShapeExt.get(), i, d);
        }
        assert(j == dims.size());
        rawTensor_ = PyObject_CallFunctionObjArgs(reshapeOp, rawTensor_, rawShapeExt.get(), NULL);
        if(!rawTensor_) return NULL;
        rawTensorExt = rawTensor_;
    }

    if(rawTensorExt) rawTensorExt.release();
    else Py_INCREF(rawTensor_); // we still have it borrowed
    return rawTensor_;
}

template<bool bIsSubset>
static PyObject* _compareOrCombine_subsetDims(
    PyModuleState* modState,
    const char* rawOpName, bool resultIsBool,
    PyObject* permuteOp, PyObject* reshapeOp, PyObject* getShapeOp, PyObject* getDtypeOp, PyObject* rawOp,
    PyObject* a, PyObject* b,
    PyObject* aRawTensor, PyObject* bRawTensor,
    PyTupleOrListStaticRef<true> aDims, PyTupleOrListStaticRef<true> bDims,
    std::vector<int>& outPermutation
) {
    // The tensor with the subset dims will be adapted to the other tensor.
    PyObjectScopedRef rawTensorExt;
    if(bIsSubset)
        rawTensorExt = _permuteAndExtend(rawOpName, permuteOp, reshapeOp, getShapeOp, b, bDims, bRawTensor, aDims, outPermutation);
    else
        rawTensorExt = _permuteAndExtend(rawOpName, permuteOp, reshapeOp, getShapeOp, a, aDims, aRawTensor, bDims, outPermutation);

    // Now create the result.
    PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(
        rawOp, bIsSubset ? aRawTensor : rawTensorExt.get(), bIsSubset ? rawTensorExt.get() : bRawTensor, NULL);
    if(!resRawTensor) return NULL;
    PyObjectScopedRef dtype = PyObject_CallFunctionObjArgs(getDtypeOp, resRawTensor.get(), NULL);
    if(!dtype) return NULL;
    PyObjectScopedRef res = tensorCopyTemplateSimple(modState, bIsSubset ? a : b, rawOpName, dtype, !resultIsBool);
    if(!res) return NULL;
    if(PyObject_SetAttrString(res, "_raw_tensor", resRawTensor) < 0) return NULL;
    return res.release();
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

template<typename IntT>
static PyObject* _cppLongVectorToPyList(const std::vector<IntT>& vector) {
    PyObjectScopedRef res = PyList_New(vector.size());
    if(!res) return NULL;
    for(size_t i = 0; i < vector.size(); ++i) {
        PyObject* v = PyLong_FromLong(vector[i]);
        if(!v) return NULL;
        PyList_SET_ITEM(res.get(), i, v);
    }
    return res.release();
}

PyObject* pyTensorGetOutPermutationsToDims(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "tensor_get_out_permutations_to_dims() takes exactly 2 args: tensor, dims");
        return NULL;
    }

    PyObjectScopedRef selfDims = PyObject_GetAttrString(args[0], "_dims");
    if(!selfDims) return NULL;
    if(!PyTuple_Check(selfDims)) {
        PyErr_Format(PyExc_TypeError, "tensor_get_out_permutations_to_dims: expected tensor.dims to be tuple, got %R", selfDims.get());
        return NULL;
    }
    PyTupleOrListStaticRef<true> selfDimsSeq(selfDims);
    PyTupleOrListRef otherDimsSeq(args[1]);
    if(!otherDimsSeq.isValid()) {
        PyErr_Format(PyExc_TypeError, "tensor_get_out_permutations_to_dims: expected dims to be tuple or list, got %R", args[1]);
        return NULL;
    }

    std::vector<int> outPermutation;
    if(_getPermutationSupersetToSubset("tensor_get_out_permutations_to_dims", selfDimsSeq, otherDimsSeq, outPermutation))
        return _cppLongVectorToPyList(outPermutation);
    return NULL;
}

// >=0 means actual set, -1 means None, -2 is error
static long _getAxis(const char* funcName, PyObject* tensor, const char* axisName) {
    PyObjectScopedRef axis = PyObject_GetAttrString(tensor, axisName);
    if(!axis) return -2;
    if(axis == Py_None) return -1;
    if(!PyLong_Check(axis)) {
        PyErr_Format(PyExc_TypeError, "%s: expected tensor.%s to be int or None, got %R", funcName, axisName, axis.get());
        return -1;
    }
    long axis_ = PyLong_AsLong(axis);
    if(axis_ < 0) {
        PyErr_Format(PyExc_ValueError, "%s: expected tensor.%s to be >= 0, got %R", funcName, axisName, axis.get());
        return -1;
    }
    return axis_;
}

static bool _setNewAxisConsistentFromPerm(
    const char* funcName, PyObject* tensor, PyObject* outTensor, bool checkExisting, const std::vector<int>& perm, const char* axisName
) {
    long axisInt = _getAxis(funcName, tensor, axisName);
    if(axisInt < -1) return false;
    long wantedOutAxisInt = -1;
    if(axisInt >= 0) {
        for(long j = 0; j < (long) perm.size(); ++j) {
            if(perm[j] == axisInt) {
                wantedOutAxisInt = j;
                break;
            }
        }
        if(wantedOutAxisInt < 0) {
            PyErr_Format(PyExc_ValueError, "%s: tensor.%s %ld is not in perm", funcName, axisName, axisInt);
            return false;
        }
    }
    if(checkExisting) {
        long outAxisInt = _getAxis(funcName, outTensor, axisName);
        if(outAxisInt < -1) return false;
        if(wantedOutAxisInt == outAxisInt)
            return true;
    }
    else {
        if(wantedOutAxisInt < 0)
            return true; // we assume the new axis is by default None
    }
    if(wantedOutAxisInt < 0) {
        if(PyObject_SetAttrString(outTensor, axisName, Py_None) < 0)
            return false;
    }
    else {
        PyObjectScopedRef wantedOutAxis = PyLong_FromLong(wantedOutAxisInt);
        if(!wantedOutAxis) return false;
        if(PyObject_SetAttrString(outTensor, axisName, wantedOutAxis) < 0)
            return false;
    }
    return true;
}

template<bool rawMode>
static PyObject* tensorCopyCompatibleToDims(const char* funcName, PyModuleState* modState, PyObject* tensor, PyObject* outDims) {
    PyTupleOrListRef outDimsSeq(outDims);
    if(!outDimsSeq.isValid()) {
        PyErr_Format(PyExc_TypeError, "%s: expected dims to be tuple or list, got %R", funcName, outDims);
        return NULL;
    }

    PyObjectScopedRef dims = PyObject_GetAttrString(tensor, "_dims");
    if(!dims) return NULL;
    if(!PyTuple_Check(dims)) {
        PyErr_Format(PyExc_TypeError, "%s: expected tensor.dims to be tuple, got %R", funcName, dims.get());
        return NULL;
    }
    PyTupleOrListStaticRef<true> dimsSeq(dims);

    PyObjectScopedRef rawTensor = PyObject_GetAttrString(tensor, "_raw_tensor");
    if(!rawTensor) return NULL;

    // follow Tensor.copy_compatible_to_dims logic

    std::vector<int> outPermutation;
    PyObjectScopedRef outRawTensor;
    if(rawTensor == Py_None) {
        if(rawMode) {
            PyErr_Format(PyExc_ValueError, "%s: tensor does not have a raw_tensor", funcName);
            return NULL;
        }
        if(!_getPermutationSupersetToSubset(funcName, dimsSeq, outDimsSeq, outPermutation))
            return NULL;
    }
    else if(modState->isTorchTensorType((PyObject*) Py_TYPE(rawTensor))) {
        PyObject* permuteOp = modState->cachedOp(TOp_Permute, BWCO_Torch);
        if(!permuteOp) return NULL;
        PyObject* reshapeOp = modState->cachedOp(TOp_Reshape, BWCO_Torch);
        if(!reshapeOp) return NULL;
        PyObject* getShapeOp = modState->cachedOp(TOp_GetShape, BWCO_Torch);
        if(!getShapeOp) return NULL;
        outRawTensor = _permuteAndExtend(funcName, permuteOp, reshapeOp, getShapeOp, tensor, dimsSeq, rawTensor, outDimsSeq, outPermutation);
        if(!outRawTensor) return NULL;
    }
    else {  // generic backend fallback
        PyObject* backend = getBackendForRawTensor(modState, rawTensor);
        PyObjectScopedRef permuteOp = PyObject_GetAttrString(backend, "transpose_raw");
        if(!permuteOp) return NULL;
        PyObjectScopedRef reshapeOp = PyObject_GetAttrString(backend, "reshape_raw");
        if(!reshapeOp) return NULL;
        PyObjectScopedRef getShapeOp = PyObject_GetAttrString(backend, "get_shape_tuple_raw");
        if(!getShapeOp) return NULL;
        outRawTensor = _permuteAndExtend(funcName, permuteOp, reshapeOp, getShapeOp, tensor, dimsSeq, rawTensor, outDimsSeq, outPermutation);
        if(!outRawTensor) return NULL;
    }

    if(rawMode) {
        assert(outRawTensor);
        return outRawTensor.release();
    }

    assert((int) outPermutation.size() == outDimsSeq.size());
    PyObjectScopedRef outDims_ = PyTuple_New(outPermutation.size());
    if(!outDims_) return NULL;
    for(int i = 0; (size_t) i < outPermutation.size(); ++i) {
        PyObject* d;
        if(outPermutation[i] >= 0) {
            d = outDimsSeq.getItem(i);
            if(!d) return NULL;
            Py_INCREF(d);
        }
        else {
            // create dummy broadcast dim
            PyObject* dim = outDimsSeq.getItem(i);
            PyObjectScopedRef kind = PyObject_GetAttrString(dim, "kind");
            if(!kind) return NULL;
            PyObjectScopedRef description = PyObject_GetAttrString(dim, "description");
            if(!description) return NULL;
            if(description == Py_None) description = PyUnicode_InternFromString("unnamed_bc_dim1");
            else description = PyUnicode_FromFormat("%S_bc_dim1", description.get());
            if(!description) return NULL;
            PyObjectScopedRef dimValue = PyLong_FromLong(1);
            if(!dimValue) return NULL;
            PyObjectScopedRef args = PyTuple_New(0);
            if(!args) return NULL;
            PyObjectScopedRef kwargs = PyDict_New();
            if(!kwargs) return NULL;
            if(PyDict_SetItemString(kwargs, "kind", kind) < 0) return NULL;
            if(PyDict_SetItemString(kwargs, "description", description) < 0) return NULL;
            if(PyDict_SetItemString(kwargs, "dimension", dimValue) < 0) return NULL;
            if(PyDict_SetItemString(kwargs, "auto_generated", Py_True) < 0) return NULL;
            d = PyObject_Call(modState->dimType(), args, kwargs);
            if(!d) return NULL;
        }
        PyTuple_SET_ITEM(outDims_.get(), i, d);
    }

    PyObjectScopedRef name = PyObject_GetAttrString(tensor, "name");
    if(!name) return NULL;
    PyObjectScopedRef dtype = PyObject_GetAttrString(tensor, "dtype");
    if(!dtype) return NULL;
    PyObjectScopedRef version = PyObject_GetAttrString(tensor, "version");
    if(!version) return NULL;
    if(!PyLong_Check(version)) {
        PyErr_Format(
            PyExc_TypeError,
            "tensorCopyTemplate: tensor.version did not return an int, from version %R", version.get());
        return NULL;
    }
    long versionInt = PyLong_AsLong(version);
    if(versionInt != 1 && versionInt != 2) {
        if(!PyErr_Occurred())
            PyErr_Format(
                PyExc_ValueError,
                "tensorCopyTemplate: tensor.version is invalid, from version %R", version.get());
        return NULL;
    }
    PyObjectScopedRef extra = PyObject_GetAttrString(tensor, "_extra");
    if(!extra) return NULL;

    PyObjectScopedRef outTensor;
    if(versionInt == 2 && extra == Py_None) {
        outTensor = PyObject_CallFunctionObjArgs(
            modState->tensorType(), name.get(), outDims_.get(), dtype.get(), NULL);
        if(!outTensor) return NULL;
    }
    else {
        PyObjectScopedRef args = PyTuple_New(0);
        if(!args) return NULL;
        PyObjectScopedRef kwargs = PyDict_New();
        if(!kwargs) return NULL;
        if(PyDict_SetItemString(kwargs, "name", name) < 0) return NULL;
        if(PyDict_SetItemString(kwargs, "dims", outDims_) < 0) return NULL;
        if(PyDict_SetItemString(kwargs, "dtype", dtype) < 0) return NULL;
        if(PyDict_SetItemString(kwargs, "version", version) < 0) return NULL;
        if(extra != Py_None)
            if(!_copyTensorExtraToKwargs(extra, kwargs)) return NULL;
        outTensor = PyObject_Call(modState->tensorType(), args, kwargs);
        if(!outTensor) return NULL;
    }

    if(outRawTensor)
        if(PyObject_SetAttrString(outTensor, "_raw_tensor", outRawTensor) < 0)
            return NULL;

    if(versionInt == 1) {
        // Directly acess tensor because the default fallback logic for *_axis might be used,
        // and we want to check for that.
        if(!_setNewAxisConsistentFromPerm(funcName, tensor, outTensor, true, outPermutation, "time_dim_axis")) return NULL;
        if(!_setNewAxisConsistentFromPerm(funcName, tensor, outTensor, true, outPermutation, "feature_dim_axis")) return NULL;
    }
    else {
        // We can directly access _feature_dim_axis, which is either None or an int.
        if(!_setNewAxisConsistentFromPerm(funcName, tensor, outTensor, false, outPermutation, "_feature_dim_axis")) return NULL;
    }

    {
        PyObjectScopedRef sparse_dim = PyObject_GetAttrString(tensor, "sparse_dim");
        if(!sparse_dim) return NULL;
        if(sparse_dim != Py_None)
            if(PyObject_SetAttrString(outTensor, "sparse_dim", sparse_dim) < 0)
                return NULL;
    }

    return outTensor.release();
}

PyObject* pyTensorCopyCompatibleToDims(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "tensor_copy_compatible_to_dims() takes exactly 2 args: tensor, dims");
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;

    return tensorCopyCompatibleToDims<false>("tensor_copy_compatible_to_dims", modState, args[0], args[1]);
}

PyObject* pyTensorCopyCompatibleToDimsRaw(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if(nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "tensor_copy_compatible_to_dims_raw() takes exactly 2 args: tensor, dims");
        return NULL;
    }

    PyModuleState* modState = (PyModuleState*) PyModule_GetState(self);
    if(!modState) return NULL;

    return tensorCopyCompatibleToDims<true>("tensor_copy_compatible_to_dims_raw", modState, args[0], args[1]);
}

static PyObject* compareOrCombine(
    PyObject* a, PyObject* b,
    bool resultIsBool,
    PyModuleState* modState,
    const char* rawOpName,
    PyObject* rawOp, PyObject* permuteOp, PyObject* reshapeOp, PyObject* getShapeOp, PyObject* getDtypeOp, PyObject* convertToTensorLikeOp,
    bool needConvertToTensor,
    bool allowBroadcastAllSources,
    PyObject* dimOrder
) {
    if(!rawOp || !permuteOp || !reshapeOp || !getShapeOp || !getDtypeOp || !convertToTensorLikeOp) return NULL;

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
            PyObjectScopedRef dtype = PyObject_CallFunctionObjArgs(getDtypeOp, resRawTensor.get(), NULL);
            if(!dtype) return NULL;
            PyObjectScopedRef res = tensorCopyTemplateSimple(modState, a_is_tensor ? a : b, rawOpName, dtype, !resultIsBool);
            if(!res) return NULL;
            if(PyObject_SetAttrString(res, "_raw_tensor", resRawTensor) < 0) return NULL;
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
    PyTupleOrListStaticRef<true> aDimsSeq(aDims);
    PyObjectScopedRef bDims = PyObject_GetAttrString(b, "_dims");
    if(!bDims) return NULL;
    if(!PyTuple_Check(bDims)) {
        PyErr_Format(PyExc_TypeError, "compareOrCombine: expected b.dims to be tuple, got %R", bDims.get());
        return NULL;
    }
    PyTupleOrListStaticRef<true> bDimsSeq(bDims);

    PyObjectScopedRef aRawTensor = PyObject_GetAttrString(a, "_raw_tensor");
    if(!aRawTensor) return NULL;
    PyObjectScopedRef bRawTensor = PyObject_GetAttrString(b, "_raw_tensor");
    if(!bRawTensor) return NULL;

    // fast path: just use `is` checks for dims.
    // allowBroadcastAllSources=false makes it easier..., we know one dims should be the superset.
    // dimOrder makes it more difficult, does not need to be a superset. but by default, we don't have that.

    PyTupleOrListRef dimOrderSeq(dimOrder);
    if(dimOrder != Py_None) {
        if(!dimOrderSeq.isValid()) {
            PyErr_Format(PyExc_TypeError, "compareOrCombine: expected dim_order to be sequence, got %R", dimOrder);
            return NULL;
        }
    }

    // first very fast path check, check exact identity of dims
    if(_isSameSeqFast(aDimsSeq, bDimsSeq) && (dimOrder == Py_None || _isSameSeqFast(aDimsSeq, dimOrderSeq))) {
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensor.get(), bRawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        PyObjectScopedRef dtype = PyObject_CallFunctionObjArgs(getDtypeOp, resRawTensor.get(), NULL);
        if(!dtype) return NULL;
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, a, rawOpName, dtype, !resultIsBool);
        if(!res) return NULL;
        if(PyObject_SetAttrString(res, "_raw_tensor", resRawTensor) < 0) return NULL;
        return res.release();
    }

    // check b is scalar
    if(bDimsSeq.size() == 0 && (dimOrder == Py_None || _isSameSeqFast(aDimsSeq, dimOrderSeq))) {
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensor.get(), bRawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        PyObjectScopedRef dtype = PyObject_CallFunctionObjArgs(getDtypeOp, resRawTensor.get(), NULL);
        if(!dtype) return NULL;
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, a, rawOpName, dtype, !resultIsBool);
        if(!res) return NULL;
        if(PyObject_SetAttrString(res, "_raw_tensor", resRawTensor) < 0) return NULL;
        return res.release();
    }

    // check a is scalar
    if(aDimsSeq.size() == 0 && (dimOrder == Py_None || _isSameSeqFast(bDimsSeq, dimOrderSeq))) {
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(rawOp, aRawTensor.get(), bRawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        PyObjectScopedRef dtype = PyObject_CallFunctionObjArgs(getDtypeOp, resRawTensor.get(), NULL);
        if(!dtype) return NULL;
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, b, rawOpName, dtype, !resultIsBool);
        if(!res) return NULL;
        if(PyObject_SetAttrString(res, "_raw_tensor", resRawTensor) < 0) return NULL;
        return res.release();
    }

    // check if bDims is a subset of aDims, in the same order (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isSeqSubsetFast(bDimsSeq, aDimsSeq, outPermutation) && (dimOrder == Py_None || _isSameSeqFast(aDimsSeq, dimOrderSeq)))
            return _compareOrCombine_subsetDims<true>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, getShapeOp, getDtypeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aDimsSeq, bDimsSeq,
                outPermutation);
    }

    // check if aDims is a subset of bDims, in the same order (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isSeqSubsetFast(aDimsSeq, bDimsSeq, outPermutation) && (dimOrder == Py_None || _isSameSeqFast(bDimsSeq, dimOrderSeq)))
            return _compareOrCombine_subsetDims<false>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, getShapeOp, getDtypeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aDimsSeq, bDimsSeq,
                outPermutation);
    }

    // check if bDims is a subset of aDims, maybe reordered (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isSeqSubsetReorderFast(bDimsSeq, aDimsSeq, outPermutation) && (dimOrder == Py_None || _isSameSeqFast(aDimsSeq, dimOrderSeq)))
            return _compareOrCombine_subsetDims<true>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, getShapeOp, getDtypeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aDimsSeq, bDimsSeq,
                outPermutation);
    }

    // check if aDims is a subset of bDims, maybe reordered (fast dim identity check only)
    {
        std::vector<int> outPermutation;
        if(_isSeqSubsetReorderFast(aDimsSeq, bDimsSeq, outPermutation) && (dimOrder == Py_None || _isSameSeqFast(bDimsSeq, dimOrderSeq)))
            return _compareOrCombine_subsetDims<false>(
                modState, rawOpName, resultIsBool,
                permuteOp, reshapeOp, getShapeOp, getDtypeOp, rawOp,
                a, b,
                aRawTensor, bRawTensor,
                aDimsSeq, bDimsSeq,
                outPermutation);
    }

    {
        // follow the bin_op_out_template code

        // collect all dims
        PyObjectScopedRef allDims = PyList_New(0);
        if(!allDims) return NULL;
        for(int i = 0; i < aDimsSeq.size() + bDimsSeq.size(); ++i) {
            PyObject* dim =
                i < aDimsSeq.size() ?
                PyTuple_GET_ITEM(aDims.get(), i) :
                PyTuple_GET_ITEM(bDims.get(), i - aDimsSeq.size());
            {
                int contains = PySequence_Contains(allDims, dim);
                if(contains < 0) return NULL;
                if(contains) continue;
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
                continue;
            }
            int c = 0;
            for(int j = 0; j < (aDimsCount >= bDimsCount ? aDimsSeq.size() : bDimsCount); ++j) {
                PyObject* dim_ = PyTuple_GET_ITEM(
                    aDimsCount >= bDimsCount ? aDims.get() : bDims.get(), j);
                if(dim_ != dim) {
                    int eq = PyObject_RichCompareBool(dim_, dim, Py_EQ);
                    if(eq < 0) return NULL;
                    if(!eq) continue;
                }
                if(PyList_Append(allDims, dim_) < 0) return NULL;
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
        PyTupleOrListStaticRef<false> allDimsSeq(allDims);

        // check if all dims are in a and b, or whether we need allowBroadcastAllSources
        bool error = false;
        bool aDimsIsSubset = _isTupleSubsetReorderList(aDims, allDims, error);
        if(error) return NULL;
        bool bDimsIsSubset = _isTupleSubsetReorderList(bDims, allDims, error);
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
            std::vector<PyObject*> outDims;
            for(int i = 0; i < allDimsSeq.size(); ++i)
                outDims.push_back(allDimsSeq.getItem(i));
            struct Cmp {
                PyTupleOrListRef dimOrderSeq;
                bool hadError;
                Cmp(PyTupleOrListRef dimOrderSeq_)
                : dimOrderSeq(dimOrderSeq_), hadError(false)
                {}
                int getIndex(PyObject* a) {
                    if(hadError) return 0;
                    for(int i = 0; i < dimOrderSeq.size(); ++i) {
                        PyObject* d = dimOrderSeq.getItem(i);
                        if(d == a) return i;
                        int eq = PyObject_RichCompareBool(a, d, Py_EQ);
                        if(eq < 0) { hadError = true; return 0; }
                        if(eq) return i;
                    }
                    return dimOrderSeq.size();
                }
                bool operator()(PyObject* a, PyObject* b) {
                    if(a == b) return false;
                    return getIndex(a) < getIndex(b);
                }
            } cmp(dimOrderSeq);
            std::stable_sort(outDims.begin(), outDims.end(), cmp);
            if(cmp.hadError) return NULL;
            for(size_t i = 0; i < outDims.size(); ++i)
                PyList_SET_ITEM(allDims.get(), i, outDims[i]);
        }

        PyObjectScopedRef resRawTensor;
        {
            std::vector<int> outPermutation;
            PyObjectScopedRef aRawTensorExt = _permuteAndExtend(
                rawOpName, permuteOp, reshapeOp, getShapeOp,
                a, aDimsSeq, aRawTensor,
                allDimsSeq, outPermutation);
            if(!aRawTensorExt) return NULL;
            outPermutation.clear();
            PyObjectScopedRef bRawTensorExt = _permuteAndExtend(
                rawOpName, permuteOp, reshapeOp, getShapeOp,
                b, bDimsSeq, bRawTensor,
                allDimsSeq, outPermutation);
            if(!bRawTensorExt) return NULL;
            resRawTensor = PyObject_CallFunctionObjArgs(
                rawOp, aRawTensorExt.get(), bRawTensorExt.get(), NULL);
            if(!resRawTensor) return NULL;
        }

        PyObjectScopedRef res;
        {
            PyObjectScopedRef name = PyUnicode_FromString(rawOpName);
            if(!name) return NULL;
            PyObjectScopedRef dtype = PyObject_CallFunctionObjArgs(getDtypeOp, resRawTensor.get(), NULL);
            if(!dtype) return NULL;
            res = PyObject_CallFunctionObjArgs(
                modState->tensorType(), name.get(), allDims.get(), dtype.get(), NULL);
            if(!res) return NULL;
            if(PyObject_SetAttrString(res, "_raw_tensor", resRawTensor) < 0) return NULL;
        }

        {
            PyObjectScopedRef featureDim = _consistentFeatureDim(a, b);
            if(!featureDim) return NULL;
            if(featureDim != Py_None)
                if(PyObject_SetAttrString(res, "feature_dim", featureDim) < 0)
                    return NULL;
        }

        if(!resultIsBool) {
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
        case TOp_Maximum:
        case TOp_Minimum:
        case TOp_Eq:
        case TOp_Ne:
        case TOp_Lt:
        case TOp_Le:
        case TOp_Gt:
        case TOp_Ge:
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
        modState->cachedOp(TOp_GetDType, backendId),
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
        PyObjectScopedRef resRawTensor = PyObject_CallFunctionObjArgs(func, rawTensor.get(), NULL);
        if(!resRawTensor) return NULL;
        PyObject* getDtypeOp = modState->cachedOp(TOp_GetDType, BWCO_Torch);
        if(!getDtypeOp) return NULL;
        // Just overtake the result dtype. In case of abs(), it might change, but maybe also in other cases.
        PyObjectScopedRef dtype = PyObject_CallFunctionObjArgs(getDtypeOp, resRawTensor.get(), NULL);
        if(!dtype) return NULL;
        PyObjectScopedRef res = tensorCopyTemplateSimple(modState, tensor, rawOpName(op), dtype);
        if(!res) return NULL;
        if(!_checkTensorRawTensorAssignForBackendWithCachedOps(modState, BWCO_Torch, rawOpName(op), res, resRawTensor, false))
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

