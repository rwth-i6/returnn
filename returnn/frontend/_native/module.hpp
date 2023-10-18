#ifndef __RETURNN_MODULE_HPP__
#define __RETURNN_MODULE_HPP__

#include <Python.h>

// raw ops
enum RawOp {
    TOp_ConvertToTensor,
    TOp_ConvertToTensorLike,
    TOp_Permute,
    TOp_Reshape,
    TOp_GetShape,
    TOp_GetDType,

    // binary compare

    TOp_Eq,
    TOp_Ne,
    TOp_Lt,
    TOp_Le,
    TOp_Gt,
    TOp_Ge,

    // binary combine

    TOp_Add,
    TOp_Sub,
    TOp_Mul,
    TOp_TrueDiv,
    TOp_FloorDiv,
    TOp_Mod,
    TOp_Pow,

    TOp_Maximum,
    TOp_Minimum,
    TOp_SquaredDifference,

    TOp_And,
    TOp_Or,

    // unary

    TOp_Neg,
    TOp_Not,
    TOp_Abs,
    TOp_Ceil,
    TOp_Floor,

    NumTOps,
};

const char* rawOpName(RawOp op);

// all backends where we want to cache the ops, to support very efficient inlined code
enum BackendWithCachedOps {
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

    int pyTraverse(visitproc visit, void *arg) {
        Py_VISIT(_notSpecified);
        for(unsigned int i = 0; i < sizeof(_rawTensorTypes)/sizeof(_rawTensorTypes[0]); ++i)
            Py_VISIT(_rawTensorTypes[i]);
        Py_VISIT(_tensorType);
        Py_VISIT(_dimType);
        Py_VISIT(_globalBackend);
        Py_VISIT(_backendTensorTypeDispatchTable);
        for(unsigned int i = 0; i < NumBackendsWithCachedOps * NumTOps; ++i)
            Py_VISIT(_cachedOps[i]);
        Py_VISIT(_torchTensorType);
        Py_VISIT(_torchBackend);
        for(unsigned int i = 0; i < sizeof(_torchTensorDTypes)/sizeof(_torchTensorDTypes[0]); ++i) {
            Py_VISIT(_torchTensorDTypes[i].dtype);
            Py_VISIT(_torchTensorDTypes[i].name);
        }
        Py_VISIT(_torchTensorNameToDTypeDict);
        return 0;
    }

    int pyClear() {
        Py_CLEAR(_notSpecified);
        _rawTensorTypesLen = 0;
        for(unsigned int i = 0; i < sizeof(_rawTensorTypes)/sizeof(_rawTensorTypes[0]); ++i)
            Py_CLEAR(_rawTensorTypes[i]);
        Py_CLEAR(_tensorType);
        Py_CLEAR(_dimType);
        Py_CLEAR(_globalBackend);
        Py_CLEAR(_backendTensorTypeDispatchTable);
        for(unsigned int i = 0; i < NumBackendsWithCachedOps * NumTOps; ++i)
            Py_CLEAR(_cachedOps[i]);
        Py_CLEAR(_torchTensorType);
        Py_CLEAR(_torchBackend);
        for(unsigned int i = 0; i < sizeof(_torchTensorDTypes)/sizeof(_torchTensorDTypes[0]); ++i) {
            Py_CLEAR(_torchTensorDTypes[i].dtype);
            Py_CLEAR(_torchTensorDTypes[i].name);
        }
        Py_CLEAR(_torchTensorNameToDTypeDict);
        _module = NULL;
        return 0;
    }

    int pyInitModuleExec(PyObject* module);

    inline /*borrowed*/ PyObject* notSpecified() const { return _notSpecified; }
    inline /*borrowed*/ PyObject* tensorType() const { return _tensorType; }
    inline /*borrowed*/ PyObject* dimType() const { return _dimType; }
    inline /*borrowed*/ PyObject* globalBackend() const { return _globalBackend; }
    inline /*borrowed*/ PyObject* cachedOp(RawOp op, BackendWithCachedOps backend) {
        if(!_cachedOps[backend * NumTOps + op])
            if(!_cachedOpInit(backend))
                return NULL;
        PyObject* func = _cachedOps[backend * NumTOps + op];
        if(!func)
            PyErr_Format(PyExc_RuntimeError, "RETURNN frontend _native: invalid backend %d, op %d, '%s'", backend, op, rawOpName(op));
        return func;
    }
    inline int rawTensorTypesLen() const { return _rawTensorTypesLen; }
    inline /*borrowed*/ PyObject* rawTensorType(int i) const { return _rawTensorTypes[i]; }
    inline /*borrowed*/ PyObject* torchTensorDTypeName(PyObject* dtype, bool& error) {
        if(!_torchTensorDTypes[0].dtype) {
            if(!_torchTensorDTypesInit()) {
                error = true;
                return NULL;
            }
        }
        for(unsigned int i = 0; i < sizeof(_torchTensorDTypes)/sizeof(_torchTensorDTypes[0]); ++i) {
            if(_torchTensorDTypes[i].dtype == dtype)
                return _torchTensorDTypes[i].name;
        }
        return NULL;
    }
    inline /*borrowed*/ PyObject* torchTensorNameToDTypeDict() {
        if(!_torchTensorNameToDTypeDict)
            _torchTensorNameToDTypeDict = PyDict_New();
        return _torchTensorNameToDTypeDict;
    }

private:
    PyObject* _module; // weak
    PyObject* _notSpecified;
    int _rawTensorTypesLen;
    PyObject* _rawTensorTypes[10]; // copy of rf.RawTensorTypes: numpy.ndarray and co; size is just arbitrary upper limit
    PyObject* _tensorType;
    PyObject* _dimType;
    PyObject* _globalBackend;
    PyObject* _backendTensorTypeDispatchTable;
    PyObject* _cachedOps[NumBackendsWithCachedOps * NumTOps];
    PyObject* _torchTensorType;
    PyObject* _torchBackend;
    struct { PyObject* dtype; PyObject* name; } _torchTensorDTypes[5]; // size is just arbitrary; check _torchTensorDTypesInit
    PyObject* _torchTensorNameToDTypeDict;

    bool _torchTensorTypeMaybeInit(PyObject* obj);
    bool _torchTensorInit();
    bool _cachedOpInit(BackendWithCachedOps backend);
    bool _cachedOpInitTorch();
    bool _torchTensorDTypesInit();
};

#endif
