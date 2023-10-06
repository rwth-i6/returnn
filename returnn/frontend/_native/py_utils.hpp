#ifndef __RETURNN_PY_UTILS_HPP__
#define __RETURNN_PY_UTILS_HPP__

#include <Python.h>

/* When you call any Python API which returns a new reference, e.g. PyObject_Call or whatever,
 * you need to Py_DECREF it at some point.
 * This class is a simple wrapper which does that automatically when it goes out of scope.
 * Example:
 *   PyObjectScopedRef result = PyObject_Call(...);
 *   // do something with ref
 *   // no need to Py_DECREF(ref) explicitly
*/
class PyObjectScopedRef {
public:
    PyObjectScopedRef(PyObject* obj = NULL) : _obj(obj) {}
    PyObjectScopedRef(const PyObjectScopedRef&) = delete;
    PyObjectScopedRef(PyObjectScopedRef&& other) : _obj(other._obj) { other._obj = NULL; }
    void operator=(PyObject* obj) { Py_CLEAR(_obj); _obj = obj; }
    void operator=(const PyObjectScopedRef&) = delete;
    operator PyObject*() const { return _obj; }
    PyObject* get() const { return _obj; }
    PyObject* release() { PyObject* obj = _obj; _obj = NULL; return obj; }
    ~PyObjectScopedRef() { Py_CLEAR(_obj); }

private:
    PyObject* _obj;
};

/*
Sequence interface contract:

- Holds borrowed reference to PyObject*.
- Copy object itself is supposed to be fast, small object.

Methods:

size() - returns the size of the sequence, fast op, cached
getItem(i) - returns the i-th item, fast op, no bound checks
get() - return PyObject* of the sequence
*/

template<bool isTuple /*false means list*/>
class PyTupleOrListStaticRef {
    PyObject* _obj;
    int _size;

public:
    PyTupleOrListStaticRef(PyObject* obj) : _obj(obj) {
#ifdef DEBUG
        assert(obj);
        if(isTuple) assert(PyTuple_Check(obj));
        else assert(PyList_Check(obj));
#endif
        if(isTuple) _size = PyTuple_GET_SIZE(obj);
        else _size = PyList_GET_SIZE(obj);
#ifdef DEBUG
        assert(_size >= 0);
#endif
    }

    int size() const { return _size; }
    PyObject* getItem(int i) const {
#ifdef DEBUG
        assert(i >= 0 && i < _size);
#endif
        if(isTuple) return PyTuple_GET_ITEM(_obj, i);
        else return PyList_GET_ITEM(_obj, i);
    }
    PyObject* get() const { return _obj; }
};

class PyTupleOrListRef {
    PyObject* _obj;
    enum { TupleType, ListType, UnknownType } _type;
    int _size;

public:
    PyTupleOrListRef(PyObject* obj) : _obj(obj) {
        if(!obj || obj == Py_None) _type = UnknownType;
        else if(PyTuple_Check(obj)) _type = TupleType;
        else if(PyList_Check(obj)) _type = ListType;
        else _type = UnknownType;
        if(_type == TupleType) _size = PyTuple_GET_SIZE(obj);
        else if(_type == ListType) _size = PyList_GET_SIZE(obj);
        else _size = -1;
#ifdef DEBUG
        if(_type != UnknownType)
            assert(_size >= 0);
#endif
    }

    bool isValid() const { return _type != UnknownType; }
    int size() const { return _size; }
    PyObject* getItem(int i) const {
#ifdef DEBUG
        assert(i >= 0 && i < _size);
#endif
        if(_type == TupleType) return PyTuple_GET_ITEM(_obj, i);
        else if(_type == ListType) return PyList_GET_ITEM(_obj, i);
        else return NULL;
    }
    PyObject* get() const { return _obj; }
};

#endif
