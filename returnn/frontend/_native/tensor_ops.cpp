
#include <Python.h>
#include "tensor_ops.hpp"

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
