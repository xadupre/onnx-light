// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "_onnxpy_numpy_api.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ONNX_LIGHT_KERNELS_ARRAY_API
#include <numpy/arrayobject.h>

namespace {

PyArrayObject *Array(OnnxLightNumpyArray *array) {
  return reinterpret_cast<PyArrayObject *>(array);
}

PyArray_Descr *Dtype(OnnxLightNumpyDtype *dtype) {
  return reinterpret_cast<PyArray_Descr *>(dtype);
}

} // namespace

int OnnxLightImportNumpy() { return _import_array(); }

PyObject *OnnxLightNumpyFromAny(PyObject *value) {
  return PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY_RO | NPY_ARRAY_NOTSWAPPED, nullptr);
}

OnnxLightNumpyArray *OnnxLightNumpyArrayCast(PyObject *value) {
  return reinterpret_cast<OnnxLightNumpyArray *>(value);
}

OnnxLightNumpyDtype *OnnxLightNumpyArrayDtype(OnnxLightNumpyArray *array) {
  return reinterpret_cast<OnnxLightNumpyDtype *>(PyArray_DESCR(Array(array)));
}

char OnnxLightNumpyDtypeKind(OnnxLightNumpyDtype *dtype) { return Dtype(dtype)->kind; }

std::ptrdiff_t OnnxLightNumpyDtypeSize(OnnxLightNumpyDtype *dtype) {
  return PyDataType_ELSIZE(Dtype(dtype));
}

PyObject *OnnxLightNumpyDtypeObject(OnnxLightNumpyDtype *dtype) {
  return reinterpret_cast<PyObject *>(dtype);
}

int OnnxLightNumpyArrayRank(OnnxLightNumpyArray *array) { return PyArray_NDIM(Array(array)); }

std::ptrdiff_t OnnxLightNumpyArrayDimension(OnnxLightNumpyArray *array, int dimension) {
  return PyArray_DIM(Array(array), dimension);
}

std::ptrdiff_t OnnxLightNumpyArraySize(OnnxLightNumpyArray *array) {
  return PyArray_SIZE(Array(array));
}

std::ptrdiff_t OnnxLightNumpyArrayItemSize(OnnxLightNumpyArray *array) {
  return PyArray_ITEMSIZE(Array(array));
}

std::ptrdiff_t OnnxLightNumpyArrayByteSize(OnnxLightNumpyArray *array) {
  return PyArray_NBYTES(Array(array));
}

void *OnnxLightNumpyArrayData(OnnxLightNumpyArray *array) { return PyArray_DATA(Array(array)); }

PyObject *OnnxLightNumpyArrayGetItem(OnnxLightNumpyArray *array, char *pointer) {
  return PyArray_GETITEM(Array(array), pointer);
}

int OnnxLightNumpyArraySetItem(OnnxLightNumpyArray *array, char *pointer, PyObject *value) {
  return PyArray_SETITEM(Array(array), pointer, value);
}

OnnxLightNumpyDtype *OnnxLightNumpyDtypeFromName(const char *name) {
  PyObject *name_pointer = PyUnicode_FromString(name);
  if (name_pointer == nullptr)
    return nullptr;
  PyArray_Descr *dtype = nullptr;
  const int result = PyArray_DescrConverter(name_pointer, &dtype);
  Py_DECREF(name_pointer);
  return result == NPY_FAIL ? nullptr : reinterpret_cast<OnnxLightNumpyDtype *>(dtype);
}

OnnxLightNumpyDtype *OnnxLightNumpyDtypeFromType(OnnxLightNumpyType type) {
  int numpy_type = NPY_NOTYPE;
  switch (type) {
  case OnnxLightNumpyType::kObject:
    numpy_type = NPY_OBJECT;
    break;
  case OnnxLightNumpyType::kFloat16:
    numpy_type = NPY_FLOAT16;
    break;
  case OnnxLightNumpyType::kFloat32:
    numpy_type = NPY_FLOAT32;
    break;
  case OnnxLightNumpyType::kFloat64:
    numpy_type = NPY_FLOAT64;
    break;
  case OnnxLightNumpyType::kInt8:
    numpy_type = NPY_INT8;
    break;
  case OnnxLightNumpyType::kInt16:
    numpy_type = NPY_INT16;
    break;
  case OnnxLightNumpyType::kInt32:
    numpy_type = NPY_INT32;
    break;
  case OnnxLightNumpyType::kInt64:
    numpy_type = NPY_INT64;
    break;
  case OnnxLightNumpyType::kUint8:
    numpy_type = NPY_UINT8;
    break;
  case OnnxLightNumpyType::kUint16:
    numpy_type = NPY_UINT16;
    break;
  case OnnxLightNumpyType::kUint32:
    numpy_type = NPY_UINT32;
    break;
  case OnnxLightNumpyType::kUint64:
    numpy_type = NPY_UINT64;
    break;
  case OnnxLightNumpyType::kBool:
    numpy_type = NPY_BOOL;
    break;
  case OnnxLightNumpyType::kComplex64:
    numpy_type = NPY_COMPLEX64;
    break;
  case OnnxLightNumpyType::kComplex128:
    numpy_type = NPY_COMPLEX128;
    break;
  }
  return reinterpret_cast<OnnxLightNumpyDtype *>(PyArray_DescrFromType(numpy_type));
}

PyObject *OnnxLightNumpyNewArray(OnnxLightNumpyDtype *dtype, const std::vector<int64_t> &shape,
                                 void *data, PyObject *owner) {
  std::vector<npy_intp> dimensions(shape.begin(), shape.end());
  PyObject *array = PyArray_NewFromDescr(
      &PyArray_Type, Dtype(dtype), static_cast<int>(dimensions.size()), dimensions.data(), nullptr,
      data, data == nullptr ? 0 : NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, nullptr);
  if (array == nullptr)
    return nullptr;
  if (owner != nullptr &&
      PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(array), owner) != 0) {
    Py_DECREF(array);
    return nullptr;
  }
  return array;
}
