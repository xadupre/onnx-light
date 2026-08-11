// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <vector>

struct OnnxLightNumpyArray;
struct OnnxLightNumpyDtype;

enum class OnnxLightNumpyType {
  kObject,
  kFloat16,
  kFloat32,
  kFloat64,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUint8,
  kUint16,
  kUint32,
  kUint64,
  kBool,
  kComplex64,
  kComplex128,
};

int OnnxLightImportNumpy();
PyObject *OnnxLightNumpyFromAny(PyObject *value);
OnnxLightNumpyArray *OnnxLightNumpyArrayCast(PyObject *value);
OnnxLightNumpyDtype *OnnxLightNumpyArrayDtype(OnnxLightNumpyArray *array);
char OnnxLightNumpyDtypeKind(OnnxLightNumpyDtype *dtype);
std::ptrdiff_t OnnxLightNumpyDtypeSize(OnnxLightNumpyDtype *dtype);
PyObject *OnnxLightNumpyDtypeObject(OnnxLightNumpyDtype *dtype);
int OnnxLightNumpyArrayRank(OnnxLightNumpyArray *array);
std::ptrdiff_t OnnxLightNumpyArrayDimension(OnnxLightNumpyArray *array, int dimension);
std::ptrdiff_t OnnxLightNumpyArraySize(OnnxLightNumpyArray *array);
std::ptrdiff_t OnnxLightNumpyArrayItemSize(OnnxLightNumpyArray *array);
std::ptrdiff_t OnnxLightNumpyArrayByteSize(OnnxLightNumpyArray *array);
void *OnnxLightNumpyArrayData(OnnxLightNumpyArray *array);
PyObject *OnnxLightNumpyArrayGetItem(OnnxLightNumpyArray *array, char *pointer);
int OnnxLightNumpyArraySetItem(OnnxLightNumpyArray *array, char *pointer, PyObject *value);
OnnxLightNumpyDtype *OnnxLightNumpyDtypeFromName(const char *name);
OnnxLightNumpyDtype *OnnxLightNumpyDtypeFromType(OnnxLightNumpyType type);
PyObject *OnnxLightNumpyNewArray(OnnxLightNumpyDtype *dtype, const std::vector<int64_t> &shape,
                                 void *data, PyObject *owner);
