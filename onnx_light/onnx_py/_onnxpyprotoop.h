#pragma once

#include <nanobind/nanobind.h>

void AddOnnxPyProto(nanobind::module_ &m);
void AddOnnxPyOp(nanobind::module_ &m);