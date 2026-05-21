#pragma once

#include <nanobind/nanobind.h>

void AddOnnxPyProto(nanobind::module_ &m);
void AddOnnxPyLib(nanobind::module_ &m);
void AddOnnxPyExpressions(nanobind::module_ &m);
void AddOnnxPyBackend(nanobind::module_ &m);
