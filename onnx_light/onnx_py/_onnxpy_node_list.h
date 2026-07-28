#pragma once

#include "onnx.h"
#include <nanobind/nanobind.h>

namespace ONNX_LIGHT_NAMESPACE {

// Invokes ``fn`` with the Python ``nodes`` argument viewed as a
// ``RepeatedProtoField<NodeProto>``. When ``nodes`` already is such a field it
// is forwarded without copying; any other iterable (e.g. a plain ``list`` of
// ``NodeProto``) is materialised once into a temporary field. This keeps the
// node-consuming bindings on the pointer-backed storage without forcing the
// caller-side ``std::vector<NodeProto>`` -> ``RepeatedProtoField`` copy.
template <typename Fn> decltype(auto) WithNodeList(nanobind::handle nodes, Fn &&fn) {
  namespace nb = nanobind;
  if (nb::isinstance<utils::RepeatedProtoField<NodeProto>>(nodes)) {
    return fn(nb::cast<utils::RepeatedProtoField<NodeProto> &>(nodes));
  }
  utils::RepeatedProtoField<NodeProto> copied;
  for (nb::handle h : nb::borrow<nb::iterable>(nodes)) {
    copied.push_back(nb::cast<const NodeProto &>(h));
  }
  return fn(copied);
}

} // namespace ONNX_LIGHT_NAMESPACE
