#include "onnx.h"
#include "onnx_proto/onnx_dlpack.h"
#include "onnx_proto/onnx_helper.h"
#include <stdexcept>

int main() {
  namespace proto = ONNX_LIGHT_NAMESPACE;
  proto::ModelProto model;
  model.set_ir_version(7);
  model.ref_graph().set_name("standalone");
  model.ref_graph().add_node(proto::MakeNode("Identity", {"X"}, {"Y"}));
  proto::ModelProto parsed;
  if (!parsed.ParseFromString(model.SerializeAsString()) ||
      parsed.ref_graph().ref_name() != "standalone" || parsed.ref_graph().ref_node().size() != 1 ||
      parsed.ref_graph().ref_node()[0].ref_op_type() != "Identity") {
    throw std::runtime_error("Standalone proto round-trip failed.");
  }
  const auto dtype = proto::DLPackDataTypeFromOnnx(proto::TensorProto::FLOAT);
  if (dtype.code != kDLFloat || dtype.bits != 32 || dtype.lanes != 1) {
    throw std::runtime_error("Standalone DLPack linkage failed.");
  }
}
