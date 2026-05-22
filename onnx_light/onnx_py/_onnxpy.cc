#include "../onnx_proto/_onnxpy.h"
#include "onnx.h"
#include "onnx_lib/onnx-data.pb.h"

using namespace ONNX_LIGHT_NAMESPACE;

NB_MODULE(_onnxpy, m) {
  m.doc() = "onnx from python without protobuf but using the same format";
  m.attr("IR_VERSION") = static_cast<int>(IR_VERSION);

  AddOnnxPyProto(m);
  AddOnnxPyLib(m);
  AddOnnxPyExpressions(m);
  AddOnnxPyBackend(m);
  AddOnnxPyOp(m);
}
