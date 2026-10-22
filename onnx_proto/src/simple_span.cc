#include "simple_span.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

std::shared_ptr<void> ByteSpan::acquire_export_guard() const {
  auto guard = export_guard_.lock();
  if (!guard) {
    guard = std::make_shared<OwnedByteBuffer>(owned_.SharedCopy());
    export_guard_ = guard;
  }
  return guard;
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
