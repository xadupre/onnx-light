#include "simple_span.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

std::shared_ptr<void> ByteSpan::acquire_export_guard() const {
  if (borrowed_)
    return {};
  auto guard = export_guard_.lock();
  if (!guard || static_cast<const OwnedByteBuffer &>(*guard).data() != owned_.data()) {
    guard = std::make_shared<OwnedByteBuffer>(owned_.SharedCopy());
    export_guard_ = guard;
  }
  return guard;
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
