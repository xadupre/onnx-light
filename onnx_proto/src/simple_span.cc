#include "simple_span.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

std::shared_ptr<void> ByteSpan::acquire_export_guard() const {
  if (borrowed_)
    return owner_;
  auto guard = export_guard_.lock();
  if (!guard || guard->first.get() != static_cast<const OwnedByteBuffer &>(owned_).data()) {
    // The lease retains owned bytes even if their ByteSpan is subsequently replaced.
    guard = std::make_shared<ExportGuard>(owned_.storage_owner(), owner_);
    export_guard_ = guard;
  }
  return guard;
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
