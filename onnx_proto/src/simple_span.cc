#include "simple_span.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

namespace detail {
namespace {
thread_local size_t shared_payload_copy_depth = 0;
}

SharedPayloadCopyScope::SharedPayloadCopyScope() noexcept { ++shared_payload_copy_depth; }
SharedPayloadCopyScope::~SharedPayloadCopyScope() { --shared_payload_copy_depth; }
bool IsSharedPayloadCopy() noexcept { return shared_payload_copy_depth != 0; }
} // namespace detail

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
