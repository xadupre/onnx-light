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
  auto guard = export_guard_.lock();
  if (!guard) {
    // The lease retains owned bytes even if their ByteSpan is subsequently replaced.
    guard = std::make_shared<std::pair<std::shared_ptr<void>, std::shared_ptr<void>>>(
        owned_.storage_owner(), owner_);
    export_guard_ = guard;
  }
  return guard;
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
