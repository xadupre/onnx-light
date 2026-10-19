#include "onnx_threads/cpu_descriptor.h"
#include "onnx_threads/cpu_executor.h"
#include "onnx_threads/parallel_for.h"
#include "onnx_threads/thread_pool.h"

#include <atomic>

int main() {
  namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
  runtime::CpuExecutionPolicy policy;
  policy.num_threads = 2;
  runtime::CpuExecutorRegistry registry(1);
  auto executor = registry.Acquire(policy);
  std::atomic<int> count{0};
  ONNX_LIGHT_NAMESPACE::utils::ThreadPool io;
  io.Start(1);
  io.SubmitTask([&count]() { count.fetch_add(10); });
  {
    runtime::CpuExecutorScope scope(executor.get());
    runtime::ParallelFor(20, 1, [&count](int64_t begin, int64_t end) {
      count.fetch_add(static_cast<int>(end - begin));
    });
  }
  io.Wait();
  return count == 30 &&
                 !ONNX_LIGHT_NAMESPACE::core::platform::GetCpuDescriptor().architecture.empty()
             ? 0
             : 1;
}
