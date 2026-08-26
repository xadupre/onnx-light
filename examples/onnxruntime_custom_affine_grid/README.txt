ONNX Runtime custom AffineGrid
================================

This example exposes onnx-light's FLOAT AffineGrid implementation as an ONNX
Runtime Lite Custom Op. It does not reimplement AffineGrid and deliberately
creates no thread pool.

For each invocation, the adapter creates an external CpuExecutor whose
dispatcher calls Ort::KernelContext::ParallelFor. The existing onnx-light
AffineGrid kernel therefore keeps calling its normal ParallelFor helper, while
the resulting blocks run on the intra-op workers owned by the current ONNX
Runtime session.

KernelContext::ParallelFor was added in ONNX Runtime API version 17, so ONNX
Runtime 1.17 or newer is required.

Install onnx-light
------------------

The custom-op library links to onnx_light::lib_onnx_kernels:

  cmake -S . -B build-install \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNX_LIGHT_BUILD_PYTHON=OFF \
        -DCMAKE_INSTALL_PREFIX=/path/to/onnx-light-install
  cmake --build build-install
  cmake --install build-install

Build the custom op
-------------------

Point CMake at an extracted ONNX Runtime CPU release:

  cmake -S examples/onnxruntime_custom_affine_grid \
        -B build-ort-affine-grid \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/path/to/onnx-light-install \
        -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
  cmake --build build-ort-affine-grid

When ONNXRUNTIME_ROOT_DIR is omitted, the example reuses the repository's
FindOrt module and downloads its default ONNX Runtime release.

Run
---

Generate the small custom-domain model with the installed ``onnx_light``
package:

  python examples/onnxruntime_custom_affine_grid/generate_model.py \
         build-ort-affine-grid/affine_grid_custom.onnx

Linux:

  build-ort-affine-grid/run_ort_affine_grid \
    build-ort-affine-grid/affine_grid_custom.onnx \
    build-ort-affine-grid/libort_affine_grid_custom.so

macOS uses libort_affine_grid_custom.dylib and Windows uses
ort_affine_grid_custom.dll.

The runner configures four intra-op threads. The external CpuExecutor receives
the OrtKernelContext for the invocation, so ONNX Runtime selects and reuses
those session workers. Setting SetIntraOpNumThreads(1) makes the same
onnx-light kernel execute serially without changing the custom operator.
