ONNX Runtime custom AffineGrid
================================

This example implements a FLOAT AffineGrid custom operator for 2D and 3D
inputs. It deliberately creates no thread pool. Its Compute method calls
Ort::KernelContext::ParallelFor, which schedules rows on the intra-op workers
owned by the current ONNX Runtime session.

KernelContext::ParallelFor was added in ONNX Runtime API version 17, so ONNX
Runtime 1.17 or newer is required.

Build
-----

Point CMake at an extracted ONNX Runtime CPU release:

  cmake -S examples/onnxruntime_custom_affine_grid \
        -B build-ort-affine-grid \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
  cmake --build build-ort-affine-grid

When ONNXRUNTIME_ROOT_DIR is omitted, the example reuses the repository's
FindOrt module and downloads its default ONNX Runtime release.

Run
---

Generate the small custom-domain model (the Python ``onnx`` package is needed
only for this step):

  python examples/onnxruntime_custom_affine_grid/generate_model.py \
         build-ort-affine-grid/affine_grid_custom.onnx

Linux:

  build-ort-affine-grid/run_ort_affine_grid \
    build-ort-affine-grid/affine_grid_custom.onnx \
    build-ort-affine-grid/libort_affine_grid_custom.so

macOS uses libort_affine_grid_custom.dylib and Windows uses
ort_affine_grid_custom.dll.

The runner configures four intra-op threads. ParallelFor receives the
OrtKernelContext for this invocation, so ONNX Runtime selects and reuses those
session workers. Setting SetIntraOpNumThreads(1) makes the same kernel execute
serially without changing the custom operator.
