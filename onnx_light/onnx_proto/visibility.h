#pragma once

/**
 * Marks symbols that form the cross-library and public onnx-light ABI.
 *
 * Windows keeps WINDOWS_EXPORT_ALL_SYMBOLS for compatibility. ELF and Mach-O
 * builds hide everything by default and expose only declarations carrying this
 * annotation.
 */
#if defined(_WIN32)
#define ONNX_LIGHT_API
#elif defined(__GNUC__) || defined(__clang__)
#define ONNX_LIGHT_API __attribute__((visibility("default")))
#else
#define ONNX_LIGHT_API
#endif

#define ONNX_LIGHT_PROTO_API ONNX_LIGHT_API

#if defined(_WIN32) && defined(ONNX_LIGHT_SHARED_LIBS)
#if defined(lib_onnx_core_EXPORTS)
#define ONNX_LIGHT_CORE_API __declspec(dllexport)
#else
#define ONNX_LIGHT_CORE_API __declspec(dllimport)
#endif
#if defined(lib_onnx_lib_EXPORTS)
#define ONNX_LIGHT_LIB_API __declspec(dllexport)
#else
#define ONNX_LIGHT_LIB_API __declspec(dllimport)
#endif
#else
#define ONNX_LIGHT_CORE_API ONNX_LIGHT_API
#define ONNX_LIGHT_LIB_API ONNX_LIGHT_API
#endif
