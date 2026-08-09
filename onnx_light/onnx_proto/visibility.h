#pragma once

/**
 * Marks symbols that form the cross-library and public lib_onnx_proto ABI.
 *
 * Windows keeps WINDOWS_EXPORT_ALL_SYMBOLS for compatibility. ELF and Mach-O
 * builds hide everything by default and expose only declarations carrying this
 * annotation.
 */
#if defined(_WIN32)
#define ONNX_LIGHT_PROTO_API
#elif defined(__GNUC__) || defined(__clang__)
#define ONNX_LIGHT_PROTO_API __attribute__((visibility("default")))
#else
#define ONNX_LIGHT_PROTO_API
#endif
