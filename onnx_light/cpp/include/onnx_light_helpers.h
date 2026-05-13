#pragma once

#include <algorithm>
#include <cstdint>
#include <float.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace onnx_light_helpers {

/** Returns a version string that exercises the MakeString helpers. */
std::string Version();

/**
 * Abstract string builder used by the MakeString helper functions.
 * Concrete subclasses override each typed append method so that the
 * same template-based formatting code can target different output sinks.
 */
class StringStream {
public:
  /** Constructs an empty stream. */
  StringStream();
  /** Destroys the stream and releases any owned resources. */
  virtual ~StringStream();
  /** Appends an unsigned 16-bit integer value. */
  virtual StringStream &append_uint16(const uint16_t &obj);
  /** Appends an unsigned 32-bit integer value. */
  virtual StringStream &append_uint32(const uint32_t &obj);
  /** Appends an unsigned 64-bit integer value. */
  virtual StringStream &append_uint64(const uint64_t &obj);
  /** Appends a signed 16-bit integer value. */
  virtual StringStream &append_int16(const int16_t &obj);
  /** Appends a signed 32-bit integer value. */
  virtual StringStream &append_int32(const int32_t &obj);
  /** Appends a signed 64-bit integer value. */
  virtual StringStream &append_int64(const int64_t &obj);
  /** Appends a single-precision floating-point value. */
  virtual StringStream &append_float(const float &obj);
  /** Appends a double-precision floating-point value. */
  virtual StringStream &append_double(const double &obj);
  /** Appends a single character. */
  virtual StringStream &append_char(const char &obj);
  /** Appends a standard string. */
  virtual StringStream &append_string(const std::string &obj);
  /** Appends a null-terminated character array. */
  virtual StringStream &append_charp(const char *obj);
  /** Returns the accumulated content as a standard string. */
  virtual std::string str();
  /** Allocates and returns a new concrete StringStream instance. */
  static StringStream *NewStream();
};

/** Splits @p input into substrings at each occurrence of @p delimiter. */
std::vector<std::string> SplitString(const std::string &input, char delimiter);

/** Appends a null-terminated C string to @p ss. */
void MakeStringInternalElement(StringStream &ss, const char *t);

/** Appends a standard string to @p ss. */
void MakeStringInternalElement(StringStream &ss, const std::string &t);

/** Appends a single character to @p ss. */
void MakeStringInternalElement(StringStream &ss, const char &t);

/** Appends an unsigned 16-bit integer to @p ss. */
void MakeStringInternalElement(StringStream &ss, const uint16_t &t);
/** Appends an unsigned 32-bit integer to @p ss. */
void MakeStringInternalElement(StringStream &ss, const uint32_t &t);
/** Appends an unsigned 64-bit integer to @p ss. */
void MakeStringInternalElement(StringStream &ss, const uint64_t &t);

/** Appends a signed 16-bit integer to @p ss. */
void MakeStringInternalElement(StringStream &ss, const int16_t &t);
/** Appends a signed 32-bit integer to @p ss. */
void MakeStringInternalElement(StringStream &ss, const int32_t &t);
/** Appends a signed 64-bit integer to @p ss. */
void MakeStringInternalElement(StringStream &ss, const int64_t &t);

/** Appends a textual description of the const uint64_t pointer reference to @p ss; outputs a null
 * marker when the pointer is null. */
void MakeStringInternalElement(StringStream &ss, const uint64_t *&t);
/** Appends a textual description of the const uint64_t pointer to @p ss; outputs a null marker when
 * the pointer is null. */
void MakeStringInternalElement(StringStream &ss, const uint64_t *t);

/** Appends a single-precision floating-point value to @p ss. */
void MakeStringInternalElement(StringStream &ss, const float &t);

/** Appends a double-precision floating-point value to @p ss. */
void MakeStringInternalElement(StringStream &ss, const double &t);

/** Appends each element of a uint16 vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<uint16_t> &t);

/** Appends each element of a uint32 vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<uint32_t> &t);

/** Appends each element of a uint64 vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<uint64_t> &t);

/** Appends each element of an int16 vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<int16_t> &t);

/** Appends each element of an int32 vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<int32_t> &t);

/** Appends each element of an int64 vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<int64_t> &t);

/** Appends each element of a float vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<float> &t);

/** Appends each element of a double vector to @p ss, prefixed with "x". */
void MakeStringInternalElement(StringStream &ss, const std::vector<double> &t);

/**
 * Catch-all overload for integer types not covered by the explicit declarations above.
 *
 * On POSIX platforms (macOS, Linux) `unsigned long` (= `size_t`) and `long` (= `ssize_t`)
 * are distinct from `uint64_t` (`unsigned long long`) and `int64_t` (`long long`).  Without
 * this overload the call would be ambiguous across the fixed-width integer overloads.
 * Non-template overloads take priority, so on platforms where these types coincide
 * (e.g. Windows where `size_t` == `uint64_t`) the explicit overloads remain selected.
 *
 * @tparam T An integral type that is not `bool`, `char`, or any of the fixed-width types
 *           already covered by explicit overloads.
 * @param ss The stream to append to.
 * @param t  The value to append.
 */
template <
    typename T,
    std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value &&
                         !std::is_same<T, char>::value && !std::is_same<T, uint16_t>::value &&
                         !std::is_same<T, uint32_t>::value && !std::is_same<T, uint64_t>::value &&
                         !std::is_same<T, int16_t>::value && !std::is_same<T, int32_t>::value &&
                         !std::is_same<T, int64_t>::value,
                     int> = 0>
inline void MakeStringInternalElement(StringStream &ss, const T &t) {
  if constexpr (std::is_unsigned<T>::value)
    ss.append_uint64(static_cast<uint64_t>(t));
  else
    ss.append_int64(static_cast<int64_t>(t));
}

/** Base case of MakeStringInternal: does nothing when there are no remaining arguments. */
void MakeStringInternal(StringStream &ss);

/**
 * Appends the single value @p t to @p ss.
 * @tparam T Type of the value to append; must have a corresponding MakeStringInternalElement
 * overload.
 */
template <typename T, typename... Args>
inline void MakeStringInternal(StringStream &ss, const T &t) {
  MakeStringInternalElement(ss, t);
}

/**
 * Appends @p t and all remaining @p args to @p ss.
 * @tparam T    Type of the first value.
 * @tparam Args Types of the remaining values.
 */
template <typename T, typename... Args>
inline void MakeStringInternal(StringStream &ss, const T &t, const Args &...args) {
  MakeStringInternalElement(ss, t);
  MakeStringInternal(ss, args...);
}

/**
 * Formats all arguments as a single concatenated string.
 *
 * Allocates a temporary StringStream, appends each argument via MakeStringInternal,
 * and returns the resulting string.
 *
 * @tparam Args Types of the values to format.
 * @param  args Values to format.
 * @returns     The concatenated string representation of all arguments.
 */
template <typename... Args> inline std::string MakeString(const Args &...args) {
  StringStream *ss = StringStream::NewStream();
  MakeStringInternal(*ss, args...);
  std::string res = ss->str();
  delete ss;
  return res;
}

/**
 * @def EXT_THROW(...)
 * Throws a `std::runtime_error` whose message is built by MakeString from @p __VA_ARGS__,
 * prefixed with "[onnx-light]".
 */
#if !defined(_THROW_DEFINED)
#define EXT_THROW(...)                                                                             \
  throw std::runtime_error(onnx_light_helpers::MakeString(                                         \
      "[onnx-light] ", onnx_light_helpers::MakeString(__VA_ARGS__)));
#define _THROW_DEFINED
#endif

/**
 * @def EXT_ENFORCE(cond, ...)
 * Evaluates @p cond and throws a `std::runtime_error` when it is false.
 * The error message includes the stringified condition and the message built
 * from @p __VA_ARGS__ via MakeString, prefixed with "[onnx-light]".
 */
#if !defined(_ENFORCE_DEFINED)
#define EXT_ENFORCE(cond, ...)                                                                     \
  if (!(cond))                                                                                     \
    throw std::runtime_error(onnx_light_helpers::MakeString(                                       \
        "`", #cond, "` failed. ",                                                                  \
        onnx_light_helpers::MakeString("[onnx-light] ",                                            \
                                       onnx_light_helpers::MakeString(__VA_ARGS__))));
#define _ENFORCE_DEFINED
#endif

/** Returns true when @p value is a power of two and strictly positive. */
inline bool IsPowerOfTwo(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

/** Validates an alignment option: value must be >=0 and a power of two when >0. */
inline void ValidateAlignmentOption(int64_t alignment, const char *option_name) {
  EXT_ENFORCE(alignment >= 0, option_name, " must be >= 0.");
  EXT_ENFORCE(alignment <= 1 || IsPowerOfTwo(alignment), option_name,
              " must be a power of two when > 0, got ", alignment, ".");
}

} // namespace onnx_light_helpers
