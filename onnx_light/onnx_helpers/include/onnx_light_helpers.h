#pragma once

#include "onnx_proto/visibility.h"
#include <algorithm>
#include <cstdint>
#include <float.h>
#include <fstream>
#include <iterator>
#include <source_location>
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
ONNX_LIGHT_PROTO_API std::string Version();

/**
 * Abstract string builder used by the MakeString helper functions.
 * Concrete subclasses override each typed append method so that the
 * same template-based formatting code can target different output sinks.
 */
class ONNX_LIGHT_PROTO_API StringStream {
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
ONNX_LIGHT_PROTO_API std::vector<std::string> SplitString(const std::string &input, char delimiter);

/** Appends a null-terminated C string to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const char *t);

/** Appends a standard string to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const std::string &t);

/** Appends a single character to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const char &t);

/** Appends an unsigned 16-bit integer to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const uint16_t &t);
/** Appends an unsigned 32-bit integer to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const uint32_t &t);
/** Appends an unsigned 64-bit integer to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const uint64_t &t);

/** Appends a signed 16-bit integer to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const int16_t &t);
/** Appends a signed 32-bit integer to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const int32_t &t);
/** Appends a signed 64-bit integer to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const int64_t &t);

/** Appends a textual description of the const uint64_t pointer reference to @p ss; outputs a null
 * marker when the pointer is null. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const uint64_t *&t);
/** Appends a textual description of the const uint64_t pointer to @p ss; outputs a null marker when
 * the pointer is null. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const uint64_t *t);

/** Appends a single-precision floating-point value to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const float &t);

/** Appends a double-precision floating-point value to @p ss. */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const double &t);

/** Appends each element of a uint16 vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss,
                                                    const std::vector<uint16_t> &t);

/** Appends each element of a uint32 vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss,
                                                    const std::vector<uint32_t> &t);

/** Appends each element of a uint64 vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss,
                                                    const std::vector<uint64_t> &t);

/** Appends each element of an int16 vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss,
                                                    const std::vector<int16_t> &t);

/** Appends each element of an int32 vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss,
                                                    const std::vector<int32_t> &t);

/** Appends each element of an int64 vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss,
                                                    const std::vector<int64_t> &t);

/** Appends each element of a float vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const std::vector<float> &t);

/** Appends each element of a double vector to @p ss, prefixed with "x". */
ONNX_LIGHT_PROTO_API void MakeStringInternalElement(StringStream &ss, const std::vector<double> &t);

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
ONNX_LIGHT_PROTO_API void MakeStringInternal(StringStream &ss);

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

/**
 * @def EXT_THROW_INVALID(...)
 * Throws a `std::invalid_argument` whose message is built by MakeString from @p __VA_ARGS__,
 * prefixed with "[onnx-light]".
 */
#if !defined(_THROW_INVALID_DEFINED)
#define EXT_THROW_INVALID(...)                                                                     \
  throw std::invalid_argument(onnx_light_helpers::MakeString(                                      \
      "[onnx-light] ", onnx_light_helpers::MakeString(__VA_ARGS__)));
#define _THROW_INVALID_DEFINED
#endif

/**
 * @class ParseLimitExceeded
 * Thrown when parsing hits a deliberate, configurable resource guard --
 * ParseOptions::max_recursion_depth, ParseOptions::max_tensor_size_bytes, or
 * ParseOptions::alignment -- rather than genuinely malformed/corrupted wire
 * bytes. Kept as a distinct type (derived from std::runtime_error) so the
 * top-level ParseFromString/ParseFromArray entry point can let these
 * deliberate policy violations continue to propagate as exceptions (the
 * caller configured the limit and needs to know it was hit) while still
 * catching genuine wire-format corruption and reporting it via a plain
 * `false` return, matching the protobuf API contract.
 */
class ParseLimitExceeded : public std::runtime_error {
public:
  explicit ParseLimitExceeded(const std::string &message) : std::runtime_error(message) {}
};

/**
 * @def EXT_THROW_LIMIT(...)
 * Throws a `ParseLimitExceeded` whose message is built by MakeString from @p __VA_ARGS__,
 * prefixed with "[onnx-light]". Use for deliberate, configurable resource guards
 * (recursion depth, tensor size, alignment) as opposed to wire-format corruption.
 */
#if !defined(_THROW_LIMIT_DEFINED)
#define EXT_THROW_LIMIT(...)                                                                       \
  throw onnx_light_helpers::ParseLimitExceeded(onnx_light_helpers::MakeString(                     \
      "[onnx-light] ", onnx_light_helpers::MakeString(__VA_ARGS__)));
#define _THROW_LIMIT_DEFINED
#endif

/**
 * @def EXT_ENFORCE_LIMIT(cond, ...)
 * Evaluates @p cond and throws a `ParseLimitExceeded` when it is false. Behaves
 * like EXT_ENFORCE but is used for deliberate, configurable resource guards
 * (recursion depth, tensor size, alignment) as opposed to wire-format corruption.
 */
#if !defined(_ENFORCE_LIMIT_DEFINED)
#define EXT_ENFORCE_LIMIT(cond, ...)                                                               \
  if (!(cond))                                                                                     \
    throw onnx_light_helpers::ParseLimitExceeded(onnx_light_helpers::MakeString(                   \
        "`", #cond, "` failed. ",                                                                  \
        onnx_light_helpers::MakeString("[onnx-light] ",                                            \
                                       onnx_light_helpers::MakeString(__VA_ARGS__))));
#define _ENFORCE_LIMIT_DEFINED
#endif

/**
 * @def EXT_ENFORCE_INVALID(cond, ...)
 * Evaluates @p cond and throws a `std::invalid_argument` when it is false.
 * The error message includes the stringified condition and the message built
 * from @p __VA_ARGS__ via MakeString, prefixed with "[onnx-light]".
 * Behaves like EXT_ENFORCE but raises std::invalid_argument instead of
 * std::runtime_error; intended to replace `if (cond) throw std::invalid_argument(...)`
 * patterns.
 */
#if !defined(_ENFORCE_INVALID_DEFINED)
#define EXT_ENFORCE_INVALID(cond, ...)                                                             \
  if (!(cond))                                                                                     \
    throw std::invalid_argument(onnx_light_helpers::MakeString(                                    \
        "`", #cond, "` failed. ",                                                                  \
        onnx_light_helpers::MakeString("[onnx-light] ",                                            \
                                       onnx_light_helpers::MakeString(__VA_ARGS__))));
#define _ENFORCE_INVALID_DEFINED
#endif

/** Returns true when @p value is a power of two and strictly positive. */
inline bool IsPowerOfTwo(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

/**
 * Validates an alignment option value.
 * @param alignment Alignment value to validate.
 * @param option_name Name of the option used in error messages.
 * Throws ParseLimitExceeded when alignment is negative or not a power of two when positive:
 * this rejects a caller-supplied configuration value, not wire-format-corrupted input, so it
 * must keep propagating as an exception through ParseFromString/ParseFromArray.
 */
inline void ValidateAlignmentOption(int64_t alignment, const char *option_name) {
  EXT_ENFORCE_LIMIT(alignment >= 0, option_name, " must be >= 0.");
  EXT_ENFORCE_LIMIT(alignment <= 1 || IsPowerOfTwo(alignment), option_name,
                    " must be a power of two when > 0, got ", alignment, ".");
}

/**
 * Simple single-destination logger backed by the standard library only.
 *
 * The logging destination is selected in the following priority order:
 *  1. The @p destination argument passed to the constructor (when non-empty).
 *  2. The value of the `ONNX_LIGHT_LOG` environment variable.
 *  3. Logging is disabled when neither source provides a destination.
 *
 * Destination semantics:
 *  - `"1"` — messages are written to stdout.
 *  - Any other non-empty string — messages are written to the file whose path equals that string.
 *
 * @warning This class is **not thread-safe**.  Concurrent calls to log() from
 *          multiple threads result in undefined behavior.  Callers that require
 *          thread-safe logging must provide their own synchronisation.
 */
class ONNX_LIGHT_PROTO_API Logger {
public:
  /**
   * Constructs a Logger and opens the configured destination.
   *
   * @param destination Overrides `ONNX_LIGHT_LOG` when non-empty.  Pass `"1"` to
   *                    redirect to stdout, or a file path to write to a file.
   *                    An empty string causes the environment variable to be
   *                    consulted instead.
   */
  explicit Logger(const std::string &destination = "");

  /** Destroys the Logger; flushes and closes any open file stream. */
  ~Logger();

  /**
   * Writes @p message followed by a newline to the configured destination and flushes
   * immediately.  Does nothing when logging is disabled.
   *
   * The source location is captured automatically at the call site and prepended to the
   * message as `[file:line] message`.
   *
   * @param message The text to log.
   * @param loc     Call-site location; defaults to `std::source_location::current()`.
   */
  void log(const std::string &message,
           const std::source_location loc = std::source_location::current());

  /** Returns true when a destination is configured and the logger is active. */
  bool enabled() const;

  /**
   * Returns a reference to the process-wide static Logger instance.
   *
   * The instance is constructed on the first call using the `ONNX_LIGHT_LOG`
   * environment variable as its destination.
   *
   * When @p message is non-null, it is logged (with source location) before the instance
   * is returned, allowing the one-liner `Logger::Instance("initialising")`.
   *
   * @param message Optional message to log; pass `nullptr` (default) to skip logging.
   * @param loc     Call-site location; defaults to `std::source_location::current()`.
   *
   * @warning The static instance is **not thread-safe**; see the class-level warning.
   */
  static Logger &Instance(const char *message = nullptr,
                          const std::source_location loc = std::source_location::current());

private:
  bool to_stdout_;
  bool enabled_;
  std::ofstream file_stream_;
};

} // namespace onnx_light_helpers
