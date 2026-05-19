// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_pb.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE {
namespace Common {

/**
 * @brief Identifies the subsystem that generated a non-OK @c Status.
 *
 * Each enumerator corresponds to a distinct functional area of the ONNX
 * library so that error handlers can quickly determine the origin of a
 * failure without parsing the error message.
 */
enum class StatusCategory : std::uint8_t {
  /** No category; used by OK (success) statuses. */
  NONE = 0,
  /** Error originated in the model checker. */
  CHECKER = 1,
  /** Error originated in the optimizer. */
  OPTIMIZER = 2,
};

/**
 * @brief Distinguishes the type of error carried by a non-OK @c Status.
 */
enum class StatusCode : std::uint8_t {
  /** Operation completed successfully. */
  OK = 0,
  /** Generic, unclassified failure. */
  FAIL = 1,
  /** A caller-supplied argument was invalid. */
  INVALID_ARGUMENT = 2,
  /** The serialized protobuf data was malformed or otherwise unreadable. */
  INVALID_PROTOBUF = 3,
};

/**
 * @brief Represents the outcome of an operation.
 *
 * A default-constructed (or @c OK()-returned) @c Status represents success and
 * holds no heap allocation.  Any non-OK status stores a @c StatusCategory,
 * a @c StatusCode, and an optional human-readable error message via a
 * heap-allocated @c State object.
 *
 * @par Usage example
 * @code{.cpp}
 * using namespace ONNX_LIGHT_NAMESPACE::Common;
 *
 * Status ok;                    // success
 * assert(ok.IsOK());
 *
 * Status err{StatusCategory::CHECKER, StatusCode::FAIL, "bad model"};
 * assert(!err.IsOK());
 * std::cout << err.ToString();  // "CHECKER:1:bad model"
 * @endcode
 */
class Status {
public:
  /**
   * @brief Constructs a success (OK) status.
   *
   * The resulting object has no heap allocation; @c IsOK() returns @c true.
   */
  Status() noexcept = default;

  /**
   * @brief Constructs a non-OK status with a category, code, and message.
   *
   * @param category  Subsystem that generated the error.
   * @param code      Error code; must not be @c StatusCode::OK (asserted in debug builds).
   * @param msg       Human-readable description of the error.
   */
  Status(StatusCategory category, StatusCode code, const std::string &msg);

  /**
   * @brief Constructs a non-OK status with a category and code but no message.
   *
   * @param category  Subsystem that generated the error.
   * @param code      Error code; must not be @c StatusCode::OK (asserted in debug builds).
   */
  Status(StatusCategory category, StatusCode code);

  /**
   * @brief Copy-constructs a @c Status from another.
   *
   * @param other Source status to copy.
   */
  Status(const Status &other) { *this = other; }

  /**
   * @brief Copy-assigns a @c Status from another.
   *
   * Performs a deep copy of the internal @c State when @p other is non-OK.
   * Self-assignment is handled safely.
   *
   * @param other Source status to copy.
   * @return Reference to @c *this.
   */
  Status &operator=(const Status &other) {
    if (&other != this) {
      if (other.state_ == nullptr) {
        state_.reset();
      } else if (state_ != other.state_) {
        state_ = std::make_unique<State>(*other.state_);
      }
    }
    return *this;
  }

  /** @brief Move-constructs a @c Status, leaving the source in an OK state. */
  Status(Status &&) = default;

  /** @brief Move-assigns a @c Status, leaving the source in an OK state. */
  Status &operator=(Status &&) = default;

  /** @brief Destroys the @c Status, releasing any heap-allocated state. */
  ~Status() = default;

  /**
   * @brief Returns @c true if the status represents success.
   *
   * An OK status has no heap-allocated state; this check is therefore
   * equivalent to testing whether the internal pointer is null.
   */
  bool IsOK() const noexcept;

  /**
   * @brief Returns the status code.
   *
   * Returns @c StatusCode::OK when @c IsOK() is @c true.
   */
  StatusCode Code() const noexcept;

  /**
   * @brief Returns the status category.
   *
   * Returns @c StatusCategory::NONE when @c IsOK() is @c true.
   */
  StatusCategory Category() const noexcept;

  /**
   * @brief Returns the human-readable error message.
   *
   * Returns a reference to an empty string when @c IsOK() is @c true or when
   * no message was supplied to the constructor.
   */
  const std::string &ErrorMessage() const;

  /**
   * @brief Returns a human-readable string representation of the status.
   *
   * For OK statuses returns @c "OK".  For non-OK statuses the format is
   * @c "[CategoryError] : <code> : <code_name> : <message>", where the
   * category tag is @c [CheckerError] or @c [OptimizerError] as appropriate.
   */
  std::string ToString() const;

  /**
   * @brief Returns @c true if two statuses are equal.
   *
   * Two statuses are considered equal when they share the same @c State
   * pointer (identity) or when their @c ToString() representations match.
   *
   * @param other Status to compare against.
   */
  bool operator==(const Status &other) const {
    return (this->state_ == other.state_) || (ToString() == other.ToString());
  }

  /**
   * @brief Returns @c true if two statuses are not equal.
   *
   * @param other Status to compare against.
   */
  bool operator!=(const Status &other) const { return !(*this == other); }

  /**
   * @brief Returns a reference to a process-lifetime singleton OK status.
   *
   * Useful when an API must return a @c const Status& without constructing a
   * new object on each call.  The singleton is initialized on first use via a
   * function-local static, which is thread-safe since C++11.
   */
  static const Status &OK() noexcept;

private:
  /** @brief Internal heap-allocated storage for non-OK status information. */
  struct State {
    /**
     * @brief Constructs a State with the given category, code, and message.
     *
     * @param cat_  Status category.
     * @param code_ Status code.
     * @param msg_  Error message string (moved in).
     */
    State(StatusCategory cat_, StatusCode code_, std::string msg_)
        : category(cat_), code(code_), msg(std::move(msg_)) {}

    /** @brief Subsystem that generated the error. */
    StatusCategory category = StatusCategory::NONE;
    /** @brief Numeric error code. */
    StatusCode code{};
    /** @brief Human-readable error message. */
    std::string msg;
  };

  /** @brief Returns a reference to a process-lifetime empty string.
   *
   * Initialized on first use via a function-local static; thread-safe since C++11.
   */
  static const std::string &EmptyString();

  /** @brief Heap-allocated state; null when the status is OK. */
  std::unique_ptr<State> state_;
};

/**
 * @brief Streams a human-readable representation of @p status to @p out.
 *
 * Equivalent to writing @c status.ToString() to the stream.
 *
 * @param out    Destination output stream.
 * @param status Status value to write.
 * @return Reference to @p out to allow chaining.
 */
inline std::ostream &operator<<(std::ostream &out, const Status &status) {
  return out << status.ToString();
}

} // namespace Common
} // namespace ONNX_LIGHT_NAMESPACE
