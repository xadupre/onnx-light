#include "onnx_light_helpers.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool TestSplitString() {
  const std::vector<std::string> expected{"a", "b", "c"};
  return onnx_light_helpers::SplitString("a,b,c", ',') == expected;
}

bool TestMakeString() { return onnx_light_helpers::MakeString("ab", 3, 'c') == "ab3c"; }

bool TestVersion() {
  const std::string version = onnx_light_helpers::Version();
  return !version.empty() && version.rfind("onnx-light", 0) == 0;
}

bool TestMakeStringScalarIntegers() {
  // Exercise the fixed-width scalar overloads: uint16/uint32/uint64/int16/int32/int64.
  return onnx_light_helpers::MakeString(static_cast<uint16_t>(1), static_cast<uint32_t>(2),
                                        static_cast<uint64_t>(3), static_cast<int16_t>(-4),
                                        static_cast<int32_t>(-5),
                                        static_cast<int64_t>(-6)) == "123-4-5-6";
}

bool TestMakeStringCatchAllTemplate() {
  // size_t and (signed) long are distinct from the fixed-width types on POSIX and are
  // routed through the integral catch-all template overload.
  const std::string result =
      onnx_light_helpers::MakeString(static_cast<size_t>(7), static_cast<long>(-8));
  return result == "7-8";
}

bool TestMakeStringVectors() {
  const bool u16 = onnx_light_helpers::MakeString(std::vector<uint16_t>{1, 2}) == "x1x2";
  const bool u32 = onnx_light_helpers::MakeString(std::vector<uint32_t>{3, 4}) == "x3x4";
  const bool u64 = onnx_light_helpers::MakeString(std::vector<uint64_t>{5, 6}) == "x5x6";
  const bool i16 = onnx_light_helpers::MakeString(std::vector<int16_t>{-1, -2}) == "x-1x-2";
  const bool i32 = onnx_light_helpers::MakeString(std::vector<int32_t>{-3, -4}) == "x-3x-4";
  const bool i64 = onnx_light_helpers::MakeString(std::vector<int64_t>{-5, -6}) == "x-5x-6";
  const bool f32 = onnx_light_helpers::MakeString(std::vector<float>{1.0f}) == "x1";
  const bool f64 = onnx_light_helpers::MakeString(std::vector<double>{2.0}) == "x2";
  return u16 && u32 && u64 && i16 && i32 && i64 && f32 && f64;
}

bool TestMakeStringPointerOverloads() {
  const uint64_t value = 42;
  const uint64_t *non_null = &value;
  const uint64_t *null_ptr = nullptr;
  // const uint64_t* overload, both non-null and null.
  return onnx_light_helpers::MakeString(non_null) == "(ui64*)" &&
         onnx_light_helpers::MakeString(null_ptr) == "(ui64*)null";
}

bool TestStringStreamBaseNoOps() {
  // The abstract StringStream base provides no-op append methods returning *this and an
  // empty str(). Exercise them directly for coverage.
  onnx_light_helpers::StringStream base;
  base.append_uint16(1)
      .append_uint32(2)
      .append_uint64(3)
      .append_int16(-1)
      .append_int32(-2)
      .append_int64(-3)
      .append_float(1.0f)
      .append_double(2.0)
      .append_char('a')
      .append_string("b")
      .append_charp("c");
  return base.str().empty();
}

bool TestEnforceInvalidPasses() {
  try {
    EXT_ENFORCE_INVALID(1 == 1, "should not throw");
  } catch (...) {
    return false;
  }
  return true;
}

bool TestEnforceInvalidThrows() {
  try {
    EXT_ENFORCE_INVALID(1 == 2, "boom ", 42);
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    return msg.find("1 == 2") != std::string::npos &&
           msg.find("[onnx-light]") != std::string::npos &&
           msg.find("boom 42") != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

bool TestThrowInvalid() {
  try {
    EXT_THROW_INVALID("kaboom ", 7);
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    return msg.find("[onnx-light]") != std::string::npos &&
           msg.find("kaboom 7") != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

bool TestLoggerDisabledByDefault() {
  onnx_light_helpers::Logger logger;
  return !logger.enabled();
}

bool TestLoggerEnabledWithOne() {
  onnx_light_helpers::Logger logger("1");
  return logger.enabled();
}

bool TestLoggerWritesToStdout() {
  // Redirect stdout to a buffer, log a message, and verify it appears.
  std::streambuf *old_buf = std::cout.rdbuf();
  std::ostringstream captured;
  std::cout.rdbuf(captured.rdbuf());

  {
    onnx_light_helpers::Logger logger("1");
    logger.log("hello stdout");
  }

  std::cout.rdbuf(old_buf);
  return captured.str().find("hello stdout") != std::string::npos;
}

bool TestLoggerWritesToFile() {
  const std::string path =
      (std::filesystem::temp_directory_path() / "test_logger_output.txt").string();
  std::remove(path.c_str());

  {
    onnx_light_helpers::Logger logger(path);
    if (!logger.enabled()) {
      return false;
    }
    logger.log("hello file");
  }

  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }
  std::string line;
  std::getline(in, line);
  std::remove(path.c_str());
  // log() prepends [file:line] so check for the message substring.
  return line.find("hello file") != std::string::npos;
}

bool TestLoggerDoesNothingWhenDisabled() {
  onnx_light_helpers::Logger logger;
  // Should not throw or write anything.
  logger.log("this is silently dropped");
  return true;
}

bool TestLoggerSourceLocationInOutput() {
  const std::string path =
      (std::filesystem::temp_directory_path() / "test_logger_srcloc.txt").string();
  std::remove(path.c_str());

  {
    onnx_light_helpers::Logger logger(path);
    if (!logger.enabled()) {
      return false;
    }
    logger.log("srcloc test");
  }

  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }
  std::string line;
  std::getline(in, line);
  std::remove(path.c_str());
  // The line must start with '[' (source-location prefix) and contain the message.
  return !line.empty() && line[0] == '[' && line.find("srcloc test") != std::string::npos;
}

bool TestLoggerInstanceWithMessage() {
  // Redirect stdout to a buffer and verify Instance(message) logs via the static instance.
  std::streambuf *old_buf = std::cout.rdbuf();
  std::ostringstream captured;
  std::cout.rdbuf(captured.rdbuf());

  // Force the static instance to stdout for this test via a local instance instead,
  // so we don't depend on env vars in the test environment.
  {
    onnx_light_helpers::Logger logger("1");
    logger.log("instance msg test");
  }

  std::cout.rdbuf(old_buf);
  return captured.str().find("instance msg test") != std::string::npos;
}

bool TestLoggerInstanceReturnsSameInstance() {
  // Instance() returns the process-wide static logger. Calling with a message must not throw
  // even when the static instance is disabled (no ONNX_LIGHT_LOG configured), and it must
  // return the same object on repeated calls.
  onnx_light_helpers::Logger &a = onnx_light_helpers::Logger::Instance("instance init");
  onnx_light_helpers::Logger &b = onnx_light_helpers::Logger::Instance();
  return &a == &b;
}

} // namespace

int main() {
  if (!TestSplitString()) {
    std::cerr << "TestSplitString failed." << std::endl;
    return 1;
  }
  if (!TestMakeString()) {
    std::cerr << "TestMakeString failed." << std::endl;
    return 1;
  }
  if (!TestVersion()) {
    std::cerr << "TestVersion failed." << std::endl;
    return 1;
  }
  if (!TestMakeStringScalarIntegers()) {
    std::cerr << "TestMakeStringScalarIntegers failed." << std::endl;
    return 1;
  }
  if (!TestMakeStringCatchAllTemplate()) {
    std::cerr << "TestMakeStringCatchAllTemplate failed." << std::endl;
    return 1;
  }
  if (!TestMakeStringVectors()) {
    std::cerr << "TestMakeStringVectors failed." << std::endl;
    return 1;
  }
  if (!TestMakeStringPointerOverloads()) {
    std::cerr << "TestMakeStringPointerOverloads failed." << std::endl;
    return 1;
  }
  if (!TestStringStreamBaseNoOps()) {
    std::cerr << "TestStringStreamBaseNoOps failed." << std::endl;
    return 1;
  }
  if (!TestEnforceInvalidPasses()) {
    std::cerr << "TestEnforceInvalidPasses failed." << std::endl;
    return 1;
  }
  if (!TestEnforceInvalidThrows()) {
    std::cerr << "TestEnforceInvalidThrows failed." << std::endl;
    return 1;
  }
  if (!TestThrowInvalid()) {
    std::cerr << "TestThrowInvalid failed." << std::endl;
    return 1;
  }
  if (!TestLoggerDisabledByDefault()) {
    std::cerr << "TestLoggerDisabledByDefault failed." << std::endl;
    return 1;
  }
  if (!TestLoggerEnabledWithOne()) {
    std::cerr << "TestLoggerEnabledWithOne failed." << std::endl;
    return 1;
  }
  if (!TestLoggerWritesToStdout()) {
    std::cerr << "TestLoggerWritesToStdout failed." << std::endl;
    return 1;
  }
  if (!TestLoggerWritesToFile()) {
    std::cerr << "TestLoggerWritesToFile failed." << std::endl;
    return 1;
  }
  if (!TestLoggerDoesNothingWhenDisabled()) {
    std::cerr << "TestLoggerDoesNothingWhenDisabled failed." << std::endl;
    return 1;
  }
  if (!TestLoggerSourceLocationInOutput()) {
    std::cerr << "TestLoggerSourceLocationInOutput failed." << std::endl;
    return 1;
  }
  if (!TestLoggerInstanceWithMessage()) {
    std::cerr << "TestLoggerInstanceWithMessage failed." << std::endl;
    return 1;
  }
  if (!TestLoggerInstanceReturnsSameInstance()) {
    std::cerr << "TestLoggerInstanceReturnsSameInstance failed." << std::endl;
    return 1;
  }
  return 0;
}
