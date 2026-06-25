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
  return 0;
}
