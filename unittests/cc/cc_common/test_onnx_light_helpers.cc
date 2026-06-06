#include "onnx_light_helpers.h"

#include <iostream>
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
  return 0;
}
