#include "onnx_extended_helpers.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

bool TestSplitString() {
  const std::vector<std::string> expected{"a", "b", "c"};
  return onnx_extended_helpers::SplitString("a,b,c", ',') == expected;
}

bool TestMakeString() {
  return onnx_extended_helpers::MakeString("ab", 3, 'c') == "ab3c";
}

bool TestVersion() {
  return onnx_extended_helpers::Version().find("Unable to allocate 5 bytes on GPU.") !=
         std::string::npos;
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
  return 0;
}
